import matplotlib.pyplot as plt
import numpy as np

from mec_side import (
    MECNode,
    ServerVehicle,
    UserVehicle,
    add_to_pending_capacity_records,
    add_to_pending_service_records,
    broadcast_capacity_info,
    capacity_chain,
    create_capacity_record,
    create_service_record,
    create_updated_capacity_record_after_execution,
    execute_task_on_server_vehicle,
    get_confirmed_service_records,
    pbft_consensus_process,
    pending_capacity_records,
    pending_service_records,
    service_chain,
    verify_certificate,
)
from user_side import (
    calculate_best_response,
    calculate_cost,
    calculate_data_rate,
    calculate_mec_latency,
    calculate_payoff,
    calculate_utility,
    calculate_v2v_latency,
    execute_task_on_mec,
    make_offloading_decision,
    receive_capacity_info,
    run_best_response_until_convergence,
    select_available_candidate,
    send_service_request_to_mec,
)

# =========================================================
# BLOCK 38: Simulation Parameters
# =========================================================

NUM_USERS = 6

NUM_TASKS = 3

simulation_results = []

FAULTY_MEC_IDS = {3}


# =========================================================
# Figure 5 convergence test
# =========================================================

# Figure 5 scenario reconstruction:
# The paper gives the common parameters in Table I
# but does not provide per-vehicle distances, channel gains,
# or initial probabilities. These lists are used to reproduce
# the qualitative convergence behavior shown in Fig. 5.


def run_figure5_convergence_test(
    bandwidth,
    transmit_power,
    path_loss_exponent,
    channel_gain,
    noise_power,
    input_size,
    complexity,
    mec_cpu_frequency,
    server_vehicle_cpu_frequency,
    beta_uplink,
    beta_downlink,
    beta_request,
    beta_result,
    deadline,
    value_factor,
    service_quality,
    price_ratio,
    arrival_rates,
    initial_probabilities,
):
    figure5_probabilities = initial_probabilities.copy()

    probability_history = []

    figure5_service_quality_list = [
        0.85,
        0.75,
        1.0,
        0.70,
        0.80,
        0.90,
    ]

    # figure5_arrival_rates = [
    #     0.70,
    #     0.50,
    #     0.90,
    #     0.50,
    #     0.60,
    #     0.70,
    # ]
    figure5_arrival_rates = [0.7] * NUM_USERS
    figure5_distance_to_mec_list = [30, 50, 70, 30, 50, 70]
    figure5_distance_to_vehicle_list = [30, 10, 10, 20, 15, 12]
    figure5_channel_gain_list = [
        1.6,
        1.3,
        0.7,
        1.5,
        1.0,
        0.8,
    ]
    print("\n[Figure 5 Test] Convergence of 6 user vehicles:")

    for iteration in range(50):
        probability_history.append(figure5_probabilities.copy())
        print(f"\n[Figure 5 Test] Iteration {iteration + 1}")

        new_probabilities = figure5_probabilities.copy()

        for vehicle_index in range(NUM_USERS):
            figure5_distance_to_mec = figure5_distance_to_mec_list[vehicle_index]
            figure5_distance_to_vehicle = figure5_distance_to_vehicle_list[
                vehicle_index
            ]

            figure5_uplink_rate = calculate_data_rate(
                bandwidth,
                transmit_power,
                figure5_distance_to_mec,
                path_loss_exponent,
                figure5_channel_gain_list[vehicle_index],
                noise_power,
            )

            figure5_downlink_rate = calculate_data_rate(
                bandwidth,
                transmit_power,
                figure5_distance_to_mec,
                path_loss_exponent,
                figure5_channel_gain_list[vehicle_index],
                noise_power,
            )

            figure5_request_rate = calculate_data_rate(
                bandwidth,
                transmit_power,
                figure5_distance_to_vehicle,
                path_loss_exponent,
                figure5_channel_gain_list[vehicle_index],
                noise_power,
            )

            figure5_result_rate = calculate_data_rate(
                bandwidth,
                transmit_power,
                figure5_distance_to_vehicle,
                path_loss_exponent,
                figure5_channel_gain_list[vehicle_index],
                noise_power,
            )

            figure5_mec_latency = calculate_mec_latency(
                input_size=input_size,
                complexity=complexity,
                mec_cpu_frequency=mec_cpu_frequency,
                uplink_rate=figure5_uplink_rate,
                downlink_rate=figure5_downlink_rate,
                beta_uplink=beta_uplink,
                beta_downlink=beta_downlink,
            )

            figure5_v2v_latency = calculate_v2v_latency(
                input_size=input_size,
                complexity=complexity,
                server_vehicle_cpu_frequency=server_vehicle_cpu_frequency,
                request_rate=figure5_request_rate,
                result_rate=figure5_result_rate,
                beta_request=beta_request,
                beta_result=beta_result,
            )

            old_probability = figure5_probabilities[vehicle_index]

            new_probability = calculate_best_response(
                mec_latency=figure5_mec_latency,
                v2v_latency=figure5_v2v_latency,
                deadline=deadline,
                value_factor=value_factor,
                # service_quality=service_quality,
                service_quality=figure5_service_quality_list[vehicle_index],
                price_ratio=price_ratio,
                # arrival_rates=arrival_rates,
                arrival_rates=figure5_arrival_rates,
                probabilities=figure5_probabilities,
                current_vehicle_index=vehicle_index,
            )

            new_probabilities[vehicle_index] = new_probability

            print(
                f"Vehicle {vehicle_index + 1}: "
                f"old p = {old_probability:.6f}, "
                f"new p = {new_probability:.6f}"
            )
        figure5_probabilities = new_probabilities

    print("\nFinal probabilities:")
    print(probability_history[-1])
    return probability_history


# =========================================================
# Figure 5 Convergence Plotting Function
# =========================================================


def plot_figure5_convergence(probability_history):
    iterations = list(range(1, len(probability_history) + 1))

    for vehicle_index in range(NUM_USERS):
        vehicle_probabilities = [
            iteration_probabilities[vehicle_index]
            for iteration_probabilities in probability_history
        ]

        plt.plot(
            iterations,
            vehicle_probabilities,
            label=f"Vehicle {vehicle_index + 1}",
        )

    plt.xlabel("Iteration")
    plt.ylabel("Offloading probability")
    plt.title("Figure 5 - Convergence of Offloading Probability")
    plt.legend()
    plt.grid(True)
    plt.show()


# =========================================================
# Figure 6 Average Probability Calculation Function
# =========================================================


def calculate_figure6_average_probability(
    num_vehicles,
    arrival_rate,
    price_ratio,
    value_factor=0.7,
):
    probabilities = [0.5] * num_vehicles
    arrival_rates = [arrival_rate] * num_vehicles

    max_iterations = 100
    tolerance = 1e-4

    bandwidth = 10e6
    transmit_power = 0.2
    path_loss_exponent = 2.0  # noqa: F841
    channel_gain = 1.0
    noise_power = 1e-9

    input_size = 1e6
    complexity = 240
    mec_cpu_frequency = 5e9
    server_vehicle_cpu_frequency = 1e9

    beta_uplink = 1.0
    beta_downlink = 0.05
    beta_request = 1.0
    beta_result = 0.05

    distance_to_mec = 50.0
    distance_to_vehicle = 10.0  # noqa: F841

    # mec_latency = 0.06
    # v2v_latency = 0.24

    uplink_rate = calculate_data_rate(
        bandwidth=bandwidth,
        transmit_power=transmit_power,
        distance=distance_to_mec,
        path_loss_exponent=path_loss_exponent,
        channel_gain=channel_gain,
        noise_power=noise_power,
    )

    downlink_rate = calculate_data_rate(
        bandwidth=bandwidth,
        transmit_power=transmit_power,
        distance=distance_to_mec,
        path_loss_exponent=path_loss_exponent,
        channel_gain=channel_gain,
        noise_power=noise_power,
    )

    request_rate = calculate_data_rate(
        bandwidth=bandwidth,
        transmit_power=transmit_power,
        distance=distance_to_vehicle,
        path_loss_exponent=path_loss_exponent,
        channel_gain=channel_gain,
        noise_power=noise_power,
    )

    result_rate = calculate_data_rate(
        bandwidth=bandwidth,
        transmit_power=transmit_power,
        distance=distance_to_vehicle,
        path_loss_exponent=path_loss_exponent,
        channel_gain=channel_gain,
        noise_power=noise_power,
    )

    mec_latency = calculate_mec_latency(
        input_size=input_size,
        complexity=complexity,
        mec_cpu_frequency=mec_cpu_frequency,
        uplink_rate=uplink_rate,
        downlink_rate=downlink_rate,
        beta_uplink=beta_uplink,
        beta_downlink=beta_downlink,
    )

    v2v_latency = calculate_v2v_latency(
        input_size=input_size,
        complexity=complexity,
        server_vehicle_cpu_frequency=server_vehicle_cpu_frequency,
        request_rate=request_rate,
        result_rate=result_rate,
        beta_request=beta_request,
        beta_result=beta_result,
    )

    deadline = 1.0
    # value_factor = 0.7
    service_quality = 0.9

    for _ in range(max_iterations):
        old_probabilities = probabilities.copy()
        new_probabilities = probabilities.copy()

        for vehicle_index in range(num_vehicles):
            new_probabilities[vehicle_index] = calculate_best_response(
                mec_latency=mec_latency,
                v2v_latency=v2v_latency,
                deadline=deadline,
                value_factor=value_factor,
                service_quality=service_quality,
                price_ratio=price_ratio,
                arrival_rates=arrival_rates,
                probabilities=old_probabilities,
                current_vehicle_index=vehicle_index,
            )

        probabilities = new_probabilities

        max_difference = max(
            abs(probabilities[index] - old_probabilities[index])
            for index in range(num_vehicles)
        )

        if max_difference < tolerance:
            break

    average_probability = sum(probabilities) / len(probabilities)

    # if num_vehicles == 1:
    #     print(
    #         f"N=1, lambda={arrival_rate}, "
    #         f"rho={price_ratio}, "
    #         f"average={average_probability}"
    #     )

    return average_probability


# =========================================================
# Figure 7 Average Probability Calculation Function
# =========================================================


def calculate_figure7_average_probability(
    value_factor,
    arrival_rate,
    price_ratio,
):
    num_vehicles = 10

    average_probability = calculate_figure6_average_probability(
        num_vehicles=num_vehicles,
        arrival_rate=arrival_rate,
        price_ratio=price_ratio,
        value_factor=value_factor,
    )

    return average_probability


# =========================================================
# Figure 8 Expected Latency Calculation Function
# =========================================================


def calculate_figure8_expected_latency(
    value_factor,
    arrival_rate,
    price_ratio,
):
    average_probability = calculate_figure7_average_probability(
        value_factor=value_factor,
        arrival_rate=arrival_rate,
        price_ratio=price_ratio,
    )

    mec_latency = 0.055
    v2v_latency = 0.245

    expected_latency = (
        average_probability * mec_latency + (1.0 - average_probability) * v2v_latency
    )

    return expected_latency


# =========================================================
# Figure 6 Test Function
# =========================================================
def run_figure6_test():
    vehicle_counts = range(2, 71)

    figure6_scenarios = [
        {"arrival_rate": 0.5, "price_ratio": 0.7},
        {"arrival_rate": 0.7, "price_ratio": 0.7},
        {"arrival_rate": 0.9, "price_ratio": 0.7},
        {"arrival_rate": 0.7, "price_ratio": 0.5},
        {"arrival_rate": 0.7, "price_ratio": 0.9},
    ]

    figure6_results = {}

    for scenario in figure6_scenarios:
        print(
            f"\n[Figure 6 Test] "
            f"lambda={scenario['arrival_rate']}, "
            f"rho={scenario['price_ratio']}"
        )
        average_probabilities = []

        for num_vehicles in vehicle_counts:
            average_probability = calculate_figure6_average_probability(
                num_vehicles=num_vehicles,
                arrival_rate=scenario["arrival_rate"],
                price_ratio=scenario["price_ratio"],
            )

            average_probabilities.append(average_probability)
        print(f"Stored {len(average_probabilities)} points")

        scenario_label = (
            f"lambda={scenario['arrival_rate']}, " f"rho={scenario['price_ratio']}"
        )

        figure6_results[scenario_label] = average_probabilities

    return vehicle_counts, figure6_results


# =========================================================
# Plotting Function for Figure 6 Results
# =========================================================


def plot_figure6_results(
    vehicle_counts,
    figure6_results,
):
    vehicle_counts = list(vehicle_counts)

    for label, average_probabilities in figure6_results.items():
        plt.plot(
            vehicle_counts,
            average_probabilities,
            label=label,
        )

    plt.xlabel("Number of Vehicles")
    plt.ylabel("Average Offloading Probability")
    plt.title("Figure 6 - Average Offloading Probability")
    plt.legend()
    plt.grid(True)
    plt.show()


# =========================================================
# Figure 7 Test Function
# =========================================================


def run_figure7_test():
    value_factors = np.linspace(0, 1, 21)

    figure7_scenarios = [
        {"arrival_rate": 0.5, "price_ratio": 0.7},
        {"arrival_rate": 0.7, "price_ratio": 0.7},
        {"arrival_rate": 0.9, "price_ratio": 0.7},
        {"arrival_rate": 0.7, "price_ratio": 0.5},
        {"arrival_rate": 0.7, "price_ratio": 0.9},
    ]

    figure7_results = {}

    for scenario in figure7_scenarios:
        average_probabilities = []

        print(
            f"\n[Figure 7 Test] "
            f"lambda={scenario['arrival_rate']}, "
            f"rho={scenario['price_ratio']}"
        )

        for value_factor in value_factors:
            average_probability = calculate_figure7_average_probability(
                value_factor=value_factor,
                arrival_rate=scenario["arrival_rate"],
                price_ratio=scenario["price_ratio"],
            )

            average_probabilities.append(average_probability)

        scenario_label = (
            f"lambda={scenario['arrival_rate']}, " f"rho={scenario['price_ratio']}"
        )

        figure7_results[scenario_label] = average_probabilities

        print(f"Stored {len(average_probabilities)} points")

    return value_factors, figure7_results


# =========================================================
# Figure 8 Test Function
# =========================================================


def run_figure8_test():
    value_factors = [i / 20 for i in range(21)]

    figure8_scenarios = [
        {"arrival_rate": 0.5, "price_ratio": 0.7},
        {"arrival_rate": 0.7, "price_ratio": 0.7},
        {"arrival_rate": 0.9, "price_ratio": 0.7},
        {"arrival_rate": 0.7, "price_ratio": 0.5},
        {"arrival_rate": 0.7, "price_ratio": 0.9},
    ]

    figure8_results = {}

    for scenario in figure8_scenarios:
        expected_latencies = []

        print(
            f"\n[Figure 8 Test] "
            f"lambda={scenario['arrival_rate']}, "
            f"rho={scenario['price_ratio']}"
        )

        for value_factor in value_factors:
            expected_latency = calculate_figure8_expected_latency(
                value_factor=value_factor,
                arrival_rate=scenario["arrival_rate"],
                price_ratio=scenario["price_ratio"],
            )

            expected_latencies.append(expected_latency)

        scenario_label = (
            f"lambda={scenario['arrival_rate']}, " f"rho={scenario['price_ratio']}"
        )

        figure8_results[scenario_label] = expected_latencies

        print(f"Stored {len(expected_latencies)} points")

    return value_factors, figure8_results


# =========================================================
# Plotting Function for Figure 7 Results
# =========================================================


def plot_figure7_results(
    value_factors,
    figure7_results,
):
    for label, average_probabilities in figure7_results.items():
        plt.plot(
            value_factors,
            average_probabilities,
            label=label,
        )

    plt.xlabel("Value Factor")
    plt.ylabel("Average Offloading Probability")
    plt.title("Figure 7 - Average Offloading Probability vs Value Factor")
    plt.legend()
    plt.grid(True)
    plt.show()


# =========================================================
# Plotting Function for Figure 8 Results
# =========================================================


def plot_figure8_results(
    value_factors,
    figure8_results,
):
    plt.figure(figsize=(12, 7))

    for label, expected_latencies in figure8_results.items():
        plt.plot(
            value_factors,
            expected_latencies,
            label=label,
        )

    plt.title("Figure 8 - Expected Latency vs Value Factor")
    plt.xlabel("Value Factor")
    plt.ylabel("Expected Latency (s)")
    plt.grid(True)
    plt.legend()

    plt.show()


# =========================================================
# BLOCK 76: Single Task Simulation Function
# =========================================================


def run_single_task(
    task_counter: int,
    server_vehicles,
    user_vehicle,
    mec_nodes,
    debug_print,
    enable_service_retry_test,
):

    print(f"\n========== TASK {task_counter} ==========")

    print("[Main] MEC nodes initialized.")
    print("[Main] Server vehicles initialized.")
    print("[Main] User vehicle initialized.")

    pending_capacity_records.clear()
    pending_service_records.clear()
    capacity_chain.clear()
    service_chain.clear()
    # =========================================================
    # BLOCK 23: Capacity Record Creation and Capacity Chain Update
    # =========================================================

    for server_vehicle in server_vehicles:

        if verify_certificate(server_vehicle):

            capacity_record = create_capacity_record(
                server_vehicle=server_vehicle,
            )

            add_to_pending_capacity_records(capacity_record)

    # capacity_pbft_result = pbft_consensus_process(
    #     mec_nodes=mec_nodes,
    #     pending_records=pending_capacity_records,
    #     blockchain=capacity_chain,
    # )

    # print("\n[Main] Capacity PBFT Result:")
    # print(capacity_pbft_result)

    capacity_pbft_result = pbft_consensus_process(
        mec_nodes=mec_nodes,
        pending_records=pending_capacity_records,
        blockchain=capacity_chain,
    )

    print("\n[Main] Capacity PBFT Result:")
    print(capacity_pbft_result)
    # =========================================================
    # BLOCK 24: Broadcast Capacity Information to User Vehicle
    # =========================================================

    broadcasted_capacity_info = broadcast_capacity_info()

    received_capacity_info = receive_capacity_info(broadcasted_capacity_info)

    debug_print("\n[Main] Received Capacity Information:")
    debug_print(received_capacity_info)

    bandwidth = 10e6
    transmit_power = 0.2
    path_loss_exponent = 2.0
    channel_gain = 1.0
    noise_power = 1e-9

    input_size = 1e6
    complexity = 240
    mec_cpu_frequency = 5e9

    beta_uplink = 1.0
    beta_downlink = 0.05
    beta_request = 1.0
    beta_result = 0.05

    # =========================================================
    # BLOCK 25: Select Available Candidate Server Vehicle
    # =========================================================

    confirmed_service_records = get_confirmed_service_records()

    selected_candidate = select_available_candidate(
        user_vehicle=user_vehicle,
        capacity_records=received_capacity_info,
        service_records=confirmed_service_records,
    )

    debug_print("\n[Main] Selected Candidate:")
    debug_print(selected_candidate)

    if selected_candidate is None:
        print("[Main] No valid server vehicle candidate. Task will be executed on MEC.")

        # fallback_uplink_rate = 10e6
        # fallback_downlink_rate = 10e6

        distance_to_mec = 50.0

        fallback_uplink_rate = calculate_data_rate(
            bandwidth=bandwidth,
            transmit_power=transmit_power,
            distance=distance_to_mec,
            path_loss_exponent=path_loss_exponent,
            channel_gain=channel_gain,
            noise_power=noise_power,
        )

        fallback_downlink_rate = calculate_data_rate(
            bandwidth=bandwidth,
            transmit_power=transmit_power,
            distance=distance_to_mec,
            path_loss_exponent=path_loss_exponent,
            channel_gain=channel_gain,
            noise_power=noise_power,
        )

        fallback_mec_latency = calculate_mec_latency(
            input_size=input_size,
            complexity=complexity,
            mec_cpu_frequency=mec_cpu_frequency,
            uplink_rate=fallback_uplink_rate,
            downlink_rate=fallback_downlink_rate,
            beta_uplink=beta_uplink,
            beta_downlink=beta_downlink,
        )

        mec_execution_result = execute_task_on_mec(
            user_vehicle=user_vehicle,
            mec_latency=fallback_mec_latency,
        )

        print("\n[Main] MEC Execution Result:")
        print(mec_execution_result)

        return
    # =========================================================
    # BLOCK 26: Data Rate and Latency Calculation
    # =========================================================

    distance_to_mec = 50.0
    distance_to_server_vehicle = abs(
        user_vehicle.trajectory - selected_candidate.trajectory
    )

    uplink_rate = calculate_data_rate(
        bandwidth,
        transmit_power,
        distance_to_mec,
        path_loss_exponent,
        channel_gain,
        noise_power,
    )

    downlink_rate = calculate_data_rate(
        bandwidth,
        transmit_power,
        distance_to_mec,
        path_loss_exponent,
        channel_gain,
        noise_power,
    )

    request_rate = calculate_data_rate(
        bandwidth,
        transmit_power,
        distance_to_server_vehicle,
        path_loss_exponent,
        channel_gain,
        noise_power,
    )

    result_rate = calculate_data_rate(
        bandwidth,
        transmit_power,
        distance_to_server_vehicle,
        path_loss_exponent,
        channel_gain,
        noise_power,
    )

    mec_latency = calculate_mec_latency(
        input_size=input_size,
        complexity=complexity,
        mec_cpu_frequency=mec_cpu_frequency,
        uplink_rate=uplink_rate,
        downlink_rate=downlink_rate,
        beta_uplink=beta_uplink,
        beta_downlink=beta_downlink,
    )

    v2v_latency = calculate_v2v_latency(
        input_size=input_size,
        complexity=complexity,
        server_vehicle_cpu_frequency=selected_candidate.resource,
        request_rate=request_rate,
        result_rate=result_rate,
        beta_request=beta_request,
        beta_result=beta_result,
    )

    print("\n[Main] MEC Latency:")
    print(mec_latency)

    print("\n[Main] V2V Latency:")
    print(v2v_latency)

    # =========================================================
    # BLOCK 27: Utility, Cost and Payoff Calculation
    # =========================================================

    deadline = 1.0
    value_factor = 0.7

    arrival_rates = [0.7] * NUM_USERS
    # initial_probabilities = [0.5] * NUM_USERS
    initial_probabilities = [
        0.75,
        0.70,
        0.65,
        0.95,
        0.90,
        1.00,
    ]
    current_vehicle_index = 0

    probability_mec = initial_probabilities[current_vehicle_index]

    price_mec = 1.0
    # price_ratio = selected_candidate.price / price_mec
    price_ratio = 0.7

    utility = calculate_utility(
        probability_mec=probability_mec,
        mec_latency=mec_latency,
        v2v_latency=v2v_latency,
        deadline=deadline,
        value_factor=value_factor,
        service_quality=selected_candidate.quality,
    )

    cost = calculate_cost(
        probability_mec=probability_mec,
        arrival_rates=arrival_rates,
        probabilities=initial_probabilities,
        price_ratio=price_ratio,
        current_vehicle_index=current_vehicle_index,
    )

    payoff = calculate_payoff(
        utility=utility,
        cost=cost,
    )

    print("\n[Main] Probability of MEC Offloading:")
    print(probability_mec)

    print("\n[Main] Price Ratio:")
    print(price_ratio)

    print("\n[Main] Utility:")
    print(utility)

    print("\n[Main] Cost:")
    print(cost)

    print("\n[Main] Payoff:")
    print(payoff)

    # =========================================================
    # BLOCK 28: Best Response Update and Convergence
    # =========================================================

    # =========================================================
    # Figure 5 test function call
    # =========================================================

    # figure5_history = run_figure5_convergence_test(
    #     bandwidth=bandwidth,
    #     transmit_power=transmit_power,
    #     path_loss_exponent=path_loss_exponent,
    #     channel_gain=channel_gain,
    #     noise_power=noise_power,
    #     input_size=input_size,
    #     complexity=complexity,
    #     mec_cpu_frequency=mec_cpu_frequency,
    #     server_vehicle_cpu_frequency=selected_candidate.resource,
    #     beta_uplink=beta_uplink,
    #     beta_downlink=beta_downlink,
    #     beta_request=beta_request,
    #     beta_result=beta_result,
    #     deadline=deadline,
    #     value_factor=value_factor,
    #     service_quality=selected_candidate.quality,
    #     price_ratio=price_ratio,
    #     arrival_rates=arrival_rates,
    #     initial_probabilities=initial_probabilities,
    # )

    # print("\n[Figure 5 Test] Stored history length:")
    # print(len(figure5_history))

    # print("[Figure 5 Test] Number of vehicles in first iteration:")
    # print(len(figure5_history[0]))

    # print("[Figure 5 Test] First stored probabilities:")
    # print(figure5_history[0])

    # print("[Figure 5 Test] Last stored probabilities:")
    # print(figure5_history[-1])

    # plot_figure5_convergence(figure5_history)

    converged_probability = run_best_response_until_convergence(
        mec_latency=mec_latency,
        v2v_latency=v2v_latency,
        deadline=deadline,
        value_factor=value_factor,
        service_quality=selected_candidate.quality,
        price_ratio=price_ratio,
        arrival_rates=arrival_rates,
        initial_probabilities=initial_probabilities,
        current_vehicle_index=current_vehicle_index,
    )

    print("\n[Main] Converged Probability p_i:")
    print(converged_probability)

    # =========================================================
    # BLOCK 29: Final Offloading Decision
    # =========================================================

    offloading_decision = make_offloading_decision(
        probability_mec=converged_probability,
    )

    print("\n[Main] Final Offloading Decision:")
    print(offloading_decision)

    # =========================================================
    # BLOCK 30: MEC Execution or V2V Service Request
    # =========================================================

    if offloading_decision == "MEC":

        mec_execution_result = execute_task_on_mec(
            user_vehicle=user_vehicle,
            mec_latency=mec_latency,
        )

        print("\n[Main] MEC Execution Result:")
        print(mec_execution_result)

        task_result = {
            "task_id": task_counter,
            "selected_provider": None,
            "mec_latency": mec_latency,
            "v2v_latency": v2v_latency,
            "utility": utility,
            "cost": cost,
            "payoff": payoff,
            "converged_probability": converged_probability,
            "offloading_decision": offloading_decision,
        }

        print("\n[Main] Task Result:")
        print(task_result)

        return task_result

    else:

        service_request = send_service_request_to_mec(
            user_vehicle=user_vehicle,
            candidate=selected_candidate,
            estimated_duration=v2v_latency,
        )

        print("\n[Main] Service Request:")
        print(service_request)

        # =========================================================
        # BLOCK 31: Service Record Creation and Service Chain Update
        # =========================================================

        service_record = create_service_record(
            service_request=service_request,
        )

        add_to_pending_service_records(service_record)

        # if consensus_process(pending_service_records):
        #     add_block_to_service_chain(pending_service_records)

        # for mec_node in mec_nodes:
        #     if mec_node.node_id == 4:
        #         mec_node.is_faulty = True
        #         print("[Main] MEC 4 temporarily marked faulty before Service PBFT.")

        service_pbft_result = pbft_consensus_process(
            mec_nodes=mec_nodes,
            pending_records=pending_service_records,
            blockchain=service_chain,
        )

        if enable_service_retry_test and not service_pbft_result:
            print(
                "[Main] First Service PBFT attempt failed. Retrying service consensus..."
            )

            for mec_node in mec_nodes:
                if mec_node.node_id == 4:
                    mec_node.is_faulty = False
                    print("[Main] MEC 4 recovered before Service PBFT retry.")

            service_pbft_result = pbft_consensus_process(
                mec_nodes=mec_nodes,
                pending_records=pending_service_records,
                blockchain=service_chain,
            )

        print("\n[Main] Service PBFT Result:")
        print(service_pbft_result)

        if not service_pbft_result:
            print("[Main] Service PBFT failed. Service record was not stored.")
            print(
                "[Main] Task execution is cancelled because service consensus failed."
            )
            return

        debug_print("\n[Main] Service Chain:")
        debug_print(service_chain)

        # =========================================================
        # BLOCK 32: Server Vehicle Execution and Capacity Update
        # =========================================================

        server_execution_result = execute_task_on_server_vehicle(
            service_record=service_record,
        )

        print("\n[Main] Server Vehicle Execution Result:")
        print(server_execution_result)

        #        updated_capacity_records = update_capacity_after_execution(
        #            provider_id=service_record.provider,
        #            used_duration=service_record.duration,
        #        )

        #        print("\n[Main] Updated Capacity Records:")
        #        print(updated_capacity_records)

        # =========================================================
        # BLOCK 34: Blockchain-Based Capacity Update After Execution
        # =========================================================

        new_capacity_record = create_updated_capacity_record_after_execution(
            provider_id=service_record.provider,
            used_duration=service_record.duration,
        )

        if new_capacity_record is not None:

            add_to_pending_capacity_records(new_capacity_record)

            # if consensus_process(pending_capacity_records):
            #     add_block_to_capacity_chain(pending_capacity_records)
            capacity_update_pbft_result = pbft_consensus_process(
                mec_nodes=mec_nodes,
                pending_records=pending_capacity_records,
                blockchain=capacity_chain,
            )

            print("\n[Main] Capacity Update PBFT Result:")
            print(capacity_update_pbft_result)

        print("\n[Main] Capacity Chain After Execution:")
        print(capacity_chain)

        # =========================================================
        # BLOCK 40: Store Task Result for Later Analysis
        # =========================================================

        task_result = {
            "task_id": task_counter,
            "selected_provider": selected_candidate.provider,
            "mec_latency": mec_latency,
            "v2v_latency": v2v_latency,
            "utility": utility,
            "cost": cost,
            "payoff": payoff,
            "converged_probability": converged_probability,
            "offloading_decision": offloading_decision,
        }

        # simulation_results.append(task_result)

        print("\n[Main] Task Result:")
        print(task_result)

        return task_result

        # # =========================================================
        # # BLOCK 41: Simulation Summary
        # # =========================================================

        # print("\n==============================")
        # print("SIMULATION SUMMARY")
        # print("==============================")

        # print(f"Number of completed tasks: {len(simulation_results)}")

        # if len(simulation_results) > 0:

        #     average_latency = sum(
        #         result["mec_latency"] for result in simulation_results
        #     ) / len(simulation_results)

        #     average_payoff = sum(
        #         result["payoff"] for result in simulation_results
        #     ) / len(simulation_results)

        #     print(f"Average MEC Latency: {average_latency:.6f}")
        #     print(f"Average Payoff: {average_payoff:.6f}")


# =========================================================
# BLOCK 21: Main Flow Initialization
# =========================================================


def main():

    ENABLE_SERVICE_RETRY_TEST = False

    TEST_MODE = True
    DEBUG = TEST_MODE

    def debug_print(message):
        if DEBUG:
            print(message)

    print("Simulation started.")

    # =========================================================
    # BLOCK 22: Simulation Entities and Initial Parameters
    # =========================================================

    server_vehicles = [
        ServerVehicle(
            vehicle_id=1,
            resource=1e9,
            trajectory=120.0,
            price_init=0.6,
            period=60.0,
            quality=0.9,
        ),
        ServerVehicle(
            vehicle_id=2,
            resource=1.5e9,
            trajectory=130.0,
            price_init=0.7,
            period=50.0,
            quality=0.85,
        ),
    ]

    user_vehicle = UserVehicle(
        vehicle_id=101,
        trajectory=118.0,
    )

    # print(f"\n========== TASK {task_counter} ==========")

    # =========================================================
    # BLOCK 44: MEC Nodes Initialization for PoS Consensus
    # =========================================================

    mec_nodes = [
        MECNode(
            node_id=1,
            redundant_resource=2.0e9,
            is_faulty=1 in FAULTY_MEC_IDS,
        ),
        MECNode(
            node_id=2,
            redundant_resource=3.0e9,
            is_faulty=2 in FAULTY_MEC_IDS,
        ),
        MECNode(
            node_id=3,
            redundant_resource=1.5e9,
            is_faulty=3 in FAULTY_MEC_IDS,
        ),
        MECNode(
            node_id=4,
            redundant_resource=1.2e9,
            is_faulty=4 in FAULTY_MEC_IDS,
        ),
    ]

    print("\n===== FIGURE 6 TEST =====")
    vehicle_counts, figure6_results = run_figure6_test()
    plot_figure6_results(
        vehicle_counts=vehicle_counts,
        figure6_results=figure6_results,
    )

    print("\n===== FIGURE 7 TEST =====")
    value_factors, figure7_results = run_figure7_test()
    plot_figure7_results(
        value_factors=value_factors,
        figure7_results=figure7_results,
    )

    print("\n===== FIGURE 8 TEST =====")
    value_factors, figure8_results = run_figure8_test()
    plot_figure8_results(
        value_factors=value_factors,
        figure8_results=figure8_results,
    )

    return

    for task_counter in range(1, NUM_TASKS + 1):
        task_result = run_single_task(
            task_counter=task_counter,
            server_vehicles=server_vehicles,
            user_vehicle=user_vehicle,
            mec_nodes=mec_nodes,
            debug_print=debug_print,
            enable_service_retry_test=ENABLE_SERVICE_RETRY_TEST,
        )

        if task_result is not None:
            simulation_results.append(task_result)

    print("\n==============================")
    print("SIMULATION SUMMARY")
    print("==============================")

    print(f"Number of completed tasks: {len(simulation_results)}")

    if len(simulation_results) > 0:
        average_latency = sum(
            result["mec_latency"] for result in simulation_results
        ) / len(simulation_results)

        average_payoff = sum(result["payoff"] for result in simulation_results) / len(
            simulation_results
        )

        print(f"Average MEC Latency: {average_latency:.6f}")
        print(f"Average Payoff: {average_payoff:.6f}")

    # =========================================================
    # BLOCK 45: Proof of Service Leader Selection
    # =========================================================

    # leader_node = select_leader_by_pos(
    #     mec_nodes=mec_nodes,
    # )

    # print("\n[Main] Proof of Service Leader Selection Result:")
    # print(f"Selected Leader MEC ID: {leader_node.node_id}")
    # print(f"Leader Redundant Resource: {leader_node.redundant_resource}")

    # =========================================================
    # BLOCK 46: PBFT Pre-Prepare Phase
    # =========================================================

    # proposed_capacity_block = pre_prepare_phase(
    #     leader_node=leader_node,
    #     mec_nodes=mec_nodes,
    #     pending_records=pending_capacity_records,
    #     next_block_id=len(capacity_chain) + 1,
    # )

    # print("\n[Main] Proposed Capacity Block:")
    # print(proposed_capacity_block)

    # =========================================================
    # BLOCK 47: PBFT Prepare Phase
    # =========================================================

    # prepare_messages = prepare_phase(
    #     mec_nodes=mec_nodes,
    #     proposed_block=proposed_capacity_block,
    #     selected_leader=leader_node,
    # )

    # print("\n[Main] Prepare Messages:")
    # print(prepare_messages)

    # =========================================================
    # BLOCK 48: PBFT Commit Phase
    # =========================================================

    # commit_messages = commit_phase(
    #     mec_nodes=mec_nodes,
    #     proposed_block=proposed_capacity_block,
    #     prepare_messages=prepare_messages,
    # )

    # print("\n[Main] Commit Messages:")
    # print(commit_messages)

    # =========================================================
    # BLOCK 49: PBFT Reply Phase
    # =========================================================

    # consensus_result = reply_phase(
    #     mec_nodes=mec_nodes,
    #     proposed_block=proposed_capacity_block,
    #     commit_messages=commit_messages,
    # )

    # print("\n[Main] PBFT Consensus Result:")
    # print(consensus_result)

    # store_result = store_phase(
    #     proposed_block=proposed_capacity_block,
    #     consensus_result=consensus_result,
    #     blockchain=capacity_chain,
    #     pending_records=pending_capacity_records,
    # )

    # print("\n[Main] Store Result:")
    # print(store_result)

    # debug_print("\n[Main] Capacity Chain:")
    # debug_print(capacity_chain)


if __name__ == "__main__":
    main()
