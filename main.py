from mec_side import (
    MECNode,
    ServerVehicle,
    UserVehicle,
    add_block_to_capacity_chain,
    add_block_to_service_chain,
    add_to_pending_capacity_records,
    add_to_pending_service_records,
    broadcast_capacity_info,
    capacity_chain,
    commit_phase,
    consensus_process,
    create_capacity_record,
    create_service_record,
    create_updated_capacity_record_after_execution,
    execute_task_on_server_vehicle,
    get_confirmed_service_records,
    pending_capacity_records,
    pending_service_records,
    pre_prepare_phase,
    prepare_phase,
    reply_phase,
    select_leader_by_pos,
    service_chain,
    store_phase,
    verify_certificate,
)
from user_side import (
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

NUM_TASKS = 100

simulation_results = []

# =========================================================
# BLOCK 21: Main Flow Initialization
# =========================================================


def main():

    print("Simulation started.")

    task_counter = 1

    # =========================================================
    # BLOCK 22: Simulation Entities and Initial Parameters
    # =========================================================

    server_vehicles = [
        ServerVehicle(
            vehicle_id=1,
            resource=2e9,
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

    print("[Main] Server vehicles initialized.")
    print("[Main] User vehicle initialized.")

    print(f"\n========== TASK {task_counter} ==========")

    # =========================================================
    # BLOCK 44: MEC Nodes Initialization for PoS Consensus
    # =========================================================

    mec_nodes = [
        MECNode(
            node_id=1,
            redundant_resource=2.0e9,
        ),
        MECNode(
            node_id=2,
            redundant_resource=3.0e9,
        ),
        MECNode(
            node_id=3,
            redundant_resource=1.5e9,
        ),
    ]

    print("[Main] MEC nodes initialized.")

    # =========================================================
    # BLOCK 23: Capacity Record Creation and Capacity Chain Update
    # =========================================================

    for server_vehicle in server_vehicles:

        if verify_certificate(server_vehicle):

            capacity_record = create_capacity_record(
                server_vehicle=server_vehicle,
            )

            add_to_pending_capacity_records(capacity_record)

    # =========================================================
    # BLOCK 45: Proof of Service Leader Selection
    # =========================================================

    leader_node = select_leader_by_pos(
        mec_nodes=mec_nodes,
    )

    print("\n[Main] Proof of Service Leader Selection Result:")
    print(f"Selected Leader MEC ID: {leader_node.node_id}")
    print(f"Leader Redundant Resource: {leader_node.redundant_resource}")

    # =========================================================
    # BLOCK 46: PBFT Pre-Prepare Phase
    # =========================================================

    proposed_capacity_block = pre_prepare_phase(
        leader_node=leader_node,
        mec_nodes=mec_nodes,
        pending_records=pending_capacity_records,
        next_block_id=len(capacity_chain) + 1,
    )

    print("\n[Main] Proposed Capacity Block:")
    print(proposed_capacity_block)

    # =========================================================
    # BLOCK 47: PBFT Prepare Phase
    # =========================================================

    prepare_messages = prepare_phase(
        mec_nodes=mec_nodes,
        proposed_block=proposed_capacity_block,
        selected_leader=leader_node,
    )

    print("\n[Main] Prepare Messages:")
    print(prepare_messages)

    # =========================================================
    # BLOCK 48: PBFT Commit Phase
    # =========================================================

    commit_messages = commit_phase(
        mec_nodes=mec_nodes,
        proposed_block=proposed_capacity_block,
        prepare_messages=prepare_messages,
    )

    print("\n[Main] Commit Messages:")
    print(commit_messages)

    # =========================================================
    # BLOCK 49: PBFT Reply Phase
    # =========================================================

    consensus_result = reply_phase(
        mec_nodes=mec_nodes,
        proposed_block=proposed_capacity_block,
        commit_messages=commit_messages,
    )

    print("\n[Main] PBFT Consensus Result:")
    print(consensus_result)

    store_result = store_phase(
        proposed_block=proposed_capacity_block,
        consensus_result=consensus_result,
        blockchain=capacity_chain,
        pending_records=pending_capacity_records,
    )

    print("\n[Main] Store Result:")
    print(store_result)

    print("\n[Main] Capacity Chain:")
    print(capacity_chain)

    # =========================================================
    # BLOCK 24: Broadcast Capacity Information to User Vehicle
    # =========================================================

    broadcasted_capacity_info = broadcast_capacity_info()

    received_capacity_info = receive_capacity_info(broadcasted_capacity_info)

    print("\n[Main] Received Capacity Information:")
    print(received_capacity_info)

    # =========================================================
    # BLOCK 25: Select Available Candidate Server Vehicle
    # =========================================================

    confirmed_service_records = get_confirmed_service_records()

    selected_candidate = select_available_candidate(
        user_vehicle=user_vehicle,
        capacity_records=received_capacity_info,
        service_records=confirmed_service_records,
    )

    print("\n[Main] Selected Candidate:")
    print(selected_candidate)

    # =========================================================
    # BLOCK 26: Data Rate and Latency Calculation
    # =========================================================

    bandwidth = 10e6
    transmit_power = 0.5
    path_loss_exponent = 3.0
    channel_gain = 1.0
    noise_power = 1e-9

    input_size = 1e6
    complexity = 240
    mec_cpu_frequency = 10e9

    beta_uplink = 1.0
    beta_downlink = 0.2
    beta_request = 1.0
    beta_result = 0.2

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
    value_factor = 0.3

    arrival_rates = [0.4, 0.5, 0.3]
    initial_probabilities = [0.6, 0.5, 0.4]
    current_vehicle_index = 0

    probability_mec = initial_probabilities[current_vehicle_index]

    price_mec = 1.0
    price_ratio = selected_candidate.price / price_mec

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

        if consensus_process(pending_service_records):
            add_block_to_service_chain(pending_service_records)

        print("\n[Main] Service Chain:")
        print(service_chain)

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

            if consensus_process(pending_capacity_records):
                add_block_to_capacity_chain(pending_capacity_records)

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

        simulation_results.append(task_result)

        print("\n[Main] Task Result:")
        print(task_result)

        # =========================================================
        # BLOCK 41: Simulation Summary
        # =========================================================

        print("\n==============================")
        print("SIMULATION SUMMARY")
        print("==============================")

        print(f"Number of completed tasks: {len(simulation_results)}")

        if len(simulation_results) > 0:

            average_latency = sum(
                result["mec_latency"] for result in simulation_results
            ) / len(simulation_results)

            average_payoff = sum(
                result["payoff"] for result in simulation_results
            ) / len(simulation_results)

            print(f"Average MEC Latency: {average_latency:.6f}")
            print(f"Average Payoff: {average_payoff:.6f}")


if __name__ == "__main__":
    main()
