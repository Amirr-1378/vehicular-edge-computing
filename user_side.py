# =========================================================
# BLOCK 5: Receive Capacity Info and Select Candidate Vehicle
# =========================================================

from mec_side import CapacityRecord, UserVehicle


def receive_capacity_info(
    capacity_records: list[CapacityRecord],
) -> list[CapacityRecord]:

    if len(capacity_records) == 0:
        print("[User Side] No capacity information received.")
        return []

    print(f"[User Side] Received {len(capacity_records)} capacity record(s).")

    return capacity_records


def select_candidate_server_vehicle(
    user_vehicle: UserVehicle,
    capacity_records: list[CapacityRecord],
) -> CapacityRecord | None:

    if len(capacity_records) == 0:
        print("[User Side] No candidate server vehicle available.")
        return None

    selected_record = min(
        capacity_records,
        key=lambda record: abs(user_vehicle.trajectory - record.trajectory),
    )

    print(
        f"[User Side] Selected Provider {selected_record.provider} "
        f"with trajectory {selected_record.trajectory}."
    )

    return selected_record


# =========================================================
# BLOCK 6: Check Candidate Occupancy in Service Chain
# =========================================================

from mec_side import ServiceRecord


def is_candidate_occupied(
    candidate: CapacityRecord,
    service_records: list[ServiceRecord],
) -> bool:

    for service in service_records:

        if service.provider == candidate.provider:
            print(
                f"[User Side] Provider {candidate.provider} " f"is currently occupied."
            )
            return True

    print(f"[User Side] Provider {candidate.provider} " f"is available.")

    return False


# =========================================================
# BLOCK 7: Select Available Candidate or Fallback to MEC
# =========================================================


def select_available_candidate(
    user_vehicle: UserVehicle,
    capacity_records: list[CapacityRecord],
    service_records: list[ServiceRecord],
) -> CapacityRecord | None:

    if len(capacity_records) == 0:
        print("[User Side] No server vehicle candidates found.")
        return None

    sorted_candidates = sorted(
        capacity_records,
        key=lambda record: abs(user_vehicle.trajectory - record.trajectory),
    )

    for candidate in sorted_candidates:

        if not is_candidate_occupied(candidate, service_records):
            print(
                f"[User Side] Available candidate selected: "
                f"Provider {candidate.provider}"
            )
            return candidate

        print(
            f"[User Side] Provider {candidate.provider} skipped "
            f"because it is occupied."
        )

    print("[User Side] All candidates are occupied. Fallback to MEC.")

    return None


# =========================================================
# BLOCK 8: Communication Model and Data Rate Calculation
# =========================================================

import math


def calculate_data_rate(
    bandwidth: float,
    transmit_power: float,
    distance: float,
    path_loss_exponent: float,
    channel_gain: float,
    noise_power: float,
) -> float:

    signal_to_noise_ratio = (
        transmit_power * (distance ** (-path_loss_exponent)) * (abs(channel_gain) ** 2)
    ) / noise_power

    data_rate = bandwidth * math.log2(1 + signal_to_noise_ratio)

    return data_rate


# =========================================================
# BLOCK 9: Computation Offloading Latency Calculation
# =========================================================


def calculate_mec_latency(
    input_size: float,
    complexity: float,
    mec_cpu_frequency: float,
    uplink_rate: float,
    downlink_rate: float,
    beta_uplink: float,
    beta_downlink: float,
) -> float:

    uplink_time = (beta_uplink * input_size) / uplink_rate

    execution_time = (complexity * input_size) / mec_cpu_frequency

    downlink_time = (beta_downlink * input_size) / downlink_rate

    total_latency = uplink_time + execution_time + downlink_time

    return total_latency


def calculate_v2v_latency(
    input_size: float,
    complexity: float,
    server_vehicle_cpu_frequency: float,
    request_rate: float,
    result_rate: float,
    beta_request: float,
    beta_result: float,
) -> float:

    request_time = (beta_request * input_size) / request_rate

    execution_time = (complexity * input_size) / server_vehicle_cpu_frequency

    result_time = (beta_result * input_size) / result_rate

    total_latency = request_time + execution_time + result_time

    return total_latency


# =========================================================
# BLOCK 10: Value Function and Utility Calculation
# =========================================================


def calculate_value(
    latency: float,
    deadline: float,
    value_factor: float,
) -> float:

    if latency > deadline:
        return 0.0

    value = (
        2 * deadline * (latency + value_factor * deadline)
        - (latency + value_factor * deadline) ** 2
    )

    return value


def calculate_max_value(
    deadline: float,
    value_factor: float,
) -> float:

    max_latency_point = (1 - value_factor) * deadline

    return calculate_value(
        latency=max_latency_point,
        deadline=deadline,
        value_factor=value_factor,
    )


def calculate_utility(
    probability_mec: float,
    mec_latency: float,
    v2v_latency: float,
    deadline: float,
    value_factor: float,
    service_quality: float,
) -> float:

    max_value = calculate_max_value(
        deadline=deadline,
        value_factor=value_factor,
    )

    mec_value = calculate_value(
        latency=mec_latency,
        deadline=deadline,
        value_factor=value_factor,
    )

    v2v_value = calculate_value(
        latency=v2v_latency,
        deadline=deadline,
        value_factor=value_factor,
    )

    utility = probability_mec * (mec_value / max_value) + (
        1 - probability_mec
    ) * service_quality * (v2v_value / max_value)

    return utility


# =========================================================
# BLOCK 11: Cost Function and Payoff Calculation
# =========================================================


def calculate_cost(
    probability_mec: float,
    arrival_rates: list[float],
    probabilities: list[float],
    price_ratio: float,
    current_vehicle_index: int,
) -> float:

    competition_term = 1.0

    for index, (arrival_rate, probability) in enumerate(
        zip(arrival_rates, probabilities)
    ):

        if index == current_vehicle_index:
            continue

        competition_term *= 1 - arrival_rate * probability

    competition_cost = (probability_mec**2) * (1 - competition_term)

    v2v_cost = (1 - probability_mec) * price_ratio

    total_cost = competition_cost + v2v_cost

    return total_cost


def calculate_payoff(
    utility: float,
    cost: float,
) -> float:

    payoff = utility - cost

    return payoff


# =========================================================
# BLOCK 12: Best Response Update for Offloading Probability
# =========================================================


def clamp_probability(value: float) -> float:

    return max(0.0, min(1.0, value))


def calculate_best_response(
    mec_latency: float,
    v2v_latency: float,
    deadline: float,
    value_factor: float,
    service_quality: float,
    price_ratio: float,
    arrival_rates: list[float],
    probabilities: list[float],
    current_vehicle_index: int,
) -> float:

    max_value = calculate_max_value(
        deadline=deadline,
        value_factor=value_factor,
    )

    # mec_value = (
    #     calculate_value(
    #         latency=mec_latency,
    #         deadline=deadline,
    #         value_factor=value_factor,
    #     )
    #     / max_value
    # )

    # v2v_value = (
    #     service_quality
    #     * calculate_value(
    #         latency=v2v_latency,
    #         deadline=deadline,
    #         value_factor=value_factor,
    #     )
    # ) / max_value

    mec_score = (
        calculate_value(
            latency=mec_latency,
            deadline=deadline,
            value_factor=value_factor,
        )
        / max_value
    )

    v2v_score = (
        service_quality
        * calculate_value(
            latency=v2v_latency,
            deadline=deadline,
            value_factor=value_factor,
        )
        / max_value
    )

    competition_term = 1.0

    for index, (arrival_rate, probability) in enumerate(
        zip(arrival_rates, probabilities)
    ):

        if index == current_vehicle_index:
            continue

        competition_term *= 1 - arrival_rate * probability

    denominator = 2 * (1 - competition_term + 1e-6)

    if denominator == 0:
        return 1.0

    # if current_vehicle_index == 0:
    #     print(
    #         f"mec={mec_value:.4f}, "
    #         f"v2v={v2v_value:.4f}, "
    #         f"competition={competition_term:.4f}, "
    #         f"denominator={denominator:.4f}"
    #     )

    # best_response = (mec_value - v2v_value + price_ratio) / denominator
    # mec_gain = 1 / mec_latency
    # v2v_gain = service_quality / v2v_latency

    # numerator = mec_score - price_ratio * v2v_score  ----->   this line was incorrect according to the base paper
    numerator = mec_score - v2v_score + price_ratio
    best_response = numerator / denominator
    return clamp_probability(best_response)


# =========================================================
# BLOCK 13: Iterative Best Response and Convergence Check
# =========================================================


def run_best_response_until_convergence(
    mec_latency: float,
    v2v_latency: float,
    deadline: float,
    value_factor: float,
    service_quality: float,
    price_ratio: float,
    arrival_rates: list[float],
    initial_probabilities: list[float],
    current_vehicle_index: int,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
) -> float:

    probabilities = initial_probabilities.copy()

    for iteration in range(max_iterations):

        old_probability = probabilities[current_vehicle_index]

        new_probability = calculate_best_response(
            mec_latency=mec_latency,
            v2v_latency=v2v_latency,
            deadline=deadline,
            value_factor=value_factor,
            service_quality=service_quality,
            price_ratio=price_ratio,
            arrival_rates=arrival_rates,
            probabilities=probabilities,
            current_vehicle_index=current_vehicle_index,
        )

        probabilities[current_vehicle_index] = new_probability

        difference = abs(new_probability - old_probability)

        print(
            f"[Best Response] Iteration {iteration + 1}: "
            f"old p = {old_probability:.6f}, "
            f"new p = {new_probability:.6f}, "
            f"difference = {difference:.6f}"
        )

        if difference < tolerance:
            print("[Best Response] Converged.")
            return new_probability

    print("[Best Response] Maximum iterations reached.")

    return probabilities[current_vehicle_index]


# =========================================================
# BLOCK 14: Final Offloading Decision Based on Probability
# =========================================================

import random


def make_offloading_decision(probability_mec: float) -> str:

    random_number = random.random()

    print(f"[Decision] Random number: {random_number:.6f}")
    print(f"[Decision] MEC probability p_i: {probability_mec:.6f}")

    if random_number > probability_mec:
        print("[Decision] V2V offloading selected.")
        return "V2V"

    print("[Decision] MEC offloading selected.")
    return "MEC"


# =========================================================
# BLOCK 15: Send Service Request or Execute on MEC
# =========================================================


def send_service_request_to_mec(
    user_vehicle: UserVehicle,
    candidate: CapacityRecord,
    estimated_duration: float,
) -> dict:

    service_request = {
        "requester": user_vehicle.vehicle_id,
        "provider": candidate.provider,
        "estimated_duration": estimated_duration,
    }

    print(
        f"[User Side] Service request sent to MEC: "
        f"Requester {user_vehicle.vehicle_id} -> Provider {candidate.provider}"
    )

    return service_request


def execute_task_on_mec(
    user_vehicle: UserVehicle,
    mec_latency: float,
) -> dict:

    execution_result = {
        "executor": "MEC",
        "requester": user_vehicle.vehicle_id,
        "latency": mec_latency,
        "status": "completed",
    }

    print(
        f"[MEC Execution] Task of User {user_vehicle.vehicle_id} "
        f"executed on MEC with latency {mec_latency:.6f}."
    )

    return execution_result


# =========================================================
# BLOCK 20: Service Quality Evaluation Placeholder
# =========================================================


def evaluate_service_quality_placeholder(
    current_quality: float,
) -> float:

    print(
        "[Evaluation] Service quality update is skipped "
        "in the current simplified version."
    )

    return current_quality
