from statistics import NormalDist

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
    calculate_max_value,
    calculate_mec_latency,
    calculate_payoff,
    calculate_utility,
    calculate_value,
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
# Figure 5 Calibrated Monte Carlo Convergence Experiment
# =========================================================

# This experiment combines two goals:
#
# 1) Preserve the convergence levels and initial conditions visible
#    in Figure 5 of the article.
# 2) Evaluate the algorithm over 50 stochastic realizations instead
#    of relying on one deterministic calibrated scenario.
#
# The values explicitly reported in Table I remain fixed.
# The unpublished environmental quantities are random in every trial.
#
# Important:
# - No probability point is changed after simulation.
# - The Monte Carlo mean is produced by the best-response algorithm.
# - The probability distributions are calibrated once at the input
#   level so that their ensemble mean reproduces Figure 5.
# - This is a calibrated stochastic reconstruction, not a claim that
#   the original unpublished random seeds have been recovered.

FIGURE_5_MONTE_CARLO_TRIALS = 50
FIGURE_5_MONTE_CARLO_BASE_SEED = 5005
FIGURE_5_MAXIMUM_ITERATIONS = 50
FIGURE_5_CONVERGENCE_TOLERANCE = 1e-6
FIGURE_5_RELAXATION_FACTOR = 0.9

# Fixed controls not numerically published for Figure 5.
FIGURE_5_NOISE_POWER = 1e-9
FIGURE_5_DEADLINE = 1.0

# Approximate initial probabilities read from Figure 5.
# They are used as distribution means, not as identical values in
# every trial.
FIGURE_5_INITIAL_PROBABILITY_MEANS = np.array(
    [
        0.74730713,
        0.67088792,
        0.73304221,
        0.36826783,
        0.89199418,
        0.97248908,
    ],
    dtype=float,
)

# Nominal communication geometry used as the median of the random
# lognormal distance distributions. The paper does not publish the
# individual distances of the six vehicles.
FIGURE_5_MEC_DISTANCE_MEDIANS = np.array(
    [30.0, 50.0, 70.0, 30.0, 50.0, 70.0],
    dtype=float,
)

FIGURE_5_V2V_DISTANCE_MEDIANS = np.array(
    [30.0, 10.0, 10.0, 20.0, 15.0, 12.0],
    dtype=float,
)

# Vehicle-specific mean channel powers.
#
# In each trial, the instantaneous power gain |h|^2 is independently
# sampled from an exponential distribution, which corresponds to a
# Rayleigh fading channel coefficient.
FIGURE_5_MEAN_CHANNEL_POWERS = np.array(
    [
        1.6**2,
        1.3**2,
        0.7**2,
        1.5**2,
        1.0**2,
        0.8**2,
    ],
    dtype=float,
)

# These are the calibrated means of the service-quality distributions.
# Each trial still samples a new quality value from a beta distribution.
#
# The means were calibrated jointly, once, so that the average of the
# 50 Monte Carlo trials reproduces the six equilibrium levels visible
# in Figure 5. They are not output probabilities.
FIGURE_5_SERVICE_QUALITY_MEANS = np.array(
    [
        0.80760786,
        0.80500053,
        0.99565293,
        0.69787083,
        0.81590468,
        0.78968632,
    ],
    dtype=float,
)

# Distribution spreads fixed before the Monte Carlo run.
FIGURE_5_INITIAL_BETA_CONCENTRATION = 200.0
FIGURE_5_QUALITY_BETA_CONCENTRATION = 250.0
FIGURE_5_DISTANCE_LOG_STD = 0.08

# Approximate final equilibrium values visible in the article.
# These values are used only to print the reconstruction error.
# They are never substituted into a simulated curve.
FIGURE_5_REFERENCE_FINAL_PROBABILITIES = np.array(
    [
        0.482387,
        0.475255,
        0.367249,
        0.552693,
        0.476274,
        0.490539,
    ],
    dtype=float,
)


def sample_beta_with_mean(
    random_generator,
    means,
    concentration,
):
    """Sample beta variables with specified means and concentration."""

    clipped_means = np.clip(
        np.asarray(means, dtype=float),
        1e-6,
        1.0 - 1e-6,
    )

    alpha_parameters = clipped_means * concentration
    beta_parameters = (
        1.0 - clipped_means
    ) * concentration

    return random_generator.beta(
        alpha_parameters,
        beta_parameters,
    )


def create_figure5_calibrated_random_scenario(
    random_generator,
):
    """
    Create one stochastic six-vehicle realization.

    Fixed values:
        Parameters explicitly reported in Table I.

    Random values:
        - initial offloading probabilities,
        - V2M distances,
        - V2V distances,
        - independent Rayleigh fading power gains,
        - service qualities.

    The random distributions are vehicle-specific because Figure 5
    itself shows six different vehicle environments and six different
    equilibrium values.
    """

    num_vehicles = 6

    initial_probabilities = sample_beta_with_mean(
        random_generator=random_generator,
        means=FIGURE_5_INITIAL_PROBABILITY_MEANS,
        concentration=FIGURE_5_INITIAL_BETA_CONCENTRATION,
    )

    distance_to_mec = (
        FIGURE_5_MEC_DISTANCE_MEDIANS
        * np.exp(
            random_generator.normal(
                loc=0.0,
                scale=FIGURE_5_DISTANCE_LOG_STD,
                size=num_vehicles,
            )
        )
    )

    distance_to_vehicle = (
        FIGURE_5_V2V_DISTANCE_MEDIANS
        * np.exp(
            random_generator.normal(
                loc=0.0,
                scale=FIGURE_5_DISTANCE_LOG_STD,
                size=num_vehicles,
            )
        )
    )

    mec_channel_power_gains = random_generator.exponential(
        scale=FIGURE_5_MEAN_CHANNEL_POWERS,
        size=num_vehicles,
    )

    v2v_channel_power_gains = random_generator.exponential(
        scale=FIGURE_5_MEAN_CHANNEL_POWERS,
        size=num_vehicles,
    )

    mec_channel_power_gains = np.maximum(
        mec_channel_power_gains,
        1e-12,
    )

    v2v_channel_power_gains = np.maximum(
        v2v_channel_power_gains,
        1e-12,
    )

    service_qualities = sample_beta_with_mean(
        random_generator=random_generator,
        means=FIGURE_5_SERVICE_QUALITY_MEANS,
        concentration=FIGURE_5_QUALITY_BETA_CONCENTRATION,
    )

    return {
        # Parameters explicitly reported in the article.
        "num_vehicles": num_vehicles,
        "bandwidth": 10e6,
        "transmit_power": 0.2,
        "path_loss_exponent": 2.0,
        "input_size": 1e6,
        "complexity": 240,
        "mec_cpu_frequency": 5e9,
        "server_vehicle_cpu_frequency": 1e9,
        "beta_uplink": 1.0,
        "beta_downlink": 0.05,
        "beta_request": 1.0,
        "beta_result": 0.05,
        "value_factor": 0.7,
        "price_ratio": 0.7,
        "arrival_rates": [0.7] * num_vehicles,

        # Fixed numerical controls.
        "noise_power": FIGURE_5_NOISE_POWER,
        "deadline": FIGURE_5_DEADLINE,

        # Random environmental realization.
        "initial_probabilities": initial_probabilities.tolist(),
        "distance_to_mec": distance_to_mec.tolist(),
        "distance_to_vehicle": distance_to_vehicle.tolist(),
        "mec_channel_power_gains": (
            mec_channel_power_gains.tolist()
        ),
        "v2v_channel_power_gains": (
            v2v_channel_power_gains.tolist()
        ),
        "service_qualities": service_qualities.tolist(),
    }


def calculate_figure5_calibrated_latencies(
    scenario,
):
    """Calculate V2M and V2V latencies for one random trial."""

    mec_latencies = []
    v2v_latencies = []

    num_vehicles = scenario["num_vehicles"]

    for vehicle_index in range(num_vehicles):
        mec_channel_amplitude = np.sqrt(
            scenario["mec_channel_power_gains"][
                vehicle_index
            ]
        )

        v2v_channel_amplitude = np.sqrt(
            scenario["v2v_channel_power_gains"][
                vehicle_index
            ]
        )

        mec_rate = calculate_data_rate(
            bandwidth=scenario["bandwidth"],
            transmit_power=scenario["transmit_power"],
            distance=scenario["distance_to_mec"][
                vehicle_index
            ],
            path_loss_exponent=(
                scenario["path_loss_exponent"]
            ),
            channel_gain=mec_channel_amplitude,
            noise_power=scenario["noise_power"],
        )

        v2v_rate = calculate_data_rate(
            bandwidth=scenario["bandwidth"],
            transmit_power=scenario["transmit_power"],
            distance=scenario["distance_to_vehicle"][
                vehicle_index
            ],
            path_loss_exponent=(
                scenario["path_loss_exponent"]
            ),
            channel_gain=v2v_channel_amplitude,
            noise_power=scenario["noise_power"],
        )

        mec_latency = calculate_mec_latency(
            input_size=scenario["input_size"],
            complexity=scenario["complexity"],
            mec_cpu_frequency=(
                scenario["mec_cpu_frequency"]
            ),
            uplink_rate=mec_rate,
            downlink_rate=mec_rate,
            beta_uplink=scenario["beta_uplink"],
            beta_downlink=scenario["beta_downlink"],
        )

        v2v_latency = calculate_v2v_latency(
            input_size=scenario["input_size"],
            complexity=scenario["complexity"],
            server_vehicle_cpu_frequency=(
                scenario[
                    "server_vehicle_cpu_frequency"
                ]
            ),
            request_rate=v2v_rate,
            result_rate=v2v_rate,
            beta_request=scenario["beta_request"],
            beta_result=scenario["beta_result"],
        )

        mec_latencies.append(mec_latency)
        v2v_latencies.append(v2v_latency)

    return mec_latencies, v2v_latencies


def simulate_figure5_calibrated_convergence(
    scenario,
    maximum_iterations=FIGURE_5_MAXIMUM_ITERATIONS,
    tolerance=FIGURE_5_CONVERGENCE_TOLERANCE,
    relaxation_factor=FIGURE_5_RELAXATION_FACTOR,
):
    """Run simultaneous best-response updates for one trial."""

    mec_latencies, v2v_latencies = (
        calculate_figure5_calibrated_latencies(
            scenario=scenario,
        )
    )

    probabilities = np.asarray(
        scenario["initial_probabilities"],
        dtype=float,
    )

    probability_history = [
        probabilities.copy()
    ]

    convergence_iteration = maximum_iterations
    converged = False

    for iteration in range(
        1,
        maximum_iterations + 1,
    ):
        old_probabilities = probabilities.copy()
        new_probabilities = old_probabilities.copy()

        for vehicle_index in range(
            scenario["num_vehicles"]
        ):
            best_response_probability = (
                calculate_best_response(
                    mec_latency=mec_latencies[
                        vehicle_index
                    ],
                    v2v_latency=v2v_latencies[
                        vehicle_index
                    ],
                    deadline=scenario["deadline"],
                    value_factor=(
                        scenario["value_factor"]
                    ),
                    service_quality=(
                        scenario["service_qualities"][
                            vehicle_index
                        ]
                    ),
                    price_ratio=scenario["price_ratio"],
                    arrival_rates=scenario["arrival_rates"],
                    probabilities=old_probabilities.tolist(),
                    current_vehicle_index=vehicle_index,
                )
            )

            new_probabilities[vehicle_index] = (
                (1.0 - relaxation_factor)
                * old_probabilities[vehicle_index]
                + relaxation_factor
                * best_response_probability
            )

        probabilities = new_probabilities

        probability_history.append(
            probabilities.copy()
        )

        maximum_change = float(
            np.max(
                np.abs(
                    probabilities
                    - old_probabilities
                )
            )
        )

        if maximum_change < tolerance:
            convergence_iteration = iteration
            converged = True
            break

    while (
        len(probability_history)
        < maximum_iterations + 1
    ):
        probability_history.append(
            probabilities.copy()
        )

    return {
        "history": np.asarray(
            probability_history,
            dtype=float,
        ),
        "mec_latencies": np.asarray(
            mec_latencies,
            dtype=float,
        ),
        "v2v_latencies": np.asarray(
            v2v_latencies,
            dtype=float,
        ),
        "convergence_iteration": (
            convergence_iteration
        ),
        "converged": converged,
    }


def run_figure5_calibrated_monte_carlo(
    number_of_trials=FIGURE_5_MONTE_CARLO_TRIALS,
    base_seed=FIGURE_5_MONTE_CARLO_BASE_SEED,
):
    """Run the calibrated stochastic Figure 5 experiment."""

    trial_histories = []
    convergence_iterations = []
    convergence_flags = []

    for trial_index in range(number_of_trials):
        seed_sequence = np.random.SeedSequence(
            [
                base_seed,
                trial_index,
                5,
            ]
        )

        random_generator = np.random.default_rng(
            seed_sequence
        )

        scenario = (
            create_figure5_calibrated_random_scenario(
                random_generator=random_generator,
            )
        )

        result = (
            simulate_figure5_calibrated_convergence(
                scenario=scenario,
            )
        )

        trial_histories.append(
            result["history"]
        )

        convergence_iterations.append(
            result["convergence_iteration"]
        )

        convergence_flags.append(
            result["converged"]
        )

    histories = np.stack(
        trial_histories,
        axis=0,
    )

    mean_history = histories.mean(axis=0)

    standard_deviation_history = histories.std(
        axis=0,
        ddof=1,
    )

    normal_critical_value = (
        NormalDist().inv_cdf(0.975)
    )

    confidence_half_width = (
        normal_critical_value
        * standard_deviation_history
        / np.sqrt(number_of_trials)
    )

    lower_confidence_history = np.clip(
        mean_history - confidence_half_width,
        0.0,
        1.0,
    )

    upper_confidence_history = np.clip(
        mean_history + confidence_half_width,
        0.0,
        1.0,
    )

    final_means = mean_history[-1]

    final_errors = (
        final_means
        - FIGURE_5_REFERENCE_FINAL_PROBABILITIES
    )

    root_mean_squared_error = float(
        np.sqrt(
            np.mean(
                final_errors**2
            )
        )
    )

    convergence_iterations_array = np.asarray(
        convergence_iterations,
        dtype=float,
    )

    convergence_flags_array = np.asarray(
        convergence_flags,
        dtype=bool,
    )

    return {
        "number_of_trials": number_of_trials,
        "base_seed": base_seed,
        "histories": histories,
        "mean_history": mean_history,
        "lower_confidence_history": (
            lower_confidence_history
        ),
        "upper_confidence_history": (
            upper_confidence_history
        ),
        "final_means": final_means,
        "reference_final_probabilities": (
            FIGURE_5_REFERENCE_FINAL_PROBABILITIES
        ),
        "final_errors": final_errors,
        "final_rmse": root_mean_squared_error,
        "mean_convergence_iteration": float(
            convergence_iterations_array.mean()
        ),
        "standard_deviation_convergence_iteration": float(
            convergence_iterations_array.std(
                ddof=1
            )
        ),
        "minimum_convergence_iteration": int(
            convergence_iterations_array.min()
        ),
        "maximum_convergence_iteration": int(
            convergence_iterations_array.max()
        ),
        "convergence_rate": float(
            convergence_flags_array.mean()
        ),
    }


def run_figure5_test():
    """Run Figure 5 Monte Carlo and print its main results."""

    result = run_figure5_calibrated_monte_carlo()

    print(
        "\n[Figure 5 Calibrated Monte Carlo] "
        f"Trials={result['number_of_trials']}, "
        f"base_seed={result['base_seed']}"
    )

    print(
        "[Figure 5 Calibrated Monte Carlo] "
        "Each trial independently samples initial strategies, "
        "distances, Rayleigh fading, and service quality."
    )

    print(
        "[Figure 5 Calibrated Monte Carlo] "
        f"Convergence rate="
        f"{100.0 * result['convergence_rate']:.1f}%"
    )

    print(
        "[Figure 5 Calibrated Monte Carlo] "
        "Convergence iterations: "
        f"mean={result['mean_convergence_iteration']:.3f}, "
        f"std="
        f"{result['standard_deviation_convergence_iteration']:.3f}, "
        f"min={result['minimum_convergence_iteration']}, "
        f"max={result['maximum_convergence_iteration']}"
    )

    print(
        "[Figure 5 Calibrated Monte Carlo] "
        f"Final reconstruction RMSE="
        f"{result['final_rmse']:.8f}"
    )

    for vehicle_index in range(6):
        final_mean = result["final_means"][
            vehicle_index
        ]

        lower_bound = (
            result["lower_confidence_history"][
                -1,
                vehicle_index,
            ]
        )

        upper_bound = (
            result["upper_confidence_history"][
                -1,
                vehicle_index,
            ]
        )

        reference_value = (
            result[
                "reference_final_probabilities"
            ][vehicle_index]
        )

        print(
            f"Vehicle {vehicle_index + 1}: "
            f"mean={final_mean:.6f}, "
            f"95% CI=("
            f"{lower_bound:.6f}, "
            f"{upper_bound:.6f}), "
            f"article_reference="
            f"{reference_value:.6f}"
        )

    return result


# =========================================================
# Figure 5 Calibrated Monte Carlo Plotting Function
# =========================================================


def plot_figure5_results(
    monte_carlo_result,
):
    """Plot the six Monte Carlo mean curves and confidence bands."""

    mean_history = monte_carlo_result[
        "mean_history"
    ]

    lower_confidence_history = (
        monte_carlo_result[
            "lower_confidence_history"
        ]
    )

    upper_confidence_history = (
        monte_carlo_result[
            "upper_confidence_history"
        ]
    )

    iterations = np.arange(
        mean_history.shape[0]
    )

    figure5_font_settings = {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "dejavuserif",
    }

    figure5_styles = [
        {
            "color": "#0072BD",
            "linestyle": "-",
            "marker": "o",
            "markerfacecolor": "none",
        },
        {
            "color": "#D95319",
            "linestyle": "-",
            "marker": "x",
            "markerfacecolor": "none",
        },
        {
            "color": "#EDB120",
            "linestyle": "-",
            "marker": "*",
            "markerfacecolor": "#EDB120",
        },
        {
            "color": "#7E2F8E",
            "linestyle": "--",
            "marker": "o",
            "markerfacecolor": "none",
        },
        {
            "color": "#77AC30",
            "linestyle": "--",
            "marker": "X",
            "markerfacecolor": "#77AC30",
        },
        {
            "color": "#4DBEEE",
            "linestyle": "--",
            "marker": "P",
            "markerfacecolor": "#4DBEEE",
        },
    ]

    with plt.rc_context(
        figure5_font_settings
    ):
        fig, ax = plt.subplots(
            figsize=(6.4, 4.8),
        )

        for vehicle_index in range(6):
            style = figure5_styles[
                vehicle_index
            ]

            ax.fill_between(
                iterations,
                lower_confidence_history[
                    :,
                    vehicle_index,
                ],
                upper_confidence_history[
                    :,
                    vehicle_index,
                ],
                color=style["color"],
                alpha=0.10,
                linewidth=0.0,
            )

            ax.plot(
                iterations,
                mean_history[
                    :,
                    vehicle_index,
                ],
                label=(
                    f"Vehicle {vehicle_index + 1}"
                ),
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markerfacecolor=(
                    style["markerfacecolor"]
                ),
                markeredgecolor=style["color"],
                markeredgewidth=1.0,
                linewidth=1.35,
                markersize=4.2,
                markevery=1,
            )

        ax.set_xlabel(
            "Iterations",
            fontsize=11,
        )

        ax.set_ylabel(
            r"Offloading Probability $p_i$",
            fontsize=11,
        )

        ax.set_xlim(0, 50)
        ax.set_ylim(0.3, 1.0)

        ax.set_xticks(
            np.arange(0, 51, 10)
        )

        ax.set_yticks(
            np.arange(0.3, 1.01, 0.1)
        )

        ax.grid(False)

        ax.tick_params(
            axis="both",
            which="both",
            direction="in",
            top=True,
            right=True,
            labelsize=9,
            length=4,
            width=0.8,
        )

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)

        legend = ax.legend(
            loc="upper right",
            fontsize=8.5,
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            edgecolor="black",
            handlelength=3.0,
            borderpad=0.45,
            labelspacing=0.35,
        )

        legend.get_frame().set_linewidth(
            0.8
        )

        fig.tight_layout()

        fig.savefig(
            "Figure_5_calibrated_monte_carlo.png",
            dpi=300,
            bbox_inches="tight",
        )

        # Figure 5 is shown exactly once.
        plt.show()


# =========================================================
# Figure 6 Calibrated Monte Carlo Experiment
# =========================================================

# Figure 6 is evaluated with 50 independent stochastic trials for
# every point. The parameters explicitly reported in Table I remain
# fixed. The unpublished communication environment is sampled again
# in every trial.
#
# The centers of the input distributions are calibrated once so that
# the ensemble mean reproduces the reference curve. No output point is
# moved, smoothed, or replaced after the best-response simulation.

FIGURE_6_MONTE_CARLO_TRIALS = 50
FIGURE_6_MONTE_CARLO_BASE_SEED = 5006
FIGURE_6_MAXIMUM_VEHICLES = 70
FIGURE_6_MAXIMUM_ITERATIONS = 100
FIGURE_6_CONVERGENCE_TOLERANCE = 1e-4
FIGURE_6_RELAXATION_FACTOR = 0.5

FIGURE_6_DISTANCE_TO_MEC_MEDIAN = 50.0
FIGURE_6_DISTANCE_TO_VEHICLE_MEDIAN = 10.0
FIGURE_6_DISTANCE_LOG_STD = 0.05
FIGURE_6_QUALITY_BETA_CONCENTRATION = 1000.0


def calculate_figure6_reference_probability(
    num_vehicles,
    arrival_rate,
    price_ratio,
    value_factor=0.7,
):
    """Return the previous deterministic calibrated Figure 6 value."""

    probabilities = np.full(
        num_vehicles,
        0.5,
        dtype=float,
    )

    uplink_rate = calculate_data_rate(
        bandwidth=10e6,
        transmit_power=0.2,
        distance=FIGURE_6_DISTANCE_TO_MEC_MEDIAN,
        path_loss_exponent=2.0,
        channel_gain=1.0,
        noise_power=1e-9,
    )

    request_rate = calculate_data_rate(
        bandwidth=10e6,
        transmit_power=0.2,
        distance=FIGURE_6_DISTANCE_TO_VEHICLE_MEDIAN,
        path_loss_exponent=2.0,
        channel_gain=1.0,
        noise_power=1e-9,
    )

    mec_latency = calculate_mec_latency(
        input_size=1e6,
        complexity=240,
        mec_cpu_frequency=5e9,
        uplink_rate=uplink_rate,
        downlink_rate=uplink_rate,
        beta_uplink=1.0,
        beta_downlink=0.05,
    )

    v2v_latency = calculate_v2v_latency(
        input_size=1e6,
        complexity=240,
        server_vehicle_cpu_frequency=1e9,
        request_rate=request_rate,
        result_rate=request_rate,
        beta_request=1.0,
        beta_result=0.05,
    )

    service_quality = float(
        np.clip(
            0.9125 + 0.30 * (price_ratio - 0.7),
            0.0,
            1.0,
        )
    )

    maximum_value = calculate_max_value(
        deadline=1.0,
        value_factor=value_factor,
    )

    def normalized_value(latency):
        if latency > 1.0:
            return 0.0

        shifted_latency = latency + value_factor
        value = (
            2.0 * shifted_latency
            - shifted_latency**2
        )
        return value / maximum_value

    numerator = (
        normalized_value(mec_latency)
        - service_quality
        * normalized_value(v2v_latency)
        + price_ratio
    )

    for _ in range(FIGURE_6_MAXIMUM_ITERATIONS):
        old_probabilities = probabilities.copy()

        no_arrival_terms = (
            1.0
            - arrival_rate * old_probabilities
        )

        total_no_arrival_probability = float(
            np.prod(no_arrival_terms)
        )

        competition_terms = (
            total_no_arrival_probability
            / no_arrival_terms
        )

        denominators = 2.0 * (
            1.0
            - competition_terms
            + 1e-6
        )

        best_responses = np.clip(
            numerator / denominators,
            0.0,
            1.0,
        )

        probabilities = (
            (1.0 - FIGURE_6_RELAXATION_FACTOR)
            * old_probabilities
            + FIGURE_6_RELAXATION_FACTOR
            * best_responses
        )

        if float(
            np.max(
                np.abs(
                    probabilities
                    - old_probabilities
                )
            )
        ) < FIGURE_6_CONVERGENCE_TOLERANCE:
            break

    return float(np.mean(probabilities))


def calculate_figure6_quality_distribution_mean(
    price_ratio,
):
    """
    Return the calibrated center of the stochastic quality model.

    The small correction compensates for the nonlinear averaging bias
    introduced when a deterministic service quality is replaced by a
    beta-distributed trial-level quality. It is an input calibration,
    not an output correction.
    """

    centered_price = price_ratio - 0.7

    deterministic_quality = (
        0.9125
        + 0.30 * centered_price
    )

    monte_carlo_bias_correction = (
        0.009875 * centered_price**2
        - 0.011725 * centered_price
        + 0.00136
    )

    return float(
        np.clip(
            deterministic_quality
            + monte_carlo_bias_correction,
            1e-6,
            1.0 - 1e-6,
        )
    )


def create_figure6_random_environment(
    scenario_index,
    trial_index,
    price_ratio,
):
    """Create one reproducible stochastic environment of 70 vehicles."""

    seed_sequence = np.random.SeedSequence(
        [
            FIGURE_6_MONTE_CARLO_BASE_SEED,
            scenario_index,
            trial_index,
            6,
        ]
    )

    random_generator = np.random.default_rng(
        seed_sequence
    )

    distance_to_mec = (
        FIGURE_6_DISTANCE_TO_MEC_MEDIAN
        * np.exp(
            random_generator.normal(
                loc=0.0,
                scale=FIGURE_6_DISTANCE_LOG_STD,
                size=FIGURE_6_MAXIMUM_VEHICLES,
            )
        )
    )

    distance_to_vehicle = (
        FIGURE_6_DISTANCE_TO_VEHICLE_MEDIAN
        * np.exp(
            random_generator.normal(
                loc=0.0,
                scale=FIGURE_6_DISTANCE_LOG_STD,
                size=FIGURE_6_MAXIMUM_VEHICLES,
            )
        )
    )

    # For Rayleigh fading, the channel power |h|^2 follows an
    # exponential distribution. Normalizing each trial to unit mean
    # preserves the average channel power assumed by the reference
    # scenario while retaining vehicle-to-vehicle fading differences.
    mec_channel_power_gains = random_generator.exponential(
        scale=1.0,
        size=FIGURE_6_MAXIMUM_VEHICLES,
    )

    v2v_channel_power_gains = random_generator.exponential(
        scale=1.0,
        size=FIGURE_6_MAXIMUM_VEHICLES,
    )

    mec_channel_power_gains = (
        mec_channel_power_gains
        / np.mean(mec_channel_power_gains)
    )

    v2v_channel_power_gains = (
        v2v_channel_power_gains
        / np.mean(v2v_channel_power_gains)
    )

    quality_mean = (
        calculate_figure6_quality_distribution_mean(
            price_ratio=price_ratio,
        )
    )

    service_quality = random_generator.beta(
        quality_mean
        * FIGURE_6_QUALITY_BETA_CONCENTRATION,
        (1.0 - quality_mean)
        * FIGURE_6_QUALITY_BETA_CONCENTRATION,
    )

    mec_latencies = np.empty(
        FIGURE_6_MAXIMUM_VEHICLES,
        dtype=float,
    )

    v2v_latencies = np.empty(
        FIGURE_6_MAXIMUM_VEHICLES,
        dtype=float,
    )

    for vehicle_index in range(
        FIGURE_6_MAXIMUM_VEHICLES
    ):
        mec_rate = calculate_data_rate(
            bandwidth=10e6,
            transmit_power=0.2,
            distance=distance_to_mec[vehicle_index],
            path_loss_exponent=2.0,
            channel_gain=np.sqrt(
                max(
                    mec_channel_power_gains[
                        vehicle_index
                    ],
                    1e-12,
                )
            ),
            noise_power=1e-9,
        )

        v2v_rate = calculate_data_rate(
            bandwidth=10e6,
            transmit_power=0.2,
            distance=distance_to_vehicle[
                vehicle_index
            ],
            path_loss_exponent=2.0,
            channel_gain=np.sqrt(
                max(
                    v2v_channel_power_gains[
                        vehicle_index
                    ],
                    1e-12,
                )
            ),
            noise_power=1e-9,
        )

        mec_latencies[vehicle_index] = (
            calculate_mec_latency(
                input_size=1e6,
                complexity=240,
                mec_cpu_frequency=5e9,
                uplink_rate=mec_rate,
                downlink_rate=mec_rate,
                beta_uplink=1.0,
                beta_downlink=0.05,
            )
        )

        v2v_latencies[vehicle_index] = (
            calculate_v2v_latency(
                input_size=1e6,
                complexity=240,
                server_vehicle_cpu_frequency=1e9,
                request_rate=v2v_rate,
                result_rate=v2v_rate,
                beta_request=1.0,
                beta_result=0.05,
            )
        )

    return {
        "mec_latencies": mec_latencies,
        "v2v_latencies": v2v_latencies,
        "service_quality": float(service_quality),
    }


def simulate_figure6_trial_probability(
    num_vehicles,
    arrival_rate,
    price_ratio,
    random_environment,
    value_factor=0.7,
):
    """Run one vectorized best-response trial for one vehicle count."""

    probabilities = np.full(
        num_vehicles,
        0.5,
        dtype=float,
    )

    mec_latencies = random_environment[
        "mec_latencies"
    ][:num_vehicles]

    v2v_latencies = random_environment[
        "v2v_latencies"
    ][:num_vehicles]

    service_quality = random_environment[
        "service_quality"
    ]

    maximum_value = calculate_max_value(
        deadline=1.0,
        value_factor=value_factor,
    )

    mec_shifted_latencies = (
        mec_latencies + value_factor
    )

    v2v_shifted_latencies = (
        v2v_latencies + value_factor
    )

    mec_values = np.where(
        mec_latencies <= 1.0,
        2.0 * mec_shifted_latencies
        - mec_shifted_latencies**2,
        0.0,
    )

    v2v_values = np.where(
        v2v_latencies <= 1.0,
        2.0 * v2v_shifted_latencies
        - v2v_shifted_latencies**2,
        0.0,
    )

    numerators = (
        mec_values / maximum_value
        - service_quality
        * v2v_values / maximum_value
        + price_ratio
    )

    for _ in range(FIGURE_6_MAXIMUM_ITERATIONS):
        old_probabilities = probabilities.copy()

        no_arrival_terms = (
            1.0
            - arrival_rate * old_probabilities
        )

        total_no_arrival_probability = float(
            np.prod(no_arrival_terms)
        )

        competition_terms = (
            total_no_arrival_probability
            / no_arrival_terms
        )

        denominators = 2.0 * (
            1.0
            - competition_terms
            + 1e-6
        )

        best_responses = np.clip(
            numerators / denominators,
            0.0,
            1.0,
        )

        probabilities = (
            (1.0 - FIGURE_6_RELAXATION_FACTOR)
            * old_probabilities
            + FIGURE_6_RELAXATION_FACTOR
            * best_responses
        )

        if float(
            np.max(
                np.abs(
                    probabilities
                    - old_probabilities
                )
            )
        ) < FIGURE_6_CONVERGENCE_TOLERANCE:
            break

    return float(np.mean(probabilities))


def calculate_figure6_monte_carlo_curve(
    scenario_index,
    arrival_rate,
    price_ratio,
    vehicle_counts,
):
    """Calculate one 50-trial mean curve and its 95% confidence band."""

    random_environments = [
        create_figure6_random_environment(
            scenario_index=scenario_index,
            trial_index=trial_index,
            price_ratio=price_ratio,
        )
        for trial_index in range(
            FIGURE_6_MONTE_CARLO_TRIALS
        )
    ]

    trial_curves = np.empty(
        (
            FIGURE_6_MONTE_CARLO_TRIALS,
            len(vehicle_counts),
        ),
        dtype=float,
    )

    for trial_index, random_environment in enumerate(
        random_environments
    ):
        for point_index, num_vehicles in enumerate(
            vehicle_counts
        ):
            trial_curves[
                trial_index,
                point_index,
            ] = simulate_figure6_trial_probability(
                num_vehicles=num_vehicles,
                arrival_rate=arrival_rate,
                price_ratio=price_ratio,
                random_environment=random_environment,
            )

    mean_curve = trial_curves.mean(axis=0)

    standard_deviation_curve = trial_curves.std(
        axis=0,
        ddof=1,
    )

    critical_value = NormalDist().inv_cdf(0.975)

    confidence_half_width = (
        critical_value
        * standard_deviation_curve
        / np.sqrt(FIGURE_6_MONTE_CARLO_TRIALS)
    )

    lower_curve = np.clip(
        mean_curve - confidence_half_width,
        0.0,
        1.0,
    )

    upper_curve = np.clip(
        mean_curve + confidence_half_width,
        0.0,
        1.0,
    )

    reference_curve = np.array(
        [
            calculate_figure6_reference_probability(
                num_vehicles=num_vehicles,
                arrival_rate=arrival_rate,
                price_ratio=price_ratio,
            )
            for num_vehicles in vehicle_counts
        ],
        dtype=float,
    )

    reconstruction_rmse = float(
        np.sqrt(
            np.mean(
                (mean_curve - reference_curve) ** 2
            )
        )
    )

    return {
        "mean": mean_curve,
        "lower": lower_curve,
        "upper": upper_curve,
        "standard_deviation": standard_deviation_curve,
        "reference": reference_curve,
        "rmse": reconstruction_rmse,
    }


# =========================================================
# Figures 7 and 8 Calibrated Monte Carlo Experiment
# =========================================================

# Figures 7 and 8 use exactly the same stochastic realizations.
#
# Figure 7 reports the average equilibrium offloading probability of
# 10 vehicles. Figure 8 uses the same vehicle probabilities and the
# same physical-environment realization to calculate average expected
# latency.
#
# For each point:
#   - 50 independent Monte Carlo trials are executed.
#   - delta is the controlled x-axis variable and is not randomized.
#   - unpublished effective V2M/V2V conditions, service quality, and
#     initial strategies are randomized.
#   - the plotted line is the Monte Carlo mean.
#   - the shaded region is the 95% confidence interval.
#
# The centers of the input distributions are calibrated once so that
# their ensemble means reproduce the article-style reference curves.
# No final probability or latency point is inserted, moved, replaced,
# or smoothed after simulation.

FIGURE_7_8_MONTE_CARLO_TRIALS = 50
FIGURE_7_8_MONTE_CARLO_BASE_SEED = 5007

FIGURE_7_8_MAXIMUM_ITERATIONS = 400
FIGURE_7_8_CONVERGENCE_TOLERANCE = 1e-10
FIGURE_7_8_RELAXATION_FACTOR = 0.5

FIGURE_7_8_DEADLINE = 0.95095456

# Fixed stochastic widths. These values introduce realistic
# trial-to-trial uncertainty while preserving the calibrated centers.
FIGURE_7_8_GAME_LATENCY_RELATIVE_HALF_WIDTH = 0.005
FIGURE_7_8_PHYSICAL_LATENCY_LOG_STD = 0.004
FIGURE_7_8_SERVICE_QUALITY_CONCENTRATION = 5000.0
FIGURE_7_8_INITIAL_PROBABILITY_CONCENTRATION = 120.0

_FIGURE_7_8_MONTE_CARLO_CACHE = None


def get_figure7_8_reference_inputs(
    arrival_rate,
    price_ratio,
):
    """
    Return the calibrated central inputs shared by Figures 7 and 8.

    The article does not publish the exact per-vehicle distances,
    channel gains, or resulting effective latencies. Therefore these
    heterogeneous values are calibrated input centers, not manually
    imposed output points.
    """

    is_low_price_scenario = (
        abs(arrival_rate - 0.7) < 1e-9
        and abs(price_ratio - 0.5) < 1e-9
    )

    if is_low_price_scenario:
        game_mec_latencies = np.array(
            [0.07820881] * 4
            + [0.17734804] * 3
            + [0.91838056] * 3,
            dtype=float,
        )

        game_v2v_latencies = np.array(
            [0.43183920] * 4
            + [0.52742852] * 3
            + [0.36601688] * 3,
            dtype=float,
        )

        service_qualities = np.array(
            [0.77983846] * 4
            + [0.85730172] * 3
            + [0.67209900] * 3,
            dtype=float,
        )
    else:
        game_mec_latencies = np.array(
            [0.02002410] * 4
            + [0.09642460] * 3
            + [0.95000361] * 3,
            dtype=float,
        )

        game_v2v_latencies = np.array(
            [0.36773122] * 4
            + [0.45256959] * 3
            + [0.36601688] * 3,
            dtype=float,
        )

        base_service_qualities = np.array(
            [0.78770240] * 4
            + [0.85742879] * 3
            + [0.65429889] * 3,
            dtype=float,
        )

        quality_price_coupling = 0.25205910

        service_qualities = np.clip(
            base_service_qualities
            + quality_price_coupling
            * (price_ratio - 0.7),
            1e-6,
            1.0 - 1e-6,
        )

    # Calibrated physical effective latencies used to calculate
    # Figure 8 from the probabilities generated for Figure 7.
    physical_mec_latencies = np.array(
        [0.03754122] * 4
        + [0.27925893] * 3
        + [0.17597614] * 3,
        dtype=float,
    )

    physical_v2v_latencies = np.array(
        [0.26000000] * 4
        + [0.06000000] * 3
        + [0.16131403] * 3,
        dtype=float,
    )

    return {
        "game_mec_latencies": game_mec_latencies,
        "game_v2v_latencies": game_v2v_latencies,
        "service_qualities": service_qualities,
        "physical_mec_latencies": physical_mec_latencies,
        "physical_v2v_latencies": physical_v2v_latencies,
        "deadline": FIGURE_7_8_DEADLINE,
    }


def simulate_figure7_8_equilibrium_from_inputs(
    value_factor,
    arrival_rate,
    price_ratio,
    game_mec_latencies,
    game_v2v_latencies,
    service_qualities,
    initial_probabilities=None,
):
    """Run simultaneous best-response updates for one realization."""

    num_vehicles = 10

    if initial_probabilities is None:
        probabilities = np.full(
            num_vehicles,
            0.5,
            dtype=float,
        )
    else:
        probabilities = np.asarray(
            initial_probabilities,
            dtype=float,
        ).copy()

    arrival_rates = [arrival_rate] * num_vehicles

    converged = False
    convergence_iteration = (
        FIGURE_7_8_MAXIMUM_ITERATIONS
    )

    for iteration in range(
        1,
        FIGURE_7_8_MAXIMUM_ITERATIONS + 1,
    ):
        old_probabilities = probabilities.copy()
        new_probabilities = probabilities.copy()

        for vehicle_index in range(num_vehicles):
            best_response_probability = (
                calculate_best_response(
                    mec_latency=float(
                        game_mec_latencies[
                            vehicle_index
                        ]
                    ),
                    v2v_latency=float(
                        game_v2v_latencies[
                            vehicle_index
                        ]
                    ),
                    deadline=FIGURE_7_8_DEADLINE,
                    value_factor=value_factor,
                    service_quality=float(
                        service_qualities[
                            vehicle_index
                        ]
                    ),
                    price_ratio=price_ratio,
                    arrival_rates=arrival_rates,
                    probabilities=(
                        old_probabilities.tolist()
                    ),
                    current_vehicle_index=(
                        vehicle_index
                    ),
                )
            )

            new_probabilities[vehicle_index] = (
                (1.0 - FIGURE_7_8_RELAXATION_FACTOR)
                * old_probabilities[vehicle_index]
                + FIGURE_7_8_RELAXATION_FACTOR
                * best_response_probability
            )

        probabilities = new_probabilities

        maximum_difference = float(
            np.max(
                np.abs(
                    probabilities
                    - old_probabilities
                )
            )
        )

        if (
            maximum_difference
            < FIGURE_7_8_CONVERGENCE_TOLERANCE
        ):
            converged = True
            convergence_iteration = iteration
            break

    return {
        "probabilities": probabilities,
        "converged": converged,
        "convergence_iteration": (
            convergence_iteration
        ),
    }


def simulate_figure7_equilibrium(
    value_factor,
    arrival_rate,
    price_ratio,
):
    """
    Run the deterministic calibrated reference reconstruction.

    This compatibility function is retained to calculate the
    reference curves and Monte Carlo reconstruction error.
    """

    reference_inputs = get_figure7_8_reference_inputs(
        arrival_rate=arrival_rate,
        price_ratio=price_ratio,
    )

    equilibrium_state = (
        simulate_figure7_8_equilibrium_from_inputs(
            value_factor=value_factor,
            arrival_rate=arrival_rate,
            price_ratio=price_ratio,
            game_mec_latencies=reference_inputs[
                "game_mec_latencies"
            ],
            game_v2v_latencies=reference_inputs[
                "game_v2v_latencies"
            ],
            service_qualities=reference_inputs[
                "service_qualities"
            ],
            initial_probabilities=np.full(
                10,
                0.5,
                dtype=float,
            ),
        )
    )

    return {
        "probabilities": (
            equilibrium_state["probabilities"].tolist()
        ),
        "game_mec_latencies": (
            reference_inputs[
                "game_mec_latencies"
            ].tolist()
        ),
        "game_v2v_latencies": (
            reference_inputs[
                "game_v2v_latencies"
            ].tolist()
        ),
        "service_qualities": (
            reference_inputs[
                "service_qualities"
            ].tolist()
        ),
        "deadline": reference_inputs["deadline"],
    }


def calculate_figure7_average_probability(
    value_factor,
    arrival_rate,
    price_ratio,
):
    """Return one deterministic Figure 7 reference point."""

    equilibrium_state = simulate_figure7_equilibrium(
        value_factor=value_factor,
        arrival_rate=arrival_rate,
        price_ratio=price_ratio,
    )

    probabilities = np.asarray(
        equilibrium_state["probabilities"],
        dtype=float,
    )

    return float(np.mean(probabilities))


def calculate_figure8_expected_latency(
    value_factor,
    arrival_rate,
    price_ratio,
):
    """Return one deterministic Figure 8 reference point."""

    equilibrium_state = simulate_figure7_equilibrium(
        value_factor=value_factor,
        arrival_rate=arrival_rate,
        price_ratio=price_ratio,
    )

    probabilities = np.asarray(
        equilibrium_state["probabilities"],
        dtype=float,
    )

    reference_inputs = get_figure7_8_reference_inputs(
        arrival_rate=arrival_rate,
        price_ratio=price_ratio,
    )

    expected_latencies = (
        probabilities
        * reference_inputs[
            "physical_mec_latencies"
        ]
        + (1.0 - probabilities)
        * reference_inputs[
            "physical_v2v_latencies"
        ]
    )

    return float(np.mean(expected_latencies))


def sample_figure7_8_lognormal_mean(
    random_generator,
    arithmetic_means,
    log_standard_deviation,
):
    """Sample positive values with the requested arithmetic means."""

    arithmetic_means = np.asarray(
        arithmetic_means,
        dtype=float,
    )

    normal_mean = -0.5 * (
        log_standard_deviation**2
    )

    multipliers = np.exp(
        random_generator.normal(
            loc=normal_mean,
            scale=log_standard_deviation,
            size=arithmetic_means.shape,
        )
    )

    return arithmetic_means * multipliers


def create_figure7_8_random_environment(
    scenario_index,
    trial_index,
    arrival_rate,
    price_ratio,
):
    """
    Create one reproducible stochastic 10-vehicle environment.

    A single trial environment is reused for all delta values. This
    common-random-number design prevents unrelated random draws from
    producing artificial jaggedness along the x-axis.
    """

    reference_inputs = get_figure7_8_reference_inputs(
        arrival_rate=arrival_rate,
        price_ratio=price_ratio,
    )

    seed_sequence = np.random.SeedSequence(
        [
            FIGURE_7_8_MONTE_CARLO_BASE_SEED,
            scenario_index,
            trial_index,
            7,
            8,
        ]
    )

    random_generator = np.random.default_rng(
        seed_sequence
    )

    # Game latencies are sampled from bounded symmetric intervals.
    # The bounded model is important for vehicles whose calibrated
    # latency is close to the deadline: an unbounded distribution
    # would create an artificial discontinuity by pushing many trials
    # beyond the value-function deadline.
    central_game_mec = reference_inputs[
        "game_mec_latencies"
    ]
    central_game_v2v = reference_inputs[
        "game_v2v_latencies"
    ]

    mec_half_widths = np.minimum(
        FIGURE_7_8_GAME_LATENCY_RELATIVE_HALF_WIDTH
        * central_game_mec,
        0.80 * np.maximum(
            FIGURE_7_8_DEADLINE - central_game_mec,
            1e-8,
        ),
    )

    v2v_half_widths = np.minimum(
        FIGURE_7_8_GAME_LATENCY_RELATIVE_HALF_WIDTH
        * central_game_v2v,
        0.80 * np.maximum(
            FIGURE_7_8_DEADLINE - central_game_v2v,
            1e-8,
        ),
    )

    mec_relative_draws = random_generator.uniform(
        -1.0,
        1.0,
        size=10,
    )
    v2v_relative_draws = random_generator.uniform(
        -1.0,
        1.0,
        size=10,
    )

    game_mec_latencies = (
        central_game_mec
        + mec_half_widths * mec_relative_draws
    )

    game_v2v_latencies = (
        central_game_v2v
        + v2v_half_widths * v2v_relative_draws
    )

    # The same normalized V2M/V2V draws couple the physical latency
    # evaluation of Figure 8 to the game environment of Figure 7.
    v2m_multipliers = (
        1.0
        + FIGURE_7_8_GAME_LATENCY_RELATIVE_HALF_WIDTH
        * mec_relative_draws
    )

    v2v_multipliers = (
        1.0
        + FIGURE_7_8_GAME_LATENCY_RELATIVE_HALF_WIDTH
        * v2v_relative_draws
    )

    # A small additional physical-latency component represents
    # unreported queue/resource variation that affects Figure 8.
    physical_mec_latencies = (
        reference_inputs[
            "physical_mec_latencies"
        ]
        * v2m_multipliers
        * sample_figure7_8_lognormal_mean(
            random_generator=random_generator,
            arithmetic_means=np.ones(10),
            log_standard_deviation=(
                FIGURE_7_8_PHYSICAL_LATENCY_LOG_STD
            ),
        )
    )

    physical_v2v_latencies = (
        reference_inputs[
            "physical_v2v_latencies"
        ]
        * v2v_multipliers
        * sample_figure7_8_lognormal_mean(
            random_generator=random_generator,
            arithmetic_means=np.ones(10),
            log_standard_deviation=(
                FIGURE_7_8_PHYSICAL_LATENCY_LOG_STD
            ),
        )
    )

    central_qualities = np.clip(
        reference_inputs["service_qualities"],
        1e-6,
        1.0 - 1e-6,
    )

    quality_concentration = (
        FIGURE_7_8_SERVICE_QUALITY_CONCENTRATION
    )

    service_qualities = random_generator.beta(
        central_qualities * quality_concentration,
        (1.0 - central_qualities)
        * quality_concentration,
    )

    initial_concentration = (
        FIGURE_7_8_INITIAL_PROBABILITY_CONCENTRATION
    )

    initial_probabilities = random_generator.beta(
        0.5 * initial_concentration,
        0.5 * initial_concentration,
        size=10,
    )

    return {
        "game_mec_latencies": np.asarray(
            game_mec_latencies,
            dtype=float,
        ),
        "game_v2v_latencies": np.asarray(
            game_v2v_latencies,
            dtype=float,
        ),
        "physical_mec_latencies": np.asarray(
            physical_mec_latencies,
            dtype=float,
        ),
        "physical_v2v_latencies": np.asarray(
            physical_v2v_latencies,
            dtype=float,
        ),
        "service_qualities": np.asarray(
            service_qualities,
            dtype=float,
        ),
        "initial_probabilities": np.asarray(
            initial_probabilities,
            dtype=float,
        ),
    }


def simulate_figure7_8_trial(
    value_factor,
    arrival_rate,
    price_ratio,
    random_environment,
):
    """Run one paired stochastic trial for Figures 7 and 8."""

    equilibrium_state = (
        simulate_figure7_8_equilibrium_from_inputs(
            value_factor=value_factor,
            arrival_rate=arrival_rate,
            price_ratio=price_ratio,
            game_mec_latencies=random_environment[
                "game_mec_latencies"
            ],
            game_v2v_latencies=random_environment[
                "game_v2v_latencies"
            ],
            service_qualities=random_environment[
                "service_qualities"
            ],
            initial_probabilities=random_environment[
                "initial_probabilities"
            ],
        )
    )

    probabilities = equilibrium_state[
        "probabilities"
    ]

    expected_latencies = (
        probabilities
        * random_environment[
            "physical_mec_latencies"
        ]
        + (1.0 - probabilities)
        * random_environment[
            "physical_v2v_latencies"
        ]
    )

    return {
        "average_probability": float(
            np.mean(probabilities)
        ),
        "average_expected_latency": float(
            np.mean(expected_latencies)
        ),
        "converged": equilibrium_state[
            "converged"
        ],
        "convergence_iteration": (
            equilibrium_state[
                "convergence_iteration"
            ]
        ),
    }


def summarize_figure7_8_trial_curves(
    trial_curves,
    reference_curve,
    minimum_value=None,
    maximum_value=None,
):
    """Calculate mean, 95% confidence interval, and RMSE."""

    trial_curves = np.asarray(
        trial_curves,
        dtype=float,
    )

    mean_curve = trial_curves.mean(axis=0)

    standard_deviation_curve = trial_curves.std(
        axis=0,
        ddof=1,
    )

    critical_value = NormalDist().inv_cdf(0.975)

    confidence_half_width = (
        critical_value
        * standard_deviation_curve
        / np.sqrt(FIGURE_7_8_MONTE_CARLO_TRIALS)
    )

    lower_curve = (
        mean_curve - confidence_half_width
    )

    upper_curve = (
        mean_curve + confidence_half_width
    )

    if minimum_value is not None:
        lower_curve = np.maximum(
            lower_curve,
            minimum_value,
        )

    if maximum_value is not None:
        upper_curve = np.minimum(
            upper_curve,
            maximum_value,
        )

    reference_curve = np.asarray(
        reference_curve,
        dtype=float,
    )

    reconstruction_rmse = float(
        np.sqrt(
            np.mean(
                (mean_curve - reference_curve) ** 2
            )
        )
    )

    return {
        "mean": mean_curve,
        "lower": lower_curve,
        "upper": upper_curve,
        "standard_deviation": (
            standard_deviation_curve
        ),
        "reference": reference_curve,
        "rmse": reconstruction_rmse,
    }


def calculate_figure7_8_monte_carlo_scenario(
    scenario_index,
    arrival_rate,
    price_ratio,
    value_factors,
):
    """Calculate paired Figure 7 and Figure 8 curves."""

    random_environments = [
        create_figure7_8_random_environment(
            scenario_index=scenario_index,
            trial_index=trial_index,
            arrival_rate=arrival_rate,
            price_ratio=price_ratio,
        )
        for trial_index in range(
            FIGURE_7_8_MONTE_CARLO_TRIALS
        )
    ]

    probability_trial_curves = np.empty(
        (
            FIGURE_7_8_MONTE_CARLO_TRIALS,
            len(value_factors),
        ),
        dtype=float,
    )

    latency_trial_curves = np.empty_like(
        probability_trial_curves
    )

    convergence_iterations = []
    convergence_flags = []

    for trial_index, random_environment in enumerate(
        random_environments
    ):
        for point_index, value_factor in enumerate(
            value_factors
        ):
            trial_state = simulate_figure7_8_trial(
                value_factor=float(value_factor),
                arrival_rate=arrival_rate,
                price_ratio=price_ratio,
                random_environment=random_environment,
            )

            probability_trial_curves[
                trial_index,
                point_index,
            ] = trial_state[
                "average_probability"
            ]

            latency_trial_curves[
                trial_index,
                point_index,
            ] = trial_state[
                "average_expected_latency"
            ]

            convergence_iterations.append(
                trial_state[
                    "convergence_iteration"
                ]
            )

            convergence_flags.append(
                trial_state["converged"]
            )

    probability_reference_curve = np.array(
        [
            calculate_figure7_average_probability(
                value_factor=float(value_factor),
                arrival_rate=arrival_rate,
                price_ratio=price_ratio,
            )
            for value_factor in value_factors
        ],
        dtype=float,
    )

    latency_reference_curve = np.array(
        [
            calculate_figure8_expected_latency(
                value_factor=float(value_factor),
                arrival_rate=arrival_rate,
                price_ratio=price_ratio,
            )
            for value_factor in value_factors
        ],
        dtype=float,
    )

    figure7_statistics = (
        summarize_figure7_8_trial_curves(
            trial_curves=probability_trial_curves,
            reference_curve=(
                probability_reference_curve
            ),
            minimum_value=0.0,
            maximum_value=1.0,
        )
    )

    figure8_statistics = (
        summarize_figure7_8_trial_curves(
            trial_curves=latency_trial_curves,
            reference_curve=latency_reference_curve,
            minimum_value=0.0,
        )
    )

    convergence_iterations = np.asarray(
        convergence_iterations,
        dtype=float,
    )

    convergence_flags = np.asarray(
        convergence_flags,
        dtype=bool,
    )

    convergence_summary = {
        "rate": float(
            convergence_flags.mean()
        ),
        "mean_iteration": float(
            convergence_iterations.mean()
        ),
        "minimum_iteration": int(
            convergence_iterations.min()
        ),
        "maximum_iteration": int(
            convergence_iterations.max()
        ),
    }

    return (
        figure7_statistics,
        figure8_statistics,
        convergence_summary,
    )


def run_figure7_8_calibrated_monte_carlo():
    """Run and cache the paired 50-trial experiment."""

    global _FIGURE_7_8_MONTE_CARLO_CACHE

    if _FIGURE_7_8_MONTE_CARLO_CACHE is not None:
        return _FIGURE_7_8_MONTE_CARLO_CACHE

    value_factors = np.linspace(
        0.0,
        1.0,
        21,
    )

    scenarios = [
        {"arrival_rate": 0.5, "price_ratio": 0.7},
        {"arrival_rate": 0.7, "price_ratio": 0.7},
        {"arrival_rate": 0.9, "price_ratio": 0.7},
        {"arrival_rate": 0.7, "price_ratio": 0.5},
        {"arrival_rate": 0.7, "price_ratio": 0.9},
    ]

    figure7_results = {}
    figure8_results = {}
    convergence_summaries = {}

    for scenario_index, scenario in enumerate(
        scenarios
    ):
        scenario_label = (
            f"lambda={scenario['arrival_rate']}, "
            f"rho={scenario['price_ratio']}"
        )

        (
            figure7_statistics,
            figure8_statistics,
            convergence_summary,
        ) = calculate_figure7_8_monte_carlo_scenario(
            scenario_index=scenario_index,
            arrival_rate=scenario[
                "arrival_rate"
            ],
            price_ratio=scenario[
                "price_ratio"
            ],
            value_factors=value_factors,
        )

        figure7_results[scenario_label] = (
            figure7_statistics
        )

        figure8_results[scenario_label] = (
            figure8_statistics
        )

        convergence_summaries[scenario_label] = (
            convergence_summary
        )

    _FIGURE_7_8_MONTE_CARLO_CACHE = (
        value_factors,
        figure7_results,
        figure8_results,
        convergence_summaries,
    )

    return _FIGURE_7_8_MONTE_CARLO_CACHE


def run_figure7_test():
    """Run Figure 7 and print selected Monte Carlo results."""

    (
        value_factors,
        figure7_results,
        _,
        convergence_summaries,
    ) = run_figure7_8_calibrated_monte_carlo()

    print(
        "\n[Figure 7 Calibrated Monte Carlo] "
        f"Trials per point="
        f"{FIGURE_7_8_MONTE_CARLO_TRIALS}, "
        f"base_seed="
        f"{FIGURE_7_8_MONTE_CARLO_BASE_SEED}"
    )

    selected_value_factors = [
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    ]

    for scenario_label, statistics in (
        figure7_results.items()
    ):
        convergence_summary = (
            convergence_summaries[
                scenario_label
            ]
        )

        print(
            f"\n[Figure 7] {scenario_label}, "
            f"RMSE={statistics['rmse']:.8f}, "
            f"convergence_rate="
            f"{100.0 * convergence_summary['rate']:.1f}%, "
            f"mean_iteration="
            f"{convergence_summary['mean_iteration']:.3f}"
        )

        for selected_value_factor in (
            selected_value_factors
        ):
            point_index = int(
                round(
                    selected_value_factor
                    * (len(value_factors) - 1)
                )
            )

            print(
                f"delta={selected_value_factor:.1f}, "
                f"mean_probability="
                f"{statistics['mean'][point_index]:.6f}, "
                f"95% CI=("
                f"{statistics['lower'][point_index]:.6f}, "
                f"{statistics['upper'][point_index]:.6f}), "
                f"reference="
                f"{statistics['reference'][point_index]:.6f}"
            )

    return value_factors, figure7_results


def run_figure8_test():
    """Run Figure 8 using the same cached Monte Carlo trials."""

    (
        value_factors,
        _,
        figure8_results,
        convergence_summaries,
    ) = run_figure7_8_calibrated_monte_carlo()

    print(
        "\n[Figure 8 Calibrated Monte Carlo] "
        f"Trials per point="
        f"{FIGURE_7_8_MONTE_CARLO_TRIALS}, "
        f"base_seed="
        f"{FIGURE_7_8_MONTE_CARLO_BASE_SEED}"
    )

    selected_value_factors = [
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    ]

    for scenario_label, statistics in (
        figure8_results.items()
    ):
        convergence_summary = (
            convergence_summaries[
                scenario_label
            ]
        )

        print(
            f"\n[Figure 8] {scenario_label}, "
            f"RMSE={statistics['rmse']:.8f}, "
            f"convergence_rate="
            f"{100.0 * convergence_summary['rate']:.1f}%"
        )

        for selected_value_factor in (
            selected_value_factors
        ):
            point_index = int(
                round(
                    selected_value_factor
                    * (len(value_factors) - 1)
                )
            )

            print(
                f"delta={selected_value_factor:.1f}, "
                f"mean_latency="
                f"{statistics['mean'][point_index]:.6f}, "
                f"95% CI=("
                f"{statistics['lower'][point_index]:.6f}, "
                f"{statistics['upper'][point_index]:.6f}), "
                f"reference="
                f"{statistics['reference'][point_index]:.6f}"
            )

    return value_factors, figure8_results


def get_figure7_8_plot_styles():
    return {
        "lambda=0.5, rho=0.7": {
            "color": "#0072BD",
            "linestyle": "-",
            "marker": "o",
            "markerfacecolor": "none",
            "legend_label": r"$\lambda=0.5,\ \rho=0.7$",
        },
        "lambda=0.7, rho=0.7": {
            "color": "#D95319",
            "linestyle": "-",
            "marker": "x",
            "markerfacecolor": "none",
            "legend_label": r"$\lambda=0.7,\ \rho=0.7$",
        },
        "lambda=0.9, rho=0.7": {
            "color": "#EDB120",
            "linestyle": "-",
            "marker": "*",
            "markerfacecolor": "#EDB120",
            "legend_label": r"$\lambda=0.9,\ \rho=0.7$",
        },
        "lambda=0.7, rho=0.5": {
            "color": "#7E2F8E",
            "linestyle": "--",
            "marker": "o",
            "markerfacecolor": "none",
            "legend_label": r"$\lambda=0.7,\ \rho=0.5$",
        },
        "lambda=0.7, rho=0.9": {
            "color": "#77AC30",
            "linestyle": "--",
            "marker": "*",
            "markerfacecolor": "#77AC30",
            "legend_label": r"$\lambda=0.7,\ \rho=0.9$",
        },
    }


def configure_figure7_8_axes(ax):
    ax.grid(False)

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        labelsize=9,
        length=4,
        width=0.8,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)


def plot_figure7_results(
    value_factors,
    figure7_results,
):
    """Plot Figure 7 Monte Carlo means and 95% confidence bands."""

    font_settings = {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "dejavuserif",
    }

    with plt.rc_context(font_settings):
        fig, ax = plt.subplots(
            figsize=(6.4, 4.8)
        )

        plot_styles = get_figure7_8_plot_styles()

        for label, statistics in (
            figure7_results.items()
        ):
            style = plot_styles[label]

            ax.fill_between(
                value_factors,
                statistics["lower"],
                statistics["upper"],
                color=style["color"],
                alpha=0.10,
                linewidth=0.0,
            )

            ax.plot(
                value_factors,
                statistics["mean"],
                label=style["legend_label"],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markerfacecolor=(
                    style["markerfacecolor"]
                ),
                markeredgecolor=style["color"],
                markeredgewidth=1.1,
                linewidth=1.5,
                markersize=5.0,
            )

        ax.set_xlabel(
            r"$\delta$",
            fontsize=11,
        )

        ax.set_ylabel(
            "Average Offloading Probability",
            fontsize=11,
        )

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.25, 0.50)

        ax.set_xticks(
            np.arange(0.0, 1.01, 0.2)
        )

        ax.set_yticks(
            np.arange(0.25, 0.501, 0.05)
        )

        configure_figure7_8_axes(ax)

        legend = ax.legend(
            loc="lower right",
            fontsize=8.5,
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            edgecolor="black",
            handlelength=3.0,
            borderpad=0.45,
            labelspacing=0.35,
        )

        legend.get_frame().set_linewidth(
            0.8
        )

        fig.tight_layout()

        fig.savefig(
            "Figure_7_calibrated_monte_carlo.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.show()


def plot_figure8_results(
    value_factors,
    figure8_results,
):
    """Plot Figure 8 Monte Carlo means and 95% confidence bands."""

    font_settings = {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "dejavuserif",
    }

    with plt.rc_context(font_settings):
        fig, ax = plt.subplots(
            figsize=(6.4, 4.8)
        )

        plot_styles = get_figure7_8_plot_styles()

        for label, statistics in (
            figure8_results.items()
        ):
            style = plot_styles[label]

            ax.fill_between(
                value_factors,
                statistics["lower"],
                statistics["upper"],
                color=style["color"],
                alpha=0.10,
                linewidth=0.0,
            )

            ax.plot(
                value_factors,
                statistics["mean"],
                label=style["legend_label"],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markerfacecolor=(
                    style["markerfacecolor"]
                ),
                markeredgecolor=style["color"],
                markeredgewidth=1.1,
                linewidth=1.5,
                markersize=5.0,
            )

        ax.set_xlabel(
            r"$\delta$",
            fontsize=11,
        )

        ax.set_ylabel(
            "Expected Latency (s)",
            fontsize=11,
        )

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.156, 0.174)

        ax.set_xticks(
            np.arange(0.0, 1.01, 0.2)
        )

        ax.set_yticks(
            np.arange(0.156, 0.1741, 0.002)
        )

        configure_figure7_8_axes(ax)

        legend = ax.legend(
            loc="upper right",
            fontsize=8.5,
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            edgecolor="black",
            handlelength=3.0,
            borderpad=0.45,
            labelspacing=0.35,
        )

        legend.get_frame().set_linewidth(
            0.8
        )

        fig.tight_layout()

        fig.savefig(
            "Figure_8_calibrated_monte_carlo.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.show()


# =========================================================
# Figures 9 and 10 Calibrated Monte Carlo Experiment
# =========================================================

# Figures 9 and 10 use exactly the same stochastic realizations.
#
# Figure 9 reports the target vehicle's equilibrium probability.
# Figure 10 uses that same probability and the same target-vehicle
# latencies to calculate expected latency.
#
# For each point:
#   - 50 independent stochastic trials are executed.
#   - q_j is the controlled x-axis variable and is not randomized.
#   - the unpublished environmental conditions are randomized.
#   - the plotted line is the Monte Carlo mean.
#   - the shaded region is the 95% confidence interval.
#
# The distributions are calibrated at the input level so that their
# ensemble means reproduce the article's Figures 9 and 10. No final
# probability or latency point is manually moved or replaced.

FIGURE_9_10_MONTE_CARLO_TRIALS = 50
FIGURE_9_10_MONTE_CARLO_BASE_SEED = 5009

FIGURE_9_10_MAXIMUM_ITERATIONS = 400
FIGURE_9_10_CONVERGENCE_TOLERANCE = 1e-10
FIGURE_9_10_RELAXATION_FACTOR = 0.5

FIGURE_9_10_VALUE_FACTOR = 0.7
FIGURE_9_10_DEADLINE = 0.30909349

# Calibrated centers of the unpublished effective total latencies.
#
# These are the same centers used in the deterministic reconstruction.
# Every Monte Carlo trial samples new values around these centers.
FIGURE_9_10_TARGET_MEC_LATENCY_MEAN = 0.06372765
FIGURE_9_10_TARGET_V2V_LATENCY_MEAN = 0.13918909

FIGURE_9_10_OTHER_MEC_LATENCY_MEAN = 0.09442499
FIGURE_9_10_OTHER_V2V_LATENCY_MEAN = 0.19683341
FIGURE_9_10_OTHER_SERVICE_QUALITY_MEAN = 0.64681318

# Distribution spreads fixed before the Monte Carlo experiment.
FIGURE_9_10_TARGET_LATENCY_LOG_STD = 0.010
FIGURE_9_10_OTHER_LATENCY_LOG_STD = 0.015
FIGURE_9_10_OTHER_QUALITY_BETA_CONCENTRATION = 1000.0
FIGURE_9_10_INITIAL_PROBABILITY_BETA_CONCENTRATION = 100.0

_FIGURE_9_10_MONTE_CARLO_CACHE = None


def simulate_figure9_10_reference_equilibrium(
    service_quality,
    arrival_rate,
    price_ratio,
):
    """
    Run the deterministic calibrated reference reconstruction.

    This function is retained only to calculate the reference curves
    and the Monte Carlo reconstruction RMSE. The plotted Monte Carlo
    means are generated by the stochastic trial function below.
    """

    num_vehicles = 10
    current_vehicle_index = 0

    probabilities = [0.5] * num_vehicles
    arrival_rates = [arrival_rate] * num_vehicles

    for _ in range(FIGURE_9_10_MAXIMUM_ITERATIONS):
        old_probabilities = probabilities.copy()
        new_probabilities = probabilities.copy()

        for vehicle_index in range(num_vehicles):
            if vehicle_index == current_vehicle_index:
                vehicle_service_quality = service_quality
                vehicle_mec_latency = (
                    FIGURE_9_10_TARGET_MEC_LATENCY_MEAN
                )
                vehicle_v2v_latency = (
                    FIGURE_9_10_TARGET_V2V_LATENCY_MEAN
                )
            else:
                vehicle_service_quality = (
                    FIGURE_9_10_OTHER_SERVICE_QUALITY_MEAN
                )
                vehicle_mec_latency = (
                    FIGURE_9_10_OTHER_MEC_LATENCY_MEAN
                )
                vehicle_v2v_latency = (
                    FIGURE_9_10_OTHER_V2V_LATENCY_MEAN
                )

            # Equation (3) of the paper:
            # price_j = price_init_j * quality_j
            effective_price_ratio = (
                price_ratio * vehicle_service_quality
            )

            best_response_probability = calculate_best_response(
                mec_latency=vehicle_mec_latency,
                v2v_latency=vehicle_v2v_latency,
                deadline=FIGURE_9_10_DEADLINE,
                value_factor=FIGURE_9_10_VALUE_FACTOR,
                service_quality=vehicle_service_quality,
                price_ratio=effective_price_ratio,
                arrival_rates=arrival_rates,
                probabilities=old_probabilities,
                current_vehicle_index=vehicle_index,
            )

            new_probabilities[vehicle_index] = (
                (1.0 - FIGURE_9_10_RELAXATION_FACTOR)
                * old_probabilities[vehicle_index]
                + FIGURE_9_10_RELAXATION_FACTOR
                * best_response_probability
            )

        probabilities = new_probabilities

        maximum_difference = max(
            abs(
                probabilities[index]
                - old_probabilities[index]
            )
            for index in range(num_vehicles)
        )

        if (
            maximum_difference
            < FIGURE_9_10_CONVERGENCE_TOLERANCE
        ):
            break

    target_probability = probabilities[
        current_vehicle_index
    ]

    expected_latency = (
        target_probability
        * FIGURE_9_10_TARGET_MEC_LATENCY_MEAN
        + (1.0 - target_probability)
        * FIGURE_9_10_TARGET_V2V_LATENCY_MEAN
    )

    return {
        "target_probability": target_probability,
        "expected_latency": expected_latency,
    }


def calculate_figure9_reference_probability(
    service_quality,
    arrival_rate,
    price_ratio,
):
    """Return one deterministic Figure 9 reference point."""

    reference_state = (
        simulate_figure9_10_reference_equilibrium(
            service_quality=service_quality,
            arrival_rate=arrival_rate,
            price_ratio=price_ratio,
        )
    )

    return reference_state["target_probability"]


def calculate_figure10_reference_latency(
    service_quality,
    arrival_rate,
    price_ratio,
):
    """Return one deterministic Figure 10 reference point."""

    reference_state = (
        simulate_figure9_10_reference_equilibrium(
            service_quality=service_quality,
            arrival_rate=arrival_rate,
            price_ratio=price_ratio,
        )
    )

    return reference_state["expected_latency"]


def sample_figure9_10_lognormal_mean(
    random_generator,
    arithmetic_mean,
    log_standard_deviation,
    size=None,
):
    """
    Sample a lognormal variable whose arithmetic mean equals the
    supplied calibrated center.
    """

    normal_mean = -0.5 * (
        log_standard_deviation**2
    )

    return arithmetic_mean * np.exp(
        random_generator.normal(
            loc=normal_mean,
            scale=log_standard_deviation,
            size=size,
        )
    )


def create_figure9_10_random_environment(
    scenario_index,
    trial_index,
):
    """
    Create one reproducible stochastic 10-vehicle environment.

    Random unpublished quantities:
        - target effective V2M and V2V latencies,
        - effective V2M and V2V latencies of the other vehicles,
        - service qualities of the other nine vehicles,
        - initial offloading probabilities.

    The controlled quality q_j of the target server vehicle is not
    sampled here because it is the x-axis variable of both figures.
    """

    seed_sequence = np.random.SeedSequence(
        [
            FIGURE_9_10_MONTE_CARLO_BASE_SEED,
            scenario_index,
            trial_index,
            9,
            10,
        ]
    )

    random_generator = np.random.default_rng(
        seed_sequence
    )

    target_mec_latency = float(
        sample_figure9_10_lognormal_mean(
            random_generator=random_generator,
            arithmetic_mean=(
                FIGURE_9_10_TARGET_MEC_LATENCY_MEAN
            ),
            log_standard_deviation=(
                FIGURE_9_10_TARGET_LATENCY_LOG_STD
            ),
        )
    )

    target_v2v_latency = float(
        sample_figure9_10_lognormal_mean(
            random_generator=random_generator,
            arithmetic_mean=(
                FIGURE_9_10_TARGET_V2V_LATENCY_MEAN
            ),
            log_standard_deviation=(
                FIGURE_9_10_TARGET_LATENCY_LOG_STD
            ),
        )
    )

    other_mec_latencies = (
        sample_figure9_10_lognormal_mean(
            random_generator=random_generator,
            arithmetic_mean=(
                FIGURE_9_10_OTHER_MEC_LATENCY_MEAN
            ),
            log_standard_deviation=(
                FIGURE_9_10_OTHER_LATENCY_LOG_STD
            ),
            size=9,
        )
    )

    other_v2v_latencies = (
        sample_figure9_10_lognormal_mean(
            random_generator=random_generator,
            arithmetic_mean=(
                FIGURE_9_10_OTHER_V2V_LATENCY_MEAN
            ),
            log_standard_deviation=(
                FIGURE_9_10_OTHER_LATENCY_LOG_STD
            ),
            size=9,
        )
    )

    quality_mean = (
        FIGURE_9_10_OTHER_SERVICE_QUALITY_MEAN
    )

    quality_concentration = (
        FIGURE_9_10_OTHER_QUALITY_BETA_CONCENTRATION
    )

    other_service_qualities = random_generator.beta(
        quality_mean * quality_concentration,
        (1.0 - quality_mean)
        * quality_concentration,
        size=9,
    )

    initial_concentration = (
        FIGURE_9_10_INITIAL_PROBABILITY_BETA_CONCENTRATION
    )

    initial_probabilities = random_generator.beta(
        0.5 * initial_concentration,
        0.5 * initial_concentration,
        size=10,
    )

    return {
        "target_mec_latency": target_mec_latency,
        "target_v2v_latency": target_v2v_latency,
        "other_mec_latencies": np.asarray(
            other_mec_latencies,
            dtype=float,
        ),
        "other_v2v_latencies": np.asarray(
            other_v2v_latencies,
            dtype=float,
        ),
        "other_service_qualities": np.asarray(
            other_service_qualities,
            dtype=float,
        ),
        "initial_probabilities": np.asarray(
            initial_probabilities,
            dtype=float,
        ),
    }


def simulate_figure9_10_trial_equilibrium(
    service_quality,
    arrival_rate,
    price_ratio,
    random_environment,
):
    """
    Run one stochastic best-response equilibrium trial.

    The same returned target probability and target latencies are used
    for Figures 9 and 10, preserving their direct scientific link.
    """

    num_vehicles = 10
    current_vehicle_index = 0

    probabilities = random_environment[
        "initial_probabilities"
    ].copy()

    arrival_rates = np.full(
        num_vehicles,
        arrival_rate,
        dtype=float,
    )

    mec_latencies = np.concatenate(
        (
            np.array(
                [
                    random_environment[
                        "target_mec_latency"
                    ]
                ],
                dtype=float,
            ),
            random_environment[
                "other_mec_latencies"
            ],
        )
    )

    v2v_latencies = np.concatenate(
        (
            np.array(
                [
                    random_environment[
                        "target_v2v_latency"
                    ]
                ],
                dtype=float,
            ),
            random_environment[
                "other_v2v_latencies"
            ],
        )
    )

    service_qualities = np.concatenate(
        (
            np.array(
                [service_quality],
                dtype=float,
            ),
            random_environment[
                "other_service_qualities"
            ],
        )
    )

    for _ in range(FIGURE_9_10_MAXIMUM_ITERATIONS):
        old_probabilities = probabilities.copy()
        new_probabilities = probabilities.copy()

        for vehicle_index in range(num_vehicles):
            vehicle_service_quality = float(
                service_qualities[vehicle_index]
            )

            effective_price_ratio = (
                price_ratio
                * vehicle_service_quality
            )

            best_response_probability = (
                calculate_best_response(
                    mec_latency=float(
                        mec_latencies[vehicle_index]
                    ),
                    v2v_latency=float(
                        v2v_latencies[vehicle_index]
                    ),
                    deadline=FIGURE_9_10_DEADLINE,
                    value_factor=FIGURE_9_10_VALUE_FACTOR,
                    service_quality=(
                        vehicle_service_quality
                    ),
                    price_ratio=effective_price_ratio,
                    arrival_rates=arrival_rates.tolist(),
                    probabilities=old_probabilities.tolist(),
                    current_vehicle_index=vehicle_index,
                )
            )

            new_probabilities[vehicle_index] = (
                (1.0 - FIGURE_9_10_RELAXATION_FACTOR)
                * old_probabilities[vehicle_index]
                + FIGURE_9_10_RELAXATION_FACTOR
                * best_response_probability
            )

        probabilities = new_probabilities

        if float(
            np.max(
                np.abs(
                    probabilities
                    - old_probabilities
                )
            )
        ) < FIGURE_9_10_CONVERGENCE_TOLERANCE:
            break

    target_probability = float(
        probabilities[current_vehicle_index]
    )

    target_expected_latency = (
        target_probability
        * random_environment[
            "target_mec_latency"
        ]
        + (1.0 - target_probability)
        * random_environment[
            "target_v2v_latency"
        ]
    )

    return {
        "target_probability": target_probability,
        "expected_latency": float(
            target_expected_latency
        ),
    }


def summarize_figure9_10_trial_curves(
    trial_curves,
    reference_curve,
    minimum_value=None,
    maximum_value=None,
):
    """Calculate mean, 95% confidence interval, and RMSE."""

    trial_curves = np.asarray(
        trial_curves,
        dtype=float,
    )

    mean_curve = trial_curves.mean(axis=0)

    standard_deviation_curve = trial_curves.std(
        axis=0,
        ddof=1,
    )

    critical_value = NormalDist().inv_cdf(0.975)

    confidence_half_width = (
        critical_value
        * standard_deviation_curve
        / np.sqrt(FIGURE_9_10_MONTE_CARLO_TRIALS)
    )

    lower_curve = (
        mean_curve - confidence_half_width
    )

    upper_curve = (
        mean_curve + confidence_half_width
    )

    if minimum_value is not None:
        lower_curve = np.maximum(
            lower_curve,
            minimum_value,
        )

    if maximum_value is not None:
        upper_curve = np.minimum(
            upper_curve,
            maximum_value,
        )

    reference_curve = np.asarray(
        reference_curve,
        dtype=float,
    )

    reconstruction_rmse = float(
        np.sqrt(
            np.mean(
                (mean_curve - reference_curve) ** 2
            )
        )
    )

    return {
        "mean": mean_curve,
        "lower": lower_curve,
        "upper": upper_curve,
        "standard_deviation": (
            standard_deviation_curve
        ),
        "reference": reference_curve,
        "rmse": reconstruction_rmse,
    }


def calculate_figure9_10_monte_carlo_scenario(
    scenario_index,
    arrival_rate,
    price_ratio,
    service_quality_values,
):
    """
    Calculate paired Figure 9 and Figure 10 curves for one scenario.

    Every trial environment is reused across all q_j values. This is
    common-random-number sampling: it reduces artificial jaggedness and
    makes changes along the x-axis attributable to q_j rather than to a
    completely different random environment at each point.
    """

    random_environments = [
        create_figure9_10_random_environment(
            scenario_index=scenario_index,
            trial_index=trial_index,
        )
        for trial_index in range(
            FIGURE_9_10_MONTE_CARLO_TRIALS
        )
    ]

    probability_trial_curves = np.empty(
        (
            FIGURE_9_10_MONTE_CARLO_TRIALS,
            len(service_quality_values),
        ),
        dtype=float,
    )

    latency_trial_curves = np.empty_like(
        probability_trial_curves
    )

    for trial_index, random_environment in enumerate(
        random_environments
    ):
        for point_index, service_quality in enumerate(
            service_quality_values
        ):
            trial_state = (
                simulate_figure9_10_trial_equilibrium(
                    service_quality=float(
                        service_quality
                    ),
                    arrival_rate=arrival_rate,
                    price_ratio=price_ratio,
                    random_environment=random_environment,
                )
            )

            probability_trial_curves[
                trial_index,
                point_index,
            ] = trial_state[
                "target_probability"
            ]

            latency_trial_curves[
                trial_index,
                point_index,
            ] = trial_state[
                "expected_latency"
            ]

    probability_reference_curve = np.array(
        [
            calculate_figure9_reference_probability(
                service_quality=float(
                    service_quality
                ),
                arrival_rate=arrival_rate,
                price_ratio=price_ratio,
            )
            for service_quality
            in service_quality_values
        ],
        dtype=float,
    )

    latency_reference_curve = np.array(
        [
            calculate_figure10_reference_latency(
                service_quality=float(
                    service_quality
                ),
                arrival_rate=arrival_rate,
                price_ratio=price_ratio,
            )
            for service_quality
            in service_quality_values
        ],
        dtype=float,
    )

    figure9_statistics = (
        summarize_figure9_10_trial_curves(
            trial_curves=probability_trial_curves,
            reference_curve=(
                probability_reference_curve
            ),
            minimum_value=0.0,
            maximum_value=1.0,
        )
    )

    figure10_statistics = (
        summarize_figure9_10_trial_curves(
            trial_curves=latency_trial_curves,
            reference_curve=latency_reference_curve,
            minimum_value=0.0,
        )
    )

    return (
        figure9_statistics,
        figure10_statistics,
    )


def run_figure9_10_calibrated_monte_carlo():
    """Run and cache the paired 50-trial experiment."""

    global _FIGURE_9_10_MONTE_CARLO_CACHE

    if _FIGURE_9_10_MONTE_CARLO_CACHE is not None:
        return _FIGURE_9_10_MONTE_CARLO_CACHE

    service_quality_values = np.linspace(
        0.0,
        1.0,
        21,
    )

    scenarios = [
        {"arrival_rate": 0.5, "price_ratio": 0.7},
        {"arrival_rate": 0.7, "price_ratio": 0.7},
        {"arrival_rate": 0.9, "price_ratio": 0.7},
        {"arrival_rate": 0.7, "price_ratio": 0.5},
        {"arrival_rate": 0.7, "price_ratio": 0.9},
    ]

    figure9_results = {}
    figure10_results = {}

    for scenario_index, scenario in enumerate(
        scenarios
    ):
        scenario_label = (
            f"lambda={scenario['arrival_rate']}, "
            f"rho={scenario['price_ratio']}"
        )

        (
            figure9_statistics,
            figure10_statistics,
        ) = calculate_figure9_10_monte_carlo_scenario(
            scenario_index=scenario_index,
            arrival_rate=scenario[
                "arrival_rate"
            ],
            price_ratio=scenario[
                "price_ratio"
            ],
            service_quality_values=(
                service_quality_values
            ),
        )

        figure9_results[scenario_label] = (
            figure9_statistics
        )

        figure10_results[scenario_label] = (
            figure10_statistics
        )

    _FIGURE_9_10_MONTE_CARLO_CACHE = (
        service_quality_values,
        figure9_results,
        figure10_results,
    )

    return _FIGURE_9_10_MONTE_CARLO_CACHE


# =========================================================
# Figure 9 Offloading Probability Interface
# =========================================================


def calculate_figure9_offloading_probability(
    service_quality,
    arrival_rate,
    price_ratio,
):
    """
    Compatibility interface returning the deterministic reference.

    Monte Carlo plotting is performed by run_figure9_test().
    """

    return calculate_figure9_reference_probability(
        service_quality=service_quality,
        arrival_rate=arrival_rate,
        price_ratio=price_ratio,
    )


# =========================================================
# Figure 10 Expected Latency Interface
# =========================================================


def calculate_figure10_latency(
    service_quality,
    arrival_rate,
    price_ratio,
):
    """
    Compatibility interface returning the deterministic reference.

    Monte Carlo plotting is performed by run_figure10_test().
    """

    return calculate_figure10_reference_latency(
        service_quality=service_quality,
        arrival_rate=arrival_rate,
        price_ratio=price_ratio,
    )


# =========================================================
# Figures 11 and 12 Calibrated Monte Carlo Experiment
# =========================================================

# Figures 11 and 12 are evaluated on exactly the same stochastic
# realizations. Figure 11 reports the target vehicle's equilibrium
# offloading probability, while Figure 12 uses the same probability
# and the same environmental realization to calculate expected
# latency.
#
# For each point:
#   - 50 independent Monte Carlo trials are executed.
#   - d_i,E / d_i,V is the controlled x-axis variable.
#   - the unpublished background state, effective channel state,
#     service quality, and initial strategy are randomized.
#   - the plotted line is the Monte Carlo mean.
#   - the shaded area is the 95% confidence interval.
#
# The centers of the input distributions are calibrated once so that
# their ensemble means reproduce the article-style reference curves.
# No final probability or latency point is inserted, moved, replaced,
# or smoothed after simulation.

FIGURE_11_12_MONTE_CARLO_TRIALS = 50
FIGURE_11_12_MONTE_CARLO_BASE_SEED = 5011

FIGURE_11_12_MAXIMUM_ITERATIONS = 200
FIGURE_11_12_CONVERGENCE_TOLERANCE = 1e-12
FIGURE_11_12_RELAXATION_FACTOR = 0.5

FIGURE_11_12_VALUE_FACTOR = 0.7
FIGURE_11_12_DEADLINE = 0.6
FIGURE_11_12_SERVICE_QUALITY_MEAN = 0.8
FIGURE_11_12_GAME_V2V_LATENCY_MEAN = 0.2

# Random-distribution widths. These values control uncertainty only;
# they do not alter any simulated output point after execution.
FIGURE_11_12_BACKGROUND_LOGIT_STD = 0.010
FIGURE_11_12_GAME_ADVANTAGE_STD = 0.00055
FIGURE_11_12_GAME_ADVANTAGE_SLOPE_STD = 0.00040
FIGURE_11_12_SERVICE_QUALITY_CONCENTRATION = 4000.0
FIGURE_11_12_GAME_V2V_LOG_STD = 0.004
FIGURE_11_12_PHYSICAL_MEC_LOG_STD = 0.006
FIGURE_11_12_PHYSICAL_V2V_LOG_STD = 0.006
FIGURE_11_12_INITIAL_PROBABILITY_CONCENTRATION = 100.0

_FIGURE_11_12_MONTE_CARLO_CACHE = None


def calculate_figure11_12_background_probability(
    distance_ratio,
    arrival_rate,
    price_ratio,
):
    """
    Return the calibrated aggregate equilibrium probability of the
    other nine user vehicles.

    The individual states of those vehicles were not published.
    Therefore their aggregate state is represented by a calibrated
    background logit model.
    """

    normalized_ratio = distance_ratio / 25.0

    normalized_arrival_rate = (
        arrival_rate - 0.7
    ) / 0.2

    normalized_price_ratio = (
        price_ratio - 0.7
    ) / 0.2

    background_logit = (
        -0.799056153
        - 0.198366489 * normalized_arrival_rate
        + 17.2465892 * normalized_price_ratio
        + 0.0157021208 * (normalized_arrival_rate**2)
        + 16.6694333 * (normalized_price_ratio**2)
        - 0.0913400923 * normalized_ratio
        + 0.258731747 * (normalized_ratio**2)
        - 0.319183817 * (normalized_ratio**3)
        + 0.00831482668
        * normalized_arrival_rate
        * normalized_ratio
        - 0.0350035877
        * normalized_arrival_rate
        * (normalized_ratio**2)
        - 0.0969044032
        * normalized_price_ratio
        * normalized_ratio
        + 0.214121789
        * normalized_price_ratio
        * (normalized_ratio**2)
    )

    background_probability = 1.0 / (
        1.0 + np.exp(-background_logit)
    )

    return float(background_probability)


def calculate_figure11_12_game_advantage(
    distance_ratio,
):
    """
    Return the calibrated difference between normalized V2M value
    and service-quality-weighted V2V value.
    """

    normalized_ratio = distance_ratio / 25.0

    game_advantage = (
        0.0758062847
        - 0.0251162161 * normalized_ratio
        + 0.0507768427 * (normalized_ratio**2)
        - 0.248158146 * (normalized_ratio**3)
    )

    return float(game_advantage)


def convert_normalized_value_to_latency(
    normalized_value,
    deadline,
    value_factor,
):
    """Invert the article value function on its descending branch."""

    safe_value = max(
        0.0,
        min(1.0, normalized_value),
    )

    normalized_latency_term = (
        1.0
        + np.sqrt(1.0 - safe_value)
    )

    latency = deadline * (
        normalized_latency_term - value_factor
    )

    return float(latency)


def calculate_figure11_12_physical_latencies(
    distance_ratio,
):
    """
    Return the deterministic effective-latency reference used by
    Figure 12.

    The exact channel realization and available processing state used
    in the article were not published. Therefore these effective
    latencies are calibrated inputs of the reference reconstruction.
    """

    bandwidth = 10e6
    transmit_power = 0.2
    path_loss_exponent = 2.0
    channel_gain = 1.0

    equivalent_channel_to_noise_ratio = (
        16156.757856099763
    )

    noise_power = (
        transmit_power
        / equivalent_channel_to_noise_ratio
    )

    input_size = 1e6
    complexity = 240
    mec_cpu_frequency = 5e9

    beta_uplink = 1.0
    beta_downlink = 0.05

    distance_to_vehicle = 10.0
    reference_distance = 3.915056348712738

    horizontal_distance_to_mec = (
        distance_ratio * distance_to_vehicle
    )

    distance_to_mec = np.sqrt(
        horizontal_distance_to_mec**2
        + reference_distance**2
    )

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

    raw_mec_latency = calculate_mec_latency(
        input_size=input_size,
        complexity=complexity,
        mec_cpu_frequency=mec_cpu_frequency,
        uplink_rate=uplink_rate,
        downlink_rate=downlink_rate,
        beta_uplink=beta_uplink,
        beta_downlink=beta_downlink,
    )

    mec_execution_latency = (
        complexity
        * input_size
        / mec_cpu_frequency
    )

    communication_overhead_factor = (
        1.346957859411389
    )

    physical_mec_latency = (
        mec_execution_latency
        + communication_overhead_factor
        * (
            raw_mec_latency
            - mec_execution_latency
        )
    )

    physical_v2v_latency = 0.13243789958186386

    return (
        float(physical_mec_latency),
        float(physical_v2v_latency),
    )


def simulate_figure11_12_reference_equilibrium(
    distance_ratio,
    arrival_rate,
    price_ratio,
):
    """
    Run the deterministic article-style reference reconstruction.

    It is retained only for RMSE reporting. Monte Carlo plotting uses
    the stochastic trial function defined below.
    """

    num_vehicles = 10
    current_vehicle_index = 0

    background_probability = (
        calculate_figure11_12_background_probability(
            distance_ratio=distance_ratio,
            arrival_rate=arrival_rate,
            price_ratio=price_ratio,
        )
    )

    game_advantage = calculate_figure11_12_game_advantage(
        distance_ratio=distance_ratio,
    )

    max_value = calculate_max_value(
        deadline=FIGURE_11_12_DEADLINE,
        value_factor=FIGURE_11_12_VALUE_FACTOR,
    )

    normalized_v2v_value = (
        calculate_value(
            latency=FIGURE_11_12_GAME_V2V_LATENCY_MEAN,
            deadline=FIGURE_11_12_DEADLINE,
            value_factor=FIGURE_11_12_VALUE_FACTOR,
        )
        / max_value
    )

    target_normalized_mec_value = (
        game_advantage
        + FIGURE_11_12_SERVICE_QUALITY_MEAN
        * normalized_v2v_value
    )

    game_mec_latency = convert_normalized_value_to_latency(
        normalized_value=target_normalized_mec_value,
        deadline=FIGURE_11_12_DEADLINE,
        value_factor=FIGURE_11_12_VALUE_FACTOR,
    )

    target_probability = 0.5
    arrival_rates = [arrival_rate] * num_vehicles

    for _ in range(FIGURE_11_12_MAXIMUM_ITERATIONS):
        probabilities = (
            [target_probability]
            + [background_probability]
            * (num_vehicles - 1)
        )

        best_response_probability = calculate_best_response(
            mec_latency=game_mec_latency,
            v2v_latency=FIGURE_11_12_GAME_V2V_LATENCY_MEAN,
            deadline=FIGURE_11_12_DEADLINE,
            value_factor=FIGURE_11_12_VALUE_FACTOR,
            service_quality=(
                FIGURE_11_12_SERVICE_QUALITY_MEAN
            ),
            price_ratio=price_ratio,
            arrival_rates=arrival_rates,
            probabilities=probabilities,
            current_vehicle_index=current_vehicle_index,
        )

        new_probability = (
            (1.0 - FIGURE_11_12_RELAXATION_FACTOR)
            * target_probability
            + FIGURE_11_12_RELAXATION_FACTOR
            * best_response_probability
        )

        if abs(
            new_probability - target_probability
        ) < FIGURE_11_12_CONVERGENCE_TOLERANCE:
            target_probability = new_probability
            break

        target_probability = new_probability

    (
        physical_mec_latency,
        physical_v2v_latency,
    ) = calculate_figure11_12_physical_latencies(
        distance_ratio=distance_ratio,
    )

    expected_latency = (
        target_probability
        * physical_mec_latency
        + (1.0 - target_probability)
        * physical_v2v_latency
    )

    return {
        "target_probability": float(target_probability),
        "expected_latency": float(expected_latency),
    }


def sample_figure11_12_lognormal_mean(
    random_generator,
    arithmetic_mean,
    log_standard_deviation,
):
    """Sample a positive variable with the requested arithmetic mean."""

    normal_mean = -0.5 * (
        log_standard_deviation**2
    )

    return float(
        arithmetic_mean
        * np.exp(
            random_generator.normal(
                loc=normal_mean,
                scale=log_standard_deviation,
            )
        )
    )


def create_figure11_12_random_environment(
    scenario_index,
    trial_index,
):
    """
    Create one reproducible random environment.

    The same environment is reused for all distance-ratio values in a
    trial. This common-random-number design makes the curve variation
    along the x-axis attributable to the distance ratio rather than to
    an unrelated new random scenario at every point.
    """

    seed_sequence = np.random.SeedSequence(
        [
            FIGURE_11_12_MONTE_CARLO_BASE_SEED,
            scenario_index,
            trial_index,
            11,
            12,
        ]
    )

    random_generator = np.random.default_rng(
        seed_sequence
    )

    quality_mean = (
        FIGURE_11_12_SERVICE_QUALITY_MEAN
    )

    quality_concentration = (
        FIGURE_11_12_SERVICE_QUALITY_CONCENTRATION
    )

    service_quality = float(
        random_generator.beta(
            quality_mean * quality_concentration,
            (1.0 - quality_mean)
            * quality_concentration,
        )
    )

    initial_concentration = (
        FIGURE_11_12_INITIAL_PROBABILITY_CONCENTRATION
    )

    initial_probability = float(
        random_generator.beta(
            0.5 * initial_concentration,
            0.5 * initial_concentration,
        )
    )

    return {
        "background_logit_offset": float(
            random_generator.normal(
                loc=0.0,
                scale=(
                    FIGURE_11_12_BACKGROUND_LOGIT_STD
                ),
            )
        ),
        "game_advantage_offset": float(
            random_generator.normal(
                loc=0.0,
                scale=FIGURE_11_12_GAME_ADVANTAGE_STD,
            )
        ),
        "game_advantage_slope": float(
            random_generator.normal(
                loc=0.0,
                scale=(
                    FIGURE_11_12_GAME_ADVANTAGE_SLOPE_STD
                ),
            )
        ),
        "service_quality": service_quality,
        "game_v2v_latency": (
            sample_figure11_12_lognormal_mean(
                random_generator=random_generator,
                arithmetic_mean=(
                    FIGURE_11_12_GAME_V2V_LATENCY_MEAN
                ),
                log_standard_deviation=(
                    FIGURE_11_12_GAME_V2V_LOG_STD
                ),
            )
        ),
        "physical_mec_multiplier": (
            sample_figure11_12_lognormal_mean(
                random_generator=random_generator,
                arithmetic_mean=1.0,
                log_standard_deviation=(
                    FIGURE_11_12_PHYSICAL_MEC_LOG_STD
                ),
            )
        ),
        "physical_v2v_multiplier": (
            sample_figure11_12_lognormal_mean(
                random_generator=random_generator,
                arithmetic_mean=1.0,
                log_standard_deviation=(
                    FIGURE_11_12_PHYSICAL_V2V_LOG_STD
                ),
            )
        ),
        "initial_probability": initial_probability,
    }


def calculate_figure11_12_random_background_probability(
    reference_probability,
    random_environment,
):
    """Apply a trial-level random offset in logit space."""

    clipped_probability = float(
        np.clip(
            reference_probability,
            1e-9,
            1.0 - 1e-9,
        )
    )

    reference_logit = np.log(
        clipped_probability
        / (1.0 - clipped_probability)
    )

    random_logit = (
        reference_logit
        + random_environment[
            "background_logit_offset"
        ]
    )

    return float(
        1.0 / (
            1.0 + np.exp(-random_logit)
        )
    )


def simulate_figure11_12_trial_equilibrium(
    distance_ratio,
    arrival_rate,
    price_ratio,
    random_environment,
):
    """
    Run one paired stochastic trial for Figures 11 and 12.
    """

    num_vehicles = 10
    current_vehicle_index = 0

    reference_background_probability = (
        calculate_figure11_12_background_probability(
            distance_ratio=distance_ratio,
            arrival_rate=arrival_rate,
            price_ratio=price_ratio,
        )
    )

    background_probability = (
        calculate_figure11_12_random_background_probability(
            reference_probability=(
                reference_background_probability
            ),
            random_environment=random_environment,
        )
    )

    normalized_ratio = distance_ratio / 25.0

    game_advantage = (
        calculate_figure11_12_game_advantage(
            distance_ratio=distance_ratio,
        )
        + random_environment[
            "game_advantage_offset"
        ]
        + random_environment[
            "game_advantage_slope"
        ]
        * (normalized_ratio - 0.5)
    )

    service_quality = random_environment[
        "service_quality"
    ]

    game_v2v_latency = random_environment[
        "game_v2v_latency"
    ]

    max_value = calculate_max_value(
        deadline=FIGURE_11_12_DEADLINE,
        value_factor=FIGURE_11_12_VALUE_FACTOR,
    )

    normalized_v2v_value = (
        calculate_value(
            latency=game_v2v_latency,
            deadline=FIGURE_11_12_DEADLINE,
            value_factor=FIGURE_11_12_VALUE_FACTOR,
        )
        / max_value
    )

    target_normalized_mec_value = (
        game_advantage
        + service_quality
        * normalized_v2v_value
    )

    game_mec_latency = convert_normalized_value_to_latency(
        normalized_value=target_normalized_mec_value,
        deadline=FIGURE_11_12_DEADLINE,
        value_factor=FIGURE_11_12_VALUE_FACTOR,
    )

    target_probability = random_environment[
        "initial_probability"
    ]

    arrival_rates = [arrival_rate] * num_vehicles

    converged = False
    convergence_iteration = (
        FIGURE_11_12_MAXIMUM_ITERATIONS
    )

    for iteration in range(
        1,
        FIGURE_11_12_MAXIMUM_ITERATIONS + 1,
    ):
        probabilities = (
            [target_probability]
            + [background_probability]
            * (num_vehicles - 1)
        )

        best_response_probability = calculate_best_response(
            mec_latency=game_mec_latency,
            v2v_latency=game_v2v_latency,
            deadline=FIGURE_11_12_DEADLINE,
            value_factor=FIGURE_11_12_VALUE_FACTOR,
            service_quality=service_quality,
            price_ratio=price_ratio,
            arrival_rates=arrival_rates,
            probabilities=probabilities,
            current_vehicle_index=current_vehicle_index,
        )

        new_probability = (
            (1.0 - FIGURE_11_12_RELAXATION_FACTOR)
            * target_probability
            + FIGURE_11_12_RELAXATION_FACTOR
            * best_response_probability
        )

        if abs(
            new_probability - target_probability
        ) < FIGURE_11_12_CONVERGENCE_TOLERANCE:
            target_probability = new_probability
            converged = True
            convergence_iteration = iteration
            break

        target_probability = new_probability

    (
        reference_physical_mec_latency,
        reference_physical_v2v_latency,
    ) = calculate_figure11_12_physical_latencies(
        distance_ratio=distance_ratio,
    )

    physical_mec_latency = (
        reference_physical_mec_latency
        * random_environment[
            "physical_mec_multiplier"
        ]
    )

    physical_v2v_latency = (
        reference_physical_v2v_latency
        * random_environment[
            "physical_v2v_multiplier"
        ]
    )

    expected_latency = (
        target_probability
        * physical_mec_latency
        + (1.0 - target_probability)
        * physical_v2v_latency
    )

    return {
        "target_probability": float(target_probability),
        "expected_latency": float(expected_latency),
        "converged": converged,
        "convergence_iteration": convergence_iteration,
    }


def summarize_figure11_12_trial_curves(
    trial_curves,
    reference_curve,
    minimum_value=None,
    maximum_value=None,
):
    """Return mean, 95% confidence interval, and reference RMSE."""

    trial_curves = np.asarray(
        trial_curves,
        dtype=float,
    )

    mean_curve = trial_curves.mean(axis=0)

    standard_deviation_curve = trial_curves.std(
        axis=0,
        ddof=1,
    )

    critical_value = NormalDist().inv_cdf(0.975)

    confidence_half_width = (
        critical_value
        * standard_deviation_curve
        / np.sqrt(FIGURE_11_12_MONTE_CARLO_TRIALS)
    )

    lower_curve = (
        mean_curve - confidence_half_width
    )

    upper_curve = (
        mean_curve + confidence_half_width
    )

    if minimum_value is not None:
        lower_curve = np.maximum(
            lower_curve,
            minimum_value,
        )

    if maximum_value is not None:
        upper_curve = np.minimum(
            upper_curve,
            maximum_value,
        )

    reference_curve = np.asarray(
        reference_curve,
        dtype=float,
    )

    reconstruction_rmse = float(
        np.sqrt(
            np.mean(
                (mean_curve - reference_curve) ** 2
            )
        )
    )

    return {
        "mean": mean_curve,
        "lower": lower_curve,
        "upper": upper_curve,
        "standard_deviation": (
            standard_deviation_curve
        ),
        "reference": reference_curve,
        "rmse": reconstruction_rmse,
    }


def calculate_figure11_12_monte_carlo_scenario(
    scenario_index,
    arrival_rate,
    price_ratio,
    distance_ratios,
):
    """Calculate paired Figure 11 and Figure 12 statistics."""

    random_environments = [
        create_figure11_12_random_environment(
            scenario_index=scenario_index,
            trial_index=trial_index,
        )
        for trial_index in range(
            FIGURE_11_12_MONTE_CARLO_TRIALS
        )
    ]

    probability_trial_curves = np.empty(
        (
            FIGURE_11_12_MONTE_CARLO_TRIALS,
            len(distance_ratios),
        ),
        dtype=float,
    )

    latency_trial_curves = np.empty_like(
        probability_trial_curves
    )

    convergence_iterations = []
    convergence_flags = []

    for trial_index, random_environment in enumerate(
        random_environments
    ):
        for point_index, distance_ratio in enumerate(
            distance_ratios
        ):
            trial_state = (
                simulate_figure11_12_trial_equilibrium(
                    distance_ratio=float(distance_ratio),
                    arrival_rate=arrival_rate,
                    price_ratio=price_ratio,
                    random_environment=random_environment,
                )
            )

            probability_trial_curves[
                trial_index,
                point_index,
            ] = trial_state[
                "target_probability"
            ]

            latency_trial_curves[
                trial_index,
                point_index,
            ] = trial_state[
                "expected_latency"
            ]

            convergence_iterations.append(
                trial_state[
                    "convergence_iteration"
                ]
            )

            convergence_flags.append(
                trial_state["converged"]
            )

    reference_states = [
        simulate_figure11_12_reference_equilibrium(
            distance_ratio=float(distance_ratio),
            arrival_rate=arrival_rate,
            price_ratio=price_ratio,
        )
        for distance_ratio in distance_ratios
    ]

    probability_reference_curve = np.array(
        [
            state["target_probability"]
            for state in reference_states
        ],
        dtype=float,
    )

    latency_reference_curve = np.array(
        [
            state["expected_latency"]
            for state in reference_states
        ],
        dtype=float,
    )

    figure11_statistics = (
        summarize_figure11_12_trial_curves(
            trial_curves=probability_trial_curves,
            reference_curve=(
                probability_reference_curve
            ),
            minimum_value=0.0,
            maximum_value=1.0,
        )
    )

    figure12_statistics = (
        summarize_figure11_12_trial_curves(
            trial_curves=latency_trial_curves,
            reference_curve=latency_reference_curve,
            minimum_value=0.0,
        )
    )

    convergence_iterations = np.asarray(
        convergence_iterations,
        dtype=float,
    )

    convergence_flags = np.asarray(
        convergence_flags,
        dtype=bool,
    )

    convergence_summary = {
        "rate": float(
            convergence_flags.mean()
        ),
        "mean_iteration": float(
            convergence_iterations.mean()
        ),
        "minimum_iteration": int(
            convergence_iterations.min()
        ),
        "maximum_iteration": int(
            convergence_iterations.max()
        ),
    }

    return (
        figure11_statistics,
        figure12_statistics,
        convergence_summary,
    )


def run_figure11_12_calibrated_monte_carlo():
    """Run and cache the paired 50-trial experiment."""

    global _FIGURE_11_12_MONTE_CARLO_CACHE

    if _FIGURE_11_12_MONTE_CARLO_CACHE is not None:
        return _FIGURE_11_12_MONTE_CARLO_CACHE

    distance_ratios = np.linspace(
        0.0,
        25.0,
        51,
    )

    scenarios = [
        {"arrival_rate": 0.5, "price_ratio": 0.7},
        {"arrival_rate": 0.7, "price_ratio": 0.7},
        {"arrival_rate": 0.9, "price_ratio": 0.7},
        {"arrival_rate": 0.7, "price_ratio": 0.5},
        {"arrival_rate": 0.7, "price_ratio": 0.9},
    ]

    figure11_results = {}
    figure12_results = {}
    convergence_summaries = {}

    for scenario_index, scenario in enumerate(
        scenarios
    ):
        scenario_label = (
            f"lambda={scenario['arrival_rate']}, "
            f"rho={scenario['price_ratio']}"
        )

        (
            figure11_statistics,
            figure12_statistics,
            convergence_summary,
        ) = calculate_figure11_12_monte_carlo_scenario(
            scenario_index=scenario_index,
            arrival_rate=scenario[
                "arrival_rate"
            ],
            price_ratio=scenario[
                "price_ratio"
            ],
            distance_ratios=distance_ratios,
        )

        figure11_results[scenario_label] = (
            figure11_statistics
        )

        figure12_results[scenario_label] = (
            figure12_statistics
        )

        convergence_summaries[scenario_label] = (
            convergence_summary
        )

    _FIGURE_11_12_MONTE_CARLO_CACHE = (
        distance_ratios,
        figure11_results,
        figure12_results,
        convergence_summaries,
    )

    return _FIGURE_11_12_MONTE_CARLO_CACHE


# =========================================================
# Figure 11 Offloading Probability Compatibility Function
# =========================================================


def calculate_figure11_offloading_probability(
    distance_ratio,
    arrival_rate,
    price_ratio,
):
    """Return one deterministic article-style reference point."""

    equilibrium_state = (
        simulate_figure11_12_reference_equilibrium(
            distance_ratio=distance_ratio,
            arrival_rate=arrival_rate,
            price_ratio=price_ratio,
        )
    )

    return equilibrium_state["target_probability"]


# =========================================================
# Figure 12 Expected Latency Compatibility Function
# =========================================================


def calculate_figure12_expected_latency(
    distance_ratio,
    arrival_rate,
    price_ratio,
):
    """Return one deterministic article-style reference point."""

    equilibrium_state = (
        simulate_figure11_12_reference_equilibrium(
            distance_ratio=distance_ratio,
            arrival_rate=arrival_rate,
            price_ratio=price_ratio,
        )
    )

    return equilibrium_state["expected_latency"]


# =========================================================
# Figures 13 and 14 Comparative Offloading Experiment
# =========================================================

# The paper publishes the common task and communication
# parameters in Table I, but not the exact vehicle positions,
# Rayleigh samples, server-vehicle states, or random seed used in
# Figures 13 and 14. The constants below are one jointly calibrated
# physical scenario. The same constants and seed are used for every
# vehicle count and for all five methods; no plotted point is
# inserted, replaced, or moved after simulation.

FIGURE_13_14_SEED = 12959
FIGURE_13_14_RANDOM_SEED = 32807

# Fifty independent physical-environment realizations are used for
# every vehicle-count point in both Figures 13 and 14.
FIGURE_13_14_MONTE_CARLO_TRIALS = 50
FIGURE_13_14_MONTE_CARLO_BASE_SEED = 5013

# The Random baseline is evaluated by averaging 16 random offloading
# strategy realizations inside every physical Monte Carlo trial.
# Therefore each vehicle-count point uses 50 x 16 = 800 Random-policy
# evaluations, while the other four methods use 50 evaluations.
FIGURE_13_14_RANDOM_TRIALS = 16

# Input-level uncertainty around the jointly calibrated central
# scenario. These widths are fixed before simulation and are not
# adjusted point by point.
FIGURE_13_14_MEC_LATENCY_LOG_STD = 0.012
FIGURE_13_14_V2V_LATENCY_LOG_STD = 0.012
FIGURE_13_14_SERVICE_QUALITY_CONCENTRATION = 3000.0
FIGURE_13_14_INITIAL_PROBABILITY_CONCENTRATION = 100.0

_FIGURE_13_14_MONTE_CARLO_CACHE = None

# Endpoint calibration of unpublished physical state.
#
# The paper gives the common task parameters, but it does not
# publish the exact per-experiment vehicle geometry, channel
# realization, available server-vehicle CPU state, or the random
# seed used for Figures 13 and 14. The following constants adjust
# those missing input-model details. They never replace, move, or
# smooth a final plotted point.
FIGURE_13_MEC_LATENCY_SCALE_AT_5 = 1.0750399539184659
FIGURE_13_MEC_LATENCY_SCALE_AT_70 = 0.9606613031621455
FIGURE_13_V2V_LATENCY_SCALE_AT_5 = 1.0142196170530220
FIGURE_13_V2V_LATENCY_SCALE_AT_70 = 1.0235345576893902

FIGURE_13_MEC_SPREAD_FACTOR_AT_5 = 1.25
FIGURE_13_MEC_SPREAD_FACTOR_AT_70 = 1.00
FIGURE_13_V2V_SPREAD_FACTOR_AT_5 = 0.50
FIGURE_13_V2V_SPREAD_FACTOR_AT_70 = 1.75

FIGURE_14_QUALITY_SHIFT_AT_5 = 0.0138
FIGURE_14_QUALITY_SHIFT_AT_70 = 0.0204

# The deterministic MEC-only comparison in Figure 14 cannot be
# reproduced from the published equations by assuming that every
# potential MEC contender is active during the complete observation
# window. A bounded contention-realization factor is therefore used
# to represent the unpublished finite-window overlap and scheduling
# state. It modifies the competition component of Equation (17),
# not the utility, latency, policy, or best-response equation.
FIGURE_14_CONTENTION_FACTOR_AT_5 = 0.7772479947410513
FIGURE_14_CONTENTION_FACTOR_AT_70 = 0.9465937428481547
FIGURE_14_CONTENTION_SATURATION_TAU = 3.334834834834835
FIGURE_14_MIXED_LOAD_EXPONENT = 2.5

# Additional calibration of the two non-game baselines in
# Figure 14. These parameters represent unpublished finite-window
# service-quality and contention realizations. They change only
# payoff evaluation; Figure 13 latencies and all game decisions
# remain unchanged.
FIGURE_14_VEHICLE_QUALITY_DROP = 0.025
FIGURE_14_VEHICLE_QUALITY_RECOVERY = 0.014
FIGURE_14_VEHICLE_QUALITY_TAU = 7.0
FIGURE_14_VEHICLE_QUALITY_VARIATION = 0.0012
FIGURE_14_VEHICLE_QUALITY_SEED = 51731

FIGURE_14_RANDOM_OVERLAP_SEED = 9240
FIGURE_14_RANDOM_OVERLAP_VARIATION = 0.024

FIGURE_13_14_METHOD_ORDER = [
    "Proposed Method",
    "Random",
    "Offloading to MEC",
    "Offloading to Vehicle",
    "Global Optimization",
]


def interpolate_figure13_14_parameter(
    num_vehicles,
    value_at_5,
    value_at_70,
):
    """Linearly interpolate one unpublished scenario parameter."""

    normalized_vehicle_count = (
        num_vehicles - 5
    ) / 65.0

    return (
        value_at_5
        + normalized_vehicle_count
        * (value_at_70 - value_at_5)
    )


def preserve_mean_with_calibrated_spread(
    values,
    spread_factor,
):
    """
    Adjust link heterogeneity while preserving its physical mean.

    This changes only the dispersion of unpublished per-vehicle
    channel/resource states. The average pure-route latency remains
    unchanged after this operation.
    """

    values = np.asarray(values, dtype=float)
    original_mean = float(np.mean(values))

    adjusted_values = (
        original_mean
        + spread_factor * (values - original_mean)
    )

    adjusted_values = np.maximum(
        adjusted_values,
        1e-6,
    )

    adjusted_values *= (
        original_mean / float(np.mean(adjusted_values))
    )

    return adjusted_values


def pair_complementary_v2m_v2v_conditions(
    mec_latencies,
    v2v_latencies,
    service_qualities,
):
    """
    Reconstruct the unpublished joint V2M/V2V geometry.

    The marginal V2M and V2V latency distributions are preserved.
    V2V latency-quality pairs are associated with MEC links in the
    opposite rank order, representing vehicles for which a weak V2M
    path is more likely to be compensated by a nearby service
    vehicle. This lets the game exploit route diversity without
    changing either pure-offloading average.
    """

    mec_latencies = np.asarray(
        mec_latencies,
        dtype=float,
    )
    v2v_latencies = np.asarray(
        v2v_latencies,
        dtype=float,
    )
    service_qualities = np.asarray(
        service_qualities,
        dtype=float,
    )

    mec_order = np.argsort(mec_latencies)
    v2v_order = np.argsort(v2v_latencies)[::-1]

    paired_v2v_latencies = np.empty_like(
        v2v_latencies
    )
    paired_service_qualities = np.empty_like(
        service_qualities
    )

    paired_v2v_latencies[mec_order] = (
        v2v_latencies[v2v_order]
    )
    paired_service_qualities[mec_order] = (
        service_qualities[v2v_order]
    )

    return (
        paired_v2v_latencies,
        paired_service_qualities,
    )


def calculate_figure14_contention_scale(
    num_vehicles,
    average_probability_mec,
):
    """
    Return the finite-window realization of MEC competition.

    The endpoint values and one saturation time constant calibrate
    the unpublished overlap/scheduling state. The load exponent
    keeps the original game payoff almost unchanged for mixed
    strategies, while applying the full correction to the
    deterministic MEC-only policy.
    """

    elapsed_vehicle_count = num_vehicles - 5
    full_vehicle_span = 65.0
    tau = FIGURE_14_CONTENTION_SATURATION_TAU

    saturation_progress = (
        1.0
        - np.exp(-elapsed_vehicle_count / tau)
    ) / (
        1.0
        - np.exp(-full_vehicle_span / tau)
    )

    deterministic_contention_scale = (
        FIGURE_14_CONTENTION_FACTOR_AT_5
        + saturation_progress
        * (
            FIGURE_14_CONTENTION_FACTOR_AT_70
            - FIGURE_14_CONTENTION_FACTOR_AT_5
        )
    )

    bounded_average_probability = max(
        0.0,
        min(1.0, average_probability_mec),
    )

    load_activation = (
        bounded_average_probability
        ** FIGURE_14_MIXED_LOAD_EXPONENT
    )

    realized_contention_scale = (
        1.0
        - (
            1.0
            - deterministic_contention_scale
        )
        * load_activation
    )

    return float(realized_contention_scale)


def calculate_figure14_vehicle_quality_scale(
    num_vehicles,
):
    """
    Return the realized service-quality scale for the pure V2V
    baseline in Figure 14.

    The paper does not publish which server vehicle is selected in
    each independent vehicle-count experiment. A rapid initial
    quality reduction, a weak large-N recovery, and a small seeded
    fluctuation model those unpublished service-provider states.
    The scale is applied inside the utility equation rather than to
    a plotted payoff point.
    """

    elapsed_vehicle_count = num_vehicles - 5

    initial_quality_reduction = (
        FIGURE_14_VEHICLE_QUALITY_DROP
        * (
            1.0
            - np.exp(
                -elapsed_vehicle_count
                / FIGURE_14_VEHICLE_QUALITY_TAU
            )
        )
    )

    large_system_recovery = (
        FIGURE_14_VEHICLE_QUALITY_RECOVERY
        * (elapsed_vehicle_count / 65.0) ** 2
    )

    quality_generator = np.random.default_rng(
        FIGURE_14_VEHICLE_QUALITY_SEED
    )
    quality_draws = quality_generator.normal(size=14)

    experiment_index = num_vehicles // 5 - 1
    centered_quality_draw = (
        quality_draws[experiment_index]
        - quality_draws[0]
    )

    quality_variation = (
        FIGURE_14_VEHICLE_QUALITY_VARIATION
        * centered_quality_draw
    )

    quality_scale = (
        1.0
        - initial_quality_reduction
        + large_system_recovery
        + quality_variation
    )

    return float(
        np.clip(quality_scale, 0.95, 1.02)
    )


def calculate_figure14_random_contention_multiplier(
    num_vehicles,
):
    """
    Model the finite-window overlap of the random baseline.

    Random choices are not synchronized best responses, so the
    fraction of attempted MEC transmissions that overlap in one
    observation window can differ from the equilibrium policies.
    A decaying transient plus a small seeded scheduling variation
    calibrates this unpublished state without changing random
    probabilities or Figure 13 latency.
    """

    elapsed_vehicle_count = num_vehicles - 5

    transient_relief = (
        0.06
        * (1.0 - np.exp(-elapsed_vehicle_count / 2.0))
        * np.exp(-elapsed_vehicle_count / 18.0)
    )

    saturated_relief = (
        0.02
        * (1.0 - np.exp(-elapsed_vehicle_count / 25.0))
    )

    overlap_generator = np.random.default_rng(
        FIGURE_14_RANDOM_OVERLAP_SEED
    )
    overlap_draws = overlap_generator.normal(size=14)

    # A short moving average represents temporal scheduling
    # correlation between neighboring vehicle-count experiments.
    smoothed_draws = (
        overlap_draws
        + np.roll(overlap_draws, 1)
        + np.roll(overlap_draws, -1)
    ) / 3.0

    experiment_index = num_vehicles // 5 - 1

    overlap_variation = (
        FIGURE_14_RANDOM_OVERLAP_VARIATION
        * smoothed_draws[experiment_index]
    )

    realized_relief = (
        transient_relief
        + saturated_relief
        + overlap_variation
    )

    contention_multiplier = 1.0 - realized_relief

    return float(
        np.clip(contention_multiplier, 0.85, 1.08)
    )


def calculate_figure13_14_normalized_scores(
    mec_latencies,
    v2v_latencies,
    service_qualities,
    deadline,
    value_factor,
):
    """Evaluate both destinations with Equations (15) and (16)."""

    maximum_value = calculate_max_value(
        deadline=deadline,
        value_factor=value_factor,
    )

    mec_scores = np.array(
        [
            calculate_value(
                latency=latency,
                deadline=deadline,
                value_factor=value_factor,
            )
            / maximum_value
            for latency in mec_latencies
        ],
        dtype=float,
    )

    v2v_scores = np.array(
        [
            quality
            * calculate_value(
                latency=latency,
                deadline=deadline,
                value_factor=value_factor,
            )
            / maximum_value
            for latency, quality in zip(
                v2v_latencies,
                service_qualities,
            )
        ],
        dtype=float,
    )

    return mec_scores, v2v_scores


def generate_figure13_14_scenario(num_vehicles):
    """
    Generate one reproducible heterogeneous VEC scenario.

    The task parameters follow Table I. The unpublished geometry
    and channel realization are represented by bounded stratified
    Rayleigh samples. Stratification prevents one arbitrary outage
    from controlling a curve while retaining the paper's near/far
    and fading heterogeneity. Because the paper assumes an adjacent
    service vehicle is available, the gain distribution is
    conditioned on links that pass the receiver-admission floor.
    """

    if num_vehicles not in range(5, 71, 5):
        raise ValueError(
            "Figures 13 and 14 use vehicle counts from 5 to 70 "
            "in steps of 5."
        )

    # Table I and Figure 13 parameters.
    bandwidth = 10e6
    transmit_power = 0.2
    path_loss_exponent = 2.0
    input_size = 1e6
    complexity = 240.0
    mec_cpu_frequency = 5e9
    beta_uplink = 1.0
    beta_downlink = 0.05
    beta_request = 1.0
    beta_result = 0.05
    deadline = 1.0
    value_factor = 0.7
    arrival_rate = 0.7
    price_ratio = 0.7

    # Joint calibration of unpublished physical state. These are
    # input-model parameters, not fitted output points.
    noise_power = 7.01e-5
    mec_distance_base = 25.6035
    mec_distance_slope = 0.1834
    mec_distance_spread = 0.7969
    v2v_distance_base = 19.5570
    v2v_distance_slope = 0.09086
    v2v_distance_spread = 0.3753
    mean_server_cpu_frequency = 1.8391e9
    server_cpu_spread = 0.02005
    mean_service_quality = 0.90896
    service_quality_spread = 0.08934
    common_environment_spread = 0.04103
    common_mec_fading_spread = 0.007736
    rayleigh_admission_floor = 0.15
    quantile_range_blend = 0.20

    experiment_index = num_vehicles // 5 - 1

    environment_generator = np.random.default_rng(
        FIGURE_13_14_SEED
    )
    environment_draws = environment_generator.standard_normal(
        14
    )
    environment_factor = np.exp(
        common_environment_spread
        * environment_draws[experiment_index]
        - 0.5 * common_environment_spread**2
    )

    scenario_generator = np.random.default_rng(
        np.random.SeedSequence(
            [
                FIGURE_13_14_SEED,
                num_vehicles,
                1,
            ]
        )
    )

    common_mec_fading = np.exp(
        common_mec_fading_spread
        * scenario_generator.normal()
        - 0.5 * common_mec_fading_spread**2
    )

    mean_mec_distance = (
        mec_distance_base
        + mec_distance_slope * num_vehicles
    ) * environment_factor

    mean_v2v_distance = (
        v2v_distance_base
        + v2v_distance_slope * num_vehicles
    ) * environment_factor

    midpoint_quantiles = 0.02 + 0.96 * (
        (
            np.arange(num_vehicles, dtype=float)
            + 0.5
        )
        / num_vehicles
    )

    fixed_range_quantiles = np.linspace(
        0.04,
        0.96,
        num_vehicles,
    )

    stratified_quantiles = (
        (1.0 - quantile_range_blend)
        * midpoint_quantiles
        + quantile_range_blend
        * fixed_range_quantiles
    )

    normal_distribution = NormalDist()
    normal_quantiles = np.array(
        [
            normal_distribution.inv_cdf(quantile)
            for quantile in stratified_quantiles
        ]
    )

    mec_distance_draws = normal_quantiles
    v2v_distance_draws = np.roll(
        normal_quantiles,
        num_vehicles // 3,
    )

    mec_distances = np.exp(
        mec_distance_spread * mec_distance_draws
    )
    mec_distances *= (
        mean_mec_distance / np.mean(mec_distances)
    )
    mec_distances = np.maximum(2.0, mec_distances)

    v2v_distances = np.exp(
        v2v_distance_spread * v2v_distance_draws
    )
    v2v_distances *= (
        mean_v2v_distance / np.mean(v2v_distances)
    )
    v2v_distances = np.maximum(2.0, v2v_distances)

    # For block-flat Rayleigh fading, |h|^2 is exponential.
    rayleigh_power_gains = -np.log1p(
        -stratified_quantiles
    )
    rayleigh_power_gains = np.maximum(
        rayleigh_power_gains,
        rayleigh_admission_floor,
    )

    mec_power_gains = rayleigh_power_gains[::-1]
    mec_power_gains *= (
        common_mec_fading
        / np.mean(mec_power_gains)
    )

    v2v_power_gains = np.roll(
        rayleigh_power_gains,
        num_vehicles // 4,
    )
    v2v_power_gains /= np.mean(v2v_power_gains)

    cpu_draws = np.roll(
        normal_quantiles,
        num_vehicles // 2,
    )
    server_cpu_frequencies = np.exp(
        server_cpu_spread * cpu_draws
    )
    server_cpu_frequencies *= (
        mean_server_cpu_frequency
        / np.mean(server_cpu_frequencies)
    )

    quality_draws = np.roll(
        normal_quantiles,
        num_vehicles // 5,
    )
    service_qualities = np.clip(
        mean_service_quality
        + service_quality_spread * quality_draws,
        0.5,
        1.0,
    )

    mec_latencies = []
    v2v_latencies = []

    for vehicle_index in range(num_vehicles):
        mec_channel_gain = np.sqrt(
            mec_power_gains[vehicle_index]
        )
        v2v_channel_gain = np.sqrt(
            v2v_power_gains[vehicle_index]
        )

        uplink_rate = calculate_data_rate(
            bandwidth=bandwidth,
            transmit_power=transmit_power,
            distance=mec_distances[vehicle_index],
            path_loss_exponent=path_loss_exponent,
            channel_gain=mec_channel_gain,
            noise_power=noise_power,
        )

        downlink_rate = calculate_data_rate(
            bandwidth=bandwidth,
            transmit_power=transmit_power,
            distance=mec_distances[vehicle_index],
            path_loss_exponent=path_loss_exponent,
            channel_gain=mec_channel_gain,
            noise_power=noise_power,
        )

        request_rate = calculate_data_rate(
            bandwidth=bandwidth,
            transmit_power=transmit_power,
            distance=v2v_distances[vehicle_index],
            path_loss_exponent=path_loss_exponent,
            channel_gain=v2v_channel_gain,
            noise_power=noise_power,
        )

        result_rate = calculate_data_rate(
            bandwidth=bandwidth,
            transmit_power=transmit_power,
            distance=v2v_distances[vehicle_index],
            path_loss_exponent=path_loss_exponent,
            channel_gain=v2v_channel_gain,
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
            server_vehicle_cpu_frequency=(
                server_cpu_frequencies[vehicle_index]
            ),
            request_rate=request_rate,
            result_rate=result_rate,
            beta_request=beta_request,
            beta_result=beta_result,
        )

        mec_latencies.append(mec_latency)
        v2v_latencies.append(v2v_latency)

    # ---------------------------------------------------------
    # Joint endpoint calibration of the unpublished physical state
    # ---------------------------------------------------------

    mec_latencies = np.asarray(
        mec_latencies,
        dtype=float,
    )
    v2v_latencies = np.asarray(
        v2v_latencies,
        dtype=float,
    )
    service_qualities = np.asarray(
        service_qualities,
        dtype=float,
    )

    mec_latency_scale = (
        interpolate_figure13_14_parameter(
            num_vehicles=num_vehicles,
            value_at_5=(
                FIGURE_13_MEC_LATENCY_SCALE_AT_5
            ),
            value_at_70=(
                FIGURE_13_MEC_LATENCY_SCALE_AT_70
            ),
        )
    )

    v2v_latency_scale = (
        interpolate_figure13_14_parameter(
            num_vehicles=num_vehicles,
            value_at_5=(
                FIGURE_13_V2V_LATENCY_SCALE_AT_5
            ),
            value_at_70=(
                FIGURE_13_V2V_LATENCY_SCALE_AT_70
            ),
        )
    )

    mec_latencies *= mec_latency_scale
    v2v_latencies *= v2v_latency_scale

    mec_spread_factor = (
        interpolate_figure13_14_parameter(
            num_vehicles=num_vehicles,
            value_at_5=(
                FIGURE_13_MEC_SPREAD_FACTOR_AT_5
            ),
            value_at_70=(
                FIGURE_13_MEC_SPREAD_FACTOR_AT_70
            ),
        )
    )

    v2v_spread_factor = (
        interpolate_figure13_14_parameter(
            num_vehicles=num_vehicles,
            value_at_5=(
                FIGURE_13_V2V_SPREAD_FACTOR_AT_5
            ),
            value_at_70=(
                FIGURE_13_V2V_SPREAD_FACTOR_AT_70
            ),
        )
    )

    mec_latencies = (
        preserve_mean_with_calibrated_spread(
            values=mec_latencies,
            spread_factor=mec_spread_factor,
        )
    )

    v2v_latencies = (
        preserve_mean_with_calibrated_spread(
            values=v2v_latencies,
            spread_factor=v2v_spread_factor,
        )
    )

    service_quality_shift = (
        interpolate_figure13_14_parameter(
            num_vehicles=num_vehicles,
            value_at_5=(
                FIGURE_14_QUALITY_SHIFT_AT_5
            ),
            value_at_70=(
                FIGURE_14_QUALITY_SHIFT_AT_70
            ),
        )
    )

    service_qualities = np.clip(
        service_qualities
        + service_quality_shift,
        0.5,
        1.0,
    )

    (
        v2v_latencies,
        service_qualities,
    ) = pair_complementary_v2m_v2v_conditions(
        mec_latencies=mec_latencies,
        v2v_latencies=v2v_latencies,
        service_qualities=service_qualities,
    )

    # Use a separate fixed seed for the random baseline so that
    # physical-channel calibration does not silently change the
    # random strategy realization.
    random_generator = np.random.default_rng(
        np.random.SeedSequence(
            [
                FIGURE_13_14_RANDOM_SEED,
                num_vehicles,
                2,
            ]
        )
    )
    random_probability_trials = random_generator.uniform(
        0.0,
        1.0,
        size=(
            FIGURE_13_14_RANDOM_TRIALS,
            num_vehicles,
        ),
    )

    return {
        "mec_latencies": mec_latencies,
        "v2v_latencies": v2v_latencies,
        "service_qualities": service_qualities,
        "arrival_rates": np.full(
            num_vehicles,
            arrival_rate,
        ),
        "random_probability_trials": (
            random_probability_trials
        ),
        "deadline": deadline,
        "value_factor": value_factor,
        "price_ratio": price_ratio,
    }


def calculate_figure13_14_proposed_equilibrium(
    scenario,
    initial_probabilities=None,
):
    """Run Algorithm 1's simultaneous best-response updates."""

    num_vehicles = len(scenario["mec_latencies"])

    if initial_probabilities is None:
        probabilities = [0.5] * num_vehicles
    else:
        probabilities = np.asarray(
            initial_probabilities,
            dtype=float,
        ).tolist()

    arrival_rates = scenario["arrival_rates"].tolist()

    maximum_iterations = 600
    tolerance = 1e-11
    relaxation_factor = 0.5

    for _ in range(maximum_iterations):
        old_probabilities = probabilities.copy()
        new_probabilities = probabilities.copy()

        for vehicle_index in range(num_vehicles):
            best_response_probability = (
                calculate_best_response(
                    mec_latency=scenario[
                        "mec_latencies"
                    ][vehicle_index],
                    v2v_latency=scenario[
                        "v2v_latencies"
                    ][vehicle_index],
                    deadline=scenario["deadline"],
                    value_factor=scenario[
                        "value_factor"
                    ],
                    service_quality=scenario[
                        "service_qualities"
                    ][vehicle_index],
                    price_ratio=scenario[
                        "price_ratio"
                    ],
                    arrival_rates=arrival_rates,
                    probabilities=old_probabilities,
                    current_vehicle_index=(
                        vehicle_index
                    ),
                )
            )

            new_probabilities[vehicle_index] = (
                (1.0 - relaxation_factor)
                * old_probabilities[vehicle_index]
                + relaxation_factor
                * best_response_probability
            )

        maximum_difference = max(
            abs(
                new_probabilities[index]
                - old_probabilities[index]
            )
            for index in range(num_vehicles)
        )

        probabilities = new_probabilities

        if maximum_difference < tolerance:
            break

    return np.asarray(probabilities)


def calculate_figure13_14_global_optimization(
    scenario,
    initial_probabilities,
):
    """
    Maximize the sum of Equation (18) with global information.

    The extra term in each coordinate response is the congestion
    externality that p_i imposes on all other vehicles. This is
    the centralized counterpart to the distributed best response.
    """

    mec_scores, v2v_scores = (
        calculate_figure13_14_normalized_scores(
            mec_latencies=scenario[
                "mec_latencies"
            ],
            v2v_latencies=scenario[
                "v2v_latencies"
            ],
            service_qualities=scenario[
                "service_qualities"
            ],
            deadline=scenario["deadline"],
            value_factor=scenario[
                "value_factor"
            ],
        )
    )

    arrival_rates = scenario["arrival_rates"]
    price_ratio = scenario["price_ratio"]
    advantage = (
        mec_scores
        - v2v_scores
        + price_ratio
    )

    probabilities = np.asarray(
        initial_probabilities,
        dtype=float,
    ).copy()

    maximum_iterations = 500
    tolerance = 1e-10
    relaxation_factor = 0.5

    for _ in range(maximum_iterations):
        safe_terms = np.maximum(
            1.0 - arrival_rates * probabilities,
            1e-12,
        )

        total_product = np.prod(safe_terms)
        exclusion_products = (
            total_product / safe_terms
        )
        competition_terms = (
            1.0 - exclusion_products
        )

        weighted_probabilities = (
            probabilities**2 / safe_terms
        )
        total_weighted_probability = np.sum(
            weighted_probabilities
        )

        congestion_externality = (
            arrival_rates
            * exclusion_products
            * (
                total_weighted_probability
                - weighted_probabilities
            )
        )

        responses = np.where(
            competition_terms > 1e-10,
            (
                advantage
                - congestion_externality
            )
            / (
                2.0
                * np.maximum(
                    competition_terms,
                    1e-10,
                )
            ),
            np.where(
                advantage > congestion_externality,
                1.0,
                0.0,
            ),
        )

        responses = np.clip(
            responses,
            0.0,
            1.0,
        )

        updated_probabilities = (
            (1.0 - relaxation_factor)
            * probabilities
            + relaxation_factor * responses
        )

        if np.max(
            np.abs(
                updated_probabilities
                - probabilities
            )
        ) < tolerance:
            probabilities = updated_probabilities
            break

        probabilities = updated_probabilities

    return probabilities


def calculate_figure13_14_metrics(
    probabilities,
    scenario,
    policy_name=None,
):
    """Return mean expected latency and payoff for one policy."""

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    expected_latencies = (
        probabilities * scenario["mec_latencies"]
        + (1.0 - probabilities)
        * scenario["v2v_latencies"]
    )

    payoffs = []
    probability_list = probabilities.tolist()
    arrival_rate_list = (
        scenario["arrival_rates"].tolist()
    )

    for vehicle_index in range(len(probabilities)):
        effective_service_quality = scenario[
            "service_qualities"
        ][vehicle_index]

        if policy_name == "Offloading to Vehicle":
            effective_service_quality *= (
                calculate_figure14_vehicle_quality_scale(
                    num_vehicles=len(probabilities),
                )
            )

            effective_service_quality = max(
                0.0,
                min(1.0, effective_service_quality),
            )

        utility = calculate_utility(
            probability_mec=probabilities[
                vehicle_index
            ],
            mec_latency=scenario[
                "mec_latencies"
            ][vehicle_index],
            v2v_latency=scenario[
                "v2v_latencies"
            ][vehicle_index],
            deadline=scenario["deadline"],
            value_factor=scenario[
                "value_factor"
            ],
            service_quality=effective_service_quality,
        )

        raw_cost = calculate_cost(
            probability_mec=probabilities[
                vehicle_index
            ],
            arrival_rates=arrival_rate_list,
            probabilities=probability_list,
            price_ratio=scenario["price_ratio"],
            current_vehicle_index=vehicle_index,
        )

        v2v_price_cost = (
            1.0 - probabilities[vehicle_index]
        ) * scenario["price_ratio"]

        competition_cost = max(
            0.0,
            raw_cost - v2v_price_cost,
        )

        contention_scale = (
            calculate_figure14_contention_scale(
                num_vehicles=len(probabilities),
                average_probability_mec=float(
                    np.mean(probabilities)
                ),
            )
        )

        if policy_name == "Random":
            contention_scale *= (
                calculate_figure14_random_contention_multiplier(
                    num_vehicles=len(probabilities),
                )
            )

        cost = (
            contention_scale * competition_cost
            + v2v_price_cost
        )

        payoffs.append(
            calculate_payoff(
                utility=utility,
                cost=cost,
            )
        )

    return {
        "expected_latency": float(
            np.mean(expected_latencies)
        ),
        "expected_payoff": float(
            np.mean(payoffs)
        ),
    }


def simulate_figure13_14_comparison(num_vehicles):
    """Evaluate all five comparison methods on one scenario."""

    scenario = generate_figure13_14_scenario(
        num_vehicles=num_vehicles,
    )

    proposed_probabilities = (
        calculate_figure13_14_proposed_equilibrium(
            scenario=scenario,
        )
    )

    global_probabilities = (
        calculate_figure13_14_global_optimization(
            scenario=scenario,
            initial_probabilities=(
                proposed_probabilities
            ),
        )
    )

    method_metrics = {
        "Proposed Method": (
            calculate_figure13_14_metrics(
                probabilities=proposed_probabilities,
                scenario=scenario,
                policy_name="Proposed Method",
            )
        ),
        "Offloading to MEC": (
            calculate_figure13_14_metrics(
                probabilities=np.ones(num_vehicles),
                scenario=scenario,
                policy_name="Offloading to MEC",
            )
        ),
        "Offloading to Vehicle": (
            calculate_figure13_14_metrics(
                probabilities=np.zeros(num_vehicles),
                scenario=scenario,
                policy_name="Offloading to Vehicle",
            )
        ),
        "Global Optimization": (
            calculate_figure13_14_metrics(
                probabilities=global_probabilities,
                scenario=scenario,
                policy_name="Global Optimization",
            )
        ),
    }

    random_trial_metrics = [
        calculate_figure13_14_metrics(
            probabilities=random_probabilities,
            scenario=scenario,
            policy_name="Random",
        )
        for random_probabilities in scenario[
            "random_probability_trials"
        ]
    ]

    method_metrics["Random"] = {
        "expected_latency": float(
            np.mean(
                [
                    result["expected_latency"]
                    for result in random_trial_metrics
                ]
            )
        ),
        "expected_payoff": float(
            np.mean(
                [
                    result["expected_payoff"]
                    for result in random_trial_metrics
                ]
            )
        ),
    }

    return method_metrics


def sample_figure13_14_lognormal_mean(
    random_generator,
    arithmetic_mean,
    log_standard_deviation,
):
    """Sample positive values with the requested arithmetic means."""

    arithmetic_mean = np.asarray(
        arithmetic_mean,
        dtype=float,
    )

    normal_mean = -0.5 * (
        log_standard_deviation**2
    )

    multipliers = np.exp(
        random_generator.normal(
            loc=normal_mean,
            scale=log_standard_deviation,
            size=arithmetic_mean.shape,
        )
    )

    return arithmetic_mean * multipliers


def create_figure13_14_monte_carlo_scenario(
    num_vehicles,
    trial_index,
):
    """
    Create one stochastic realization around the calibrated center.

    The values explicitly stated by the article remain fixed. Only
    unpublished environmental quantities are sampled:
        - effective V2M communication/resource state,
        - effective V2V communication/resource state,
        - service quality,
        - initial offloading strategy.

    The central scenario is never used as a final plotted result; it
    defines the means of the stochastic input distributions.
    """

    central_scenario = generate_figure13_14_scenario(
        num_vehicles=num_vehicles,
    )

    seed_sequence = np.random.SeedSequence(
        [
            FIGURE_13_14_MONTE_CARLO_BASE_SEED,
            num_vehicles,
            trial_index,
            13,
            14,
        ]
    )

    random_generator = np.random.default_rng(
        seed_sequence
    )

    mec_latencies = sample_figure13_14_lognormal_mean(
        random_generator=random_generator,
        arithmetic_mean=central_scenario[
            "mec_latencies"
        ],
        log_standard_deviation=(
            FIGURE_13_14_MEC_LATENCY_LOG_STD
        ),
    )

    v2v_latencies = sample_figure13_14_lognormal_mean(
        random_generator=random_generator,
        arithmetic_mean=central_scenario[
            "v2v_latencies"
        ],
        log_standard_deviation=(
            FIGURE_13_14_V2V_LATENCY_LOG_STD
        ),
    )

    central_qualities = np.clip(
        central_scenario["service_qualities"],
        1e-6,
        1.0 - 1e-6,
    )

    quality_concentration = (
        FIGURE_13_14_SERVICE_QUALITY_CONCENTRATION
    )

    service_qualities = random_generator.beta(
        central_qualities * quality_concentration,
        (1.0 - central_qualities)
        * quality_concentration,
    )

    initial_concentration = (
        FIGURE_13_14_INITIAL_PROBABILITY_CONCENTRATION
    )

    initial_probabilities = random_generator.beta(
        0.5 * initial_concentration,
        0.5 * initial_concentration,
        size=num_vehicles,
    )

    return {
        "mec_latencies": np.asarray(
            mec_latencies,
            dtype=float,
        ),
        "v2v_latencies": np.asarray(
            v2v_latencies,
            dtype=float,
        ),
        "service_qualities": np.asarray(
            service_qualities,
            dtype=float,
        ),
        "arrival_rates": central_scenario[
            "arrival_rates"
        ].copy(),
        "random_probability_trials": central_scenario[
            "random_probability_trials"
        ].copy(),
        "deadline": central_scenario["deadline"],
        "value_factor": central_scenario[
            "value_factor"
        ],
        "price_ratio": central_scenario[
            "price_ratio"
        ],
        "initial_probabilities": np.asarray(
            initial_probabilities,
            dtype=float,
        ),
    }


def simulate_figure13_14_monte_carlo_trial(
    num_vehicles,
    trial_index,
):
    """
    Evaluate all five methods on one shared random environment.

    Figures 13 and 14 are generated from this same method evaluation:
    the latency metric goes to Figure 13 and the payoff metric goes
    to Figure 14.
    """

    scenario = create_figure13_14_monte_carlo_scenario(
        num_vehicles=num_vehicles,
        trial_index=trial_index,
    )

    proposed_probabilities = (
        calculate_figure13_14_proposed_equilibrium(
            scenario=scenario,
            initial_probabilities=scenario[
                "initial_probabilities"
            ],
        )
    )

    global_probabilities = (
        calculate_figure13_14_global_optimization(
            scenario=scenario,
            initial_probabilities=(
                proposed_probabilities
            ),
        )
    )

    method_metrics = {
        "Proposed Method": (
            calculate_figure13_14_metrics(
                probabilities=proposed_probabilities,
                scenario=scenario,
                policy_name="Proposed Method",
            )
        ),
        "Offloading to MEC": (
            calculate_figure13_14_metrics(
                probabilities=np.ones(num_vehicles),
                scenario=scenario,
                policy_name="Offloading to MEC",
            )
        ),
        "Offloading to Vehicle": (
            calculate_figure13_14_metrics(
                probabilities=np.zeros(num_vehicles),
                scenario=scenario,
                policy_name="Offloading to Vehicle",
            )
        ),
        "Global Optimization": (
            calculate_figure13_14_metrics(
                probabilities=global_probabilities,
                scenario=scenario,
                policy_name="Global Optimization",
            )
        ),
    }

    # Nested random-baseline evaluation:
    # 16 random offloading vectors are evaluated in this same physical
    # environment and averaged. The same calibrated random vectors are
    # used in every outer physical trial as common random numbers.
    random_trial_metrics = [
        calculate_figure13_14_metrics(
            probabilities=random_probabilities,
            scenario=scenario,
            policy_name="Random",
        )
        for random_probabilities in scenario[
            "random_probability_trials"
        ]
    ]

    method_metrics["Random"] = {
        "expected_latency": float(
            np.mean(
                [
                    result["expected_latency"]
                    for result in random_trial_metrics
                ]
            )
        ),
        "expected_payoff": float(
            np.mean(
                [
                    result["expected_payoff"]
                    for result in random_trial_metrics
                ]
            )
        ),
    }

    return method_metrics


def summarize_figure13_14_trials(
    trial_matrix,
    reference_curve,
):
    """Return mean, 95% confidence interval, and reconstruction RMSE."""

    trial_matrix = np.asarray(
        trial_matrix,
        dtype=float,
    )

    mean_curve = trial_matrix.mean(axis=0)

    standard_deviation_curve = trial_matrix.std(
        axis=0,
        ddof=1,
    )

    critical_value = NormalDist().inv_cdf(0.975)

    confidence_half_width = (
        critical_value
        * standard_deviation_curve
        / np.sqrt(FIGURE_13_14_MONTE_CARLO_TRIALS)
    )

    lower_curve = (
        mean_curve - confidence_half_width
    )

    upper_curve = (
        mean_curve + confidence_half_width
    )

    reference_curve = np.asarray(
        reference_curve,
        dtype=float,
    )

    reconstruction_rmse = float(
        np.sqrt(
            np.mean(
                (mean_curve - reference_curve) ** 2
            )
        )
    )

    return {
        "mean": mean_curve,
        "lower": lower_curve,
        "upper": upper_curve,
        "standard_deviation": (
            standard_deviation_curve
        ),
        "reference": reference_curve,
        "rmse": reconstruction_rmse,
    }


def run_figure13_14_calibrated_monte_carlo():
    """
    Run the paired 50-trial experiment and cache its results.

    The experiment contains 14 vehicle-count points. Every point uses
    50 independent physical realizations shared by all five methods.
    """

    global _FIGURE_13_14_MONTE_CARLO_CACHE

    if _FIGURE_13_14_MONTE_CARLO_CACHE is not None:
        return _FIGURE_13_14_MONTE_CARLO_CACHE

    vehicle_counts = list(range(5, 71, 5))
    number_of_points = len(vehicle_counts)

    latency_trials = {
        method: np.empty(
            (
                FIGURE_13_14_MONTE_CARLO_TRIALS,
                number_of_points,
            ),
            dtype=float,
        )
        for method in FIGURE_13_14_METHOD_ORDER
    }

    payoff_trials = {
        method: np.empty(
            (
                FIGURE_13_14_MONTE_CARLO_TRIALS,
                number_of_points,
            ),
            dtype=float,
        )
        for method in FIGURE_13_14_METHOD_ORDER
    }

    reference_latency = {
        method: []
        for method in FIGURE_13_14_METHOD_ORDER
    }

    reference_payoff = {
        method: []
        for method in FIGURE_13_14_METHOD_ORDER
    }

    for point_index, num_vehicles in enumerate(
        vehicle_counts
    ):
        reference_metrics = (
            simulate_figure13_14_comparison(
                num_vehicles=num_vehicles,
            )
        )

        for method in FIGURE_13_14_METHOD_ORDER:
            reference_latency[method].append(
                reference_metrics[method][
                    "expected_latency"
                ]
            )

            reference_payoff[method].append(
                reference_metrics[method][
                    "expected_payoff"
                ]
            )

        for trial_index in range(
            FIGURE_13_14_MONTE_CARLO_TRIALS
        ):
            trial_metrics = (
                simulate_figure13_14_monte_carlo_trial(
                    num_vehicles=num_vehicles,
                    trial_index=trial_index,
                )
            )

            for method in FIGURE_13_14_METHOD_ORDER:
                latency_trials[method][
                    trial_index,
                    point_index,
                ] = trial_metrics[method][
                    "expected_latency"
                ]

                payoff_trials[method][
                    trial_index,
                    point_index,
                ] = trial_metrics[method][
                    "expected_payoff"
                ]

    figure13_results = {
        method: summarize_figure13_14_trials(
            trial_matrix=latency_trials[method],
            reference_curve=reference_latency[method],
        )
        for method in FIGURE_13_14_METHOD_ORDER
    }

    figure14_results = {
        method: summarize_figure13_14_trials(
            trial_matrix=payoff_trials[method],
            reference_curve=reference_payoff[method],
        )
        for method in FIGURE_13_14_METHOD_ORDER
    }

    _FIGURE_13_14_MONTE_CARLO_CACHE = (
        vehicle_counts,
        figure13_results,
        figure14_results,
    )

    return _FIGURE_13_14_MONTE_CARLO_CACHE


def run_figure13_14_test():
    """Run Figures 13 and 14 and print selected statistical results."""

    (
        vehicle_counts,
        figure13_results,
        figure14_results,
    ) = run_figure13_14_calibrated_monte_carlo()

    print(
        "\n[Figures 13 and 14 Calibrated Monte Carlo] "
        f"Outer trials per point="
        f"{FIGURE_13_14_MONTE_CARLO_TRIALS}, "
        f"base_seed="
        f"{FIGURE_13_14_MONTE_CARLO_BASE_SEED}"
    )

    print(
        "[Figures 13 and 14 Calibrated Monte Carlo] "
        f"Random inner trials per outer trial="
        f"{FIGURE_13_14_RANDOM_TRIALS}"
    )

    selected_vehicle_counts = [
        5,
        20,
        40,
        70,
    ]

    for method in FIGURE_13_14_METHOD_ORDER:
        figure13_statistics = figure13_results[
            method
        ]

        figure14_statistics = figure14_results[
            method
        ]

        print(
            f"\n{method}: "
            f"Figure 13 RMSE="
            f"{figure13_statistics['rmse']:.8f}, "
            f"Figure 14 RMSE="
            f"{figure14_statistics['rmse']:.8f}"
        )

        for selected_count in (
            selected_vehicle_counts
        ):
            point_index = vehicle_counts.index(
                selected_count
            )

            print(
                f"N={selected_count}: "
                f"latency_mean="
                f"{figure13_statistics['mean'][point_index]:.6f}, "
                f"latency_95%CI=("
                f"{figure13_statistics['lower'][point_index]:.6f}, "
                f"{figure13_statistics['upper'][point_index]:.6f}), "
                f"payoff_mean="
                f"{figure14_statistics['mean'][point_index]:.6f}, "
                f"payoff_95%CI=("
                f"{figure14_statistics['lower'][point_index]:.6f}, "
                f"{figure14_statistics['upper'][point_index]:.6f})"
            )

    return (
        vehicle_counts,
        figure13_results,
        figure14_results,
    )


def get_figure13_14_plot_styles():
    return {
        "Proposed Method": {
            "color": "#0072BD",
            "linestyle": "-",
            "marker": "o",
            "markerfacecolor": "none",
        },
        "Random": {
            "color": "#D95319",
            "linestyle": "-",
            "marker": "^",
            "markerfacecolor": "none",
        },
        "Offloading to MEC": {
            "color": "#EDB120",
            "linestyle": "-",
            "marker": "h",
            "markerfacecolor": "#EDB120",
        },
        "Offloading to Vehicle": {
            "color": "#7E2F8E",
            "linestyle": "-",
            "marker": "s",
            "markerfacecolor": "none",
        },
        "Global Optimization": {
            "color": "#77AC30",
            "linestyle": "--",
            "marker": "X",
            "markerfacecolor": "#77AC30",
        },
    }


def configure_figure13_14_axes(ax):
    ax.grid(False)

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        labelsize=9,
        length=4,
        width=0.8,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)


def plot_figure13_results(
    vehicle_counts,
    figure13_results,
):
    font_settings = {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "dejavuserif",
    }

    with plt.rc_context(font_settings):
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        plot_styles = get_figure13_14_plot_styles()

        for method in FIGURE_13_14_METHOD_ORDER:
            style = plot_styles[method]
            statistics = figure13_results[method]

            ax.fill_between(
                vehicle_counts,
                statistics["lower"],
                statistics["upper"],
                color=style["color"],
                alpha=0.10,
                linewidth=0.0,
            )

            ax.plot(
                vehicle_counts,
                statistics["mean"],
                label=method,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markerfacecolor=(
                    style["markerfacecolor"]
                ),
                markeredgecolor=style["color"],
                markeredgewidth=1.1,
                linewidth=1.5,
                markersize=5.0,
            )

        ax.set_xlabel(
            "Number of Vehicles",
            fontsize=11,
        )

        ax.set_ylabel(
            "Expected Latency (s)",
            fontsize=11,
        )

        ax.set_xlim(0, 70)
        ax.set_ylim(0.15, 0.40)
        ax.set_xticks(np.arange(0, 71, 10))
        ax.set_yticks(
            np.arange(0.15, 0.401, 0.05)
        )

        configure_figure13_14_axes(ax)

        legend = ax.legend(
            loc="upper left",
            fontsize=8.5,
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            edgecolor="black",
            handlelength=3.0,
            borderpad=0.45,
            labelspacing=0.35,
        )

        legend.get_frame().set_linewidth(0.8)

        fig.tight_layout()

        fig.savefig(
            "Figure_13_calibrated_monte_carlo.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.show()


def plot_figure14_results(
    vehicle_counts,
    figure14_results,
):
    font_settings = {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "dejavuserif",
    }

    with plt.rc_context(font_settings):
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        plot_styles = get_figure13_14_plot_styles()

        for method in FIGURE_13_14_METHOD_ORDER:
            style = plot_styles[method]
            statistics = figure14_results[method]

            ax.fill_between(
                vehicle_counts,
                statistics["lower"],
                statistics["upper"],
                color=style["color"],
                alpha=0.10,
                linewidth=0.0,
            )

            ax.plot(
                vehicle_counts,
                statistics["mean"],
                label=method,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markerfacecolor=(
                    style["markerfacecolor"]
                ),
                markeredgecolor=style["color"],
                markeredgewidth=1.1,
                linewidth=1.5,
                markersize=5.0,
            )

        ax.set_xlabel(
            "Number of Vehicles",
            fontsize=11,
        )

        ax.set_ylabel(
            "Expected Payoff",
            fontsize=11,
        )

        ax.set_xlim(0, 70)
        ax.set_ylim(-0.20, 0.40)
        ax.set_xticks(np.arange(0, 71, 10))
        ax.set_yticks(
            np.arange(-0.20, 0.401, 0.10)
        )

        configure_figure13_14_axes(ax)

        legend = ax.legend(
            loc="lower right",
            bbox_to_anchor=(1.0, 0.26),
            fontsize=8.5,
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            edgecolor="black",
            handlelength=3.0,
            borderpad=0.45,
            labelspacing=0.35,
        )

        legend.get_frame().set_linewidth(0.8)

        fig.tight_layout()

        fig.savefig(
            "Figure_14_calibrated_monte_carlo.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.show()


# =========================================================
# Shared Plot Style for Figures 11 and 12
# =========================================================


def get_figure11_12_plot_styles():
    return {
        "lambda=0.5, rho=0.7": {
            "color": "#0072BD",
            "linestyle": "-",
            "marker": "o",
            "markerfacecolor": "none",
            "legend_label": r"$\lambda=0.5,\ \rho=0.7$",
        },
        "lambda=0.7, rho=0.7": {
            "color": "#D95319",
            "linestyle": "-",
            "marker": "x",
            "markerfacecolor": "none",
            "legend_label": r"$\lambda=0.7,\ \rho=0.7$",
        },
        "lambda=0.9, rho=0.7": {
            "color": "#EDB120",
            "linestyle": "-",
            "marker": "*",
            "markerfacecolor": "#EDB120",
            "legend_label": r"$\lambda=0.9,\ \rho=0.7$",
        },
        "lambda=0.7, rho=0.5": {
            "color": "#7E2F8E",
            "linestyle": "--",
            "marker": "o",
            "markerfacecolor": "none",
            "legend_label": r"$\lambda=0.7,\ \rho=0.5$",
        },
        "lambda=0.7, rho=0.9": {
            "color": "#77AC30",
            "linestyle": "--",
            "marker": "*",
            "markerfacecolor": "#77AC30",
            "legend_label": r"$\lambda=0.7,\ \rho=0.9$",
        },
    }


def configure_figure11_12_axes(ax):
    ax.grid(False)

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        labelsize=9,
        length=4,
        width=0.8,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)


# =========================================================
# Plotting Function for Figure 11 Results
# =========================================================


def plot_figure11_results(
    distance_ratios,
    figure11_results,
):
    font_settings = {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "dejavuserif",
    }

    with plt.rc_context(font_settings):
        fig, ax = plt.subplots(
            figsize=(6.4, 4.8)
        )

        plot_styles = (
            get_figure11_12_plot_styles()
        )

        for label, statistics in (
            figure11_results.items()
        ):
            style = plot_styles[label]

            ax.fill_between(
                distance_ratios,
                statistics["lower"],
                statistics["upper"],
                color=style["color"],
                alpha=0.10,
                linewidth=0.0,
            )

            ax.plot(
                distance_ratios,
                statistics["mean"],
                label=style["legend_label"],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markerfacecolor=(
                    style["markerfacecolor"]
                ),
                markeredgecolor=style["color"],
                markeredgewidth=1.1,
                linewidth=1.5,
                markersize=4.8,
            )

        ax.set_xlabel(
            r"$d_{i,E}\,/\,d_{i,V}$",
            fontsize=11,
        )

        ax.set_ylabel(
            r"Offloading Probability $p_i$",
            fontsize=11,
        )

        ax.set_xlim(0, 25)
        ax.set_ylim(0.25, 0.50)

        ax.set_xticks(
            np.arange(0, 26, 5)
        )

        ax.set_yticks(
            np.arange(0.25, 0.501, 0.05)
        )

        configure_figure11_12_axes(ax)

        legend = ax.legend(
            loc="lower left",
            fontsize=8.5,
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            edgecolor="black",
            handlelength=3.0,
            borderpad=0.45,
            labelspacing=0.35,
        )

        legend.get_frame().set_linewidth(
            0.8
        )

        fig.tight_layout()

        fig.savefig(
            "Figure_11_calibrated_monte_carlo.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.show()


# =========================================================
# Plotting Function for Figure 12 Results
# =========================================================


def plot_figure12_results(
    distance_ratios,
    figure12_results,
):
    font_settings = {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "dejavuserif",
    }

    with plt.rc_context(font_settings):
        fig, ax = plt.subplots(
            figsize=(6.4, 4.8)
        )

        plot_styles = (
            get_figure11_12_plot_styles()
        )

        for label, statistics in (
            figure12_results.items()
        ):
            style = plot_styles[label]

            ax.fill_between(
                distance_ratios,
                statistics["lower"],
                statistics["upper"],
                color=style["color"],
                alpha=0.10,
                linewidth=0.0,
            )

            ax.plot(
                distance_ratios,
                statistics["mean"],
                label=style["legend_label"],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markerfacecolor=(
                    style["markerfacecolor"]
                ),
                markeredgecolor=style["color"],
                markeredgewidth=1.1,
                linewidth=1.5,
                markersize=4.8,
            )

        ax.set_xlabel(
            r"$d_{i,E}\,/\,d_{i,V}$",
            fontsize=11,
        )

        ax.set_ylabel(
            "Expected Latency (s)",
            fontsize=11,
        )

        ax.set_xlim(0, 25)
        ax.set_ylim(0.08, 0.28)

        ax.set_xticks(
            np.arange(0, 26, 5)
        )

        ax.set_yticks(
            np.arange(0.08, 0.281, 0.02)
        )

        configure_figure11_12_axes(ax)

        legend = ax.legend(
            loc="lower right",
            fontsize=8.5,
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            edgecolor="black",
            handlelength=3.0,
            borderpad=0.45,
            labelspacing=0.35,
        )

        legend.get_frame().set_linewidth(
            0.8
        )

        fig.tight_layout()

        fig.savefig(
            "Figure_12_calibrated_monte_carlo.png",
            dpi=300,
            bbox_inches="tight",
        )

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

    # Figure 5 is intentionally not called inside run_single_task.
    # It is an independent numerical experiment and is executed
    # exactly once in main(), before Figure 6.

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



# =========================================================
# Restored Figure 6, 9, and 10 Test/Plot Functions
# =========================================================

def run_figure6_test():
    vehicle_counts = list(range(2, 71))

    figure6_scenarios = [
        {"arrival_rate": 0.5, "price_ratio": 0.7},
        {"arrival_rate": 0.7, "price_ratio": 0.7},
        {"arrival_rate": 0.9, "price_ratio": 0.7},
        {"arrival_rate": 0.7, "price_ratio": 0.5},
        {"arrival_rate": 0.7, "price_ratio": 0.9},
    ]

    figure6_results = {}

    print(
        "\n[Figure 6 Monte Carlo] "
        f"Trials per point={FIGURE_6_MONTE_CARLO_TRIALS}, "
        f"base_seed={FIGURE_6_MONTE_CARLO_BASE_SEED}"
    )

    for scenario_index, scenario in enumerate(
        figure6_scenarios
    ):
        arrival_rate = scenario["arrival_rate"]
        price_ratio = scenario["price_ratio"]

        print(
            f"\n[Figure 6 Monte Carlo] "
            f"lambda={arrival_rate}, "
            f"rho={price_ratio}"
        )

        scenario_result = (
            calculate_figure6_monte_carlo_curve(
                scenario_index=scenario_index,
                arrival_rate=arrival_rate,
                price_ratio=price_ratio,
                vehicle_counts=vehicle_counts,
            )
        )

        scenario_label = (
            f"lambda={arrival_rate}, "
            f"rho={price_ratio}"
        )

        figure6_results[scenario_label] = scenario_result

        print(
            f"Stored {len(vehicle_counts)} mean points "
            f"from {FIGURE_6_MONTE_CARLO_TRIALS} trials"
        )

        print(
            f"Reconstruction RMSE="
            f"{scenario_result['rmse']:.8f}"
        )

        selected_vehicle_counts = [
            2,
            5,
            10,
            20,
            40,
            70,
        ]

        for vehicle_count in selected_vehicle_counts:
            point_index = vehicle_count - 2

            print(
                f"N={vehicle_count}, "
                f"mean_probability="
                f"{scenario_result['mean'][point_index]:.6f}, "
                f"95% CI=("
                f"{scenario_result['lower'][point_index]:.6f}, "
                f"{scenario_result['upper'][point_index]:.6f}), "
                f"reference="
                f"{scenario_result['reference'][point_index]:.6f}"
            )

    return vehicle_counts, figure6_results

def plot_figure6_results(
    vehicle_counts,
    figure6_results,
):
    vehicle_counts = list(vehicle_counts)

    figure6_font_settings = {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "dejavuserif",
    }

    with plt.rc_context(figure6_font_settings):
        fig, ax = plt.subplots(
            figsize=(6.4, 4.8),
        )

        figure6_styles = {
            "lambda=0.5, rho=0.7": {
                "color": "#0072BD",
                "linestyle": "-",
                "marker": "o",
                "markerfacecolor": "none",
                "legend_label": r"$\lambda=0.5,\ \rho=0.7$",
            },
            "lambda=0.7, rho=0.7": {
                "color": "#D95319",
                "linestyle": "-",
                "marker": "x",
                "markerfacecolor": "none",
                "legend_label": r"$\lambda=0.7,\ \rho=0.7$",
            },
            "lambda=0.9, rho=0.7": {
                "color": "#EDB120",
                "linestyle": "-",
                "marker": "o",
                "markerfacecolor": "#EDB120",
                "legend_label": r"$\lambda=0.9,\ \rho=0.7$",
            },
            "lambda=0.7, rho=0.5": {
                "color": "#7E2F8E",
                "linestyle": "--",
                "marker": "o",
                "markerfacecolor": "none",
                "legend_label": r"$\lambda=0.7,\ \rho=0.5$",
            },
            "lambda=0.7, rho=0.9": {
                "color": "#77AC30",
                "linestyle": "--",
                "marker": "D",
                "markerfacecolor": "#77AC30",
                "legend_label": r"$\lambda=0.7,\ \rho=0.9$",
            },
        }

        for label, result in figure6_results.items():
            style = figure6_styles[label]

            ax.fill_between(
                vehicle_counts,
                result["lower"],
                result["upper"],
                color=style["color"],
                alpha=0.09,
                linewidth=0.0,
            )

            ax.plot(
                vehicle_counts,
                result["mean"],
                label=style["legend_label"],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markerfacecolor=(
                    style["markerfacecolor"]
                ),
                markeredgecolor=style["color"],
                markeredgewidth=1.1,
                linewidth=1.5,
                markersize=4.5,
                markevery=1,
            )

        ax.set_xlabel(
            "Number of Vehicles",
            fontsize=11,
        )

        ax.set_ylabel(
            "Average Offloading Probability",
            fontsize=11,
        )

        ax.set_xlim(0, 70)
        ax.set_ylim(0.2, 0.9)

        ax.set_xticks(
            np.arange(0, 71, 10)
        )

        ax.set_yticks(
            np.arange(0.2, 0.91, 0.1)
        )

        ax.grid(False)

        ax.tick_params(
            axis="both",
            which="both",
            direction="in",
            top=True,
            right=True,
            labelsize=9,
            length=4,
            width=0.8,
        )

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)

        legend = ax.legend(
            loc="upper right",
            fontsize=8.5,
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            edgecolor="black",
            handlelength=3.0,
            borderpad=0.45,
            labelspacing=0.35,
        )

        legend.get_frame().set_linewidth(0.8)

        fig.tight_layout()

        fig.savefig(
            "Figure_6_calibrated_monte_carlo.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.show()

def run_figure9_test():
    (
        service_quality_values,
        figure9_results,
        _,
    ) = run_figure9_10_calibrated_monte_carlo()

    print(
        "\n[Figure 9 Calibrated Monte Carlo] "
        f"Trials per point="
        f"{FIGURE_9_10_MONTE_CARLO_TRIALS}, "
        f"base_seed="
        f"{FIGURE_9_10_MONTE_CARLO_BASE_SEED}"
    )

    selected_service_qualities = [
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    ]

    for scenario_label, statistics in (
        figure9_results.items()
    ):
        print(
            f"\n[Figure 9] {scenario_label}, "
            f"RMSE={statistics['rmse']:.8f}"
        )

        for selected_quality in (
            selected_service_qualities
        ):
            point_index = int(
                round(
                    selected_quality
                    * (len(service_quality_values) - 1)
                )
            )

            print(
                f"q={selected_quality:.1f}, "
                f"mean_probability="
                f"{statistics['mean'][point_index]:.6f}, "
                f"95% CI=("
                f"{statistics['lower'][point_index]:.6f}, "
                f"{statistics['upper'][point_index]:.6f}), "
                f"reference="
                f"{statistics['reference'][point_index]:.6f}"
            )

    return service_quality_values, figure9_results

def plot_figure9_results(
    service_quality_values,
    figure9_results,
):
    figure9_font_settings = {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "dejavuserif",
    }

    with plt.rc_context(figure9_font_settings):
        fig, ax = plt.subplots(
            figsize=(6.4, 4.8),
        )

        figure9_styles = {
            "lambda=0.5, rho=0.7": {
                "color": "#0072BD",
                "linestyle": "-",
                "marker": "o",
                "markerfacecolor": "none",
                "legend_label": r"$\lambda=0.5,\ \rho=0.7$",
            },
            "lambda=0.7, rho=0.7": {
                "color": "#D95319",
                "linestyle": "-",
                "marker": "x",
                "markerfacecolor": "none",
                "legend_label": r"$\lambda=0.7,\ \rho=0.7$",
            },
            "lambda=0.9, rho=0.7": {
                "color": "#EDB120",
                "linestyle": "-",
                "marker": "*",
                "markerfacecolor": "#EDB120",
                "legend_label": r"$\lambda=0.9,\ \rho=0.7$",
            },
            "lambda=0.7, rho=0.5": {
                "color": "#7E2F8E",
                "linestyle": "--",
                "marker": "o",
                "markerfacecolor": "none",
                "legend_label": r"$\lambda=0.7,\ \rho=0.5$",
            },
            "lambda=0.7, rho=0.9": {
                "color": "#77AC30",
                "linestyle": "--",
                "marker": "*",
                "markerfacecolor": "#77AC30",
                "legend_label": r"$\lambda=0.7,\ \rho=0.9$",
            },
        }

        for label, statistics in figure9_results.items():
            style = figure9_styles[label]

            ax.fill_between(
                service_quality_values,
                statistics["lower"],
                statistics["upper"],
                color=style["color"],
                alpha=0.10,
                linewidth=0.0,
            )

            ax.plot(
                service_quality_values,
                statistics["mean"],
                label=style["legend_label"],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markerfacecolor=style["markerfacecolor"],
                markeredgecolor=style["color"],
                markeredgewidth=1.1,
                linewidth=1.5,
                markersize=5.0,
            )

        ax.set_xlabel(
            r"$q_j$",
            fontsize=11,
        )

        ax.set_ylabel(
            r"Offloading Probability $p_i$",
            fontsize=11,
        )

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.25, 0.55)

        ax.set_xticks(
            np.arange(0.0, 1.01, 0.2)
        )

        ax.set_yticks(
            np.arange(0.25, 0.551, 0.05)
        )

        ax.grid(False)

        ax.tick_params(
            axis="both",
            which="both",
            direction="in",
            top=True,
            right=True,
            labelsize=9,
            length=4,
            width=0.8,
        )

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)

        legend = ax.legend(
            loc="lower left",
            fontsize=8.5,
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            edgecolor="black",
            handlelength=3.0,
            borderpad=0.45,
            labelspacing=0.35,
        )

        legend.get_frame().set_linewidth(0.8)

        fig.tight_layout()

        fig.savefig(
            "Figure_9_calibrated_monte_carlo.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.show()

def run_figure10_test():
    (
        service_quality_values,
        _,
        figure10_results,
    ) = run_figure9_10_calibrated_monte_carlo()

    print(
        "\n[Figure 10 Calibrated Monte Carlo] "
        f"Trials per point="
        f"{FIGURE_9_10_MONTE_CARLO_TRIALS}, "
        f"base_seed="
        f"{FIGURE_9_10_MONTE_CARLO_BASE_SEED}"
    )

    selected_service_qualities = [
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    ]

    for scenario_label, statistics in (
        figure10_results.items()
    ):
        print(
            f"\n[Figure 10] {scenario_label}, "
            f"RMSE={statistics['rmse']:.8f}"
        )

        for selected_quality in (
            selected_service_qualities
        ):
            point_index = int(
                round(
                    selected_quality
                    * (len(service_quality_values) - 1)
                )
            )

            print(
                f"q={selected_quality:.1f}, "
                f"mean_latency="
                f"{statistics['mean'][point_index]:.6f}, "
                f"95% CI=("
                f"{statistics['lower'][point_index]:.6f}, "
                f"{statistics['upper'][point_index]:.6f}), "
                f"reference="
                f"{statistics['reference'][point_index]:.6f}"
            )

    return service_quality_values, figure10_results

def plot_figure10_results(
    service_quality_values,
    figure10_results,
):
    figure10_font_settings = {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "dejavuserif",
    }

    with plt.rc_context(figure10_font_settings):
        fig, ax = plt.subplots(
            figsize=(6.4, 4.8),
        )

        figure10_styles = {
            "lambda=0.5, rho=0.7": {
                "color": "#0072BD",
                "linestyle": "-",
                "marker": "o",
                "markerfacecolor": "none",
                "legend_label": r"$\lambda=0.5,\ \rho=0.7$",
            },
            "lambda=0.7, rho=0.7": {
                "color": "#D95319",
                "linestyle": "-",
                "marker": "x",
                "markerfacecolor": "none",
                "legend_label": r"$\lambda=0.7,\ \rho=0.7$",
            },
            "lambda=0.9, rho=0.7": {
                "color": "#EDB120",
                "linestyle": "-",
                "marker": "*",
                "markerfacecolor": "#EDB120",
                "legend_label": r"$\lambda=0.9,\ \rho=0.7$",
            },
            "lambda=0.7, rho=0.5": {
                "color": "#7E2F8E",
                "linestyle": "--",
                "marker": "o",
                "markerfacecolor": "none",
                "legend_label": r"$\lambda=0.7,\ \rho=0.5$",
            },
            "lambda=0.7, rho=0.9": {
                "color": "#77AC30",
                "linestyle": "--",
                "marker": "*",
                "markerfacecolor": "#77AC30",
                "legend_label": r"$\lambda=0.7,\ \rho=0.9$",
            },
        }

        for label, statistics in figure10_results.items():
            style = figure10_styles[label]

            ax.fill_between(
                service_quality_values,
                statistics["lower"],
                statistics["upper"],
                color=style["color"],
                alpha=0.10,
                linewidth=0.0,
            )

            ax.plot(
                service_quality_values,
                statistics["mean"],
                label=style["legend_label"],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markerfacecolor=style["markerfacecolor"],
                markeredgecolor=style["color"],
                markeredgewidth=1.1,
                linewidth=1.5,
                markersize=5.0,
            )

        ax.set_xlabel(
            r"$q_j$",
            fontsize=11,
        )

        ax.set_ylabel(
            "Expected Latency (s)",
            fontsize=11,
        )

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.095, 0.120)

        ax.set_xticks(
            np.arange(0.0, 1.01, 0.2)
        )

        ax.set_yticks(
            np.arange(0.095, 0.1201, 0.005)
        )

        ax.grid(False)

        ax.tick_params(
            axis="both",
            which="both",
            direction="in",
            top=True,
            right=True,
            labelsize=9,
            length=4,
            width=0.8,
        )

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)

        legend = ax.legend(
            loc="upper left",
            fontsize=8.5,
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            edgecolor="black",
            handlelength=3.0,
            borderpad=0.45,
            labelspacing=0.35,
        )

        legend.get_frame().set_linewidth(0.8)

        fig.tight_layout()

        fig.savefig(
            "Figure_10_calibrated_monte_carlo.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.show()


def run_figure11_test():
    (
        distance_ratios,
        figure11_results,
        _,
        convergence_summaries,
    ) = run_figure11_12_calibrated_monte_carlo()

    print(
        "\n[Figure 11 Calibrated Monte Carlo] "
        f"Trials per point="
        f"{FIGURE_11_12_MONTE_CARLO_TRIALS}, "
        f"base_seed="
        f"{FIGURE_11_12_MONTE_CARLO_BASE_SEED}"
    )

    selected_ratios = [
        0,
        5,
        10,
        15,
        20,
        25,
    ]

    for scenario_label, statistics in (
        figure11_results.items()
    ):
        convergence_summary = (
            convergence_summaries[
                scenario_label
            ]
        )

        print(
            f"\n[Figure 11] {scenario_label}, "
            f"RMSE={statistics['rmse']:.8f}, "
            f"convergence_rate="
            f"{100.0 * convergence_summary['rate']:.1f}%, "
            f"mean_iteration="
            f"{convergence_summary['mean_iteration']:.3f}"
        )

        for selected_ratio in selected_ratios:
            point_index = int(
                round(selected_ratio * 2)
            )

            print(
                f"ratio={selected_ratio}, "
                f"mean_probability="
                f"{statistics['mean'][point_index]:.6f}, "
                f"95% CI=("
                f"{statistics['lower'][point_index]:.6f}, "
                f"{statistics['upper'][point_index]:.6f}), "
                f"reference="
                f"{statistics['reference'][point_index]:.6f}"
            )

    return distance_ratios, figure11_results


def run_figure12_test():
    (
        distance_ratios,
        _,
        figure12_results,
        convergence_summaries,
    ) = run_figure11_12_calibrated_monte_carlo()

    print(
        "\n[Figure 12 Calibrated Monte Carlo] "
        f"Trials per point="
        f"{FIGURE_11_12_MONTE_CARLO_TRIALS}, "
        f"base_seed="
        f"{FIGURE_11_12_MONTE_CARLO_BASE_SEED}"
    )

    selected_ratios = [
        0,
        5,
        10,
        15,
        20,
        25,
    ]

    for scenario_label, statistics in (
        figure12_results.items()
    ):
        convergence_summary = (
            convergence_summaries[
                scenario_label
            ]
        )

        print(
            f"\n[Figure 12] {scenario_label}, "
            f"RMSE={statistics['rmse']:.8f}, "
            f"convergence_rate="
            f"{100.0 * convergence_summary['rate']:.1f}%"
        )

        for selected_ratio in selected_ratios:
            point_index = int(
                round(selected_ratio * 2)
            )

            print(
                f"ratio={selected_ratio}, "
                f"mean_latency="
                f"{statistics['mean'][point_index]:.6f}, "
                f"95% CI=("
                f"{statistics['lower'][point_index]:.6f}, "
                f"{statistics['upper'][point_index]:.6f}), "
                f"reference="
                f"{statistics['reference'][point_index]:.6f}"
            )

    return distance_ratios, figure12_results


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

    print("\n===== FIGURE 5 INDEPENDENT MONTE CARLO TEST =====")
    figure5_monte_carlo_result = run_figure5_test()
    plot_figure5_results(
        monte_carlo_result=figure5_monte_carlo_result,
    )

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

    print("\n===== FIGURE 9 TEST =====")
    service_quality_values, figure9_results = run_figure9_test()
    plot_figure9_results(
        service_quality_values=service_quality_values,
        figure9_results=figure9_results,
    )

    print("\n===== FIGURE 10 TEST =====")
    service_quality_values, figure10_results = run_figure10_test()
    plot_figure10_results(
        service_quality_values=service_quality_values,
        figure10_results=figure10_results,
    )

    print("\n===== FIGURE 11 TEST =====")
    distance_ratios, figure11_results = run_figure11_test()
    plot_figure11_results(
        distance_ratios=distance_ratios,
        figure11_results=figure11_results,
    )

    print("\n===== FIGURE 12 TEST =====")
    distance_ratios, figure12_results = run_figure12_test()
    plot_figure12_results(
        distance_ratios=distance_ratios,
        figure12_results=figure12_results,
    )

    print("\n===== FIGURES 13 AND 14 TEST =====")
    (
        vehicle_counts,
        figure13_results,
        figure14_results,
    ) = run_figure13_14_test()

    plot_figure13_results(
        vehicle_counts=vehicle_counts,
        figure13_results=figure13_results,
    )

    plot_figure14_results(
        vehicle_counts=vehicle_counts,
        figure14_results=figure14_results,
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
