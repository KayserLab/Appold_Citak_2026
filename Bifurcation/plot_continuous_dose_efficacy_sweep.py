import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import yaml

try:
    from . import plot_style
except ImportError:
    import plot_style


plot_style.configure_plot_style()


def get_plot_config():
    return {
        "save_folder": os.path.join("data", "sweeps", "continuous_dose_sweep_rebuttal"),
        "output_path": os.path.join(
            "Bifurcation", "plots", "continuous_dose_sweep_rebuttal.pdf"
        ),
        "ttp_threshold": 71.0,
        "steps_per_hour": 20.0,
        "ratio_metric": "ttp",
        "front_velocity_metric": "ttp",
        "front_velocity_window": 51,
        "front_growth_mad_multiplier": 3.0,
        "front_growth_min_consecutive_points": 2,
        "show_plot": True,
    }


def find_project_root(current_dir, marker_file):
    current_dir = os.path.abspath(current_dir)
    while current_dir != os.path.dirname(current_dir):
        if marker_file in os.listdir(current_dir):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return None


def resolve_path(project_root, path):
    if os.path.isabs(path):
        return path
    return os.path.join(project_root, path)


def load_metadata(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _efficacy_key(value):
    return round(float(value), 8)


def find_ttp_index(size_trace, ttp_threshold):
    progression_indices = np.flatnonzero(np.asarray(size_trace) >= float(ttp_threshold))
    if progression_indices.size > 0:
        return int(progression_indices[0])
    return len(size_trace) - 1


def compute_ttp_and_ratio(
    size_trace, ratio_trace, *, start_point, steps_per_hour, ttp_threshold
):
    ttp_index = find_ttp_index(size_trace, ttp_threshold)

    ttp_hours = (ttp_index - float(start_point)) / float(steps_per_hour)
    ratio_at_ttp = float(ratio_trace[ttp_index])
    ratio_endpoint = float(ratio_trace[-1])

    return ttp_index, ttp_hours, ratio_at_ttp, ratio_endpoint


def rolling_average(data, window_size):
    data = np.asarray(data, dtype=float)
    window_size = max(1, int(window_size))
    if data.size == 0 or window_size == 1:
        return data

    window_size = min(window_size, data.size)
    left = window_size // 2
    right = window_size - left - 1
    padded_data = np.pad(data, (left, right), mode="edge")
    kernel = np.ones(window_size, dtype=float) / window_size

    return np.convolve(padded_data, kernel, mode="valid")


def sample_front_velocity(
    front_velocity_trace, ttp_index, metric, window_size, steps_per_hour
):
    front_velocity_trace = rolling_average(front_velocity_trace, window_size)
    front_velocity_trace = front_velocity_trace * float(steps_per_hour)
    if metric == "ttp":
        velocity_index = max(0, min(int(ttp_index) - 1, len(front_velocity_trace) - 1))
        return float(front_velocity_trace[velocity_index])
    if metric == "endpoint":
        return float(front_velocity_trace[-1])
    raise ValueError("front_velocity_metric must be either 'ttp' or 'endpoint'.")


def load_continuous_dose_sweep_metrics(
    save_folder,
    *,
    steps_per_hour=20.0,
    ttp_threshold=71.0,
    front_velocity_metric="ttp",
    front_velocity_window=51,
):
    params_list = np.load(os.path.join(save_folder, "params_list.npy"))
    sweep_axes = np.load(os.path.join(save_folder, "sweep_axes.npz"))
    metadata = load_metadata(os.path.join(save_folder, "metadata.yaml"))

    target_efficacy_values = np.asarray(
        sweep_axes["target_efficacy"],
        dtype=np.float32,
    )
    num_replicates = int(metadata["num_replicates"])
    total_time = int(metadata["total_time"])
    start_point = int(metadata["start_point"])
    num_sim = len(params_list)

    size = np.memmap(
        os.path.join(save_folder, "size.dat"),
        dtype=np.float32,
        mode="r",
        shape=(num_sim, total_time),
    )
    ratio = np.memmap(
        os.path.join(save_folder, "ratio.dat"),
        dtype=np.float32,
        mode="r",
        shape=(num_sim, total_time),
    )
    sensitive_front_velocity_path = os.path.join(
        save_folder, "sensitive_front_velocity.dat"
    )
    resistant_front_velocity_path = os.path.join(
        save_folder, "resistant_front_velocity.dat"
    )
    if not os.path.exists(sensitive_front_velocity_path) or not os.path.exists(
        resistant_front_velocity_path
    ):
        raise FileNotFoundError(
            "Front-velocity sweep files are missing. Re-run continuous_dose_efficacy_sweep.py to create sensitive_front_velocity.dat and resistant_front_velocity.dat."
        )
    sensitive_front_velocity = np.memmap(
        sensitive_front_velocity_path,
        dtype=np.float32,
        mode="r",
        shape=(num_sim, total_time - 1),
    )
    resistant_front_velocity = np.memmap(
        resistant_front_velocity_path,
        dtype=np.float32,
        mode="r",
        shape=(num_sim, total_time - 1),
    )
    status = np.memmap(
        os.path.join(save_folder, "status.dat"),
        dtype=np.bool_,
        mode="r",
        shape=(num_sim,),
    )

    efficacy_index = {
        _efficacy_key(value): idx for idx, value in enumerate(target_efficacy_values)
    }
    ttp_grid = np.full(
        (len(target_efficacy_values), num_replicates),
        np.nan,
        dtype=np.float32,
    )
    ratio_ttp_grid = np.full_like(ttp_grid, np.nan)
    ratio_endpoint_grid = np.full_like(ttp_grid, np.nan)
    sensitive_front_velocity_grid = np.full_like(ttp_grid, np.nan)
    resistant_front_velocity_grid = np.full_like(ttp_grid, np.nan)

    for idx, (target_efficacy, replicate, _) in enumerate(params_list):
        if not status[idx]:
            continue

        target_idx = efficacy_index[_efficacy_key(target_efficacy)]
        replicate_idx = int(replicate)
        ttp_index, ttp_hours, ratio_at_ttp, ratio_endpoint = compute_ttp_and_ratio(
            size[idx],
            ratio[idx],
            start_point=start_point,
            steps_per_hour=steps_per_hour,
            ttp_threshold=ttp_threshold,
        )
        ttp_grid[target_idx, replicate_idx] = ttp_hours
        ratio_ttp_grid[target_idx, replicate_idx] = ratio_at_ttp
        ratio_endpoint_grid[target_idx, replicate_idx] = ratio_endpoint
        sensitive_front_velocity_grid[target_idx, replicate_idx] = (
            sample_front_velocity(
                sensitive_front_velocity[idx],
                ttp_index,
                front_velocity_metric,
                front_velocity_window,
                steps_per_hour,
            )
        )
        resistant_front_velocity_grid[target_idx, replicate_idx] = (
            sample_front_velocity(
                resistant_front_velocity[idx],
                ttp_index,
                front_velocity_metric,
                front_velocity_window,
                steps_per_hour,
            )
        )

    if not np.all(status):
        missing = int(np.count_nonzero(~status))
        print(
            f"Warning: {missing} sweep entries are incomplete and will be plotted as NaN."
        )

    return (
        target_efficacy_values,
        ttp_grid,
        ratio_ttp_grid,
        ratio_endpoint_grid,
        sensitive_front_velocity_grid,
        resistant_front_velocity_grid,
        metadata,
    )


def summarize_metric_grid(metric_grid):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(metric_grid, axis=1)
        q1 = np.nanpercentile(metric_grid, 25, axis=1)
        q3 = np.nanpercentile(metric_grid, 75, axis=1)

    return median, q1, q3


def find_resistant_front_growth_onset(x_values, resistant_values, mad_multiplier=3.0, min_consecutive_points=2):
    x_values = np.asarray(x_values, dtype=float)
    resistant_values = np.asarray(resistant_values, dtype=float)
    valid_values = np.isfinite(x_values) & np.isfinite(resistant_values)
    x_values = x_values[valid_values]
    resistant_values = resistant_values[valid_values]

    if x_values.size < 4:
        return None

    order = np.argsort(x_values)
    x_values = x_values[order]
    resistant_values = resistant_values[order]
    resistant_jumps = np.diff(resistant_values)
    largest_jump_index = int(np.nanargmax(resistant_jumps))
    if resistant_jumps[largest_jump_index] <= 0.0:
        return None
    baseline_values = resistant_values[:largest_jump_index]
    if baseline_values.size < 3:
        return None

    baseline = float(np.median(baseline_values))
    baseline_mad = float(np.median(np.abs(baseline_values - baseline)))
    baseline_scale = 1.4826 * baseline_mad
    if np.isclose(baseline_scale, 0.0):
        baseline_scale = float(np.std(baseline_values))
    threshold = baseline + float(mad_multiplier) * baseline_scale
    consecutive_points = max(1, int(min_consecutive_points))

    for index in range(x_values.size - consecutive_points + 1):
        if np.all(resistant_values[index : index + consecutive_points] > threshold):
            return float(x_values[index]), float(resistant_values[index]), baseline, threshold

    return None


def find_front_velocity_crossing(x_values, sensitive_values, resistant_values):
    x_values = np.asarray(x_values, dtype=float)
    sensitive_values = np.asarray(sensitive_values, dtype=float)
    resistant_values = np.asarray(resistant_values, dtype=float)
    valid_values = (
        np.isfinite(x_values)
        & np.isfinite(sensitive_values)
        & np.isfinite(resistant_values)
    )
    x_values = x_values[valid_values]
    sensitive_values = sensitive_values[valid_values]
    resistant_values = resistant_values[valid_values]

    if x_values.size == 0:
        return None

    difference = sensitive_values - resistant_values
    for idx in range(x_values.size):
        if np.isclose(difference[idx], 0.0):
            crossing_y = 0.5 * (sensitive_values[idx] + resistant_values[idx])
            return float(x_values[idx]), float(crossing_y)
        if idx == x_values.size - 1:
            continue
        next_difference = difference[idx + 1]
        if difference[idx] * next_difference < 0.0:
            fraction = difference[idx] / (difference[idx] - next_difference)
            crossing_x = x_values[idx] + fraction * (x_values[idx + 1] - x_values[idx])
            sensitive_crossing_y = sensitive_values[idx] + fraction * (
                sensitive_values[idx + 1] - sensitive_values[idx]
            )
            resistant_crossing_y = resistant_values[idx] + fraction * (
                resistant_values[idx + 1] - resistant_values[idx]
            )
            crossing_y = 0.5 * (sensitive_crossing_y + resistant_crossing_y)
            return float(crossing_x), float(crossing_y)

    return None


def scale_metric_stats(metric_stats, scale_value):
    if not np.isfinite(scale_value) or np.isclose(scale_value, 0.0):
        raise ValueError("gamma^*_front must be finite and non-zero.")

    median, q1, q3 = metric_stats
    scaled_median = median / scale_value
    scaled_q1 = q1 / scale_value
    scaled_q3 = q3 / scale_value

    if scale_value < 0.0:
        scaled_q1, scaled_q3 = np.minimum(scaled_q1, scaled_q3), np.maximum(scaled_q1, scaled_q3)

    return scaled_median, scaled_q1, scaled_q3


def value_at_position(x_values, values, position):
    x_values = np.asarray(x_values, dtype=float)
    values = np.asarray(values, dtype=float)
    matching_indices = np.flatnonzero(np.isclose(x_values, float(position)))
    if matching_indices.size == 0:
        raise ValueError("The gamma^*_front position is not present in the sweep axis.")
    value = float(values[matching_indices[0]])
    if not np.isfinite(value):
        raise ValueError("The sensitive gamma_front at gamma^*_front is not finite.")
    return value


def plot_gamma_front_star_line(ax, gamma_front_star_position, label=None):
    if gamma_front_star_position is None or not np.isfinite(gamma_front_star_position):
        return

    ax.axvline(gamma_front_star_position, color="black", linestyle=":", linewidth=1.0, label=label, zorder=4)


def plot_ttp_ratio_panel(ax, x_values, ttp_stats, ratio_stats, ratio_ylabel, gamma_front_star_position=None):
    ttp_median, ttp_q1, ttp_q3 = ttp_stats
    ratio_median, ratio_q1, ratio_q3 = ratio_stats

    ax.set_ylabel("TTP (h)", color="c")
    ax.plot(x_values, ttp_median, color="c", linewidth=1.5)
    ax.fill_between(x_values, ttp_q1, ttp_q3, alpha=0.5, color="c", linewidth=0)

    ratio_ax = ax.twinx()
    ratio_ax.set_ylabel(ratio_ylabel, rotation=270, labelpad=10, color="m")
    ratio_ax.set_ylim(-0.05, 1.05)
    ratio_ax.plot(x_values, ratio_median, "mx", markersize=3)
    ratio_ax.fill_between(
        x_values, ratio_q1, ratio_q3, alpha=0.5, color="m", linewidth=0
    )
    plot_gamma_front_star_line(ratio_ax, gamma_front_star_position)

    x_min = float(x_values[0])
    x_max = float(x_values[-1])
    if np.isclose(x_min, x_max):
        padding = 0.05 if np.isclose(x_min, 0.0) else max(0.01, 0.05 * abs(x_min))
        ax.set_xlim(x_min - padding, x_max + padding)
    else:
        ax.set_xlim(x_min, x_max)


def plot_front_velocity_panel(ax, x_values, sensitive_stats, resistant_stats, ylabel, gamma_front_star_position=None):
    sensitive_median, sensitive_q1, sensitive_q3 = sensitive_stats
    resistant_median, resistant_q1, resistant_q3 = resistant_stats

    ax.plot(
        x_values,
        sensitive_median,
        color="royalblue",
        linewidth=1.5,
        linestyle="-",
        label="Sensitive",
    )
    ax.fill_between(
        x_values, sensitive_q1, sensitive_q3, color="royalblue", alpha=0.22, linewidth=0
    )
    ax.plot(
        x_values,
        resistant_median,
        color="goldenrod",
        linewidth=1.5,
        linestyle="-",
        label="Resistant",
    )
    ax.fill_between(
        x_values, resistant_q1, resistant_q3, color="goldenrod", alpha=0.18, linewidth=0
    )
    plot_gamma_front_star_line(ax, gamma_front_star_position, label=r"$\gamma^*_\mathrm{front}$")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", frameon=False, fontsize=6)

    x_min = float(x_values[0])
    x_max = float(x_values[-1])
    if np.isclose(x_min, x_max):
        padding = 0.05 if np.isclose(x_min, 0.0) else max(0.01, 0.05 * abs(x_min))
        ax.set_xlim(x_min - padding, x_max + padding)
    else:
        ax.set_xlim(x_min, x_max)


def plot_continuous_dose_efficacy_sweep(
    save_folder,
    *,
    output_path=None,
    show=True,
    steps_per_hour=20.0,
    ttp_threshold=71.0,
    ratio_metric="ttp",
    front_velocity_metric="ttp",
    front_velocity_window=51,
    front_growth_mad_multiplier=3.0,
    front_growth_min_consecutive_points=2,
):
    (
        target_efficacy_values,
        ttp_grid,
        ratio_ttp_grid,
        ratio_endpoint_grid,
        sensitive_front_velocity_grid,
        resistant_front_velocity_grid,
        metadata,
    ) = load_continuous_dose_sweep_metrics(
        save_folder,
        steps_per_hour=steps_per_hour,
        ttp_threshold=ttp_threshold,
        front_velocity_metric=front_velocity_metric,
        front_velocity_window=front_velocity_window,
    )

    if ratio_metric == "ttp":
        ratio_grid = ratio_ttp_grid
        ratio_ylabel = "Resistant fraction"
    elif ratio_metric == "endpoint":
        ratio_grid = ratio_endpoint_grid
        ratio_ylabel = "Final resistant fraction"
    else:
        raise ValueError("ratio_metric must be either 'ttp' or 'endpoint'.")

    if front_velocity_metric == "ttp":
        front_velocity_ylabel = r"$\gamma_\mathrm{front}/\gamma^*_\mathrm{front}$"
    elif front_velocity_metric == "endpoint":
        front_velocity_ylabel = r"Endpoint $\gamma_\mathrm{front}/\gamma^*_\mathrm{front}$"
    else:
        raise ValueError("front_velocity_metric must be either 'ttp' or 'endpoint'.")

    ttp_median, ttp_q1, ttp_q3 = summarize_metric_grid(ttp_grid)
    ratio_median, ratio_q1, ratio_q3 = summarize_metric_grid(ratio_grid)
    sensitive_front_velocity_stats = summarize_metric_grid(
        sensitive_front_velocity_grid
    )
    resistant_front_velocity_stats = summarize_metric_grid(
        resistant_front_velocity_grid
    )

    if np.all(np.isnan(ttp_median)):
        raise ValueError("The selected sweep contains only NaN TTP values.")
    if np.all(np.isnan(ratio_median)):
        raise ValueError("The selected sweep contains only NaN ratio values.")
    if np.all(np.isnan(sensitive_front_velocity_stats[0])):
        raise ValueError(
            "The selected sweep contains only NaN sensitive front-velocity values."
        )
    if np.all(np.isnan(resistant_front_velocity_stats[0])):
        raise ValueError(
            "The selected sweep contains only NaN resistant front-velocity values."
        )

    gamma_front_star = find_resistant_front_growth_onset(
        target_efficacy_values,
        resistant_front_velocity_stats[0],
        mad_multiplier=front_growth_mad_multiplier,
        min_consecutive_points=front_growth_min_consecutive_points,
    )
    if gamma_front_star is None:
        raise ValueError("The resistant front-growth onset could not be determined.")
    gamma_front_star_value = value_at_position(target_efficacy_values, sensitive_front_velocity_stats[0], gamma_front_star[0])
    sensitive_front_velocity_stats = scale_metric_stats(sensitive_front_velocity_stats, gamma_front_star_value)
    resistant_front_velocity_stats = scale_metric_stats(resistant_front_velocity_stats, gamma_front_star_value)
    print(
        f"gamma^*_front position from resistant growth onset: treatment efficacy = {gamma_front_star[0]:.6g}, sensitive gamma^*_front = {gamma_front_star_value:.6g} 1/h, resistant gamma_front = {gamma_front_star[1]:.6g} 1/h, baseline = {gamma_front_star[2]:.6g} 1/h, threshold = {gamma_front_star[3]:.6g} 1/h"
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(3.2, 3.8),
        dpi=300,
        sharex=True,
    )

    plot_ttp_ratio_panel(
        axes[0],
        target_efficacy_values,
        (ttp_median, ttp_q1, ttp_q3),
        (ratio_median, ratio_q1, ratio_q3),
        ratio_ylabel,
        gamma_front_star[0],
    )
    plot_front_velocity_panel(
        axes[1],
        target_efficacy_values,
        sensitive_front_velocity_stats,
        resistant_front_velocity_stats,
        front_velocity_ylabel,
        gamma_front_star[0],
    )

    axes[1].set_xlabel("Effective treatment strength")

    if metadata.get("num_replicates") is not None:
        axes[0].text(
            0.98,
            0.05,
            f"n = {int(metadata['num_replicates'])}\nband = IQR",
            transform=axes[0].transAxes,
            ha="right",
            va="bottom",
            fontsize=6,
            color="black",
        )

    # fig.suptitle("Continuous-dose efficacy sweep", y=0.99, fontsize=8)
    plt.tight_layout()

    if output_path is not None:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fig.savefig(output_path, transparent=True, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return gamma_front_star


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = find_project_root(current_dir, "requirements.txt") or os.getcwd()

    config = get_plot_config()

    plot_continuous_dose_efficacy_sweep(
        save_folder=resolve_path(project_root, config["save_folder"]),
        output_path=resolve_path(project_root, config["output_path"]),
        show=config["show_plot"],
        steps_per_hour=config["steps_per_hour"],
        ttp_threshold=config["ttp_threshold"],
        ratio_metric=config["ratio_metric"],
        front_velocity_metric=config["front_velocity_metric"],
        front_velocity_window=config["front_velocity_window"],
        front_growth_mad_multiplier=config["front_growth_mad_multiplier"],
        front_growth_min_consecutive_points=config["front_growth_min_consecutive_points"],
    )


if __name__ == "__main__":
    main()
