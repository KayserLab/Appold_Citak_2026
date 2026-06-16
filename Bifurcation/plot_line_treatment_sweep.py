import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import yaml

try:
    from . import plot_style
    from . import treat_core
except ImportError:
    import plot_style
    import treat_core


plot_style.configure_plot_style()


def get_metric_files():
    return {
        "last_cycle_start_efficacy": "last_cycle_start_efficacy.dat",
        "last_cycle_mean_efficacy": "last_cycle_mean_efficacy.dat",
    }


def get_metric_titles():
    return {
        "last_cycle_start_efficacy": "Last cycle-start efficacy",
        "last_cycle_mean_efficacy": "Last full-cycle average growth rate",
    }


def resolve_path(project_root, path):
    if os.path.isabs(path):
        return path
    return os.path.join(project_root, path)


def load_metadata(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def load_treatment_sweep(save_folder, metric_key="last_cycle_start_efficacy"):
    metric_files = get_metric_files()
    if metric_key not in metric_files:
        raise ValueError(f"Unsupported metric: {metric_key}")

    params_list = np.load(os.path.join(save_folder, "params_list.npy"))
    sweep_axes = np.load(os.path.join(save_folder, "sweep_axes.npz"))
    metadata = load_metadata(os.path.join(save_folder, "metadata.yaml"))

    tau_on_values = sweep_axes["tau_on"]
    tau_off_values = sweep_axes["tau_off"]
    num_replicates = int(metadata["num_replicates"])
    num_sim = len(params_list)
    efficacy = np.memmap(os.path.join(save_folder, metric_files[metric_key]), dtype=np.float32, mode="r", shape=(num_sim,))
    status = np.memmap(os.path.join(save_folder, "status.dat"), dtype=np.bool_, mode="r", shape=(num_sim,))

    tau_on_index = {int(value): idx for idx, value in enumerate(tau_on_values)}
    tau_off_index = {int(value): idx for idx, value in enumerate(tau_off_values)}
    grid = np.full((len(tau_on_values), len(tau_off_values), num_replicates), np.nan, dtype=np.float32)

    for idx, (tau_on, tau_off, replicate, _) in enumerate(params_list):
        if status[idx]:
            grid[tau_on_index[int(tau_on)], tau_off_index[int(tau_off)], int(replicate)] = efficacy[idx]

    if not np.all(status):
        missing = int(np.count_nonzero(~status))
        print(f"Warning: {missing} sweep entries are incomplete and will be plotted as NaN.")

    return grid, tau_on_values, tau_off_values, metadata


def select_tau_off_slice(grid, tau_off_values, selected_tau_off=None, steps_per_hour=20):
    if selected_tau_off is None:
        if len(tau_off_values) != 1:
            raise ValueError(
                "The sweep contains multiple tau_off values. Set selected_tau_off explicitly."
            )
        tau_off_index = 0
    else:
        target_value = (
            selected_tau_off
            if steps_per_hour is None
            else selected_tau_off * steps_per_hour
        )
        matches = np.where(
            np.isclose(tau_off_values.astype(float), float(target_value))
        )[0]
        if len(matches) == 0:
            raise ValueError(
                "The requested tau_off value is not present in the saved sweep."
            )
        tau_off_index = int(matches[0])

    return grid[:, tau_off_index, :], tau_off_values[tau_off_index]


def subset_tau_on(line_grid, tau_on_values, tau_on_limits=None, steps_per_hour=20):
    if tau_on_limits is None:
        return line_grid, tau_on_values

    scale = steps_per_hour if steps_per_hour is not None else 1
    tau_on_min, tau_on_max = tau_on_limits
    tau_on_mask = (tau_on_values >= tau_on_min * scale) & (
        tau_on_values <= tau_on_max * scale
    )

    if not np.any(tau_on_mask):
        raise ValueError("No tau_on values fall within the requested plot window.")

    return line_grid[tau_on_mask], tau_on_values[tau_on_mask]


def summarize_line(line_grid):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(line_grid, axis=1)
        variance = np.nanvar(line_grid, axis=1)

    if np.all(np.isnan(mean)):
        raise ValueError(
            "The selected line sweep contains only NaN values. "
            "This usually means no valid treatment cycle was generated for the saved tau_on/tau_off values."
        )

    return mean, variance


def format_tau_axis(values, steps_per_hour=20):
    if steps_per_hour is None:
        return values.astype(float), "steps"
    return values.astype(float) / float(steps_per_hour), "h"


def format_tau_value(value, steps_per_hour=20):
    if steps_per_hour is None:
        return str(int(value)), "steps"

    hours = float(value) / float(steps_per_hour)
    if hours.is_integer():
        return str(int(hours)), "h"
    return f"{hours:.2f}", "h"


def build_variance_band(mean, variance, clip_range=(0.0, 1.0)):
    lower = mean - variance
    upper = mean + variance

    if clip_range is not None:
        lower = np.clip(lower, clip_range[0], clip_range[1])
        upper = np.clip(upper, clip_range[0], clip_range[1])

    return lower, upper


def plot_line_sweep(save_folder, metric_key="last_cycle_start_efficacy", output_path=None, selected_tau_off=None, tau_on_limits=None, steps_per_hour=20, show=True, line_color="C0", band_color=None, band_alpha=0.25):
    grid, tau_on_values, tau_off_values, metadata = load_treatment_sweep(
        save_folder,
        metric_key=metric_key,
    )
    line_grid, tau_off_value = select_tau_off_slice(
        grid,
        tau_off_values,
        selected_tau_off=selected_tau_off,
        steps_per_hour=steps_per_hour,
    )
    line_grid, tau_on_values = subset_tau_on(
        line_grid,
        tau_on_values,
        tau_on_limits=tau_on_limits,
        steps_per_hour=steps_per_hour,
    )

    mean, variance = summarize_line(line_grid)
    band_lower, band_upper = build_variance_band(mean, variance)

    x_values, x_unit = format_tau_axis(tau_on_values, steps_per_hour=steps_per_hour)
    tau_off_label, tau_off_unit = format_tau_value(
        tau_off_value, steps_per_hour=steps_per_hour
    )
    metric_title = get_metric_titles()[metric_key]

    if band_color is None:
        band_color = line_color

    fig, ax = plt.subplots(figsize=(3.1, 2.15), dpi=300)
    ax.plot(x_values, 1 - mean, color=line_color, linewidth=1.4)
    ax.fill_between(
        x_values,
        1 - band_lower,
        1 - band_upper,
        color=band_color,
        alpha=band_alpha,
        linewidth=0,
    )

    ax.set_xlabel(rf"$\tau_{{\mathrm{{on}}}}$ ({x_unit})")
    ax.set_ylabel(r"Effective growth rate, $\xi$")
    ax.set_title(
        f"{metric_title}, "
        + rf"$\tau_{{\mathrm{{off}}}}$ = {tau_off_label} {tau_off_unit}"
    )
    ax.set_ylim(-0.02, 1.02)

    if metadata.get("noise", False):
        ax.text(
            0.98,
            0.02,
            f"n = {metadata['num_replicates']}\nband = mean +/- variance",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6,
            color="black",
        )

    plt.tight_layout()

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, transparent=True, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = (
        treat_core.find_project_root(current_dir, "requirements.txt") or os.getcwd()
    )

    config = {
        "save_folder": os.path.join("data", "sweeps", "treatment_only_sweep_rebuttal"),
        "start_output_path": os.path.join("Bifurcation", "plots", "treatment_only_sweep_rebuttal.pdf"),
        "cycle_mean_output_path": os.path.join("Bifurcation", "plots", "treatment_only_sweep_cycle_rebuttal.pdf"),
        "steps_per_hour": 20,
        "selected_tau_off_hours": None,
        "tau_on_limits_hours": None,
        "show_plot": True,
    }

    plot_line_sweep(
        save_folder=resolve_path(project_root, config["save_folder"]),
        metric_key="last_cycle_start_efficacy",
        output_path=resolve_path(project_root, config["start_output_path"]),
        selected_tau_off=config["selected_tau_off_hours"],
        tau_on_limits=config["tau_on_limits_hours"],
        steps_per_hour=config["steps_per_hour"],
        show=config["show_plot"],
        line_color="#0072B2",
        band_color="#0072B2",
    )
    plot_line_sweep(
        save_folder=resolve_path(project_root, config["save_folder"]),
        metric_key="last_cycle_mean_efficacy",
        output_path=resolve_path(project_root, config["cycle_mean_output_path"]),
        selected_tau_off=config["selected_tau_off_hours"],
        tau_on_limits=config["tau_on_limits_hours"],
        steps_per_hour=config["steps_per_hour"],
        show=config["show_plot"],
        line_color="#D55E00",
        band_color="#D55E00",
    )


if __name__ == "__main__":
    main()
