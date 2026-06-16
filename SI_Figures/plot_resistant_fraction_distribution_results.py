import csv
import pathlib as pl
import sys

import matplotlib.pyplot as plt
import numpy as np


def get_project_root():
    return pl.Path(__file__).resolve().parent.parent


def configure_import_paths():
    project_root = get_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


def configure_plot_style():
    configure_import_paths()
    from Bifurcation import plot_style
    plot_style.configure_plot_style()
    plt.rcParams["axes.linewidth"] = 0.5
    plt.rcParams["axes.titlesize"] = 7
    plt.rcParams["lines.linewidth"] = 1.0
    plt.rcParams["xtick.major.size"] = 3
    plt.rcParams["ytick.major.size"] = 3
    plt.rcParams["xtick.major.width"] = 0.5
    plt.rcParams["ytick.major.width"] = 0.5
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"


def default_config():
    project_root = get_project_root()
    result_stem = project_root / "SI_Figures" / "plots" / "resistant_fraction_distribution_rebuttal" / "final_resistant_fraction_distribution"
    return {
        "result_stem": result_stem,
        "output_dir": result_stem.parent / "distribution_replots",
        "measurement_modes": ["endpoint", "ttp"],
        "value_source": "runs_csv",
        "fallback_to_values_csv": True,
        "treatment_ids": None,
        "triptych_groups": [{"id": "continuous_transition", "treatment_ids": ["continuous_cap_0p5", "continuous_cap_0p52", "continuous_cap_0p533", "continuous_cap_0p546", "continuous_cap_0p57"]}, {"id": "schedule_transition", "treatment_ids": ["met_5_50_18", "met_5_70_18", "met_5_80_18"]}],
        "plot_individual_distributions": True,
        "plot_triptychs": True,
        "plot_continuous_metric_summary": True,
        "metric_target_efficacies": [0.5, 0.52, 0.533, 0.546, 0.57],
        "metric_target_efficacy_tolerance": 1e-9,
        "metric_columns": 3,
        "continuous_metric_keys": ["std_sample", "skewness", "binder_centered"],
        "continuous_metric_labels": {"std_sample": "Std", "scaled_std": "Scaled std", "escape_probability": "Escape probability", "skewness": "Skewness", "binder_centered": "Centered Binder"},
        "escape_probability_threshold": 0.05,
        "figure_width_pt": 460.0,
        "panels_per_figure_width": 3.0,
        "triptych_columns": 3,
        "single_panel_height_in": 2.0,
        "triptych_height_in": 2.0,
        "metric_summary_height_in": 1.85,
        "point_to_inch": 72.27,
        "dpi": 600,
        "formats": ["pdf", "png"],
        "bins": 50,
        "show_kde": False,
        "show_mean": False,
        "show_median": False,
        "show_n": True,
        "hist_alpha": 0.92,
        "hist_edge_color": "white",
        "hist_edge_linewidth": 0.35,
        "kde_color": "black",
        "kde_linewidth": 0.9,
        "metric_color": "black",
        "metric_marker_size": 3,
        "metric_linewidth": 1.0,
        "mean_linewidth": 0.8,
        "median_linewidth": 0.8,
        "grid_color": "#d9d9d9",
        "grid_linewidth": 0.4,
        "grid_alpha": 0.8,
        "continuous_color_map": "viridis",
        "continuous_color_limits": [0.2, 0.85],
        "fallback_color_map": "tab20",
        "transparent": True,
        "use_tight_bbox": False,
        "layout_pad": 0.25,
    }


def resolve_path(project_root, path):
    resolved = pl.Path(path)
    if resolved.is_absolute():
        return resolved
    return project_root / resolved


def normalize_stem(project_root, path):
    result_stem = resolve_path(project_root, path)
    if result_stem.suffix:
        result_stem = result_stem.with_suffix("")
    return result_stem


def values_csv_path(result_stem, measurement_mode):
    return result_stem.with_name(f"{result_stem.name}_{measurement_mode}_values.csv")


def runs_csv_paths(result_stem):
    return sorted(result_stem.parent.glob(f"{result_stem.name}_*_runs.csv"))


def load_csv_rows(path):
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def save_csv_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return path
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def float_or_nan(value):
    try:
        if value in ("", None):
            return np.nan
        return float(value)
    except ValueError:
        return np.nan


def treatment_sort_value(row):
    control_parameter = float_or_nan(row.get("control_parameter"))
    target_efficacy = float_or_nan(row.get("target_efficacy"))
    if np.isfinite(target_efficacy):
        return 0, target_efficacy, row.get("treatment_id", "")
    if np.isfinite(control_parameter):
        return 1, control_parameter, row.get("treatment_id", "")
    return 2, row.get("treatment_label", ""), row.get("treatment_id", "")


def first_rows_by_treatment(rows):
    first_rows = {}
    for row in rows:
        treatment_id = row.get("treatment_id")
        if treatment_id not in first_rows:
            first_rows[treatment_id] = row
    return first_rows


def ordered_treatment_ids(rows, selected_ids):
    first_rows = first_rows_by_treatment(rows)
    if selected_ids is not None:
        return [treatment_id for treatment_id in selected_ids if treatment_id in first_rows]
    ordered_rows = sorted(first_rows.values(), key=treatment_sort_value)
    return [row["treatment_id"] for row in ordered_rows]


def rows_for_treatment(rows, treatment_id):
    return [row for row in rows if row.get("treatment_id") == treatment_id]


def finite_resistant_fractions(rows):
    values = np.asarray([float_or_nan(row.get("resistant_fraction")) for row in rows], dtype=float)
    return values[np.isfinite(values)]


def calculate_distribution_metrics(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    metrics = {"n_runs": int(values.size), "mean": np.nan, "std_population": np.nan, "std_sample": np.nan, "scaled_std": np.nan, "cv_population": np.nan, "cv_sample": np.nan, "skewness": np.nan, "binder_centered": np.nan}
    if values.size == 0:
        return metrics
    mean = float(np.mean(values))
    centered = values - mean
    central_moment_2 = float(np.mean(centered**2))
    central_moment_3 = float(np.mean(centered**3))
    central_moment_4 = float(np.mean(centered**4))
    std_population = float(np.sqrt(central_moment_2))
    std_sample = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    skewness = float(central_moment_3 / (central_moment_2 ** 1.5)) if central_moment_2 > 0 else np.nan
    binder_centered = float(1.0 - central_moment_4 / (3.0 * central_moment_2 * central_moment_2)) if central_moment_2 > 0 else np.nan
    metrics.update({"mean": mean, "std_population": std_population, "std_sample": std_sample, "scaled_std": float(2.0 * std_population), "cv_population": float(std_population / mean) if mean != 0 else np.nan, "cv_sample": float(std_sample / mean) if mean != 0 else np.nan, "skewness": skewness, "binder_centered": binder_centered})
    return metrics


def calculate_escape_probability(values, threshold):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.mean(values > float(threshold)))


def rows_from_run_rows(run_rows, measurement_mode):
    selected_rows = []
    field = f"{measurement_mode}_resistant_fraction"
    for row in run_rows:
        if field not in row:
            continue
        selected = dict(row)
        selected["measurement_mode"] = measurement_mode
        selected["resistant_fraction"] = row[field]
        selected_rows.append(selected)
    return selected_rows


def load_rows_from_runs_csv(result_stem, measurement_mode):
    rows = []
    for path in runs_csv_paths(result_stem):
        rows.extend(rows_from_run_rows(load_csv_rows(path), measurement_mode))
    return rows


def load_rows_from_values_csv(result_stem, measurement_mode):
    path = values_csv_path(result_stem, measurement_mode)
    if not path.exists():
        return []
    return load_csv_rows(path)


def load_measurement_rows(result_stem, measurement_mode, config):
    if config["value_source"] == "values_csv":
        rows = load_rows_from_values_csv(result_stem, measurement_mode)
    elif config["value_source"] == "runs_csv":
        rows = load_rows_from_runs_csv(result_stem, measurement_mode)
    else:
        raise ValueError("value_source must be 'runs_csv' or 'values_csv'.")
    if rows or not bool(config["fallback_to_values_csv"]):
        return rows
    return load_rows_from_values_csv(result_stem, measurement_mode)


def treatment_label(row):
    target_efficacy = float_or_nan(row.get("target_efficacy"))
    if np.isfinite(target_efficacy):
        return f"Efficacy {target_efficacy:.3f}"
    return row.get("treatment_label", row.get("treatment_id", "Treatment"))


def is_continuous_treatment(row):
    return row.get("treatment_mode") == "continuous_constant_efficacy" or np.isfinite(float_or_nan(row.get("target_efficacy")))


def metric_efficacy_is_selected(target_efficacy, config):
    selected = config["metric_target_efficacies"]
    if selected is None:
        return True
    tolerance = float(config["metric_target_efficacy_tolerance"])
    return any(abs(float(target_efficacy) - float(value)) <= tolerance for value in selected)


def treatment_color(row, treatment_index, treatment_count, config):
    target_efficacy = float_or_nan(row.get("target_efficacy"))
    if np.isfinite(target_efficacy):
        low, high = [float(value) for value in config["continuous_color_limits"]]
        values = np.linspace(low, high, max(1, treatment_count))
        return plt.colormaps[config["continuous_color_map"]](float(values[int(treatment_index)]))
    return plt.colormaps[config["fallback_color_map"]](int(treatment_index) % 20)


def points_to_inches(points, config):
    return float(points) / float(config["point_to_inch"])


def single_panel_size(config):
    width = points_to_inches(float(config["figure_width_pt"]) / float(config["panels_per_figure_width"]), config)
    return width, float(config["single_panel_height_in"])


def triptych_size(n_rows, config):
    width = points_to_inches(float(config["figure_width_pt"]), config)
    height = float(config["triptych_height_in"]) * int(n_rows)
    return width, height


def metric_summary_size(config):
    return points_to_inches(float(config["figure_width_pt"]), config), float(config["metric_summary_height_in"])


def kde_scaled_to_counts(values, grid, bin_width):
    if values.size <= 1 or float(np.std(values)) <= 0:
        return np.zeros_like(grid)
    try:
        from scipy.stats import gaussian_kde
        return gaussian_kde(values)(grid) * float(values.size) * float(bin_width)
    except Exception:
        return np.zeros_like(grid)


def configure_axis(ax, show_ylabel, config):
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xlabel("Resistant fraction")
    ax.set_ylabel("Run count" if show_ylabel else "")
    ax.grid(axis="y", color=config["grid_color"], linewidth=float(config["grid_linewidth"]), alpha=float(config["grid_alpha"]))
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="both", length=3, width=0.5, pad=1)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


def configure_metric_axis(ax, config):
    ax.grid(axis="y", color=config["grid_color"], linewidth=float(config["grid_linewidth"]), alpha=float(config["grid_alpha"]))
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="both", length=3, width=0.5, pad=1)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


def add_summary_marks(ax, values, config):
    if values.size == 0:
        return
    if bool(config["show_mean"]):
        ax.axvline(float(np.mean(values)), color="black", linewidth=float(config["mean_linewidth"]), linestyle=":")
    if bool(config["show_median"]):
        ax.axvline(float(np.median(values)), color="black", linewidth=float(config["median_linewidth"]), linestyle="--")


def add_sample_count(ax, values, config):
    if not bool(config["show_n"]):
        return
    ax.text(0.98, 0.94, f"n={values.size}", ha="right", va="top", transform=ax.transAxes, fontsize=6)


def plot_distribution(ax, values, label, color, show_ylabel, config):
    edges = np.linspace(0.0, 1.0, int(config["bins"]) + 1)
    ax.hist(values, bins=edges, color=color, edgecolor=config["hist_edge_color"], linewidth=float(config["hist_edge_linewidth"]), alpha=float(config["hist_alpha"]))
    if bool(config["show_kde"]):
        grid = np.linspace(0.0, 1.0, 256)
        density = kde_scaled_to_counts(values, grid, float(edges[1] - edges[0]))
        if np.nanmax(density) > 0:
            ax.plot(grid, density, color=config["kde_color"], linewidth=float(config["kde_linewidth"]))
    add_summary_marks(ax, values, config)
    add_sample_count(ax, values, config)
    configure_axis(ax, show_ylabel, config)
    ax.set_title(label, fontsize=7, pad=2)


def save_figure(fig, base_path, config):
    saved_paths = []
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=float(config["layout_pad"]))
    for file_format in config["formats"]:
        path = base_path.with_suffix(f".{file_format}")
        if bool(config["use_tight_bbox"]):
            fig.savefig(path, dpi=int(config["dpi"]), transparent=bool(config["transparent"]), bbox_inches="tight")
        else:
            fig.savefig(path, dpi=int(config["dpi"]), transparent=bool(config["transparent"]))
        saved_paths.append(path)
    return saved_paths


def plot_individual_distribution(rows, treatment_id, treatment_index, treatment_count, measurement_mode, output_stem, config):
    treatment_rows = rows_for_treatment(rows, treatment_id)
    if not treatment_rows:
        return []
    values = finite_resistant_fractions(treatment_rows)
    label = treatment_label(treatment_rows[0])
    color = treatment_color(treatment_rows[0], treatment_index, treatment_count, config)
    fig, ax = plt.subplots(figsize=single_panel_size(config), dpi=int(config["dpi"]))
    plot_distribution(ax, values, label, color, True, config)
    base_path = output_stem / f"{measurement_mode}_{treatment_id}_distribution"
    saved_paths = save_figure(fig, base_path, config)
    plt.close(fig)
    return saved_paths


def plot_triptych(rows, group, measurement_mode, output_stem, config):
    treatment_ids = [treatment_id for treatment_id in group["treatment_ids"] if rows_for_treatment(rows, treatment_id)]
    if not treatment_ids:
        return []
    n_cols = max(1, int(config["triptych_columns"]))
    n_rows = int(np.ceil(len(treatment_ids) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=triptych_size(n_rows, config), dpi=int(config["dpi"]), squeeze=False)
    for index, treatment_id in enumerate(treatment_ids):
        treatment_rows = rows_for_treatment(rows, treatment_id)
        values = finite_resistant_fractions(treatment_rows)
        label = treatment_label(treatment_rows[0])
        color = treatment_color(treatment_rows[0], index, len(treatment_ids), config)
        row_index = index // n_cols
        column_index = index % n_cols
        plot_distribution(axes[row_index, column_index], values, label, color, column_index == 0, config)
    for index in range(len(treatment_ids), n_rows * n_cols):
        axes[index // n_cols, index % n_cols].axis("off")
    base_path = output_stem / f"{measurement_mode}_{group['id']}_triptych"
    saved_paths = save_figure(fig, base_path, config)
    plt.close(fig)
    return saved_paths


def continuous_metric_rows(rows, treatment_ids, config):
    metric_rows = []
    escape_threshold = float(config["escape_probability_threshold"])
    for treatment_id in treatment_ids:
        treatment_rows = rows_for_treatment(rows, treatment_id)
        if not treatment_rows or not is_continuous_treatment(treatment_rows[0]):
            continue
        values = finite_resistant_fractions(treatment_rows)
        metrics = calculate_distribution_metrics(values)
        metrics["escape_probability"] = calculate_escape_probability(values, escape_threshold)
        metrics["escape_probability_threshold"] = escape_threshold
        row = {"treatment_id": treatment_id, "treatment_label": treatment_rows[0].get("treatment_label", treatment_id), "target_efficacy": float_or_nan(treatment_rows[0].get("target_efficacy"))}
        if not np.isfinite(row["target_efficacy"]) or not metric_efficacy_is_selected(row["target_efficacy"], config):
            continue
        row.update(metrics)
        metric_rows.append(row)
    return metric_rows


def metric_plot_positions(metric_keys, config):
    n_cols = min(max(1, int(config["metric_columns"])), len(metric_keys))
    n_rows = int(np.ceil(len(metric_keys) / n_cols))
    return n_rows, n_cols


def plot_continuous_metric_summary(rows, treatment_ids, measurement_mode, output_dir, config):
    metric_rows = continuous_metric_rows(rows, treatment_ids, config)
    if not metric_rows:
        return []
    metric_rows = sorted(metric_rows, key=lambda row: row["target_efficacy"])
    csv_path = save_csv_rows(output_dir / f"{measurement_mode}_continuous_metric_summary.csv", metric_rows)
    metric_keys = list(config["continuous_metric_keys"])
    n_rows, n_cols = metric_plot_positions(metric_keys, config)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=metric_summary_size(config), dpi=int(config["dpi"]), squeeze=False)
    x_values = np.asarray([row["target_efficacy"] for row in metric_rows], dtype=float)
    for index, metric_key in enumerate(metric_keys):
        ax = axes[index // n_cols, index % n_cols]
        y_values = np.asarray([float_or_nan(row.get(metric_key)) for row in metric_rows], dtype=float)
        ax.plot(x_values, y_values, marker="o", markersize=float(config["metric_marker_size"]), color=config["metric_color"], linewidth=float(config["metric_linewidth"]))
        ax.axhline(0.0, color="#777777", linewidth=0.5, linestyle=":")
        if metric_key in {"scaled_std", "escape_probability"}:
            ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Effective treatment strength")
        ax.set_ylabel(config["continuous_metric_labels"].get(metric_key, metric_key))
        configure_metric_axis(ax, config)
    for index in range(len(metric_keys), n_rows * n_cols):
        axes[index // n_cols, index % n_cols].axis("off")
    base_path = output_dir / f"{measurement_mode}_continuous_metric_summary"
    saved_paths = save_figure(fig, base_path, config)
    plt.close(fig)
    return [csv_path] + saved_paths


def plot_measurement_mode(result_stem, measurement_mode, output_dir, config):
    rows = load_measurement_rows(result_stem, measurement_mode, config)
    if not rows:
        raise ValueError(f"No saved resistant-fraction rows were found for measurement mode '{measurement_mode}'.")
    saved_paths = []
    treatment_ids = ordered_treatment_ids(rows, config["treatment_ids"])
    if bool(config["plot_individual_distributions"]):
        for index, treatment_id in enumerate(treatment_ids):
            saved_paths.extend(plot_individual_distribution(rows, treatment_id, index, len(treatment_ids), measurement_mode, output_dir, config))
    if bool(config["plot_triptychs"]):
        for group in config["triptych_groups"]:
            saved_paths.extend(plot_triptych(rows, group, measurement_mode, output_dir, config))
    if bool(config["plot_continuous_metric_summary"]):
        saved_paths.extend(plot_continuous_metric_summary(rows, treatment_ids, measurement_mode, output_dir, config))
    return saved_paths


def main(config=None):
    configure_plot_style()
    project_root = configure_import_paths()
    run_config = default_config()
    if config is not None:
        run_config.update(config)
    result_stem = normalize_stem(project_root, run_config["result_stem"])
    output_dir = resolve_path(project_root, run_config["output_dir"])
    saved_paths = []
    for measurement_mode in run_config["measurement_modes"]:
        saved_paths.extend(plot_measurement_mode(result_stem, measurement_mode, output_dir, run_config))
    print(f"Saved {len(saved_paths)} distribution plot file(s) to {output_dir}")
    for path in saved_paths:
        print(path)
    return saved_paths


if __name__ == "__main__":
    run_config = default_config()
    main(run_config)
