import copy
import pathlib as pl
import sys

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def get_project_root():
    return pl.Path(__file__).resolve().parent.parent


def default_config():
    return {"config_path": "SI_Figures/loocv_bootstrap_plot_config.yaml", "loocv_run_dir": None, "bootstrap_run_dir": None, "loocv_results_root": "Validation/results/end_to_end", "bootstrap_results_root": "source/fit/fit_results/bootstrap_uncertainties", "output_dir": "SI_Figures/plots/loocv_bootstrap_summary", "output_stem": "loocv_bootstrap_summary", "dpi": 300, "bootstrap_ci_level": 95.0, "bootstrap_output_stem": "bootstrap_parameter_uncertainty", "loocv_prediction": {"output_stem": "loocv_clone_count_prediction", "observed_column": "test_clone_count", "predicted_column": "test_predicted_clone_mean", "prediction_std_column": "test_predicted_clone_std", "relative_error_column": "test_relative_abs_error", "color": "#287d8e", "title": "LOOCV: held-out prediction", "xlabel": "Observed resistant clone count", "ylabel": "Predicted resistant clone count"}, "loocv_headline_metrics": [{"label": "Area NRMSE", "column": "test_area_nrmse", "format": "{:.3f}"}, {"label": "Regrowth RMSE", "column": "test_regrowth_time_rmse_h", "format": "{:.1f} h"}, {"label": "Clone rel. error", "column": "test_relative_abs_error", "format": "{:.2f}"}], "loocv_area_prediction": {"output_stem": "loocv_area_trajectory_prediction", "color": "#287d8e", "title": "LOOCV: area trajectory prediction", "plot_kind": "simulated_scatter", "fold_mode": "all_folds", "timepoint_stride": 1, "area_scale_um_per_pixel": 8.648, "group_column": "held_out_no_treatment_label", "value_column": "test_area_nrmse", "secondary_value_column": "test_area_r2", "xlabel": "Observed colony area (mm^2)", "ylabel": "Predicted colony area (mm^2)", "metric_xlabel": "Area trajectory NRMSE", "label_prefix": "colony_data_", "label_suffix": "_with_clonearea.csv", "value_format": "{:.3f}", "secondary_label": "Area R2", "secondary_format": "{:.3f}"}, "bootstrap_parameters": [{"parameter": "regrowth_t0_h", "label": "Regrowth onset", "unit": "h"}, {"parameter": "regrowth_slope_um_per_h", "label": "Regrowth slope", "unit": "um/h"}, {"parameter": "diffusion_sensitive", "label": "Cell diffusion", "unit": "sim units"}, {"parameter": "uptake_rate", "label": "Nutrient uptake", "unit": "sim units"}, {"parameter": "diffusion_nutrients", "label": "Nutrient diffusion", "unit": "sim units"}, {"parameter": "start_point_from_dispersion_fit", "label": "Start point", "unit": "steps"}, {"parameter": "mutation_rate", "label": "Mutation rate", "unit": "rate units"}]}


def resolve_path(project_root, path):
    resolved = pl.Path(path)
    if resolved.is_absolute():
        return resolved
    return project_root / resolved


def load_config_file(project_root, config_path):
    path = resolve_path(project_root, config_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    return loaded


def merge_config(base_config, user_config):
    merged = copy.deepcopy(base_config)
    for key, value in user_config.items():
        merged[key] = value
    return merged


def configure_plot_style():
    project_root = get_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from Bifurcation import plot_style
    plot_style.configure_plot_style()


def load_validation_common(project_root):
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    import Validation.validation_common as vc
    return vc


def find_latest_loocv_run(project_root, config):
    if config["loocv_run_dir"] is not None:
        return resolve_path(project_root, config["loocv_run_dir"])
    results_root = resolve_path(project_root, config["loocv_results_root"])
    candidates = [path for path in results_root.glob("end_to_end_loocv_*") if (path / "fold_results.csv").exists()]
    if not candidates:
        raise FileNotFoundError(f"No LOOCV fold_results.csv files found below {results_root}.")
    return max(candidates, key=lambda path: path.name)


def find_latest_bootstrap_run(project_root, config):
    if config["bootstrap_run_dir"] is not None:
        return resolve_path(project_root, config["bootstrap_run_dir"])
    results_root = resolve_path(project_root, config["bootstrap_results_root"])
    candidates = [path.parent for path in results_root.glob("run_*/bootstrap_summary.csv")]
    if not candidates:
        raise FileNotFoundError(f"No bootstrap_summary.csv files found below {results_root}.")
    return max(candidates, key=lambda path: path.name)


def load_result_tables(loocv_run_dir, bootstrap_run_dir):
    fold_df = pd.read_csv(loocv_run_dir / "fold_results.csv")
    bootstrap_df = pd.read_csv(bootstrap_run_dir / "bootstrap_summary.csv")
    return fold_df, bootstrap_df


def require_columns(df, columns, table_name):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required column(s): {', '.join(missing)}")


def finite_values(df, column):
    values = df[column].to_numpy(dtype=float)
    return values[np.isfinite(values)]


def area_values_to_mm2(values, spec):
    scale_factor = float(spec.get("area_scale_um_per_pixel", 8.648)) ** 2 / 1.0e6
    return np.asarray(values, dtype=float) * scale_factor


def identity_limits(first_values, second_values):
    first_values = np.asarray(first_values, dtype=float)
    second_values = np.asarray(second_values, dtype=float)
    finite = np.concatenate([first_values[np.isfinite(first_values)], second_values[np.isfinite(second_values)]])
    if finite.size == 0:
        return 0.0, 1.0
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    span = upper - lower
    if span <= 0:
        span = max(abs(lower), 1.0)
    pad = span * 0.06
    return lower - pad, upper + pad


def metric_summary_text(fold_df, metrics):
    lines = []
    for metric in metrics:
        values = finite_values(fold_df, metric["column"])
        if values.size == 0:
            continue
        formatted = metric["format"].format(float(np.nanmedian(values)))
        lines.append(f"{metric['label']} median: {formatted}")
    return "\n".join(lines)


def draw_loocv_prediction(ax, fold_df, config, show_title=True):
    spec = config["loocv_prediction"]
    columns = [spec["observed_column"], spec["predicted_column"], spec["relative_error_column"]]
    if spec.get("prediction_std_column") is not None:
        columns.append(spec["prediction_std_column"])
    require_columns(fold_df, columns, "fold_results.csv")

    observed = fold_df[spec["observed_column"]].to_numpy(dtype=float)
    predicted = fold_df[spec["predicted_column"]].to_numpy(dtype=float)
    finite_mask = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[finite_mask]
    predicted = predicted[finite_mask]

    if spec.get("prediction_std_column") is not None:
        prediction_std = fold_df[spec["prediction_std_column"]].to_numpy(dtype=float)[finite_mask]
        ax.errorbar(observed, predicted, yerr=prediction_std, fmt="none", ecolor="#b7b7b7", alpha=0.35, elinewidth=0.6, capsize=0, zorder=1)

    ax.scatter(observed, predicted, s=20, color=spec["color"], edgecolor="white", linewidth=0.35, alpha=0.82, zorder=2)
    lower, upper = identity_limits(observed, predicted)
    ax.plot([lower, upper], [lower, upper], color="#3f3f3f", linestyle="--", linewidth=1.0, zorder=0)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    if show_title:
        ax.set_title(spec["title"])
    ax.set_xlabel(spec["xlabel"])
    ax.set_ylabel(spec["ylabel"])
    ax.grid(alpha=0.18, linewidth=0.7)

    headline = metric_summary_text(fold_df, config["loocv_headline_metrics"])
    annotation = f"n = {len(observed)} paired held-out folds\n{headline}"
    ax.text(0.04, 0.96, annotation, transform=ax.transAxes, ha="left", va="top", fontsize=6, color="#333333")


def select_area_simulation_rows(fold_df, spec):
    fold_mode = str(spec.get("fold_mode", "all_folds"))
    if fold_mode == "all_folds":
        return fold_df.reset_index(drop=True)
    if fold_mode != "median_per_trajectory":
        raise ValueError(f"Unsupported area fold_mode '{fold_mode}'. Use 'all_folds' or 'median_per_trajectory'.")

    selected_rows = []
    group_column = spec["group_column"]
    value_column = spec["value_column"]
    for _, group in fold_df.groupby(group_column, sort=True):
        values = group[value_column].to_numpy(dtype=float)
        finite_mask = np.isfinite(values)
        if not np.any(finite_mask):
            selected_rows.append(group.iloc[0])
            continue
        median_value = float(np.nanmedian(values[finite_mask]))
        local_index = int(np.nanargmin(np.abs(values - median_value)))
        selected_rows.append(group.iloc[local_index])
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def simulate_area_at_steps(vc, fitted_parameters, sample_steps):
    sample_steps = np.asarray(sample_steps, dtype=int)
    if sample_steps.size == 0:
        return np.array([], dtype=float)
    max_step = int(np.max(sample_steps))
    sim = vc.cr.DiffusionModel2D()
    treatment_times = np.zeros(max_step + 1, dtype=bool)
    vc.configure_dispersion_sim(sim, np.asarray(fitted_parameters, dtype=float))
    sim.params["total_time"] = int(len(treatment_times))
    sim.treatment_times = treatment_times
    sim.random_seed = 1
    sim.set_random_seed()

    nutrients, sensitive, resistant = sim.get_initial_state()
    mutation_scaling = float(sim.params["mutation_scaling"])
    areas = np.empty(len(sample_steps), dtype=float)
    step_lookup = {int(step): index for index, step in enumerate(sample_steps)}
    if 0 in step_lookup:
        areas[step_lookup[0]] = vc._area_from_state(sensitive, resistant, mutation_scaling)

    for timer in range(1, max_step + 1):
        nutrients, sensitive, resistant = sim.update(timer, nutrients, sensitive, resistant)
        if timer in step_lookup:
            areas[step_lookup[timer]] = vc._area_from_state(sensitive, resistant, mutation_scaling)
    return areas


def collect_simulated_area_points(fold_df, config, project_root):
    spec = config["loocv_area_prediction"]
    columns = [spec["group_column"], "fitted_diffusion_sensitive", "fitted_uptake_rate", "fitted_diffusion_nutrients", "fitted_start_point", spec["value_column"]]
    require_columns(fold_df, columns, "fold_results.csv")

    selected_df = select_area_simulation_rows(fold_df, spec)
    vc = load_validation_common(project_root)
    area_dataset = vc.load_no_treatment_area_dataset()
    area_map = {str(entry["label"]): np.asarray(entry["area"], dtype=float) for entry in area_dataset}
    observed_points = []
    predicted_points = []
    stride = max(1, int(spec.get("timepoint_stride", 1)))

    for row in selected_df.itertuples(index=False):
        held_out_area = area_map.get(str(getattr(row, spec["group_column"])))
        if held_out_area is None:
            continue
        fitted_parameters = np.array([float(row.fitted_diffusion_sensitive), float(row.fitted_uptake_rate), float(row.fitted_diffusion_nutrients)], dtype=float)
        start_point = int(row.fitted_start_point)
        sample_indices = np.arange(0, len(held_out_area), stride, dtype=int)
        sample_steps = start_point + sample_indices * 10
        predicted_area = simulate_area_at_steps(vc, fitted_parameters, sample_steps)
        if len(predicted_area) != len(sample_indices):
            continue
        observed_points.append(held_out_area[sample_indices])
        predicted_points.append(predicted_area)

    if not observed_points:
        return np.array([], dtype=float), np.array([], dtype=float), len(selected_df)
    return np.concatenate(observed_points), np.concatenate(predicted_points), len(selected_df)


def build_area_prediction_data(fold_df, config, project_root):
    spec = config["loocv_area_prediction"]
    if str(spec.get("plot_kind", "simulated_scatter")) != "simulated_scatter":
        return {"plot_kind": "metric_distribution"}
    observed, predicted, fold_count = collect_simulated_area_points(fold_df, config, project_root)
    return {"plot_kind": "simulated_scatter", "observed": observed, "predicted": predicted, "fold_count": fold_count}


def draw_simulated_area_scatter(ax, fold_df, config, project_root, area_data=None):
    spec = config["loocv_area_prediction"]
    if area_data is None:
        area_data = build_area_prediction_data(fold_df, config, project_root)
    observed = np.asarray(area_data["observed"], dtype=float)
    predicted = np.asarray(area_data["predicted"], dtype=float)
    fold_count = int(area_data["fold_count"])
    finite_mask = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[finite_mask]
    predicted = predicted[finite_mask]
    observed = area_values_to_mm2(observed, spec)
    predicted = area_values_to_mm2(predicted, spec)

    ax.scatter(observed, predicted, s=5, color=spec["color"], edgecolors="none", alpha=0.16, rasterized=len(observed) > 3000)
    lower, upper = identity_limits(observed, predicted)
    ax.plot([lower, upper], [lower, upper], color="#3f3f3f", linestyle="--", linewidth=1.0)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(spec["xlabel"])
    ax.set_ylabel(spec["ylabel"])

    values = finite_values(fold_df, spec["value_column"])
    secondary_column = spec.get("secondary_value_column")
    secondary_values = finite_values(fold_df, secondary_column) if secondary_column is not None else np.array([], dtype=float)
    secondary_summary = ""
    if secondary_values.size:
        secondary_summary = f"\nmedian {spec['secondary_label']} = {spec['secondary_format'].format(float(np.nanmedian(secondary_values)))}"
    annotation = f"n = {len(observed)} time points\nfolds = {fold_count}\nmedian NRMSE = {spec['value_format'].format(float(np.nanmedian(values)))}{secondary_summary}"
    ax.text(0.04, 0.96, annotation, transform=ax.transAxes, ha="left", va="top", fontsize=6, color="#333333")


def clean_configured_label(label, spec):
    text = str(label)
    prefix = str(spec.get("label_prefix", ""))
    suffix = str(spec.get("label_suffix", ""))
    if prefix and text.startswith(prefix):
        text = text[len(prefix):]
    if suffix and text.endswith(suffix):
        text = text[: -len(suffix)]
    return text.rstrip("_")


def sorted_group_labels(fold_df, group_column, spec):
    labels = list(fold_df[group_column].dropna().unique())
    return sorted(labels, key=lambda label: clean_configured_label(label, spec))


def draw_area_trajectory_prediction(ax, fold_df, config, project_root, area_data=None, show_title=True):
    spec = config["loocv_area_prediction"]
    if show_title:
        ax.set_title(spec["title"])
    if str(spec.get("plot_kind", "simulated_scatter")) == "simulated_scatter":
        draw_simulated_area_scatter(ax, fold_df, config, project_root, area_data=area_data)
        ax.grid(alpha=0.18, linewidth=0.7)
        return

    group_column = spec["group_column"]
    value_column = spec["value_column"]
    secondary_column = spec.get("secondary_value_column")
    columns = [group_column, value_column]
    if secondary_column is not None:
        columns.append(secondary_column)
    require_columns(fold_df, columns, "fold_results.csv")

    group_labels = sorted_group_labels(fold_df, group_column, spec)
    positions = np.arange(len(group_labels))
    rng = np.random.default_rng(11)
    data = [fold_df.loc[fold_df[group_column] == label, value_column].astype(float).dropna().to_numpy() for label in group_labels]

    boxplot = ax.boxplot(data, positions=positions, vert=False, widths=0.52, showfliers=False, patch_artist=True, medianprops={"color": "black", "linewidth": 1.1}, whiskerprops={"color": "#444444", "linewidth": 0.9}, capprops={"color": "#444444", "linewidth": 0.9}, boxprops={"linewidth": 0.9, "edgecolor": "#444444"})
    for patch in boxplot["boxes"]:
        patch.set_facecolor(spec["color"])
        patch.set_alpha(0.28)

    for position, values in zip(positions, data):
        jitter = rng.uniform(-0.16, 0.16, size=len(values))
        ax.scatter(values, np.full(len(values), position, dtype=float) + jitter, s=13, color=spec["color"], edgecolors="none", alpha=0.68)

    values = finite_values(fold_df, value_column)
    median_value = float(np.nanmedian(values)) if values.size else np.nan
    if np.isfinite(median_value):
        ax.axvline(median_value, color="#3f3f3f", linestyle="--", linewidth=1.0)

    ax.set_xlabel(spec.get("metric_xlabel", spec["xlabel"]))
    ax.set_yticks(positions)
    ax.set_yticklabels([clean_configured_label(label, spec) for label in group_labels])
    ax.grid(axis="x", alpha=0.2, linewidth=0.7)

    secondary_values = finite_values(fold_df, secondary_column) if secondary_column is not None else np.array([], dtype=float)
    secondary_summary = ""
    if secondary_values.size:
        secondary_summary = f"\nmedian {spec['secondary_label']} = {spec['secondary_format'].format(float(np.nanmedian(secondary_values)))}"
    annotation = f"n = {len(values)} held-out fold predictions\nmedian = {spec['value_format'].format(median_value)}{secondary_summary}"
    ax.text(0.98, 0.04, annotation, transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color="#333333")


def build_bootstrap_plot_rows(bootstrap_df, parameter_specs):
    rows = []
    require_columns(bootstrap_df, ["parameter", "point_estimate", "bootstrap_mean", "ci_lower", "ci_upper"], "bootstrap_summary.csv")
    for spec in parameter_specs:
        match = bootstrap_df.loc[bootstrap_df["parameter"] == spec["parameter"]]
        if match.empty:
            continue
        row = match.iloc[0]
        point = float(row["point_estimate"])
        mean = float(row["bootstrap_mean"])
        lower = float(row["ci_lower"])
        upper = float(row["ci_upper"])
        scale = abs(point) if np.isfinite(point) and abs(point) > 0 else 1.0
        rows.append({"label": spec["label"], "unit": spec.get("unit", ""), "point": point, "mean": mean, "lower": lower, "upper": upper, "mean_change": 100.0 * (mean - point) / scale, "lower_change": 100.0 * (lower - point) / scale, "upper_change": 100.0 * (upper - point) / scale})
    if not rows:
        raise ValueError("None of the configured bootstrap parameters were found in bootstrap_summary.csv.")
    return rows


def format_number(value):
    value = float(value)
    if not np.isfinite(value):
        return "NA"
    if abs(value) < 1.0e-6:
        return "0"
    return f"{value:.3g}"


def format_ci(row):
    unit = f" {row['unit']}" if row["unit"] else ""
    return f"{format_number(row['lower'])} to {format_number(row['upper'])}{unit}"


def draw_bootstrap_uncertainty(ax, bootstrap_df, config, show_title=True):
    rows = build_bootstrap_plot_rows(bootstrap_df, config["bootstrap_parameters"])
    rows = list(reversed(rows))
    positions = np.arange(len(rows))
    lower_values = np.asarray([row["lower_change"] for row in rows], dtype=float)
    upper_values = np.asarray([row["upper_change"] for row in rows], dtype=float)
    mean_values = np.asarray([row["mean_change"] for row in rows], dtype=float)

    for position, row in zip(positions, rows):
        ax.hlines(position, row["lower_change"], row["upper_change"], color="#4d4d4d", linewidth=1.6)
        ax.scatter(row["mean_change"], position, s=24, color="#6b5ca5", edgecolor="white", linewidth=0.4, zorder=3)
        ax.text(1.02, position, format_ci(row), transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=5.5, color="#333333")

    ax.axvline(0.0, color="#b2182b", linestyle="--", linewidth=1.0)
    ax.set_yticks(positions)
    ax.set_yticklabels([row["label"] for row in rows])
    ax.set_xlabel("Change from point estimate (%)")
    if show_title:
        ax.set_title("Bootstrap: parameter uncertainty")
    ax.grid(axis="x", alpha=0.2, linewidth=0.7)
    ax.text(1.02, 1.02, f"{float(config['bootstrap_ci_level']):.0f}% CI", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.5, color="#333333")
    mean_handle = mlines.Line2D([], [], color="#6b5ca5", marker="o", linestyle="None", markersize=4.2, label="Bootstrap mean")
    ci_handle = mlines.Line2D([], [], color="#4d4d4d", linewidth=1.6, label=f"{float(config['bootstrap_ci_level']):.0f}% CI")
    reference_handle = mlines.Line2D([], [], color="#b2182b", linestyle="--", linewidth=1.0, label="Point estimate")
    ax.legend(handles=[mean_handle, ci_handle, reference_handle], loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3, frameon=False, fontsize=5.5, handlelength=2.0, columnspacing=1.2)

    finite_limits = np.concatenate([lower_values[np.isfinite(lower_values)], upper_values[np.isfinite(upper_values)], mean_values[np.isfinite(mean_values)]])
    lower, upper = float(np.min(finite_limits)), float(np.max(finite_limits))
    span = upper - lower
    pad = max(6.0, span * 0.08)
    ax.set_xlim(lower - pad, upper + pad)
    ax.set_ylim(-0.6, len(rows) - 0.4)


def save_figure(fig, output_dir, output_stem, dpi):
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{output_stem}.pdf"
    png_path = output_dir / f"{output_stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", transparent=True)
    fig.savefig(png_path, bbox_inches="tight", transparent=True, dpi=dpi)
    plt.close(fig)
    return pdf_path, png_path


def make_summary_figure(fold_df, bootstrap_df, config, project_root, area_data=None):
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2), dpi=int(config["dpi"]), gridspec_kw={"width_ratios": [1.0, 1.0, 1.45]})
    draw_loocv_prediction(axes[0], fold_df, config, show_title=False)
    draw_area_trajectory_prediction(axes[1], fold_df, config, project_root, area_data=area_data, show_title=False)
    draw_bootstrap_uncertainty(axes[2], bootstrap_df, config, show_title=False)
    fig.tight_layout()
    return fig


def make_clone_prediction_figure(fold_df, config):
    fig, ax = plt.subplots(figsize=(3.25, 3.2), dpi=int(config["dpi"]))
    draw_loocv_prediction(ax, fold_df, config)
    fig.tight_layout()
    return fig


def make_area_prediction_figure(fold_df, config, project_root, area_data=None):
    fig, ax = plt.subplots(figsize=(3.45, 3.2), dpi=int(config["dpi"]))
    draw_area_trajectory_prediction(ax, fold_df, config, project_root, area_data=area_data)
    fig.tight_layout()
    return fig


def make_bootstrap_figure(bootstrap_df, config):
    fig, ax = plt.subplots(figsize=(5.0, 3.2), dpi=int(config["dpi"]))
    draw_bootstrap_uncertainty(ax, bootstrap_df, config)
    fig.tight_layout()
    return fig


def run_plotting(config):
    project_root = get_project_root()
    configure_plot_style()
    loocv_run_dir = find_latest_loocv_run(project_root, config)
    bootstrap_run_dir = find_latest_bootstrap_run(project_root, config)
    output_dir = resolve_path(project_root, config["output_dir"])
    fold_df, bootstrap_df = load_result_tables(loocv_run_dir, bootstrap_run_dir)
    area_data = build_area_prediction_data(fold_df, config, project_root)
    summary_fig = make_summary_figure(fold_df, bootstrap_df, config, project_root, area_data=area_data)
    clone_fig = make_clone_prediction_figure(fold_df, config)
    area_fig = make_area_prediction_figure(fold_df, config, project_root, area_data=area_data)
    bootstrap_fig = make_bootstrap_figure(bootstrap_df, config)
    summary_paths = save_figure(summary_fig, output_dir, config["output_stem"], int(config["dpi"]))
    clone_paths = save_figure(clone_fig, output_dir, config["loocv_prediction"]["output_stem"], int(config["dpi"]))
    area_paths = save_figure(area_fig, output_dir, config["loocv_area_prediction"]["output_stem"], int(config["dpi"]))
    bootstrap_paths = save_figure(bootstrap_fig, output_dir, config["bootstrap_output_stem"], int(config["dpi"]))
    print(f"Loaded LOOCV run: {loocv_run_dir}")
    print(f"Loaded bootstrap run: {bootstrap_run_dir}")
    print(f"Saved summary figure: {summary_paths[0]}")
    print(f"Saved summary figure: {summary_paths[1]}")
    print(f"Saved clone prediction figure: {clone_paths[0]}")
    print(f"Saved clone prediction figure: {clone_paths[1]}")
    print(f"Saved area trajectory figure: {area_paths[0]}")
    print(f"Saved area trajectory figure: {area_paths[1]}")
    print(f"Saved bootstrap figure: {bootstrap_paths[0]}")
    print(f"Saved bootstrap figure: {bootstrap_paths[1]}")
    return summary_paths, clone_paths, area_paths, bootstrap_paths


def main():
    project_root = get_project_root()
    config = default_config()
    config_path = sys.argv[1] if len(sys.argv) > 1 else config["config_path"]
    config = merge_config(config, load_config_file(project_root, config_path))
    run_plotting(config)


if __name__ == "__main__":
    main()
