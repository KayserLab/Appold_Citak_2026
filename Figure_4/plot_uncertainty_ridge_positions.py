import os
import pathlib as pl
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def default_config():
    result_paths = [{"path": "Uncertainty_ridge/results/uncertainty_ridge_array_27509082", "label": "baseline", "start_index": 23, "color": "darkcyan"},
                    {"path": "Uncertainty_ridge/results/uncertainty_ridge_array_half_delay_27552714", "label": "0.5x delay", "start_index": 12, "color": "tan"},
                    {"path": "Uncertainty_ridge/results/uncertainty_ridge_array_1_5_delay_27528844", "label": "1.5x delay", "start_index": 33, "color": "olivedrab"}]
    return {"result_paths": result_paths,
            "output_path": "Figure_4/uncertainty_ridge_position_iqr.pdf",
            "figsize": (3.1568, 1.7747), # (7.25/2, 7.2/6),
            "dpi": 300,
            "show": False,
            "show_legend": True,
            "transparent": True,
            "color_map": "tab10",
            "xlim": [0,40],
            "ylim": [0,20]}


def setup_plotting_params():
    rc_params = {"font.size": 7,
                 "pdf.fonttype": 42,
                 "font.family": "sans-serif",
                 "font.sans-serif": ["Arial"],
                 "mathtext.fontset": "custom",
                 "mathtext.rm": "Arial",
                 "mathtext.it": "Arial:italic",
                 "mathtext.bf": "Arial:bold"}
    plt.rcParams.update(rc_params)
    plt.rcParams["axes.labelsize"] = 7
    plt.rcParams["xtick.labelsize"] = 6
    plt.rcParams["ytick.labelsize"] = 6


def find_project_root(current_dir, marker_file):
    current_dir = os.path.abspath(str(current_dir))
    while current_dir != os.path.dirname(current_dir):
        if marker_file in os.listdir(current_dir):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return None


def resolve_path(project_root, path):
    if os.path.isabs(str(path)):
        return str(path)
    return os.path.join(str(project_root), str(path))


def normalize_entry(entry):
    if isinstance(entry, str):
        return {"path": entry, "label": None, "color": None, "linestyle": ":", "start_index": 10}

    normalized = dict(entry)
    normalized.setdefault("label", None)
    normalized.setdefault("color", None)
    normalized.setdefault("linestyle", ":")
    normalized.setdefault("start_index", 10)
    return normalized


def result_path_from_entry(entry):
    for key in ("path", "run_dir", "result_path"):
        if key in entry and entry[key] is not None:
            return entry[key]
    raise ValueError("Each result entry needs a path, run_dir, or result_path value.")


def resolve_result_dir(project_root, entry):
    result_path = pl.Path(resolve_path(project_root, result_path_from_entry(entry)))
    if result_path.is_file():
        return result_path.parent
    return result_path


def steps_per_hour_from_axes(tau_on_steps, tau_on_hours):
    valid = np.isfinite(tau_on_steps) & np.isfinite(tau_on_hours) & (np.abs(tau_on_hours) > 0.0)
    if not np.any(valid):
        raise ValueError("Could not infer steps per hour from tau_on_steps and tau_on_hours.")
    return float(np.nanmedian(tau_on_steps[valid] / tau_on_hours[valid]))


def sort_and_filter_ridge(x_hours, ridge_hours, lower_hours, upper_hours):
    x_hours = np.asarray(x_hours, dtype=float)
    ridge_hours = np.asarray(ridge_hours, dtype=float)
    lower_hours = np.asarray(lower_hours, dtype=float)
    upper_hours = np.asarray(upper_hours, dtype=float)
    finite = np.isfinite(x_hours) & np.isfinite(ridge_hours) & np.isfinite(lower_hours) & np.isfinite(upper_hours)
    order = np.argsort(x_hours[finite])
    return x_hours[finite][order], ridge_hours[finite][order], lower_hours[finite][order], upper_hours[finite][order]


def load_npz_ridge(run_dir):
    surface_path = run_dir / "aggregated_surfaces.npz"
    with np.load(surface_path) as surfaces:
        tau_off_hours = surfaces["tau_off_hours"].astype(float)
        tau_on_steps = surfaces["tau_on_steps"].astype(float)
        tau_on_hours = surfaces["tau_on_hours"].astype(float)
        ridge_tau_on_steps = surfaces["ridge_tau_on_steps"].astype(float)

    steps_per_hour = steps_per_hour_from_axes(tau_on_steps, tau_on_hours)
    ridge_tau_on_hours = ridge_tau_on_steps / steps_per_hour
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ridge_hours = np.nanmedian(ridge_tau_on_hours, axis=0)
        lower_hours = np.nanpercentile(ridge_tau_on_hours, 25, axis=0)
        upper_hours = np.nanpercentile(ridge_tau_on_hours, 75, axis=0)

    return sort_and_filter_ridge(tau_off_hours, ridge_hours, lower_hours, upper_hours)


def load_analytic_csv_ridge(run_dir):
    csv_path = run_dir / "ridge_curve.csv"
    df = pd.read_csv(csv_path)
    required_columns = {"x_hours", "mc_median_hours", "mc_q25_hours", "mc_q75_hours"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing_columns)}")
    return sort_and_filter_ridge(df["x_hours"], df["mc_median_hours"], df["mc_q25_hours"], df["mc_q75_hours"])


def iqr_columns(df):
    candidates = [("ridge_tau_on_q25_h", "ridge_tau_on_q75_h"), ("ridge_tau_on_iqr_low_h", "ridge_tau_on_iqr_high_h")]
    for lower_column, upper_column in candidates:
        if lower_column in df.columns and upper_column in df.columns:
            return lower_column, upper_column
    return None, None


def load_summary_csv_ridge(run_dir):
    csv_path = run_dir / "ridge_line_summary.csv"
    df = pd.read_csv(csv_path)
    lower_column, upper_column = iqr_columns(df)
    required_columns = {"tau_off_hours", "ridge_tau_on_median_h"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing_columns)}")
    if lower_column is None or upper_column is None:
        raise ValueError(f"{csv_path} does not store an IQR. Keep aggregated_surfaces.npz beside it or add q25/q75 columns.")
    return sort_and_filter_ridge(df["tau_off_hours"], df["ridge_tau_on_median_h"], df[lower_column], df[upper_column])


def load_ridge_result(project_root, entry):
    normalized_entry = normalize_entry(entry)
    run_dir = resolve_result_dir(project_root, normalized_entry)
    if not run_dir.exists():
        raise FileNotFoundError(f"Result path does not exist: {run_dir}")

    if (run_dir / "aggregated_surfaces.npz").exists():
        x_hours, ridge_hours, lower_hours, upper_hours = load_npz_ridge(run_dir)
    elif (run_dir / "ridge_curve.csv").exists():
        x_hours, ridge_hours, lower_hours, upper_hours = load_analytic_csv_ridge(run_dir)
    elif (run_dir / "ridge_line_summary.csv").exists():
        x_hours, ridge_hours, lower_hours, upper_hours = load_summary_csv_ridge(run_dir)
    else:
        raise FileNotFoundError(f"No supported ridge result files were found in {run_dir}")

    label = normalized_entry["label"] if normalized_entry["label"] is not None else run_dir.name
    start_index = int(normalized_entry["start_index"])
    if start_index < 0:
        raise ValueError(f"start_index must be 0 or larger for {label}.")
    return {"label": label, "color": normalized_entry["color"], "linestyle": normalized_entry["linestyle"], "start_index": start_index, "x_hours": x_hours, "ridge_hours": ridge_hours, "lower_hours": lower_hours, "upper_hours": upper_hours}


def color_for_result(index, result, run_config):
    if result["color"] is not None:
        return result["color"]
    cmap = mpl.colormaps.get_cmap(str(run_config["color_map"]))
    return cmap(index % cmap.N)


def apply_axis_limits(ax, run_config):
    if run_config["xlim"] is not None:
        ax.set_xlim(run_config["xlim"])
    if run_config["ylim"] is not None:
        ax.set_ylim(run_config["ylim"])


def plot_ridge_results(results, output_path, run_config):
    fig, ax = plt.subplots(figsize=tuple(run_config["figsize"]), dpi=int(run_config["dpi"]))

    for index, result in enumerate(results):
        color = color_for_result(index, result, run_config)
        start_index = result["start_index"]
        ax.fill_between(result["x_hours"][start_index:], result["lower_hours"][start_index:], result["upper_hours"][start_index:], color=color, alpha=0.22, linewidth=0)
        ax.plot(result["x_hours"][start_index:], result["ridge_hours"][start_index:], color=color, linewidth=1.5, linestyle=result["linestyle"], label=result["label"])

    ax.set_xlabel(r"$\tau_{\mathrm{off}}$ (h)", labelpad=0)
    ax.set_ylabel(r"$\tau_{\mathrm{on}}^*$ (h)", labelpad=0)
    apply_axis_limits(ax, run_config)

    if bool(run_config["show_legend"]):
        ax.legend(frameon=False, fontsize=6)

    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, transparent=bool(run_config["transparent"]), bbox_inches="tight")
    return fig


def main(config=None):
    run_config = default_config()
    if config is not None:
        run_config.update(config)

    setup_plotting_params()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = find_project_root(script_dir, "requirements.txt") or os.getcwd()
    if not run_config["result_paths"]:
        raise ValueError("Set result_paths in default_config or pass them to main({'result_paths': [...]})")

    results = [load_ridge_result(project_root, entry) for entry in run_config["result_paths"]]
    output_path = None if run_config["output_path"] is None else pl.Path(resolve_path(project_root, run_config["output_path"]))
    fig = plot_ridge_results(results, output_path, run_config)

    if bool(run_config["show"]):
        plt.show()
    else:
        plt.close(fig)
    return output_path


if __name__ == "__main__":
    main()
