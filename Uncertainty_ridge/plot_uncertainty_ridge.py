import os
import pathlib as pl

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import uncertainty_ridge_common as urc


plt.rcParams.update(
    {
        "font.size": 7,
        "pdf.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
    }
)
plt.rcParams["axes.labelsize"] = 7
plt.rcParams["xtick.labelsize"] = 6
plt.rcParams["ytick.labelsize"] = 6


def default_config():
    return {"run_dir": None,
            "output_summary": None,
            "output_line": None,
            "output_ratio": None,
            "show": False}


def auto_discover_run_dir(project_root):
    output_root = pl.Path(project_root) / 'Uncertainty_ridge' / 'results'
    candidates = sorted(output_root.glob("uncertainty_ridge_*"))
    if not candidates:
        raise ValueError(
            "No uncertainty-ridge result folders were found. "
            "Set run_dir explicitly in the config."
        )
    return candidates[-1]


def build_tick_positions(values):
    if len(values) == 1:
        return np.array([0]), [f"{float(values[0]):g}"]

    target_tick_count = 6
    stride = max(1, int(np.ceil(len(values) / target_tick_count)))
    positions = np.arange(0, len(values), stride)
    if positions[-1] != len(values) - 1:
        positions = np.append(positions, len(values) - 1)

    labels = [f"{float(values[pos]):g}" for pos in positions]
    return positions, labels


def plot_surface(ax, surface, tau_on_hours, tau_off_hours, cmap, colorbar_label, title, ridge_line_df=None, vmin=None, vmax=None, ridge_line_label=None, show_ridge_legend=False):
    image = ax.imshow(surface, origin="lower", aspect="auto", interpolation="none", cmap=cmap, vmin=vmin, vmax=vmax)
    x_positions, x_labels = build_tick_positions(tau_off_hours)
    y_positions, y_labels = build_tick_positions(tau_on_hours)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel(r"$\tau_{\mathrm{off}}$ (h)")
    ax.set_ylabel(r"$\tau_{\mathrm{on}}$ (h)")
    ax.set_title(title)

    if ridge_line_df is not None and not ridge_line_df.empty:
        tau_off_grid = np.interp(ridge_line_df["tau_off_hours"].to_numpy(dtype=float), tau_off_hours, np.arange(len(tau_off_hours), dtype=float))
        tau_on_grid = np.interp(ridge_line_df["ridge_tau_on_median_h"].to_numpy(dtype=float), tau_on_hours, np.arange(len(tau_on_hours), dtype=float))
        ax.plot(tau_off_grid, tau_on_grid, color="white", linewidth=1.0, linestyle="--", label=ridge_line_label)
        if show_ridge_legend and ridge_line_label is not None:
            ax.legend(loc="upper right", frameon=False, fontsize=6)

    cbar = plt.colorbar(image, ax=ax, pad=0.04, shrink=0.8)
    cbar.set_label(colorbar_label, rotation=270, labelpad=11)


def main(config=None):
    run_config = default_config()
    if config is not None:
        run_config.update(config)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = urc.find_project_root(script_dir, "requirements.txt") or os.getcwd()

    if run_config["run_dir"] is None:
        run_dir = auto_discover_run_dir(project_root)
    else:
        run_dir = pl.Path(urc.resolve_path(project_root, str(run_config["run_dir"])))

    surfaces = np.load(run_dir / "aggregated_surfaces.npz")
    ridge_line_df = pd.read_csv(run_dir / "ridge_line_summary.csv")

    tau_on_hours = surfaces["tau_on_hours"]
    tau_off_hours = surfaces["tau_off_hours"]
    ttp_mean_surface = surfaces["ttp_mean_surface"]
    ttp_sd_surface = surfaces["ttp_sd_surface"]
    ratio_ttp_sample_median = surfaces["ratio_ttp_sample_median"]
    ratio_endpoint_sample_median = surfaces["ratio_endpoint_sample_median"]
    ratio_ttp_mean_surface = surfaces["ratio_ttp_mean_surface"]
    ratio_endpoint_mean_surface = surfaces["ratio_endpoint_mean_surface"]
    ridge_probability = surfaces["ridge_probability"]
    global_optimum_probability = surfaces["global_optimum_probability"]
    n_samples = int(ratio_ttp_sample_median.shape[0])

    steps_per_hour = float(surfaces["tau_on_steps"][-1] / surfaces["tau_on_hours"][-1])
    ridge_tau_on_hours = surfaces["ridge_tau_on_steps"].astype(float) / steps_per_hour
    ridge_tau_on_q025_h = np.nanpercentile(ridge_tau_on_hours, 2.5, axis=0)
    ridge_tau_on_q975_h = np.nanpercentile(ridge_tau_on_hours, 97.5, axis=0)

    with np.errstate(invalid="ignore"):
        ratio_ttp_sd_surface = np.nanstd(ratio_ttp_sample_median, axis=0, ddof=1 if n_samples > 1 else 0)
        ratio_endpoint_sd_surface = np.nanstd(ratio_endpoint_sample_median, axis=0, ddof=1 if n_samples > 1 else 0)
        ratio_ttp_q25_surface = np.nanpercentile(ratio_ttp_sample_median, 25, axis=0)
        ratio_ttp_q75_surface = np.nanpercentile(ratio_ttp_sample_median, 75, axis=0)
        ratio_endpoint_q25_surface = np.nanpercentile(ratio_endpoint_sample_median, 25, axis=0)
        ratio_endpoint_q75_surface = np.nanpercentile(ratio_endpoint_sample_median, 75, axis=0)
    ratio_ttp_iqr_surface = ratio_ttp_q75_surface - ratio_ttp_q25_surface
    ratio_endpoint_iqr_surface = ratio_endpoint_q75_surface - ratio_endpoint_q25_surface
    resistant_fraction_cmap = mpl.colors.LinearSegmentedColormap.from_list("royalblue_goldenrod", ["royalblue", "goldenrod"], N=256)
    ridge_label = "Median TTP ridge"

    summary_output = (pl.Path(urc.resolve_path(project_root, str(run_config["output_summary"]))) if run_config["output_summary"] is not None else run_dir / "uncertainty_ridge_summary.png")
    line_output = (pl.Path(urc.resolve_path(project_root, str(run_config["output_line"]))) if run_config["output_line"] is not None else run_dir / "uncertainty_ridge_line.png")
    ratio_output = (pl.Path(urc.resolve_path(project_root, str(run_config["output_ratio"]))) if run_config["output_ratio"] is not None else run_dir / "uncertainty_ridge_ratios.png")

    fig, axes = plt.subplots(2, 2, figsize=(6.2, 5.4), dpi=300)
    plot_surface(axes[0, 0], ttp_mean_surface, tau_on_hours=tau_on_hours, tau_off_hours=tau_off_hours, cmap="viridis", colorbar_label="TTP (h)", title="Mean TTP Across Parameter Samples", ridge_line_df=ridge_line_df, ridge_line_label=ridge_label, show_ridge_legend=True)
    plot_surface(axes[0, 1], ttp_sd_surface, tau_on_hours=tau_on_hours, tau_off_hours=tau_off_hours, cmap="magma", colorbar_label="SD of TTP (h)", title="TTP Uncertainty Surface")
    plot_surface(axes[1, 0], ridge_probability, tau_on_hours=tau_on_hours, tau_off_hours=tau_off_hours, cmap="Blues", colorbar_label="Probability", title="Ridge Occupancy Probability")
    plot_surface(axes[1, 1],global_optimum_probability,tau_on_hours=tau_on_hours, tau_off_hours=tau_off_hours, cmap="Oranges", colorbar_label="Probability", title="Global Optimum Probability")
    plt.tight_layout()
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(summary_output, bbox_inches="tight")

    ratio_fig, ratio_axes = plt.subplots(2, 3, figsize=(8.7, 5.0), dpi=300)
    plot_surface(ratio_axes[0, 0], ratio_ttp_mean_surface, tau_on_hours=tau_on_hours, tau_off_hours=tau_off_hours, cmap=resistant_fraction_cmap, colorbar_label="Resistant Fraction", title="Mean Resistant Fraction at TTP", ridge_line_df=ridge_line_df, vmin=0.0, vmax=1.0, ridge_line_label=ridge_label, show_ridge_legend=True)
    plot_surface(ratio_axes[0, 1], ratio_ttp_sd_surface, tau_on_hours=tau_on_hours, tau_off_hours=tau_off_hours, cmap="magma", colorbar_label="SD", title="SD of Resistant Fraction at TTP", ridge_line_df=ridge_line_df, vmin=0.0, ridge_line_label=ridge_label)
    plot_surface(ratio_axes[0, 2], ratio_ttp_iqr_surface, tau_on_hours=tau_on_hours, tau_off_hours=tau_off_hours, cmap="inferno", colorbar_label="IQR", title="IQR of Resistant Fraction at TTP", ridge_line_df=ridge_line_df, vmin=0.0, ridge_line_label=ridge_label)
    plot_surface(ratio_axes[1, 0], ratio_endpoint_mean_surface, tau_on_hours=tau_on_hours, tau_off_hours=tau_off_hours, cmap=resistant_fraction_cmap, colorbar_label="Resistant Fraction", title="Mean Final Resistant Fraction", ridge_line_df=ridge_line_df, vmin=0.0, vmax=1.0, ridge_line_label=ridge_label, show_ridge_legend=True)
    plot_surface(ratio_axes[1, 1], ratio_endpoint_sd_surface, tau_on_hours=tau_on_hours, tau_off_hours=tau_off_hours, cmap="magma", colorbar_label="SD", title="SD of Final Resistant Fraction", ridge_line_df=ridge_line_df, vmin=0.0, ridge_line_label=ridge_label)
    plot_surface(ratio_axes[1, 2], ratio_endpoint_iqr_surface, tau_on_hours=tau_on_hours, tau_off_hours=tau_off_hours, cmap="inferno", colorbar_label="IQR", title="IQR of Final Resistant Fraction", ridge_line_df=ridge_line_df, vmin=0.0, ridge_line_label=ridge_label)
    plt.tight_layout()
    ratio_output.parent.mkdir(parents=True, exist_ok=True)
    ratio_fig.savefig(ratio_output, bbox_inches="tight")

    line_fig, line_ax = plt.subplots(figsize=(3.4, 2.4), dpi=300)
    line_ax.fill_between(tau_off_hours, ridge_tau_on_q025_h, ridge_tau_on_q975_h, color="#9ecae1", alpha=0.35, linewidth=0, label="95%")
    #line_ax.fill_between(ridge_line_df["tau_off_hours"].to_numpy(dtype=float), ridge_line_df["ridge_tau_on_q25_h"].to_numpy(dtype=float), ridge_line_df["ridge_tau_on_q75_h"].to_numpy(dtype=float), color="#3182bd", alpha=0.35, linewidth=0, label="25-75%")
    line_ax.plot(ridge_line_df["tau_off_hours"].to_numpy(dtype=float), ridge_line_df["ridge_tau_on_median_h"].to_numpy(dtype=float), color="black", linewidth=1.2, label="Median ridge")
    line_ax.set_xlabel(r"$\tau_{\mathrm{off}}$ (h)")
    line_ax.set_ylabel(r"$\tau_{\mathrm{on}}^*$ (h)")
    line_ax.set_title("Uncertainty-Aware Ridge Band")
    line_ax.legend(frameon=False, fontsize=6)
    plt.tight_layout()
    line_output.parent.mkdir(parents=True, exist_ok=True)
    line_fig.savefig(line_output, bbox_inches="tight")

    if bool(run_config["show"]):
        plt.show()
    else:
        plt.close(fig)
        plt.close(ratio_fig)
        plt.close(line_fig)
    return run_dir


if __name__ == "__main__":
    main()
