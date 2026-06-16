import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch


def build_sweep_indices(params):
    num_treatment_on_steps = int((params["treatment_on_max"] - params["treatment_on_min"]) / params["treatment_on_step"]) + 1
    treat_on = np.linspace(params["treatment_on_min"], params["treatment_on_max"], num_treatment_on_steps, dtype=np.int16)

    num_treatment_off_steps = int((params["treatment_off_max"] - params["treatment_off_min"]) / params["treatment_off_step"]) + 1
    treat_off = np.linspace(params["treatment_off_min"], params["treatment_off_max"], num_treatment_off_steps, dtype=np.int16)

    num_mutation_rate_steps = int((params["mutation_rate_max"] - params["mutation_rate_min"]) / params["mutation_rate_step"]) + 1
    mutation_rates = np.linspace(params["mutation_rate_min"], params["mutation_rate_max"], num_mutation_rate_steps)  # [0.29251213431358345, 0.5850242686271669, 0.87753640294075035]
    replicas = params["num_replicas"]
    return treat_on, treat_off, mutation_rates, replicas


def configure_matplotlib():
    rc_params = {
        "font.size": 6,
        "pdf.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
    }
    plt.rcParams.update(rc_params)
    plt.rcParams["axes.labelsize"] = 7
    plt.rcParams["xtick.labelsize"] = 6
    plt.rcParams["ytick.labelsize"] = 6


def load_iqr_array(folder, metric, mutation_index):
    q1_total = np.load(f"data/sweep_arrays/{folder}_{metric}_q1_array.npy")[mutation_index]
    q3_total = np.load(f"data/sweep_arrays/{folder}_{metric}_q3_array.npy")[mutation_index]
    return q3_total - q1_total


def build_black_green_white_cmap():
    colors = [(0, (0, 0, 0)), (0.35, (0 / 255, 128 / 255, 0 / 255)), (1, (1, 1, 1))]
    return mpl.colors.LinearSegmentedColormap.from_list("black_green_white", colors, N=256)


def build_blue_yellow_cmap():
    colors = [(65 / 255, 105 / 255, 225 / 255), (218 / 255, 165 / 255, 32 / 255)]
    return mpl.colors.LinearSegmentedColormap.from_list("blue_yellow", colors, N=256)


def format_sweep_axis(ax, treat_on, treat_off, params, show_ylabel):
    steps_x = 40 // params["treatment_off_step"]
    steps_y = 40 // params["treatment_on_step"]
    x_ticks = [i // 20 if i % 20 == 0 else i for i in treat_off]
    y_ticks = [i // 20 if i % 20 == 0 else i for i in treat_on]

    ax.set_xlabel(r"$\tau_{\mathrm{off}}$ (h)")
    ax.set_xticks(np.arange(0, len(treat_off), 2 * steps_x))
    ax.set_xticklabels(x_ticks[:: 2 * steps_x])
    ax.set_yticks(np.arange(0, len(treat_on), 2 * steps_y))
    ax.set_yticklabels(y_ticks[:: 2 * steps_y])

    if show_ylabel:
        ax.set_ylabel(r"$\tau_{\mathrm{on}}$ (h)")
    else:
        ax.tick_params(labelleft=False)


def plot_iqr_panel(ax, iqr_total, max_line, cmap, title, ridge_start_index, show_legend):
    image = ax.imshow(iqr_total, interpolation="none", cmap=cmap, origin="lower", vmin=iqr_total.min(), vmax=iqr_total.max())
    line_label = r"$\tau*$" if show_legend else None
    line_x = np.arange(len(max_line))
    ax.plot(line_x[ridge_start_index:], max_line[ridge_start_index:], color="white", linestyle=":", linewidth=1, zorder=0, label=line_label)
    ax.set_title(title, fontsize=7)
    return image


def add_colorbar(fig, ax, image, label):
    cbar = fig.colorbar(image, ax=ax, pad=0.04, shrink=0.7)
    cbar.set_label(label, color="black", rotation=270, labelpad=10)


def save_iqr_comparison(config):
    configure_matplotlib()

    folder = config["folder"]
    mutation_index = config["mutation_index"]
    ttp_iqr_total = load_iqr_array(folder, "size", mutation_index)
    ratio_iqr_total = load_iqr_array(folder, "ratio", mutation_index)
    params = torch.load(f"data/sweeps/{folder}/params.pth", map_location="cpu", weights_only=False)
    treat_on, treat_off, mutation_rates, _ = build_sweep_indices(params)
    median_ttp_total = np.load(f"data/sweep_arrays/{folder}_size_array.npy")[mutation_index]
    max_line = np.argmax(median_ttp_total, axis=0)

    fig, axs = plt.subplots(1, 2, figsize=config["figsize"], sharex=True, sharey=True)
    ttp_image = plot_iqr_panel(axs[0], ttp_iqr_total, max_line, build_black_green_white_cmap(), "TTP", config["ridge_start_index"], True)
    ratio_image = plot_iqr_panel(axs[1], ratio_iqr_total, max_line, build_blue_yellow_cmap(), "Ratio", config["ridge_start_index"], False)

    format_sweep_axis(axs[0], treat_on, treat_off, params, True)
    format_sweep_axis(axs[1], treat_on, treat_off, params, False)
    add_colorbar(fig, axs[0], ttp_image, r"IQR TTP (h)")
    add_colorbar(fig, axs[1], ratio_image, "IQR ratio")

    leg = axs[0].legend(loc="upper right", frameon=False)
    for text in leg.get_texts():
        text.set_color("white")

    plt.tight_layout()
    os.makedirs(config["output_dir"], exist_ok=True)
    output_path = os.path.join(config["output_dir"], f"{mutation_rates[mutation_index]}_iqr_{folder}_ttp_ratio_SI.pdf")
    plt.savefig(output_path, transparent=True)
    plt.show()
    return output_path


def main():
    config = {
        "folder": "rebuttal_sweep_final",
        "mutation_index": 1,
        "figsize": (7.1, 2.6),
        "ridge_start_index": 23,
        "output_dir": "SI_Figures/plots",
    }
    save_iqr_comparison(config)


if __name__ == "__main__":
    main()
