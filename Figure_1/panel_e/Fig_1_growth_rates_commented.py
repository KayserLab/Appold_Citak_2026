#!/usr/bin/env python3
"""
Estimate growth speed and doubling time from plate reader OD curves (Excel).

Workflow
- Load OD data from the second Excel sheet (sheet index 1).
- Create a synthetic time axis: 10, 20, 30, ... minutes (one per row).
- For each well:
    - keep only OD values within a user-defined "exponential phase" window
    - log-transform OD
    - fit a linear model to log(OD) vs time using curve_fit
    - interpret slope as growth speed; compute doubling time as ln(2)/slope
- Aggregate replicate wells into groups (e.g. susceptible vs resistant).
- Produce summary plots (optionally Nature-style exports).

Notes
- This script expects the Excel files and sheet layout to be consistent with the
  original data acquisition setup.
- No data cleaning beyond the OD window filter is performed.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ───────────────────────────── CONFIG ─────────────────────────────
# Project-relative IO roots
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_ROOT = os.path.join(SCRIPT_DIR, "Input_files")
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "Output_files")

# Excel input files are expected inside Input_files/ (optionally in subfolders).
# Adjust these relative paths to match your repository layout.
FILE_30C_RELPATH = os.path.join("Platereader_files", "20230706_yNA16_S_30C.xlsx")
FILE_30C2_RELPATH = os.path.join("Platereader_files", "20230808_yNA16_S_30C.xlsx")
FILE_35C_RELPATH = os.path.join("Platereader_files", "20230705_yNA16_S_35C.xlsx")

# Output filenames (saved to Output_files/)
DIV_RATE_BOXPLOT_PDF = "division_rate_boxplot.pdf"
GROWTH_SPEED_PDF = "growth_speed_plot.pdf"


# ───────────────────────────── Core I/O ─────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """
    Load OD data from the second sheet (sheet index 1) and add a time column.

    The time axis starts at 10 minutes and increases by 10 minutes per row.
    """
    df = pd.read_excel(filepath, sheet_name=1)
    df["Time"] = np.arange(10, 10 * (len(df) + 1), 10)
    return df


# ───────────────────────────── Analysis helpers ─────────────────────────────
def filter_exponential_phase(series: pd.Series, min_y: float = 0.02, max_y: float = 0.6) -> pd.Series:
    """
    Keep only values in a user-defined OD window to approximate exponential phase.
    """
    filtered = series[(series > min_y) & (series < max_y)]
    return filtered.dropna()


def apply_log_transformation(series: pd.Series) -> np.ndarray:
    """Natural-log transform the OD values."""
    return np.log(series)


def linear_fit(time: np.ndarray, log_data: np.ndarray) -> np.ndarray:
    """
    Fit log(OD) = m*time + b using curve_fit and return [m, b].
    """
    def model(t, m, b):
        return m * t + b

    popt, _ = curve_fit(model, time, log_data)
    return popt


def calculate_doubling_time(growth_speed: float) -> float:
    """
    Compute doubling time from growth speed (slope of log(OD) vs time).
    """
    if growth_speed == 0:
        return np.inf
    return np.log(2) / growth_speed


def plot_linear_fit(time: np.ndarray, log_data: np.ndarray, popt: np.ndarray, title: str) -> None:
    """
    Plot log-transformed data and the fitted line for a single well (debug/sanity plot).
    """
    plt.figure(figsize=(10, 6))
    plt.plot(time, log_data, "o-", label="Log-transformed data")
    plt.plot(time, popt[0] * time + popt[1], "r--", label=f"Fit: m={popt[0]:.4f}, b={popt[1]:.4f}")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Log(OD)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def process_and_calculate(filepath: str, well_groups: dict, plot_fits: bool = False):
    """
    For each group of wells, estimate growth speeds and doubling times per well.

    Returns
    - growth_speeds[group_name] -> list of slopes per well
    - doubling_times[group_name] -> list of doubling times per well
    """
    df = load_data(filepath)

    growth_speeds = {}
    doubling_times = {}

    for group_name, wells in well_groups.items():
        group_speeds = []
        group_doubling_times = []

        for well in wells:
            filtered_data = filter_exponential_phase(df[well])

            # Require a minimum number of usable points before fitting
            if len(filtered_data) >= 10:
                time_filtered = df.loc[filtered_data.index, "Time"].values
                log_data_filtered = apply_log_transformation(filtered_data)

                popt = linear_fit(time_filtered, log_data_filtered)

                growth_speed = popt[0]
                group_speeds.append(growth_speed)

                doubling_time = calculate_doubling_time(growth_speed)
                group_doubling_times.append(doubling_time)

                if plot_fits:
                    plot_linear_fit(
                        time_filtered,
                        log_data_filtered,
                        popt,
                        f"{well} - Linear Fit on Log-Transformed Data",
                    )
            else:
                print(f"Skipping {well}: Not enough data points in the defined range.")
                group_speeds.append(0)
                group_doubling_times.append(np.inf)

        growth_speeds[group_name] = group_speeds
        doubling_times[group_name] = group_doubling_times

    return growth_speeds, doubling_times


# ───────────────────────────── Plotting helpers ─────────────────────────────
def plot_doubling_times_over_files(doubling_times_all: dict, file_labels: list) -> None:
    """
    Plot mean ± std doubling time per condition for each group.
    """
    plt.figure(figsize=(10, 6), dpi=300)

    for set_name in doubling_times_all.keys():
        means = [np.mean(doubling_times_all[set_name][i]) for i in range(len(file_labels))]
        stds = [np.std(doubling_times_all[set_name][i]) for i in range(len(file_labels))]
        plt.errorbar(
            file_labels,
            means,
            yerr=stds,
            label=f"{set_name} Doubling Time",
            marker="o",
            capsize=5,
            linestyle="None",
        )

    plt.xlabel("Condition")
    plt.ylabel("Doubling Time (minutes)")
    plt.title("Doubling Time Across Conditions")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_growth_speeds_over_files(growth_speeds_all: dict, file_labels: list) -> None:
    """
    Plot mean ± std growth speed per condition for each group.
    """
    plt.figure(figsize=(10, 6), dpi=300)

    for set_name in growth_speeds_all.keys():
        means = [np.mean(growth_speeds_all[set_name][i]) for i in range(len(file_labels))]
        stds = [np.std(growth_speeds_all[set_name][i]) for i in range(len(file_labels))]
        plt.errorbar(
            file_labels,
            means,
            yerr=stds,
            label=f"{set_name} Growth Speed",
            marker="o",
            capsize=5,
            linestyle="None",
        )

    plt.ylim(bottom=0)
    plt.xlabel("Condition")
    plt.ylabel("Growth Speed (slope of log-transformed data)")
    plt.title("Growth Speed Across Conditions")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_combined_doubling_and_growth(
    doubling_times_all: dict,
    growth_speeds_all: dict,
    file_labels_combined: list,
) -> None:
    """
    Combine the two 30°C replicates into one condition and compare against 35°C.
    """
    colors = {
        "yNA16": "royalblue",
        "yNA16S": "gold",
    }

    fig, axs = plt.subplots(2, 1, figsize=(10, 12), dpi=300, sharex=True)

    for set_name in ["yNA16", "yNA16S"]:
        doubling_avg_30C = np.mean(doubling_times_all[set_name][:2], axis=0)
        doubling_combined = [doubling_avg_30C, doubling_times_all[set_name][2]]

        doubling_means = [np.mean(vals) for vals in doubling_combined]
        doubling_stds = [np.std(vals) for vals in doubling_combined]

        axs[0].errorbar(
            file_labels_combined,
            doubling_means,
            yerr=doubling_stds,
            label=set_name,
            marker="o",
            capsize=5,
            color=colors[set_name],
            linestyle="-",
            linewidth=2,
        )

        growth_avg_30C = np.mean(growth_speeds_all[set_name][:2], axis=0)
        growth_combined = [growth_avg_30C, growth_speeds_all[set_name][2]]

        growth_means = [np.mean(vals) for vals in growth_combined]
        growth_stds = [np.std(vals) for vals in growth_combined]

        axs[1].errorbar(
            file_labels_combined,
            growth_means,
            yerr=growth_stds,
            label=set_name,
            marker="o",
            capsize=5,
            color=colors[set_name],
            linestyle="-",
            linewidth=2,
        )

    axs[0].set_title("Doubling Time Across Conditions")
    axs[0].set_ylabel("Doubling Time (min)")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].set_title("Growth Speed Across Conditions")
    axs[1].set_ylabel("Growth Speed (slope of log-transformed OD)")
    axs[1].set_xlabel("Condition")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def plot_nature_style_growth_speed(growth_speeds_all: dict, file_labels_combined: list, save_path: str | None = None) -> None:
    """
    Nature-style plot of growth speed (30°C averaged over two replicates vs 35°C).
    """
    import matplotlib as mpl  # local import kept to preserve original behavior
    import matplotlib.pyplot as plt  # local import kept to preserve original behavior

    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial"],
        "font.size": 7,
        "axes.linewidth": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })

    colors = {
        "yNA16": "#4169E1",   # royalblue
        "yNA16S": "#DAA520",  # goldenrod
    }

    fig, ax = plt.subplots(figsize=(2, 1.25), dpi=600)

    for set_name in ["yNA16", "yNA16S"]:
        growth_avg_30C = np.mean(growth_speeds_all[set_name][:2], axis=0)
        growth_35C = growth_speeds_all[set_name][2]
        growth_combined = [growth_avg_30C, growth_35C]

        means = [np.mean(g) for g in growth_combined]
        stds = [np.std(g) for g in growth_combined]

        ax.errorbar(
            file_labels_combined,
            means,
            yerr=stds,
            label=set_name,
            marker="o",
            color=colors[set_name],
            linestyle="-",
            linewidth=0.75,
            capsize=2,
            markersize=4,
        )

    ax.set_ylabel("Growth speed\n(slope of log(OD))")
    ax.set_xlabel("Temperature\n(°C)")
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="both", which="both", length=3, width=0.5, labelsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, format="pdf", dpi=600, bbox_inches="tight")

    print("Using font:", plt.rcParams["font.sans-serif"])
    plt.show()


def plot_nature_style_divisions_per_hour_boxplot(
    doubling_times_all: dict,
    file_labels_combined: list,
    save_path: str | None = None,
) -> None:
    """
    Nature-style boxplot of divisions per hour (computed as 60 / doubling_time).
    """
    import matplotlib as mpl  # local import kept to preserve original behavior
    import matplotlib.pyplot as plt  # local import kept to preserve original behavior
    import numpy as np  # local import kept to preserve original behavior

    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial"],
        "font.size": 7,
        "axes.linewidth": 0.5,
        "xtick.major.size": 0,
        "ytick.major.size": 3,
        "xtick.minor.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })

    key_map = {
        "Susceptible": "yNA16",
        "Resistant": "yNA16S",
    }

    colors = {
        "Susceptible": "#4169E1",
        "Resistant": "#DAA520",
    }

    data = []
    positions = []
    box_colors = []
    offset = 0.2

    for i, temp_label in enumerate(file_labels_combined):
        center = i + 1
        for strain in ["Susceptible", "Resistant"]:
            orig_key = key_map[strain]

            if temp_label.startswith("30"):
                dt = np.hstack((
                    doubling_times_all[orig_key][0],
                    doubling_times_all[orig_key][1],
                ))
            else:
                dt = np.array(doubling_times_all[orig_key][2])

            div_per_hr = 60.0 / dt
            data.append(div_per_hr)
            positions.append(center + (-offset if strain == "Susceptible" else offset))
            box_colors.append(colors[strain])

    fig, ax = plt.subplots(figsize=(2.5, 1.5), dpi=600)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.35,
        patch_artist=True,
        showfliers=False,
    )

    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.4)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(0.6)
    for whisker in bp["whiskers"]:
        whisker.set_color("black")
        whisker.set_linewidth(0.5)
    for cap in bp["caps"]:
        cap.set_color("black")
        cap.set_linewidth(0.5)

    ax.set_xlim(0.5, len(file_labels_combined) + 0.5)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(file_labels_combined, fontsize=7)
    ax.xaxis.set_tick_params(which="major", pad=12)

    ax.set_xticks(positions, minor=True)
    ax.set_xticklabels(["Sensitive", "Resistant"] * len(file_labels_combined), minor=True, fontsize=6)
    ax.xaxis.set_tick_params(which="minor", pad=3)

    ax.set_ylabel("Divisions per hour", fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", width=0.5, length=3, labelsize=6, pad=2)
    ax.yaxis.labelpad = 3

    x_left_lim, x_right_lim = ax.get_xlim()
    ax.axvspan(
        (x_left_lim + x_right_lim) / 2,
        x_right_lim,
        facecolor="#bfbfbf",
        alpha=1,
        zorder=-1,
        linewidth=0,
    )
    ax.axhline(
        y=0,
        color="0.5",
        linestyle=(0, (3, 3)),
        linewidth=0.6,
        alpha=0.6,
        zorder=0.5,
    )
    ax.grid(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, format="pdf", dpi=600, bbox_inches="tight")
    plt.show()


# ───────────────────────────── Run ─────────────────────────────
def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    file_30C = os.path.join(INPUT_ROOT, FILE_30C_RELPATH)
    file_35C = os.path.join(INPUT_ROOT, FILE_35C_RELPATH)
    file_30C2 = os.path.join(INPUT_ROOT, FILE_30C2_RELPATH)

    files = [file_30C, file_30C2, file_35C]
    file_labels = ["30°C", "30°C2", "35°C"]

    well_groups = {
        "yNA16": ["A1", "B1", "C1", "A2", "B2", "C2", "A3", "B3", "C3"],
        "yNA16S": ["A6", "B6", "C6", "A7", "B7", "C7", "A8", "B8", "C8"],
    }

    doubling_times_all = {"yNA16": [], "yNA16S": []}
    growth_speeds_all = {"yNA16": [], "yNA16S": []}

    for filepath in files:
        speeds, doubling_times = process_and_calculate(filepath, well_groups, plot_fits=False)
        for set_name in well_groups.keys():
            doubling_times_all[set_name].append(doubling_times[set_name])
            growth_speeds_all[set_name].append(speeds[set_name])

    file_labels_combined = ["30°C", "35°C"]

    # Output goes directly to Output_files/
    plot_nature_style_divisions_per_hour_boxplot(
        doubling_times_all,
        file_labels_combined,
        save_path=os.path.join(OUTPUT_ROOT, DIV_RATE_BOXPLOT_PDF),
    )


if __name__ == "__main__":
    main()