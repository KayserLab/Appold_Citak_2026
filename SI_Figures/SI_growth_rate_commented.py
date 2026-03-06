#!/usr/bin/env python3
"""
Plot growth metrics across multiple temperatures (28, 29, 30, 31, 32, 35 °C),
with Nature-style errorbars and connected lines per strain.

Workflow
- Read the 2nd sheet of each XLSX file.
- Add a Time column (10 min steps).
- Filter the exponential window (OD between MIN_Y and MAX_Y).
- Fit log(OD) = m*t + b  → growth speed = m.
- Compute doubling time and divisions per hour (60 / doubling_time).

Outputs
1) Growth speed vs temperature (median ± IQR across wells)
2) Divisions per hour vs temperature (median ± IQR across wells)

Input location
- XLSX files are referenced relative to Input_files/Platereader_files/.

Output location
- Figures are saved to Output_files/ when a save filename is provided.
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ──────────────────────────────────────────────────────────────────────────────
# PROJECT-RELATIVE IO
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_ROOT = os.path.join(SCRIPT_DIR, "Input_files")
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "Output_files")
PLATEREADER_ROOT = os.path.join(INPUT_ROOT, "Platereader_files")


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
files_by_temp = {
    28: [
        os.path.join("20230704_yNA16_S_28C.xlsx"),
    ],
    29: [
        os.path.join("20230712_yNA16_S_29C.xlsx"),
    ],
    30: [
        os.path.join("20230706_yNA16_S_30C.xlsx"),
        os.path.join("20230808_yNA16_S_30C.xlsx"),
    ],
    31: [
        os.path.join("20230706_yNA16_S_31C.xlsx"),
    ],
    32: [
        os.path.join("20230706_yNA16_S_32C.xlsx"),
    ],
    35: [
        os.path.join("20230705_yNA16_S_35C.xlsx"),
    ],
}

well_groups = {
    "yNA16":  ["A1", "B1", "C1", "A2", "B2", "C2", "A3", "B3", "C3"],
    "yNA16S": ["A6", "B6", "C6", "A7", "B7", "C7", "A8", "B8", "C8"],
}

MIN_Y = 0.02
MAX_Y = 0.6
MIN_POINTS = 10

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "axes.linewidth": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.frameon": False,
    "legend.fontsize": 6,
    "lines.linewidth": 1.0,
    "savefig.dpi": 300,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "yNA16":  "#4169E1",  # royalblue
    "yNA16S": "#DAA520",  # goldenrod
}

LABELS = {
    "yNA16":  "Sensitive",
    "yNA16S": "Resistant",
}


# ──────────────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(filepath, sheet_name=1)
    df["Time"] = np.arange(10, 10 * (len(df) + 1), 10)  # minutes
    return df


def filter_exponential_phase(series: pd.Series, min_y=MIN_Y, max_y=MAX_Y) -> pd.Series:
    filtered = series[(series > min_y) & (series < max_y)]
    return filtered.dropna()


def linear_fit(time_min: np.ndarray, log_data: np.ndarray) -> tuple[float, float]:
    def model(t, m, b):
        return m * t + b

    popt, _ = curve_fit(model, time_min, log_data)
    return float(popt[0]), float(popt[1])


def calculate_doubling_time(growth_speed: float) -> float:
    if growth_speed == 0:
        return np.inf
    return float(np.log(2) / growth_speed)


def compute_metrics_for_file(filepath: str, well_groups: dict) -> dict:
    """
    Returns dict:
      metrics[strain] = {
         'growth_speeds': list[float]  (per well)
         'doubling_times': list[float] (per well)
         'div_per_hr': list[float]     (per well)
      }
    """
    df = load_data(filepath)

    metrics = {}
    for strain, wells in well_groups.items():
        gs_list = []
        dt_list = []
        dph_list = []

        for well in wells:
            if well not in df.columns:
                continue

            filtered = filter_exponential_phase(df[well], MIN_Y, MAX_Y)

            if len(filtered) >= MIN_POINTS:
                time_filtered = df.loc[filtered.index, "Time"].values.astype(float)
                log_data = np.log(filtered.values.astype(float))
                m, _b = linear_fit(time_filtered, log_data)
                gs = m
                dt = calculate_doubling_time(gs)
                dph = 60.0 / dt if np.isfinite(dt) and dt > 0 else 0.0
            else:
                gs = 0.0
                dt = np.inf
                dph = 0.0

            gs_list.append(gs)
            dt_list.append(dt)
            dph_list.append(dph)

        metrics[strain] = {
            "growth_speeds": gs_list,
            "doubling_times": dt_list,
            "div_per_hr": dph_list,
        }

    return metrics


def median_and_iqr(values):
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return np.nan, np.nan, np.nan

    med = np.median(arr)
    q25 = np.percentile(arr, 25)
    q75 = np.percentile(arr, 75)
    return med, q25, q75


# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────────────────────────────────────
def plot_metric_vs_temperature(
    results_by_temp: dict,
    metric_key: str,
    ylabel: str,
    title: str,
    save_filename: str | None = None,
):
    """
    results_by_temp[temp][strain][metric_key] -> list[float]
    """
    temps = sorted(results_by_temp.keys())

    fig, ax = plt.subplots(figsize=(3, 2), dpi=600)

    for strain in ["yNA16", "yNA16S"]:
        medians = []
        low_err = []
        high_err = []

        for t in temps:
            vals = results_by_temp[t][strain][metric_key]
            med, q25, q75 = median_and_iqr(vals)

            medians.append(med)
            low_err.append(med - q25)
            high_err.append(q75 - med)

        ax.errorbar(
            temps,
            medians,
            yerr=[low_err, high_err],
            marker="o",
            linestyle="-",
            linewidth=0.75,
            capsize=2,
            markersize=2,
            color=COLORS[strain],
            label=LABELS[strain],
        )

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4)

    ax.tick_params(axis="both", which="both", length=3, width=0.5, labelsize=6, pad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.legend(frameon=False, fontsize=7, loc="best")
    ax.set_ylim(bottom=0)

    ax.set_xticks(temps)
    ax.set_xticklabels([str(t) for t in temps])

    plt.tight_layout()

    if save_filename:
        os.makedirs(OUTPUT_ROOT, exist_ok=True)
        out = os.path.join(OUTPUT_ROOT, save_filename)
        fig.savefig(out, format="pdf", dpi=600, bbox_inches="tight")
        print(f"[saved] {out}")

    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    results_by_temp = {}

    for temp, relpaths in files_by_temp.items():
        pooled = {
            "yNA16": {"growth_speeds": [], "doubling_times": [], "div_per_hr": []},
            "yNA16S": {"growth_speeds": [], "doubling_times": [], "div_per_hr": []},
        }

        for rel in relpaths:
            path = os.path.join(PLATEREADER_ROOT, rel)
            res = compute_metrics_for_file(path, well_groups)

            for strain in pooled:
                pooled[strain]["growth_speeds"] += res[strain]["growth_speeds"]
                pooled[strain]["doubling_times"] += res[strain]["doubling_times"]
                pooled[strain]["div_per_hr"] += res[strain]["div_per_hr"]

        results_by_temp[temp] = pooled

    plot_metric_vs_temperature(
        results_by_temp,
        metric_key="growth_speeds",
        ylabel="Growth speed\n(slope of log(OD))",
        title="Growth speed vs temperature",
        save_filename="growth_speed_vs_temp.pdf",
    )

    plot_metric_vs_temperature(
        results_by_temp,
        metric_key="div_per_hr",
        ylabel="Divisions per hour",
        title="Division rate vs temperature",
        save_filename="divisions_per_hour_vs_temp.pdf",
    )


if __name__ == "__main__":
    main()