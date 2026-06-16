#!/usr/bin/env python3
"""
Plot median + IQR of escaped clones per colony across experiment times.

Workflow
- Read an Excel sheet where the first column contains positions and each subsequent
  column represents an experiment time (e.g. "12H", "18H", ...).
- Ignore empty cells.
- For each experiment time, compute median and IQR (25th–75th percentile).
- Generate a compact Nature-style plot with boxed axes and embedded fonts.
"""

import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_ROOT = os.path.join(SCRIPT_DIR, "Input_files")
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "Output_files")

CONFIG = {
    # Excel input file inside Input_files/
    "input_relpath": "Pulse_duration_tests/Number_Treatment_Failures_Pulse_Duration.xlsx",
    "sheet_name": "Tabelle1",

    # Plot appearance
    "figsize": (1.5, 0.9),
    "color": "black",

    # Output file (saved into Output_files/)
    "save_path": os.path.join(OUTPUT_ROOT, "Pulse_Tests.pdf"),
}


# ──────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB STYLE
# ──────────────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial"],

    "font.size": 7,
    "axes.titlesize": 5,
    "axes.labelsize": 5,
    "axes.titlepad": 2,
    "axes.linewidth": 0.5,

    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.pad": 1,
    "ytick.major.pad": 1,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "axes.labelpad": 1,

    "legend.frameon": False,
    "legend.fontsize": 6,

    "savefig.dpi": 300,
    "figure.dpi": 300,
})


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main(cfg: dict):
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    file_path = os.path.join(INPUT_ROOT, cfg["input_relpath"])
    df = pd.read_excel(file_path, sheet_name=cfg["sheet_name"])
    df = df.rename(columns={"Unnamed: 0": "Position"})

    # Melt and clean data
    df_melted = df.melt(id_vars="Position", var_name="ExperimentTime", value_name="EscapedClones")
    df_melted = df_melted.dropna(subset=["EscapedClones"])
    df_melted["EscapedClones"] = df_melted["EscapedClones"].astype(float)

    # Compute median + IQR
    summary = (
        df_melted.groupby("ExperimentTime")["EscapedClones"]
        .agg(["median", lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)])
        .reset_index()
    )
    summary.columns = ["ExperimentTime", "Median", "Q1", "Q3"]
    summary["ExperimentTime_h"] = summary["ExperimentTime"].str.replace("H", "").astype(float)
    summary = summary.sort_values("ExperimentTime_h")

    # Plot
    fig, ax = plt.subplots(figsize=cfg["figsize"])

    x = summary["ExperimentTime_h"].to_numpy(dtype=float)
    y_median = summary["Median"].to_numpy(dtype=float)
    y_q1 = summary["Q1"].to_numpy(dtype=float)
    y_q3 = summary["Q3"].to_numpy(dtype=float)

    ax.plot(x, y_median, color=cfg["color"], lw=1.2)
    ax.fill_between(x, y_q1, y_q3, color=cfg["color"], alpha=0.3, label="IQR", linewidth=0)

    ax.set_xlabel("Pulse length (h)")
    ax.set_ylabel("Escapes")
    ax.set_xticks(x)
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(0, 6.5)
    ax.set_yticks(np.arange(0, 7, 1))

    # Boxed axes; no grid
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.grid(False)

    fig.tight_layout()

    if cfg.get("save_path"):
        fig.savefig(cfg["save_path"], bbox_inches="tight")
        print(f"[ok] Saved plot → {cfg['save_path']}")
    else:
        plt.show()


if __name__ == "__main__":
    main(CONFIG)