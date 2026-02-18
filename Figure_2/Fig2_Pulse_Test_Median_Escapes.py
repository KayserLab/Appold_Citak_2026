#!/usr/bin/env python3
"""
Plot median + IQR of escaped clones per colony across experiment times.

- Reads Excel file with positions in first column and experiment times as headers.
- Ignores empty cells.
- Outputs a Nature-style figure (3.6 × 1.8 in) with boxed axes and Arial/Helvetica fonts.

Author: you :)
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "file_path": r"C:\Users\nappold\Desktop\Pulse_Tests.xlsx",  # adjust to your path
    "sheet_name": "Tabelle1",
    "figsize": (1.5, 0.9),
    "color": "black",
    "save_path": r"C:\Users\nappold\Desktop\Pulse_Tests.pdf",  # e.g. r"S:\Members\Nico\Figures\Escapes_Plot.pdf"
}

# ──────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB STYLE
# ──────────────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    'pdf.fonttype': 42,  # embed TrueType fonts
    'ps.fonttype': 42,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial'],

    'font.size': 7,
    'axes.titlesize': 5,
    'axes.labelsize': 5,
    'axes.titlepad': 2,
    'axes.linewidth': 0.5,



    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.pad': 1,
    'ytick.major.pad': 1,
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    'axes.labelpad': 1,

    'legend.frameon': False,
    'legend.fontsize': 6,

    'savefig.dpi': 300,
    'figure.dpi': 300,
})

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
df = pd.read_excel(CONFIG["file_path"], sheet_name=CONFIG["sheet_name"])
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

# ──────────────────────────────────────────────────────────────────────────────
# PLOT
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=CONFIG["figsize"])

x = summary["ExperimentTime_h"].to_numpy(dtype=float)
y_median = summary["Median"].to_numpy(dtype=float)
y_q1 = summary["Q1"].to_numpy(dtype=float)
y_q3 = summary["Q3"].to_numpy(dtype=float)

ax.plot(x, y_median, color=CONFIG["color"], lw=1.2)
ax.fill_between(x, y_q1, y_q3, color=CONFIG["color"], alpha=0.3, label="IQR", linewidth=0)

ax.set_xlabel("Pulse length (h)")
ax.set_ylabel("Escapes")
ax.set_xticks(x)
ax.set_xlim(min(x), max(x))
ax.set_ylim(0, 6.5)
#Have y ticks at every 1 unit
ax.set_yticks(np.arange(0, 7, 1))

# Boxed axes (no grid)
for spine in ax.spines.values():
    spine.set_visible(True)
ax.grid(False)

#ax.legend()

fig.tight_layout()

# Save or show
if CONFIG["save_path"]:
    plt.show()
    fig.savefig(CONFIG["save_path"], bbox_inches="tight")
else:
    plt.show()
