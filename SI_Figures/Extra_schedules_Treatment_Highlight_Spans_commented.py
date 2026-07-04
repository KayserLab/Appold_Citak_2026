#!/usr/bin/env python3
"""
Plot treatment highlight spans as horizontal bars per condition.

- Y axis: display names (one row per condition)
- X axis: time in hours (0-150 h), ticks every 25 h
- Input spans are in FRAMES; 1 frame = 30 min = 0.5 h
- Colors are taken from display_colors

This script does not read image or CSV data. The original absolute folder paths
were only used as dictionary keys; here, stable condition labels are used instead.
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt


# ────────────────────────── Project-relative output ──────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "Output_files")


# ────────────────────────── Plot style ──────────────────────────

def set_nature_style():
    """Update Matplotlib rcParams for manuscript-style plotting."""
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
        "axes.labelpad": 1,
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,
        "legend.frameon": False,
        "legend.fontsize": 6,
        "lines.linewidth": 1.0,
        "savefig.dpi": 300,
        "figure.dpi": 300,
        "axes.spines.top": True,
        "axes.spines.right": True,
    })


# ────────────────────────── Plotting ──────────────────────────

def plot_treatment_spans(config):
    """
    Plot treatment highlight spans as horizontal bars.

    The order is controlled by config["condition_order"].
    Spans are read from config["highlight_spans_frames"].
    """
    condition_order = list(config["condition_order"])
    highlight_spans = config["highlight_spans_frames"]
    display_colors = config["display_colors"]

    frame_to_hours = float(config.get("frame_to_hours", 0.5))
    x_min_h = int(config.get("x_min_h", 0))
    x_max_h = int(config.get("x_max_h", 150))
    xtick_step_h = int(config.get("xtick_step_h", 25))

    # Y positions are assigned top-to-bottom according to condition_order.
    y_positions = list(range(len(condition_order)))[::-1]
    label_to_y = {lab: y for lab, y in zip(condition_order, y_positions)}

    fig_w_in = float(config.get("fig_w_in", 3.2))
    fig_h_in = max(0.8, 0.267 * len(condition_order) + 0.6)
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))

    bar_height = float(config.get("bar_height", 0.55))

    for lab in condition_order:
        spans = highlight_spans.get(lab, [])
        y = label_to_y[lab]
        color = display_colors.get(lab, "0.3")

        # Draw each treatment interval as one horizontal bar.
        for f0, f1 in spans:
            x0 = f0 * frame_to_hours
            x1 = f1 * frame_to_hours
            left = max(x_min_h, x0 - 0.5)
            right = min(x_max_h, x1 - 0.5)
            width = right - left

            if width <= 0:
                continue

            ax.broken_barh(
                [(left, width)],
                (y - bar_height / 2, bar_height),
                facecolors=color,
                edgecolors=color,
                linewidth=0.0,
                alpha=0.9,
            )

    # Format axes.
    ax.set_xlim(x_min_h, x_max_h)
    ax.set_xticks(list(range(x_min_h, x_max_h + 1, xtick_step_h)))
    ax.set_xlabel("Time (h)")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(condition_order)
    ax.set_ylabel("Treatment schedule")

    y_min = min(y_positions) - 0.6
    y_max = max(y_positions) + 0.6
    ax.set_ylim(y_min, y_max)

    ax.grid(axis="x", which="major", linewidth=0.4, alpha=0.25)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if config.get("save_as_pdf", True):
        save_dir = config.get("save_path", OUTPUT_ROOT)
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, config.get("outfile_name", "treatment_spans.pdf"))
        fig.savefig(out, bbox_inches="tight", transparent=True)
        print("Saved:", out)

    if config.get("show_plot", True):
        plt.show()
    else:
        plt.close(fig)


# ────────────────────────── Main configuration ──────────────────────────

if __name__ == "__main__":
    CONFIG = {
        # Select and order the treatment schedules shown in the plot.
        # Comment/uncomment labels here to change which rows are displayed.
        "condition_order": [
            "CT",
            "9/20.5",
            "6.5/15.5",
            # "9/18",
            "6.5/18",
            "6.5/20.5",
            # "4/18",
            "NT",
        ],

        # Treatment highlight spans in frames for each schedule.
        # These values define the treatment windows; 1 frame = 0.5 h.
        "highlight_spans_frames": {
            "NT": [],
            "4/18": [
                [37, 45], [81, 89], [125, 133], [169, 177],
                [213, 221], [257, 265], [301, 309],
            ],
            "6.5/18": [
                [37, 50], [86, 99], [135, 148], [184, 197],
                [233, 246], [282, 295], [331, 344],
            ],
            "9/18": [
                [37, 55], [91, 109], [145, 163], [199, 217],
                [253, 271], [307, 325],
            ],
            "CT": [
                [37, 334],
            ],
            "6.5/20.5": [
                [37, 50], [91, 104], [145, 158], [199, 212],
                [253, 266], [307, 320], [349, 362],
            ],
            "6.5/15.5": [
                [37, 50], [81, 94], [125, 138], [169, 182],
                [213, 226], [257, 270], [301, 314], [348, 361],
            ],
            "9/20.5": [
                [37, 55], [96, 114], [155, 173],
                [214, 232], [273, 291], [332, 350],
            ],
        },

        # Colors assigned to each treatment schedule.
        "display_colors": {
            "NT": "#393b79",
            "4/18": "#637939",
            "CT": "#7b4173",
            "9/18": "#843c39",
            "6.5/18": "#8c6d31",
            "6.5/15.5": "teal",
            "6.5/20.5": "palegreen",
            "9/20.5": "plum",
        },

        # Time conversion and axis limits.
        "frame_to_hours": 0.5,
        "x_min_h": 0,
        "x_max_h": 150,
        "xtick_step_h": 25,

        # Figure layout.
        "fig_w_in": 3.2,
        "bar_height": 0.55,

        # Output settings.
        "save_as_pdf": True,
        "save_path": OUTPUT_ROOT,
        "outfile_name": "treatment_spans.pdf",
        "show_plot": True,
    }

    set_nature_style()
    plot_treatment_spans(CONFIG)
