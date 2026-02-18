#!/usr/bin/env python3
"""
Plot treatment highlight spans as horizontal bars per condition.

- Y axis: display names (one row per condition)
- X axis: time in hours (0–150 h), ticks every 25 h
- Input spans are in FRAMES; 1 frame = 30 min = 0.5 h
- Colors are taken from display_colors
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# CONFIG / DICTS (as provided)
# ─────────────────────────────────────────────────────────────

display_names = {
    r"C:\Users\nappold\Desktop\New folder\20240917_continuous_dose_2\For_Manuscript": "CT",
    r"C:\Users\nappold\Desktop\New folder\20251114_9_18\For_manuscript": "9/18",
    r"C:\Users\nappold\Desktop\New folder\20251205_6_18": "6.5/18",
    r"C:\Users\nappold\Desktop\New folder\20251121_4_18": "4/18",
    r"C:\Users\nappold\Desktop\New folder\20251024_no_treatment\For_Manuscript": "NT",
}

highlight_spans = {
    r"C:\Users\nappold\Desktop\New folder\20251024_no_treatment\For_Manuscript": [],
    r"C:\Users\nappold\Desktop\New folder\20251121_4_18": [
        [37, 45], [81, 89], [125, 133], [169, 177],
        [213, 221], [257, 265], [301, 309]
    ],
    r"C:\Users\nappold\Desktop\New folder\20251205_6_18": [
        [37, 50], [86, 99], [135, 148], [184, 197], [233, 246], [282, 295], [331, 344]
    ],
    r"C:\Users\nappold\Desktop\New folder\20251114_9_18\For_manuscript": [
        [37, 55], [91, 109], [145, 163], [199, 217],
        [253, 271], [307, 325]
    ],
    r"C:\Users\nappold\Desktop\New folder\20240917_continuous_dose_2\For_Manuscript": [[37, 334]],
}

display_colors = {
    "NT": "#393b79",  # dark indigo
    "4/18": "#637939",         # dark olive
    "CT": "#7b4173",   # dark purple
    "9/18": "#843c39",
    "6.5/18": "#8c6d31",
}

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.size': 7,
    'axes.titlesize': 7,
    'axes.labelsize': 7,
    'axes.linewidth': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'axes.labelpad': 1,
    'xtick.major.pad': 2,
    'ytick.major.pad': 2,
    'legend.frameon': False,
    'legend.fontsize': 6,
    'lines.linewidth': 1.0,
    'savefig.dpi': 300,
    'figure.dpi': 300,
    'axes.spines.top': True,
    'axes.spines.right': True,
})

# ─────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────

FRAME_TO_HOURS = 0.5  # 30 minutes per frame
X_MIN_H, X_MAX_H = 0, 150
XTICK_STEP_H = 25

# Keep a stable order (same as display_names dict insertion order)
folder_paths = list(display_names.keys())
labels = [display_names[p] for p in folder_paths]

# Map label -> spans (frames)
label_to_spans_frames = {display_names[p]: highlight_spans.get(p, []) for p in folder_paths}

# Y positions (top-to-bottom)
y_positions = list(range(len(labels)))[::-1]
label_to_y = {lab: y for lab, y in zip(labels, y_positions)}

fig_w_in = 3.2
fig_h_in = max(0.8, 0.267 * len(labels) + 0.6)
fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))

bar_height = 0.55

for lab in labels:
    spans = label_to_spans_frames.get(lab, [])
    y = label_to_y[lab]
    color = display_colors.get(lab, "0.3")

    # Draw each span as a horizontal bar
    for f0, f1 in spans:
        x0 = f0 * FRAME_TO_HOURS
        x1 = f1 * FRAME_TO_HOURS
        left = max(X_MIN_H, x0-0.5)
        right = min(X_MAX_H, x1-0.5)
        width = right - left
        if width <= 0:
            continue

        ax.broken_barh(
            [(left, width)],
            (y - bar_height / 2, bar_height),
            facecolors=color,
            edgecolors=color,
            linewidth=0.0,
            alpha=0.9
        )

# Axes formatting
ax.set_xlim(X_MIN_H, X_MAX_H)
ax.set_xticks(list(range(X_MIN_H, X_MAX_H + 1, XTICK_STEP_H)))
ax.set_xlabel("Time (h)")

ax.set_yticks(y_positions)
ax.set_yticklabels(labels)
ax.set_ylabel("Treatment schedule")
y_min = min(y_positions) - 0.6
y_max = max(y_positions) + 0.6
ax.set_ylim(y_min, y_max)

# Nice spacing / grid (optional subtle x grid)
ax.grid(axis="x", which="major", linewidth=0.4, alpha=0.25)
ax.set_axisbelow(True)

plt.tight_layout()

# Save if you want:
out = r"C:\Users\nappold\Desktop\Manuscript Figures\Fig5\treatment_spans.pdf"
fig.savefig(out, bbox_inches="tight", transparent=True)
print("Saved:", out)

plt.show()
