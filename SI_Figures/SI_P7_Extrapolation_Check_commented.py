#!/usr/bin/env python3
"""
Plot-only P7 extrapolation check for manuscript use.

This script is restricted to one identifier, P7, and does not save extrapolated
CSV files by default.

For the selected timelapse colony CSV:

  1. Detect the first frame where the colony contour touches the image border.
  2. Define the last trusted frame as fit_end = touch_frame - 1.
     If no touch is detected, use the last frame as fit_end.
  3. Find the matching final CSV for P7.
  4. Convert timelapse and final areas to mm² using their respective pixel sizes.
  5. Linearly interpolate/extrapolate from the colony/clone area at fit_end to
     the final area at frame = last_timelapse_frame + 1.
  6. Plot measured timelapse data across the full timelapse as solid lines.
  7. Plot the extrapolated/interpolated region as dashed lines.
  8. Show a colony-only ±3.118% band around the extrapolated colony curve only.
     Show an analogous clone-only ±41.4% band around the extrapolated clone curve only.
     The measured timelapse values and the final target points are not given bands
     or error bars.
  9. Use hours on the x-axis, with 2 frames = 1 hour.
 10. Print the time positions where the colony extrapolation and its lower/upper
     uncertainty bounds cross 71 mm², rounded up to the next full hour.
 11. Save the same 71 mm² crossing output as a TXT file next to the PDF plot.
 12. Mark the 71 mm² crossing positions in the colony panel.
"""

import os
import re
import ast
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ───────────────────────────── Project-relative paths ─────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_ROOT = os.path.join(SCRIPT_DIR, "Input_files")
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "Output_files")

# ───────────────────────────── CONFIG ─────────────────────────────

# Set the input folders containing the timelapse and final CSV files.
DEFAULT_TIMELAPSE_FOLDER = os.path.join(INPUT_ROOT, "9h_18h")
DEFAULT_FINALS_FOLDER = os.path.join(INPUT_ROOT, "9h_18h_endpoint")

# Set the output folder for the PDF plot and 71 mm² crossing-time TXT file.
DEFAULT_OUTPUT_FOLDER = OUTPUT_ROOT

# Select the identifier analysed in this plot-only validation.
TARGET_PID = 7

# Set the frame-to-hour conversion used throughout the plot and output.
FRAMES_PER_HOUR = 2.0

# Set image dimensions and border-touch detection settings.
IMG_WIDTH = 1376
IMG_HEIGHT = 1104
PAD = 1
TOUCH_DETECTION_START_FRAME = 20   # start checking only after this frame

# Set pixel sizes for timelapse segmentation and final-image measurements.
SEG_SCALE_UM_PER_PX   = 8.648   # timelapse
FINAL_SCALE_UM_PER_PX = 14.424  # finals

# Colony-only uncertainty/tolerance around the extrapolated colony curve.
# Interpreted as ±3.118%, not as mathematical variance (sigma²).
# Important: this band is NOT applied to measured values or the final point.
COLONY_AREA_VARIANCE_PERCENT = 0.78
COLONY_AREA_VARIANCE_FRACTION = COLONY_AREA_VARIANCE_PERCENT / 100.0

# Clone-only uncertainty/tolerance around the extrapolated clone curve.
# Interpreted as ±41.4%, not as mathematical variance (sigma²).
# Important: this band is NOT applied to measured values or the final point.
CLONE_AREA_VARIANCE_PERCENT = 41.4
CLONE_AREA_VARIANCE_FRACTION = CLONE_AREA_VARIANCE_PERCENT / 100.0

# Set the target colony area used for crossing-time reporting.
TARGET_COLONY_AREA_MM2 = 71.0
SHOW_TARGET_CROSSING_ON_PLOT = True
SAVE_CROSSING_OUTPUT_TXT = True
# Set the manuscript colors for colony and clone curves.
COLONY_COLOR = "dimgray"
CLONE_COLOR = "goldenrod"

# Set whether plots are saved and/or shown interactively.
SAVE_PLOTS = True
SHOW_PLOTS = True

# Do not save extrapolated CSV files in this variant.
SAVE_EXTRAPOLATED_CSV = False


# ───────────────────────────── HELPERS ─────────────────────────────

def px2_to_mm2(px2, pixel_size_um):
    """Convert area from px² to mm² given pixel size in µm/px."""
    return px2 * (pixel_size_um * 1e-3) ** 2


def mm2_to_px2(mm2, pixel_size_um):
    """Convert area from mm² to px² given pixel size in µm/px."""
    return mm2 / (pixel_size_um * 1e-3) ** 2


def extract_pid(path_or_name):
    """Extract P-id as int from a filename/path, e.g. P7 -> 7."""
    m = re.search(r"P(\d{1,2})(?!\d)", os.path.basename(path_or_name))
    return int(m.group(1)) if m else None


def frame_to_hour(frame_value):
    """Convert frame index/value to hours."""
    return float(frame_value) / FRAMES_PER_HOUR


def frames_to_hours(frame_values):
    """Convert an array-like of frame values to hours."""
    return np.asarray(frame_values, dtype=float) / FRAMES_PER_HOUR


def crossing_to_next_full_hour(x_frame):
    """Convert a crossing frame to hours and round up to the next full hour."""
    if x_frame is None or not np.isfinite(x_frame):
        return None
    hour = frame_to_hour(x_frame)
    # Small epsilon avoids rounding 35.0000000001 up to 36 because of float noise.
    return int(np.ceil(hour - 1e-9))


def format_hour_value(x_frame):
    """Format a crossing frame-value as rounded-up hours for printed output."""
    rounded_hour = crossing_to_next_full_hour(x_frame)
    if rounded_hour is None:
        return "not reached"
    return f"{rounded_hour:d} h"


def format_exact_hour_value(x_frame):
    """Format a crossing frame-value as exact, unrounded hours for the TXT output."""
    if x_frame is None or not np.isfinite(x_frame):
        return "not reached"
    return f"{frame_to_hour(x_frame):.3f} h"


def interpolate_x_at_y(x_values, y_values, target_y):
    """
    Return the first x-position where a polyline crosses target_y.

    Linear interpolation is used between neighbouring points. Works for both
    increasing and decreasing y-values. Returns None if the target is not crossed.
    """
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) == 0:
        return None

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    for i in range(len(x) - 1):
        x0, x1 = x[i], x[i + 1]
        y0, y1 = y[i], y[i + 1]

        if y0 == target_y:
            return float(x0)

        # True if target_y lies between y0 and y1, inclusive.
        if (target_y - y0) * (target_y - y1) <= 0:
            if y1 == y0:
                return float(x0)
            t = (target_y - y0) / (y1 - y0)
            return float(x0 + t * (x1 - x0))

    if y[-1] == target_y:
        return float(x[-1])

    return None


def first_touch_frame(
    colony_df,
    width=IMG_WIDTH,
    height=IMG_HEIGHT,
    pad=PAD,
    tol=1e-6,
    colony_label=None,
    start_frame=TOUCH_DETECTION_START_FRAME,
):
    """
    Return the first frame where the colony contour touches the image border,
    considering only frames >= start_frame.
    """
    # pad is kept in the signature for compatibility with older script versions.
    _ = pad

    border_x_min, border_y_min = 0.5, 0.5
    border_x_max, border_y_max = width + 0.5, height + 0.5

    frames = colony_df["frame"].to_numpy() if "frame" in colony_df.columns else np.arange(len(colony_df))

    for pos in range(len(colony_df)):
        frame_val = int(frames[pos])
        if frame_val < start_frame:
            continue

        try:
            contour = colony_df.iloc[pos]["colony_contour"]
            if isinstance(contour, str):
                contour = ast.literal_eval(contour)
            arr = np.asarray(contour, dtype=float)
            if arr.size == 0:
                continue

            y, x = arr[:, 0], arr[:, 1]
            touches = (
                (y <= border_y_min + tol).any()
                or (y >= border_y_max - tol).any()
                or (x <= border_x_min + tol).any()
                or (x >= border_x_max - tol).any()
            )
            if touches:
                print(f"[touch] {colony_label or '<unknown>'} touches border at frame {frame_val} ({frame_to_hour(frame_val):.1f} h)")
                return frame_val
        except Exception:
            continue

    return None


def detect_colony_files(folder, target_pid=TARGET_PID):
    """Return only the timelapse colony CSV for the requested P-id."""
    files = []
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".csv"):
            continue
        if "colony" not in fname.lower() or "clonearea" not in fname.lower():
            continue
        if extract_pid(fname) != target_pid:
            continue
        files.append(os.path.join(folder, fname))

    files = sorted(files)
    print(f"[info] Found {len(files)} timelapse colony file(s) for P{target_pid}.")
    return files


def build_finals_map(finals_folder):
    """
    Build a mapping from P-id (int) to final CSV path.

    It expects that final CSV filenames contain 'P<id>', e.g. '20251114_P8_0_332-10.csv'.
    For each id, the last file found wins if there are multiple.
    """
    mapping = {}
    if not os.path.isdir(finals_folder):
        print(f"[warn] Finals folder does not exist: {finals_folder}")
        return mapping

    for fname in os.listdir(finals_folder):
        if not fname.lower().endswith(".csv"):
            continue
        pid = extract_pid(fname)
        if pid is None:
            continue
        mapping[pid] = os.path.join(finals_folder, fname)

    print(f"[info] Found {len(mapping)} final CSV files with P<id> in name.")
    return mapping


def setup_matplotlib():
    """Manuscript-style rcParams."""
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
        "legend.frameon": True,
        "legend.fontsize": 6,
        "lines.linewidth": 1.0,
        "savefig.dpi": 300,
        "figure.dpi": 300,
        "axes.spines.top": True,
        "axes.spines.right": True,
    })


def build_interp_series(frames, values, fit_end, final_frame_index=None, final_value=None):
    """
    Build a series for plotting/crossing from fit_end onward, optionally appending
    the separate final target point.
    """
    frames = np.asarray(frames, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = frames >= fit_end

    x = frames[mask].copy()
    y = values[mask].copy()

    if final_frame_index is not None and final_value is not None and np.isfinite(final_value):
        x = np.append(x, float(final_frame_index))
        y = np.append(y, float(final_value))

    return x, y


def compute_crossings_for_target(
    frames,
    extrap_colony_mm2,
    extrap_colony_lower_mm2,
    extrap_colony_upper_mm2,
    fit_end,
    final_frame_index,
    final_colony_mm2,
    target_mm2=TARGET_COLONY_AREA_MM2,
):
    """
    Compute x/frame positions where the central extrapolation and its lower/upper
    uncertainty bounds cross target_mm2.

    Central curve includes the final target point. The lower/upper uncertainty
    bounds are only the extrapolated-band values; the final measured point is not
    given uncertainty and is therefore not appended to lower/upper.
    """
    x_central, y_central = build_interp_series(
        frames, extrap_colony_mm2, fit_end, final_frame_index, final_colony_mm2
    )
    x_lower, y_lower = build_interp_series(frames, extrap_colony_lower_mm2, fit_end)
    x_upper, y_upper = build_interp_series(frames, extrap_colony_upper_mm2, fit_end)

    return {
        "central": interpolate_x_at_y(x_central, y_central, target_mm2),
        "lower": interpolate_x_at_y(x_lower, y_lower, target_mm2),
        "upper": interpolate_x_at_y(x_upper, y_upper, target_mm2),
    }


def build_crossing_output_lines(
    crossings,
    target_mm2=TARGET_COLONY_AREA_MM2,
    colony_file=None,
    include_exact_values=True,
):
    """Build printable/TXT lines for target crossings in hours."""
    lines = []
    lines.append(f"[{target_mm2:g} mm² crossing times]")
    if colony_file is not None:
        lines.append(f"file: {os.path.basename(colony_file)}")
    lines.append(f"time conversion: {FRAMES_PER_HOUR:g} frames = 1 h")
    lines.append("rounded values: rounded up to the next full hour")
    lines.append("")
    lines.append(f"central extrapolation: {format_hour_value(crossings.get('central'))}")
    lines.append(f"lower band (-{COLONY_AREA_VARIANCE_PERCENT:g}%): {format_hour_value(crossings.get('lower'))}")
    lines.append(f"upper band (+{COLONY_AREA_VARIANCE_PERCENT:g}%): {format_hour_value(crossings.get('upper'))}")

    if include_exact_values:
        lines.append("")
        lines.append("exact unrounded values:")
        lines.append(f"central extrapolation: {format_exact_hour_value(crossings.get('central'))}")
        lines.append(f"lower band (-{COLONY_AREA_VARIANCE_PERCENT:g}%): {format_exact_hour_value(crossings.get('lower'))}")
        lines.append(f"upper band (+{COLONY_AREA_VARIANCE_PERCENT:g}%): {format_exact_hour_value(crossings.get('upper'))}")

    lines.append("")
    lines.append("Note: lower/upper values refer only to the extrapolated colony band; the final point has no error bar.")
    return lines


def print_and_save_crossings(
    crossings,
    target_mm2=TARGET_COLONY_AREA_MM2,
    colony_file=None,
    output_txt_path=None,
):
    """Print rounded-up crossing times and optionally save them as a TXT file."""
    lines = build_crossing_output_lines(
        crossings=crossings,
        target_mm2=target_mm2,
        colony_file=colony_file,
        include_exact_values=True,
    )
    print("\n" + "\n".join(lines))

    if output_txt_path is not None:
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[txt saved] {output_txt_path}")


def mark_target_crossings(ax, crossings, target_mm2=TARGET_COLONY_AREA_MM2):
    """Mark the target-area crossing positions in the colony panel."""
    finite_x = [frame_to_hour(x) for x in crossings.values() if x is not None and np.isfinite(x)]
    if not finite_x:
        return

    ymin, ymax = ax.get_ylim()
    target = target_mm2

    # Ensure the target line is visible even if it lies outside the current y-limits.
    if target < ymin or target > ymax:
        all_y = [ymin, ymax, target]
        pad = 0.04 * (max(all_y) - min(all_y)) if max(all_y) > min(all_y) else 1.0
        ax.set_ylim(0, max(all_y) + pad)
        ymin, ymax = ax.get_ylim()

    ax.axhline(
        target,
        color="green",
        linestyle="--",
        linewidth=0.6,
        label="_nolegend_",
        zorder=0,
    )

    # Central crossing: black; uncertainty crossings: grey.
    style_map = {
        "central": {"color": "black", "linestyle": "--", "marker": "o", "s": 9},
        "lower": {"color": "0.45", "linestyle": ":", "marker": "o", "s": 7},
        "upper": {"color": "0.45", "linestyle": ":", "marker": "o", "s": 7},
    }

    for key in ("lower", "upper", "central"):
        x_frame = crossings.get(key)
        if x_frame is None or not np.isfinite(x_frame):
            continue
        x = frame_to_hour(x_frame)
        st = style_map[key]
        ax.vlines(
            x,
            ymin=ymin,
            ymax=target,
            color=st["color"],
            linestyle=st["linestyle"],
            linewidth=0.6,
            label="_nolegend_",
            zorder=1,
        )
        ax.scatter(
            [x],
            [target],
            marker=st["marker"],
            s=st["s"],
            color=st["color"],
            linewidths=0.5,
            label="_nolegend_",
            zorder=6,
        )

    # Small unobtrusive labels. Comment these three lines out if they are too busy.
    ax.text(
        0.99,
        0.02,
        f"{target_mm2:g} mm²",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        color="0.35",
    )


def sanity_plot(
    colony_file,
    frames,
    colony_mm2,
    extrap_colony_mm2,
    extrap_colony_mm2_lower,
    extrap_colony_mm2_upper,
    clone_mm2,
    extrap_clone_mm2,
    extrap_clone_mm2_lower,
    extrap_clone_mm2_upper,
    touch_frame,
    fit_end,
    final_frame_index,
    final_colony_mm2,
    final_clone_mm2,
    crossings,
):
    """
    Create a 2-panel manuscript-style plot (colony + clone area in mm²) and save as PDF.
    """
    if not SAVE_PLOTS:
        return

    setup_matplotlib()

    frames = np.asarray(frames, dtype=float)
    frame_hours = frames_to_hours(frames)
    colony_mm2 = np.asarray(colony_mm2, dtype=float)
    extrap_colony_mm2 = np.asarray(extrap_colony_mm2, dtype=float)
    extrap_colony_mm2_lower = np.asarray(extrap_colony_mm2_lower, dtype=float)
    extrap_colony_mm2_upper = np.asarray(extrap_colony_mm2_upper, dtype=float)
    clone_mm2 = np.asarray(clone_mm2, dtype=float)
    extrap_clone_mm2 = np.asarray(extrap_clone_mm2, dtype=float)
    extrap_clone_mm2_lower = np.asarray(extrap_clone_mm2_lower, dtype=float)
    extrap_clone_mm2_upper = np.asarray(extrap_clone_mm2_upper, dtype=float)

    # Measured timelapse data stay visible across the full timelapse.
    measured_mask = np.ones_like(frames, dtype=bool)

    # Interpolation arrays, with the final target appended for visual continuity.
    colony_interp_frames, colony_interp_values = build_interp_series(
        frames, extrap_colony_mm2, fit_end, final_frame_index, final_colony_mm2
    )
    colony_interp_hours = frames_to_hours(colony_interp_frames)
    clone_interp_frames, clone_interp_values = build_interp_series(
        frames, extrap_clone_mm2, fit_end, final_frame_index, final_clone_mm2
    )
    clone_interp_hours = frames_to_hours(clone_interp_frames)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(3.6, 3.2),
        dpi=300,
        sharex=True,
    )
    ax_col, ax_clone = axes

    # ── Colony area plot ─────────────────────────────────────────────
    ax_col.plot(
        frame_hours[measured_mask],
        colony_mm2[measured_mask],
        color=COLONY_COLOR,
        linestyle="-",
        label="Colony area (measured)",
        zorder=3,
    )

    # Uncertainty band only for extrapolated timelapse frames after fit_end.
    # The measured points and final target point are not included.
    band_mask = (
        (frames > fit_end)
        & np.isfinite(extrap_colony_mm2_lower)
        & np.isfinite(extrap_colony_mm2_upper)
    )
    if band_mask.any():
        ax_col.fill_between(
            frame_hours[band_mask],
            extrap_colony_mm2_lower[band_mask],
            extrap_colony_mm2_upper[band_mask],
            color=COLONY_COLOR,
            alpha=0.18,
            linewidth=0,
            label="_nolegend_",
            zorder=1,
        )

    ax_col.plot(
        colony_interp_hours,
        colony_interp_values,
        color=COLONY_COLOR,
        linestyle="--",
        label="Colony area (interp)",
        zorder=4,
    )

    if final_colony_mm2 is not None:
        ax_col.scatter(
            frame_to_hour(final_frame_index),
            final_colony_mm2,
            marker="x",
            s=16,
            linewidths=0.9,
            color="black",
            label="Final area",
            zorder=7,
        )

    if touch_frame is not None:
        ax_col.axvline(
            frame_to_hour(touch_frame),
            color="0.35",
            linestyle=":",
            linewidth=0.8,
            label="Border hit",
            zorder=2,
        )

    if SHOW_TARGET_CROSSING_ON_PLOT:
        mark_target_crossings(ax_col, crossings, TARGET_COLONY_AREA_MM2)

    ax_col.set_ylabel("Colony area (mm²)")
    ax_col.legend(
        frameon=True,
        fontsize=6,
        ncol=2,
        handlelength=1.8,
        columnspacing=0.8,
        borderpad=0.3,
    )

    # ── Clone area plot ──────────────────────────────────────────────
    ax_clone.plot(
        frame_hours[measured_mask],
        clone_mm2[measured_mask],
        color=CLONE_COLOR,
        linestyle="-",
        label="Clone area (measured)",
        zorder=3,
    )

    # Uncertainty band only for extrapolated timelapse frames after fit_end.
    # The measured points and final target point are not included.
    clone_band_mask = (
        (frames > fit_end)
        & np.isfinite(extrap_clone_mm2_lower)
        & np.isfinite(extrap_clone_mm2_upper)
    )
    if clone_band_mask.any():
        ax_clone.fill_between(
            frame_hours[clone_band_mask],
            extrap_clone_mm2_lower[clone_band_mask],
            extrap_clone_mm2_upper[clone_band_mask],
            color=CLONE_COLOR,
            alpha=0.18,
            linewidth=0,
            label="_nolegend_",
            zorder=1,
        )

    ax_clone.plot(
        clone_interp_hours,
        clone_interp_values,
        color=CLONE_COLOR,
        linestyle="--",
        label="Clone area (interp)",
        zorder=4,
    )

    if final_clone_mm2 is not None:
        ax_clone.scatter(
            frame_to_hour(final_frame_index),
            final_clone_mm2,
            marker="x",
            s=16,
            linewidths=0.9,
            color="black",
            label="Final area",
            zorder=7,
        )

    if touch_frame is not None:
        ax_clone.axvline(
            frame_to_hour(touch_frame),
            color="0.35",
            linestyle=":",
            linewidth=0.8,
            label="Border hit",
            zorder=2,
        )

    ax_clone.set_xlabel("Time (h)")
    ax_clone.set_ylabel("Clone area (mm²)")
    ax_clone.legend(
        frameon=True,
        fontsize=6,
        ncol=2,
        handlelength=1.8,
        columnspacing=0.8,
        borderpad=0.3,
    )

    for ax in axes:
        _, ymax = ax.get_ylim()
        ax.set_ylim(bottom=0, top=ymax)
        ax.tick_params(width=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    plt.tight_layout()

    os.makedirs(DEFAULT_OUTPUT_FOLDER, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(colony_file))[0]
    plot_path = os.path.join(
        DEFAULT_OUTPUT_FOLDER,
        f"{base_name}_P{TARGET_PID}_plot_only_extrapolation_hours.pdf"
    )
    plt.savefig(plot_path, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    print(f"[plot saved] {plot_path}")


# ───────────────────────────── CORE ─────────────────────────────

def process_colony_file(colony_file, finals_map):
    print(f"\nProcessing timelapse file: {os.path.basename(colony_file)}")

    pid = extract_pid(colony_file)
    if pid != TARGET_PID:
        print(f"[skip] This script is restricted to P{TARGET_PID}; found P{pid}.")
        return

    colony_df = pd.read_csv(colony_file)

    required = {"colony_contour", "colony_area", "total_clone_area"}
    if not required.issubset(colony_df.columns):
        print(f"[warn] Missing {required - set(colony_df.columns)} in {os.path.basename(colony_file)} — skipping.")
        return

    # Ensure 'frame' column exists.
    if "frame" not in colony_df.columns:
        colony_df = colony_df.reset_index().rename(columns={"index": "frame"})

    frames = colony_df["frame"].to_numpy()
    frame_min, frame_max = frames.min(), frames.max()

    # Detect border touch.
    touch_frame = first_touch_frame(colony_df, colony_label=os.path.basename(colony_file))

    if touch_frame is None:
        print("  No border touch detected — using last frame as start for any extrapolation.")
        fit_end = frame_max
    else:
        fit_end = touch_frame - 1
        if fit_end < frame_min:
            fit_end = frame_min
        print(f"  Using frame {fit_end} as last trusted frame (fit_end).")

    # Final point in time is always 1 timepoint after the last timelapse frame.
    final_frame_index = frame_max + 1
    print(f"  Final target time set to {frame_to_hour(final_frame_index):.1f} h (frame {final_frame_index}, last_frame + 1).")

    # Basic arrays in px².
    colony_px2 = colony_df["colony_area"].to_numpy(dtype=float)
    clone_px2 = colony_df["total_clone_area"].to_numpy(dtype=float)

    # Convert timelapse areas to mm².
    colony_mm2 = px2_to_mm2(colony_px2, SEG_SCALE_UM_PER_PX)
    clone_mm2 = px2_to_mm2(clone_px2, SEG_SCALE_UM_PER_PX)

    # Prepare default: no change.
    extrap_colony_mm2 = colony_mm2.copy()
    extrap_clone_mm2 = clone_mm2.copy()

    # Lower/upper are NaN outside the extrapolated region, so the measured region
    # cannot accidentally be interpreted as uncertainty-adjusted data.
    extrap_colony_mm2_lower = np.full_like(colony_mm2, np.nan, dtype=float)
    extrap_colony_mm2_upper = np.full_like(colony_mm2, np.nan, dtype=float)

    extrap_clone_mm2_lower = np.full_like(clone_mm2, np.nan, dtype=float)
    extrap_clone_mm2_upper = np.full_like(clone_mm2, np.nan, dtype=float)

    final_colony_mm2 = None
    final_clone_mm2 = None

    if pid not in finals_map:
        print(f"[warn] No final CSV found for P{pid} — no extrapolation to final will be performed.")
    else:
        final_file = finals_map[pid]
        print(f"[info] Using final CSV for P{pid}: {os.path.basename(final_file)}")
        final_df = pd.read_csv(final_file)

        if "colony_area" not in final_df.columns:
            print(f"[warn] Final CSV {os.path.basename(final_file)} has no 'colony_area' — skipping extrapolation.")
        else:
            # Final areas: take the last row by default.
            final_colony_px2 = float(final_df["colony_area"].iloc[-1])

            if "total_clone_area" in final_df.columns:
                final_clone_px2 = float(final_df["total_clone_area"].iloc[-1])
            else:
                final_clone_px2 = None
                print(
                    f"[warn] Final CSV {os.path.basename(final_file)} has no 'total_clone_area'; "
                    "clone area will not be adjusted to final."
                )

            # Convert final areas to mm² using final scale.
            final_colony_mm2 = px2_to_mm2(final_colony_px2, FINAL_SCALE_UM_PER_PX)
            if final_clone_px2 is not None:
                final_clone_mm2 = px2_to_mm2(final_clone_px2, FINAL_SCALE_UM_PER_PX)

            # Ensure fit_end exists in frames.
            if fit_end in frames:
                idx_fit_end = np.where(frames == fit_end)[0][0]
            else:
                # In case of non-contiguous frames, take the last frame < fit_end.
                idx_candidates = np.where(frames < fit_end)[0]
                if len(idx_candidates) == 0:
                    idx_fit_end = 0
                    fit_end = frames[0]
                else:
                    idx_fit_end = idx_candidates[-1]
                    fit_end = frames[idx_fit_end]
                print(f"  Adjusted fit_end to existing frame {fit_end} (index {idx_fit_end}).")

            start_colony_mm2 = colony_mm2[idx_fit_end]
            start_clone_mm2 = clone_mm2[idx_fit_end]

            # Duration from fit_end to final_frame_index.
            duration = final_frame_index - fit_end
            if duration <= 0:
                print("  Warning: final_frame_index <= fit_end; no extrapolation performed.")
            else:
                mask_after = frames > fit_end
                if mask_after.any():
                    frames_after = frames[mask_after]
                    frames_clamped = np.minimum(frames_after, final_frame_index)
                    t = (frames_clamped - fit_end) / duration

                    # Colony: interpolate towards final_colony_mm2.
                    extrap_colony_mm2[mask_after] = (
                        start_colony_mm2 + t * (final_colony_mm2 - start_colony_mm2)
                    )

                    # Colony-only uncertainty band around the extrapolated curve.
                    # The measured region and final point remain unaffected.
                    extrap_colony_mm2_lower[mask_after] = (
                        extrap_colony_mm2[mask_after] * (1.0 - COLONY_AREA_VARIANCE_FRACTION)
                    )
                    extrap_colony_mm2_upper[mask_after] = (
                        extrap_colony_mm2[mask_after] * (1.0 + COLONY_AREA_VARIANCE_FRACTION)
                    )

                    # Clones: only adjust if we have a final clone area.
                    if final_clone_mm2 is not None:
                        extrap_clone_mm2[mask_after] = (
                            start_clone_mm2 + t * (final_clone_mm2 - start_clone_mm2)
                        )

                        # Clone-only uncertainty band around the extrapolated curve.
                        # The measured region and final point remain unaffected.
                        extrap_clone_mm2_lower[mask_after] = (
                            extrap_clone_mm2[mask_after] * (1.0 - CLONE_AREA_VARIANCE_FRACTION)
                        )
                        extrap_clone_mm2_upper[mask_after] = (
                            extrap_clone_mm2[mask_after] * (1.0 + CLONE_AREA_VARIANCE_FRACTION)
                        )
                    else:
                        extrap_clone_mm2 = clone_mm2

    crossings = compute_crossings_for_target(
        frames=frames,
        extrap_colony_mm2=extrap_colony_mm2,
        extrap_colony_lower_mm2=extrap_colony_mm2_lower,
        extrap_colony_upper_mm2=extrap_colony_mm2_upper,
        fit_end=fit_end,
        final_frame_index=final_frame_index,
        final_colony_mm2=final_colony_mm2,
        target_mm2=TARGET_COLONY_AREA_MM2,
    )

    if SAVE_CROSSING_OUTPUT_TXT:
        os.makedirs(DEFAULT_OUTPUT_FOLDER, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(colony_file))[0]
        crossing_txt_path = os.path.join(
            DEFAULT_OUTPUT_FOLDER,
            f"{base_name}_P{TARGET_PID}_{TARGET_COLONY_AREA_MM2:g}mm2_crossing_times_hours.txt"
        )
    else:
        crossing_txt_path = None
    print_and_save_crossings(
        crossings=crossings,
        target_mm2=TARGET_COLONY_AREA_MM2,
        colony_file=colony_file,
        output_txt_path=crossing_txt_path,
    )

    if SAVE_EXTRAPOLATED_CSV:
        # Kept only as an optional escape hatch. Default is False for this script.
        colony_df["extrapolated_total_area"] = mm2_to_px2(extrap_colony_mm2, SEG_SCALE_UM_PER_PX)
        colony_df["extrapolated_clone_area"] = mm2_to_px2(extrap_clone_mm2, SEG_SCALE_UM_PER_PX)
        colony_df["extrapolated_total_area_lower_3p118pct"] = mm2_to_px2(
            extrap_colony_mm2_lower, SEG_SCALE_UM_PER_PX
        )
        colony_df["extrapolated_total_area_upper_3p118pct"] = mm2_to_px2(
            extrap_colony_mm2_upper, SEG_SCALE_UM_PER_PX
        )
        colony_df["extrapolated_clone_area_lower_41p4pct"] = mm2_to_px2(
            extrap_clone_mm2_lower, SEG_SCALE_UM_PER_PX
        )
        colony_df["extrapolated_clone_area_upper_41p4pct"] = mm2_to_px2(
            extrap_clone_mm2_upper, SEG_SCALE_UM_PER_PX
        )
        os.makedirs(DEFAULT_OUTPUT_FOLDER, exist_ok=True)
        base_name, ext = os.path.splitext(os.path.basename(colony_file))
        out_path = os.path.join(DEFAULT_OUTPUT_FOLDER, f"{base_name}_with_extrapolation_to_final{ext}")
        colony_df.to_csv(out_path, index=False)
        print(f"[saved] {out_path}")
    else:
        print("[info] CSV writing disabled; only the plot and printed crossing values are produced.")

    sanity_plot(
        colony_file=colony_file,
        frames=frames,
        colony_mm2=colony_mm2,
        extrap_colony_mm2=extrap_colony_mm2,
        extrap_colony_mm2_lower=extrap_colony_mm2_lower,
        extrap_colony_mm2_upper=extrap_colony_mm2_upper,
        clone_mm2=clone_mm2,
        extrap_clone_mm2=extrap_clone_mm2,
        extrap_clone_mm2_lower=extrap_clone_mm2_lower,
        extrap_clone_mm2_upper=extrap_clone_mm2_upper,
        touch_frame=touch_frame,
        fit_end=fit_end,
        final_frame_index=final_frame_index,
        final_colony_mm2=final_colony_mm2,
        final_clone_mm2=final_clone_mm2,
        crossings=crossings,
    )


def main(timelapse_folder=None, finals_folder=None):
    if timelapse_folder is None or not os.path.isdir(timelapse_folder):
        timelapse_folder = input("Enter path to folder with timelapse colony CSVs: ").strip()
        if not os.path.isdir(timelapse_folder):
            print("[error] Invalid timelapse folder.")
            return

    if finals_folder is None or not os.path.isdir(finals_folder):
        finals_folder = input("Enter path to folder with FINAL CSVs: ").strip()
        if not os.path.isdir(finals_folder):
            print("[error] Invalid finals folder.")
            return

    finals_map = build_finals_map(finals_folder)
    if not finals_map:
        print("[warn] No usable final CSVs found; no extrapolation to final will be performed.")

    colony_files = detect_colony_files(timelapse_folder, TARGET_PID)
    if not colony_files:
        print(f"[warn] No timelapse colony file found for P{TARGET_PID}.")
        return

    if len(colony_files) > 1:
        print(f"[warn] Multiple P{TARGET_PID} files found. The script will process all matching P{TARGET_PID} files:")
        for cf in colony_files:
            print(f"  - {os.path.basename(cf)}")

    for cf in colony_files:
        process_colony_file(cf, finals_map)

    print("\nAll done!")


if __name__ == "__main__":
    main(DEFAULT_TIMELAPSE_FOLDER, DEFAULT_FINALS_FOLDER)
