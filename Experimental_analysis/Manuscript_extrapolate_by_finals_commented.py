#!/usr/bin/env python3
"""
Takes files created by Manuscript_add_clone_total_commented.py and
Extrapolate colony + clone area to a final size measured from separate final CSVs,
and create sanity-check plots.

For each timelapse colony CSV (with at least 'colony_area' and 'total_clone_area'):

  1. Detect the first frame where the colony contour touches the image border.
  2. Define the last trusted frame as: fit_end = touch_frame - 1
     (if no touch is detected, use the last frame as fit_end).
  3. Find the corresponding "final" CSV in another folder based on the P<id> identifier.
  4. Convert both timelapse and final areas to mm² using their respective pixel sizes:
         SEG_SCALE_UM_PER_PX   (timelapse)
         FINAL_SCALE_UM_PER_PX (finals)
  5. From fit_end onward, linearly extrapolate in mm² from the area at fit_end
     to the final area, with the final point being at frame = last_timelapse_frame + 1.
     After that frame, the extrapolated value stays constant (plateau at the final size).
  6. Convert the extrapolated values back to px² at the timelapse scale and store:
         'extrapolated_total_area'  (for colony_area)
         'extrapolated_clone_area'  (for total_clone_area)
  7. Save a new CSV with suffix '_with_extrapolation_to_final.csv' (or overwrite).
  8. Create a sanity-check plot (PDF) showing:
         - original vs extrapolated areas (mm²)
         - touch frame, fit_end, final_frame_index
         - final target point from the final CSV.
"""

import ast
import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ───────────────────────────── CONFIG ─────────────────────────────

# Project-relative IO roots
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_ROOT = os.path.join(SCRIPT_DIR, "Input_files")
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "Output_files")

# Subfolders inside Input_files/ for the two required inputs
# - timelapse: timelapse colony CSVs (with clonearea already added)
# - finals:    final-size CSVs used as extrapolation targets
TIMELAPSE_INPUT_RELPATH = "files_for_extrapolation"
FINALS_INPUT_RELPATH = "finals_for_extrapolation"

IMG_WIDTH = 1376
IMG_HEIGHT = 1104
PAD = 1
TOUCH_DETECTION_START_FRAME = 20  # start checking only after this frame

OVERWRITE_EXISTING = False  # overwrite or create new file with suffix

# Pixel sizes (µm per pixel)
SEG_SCALE_UM_PER_PX = 8.648    # timelapse
FINAL_SCALE_UM_PER_PX = 14.424  # finals

# Plot settings
SAVE_PLOTS = True   # always save sanity plots
SHOW_PLOTS = True   # set True if you want them to pop up


# ───────────────────────────── HELPERS ─────────────────────────────

def px2_to_mm2(px2, pixel_size_um):
    """Convert area from px² to mm² given pixel size in µm/px."""
    return px2 * (pixel_size_um * 1e-3) ** 2


def mm2_to_px2(mm2, pixel_size_um):
    """Convert area from mm² to px² given pixel size in µm/px."""
    return mm2 / (pixel_size_um * 1e-3) ** 2


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
    border_x_min, border_y_min = 0.5, 0.5
    border_x_max, border_y_max = width + 0.5, height + 0.5

    frames = colony_df["frame"].to_numpy() if "frame" in colony_df.columns else np.arange(len(colony_df))

    for pos in range(len(colony_df)):
        frame_val = int(frames[pos])
        if frame_val < start_frame:
            continue  # skip early frames (noisy)

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
                print(f"[touch] {colony_label or '<unknown>'} touches border at frame {frame_val}")
                return frame_val
        except Exception:
            continue

    return None


def detect_colony_files(folder):
    """
    Return all timelapse colony CSVs in the folder.

    Note: The filename filtering is kept identical to the original script.
    """
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".csv") and "colony" and "clonearea" in f.lower()
    ]
    files = sorted(
        files,
        key=lambda x: int(re.search(r"P(\d{1,2})", x).group(1)) if re.search(r"P(\d{1,2})", x) else 0,
    )
    print(f"[info] Found {len(files)} timelapse colony files.")
    return files


def build_finals_map(finals_folder):
    """
    Build a mapping from P-id (int) to final CSV path.

    It expects that final CSV filenames contain 'P<id>', e.g. '20251114_P8_0_332-10.csv'.
    For each id, the last file found wins (in case there are multiple).
    """
    mapping = {}
    if not os.path.isdir(finals_folder):
        print(f"[warn] Finals folder does not exist: {finals_folder}")
        return mapping

    for fname in os.listdir(finals_folder):
        if not fname.lower().endswith(".csv"):
            continue
        m = re.search(r"P(\d{1,2})(?!\d)", fname)
        if not m:
            continue
        pid = int(m.group(1))
        full_path = os.path.join(finals_folder, fname)
        mapping[pid] = full_path

    print(f"[info] Found {len(mapping)} final CSV files with P<id> in name.")
    return mapping


def setup_matplotlib():
    """Nature-ish style rcParams for sanity plots."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "axes.linewidth": 0.5,
        "xtick.major.size": 0,
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


def sanity_plot(
    title,
    plot_path,
    frames,
    colony_mm2,
    extrap_colony_mm2,
    clone_mm2,
    extrap_clone_mm2,
    touch_frame,
    fit_end,
    final_frame_index,
    final_colony_mm2,
    final_clone_mm2,
):
    """Create a 2-panel sanity plot (colony + clone area in mm²) and save as PDF."""
    if not SAVE_PLOTS:
        return

    setup_matplotlib()

    fig, axes = plt.subplots(
        nrows=2, ncols=1,
        figsize=(3.6, 3.2),
        dpi=300,
        sharex=True,
    )
    ax_col, ax_clone = axes

    fig.suptitle(title, fontsize=7)

    # Colony area plot
    ax_col.plot(frames, colony_mm2, label="colony_area (orig)", linestyle="-")
    ax_col.plot(frames, extrap_colony_mm2, label="colony_area (extrap)", linestyle="--")
    if final_colony_mm2 is not None:
        ax_col.scatter(final_frame_index, final_colony_mm2, marker="x", s=20, label="final colony (target)")

    if touch_frame is not None:
        ax_col.axvline(touch_frame, color="red", linestyle=":", linewidth=0.8, label=f"touch @ {touch_frame}")
    ax_col.axvline(fit_end, color="grey", linestyle="--", linewidth=0.8, label=f"fit_end @ {fit_end}")
    ax_col.axvline(final_frame_index, color="green", linestyle=":", linewidth=0.8, label=f"final_frame @ {final_frame_index}")

    ax_col.set_ylabel("Colony area (mm²)")
    ax_col.legend(frameon=False, fontsize=6, ncol=2)

    # Clone area plot
    ax_clone.plot(frames, clone_mm2, label="clone_area (orig)", linestyle="-")
    ax_clone.plot(frames, extrap_clone_mm2, label="clone_area (extrap)", linestyle="--")
    if final_clone_mm2 is not None:
        ax_clone.scatter(final_frame_index, final_clone_mm2, marker="x", s=20, label="final clone (target)")

    if touch_frame is not None:
        ax_clone.axvline(touch_frame, color="red", linestyle=":", linewidth=0.8, label=f"touch @ {touch_frame}")
    ax_clone.axvline(fit_end, color="grey", linestyle="--", linewidth=0.8, label=f"fit_end @ {fit_end}")
    ax_clone.axvline(final_frame_index, color="green", linestyle=":", linewidth=0.8, label=f"final_frame @ {final_frame_index}")

    ax_clone.set_xlabel("Frame")
    ax_clone.set_ylabel("Clone area (mm²)")
    ax_clone.legend(frameon=False, fontsize=6, ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(plot_path)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    print(f"[plot saved] {plot_path}")


# ───────────────────────────── CORE ─────────────────────────────

def process_colony_file(colony_file, finals_map, out_dir, overwrite=False):
    print(f"\nProcessing timelapse file: {os.path.basename(colony_file)}")
    colony_df = pd.read_csv(colony_file)

    required = {"colony_contour", "colony_area", "total_clone_area"}
    if not required.issubset(colony_df.columns):
        print(f"[warn] Missing {required - set(colony_df.columns)} in {os.path.basename(colony_file)} — skipping.")
        return

    # Ensure 'frame' column exists
    if "frame" not in colony_df.columns:
        colony_df = colony_df.reset_index().rename(columns={"index": "frame"})

    frames = colony_df["frame"].to_numpy()
    frame_min, frame_max = frames.min(), frames.max()

    # Detect border touch (for plotting + logic)
    touch_frame = first_touch_frame(colony_df, colony_label=os.path.basename(colony_file))

    if touch_frame is None:
        print("  No border touch detected — using last frame as start for any extrapolation.")
        fit_end = frame_max
    else:
        fit_end = touch_frame - 1
        if fit_end < frame_min:
            fit_end = frame_min
        print(f"  Using frame {fit_end} as last trusted frame (fit_end).")

    # Final point in time is always 1 timepoint after the last timelapse frame
    final_frame_index = frame_max + 1
    print(f"  Final (target) frame index set to {final_frame_index} (last_frame + 1).")

    # Extract P-id from filename
    m = re.search(r"P(\d{1,2})", os.path.basename(colony_file))
    if not m:
        print(f"[warn] Could not find P<id> in filename {os.path.basename(colony_file)} — skipping.")
        return
    pid = int(m.group(1))

    # Basic arrays in px²
    colony_px2 = colony_df["colony_area"].to_numpy(dtype=float)
    clone_px2 = colony_df["total_clone_area"].to_numpy(dtype=float)

    # Convert timelapse areas to mm² using seg scale (for plotting + extrapolation)
    colony_mm2 = px2_to_mm2(colony_px2, SEG_SCALE_UM_PER_PX)
    clone_mm2 = px2_to_mm2(clone_px2, SEG_SCALE_UM_PER_PX)

    # Prepare default: no change (extrap = original)
    extrap_colony_mm2 = colony_mm2.copy()
    extrap_clone_mm2 = clone_mm2.copy()
    final_colony_mm2 = None
    final_clone_mm2 = None

    if pid not in finals_map:
        print(f"[warn] No final CSV found for P{pid} — keeping areas as-is, no extrapolation to final.")
    else:
        final_file = finals_map[pid]
        print(f"[info] Using final CSV for P{pid}: {os.path.basename(final_file)}")
        final_df = pd.read_csv(final_file)

        if "colony_area" not in final_df.columns:
            print(f"[warn] Final CSV {os.path.basename(final_file)} has no 'colony_area' — skipping extrapolation.")
        else:
            # Final areas (take the last row by default)
            final_colony_px2 = float(final_df["colony_area"].iloc[-1])

            if "total_clone_area" in final_df.columns:
                final_clone_px2 = float(final_df["total_clone_area"].iloc[-1])
            else:
                final_clone_px2 = None
                print(
                    f"[warn] Final CSV {os.path.basename(final_file)} has no 'total_clone_area'; "
                    f"clone area will not be adjusted to final."
                )

            # Convert final areas to mm² using final scale
            final_colony_mm2 = px2_to_mm2(final_colony_px2, FINAL_SCALE_UM_PER_PX)
            if final_clone_px2 is not None:
                final_clone_mm2 = px2_to_mm2(final_clone_px2, FINAL_SCALE_UM_PER_PX)

            # Ensure fit_end exists in frames
            if fit_end in frames:
                idx_fit_end = np.where(frames == fit_end)[0][0]
            else:
                # in case of weird non-contiguous frames, take the last frame < fit_end
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

            # Duration from fit_end to final_frame_index
            duration = final_frame_index - fit_end
            if duration <= 0:
                print("  Warning: final_frame_index <= fit_end; no extrapolation performed.")
            else:
                mask_after = frames > fit_end
                if mask_after.any():
                    frames_after = frames[mask_after]
                    frames_clamped = np.minimum(frames_after, final_frame_index)
                    t = (frames_clamped - fit_end) / duration  # 0 at fit_end, 1 at final_frame_index

                    # Colony: always extrapolate towards final_colony_mm2
                    extrap_colony_mm2[mask_after] = start_colony_mm2 + t * (final_colony_mm2 - start_colony_mm2)

                    # Clones: only adjust if we have a final clone area
                    if final_clone_mm2 is not None:
                        extrap_clone_mm2[mask_after] = start_clone_mm2 + t * (final_clone_mm2 - start_clone_mm2)
                    else:
                        # keep clone_mm2 as original
                        extrap_clone_mm2 = clone_mm2

    # Convert back to px² at seg scale so downstream scripts behave as before
    extrap_colony_px2 = mm2_to_px2(extrap_colony_mm2, SEG_SCALE_UM_PER_PX)
    extrap_clone_px2 = mm2_to_px2(extrap_clone_mm2, SEG_SCALE_UM_PER_PX)

    colony_df["extrapolated_total_area"] = extrap_colony_px2
    colony_df["extrapolated_clone_area"] = extrap_clone_px2

    # Output paths (all outputs go to Output_files/)
    os.makedirs(out_dir, exist_ok=True)

    in_name = os.path.basename(colony_file)
    in_stem, in_ext = os.path.splitext(in_name)

    if overwrite:
        out_csv_path = os.path.join(out_dir, in_name)
    else:
        out_csv_path = os.path.join(out_dir, f"{in_stem}_with_extrapolation_to_final{in_ext}")

    colony_df.to_csv(out_csv_path, index=False)
    print(f"[saved] {out_csv_path}")

    plot_path = os.path.join(out_dir, f"{in_stem}_with_extrapolation_to_final_plot.pdf")

    # Sanity plot (in mm², using arrays we already have)
    sanity_plot(
        title=in_name,
        plot_path=plot_path,
        frames=frames,
        colony_mm2=colony_mm2,
        extrap_colony_mm2=extrap_colony_mm2,
        clone_mm2=clone_mm2,
        extrap_clone_mm2=extrap_clone_mm2,
        touch_frame=touch_frame,
        fit_end=fit_end,
        final_frame_index=final_frame_index,
        final_colony_mm2=final_colony_mm2,
        final_clone_mm2=final_clone_mm2,
    )


def main():
    timelapse_folder = os.path.join(INPUT_ROOT, TIMELAPSE_INPUT_RELPATH)
    finals_folder = os.path.join(INPUT_ROOT, FINALS_INPUT_RELPATH)

    if not os.path.isdir(timelapse_folder):
        print(f"[error] Timelapse input folder not found: {timelapse_folder}")
        return
    if not os.path.isdir(finals_folder):
        print(f"[error] Finals input folder not found: {finals_folder}")
        return

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    finals_map = build_finals_map(finals_folder)
    if not finals_map:
        print("[warn] No usable final CSVs found; no extrapolation to final will be performed.")

    colony_files = detect_colony_files(timelapse_folder)
    if not colony_files:
        print("[warn] No timelapse colony files found.")
        return

    # NEW
    out_dir = OUTPUT_ROOT

    for cf in colony_files:
        process_colony_file(cf, finals_map, out_dir, overwrite=OVERWRITE_EXISTING)

    print("\nAll done!")


if __name__ == "__main__":
    main()