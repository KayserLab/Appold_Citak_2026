#!/usr/bin/env python3
"""
ONLY USED FOR NT CONTROL:
Detect when a colony touches the border and extrapolate its total clone area.

For each colony CSV (already containing 'total_clone_area'):
  1. Detect the first frame where the colony contour touches the image border.
  2. Use a window of frames before the touch event for fitting.
  3. Fit a robust linear regression to 'total_clone_area' in that window.
  4. Extrapolate clone area forward in time (px²) up to EXTRAPOLATE_TO_FRAME.
  5. Add a new column 'extrapolated_clone_area' to the CSV.

Assumptions
- colony CSV contains at least: ['frame' (optional), 'colony_contour', 'total_clone_area']
- filenames contain P<id>, e.g. 'colony_data_P1_with_clonearea.csv'

Input/Output
- Input is read from Input_files/files_for_extrapolation/ (relative to this script).
- Output CSVs and optional plots are written alongside the input CSVs (same behavior as original),
  unless OVERWRITE_EXISTING is enabled.
"""

import os
import re
import ast

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import linregress


# ───────────────────────────── Project-relative IO ─────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_ROOT = os.path.join(SCRIPT_DIR, "Input_files")
DEFAULT_FOLDER = os.path.join(INPUT_ROOT, "files_for_extrapolation")


# ───────────────────────────── CONFIG ─────────────────────────────
IMG_WIDTH = 1376
IMG_HEIGHT = 1104
PAD = 1
N_FIT_FRAMES = 14
EXTRAPOLATE_TO_FRAME = 350
TOUCH_DETECTION_START_FRAME = 20
OVERWRITE_EXISTING = False
SHOW_PLOTS = True

# scale factor for conversion from px² → mm² (used only for plotting)
PIXEL_SIZE_UM = 8.648
SCALE_MM2_PER_PX2 = (PIXEL_SIZE_UM * 1e-3) ** 2
APPLY_SCALING = True  # retained for compatibility; not used in computation


# ───────────────────────────── HELPERS ─────────────────────────────
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
                print(f"[touch] {colony_label or '<unknown>'} touches border at frame {frame_val}")
                return frame_val
        except Exception:
            continue

    return None


def robust_linregress(x, y, z_thresh=2.5):
    """
    One-step robust linear regression:
      1) fit all points
      2) compute residuals
      3) drop points with |residual| > z_thresh * MAD
      4) refit on remaining points (if >=3), otherwise keep original fit

    Returns (slope, intercept, r2, mask_used).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    s0, b0, r0, _p0, _se0 = linregress(x, y)
    y_pred0 = b0 + s0 * x
    resid = y - y_pred0

    mad = np.median(np.abs(resid - np.median(resid)))
    if mad == 0 or np.isnan(mad):
        return s0, b0, r0 ** 2, np.ones_like(y, dtype=bool)

    sigma = 1.4826 * mad
    z = resid / sigma
    mask = np.abs(z) <= z_thresh

    if mask.sum() < 3:
        return s0, b0, r0 ** 2, np.ones_like(y, dtype=bool)

    s1, b1, r1, _p1, _se1 = linregress(x[mask], y[mask])
    return s1, b1, r1 ** 2, mask


def detect_colony_files(folder):
    """Return all colony CSVs in the folder."""
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".csv") and "colony" in f.lower()
    ]
    files = sorted(
        files,
        key=lambda x: int(re.search(r"P(\d{1,2})", x).group(1)) if re.search(r"P(\d{1,2})", x) else 0,
    )
    print(f"[info] Found {len(files)} colony files.")
    return files


# ───────────────────────────── CORE ─────────────────────────────
def process_colony_file(colony_file, overwrite=False, show_plot=False):
    print(f"\nProcessing {os.path.basename(colony_file)}")
    colony_df = pd.read_csv(colony_file)

    required = {"colony_contour", "total_clone_area"}
    if not required.issubset(colony_df.columns):
        print(f"[warn] Missing {required - set(colony_df.columns)} in {os.path.basename(colony_file)} — skipping.")
        return

    if "frame" not in colony_df.columns:
        colony_df = colony_df.reset_index().rename(columns={"index": "frame"})

    total_clone_px2 = colony_df["total_clone_area"].values

    touch_frame = first_touch_frame(colony_df, colony_label=os.path.basename(colony_file))

    if touch_frame is None:
        print("  No border touch detected — skipping extrapolation.")
        colony_df["extrapolated_clone_area"] = colony_df["total_clone_area"]
    else:
        fit_end = touch_frame - 1
        fit_start = max(0, fit_end - N_FIT_FRAMES)
        fit_slice = colony_df[(colony_df["frame"] >= fit_start) & (colony_df["frame"] <= fit_end)]

        if len(fit_slice) < 3:
            print(f"  Not enough frames ({len(fit_slice)}) before touch — skipping extrapolation.")
            colony_df["extrapolated_clone_area"] = colony_df["total_clone_area"]
        else:
            x = fit_slice["frame"].values
            y = fit_slice["total_clone_area"].values

            slope, intercept, r2, mask_inliers = robust_linregress(x, y, z_thresh=2.5)
            print(f"  Robust fit: slope={slope:.2f} px²/frame, intercept={intercept:.2f} px², R²={r2:.3f}")

            frames = colony_df["frame"].values
            extrapolated_px2 = np.where(
                frames <= fit_end,
                total_clone_px2,
                intercept + slope * frames,
            )
            extrapolated_px2 = np.where(frames <= EXTRAPOLATE_TO_FRAME, extrapolated_px2, np.nan)
            colony_df["extrapolated_clone_area"] = extrapolated_px2

            if show_plot:
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

                frames_plot = frames
                total_mm2 = total_clone_px2 * SCALE_MM2_PER_PX2
                extrapolated_mm2 = extrapolated_px2 * SCALE_MM2_PER_PX2

                inlier_x = x[mask_inliers]
                inlier_y_mm2 = y[mask_inliers] * SCALE_MM2_PER_PX2
                outlier_x = x[~mask_inliers]
                outlier_y_mm2 = y[~mask_inliers] * SCALE_MM2_PER_PX2

                fit_line_mm2 = (intercept + slope * inlier_x) * SCALE_MM2_PER_PX2

                plt.figure(figsize=(3.6, 1.8), dpi=300)
                plt.title(os.path.basename(colony_file))

                plt.scatter(frames_plot, total_mm2, s=5, label="total_clone_area", color="tab:blue")
                plt.scatter(
                    inlier_x, inlier_y_mm2, s=10,
                    facecolors="none", edgecolors="black",
                    linewidths=0.6, label="fit inliers"
                )

                if outlier_x.size:
                    plt.scatter(outlier_x, outlier_y_mm2, s=20, color="crimson", marker="x", label="fit outliers")

                plt.plot(inlier_x, fit_line_mm2, color="orange", lw=1.5, label=f"robust fit ({mask_inliers.sum()} frames)")
                plt.plot(frames_plot, extrapolated_mm2, "--", color="orange", lw=1, label="extrapolated")

                plt.axvline(touch_frame, color="red", linestyle=":", lw=0.8, label=f"touch @ {touch_frame}")

                plt.xlabel("Frame")
                plt.ylabel("Total clone area (mm²)")
                plt.legend(frameon=False, fontsize=6)
                plt.tight_layout()

                base, ext = os.path.splitext(colony_file)
                plot_pdf_path = f"{base}_extrapolation_plot.pdf"
                plt.savefig(plot_pdf_path)
                plt.show()

    base, ext = os.path.splitext(colony_file)
    out_path = colony_file if overwrite else f"{base}_with_extrapolation{ext}"
    colony_df.to_csv(out_path, index=False)
    print(f"[saved] {out_path}")


# ───────────────────────────── MAIN ─────────────────────────────
def main(folder=None, overwrite=False, show_plot=False):
    if folder is None or not os.path.isdir(folder):
        folder = input("Enter path to folder with colony CSVs: ").strip()
        if not os.path.isdir(folder):
            print("[error] Invalid folder.")
            return

    colony_files = detect_colony_files(folder)
    if not colony_files:
        print("[warn] No colony files found.")
        return

    for cf in colony_files:
        process_colony_file(cf, overwrite=overwrite, show_plot=show_plot)

    print("\nAll done!")


# ───────────────────────────── ENTRY ─────────────────────────────
if __name__ == "__main__":
    main(DEFAULT_FOLDER, OVERWRITE_EXISTING, SHOW_PLOTS)