#!/usr/bin/env python3
"""
Radial kymograph generator (multi-identifier, interactive angles, auto-ID detect).

For each identifier in the TIFF filenames in `folder_path`:

  - Loads all TIFFs containing that identifier (e.g. 20251114_P1_0_332-10.tiff).
  - Computes the colony centroid from frame `tc`.
  - Asks the user for a central angle (degrees) for that identifier.
  - Builds a set of angles around the central angle:
        [angle - angle_range, ..., angle + angle_range] with step `angle_step`
  - For each angle, it builds a space-time (kymograph) by sampling along the
    radial line from colony center to image border (with optional FOV shift).
  - Combines per-angle kymographs (nan-min).
  - Optionally smooths with a median filter.
  - Plots a radial kymograph with an overlay of the detected “front” and
    saves it as PDF.
  - Extracts, for each frame, the *farthest* non-background point along the line,
    converts to mm, and saves that time series as CSV.

Outputs (per identifier):
  - <output_prefix>_<identifier>_<angle>deg_kymograph.pdf
  - <output_prefix>_<identifier>_<angle>deg_max_distance.csv
"""

import os
import re
import math
import numpy as np
import pandas as pd
import h5py  # kept in case you later extend
import skimage.io as io
from skimage import measure
from scipy.ndimage import rotate, median_filter, map_coordinates
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import shutil


# ──────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────

def colony_parameters(array: np.ndarray) -> tuple[float, float]:
    """
    Finds the largest connected component in a labeled/binary-ish image and
    returns its centroid as (row, col).

    Assumes:
      - Colony pixels have labels 1 or 2
      - Background has label 3 (or something else)
    """
    array = np.where(array == 2, 1, array)
    labeled_colony = measure.label(array, background=3)
    components = measure.regionprops(labeled_colony)
    biggest_component = max(components, key=lambda x: x.area)
    return biggest_component.centroid  # (row, col) floats


def get_number(filename: str) -> int:
    """
    Extracts a numerical identifier from a filename (assumes pattern ending with '.tiff').
    """
    match = re.search(r'(\d+)\.tiff$', filename)
    return int(match.group(1)) if match else 0


def calculate_endpoint(image_width: int, image_height: int,
                       x: float, y: float, angle_degrees: float) -> tuple[float, float]:
    """
    Given image size (W,H), a starting point (x,y) in pixels and an angle (deg),
    compute where the ray first hits the image border in the *forward* direction.

    Returns:
      (end_x, end_y) in float coordinates.
    """
    ang = math.radians(angle_degrees)
    dx, dy = math.sin(ang), -math.cos(ang)
    t_candidates = []

    # Intersections with the 4 borders (in continuous coordinates)
    if dy != 0:
        # top (y=0)
        t = -y / dy
        ex, ey = x + t * dx, 0.0
        if 0 <= ex <= image_width - 1:
            t_candidates.append(t)
        # bottom (y=H-1)
        t = (image_height - 1 - y) / dy
        ex, ey = x + t * dx, image_height - 1.0
        if 0 <= ex <= image_width - 1:
            t_candidates.append(t)
    if dx != 0:
        # left (x=0)
        t = -x / dx
        ex, ey = 0.0, y + t * dy
        if 0 <= ey <= image_height - 1:
            t_candidates.append(t)
        # right (x=W-1)
        t = (image_width - 1 - x) / dx
        ex, ey = image_width - 1.0, y + t * dy
        if 0 <= ey <= image_height - 1:
            t_candidates.append(t)

    # Take the smallest positive t (forward ray)
    t_forward = min([t for t in t_candidates if t >= 0], default=0.0)
    ex, ey = x + t_forward * dx, y + t_forward * dy
    return ex, ey


def truncate_diagrams(diagrams: list[np.ndarray]) -> list[np.ndarray]:
    """
    Truncates all kymographs to the length of the shortest one (along space axis).
    """
    min_length = min(diagram.shape[1] for diagram in diagrams)
    return [diagram[:, :min_length] for diagram in diagrams]


# ──────────────────────────────────────────────────────────────
# Main worker
# ──────────────────────────────────────────────────────────────

def tiff_data_to_space_time(config: dict) -> None:
    """
    Processes TIFF images into a space-time (kymograph) diagram for multiple identifiers.
    For each identifier, interactively asks for the central angle.

    It then:
      - builds the kymograph (averaged across angles in a small range),
      - plots & saves it,
      - and computes, for each frame, the maximum radial distance (mm)
        of non-background (non-white) pixels along the line, saving this
        time series as CSV.
    """
    # Normalize folder path
    folder_path = config["folder_path"].replace('\\', '/')
    if not folder_path.endswith('/'):
        folder_path += '/'

    # --- basic config ---
    dark_bg          = config.get("dark_background", False)
    tc               = config["tc"]                 # timepoint for colony contour
    angle_range      = config["angle_range"]        # ± range around central angle (deg)
    angle_step       = config["angle_step"]         # step size (deg)
    angle_check      = config.get("angle_check", False)
    smoothing        = config.get("smoothing", False)

    # shift-compensation (one-time jump)
    shift_from       = int(config.get("shift_from_frame", 10**9))
    drow, dcol       = config.get("shift_vector_rc", (0.0, 0.0))
    if config.get("round_shift", True):
        drow, dcol = int(round(drow)), int(round(dcol))
    fill_value       = config.get("oob_fill", np.nan)

    # plotting/scaling
    scale_factor     = config["scale_factor"]       # µm per pixel
    frame_to_hour    = config.get("frame_to_hour", 0.5)
    figsize          = config.get("figsize", (5, 1.5))
    dpi              = config.get("dpi", 300)
    xlim_max         = config.get("xlim_max", None)
    ylim_max         = config.get("ylim_max", None)
    highlight_regions = config.get("highlight_regions", [])
    highlight_color   = config.get("highlight_color", "#bfbfbf")
    highlight_alpha   = config.get("highlight_alpha", 0.3)
    hline_positions   = config.get("hline_positions", [])

    display_width    = config.get("display_width", None)
    display_height   = config.get("display_height", None)

    # output
    output_dir       = config.get("output_dir", folder_path)  # default: same as TIFFs
    output_prefix    = config.get("output_prefix", "radial_kymograph")

    os.makedirs(output_dir, exist_ok=True)

    # ──────────────────────────────────────────────
    # AUTO-DETECT IDENTIFIERS FROM TIFF FILENAMES
    # ──────────────────────────────────────────────
    # If config["identifiers"] is provided and non-empty, use it.
    # Otherwise, auto-detect patterns "_P<NUM>_" in all .tiff files.
    cfg_ids = config.get("identifiers", None)
    if cfg_ids:
        identifiers = list(cfg_ids)
        print(f"[INFO] Using identifiers from config: {identifiers}")
    else:
        all_tiffs = [f for f in os.listdir(folder_path) if f.endswith(".tiff")]
        id_nums = set()
        for fname in all_tiffs:
            # Look for _P<number>_ in the filename, e.g. 20251114_P10_0_332-10.tiff
            for m in re.finditer(r'_P(\d+)_', fname):
                id_nums.add(int(m.group(1)))
        if not id_nums:
            raise RuntimeError(
                f"No identifiers found in TIFF filenames in folder: {folder_path}\n"
                f"Expected patterns like '_P1_' or '_P10_'."
            )
        identifiers = [f"P{n}" for n in sorted(id_nums)]
        print(f"[INFO] Auto-detected identifiers from TIFFs: {identifiers}")

    # Matplotlib style
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'pdf.fonttype': 42,  # embed TrueType
        'ps.fonttype': 42,
        'font.size': 6,
        'axes.titlesize': 6,
        'axes.labelsize': 6,
        'axes.linewidth': 0.5,
        'xtick.major.size': 2,
        'ytick.major.size': 2,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
        'xtick.major.pad': 2,
        'ytick.major.pad': 2,
        'axes.labelpad': 1,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.labelsize': 5,
        'ytick.labelsize': 5,
        'legend.frameon': False,
        'lines.linewidth': 1.0,
    })

    # colormap: 0 and 3 mapped to transparent white, 1 and 2 are colony states
    colors_rgba = [
        (1, 1, 1, 0),                # label 0: transparent white (background)
        (0.855, 0.647, 0.125, 1),    # label 1: goldenrod
        (0.254, 0.41, 0.882, 1),     # label 2: royal blue
        (1, 1, 1, 0)                 # label 3: transparent white (background)
    ]
    cmap = mcolors.ListedColormap(colors_rgba)
    norm = mcolors.BoundaryNorm([0, 1, 2, 3, 4], cmap.N, clip=True)

    # Loop over identifiers
    for identifier in tqdm(identifiers, desc="Processing identifiers"):
        # --- load image stack for this identifier ---
        # Be strict: require pattern "_P<num>_" to avoid P1 vs P10 confusion.
        id_pattern = f"_{identifier}_"
        tiff_files = [f for f in os.listdir(folder_path)
                      if f.endswith('.tiff') and id_pattern in f]
        tiff_files.sort(key=get_number)

        if not tiff_files:
            print(f"[WARN] No TIFFs found for identifier '{identifier}' in {folder_path}")
            continue

        images = [io.imread(os.path.join(folder_path, file)) for file in tiff_files]
        combined_array = np.stack(images)  # shape: (T, H, W)
        T, H, W = combined_array.shape

        # --- colony centroid ---
        centroid_rc = colony_parameters(combined_array[tc])  # (row, col)
        r0, c0 = float(centroid_rc[0]), float(centroid_rc[1])

        # --- ask user for angle for this identifier ---
        while True:
            angle_input = input(
                f"Enter central angle in degrees for identifier '{identifier}' "
                f"(e.g. 209), or press Enter to use config.get('default_angle'): "
            ).strip()

            if angle_input == "":
                default_angle = config.get("default_angle", None)
                if default_angle is None:
                    print("No default_angle in config; please enter a numeric angle.")
                    continue
                central_angle = float(default_angle)
                print(f"Using default_angle = {central_angle:.2f}° for '{identifier}'.")
                break
            try:
                central_angle = float(angle_input)
                break
            except ValueError:
                print("Could not parse angle. Please enter a number like 209 or 18.5.")

        # build angle list around central angle
        angle_values = np.arange(
            central_angle - angle_range,
            central_angle + angle_range + angle_step,
            angle_step
        )

        # container for per-angle kymographs
        space_time_diagrams = []

        for a in angle_values:
            # continuous endpoint on image border
            ex, ey = calculate_endpoint(W, H, c0, r0, a)
            r1, c1 = float(ey), float(ex)

            # build evenly spaced coordinates along line (step = 1 px)
            L = math.hypot(r1 - r0, c1 - c0)
            n_samples = int(math.floor(L / 1.0)) + 1
            rr_base = np.linspace(r0, r1, n_samples, dtype=np.float32)
            cc_base = np.linspace(c0, c1, n_samples, dtype=np.float32)

            # optional angle check plot (only for angle nearest central_angle)
            if angle_check and np.isclose(a, central_angle, atol=angle_step / 2):
                style = 'dark_background' if dark_bg else 'default'
                with plt.style.context(style):
                    plt.figure(figsize=(6, 6), dpi=200)
                    plt.imshow(combined_array[tc], cmap='gray')
                    # pre-shift line preview
                    plt.plot([cc_base[0], cc_base[-1]],
                             [rr_base[0], rr_base[-1]],
                             'r-', label='Pre-shift line')
                    # post-shift line preview
                    plt.plot([cc_base[0] + dcol, cc_base[-1] + dcol],
                             [rr_base[0] + drow, rr_base[-1] + drow],
                             'c--', label='Post-shift line')
                    if display_width is not None:
                        plt.xlim(0, display_width)
                    if display_height is not None:
                        plt.ylim(display_height, 0)
                    plt.title(f'{identifier}: line positions at frame {shift_from}')
                    plt.legend()
                    plt.tight_layout()
                    plt.show()

            # build kymograph with per-frame shift compensation
            rows = []
            for t in range(T):
                if t >= shift_from:
                    rr = rr_base + float(drow)
                    cc = cc_base + float(dcol)
                else:
                    rr = rr_base
                    cc = cc_base

                # bilinear interpolation at uniform arc-length coordinates
                row = map_coordinates(
                    combined_array[t],
                    [rr, cc],
                    order=1,
                    mode='constant',
                    cval=fill_value
                )
                rows.append(row)

            kymo = np.stack(rows, axis=0)  # shape: (T, n_samples)
            space_time_diagrams.append(kymo)

        # --- truncate to shortest length and combine robustly (handle NaNs) ---
        space_time_diagrams = truncate_diagrams(space_time_diagrams)  # list of (T, L')
        averaged_space_time_diagram = np.nanmin(space_time_diagrams, axis=0)  # (T, L')

        # optional smoothing
        if smoothing:
            averaged_space_time_diagram = median_filter(averaged_space_time_diagram, size=(3, 1))

            # transpose for plotting (space x time)
            kymo_T = averaged_space_time_diagram.T  # shape: (L', T)
            max_x_pixels = kymo_T.shape[0]
            max_y_frames = kymo_T.shape[1]

            # spatial axis in µm and mm (does NOT depend on time truncation)
            y_um = np.linspace(0, max_x_pixels * scale_factor, max_x_pixels + 1)
            y_mm = y_um / 1000.0

            # ──────────────────────────────────────────────
            # 1) Compute farthest valid index per frame
            # ──────────────────────────────────────────────
            # Background: labels 0 and 3 (and NaN). Colony: labels 1 and 2.
            farthest_indices = []

            for t in range(max_y_frames):
                col = kymo_T[:, t]  # shape: (L',)
                valid_mask = (~np.isnan(col)) & (col != 0) & (col != 3)

                if np.any(valid_mask):
                    farthest_idx = int(np.max(np.where(valid_mask)[0]))
                else:
                    farthest_idx = np.nan

                farthest_indices.append(farthest_idx)

            # ──────────────────────────────────────────────
            # 2) Detect first frame where front hits image border
            #    but ONLY allow truncation after min_cutoff_frame
            # ──────────────────────────────────────────────
            border_margin_px = config.get("border_margin_px", 2)
            min_cutoff_frame = config.get("min_cutoff_frame", 0)

            border_idx = max_x_pixels - 1
            threshold_idx = border_idx - border_margin_px

            border_hit_frames = [
                i for i, idx in enumerate(farthest_indices)
                if not np.isnan(idx) and idx >= threshold_idx
            ]

            # Filter hits to only those at or after min_cutoff_frame
            border_hit_frames_filtered = [
                f for f in border_hit_frames if f >= min_cutoff_frame
            ]

            if border_hit_frames_filtered:
                cutoff_frame = border_hit_frames_filtered[0]  # first valid hit after min_cutoff_frame
            else:
                cutoff_frame = max_y_frames - 1  # no valid hit → keep full time range

            # Effective number of frames to keep (0..cutoff_frame)
            effective_frames = cutoff_frame + 1

            # Truncate kymograph and farthest_indices in time
            kymo_T = kymo_T[:, :effective_frames]
            farthest_indices = farthest_indices[:effective_frames]

            # Update shapes after truncation
            max_x_pixels = kymo_T.shape[0]
            max_y_frames = kymo_T.shape[1]

            # ──────────────────────────────────────────────
            # 3) Build time axis & distance series (rounded)
            # ──────────────────────────────────────────────
            x_hours = np.linspace(0, max_y_frames * frame_to_hour, max_y_frames + 1)

            frames = list(range(max_y_frames))
            times_h = [t * frame_to_hour for t in frames]

            max_distances_mm = []
            for idx in farthest_indices:
                if np.isnan(idx):
                    max_distances_mm.append(np.nan)
                else:
                    dist_mm = (idx * scale_factor) / 1000.0
                    max_distances_mm.append(dist_mm)

            # Round distances to 3 decimals for CSV + overlay
            max_distances_mm = [
                round(x, 3) if not np.isnan(x) else np.nan
                for x in max_distances_mm
            ]

            df = pd.DataFrame({
                "frame": frames,
                "time_h": times_h,
                "max_distance_mm": max_distances_mm,
            })

        # ──────────────────────────────────────────────
        # Plot kymograph + SANITY CHECK OVERLAY + BORDER LINE
        # ──────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Highlighted regions (given in frames, converted to hours)
        for region in highlight_regions:
            t0, t1 = np.array(region) * frame_to_hour
            ax.axvspan(t0, t1, color=highlight_color, alpha=highlight_alpha, zorder=0)

        pcm = ax.pcolormesh(
            x_hours, y_mm, kymo_T,
            cmap=cmap, norm=norm,
            edgecolors='none',
            antialiased=False,
            rasterized=False
        )

        # Detected front overlay (sanity check)
        ax.plot(times_h, max_distances_mm,
                linestyle='-',
                linewidth=0.75,
                color='black',
                label='Detected front')

        # ── NEW: horizontal line at image border ──
        border_idx = max_x_pixels - 1
        border_distance_mm = (border_idx * scale_factor) / 1000.0
        ax.axhline(border_distance_mm,
                   linestyle='--',
                   linewidth=0.5,
                   color='black',
                   label='Image border')

        ax.set_xlabel('Time (h)', labelpad=2)
        ax.set_ylabel('Distance to center (mm)', labelpad=5)

        # x labels every 25 hours
        x_tick_interval = 25
        x_ticks = np.arange(0, x_hours[-1] + x_tick_interval, x_tick_interval)
        ax.set_xticks(x_ticks)

        # y labels every 1 mm
        y_tick_interval = 1
        y_ticks = np.arange(0, y_mm[-1] + y_tick_interval, y_tick_interval)
        ax.set_yticks(y_ticks)

        # Same swapped logic as before (kept for compatibility)
        if ylim_max is not None:
            ax.set_xlim(0, ylim_max)
        else:
            ax.set_xlim(0, x_hours[-1])
        if xlim_max is not None:
            ax.set_ylim(0, xlim_max)
        else:
            ax.set_ylim(0, y_mm[-1])

        for h in hline_positions:
            ax.axvline(x=h, color='black',
                       linestyle=(0, (3, 5, 1, 5)), linewidth=1)

        ax.legend(loc='upper right', fontsize=5, frameon=False)

        ax.set_title("")
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        plt.tight_layout(pad=0.2)
        plt.show()
        # ──────────────────────────────────────────────
        # Save outputs (PDF + CSV + optional code copy)
        # ──────────────────────────────────────────────
        angle_tag = f"{int(round(central_angle))}deg"
        base_name = f"{output_prefix}_{identifier}_{angle_tag}"

        pdf_path = os.path.join(output_dir, base_name + "_kymograph.pdf")
        csv_path = os.path.join(output_dir, base_name + "_max_distance.csv")

        fig.savefig(pdf_path, format='pdf',
                    bbox_inches=None, transparent=False, dpi=300)
        plt.close(fig)

        df.to_csv(csv_path, index=False)
        print(f"[INFO] Saved kymograph PDF for '{identifier}' to: {pdf_path}")
        print(f"[INFO] Saved max-distance CSV for '{identifier}' to: {csv_path}")

        # Save a copy of the current script alongside (best-effort)
        try:
            script_file = os.path.abspath(__file__)
            code_path = os.path.splitext(pdf_path)[0] + "_code.txt"
            shutil.copy(script_file, code_path)
            print(f"[INFO] Code copied to {code_path}")
        except NameError:
            print("[WARN] No __file__ available (likely running in REPL/Notebook). "
                  "Save code manually if needed.")


# ──────────────────────────────────────────────────────────────
# Main / config
# ──────────────────────────────────────────────────────────────

def main():
    config = {
        # File settings
        "folder_path": r"D:\Image_Segmentation\20241210_pulse_treatment\optimized",

        # If you leave identifiers empty or None, they are auto-detected from filenames.
        # Example filenames: 20251114_P1_0_332-10.tiff, 20251114_P10_0_332-10.tiff
        "identifiers": None,   # or [] for auto-detect

        "output_dir": r"D:\Image_Segmentation\20241210_pulse_treatment\optimized\Sus_Kymos",
        "output_prefix": "Pulse_radial",

        # Timepoint (frame index) for colony contour
        "tc": 100,

        # If user just presses Enter for angle, use this:
        "default_angle": 209,   # degrees

        # Angle range and resolution
        "angle_range": 2,       # ± range around central angle (deg)
        "angle_step": 0.5,      # angle step size (deg)
        "angle_check": True,    # show line overlay for central angle?
        "smoothing": True,      # median filter the kymograph?

        # Plot display settings
        "scale_factor": 8.648,  # µm per pixel
        "frame_to_hour": 0.5,   # frames → hours
        "figsize": (1.65, 1.1),
        "dpi": 300,
        "display_width": 1376,  # only used for preview
        "display_height": 1104, # only used for preview
        "xlim_max": 6.5,          # max x-axis in hours (note: swapped logic kept)
        "ylim_max": 160,        # max y-axis in mm (note: swapped logic kept)

        # Highlight regions in frames (will be converted to hours)
        "highlight_regions": [[37, 65]],
        "highlight_color": "#bfbfbf",
        "highlight_alpha": 1.0,
        # How close (in pixels) to the image border we consider "border hit"
        "border_margin_px": 2,
        # Do not truncate before this frame index (0-based)
        "min_cutoff_frame": 50,

        # Optional vertical lines (in hours)
        "hline_positions": [],

        # Dark background toggle for preview plot only
        "dark_background": False,

        # Compensate a one-time FOV jump
        "shift_from_frame": 0,      # first frame that is displaced (0-based)
        "shift_vector_rc": (0, 0),  # (Δrow, Δcol)
        "round_shift": True,
        "oob_fill": np.nan,
    }

    tiff_data_to_space_time(config)


if __name__ == "__main__":
    main()
