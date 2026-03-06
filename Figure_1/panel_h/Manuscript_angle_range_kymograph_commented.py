#!/usr/bin/env python3
"""
Generate radial kymographs (space–time diagrams) from segmented TIFF time-lapse stacks.

Expected input:
- Segmented TIFF frames (same HxW), one file per timepoint.
- Pixel values are assumed to be discrete labels (e.g. 0/3 background, 1 sensitive, 2 resistant).

Workflow per identifier (e.g. "P12"):
1) Load all TIFF frames containing the identifier and stack them to (T, H, W).
2) Compute colony centroid from a chosen reference frame `tc` (largest connected component).
3) For a range of angles around a central `angle`, sample intensities along a radial line
   from centroid to the image border, for every timepoint.
4) Combine kymographs across angles using pixel-wise minimum (NaN-aware).
5) Optionally apply median smoothing (visualisation only).
6) Render a two-color RGBA overlay (class 1 = goldenrod, class 2 = royal blue) and save as PDF.

Input/Output folders:
- This script expects an `Input_files/` folder next to the script.
- Output is written to `Output_files/` next to the script.
- If your data are inside a subfolder of `Input_files/`, set `input_relpath` accordingly.
"""

import os
import re
import math
import numpy as np
import skimage.io as io
from skimage import measure
from scipy.ndimage import map_coordinates, median_filter

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm


def normalize_array(x, gamma=1.0, vmin=None, vmax=None):
    """
    Normalize an array to [0, 1] with optional gamma correction.
    """
    x = np.asarray(x, dtype=np.float32)
    if vmin is None:
        vmin = np.nanmin(x)
    if vmax is None:
        vmax = np.nanmax(x)
    x = (x - vmin) / (vmax - vmin + 1e-16)
    x = np.clip(x, 0.0, 1.0)
    # Gamma correction (gamma>1 -> boosts darks less, gamma<1 -> boosts darks more)
    x = np.clip(x ** (1.0 / float(gamma)), 0.0, 1.0)
    return x


def mono_colormap(color_hex, name="mono"):
    """
    Create a simple black->color colormap.
    """
    return LinearSegmentedColormap.from_list(name, [(0, "black"), (1, color_hex)], N=256)


def apply_cmap(x01, cmap):
    """
    Apply a matplotlib colormap to a [0,1]-normalized array and return RGB (drop alpha).
    """
    rgba = cmap(x01)      # (...,4)
    return rgba[..., :3]  # (...,3)


def colony_parameters(array: np.ndarray) -> tuple[float, float]:
    """
    Find the centroid (row, col) of the largest connected component in a label image.

    Notes:
    - Values equal to 2 are treated as 1 (merged class for centroid finding).
    - Background is assumed to be label value 3 in the `measure.label` call.
    """
    array = np.where(array == 2, 1, array)
    labeled_colony = measure.label(array, background=3)
    components = measure.regionprops(labeled_colony)
    biggest_component = max(components, key=lambda x: x.area)
    return biggest_component.centroid


def get_number(filename: str) -> int:
    """
    Extract a numerical identifier from a filename (assumes pattern ending with '.tiff').

    Used for sorting timepoints.
    """
    match = re.search(r"(\d+)\.tiff$", filename)
    return int(match.group(1)) if match else 0


def calculate_endpoint(
    image_width: int,
    image_height: int,
    x: float,
    y: float,
    angle_degrees: float,
) -> tuple[float, float]:
    """
    Compute the intersection point (end_x, end_y) of a ray starting at (x,y) with direction
    given by angle_degrees, intersecting the image borders.

    Coordinates:
    - x corresponds to column, y corresponds to row (image coordinate convention).
    """
    ang = math.radians(angle_degrees)
    dx, dy = math.sin(ang), -math.cos(ang)

    t_candidates = []

    # Intersections with top/bottom borders
    if dy != 0:
        t = -y / dy  # top (y=0)
        ex = x + t * dx
        if 0 <= ex <= image_width - 1:
            t_candidates.append(t)

        t = (image_height - 1 - y) / dy  # bottom (y=H-1)
        ex = x + t * dx
        if 0 <= ex <= image_width - 1:
            t_candidates.append(t)

    # Intersections with left/right borders
    if dx != 0:
        t = -x / dx  # left (x=0)
        ey = y + t * dy
        if 0 <= ey <= image_height - 1:
            t_candidates.append(t)

        t = (image_width - 1 - x) / dx  # right (x=W-1)
        ey = y + t * dy
        if 0 <= ey <= image_height - 1:
            t_candidates.append(t)

    # Take the smallest non-negative t (forward ray)
    t_forward = min([t for t in t_candidates if t >= 0], default=0.0)
    ex, ey = x + t_forward * dx, y + t_forward * dy
    return ex, ey


def truncate_diagrams(diagrams: list[np.ndarray]) -> list[np.ndarray]:
    """
    Truncate all kymographs to the shortest distance-axis length.
    """
    min_length = min(diagram.shape[1] for diagram in diagrams)
    return [diagram[:, :min_length] for diagram in diagrams]


def fixed_axes(fig_w, fig_h, ax_w, ax_h, left=0.35, bottom=0.30, dpi=300):
    """
    Create a figure with one axes of fixed physical size.

    All units are inches.
    """
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes(
        [
            left / fig_w,
            bottom / fig_h,
            ax_w / fig_w,
            ax_h / fig_h,
        ]
    )
    return fig, ax


def tiff_data_to_space_time(config: dict) -> None:
    """
    Process TIFF stacks into a radial kymograph and save as a PDF.

    Input folder:
      <script_dir>/Input_files/<input_relpath>

    Output folder:
      <script_dir>/Output_files
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    input_root = os.path.join(script_dir, "Input_files")
    output_root = os.path.join(script_dir, "Output_files")
    os.makedirs(output_root, exist_ok=True)

    # Data location: allow referencing subfolders inside Input_files/
    input_relpath = config.get("input_relpath", "")
    folder_path = os.path.join(input_root, input_relpath)

    # --- config ---
    dark_bg = config.get("dark_background", False)
    identifiers = config["identifiers"]
    tc = config["tc"]  # reference frame index for centroid
    angle = config["angle"]
    angle_range = config["angle_range"]
    angle_step = config["angle_step"]
    angle_check = config.get("angle_check", False)
    smoothing = config.get("smoothing", False)

    # Shift-compensation (one-time jump)
    shift_from = int(config.get("shift_from_frame", 10**9))  # default: never shift
    drow, dcol = config.get("shift_vector_rc", (0.0, 0.0))   # (Δrow, Δcol)
    if config.get("round_shift", True):
        drow, dcol = int(round(drow)), int(round(dcol))
    fill_value = config.get("oob_fill", np.nan)

    # Plot scaling
    scale_factor = config["scale_factor"]  # µm per pixel
    frame_to_hour = config.get("frame_to_hour", 0.5)
    dpi = int(config.get("dpi", 300))

    highlight_regions = config.get("highlight_regions", [])
    highlight_color = config.get("highlight_color", "#bfbfbf")
    highlight_alpha = config.get("highlight_alpha", 0.3)
    hline_positions = config.get("hline_positions", [])

    # Note: these names are kept as-is to avoid behavioural changes.
    # (The script uses `ylim_max` to set x-limits and `xlim_max` to set y-limits.)
    xlim_max = config.get("xlim_max", None)
    ylim_max = config.get("ylim_max", None)

    output_filename = config.get("output_filename", "radial_kymograph.pdf")
    output_path = os.path.join(output_root, output_filename)

    # Matplotlib style
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "pdf.fonttype": 42,  # embed TrueType
            "ps.fonttype": 42,
            "font.size": 6,
            "axes.titlesize": 6,
            "axes.labelsize": 6,
            "axes.linewidth": 0.5,
            "xtick.major.size": 2,
            "ytick.major.size": 2,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "xtick.major.pad": 2,
            "ytick.major.pad": 2,
            "axes.labelpad": 1,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
            "legend.frameon": False,
            "lines.linewidth": 1.0,
        }
    )

    angle_values = np.arange(angle - angle_range, angle + angle_range + angle_step, angle_step)

    for identifier in tqdm(identifiers, desc="Processing Identifiers"):
        # Load TIFF frames for this identifier
        try:
            all_files = os.listdir(folder_path)
        except FileNotFoundError:
            print(f"[ERROR] Input folder not found: {folder_path}")
            return

        tiff_files = [f for f in all_files if f.endswith(".tiff") and identifier in f]
        tiff_files.sort(key=get_number)

        if not tiff_files:
            print(f"[WARN] No TIFFs found for identifier '{identifier}' in {folder_path}")
            continue

        images = [io.imread(os.path.join(folder_path, file)) for file in tiff_files]
        combined_array = np.stack(images)  # shape: (T, H, W)
        T, H, W = combined_array.shape

        # Colony centroid from reference frame (float coordinates)
        centroid_rc = colony_parameters(combined_array[tc])  # (row, col)
        r0, c0 = float(centroid_rc[0]), float(centroid_rc[1])

        # Build kymographs across angles and combine
        space_time_diagrams = []

        for a in angle_values:
            # Endpoint (float) on image border (x=col, y=row)
            ex, ey = calculate_endpoint(W, H, c0, r0, a)
            r1, c1 = float(ey), float(ex)

            # Uniform sampling along line (1 px steps in arc-length)
            L = math.hypot(r1 - r0, c1 - c0)
            n_samples = int(math.floor(L / 1.0)) + 1
            rr_base = np.linspace(r0, r1, n_samples, dtype=np.float32)
            cc_base = np.linspace(c0, c1, n_samples, dtype=np.float32)

            # Optional diagnostic plot for the central angle
            if angle_check and np.isclose(a, angle, atol=angle_step / 2):
                style = "dark_background" if dark_bg else "default"
                with plt.style.context(style):
                    plt.figure(figsize=(6, 6), dpi=200)
                    plt.imshow(combined_array[tc], cmap="gray")
                    plt.plot(
                        [cc_base[0], cc_base[-1]],
                        [rr_base[0], rr_base[-1]],
                        "r-",
                        label="Pre-shift line",
                    )
                    plt.plot(
                        [cc_base[0] + dcol, cc_base[-1] + dcol],
                        [rr_base[0] + drow, rr_base[-1] + drow],
                        "c--",
                        label="Post-shift line",
                    )
                    plt.xlim(0, config["display_width"])
                    plt.ylim(config["display_height"], 0)
                    plt.title(f"{identifier}: line positions at frame {shift_from}")
                    plt.legend()
                    plt.tight_layout()
                    plt.show()

            # Build kymograph for this angle with per-frame shift compensation
            rows = []
            for t in range(T):
                if t >= shift_from:
                    rr = rr_base + float(drow)
                    cc = cc_base + float(dcol)
                else:
                    rr = rr_base
                    cc = cc_base

                row = map_coordinates(
                    combined_array[t],
                    [rr, cc],
                    order=1,          # bilinear interpolation
                    mode="constant",
                    cval=fill_value,
                )
                rows.append(row)

            kymo = np.stack(rows, axis=0)  # (T, n_samples)
            space_time_diagrams.append(kymo)

        # Truncate to shortest sampled distance and combine (NaN-aware)
        space_time_diagrams = truncate_diagrams(space_time_diagrams)
        averaged_space_time_diagram = np.nanmin(space_time_diagrams, axis=0)  # (T, L')

        # Optional smoothing (visualisation only)
        if smoothing:
            averaged_space_time_diagram = median_filter(averaged_space_time_diagram, size=(3, 1))

        # Axes scaling
        kymo_T = averaged_space_time_diagram.T  # (distance, time)
        max_x_pixels = kymo_T.shape[0]
        max_y_frames = kymo_T.shape[1]
        print(kymo_T.shape)

        x_hours = np.linspace(0, max_y_frames * frame_to_hour, max_y_frames + 1)
        y_um = np.linspace(0, max_x_pixels * scale_factor, max_x_pixels + 1)
        y_mm = y_um / 1000.0

        L = config.get("fixed_layout", None)
        if L:
            fig, ax = fixed_axes(
                fig_w=L["fig_w"],
                fig_h=L["fig_h"],
                ax_w=L["ax_w"],
                ax_h=L["ax_h"],
                left=L.get("left", 0.35),
                bottom=L.get("bottom", 0.30),
                dpi=dpi,
            )
        else:
            figsize = config.get("figsize", (5, 1.5))
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Highlight regions (provided in frames, converted to hours)
        for region in highlight_regions:
            t0, t1 = np.array(region) * frame_to_hour
            ax.axvspan(
                t0 - 0.5,
                t1 - 0.5,
                color=highlight_color,
                alpha=highlight_alpha,
                zorder=0,
                linewidth=0,
            )

        extent = [0, x_hours[-1], 0, y_mm[-1]]  # [xmin, xmax, ymin, ymax]
        k = np.asarray(kymo_T)

        goldenrod = "#DAA520"
        royalblue = "#4169E1"

        mask_gold = (k == 1).astype(np.float32)
        mask_blue = (k == 2).astype(np.float32)

        gold_norm = normalize_array(mask_gold, gamma=1.0, vmin=0, vmax=1)
        blue_norm = normalize_array(mask_blue, gamma=1.0, vmin=0, vmax=1)

        gold_cmap = mono_colormap(goldenrod, "gold")
        blue_cmap = mono_colormap(royalblue, "blue")

        gold_rgb = apply_cmap(gold_norm, gold_cmap)
        blue_rgb = apply_cmap(blue_norm, blue_cmap)

        alpha_gold = gold_norm[..., None]
        alpha_blue = blue_norm[..., None]

        # "Over" compositing: gold over blue
        rgb_add = gold_rgb * alpha_gold + blue_rgb * alpha_blue * (1 - alpha_gold)

        # Transparent background for classes 0 and 3; opaque where gold or blue exist
        alpha = np.where((k == 1) | (k == 2), 1.0, 0.0).astype(np.float32)
        rgba_img = np.dstack([rgb_add, alpha])

        ax.imshow(
            rgba_img,
            origin="lower",
            aspect="auto",
            extent=extent,
            interpolation="none",
        )

        ax.set_xlabel("Time (h)", labelpad=2)
        ax.set_ylabel("Distance to center (mm)", labelpad=5)

        # Tick configuration
        x_tick_interval = 25
        x_ticks = np.arange(0, x_hours[-1] + x_tick_interval, x_tick_interval)
        ax.set_xticks(x_ticks)

        y_tick_interval = 1
        y_ticks = np.arange(0, y_mm[-1] + y_tick_interval, y_tick_interval)
        ax.set_yticks(y_ticks)

        # X-axis: time (hours)
        if xlim_max is not None:
            ax.set_xlim(0, float(xlim_max))
        else:
            ax.set_xlim(0, x_hours[-1])

        # Y-axis: distance (mm)
        if ylim_max is not None:
            ax.set_ylim(0, float(ylim_max))
        else:
            ax.set_ylim(0, y_mm[-1])

        for h in hline_positions:
            ax.axvline(x=h, color="black", linestyle=(0, (3, 5, 1, 5)), linewidth=1)

        ax.set_title("")
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        # Save output (overwrites if multiple identifiers are processed with one filename)
        fig.savefig(output_path, format="pdf", bbox_inches=None, transparent=True, dpi=300)

        plt.show()


def main():
    config = {
        # Input subfolder inside Input_files/ (leave "" if TIFFs are directly in Input_files/)
        "input_relpath": "11.5h_21h",

        # Output PDF filename inside Output_folder/
        "output_filename": "Fig1_panel_h.pdf",

        # Identifiers to process
        "identifiers": ["P13"],

        # Kymograph parameters
        "tc": 100,
        "angle": 46, #!!! or 104 for the lower plot in the panel !!!
        "angle_range": 2,
        "angle_step": 0.5,
        "angle_check": True,
        "smoothing": True,

        # Plot / scaling
        "scale_factor": 8.648,   # µm per pixel
        "frame_to_hour": 0.5,
        "figsize": (5.9, 1.5),
        "dpi": 300,
        "display_width": 1376,
        "display_height": 1104,

        # NOTE: naming kept as in original code (see note near set_xlim/set_ylim)
        "ylim_max": 4,
        "xlim_max": 70,

        # Highlight regions (in frames)
        "highlight_regions": [(37, 60), (102, 125), (167, 190), (232, 255)],
        "highlight_color": "#bfbfbf",
        "highlight_alpha": 1,

        "hline_positions": [1, 12.5, 25, 37.5, 50, 62.5],

        "dark_background": False,

        # One-time FOV jump compensation
        "shift_from_frame": 47,  # frame index at which the jump occurs (0-based)
        "shift_vector_rc": (-12.72, -16.69),  # (Δrow, Δcol) = (new_old_row - old_row, new_col - old_col)
        "round_shift": True,
        "oob_fill": np.nan,
    }

    tiff_data_to_space_time(config)


if __name__ == "__main__":
    main()