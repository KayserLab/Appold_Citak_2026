#!/usr/bin/env python3
"""
Radial kymograph generator (multi-identifier), to detect the growing edge where no mutants are present.

For each identifier:
- loads all TIFFs that contain the strict pattern "_<identifier>_" (avoids P1 vs P10 ambiguity)
- computes the colony centroid from frame `tc`
- asks the user for a central angle (degrees) (or uses `default_angle` if Enter is pressed)
- samples along a radial line from the centroid to the image border over a small angle range
- combines the per-angle kymographs via nan-min
- optionally applies a median filter for visual smoothing
- computes, for each frame, the farthest non-background point along the sampled line
  and stores this distance (mm) as a time series
- writes:
    Output_files/<output_prefix>_<identifier>_<angle>deg_kymograph.pdf
    Output_files/<output_prefix>_<identifier>_<angle>deg_max_distance.csv

Folder structure (next to this script):
- Input_files/
    - <input_relpath>/     (segmented TIFFs)
- Output_files/
"""

from __future__ import annotations

import os
import re
import math

import numpy as np
import pandas as pd
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl

from skimage import measure
from scipy.ndimage import median_filter, map_coordinates
from tqdm import tqdm


def colony_centroid(array: np.ndarray) -> tuple[float, float]:
    """
    Return the centroid (row, col) of the largest connected component.

    Assumptions about labels:
    - colony pixels are 1 or 2 (2 is merged into 1)
    - background is 3
    """
    array = np.where(array == 2, 1, array)
    labeled = measure.label(array, background=3)
    components = measure.regionprops(labeled)
    biggest = max(components, key=lambda x: x.area)
    return float(biggest.centroid[0]), float(biggest.centroid[1])


def get_number(filename: str) -> int:
    """
    Extract trailing frame number from filenames ending in '<number>.tiff' for sorting.
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
    From starting point (x, y) and direction `angle_degrees`, compute where the forward ray
    first intersects the image border.

    Returns
    -------
    end_x, end_y : float
        Intersection point on the border.
    """
    ang = math.radians(angle_degrees)
    dx, dy = math.sin(ang), -math.cos(ang)
    t_candidates: list[float] = []

    if dy != 0:
        # top (y=0)
        t = -y / dy
        ex = x + t * dx
        if 0 <= ex <= image_width - 1:
            t_candidates.append(t)
        # bottom (y=H-1)
        t = (image_height - 1 - y) / dy
        ex = x + t * dx
        if 0 <= ex <= image_width - 1:
            t_candidates.append(t)

    if dx != 0:
        # left (x=0)
        t = -x / dx
        ey = y + t * dy
        if 0 <= ey <= image_height - 1:
            t_candidates.append(t)
        # right (x=W-1)
        t = (image_width - 1 - x) / dx
        ey = y + t * dy
        if 0 <= ey <= image_height - 1:
            t_candidates.append(t)

    t_forward = min([t for t in t_candidates if t >= 0], default=0.0)
    end_x = x + t_forward * dx
    end_y = y + t_forward * dy
    return float(end_x), float(end_y)


def truncate_diagrams(diagrams: list[np.ndarray]) -> list[np.ndarray]:
    """
    Truncate all diagrams to the shortest spatial length.
    """
    min_length = min(d.shape[1] for d in diagrams)
    return [d[:, :min_length] for d in diagrams]


def tiff_data_to_space_time(config: dict) -> None:
    """
    Build kymographs and max-distance time series for each identifier in `config["identifiers"]`.
    """
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    identifiers: list[str] = config["identifiers"]

    tc = int(config["tc"])
    angle_range = float(config["angle_range"])
    angle_step = float(config["angle_step"])
    default_angle = config.get("default_angle", None)

    angle_check = bool(config.get("angle_check", False))
    smoothing = bool(config.get("smoothing", False))

    shift_from = int(config.get("shift_from_frame", 10**9))
    drow, dcol = config.get("shift_vector_rc", (0.0, 0.0))
    if config.get("round_shift", True):
        drow, dcol = int(round(drow)), int(round(dcol))
    fill_value = config.get("oob_fill", np.nan)

    scale_factor = float(config["scale_factor"])         # µm / px
    frame_to_hour = float(config.get("frame_to_hour", 0.5))

    figsize = config.get("figsize", (1.65, 1.1))
    dpi = int(config.get("dpi", 300))
    xlim_max = config.get("xlim_max", None)              # time axis limit (hours)
    ylim_max = config.get("ylim_max", None)              # distance axis limit (mm)

    highlight_regions = config.get("highlight_regions", [])
    highlight_color = config.get("highlight_color", "#bfbfbf")
    highlight_alpha = float(config.get("highlight_alpha", 0.3))
    vline_positions = config.get("vline_positions", [])

    display_width = config.get("display_width", None)
    display_height = config.get("display_height", None)

    border_margin_px = int(config.get("border_margin_px", 2))
    min_cutoff_frame = int(config.get("min_cutoff_frame", 0))

    output_prefix = config.get("output_prefix", "radial_kymograph")

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 6,
            "axes.titlesize": 6,
            "axes.labelsize": 6,
            "axes.linewidth": 0.5,
            "xtick.major.size": 2,
            "ytick.major.size": 2,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
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

    # label colormap: 0 and 3 are transparent; 1 and 2 are colored
    colors_rgba = [
        (1, 1, 1, 0),                 # 0
        (0.855, 0.647, 0.125, 1),     # 1 (goldenrod)
        (0.254, 0.41, 0.882, 1),      # 2 (royal blue)
        (1, 1, 1, 0),                 # 3
    ]
    cmap = mcolors.ListedColormap(colors_rgba)
    norm = mcolors.BoundaryNorm([0, 1, 2, 3, 4], cmap.N, clip=True)

    for identifier in tqdm(identifiers, desc="Identifiers"):
        id_pattern = f"{identifier}_"
        tiff_files = [f for f in os.listdir(input_dir) if f.endswith(".tiff") and id_pattern in f]
        tiff_files.sort(key=get_number)

        if not tiff_files:
            continue

        images = [io.imread(os.path.join(input_dir, f)) for f in tiff_files]
        stack = np.stack(images)  # (T, H, W)
        T, H, W = stack.shape

        r0, c0 = colony_centroid(stack[tc])

        # Angle input per identifier (interactive)
        while True:
            angle_input = input(
                f"Central angle for '{identifier}' in degrees "
                f"(press Enter for default_angle): "
            ).strip()

            if angle_input == "":
                if default_angle is None:
                    continue
                central_angle = float(default_angle)
                break

            try:
                central_angle = float(angle_input)
                break
            except ValueError:
                continue

        angle_values = np.arange(
            central_angle - angle_range,
            central_angle + angle_range + angle_step,
            angle_step,
        )

        space_time_diagrams: list[np.ndarray] = []

        for a in angle_values:
            end_x, end_y = calculate_endpoint(W, H, c0, r0, a)
            r1, c1 = float(end_y), float(end_x)

            L = math.hypot(r1 - r0, c1 - c0)
            n_samples = int(math.floor(L / 1.0)) + 1
            rr_base = np.linspace(r0, r1, n_samples, dtype=np.float32)
            cc_base = np.linspace(c0, c1, n_samples, dtype=np.float32)

            if angle_check and np.isclose(a, central_angle, atol=angle_step / 2):
                plt.figure(figsize=(6, 6), dpi=200)
                plt.imshow(stack[tc], cmap="gray")
                plt.plot([cc_base[0], cc_base[-1]], [rr_base[0], rr_base[-1]], "r-", label="Pre-shift")
                plt.plot(
                    [cc_base[0] + dcol, cc_base[-1] + dcol],
                    [rr_base[0] + drow, rr_base[-1] + drow],
                    "c--",
                    label="Post-shift",
                )
                if display_width is not None:
                    plt.xlim(0, display_width)
                if display_height is not None:
                    plt.ylim(display_height, 0)
                plt.title(f"{identifier}: sampling line preview")
                plt.legend()
                plt.tight_layout()
                plt.show()

            rows = []
            for t in range(T):
                if t >= shift_from:
                    rr = rr_base + float(drow)
                    cc = cc_base + float(dcol)
                else:
                    rr = rr_base
                    cc = cc_base

                row = map_coordinates(
                    stack[t],
                    [rr, cc],
                    order=1,
                    mode="constant",
                    cval=fill_value,
                )
                rows.append(row)

            space_time_diagrams.append(np.stack(rows, axis=0))  # (T, L)

        space_time_diagrams = truncate_diagrams(space_time_diagrams)
        kymo = np.nanmin(space_time_diagrams, axis=0)  # (T, L)

        if smoothing:
            kymo = median_filter(kymo, size=(3, 1))

        # Plotting uses (distance, time)
        kymo_T = kymo.T  # (L, T)
        L_px = kymo_T.shape[0]
        n_frames = kymo_T.shape[1]

        y_mm = (np.arange(L_px + 1) * scale_factor) / 1000.0
        x_hours = np.linspace(0, n_frames * frame_to_hour, n_frames + 1)

        # Farthest valid point per frame (labels 1 or 2 only)
        farthest_indices: list[float] = []
        for t in range(n_frames):
            col = kymo_T[:, t]
            valid = (~np.isnan(col)) & (col != 0) & (col != 3)
            farthest_indices.append(float(np.max(np.where(valid)[0])) if np.any(valid) else np.nan)

        # Truncate in time if the detected front hits the image border
        border_idx = L_px - 1
        threshold_idx = border_idx - border_margin_px

        border_hits = [i for i, idx in enumerate(farthest_indices) if not np.isnan(idx) and idx >= threshold_idx]
        border_hits = [f for f in border_hits if f >= min_cutoff_frame]

        cutoff_frame = border_hits[0] if border_hits else (n_frames - 1)
        effective_frames = cutoff_frame + 1

        kymo_T = kymo_T[:, :effective_frames]
        farthest_indices = farthest_indices[:effective_frames]

        n_frames = kymo_T.shape[1]
        times_h = [t * frame_to_hour for t in range(n_frames)]
        max_distances_mm = [
            round((idx * scale_factor) / 1000.0, 3) if not np.isnan(idx) else np.nan
            for idx in farthest_indices
        ]

        df = pd.DataFrame(
            {
                "time_h": times_h,
                "max_distance_mm": max_distances_mm,
            }
        )

        # Plot
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        for region in highlight_regions:
            t0, t1 = np.array(region) * frame_to_hour
            ax.axvspan(t0, t1, color=highlight_color, alpha=highlight_alpha, zorder=0, linewidth=0)

        ax.pcolormesh(
            np.linspace(0, n_frames * frame_to_hour, n_frames + 1),
            y_mm,
            kymo_T,
            cmap=cmap,
            norm=norm,
            edgecolors="none",
            antialiased=False,
        )

        ax.plot(times_h, max_distances_mm, linestyle="-", linewidth=0.75, color="black", label="Detected front")

        border_distance_mm = (border_idx * scale_factor) / 1000.0
        ax.axhline(border_distance_mm, linestyle="--", linewidth=0.5, color="black", label="Image border")

        ax.set_xlabel("Time (h)", labelpad=2)
        ax.set_ylabel("Distance to center (mm)", labelpad=5)

        if xlim_max is not None:
            ax.set_xlim(0, float(xlim_max))
        else:
            ax.set_xlim(0, n_frames * frame_to_hour)

        if ylim_max is not None:
            ax.set_ylim(0, float(ylim_max))
        else:
            ax.set_ylim(0, y_mm[-1])

        for h in vline_positions:
            ax.axvline(x=h, color="black", linestyle=(0, (3, 5, 1, 5)), linewidth=1)

        ax.legend(loc="upper right", fontsize=5, frameon=False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        plt.tight_layout(pad=0.2)

        angle_tag = f"{int(round(central_angle))}deg"
        base_name = f"{output_prefix}_{identifier}_{angle_tag}"

        pdf_path = os.path.join(output_dir, base_name + "_kymograph.pdf")
        csv_path = os.path.join(output_dir, base_name + "_max_distance.csv")

        fig.savefig(pdf_path, format="pdf", bbox_inches=None, transparent=False, dpi=300)
        plt.close(fig)

        df.to_csv(csv_path, index=False)


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))

    input_root = os.path.join(script_dir, "Input_files")
    output_root = os.path.join(script_dir, "Output_files")

    config = {
        # Input folder inside Input_files/
        "input_dir": os.path.join(input_root, "Segmented_TIFFs", "Your_Experiment_folder"),  # <-- CHANGE THIS to your folder containing TIFF stacks (can have subfolders if needed)
        # Output is always your general Output_files/ folder
        "output_dir": output_root,

        # Your identifiers (USe P1_ for P1)
        "identifiers": ["P2", "P3", "P4", "P5", "P6", "P7", "P8", "P10", "P11", "P12", "P14", "P15", "P16"],

        # Frame index used to compute the centroid
        "tc": 100,

        # Default angle if user presses Enter
        "default_angle": 209,

        # Angle sweep
        "angle_range": 2,
        "angle_step": 0.5,

        # Optional previews / smoothing
        "angle_check": True,
        "smoothing": True,

        # Unit conversion
        "scale_factor": 8.648,   # µm / px
        "frame_to_hour": 0.5,

        # Figure settings
        "figsize": (1.65, 1.1),
        "dpi": 300,
        "xlim_max": 160,   # time axis (hours)
        "ylim_max": 6.5,   # distance axis (mm)

        # Highlight regions in frames (converted to hours)
        "highlight_regions": [[37, 65]],
        "highlight_color": "#bfbfbf",
        "highlight_alpha": 1.0,

        # Border-hit truncation
        "border_margin_px": 2,
        "min_cutoff_frame": 50,

        # Optional vertical marker lines (hours)
        "vline_positions": [],

        # Optional shift compensation
        "shift_from_frame": 10**9,   # default: no shift
        "shift_vector_rc": (0, 0),
        "round_shift": True,
        "oob_fill": np.nan,

        "output_prefix": "Example",
    }

    tiff_data_to_space_time(config)


if __name__ == "__main__":
    main()