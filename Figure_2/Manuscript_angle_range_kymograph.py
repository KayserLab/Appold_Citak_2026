import os
import re
import math
import ast
import numpy as np
import h5py
import skimage.io as io
from skimage import measure
from skimage.draw import line
from scipy.ndimage import rotate
from scipy.spatial.distance import cdist
from scipy.ndimage import map_coordinates

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib as mpl
from tqdm import tqdm
from scipy.ndimage import median_filter
import shutil
from matplotlib.colors import LinearSegmentedColormap

def normalize_array(x, gamma=1.0, vmin=None, vmax=None):
    x = np.asarray(x, dtype=np.float32)
    if vmin is None:
        vmin = np.nanmin(x)
    if vmax is None:
        vmax = np.nanmax(x)
    x = (x - vmin) / (vmax - vmin + 1e-16)
    x = np.clip(x, 0.0, 1.0)
    # gamma correction (gamma>1 -> boosts darks less, gamma<1 -> boosts darks more)
    x = np.clip(x ** (1.0 / float(gamma)), 0.0, 1.0)
    return x

def mono_colormap(color_hex, name='mono'):
    return LinearSegmentedColormap.from_list(name, [(0, 'black'), (1, color_hex)], N=256)

def apply_cmap(x01, cmap):
    rgba = cmap(x01)              # (...,4)
    return rgba[..., :3]          # (...,3)

def colony_parameters(array: np.ndarray) -> tuple[float, float]:
    """
    Finds the largest connected component in a binary image and returns its centroid.

    Parameters:
    -----------
    array : np.ndarray
        Binary image.

    Returns:
    --------
    tuple[float, float]
        Centroid of the largest connected component.
    """
    array = np.where(array == 2, 1, array)
    labeled_colony = measure.label(array, background=3)
    components = measure.regionprops(labeled_colony)
    biggest_component = max(components, key=lambda x: x.area)
    return biggest_component.centroid


def get_number(filename: str) -> int:
    """
    Extracts a numerical identifier from a filename (assumes pattern ending with '.tiff').
    """
    match = re.search(r'(\d+)\.tiff$', filename)
    return int(match.group(1)) if match else 0


def calculate_endpoint(image_width: int, image_height: int, x: float, y: float, angle_degrees: float) -> tuple[float, float]:
    ang = math.radians(angle_degrees)
    dx, dy = math.sin(ang), -math.cos(ang)
    t_candidates = []

    # Intersections with the 4 borders (in continuous coordinates)
    if dy != 0:
        t = -y / dy                 # top (y=0)
        ex, ey = x + t*dx, 0.0
        if 0 <= ex <= image_width - 1: t_candidates.append(t)
        t = (image_height - 1 - y) / dy  # bottom (y=H-1)
        ex, ey = x + t*dx, image_height - 1.0
        if 0 <= ex <= image_width - 1: t_candidates.append(t)
    if dx != 0:
        t = -x / dx                 # left (x=0)
        ex, ey = 0.0, y + t*dy
        if 0 <= ey <= image_height - 1: t_candidates.append(t)
        t = (image_width - 1 - x) / dx   # right (x=W-1)
        ex, ey = image_width - 1.0, y + t*dy
        if 0 <= ey <= image_height - 1: t_candidates.append(t)

    # Take the smallest positive t (forward ray)
    t_forward = min([t for t in t_candidates if t >= 0], default=0.0)
    ex, ey = x + t_forward*dx, y + t_forward*dy
    return ex, ey

def make_line_coords(r0, c0, r1, c1, step=1.0):
    # build evenly-spaced samples along the true length
    L = math.hypot(r1 - r0, c1 - c0)
    n = int(math.floor(L / step)) + 1
    rr = np.linspace(r0, r1, n, dtype=np.float32)
    cc = np.linspace(c0, c1, n, dtype=np.float32)
    return rr, cc, L

def sample_frame_along_line_interp(frame_img, rr, cc, drow=0.0, dcol=0.0, fill_value=np.nan):
    rr_s = rr + drow
    cc_s = cc + dcol
    # sample with bilinear interpolation; outside = fill_value
    prof = map_coordinates(frame_img, [rr_s, cc_s], order=1, mode='constant', cval=fill_value)
    return prof

def truncate_diagrams(diagrams: list[np.ndarray]) -> list[np.ndarray]:
    """
    Truncates all diagrams to the length of the shortest one.
    """
    min_length = min(diagram.shape[1] for diagram in diagrams)
    return [diagram[:, :min_length] for diagram in diagrams]

def fixed_axes(fig_w, fig_h, ax_w, ax_h, left=0.35, bottom=0.30, dpi=300):
    """
    One axes with FIXED physical size (ax_w x ax_h in inches) inside a figure (fig_w x fig_h inches).
    left/bottom are margins in inches from the figure edge.
    """
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([
        left / fig_w,
        bottom / fig_h,
        ax_w / fig_w,
        ax_h / fig_h
    ])
    return fig, ax

def tiff_data_to_space_time(config: dict) -> None:
    """
    Processes TIFF images into a space-time (kymograph) diagram. It computes the space-time
    diagram along a line at a specified angle (or range of angles) from a colony centroid,
    then displays/saves the averaged diagram. Supports compensating a one-time FOV jump by
    shifting the sampling line from a given frame onward.

    Assumes helper functions exist in the file:
      - colony_parameters(array) -> (row, col) centroid of largest component
      - get_number(filename) -> numeric sorter for TIFF names
      - calculate_endpoint(W, H, x, y, angle_degrees) -> (end_x, end_y)
      - truncate_diagrams(list_of_ndarrays) -> list with equalized width
    """
    # Normalize folder path
    folder_path = config["folder_path"].replace('\\', '/')
    if not folder_path.endswith('/'):
        folder_path += '/'

    # --- config ---
    dark_bg        = config.get("dark_background", False)
    identifiers    = config["identifiers"]
    tc             = config["tc"]                 # timepoint (frame index) to use for colony contour
    angle          = config["angle"]
    angle_range    = config["angle_range"]
    angle_step     = config["angle_step"]
    angle_check    = config.get("angle_check", False)
    smoothing      = config.get("smoothing", False)

    # shift-compensation (one-time jump) ---
    shift_from     = int(config.get("shift_from_frame", 10**9))    # default: never shift
    drow, dcol     = config.get("shift_vector_rc", (0.0, 0.0))     # (Δrow, Δcol)
    if config.get("round_shift", True):
        drow, dcol = int(round(drow)), int(round(dcol))
    fill_value     = config.get("oob_fill", np.nan)

    # plotting/scaling ---
    scale_factor   = config["scale_factor"]       # µm per pixel
    frame_to_hour  = config.get("frame_to_hour", 0.5)
    figsize        = config.get("figsize", (5, 1.5))
    dpi            = config.get("dpi", 300)
    xlim_max       = config.get("xlim_max", None)
    ylim_max       = config.get("ylim_max", None)
    highlight_regions = config.get("highlight_regions", [])
    highlight_color   = config.get("highlight_color", "#bfbfbf")
    highlight_alpha   = config.get("highlight_alpha", 0.3)
    hline_positions   = config.get("hline_positions", [])
    diagram_title     = config.get("diagram_title", f"Radial Kymograph at {angle}°")

    # Matplotlib style
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'pdf.fonttype': 42,  # embed TrueType
        'ps.fonttype': 42,
        'font.size': 7,
        'axes.titlesize': 7,
        'axes.labelsize': 7,
        'axes.linewidth': 0.5,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'xtick.major.width': 1,
        'ytick.major.width': 1,
        'xtick.minor.width': 1,
        'ytick.minor.width': 1,
        'xtick.major.pad': 2,
        'ytick.major.pad': 2,
        'axes.labelpad': 1,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.labelsize': 6,  # tick‐label sizes
        'ytick.labelsize': 6,
        'legend.frameon': False,
        'lines.linewidth': 1.0,
    })

    angle_values = np.arange(angle - angle_range, angle + angle_range + angle_step, angle_step)

    for identifier in tqdm(identifiers, desc="Processing Identifiers"):
        # --- load images stack ---
        tiff_files = [f for f in os.listdir(folder_path) if f.endswith('.tiff') and identifier in f]
        tiff_files.sort(key=get_number)
        if not tiff_files:
            print(f"[WARN] No TIFFs found for identifier '{identifier}' in {folder_path}")
            continue

        images = [io.imread(os.path.join(folder_path, file)) for file in tiff_files]
        combined_array = np.stack(images)  # shape: (T, H, W)
        T, H, W = combined_array.shape

        #Rotate images for distance control degrees counter-clockwise
        #combined_array = np.array([rotate(frame, angle=45, reshape=False) for frame in combined_array])


        # --- colony centroid (use floats, no int cast) ---
        centroid_rc = colony_parameters(combined_array[tc])  # (row, col) floats
        r0, c0 = float(centroid_rc[0]), float(centroid_rc[1])

        # ────────────────────────────────────────────────────────────────
        # Angle iteration with interpolation-based, uniform arc-length sampling
        # ────────────────────────────────────────────────────────────────
        space_time_diagrams = []

        for a in angle_values:
            # continuous endpoint (float) on the image border
            ex, ey = calculate_endpoint(W, H, c0, r0, a)  # note: args = (image_width, image_height, x, y, angle)
            r1, c1 = float(ey), float(ex)

            # build evenly spaced coordinates along the true line length (step = 1 px)
            L = math.hypot(r1 - r0, c1 - c0)
            n_samples = int(math.floor(L / 1.0)) + 1
            rr_base = np.linspace(r0, r1, n_samples, dtype=np.float32)
            cc_base = np.linspace(c0, c1, n_samples, dtype=np.float32)

            # optional sanity check image for this angle
            if angle_check and np.isclose(a, angle, atol=angle_step / 2):
                style = 'dark_background' if dark_bg else 'default'
                with plt.style.context(style):
                    plt.figure(figsize=(6, 6), dpi=200)
                    plt.imshow(combined_array[tc], cmap='gray')
                    # pre-shift line preview
                    plt.plot([cc_base[0], cc_base[-1]], [rr_base[0], rr_base[-1]],
                             'r-', label='Pre-shift line')
                    # post-shift line preview
                    plt.plot([cc_base[0] + dcol, cc_base[-1] + dcol],
                             [rr_base[0] + drow, rr_base[-1] + drow],
                             'c--', label='Post-shift line')
                    plt.xlim(0, config["display_width"])
                    plt.ylim(config["display_height"], 0)
                    plt.title(f'{identifier}: line positions at frame {shift_from}')
                    plt.legend()
                    plt.tight_layout()
                    plt.show()

            # build kymograph for this angle with per-frame shift compensation
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
                    order=1,  # bilinear
                    mode='constant',
                    cval=fill_value
                )
                rows.append(row)

            kymo = np.stack(rows, axis=0)  # shape: (T, n_samples)
            space_time_diagrams.append(kymo)

        # --- truncate to shortest length and combine robustly (handle NaNs) ---
        space_time_diagrams = truncate_diagrams(space_time_diagrams)  # list of (T, L')
        averaged_space_time_diagram = np.nanmin(space_time_diagrams, axis=0)  # (T, L')

        # Optional smoothing
        if smoothing:
            averaged_space_time_diagram = median_filter(averaged_space_time_diagram, size=(3, 1))



        # 2) Axes scaling
        kymo_T = averaged_space_time_diagram.T           # shape: (L', T)
        max_x_pixels = kymo_T.shape[0]
        max_y_frames = kymo_T.shape[1]
        print(kymo_T.shape)

        x_hours = np.linspace(0, max_y_frames * frame_to_hour, max_y_frames + 1)  # time axis
        y_um    = np.linspace(0, max_x_pixels * scale_factor, max_x_pixels + 1)   # space axis
        # Convert to mm
        y_mm = y_um / 1000.0

        dpi = int(config.get("dpi", 300))
        L = config.get("fixed_layout", None)

        if L:
            fig, ax = fixed_axes(
                fig_w=L["fig_w"], fig_h=L["fig_h"],
                ax_w=L["ax_w"], ax_h=L["ax_h"],
                left=L.get("left", 0.35),
                bottom=L.get("bottom", 0.30),
                dpi=dpi
            )
        else:
            figsize = config.get("figsize", (5, 1.5))
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Highlighted regions (given in frames, converted to hours)
        for region in highlight_regions:
            t0, t1 = np.array(region) * frame_to_hour
            ax.axvspan(t0-0.5, t1-0.5, color=highlight_color, alpha=highlight_alpha, zorder=0, linewidth=0)

        # kymo_T: shape (L, T) where rows=distance, cols=time
        # extent maps pixel coords -> your physical axes
        extent = [0, x_hours[-1], 0, y_mm[-1]]  # [xmin, xmax, ymin, ymax]

        # kymo_T is (distance, time) with discrete labels 0..3 (as in your BoundaryNorm setup)
        k = np.asarray(kymo_T)

        goldenrod = '#DAA520'
        royalblue = '#4169E1'

        # Build per-class masks (float 0/1)
        mask_gold = (k == 1).astype(np.float32)  # goldenrod class
        mask_blue = (k == 2).astype(np.float32)  # royalblue class

        # Optional: soften edges slightly if you want, otherwise keep crisp
        # (comment out if you want strictly pixel-perfect class blocks)
        # from scipy.ndimage import gaussian_filter
        # mask_gold = gaussian_filter(mask_gold, sigma=0.0)
        # mask_blue = gaussian_filter(mask_blue, sigma=0.0)

        # Normalize (mostly relevant if you later use graded intensities; here it's 0/1)
        gold_norm = normalize_array(mask_gold, gamma=1.0, vmin=0, vmax=1)
        blue_norm = normalize_array(mask_blue, gamma=1.0, vmin=0, vmax=1)

        gold_cmap = mono_colormap(goldenrod, 'gold')
        blue_cmap = mono_colormap(royalblue, 'blue')

        gold_rgb = apply_cmap(gold_norm, gold_cmap)
        blue_rgb = apply_cmap(blue_norm, blue_cmap)

        alpha_gold = gold_norm[..., None]
        alpha_blue = blue_norm[..., None]

        # "Over" compositing: gold over blue (or swap if you prefer)
        rgb_add = gold_rgb * alpha_gold + blue_rgb * alpha_blue * (1 - alpha_gold)

        # Transparent background for classes 0 and 3, opaque where gold or blue exist
        alpha = np.where((k == 1) | (k == 2), 1.0, 0.0).astype(np.float32)

        rgba_img = np.dstack([rgb_add, alpha])

        im = ax.imshow(
            rgba_img,
            origin='lower',
            aspect='auto',
            extent=extent,
            interpolation='none',  # keep crisp pixels
        )

        ax.set_xlabel('Time (h)', labelpad=2)
        ax.set_ylabel('Distance to center (mm)', labelpad=5)

        # Have x labels every 5 hours
        x_tick_interval = 5
        x_ticks = np.arange(0, x_hours[-1] + x_tick_interval, x_tick_interval)
        ax.set_xticks(x_ticks)

        #Have y labels every 0.5 mm
        y_tick_interval = 1
        y_ticks = np.arange(0, y_mm[-1] + y_tick_interval, y_tick_interval)
        ax.set_yticks(y_ticks)

        # x axis is time (hours)
        if config.get("ylim_max") is not None:
            ax.set_xlim(0, float(config["ylim_max"]))
        else:
            ax.set_xlim(0, x_hours[-1])

        # y axis is distance (mm)
        if config.get("xlim_max") is not None:
            ax.set_ylim(0, float(config["xlim_max"]))
        else:
            ax.set_ylim(0, y_mm[-1])

        for h in hline_positions:
            ax.axvline(x=h, color='black', linestyle=(0, (3, 5, 1, 5)), linewidth=1)

        ax.set_title("")
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        #plt.tight_layout(pad=0.2)

        # --- save ---
        out_path = config.get("output_path")
        if out_path:
            fig.savefig(out_path, format='pdf', bbox_inches=None, transparent=True, dpi=300)

            # Save a copy of the current script alongside (best-effort)
            code_path = os.path.splitext(out_path)[0] + "_code.txt"
            try:
                script_file = os.path.abspath(__file__)
                shutil.copy(script_file, code_path)
                print(f"Code copied to {code_path}")
            except NameError:
                print("No __file__ available (likely running in REPL/Notebook). Save code manually.")

        plt.show()


def main():
    config = {
        # File settings
        "folder_path": r"D:\Image_Segmentation\20251024_no_treatment",
        "output_path": r"Z:\Members\Nico\Manuscript_Figures\For Supplement\NT_colony_pictures\NT_P13_radial_kymograph_45_big.pdf",
        "identifiers": ['P13'],  # List of identifiers to process
        "tc": 100,  # Timepoint (frame index) for colony contour
        "angle": 45,  # Central angle for line extraction
        "angle_range": 2,  # ± range around central angle (in degrees)
        "angle_step": 0.5,  # Angle step size (degrees)
        "angle_check": True,  # Whether to display a sample image with the line
        "smoothing": True,  # Whether to apply median filter for smoothing

        # Plot display settings
        "scale_factor": 8.648,  # Microns per pixel for x-axis scaling
        "frame_to_hour": 0.5,  # Conversion factor from frame (timepoint) to hours
        "figsize": (1.65, 1.1),
        "dpi": 300,
        "display_width": 1376,  # Width (pixels) for sample image display
        "display_height": 1104,  # Height (pixels) for sample image display
        "xlim_max": 5,  # Maximum x-axis value in microns for final plot
        "ylim_max": 85,  # Maximum y-axis value (hours) for final plot

        # Highlight regions: list of tuples defined in timeframes (frames)
        # They will be converted to hours using the frame_to_hour factor.
        "highlight_regions": [],  # in frames
        "highlight_color": "#bfbfbf",
        "highlight_alpha": 1,

        # Optional horizontal lines (in hours) to mark special timepoints
        "hline_positions": [1, 12.5, 25, 37.5, 50, 62.5,75],  # in hours

        "fixed_layout": {
            "fig_w": 7.2,
            "fig_h": 1.5,
            "ax_w": 6.7,
            "ax_h": 1.1,
            "left": 0.35,
            "bottom": 0.30,
        },
        # Diagram title with automatic angle
        "diagram_title": "Radial Kymograph in Adaptive Treatment at 18°",

        # Toggle dark background
        "dark_background": False,

        # Compensate a one-time FOV jump
        "shift_from_frame": 0,  # first frame that is displaced (0-based)
        "shift_vector_rc": (0, 0),  # (Δrow, Δcol) = (new_old_row - old_row, new_col - old_col)
        "round_shift": True,  # snap to pixel grid
        "oob_fill": np.nan,  # out-of-bounds fill when the shifted line leaves the image
    }

    tiff_data_to_space_time(config)

if __name__ == '__main__':
    main()