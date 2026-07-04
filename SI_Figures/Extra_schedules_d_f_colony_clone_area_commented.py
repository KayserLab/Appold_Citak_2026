#!/usr/bin/env python3
"""
Plot clone-adjusted population trajectories and resistant fractions per condition.

For each active condition folder, the script:

- loads colony CSV files for the configured identifiers;
- uses ``extrapolated_clone_area`` when available and otherwise falls back to
  ``total_clone_area``;
- calculates the clone-adjusted total area as
  ``colony_area - total_clone_area + extrapolated_clone_area``;
- computes the per-frame median and interquartile range across colonies;
- converts frame indices to hours and areas from px² to mm²; and
- creates one total/resistant-area plot and one resistant-fraction plot per condition.

Input data are read from the project-relative ``Input_files`` directory. PDF
outputs are written to the project-relative ``Output_files`` directory.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# ────────────────────────── Project-relative IO ──────────────────────────
# Place all condition folders inside "Input_files" next to this script.
# Generated PDF files are written to "Output_files" next to this script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_ROOT = os.path.join(SCRIPT_DIR, "Input_files")
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "Output_files")

# ────────────────────────── Basic helpers ──────────────────────────


def _norm(p: str) -> str:
    """Consistent path normalization for config keys."""
    return os.path.normpath(p).replace("\\", "/").rstrip("/")


def set_nature_style():
    """Update Matplotlib rcParams to mimic a Nature-style layout."""
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 7,
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
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.labelsize': 5,
        'ytick.labelsize': 5,
        'axes.labelpad': 1,
        'xtick.major.pad': 2,
        'ytick.major.pad': 2,
        'legend.frameon': False,
        'legend.fontsize': 5,
        'lines.linewidth': 1.0,
        'savefig.dpi': 300,
        'figure.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def _default_color_palette():
    """
    Fallback-Palette: dunklere Farben aus tab20b.
    Wird verwendet, wenn für einen display_name keine Farbe definiert ist.
    """
    cmap = mpl.colormaps["tab20b"]
    idxs = [0, 4, 8, 12, 16]  # dunkle Töne
    return [mpl.colors.to_hex(cmap(i)) for i in idxs]


def _get_color_for_display_name(disp_name: str, config: dict, fallback_index: int = 0) -> str:
    """
    Hole eine definierte Farbe für display_name.
    Wenn nicht vorhanden, nimm eine Farbe aus der Fallback-Palette (tab20b-dark).
    """
    display_colors: dict = config.get("display_colors", {})
    if disp_name in display_colors:
        return display_colors[disp_name]

    palette = config.get("fallback_colors", None)
    if palette is None:
        palette = _default_color_palette()
    if not palette:
        return "black"

    return palette[fallback_index % len(palette)]

def _hours_from_index(series: pd.Series) -> pd.Series:
    """Convert index of frames to hours (2 frames = 1 hour)."""
    s = series.copy()
    s.index = s.index / 2.0
    return s


def _sanitize_for_filename(name: str) -> str:
    """Turn a display name into a filesystem-friendly chunk."""
    safe = re.sub(r"[^A-Za-z0-9]+", "_", name)
    return safe.strip("_") or "condition"


def _prep_display_name_map(raw_map: dict) -> dict:
    """Normalize keys in display_names to match normalized folder paths."""
    return {_norm(k): v for k, v in raw_map.items()}


def _prep_span_map(raw_map: dict) -> dict:
    """Normalize keys in highlight_spans to match normalized folder paths."""
    return {_norm(k): v for k, v in raw_map.items()}


def _safe_display_name(path: str, display_names: dict) -> str:
    """
    Resolve a human-readable name for a folder:

      - first: use normalized full path lookup in display_names
      - fallback: basename with stripped date prefix
    """
    norm = _norm(path)
    if norm in display_names:
        return display_names[norm]

    base = os.path.basename(norm)
    # e.g. "20240917_continuous_dose_2" -> "continuous_dose_2"
    name = re.sub(r"^\d{8}_", "", base)
    return name


# ────────────────────────── File handling ──────────────────────────

def find_colony_extrap_file(folder_path: str, identifier: str) -> str | None:
    """
    Find the colony CSV with extrapolated clone area for a given identifier.

    Priority:
      1) *with_extrapolation*
      2) *with_clonearea*
      3) any file containing 'colony' and identifier
    """
    files = os.listdir(folder_path)

    candidates = [
        f for f in files
        if f.endswith(".csv")
        and "colony" in f
        and identifier in f
        and "with_extrapolation" in f
    ]

    if not candidates:
        candidates = [
            f for f in files
            if f.endswith(".csv")
            and "colony" in f
            and identifier in f
            and "with_clonearea" in f
        ]

    if not candidates:
        candidates = [
            f for f in files
            if f.endswith(".csv")
            and "colony" in f
            and identifier in f
        ]

    if not candidates:
        return None

    #print number of files used
    print(f"[info] Found {len(candidates)} candidate files for identifier '{identifier}' in folder '{folder_path}'")

    candidates.sort()
    return os.path.join(folder_path, candidates[0])


def series_from_colony_file(colony_file: str):
    """
    Load colony CSV and return three aligned pd.Series indexed by frame (in px²):

      - colony_series_px2         : raw 'colony_area'
      - clone_series_px2          : 'extrapolated_clone_area' if present,
                                    otherwise 'total_clone_area'
      - adj_colony_series_px2     : colony_area - total_clone_area + clone_series_px2
    """
    df = pd.read_csv(colony_file)

    if "colony_area" not in df.columns:
        raise ValueError(f"'colony_area' column not found in {colony_file}")
    if "total_clone_area" not in df.columns:
        raise ValueError(f"'total_clone_area' column not found in {colony_file}")

    if "extrapolated_clone_area" in df.columns:
        clone_colname = "extrapolated_clone_area"
    else:
        clone_colname = "total_clone_area"

    idx = np.arange(len(df))
    colony_series_px2 = pd.Series(df["colony_area"].values, index=idx)
    clone_raw_px2 = pd.Series(df["total_clone_area"].values, index=idx)
    clone_series_px2 = pd.Series(df[clone_colname].values, index=idx)

    adj_colony_series_px2 = colony_series_px2 - clone_raw_px2 + clone_series_px2

    return colony_series_px2, clone_series_px2, adj_colony_series_px2


# ────────────────────────── Aggregation per folder ──────────────────────────

def aggregate_folder(folder_path: str, identifiers: list[str], config: dict):
    """
    Aggregate per-folder median and IQR across identifiers.

    Returns dict with (all in px² except fraction, index in HOURS):

      - clone_med_hr, clone_q1_hr, clone_q3_hr
      - adj_med_hr,   adj_q1_hr,   adj_q3_hr
      - frac_med_hr,  frac_q1_hr,  frac_q3_hr   # resistant fraction
    """
    folder = _norm(folder_path)

    clone_df = pd.DataFrame()
    adj_df = pd.DataFrame()

    for ident in identifiers:
        colony_file = find_colony_extrap_file(folder, ident)
        if colony_file is None:
            continue

        try:
            _, clone_s, adj_s = series_from_colony_file(colony_file)
        except Exception as e:
            print(f"[warn] {os.path.basename(folder)}: failed loading '{ident}': {e}")
            continue

        clone_df[ident] = clone_s
        adj_df[ident] = adj_s

    if clone_df.empty or adj_df.empty:
        print(f"[skip] No valid clone/adjusted data in folder: {folder}")
        return None

    # --- Fraction per identifier BEFORE aggregating ---
    # Avoid division by zero -> NaN where adj==0
    frac_df = clone_df.divide(adj_df.where(adj_df > 0), axis=0)

    # Median + IQR across identifiers for each frame
    def qstats(df_):
        med = df_.quantile(0.5, axis=1, interpolation="linear")
        q1  = df_.quantile(0.25, axis=1, interpolation="linear")
        q3  = df_.quantile(0.75, axis=1, interpolation="linear")
        return med, q1, q3

    clone_med, clone_q1, clone_q3 = qstats(clone_df)
    adj_med,   adj_q1,   adj_q3   = qstats(adj_df)
    frac_med,  frac_q1,  frac_q3  = qstats(frac_df)

    smooth_window = int(config.get("smooth_window", 0) or 0)

    def _smooth(s: pd.Series) -> pd.Series:
        return s.rolling(smooth_window, center=True, min_periods=1).median()

    if smooth_window > 1:
        clone_med = _smooth(clone_med)
        clone_q1  = _smooth(clone_q1)
        clone_q3  = _smooth(clone_q3)

        adj_med = _smooth(adj_med)
        adj_q1  = _smooth(adj_q1)
        adj_q3  = _smooth(adj_q3)

        frac_med = _smooth(frac_med)
        frac_q1  = _smooth(frac_q1)
        frac_q3  = _smooth(frac_q3)

    # optional x-limit in frames
    xmax_frames = config.get("xlim", None)
    if xmax_frames is not None:
        xmax_frames = int(xmax_frames)
        sl = slice(0, xmax_frames + 1)

        clone_med = clone_med.iloc[sl]
        clone_q1  = clone_q1.iloc[sl]
        clone_q3  = clone_q3.iloc[sl]

        adj_med = adj_med.iloc[sl]
        adj_q1  = adj_q1.iloc[sl]
        adj_q3  = adj_q3.iloc[sl]

        frac_med = frac_med.iloc[sl]
        frac_q1  = frac_q1.iloc[sl]
        frac_q3  = frac_q3.iloc[sl]

    return {
        "clone_med_hr": _hours_from_index(clone_med),
        "clone_q1_hr":  _hours_from_index(clone_q1),
        "clone_q3_hr":  _hours_from_index(clone_q3),

        "adj_med_hr":   _hours_from_index(adj_med),
        "adj_q1_hr":    _hours_from_index(adj_q1),
        "adj_q3_hr":    _hours_from_index(adj_q3),

        "frac_med_hr":  _hours_from_index(frac_med),
        "frac_q1_hr":   _hours_from_index(frac_q1),
        "frac_q3_hr":   _hours_from_index(frac_q3),
    }

def fixed_axes(fig_w, fig_h, ax_w, ax_h, left=0.25, bottom=0.22):
    """
    Create a figure with ONE axes of FIXED physical size (ax_w x ax_h in inches),
    placed at (left, bottom) inches from the lower-left corner of the figure.

    Everything else (labels, ticks, title) is drawn around it and may extend
    beyond the figure if too large.
    """
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([
        left / fig_w,
        bottom / fig_h,
        ax_w / fig_w,
        ax_h / fig_h
    ])
    return fig, ax

def plot_condition_fraction(
    folder_path: str,
    identifiers: list[str],
    display_names: dict,
    highlight_spans: dict,
    config: dict,
    color_index: int = 0,
):
    """
    Plot resistant fraction over time for one condition:

      - solid line + IQR: resistant fraction (clone / adjusted total)
      - grey shaded background for highlight_spans
    """
    folder_norm = _norm(folder_path)
    disp_name = _safe_display_name(folder_norm, display_names)

    data = aggregate_folder(folder_norm, identifiers, config)
    if data is None:
        return

    frac_med_hr = data["frac_med_hr"]
    frac_q1_hr  = data["frac_q1_hr"]
    frac_q3_hr  = data["frac_q3_hr"]

    # Align and drop empty overlap
    idx = frac_med_hr.index
    if idx.empty:
        print(f"[warn] No fraction data for {disp_name}")
        return

    x_hours = idx.values
    frac_med = frac_med_hr.values
    frac_q1  = frac_q1_hr.values
    frac_q3  = frac_q3_hr.values

    color = _get_color_for_display_name(disp_name, config, color_index)

    figsize = config.get("figsize", (3.5, 2.5))
    dpi = int(config.get("dpi", 300))
    L = config.get("fixed_layout", None)

    if L:
        fig, ax = fixed_axes(
            fig_w=L["fig_w"], fig_h=L["fig_h"],
            ax_w=L["ax_w"], ax_h=L["ax_h"],
            left=L.get("left", 0.25),
            bottom=L.get("bottom", 0.22),
        )
    else:
        figsize = config.get("figsize", (3.5, 2.5))
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    fig.set_dpi(dpi)

    # Grey highlights (from highlight_spans in FRAMES)
    spans = highlight_spans.get(folder_norm, []) or []
    highlight_alpha = float(config.get("highlight_alpha", 0.4))
    for (start_frame, end_frame) in spans:
        s_hr = start_frame / 2.0
        e_hr = end_frame / 2.0
        if e_hr <= s_hr:
            continue
        ax.axvspan(
            s_hr - 0.5, e_hr - 0.5,
            color="#bfbfbf",
            alpha=highlight_alpha,
            zorder=0,
            linewidth=0,
        )

    iqr_alpha = float(config.get("iqr_alpha", 0.25))
    line_width = float(config.get("line_width", 1.8))

    ax.fill_between(
        x_hours,
        frac_q1,
        frac_q3,
        color=color,
        alpha=iqr_alpha,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        x_hours,
        frac_med,
        color=color,
        linewidth=line_width,
        linestyle="dotted",
        label="Resistant fraction",
        zorder=2,
    )

    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Resistant / Total")
    ax.set_ylim(0, 1.0)

    # --- x ticks every 25 hours ---
    if config.get("xlim") is not None:
        xmax_h = float(config["xlim"]) / 2.0
    else:
        xmax_h = float(np.nanmax(x_hours))

    tick_step = 25
    ticks = np.arange(0, xmax_h + 1e-9, tick_step)
    ax.set_xticks(ticks)

    # --- NOW enforce xlim (do this AFTER set_xticks) ---
    ax.set_xlim(0, xmax_h)

    if config.get("show_titles", False):
        ax.set_title(disp_name)
    else:
        ax.set_title("")

    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.5)
        ax.spines[side].set_edgecolor("black")

    ax.tick_params(axis="both", which="both", direction="out", length=3)
    ax.legend(loc="best", frameon=False)

    #fig.tight_layout()

    # Save
    if config.get("save_as_pdf", False):
        save_dir = config.get("save_path", ".")
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.splitext(config.get("outfile_base", "clone_adjusted_total"))[0]
        safe_name = _sanitize_for_filename(disp_name)
        out = os.path.join(save_dir, f"{base}_fraction_{safe_name}.pdf")
        fig.savefig(out, format="pdf", bbox_inches="tight", transparent=True)
        print(f"[saved] {out}")

    if config.get("show_plots", True):
        plt.show()
    else:
        plt.close(fig)



# ────────────────────────── Plotting per condition ──────────────────────────

def plot_condition(
    folder_path: str,
    identifiers: list[str],
    display_names: dict,
    highlight_spans: dict,
    config: dict,
    color_index: int = 0,
):
    """
    Plot one condition (one folder):

      - solid line + IQR: clone-adjusted total area (mm²)
      - dashed line + IQR: extrapolated clone area (mm²)
      - grey shaded background for highlight_spans (Ton intervals etc.)
    """
    folder_norm = _norm(folder_path)
    disp_name = _safe_display_name(folder_norm, display_names)

    data = aggregate_folder(folder_norm, identifiers, config)
    if data is None:
        return

    clone_med_hr = data["clone_med_hr"]
    clone_q1_hr = data["clone_q1_hr"]
    clone_q3_hr = data["clone_q3_hr"]

    adj_med_hr = data["adj_med_hr"]
    adj_q1_hr = data["adj_q1_hr"]
    adj_q3_hr = data["adj_q3_hr"]

    # Ensure aligned index in hours
    idx = adj_med_hr.index.intersection(clone_med_hr.index)
    if idx.empty:
        print(f"[warn] No overlapping frames for {disp_name}")
        return

    clone_med_hr = clone_med_hr.loc[idx]
    clone_q1_hr = clone_q1_hr.loc[idx]
    clone_q3_hr = clone_q3_hr.loc[idx]

    adj_med_hr = adj_med_hr.loc[idx]
    adj_q1_hr = adj_q1_hr.loc[idx]
    adj_q3_hr = adj_q3_hr.loc[idx]

    x_hours = adj_med_hr.index.values

    # px² → mm²
    um_per_px = float(config.get("scale_factor", 1.0))
    SCALE_MM2_PER_PX2 = (um_per_px ** 2) / 1e6

    clone_med_mm2 = clone_med_hr.values * SCALE_MM2_PER_PX2
    clone_q1_mm2 = clone_q1_hr.values * SCALE_MM2_PER_PX2
    clone_q3_mm2 = clone_q3_hr.values * SCALE_MM2_PER_PX2

    adj_med_mm2 = adj_med_hr.values * SCALE_MM2_PER_PX2
    adj_q1_mm2 = adj_q1_hr.values * SCALE_MM2_PER_PX2
    adj_q3_mm2 = adj_q3_hr.values * SCALE_MM2_PER_PX2

    # Feste Farbe pro display_name (oder Fallback aus Palette)
    color = _get_color_for_display_name(disp_name, config, color_index)

    dpi = int(config.get("dpi", 300))
    L = config.get("fixed_layout", None)

    if L:
        fig, ax = fixed_axes(
            fig_w=L["fig_w"], fig_h=L["fig_h"],
            ax_w=L["ax_w"], ax_h=L["ax_h"],
            left=L.get("left", 0.25),
            bottom=L.get("bottom", 0.22),
        )
    else:
        figsize = config.get("figsize", (3.5, 2.5))
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    fig.set_dpi(dpi)

    # Grey highlights (from highlight_spans in FRAMES)
    spans = highlight_spans.get(folder_norm, [])
    spans = spans or []
    highlight_alpha = float(config.get("highlight_alpha", 0.4))
    for (start_frame, end_frame) in spans:
        s_hr = start_frame / 2.0
        e_hr = end_frame / 2.0
        if e_hr <= s_hr:
            continue
        ax.axvspan(
            s_hr - 0.5,
            e_hr - 0.5,
            color="#bfbfbf",
            alpha=highlight_alpha,
            zorder=0,
            linewidth=0,
        )

    iqr_alpha = float(config.get("iqr_alpha", 0.25))
    line_width = float(config.get("line_width", 1.8))

    # IQR + line: clone-adjusted total colony (solid)
    ax.fill_between(
        x_hours,
        adj_q1_mm2,
        adj_q3_mm2,
        color=color,
        alpha=iqr_alpha,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        x_hours,
        adj_med_mm2,
        color=color,
        linewidth=line_width,
        linestyle="-",
        label="Total",
        zorder=2,
    )

    # IQR + line: clone area (dashed)
    ax.fill_between(
        x_hours,
        clone_q1_mm2,
        clone_q3_mm2,
        color=color,
        alpha=iqr_alpha * 0.7,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        x_hours,
        clone_med_mm2,
        color=color,
        linewidth=line_width,
        linestyle="dotted",
        label="Resistant",
        zorder=2,
    )

    # Axis formatting
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Area (mm²)")
    ax.set_ylim(bottom=0)

    if config.get("ymax") is not None:
        ax.set_ylim(top=float(config["ymax"]))
    # --- x ticks every 25 hours ---
    if config.get("xlim") is not None:
        xmax_h = float(config["xlim"]) / 2.0
    else:
        xmax_h = float(np.nanmax(x_hours))

    tick_step = 25
    ticks = np.arange(0, xmax_h + 1e-9, tick_step)
    ax.set_xticks(ticks)

    # --- NOW enforce xlim (do this AFTER set_xticks) ---
    ax.set_xlim(0, xmax_h)

    if config.get("show_titles", False):
        ax.set_title(disp_name)
    else:
        ax.set_title("")

    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.5)
        ax.spines[side].set_edgecolor("black")

    ax.tick_params(axis="both", which="both", direction="out", length=3)

    ax.legend(loc="best", frameon=False)
    #Set dashes for legend
    handles, labels = ax.get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        if label == "Resistant":
            handle.set_dashes([1.5, 1.5])
    ax.legend(handles, labels, loc="best", frameon=False)


    #fig.tight_layout()

    # Save
    if config.get("save_as_pdf", False):
        save_dir = config.get("save_path", ".")
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.splitext(config.get("outfile_base", "clone_adjusted_total"))[0]
        safe_name = _sanitize_for_filename(disp_name)
        out = os.path.join(save_dir, f"{base}_{safe_name}.pdf")
        fig.savefig(out, format="pdf", bbox_inches="tight", transparent=True)
        print(f"[saved] {out}")

    if config.get("show_plots", True):
        plt.show()
    else:
        plt.close(fig)


# ────────────────────────── Main ──────────────────────────


def main():
    # ── CONFIG ─────────────────────────────────────────────────────

    # Define the expected condition subfolders below Input_files.
    # Reviewers normally do not need to edit these entries when the supplied folder
    # structure is retained. Only the active-condition list below controls which
    # conditions are plotted.
    condition_paths = {
        "No Treatment": os.path.join(INPUT_ROOT, "No_treatment_control"),
        "Continuous": os.path.join(INPUT_ROOT, "Continuous_therapy"),
        "9/18": os.path.join(INPUT_ROOT, "9h_18h"),
        "4/18": os.path.join(INPUT_ROOT, "4h_18h"),
        "6.5/18": os.path.join(INPUT_ROOT, "6.5h_18h"),
        "6.5/15.5": os.path.join(INPUT_ROOT, "6.5h_15.5h"),
        "AT": os.path.join(INPUT_ROOT, "adaptive_therapy"),
        "6.5/20.5": os.path.join(INPUT_ROOT, "6.5h_20.5h"),
        "9/20.5": os.path.join(INPUT_ROOT, "9h_20.5h"),
    }

    config = {
        # Select the conditions used for this figure by commenting or uncommenting entries.
        "folder_paths": [
            #condition_paths["No Treatment"],
            #condition_paths["Continuous"],
            #condition_paths["9/18"],
            #condition_paths["4/18"],
            condition_paths["6.5/20.5"],
            condition_paths["6.5/18"],
            condition_paths["6.5/15.5"],
            condition_paths["9/20.5"],
            #condition_paths["AT"],
        ],

        # Specify the colony identifiers searched for within each active condition folder.
        "identifiers": [
            "P1_", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9",
            "P10", "P11", "P12", "P13", "P14", "P15", "P16", "P17",
            "P21_", "P22", "P24", "P25", "P26", "P27", "P28",
            "P29", "P210", "P211", "P213", "P215"
        ],

        # Set the microscope calibration used to convert pixel area to mm².
        "scale_factor": 8.648,  # µm per pixel

        # Assign the fixed display name associated with each condition folder.
        "display_names": {
            path: display_name for display_name, path in condition_paths.items()
        },

        # Define treatment-highlight intervals in frames for each condition.
        "highlight_spans": {
            condition_paths["No Treatment"]: [],
            condition_paths["Continuous"]: [[37, 334]],
            condition_paths["9/18"]: [
                [37, 55], [91, 109], [145, 163], [199, 217],
                [253, 271], [307, 325]
            ],
            condition_paths["4/18"]: [
                [37, 45], [81, 89], [125, 133], [169, 177],
                [213, 221], [257, 265], [301, 309]
            ],
            condition_paths["6.5/18"]: [
                [37, 50], [86, 99], [135, 148], [184, 197],
                [233, 246], [282, 295], [331, 344]
            ],
            condition_paths["6.5/15.5"]: [
                [37, 50], [81, 94], [125, 138], [169, 182],
                [213, 226], [257, 270], [301, 314], [348, 361]
            ],
            condition_paths["AT"]: [
                [37, 61], [110, 141], [189, 284]
            ],
            condition_paths["6.5/20.5"]: [
                [37, 50], [91, 104], [145, 158], [199, 212],
                [253, 266], [307, 320], [349, 362]
            ],
            condition_paths["9/20.5"]: [
                [37, 55], [96, 114], [155, 173], [214, 232],
                [273, 291], [332, 350]
            ],
        },

        # Assign the fixed plot color associated with each display name.
        "display_colors": {
            "No Treatment": "#393b79",  # dark indigo
            "4/18": "#637939",          # dark olive
            "Continuous": "#7b4173",   # dark purple
            "9/18": "#843c39",
            "6.5/18": "#8c6d31",
            "6.5/15.5": "teal",
            "6.5/20.5": "palegreen",
            "AT": "black",
            "9/20.5": "plum",
        },

        # Store the treatment timing associated with each display name.
        "treatment_schedule": {
            "No Treatment": {"Ton": 0, "Toff": np.inf},
            "Continuous": {"Ton": np.inf, "Toff": 0.0},
            "9/18": {"Ton": 9.0, "Toff": 18.0},
            "9/20.5": {"Ton": 9.0, "Toff": 20.5},
        },

        # Set the fixed physical figure and axes dimensions in inches.
        "fixed_layout": {
            "fig_w": 1.8,
            "fig_h": 1.3,
            "ax_w": 1.25,
            "ax_h": 0.85,
            "left": 0.35,
            "bottom": 0.30,
        },

        # Set smoothing, axis limits, line appearance, and plot visibility.
        "smooth_window": 9,      # rolling median window in frames
        "figsize": (1.8, 1.3),
        "dpi": 600,
        "xlim": 300,             # frames
        "ymax": 71.0,            # mm²
        "line_width": 1.0,
        "iqr_alpha": 0.6,
        "highlight_alpha": 1,
        "show_titles": False,

        # Configure PDF export to the project-relative Output_files directory.
        "save_as_pdf": True,
        "save_path": OUTPUT_ROOT,
        "outfile_base": "colony_clone",
        "show_plots": True,
    }

    # ── RUN ────────────────────────────────────────────────────────
    set_nature_style()

    display_names = _prep_display_name_map(config.get("display_names", {}))
    highlight_spans = _prep_span_map(config.get("highlight_spans", {}))

    for i, folder in enumerate(config["folder_paths"]):
        if not os.path.isdir(folder):
            print(f"[skip] Not a directory: {folder}")
            continue
        plot_condition(
            folder_path=folder,
            identifiers=config["identifiers"],
            display_names=display_names,
            highlight_spans=highlight_spans,
            config=config,
            color_index=i,   # fallback-palette index
        )
        # Create the corresponding resistant-fraction plot.
        plot_condition_fraction(
            folder_path=folder,
            identifiers=config["identifiers"],
            display_names=display_names,
            highlight_spans=highlight_spans,
            config=config,
            color_index=i,
        )


if __name__ == "__main__":
    main()
