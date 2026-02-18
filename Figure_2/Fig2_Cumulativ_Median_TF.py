#!/usr/bin/env python3
"""
Cumulative first-crossing plot for clones getting within a distance threshold
of the colony front over time.

- Loads clone CSVs by identifiers from a folder
- Adds a unique_particle id per file
- Filters by (i) initial frame window and (ii) initial distance-to-edge window
- Computes first time each clone reaches ≤ threshold (µm)
- Plots cumulative count (or %) vs. time (hours)

Author: you :)
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # IO
    "folder_path": r"S:\Members\Nico\Experiment_CSV_files\20240917_continuous_dose_2",
    "identifiers": ['P1_', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15',
                    'P16'],
    "save_path": r"C:\Users\nappold\Desktop\Manuscript "
                 r"Figures\Fig2\cumulative_crossers_10um_CT_new_unfiltered.pdf",  # or None
    # where to save the median±IQR-through-time figure
    "save_path_median_iqr_time": r"C:\Users\nappold\Desktop\Manuscript "
                                 r"Figures\Fig2\median_IQR_over_time_CT_unfiltered.pdf",

    # optional: draw faint per-identifier traces
    "show_per_identifier_traces": False,
    "export_csv": None,  # or None

    # Filtering (frames are in your raw frame units)
    "min_frame": 0,  # initial frame lower bound (inclusive)
    "max_frame": 200,  # initial frame upper bound (inclusive)
    # initial distance window BEFORE scaling (same units as CSV: pixels/arb units)
    "init_dist_min_raw": 2,
    "init_dist_max_raw": 57,

    # Plot axes and formatting
    "lower_bound": 0,  # x-axis in frames
    "upper_bound": 200,  # x-axis in frames
    "treatment_window": (37, 300),  # or None

    "theme": "bright",  # "bright" or "dark"
    "normalize": False,  # True → percentage, False → counts
    "n_bins": 20,  # number of bins to aggregate escape events (adjustable)

    # Physics/units
    "scale_factor": 8.648,  # raw → µm
    "threshold_um": 10.0,  # crossing threshold in µm
    "frames_per_hour": 2,  # your convention (2 frames = 1 hour)
    "figsize": (3.6, 1.8),

    # Classification: touch vs recovery (in µm)
    "breach_threshold_um": 10.0,  # counts as a 'touch'
    "recover_threshold_um": 60.0,  # if it later reaches ≥ this, we EXCLUDE it (recovered)

    # Optional debounce (frames)
    "min_breach_frames": 1,  # require ≥N consecutive frames ≤ breach to call it a real touch
    "min_recovery_frames": 1,  # require ≥N consecutive frames ≥ recover to call it a recovery

}


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def grab_files(folder_path, identifiers):
    """Return first CSV per identifier that contains 'clone' in the filename,
    *and* the list of identifiers for which a CSV was found."""
    folder_path = folder_path.replace('\\', '/')
    if folder_path and folder_path[-1] != '/':
        folder_path += '/'

    file_paths = []
    used_identifiers = []

    for identifier in identifiers:
        matches = [f for f in os.listdir(folder_path)
                   if identifier in f and f.endswith('.csv') and "clone_" in f]
        if len(matches) == 0:
            print(f"[warn] No CSV with identifier '{identifier}' found in {folder_path}")
            continue
        if len(matches) > 1:
            raise ValueError(f"More than one file with identifier '{identifier}' found.")
        file_paths.append(os.path.join(folder_path, matches[0]))
        used_identifiers.append(identifier)

    return file_paths, used_identifiers, folder_path



def add_unique_identifier(df, df_id):
    """unique_particle = df_id*10000 + particle"""
    if "particle" not in df.columns:
        raise ValueError("CSV must contain a 'particle' column.")
    df = df.copy()
    df["unique_particle"] = df_id * 10000 + df["particle"]
    return df


def filter_by_initial_frame_and_distance(df,
                                         min_frame, max_frame,
                                         init_dist_min_raw, init_dist_max_raw):
    """
    Keep only clones whose FIRST frame is within [min_frame, max_frame]
    AND whose initial distance_to_edge is within [init_dist_min_raw, init_dist_max_raw]
    (raw units before any scaling).
    """
    req = {"unique_particle", "frame", "distance_to_edge"}
    if not req.issubset(df.columns):
        missing = req - set(df.columns)
        raise ValueError(f"df missing columns: {missing}")

    # first observation per clone
    first_rows = df.sort_values(["unique_particle", "frame"]).groupby("unique_particle").first()
    good = first_rows[
        (first_rows["frame"].between(min_frame, max_frame)) &
        (first_rows["distance_to_edge"].between(init_dist_min_raw, init_dist_max_raw))
    ].index

    return df[df["unique_particle"].isin(good)].copy()

def has_consecutive(arr_bool: np.ndarray, run_len: int) -> bool:
    """True if there is a run of >= run_len consecutive Trues in arr_bool."""
    if run_len <= 1:
        return arr_bool.any()
    if arr_bool.size == 0:
        return False
    # sliding sum trick
    s = np.convolve(arr_bool.astype(int), np.ones(run_len, dtype=int), mode='valid')
    return (s >= run_len).any()

def compute_first_nonrecovering_crossing_frames(
    df,
    *,
    breach_threshold_um: float,
    recover_threshold_um: float,
    scale_factor: float,
    min_breach_frames: int = 1,
    min_recovery_frames: int = 1,
):
    """
    For each clone:
      - dist_um = distance_to_edge * scale_factor
      - 'breach' = ≥ min_breach_frames consecutive frames with dist_um ≤ breach_threshold_um
      - after the first breach, 'recovery' = ≥ min_recovery_frames consecutive frames with dist_um ≥ recover_threshold_um
      - Count as FAILURE iff: breached AND NEVER recovered afterwards.
      - Record the first breach frame for failures.

    Returns:
      frames:       sorted unique frames present in df
      cum_counts:   cumulative #non-recovering failures by each frame
      all_clones:   total #clones considered (denominator for %)
      failed_ids:   list of unique_particle ids counted as failures
    """
    req = {"unique_particle", "frame", "distance_to_edge"}
    if not req.issubset(df.columns):
        missing = req - set(df.columns)
        raise ValueError(f"df missing columns: {missing}")

    d = df.sort_values(["unique_particle", "frame"]).copy()
    d["dist_um"] = d["distance_to_edge"] * scale_factor

    all_clones = d["unique_particle"].nunique()
    frames_all = np.sort(d["frame"].unique())

    failure_first_frames = []
    failed_ids = []

    for uid, g in d.groupby("unique_particle", sort=False):
        f = g["frame"].to_numpy()
        y = g["dist_um"].to_numpy()

        # find first sustained breach
        breach_bool = (y <= breach_threshold_um)
        # locate first index where a run of min_breach_frames starts
        breach_idx = None
        if min_breach_frames <= 1:
            if breach_bool.any():
                breach_idx = int(np.where(breach_bool)[0][0])
        else:
            # run-length via sliding sum
            run = np.convolve(breach_bool.astype(int),
                              np.ones(min_breach_frames, dtype=int),
                              mode="valid")
            starts = np.where(run >= min_breach_frames)[0]
            if starts.size > 0:
                breach_idx = int(starts[0])

        if breach_idx is None:
            # never breached → not a failure
            continue

        # after breach, check sustained recovery
        post = slice(breach_idx + 1, None)
        rec_bool = (y[post] >= recover_threshold_um) if breach_idx + 1 < len(y) else np.array([], dtype=bool)

        recovered = False
        if rec_bool.size:
            if min_recovery_frames <= 1:
                recovered = rec_bool.any()
            else:
                run_rec = np.convolve(rec_bool.astype(int),
                                      np.ones(min_recovery_frames, dtype=int),
                                      mode="valid")
                recovered = (run_rec >= min_recovery_frames).any()

        if not recovered:
            # non-recovering failure: use the *first* breach frame
            failure_first_frames.append(int(f[breach_idx]))
            failed_ids.append(uid)

    if len(failure_first_frames) == 0:
        return frames_all, np.zeros_like(frames_all, dtype=int), all_clones, []

    first_fail_sorted = np.sort(np.asarray(failure_first_frames, dtype=int))
    # cumulative by frame
    cum_counts = np.searchsorted(first_fail_sorted, frames_all, side="right")
    return frames_all, cum_counts, all_clones, failed_ids

def plot_cumulative(frames, cum_counts, all_clones,
                    *, frames_per_hour=2,
                    lower_bound=None, upper_bound=None,
                    treatment_window=None, normalize=False,
                    theme="bright", save_path=None, figsize=(3.5, 1.8)):
    """Cumulative plot (count or %) vs time (hours)."""
    # rcParams
    mpl.rcParams.update({
        'pdf.fonttype': 42,  # embed TrueType fonts
        'ps.fonttype': 42,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial'],

        'font.size': 7,
        'axes.titlesize': 7,
        'axes.labelsize': 7,
        # adjust padding
        'axes.titlepad': 2,
        'axes.linewidth': 0.5,

        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'xtick.direction': 'out',
        'ytick.direction': 'out',

        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'axes.labelpad': 2,


        'legend.frameon': False,
        'legend.fontsize': 6,

        'savefig.dpi': 300,
        'figure.dpi': 300,

        # We won't globally kill top/right spines because for you we want the full box.
        # You can still turn them off per-axis if you want.
    })

    if theme.lower() == "dark":
        fig_face, ax_face = "black", "black"
        spine_color = tick_color = label_color = "white"
        line_color = "yellow"
    elif theme.lower() == "bright":
        fig_face, ax_face = "white", "white"
        spine_color = tick_color = label_color = "black"
        line_color = "orangered"
    else:
        raise ValueError("theme must be 'bright' or 'dark'")

    # restrict x if requested
    if lower_bound is None:
        lower_bound = int(frames.min())
    if upper_bound is None:
        upper_bound = int(frames.max())
    mask = (frames >= lower_bound) & (frames <= upper_bound)
    x = frames[mask]
    y_raw = cum_counts[mask]

    if normalize and all_clones > 0:
        y = (y_raw / all_clones) * 100.0
        y_label = "Clones crossing threshold (%)"
    else:
        y = y_raw
        y_label = "Escapes (Cumulative)"

    fig, ax = plt.subplots(figsize=figsize, dpi=300, facecolor=fig_face)
    ax.set_facecolor(ax_face)
    for s in ax.spines.values():
        s.set_color(spine_color)
    ax.tick_params(axis="x", colors=tick_color)
    ax.tick_params(axis="y", colors=tick_color)

    # ticks in hours (2 frames = 1 h by default)
    frames_per_5h = 5 * frames_per_hour
    ax.set_xlim(lower_bound, upper_bound)
    ax.set_ylim(bottom=0, top=69)
    xticks = np.arange(lower_bound, upper_bound + 1, frames_per_5h)
    if len(xticks) == 0:
        xticks = np.array([lower_bound, upper_bound])
    ax.set_xticks(xticks)
    ax.set_xticklabels([int(t / frames_per_hour) for t in xticks], color=tick_color)

    if treatment_window is not None:
        ax.axvspan(treatment_window[0]-1, treatment_window[1]-1, color="#bfbfbf", alpha=1, linewidth=0)

    ax.plot(x, y, linewidth=1.5, color=line_color, label="≤ threshold")
    ax.set_xlabel("Time (h)", color=label_color, fontsize=7)
    ax.set_ylabel(y_label, color=label_color, fontsize=7)
    #ax.legend(frameon=False, fontsize=6, labelcolor=label_color)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig_face, transparent=True)
        print(f"[ok] Saved plot → {save_path}")
    else:
        plt.show()

def first_failure_frames_per_identifier(
    df: pd.DataFrame,
    *,
    breach_threshold_um: float,
    recover_threshold_um: float,
    scale_factor: float,
    min_breach_frames: int = 1,
    min_recovery_frames: int = 1,
    identifiers: list[str]
) -> dict:
    """
    Returns:
      {
        identifier: {
          "first_fail_frames": np.ndarray[int],  # sorted first-breach frames for non-recovering clones
          "n_clones": int                        # clones considered (denominator for %)
        }, ...
      }
    """
    req = {"unique_particle", "frame", "distance_to_edge"}
    if not req.issubset(df.columns):
        missing = req - set(df.columns)
        raise ValueError(f"df missing columns: {missing}")

    out = {ident: {"first_fail_frames": np.array([], dtype=int), "n_clones": 0}
           for ident in identifiers}

    d = df.sort_values(["unique_particle", "frame"]).copy()
    d["dist_um"] = d["distance_to_edge"] * scale_factor

    # Pre-split by identifier using your unique_particle scheme (df_id*10000 + particle)
    for df_id, g_id in d.groupby(d["unique_particle"] // 10000, sort=False):
        df_id = int(df_id)
        if not (0 <= df_id < len(identifiers)):
            continue
        ident = identifiers[df_id]

        n_clones = g_id["unique_particle"].nunique()
        failure_first_frames = []

        for uid, g in g_id.groupby("unique_particle", sort=False):
            f = g["frame"].to_numpy()
            y = g["dist_um"].to_numpy()

            breach_bool = (y <= breach_threshold_um)
            breach_idx = None
            if min_breach_frames <= 1:
                if breach_bool.any():
                    breach_idx = int(np.where(breach_bool)[0][0])
            else:
                run = np.convolve(breach_bool.astype(int),
                                  np.ones(min_breach_frames, dtype=int),
                                  mode="valid")
                starts = np.where(run >= min_breach_frames)[0]
                if starts.size > 0:
                    breach_idx = int(starts[0])

            if breach_idx is None:
                continue

            post = slice(breach_idx + 1, None)
            rec_bool = (y[post] >= recover_threshold_um) if breach_idx + 1 < len(y) else np.array([], dtype=bool)

            recovered = False
            if rec_bool.size:
                if min_recovery_frames <= 1:
                    recovered = rec_bool.any()
                else:
                    run_rec = np.convolve(rec_bool.astype(int),
                                          np.ones(min_recovery_frames, dtype=int),
                                          mode="valid")
                    recovered = (run_rec >= min_recovery_frames).any()

            if not recovered:
                failure_first_frames.append(int(f[breach_idx]))

        first_sorted = np.sort(np.asarray(failure_first_frames, dtype=int))
        out[ident] = {"first_fail_frames": first_sorted, "n_clones": int(n_clones)}

    return out


def cumulative_series_from_firsts(first_fail_frames: np.ndarray, frame_grid: np.ndarray) -> np.ndarray:
    """
    Step function: at each t in frame_grid, how many failures occurred at or before t?
    """
    if first_fail_frames.size == 0:
        return np.zeros_like(frame_grid, dtype=int)
    # searchsorted with 'right' gives count <= t
    return np.searchsorted(first_fail_frames, frame_grid, side="right")


# ──────────────────────────────────────────────────────────────────────────────
# Median + IQR across identifiers through time
# ──────────────────────────────────────────────────────────────────────────────
def aggregate_median_iqr_over_time(
    per_id: dict,
    frame_grid: np.ndarray,
    *,
    normalize: bool = False
) -> dict:
    """
    Build a matrix of shape (n_id, T), then compute median & IQR across identifiers at each time.
    Returns dict with 'median', 'q1', 'q3', and also per-identifier matrix and id order.
    """
    idents = list(per_id.keys())
    mats = []
    for ident in idents:
        firsts = per_id[ident]["first_fail_frames"]
        series = cumulative_series_from_firsts(firsts, frame_grid).astype(float)
        if normalize:
            denom = per_id[ident]["n_clones"]
            series = (series / denom * 100.0) if denom > 0 else np.full_like(series, np.nan, dtype=float)
        mats.append(series)

    M = np.vstack(mats) if mats else np.zeros((0, frame_grid.size), dtype=float)
    median = np.nanmedian(M, axis=0) if M.size else np.array([])
    q1 = np.nanpercentile(M, 25, axis=0) if M.size else np.array([])
    q3 = np.nanpercentile(M, 75, axis=0) if M.size else np.array([])
    return {"idents": idents, "matrix": M, "median": median, "q1": q1, "q3": q3}

def plot_group_median_iqr_over_time(
    frame_grid: np.ndarray,
    agg: dict,
    *,
    frames_per_hour: float = 2.0,
    theme: str = "bright",
    treatment_window: tuple[int, int] | None = None,
    show_per_identifier: bool = False,
    figsize=(3.6, 1.8),
    save_path: str | None = None,
    ylabel: str | None = None,
    lower_bound: int | None = None,
    upper_bound: int | None = None,
):
    # Nature-style rcParams
    mpl.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial'],
        'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7,
        'axes.titlepad': 2, 'axes.linewidth': 0.5,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.minor.size': 3, 'ytick.minor.size': 3,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'axes.labelpad': 2, 'legend.frameon': False, 'legend.fontsize': 6,
        'savefig.dpi': 300, 'figure.dpi': 300,
    })

    if theme.lower() == "dark":
        fig_face, ax_face = "black", "black"
        spine_color = tick_color = label_color = "white"
        med_color = "#DAA520"  # golden
        band_alpha = 0.25
        per_id_color = "#888888"
    else:
        fig_face, ax_face = "white", "white"
        spine_color = tick_color = label_color = "black"
        med_color = "orangered"
        band_alpha = 0.3
        per_id_color = "blue"

    fig, ax = plt.subplots(figsize=(3.55,1.8), dpi=300, facecolor=fig_face)
    ax.set_facecolor(ax_face)
    for s in ax.spines.values():
        s.set_color(spine_color)
    ax.tick_params(axis="x", colors=tick_color)
    ax.tick_params(axis="y", colors=tick_color)

    # X in hours with 5h ticks
    x = frame_grid
    xticks = np.arange(x.min(), x.max() + 1, int(5 * frames_per_hour))
    if xticks.size == 0:
        xticks = np.array([x.min(), x.max()])
    ax.set_xlim(lower_bound, upper_bound)
    ax.set_xticks(xticks)
    ax.set_xticklabels([int(t / frames_per_hour) for t in xticks], color=tick_color)
    ax.set_xlabel("Time (h)", color=label_color, fontsize=7)

    ax.set_ylim(bottom=0, top=5.5)
    if ylabel is None:
        ylabel = "Median escapes"  # or "(%)" if normalized in caller
    ax.set_ylabel(ylabel, color=label_color, fontsize=7)

    if treatment_window is not None:
        ax.axvspan(treatment_window[0]-1, treatment_window[1]-1, color="#bfbfbf", alpha=1, linewidth=0)

    # Optional: faint per-identifier traces (helps show spread)
    M = agg["matrix"]  # shape (n_id, T)
    if show_per_identifier and M.size:
        for row in M:
            ax.plot(x, row, lw=0.6, color=per_id_color, alpha=0.6, zorder=1)

    # IQR band
    ax.fill_between(x, agg["q1"], agg["q3"], alpha=band_alpha, color=med_color, linewidth=0, zorder=2)

    # Median line
    ax.plot(x, agg["median"], lw=1.6, color=med_color, zorder=3)

    # Boxed axes
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig_face, transparent =True)
        print(f"[ok] Saved median±IQR over time → {save_path}")
    else:
        plt.show()

def compute_clone_counts_per_identifier_over_time(
    df: pd.DataFrame,
    identifiers: list[str],
    frame_grid: np.ndarray
) -> dict:
    """
    For each identifier and each frame in frame_grid, count how many clones
    (unique_particle) are present in that frame.

    Returns:
      {
        "idents": [list of identifiers in order],
        "matrix": np.ndarray shape (n_id, T) with counts
      }
    """
    req = {"unique_particle", "frame"}
    if not req.issubset(df.columns):
        missing = req - set(df.columns)
        raise ValueError(f"df missing columns for clone count: {missing}")

    # Extract identifier index from unique_particle = df_id*10000 + particle
    d = df[["unique_particle", "frame"]].copy()
    d["id_index"] = (d["unique_particle"] // 10000).astype(int)

    # Keep only valid identifier indices
    n_id = len(identifiers)
    d = d[(d["id_index"] >= 0) & (d["id_index"] < n_id)]

    # Group: for each (id_index, frame) count how many clones are present
    grouped = d.groupby(["id_index", "frame"])["unique_particle"].nunique()

    # Build matrix [n_id x T], where T = len(frame_grid)
    T = frame_grid.size
    M = np.zeros((n_id, T), dtype=int)

    for i_id in range(n_id):
        for j, fr in enumerate(frame_grid):
            M[i_id, j] = grouped.get((i_id, int(fr)), 0)

    return {"idents": identifiers, "matrix": M}

def plot_per_identifier_clone_counts(
    frame_grid: np.ndarray,
    counts_matrix: np.ndarray,
    identifiers: list[str],
    *,
    frames_per_hour: float = 2.0,
    theme: str = "bright",
    figsize=(3.6, 1.8),
):
    """
    Debug plot: for each identifier, plot how many clones are present
    at each frame as a line. This plot is only shown, not saved.
    """
    # Nature-style rcParams (same vibe as your other plots)
    mpl.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial'],
        'font.size': 7,
        'axes.titlesize': 7, 'axes.labelsize': 7,
        'axes.titlepad': 2, 'axes.linewidth': 0.5,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.minor.size': 3, 'ytick.minor.size': 3,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'axes.labelpad': 2,
        'legend.frameon': False, 'legend.fontsize': 6,
        'savefig.dpi': 300, 'figure.dpi': 300,
    })

    if theme.lower() == "dark":
        fig_face, ax_face = "black", "black"
        spine_color = tick_color = label_color = "white"
    else:
        fig_face, ax_face = "white", "white"
        spine_color = tick_color = label_color = "black"

    fig, ax = plt.subplots(figsize=figsize, dpi=300, facecolor=fig_face)
    ax.set_facecolor(ax_face)
    for s in ax.spines.values():
        s.set_color(spine_color)
    ax.tick_params(axis="x", colors=tick_color)
    ax.tick_params(axis="y", colors=tick_color)

    # X in hours
    x = frame_grid
    xticks = np.arange(x.min(), x.max() + 1, int(5 * frames_per_hour))
    if xticks.size == 0:
        xticks = np.array([x.min(), x.max()])
    ax.set_xlim(x.min(), x.max())
    ax.set_xticks(xticks)
    ax.set_xticklabels([int(t / frames_per_hour) for t in xticks], color=tick_color)
    ax.set_xlabel("Time (h)", color=label_color, fontsize=7)

    ax.set_ylabel("Clones present", color=label_color, fontsize=7)

    # One line per identifier
    n_id = counts_matrix.shape[0]
    for i in range(n_id):
        # thin lines, slight alpha to see overlaps
        ax.plot(
            x,
            counts_matrix[i, :],
            lw=0.8,
            alpha=0.8,
            label=identifiers[i]
        )

    # Full box
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    # Legend (small, multi-column)
    ax.legend(loc="upper left", ncol=4, fontsize=5)

    plt.tight_layout()
    plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main(cfg: dict):
    # gather files
    files, active_identifiers, folder_path = grab_files(cfg["folder_path"], cfg["identifiers"])
    if not files:
        raise SystemExit("[error] No files found. Check folder_path/identifiers.")

    # From here on, only use identifiers that actually have data
    identifiers = active_identifiers
    print("\n[info] Using identifiers with CSV data only:")
    for ident in identifiers:
        print(f"  - {ident}")


    # load + attach unique ids
    dfs = []
    for i, fp in enumerate(files):
        df = pd.read_csv(fp)
        # required columns?
        need = {"particle", "frame", "distance_to_edge"}
        miss = need - set(df.columns)
        if miss:
            raise ValueError(f"{os.path.basename(fp)} is missing columns: {miss}")
        df = add_unique_identifier(df, i)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # filter clones by first frame + initial distance window
    combined = filter_by_initial_frame_and_distance(
        combined,
        cfg["min_frame"], cfg["max_frame"],
        cfg["init_dist_min_raw"], cfg["init_dist_max_raw"]
    )

    if combined.empty:
        raise SystemExit("[warn] After filtering, no data remains. Adjust config.")

    # ── NEW: check how many clones each identifier has at each timeframe ──
    frame_grid_counts = np.arange(cfg["lower_bound"], cfg["upper_bound"] + 1, dtype=int)

    counts_info = compute_clone_counts_per_identifier_over_time(
        combined,
        identifiers,
        frame_grid_counts
    )

    # Optional quick text check
    # (prints total clones per identifier after filtering)
    print("\n[info] Clones per identifier (total unique clones after filtering):")
    for ident, total in zip(
        counts_info["idents"],
        counts_info["matrix"].max(axis=1)  # max over time ≈ max clones present at once
    ):
        print(f"  {ident}: {int(total)} clones (max present in a single frame)")

    # Debug plot: clones per identifier vs time (not saved)
    plot_per_identifier_clone_counts(
        frame_grid_counts,
        counts_info["matrix"],
        counts_info["idents"],
        frames_per_hour=cfg["frames_per_hour"],
        theme=cfg["theme"],
        figsize=cfg.get("figsize", (3.6, 1.8)),
    )

    # compute first *non-recovering* failures + cumulative
    frames, cum_counts, all_clones, failed_ids = compute_first_nonrecovering_crossing_frames(
        combined,
        breach_threshold_um=cfg["breach_threshold_um"],
        recover_threshold_um=cfg["recover_threshold_um"],
        scale_factor=cfg["scale_factor"],
        min_breach_frames=cfg.get("min_breach_frames", 1),
        min_recovery_frames=cfg.get("min_recovery_frames", 1),
    )



    # optional CSV export (frame, hour, cumulative_count, percent)
    if cfg.get("export_csv"):
        out = pd.DataFrame({
            "frame": frames,
            "hour": frames / cfg["frames_per_hour"],
            "cumulative_count": cum_counts,
            "cumulative_percent": (cum_counts / all_clones * 100.0) if all_clones > 0 else np.nan
        })
        out.to_csv(cfg["export_csv"], index=False)
        print(f"[ok] Saved data → {cfg['export_csv']}")

    # plot
    plot_cumulative(
        frames, cum_counts, all_clones,
        frames_per_hour=cfg["frames_per_hour"],
        lower_bound=cfg["lower_bound"],
        upper_bound=cfg["upper_bound"],
        treatment_window=cfg["treatment_window"],
        normalize=cfg["normalize"],
        theme=cfg["theme"],
        save_path=cfg["save_path"],
        figsize=cfg["figsize"]
    )

    # optional: save a copy of this script next to the figure for provenance
    if cfg.get("save_path"):
        try:
            code_path = os.path.splitext(cfg["save_path"])[0] + "_code.txt"
            script_file = os.path.abspath(__file__)
            shutil.copy(script_file, code_path)
            print(f"[ok] Copied code → {code_path}")
        except NameError:
            print("[info] __file__ not available (likely run in REPL/Notebook).")

        # --- NEW: per-identifier cumulative series, then group median±IQR through time
        # frame grid (respect your plot window)
    frame_grid = np.arange(cfg["lower_bound"], cfg["upper_bound"] + 1, dtype=int)

    per_id = first_failure_frames_per_identifier(
        combined,
        breach_threshold_um=cfg["breach_threshold_um"],
        recover_threshold_um=cfg["recover_threshold_um"],
        scale_factor=cfg["scale_factor"],
        min_breach_frames=cfg.get("min_breach_frames", 1),
        min_recovery_frames=cfg.get("min_recovery_frames", 1),
        identifiers=identifiers,  # <- only those with CSV
    )

    agg = aggregate_median_iqr_over_time(
        per_id,
        frame_grid,
        normalize=cfg.get("normalize", False)
    )

    ylabel = "Median escapes per colony"
    if cfg.get("normalize", False):
        ylabel = "Escapes per position (%) (median, IQR)"

    plot_group_median_iqr_over_time(
        frame_grid,
        agg,
        frames_per_hour=cfg["frames_per_hour"],
        theme=cfg["theme"],
        treatment_window=cfg["treatment_window"],
        show_per_identifier=cfg.get("show_per_identifier_traces", False),
        figsize=cfg.get("figsize", (3.5, 1.8)),
        save_path=cfg.get("save_path_median_iqr_time"),
        ylabel=ylabel
    )


if __name__ == "__main__":
    main(CONFIG)
