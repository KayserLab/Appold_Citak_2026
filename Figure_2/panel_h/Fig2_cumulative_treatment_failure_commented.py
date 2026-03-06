#!/usr/bin/env python3
"""
Cumulative first-crossing plot for clones reaching a distance threshold to the colony front.

Workflow
- Load one clone CSV per identifier from a folder.
- Create a dataset-wide clone id (unique_particle).
- Filter clones by:
    (i) the first observed frame window
    (ii) the initial distance-to-edge window (raw units from CSV)
- For each clone, detect the first sustained "breach" event (distance ≤ threshold, in µm)
  and exclude clones that later show sustained "recovery" (distance ≥ recovery threshold, in µm).
- Plot cumulative number (or %) of non-recovering breaches vs time.
- Compute per-identifier cumulative series and plot median ± IQR across identifiers over time.
"""

import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# PROJECT-RELATIVE IO
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_ROOT = os.path.join(SCRIPT_DIR, "Input_files")
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "Output_files")


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # Input folder inside Input_files/
    "input_relpath": "14h_Pulse",

    "identifiers": [
        "P1_", "P2", "P3", "P4", "P5", "P6", "P7", "P8",
        "P9", "P10", "P11", "P12", "P13", "P14", "P15", "P16",
    ],

    # Outputs go into Output_files/
    "save_path": os.path.join(OUTPUT_ROOT, "cumulative_crossers_10um_Pulse_new_unfiltered.pdf"),
    "save_path_median_iqr_time": os.path.join(OUTPUT_ROOT, "Fig2_panel_h_median_IQR_over_time_Pulse_unfiltered.pdf"),
    "export_csv": None,

    # Optional: draw faint per-identifier traces in the median±IQR plot
    "show_per_identifier_traces": False,

    # Filtering (frames are in raw frame units)
    "min_frame": 0,
    "max_frame": 200,

    # Initial distance window BEFORE scaling (raw units from CSV)
    "init_dist_min_raw": 2,
    "init_dist_max_raw": 57,

    # Plot axes and formatting
    "lower_bound": 0,     # x-axis in frames
    "upper_bound": 200,   # x-axis in frames
    "treatment_window": (37, 300),

    "theme": "bright",    # "bright" or "dark"
    "normalize": False,   # True → percentage, False → counts
    "figsize": (3.6, 1.8),

    # Units
    "scale_factor": 8.648,      # raw → µm
    "frames_per_hour": 2,       # 2 frames = 1 hour

    # Classification: breach vs recovery (in µm)
    "breach_threshold_um": 10.0,
    "recover_threshold_um": 60.0,

    # Debounce (frames)
    "min_breach_frames": 1,     # require ≥N consecutive frames ≤ breach to call a breach
    "min_recovery_frames": 1,   # require ≥N consecutive frames ≥ recovery to call a recovery
}


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def grab_files(folder_path, identifiers):
    """
    Return first CSV per identifier that contains 'clone_' in the filename,
    and the list of identifiers for which a CSV was found.
    """
    folder_path = folder_path.replace("\\", "/")
    if folder_path and folder_path[-1] != "/":
        folder_path += "/"

    file_paths = []
    used_identifiers = []

    for identifier in identifiers:
        matches = [
            f for f in os.listdir(folder_path)
            if identifier in f and f.endswith(".csv") and "clone_" in f
        ]
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


def filter_by_initial_frame_and_distance(df, min_frame, max_frame, init_dist_min_raw, init_dist_max_raw):
    """
    Keep only clones whose FIRST frame is within [min_frame, max_frame]
    AND whose initial distance_to_edge is within [init_dist_min_raw, init_dist_max_raw]
    (raw units before any scaling).
    """
    req = {"unique_particle", "frame", "distance_to_edge"}
    if not req.issubset(df.columns):
        missing = req - set(df.columns)
        raise ValueError(f"df missing columns: {missing}")

    first_rows = df.sort_values(["unique_particle", "frame"]).groupby("unique_particle").first()
    good = first_rows[
        (first_rows["frame"].between(min_frame, max_frame))
        & (first_rows["distance_to_edge"].between(init_dist_min_raw, init_dist_max_raw))
    ].index

    return df[df["unique_particle"].isin(good)].copy()


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
      - breach = ≥ min_breach_frames consecutive frames with dist_um ≤ breach_threshold_um
      - after breach, recovery = ≥ min_recovery_frames consecutive frames with dist_um ≥ recover_threshold_um
      - Count as event iff: breached AND never recovered afterwards.
      - Record the first breach frame for events.

    Returns:
      frames_all:   sorted unique frames present in df
      cum_counts:   cumulative #events by each frame
      all_clones:   total #clones considered (denominator for %)
      failed_ids:   list of unique_particle ids counted as events
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

        breach_bool = (y <= breach_threshold_um)
        breach_idx = None

        if min_breach_frames <= 1:
            if breach_bool.any():
                breach_idx = int(np.where(breach_bool)[0][0])
        else:
            run = np.convolve(
                breach_bool.astype(int),
                np.ones(min_breach_frames, dtype=int),
                mode="valid",
            )
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
                run_rec = np.convolve(
                    rec_bool.astype(int),
                    np.ones(min_recovery_frames, dtype=int),
                    mode="valid",
                )
                recovered = (run_rec >= min_recovery_frames).any()

        if not recovered:
            failure_first_frames.append(int(f[breach_idx]))
            failed_ids.append(uid)

    if len(failure_first_frames) == 0:
        return frames_all, np.zeros_like(frames_all, dtype=int), all_clones, []

    first_fail_sorted = np.sort(np.asarray(failure_first_frames, dtype=int))
    cum_counts = np.searchsorted(first_fail_sorted, frames_all, side="right")
    return frames_all, cum_counts, all_clones, failed_ids


def plot_cumulative(
    frames,
    cum_counts,
    all_clones,
    *,
    frames_per_hour=2,
    lower_bound=None,
    upper_bound=None,
    treatment_window=None,
    normalize=False,
    theme="bright",
    save_path=None,
    figsize=(3.5, 1.8),
):
    """Cumulative plot (count or %) vs time (hours)."""
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial"],
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "axes.titlepad": 2,
        "axes.linewidth": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.labelpad": 2,
        "legend.frameon": False,
        "legend.fontsize": 6,
        "savefig.dpi": 300,
        "figure.dpi": 300,
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

    frames_per_5h = 5 * frames_per_hour
    ax.set_xlim(lower_bound, upper_bound)
    ax.set_ylim(bottom=0, top=69)
    xticks = np.arange(lower_bound, upper_bound + 1, frames_per_5h)
    if len(xticks) == 0:
        xticks = np.array([lower_bound, upper_bound])
    ax.set_xticks(xticks)
    ax.set_xticklabels([int(t / frames_per_hour) for t in xticks], color=tick_color)

    if treatment_window is not None:
        ax.axvspan(treatment_window[0] - 1, treatment_window[1] - 1, color="#bfbfbf", alpha=1, linewidth=0)

    ax.plot(x, y, linewidth=1.5, color=line_color, label="≤ threshold")
    ax.set_xlabel("Time (h)", color=label_color, fontsize=7)
    ax.set_ylabel(y_label, color=label_color, fontsize=7)

    plt.tight_layout()
    if save_path:
        os.makedirs(OUTPUT_ROOT, exist_ok=True)
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
    identifiers: list[str],
) -> dict:
    """Compute per-identifier first breach frames for non-recovering clones."""
    req = {"unique_particle", "frame", "distance_to_edge"}
    if not req.issubset(df.columns):
        missing = req - set(df.columns)
        raise ValueError(f"df missing columns: {missing}")

    out = {ident: {"first_fail_frames": np.array([], dtype=int), "n_clones": 0} for ident in identifiers}

    d = df.sort_values(["unique_particle", "frame"]).copy()
    d["dist_um"] = d["distance_to_edge"] * scale_factor

    for df_id, g_id in d.groupby(d["unique_particle"] // 10000, sort=False):
        df_id = int(df_id)
        if not (0 <= df_id < len(identifiers)):
            continue
        ident = identifiers[df_id]

        n_clones = g_id["unique_particle"].nunique()
        failure_first_frames = []

        for _uid, g in g_id.groupby("unique_particle", sort=False):
            f = g["frame"].to_numpy()
            y = g["dist_um"].to_numpy()

            breach_bool = (y <= breach_threshold_um)
            breach_idx = None
            if min_breach_frames <= 1:
                if breach_bool.any():
                    breach_idx = int(np.where(breach_bool)[0][0])
            else:
                run = np.convolve(
                    breach_bool.astype(int),
                    np.ones(min_breach_frames, dtype=int),
                    mode="valid",
                )
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
                    run_rec = np.convolve(
                        rec_bool.astype(int),
                        np.ones(min_recovery_frames, dtype=int),
                        mode="valid",
                    )
                    recovered = (run_rec >= min_recovery_frames).any()

            if not recovered:
                failure_first_frames.append(int(f[breach_idx]))

        out[ident] = {"first_fail_frames": np.sort(np.asarray(failure_first_frames, dtype=int)), "n_clones": int(n_clones)}

    return out


def cumulative_series_from_firsts(first_fail_frames: np.ndarray, frame_grid: np.ndarray) -> np.ndarray:
    """Step function: at each t in frame_grid, how many events occurred at or before t?"""
    if first_fail_frames.size == 0:
        return np.zeros_like(frame_grid, dtype=int)
    return np.searchsorted(first_fail_frames, frame_grid, side="right")


def aggregate_median_iqr_over_time(per_id: dict, frame_grid: np.ndarray, *, normalize: bool = False) -> dict:
    """Compute median and IQR across identifiers at each time point."""
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
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial"],
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "axes.titlepad": 2,
        "axes.linewidth": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.labelpad": 2,
        "legend.frameon": False,
        "legend.fontsize": 6,
        "savefig.dpi": 300,
        "figure.dpi": 300,
    })

    if theme.lower() == "dark":
        fig_face, ax_face = "black", "black"
        spine_color = tick_color = label_color = "white"
        med_color = "#DAA520"
        band_alpha = 0.25
        per_id_color = "#888888"
    else:
        fig_face, ax_face = "white", "white"
        spine_color = tick_color = label_color = "black"
        med_color = "orangered"
        band_alpha = 0.3
        per_id_color = "blue"

    fig, ax = plt.subplots(figsize=(3.55, 1.8), dpi=300, facecolor=fig_face)
    ax.set_facecolor(ax_face)
    for s in ax.spines.values():
        s.set_color(spine_color)
    ax.tick_params(axis="x", colors=tick_color)
    ax.tick_params(axis="y", colors=tick_color)

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
        ylabel = "Median escapes"
    ax.set_ylabel(ylabel, color=label_color, fontsize=7)

    if treatment_window is not None:
        ax.axvspan(treatment_window[0] - 1, treatment_window[1] - 1, color="#bfbfbf", alpha=1, linewidth=0)

    M = agg["matrix"]
    if show_per_identifier and M.size:
        for row in M:
            ax.plot(x, row, lw=0.6, color=per_id_color, alpha=0.6, zorder=1)

    ax.fill_between(x, agg["q1"], agg["q3"], alpha=band_alpha, color=med_color, linewidth=0, zorder=2)
    ax.plot(x, agg["median"], lw=1.6, color=med_color, zorder=3)

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    plt.tight_layout()
    if save_path:
        os.makedirs(OUTPUT_ROOT, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig_face, transparent=True)
        print(f"[ok] Saved median±IQR over time → {save_path}")
    else:
        plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main(cfg: dict):
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    folder_path = os.path.join(INPUT_ROOT, cfg["input_relpath"])
    files, active_identifiers, _ = grab_files(folder_path, cfg["identifiers"])
    if not files:
        raise SystemExit("[error] No files found. Check input_relpath/identifiers.")

    identifiers = active_identifiers
    print("\n[info] Using identifiers with CSV data only:")
    for ident in identifiers:
        print(f"  - {ident}")

    dfs = []
    for i, fp in enumerate(files):
        df = pd.read_csv(fp)
        need = {"particle", "frame", "distance_to_edge"}
        miss = need - set(df.columns)
        if miss:
            raise ValueError(f"{os.path.basename(fp)} is missing columns: {miss}")
        dfs.append(add_unique_identifier(df, i))

    combined = pd.concat(dfs, ignore_index=True)

    combined = filter_by_initial_frame_and_distance(
        combined,
        cfg["min_frame"], cfg["max_frame"],
        cfg["init_dist_min_raw"], cfg["init_dist_max_raw"],
    )

    if combined.empty:
        raise SystemExit("[warn] After filtering, no data remains. Adjust config.")

    frames, cum_counts, all_clones, _failed_ids = compute_first_nonrecovering_crossing_frames(
        combined,
        breach_threshold_um=cfg["breach_threshold_um"],
        recover_threshold_um=cfg["recover_threshold_um"],
        scale_factor=cfg["scale_factor"],
        min_breach_frames=cfg.get("min_breach_frames", 1),
        min_recovery_frames=cfg.get("min_recovery_frames", 1),
    )

    if cfg.get("export_csv"):
        out = pd.DataFrame({
            "frame": frames,
            "hour": frames / cfg["frames_per_hour"],
            "cumulative_count": cum_counts,
            "cumulative_percent": (cum_counts / all_clones * 100.0) if all_clones > 0 else np.nan,
        })
        out.to_csv(cfg["export_csv"], index=False)
        print(f"[ok] Saved data → {cfg['export_csv']}")

    plot_cumulative(
        frames, cum_counts, all_clones,
        frames_per_hour=cfg["frames_per_hour"],
        lower_bound=cfg["lower_bound"],
        upper_bound=cfg["upper_bound"],
        treatment_window=cfg["treatment_window"],
        normalize=cfg["normalize"],
        theme=cfg["theme"],
        save_path=cfg["save_path"],
        figsize=cfg["figsize"],
    )

    frame_grid = np.arange(cfg["lower_bound"], cfg["upper_bound"] + 1, dtype=int)

    per_id = first_failure_frames_per_identifier(
        combined,
        breach_threshold_um=cfg["breach_threshold_um"],
        recover_threshold_um=cfg["recover_threshold_um"],
        scale_factor=cfg["scale_factor"],
        min_breach_frames=cfg.get("min_breach_frames", 1),
        min_recovery_frames=cfg.get("min_recovery_frames", 1),
        identifiers=identifiers,
    )

    agg = aggregate_median_iqr_over_time(per_id, frame_grid, normalize=cfg.get("normalize", False))

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
        ylabel=ylabel,
        lower_bound=cfg["lower_bound"],
        upper_bound=cfg["upper_bound"],
    )


if __name__ == "__main__":
    main(CONFIG)