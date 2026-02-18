#!/usr/bin/env python3
"""
Unified clone analysis:
- load CSVs
- filter clones (vectorfield-style filtering)
- classify trajectories as "Escaped" vs "Confined"
- plot (1) trajectories panel
- plot (2) pie chart

This script is meant to be publication-ish:
Helvetica/Arial, thin lines, small font, PDF-friendly.
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter
import shutil


# ────────────────────────── CONFIG ──────────────────────────
CONFIG = {
    # IO
    "folder_path": r"S:\Members\Nico\Experiment_CSV_files\20240917_continuous_dose_2",
    "identifiers": ['P1_', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8',
                    'P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16'],

    # Output paths
    "save_traj_pdf": r"C:\Users\nappold\Desktop\Manuscript Figures\Fig2\Trajectories_CD_long_18H.pdf",
    "save_pie_pdf": r"C:\Users\nappold\Desktop\Manuscript Figures\Fig2\Piechart_CD_long_18H.pdf",

    # Scale (px → µm)
    "scale_factor_um_per_px": 8.648,

    # Filtering (this matches the pipeline you said 'worked better')
    # 1. keep only particles whose FIRST frame is between these:
    "min_frame_first_seen": 36,
    "max_frame_first_seen": 200,
    # 2. Keep only particles that are (at any time) in this px distance band:
    "min_dist_px_first": 2,
    "max_dist_px_first": 57,  # ca. 500 µm / 8.648 µm/px
    # 3. Require at least this many rows after filtering:
    "min_track_len": 20,

    # Classification thresholds (µm)
    # breach = went this close or closer at least once
    "breach_threshold_um": 20.0,
    # recovery = after breach, did it ever go back out at/above this?
    "recover_threshold_um": 80.0,

    # Plot styling for trajectories
    "theme": "bright",  # "bright" or "dark"
    "treatment_window_frames": (37, 220),  # shaded span in frames
    "x_lower_bound_frames": 0,
    "x_upper_bound_frames": 200,
    "y_upper_bound_um": 500,  # y-limit for trajectories plot (µm)
    "frames_per_hour": 2,  # 2 frames per hour = 30 min per frame
}


# ────────────────────────── STYLE HELPER ──────────────────────────
def apply_nature_style(theme: str):
    """
    Apply Nature-ish rcParams for consistent typography.
    """
    if theme.lower() == "dark":
        text_color = "white"
    else:
        text_color = "black"

    mpl.rcParams.update({
        'pdf.fonttype': 42,    # embed TrueType fonts
        'ps.fonttype': 42,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial'],

        'font.size': 7,
        'axes.titlesize': 7,
        'axes.labelsize': 7,
        #adjust padding
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

        'legend.frameon': False,
        'legend.fontsize': 6,

        'savefig.dpi': 300,
        'figure.dpi': 300,

        # We won't globally kill top/right spines because for you we want the full box.
        # You can still turn them off per-axis if you want.
    })

    return text_color


# ────────────────────────── DATA HELPERS ──────────────────────────
def grab_files(folder_path, identifiers):
    """
    For each identifier like P1_, P2, ... find the first matching clone CSV.
    """
    file_paths = []
    for identifier in identifiers:
        candidates = [f for f in os.listdir(folder_path)
                      if identifier in f and f.endswith('.csv') and "clone_" in f]
        if len(candidates) > 1:
            raise ValueError(f"More than one file with identifier {identifier} in {folder_path}")
        if len(candidates) == 0:
            print(f"[grab_files] No file with identifier {identifier}")
            continue
        file_paths.append(os.path.join(folder_path, candidates[0]))
    return file_paths


def add_unique_identifier(df, df_id):
    """
    Make a dataset-global clone ID: unique_particle = df_id*10000 + particle
    """
    out = df.copy()
    out['unique_particle'] = df_id * 10000 + out['particle']
    return out


def load_and_combine(folder_path, identifiers):
    """
    Read all clone CSVs and stitch them together with unique_particle IDs.
    """
    dfs = []
    file_paths = grab_files(folder_path, identifiers)
    for i, path in enumerate(file_paths):
        df = pd.read_csv(path)
        df = add_unique_identifier(df, i)
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No CSVs found for given identifiers.")
    return pd.concat(dfs, ignore_index=True)


def vectorstyle_filter(df, cfg):
    """
    Updated filtering logic:

    1. Get each particle's FIRST observed frame.
    2. Keep only particles whose first frame is between [min_frame_first_seen, max_frame_first_seen].
    3. From those, keep only particles whose distance_to_edge AT THEIR FIRST FRAME is within:
       min_dist_px_anytime ≤ distance_to_edge ≤ max_dist_px_anytime (in px).
    4. Enforce a minimum track length.
    """
    df = df.copy()

    # ── 1) first frame each particle appears
    first_frame_df = (
        df.groupby('unique_particle')['frame']
          .min()
          .reset_index()
          .rename(columns={'frame': 'first_frame'})
    )

    # ── 2) keep only particles that start in the time window
    allowed_particles = first_frame_df[
        (first_frame_df['first_frame'] >= cfg["min_frame_first_seen"]) &
        (first_frame_df['first_frame'] <= cfg["max_frame_first_seen"])
    ][['unique_particle', 'first_frame']]

    # Restrict df to those particles early (faster and avoids surprises)
    df = df[df['unique_particle'].isin(allowed_particles['unique_particle'])]

    # ── 3) distance band check at the FIRST frame (per particle)
    # Merge first_frame onto df, then pick only rows that correspond to that first frame.
    df_first = df.merge(allowed_particles, on='unique_particle', how='inner')
    df_first = df_first[df_first['frame'] == df_first['first_frame']]

    # Now apply the distance band condition to those first-frame rows
    in_band_particles = df_first[
        (df_first['distance_to_edge'] >= cfg["min_dist_px_first"]) &
        (df_first['distance_to_edge'] <= cfg["max_dist_px_first"])
    ]['unique_particle'].unique()

    df = df[df['unique_particle'].isin(in_band_particles)]

    # ── 4) enforce min track length (after filtering)
    counts = df.groupby('unique_particle')['frame'].count()
    keep_long_enough = counts[counts >= cfg["min_track_len"]].index
    df = df[df['unique_particle'].isin(keep_long_enough)]

    return df



def classify_escape_status(df, cfg, smoothing=True):
    """
    Per unique_particle, return:
    unique_particle, escaped(bool), classification("Escaped"/"Confined"),
    n_frames, min_distance_um

    Escape logic:
    - scale px → µm
    - breach = ever <= breach_threshold_um
    - recovered = after first breach, ever >= recover_threshold_um
    - Escaped = breached AND NOT recovered
    - Confined otherwise
    """
    df = df.copy()
    scale = cfg["scale_factor_um_per_px"]

    results = []
    for uid, g in df.groupby("unique_particle"):
        g = g.sort_values("frame")

        # distance in µm over time
        dist_um = g["distance_to_edge"].to_numpy() * scale

        # optional smoothing just for nicer plotting (not classification)
        if smoothing and len(dist_um) >= 5:
            wl = min(11, len(dist_um) if len(dist_um) % 2 else len(dist_um) - 1)
            dist_um_smooth = savgol_filter(dist_um, window_length=wl, polyorder=2)
        else:
            dist_um_smooth = dist_um.copy()

        breach_thr = cfg["breach_threshold_um"]
        recover_thr = cfg["recover_threshold_um"]

        breached_any = np.any(dist_um <= breach_thr)
        if breached_any:
            first_breach_idx = np.where(dist_um <= breach_thr)[0][0]
            recovered = np.any(dist_um[first_breach_idx + 1:] >= recover_thr) \
                        if first_breach_idx + 1 < len(dist_um) else False
            if recovered:
                classification = "Confined"
                escaped = False
            else:
                classification = "Escaped"
                escaped = True
        else:
            classification = "Confined"
            escaped = False

        results.append({
            "unique_particle": uid,
            "escaped": escaped,
            "classification": classification,
            "n_frames": len(g),
            "min_distance_um": float(np.min(dist_um)),
            "frame": g["frame"].to_numpy(),
            "dist_um_smooth": dist_um_smooth
        })

    return results


# ────────────────────────── PLOTTING ──────────────────────────
def plot_trajectories(results, cfg, text_color):
    """
    Plot all trajectories (no arrows anymore),
    color-coded by classification.
    """
    # theme colors
    if cfg["theme"].lower() == "dark":
        fig_face, ax_face = "black", "black"
        spine_color = "white"
        tick_color = "white"
        confined_color = "dimgrey"
        escaped_color = "orangered"
    else:
        fig_face, ax_face = "white", "white"
        spine_color = "black"
        tick_color = "black"
        confined_color = "dimgrey"
        escaped_color = "orangered"

    fig, ax = plt.subplots(figsize=(3.5, 1.8), dpi=300, facecolor=fig_face)
    ax.set_facecolor(ax_face)

    # box frame in Nature-ish thin lines
    for spine in ax.spines.values():
        spine.set_color(spine_color)
        spine.set_linewidth(0.5)

    # plot confined first then escaped last so orange is on top
    for r in results:
        x = r["frame"]
        y = r["dist_um_smooth"]  # already in µm
        if r["classification"] == "Confined":
            ax.plot(x, y, color=confined_color, linewidth=0.5, alpha=1)
    for r in results:
        x = r["frame"]
        y = r["dist_um_smooth"]
        if r["classification"] == "Escaped":
            ax.plot(x, y, color=escaped_color, linewidth=0.8, alpha=1)

    # x-axis as hours
    frames_per_hour = cfg["frames_per_hour"]
    lb = cfg["x_lower_bound_frames"]
    ub = cfg["x_upper_bound_frames"]

    ax.set_xlim(lb, ub)
    ax.set_ylim(0, cfg["y_upper_bound_um"])

    frames_per_5h = 5 * frames_per_hour
    ticks = np.arange(lb, ub + 1, frames_per_5h)
    hour_labels = [int(t / frames_per_hour) for t in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(hour_labels, color=tick_color, fontsize=6)

    # y ticks color
    ax.tick_params(axis="y", colors=tick_color, labelsize=6)

    # shaded treatment window
    t0, t1 = cfg["treatment_window_frames"]
    ax.axvspan(t0 - 1, t1 - 1, color='#bfbfbf', alpha=1, linewidth=0)

    # legend
    legend_elements = [
        Line2D([0], [0], color=confined_color, lw=1, label="Confined"),
        Line2D([0], [0], color=escaped_color, lw=1, label="Escaped")
    ]
    ax.legend(handles=legend_elements,
              loc="lower left",
              frameon=False,
              fontsize=6,
              labelcolor=text_color)

    # labels
    ax.set_xlabel("Time (h)", color=text_color, fontsize=7)
    ax.set_ylabel("Distance to Edge (µm)", color=text_color, fontsize=7)

    plt.tight_layout(pad=0.2)
    return fig, ax


def plot_pie(results, cfg, text_color):
    """
    Pie chart of Escaped vs Confined.
    Uses the same classification results passed to plot_trajectories.
    """
    # count classes
    escaped_count = sum(r["escaped"] for r in results)
    confined_count = len(results) - escaped_count

    if escaped_count + confined_count == 0:
        print("[plot_pie] nothing to plot")
        return None, None

    # color mapping matches trajectories
    confined_color = "dimgrey"
    escaped_color = "orangered"

    fig_face = "black" if cfg["theme"].lower() == "dark" else "white"

    fig, ax = plt.subplots(figsize=(0.6, 0.6), dpi=300, facecolor=fig_face)
    ax.set_facecolor(fig_face)

    values = [confined_count, escaped_count]
    labels = ["Confined", "Escaped"]
    colors = [confined_color, escaped_color]

    wedges, _texts, _auto = ax.pie(
        values,
        labels=None,
        autopct=lambda pct: '',  # we'll do manual labels for nicer control
        startangle=90,
        counterclock=False,
        colors=colors,
        wedgeprops=dict(linewidth=0.6, edgecolor=fig_face)
    )

    total = sum(values)
    # place our custom text a bit inside each wedge
    for i, (wedge, val) in enumerate(zip(wedges, values)):
        ang = 0.5 * (wedge.theta2 + wedge.theta1)
        r = 0.55  # radial position of label
        x = np.cos(np.deg2rad(ang)) * r
        y = np.sin(np.deg2rad(ang)) * r
        pct = 100.0 * val / total
        ax.text(x, y,
                f"{pct:.0f}%",
                ha="center", va="center",
                color="white", fontsize=6, weight="regular")

    ax.set_aspect("equal")
    plt.tight_layout(pad=0.2)
    return fig, ax


# ────────────────────────── MAIN ──────────────────────────
def main(cfg):
    # normalize path
    folder = cfg["folder_path"].replace('\\', '/')
    if not folder.endswith('/'):
        folder += '/'

    # apply style
    text_color = apply_nature_style(cfg["theme"])

    # load
    df_all = load_and_combine(folder, cfg["identifiers"])

    # filter once (vectorfield-style)
    df_filt = vectorstyle_filter(df_all, cfg)

    if df_filt.empty:
        print("No particles left after filtering. Check thresholds.")
        return

    # classify (gives us both plotting y-values and escape labels)
    results = classify_escape_status(df_filt, cfg, smoothing=True)

    print(f"Number of trajectories plotted: {len(results)}")

    # (1) trajectories plot
    fig_traj, ax_traj = plot_trajectories(results, cfg, text_color)
    if cfg.get("save_traj_pdf"):
        fig_traj.savefig(cfg["save_traj_pdf"],
                         dpi=300,
                         bbox_inches="tight",
                         facecolor=fig_traj.get_facecolor(),
                         transparent=True)
        print(f"Saved trajectories plot to {cfg['save_traj_pdf']}")

    # (2) pie plot
    fig_pie, ax_pie = plot_pie(results, cfg, text_color)
    if fig_pie is not None and cfg.get("save_pie_pdf"):
        fig_pie.savefig(cfg["save_pie_pdf"],
                        dpi=300,
                        bbox_inches="tight",
                        facecolor=fig_pie.get_facecolor(),
                        transparent=True)
        print(f"Saved pie chart to {cfg['save_pie_pdf']}")

    # save a copy of this script alongside, like you like to do
    try:
        code_out = os.path.splitext(cfg["save_traj_pdf"])[0] + "_code.txt"
        this_script = os.path.abspath(__file__)
        shutil.copy(this_script, code_out)
        print(f"Code copied to {code_out}")
    except NameError:
        print("Running in environment without __file__; not copying script.")


if __name__ == "__main__":
    main(CONFIG)
