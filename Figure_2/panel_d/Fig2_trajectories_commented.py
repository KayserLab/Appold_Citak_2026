#!/usr/bin/env python3
"""
Unified clone analysis:
- load CSVs
- filter clones (vectorfield-style filtering)
- classify trajectories as "Escaped" vs "Confined"
- plot trajectories panel

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


# ────────────────────────── CONFIG ──────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_ROOT = os.path.join(SCRIPT_DIR, "Input_files")
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "Output_files")

CONFIG = {
    # IO
    "input_relpath": "Continuous_therapy",
    "identifiers": [
        "P1_", "P2", "P3", "P4", "P5", "P6", "P7", "P8",
        "P9", "P10", "P11", "P12", "P13", "P14", "P15", "P16",
    ],

    # Output paths (saved into Output_files/)
    "save_traj_pdf": os.path.join(OUTPUT_ROOT, "Fig2_panel_d_trajectories.pdf"),

    # Scale (px → µm)
    "scale_factor_um_per_px": 8.648,

    # Filtering (vectorfield-style filtering)
    "min_frame_first_seen": 50,
    "max_frame_first_seen": 200,
    "min_dist_px_first": 2,
    "max_dist_px_first": 57,  # ca. 500 µm / 8.648 µm/px
    "min_track_len": 20,

    # Classification thresholds (µm)
    "breach_threshold_um": 20.0,
    "recover_threshold_um": 80.0,

    # Plot styling for trajectories
    "theme": "bright",  # "bright" or "dark"
    "treatment_window_frames": (37, 220),
    "x_lower_bound_frames": 0,
    "x_upper_bound_frames": 200,
    "y_upper_bound_um": 500,
    "frames_per_hour": 2,  # 2 frames per hour = 30 min per frame
}


# ────────────────────────── STYLE HELPER ──────────────────────────
def apply_nature_style(theme: str):
    """Apply Nature-ish rcParams for consistent typography."""
    text_color = "white" if theme.lower() == "dark" else "black"

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

        "legend.frameon": False,
        "legend.fontsize": 6,

        "savefig.dpi": 300,
        "figure.dpi": 300,
    })

    return text_color


# ────────────────────────── DATA HELPERS ──────────────────────────
def grab_files(folder_path, identifiers):
    """For each identifier like P1_, P2, ... find the first matching clone CSV."""
    file_paths = []
    for identifier in identifiers:
        candidates = [
            f for f in os.listdir(folder_path)
            if identifier in f and f.endswith(".csv") and "clone_" in f
        ]
        if len(candidates) > 1:
            raise ValueError(f"More than one file with identifier {identifier} in {folder_path}")
        if len(candidates) == 0:
            print(f"[grab_files] No file with identifier {identifier}")
            continue
        file_paths.append(os.path.join(folder_path, candidates[0]))
    return file_paths


def add_unique_identifier(df, df_id):
    """Make a dataset-global clone ID: unique_particle = df_id*10000 + particle."""
    out = df.copy()
    out["unique_particle"] = df_id * 10000 + out["particle"]
    return out


def load_and_combine(folder_path, identifiers):
    """Read all clone CSVs and stitch them together with unique_particle IDs."""
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
    Filtering logic:
    1) Keep only particles whose first frame is within [min_frame_first_seen, max_frame_first_seen].
    2) Keep only particles whose distance_to_edge at their first frame is within [min_dist_px_first, max_dist_px_first].
    3) Enforce a minimum track length after filtering.
    """
    df = df.copy()

    first_frame_df = (
        df.groupby("unique_particle")["frame"]
        .min()
        .reset_index()
        .rename(columns={"frame": "first_frame"})
    )

    allowed_particles = first_frame_df[
        (first_frame_df["first_frame"] >= cfg["min_frame_first_seen"])
        & (first_frame_df["first_frame"] <= cfg["max_frame_first_seen"])
    ][["unique_particle", "first_frame"]]

    df = df[df["unique_particle"].isin(allowed_particles["unique_particle"])]

    df_first = df.merge(allowed_particles, on="unique_particle", how="inner")
    df_first = df_first[df_first["frame"] == df_first["first_frame"]]

    in_band_particles = df_first[
        (df_first["distance_to_edge"] >= cfg["min_dist_px_first"])
        & (df_first["distance_to_edge"] <= cfg["max_dist_px_first"])
    ]["unique_particle"].unique()

    df = df[df["unique_particle"].isin(in_band_particles)]

    counts = df.groupby("unique_particle")["frame"].count()
    keep_long_enough = counts[counts >= cfg["min_track_len"]].index
    df = df[df["unique_particle"].isin(keep_long_enough)]

    return df


def classify_escape_status(df, cfg, smoothing=True):
    """
    Per unique_particle, return:
      unique_particle, escaped(bool), classification("Escaped"/"Confined"),
      n_frames, min_distance_um, frame array, and smoothed distance trajectory.

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

        dist_um = g["distance_to_edge"].to_numpy() * scale

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
            recovered = (
                np.any(dist_um[first_breach_idx + 1:] >= recover_thr)
                if first_breach_idx + 1 < len(dist_um)
                else False
            )
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
            "dist_um_smooth": dist_um_smooth,
        })

    return results


# ────────────────────────── PLOTTING ──────────────────────────
def plot_trajectories(results, cfg, text_color):
    """Plot all trajectories, color-coded by classification."""
    if cfg["theme"].lower() == "dark":
        fig_face, ax_face = "black", "black"
        spine_color = tick_color = "white"
        confined_color = "dimgrey"
        escaped_color = "orangered"
    else:
        fig_face, ax_face = "white", "white"
        spine_color = tick_color = "black"
        confined_color = "dimgrey"
        escaped_color = "orangered"

    fig, ax = plt.subplots(figsize=(3.5, 1.8), dpi=300, facecolor=fig_face)
    ax.set_facecolor(ax_face)

    for spine in ax.spines.values():
        spine.set_color(spine_color)
        spine.set_linewidth(0.5)

    for r in results:
        x = r["frame"]
        y = r["dist_um_smooth"]
        if r["classification"] == "Confined":
            ax.plot(x, y, color=confined_color, linewidth=0.5, alpha=1)

    for r in results:
        x = r["frame"]
        y = r["dist_um_smooth"]
        if r["classification"] == "Escaped":
            ax.plot(x, y, color=escaped_color, linewidth=0.8, alpha=1)

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

    ax.tick_params(axis="y", colors=tick_color, labelsize=6)

    t0, t1 = cfg["treatment_window_frames"]
    ax.axvspan(t0 - 1, t1 - 1, color="#bfbfbf", alpha=1, linewidth=0)

    legend_elements = [
        Line2D([0], [0], color=confined_color, lw=1, label="Confined"),
        Line2D([0], [0], color=escaped_color, lw=1, label="Escaped"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower left",
        frameon=False,
        fontsize=6,
        labelcolor=text_color,
    )

    ax.set_xlabel("Time (h)", color=text_color, fontsize=7)
    ax.set_ylabel("Distance to Edge (µm)", color=text_color, fontsize=7)

    plt.tight_layout(pad=0.2)
    return fig, ax


# ────────────────────────── MAIN ──────────────────────────
def main(cfg):
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    folder = os.path.join(INPUT_ROOT, cfg["input_relpath"])
    folder = folder.replace("\\", "/")
    if not folder.endswith("/"):
        folder += "/"

    text_color = apply_nature_style(cfg["theme"])

    df_all = load_and_combine(folder, cfg["identifiers"])
    df_filt = vectorstyle_filter(df_all, cfg)

    if df_filt.empty:
        print("No particles left after filtering. Check thresholds.")
        return

    results = classify_escape_status(df_filt, cfg, smoothing=True)

    print(f"Number of trajectories plotted: {len(results)}")

    fig_traj, _ax_traj = plot_trajectories(results, cfg, text_color)
    if cfg.get("save_traj_pdf"):
        fig_traj.savefig(
            cfg["save_traj_pdf"],
            dpi=300,
            bbox_inches="tight",
            facecolor=fig_traj.get_facecolor(),
            transparent=True,
        )
        print(f"Saved trajectories plot to {cfg['save_traj_pdf']}")
    else:
        plt.show()


if __name__ == "__main__":
    main(CONFIG)