#!/usr/bin/env python3
"""
Initial distance at first detection for all clones (no escape filter).

Workflow
1) Load one clone CSV per identifier.
2) Keep only clones tracked for >= min_duration_frames detections.
3) For each surviving clone, record:
      init_frame   = first frame observed
      init_dist_um = distance_to_edge at that first frame (scaled to µm)
4) Bin clones by init_frame into n_bins between [lower_bound, upper_bound].
5) For each bin, compute median and IQR of init_dist_um.
6) Create an unbinned scatter plot of init_dist_um vs time (hours), overlay:
      - treatment window shading
      - rolling median and IQR band
      - ReLU-style linear fit on the active region
"""

import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.optimize import curve_fit
from scipy.stats import pearsonr


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_ROOT = os.path.join(SCRIPT_DIR, "Input_files")
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "Output_files")

CONFIG = {
    # Folder inside Input_files/ containing the clone CSVs
    "input_relpath": os.path.join("Continuous_therapy"),

    "identifiers": [
        "P1_", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12", "P13", "P14", "P15", "P16",
    ],

    # Outputs go directly into Output_files/
    "save_path": os.path.join(OUTPUT_ROOT, "init_distance_at_first_detection_binned_scatter_CT.pdf"),
    "export_csv": None,

    # Filtering
    "min_duration_frames": 10,

    # Plot / binning
    "lower_bound": 70,
    "upper_bound": 230,
    "frames_per_hour": 2,
    "n_bins": 12,
    "treatment_window": (37, 300),
    "theme": "bright",          # "bright" or "dark"
    "plot_mode": "scatter",     # kept for compatibility (not used in current plotting)
    "point_alpha": 0.4,
    "point_size": 6,
    "jitter_width": 0.3,        # kept for compatibility (not used in current plotting)
    "identifier_save": "CT",

    # Units
    "scale_factor": 8.648,      # raw distance_to_edge → µm
}


# ──────────────────────────────────────────────────────────────────────────────
# Utilities (fit + rolling stats)
# ──────────────────────────────────────────────────────────────────────────────
def relu_model(t, t0, m):
    """ReLU with zero baseline: y(t) = max(0, m * (t - t0))"""
    return np.maximum(0, m * (t - t0))


def rolling_median(x_h, y, window_h=5.0):
    """Local/rolling median of y vs x_h using a ±window_h/2 neighborhood."""
    x_sorted = np.asarray(x_h, dtype=float)
    y_sorted = np.asarray(y, dtype=float)
    order = np.argsort(x_sorted)
    x_sorted = x_sorted[order]
    y_sorted = y_sorted[order]

    half = window_h / 2.0
    x_out = []
    y_out = []

    for xi in x_sorted:
        mask = (x_sorted >= xi - half) & (x_sorted <= xi + half)
        if np.any(mask):
            x_out.append(xi)
            y_out.append(np.median(y_sorted[mask]))

    return np.asarray(x_out), np.asarray(y_out)


def rolling_percentiles(x_h, y, window_h=5.0, q_low=25, q_high=75):
    """Local/rolling percentiles of y vs x_h using a ±window_h/2 neighborhood."""
    x_sorted = np.asarray(x_h, dtype=float)
    y_sorted = np.asarray(y, dtype=float)
    order = np.argsort(x_sorted)
    x_sorted = x_sorted[order]
    y_sorted = y_sorted[order]

    half = window_h / 2.0
    x_out = []
    qlow_out = []
    qhigh_out = []

    for xi in x_sorted:
        mask = (x_sorted >= xi - half) & (x_sorted <= xi + half)
        if np.any(mask):
            x_out.append(xi)
            qlow_out.append(np.percentile(y_sorted[mask], q_low))
            qhigh_out.append(np.percentile(y_sorted[mask], q_high))

    return np.asarray(x_out), np.asarray(qlow_out), np.asarray(qhigh_out)


def fit_relu_to_points_active(x_h, y_um, p0=(70.0, 20.0), n_iter=5):
    """
    Iterative ReLU fit:
      - initialize (t0, m)
      - repeatedly fit only points with x >= current t0
    """
    x_h = np.asarray(x_h, float)
    y_um = np.asarray(y_um, float)

    bounds_lower = [0.0, 0.0]
    bounds_upper = [200.0, 200.0]

    t0, m = p0
    for _ in range(n_iter):
        sel = x_h >= t0
        if sel.sum() < 5:
            break

        popt, _ = curve_fit(
            relu_model,
            x_h[sel], y_um[sel],
            p0=[t0, m],
            bounds=(bounds_lower, bounds_upper),
            maxfev=20000,
        )
        t0, m = popt

    return float(t0), float(m)


def plot_initial_distance_scatter_with_relu(
    init_df,
    *,
    frames_per_hour=2,
    theme="bright",
    treatment_window=(73, 230),
    point_size=3,
    point_alpha=0.6,
    rolling_window_h=5.0,
    save_path=None,
):
    """
    Scatter plot of distance at first detection vs time, with:
      - treatment window shading
      - rolling median and IQR band (for x >= fitted t0)
      - ReLU-style fit line (for x >= fitted t0)
    """
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial"],
        "font.size": 7,
        "axes.linewidth": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
    })

    if theme.lower() == "dark":
        fig_face = ax_face = "black"
        spine_color = tick_color = label_color = "white"
        dot_color = "goldenrod"
        median_color = "darkgoldenrod"
        fit_color = "white"
        iqr_fill = "darkgoldenrod"
    else:
        fig_face = ax_face = "white"
        spine_color = tick_color = label_color = "black"
        dot_color = "goldenrod"
        median_color = "dimgray"
        fit_color = "black"
        iqr_fill = "dimgray"

    fig, ax = plt.subplots(figsize=(3, 1.8), dpi=300)
    fig.patch.set_facecolor(fig_face)
    ax.set_facecolor(ax_face)
    for s in ax.spines.values():
        s.set_color(spine_color)
    ax.tick_params(axis="x", colors=tick_color)
    ax.tick_params(axis="y", colors=tick_color)
    ax.tick_params(labelsize=6)

    x_h_all = init_df["init_frame"].values / frames_per_hour
    y_um_all = init_df["init_dist_um"].values

    # Keep identical behavior to original script
    mask = y_um_all > 100
    x_h, y_um = x_h_all[mask], y_um_all[mask]

    if treatment_window is not None:
        t_start_h = treatment_window[0] / frames_per_hour
        t_end_h = treatment_window[1] / frames_per_hour
        ax.axvspan(t_start_h - 0.5, t_end_h - 0.5, color="#bfbfbf", alpha=1, linewidth=0, zorder=0)

    ax.scatter(x_h, y_um, s=point_size, color=dot_color, alpha=point_alpha, edgecolors="none", zorder=2)

    x_trend_raw, y_trend_raw = rolling_median(x_h, y_um, window_h=rolling_window_h)
    x_iqr_raw, y_q25_raw, y_q75_raw = rolling_percentiles(x_h, y_um, window_h=rolling_window_h, q_low=25, q_high=75)

    mask_t = (x_trend_raw >= 0) & (x_trend_raw <= 230)
    x_trend_raw, y_trend_raw = x_trend_raw[mask_t], y_trend_raw[mask_t]

    mask_i = (x_iqr_raw >= 0) & (x_iqr_raw <= 230)
    x_iqr_raw, y_q25_raw, y_q75_raw = x_iqr_raw[mask_i], y_q25_raw[mask_i], y_q75_raw[mask_i]

    t0_fit, m_fit = np.nan, np.nan
    r_val = np.nan

    if len(x_trend_raw) > 5:
        t0_fit, m_fit = fit_relu_to_points_active(x_h, y_um, p0=(70.0, 20.0), n_iter=5)

        sel = x_trend_raw >= t0_fit
        x_trend, y_trend = x_trend_raw[sel], y_trend_raw[sel]

        ax.fill_between(
            x_iqr_raw[x_iqr_raw >= t0_fit],
            y_q25_raw[x_iqr_raw >= t0_fit],
            y_q75_raw[x_iqr_raw >= t0_fit],
            color=iqr_fill,
            alpha=0.3,
            linewidth=0,
            zorder=3,
        )

        ax.plot(x_trend, y_trend, color=median_color, lw=1.2, alpha=1, zorder=4, label="Median")

        x_fit = np.linspace(t0_fit, 230, 300)
        y_fit = relu_model(x_fit, t0_fit, m_fit)

        y_fit_interp = relu_model(x_trend, t0_fit, m_fit)
        if len(x_trend) > 2:
            r_val, _ = pearsonr(y_trend, y_fit_interp)

        ax.plot(
            x_fit, y_fit,
            color=fit_color, lw=1.0, linestyle="--", zorder=5,
            label=(f"Linear fit (r={r_val:.2f}, t0={t0_fit:.1f} h, m={m_fit:.1f} µm/h)"),
        )

    ax.set_xlim(0, 130)
    ax.set_ylim(bottom=0, top=np.max(y_um) * 1.05 if len(y_um) else 1)

    ax.set_xlabel("Time (h)", color=label_color, fontsize=7, labelpad=2)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, pos: f"{v / 1000:.1f}"))
    ax.yaxis.set_major_locator(mtick.MultipleLocator(500))
    ax.set_ylabel("Distance at first detection (mm)", color=label_color, fontsize=7, labelpad=2)
    ax.legend(frameon=False, fontsize=6, loc="upper left")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig_face, transparent=True)
        print(f"[ok] Saved scatter+LinearFit → {save_path}")
    else:
        plt.show()

    return {"t0_h": t0_fit, "slope_um_per_h": m_fit, "pearson_r": r_val}


def grab_files(folder_path, identifiers):
    """Return first CSV per identifier that contains 'clone_' in the filename."""
    folder_path = folder_path.replace("\\", "/")
    if folder_path and folder_path[-1] != "/":
        folder_path += "/"

    file_paths = []
    for identifier in identifiers:
        matches = [f for f in os.listdir(folder_path) if identifier in f and f.endswith(".csv") and "clone_" in f]
        if len(matches) == 0:
            print(f"[warn] No CSV with identifier '{identifier}' found in {folder_path}")
            continue
        if len(matches) > 1:
            raise ValueError(f"More than one file with identifier '{identifier}' found.")
        file_paths.append(os.path.join(folder_path, matches[0]))
    return file_paths, folder_path


def add_unique_identifier(df, df_id):
    """unique_particle = df_id*10000 + particle"""
    if "particle" not in df.columns:
        raise ValueError("CSV must contain a 'particle' column.")
    df = df.copy()
    df["unique_particle"] = df_id * 10000 + df["particle"]
    return df


def extract_initial_distances(df, *, min_duration_frames, scale_factor):
    """
    From combined dataframe with columns [unique_particle, frame, distance_to_edge]:
      - keep clones with >= min_duration_frames detections
      - for each kept clone, return first detection frame and initial distance (µm)

    Returns a DataFrame with columns: ['unique_particle','init_frame','init_dist_um'].
    """
    req = {"unique_particle", "frame", "distance_to_edge"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"df missing columns: {missing}")

    d = df[["unique_particle", "frame", "distance_to_edge"]].copy()
    d["init_dist_um_tmp"] = d["distance_to_edge"] * scale_factor

    counts = d.groupby("unique_particle")["frame"].count()
    good_ids = counts[counts >= min_duration_frames].index

    if len(good_ids) == 0:
        return pd.DataFrame(columns=["unique_particle", "init_frame", "init_dist_um"])

    kept = d[d["unique_particle"].isin(good_ids)]

    first_rows = (
        kept.sort_values(["unique_particle", "frame"])
        .groupby("unique_particle")
        .first()
        .reset_index()
    )
    out = first_rows.rename(columns={
        "frame": "init_frame",
        "init_dist_um_tmp": "init_dist_um",
    })
    return out[["unique_particle", "init_frame", "init_dist_um"]]


def bin_by_frames_distlist(init_df, *, n_bins, frame_min=None, frame_max=None):
    """
    Bin initial detections along init_frame into n_bins equal-width bins.
    For each bin, store the full distance list and compute median and IQR.
    """
    if init_df.empty:
        return np.array([]), pd.DataFrame(columns=[
            "bin", "bin_center", "dist_list", "median_um", "q25_um", "q75_um", "n_clones"
        ])

    fmin = int(init_df["init_frame"].min()) if frame_min is None else int(frame_min)
    fmax = int(init_df["init_frame"].max()) if frame_max is None else int(frame_max)
    if fmax <= fmin:
        fmax = fmin + 1

    bins = np.linspace(fmin, fmax, n_bins + 1)
    bin_idx = np.digitize(init_df["init_frame"], bins) - 1

    dfc = init_df.copy()
    dfc["bin"] = bin_idx

    grouped = dfc.groupby("bin")["init_dist_um"].apply(list)

    rows = []
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    for b, dist_list in grouped.items():
        dist_arr = np.array(dist_list, dtype=float)
        rows.append({
            "bin": b,
            "bin_center": bin_centers[b] if 0 <= b < len(bin_centers) else np.nan,
            "dist_list": dist_arr,
            "median_um": np.median(dist_arr),
            "q25_um": np.percentile(dist_arr, 25),
            "q75_um": np.percentile(dist_arr, 75),
            "n_clones": len(dist_arr),
        })

    agg_df = pd.DataFrame(rows).sort_values("bin").reset_index(drop=True)
    return bin_centers, agg_df


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main(cfg: dict):
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    folder_path = os.path.join(INPUT_ROOT, cfg["input_relpath"])
    files, _ = grab_files(folder_path, cfg["identifiers"])
    if not files:
        raise SystemExit("[error] No files found. Check input_relpath/identifiers.")

    dfs = []
    for i, fp in enumerate(files):
        df = pd.read_csv(fp)
        need = {"particle", "frame", "distance_to_edge"}
        miss = need - set(df.columns)
        if miss:
            raise ValueError(f"{os.path.basename(fp)} missing columns: {miss}")
        dfs.append(add_unique_identifier(df, i))

    combined = pd.concat(dfs, ignore_index=True)

    init_df = extract_initial_distances(
        combined,
        min_duration_frames=cfg["min_duration_frames"],
        scale_factor=cfg["scale_factor"],
    )
    if init_df.empty:
        raise SystemExit("[warn] No clones met min_duration_frames. Adjust config.")

    bin_centers, agg = bin_by_frames_distlist(
        init_df,
        n_bins=cfg["n_bins"],
        frame_min=cfg["lower_bound"],
        frame_max=cfg["upper_bound"],
    )

    print("\n=== Aggregated bin data (median/IQR) ===")
    print(agg[["bin", "bin_center", "median_um", "q25_um", "q75_um", "n_clones"]])
    print("bin_centers (hours):", bin_centers / cfg["frames_per_hour"])
    print("=======================================\n")

    if cfg.get("export_csv"):
        out = agg.copy()
        out["hour_center"] = out["bin_center"] / cfg["frames_per_hour"]
        out.to_csv(cfg["export_csv"], index=False)
        print(f"[ok] Saved data → {cfg['export_csv']}")

    ident = str(cfg["identifier_save"])
    save_path = os.path.join(OUTPUT_ROOT, f"init_distance_scatter_relu_{ident}.pdf")

    fit_info = plot_initial_distance_scatter_with_relu(
        init_df,
        frames_per_hour=cfg["frames_per_hour"],
        theme=cfg["theme"],
        treatment_window=cfg["treatment_window"],
        point_size=5,
        point_alpha=0.6,
        rolling_window_h=5.0,
        save_path=save_path,
    )

    print("ReLU fit:", fit_info)


if __name__ == "__main__":
    main(CONFIG)