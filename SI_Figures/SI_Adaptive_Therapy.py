#!/usr/bin/env python3
"""
Reduced plotting script (3 separate transparent figures):
1) Clone-adjusted total area (mm²)
2) Resistant/Total = clone_area / adjusted_colony_area
3) Highlight spans strip (same figure size as the other plots)

- x-axis: 0..125 hours, ticks every 25 hours
- highlight_spans are in FRAMES and are shifted by -1 frame (= -0.5 h) for plotting
- prints for each display name which identifier is the "median colony"
  (closest to median adjusted area at the last common timepoint in range)

NOTE: For Illustrator: background is transparent (figure + axes).
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ────────────────────────── Style ──────────────────────────

def set_nature_style():
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 7,
        'axes.titlesize': 7,
        'axes.labelsize': 7,
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
        'lines.linewidth': 1.0,
        'savefig.dpi': 300,
        'figure.dpi': 300,
        'axes.spines.top': True,
        'axes.spines.right': True,
    })

def set_time_axis_0_125h(ax):
    ax.set_xlim(0, 125)
    ax.xaxis.set_major_locator(mticker.FixedLocator(np.arange(0, 126, 25)))
    ax.xaxis.set_minor_locator(mticker.FixedLocator(np.arange(0, 126, 5)))

def make_transparent(fig, ax=None):
    """Transparent fig + axes background for Illustrator moving/stacking."""
    fig.patch.set_alpha(0.0)
    if ax is None:
        for a in fig.axes:
            a.set_facecolor((0, 0, 0, 0))
    else:
        ax.set_facecolor((0, 0, 0, 0))


# ────────────────────────── Helpers ──────────────────────────

def _norm(p: str) -> str:
    return os.path.normpath(p).replace("\\", "/").rstrip("/")


def _prep_name_map(cfg):
    raw = cfg.get("display_names", {})
    full = {}
    base = {}
    for k, v in raw.items():
        kn = _norm(k)
        full[kn] = v
        base[os.path.basename(kn)] = v
    return full, base


def _prep_style_map(cfg):
    raw = cfg.get("plot_styles", {})
    full = {}
    base = {}
    for k, v in raw.items():
        kn = _norm(k)
        full[kn] = v
        base[os.path.basename(kn)] = v
    return full, base


def _prep_highlight_map(cfg):
    raw = cfg.get("highlight_spans", {})
    return {_norm(k): v for k, v in raw.items()}


def find_colony_file(folder_path: str, identifier: str) -> str | None:
    """Prefer 'with_extrapolation' if present, else fallback to other colony csvs."""
    if not os.path.isdir(folder_path):
        return None
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(".csv")]

    def is_candidate(name: str) -> bool:
        n = name.lower()
        return ("colony" in n) and (identifier in name)

    candidates = [f for f in files if is_candidate(f)]
    if not candidates:
        return None

    def sort_key(name: str):
        n = name.lower()
        return (
            "with_extrapolation" not in n,  # False first
            "with_clonearea" not in n,      # False first
            name
        )

    candidates.sort(key=sort_key)
    return os.path.join(folder_path, candidates[0])


def load_series_px2(colony_csv: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Returns (colony_area_px2, clone_area_px2, adjusted_colony_area_px2),
    indexed by frame (0..N-1).
    """
    df = pd.read_csv(colony_csv)

    required = ["colony_area", "total_clone_area"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {colony_csv}")

    clone_col = "extrapolated_clone_area" if "extrapolated_clone_area" in df.columns else "total_clone_area"

    idx = np.arange(len(df))
    colony = pd.Series(df["colony_area"].to_numpy(dtype=float), index=idx)
    clone_raw = pd.Series(df["total_clone_area"].to_numpy(dtype=float), index=idx)
    clone = pd.Series(df[clone_col].to_numpy(dtype=float), index=idx)

    # adjusted colony: replace raw clone area by extrapolated clone area (if available)
    adj = colony - clone_raw + clone
    return colony, clone, adj


def frames_to_hours(frames: np.ndarray) -> np.ndarray:
    # 2 frames = 1 hour
    return frames / 2.0


def px2_to_mm2(px2: np.ndarray, um_per_px: float) -> np.ndarray:
    return px2 * (um_per_px ** 2) / 1e6


def _force_spines_black(ax):
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_edgecolor("black")


# ────────────────────────── Core computation ──────────────────────────

def compute_folder_stats(folder: str, identifiers: list[str], cfg: dict):
    """
    Loads all identifiers in folder and returns:
      - exp_name
      - style (color/linestyle/alpha/linewidth)
      - time_hr (shared x for med/q1/q3)
      - adj_total_mm2 (median/q1/q3)
      - resistant_total (median/q1/q3) where resistant_total = clone / adjusted_colony
      - median_identifier
      - highlight spans (frames)
    """
    folder_norm = _norm(folder)
    base = os.path.basename(folder_norm)

    name_full, name_base = _prep_name_map(cfg)
    style_full, style_base = _prep_style_map(cfg)
    hl_map = _prep_highlight_map(cfg)

    exp_name = name_full.get(folder_norm) or name_base.get(base) or re.sub(r"^\d{8}_", "", base)

    style_cfg = style_full.get(folder_norm) or style_base.get(base) or {}
    color = style_cfg.get("color", None)
    linestyle = style_cfg.get("linestyle", "-")
    alpha = float(style_cfg.get("alpha", 1.0))
    linewidth = float(style_cfg.get("linewidth", cfg.get("line_width", 1.0)))

    # highlight spans lookup (full path preferred, then basename key, then display name)
    spans_frames = hl_map.get(folder_norm)
    if spans_frames is None:
        spans_frames = hl_map.get(base)
    if spans_frames is None:
        spans_frames = hl_map.get(exp_name)
    spans_frames = spans_frames or []

    # time window: 0..125 h => frames 0..250
    max_hours = 125
    max_frame = int(max_hours * 2)

    adj_df = pd.DataFrame()
    frac_df = pd.DataFrame()

    for ident in identifiers:
        csv_path = find_colony_file(folder, ident)
        if csv_path is None:
            continue
        try:
            _, clone_s, adj_s = load_series_px2(csv_path)
        except Exception as e:
            print(f"[warn] {exp_name}: failed '{ident}' -> {os.path.basename(csv_path)}: {e}")
            continue

        adj_s = adj_s.loc[:max_frame].copy()
        clone_s = clone_s.loc[:max_frame].copy()

        frac_s = clone_s / adj_s
        frac_s.replace([np.inf, -np.inf], np.nan, inplace=True)

        adj_df[ident] = adj_s
        frac_df[ident] = frac_s

    if adj_df.empty or frac_df.empty:
        return None

    # only frames present across all loaded identifiers (common index)
    common_index = adj_df.dropna(how="any").index.intersection(frac_df.dropna(how="any").index)
    common_index = common_index.sort_values()
    if len(common_index) < max(int(cfg.get("min_frames", 10)), 2):
        print(f"[skip] {exp_name}: too few common frames after filtering.")
        return None

    adj_df = adj_df.loc[common_index]
    frac_df = frac_df.loc[common_index]

    # optional smoothing
    smooth_window = int(cfg.get("smooth_window", 0))
    if smooth_window and smooth_window > 1:
        adj_df = adj_df.rolling(smooth_window, center=True, min_periods=1).median()
        frac_df = frac_df.rolling(smooth_window, center=True, min_periods=1).median()

    # quantiles across identifiers at each timepoint
    adj_med = adj_df.quantile(0.5, axis=1)
    adj_q1 = adj_df.quantile(0.25, axis=1)
    adj_q3 = adj_df.quantile(0.75, axis=1)

    frac_med = frac_df.quantile(0.5, axis=1)
    frac_q1 = frac_df.quantile(0.25, axis=1)
    frac_q3 = frac_df.quantile(0.75, axis=1)

    # x axis
    frames = common_index.to_numpy(dtype=float)
    time_hr = frames_to_hours(frames)

    # adjusted area to mm²
    um_per_px = float(cfg.get("scale_factor", 1.0))
    adj_med_mm2 = px2_to_mm2(adj_med.to_numpy(dtype=float), um_per_px)
    adj_q1_mm2 = px2_to_mm2(adj_q1.to_numpy(dtype=float), um_per_px)
    adj_q3_mm2 = px2_to_mm2(adj_q3.to_numpy(dtype=float), um_per_px)

    # "median colony" identifier at the last common timepoint
    last_frame = common_index.max()
    last_vals = adj_df.loc[last_frame].astype(float)
    med_val = float(last_vals.median())
    median_ident = (last_vals - med_val).abs().idxmin()

    return {
        "exp_name": exp_name,
        "folder_norm": folder_norm,
        "color": color,
        "linestyle": linestyle,
        "alpha": alpha,
        "linewidth": linewidth,
        "time_hr": time_hr,
        "adj_med_mm2": adj_med_mm2,
        "adj_q1_mm2": adj_q1_mm2,
        "adj_q3_mm2": adj_q3_mm2,
        "frac_med": frac_med.to_numpy(dtype=float),
        "frac_q1": frac_q1.to_numpy(dtype=float),
        "frac_q3": frac_q3.to_numpy(dtype=float),
        "median_ident": median_ident,
        "median_frame": int(last_frame),
        "median_hour": float(last_frame) / 2.0,
        "spans_frames": spans_frames,
    }


# ────────────────────────── Plotting (separate figures) ──────────────────────────

def _apply_x_axis(ax):
    ax.set_xlim(0, 125)
    ax.xaxis.set_major_locator(mticker.FixedLocator(np.arange(0, 126, 25)))
    ax.set_xlabel("Hours")


def plot_adjusted_total(entries: list[dict], cfg: dict):
    figsize = cfg.get("figsize", (3.5, 2.5))
    dpi = cfg.get("dpi", 300)
    iqr_alpha = float(cfg.get("iqr_alpha", 0.25))

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    make_transparent(fig, ax)

    for e in entries:
        x = e["time_hr"]
        ax.fill_between(x, e["adj_q1_mm2"], e["adj_q3_mm2"],
                        alpha=iqr_alpha * e["alpha"], color=e["color"], linewidth=0)
        ax.plot(x, e["adj_med_mm2"], e["linestyle"],
                color=e["color"], alpha=e["alpha"], linewidth=e["linewidth"],
                label=e["exp_name"])

    _apply_x_axis(ax)
    ax.set_ylabel("Total area (mm²)")
    ax.set_ylim(bottom=0)
    if cfg.get("ymax") is not None:
        ax.set_ylim(top=float(cfg["ymax"]))
    for h in cfg.get("hline_positions"):
        ax.axvline(x=h, color='black', linestyle=(0, (3, 5, 1, 5)), linewidth=1)

    set_time_axis_0_125h(ax)
    _force_spines_black(ax)
    ax.legend(loc="best", ncol=int(cfg.get("legend_ncol", 1)))
    _force_spines_black(ax)

    if cfg.get("save_as_pdf", False):
        out_dir = cfg.get("save_path", ".")
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(cfg.get("outfile_name", "plots.pdf"))[0]
        out = os.path.join(out_dir, f"{base}_adj_total.pdf")
        fig.savefig(out, bbox_inches="tight", transparent=True)
        print(f"[saved] {out}")

    plt.show()


def plot_resistant_total(entries: list[dict], cfg: dict):
    figsize = cfg.get("figsize", (3.5, 2.5))
    dpi = cfg.get("dpi", 300)
    iqr_alpha = float(cfg.get("iqr_alpha", 0.25))

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    make_transparent(fig, ax)

    for e in entries:
        x = e["time_hr"]
        ax.fill_between(x, e["frac_q1"], e["frac_q3"],
                        alpha=iqr_alpha * e["alpha"], color=e["color"], linewidth=0)
        ax.plot(x, e["frac_med"], e["linestyle"],
                color=e["color"], alpha=e["alpha"], linewidth=e["linewidth"],
                label=e["exp_name"])

    _apply_x_axis(ax)
    ax.set_ylabel("Resistant / Total")
    ax.set_ylim(bottom=0)
    if cfg.get("ymax_freq") is not None:
        ax.set_ylim(top=float(cfg["ymax_freq"]))

    for h in cfg.get("hline_positions"):
        ax.axvline(x=h, color='black', linestyle=(0, (3, 5, 1, 5)), linewidth=1)
    _force_spines_black(ax)

    set_time_axis_0_125h(ax)
    if cfg.get("save_as_pdf", False):
        out_dir = cfg.get("save_path", ".")
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(cfg.get("outfile_name", "plots.pdf"))[0]
        out = os.path.join(out_dir, f"{base}_resistant_total.pdf")
        fig.savefig(out, bbox_inches="tight", transparent=True)
        print(f"[saved] {out}")

    plt.show()


def plot_highlight_strip(entries: list[dict], cfg: dict):
    """
    Highlight strip as its own figure with SAME figsize/dpi.
    Each experiment is one horizontal row; its spans are drawn in the experiment color.

    IMPORTANT: spans are given in FRAMES and must be shifted by -1 frame (= -0.5 h).
    So we do: (frame/2.0) - 0.5
    """
    figsize = cfg.get("figsize", (3.5, 2.5))
    dpi = cfg.get("dpi", 300)

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    make_transparent(fig, ax)

    label_spacing = float(cfg.get("strip_label_spacing", 1.0))
    lw = float(cfg.get("strip_linewidth", 3))
    show_labels = bool(cfg.get("strip_show_labels", True))
    label_pad = float(cfg.get("strip_label_pad", 2))

    n = len(entries)
    y_positions = np.arange(n) * label_spacing

    ax.set_xlim(0, 125)
    ax.xaxis.set_major_locator(mticker.FixedLocator(np.arange(0, 126, 25)))
    ax.set_xlabel("Hours")

    ax.set_ylim(y_positions[0] - 0.5 * label_spacing, y_positions[-1] + 0.5 * label_spacing)
    ax.set_yticks(y_positions)
    if show_labels:
        ax.set_yticklabels([e["exp_name"] for e in entries])
        ax.tick_params(axis="y", pad=label_pad)
    else:
        ax.set_yticklabels([])

    #ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.6)

    # draw spans
    for i, e in enumerate(entries):
        y = y_positions[i]
        spans = e.get("spans_frames", []) or []
        for (s_f, e_f) in spans:
            # frames -> hours and shift left by 0.5 h (=-1 frame)
            s_hr = (s_f / 2.0) - 0.5
            e_hr = (e_f / 2.0) - 0.5

            # clip to visible range
            s_hr = max(s_hr, 0.0)
            e_hr = min(e_hr, 125.0)
            if e_hr <= s_hr:
                continue

            ax.plot([s_hr, e_hr], [y, y],
                    "-", linewidth=lw, color=e["color"], alpha=e["alpha"],
                    solid_capstyle="butt")

    _force_spines_black(ax)

    if cfg.get("save_as_pdf", False):
        out_dir = cfg.get("save_path", ".")
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(cfg.get("outfile_name", "plots.pdf"))[0]
        out = os.path.join(out_dir, f"{base}_highlight_strip.pdf")
        fig.savefig(out, bbox_inches="tight", transparent=True)
        print(f"[saved] {out}")
    set_time_axis_0_125h(ax)
    plt.show()


# ────────────────────────── Main ──────────────────────────

def main(cfg: dict):
    set_nature_style()

    folder_paths = cfg["folder_paths"]
    identifiers = cfg["identifiers"]

    entries = []
    for folder in folder_paths:
        out = compute_folder_stats(folder, identifiers, cfg)
        if out is None:
            continue
        entries.append(out)

    if not entries:
        print("[warn] Nothing to plot.")
        return

    # If some colors are None, assign a simple cycle
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["black"])
    cidx = 0
    for e in entries:
        if e["color"] is None:
            e["color"] = cycle[cidx % len(cycle)]
            cidx += 1

    # Print median colony per display name
    print("\nMedian colony per experiment (closest to median adjusted area at last common timepoint):")
    for e in entries:
        print(f"  - {e['exp_name']}: {e['median_ident']}  "
              f"(t={e['median_hour']:.1f} h, frame={e['median_frame']})")
    print("")

    # 3 separate figures
    plot_adjusted_total(entries, cfg)
    plot_resistant_total(entries, cfg)
    plot_highlight_strip(entries, cfg)


if __name__ == "__main__":
    # Plug in your config; this script uses: folder_paths, identifiers, scale_factor,
    # display_names, plot_styles, highlight_spans, plus plotting params below.
    config = {
        "folder_paths": [
            r"C:\Users\nappold\Desktop\New folder\20251024_no_treatment\For_Manuscript",
            r"C:\Users\nappold\Desktop\New folder\20240917_continuous_dose_2\For_Manuscript",
            r"C:\Users\nappold\Desktop\New folder\20251205_6_18",
            r"C:\Users\nappold\Desktop\New folder\20240227_adaptivetherapy",
        ],
        "identifiers": ["P1_", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12", "P13", "P14", "P15", "P16", "P17"],
        "min_frames": 10,
        "scale_factor": 8.648,

        "display_names": {
            r"C:\Users\nappold\Desktop\New folder\20251024_no_treatment\For_Manuscript": "No Treatment",
            r"C:\Users\nappold\Desktop\New folder\20240917_continuous_dose_2\For_Manuscript": "Continuous",
            r"C:\Users\nappold\Desktop\New folder\20251205_6_18": "6/18",
            r"C:\Users\nappold\Desktop\New folder\20240227_adaptivetherapy": "Adaptive Therapy",
        },

        "plot_styles": {
            r"C:\Users\nappold\Desktop\New folder\20240917_continuous_dose_2\For_Manuscript": {"color": "#7b4173", "linestyle": "-", "alpha": 1, "linewidth": 1.0},
            r"C:\Users\nappold\Desktop\New folder\20251024_no_treatment\For_Manuscript": {"color": "#393b79", "linestyle": "-", "alpha": 1, "linewidth": 1.0},
            r"C:\Users\nappold\Desktop\New folder\20251205_6_18": {"color": "#8c6d31", "linestyle": "-", "alpha": 1, "linewidth": 1.0},
            r"C:\Users\nappold\Desktop\New folder\20240227_adaptivetherapy": {"color": "black", "linestyle": "-", "alpha": 1, "linewidth": 1.0},
        },
        # highlight spans in FRAMES (will be shifted -1 frame = -0.5h in plot)
        "highlight_spans": {
            r"C:\Users\nappold\Desktop\New folder\20251024_no_treatment\For_Manuscript": [],
            r"C:\Users\nappold\Desktop\New folder\20240917_continuous_dose_2\For_Manuscript": [[37, 334]],
            r"C:\Users\nappold\Desktop\New folder\20251205_6_18": [[37, 50], [86, 99], [135, 148], [184, 197],
                                                                   [233, 246], [282, 295]],
            r"C:\Users\nappold\Desktop\New folder\20240227_adaptivetherapy": [[37, 61], [110, 141], [189, 251]],
        },

        # plotting params
        "iqr_alpha": 0.25,
        "figsize": (3.5, 1.5),
        "dpi": 600,
        "smooth_window": 8,
        "ymax": 71,  # mm²
        "ymax_freq": 0.9,
        "legend_ncol": 1,
        "line_width": 1.0,
        "hline_positions": [],

        # strip params (same fig size, but this affects only the strip plot)
        "strip_linewidth": 3,
        "strip_label_spacing": 1.0,
        "strip_label_pad": 2,
        "strip_show_labels": True,

        # saving
        "save_as_pdf": True,
        "save_path": r"C:\Users\nappold\Desktop\Manuscript Figures\For SupplementAdaptive",
        "outfile_name": "adj_total_resistant_strip.pdf",
    }

    main(config)