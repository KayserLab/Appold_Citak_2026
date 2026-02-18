#!/usr/bin/env python3
"""
Compute and plot TIME TO PROGRESSION for clone-adjusted colony size.

Definition:
- For each folder (treatment) and identifier (P1_, P2, ...),
  we load the colony CSV that contains clone information.
- We build the "clone-adjusted colony area":
      adj_colony_area = colony_area - total_clone_area + extrapolated_clone_area
  (falling back to total_clone_area if extrapolated_clone_area is missing).
- We convert this to mm² using the given scale factor (µm per pixel).
- We find the FIRST frame where adj_colony_area >= progression_threshold_mm2.
- Time to progression [h] = frame_index / 2.0 (since 2 frames = 1 hour).

We then plot a boxplot:
- one box per experiment/treatment
- x-tick labels = your display names
- colors = from plot_styles (if given), else gray.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


# ────────────────────────── Helpers ──────────────────────────

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
        'legend.frameon': True,
        'legend.fontsize': 6,
        'lines.linewidth': 1.0,
        'savefig.dpi': 300,
        'figure.dpi': 300,
        'axes.spines.top': True,
        'axes.spines.right': True,
    })


def _norm(p: str) -> str:
    return os.path.normpath(p).replace("\\", "/").rstrip("/")


def _prep_name_map(cfg):
    raw = cfg.get("display_names", {})
    norm_map, base_map = {}, {}
    for k, v in raw.items():
        kn = _norm(k)
        norm_map[kn] = v
        base_map[os.path.basename(kn)] = v
    return norm_map, base_map


def _prep_style_map(cfg):
    raw = cfg.get("plot_styles", {})
    norm_map, base_map = {}, {}
    for k, v in raw.items():
        kn = _norm(k)
        norm_map[kn] = v
        base_map[os.path.basename(kn)] = v
    return norm_map, base_map

def compare_vs_control(values_by_exp, control_name="CT",
                       ignore=("NT",), alpha=0.05):

    control = values_by_exp.get(control_name, [])
    if len(control) == 0:
        raise ValueError("Control group empty!")

    results = []
    pvals = []
    labels = []

    for name, vals in values_by_exp.items():
        if name == control_name or name in ignore:
            continue

        stat, p = mannwhitneyu(
            vals, control,
            alternative="greater"  # <-- IMPORTANT
        )
        # "greater" = test if treatment > CT (better performance)

        results.append((name, stat, p))
        pvals.append(p)
        labels.append(name)

    # multiple testing correction
    reject, p_corr, _, _ = multipletests(
        pvals, alpha=alpha, method="holm"
    )

    print("\n=== Statistical comparison vs CT ===")
    for i, name in enumerate(labels):
        sig = "YES" if reject[i] else "no"
        print(f"{name:6s} vs CT | "
              f"raw p = {pvals[i]:.4g} | "
              f"corr p = {p_corr[i]:.4g} | "
              f"significant: {sig}")

    return labels, pvals, p_corr, reject

def find_colony_extrap_file(folder_path, identifier):
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

    candidates.sort()
    return os.path.join(folder_path, candidates[0])


def series_from_colony_file(colony_file):
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

    colony_px2 = pd.Series(df["colony_area"].values, index=idx)
    clone_raw_px2 = pd.Series(df["total_clone_area"].values, index=idx)
    clone_px2 = pd.Series(df[clone_colname].values, index=idx)

    adj_colony_px2 = colony_px2 - clone_raw_px2 + clone_px2

    return colony_px2, clone_px2, adj_colony_px2


# ────────────────────────── Core computation ──────────────────────────

def compute_progression_metrics(folder_paths, identifiers, config):
    """
    Returns:
      ttp_by_exp:  dict {exp_name: [hours, ...]}
      frac_by_exp: dict {exp_name: [clonal_fraction_at_progression, ...]}
      style_by_exp:dict {exp_name: style_cfg}
      ttp_detail_by_exp: dict {exp_name: [(identifier, ttp_hours, f_prog), ...]}
    """
    sf = float(config.get("scale_factor", 1.0))  # µm/px
    thresh_mm2 = float(config.get("progression_threshold_mm2", 71.0))
    SCALE_MM2_PER_PX2 = (sf ** 2) / 1e6

    name_full, name_base = _prep_name_map(config)
    style_full, style_base = _prep_style_map(config)

    ttp_by_exp = {}
    frac_by_exp = {}
    style_by_exp = {}
    ttp_detail_by_exp = {}

    for folder in folder_paths:
        folder_norm = _norm(folder)
        base = os.path.basename(folder_norm)

        exp_name = (
            name_full.get(folder_norm) or
            name_base.get(base) or
            re.sub(r"^\d{8}_", "", base)
        )

        style_cfg = (
            style_full.get(folder_norm) or
            style_base.get(exp_name) or
            style_base.get(base) or
            {}
        )
        style_by_exp.setdefault(exp_name, style_cfg)

        for ident in identifiers:
            colony_file = find_colony_extrap_file(folder, ident)
            if colony_file is None:
                continue

            try:
                _, clone_px2, adj_px2 = series_from_colony_file(colony_file)
            except Exception as e:
                print(f"[warn] {exp_name}: failed for {ident} from {colony_file}: {e}")
                continue

            if adj_px2.empty or clone_px2.empty:
                continue

            adj_mm2 = adj_px2.values * SCALE_MM2_PER_PX2
            frames = adj_px2.index.values

            hit = np.where(adj_mm2 >= thresh_mm2)[0]
            if hit.size == 0:
                f_prog = frames[-1]
                print(f"[info] {exp_name} {ident}: no progression reached, "
                      f"using last frame {f_prog} for TTP calculation.")
            else:
                f_prog = frames[hit[0]]

            ttp_h = f_prog / 2.0

            # clonal fraction at that frame
            adj_at_prog = float(adj_px2.loc[f_prog])
            clone_at_prog = float(clone_px2.loc[f_prog])

            if adj_at_prog <= 0:
                frac = np.nan
            else:
                frac = clone_at_prog / adj_at_prog

            ttp_by_exp.setdefault(exp_name, []).append(ttp_h)
            frac_by_exp.setdefault(exp_name, []).append(frac)
            ttp_detail_by_exp.setdefault(exp_name, []).append((ident, float(ttp_h), int(f_prog)))

    # clean NaNs in fractions
    for k in list(frac_by_exp.keys()):
        frac_by_exp[k] = [v for v in frac_by_exp[k] if np.isfinite(v)]

    # ---- NEW: pretty print TTPs + mark median colony (the one shown by the boxplot) ----
    print("\n=== TTP per colony (and which one is the median) ===")
    for exp_name in sorted(ttp_detail_by_exp.keys()):
        details = ttp_detail_by_exp[exp_name]
        if len(details) == 0:
            continue

        # sort by TTP, then by identifier for stable ordering
        details_sorted = sorted(details, key=lambda x: (x[1], x[0]))

        # Matplotlib boxplot median is np.median of the sample values
        ttp_vals = np.array([d[1] for d in details_sorted], dtype=float)
        med = float(np.median(ttp_vals))

        # Choose a *specific colony* to label as the median:
        # - if an exact match exists, pick the first (stable)
        # - else, pick the closest value (rare if half-frames etc.)
        idx_exact = [i for i, (_, t, _) in enumerate(details_sorted) if np.isclose(t, med)]
        if idx_exact:
            med_idx = idx_exact[0]
        else:
            med_idx = int(np.argmin(np.abs(ttp_vals - med)))

        print(f"\n{exp_name}: n={len(details_sorted)} | median={med:.2f} h")
        for i, (ident, ttp_h, f_prog) in enumerate(details_sorted):
            tag = "  <== MEDIAN (boxplot line)" if i == med_idx else ""
            print(f"  {ident:>4s}  TTP={ttp_h:7.2f} h   (frame={f_prog:4d}){tag}")

    # keep your summary line if you want
    for name in ttp_by_exp:
        print(f"{name}: n={len(ttp_by_exp[name])} TTP values, "
              f"n_frac={len(frac_by_exp.get(name, []))} fractions")

    return ttp_by_exp, frac_by_exp, style_by_exp, ttp_detail_by_exp

# ────────────────────────── Plotting ──────────────────────────

def _ordered_names(folder_paths, progression_dict, config):
    name_full, name_base = _prep_name_map(config)
    ordered = []
    seen = set()

    for folder in folder_paths:
        fn = _norm(folder)
        base = os.path.basename(fn)
        nm = (
            name_full.get(fn) or
            name_base.get(base) or
            re.sub(r"^\d{8}_", "", base)
        )
        if nm in progression_dict and nm not in seen:
            ordered.append(nm)
            seen.add(nm)

    for nm in progression_dict.keys():
        if nm not in seen:
            ordered.append(nm)

    return ordered


def plot_horizontal_boxplot(values_by_exp, style_by_exp, config,
                            xlabel, outfile_key, xlim=None):
    """
    Horizontal boxplot (categories on y-axis, values on x-axis).
    """
    if not values_by_exp:
        print(f"[warn] No data to plot for {xlabel}.")
        return

    set_nature_style()

    folder_paths = config.get("folder_paths", [])
    ordered_names = _ordered_names(folder_paths, values_by_exp, config)
    data = [values_by_exp[nm] for nm in ordered_names]

    figsize = config.get("figsize_boxplot", (2.33, 1.4))
    dpi = config.get("dpi", 600)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    bp = ax.boxplot(
        data,
        vert=False,                      # <-- horizontal
        positions=np.arange(1, len(ordered_names) + 1),
        widths=0.55,
        patch_artist=True,
        showfliers=True,
        medianprops={"linewidth": 1.0, "color": "black"},
        boxprops={"linewidth": 0.5},
        whiskerprops={"linewidth": 0.5},
        capprops={"linewidth": 0.5},
        flierprops={
            "marker": "o",
            "markersize": 2.5,      # <-- change this
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "alpha": 0.8
        }
    )

    for patch, nm in zip(bp["boxes"], ordered_names):
        style_cfg = style_by_exp.get(nm, {})
        facecolor = style_cfg.get("color", "lightgray")
        alpha = float(style_cfg.get("alpha", 1.0))
        patch.set_facecolor(facecolor)
        patch.set_alpha(alpha)

    # y ticks = treatments
    ax.set_yticks(np.arange(1, len(ordered_names) + 1))
    ax.set_yticklabels(ordered_names)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Treatment schedule")


    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_edgecolor("black")

    ax.tick_params(axis="x", direction="out")
    ax.tick_params(axis="y", direction="out")

    if xlim is not None:
        ax.set_xlim(xlim)

    fig.tight_layout()

    if config.get("save_as_pdf", False):
        save_dir = config.get("save_path", ".")
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, config.get(outfile_key))
        fig.savefig(out, format="pdf", bbox_inches="tight")
        print(f"[saved] {out}")

    plt.show()
# ────────────────────────── Main ──────────────────────────

if __name__ == "__main__":
    CONFIG = {
        "folder_paths": [
            r"C:\Users\nappold\Desktop\New folder\20251024_no_treatment\For_Manuscript",
            #r"C:\Users\nappold\Desktop\New folder\20250909_metr_undertreat\For_Manuscript",
            #r"C:\Users\nappold\Desktop\New folder\20250902_metronomic_overtreat\For_Manuscript",
            #r"C:\Users\nappold\Desktop\New folder\20250930_metronomic_6_18\For_Manuscript",
            r"C:\Users\nappold\Desktop\New folder\20251121_4_18",
            #r"C:\Users\nappold\Desktop\New folder\20251007_metr_7_18\For_Manuscript",
            r"C:\Users\nappold\Desktop\New folder\20251205_6_18",
            #r"C:\Users\nappold\Desktop\New folder\20240227_adaptivetherapy",
            r"C:\Users\nappold\Desktop\New folder\20251114_9_18\For_manuscript",
            r"C:\Users\nappold\Desktop\New folder\20240917_continuous_dose_2\For_Manuscript",
        ],
        "identifiers": [
            "P1_", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9",
            "P10", "P11", "P12", "P13", "P14", "P15", "P16", "P17"
        ],
        "scale_factor": 8.648,          # µm per pixel
        "progression_threshold_mm2": 71.0,

        "display_names": {
            r"C:\Users\nappold\Desktop\New folder\20251024_no_treatment\For_Manuscript": "NT",
            r"C:\Users\nappold\Desktop\New folder\20250930_metronomic_6_18\For_Manuscript": "6/18",
            r"C:\Users\nappold\Desktop\New folder\20251007_metr_7_18\For_Manuscript": "7/18",
            r"C:\Users\nappold\Desktop\New folder\20240917_continuous_dose_2\For_Manuscript": "CT",
            r"C:\Users\nappold\Desktop\New folder\20251114_9_18\For_manuscript": "9/18",
            r"C:\Users\nappold\Desktop\New folder\20251121_4_18": "4/18",
            r"C:\Users\nappold\Desktop\New folder\20251205_6_18": "6.5/18",
            r"C:\Users\nappold\Desktop\New folder\20240227_adaptivetherapy": "AT",
            #r"C:\Users\nappold\Desktop\New folder\20250909_metr_undertreat\For_Manuscript": "2/21",
            #r"C:\Users\nappold\Desktop\New folder\20250902_metronomic_overtreat\For_Manuscript": "20/21",

        },

        # BACK AGAIN: plot_styles drives colors
        "plot_styles": {
            r"C:\Users\nappold\Desktop\New folder\20240917_continuous_dose_2\For_Manuscript": {
                "color": "#7b4173",
                "alpha": 1.0,
            },
            r"C:\Users\nappold\Desktop\New folder\20251024_no_treatment\For_Manuscript": {
                "color": "#393b79",
                "alpha": 1.0,
            },
            r"C:\Users\nappold\Desktop\New folder\20250930_metronomic_6_18\For_Manuscript": {
                "color": "#637939",
                "alpha": 1.0,
            },
            r"C:\Users\nappold\Desktop\New folder\20251007_metr_7_18\For_Manuscript": {
                "color": "#8c6d31",
                "alpha": 1.0,
            },
            r"C:\Users\nappold\Desktop\New folder\20251205_6_18": {
                "color": "#8c6d31",
                "alpha": 1.0,
            },
            r"C:\Users\nappold\Desktop\New folder\20250909_metr_undertreat\For_Manuscript": {
                "color": "#843c39",
                "alpha": 1.0,
            },
            r"C:\Users\nappold\Desktop\New folder\20250902_metronomic_overtreat\For_Manuscript": {
                "color": "darkorange",
                "alpha": 1.0,
            },
            r"C:\Users\nappold\Desktop\New folder\20251114_9_18\For_manuscript": {
                "color": "#843c39", #????
                "alpha": 1.0,
            },
            r"C:\Users\nappold\Desktop\New folder\20251121_4_18": {
                "color": "#637939",
                "alpha": 1.0,
            },
            r"C:\Users\nappold\Desktop\New folder\20240227_adaptivetherapy": {
                "color": "teal",
                "alpha": 1.0,
            },
        },

        # Plot appearance
        "figsize_boxplot": (2, 2),
        "dpi": 600,
        "ymax_progression": None,

        # Optional styling toggle
        "color_whiskers_like_boxes": False,

        # Saving
        "save_as_pdf": True,
        "save_path": r"C:\Users\nappold\Desktop\Manuscript Figures\Fig5",
        "outfile_name_ttp": "time_to_progression_clone_adjusted_boxplot_horizontal_AT.pdf",
        "outfile_name_frac": "clonal_fraction_at_progression_boxplot_horizontal_AT.pdf",
    }

    ttp_by_exp, frac_by_exp, style_by_exp, ttp_detail_by_exp = compute_progression_metrics(
        CONFIG["folder_paths"],
        CONFIG["identifiers"],
        CONFIG,
    )

    # 1) Horizontal TTP boxplot
    plot_horizontal_boxplot(
        ttp_by_exp,
        style_by_exp,
        CONFIG,
        xlabel="Time to progression (h)",
        outfile_key="outfile_name_ttp",
        xlim=(50, 170),
    )

    # 2) Horizontal clonal fraction at progression boxplot
    plot_horizontal_boxplot(
        frac_by_exp,
        style_by_exp,
        CONFIG,
        xlabel="Clonal fraction at progression",
        outfile_key="outfile_name_frac",
        xlim=(0, 1),  # fractions live in [0,1]
    )

    compare_vs_control(
        ttp_by_exp,
        control_name="CT",
        ignore=("NT",),
        alpha=0.05
    )