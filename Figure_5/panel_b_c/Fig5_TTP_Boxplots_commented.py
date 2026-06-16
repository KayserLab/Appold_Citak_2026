#!/usr/bin/env python3
"""
Compute and plot time-to-progression for clone-adjusted colony size.

Definition
- For each condition folder and identifier (P1_, P2, ...), load the colony CSV that contains clone information.
- Compute the clone-adjusted colony area:
      adj_colony_area = colony_area - total_clone_area + extrapolated_clone_area
  If 'extrapolated_clone_area' is not present, 'total_clone_area' is used instead.
- Convert areas from px² to mm² using the provided pixel size (µm per pixel).
- Determine the first frame where adj_colony_area >= progression_threshold_mm2.
- Time-to-progression (TTP) in hours is computed as: frame_index / 2.0 (2 frames = 1 hour).

Outputs
- Horizontal boxplot of TTP per condition.
- Horizontal boxplot of clonal fraction at progression per condition.
- Console output listing per-colony TTPs with the median colony highlighted, and significance tests vs. CT.
"""

import os
import re

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


# ────────────────────────── Project-relative IO ──────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_ROOT = os.path.join(SCRIPT_DIR, "Input_files")
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "Output_files")


# ────────────────────────── Helpers ──────────────────────────
def set_nature_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "axes.linewidth": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.frameon": True,
        "legend.fontsize": 6,
        "lines.linewidth": 1.0,
        "savefig.dpi": 300,
        "figure.dpi": 300,
        "axes.spines.top": True,
        "axes.spines.right": True,
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


def compare_vs_control(values_by_exp, control_name="CT", ignore=("NT",), alpha=0.05):
    """
    Compare each condition against the control group using a one-sided Mann–Whitney U test.

    Hypothesis
    - alternative="greater": tests whether the condition has larger values than the control.

    Multiple testing correction
    - Holm correction is applied across all comparisons.

    Output
    - Prints raw and corrected p-values and a significance flag for each comparison.
    """
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
            alternative="greater"  # IMPORTANT: treatment > CT
        )

        results.append((name, stat, p))
        pvals.append(p)
        labels.append(name)

    reject, p_corr, _, _ = multipletests(pvals, alpha=alpha, method="holm")

    print("\n=== Statistical comparison vs CT ===")
    for i, name in enumerate(labels):
        sig = "YES" if reject[i] else "no"
        print(
            f"{name:6s} vs CT | "
            f"raw p = {pvals[i]:.4g} | "
            f"corr p = {p_corr[i]:.4g} | "
            f"significant: {sig}"
        )

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

    clone_colname = "extrapolated_clone_area" if "extrapolated_clone_area" in df.columns else "total_clone_area"

    idx = np.arange(len(df))

    colony_px2 = pd.Series(df["colony_area"].values, index=idx)
    clone_raw_px2 = pd.Series(df["total_clone_area"].values, index=idx)
    clone_px2 = pd.Series(df[clone_colname].values, index=idx)

    adj_colony_px2 = colony_px2 - clone_raw_px2 + clone_px2

    return colony_px2, clone_px2, adj_colony_px2


# ────────────────────────── Core computation ──────────────────────────
def compute_progression_metrics(folder_paths, identifiers, config):
    """
    Compute time-to-progression (TTP) and clonal fraction at progression for each condition.

    Returns
    - ttp_by_exp: dict {condition_name: [ttp_hours, ...]}
    - frac_by_exp: dict {condition_name: [clone_fraction_at_progression, ...]} (NaNs removed)
    - style_by_exp: dict {condition_name: style_cfg}
    - ttp_detail_by_exp: dict {condition_name: [(identifier, ttp_hours, progression_frame), ...]}

    Notes
    - If the progression threshold is never reached, the last available frame is used.
    - The console output includes per-colony TTP values and marks the colony corresponding to the median.
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
            name_full.get(folder_norm)
            or name_base.get(base)
            or re.sub(r"^\d{8}_", "", base)
        )

        style_cfg = (
            style_full.get(folder_norm)
            or style_base.get(exp_name)
            or style_base.get(base)
            or {}
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
                print(
                    f"[info] {exp_name} {ident}: no progression reached, "
                    f"using last frame {f_prog} for TTP calculation."
                )
            else:
                f_prog = frames[hit[0]]

            ttp_h = f_prog / 2.0

            adj_at_prog = float(adj_px2.loc[f_prog])
            clone_at_prog = float(clone_px2.loc[f_prog])
            frac = np.nan if adj_at_prog <= 0 else (clone_at_prog / adj_at_prog)

            ttp_by_exp.setdefault(exp_name, []).append(ttp_h)
            frac_by_exp.setdefault(exp_name, []).append(frac)
            ttp_detail_by_exp.setdefault(exp_name, []).append((ident, float(ttp_h), int(f_prog)))

    for k in list(frac_by_exp.keys()):
        frac_by_exp[k] = [v for v in frac_by_exp[k] if np.isfinite(v)]

    # Print per-colony TTPs and mark which colony corresponds to the median boxplot line
    print("\n=== TTP per colony (and which one is the median) ===")
    for exp_name in sorted(ttp_detail_by_exp.keys()):
        details = ttp_detail_by_exp[exp_name]
        if len(details) == 0:
            continue

        details_sorted = sorted(details, key=lambda x: (x[1], x[0]))
        ttp_vals = np.array([d[1] for d in details_sorted], dtype=float)
        med = float(np.median(ttp_vals))

        idx_exact = [i for i, (_, t, _) in enumerate(details_sorted) if np.isclose(t, med)]
        if idx_exact:
            med_idx = idx_exact[0]
        else:
            med_idx = int(np.argmin(np.abs(ttp_vals - med)))

        print(f"\n{exp_name}: n={len(details_sorted)} | median={med:.2f} h")
        for i, (ident, ttp_h, f_prog) in enumerate(details_sorted):
            tag = "  <== MEDIAN (boxplot line)" if i == med_idx else ""
            print(f"  {ident:>4s}  TTP={ttp_h:7.2f} h   (frame={f_prog:4d}){tag}")

    for name in ttp_by_exp:
        print(
            f"{name}: n={len(ttp_by_exp[name])} TTP values, "
            f"n_frac={len(frac_by_exp.get(name, []))} fractions"
        )

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
            name_full.get(fn)
            or name_base.get(base)
            or re.sub(r"^\d{8}_", "", base)
        )
        if nm in progression_dict and nm not in seen:
            ordered.append(nm)
            seen.add(nm)

    for nm in progression_dict.keys():
        if nm not in seen:
            ordered.append(nm)

    return ordered


def plot_horizontal_boxplot(values_by_exp, style_by_exp, config, xlabel, outfile_key, xlim=None):
    """
    Create a horizontal boxplot with conditions on the y-axis and values on the x-axis.

    Styling
    - Box colors are taken from 'style_by_exp' (e.g. config['plot_styles']).
    - Outliers are shown as small open circles.

    Saving
    - If config['save_as_pdf'] is True, the figure is written to Output_files/ using config[outfile_key].
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
        vert=False,
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
            "markersize": 2.5,
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "alpha": 0.8,
        },
    )

    for patch, nm in zip(bp["boxes"], ordered_names):
        style_cfg = style_by_exp.get(nm, {})
        facecolor = style_cfg.get("color", "lightgray")
        alpha = float(style_cfg.get("alpha", 1.0))
        patch.set_facecolor(facecolor)
        patch.set_alpha(alpha)

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
        os.makedirs(OUTPUT_ROOT, exist_ok=True)
        out = os.path.join(OUTPUT_ROOT, config.get(outfile_key))
        fig.savefig(out, format="pdf", bbox_inches="tight")
        print(f"[saved] {out}")

    plt.show()


# ────────────────────────── Main ──────────────────────────
if __name__ == "__main__":
    CONFIG = {
        # Only folders that are not commented out in the original script
        # These are subfolders under Input_files/
        "folder_paths": [
            os.path.join(INPUT_ROOT, "No_treatment_control"),
            os.path.join(INPUT_ROOT, "4h_18h"),
            os.path.join(INPUT_ROOT, "6.5h_18h"),
            os.path.join(INPUT_ROOT, "9h_18h"),
            os.path.join(INPUT_ROOT, "Continuous_therapy"),
        ],

        "identifiers": [
            "P1_", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9",
            "P10", "P11", "P12", "P13", "P14", "P15", "P16", "P17",
        ],

        "scale_factor": 8.648,
        "progression_threshold_mm2": 71.0,

        # Display names only for the active folders above
        "display_names": {
            _norm(os.path.join(INPUT_ROOT, "NT_csvs")): "NT",
            _norm(os.path.join(INPUT_ROOT, "CT_csvs")): "CT",
            _norm(os.path.join(INPUT_ROOT, "9_18_csvs")): "9/18",
            _norm(os.path.join(INPUT_ROOT, "4_18_csvs")): "4/18",
            _norm(os.path.join(INPUT_ROOT, "6_18_csvs")): "6.5/18",
        },

        # plot_styles only for the active folders above
        "plot_styles": {
            _norm(os.path.join(INPUT_ROOT, "CT_csvs")): {"color": "#7b4173", "alpha": 1.0},
            _norm(os.path.join(INPUT_ROOT, "NT_csvs")): {"color": "#393b79", "alpha": 1.0},
            _norm(os.path.join(INPUT_ROOT, "6_18_csvs")): {"color": "#8c6d31", "alpha": 1.0},
            _norm(os.path.join(INPUT_ROOT, "9_18_csvs")): {"color": "#843c39", "alpha": 1.0},
            _norm(os.path.join(INPUT_ROOT, "4_18_csvs")): {"color": "#637939", "alpha": 1.0},
        },

        # Plot appearance
        "figsize_boxplot": (2, 2),
        "dpi": 600,
        "ymax_progression": None,

        # Saving (to Output_files/)
        "save_as_pdf": True,
        "outfile_name_ttp": "time_to_progression_clone_adjusted_boxplot_horizontal.pdf",
        "outfile_name_frac": "clonal_fraction_at_progression_boxplot_horizontal.pdf",
    }

    ttp_by_exp, frac_by_exp, style_by_exp, _ttp_detail_by_exp = compute_progression_metrics(
        CONFIG["folder_paths"],
        CONFIG["identifiers"],
        CONFIG,
    )

    plot_horizontal_boxplot(
        ttp_by_exp,
        style_by_exp,
        CONFIG,
        xlabel="Time to progression (h)",
        outfile_key="outfile_name_ttp",
        xlim=(50, 170),
    )

    plot_horizontal_boxplot(
        frac_by_exp,
        style_by_exp,
        CONFIG,
        xlabel="Clonal fraction at progression",
        outfile_key="outfile_name_frac",
        xlim=(0, 1),
    )

    compare_vs_control(
        ttp_by_exp,
        control_name="CT",
        ignore=("NT",),
        alpha=0.05,
    )