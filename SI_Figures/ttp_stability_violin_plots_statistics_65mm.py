#!/usr/bin/env python3
"""
Compute and plot:
1) Time to progression (TTP)
2) Clonal fraction at progression / shared censor time

Plot style:
- horizontal violin-like distributions
- single colonies shown as jittered points
- IQR shown as a thick horizontal bar
- median shown as a marker + short line

Corrected TTP censoring:
- The shared censor frame is defined by the shortest non-progressing colony.
- Schedules listed in CONFIG["exclude_from_censor_definition"], e.g. NT,
  are skipped when defining this shared censor frame.
- NT is still loaded, included, and plotted; it simply cannot define the censor frame.
- After the shared censor frame is defined, ALL colonies are clipped to that frame
  before progression is evaluated.
- Therefore, colonies that only reach progression after the shared censor frame are
  treated as censored at the shared censor frame.

Statistics:
- Mann-Whitney U tests with Holm correction
- median-difference effect sizes for selected TTP schedule comparisons
- 95% bootstrap confidence intervals for those median differences
- optional CSV export of the effect-size table

Effect size definition:
    median difference = median(TTP of group A) - median(TTP of group B)

For example:
    group_A = "6.5/18", group_B = "CT"
    positive effect size means the 6.5/18 schedule had a higher median TTP than CT.

Bootstrap CI:
- Non-parametric percentile bootstrap
- Resamples colonies within each schedule with replacement
- Computes median(group A bootstrap sample) - median(group B bootstrap sample)
- Repeats n_bootstrap times
- Reports the central 95% interval by default
"""

import os
import re
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


# ────────────────────────── Project-relative IO ──────────────────────────
# Place the condition folders inside "Input_files" next to this script.
# Generated plots and optional CSV tables are written to "Output_files".
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_ROOT = os.path.join(SCRIPT_DIR, "Input_files")
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "Output_files")


# ────────────────────────── Style ──────────────────────────

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


# ────────────────────────── Helpers ──────────────────────────

def _norm(path):
    return os.path.normpath(path).replace("\\", "/").rstrip("/")


def _prep_name_map(cfg):
    raw = cfg.get("display_names", {})
    norm_map = {}
    base_map = {}

    for k, v in raw.items():
        kn = _norm(k)
        norm_map[kn] = v
        base_map[os.path.basename(kn)] = v

    return norm_map, base_map


def _prep_style_map(cfg):
    raw = cfg.get("plot_styles", {})
    norm_map = {}
    base_map = {}

    for k, v in raw.items():
        kn = _norm(k)
        norm_map[kn] = v
        base_map[os.path.basename(kn)] = v

    return norm_map, base_map

def identifier_matches_filename(filename, identifier):
    """
    Match P2 without accidentally matching P21 or P210.

    The configured identifier list uses P1_ for P1, while most other IDs are P2, P3, ...
    This helper strips the trailing underscore and requires that no digit follows
    the identifier.
    """
    ident = identifier.rstrip("_")
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(ident)}(?!\d)", filename) is not None

def find_colony_extrap_file(folder_path, identifier):
    files = os.listdir(folder_path)

    candidates = [
        f for f in files
        if f.endswith(".csv")
        and "colony" in f
        and identifier_matches_filename(f, identifier)
        and "with_extrapolation" in f
    ]

    if not candidates:
        candidates = [
            f for f in files
            if f.endswith(".csv")
            and "colony" in f
            and identifier_matches_filename(f, identifier)
            and "with_clonearea" in f
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


def _clean_values(vals):
    vals = np.asarray(vals, dtype=float)
    return vals[np.isfinite(vals)]


# ────────────────────────── Mann-Whitney statistics ──────────────────────────

def compare_vs_control(values_by_exp, control_name="CT",
                       ignore=("NT",), alpha=0.05,
                       alternative="greater",
                       metric_name="metric"):

    control = _clean_values(values_by_exp.get(control_name, []))
    if len(control) == 0:
        raise ValueError(f"Control group '{control_name}' is empty!")

    pvals = []
    labels = []

    for name, vals in values_by_exp.items():
        if name == control_name or name in ignore:
            continue

        vals = _clean_values(vals)
        if len(vals) == 0:
            continue

        _, p = mannwhitneyu(vals, control, alternative=alternative)
        pvals.append(p)
        labels.append(name)

    if len(pvals) == 0:
        print(f"\n=== Statistical comparison vs {control_name} for {metric_name} ===")
        print("No valid comparison groups found.")
        return [], [], [], []

    reject, p_corr, _, _ = multipletests(pvals, alpha=alpha, method="holm")

    print(f"\n=== Statistical comparison vs {control_name} for {metric_name} ===")
    print(f"Alternative: treatment {alternative} {control_name}")

    for i, name in enumerate(labels):
        sig = "YES" if reject[i] else "no"
        print(f"{name:10s} vs {control_name:10s} | "
              f"raw p = {pvals[i]:.4g} | "
              f"corr p = {p_corr[i]:.4g} | "
              f"significant: {sig}")

    return labels, pvals, p_corr, reject


def compare_reference_vs_others(values_by_exp, reference_name,
                                include_only=None,
                                ignore=("NT",),
                                alpha=0.05,
                                alternative="greater",
                                metric_name="metric"):
    reference = _clean_values(values_by_exp.get(reference_name, []))
    if len(reference) == 0:
        raise ValueError(f"Reference group '{reference_name}' is empty!")

    pvals = []
    labels = []

    for name, vals in values_by_exp.items():
        if name == reference_name or name in ignore:
            continue
        if include_only is not None and name not in include_only:
            continue

        vals = _clean_values(vals)
        if len(vals) == 0:
            continue

        _, p = mannwhitneyu(reference, vals, alternative=alternative)
        pvals.append(p)
        labels.append(name)

    if len(pvals) == 0:
        print(f"\n=== Statistical comparison for {reference_name} ({metric_name}) ===")
        print("No valid comparison groups found.")
        return [], [], [], []

    reject, p_corr, _, _ = multipletests(pvals, alpha=alpha, method="holm")

    print(f"\n=== Statistical comparison: {reference_name} vs others for {metric_name} ===")
    print(f"Alternative: reference {alternative} comparator")

    for i, name in enumerate(labels):
        sig = "YES" if reject[i] else "no"
        print(f"{reference_name:10s} vs {name:10s} | "
              f"raw p = {pvals[i]:.4g} | "
              f"corr p = {p_corr[i]:.4g} | "
              f"significant: {sig}")

    return labels, pvals, p_corr, reject


# ────────────────────────── Effect sizes + bootstrap CIs ──────────────────────────

def bootstrap_median_difference_ci(vals_a, vals_b,
                                   n_bootstrap=10000,
                                   ci_percent=95,
                                   rng_seed=12345):
    """
    Non-parametric percentile bootstrap CI for:
        median(vals_a) - median(vals_b)

    Returns:
        observed_diff, ci_low, ci_high
    """
    vals_a = _clean_values(vals_a)
    vals_b = _clean_values(vals_b)

    if len(vals_a) == 0 or len(vals_b) == 0:
        return np.nan, np.nan, np.nan

    observed_diff = float(np.median(vals_a) - np.median(vals_b))

    rng = np.random.default_rng(rng_seed)
    boot_diffs = np.empty(int(n_bootstrap), dtype=float)

    for i in range(int(n_bootstrap)):
        sample_a = rng.choice(vals_a, size=len(vals_a), replace=True)
        sample_b = rng.choice(vals_b, size=len(vals_b), replace=True)
        boot_diffs[i] = np.median(sample_a) - np.median(sample_b)

    alpha = (100.0 - float(ci_percent)) / 2.0
    ci_low, ci_high = np.percentile(boot_diffs, [alpha, 100.0 - alpha])

    return observed_diff, float(ci_low), float(ci_high)


def _parse_comparison_entry(entry):
    """
    Accepts:
      ("6.5/18", "CT")
      ["6.5/18", "CT"]
      {"group_a": "6.5/18", "group_b": "CT"}
      {"A": "6.5/18", "B": "CT"}
    """
    if isinstance(entry, dict):
        group_a = entry.get("group_a", entry.get("A"))
        group_b = entry.get("group_b", entry.get("B"))
        return group_a, group_b

    if isinstance(entry, (list, tuple)) and len(entry) == 2:
        return entry[0], entry[1]

    raise ValueError(f"Invalid comparison entry: {entry}")


def build_effect_size_comparisons(values_by_exp, config):
    """
    Build the schedule comparisons used for the effect-size table.

    Default manuscript configuration:
      CONFIG["ttp_effect_size_comparisons"] = None
      CONFIG["effect_size_comparison_mode"] = "all_pairwise_main"

    This gives all pairwise comparisons among:
      CONFIG["main_schedules"] = ["4/18", "6.5/18", "9/18", "CT"]

    To use only specific comparisons, set, for example:
      "ttp_effect_size_comparisons": [
          ("4/18", "CT"),
          ("6.5/18", "CT"),
          ("9/18", "CT"),
          ("6.5/18", "4/18"),
          ("6.5/18", "9/18"),
      ]
    """
    explicit = config.get("ttp_effect_size_comparisons", None)

    if explicit is not None:
        return [_parse_comparison_entry(x) for x in explicit]

    main = list(config.get("main_schedules", []))
    main = [x for x in main if x in values_by_exp and len(_clean_values(values_by_exp[x])) > 0]

    mode = config.get("effect_size_comparison_mode", "all_pairwise_main")

    if mode == "all_pairwise_main":
        return list(combinations(main, 2))

    if mode == "main_vs_control":
        control = config.get("effect_size_control_name", "CT")
        return [(x, control) for x in main if x != control]

    raise ValueError(
        "Unknown effect_size_comparison_mode. Use 'all_pairwise_main' "
        "or 'main_vs_control', or provide ttp_effect_size_comparisons explicitly."
    )


def report_ttp_median_effect_sizes(values_by_exp, config,
                                   metric_name="TTP",
                                   units="h"):
    """
    Print and optionally save median-difference effect sizes and bootstrap CIs.

    The table is intended for Fig. 5 reporting/caption:
      median difference in TTP [h], 95% bootstrap CI, and optional MWU p-values.
    """
    comparisons = build_effect_size_comparisons(values_by_exp, config)

    n_bootstrap = int(config.get("bootstrap_n", 10000))
    ci_percent = float(config.get("bootstrap_ci_percent", 95))
    base_seed = int(config.get("bootstrap_rng_seed", 12345))
    alpha = float(config.get("alpha", 0.05))
    mwu_alternative = config.get("effect_size_mwu_alternative", "two-sided")

    rows = []
    raw_pvals = []

    for idx, (group_a, group_b) in enumerate(comparisons):
        vals_a = _clean_values(values_by_exp.get(group_a, []))
        vals_b = _clean_values(values_by_exp.get(group_b, []))

        if len(vals_a) == 0 or len(vals_b) == 0:
            print(
                f"[warn] Skipping effect-size comparison {group_a} - {group_b}: "
                "one of the groups is empty."
            )
            continue

        diff, ci_low, ci_high = bootstrap_median_difference_ci(
            vals_a,
            vals_b,
            n_bootstrap=n_bootstrap,
            ci_percent=ci_percent,
            rng_seed=base_seed + idx,
        )

        try:
            _, raw_p = mannwhitneyu(vals_a, vals_b, alternative=mwu_alternative)
        except ValueError:
            raw_p = np.nan

        raw_pvals.append(raw_p)

        rows.append({
            "metric": metric_name,
            "group_A": group_a,
            "group_B": group_b,
            "comparison": f"{group_a} - {group_b}",
            "n_A": len(vals_a),
            "n_B": len(vals_b),
            f"median_A_{units}": float(np.median(vals_a)),
            f"median_B_{units}": float(np.median(vals_b)),
            f"median_difference_A_minus_B_{units}": diff,
            f"ci{int(ci_percent)}_low_{units}": ci_low,
            f"ci{int(ci_percent)}_high_{units}": ci_high,
            "bootstrap_n": n_bootstrap,
            "bootstrap_ci_method": "percentile",
            "mwu_alternative": mwu_alternative,
            "raw_p_mwu": raw_p,
        })

    if len(rows) == 0:
        print(f"\n=== Median-difference effect sizes for {metric_name} ===")
        print("No valid comparisons found.")
        return pd.DataFrame()

    raw_pvals_arr = np.asarray(raw_pvals, dtype=float)
    finite_mask = np.isfinite(raw_pvals_arr)

    p_holm = np.full_like(raw_pvals_arr, np.nan, dtype=float)
    reject = np.full_like(raw_pvals_arr, False, dtype=bool)

    if np.any(finite_mask):
        reject_f, p_holm_f, _, _ = multipletests(
            raw_pvals_arr[finite_mask],
            alpha=alpha,
            method="holm"
        )
        p_holm[finite_mask] = p_holm_f
        reject[finite_mask] = reject_f

    for i, row in enumerate(rows):
        row["holm_p_mwu"] = float(p_holm[i]) if np.isfinite(p_holm[i]) else np.nan
        row[f"significant_holm_{alpha}"] = bool(reject[i])

    df = pd.DataFrame(rows)

    print(f"\n=== Median-difference effect sizes for {metric_name} ===")
    print(
        f"Effect size: median(group A) - median(group B) [{units}]\n"
        f"CI: {ci_percent:.0f}% non-parametric percentile bootstrap "
        f"({n_bootstrap} resamples)\n"
        f"Mann-Whitney U alternative for p-values: {mwu_alternative}"
    )

    diff_col = f"median_difference_A_minus_B_{units}"
    ci_low_col = f"ci{int(ci_percent)}_low_{units}"
    ci_high_col = f"ci{int(ci_percent)}_high_{units}"

    for _, row in df.iterrows():
        sig = "YES" if row[f"significant_holm_{alpha}"] else "no"
        print(
            f"{row['comparison']:18s} | "
            f"n={int(row['n_A'])}/{int(row['n_B'])} | "
            f"median diff = {row[diff_col]:7.2f} {units} | "
            f"{ci_percent:.0f}% CI [{row[ci_low_col]:7.2f}, {row[ci_high_col]:7.2f}] {units} | "
            f"raw p = {row['raw_p_mwu']:.4g} | "
            f"Holm p = {row['holm_p_mwu']:.4g} | "
            f"sig: {sig}"
        )

    if config.get("save_stats_csv", True):
        save_dir = config.get("save_path", ".")
        os.makedirs(save_dir, exist_ok=True)
        outfile = os.path.join(
            save_dir,
            config.get(
                "outfile_name_ttp_effect_sizes",
                "ttp_median_difference_effect_sizes_bootstrap_ci.csv"
            )
        )
        df.to_csv(outfile, index=False)
        print(f"[saved] {outfile}")

    return df


def report_reference_vs_others_median_effect_sizes(values_by_exp, config,
                                                   reference_name=None,
                                                   include_only=None,
                                                   ignore=None,
                                                   alternative=None,
                                                   metric_name="TTP",
                                                   units="h"):
    """
    Print and optionally save median-difference effect sizes and bootstrap CIs
    for a directional reference-vs-others comparison.

    This mirrors compare_reference_vs_others(), but adds:
      - median-difference effect size:
            median(reference) - median(comparator)
      - non-parametric percentile bootstrap CI for that difference
      - one-sided Mann-Whitney U p-values if alternative="greater" or "less"
      - Holm correction across the reference-vs-others comparison family

    For the manuscript TTP question:
      reference_name = "6.5/18"
      alternative = "greater"

    Then positive median differences mean that 6.5/18 has a longer TTP than
    the comparator, and the p-value tests:
      TTP(6.5/18) > TTP(comparator)
    """
    if reference_name is None:
        reference_name = config.get("reference_effect_size_reference_name", "6.5/18")

    if include_only is None:
        include_only = config.get("reference_effect_size_include_only", None)

    if ignore is None:
        ignore = tuple(config.get("reference_effect_size_ignore", ("NT",)))

    if alternative is None:
        alternative = config.get("reference_effect_size_mwu_alternative", "greater")

    reference_vals = _clean_values(values_by_exp.get(reference_name, []))
    if len(reference_vals) == 0:
        raise ValueError(f"Reference group '{reference_name}' is empty!")

    n_bootstrap = int(config.get("bootstrap_n", 10000))
    ci_percent = float(config.get("bootstrap_ci_percent", 95))
    base_seed = int(config.get("bootstrap_rng_seed", 12345))
    alpha = float(config.get("alpha", 0.05))

    rows = []
    raw_pvals = []
    comparison_index = 0

    for name, vals in values_by_exp.items():
        if name == reference_name or name in ignore:
            continue
        if include_only is not None and name not in include_only:
            continue

        comparator_vals = _clean_values(vals)
        if len(comparator_vals) == 0:
            continue

        diff, ci_low, ci_high = bootstrap_median_difference_ci(
            reference_vals,
            comparator_vals,
            n_bootstrap=n_bootstrap,
            ci_percent=ci_percent,
            rng_seed=base_seed + 10000 + comparison_index,
        )

        try:
            _, raw_p = mannwhitneyu(
                reference_vals,
                comparator_vals,
                alternative=alternative
            )
        except ValueError:
            raw_p = np.nan

        raw_pvals.append(raw_p)

        rows.append({
            "metric": metric_name,
            "reference_group": reference_name,
            "comparator_group": name,
            "comparison": f"{reference_name} - {name}",
            "n_reference": len(reference_vals),
            "n_comparator": len(comparator_vals),
            f"median_reference_{units}": float(np.median(reference_vals)),
            f"median_comparator_{units}": float(np.median(comparator_vals)),
            f"median_difference_reference_minus_comparator_{units}": diff,
            f"ci{int(ci_percent)}_low_{units}": ci_low,
            f"ci{int(ci_percent)}_high_{units}": ci_high,
            "bootstrap_n": n_bootstrap,
            "bootstrap_ci_method": "percentile",
            "mwu_alternative": alternative,
            "raw_p_mwu": raw_p,
        })

        comparison_index += 1

    if len(rows) == 0:
        print(f"\n=== Reference-vs-others median-difference effect sizes for {metric_name} ===")
        print("No valid comparisons found.")
        return pd.DataFrame()

    raw_pvals_arr = np.asarray(raw_pvals, dtype=float)
    finite_mask = np.isfinite(raw_pvals_arr)

    p_holm = np.full_like(raw_pvals_arr, np.nan, dtype=float)
    reject = np.full_like(raw_pvals_arr, False, dtype=bool)

    if np.any(finite_mask):
        reject_f, p_holm_f, _, _ = multipletests(
            raw_pvals_arr[finite_mask],
            alpha=alpha,
            method="holm"
        )
        p_holm[finite_mask] = p_holm_f
        reject[finite_mask] = reject_f

    for i, row in enumerate(rows):
        row["holm_p_mwu"] = float(p_holm[i]) if np.isfinite(p_holm[i]) else np.nan
        row[f"significant_holm_{alpha}"] = bool(reject[i])

    df = pd.DataFrame(rows)

    print(f"\n=== Reference-vs-others median-difference effect sizes for {metric_name} ===")
    print(
        f"Reference group: {reference_name}\n"
        f"Effect size: median(reference) - median(comparator) [{units}]\n"
        f"CI: {ci_percent:.0f}% non-parametric percentile bootstrap "
        f"({n_bootstrap} resamples)\n"
        f"Mann-Whitney U alternative for p-values: reference {alternative} comparator"
    )

    diff_col = f"median_difference_reference_minus_comparator_{units}"
    ci_low_col = f"ci{int(ci_percent)}_low_{units}"
    ci_high_col = f"ci{int(ci_percent)}_high_{units}"

    for _, row in df.iterrows():
        sig = "YES" if row[f"significant_holm_{alpha}"] else "no"
        print(
            f"{row['comparison']:24s} | "
            f"n={int(row['n_reference'])}/{int(row['n_comparator'])} | "
            f"median diff = {row[diff_col]:7.2f} {units} | "
            f"{ci_percent:.0f}% CI [{row[ci_low_col]:7.2f}, {row[ci_high_col]:7.2f}] {units} | "
            f"raw p = {row['raw_p_mwu']:.4g} | "
            f"Holm p = {row['holm_p_mwu']:.4g} | "
            f"sig: {sig}"
        )

    if config.get("save_stats_csv", True):
        save_dir = config.get("save_path", ".")
        os.makedirs(save_dir, exist_ok=True)
        outfile = os.path.join(
            save_dir,
            config.get(
                "outfile_name_ttp_reference_effect_sizes",
                f"ttp_{reference_name.replace('/', '_')}_vs_others_one_sided_median_difference_effect_sizes_bootstrap_ci.csv"
            )
        )
        df.to_csv(outfile, index=False)
        print(f"[saved] {outfile}")

    return df


def report_group_summary(values_by_exp, config,
                         metric_name="TTP",
                         units="h",
                         outfile_name="ttp_group_summary.csv"):
    """
    Optional group summary table:
    n, median, IQR, min, max for each schedule.
    """
    rows = []

    for name, vals in values_by_exp.items():
        vals = _clean_values(vals)
        if len(vals) == 0:
            continue

        q1, med, q3 = np.percentile(vals, [25, 50, 75])

        rows.append({
            "metric": metric_name,
            "schedule": name,
            "n": len(vals),
            f"median_{units}": float(med),
            f"q1_{units}": float(q1),
            f"q3_{units}": float(q3),
            f"iqr_{units}": float(q3 - q1),
            f"min_{units}": float(np.min(vals)),
            f"max_{units}": float(np.max(vals)),
        })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        return df

    folder_paths = config.get("folder_paths", [])
    ordered = _ordered_names(folder_paths, values_by_exp, config)
    order_lookup = {name: i for i, name in enumerate(ordered)}
    df["_order"] = df["schedule"].map(order_lookup).fillna(9999)
    df = df.sort_values(["_order", "schedule"]).drop(columns=["_order"])

    print(f"\n=== Group summary for {metric_name} ===")
    print(df.to_string(index=False))

    if config.get("save_stats_csv", True):
        save_dir = config.get("save_path", ".")
        os.makedirs(save_dir, exist_ok=True)
        outfile = os.path.join(save_dir, outfile_name)
        df.to_csv(outfile, index=False)
        print(f"[saved] {outfile}")

    return df


# ────────────────────────── Core computation ──────────────────────────

def compute_progression_metrics(folder_paths, identifiers, config):
    """
    Returns:
      ttp_by_exp:        dict {exp_name: [hours, ...]}
      frac_by_exp:       dict {exp_name: [fraction_at_progression_or_censor, ...]}
      style_by_exp:      dict {exp_name: style_cfg}
      ttp_detail_by_exp: dict {
          exp_name: [
              (
                  identifier,
                  ttp_hours,
                  eval_frame,
                  progressed_within_shared_censor,
                  last_frame,
                  full_first_hit_frame,
                  status
              ),
              ...
          ]
      }

    Corrected censoring logic:
    1) First, the script checks the full time series only to find colonies that
       never reach progression.
    2) The shared censor frame is defined as the shortest last_frame among those
       non-progressing colonies, while skipping experiments listed in
       CONFIG["exclude_from_censor_definition"], e.g. ("NT",).
    3) Then ALL colonies, including NT, are evaluated only up to the shared
       censor frame. This means colonies that would progress only after the
       shared censor frame are treated as censored.
    """
    sf = float(config.get("scale_factor", 1.0))
    thresh_mm2 = float(config.get("progression_threshold_mm2", 71.0))
    frames_per_hour = float(config.get("frames_per_hour", 2.0))
    scale_mm2_per_px2 = (sf ** 2) / 1e6

    exclude_from_censor_definition = set(
        config.get("exclude_from_censor_definition", ("NT",))
    )

    name_full, name_base = _prep_name_map(config)
    style_full, style_base = _prep_style_map(config)

    style_by_exp = {}
    raw_records = []

    # ──────────────────────────
    # First pass:
    # Load every colony and determine full-series progression.
    #
    # Important:
    # This full-series progression is ONLY used to define which colonies are
    # non-progressors for selecting the shared censor frame.
    # The final TTP is recalculated later after clipping every colony to the
    # shared censor frame.
    # ──────────────────────────
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

            adj_mm2_full = adj_px2.values * scale_mm2_per_px2
            frames = adj_px2.index.values.astype(int)
            last_frame = int(frames[-1])

            full_hit = np.where(adj_mm2_full >= thresh_mm2)[0]
            full_progressed = full_hit.size > 0
            full_first_hit_frame = int(frames[full_hit[0]]) if full_progressed else None

            raw_records.append({
                "exp_name": exp_name,
                "identifier": ident,
                "colony_file": colony_file,
                "adj_px2": adj_px2,
                "clone_px2": clone_px2,
                "frames": frames,
                "last_frame": last_frame,
                "full_progressed": bool(full_progressed),
                "full_first_hit_frame": full_first_hit_frame,
            })

    if len(raw_records) == 0:
        print("[warn] No valid colony CSV files found.")
        return {}, {}, style_by_exp, {}

    # ──────────────────────────
    # Define shared censor frame.
    #
    # This is now based on the shortest non-progressing COLONY length,
    # not on the shortest experiment that has a non-progressor.
    #
    # NT or other excluded schedules can still be loaded, plotted, and analysed
    # as data points, but they cannot define the shared censor frame.
    # ──────────────────────────
    censor_candidates = [
        rec for rec in raw_records
        if (
            not rec["full_progressed"]
            and rec["exp_name"] not in exclude_from_censor_definition
        )
    ]

    shared_censor_frame = None

    print("\n=== Shared censoring definition ===")
    print(
        "Schedules excluded from defining the shared censor frame: "
        + ", ".join(sorted(exclude_from_censor_definition))
    )
    print(
        "These schedules are still loaded and plotted; they are skipped only "
        "when selecting the shared censor frame."
    )

    if len(censor_candidates) == 0:
        print(
            "No non-progressing colonies found outside the excluded schedules. "
            "No shared censor frame is applied."
        )
    else:
        shared_censor_frame = min(rec["last_frame"] for rec in censor_candidates)

        defining_records = [
            rec for rec in censor_candidates
            if rec["last_frame"] == shared_censor_frame
        ]

        print(
            f"Using shared censor frame {shared_censor_frame} "
            f"({shared_censor_frame / frames_per_hour:.2f} h)."
        )
        print("Defined by the shortest non-progressing colony/colonies outside excluded schedules:")

        for rec in sorted(defining_records, key=lambda r: (r["exp_name"], r["identifier"])):
            print(
                f"  {rec['exp_name']} / {rec['identifier']} "
                f"/ last_frame={rec['last_frame']} "
                f"/ file={os.path.basename(rec['colony_file'])}"
            )

        print("\nAll non-progressing censor candidates outside excluded schedules:")
        for rec in sorted(censor_candidates, key=lambda r: (r["last_frame"], r["exp_name"], r["identifier"])):
            print(
                f"  {rec['exp_name']:12s} {rec['identifier']:>4s} "
                f"last_frame={rec['last_frame']:4d} "
                f"last_time={rec['last_frame'] / frames_per_hour:7.2f} h"
            )

        skipped_candidates = [
            rec for rec in raw_records
            if (
                not rec["full_progressed"]
                and rec["exp_name"] in exclude_from_censor_definition
            )
        ]

        if skipped_candidates:
            print("\nNon-progressing colonies skipped for censor-frame definition:")
            for rec in sorted(skipped_candidates, key=lambda r: (r["exp_name"], r["last_frame"], r["identifier"])):
                print(
                    f"  {rec['exp_name']:12s} {rec['identifier']:>4s} "
                    f"last_frame={rec['last_frame']:4d} "
                    f"last_time={rec['last_frame'] / frames_per_hour:7.2f} h"
                )

    # ──────────────────────────
    # Second pass:
    # Apply shared censoring to ALL colonies before evaluating progression.
    #
    # This is the core correction:
    # A colony that reaches 71 mm² after shared_censor_frame is NOT counted as
    # progressed. It is censored at shared_censor_frame.
    # ──────────────────────────
    ttp_by_exp = {}
    frac_by_exp = {}
    ttp_detail_by_exp = {}

    for rec in raw_records:
        exp_name = rec["exp_name"]
        ident = rec["identifier"]
        adj_px2 = rec["adj_px2"]
        clone_px2 = rec["clone_px2"]
        frames = rec["frames"]
        last_frame = rec["last_frame"]
        full_first_hit_frame = rec["full_first_hit_frame"]

        if shared_censor_frame is None:
            max_eval_frame = last_frame
            censor_limit_status = "no shared censor frame"
        elif shared_censor_frame <= last_frame:
            max_eval_frame = shared_censor_frame
            censor_limit_status = f"shared censor frame {shared_censor_frame}"
        else:
            # This can happen for an excluded schedule such as NT if it is shorter
            # than the shared censor frame. It cannot be evaluated beyond its own
            # available data.
            max_eval_frame = last_frame
            censor_limit_status = (
                f"local last frame {last_frame} "
                f"(shorter than shared censor frame {shared_censor_frame})"
            )
            print(
                f"[warn] {exp_name} {ident}: shared censor frame {shared_censor_frame} "
                f"exceeds available last frame {last_frame}. Falling back to local last frame."
            )

        valid_frames = frames[frames <= max_eval_frame]

        if len(valid_frames) == 0:
            print(f"[warn] {exp_name} {ident}: no frames <= {max_eval_frame}. Skipping.")
            continue

        adj_eval_mm2 = adj_px2.loc[valid_frames].values * scale_mm2_per_px2
        hit = np.where(adj_eval_mm2 >= thresh_mm2)[0]

        if hit.size > 0:
            eval_frame = int(valid_frames[hit[0]])
            progressed_used = True
            status = "progressed within censor window"
        else:
            eval_frame = int(max_eval_frame)
            progressed_used = False

            if (
                full_first_hit_frame is not None
                and shared_censor_frame is not None
                and full_first_hit_frame > max_eval_frame
            ):
                status = (
                    f"censored at {censor_limit_status}; "
                    f"would have progressed later at frame {full_first_hit_frame}"
                )
            else:
                status = f"censored at {censor_limit_status}"

        ttp_h = eval_frame / frames_per_hour

        adj_at_eval = float(adj_px2.loc[eval_frame])
        clone_at_eval = float(clone_px2.loc[eval_frame])

        if adj_at_eval <= 0:
            frac = np.nan
        else:
            frac = clone_at_eval / adj_at_eval

        ttp_by_exp.setdefault(exp_name, []).append(float(ttp_h))
        frac_by_exp.setdefault(exp_name, []).append(float(frac) if np.isfinite(frac) else np.nan)
        ttp_detail_by_exp.setdefault(exp_name, []).append(
            (
                ident,
                float(ttp_h),
                int(eval_frame),
                bool(progressed_used),
                int(last_frame),
                full_first_hit_frame,
                status,
            )
        )

    for k in list(frac_by_exp.keys()):
        frac_by_exp[k] = [v for v in frac_by_exp[k] if np.isfinite(v)]

    print("\n=== TTP per colony after censoring was applied to ALL colonies ===")

    for exp_name in sorted(ttp_detail_by_exp.keys()):
        details = ttp_detail_by_exp[exp_name]
        if len(details) == 0:
            continue

        details_sorted = sorted(details, key=lambda x: (x[1], x[0]))
        ttp_vals = np.array([d[1] for d in details_sorted], dtype=float)
        med = float(np.median(ttp_vals))

        idx_exact = [
            i for i, (_, t, _, _, _, _, _) in enumerate(details_sorted)
            if np.isclose(t, med)
        ]

        if idx_exact:
            med_idx = idx_exact[0]
        else:
            med_idx = int(np.argmin(np.abs(ttp_vals - med)))

        print(f"\n{exp_name}: n={len(details_sorted)} | median={med:.2f} h")

        for i, detail in enumerate(details_sorted):
            (
                ident,
                ttp_h,
                eval_frame,
                progressed_used,
                last_frame,
                full_first_hit_frame,
                status,
            ) = detail

            if full_first_hit_frame is None:
                full_hit_text = "full_hit=none"
            else:
                full_hit_text = f"full_hit={full_first_hit_frame}"

            tag = "  <== MEDIAN" if i == med_idx else ""

            print(
                f"  {ident:>4s}  TTP={ttp_h:7.2f} h   "
                f"(eval_frame={eval_frame:4d}, last_frame={last_frame:4d}, "
                f"{full_hit_text}, progressed_used={progressed_used}, {status}){tag}"
            )

    for name in ttp_by_exp:
        print(
            f"{name}: n={len(ttp_by_exp[name])} TTP values, "
            f"n_frac={len(frac_by_exp.get(name, []))}"
        )

    return ttp_by_exp, frac_by_exp, style_by_exp, ttp_detail_by_exp


# ────────────────────────── Plotting ──────────────────────────

def _ordered_names(folder_paths, values_by_exp, config):
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

        if nm in values_by_exp and len(values_by_exp[nm]) > 0 and nm not in seen:
            ordered.append(nm)
            seen.add(nm)

    for nm, vals in values_by_exp.items():
        if len(vals) > 0 and nm not in seen:
            ordered.append(nm)

    return ordered


def plot_horizontal_violin_cloud(values_by_exp, style_by_exp, config,
                                 xlabel, outfile_key, xlim=None):
    """
    Horizontal violin-style plot with:
    - jittered single-colony points
    - IQR bar
    - median marker
    """
    if not values_by_exp:
        print(f"[warn] No data to plot for {xlabel}.")
        return

    set_nature_style()

    folder_paths = config.get("folder_paths", [])
    ordered_names = _ordered_names(folder_paths, values_by_exp, config)

    if len(ordered_names) == 0:
        print(f"[warn] No non-empty groups to plot for {xlabel}.")
        return

    figsize = config.get("figsize_violin", (2, 2))
    dpi = config.get("dpi", 600)

    point_size = float(config.get("point_size", 12))
    point_alpha = float(config.get("point_alpha", 0.75))
    jitter_sd = float(config.get("point_jitter_sd", 0.06))
    violin_width = float(config.get("violin_width", 0.8))
    rng_seed = int(config.get("rng_seed", 12345))

    rng = np.random.default_rng(rng_seed)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for pos, nm in enumerate(ordered_names, start=1):
        vals = _clean_values(values_by_exp[nm])
        if len(vals) == 0:
            continue

        style_cfg = style_by_exp.get(nm, {})
        facecolor = style_cfg.get("color", "lightgray")
        alpha = float(style_cfg.get("alpha", 1.0))

        if len(vals) >= 2 and len(np.unique(vals)) >= 2:
            vp = ax.violinplot(
                [vals],
                positions=[pos],
                vert=False,
                widths=violin_width,
                showmeans=False,
                showmedians=False,
                showextrema=False
            )

            for body in vp["bodies"]:
                body.set_facecolor(facecolor)
                body.set_edgecolor("black")
                body.set_linewidth(0.5)
                body.set_alpha(alpha)

        y_jitter = rng.normal(loc=pos, scale=jitter_sd, size=len(vals))

        ax.scatter(
            vals,
            y_jitter,
            s=point_size,
            marker="x",
            c="#b0b0b0",
            linewidths=0.6,
            alpha=point_alpha,
            zorder=3
        )

        q1, med, q3 = np.percentile(vals, [25, 50, 75])

        ax.hlines(
            y=pos,
            xmin=q1,
            xmax=q3,
            colors="black",
            linewidth=2.2,
            zorder=4,
            alpha=0.5
        )

        ax.vlines(
            x=med,
            ymin=pos - 0.16,
            ymax=pos + 0.16,
            colors="black",
            linewidth=1.0,
            zorder=5
        )

        ax.scatter(
            [med],
            [pos],
            s=14,
            facecolors="black",
            edgecolors="black",
            linewidths=0.4,
            zorder=6
        )

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


# ────────────────────────── Main configuration ──────────────────────────

if __name__ == "__main__":
    # Define each available condition folder once relative to INPUT_ROOT.
    # Folder names or nested subfolders only need to be changed in this block.
    condition_paths = {
        "nt": os.path.join(INPUT_ROOT, "No_treatment_control"),
        "schedule_4_18": os.path.join(INPUT_ROOT, "4h_18h"),
        "schedule_6_5_20_5": os.path.join(INPUT_ROOT, "6.5h_20.5h"),
        "schedule_6_5_18": os.path.join(INPUT_ROOT, "6.5h_18h"),
        "schedule_6_5_15_5": os.path.join(INPUT_ROOT, "6.5h_15.5h"),
        "schedule_9_20_5": os.path.join(INPUT_ROOT, "9h_20.5h"),
        "schedule_9_18": os.path.join(INPUT_ROOT, "9h_18h"),
        "ct": os.path.join(INPUT_ROOT, "Continuous_therapy"),
    }

    CONFIG = {
        # Select the condition folders included in this run by commenting or
        # uncommenting entries below. No other path-dependent block needs editing.
        "folder_paths": [
            condition_paths["nt"],
            condition_paths["schedule_4_18"],
            #condition_paths["schedule_6_5_20_5"],
            condition_paths["schedule_6_5_18"],
            #condition_paths["schedule_6_5_15_5"],
            #condition_paths["schedule_9_20_5"],
            condition_paths["schedule_9_18"],
            condition_paths["ct"],
        ],

        # Colony identifiers searched for within every active condition folder.
        "identifiers": [
            "P1_", "P2_", "P3", "P4", "P5", "P6", "P7", "P8", "P9",
            "P10", "P11", "P12", "P13", "P14", "P15", "P16", "P17",
            "P21_", "P22", "P24", "P25", "P26", "P27", "P28",
            "P29", "P210", "P211", "P213", "P215"
        ],

        # Conversion factors and progression threshold used for the analysis.
        "scale_factor": 8.648,
        "progression_threshold_mm2": 65.0,
        "frames_per_hour": 2.0,

        # Schedules excluded only from defining the shared censor frame.
        # They remain loaded, analysed, and plotted.
        "exclude_from_censor_definition": ("NT",),

        # Display labels linked to the condition definitions above.
        "display_names": {
            condition_paths["nt"]: "NT",
            condition_paths["ct"]: "CT",
            condition_paths["schedule_9_18"]: "9/18",
            condition_paths["schedule_4_18"]: "4/18",
            condition_paths["schedule_6_5_18"]: "6.5/18",
            condition_paths["schedule_6_5_20_5"]: "6.5/20.5",
            condition_paths["schedule_6_5_15_5"]: "6.5/15.5",
            condition_paths["schedule_9_20_5"]: "9/20.5",
        },

        # Plot colors and transparency linked to the same condition definitions.
        "plot_styles": {
            condition_paths["ct"]: {
                "color": "#7b4173",
                "alpha": 1.0,
            },
            condition_paths["nt"]: {
                "color": "#393b79",
                "alpha": 1.0,
            },
            condition_paths["schedule_4_18"]: {
                "color": "#637939",
                "alpha": 1.0,
            },
            condition_paths["schedule_6_5_18"]: {
                "color": "#8c6d31",
                "alpha": 1.0,
            },
            condition_paths["schedule_9_18"]: {
                "color": "#843c39",
                "alpha": 1.0,
            },
            condition_paths["schedule_6_5_20_5"]: {
                "color": "palegreen",
                "alpha": 1.0,
            },
            condition_paths["schedule_6_5_15_5"]: {
                "color": "teal",
                "alpha": 1.0,
            },
            condition_paths["schedule_9_20_5"]: {
                "color": "plum",
                "alpha": 1.0,
            }
        },

        # Violin-cloud plot dimensions and point appearance.
        "figsize_violin": (2, 2),
        "dpi": 600,
        "point_size": 12,
        "point_alpha": 0.8,
        "point_jitter_sd": 0.055,
        "violin_width": 0.75,
        "rng_seed": 12345,

        # Plot saving settings. Files are written to OUTPUT_ROOT.
        "save_as_pdf": False,
        "save_path": OUTPUT_ROOT,
        "outfile_name_ttp": "time_to_progression_violin_cloud_horizontal_shared_censor_all_colonies_AT.pdf",
        "outfile_name_frac": "clonal_fraction_violin_cloud_horizontal_shared_censor_all_colonies_AT.pdf",

        # Effect-size and confidence-interval output settings.
        "save_stats_csv": False,
        "outfile_name_ttp_effect_sizes": "ttp_median_difference_effect_sizes_bootstrap_ci.csv",
        "outfile_name_ttp_reference_effect_sizes": (
            "ttp_6.5_18_vs_others_one_sided_median_difference_"
            "effect_sizes_bootstrap_ci.csv"
        ),
        "outfile_name_ttp_group_summary": "ttp_group_summary.csv",
        "outfile_name_frac_group_summary": "clonal_fraction_group_summary.csv",

        # Main schedules used for the Fig. 5 effect-size comparisons.
        "main_schedules": ["4/18", "6.5/18", "9/18", "CT"],

        # Comparison mode used when no explicit comparison list is supplied.
        "effect_size_comparison_mode": "all_pairwise_main",

        # Alternative configuration:
        # "effect_size_comparison_mode": "main_vs_control",
        # "effect_size_control_name": "CT",

        # Explicit TTP comparisons used by the current analysis.
        "ttp_effect_size_comparisons": [
            ("4/18", "CT"),
            ("6.5/18", "CT"),
            ("9/18", "CT"),
            ("6.5/15.5", "CT"),
            ("6.5/20.5", "CT"),
            ("9/20.5", "CT"),
            ("4/18", "6.5/18"),
            ("9/18", "6.5/18"),
            ("6.5/15.5", "6.5/18"),
            ("6.5/20.5", "6.5/18"),
            ("9/20.5", "6.5/18"),
        ],

        # Bootstrap sample count, confidence level, and random seed.
        "bootstrap_n": 10000,
        "bootstrap_ci_percent": 95,
        "bootstrap_rng_seed": 12345,

        # Two-sided p-values accompanying the general pairwise effect sizes.
        "effect_size_mwu_alternative": "two-sided",

        # Directional 6.5/18-versus-others effect-size comparison settings.
        # Positive differences indicate a longer TTP for 6.5/18.
        "reference_effect_size_reference_name": "6.5/18",
        "reference_effect_size_include_only": (
            "4/18", "9/18", "CT", "6.5/20.5",
            "6/18", "6.5/15.5", "9/20.5"
        ),
        "reference_effect_size_ignore": ("NT",),
        "reference_effect_size_mwu_alternative": "greater",

        # Significance threshold used for Holm correction.
        "alpha": 0.05,
    }

    ttp_by_exp, frac_by_exp, style_by_exp, ttp_detail_by_exp = compute_progression_metrics(
        CONFIG["folder_paths"],
        CONFIG["identifiers"],
        CONFIG,
    )

    plot_horizontal_violin_cloud(
        ttp_by_exp,
        style_by_exp,
        CONFIG,
        xlabel="Time to progression (h)",
        outfile_key="outfile_name_ttp",
        xlim=(50, 170),
    )

    plot_horizontal_violin_cloud(
        frac_by_exp,
        style_by_exp,
        CONFIG,
        xlabel="Clonal fraction at progression / shared censor time",
        outfile_key="outfile_name_frac",
        xlim=(0, 1),
    )

    # One-sided TTP comparisons against continuous therapy.
    compare_vs_control(
        ttp_by_exp,
        control_name="CT",
        ignore=("NT",),
        alpha=CONFIG["alpha"],
        alternative="greater",
        metric_name="TTP"
    )

    # One-sided TTP comparisons using 6.5/18 as the reference schedule.
    compare_reference_vs_others(
        ttp_by_exp,
        reference_name="6.5/18",
        include_only=CONFIG["reference_effect_size_include_only"],
        ignore=CONFIG["reference_effect_size_ignore"],
        alpha=CONFIG["alpha"],
        alternative=CONFIG["reference_effect_size_mwu_alternative"],
        metric_name="TTP"
    )

    # Directional 6.5/18-versus-others TTP effect sizes and bootstrap CIs.
    report_reference_vs_others_median_effect_sizes(
        ttp_by_exp,
        CONFIG,
        reference_name=CONFIG["reference_effect_size_reference_name"],
        include_only=CONFIG["reference_effect_size_include_only"],
        ignore=CONFIG["reference_effect_size_ignore"],
        alternative=CONFIG["reference_effect_size_mwu_alternative"],
        metric_name="TTP",
        units="h",
    )

    # Group-level TTP summary and configured pairwise effect sizes.
    report_group_summary(
        ttp_by_exp,
        CONFIG,
        metric_name="TTP",
        units="h",
        outfile_name=CONFIG["outfile_name_ttp_group_summary"],
    )

    report_ttp_median_effect_sizes(
        ttp_by_exp,
        CONFIG,
        metric_name="TTP",
        units="h",
    )

    # Clonal-fraction comparisons against continuous therapy.
    compare_vs_control(
        frac_by_exp,
        control_name="CT",
        ignore=("NT",),
        alpha=CONFIG["alpha"],
        alternative="less",
        metric_name="clonal fraction at progression / shared censor time"
    )

    # Two-sided clonal-fraction comparisons using 6.5/18 as the reference.
    compare_reference_vs_others(
        frac_by_exp,
        reference_name="6.5/18",
        include_only=("4/18", "9/18", "CT", "6.5/20.5", "6/18", "6.5/15.5", "6.5/20.5", "9/20.5"),
        ignore=("NT",),
        alpha=CONFIG["alpha"],
        alternative="two-sided",
        metric_name="clonal fraction at progression / shared censor time"
    )

    # Group-level clonal-fraction summary.
    report_group_summary(
        frac_by_exp,
        CONFIG,
        metric_name="clonal fraction at progression / shared censor time",
        units="fraction",
        outfile_name=CONFIG["outfile_name_frac_group_summary"],
    )
