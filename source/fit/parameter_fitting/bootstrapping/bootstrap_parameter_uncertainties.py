import concurrent.futures as cf
import datetime as dt
import json
import logging
import os
import pathlib as pl
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import yaml
import source.fit.parameter_fitting.calculate_mean_cell_area_from_csv as mean_cell_area
import source.fit.parameter_fitting.delay_parameter_estimation as delay_fit
import Validation.validation_common as vc


def scalarize_parameter(value):
    return float(np.atleast_1d(value)[0])


def build_run_args(config):
    run_config = dict(config)
    run_config["targets"] = [str(target) for target in run_config["targets"]]
    run_config["output_dir"] = pl.Path(run_config["output_dir"])
    run_config["single_cell_area_csv"] = pl.Path(run_config["single_cell_area_csv"])
    if run_config["dispersion_initial"] is not None:
        run_config["dispersion_initial"] = [float(value) for value in run_config["dispersion_initial"]]
        if len(run_config["dispersion_initial"]) != 3:
            raise ValueError("dispersion_initial must contain exactly three values.")
    return SimpleNamespace(**run_config)


def create_run_directory(base_dir):
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def setup_logger(log_path):
    logger = logging.getLogger(f"bootstrap_parameter_uncertainties_{log_path.stem}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def json_ready(value):
    if isinstance(value, pl.Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def encode_json_cell(value):
    return json.dumps(json_ready(value), separators=(",", ":"))


def bootstrap_indices(length, rng):
    length = int(length)
    if length <= 0:
        return []
    return [int(index) for index in rng.integers(0, length, size=length)]


def log_progress(logger, label, current, total):
    if total <= 0:
        return
    checkpoint = max(1, total // 5)
    if current == 1 or current == total or current % checkpoint == 0:
        logger.info("%s bootstrap %d/%d", label, current, total)


def log_exception(logger, message, *args):
    if logger is not None:
        logger.exception(message, *args)


def summarize_samples(parameter, point_estimate, samples, ci_level, n_attempted, n_converged, method, notes="", point_fit_converged=None):
    valid_samples = np.asarray(samples, dtype=float)
    valid_samples = valid_samples[np.isfinite(valid_samples)]
    alpha = (100.0 - ci_level) / 2.0

    summary = {"parameter": parameter,
               "point_estimate": float(point_estimate),
               "bootstrap_mean": np.nan,
               "bootstrap_std": np.nan,
               "ci_lower": np.nan,
               "ci_upper": np.nan,
               "ci_level": float(ci_level),
               "n_attempted": int(n_attempted),
               "n_success": int(valid_samples.size),
               "n_converged": int(n_converged),
               "point_fit_converged": point_fit_converged,
               "method": method,
               "notes": notes}

    if valid_samples.size:
        summary["bootstrap_mean"] = float(np.mean(valid_samples))
        summary["bootstrap_std"] = (float(np.std(valid_samples, ddof=1)) if valid_samples.size > 1 else 0.0)
        summary["ci_lower"] = float(np.percentile(valid_samples, alpha))
        summary["ci_upper"] = float(np.percentile(valid_samples, 100.0 - alpha))

    return summary


def summarize_point_only(parameter, point_estimate, method, notes):
    return {"parameter": parameter,
            "point_estimate": float(point_estimate),
            "bootstrap_mean": np.nan,
            "bootstrap_std": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "ci_level": np.nan,
            "n_attempted": 0,
            "n_success": 0,
            "n_converged": 0,
            "point_fit_converged": None,
            "method": method,
            "notes": notes}


def fits_are_finite(result):
    return np.all(np.isfinite(np.asarray(result.x, dtype=float))) and np.isfinite(float(result.fun))


def constant_sample_array(point_value, size):
    return np.full(int(size), float(point_value), dtype=float)


def sample_input_distribution(mean_value, sem_value, size, rng, positive_only):
    draws = np.full(int(size), np.nan, dtype=float)
    if sem_value is None or float(sem_value) <= 0:
        return draws, 0

    sampled = rng.normal(float(mean_value), float(sem_value), size=int(size))
    valid_mask = np.isfinite(sampled)
    if positive_only:
        valid_mask &= sampled > 0
    draws[valid_mask] = sampled[valid_mask]
    return draws, int(np.count_nonzero(valid_mask))


def summarize_joint_samples(parameter, point_estimate, sample_values, ci_level, n_attempted, method, notes, point_fit_converged=None, n_converged=None):
    finite_count = int(np.count_nonzero(np.isfinite(np.asarray(sample_values, dtype=float))))
    converged = finite_count if n_converged is None else int(n_converged)
    return summarize_samples(parameter, point_estimate=point_estimate, samples=sample_values, ci_level=ci_level, n_attempted=n_attempted, n_converged=converged, method=method, notes=notes, point_fit_converged=point_fit_converged)


def regrowth_frames_from_calibration(offsets_in_sim_pixels, t0_h, slope_um_per_h, scale_factor=8.648, frames_per_hour=2.0, sim_pixel_size_um=13.76):
    offsets = np.asarray(offsets_in_sim_pixels, dtype=float)
    t0_h = np.asarray(t0_h, dtype=float)
    slope_um_per_h = np.asarray(slope_um_per_h, dtype=float)
    offset_um = offsets * float(sim_pixel_size_um) * float(scale_factor)
    with np.errstate(divide="ignore", invalid="ignore"):
        detection_time_h = offset_um / slope_um_per_h + t0_h
    return detection_time_h * float(frames_per_hour)


def build_growth_rate_draws(growth_time, growth_time_sem, n_bootstrap, rng, notes):
    point_estimate = float(growth_rate(growth_time))
    if growth_time_sem is None or float(growth_time_sem) <= 0:
        return {"point_estimate": point_estimate,
                "samples": constant_sample_array(point_estimate, n_bootstrap),
                "summary_kind": "point_only",
                "method": "deterministic_formula",
                "notes": notes,
                "n_converged": 0}

    time_draws, valid_count = sample_input_distribution(growth_time, growth_time_sem, n_bootstrap, rng, positive_only=True)
    rate_draws = np.full(int(n_bootstrap), np.nan, dtype=float)
    valid_mask = np.isfinite(time_draws)
    rate_draws[valid_mask] = growth_rate(time_draws[valid_mask])
    return {"point_estimate": point_estimate,
            "samples": rate_draws,
            "summary_kind": "sampled",
            "method": "input_sampling",
            "notes": notes,
            "n_converged": valid_count}


def build_derived_parameter_draws(args, rng):
    n_bootstrap = int(args.n_bootstrap)
    derived_specs = {}
    diagnostics = {}

    single_cell_values, _, _ = mean_cell_area.load_area_values(csv_path=args.single_cell_area_csv, only_accepted=not args.single_cell_area_include_rejected)
    single_cell_summary = mean_cell_area.summarize_values(single_cell_values)
    sim_pixel_area_um2 = mean_cell_area.load_sim_pixel_area_um2(params_path=vc.find_project_root(os.getcwd(), "setup.py") / "params.yaml", exp_pixel_size_um=args.single_cell_exp_pixel_size_um)
    mutation_scaling_point, mutation_scaling_sem = (mean_cell_area.calculate_scaling_factor(mean_area=float(single_cell_summary["mean"]), area_sem=float(single_cell_summary["sem"]), sim_pixel_area_um2=sim_pixel_area_um2))

    if mutation_scaling_sem <= 0:
        mutation_scaling_samples = constant_sample_array(mutation_scaling_point, n_bootstrap)
        derived_specs["mutation_scaling"] = {"point_estimate": float(mutation_scaling_point),
                                             "samples": mutation_scaling_samples,
                                             "summary_kind": "point_only",
                                             "method": "deterministic_formula",
                                             "notes": ("Derived from the single-cell area CSV without refitting. No uncertainty attached because the mean-area SEM was zero."),
                                             "n_converged": 0}
    else:
        mean_area_draws, valid_count = sample_input_distribution(float(single_cell_summary["mean"]), float(single_cell_summary["sem"]), n_bootstrap, rng, positive_only=True)
        mutation_scaling_samples = np.full(int(n_bootstrap), np.nan, dtype=float)
        valid_mask = np.isfinite(mean_area_draws)
        mutation_scaling_samples[valid_mask] = (sim_pixel_area_um2 / mean_area_draws[valid_mask])
        derived_specs["mutation_scaling"] = {"point_estimate": float(mutation_scaling_point),
                                             "samples": mutation_scaling_samples,
                                             "summary_kind": "sampled",
                                             "method": "input_sampling",
                                             "notes": ("Derived from the single-cell area CSV by sampling the mean-area uncertainty instead of refitting the legacy mutation-scaling model."),
                                             "n_converged": valid_count}

    diagnostics["single_cell_area"] = {"n_cells": int(single_cell_summary["n"]),
                                       "mean_area": float(single_cell_summary["mean"]),
                                       "sem_area": float(single_cell_summary["sem"]),
                                       "sim_pixel_area_um2": float(sim_pixel_area_um2),
                                       "mutation_scaling_point": float(mutation_scaling_point),
                                       "mutation_scaling_sem": float(mutation_scaling_sem)}

    if args.independent_growth_rates:
        derived_specs["sensitive_growth_rate"] = build_growth_rate_draws(args.sensitive_growth_time, args.sensitive_growth_time_sem, n_bootstrap, rng, notes="Approximate uncertainty obtained by sampling the sensitive growth-time input while keeping the formula fixed.")
        derived_specs["resistant_growth_rate"] = build_growth_rate_draws(args.resistant_growth_time, args.resistant_growth_time_sem, n_bootstrap, rng, notes="Approximate uncertainty obtained by sampling the resistant growth-time input while keeping the formula fixed.")
    else:
        shared_spec = build_growth_rate_draws(args.sensitive_growth_time, args.sensitive_growth_time_sem, n_bootstrap, rng, notes=("Approximate uncertainty obtained by sampling the shared growth-time input while keeping the formula fixed. resistant_growth_rate is tied to the same draw."))
        derived_specs["sensitive_growth_rate"] = shared_spec
        derived_specs["resistant_growth_rate"] = {**shared_spec, "notes": ("Tied to sensitive_growth_rate to match the equal-growth-rate assumption in params.yaml.")}

    effective_resistant_growth_time = (float(args.resistant_growth_time) if args.independent_growth_rates else float(args.sensitive_growth_time))
    effective_resistant_growth_time_sem = (None if (args.resistant_growth_time_sem is None if args.independent_growth_rates else args.sensitive_growth_time_sem is None) else (float(args.resistant_growth_time_sem) if args.independent_growth_rates else float(args.sensitive_growth_time_sem)))

    diagnostics["growth_rate_inputs"] = {"independent_growth_rates": bool(args.independent_growth_rates),
                                         "sensitive_growth_time": float(args.sensitive_growth_time),
                                         "sensitive_growth_time_sem": (None if args.sensitive_growth_time_sem is None else float(args.sensitive_growth_time_sem)),
                                         "resistant_growth_time": effective_resistant_growth_time,
                                         "resistant_growth_time_sem": effective_resistant_growth_time_sem}

    offset_draws = {}
    for offset in args.regrowth_offsets:
        parameter_name = (f"res_regrowth_frame_from_offset_{format_number_for_name(offset)}")
        offset_draws[parameter_name] = constant_sample_array(offset, n_bootstrap)
        diagnostics[parameter_name] = {"offset_sim_pixels": float(offset),
                                       "offset_sem": None}

    return derived_specs, offset_draws, diagnostics


def build_delay_parameter_draws(args, params_yaml, rng):
    n_bootstrap = int(args.n_bootstrap)
    analysis = delay_fit.analyze_delay_parameters()
    center_values = {"treatment_delay": float(params_yaml["treatment_delay"]),
                     "release_delay": float(params_yaml["release_delay"]),
                     "lag_steps": float(params_yaml["lag_steps"]),
                     "overshoot_steps": float(params_yaml["overshoot_steps"])}
    
    sampled_values = delay_fit.sample_delay_parameter_sets(analysis, n_bootstrap, rng, center_values=center_values)

    specs = {}
    for parameter_name, point_estimate in center_values.items():
        notes = ("Point estimate kept at params.yaml while uncertainty was sampled from the shared delay-fit analysis used by SI_Figures/sim_val/delay_plot.py.")
        if parameter_name in ("lag_steps", "overshoot_steps"):
            notes += " The uncertainty shape comes from the weighted 4-segment grid-fit candidates."
        else:
            notes += " The uncertainty shape comes from the least-squares covariance of the experimental fits."

        samples = np.asarray(sampled_values[parameter_name], dtype=float)
        specs[parameter_name] = {"point_estimate": float(point_estimate),
                                 "samples": samples,
                                 "summary_kind": "sampled",
                                 "method": "analytic_fit_sampling",
                                 "notes": notes,
                                 "n_converged": int(np.count_nonzero(np.isfinite(samples)))}

    diagnostics = {"point_estimates_from_params_yaml": center_values,
                   "fitted_estimates_from_delay_analysis": {key: float(spec["estimate"]) for key, spec in analysis["parameter_estimates"].items()},
                   "fitted_standard_errors": {key: float(spec["stderr"]) for key, spec in analysis["parameter_estimates"].items()},
                   "legacy_rounded_delay_values": {key: float(spec["legacy_rounded"]) for key, spec in analysis["parameter_estimates"].items()},
                   "four_segment_candidate_count": int(analysis["fits"]["four_segment"]["n_candidates"])}
    return specs, diagnostics


def build_upstream_simulation_overrides(derived_values):
    upstream_parameter_names = ("mutation_scaling", "sensitive_growth_rate", "resistant_growth_rate", "treatment_delay", "release_delay", "lag_steps", "overshoot_steps")
    return {name: float(derived_values[name]) for name in upstream_parameter_names if name in derived_values and np.isfinite(float(derived_values[name]))}


def growth_rate(measurement):
    return np.log(2.0) / measurement


def format_number_for_name(value):
    value_str = f"{value:g}"
    return value_str.replace("-", "m").replace(".", "p")


def build_hierarchical_bootstrap_tasks(args, params_yaml, n_bootstrap, need_regrowth, need_dispersion, need_mutation_rate, continuous_wells, clone_count_map, no_treatment_labels, no_treatment_area_observations, derived_specs, regrowth_point_fit, point_dispersion_estimates, point_mutation_rate, rng):
    sample_seeds = [int(seed) for seed in rng.integers(0, np.iinfo(np.int64).max, size=int(n_bootstrap), dtype=np.int64)]
    tasks = []
    for sample_idx, sample_seed in enumerate(sample_seeds):
        tasks.append({"sample_id": int(sample_idx),
                "sample_label": f"bootstrap_joint_{sample_idx:03d}",
                "sample_seed": int(sample_seed),
                "derived_values": {parameter_name: float(spec["samples"][sample_idx]) for parameter_name, spec in derived_specs.items()},
                "params_yaml": params_yaml,
                "maxiter": int(args.maxiter),
                "mutation_rate_sim_replicas": int(args.mutation_rate_sim_replicas),
                "regrowth_min_duration_frames": int(args.regrowth_min_duration_frames),
                "regrowth_min_distance_um": float(args.regrowth_min_distance_um),
                "need_regrowth": bool(need_regrowth),
                "need_dispersion": bool(need_dispersion),
                "need_mutation_rate": bool(need_mutation_rate),
                "continuous_wells": list(continuous_wells),
                "clone_count_map": dict(clone_count_map),
                "no_treatment_labels": list(no_treatment_labels),
                "no_treatment_area_observations": [np.asarray(area_observation, dtype=float) for area_observation in no_treatment_area_observations],
                "point_regrowth_p0": ([float(regrowth_point_fit["t0_h"]), float(regrowth_point_fit["slope_um_per_h"])] if regrowth_point_fit is not None else None),
                "point_dispersion_estimates": (np.asarray(point_dispersion_estimates, dtype=float) if point_dispersion_estimates is not None else None),
                "point_mutation_rate": (float(point_mutation_rate) if point_mutation_rate is not None else None)})
    return tasks


def run_single_hierarchical_bootstrap_sample(task, logger=None):
    sample_rng = np.random.default_rng(int(task["sample_seed"]))

    sample_row = {"sample_id": int(task["sample_id"]),
                  "sample_label": str(task["sample_label"]),
                  "sample_source": "hierarchical_bootstrap_refit",
                  "sample_seed": int(task["sample_seed"]),
                  "bootstrap_continuous_well_indices": encode_json_cell([]),
                  "bootstrap_continuous_well_labels": encode_json_cell([]),
                  "bootstrap_no_treatment_indices": encode_json_cell([]),
                  "bootstrap_no_treatment_labels": encode_json_cell([]),
                  "regrowth_t0_h": np.nan,
                  "regrowth_slope_um_per_h": np.nan,
                  "diffusion_sensitive": np.nan,
                  "diffusion_resistant": np.nan,
                  "uptake_rate": np.nan,
                  "diffusion_nutrients": np.nan,
                  "start_point": np.nan,
                  "mutation_rate": np.nan,
                  "regrowth_fit_converged": False,
                  "dispersion_fit_converged": False,
                  "mutation_rate_fit_converged": False,
                  "regrowth_training_points": np.nan,
                  "regrowth_error": None,
                  "dispersion_objective": np.nan,
                  "dispersion_fit_message": None,
                  "dispersion_error": None,
                  "mutation_rate_objective": np.nan,
                  "mutation_rate_fit_message": None,
                  "mutation_rate_error": None,
                  "mutation_rate_skip_reason": None,
                  "mutation_rate_start_point_used": np.nan,
                  "mutation_rate_num_wells": np.nan}

    for parameter_name, parameter_value in task["derived_values"].items():
        sample_row[parameter_name] = float(parameter_value)

    upstream_simulation_overrides = build_upstream_simulation_overrides(task["derived_values"])
    continuous_well_indices = []
    bootstrap_wells = []
    bootstrap_clone_counts = None
    if task["need_regrowth"] or task["need_mutation_rate"]:
        continuous_wells = list(task["continuous_wells"])
        continuous_well_indices = bootstrap_indices(len(continuous_wells), sample_rng)
        bootstrap_wells = [continuous_wells[index] for index in continuous_well_indices]
        bootstrap_clone_counts = np.asarray([task["clone_count_map"][well] for well in bootstrap_wells], dtype=float)
        sample_row["bootstrap_continuous_well_indices"] = encode_json_cell(continuous_well_indices)
        sample_row["bootstrap_continuous_well_labels"] = encode_json_cell(bootstrap_wells)

    regrowth_fit = None
    mutation_simulation_overrides = None
    if task["need_regrowth"]:
        try:
            regrowth_fit = vc.fit_regrowth_calibration_from_continuous_therapy(wells=bootstrap_wells, min_duration_frames=int(task["regrowth_min_duration_frames"]), min_distance_um=float(task["regrowth_min_distance_um"]), p0=tuple(task["point_regrowth_p0"]))
            sample_row["regrowth_t0_h"] = float(regrowth_fit["t0_h"])
            sample_row["regrowth_slope_um_per_h"] = float(regrowth_fit["slope_um_per_h"])
            sample_row["regrowth_training_points"] = int(regrowth_fit["n_points"])
            sample_row["regrowth_fit_converged"] = bool(np.isfinite(sample_row["regrowth_t0_h"]) and np.isfinite(sample_row["regrowth_slope_um_per_h"]))
        except Exception as exc:
            sample_row["regrowth_error"] = f"{type(exc).__name__}: {exc}"
            log_exception(logger, "regrowth bootstrap replicate %d failed.", sample_row["sample_id"] + 1)

    if task["need_dispersion"]:
        if regrowth_fit is None or not sample_row["regrowth_fit_converged"]:
            sample_row["dispersion_error"] = ("Skipped because regrowth calibration was unavailable for this replicate.")
        else:
            no_treatment_indices = bootstrap_indices(len(task["no_treatment_area_observations"]), sample_rng)
            bootstrap_area = [np.asarray(task["no_treatment_area_observations"][index], dtype=float) for index in no_treatment_indices]
            no_treatment_labels = [task["no_treatment_labels"][index] for index in no_treatment_indices]
            sample_row["bootstrap_no_treatment_indices"] = encode_json_cell(no_treatment_indices)
            sample_row["bootstrap_no_treatment_labels"] = encode_json_cell(no_treatment_labels)
            try:
                bootstrap_fit = vc.fit_dispersion_dataset(area_observations=bootstrap_area, initial_guess=np.asarray(task["point_dispersion_estimates"], dtype=float), params_yaml=task["params_yaml"], regrowth_calibration=regrowth_fit, maxiter=int(task["maxiter"]), simulation_param_overrides=upstream_simulation_overrides)
                sample_row["dispersion_fit_converged"] = bool(bootstrap_fit.result.success)
                sample_row["dispersion_objective"] = float(bootstrap_fit.result.fun)
                sample_row["dispersion_fit_message"] = str(bootstrap_fit.result.message)
                if fits_are_finite(bootstrap_fit.result):
                    bootstrap_params = np.asarray(bootstrap_fit.result.x, dtype=float)
                    mutation_simulation_overrides = dict(upstream_simulation_overrides)
                    mutation_simulation_overrides.update(vc.build_dispersion_parameter_overrides(bootstrap_params))
                    sample_row["diffusion_sensitive"] = float(bootstrap_params[0])
                    sample_row["diffusion_resistant"] = float(bootstrap_params[0])
                    sample_row["uptake_rate"] = float(bootstrap_params[1])
                    sample_row["diffusion_nutrients"] = float(bootstrap_params[2])
                    sample_row["start_point"] = float(bootstrap_fit.metadata["start_point"])
            except Exception as exc:
                sample_row["dispersion_error"] = f"{type(exc).__name__}: {exc}"
                log_exception(logger, "dispersion bootstrap replicate %d failed.", sample_row["sample_id"] + 1)

    if task["need_mutation_rate"]:
        if task["need_dispersion"] and not np.isfinite(sample_row["start_point"]):
            sample_row["mutation_rate_skip_reason"] = ("Skipped because no dispersion-derived start_point was available.")
        else:
            mutation_start_point = (int(round(sample_row["start_point"])) if task["need_dispersion"] else None)
            if mutation_simulation_overrides is None:
                mutation_simulation_overrides = dict(upstream_simulation_overrides)
            try:
                bootstrap_fit = vc.fit_mutation_rate_dataset(clone_counts=bootstrap_clone_counts, initial_guess=float(task["point_mutation_rate"]), params_yaml=task["params_yaml"], maxiter=int(task["maxiter"]), sim_replicas=int(task["mutation_rate_sim_replicas"]), start_point_override=mutation_start_point, simulation_param_overrides=mutation_simulation_overrides)
                sample_row["mutation_rate_fit_converged"] = bool(bootstrap_fit.result.success)
                sample_row["mutation_rate_objective"] = float(bootstrap_fit.result.fun)
                sample_row["mutation_rate_fit_message"] = str(bootstrap_fit.result.message)
                sample_row["mutation_rate_start_point_used"] = int(bootstrap_fit.metadata["start_point_used"])
                sample_row["mutation_rate_num_wells"] = int(bootstrap_fit.metadata["num_wells"])
                if fits_are_finite(bootstrap_fit.result):
                    sample_row["mutation_rate"] = scalarize_parameter(bootstrap_fit.result.x)
            except Exception as exc:
                sample_row["mutation_rate_error"] = f"{type(exc).__name__}: {exc}"
                log_exception(logger, "mutation_rate bootstrap replicate %d failed.", sample_row["sample_id"] + 1)

    return sample_row


def run_hierarchical_bootstrap_tasks(sample_tasks, worker_count, logger):
    total_tasks = len(sample_tasks)
    if worker_count == 1:
        rows = []
        for task in sample_tasks:
            log_progress(logger, "hierarchical", task["sample_id"] + 1, total_tasks)
            row = run_single_hierarchical_bootstrap_sample(task, logger=logger)
            logger.info("Completed hierarchical bootstrap replicate %s (regrowth=%s, dispersion=%s, mutation_rate=%s).", row["sample_label"], row["regrowth_fit_converged"], row["dispersion_fit_converged"], row["mutation_rate_fit_converged"])
            rows.append(row)
        return rows

    logger.info("Submitting %d hierarchical bootstrap replicate(s) to the process pool. Worker logs are muted so the main log stays readable.", total_tasks)
    rows = []
    completed = 0
    with cf.ProcessPoolExecutor(max_workers=int(worker_count)) as executor:
        future_map = {executor.submit(run_single_hierarchical_bootstrap_sample, task): task for task in sample_tasks}
        logger.info("All bootstrap replicates submitted; waiting for completed replicates.")
        for future in cf.as_completed(future_map):
            task = future_map[future]
            try:
                row = future.result()
            except Exception:
                logger.exception("Hierarchical bootstrap replicate %s crashed.", task["sample_label"])
                raise
            completed += 1
            logger.info("Completed hierarchical bootstrap replicate %s (%d/%d finished; regrowth=%s, dispersion=%s, mutation_rate=%s).", row["sample_label"], completed, total_tasks, row["regrowth_fit_converged"], row["dispersion_fit_converged"], row["mutation_rate_fit_converged"])
            rows.append(row)

    rows.sort(key=lambda row: row["sample_id"])
    return rows


def run_hierarchical_bootstrap(args, params_yaml, rng, logger, worker_count):
    need_regrowth = ("dispersion" in args.targets) or ("derived" in args.targets)
    need_dispersion = "dispersion" in args.targets
    need_mutation_rate = "mutation_rate" in args.targets
    n_bootstrap = int(args.n_bootstrap)

    summary_rows = []
    raw_samples = {}
    diagnostics = {}
    derived_specs = {}
    regrowth_offset_draws = {}
    upstream_parameter_draws_needed = ("derived" in args.targets) or need_dispersion or need_mutation_rate

    if upstream_parameter_draws_needed:
        derived_specs, regrowth_offset_draws, derived_diagnostics = build_derived_parameter_draws(args, rng)
        diagnostics["derived"] = derived_diagnostics

    if upstream_parameter_draws_needed:
        delay_specs, delay_diagnostics = build_delay_parameter_draws(args, params_yaml, rng)
        derived_specs.update(delay_specs)
        diagnostics["delay_parameters"] = delay_diagnostics

    continuous_dataset = (vc.load_mutation_rate_dataset() if (need_regrowth or need_mutation_rate) else [])
    no_treatment_dataset = (vc.load_no_treatment_area_dataset() if need_dispersion else [])

    continuous_wells = [str(entry["label"]) for entry in continuous_dataset]
    clone_count_map = {str(entry["label"]): float(entry["clone_count"]) for entry in continuous_dataset}
    no_treatment_labels = [str(entry["label"]) for entry in no_treatment_dataset]
    full_clone_counts = np.asarray([clone_count_map[well] for well in continuous_wells], dtype=float)
    full_area_observations = [np.asarray(entry["area"], dtype=float) for entry in no_treatment_dataset]

    regrowth_point_fit = None
    if need_regrowth:
        logger.info("Fitting continuous-therapy regrowth calibration on the observed dataset.")
        regrowth_point_fit = vc.fit_regrowth_calibration_from_continuous_therapy(wells=continuous_wells, min_duration_frames=int(args.regrowth_min_duration_frames), min_distance_um=float(args.regrowth_min_distance_um))
        diagnostics["regrowth_calibration"] = {"point_estimate": {"t0_h": float(regrowth_point_fit["t0_h"]),
                                                                  "slope_um_per_h": float(regrowth_point_fit["slope_um_per_h"])},
                                               "n_wells": int(len(continuous_wells)),
                                               "n_points": int(regrowth_point_fit["n_points"])}

    dispersion_initial = (np.array(args.dispersion_initial, dtype=float) if args.dispersion_initial is not None else np.array([float(params_yaml["diffusion_sensitive"]), float(params_yaml["uptake_rate"]), float(params_yaml["diffusion_nutrients"])], dtype=float))
    point_dispersion_fit = None
    point_dispersion_estimates = None
    point_start_point = int(params_yaml["start_point"])
    point_mutation_simulation_overrides = None
    if need_dispersion:
        logger.info("Fitting dispersion and nutrient parameters on the observed dataset.")
        point_dispersion_fit = vc.fit_dispersion_dataset(area_observations=full_area_observations, initial_guess=dispersion_initial, params_yaml=params_yaml, regrowth_calibration=regrowth_point_fit, maxiter=args.maxiter)
        point_dispersion_estimates = np.asarray(point_dispersion_fit.result.x, dtype=float)
        point_start_point = int(point_dispersion_fit.metadata["start_point"])
        point_mutation_simulation_overrides = vc.build_dispersion_parameter_overrides(point_dispersion_estimates)
        diagnostics["dispersion"] = {"point_estimate": point_dispersion_estimates.tolist(),
                                     "point_objective": float(point_dispersion_fit.result.fun),
                                     "point_fit_converged": bool(point_dispersion_fit.result.success),
                                     "point_fit_message": str(point_dispersion_fit.result.message),
                                     "start_point": point_start_point,
                                     "regrowth_t0_h": float(regrowth_point_fit["t0_h"]),
                                     "regrowth_slope_um_per_h": float(regrowth_point_fit["slope_um_per_h"])}

    mutation_rate_initial = (float(args.mutation_rate_initial) if args.mutation_rate_initial is not None else float(params_yaml["mutation_rate"]))
    point_mutation_fit = None
    point_mutation_rate = float(params_yaml["mutation_rate"])
    if need_mutation_rate:
        logger.info("Fitting mutation_rate on the observed dataset.")
        point_mutation_fit = vc.fit_mutation_rate_dataset(clone_counts=full_clone_counts, initial_guess=mutation_rate_initial, params_yaml=params_yaml, maxiter=args.maxiter, sim_replicas=args.mutation_rate_sim_replicas, start_point_override=point_start_point if need_dispersion else None, simulation_param_overrides=point_mutation_simulation_overrides)

        point_mutation_rate = scalarize_parameter(point_mutation_fit.result.x)
        diagnostics["mutation_rate"] = {"point_estimate": point_mutation_rate,
                                        "point_objective": float(point_mutation_fit.result.fun),
                                        "point_fit_converged": bool(point_mutation_fit.result.success),
                                        "point_fit_message": str(point_mutation_fit.result.message),
                                        "num_wells": int(point_mutation_fit.metadata["num_wells"]),
                                        "start_point_used": int(point_mutation_fit.metadata["start_point_used"])}
    diagnostics["bootstrap_execution"] = {"worker_count": int(worker_count),
                                          "parallel_execution": bool(worker_count > 1),
                                          "n_bootstrap": int(n_bootstrap)}

    sample_tasks = build_hierarchical_bootstrap_tasks(args=args, params_yaml=params_yaml, n_bootstrap=n_bootstrap, need_regrowth=need_regrowth, need_dispersion=need_dispersion, need_mutation_rate=need_mutation_rate, continuous_wells=continuous_wells, clone_count_map=clone_count_map, no_treatment_labels=no_treatment_labels, no_treatment_area_observations=full_area_observations, derived_specs=derived_specs, regrowth_point_fit=regrowth_point_fit, point_dispersion_estimates=point_dispersion_estimates, point_mutation_rate=point_mutation_rate if need_mutation_rate else None, rng=rng)
    joint_rows = run_hierarchical_bootstrap_tasks(sample_tasks, worker_count, logger)
    joint_df = pd.DataFrame(joint_rows)

    if need_regrowth:
        regrowth_success = int(joint_df["regrowth_fit_converged"].sum())
        raw_samples["regrowth_t0_h"] = joint_df["regrowth_t0_h"].to_numpy(dtype=float)
        raw_samples["regrowth_slope_um_per_h"] = joint_df["regrowth_slope_um_per_h"].to_numpy(dtype=float)
        summary_rows.append(summarize_joint_samples("regrowth_t0_h", point_estimate=float(regrowth_point_fit["t0_h"]), sample_values=raw_samples["regrowth_t0_h"], ci_level=args.ci_level, n_attempted=n_bootstrap, n_converged=regrowth_success, method="bootstrap_refit", notes="Continuous-therapy wells were resampled with replacement and refit with the ReLU regrowth calibration.", point_fit_converged=True))
        summary_rows.append(summarize_joint_samples("regrowth_slope_um_per_h", point_estimate=float(regrowth_point_fit["slope_um_per_h"]), sample_values=raw_samples["regrowth_slope_um_per_h"], ci_level=args.ci_level, n_attempted=n_bootstrap, n_converged=regrowth_success, method="bootstrap_refit", notes="Continuous-therapy wells were resampled with replacement and refit with the ReLU regrowth calibration.", point_fit_converged=True))

    if need_dispersion:
        dispersion_success = int(joint_df["dispersion_fit_converged"].sum())
        raw_samples["diffusion_sensitive"] = joint_df["diffusion_sensitive"].to_numpy(dtype=float)
        raw_samples["diffusion_resistant"] = joint_df["diffusion_resistant"].to_numpy(dtype=float)
        raw_samples["uptake_rate"] = joint_df["uptake_rate"].to_numpy(dtype=float)
        raw_samples["diffusion_nutrients"] = joint_df["diffusion_nutrients"].to_numpy(dtype=float)
        raw_samples["start_point_from_dispersion_fit"] = joint_df["start_point"].to_numpy(dtype=float)

        summary_rows.append(summarize_joint_samples("diffusion_sensitive", point_estimate=float(point_dispersion_estimates[0]), sample_values=raw_samples["diffusion_sensitive"], ci_level=args.ci_level, n_attempted=n_bootstrap, n_converged=dispersion_success, method="bootstrap_refit", notes=("No-treatment colony-area trajectories were resampled with replacement. Each replicate first refit the continuous-therapy regrowth calibration."), point_fit_converged=bool(point_dispersion_fit.result.success)))
        summary_rows.append(summarize_joint_samples("diffusion_resistant", point_estimate=float(point_dispersion_estimates[0]), sample_values=raw_samples["diffusion_resistant"], ci_level=args.ci_level, n_attempted=n_bootstrap, n_converged=dispersion_success, method="bootstrap_refit", notes=("Kept identical to diffusion_sensitive because the current fit ties both diffusion parameters together."), point_fit_converged=bool(point_dispersion_fit.result.success)))
        summary_rows.append(summarize_joint_samples("uptake_rate", point_estimate=float(point_dispersion_estimates[1]), sample_values=raw_samples["uptake_rate"], ci_level=args.ci_level, n_attempted=n_bootstrap, n_converged=dispersion_success, method="bootstrap_refit", notes=("No-treatment colony-area trajectories were resampled with replacement. Each replicate first refit the continuous-therapy regrowth calibration."), point_fit_converged=bool(point_dispersion_fit.result.success)))
        summary_rows.append(summarize_joint_samples("diffusion_nutrients", point_estimate=float(point_dispersion_estimates[2]), sample_values=raw_samples["diffusion_nutrients"], ci_level=args.ci_level, n_attempted=n_bootstrap, n_converged=dispersion_success, method="bootstrap_refit",notes=("No-treatment colony-area trajectories were resampled with replacement. Each replicate first refit the continuous-therapy regrowth calibration."), point_fit_converged=bool(point_dispersion_fit.result.success)))
        summary_rows.append(summarize_joint_samples("start_point_from_dispersion_fit", point_estimate=float(point_start_point), sample_values=raw_samples["start_point_from_dispersion_fit"], ci_level=args.ci_level, n_attempted=n_bootstrap, n_converged=dispersion_success, method="bootstrap_refit", notes=("Computed in-memory during each hierarchical replicate. The bootstrap never mutates params.yaml."), point_fit_converged=bool(point_dispersion_fit.result.success)))

    if need_mutation_rate:
        mutation_success = int(joint_df["mutation_rate_fit_converged"].sum())
        raw_samples["mutation_rate"] = joint_df["mutation_rate"].to_numpy(dtype=float)
        mutation_notes = "Continuous-therapy wells were resampled with replacement."
        if need_dispersion:
            mutation_notes += (" Each replicate used that replicate's dispersion-derived start_point.")
        if "treatment_delay" in derived_specs:
            mutation_notes += " Sampled upstream simulation parameters were propagated into each replicate's simulation parameters."
        summary_rows.append(summarize_joint_samples("mutation_rate", point_estimate=point_mutation_rate, sample_values=raw_samples["mutation_rate"], ci_level=args.ci_level, n_attempted=n_bootstrap, n_converged=mutation_success, method="bootstrap_refit", notes=mutation_notes, point_fit_converged=bool(point_mutation_fit.result.success)))

    if "derived" in args.targets:
        for parameter_name, spec in derived_specs.items():
            raw_samples[parameter_name] = np.asarray(spec["samples"], dtype=float)
            if spec["summary_kind"] == "point_only":
                summary_rows.append(
                    summarize_point_only(parameter_name, spec["point_estimate"], method=spec["method"], notes=spec["notes"]))
            else:
                summary_rows.append(summarize_joint_samples(parameter_name, point_estimate=spec["point_estimate"], sample_values=raw_samples[parameter_name], ci_level=args.ci_level, n_attempted=n_bootstrap, n_converged=spec["n_converged"], method=spec["method"], notes=spec["notes"]))

        regrowth_t0_samples = joint_df["regrowth_t0_h"].to_numpy(dtype=float)
        regrowth_slope_samples = joint_df["regrowth_slope_um_per_h"].to_numpy(dtype=float)
        for offset in args.regrowth_offsets:
            parameter_name = (f"res_regrowth_frame_from_offset_{format_number_for_name(offset)}")
            offset_draws = regrowth_offset_draws[parameter_name]
            sample_values = np.full(int(n_bootstrap), np.nan, dtype=float)
            valid_mask = (np.isfinite(regrowth_t0_samples) & np.isfinite(regrowth_slope_samples) & (regrowth_slope_samples > 0) & np.isfinite(offset_draws))
            sample_values[valid_mask] = regrowth_frames_from_calibration(offset_draws[valid_mask], regrowth_t0_samples[valid_mask], regrowth_slope_samples[valid_mask], scale_factor=float(regrowth_point_fit["scale_factor"]), frames_per_hour=float(regrowth_point_fit["frames_per_hour"]))
            raw_samples[parameter_name] = sample_values
            point_value = float(regrowth_frames_from_calibration(offset, float(regrowth_point_fit["t0_h"]), float(regrowth_point_fit["slope_um_per_h"]), scale_factor=float(regrowth_point_fit["scale_factor"]), frames_per_hour=float(regrowth_point_fit["frames_per_hour"])))
            notes = "Computed from the bootstrap regrowth calibration instead of the old hard-coded constants."
            summary_rows.append(summarize_joint_samples(parameter_name, point_estimate=point_value, sample_values=sample_values, ci_level=args.ci_level, n_attempted=n_bootstrap, n_converged=int(np.count_nonzero(valid_mask)), method="bootstrap_refit", notes=notes, point_fit_converged=True))

    return summary_rows, raw_samples, diagnostics, joint_df


def save_outputs(run_dir, args, summary_rows, raw_samples, diagnostics, joint_samples=None):
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(run_dir / "bootstrap_summary.csv", index=False)
    np.savez(run_dir / "raw_bootstrap_samples.npz", **raw_samples)
    if joint_samples is not None and not joint_samples.empty:
        joint_samples.to_csv(run_dir / "joint_parameter_samples.csv", index=False)

    with (run_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready(vars(args)), handle, indent=2)

    with (run_dir / "fit_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready(diagnostics), handle, indent=2)


def main():
    project_root = vc.find_project_root(os.getcwd(), "setup.py")
    config = {"targets": ["mutation_rate", "dispersion", "derived"],
              "n_bootstrap": 1000,
              "seed": 0,
              "maxiter": 800,
              "mutation_rate_initial": None,
              "dispersion_initial": None,
              "mutation_rate_sim_replicas": 15,
              "ci_level": 95.0,
              "output_dir": project_root / "source" / "fit" / "fit_results" / "bootstrap_uncertainties",
              "num_workers": 24,
              "sensitive_growth_time": 84.92 / 3.0,
              "resistant_growth_time": 80.49 / 3.0,
              "sensitive_growth_time_sem": 1.75 / 3.0,
              "resistant_growth_time_sem": 1.21 / 3.0,
              "independent_growth_rates": False,
              "regrowth_offsets": [2.0, 4.0, 6.0, 8.0],
              "single_cell_area_csv": project_root / "data" / "Single_cell_resolution_yNA16_cell_area" / "Single_cell_resolution_yNA16_cell_measurements.csv",
              "single_cell_exp_pixel_size_um": 8.648,
              "single_cell_area_include_rejected": False,
              "regrowth_min_duration_frames": 10,
              "regrowth_min_distance_um": 100.0}
    
    args = build_run_args(config)
    params_yaml = vc.load_params_yaml('params.yaml')
    worker_count = args.num_workers
    run_dir = create_run_directory(args.output_dir)
    logger = setup_logger(run_dir / "bootstrap.log")
    rng = np.random.default_rng(args.seed)

    logger.info("Bootstrap run directory: %s", run_dir)
    logger.info("Targets: %s", ", ".join(args.targets))
    logger.info("Using %d worker process(es) for bootstrap replicates.", worker_count)

    summary_rows, raw_samples, diagnostics, joint_df = run_hierarchical_bootstrap(args, params_yaml, rng, logger, worker_count)
    save_outputs(run_dir=run_dir, args=args, summary_rows=summary_rows, raw_samples=raw_samples, diagnostics=diagnostics, joint_samples=joint_df)

    logger.info("Saved bootstrap summary to %s", run_dir / "bootstrap_summary.csv")
    logger.info("Saved raw samples to %s", run_dir / "raw_bootstrap_samples.npz")
    if not joint_df.empty:
        logger.info("Saved joint parameter samples to %s", run_dir / "joint_parameter_samples.csv")



if __name__ == "__main__":
    main()
