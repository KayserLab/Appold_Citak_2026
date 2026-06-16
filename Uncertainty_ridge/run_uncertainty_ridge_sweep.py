import concurrent.futures as cf
import copy
import os
import pathlib as pl
import time
import warnings
import numpy as np
import pandas as pd
import tqdm
import uncertainty_ridge_common as urc


def default_config():
    return {"params_path": "params.yaml",
            "output_root": "Uncertainty_ridge/results",
            "bootstrap_joint": None,
            "num_parameter_samples": 200,
            "num_replicates": 1,
            "steps_per_hour": 20.0,
            "ttp_threshold_mm2": 71.0,
            "ridge_tolerance_hours": 0.5,
            "num_workers": None,
            "job_id": int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)),
            "num_jobs": int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)),
            "run_dir": None,
            "show_progress": True,
            "wait_poll_seconds": 1.0,
            "random_seed": 0,
            "total_time": None,
            "treatment_delay_shift": 0.0,
            "release_delay_shift": 0.0,
            "lag_steps_shift": 0.0,
            "overshoot_steps_shift": 0.0}


def apply_delay_shifts(sample_df, run_config):
    shifted_df = sample_df.copy()
    shift_map = {"treatment_delay": float(run_config["treatment_delay_shift"]),
                 "release_delay": float(run_config["release_delay_shift"]),
                 "lag_steps": float(run_config["lag_steps_shift"]),
                 "overshoot_steps": float(run_config["overshoot_steps_shift"])}

    for key, shift in shift_map.items():
        if key in shifted_df.columns and shift != 0.0:
            shifted_df[key] = shifted_df[key].astype(float) + shift

    return shifted_df



def auto_discover_sample_source(project_root):
    bootstrap_root = pl.Path(project_root) / "source" / "fit" / "fit_results" / "bootstrap_uncertainties"
    if bootstrap_root.exists():
        joint_candidates = sorted(bootstrap_root.glob("run_*/joint_parameter_samples.csv"))
        if joint_candidates:
            return "bootstrap_joint", str(joint_candidates[-1])

    return None, None


def build_parameter_samples(project_root, base_params, bootstrap_joint, num_parameter_samples, random_seed):
    selected_source = None
    selected_path = None
    if bootstrap_joint is not None:
        selected_source = "bootstrap_joint"
        selected_path = urc.resolve_path(project_root, bootstrap_joint)
    else:
        selected_source, selected_path = auto_discover_sample_source(project_root)

    if selected_source is None or selected_path is None:
        raise ValueError("No parameter-sample source was provided or auto-discovered. Set bootstrap_joint in the config.")

    raw_df = urc.load_parameter_samples_from_bootstrap_joint(selected_path, base_params=base_params, num_samples=num_parameter_samples, random_seed=random_seed)

    final_df = urc.finalize_parameter_samples(raw_df, base_params=base_params)
    final_df["source_path"] = str(selected_path)
    return final_df


def _quiet_tqdm(iterable, *args, **kwargs):
    return iterable


def _progress_bar(*, total, run_config):
    if not bool(run_config.get("show_progress", True)):
        return None

    return tqdm.tqdm(total=int(total), desc="Uncertainty ridge runs", unit="run")


def _resolve_run_dir(project_root, run_config):
    explicit_run_dir = run_config.get("run_dir")
    if explicit_run_dir is not None:
        return pl.Path(urc.resolve_path(project_root, str(explicit_run_dir)))

    output_root = urc.resolve_path(project_root, str(run_config["output_root"]))
    num_jobs = max(1, int(run_config.get("num_jobs", 1)))
    if num_jobs == 1:
        return urc.create_run_directory(output_root, "uncertainty_ridge")

    shared_job_id = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID")
    if shared_job_id is None:
        raise ValueError("When num_jobs > 1, set run_dir in the config or submit through Slurm so a shared run directory can be inferred.")
    return pl.Path(output_root) / f"uncertainty_ridge_array_{shared_job_id}"


def _shared_result_paths(run_dir):
    run_dir = pl.Path(run_dir)
    return {"parameter_samples": run_dir / "parameter_samples.csv",
            "run_manifest": run_dir / "run_manifest.json",
            "ttp_hours": run_dir / "ttp_hours.dat",
            "ratio_ttp": run_dir / "ratio_ttp.dat",
            "ratio_endpoint": run_dir / "ratio_endpoint.dat",
            "status": run_dir / "status.dat"}


def _init_shared_results(run_dir, sample_df, n_samples, n_tau_on, n_tau_off, n_replicates, params_path, run_config):
    run_dir = pl.Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    sample_df.to_csv(run_dir / "parameter_samples.csv", index=False)
    total_simulations = int(n_samples * n_tau_on * n_tau_off * n_replicates)
    urc.save_json(run_dir / "run_manifest.json", {"params_path": str(params_path),
                                                  "num_parameter_samples": int(n_samples),
                                                  "num_tau_on": int(n_tau_on),
                                                  "num_tau_off": int(n_tau_off),
                                                  "num_replicates": int(n_replicates),
                                                  "total_simulations": int(total_simulations),
                                                  "job_id": int(run_config.get("job_id", 0)),
                                                  "num_jobs": int(run_config.get("num_jobs", 1))})

    grid_shape = (int(n_samples), int(n_tau_on), int(n_tau_off), int(n_replicates))
    for key in ("ttp_hours", "ratio_ttp", "ratio_endpoint"):
        mmap = np.memmap(run_dir / f"{key}.dat", dtype=np.float32, mode="w+", shape=grid_shape)
        mmap[:] = np.nan
        mmap.flush()

    status = np.memmap(run_dir / "status.dat", dtype=np.bool_, mode="w+", shape=(total_simulations,))
    status[:] = False
    status.flush()


def _wait_for_shared_results(run_dir, *, poll_seconds):
    paths = _shared_result_paths(run_dir)
    while not all(path.exists() for path in paths.values()):
        time.sleep(float(poll_seconds))
    return paths


def _count_completed_status(status_path, total_simulations):
    status = np.memmap(status_path, dtype=np.bool_, mode="r", shape=(int(total_simulations)))
    return int(np.count_nonzero(status))


def _simulate_schedule_task(task):
    project_root = task["project_root"]
    if os.getcwd() != project_root:
        os.chdir(project_root)

    from source import core as cr

    cr.tqdm.tqdm = _quiet_tqdm

    params = copy.deepcopy(task["params"])
    sim = cr.DiffusionModel2D()
    sim.params = params
    start_point = 0 if bool(params.get("gaussian", False)) else int(params["start_point"])
    sim.treatment_times = urc.build_treatment_schedule(total_time=int(params["total_time"]), treatment_start=int(params["treatment_start"]), start_point=start_point, tau_on=int(task["tau_on"]), tau_off=int(task["tau_off"]))
    sim.random_seed = int(task["seed"])
    sim.rng = None
    sim.treatment_efficacy = 0.0
    sim.treatment_temp = 0
    sim.save_treat_efficacy = [0.0]
    sim.save_size = [urc.initial_area_mm2(params)]
    sim.save_ratio = [0.0]
    sim.mutation_count = 0
    sim.prev_treatment = False
    sim.extra_steps_remaining = 0
    sim.lag_steps_remaining = 0
    sim.params["save_in_core"] = False
    sim.params["return_all"] = False
    sim.params["treatment_on_duration"] = int(task["tau_on"])

    _, _, _, _, _, size_trace, ratio_trace = sim.run_simulation(save_without_asking=True, stop_with_size=True)
    ttp_hours, ratio_at_ttp, ratio_endpoint = urc.compute_progression_metrics(size_trace=size_trace, ratio_trace=ratio_trace, start_point=int(params["start_point"]), steps_per_hour=float(task["steps_per_hour"]), ttp_threshold_mm2=float(task["ttp_threshold_mm2"]))

    return (int(task["task_index"]), int(task["sample_index"]), int(task["tau_on_index"]), int(task["tau_off_index"]), int(task["replicate_index"]), float(ttp_hours), float(ratio_at_ttp), float(ratio_endpoint))


def compute_ridge_probability(ttp_grid, tau_on_steps, tolerance_hours):
    n_samples, n_tau_on, n_tau_off = ttp_grid.shape
    ridge_probability = np.zeros((n_tau_on, n_tau_off), dtype=np.float32)
    ridge_tau_on_hours = np.full((n_samples, n_tau_off), np.nan, dtype=np.float32)

    for sample_idx in range(n_samples):
        for off_idx in range(n_tau_off):
            column = np.asarray(ttp_grid[sample_idx, :, off_idx], dtype=float)
            finite_mask = np.isfinite(column)
            if not np.any(finite_mask):
                continue

            best_value = float(np.nanmax(column))
            candidate_mask = finite_mask & (column >= (best_value - float(tolerance_hours)))
            candidate_indices = np.flatnonzero(candidate_mask)
            if candidate_indices.size == 0:
                continue

            ridge_probability[candidate_indices, off_idx] += (1.0 / float(candidate_indices.size) / float(n_samples))
            ridge_tau_on_hours[sample_idx, off_idx] = float(tau_on_steps[int(np.min(candidate_indices))])

    return ridge_probability, ridge_tau_on_hours


def compute_global_optimum_probability(ttp_grid, tolerance_hours):
    n_samples, n_tau_on, n_tau_off = ttp_grid.shape
    probability = np.zeros((n_tau_on, n_tau_off), dtype=np.float32)

    for sample_idx in range(n_samples):
        surface = np.asarray(ttp_grid[sample_idx], dtype=float)
        if not np.isfinite(surface).any():
            continue

        best_value = float(np.nanmax(surface))
        candidate_mask = np.isfinite(surface) & (surface >= (best_value - float(tolerance_hours)))
        candidate_indices = np.argwhere(candidate_mask)
        if candidate_indices.size == 0:
            continue

        weight = 1.0 / float(len(candidate_indices)) / float(n_samples)
        for row_idx, col_idx in candidate_indices:
            probability[int(row_idx), int(col_idx)] += float(weight)

    return probability


def summarize_ridge_line(ridge_tau_on_steps, tau_off_steps, steps_per_hour):
    rows = []
    tau_off_hours = urc.steps_to_hours(tau_off_steps, steps_per_hour)
    ridge_tau_on_hours = ridge_tau_on_steps / float(steps_per_hour)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for off_idx, tau_off_h in enumerate(tau_off_hours):
            column = ridge_tau_on_hours[:, off_idx]
            finite = column[np.isfinite(column)]
            if finite.size == 0:
                rows.append({"tau_off_steps": int(tau_off_steps[off_idx]),
                             "tau_off_hours": float(tau_off_h),
                             "ridge_tau_on_mean_h": np.nan,
                             "ridge_tau_on_median_h": np.nan,
                             "ridge_tau_on_q25_h": np.nan,
                             "ridge_tau_on_q75_h": np.nan,
                             "ridge_tau_on_q10_h": np.nan,
                             "ridge_tau_on_q90_h": np.nan,
                             "sample_count": 0})
                continue

            rows.append({"tau_off_steps": int(tau_off_steps[off_idx]),
                         "tau_off_hours": float(tau_off_h),
                         "ridge_tau_on_mean_h": float(np.mean(finite)),
                         "ridge_tau_on_median_h": float(np.median(finite)),
                         "ridge_tau_on_q25_h": float(np.percentile(finite, 25)),
                         "ridge_tau_on_q75_h": float(np.percentile(finite, 75)),
                         "ridge_tau_on_q10_h": float(np.percentile(finite, 10)),
                         "ridge_tau_on_q90_h": float(np.percentile(finite, 90)),
                         "sample_count": int(finite.size)})
    return pd.DataFrame(rows)


def task_iterator(sample_payloads, tau_on_steps, tau_off_steps, n_replicates, project_root, steps_per_hour, run_config):
    task_seed = int(run_config["random_seed"])
    task_index = 0
    for payload in sample_payloads:
        for tau_on_index, tau_on in enumerate(tau_on_steps):
            for tau_off_index, tau_off in enumerate(tau_off_steps):
                for replicate_index in range(n_replicates):
                    yield {"task_index": int(task_index),
                           "project_root": project_root,
                           "params": payload["params"],
                           "sample_index": int(payload["sample_index"]),
                           "tau_on_index": int(tau_on_index),
                           "tau_off_index": int(tau_off_index),
                           "replicate_index": int(replicate_index),
                           "tau_on": int(tau_on),
                           "tau_off": int(tau_off),
                           "seed": int(task_seed),
                           "steps_per_hour": steps_per_hour,
                           "ttp_threshold_mm2": float(run_config["ttp_threshold_mm2"])}
                    task_seed += 1
                    task_index += 1


def save_task_result(result, ttp_hours, ratio_ttp, ratio_endpoint, status):
    task_index, sample_idx, on_idx, off_idx, rep_idx, ttp_val, ratio_ttp_val, ratio_endpoint_val = result
    ttp_hours[sample_idx, on_idx, off_idx, rep_idx] = np.float32(ttp_val)
    ratio_ttp[sample_idx, on_idx, off_idx, rep_idx] = np.float32(ratio_ttp_val)
    ratio_endpoint[sample_idx, on_idx, off_idx, rep_idx] = np.float32(ratio_endpoint_val)
    ttp_hours.flush()
    ratio_ttp.flush()
    ratio_endpoint.flush()
    status[int(task_index)] = True
    status.flush()


def main():
    run_config = default_config()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = urc.find_project_root(script_dir, "requirements.txt") or os.getcwd()
    params_path = urc.resolve_path(project_root, str(run_config["params_path"]))
    base_params = urc.load_params(params_path)
    if run_config["total_time"] is not None:
        base_params["total_time"] = int(run_config["total_time"])

    steps_per_hour = float(run_config["steps_per_hour"])

    tau_on_steps = urc.build_duration_axis_from_params(base_params, axis_name="tau_on")
    tau_off_steps = urc.build_duration_axis_from_params(base_params, axis_name="tau_off")

    sample_df = build_parameter_samples(project_root, base_params, run_config["bootstrap_joint"], int(run_config["num_parameter_samples"]), int(run_config["random_seed"]))
    sample_df = apply_delay_shifts(sample_df, run_config)


    n_samples = len(sample_df)
    n_tau_on = len(tau_on_steps)
    n_tau_off = len(tau_off_steps)
    n_replicates = int(run_config["num_replicates"])
    job_id = int(run_config.get("job_id", 0))
    num_jobs = int(run_config.get("num_jobs", 1))
    if num_jobs < 1:
        raise ValueError("num_jobs must be at least 1.")
    if job_id < 0 or job_id >= num_jobs:
        raise ValueError("job_id must satisfy 0 <= job_id < num_jobs.")

    wait_poll_seconds = max(0.1, float(run_config.get("wait_poll_seconds", 1.0)))
    run_dir = _resolve_run_dir(project_root, run_config)

    sample_payloads = []
    for sample_index, sample_row in sample_df.iterrows():
        params = copy.deepcopy(base_params)
        for key in sample_df.columns:
            if key in params:
                params[key] = sample_row[key]
        params["start_point"] = int(sample_row["start_point"])
        params["diffusion_sensitive"] = float(sample_row["diffusion_sensitive"])
        params["diffusion_resistant"] = float(sample_row["diffusion_resistant"])
        params["uptake_rate"] = float(sample_row["uptake_rate"])
        params["diffusion_nutrients"] = float(sample_row["diffusion_nutrients"])
        params["mutation_rate"] = float(sample_row["mutation_rate"])
        params["mutation_scaling"] = float(sample_row["mutation_scaling"])
        params["sensitive_growth_rate"] = float(sample_row["sensitive_growth_rate"])
        params["resistant_growth_rate"] = float(sample_row["resistant_growth_rate"])
        sample_payloads.append({"sample_index": int(sample_index),
                                "params": params})

    if num_jobs == 1 or job_id == 0:
        _init_shared_results(run_dir, sample_df, n_samples, n_tau_on, n_tau_off, n_replicates, params_path, run_config)

    shared_paths = _wait_for_shared_results(run_dir, poll_seconds=wait_poll_seconds)

    worker_count = (max(1, int(os.cpu_count() or 1) - 1) if run_config["num_workers"] is None else max(1, int(run_config["num_workers"])))
    total_simulations = int(n_samples * n_tau_on * n_tau_off * n_replicates)
    grid_shape = (n_samples, n_tau_on, n_tau_off, n_replicates)
    ttp_hours = np.memmap(shared_paths["ttp_hours"], dtype=np.float32, mode="r+", shape=grid_shape)
    ratio_ttp = np.memmap(shared_paths["ratio_ttp"], dtype=np.float32, mode="r+", shape=grid_shape)
    ratio_endpoint = np.memmap(shared_paths["ratio_endpoint"], dtype=np.float32, mode="r+", shape=grid_shape)
    status = np.memmap(shared_paths["status"], dtype=np.bool_, mode="r+", shape=(total_simulations,))

    jobs = []
    for task in task_iterator(sample_payloads, tau_on_steps, tau_off_steps, n_replicates, project_root, steps_per_hour, run_config):
        task_index = int(task["task_index"])
        if bool(status[task_index]):
            continue
        if task_index % num_jobs != job_id:
            continue
        jobs.append(task)

    if worker_count == 1:
        progress_bar = _progress_bar(total=len(jobs), run_config=run_config)
        try:
            for task in jobs:
                result = _simulate_schedule_task(task)
                save_task_result(result, ttp_hours, ratio_ttp, ratio_endpoint, status)
                if progress_bar is not None:
                    progress_bar.update(1)
        finally:
            if progress_bar is not None:
                progress_bar.close()
    else:
        with cf.ProcessPoolExecutor(max_workers=worker_count) as executor:
            result_iterator = executor.map(_simulate_schedule_task, jobs, chunksize=1)
            progress_bar = _progress_bar(total=len(jobs), run_config=run_config)
            try:
                for result in result_iterator:
                    save_task_result(result, ttp_hours, ratio_ttp, ratio_endpoint, status)
                    if progress_bar is not None:
                        progress_bar.update(1)
            finally:
                if progress_bar is not None:
                    progress_bar.close()

    if num_jobs > 1 and job_id != 0:
        return run_dir

    if num_jobs > 1:
        while _count_completed_status(shared_paths["status"], total_simulations) < total_simulations:
            time.sleep(wait_poll_seconds)

    ttp_hours = np.memmap(shared_paths["ttp_hours"], dtype=np.float32, mode="r", shape=grid_shape)
    ratio_ttp = np.memmap(shared_paths["ratio_ttp"], dtype=np.float32, mode="r", shape=grid_shape)
    ratio_endpoint = np.memmap(shared_paths["ratio_endpoint"], dtype=np.float32, mode="r", shape=grid_shape)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ttp_sample_median = np.nanmedian(ttp_hours, axis=3)
        ratio_ttp_sample_median = np.nanmedian(ratio_ttp, axis=3)
        ratio_endpoint_sample_median = np.nanmedian(ratio_endpoint, axis=3)
        ttp_mean_surface = np.nanmean(ttp_sample_median, axis=0)
        ttp_sd_surface = np.nanstd(ttp_sample_median, axis=0, ddof=1 if n_samples > 1 else 0)
        ratio_ttp_mean_surface = np.nanmean(ratio_ttp_sample_median, axis=0)
        ratio_endpoint_mean_surface = np.nanmean(ratio_endpoint_sample_median, axis=0)

    ridge_probability, ridge_tau_on_steps = compute_ridge_probability(ttp_sample_median, np.asarray(tau_on_steps, dtype=np.int32), float(run_config["ridge_tolerance_hours"]))
    global_optimum_probability = compute_global_optimum_probability(ttp_sample_median, float(run_config["ridge_tolerance_hours"]))
    ridge_summary_df = summarize_ridge_line(ridge_tau_on_steps, tau_off_steps, steps_per_hour)
    ridge_summary_df.to_csv(run_dir / "ridge_line_summary.csv", index=False)

    np.savez_compressed(run_dir / "aggregated_surfaces.npz", tau_on_steps=np.asarray(tau_on_steps, dtype=np.int32), tau_off_steps=np.asarray(tau_off_steps, dtype=np.int32), tau_on_hours=urc.steps_to_hours(tau_on_steps, steps_per_hour), tau_off_hours=urc.steps_to_hours(tau_off_steps, steps_per_hour), ttp_sample_median=ttp_sample_median.astype(np.float32), ratio_ttp_sample_median=ratio_ttp_sample_median.astype(np.float32), ratio_endpoint_sample_median=ratio_endpoint_sample_median.astype(np.float32), ttp_mean_surface=ttp_mean_surface.astype(np.float32), ttp_sd_surface=ttp_sd_surface.astype(np.float32), ratio_ttp_mean_surface=ratio_ttp_mean_surface.astype(np.float32), ratio_endpoint_mean_surface=ratio_endpoint_mean_surface.astype(np.float32), ridge_probability=ridge_probability.astype(np.float32), global_optimum_probability=global_optimum_probability.astype(np.float32), ridge_tau_on_steps=ridge_tau_on_steps.astype(np.float32))

    ttp_surface_summary = urc.summarize_vector(ttp_mean_surface)
    ridge_probability_summary = urc.summarize_vector(ridge_probability)
    global_optimum_summary = urc.summarize_vector(global_optimum_probability)

    urc.save_json(run_dir / "summary.json",{"params_path": params_path,
                                            "sample_source": str(sample_df["sample_source"].iloc[0]),
                                            "source_path": str(sample_df["source_path"].iloc[0]),
                                            "num_parameter_samples": int(n_samples),
                                            "num_tau_on": int(n_tau_on),
                                            "num_tau_off": int(n_tau_off),
                                            "num_replicates": int(n_replicates),
                                            "num_workers": int(worker_count),
                                            "total_simulations": int(total_simulations),
                                            "steps_per_hour": steps_per_hour,
                                            "ttp_threshold_mm2": float(run_config["ttp_threshold_mm2"]),
                                            "ridge_tolerance_hours": float(run_config["ridge_tolerance_hours"]),
                                            "total_time": int(base_params["total_time"]),
                                            "tau_on_hours_range": [float(urc.steps_to_hours(tau_on_steps, steps_per_hour)[0]), float(urc.steps_to_hours(tau_on_steps, steps_per_hour)[-1])],
                                            "tau_off_hours_range": [float(urc.steps_to_hours(tau_off_steps, steps_per_hour)[0]), float(urc.steps_to_hours(tau_off_steps, steps_per_hour)[-1])],
                                            "surface_summaries": {"ttp_mean_surface": ttp_surface_summary,
                                                                "ridge_probability": ridge_probability_summary,
                                                                "global_optimum_probability": global_optimum_summary},
                                            "notes": ["The saved ttp_sample_median array stores the replicate-median TTP surface for each parameter sample.", "ridge_probability is column-wise: for each tau_off and parameter sample, mass is assigned to the schedule(s) within ridge_tolerance_hours of the best TTP.", "global_optimum_probability counts which cells are within ridge_tolerance_hours of the best TTP anywhere in the full tau_on / tau_off surface for each parameter sample.", "This workflow is intentionally summary-only and does not save full size.dat or ratio.dat traces for every parameter sample."]})
    return run_dir


if __name__ == "__main__":
    main()
