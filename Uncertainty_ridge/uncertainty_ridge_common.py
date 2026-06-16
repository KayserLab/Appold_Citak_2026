import json
import os
import datetime as dt
import pathlib as pl

import numpy as np
import pandas as pd
import yaml


def forward_parameter_keys():
    return ("diffusion_sensitive", "diffusion_resistant", "uptake_rate", "diffusion_nutrients", "start_point", "mutation_rate", "mutation_scaling", "sensitive_growth_rate", "resistant_growth_rate")


def delay_parameter_keys():
    return ("treatment_delay", "release_delay", "lag_steps", "overshoot_steps")


def end_to_end_column_map():
    return {
        "diffusion_sensitive": "fitted_diffusion_sensitive",
        "uptake_rate": "fitted_uptake_rate",
        "diffusion_nutrients": "fitted_diffusion_nutrients",
        "start_point": "fitted_start_point",
        "mutation_rate": "fitted_mutation_rate",
    }


def bootstrap_summary_parameter_map():
    return {
        "diffusion_sensitive": "diffusion_sensitive",
        "uptake_rate": "uptake_rate",
        "diffusion_nutrients": "diffusion_nutrients",
        "start_point": "start_point_from_dispersion_fit",
        "mutation_rate": "mutation_rate",
        "mutation_scaling": "mutation_scaling",
        "sensitive_growth_rate": "sensitive_growth_rate",
        "resistant_growth_rate": "resistant_growth_rate",
        "treatment_delay": "treatment_delay",
        "release_delay": "release_delay",
        "lag_steps": "lag_steps",
        "overshoot_steps": "overshoot_steps",
    }


def find_project_root(current_dir, marker_file):
    current_dir = os.path.abspath(str(current_dir))
    while current_dir != os.path.dirname(current_dir):
        if marker_file in os.listdir(current_dir):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return None


def resolve_path(project_root, path):
    if os.path.isabs(path):
        return str(path)
    return os.path.join(str(project_root), str(path))


def create_run_directory(base_dir, prefix):
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = pl.Path(base_dir) / f"{prefix}_{timestamp}"
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2)


def json_ready(value):
    if isinstance(value, pl.Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def load_params(params_path):
    with open(params_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build_step_axis_from_steps(axis_min, axis_max, axis_step, axis_name):
    if axis_step <= 0:
        raise ValueError(f"{axis_name} step must be positive.")
    if axis_max < axis_min:
        raise ValueError(f"{axis_name} max must be >= min.")

    num_steps = (int(axis_max) - int(axis_min)) / int(axis_step)
    if not np.isclose(num_steps, round(num_steps)):
        raise ValueError(f"{axis_name} step does not divide the requested range cleanly.")

    return np.arange(int(axis_min), int(axis_max) + int(axis_step), int(axis_step), dtype=np.int32)


def build_duration_axis_from_params(params, axis_name):
    if axis_name == "tau_on":
        return _build_step_axis_from_steps(int(params["treatment_on_min"]), int(params["treatment_on_max"]), int(params["treatment_on_step"]), axis_name)
    if axis_name == "tau_off":
        return _build_step_axis_from_steps(int(params["treatment_off_min"]), int(params["treatment_off_max"]), int(params["treatment_off_step"]), axis_name)
    raise ValueError(f"Unsupported axis_name: {axis_name}")


def initial_area_mm2(params):
    return ((eval(params["sim_pixel_to_exp_pixel_factor"]) ** 2) * (8.648**2) / 1e6)


def build_treatment_schedule(total_time, treatment_start, start_point, tau_on, tau_off):
    treatment_times = np.zeros(int(total_time), dtype=bool)
    first_start = int(treatment_start) + int(start_point)
    treatment_length = int(tau_on)

    if tau_off == 0:
        treatment_starts = [first_start]
        treatment_length = int(total_time) - first_start
        if tau_on == 0:
            treatment_starts = []
    elif tau_on == 0:
        treatment_starts = []
    else:
        treatment_starts = [current_start for current_start in range(first_start, int(total_time), int(tau_on) + int(tau_off))]

    for current_start in treatment_starts:
        treatment_times[current_start : current_start + treatment_length] = True

    return treatment_times


def compute_progression_metrics(size_trace, ratio_trace, start_point, steps_per_hour, ttp_threshold_mm2):
    progression_indices = np.flatnonzero(np.asarray(size_trace, dtype=float) >= float(ttp_threshold_mm2))
    if progression_indices.size > 0:
        ttp_index = int(progression_indices[0])
    else:
        ttp_index = int(len(size_trace) - 1)

    ttp_hours = (float(ttp_index) - float(start_point)) / float(steps_per_hour)
    ratio_at_ttp = float(ratio_trace[ttp_index])
    ratio_endpoint = float(ratio_trace[-1])
    return float(ttp_hours), ratio_at_ttp, ratio_endpoint


def summarize_vector(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    summary = {"count": int(finite.size),
               "mean": np.nan,
               "std": np.nan,
               "median": np.nan,
               "min": np.nan,
               "max": np.nan}
    if finite.size == 0:
        return summary

    summary["mean"] = float(np.mean(finite))
    summary["std"] = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    summary["median"] = float(np.median(finite))
    summary["min"] = float(np.min(finite))
    summary["max"] = float(np.max(finite))
    return summary


def _clean_numeric_array(raw):
    values = np.asarray(raw, dtype=float)
    values = values[np.isfinite(values)]
    return values


def _choose_indices(total_count, requested_count, rng):
    if total_count <= 0:
        return np.asarray([], dtype=int)
    if requested_count is None or requested_count >= total_count:
        return np.arange(total_count, dtype=int)
    return np.sort(rng.choice(total_count, size=int(requested_count), replace=False))


def load_parameter_samples_from_bootstrap_joint(joint_samples_path, base_params, num_samples, random_seed):
    joint_df = pd.read_csv(joint_samples_path)
    if joint_df.empty:
        raise ValueError("The supplied joint_parameter_samples.csv is empty.")

    rng = np.random.default_rng(random_seed)
    selected_indices = _choose_indices(len(joint_df), num_samples, rng)

    rows = []
    for sample_idx, row_index in enumerate(selected_indices):
        row = joint_df.iloc[int(row_index)]
        sample_row = {"sample_id": int(sample_idx),
                      "sample_label": str(row.get("sample_label", f"bootstrap_joint_{int(row_index):03d}")),
                      "sample_source": str(row.get("sample_source", "bootstrap_joint_empirical"))}
        for parameter_name in forward_parameter_keys():
            sample_row[parameter_name] = row.get(parameter_name, base_params[parameter_name])
        for parameter_name in delay_parameter_keys():
            if parameter_name in row:
                sample_row[parameter_name] = row.get(parameter_name)
        rows.append(sample_row)

    return pd.DataFrame(rows)


def finalize_parameter_samples(sample_df, base_params):
    if sample_df.empty:
        raise ValueError("No parameter samples were created.")

    final_df = sample_df.copy()
    parameter_keys = forward_parameter_keys()
    for key in parameter_keys:
        if key not in final_df.columns:
            final_df[key] = float(base_params[key])

    final_df["diffusion_resistant"] = final_df["diffusion_resistant"].fillna(final_df["diffusion_sensitive"])
    final_df["start_point"] = final_df["start_point"].round().astype(int)

    float_columns = [key for key in parameter_keys if key != "start_point"]
    for key in float_columns:
        final_df[key] = final_df[key].astype(float)
        invalid_mask = ~np.isfinite(final_df[key]) | (final_df[key] <= 0)
        final_df.loc[invalid_mask, key] = float(base_params[key])

    final_df["start_point"] = final_df["start_point"].clip(lower=0).round().astype(int)

    delay_columns = [key for key in delay_parameter_keys() if key in final_df.columns]
    ordered_columns = ["sample_id", "sample_label", "sample_source", *parameter_keys, *delay_columns]
    return final_df[ordered_columns]


def steps_to_hours(values, steps_per_hour):
    return np.asarray(values, dtype=float) / float(steps_per_hour)


def _read_raw_sample_value(raw_samples, key, raw_row_idx, fallback, *, positive_only):
    if key not in raw_samples:
        return fallback

    value = float(np.asarray(raw_samples[key], dtype=float).reshape(-1)[raw_row_idx])
    if not np.isfinite(value):
        return fallback
    if positive_only and value <= 0:
        return fallback
    return value
