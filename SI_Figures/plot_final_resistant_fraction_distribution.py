import copy
import csv
import concurrent.futures as cf
import os
import pathlib as pl
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm


def get_project_root():
    return pl.Path(__file__).resolve().parent.parent


def configure_import_paths():
    project_root = get_project_root()
    uncertainty_root = project_root / "Uncertainty_ridge"
    for path in (project_root, uncertainty_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return project_root


def load_uncertainty_common():
    configure_import_paths()
    import uncertainty_ridge_common as urc
    return urc


def configure_plot_style():
    plt.rcParams.update({
        "font.size": 7,
        "pdf.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
    })
    plt.rcParams["axes.labelsize"] = 7
    plt.rcParams["xtick.labelsize"] = 6
    plt.rcParams["ytick.labelsize"] = 6


def default_schedule_treatments():
    color_a = mpl.colors.to_hex(mpl.colormaps.get_cmap("tab20b").colors[4])
    color_b = mpl.colors.to_hex(mpl.colormaps.get_cmap("tab20b").colors[12])
    return [
        {"id": "met_5_50_18", "label": "5.65h / 18h", "treat_on": 113, "treat_off": 360, "pulse": False, "pulse_duration": None, "color": color_a},
        {"id": "met_5_70_18", "label": "5.70h / 18h", "treat_on": 114, "treat_off": 360, "pulse": False, "pulse_duration": None, "color": color_b},
        {"id": "met_5_80_18", "label": "5.75h / 18h", "treat_on": 115, "treat_off": 360, "pulse": False, "pulse_duration": None, "color": color_b},
    ]


def default_config():
    project_root = get_project_root()
    return {
        "params_path": "params.yaml",
        "output_stem": project_root / "SI_Figures" / "plots" / "resistant_fraction_distribution_rebuttal" / "final_resistant_fraction_distribution",
        "num_runs": 400,
        "max_workers": 20,
        "progress_interval": 10,
        "resume_from_outputs": False,
        "sample_parameter_uncertainty": False,
        "bootstrap_joint": None,
        "bootstrap_raw": None,
        "bootstrap_summary": None,
        "fold_results": None,
        "parameter_sample_seed": 0,
        "parameter_sample_keys": ["diffusion_sensitive", "diffusion_resistant", "sensitive_growth_rate", "resistant_growth_rate"],
        "run_schedule_treatments": False,
        "schedule_treatments": default_schedule_treatments(),
        "run_continuous_efficacies": True,
        "continuous_target_efficacies": [0.53, 0.535],
        "continuous_label_template": "Continuous cap {target_efficacy:.3f}",
        "continuous_color_map": "viridis",
        "continuous_color_limits": [0.2, 0.85],
        "analysis_measurement_modes": ["endpoint", "ttp"],
        "ttp_threshold_mm2": 71.0,
        "steps_per_hour": 20.0,
        "total_time": None,
        "bins": 50,
        "kde_grid_points": 256,
        "bimodality_peak_prominence_fraction": 0.05,
        "show": False,
        "save_endpoint_arrays": False,
        "endpoint_array_fields": ["sensitive", "resistant", "nutrients"],
    }


def merge_config(config):
    merged = copy.deepcopy(default_config())
    if config is None:
        return normalize_config(merged)
    for key, value in config.items():
        merged[key] = value
    return normalize_config(merged)


def normalize_config(config):
    normalized = copy.deepcopy(config)
    if isinstance(normalized["analysis_measurement_modes"], str):
        normalized["analysis_measurement_modes"] = [normalized["analysis_measurement_modes"]]
    normalized["endpoint_array_fields"] = list(normalized["endpoint_array_fields"])
    return normalized


def resolve_max_workers(max_workers):
    if max_workers is None:
        return os.cpu_count() or 1
    return int(max_workers)


def resolve_path(project_root, path):
    resolved = pl.Path(path)
    if resolved.is_absolute():
        return resolved
    return project_root / resolved


def normalize_output_stem(project_root, path):
    output_stem = resolve_path(project_root, path)
    if output_stem.suffix:
        output_stem = output_stem.with_suffix("")
    return output_stem


def format_efficacy_slug(target_efficacy):
    return f"{float(target_efficacy):.8f}".rstrip("0").rstrip(".").replace(".", "p")


def color_from_map(cmap_name, position):
    return mpl.colors.to_hex(mpl.colormaps.get_cmap(cmap_name)(float(position)))


def build_continuous_treatments(config):
    values = [float(value) for value in config["continuous_target_efficacies"]]
    if not values:
        return []
    low, high = [float(value) for value in config["continuous_color_limits"]]
    color_positions = np.linspace(low, high, len(values))
    treatments = []
    for value, color_position in zip(values, color_positions):
        if value < 0.0 or value > 1.0:
            raise ValueError("Continuous target efficacies must stay within [0, 1].")
        treatment_id = f"continuous_cap_{format_efficacy_slug(value)}"
        treatments.append({"id": treatment_id, "label": config["continuous_label_template"].format(target_efficacy=value), "mode": "continuous_constant_efficacy", "target_efficacy": value, "color": color_from_map(config["continuous_color_map"], color_position)})
    return treatments


def build_treatments(config):
    treatments = []
    if bool(config["run_schedule_treatments"]):
        for treatment in config["schedule_treatments"]:
            schedule = dict(treatment)
            schedule["mode"] = "schedule"
            schedule["pulse"] = bool(schedule.get("pulse", False))
            schedule["pulse_duration"] = schedule.get("pulse_duration")
            treatments.append(schedule)
    if bool(config["run_continuous_efficacies"]):
        treatments.extend(build_continuous_treatments(config))
    validate_treatments(treatments)
    return treatments


def validate_treatments(treatments):
    if not treatments:
        raise ValueError("No treatments are enabled. Enable schedules, continuous efficacies, or both.")
    seen = set()
    for treatment in treatments:
        treatment_id = str(treatment["id"])
        if treatment_id in seen:
            raise ValueError(f"Treatment id '{treatment_id}' appears more than once.")
        seen.add(treatment_id)
        mode = treatment.get("mode", "schedule")
        if mode not in {"schedule", "continuous_constant_efficacy"}:
            raise ValueError(f"Treatment '{treatment_id}' uses unsupported mode '{mode}'.")
        if mode == "schedule":
            for key in ("treat_on", "treat_off"):
                if key not in treatment:
                    raise ValueError(f"Schedule treatment '{treatment_id}' is missing '{key}'.")


def apply_parameter_overrides(params, parameter_overrides):
    if parameter_overrides is None:
        return
    for key, value in parameter_overrides.items():
        if key == "start_point":
            params[key] = int(round(float(value)))
        else:
            params[key] = float(value)


def build_schedule_treatment_times(sim, treatment):
    time = int(sim.params["total_time"])
    start_point = 0 if bool(sim.params["gaussian"]) else int(sim.params["start_point"])
    first_start = int(sim.params["treatment_start"]) + start_point
    treat_on = int(treatment["treat_on"])
    treat_off = int(treatment["treat_off"])
    pulse = bool(treatment.get("pulse", False))
    pulse_duration = treatment.get("pulse_duration")
    treatment_times = np.zeros(time, dtype=np.bool_)
    treatment_length = treat_on
    if treat_off == 0:
        treatment_starts = [first_start]
        treatment_length = time - first_start
        if treat_on == 0:
            treatment_starts = []
    elif treat_on == 0:
        treatment_starts = []
    else:
        treatment_starts = list(range(first_start, time, treat_off + treat_on))
    for treatment_start in treatment_starts:
        treatment_times[treatment_start : treatment_start + treatment_length] = True
    if pulse and pulse_duration is not None:
        treatment_times[first_start : first_start + int(pulse_duration)] = True
    return treatment_times


def configure_simulation(treatment, run_index, base_params, parameter_overrides):
    configure_import_paths()
    from source import core as cr
    params = copy.deepcopy(base_params)
    apply_parameter_overrides(params, parameter_overrides)
    if treatment["mode"] == "continuous_constant_efficacy":
        from Bifurcation.continuous_dose_efficacy_sweep import CappedContinuousDoseModel, build_continuous_treatment_schedule
        sim = CappedContinuousDoseModel(float(treatment["target_efficacy"]), params=params)
        sim.params["continuous_target_efficacy"] = float(sim.target_treatment_efficacy)
        sim.treatment_times, treatment_start = build_continuous_treatment_schedule(sim.params)
        sim.params["treatment_on_duration"] = max(0, int(sim.params["total_time"]) - int(treatment_start))
    else:
        sim = cr.DiffusionModel2D()
        sim.params = params
        sim.treatment_times = build_schedule_treatment_times(sim, treatment)
        sim.params["treatment_on_duration"] = int(treatment.get("pulse_duration") or treatment["treat_on"])
    sim.random_seed = int(run_index)
    sim.set_random_seed()
    sim.params["save_in_core"] = False
    sim.params["return_all"] = False
    return sim


def apply_fixed_mutation_if_needed(sim, timer, sensitive, resistant):
    if not sim.params["set_mut_pos"] or sim.params["mutations_active"]:
        return
    if timer != sim.params["mutation_pos_time"]:
        return
    scaling = 1 / sim.params["mutation_scaling"]
    mutation_position = sim.params["mutation_position"]
    sensitive[mutation_position[0], mutation_position[1]] -= scaling
    resistant[mutation_position[0], mutation_position[1]] += scaling
    sensitive[mutation_position[1], mutation_position[0]] -= scaling
    resistant[mutation_position[1], mutation_position[0]] += scaling
    sensitive[-mutation_position[0], -mutation_position[1]] -= scaling
    resistant[-mutation_position[0], -mutation_position[1]] += scaling
    sensitive[-mutation_position[1], -mutation_position[0]] -= scaling
    resistant[-mutation_position[1], -mutation_position[0]] += scaling


def area_per_pixel_mm2(params):
    return (float(params["sim_pixel_to_exp_pixel_factor"]) ** 2) * (8.648**2) / 1e6


def initial_scalar_summary(params):
    return {"sensitive_count_px": 1, "resistant_count_px": 0, "resistant_dominant_count_px": 0, "total_count_px": 1, "area_mm2": area_per_pixel_mm2(params), "resistant_fraction": 0.0}


def scalar_summary_from_state(params, sensitive, resistant):
    threshold = 1.0 / float(params["mutation_scaling"])
    sensitive_occupied = sensitive > threshold
    resistant_occupied = resistant > threshold
    total_occupied = sensitive_occupied | resistant_occupied
    total_count = int(np.count_nonzero(total_occupied))
    sensitive_ratio = np.where(sensitive_occupied, sensitive, 0)
    resistant_ratio = np.where(resistant_occupied, resistant, 0)
    resistant_dominant = resistant_ratio > sensitive_ratio
    resistant_dominant_count = int(np.count_nonzero(resistant_dominant))
    resistant_fraction = float(resistant_dominant_count / total_count) if total_count > 0 else 0.0
    return {"sensitive_count_px": int(np.count_nonzero(sensitive_occupied)), "resistant_count_px": int(np.count_nonzero(resistant_occupied)), "resistant_dominant_count_px": resistant_dominant_count, "total_count_px": total_count, "area_mm2": float(total_count * area_per_pixel_mm2(params)), "resistant_fraction": resistant_fraction}


def make_empty_series(total_time):
    return {
        "area_mm2": np.full(total_time, np.nan, dtype=np.float32),
        "resistant_fraction": np.full(total_time, np.nan, dtype=np.float32),
        "sensitive_count_px": np.full(total_time, -1, dtype=np.int32),
        "resistant_count_px": np.full(total_time, -1, dtype=np.int32),
        "resistant_dominant_count_px": np.full(total_time, -1, dtype=np.int32),
        "total_count_px": np.full(total_time, -1, dtype=np.int32),
        "treatment_efficacy": np.full(total_time, np.nan, dtype=np.float32),
    }


def record_scalar_summary(series, step, summary, treatment_efficacy):
    series["area_mm2"][step] = float(summary["area_mm2"])
    series["resistant_fraction"][step] = float(summary["resistant_fraction"])
    series["sensitive_count_px"][step] = int(summary["sensitive_count_px"])
    series["resistant_count_px"][step] = int(summary["resistant_count_px"])
    series["resistant_dominant_count_px"][step] = int(summary["resistant_dominant_count_px"])
    series["total_count_px"][step] = int(summary["total_count_px"])
    series["treatment_efficacy"][step] = float(treatment_efficacy)


def find_ttp_index(area_trace, threshold_mm2):
    progression_indices = np.flatnonzero(np.asarray(area_trace, dtype=float) >= float(threshold_mm2))
    if progression_indices.size > 0:
        return int(progression_indices[0]), True
    return int(len(area_trace) - 1), False


def derive_measurement_values(series, params, threshold_mm2, steps_per_hour):
    ttp_index, ttp_reached = find_ttp_index(series["area_mm2"], threshold_mm2)
    endpoint_index = int(len(series["resistant_fraction"]) - 1)
    start_point = int(params["start_point"])
    return {
        "endpoint_step": endpoint_index,
        "endpoint_hours": (float(endpoint_index) - float(start_point)) / float(steps_per_hour),
        "endpoint_area_mm2": float(series["area_mm2"][endpoint_index]),
        "endpoint_total_count_px": int(series["total_count_px"][endpoint_index]),
        "endpoint_sensitive_count_px": int(series["sensitive_count_px"][endpoint_index]),
        "endpoint_resistant_count_px": int(series["resistant_count_px"][endpoint_index]),
        "endpoint_resistant_dominant_count_px": int(series["resistant_dominant_count_px"][endpoint_index]),
        "endpoint_resistant_fraction": float(series["resistant_fraction"][endpoint_index]),
        "ttp_reached": bool(ttp_reached),
        "ttp_step": ttp_index,
        "ttp_hours": (float(ttp_index) - float(start_point)) / float(steps_per_hour),
        "ttp_threshold_mm2": float(threshold_mm2),
        "ttp_area_mm2": float(series["area_mm2"][ttp_index]),
        "ttp_total_count_px": int(series["total_count_px"][ttp_index]),
        "ttp_sensitive_count_px": int(series["sensitive_count_px"][ttp_index]),
        "ttp_resistant_count_px": int(series["resistant_count_px"][ttp_index]),
        "ttp_resistant_dominant_count_px": int(series["resistant_dominant_count_px"][ttp_index]),
        "ttp_resistant_fraction": float(series["resistant_fraction"][ttp_index]),
    }


def treatment_metadata(treatment, steps_per_hour):
    metadata = {"treatment_id": treatment["id"], "treatment_label": treatment["label"], "treatment_mode": treatment["mode"]}
    if treatment["mode"] == "continuous_constant_efficacy":
        metadata["target_efficacy"] = float(treatment["target_efficacy"])
        metadata["control_parameter"] = float(treatment["target_efficacy"])
        metadata["control_parameter_name"] = "target_efficacy"
        return metadata
    treat_on = int(treatment["treat_on"])
    treat_off = int(treatment["treat_off"])
    period = treat_on + treat_off
    metadata["treat_on"] = treat_on
    metadata["treat_off"] = treat_off
    metadata["treat_on_hours"] = float(treat_on) / float(steps_per_hour)
    metadata["treat_off_hours"] = float(treat_off) / float(steps_per_hour)
    metadata["pulse"] = bool(treatment.get("pulse", False))
    metadata["pulse_duration"] = "" if treatment.get("pulse_duration") is None else int(treatment["pulse_duration"])
    metadata["duty_cycle"] = float(treat_on / period) if period > 0 else 0.0
    metadata["control_parameter"] = metadata["duty_cycle"]
    metadata["control_parameter_name"] = "duty_cycle"
    return metadata


def add_parameter_sample_metadata(row, parameter_sample):
    if parameter_sample is None:
        return row
    row["parameter_sample_index"] = int(parameter_sample["parameter_sample_index"])
    row["parameter_sample_label"] = str(parameter_sample["parameter_sample_label"])
    row["parameter_sample_source"] = str(parameter_sample["parameter_sample_source"])
    row["parameter_sample_path"] = str(parameter_sample["parameter_sample_path"])
    for key, value in parameter_sample["parameter_overrides"].items():
        row[key] = value
    return row


def build_run_row(treatment, run_index, params, series, config, data_source, parameter_sample):
    row = treatment_metadata(treatment, float(config["steps_per_hour"]))
    row["run_name"] = f"{treatment['id']}_{int(run_index)}"
    row["run_index"] = int(run_index)
    row["data_source"] = data_source
    row.update(derive_measurement_values(series, params, float(config["ttp_threshold_mm2"]), float(config["steps_per_hour"])))
    return add_parameter_sample_metadata(row, parameter_sample)


def get_series_fields():
    return ["area_mm2", "resistant_fraction", "sensitive_count_px", "resistant_count_px", "resistant_dominant_count_px", "total_count_px", "treatment_efficacy"]


def get_count_series_fields():
    return {"sensitive_count_px", "resistant_count_px", "resistant_dominant_count_px", "total_count_px"}


def validate_endpoint_fields(endpoint_fields):
    supported_fields = {"sensitive", "resistant", "nutrients"}
    for field in endpoint_fields:
        if field not in supported_fields:
            raise ValueError(f"Unsupported endpoint field '{field}'. Choose from sensitive, resistant, nutrients.")


def schedule_output_stem(output_stem, treatment):
    return output_stem.with_name(f"{output_stem.name}_{treatment['id']}")


def get_output_paths(output_stem, treatment):
    stem = schedule_output_stem(output_stem, treatment)
    paths = {"values_csv": stem.with_name(f"{stem.name}_runs").with_suffix(".csv"), "series_status": stem.with_name(f"{stem.name}_series_status").with_suffix(".npy"), "series_metadata": stem.with_name(f"{stem.name}_series_metadata").with_suffix(".npz"), "endpoint_status": stem.with_name(f"{stem.name}_endpoint_status").with_suffix(".npy")}
    for field in get_series_fields():
        paths[f"series_{field}"] = stem.with_name(f"{stem.name}_series_{field}").with_suffix(".npy")
    for field in ("sensitive", "resistant", "nutrients"):
        paths[f"endpoint_{field}"] = stem.with_name(f"{stem.name}_endpoint_{field}").with_suffix(".npy")
    return paths


def ensure_npy_memmap(path, shape, dtype, fill_value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = np.lib.format.open_memmap(path, mode="r+")
        if existing.shape == tuple(shape) and existing.dtype == np.dtype(dtype):
            del existing
            return
        del existing
        path.unlink()
    array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    array[...] = fill_value
    array.flush()
    del array


def ensure_scalar_store(output_stem, treatment, num_runs, total_time, config):
    paths = get_output_paths(output_stem, treatment)
    ensure_npy_memmap(paths["series_status"], (num_runs,), np.bool_, False)
    for field in get_series_fields():
        dtype = np.int32 if field in get_count_series_fields() else np.float32
        fill_value = -1 if field in get_count_series_fields() else np.nan
        ensure_npy_memmap(paths[f"series_{field}"], (num_runs, total_time), dtype, fill_value)
    np.savez(paths["series_metadata"], treatment_id=np.asarray([treatment["id"]], dtype="<U128"), total_time=np.asarray([total_time], dtype=np.int32), num_runs=np.asarray([num_runs], dtype=np.int32), ttp_threshold_mm2=np.asarray([float(config["ttp_threshold_mm2"])], dtype=np.float32), steps_per_hour=np.asarray([float(config["steps_per_hour"])], dtype=np.float32), fields=np.asarray(get_series_fields(), dtype="<U64"))
    return paths


def ensure_endpoint_store(output_stem, treatment, num_runs, image_size, endpoint_fields):
    paths = get_output_paths(output_stem, treatment)
    ensure_npy_memmap(paths["endpoint_status"], (num_runs,), np.bool_, False)
    for field in endpoint_fields:
        ensure_npy_memmap(paths[f"endpoint_{field}"], (num_runs, image_size, image_size), np.float32, np.nan)
    return paths


def write_time_series(paths, run_index, series):
    for field in get_series_fields():
        array = np.lib.format.open_memmap(paths[f"series_{field}"], mode="r+")
        array[int(run_index)] = series[field]
        array.flush()
        del array
    status = np.lib.format.open_memmap(paths["series_status"], mode="r+")
    status[int(run_index)] = True
    status.flush()
    del status


def write_endpoint_state(paths, run_index, endpoint_state, endpoint_fields):
    if not endpoint_fields:
        return
    for field in endpoint_fields:
        array = np.lib.format.open_memmap(paths[f"endpoint_{field}"], mode="r+")
        array[int(run_index)] = np.asarray(endpoint_state[field], dtype=np.float32)
        array.flush()
        del array
    status = np.lib.format.open_memmap(paths["endpoint_status"], mode="r+")
    status[int(run_index)] = True
    status.flush()
    del status


def load_series_status(paths):
    if not paths["series_status"].exists():
        return np.asarray([], dtype=np.bool_)
    return np.asarray(np.load(paths["series_status"], mmap_mode="r"), dtype=np.bool_)


def load_time_series(paths, run_index):
    series = {}
    for field in get_series_fields():
        series[field] = np.asarray(np.load(paths[f"series_{field}"], mmap_mode="r")[int(run_index)])
    return series


def simulate_run_task(task):
    project_root = pl.Path(task["project_root"])
    if pl.Path.cwd() != project_root:
        os.chdir(project_root)
    treatment = task["treatment"]
    run_index = int(task["run_index"])
    parameter_sample = task.get("parameter_sample")
    parameter_overrides = None if parameter_sample is None else parameter_sample.get("parameter_overrides")
    sim = configure_simulation(treatment, run_index, task["base_params"], parameter_overrides)
    total_time = int(sim.params["total_time"])
    nutrients, sensitive, resistant = sim.get_initial_state()
    series = make_empty_series(total_time)
    record_scalar_summary(series, 0, initial_scalar_summary(sim.params), sim.treatment_efficacy)
    for timer in range(1, total_time):
        nutrients, sensitive, resistant = sim.update(timer, nutrients, sensitive, resistant)
        apply_fixed_mutation_if_needed(sim, timer, sensitive, resistant)
        record_scalar_summary(series, timer, scalar_summary_from_state(sim.params, sensitive, resistant), sim.treatment_efficacy)
    write_time_series(task["series_paths"], run_index, series)
    endpoint_state = {"nutrients": np.asarray(nutrients, dtype=np.float32), "sensitive": np.asarray(sensitive, dtype=np.float32), "resistant": np.asarray(resistant, dtype=np.float32)}
    write_endpoint_state(task.get("endpoint_paths"), run_index, endpoint_state, task.get("endpoint_fields", []))
    row = build_run_row(treatment, run_index, sim.params, series, task["config"], "simulated_in_memory", parameter_sample)
    return row


def auto_discover_sample_source(project_root):
    bootstrap_root = project_root / "source" / "fit" / "fit_results" / "bootstrap_uncertainties"
    if bootstrap_root.exists():
        joint_candidates = sorted(bootstrap_root.glob("run_*/joint_parameter_samples.csv"))
        if joint_candidates:
            return "bootstrap_joint", joint_candidates[-1]
        raw_candidates = sorted(bootstrap_root.glob("run_*/raw_bootstrap_samples.npz"))
        if raw_candidates:
            return "bootstrap_raw", raw_candidates[-1]
    validation_root = project_root / "Validation" / "results" / "end_to_end"
    if validation_root.exists():
        fold_candidates = sorted(validation_root.glob("end_to_end_loocv_*/fold_results.csv"))
        if fold_candidates:
            return "fold_results", fold_candidates[-1]
    if bootstrap_root.exists():
        summary_candidates = sorted(bootstrap_root.glob("run_*/bootstrap_summary.csv"))
        if summary_candidates:
            return "bootstrap_summary", summary_candidates[-1]
    return None, None


def build_parameter_samples(project_root, base_params, config):
    urc = load_uncertainty_common()
    selected_source = None
    selected_path = None
    if config["bootstrap_joint"] is not None:
        selected_source = "bootstrap_joint"
        selected_path = resolve_path(project_root, config["bootstrap_joint"])
    elif config["bootstrap_raw"] is not None:
        selected_source = "bootstrap_raw"
        selected_path = resolve_path(project_root, config["bootstrap_raw"])
    elif config["fold_results"] is not None:
        selected_source = "fold_results"
        selected_path = resolve_path(project_root, config["fold_results"])
    elif config["bootstrap_summary"] is not None:
        selected_source = "bootstrap_summary"
        selected_path = resolve_path(project_root, config["bootstrap_summary"])
    else:
        selected_source, selected_path = auto_discover_sample_source(project_root)
    if selected_source is None or selected_path is None:
        raise ValueError("No bootstrap uncertainty source was provided or auto-discovered.")
    num_samples = int(config["num_runs"])
    random_seed = int(config["parameter_sample_seed"])
    if selected_source == "bootstrap_joint":
        raw_df = urc.load_parameter_samples_from_bootstrap_joint(str(selected_path), base_params=base_params, num_samples=num_samples, random_seed=random_seed)
    elif selected_source == "bootstrap_raw":
        raw_df = urc.load_parameter_samples_from_bootstrap_raw(str(selected_path), base_params=base_params, num_samples=num_samples, random_seed=random_seed)
    elif selected_source == "fold_results":
        raw_df = urc.load_parameter_samples_from_fold_results(str(selected_path), base_params=base_params, num_samples=num_samples, random_seed=random_seed)
    else:
        raw_df = urc.load_parameter_samples_from_bootstrap_summary(str(selected_path), base_params=base_params, num_samples=num_samples, random_seed=random_seed)
    final_df = urc.finalize_parameter_samples(raw_df, base_params=base_params)
    final_df["source_path"] = str(selected_path)
    return final_df


def prepare_parameter_sample_lookup(sample_df, parameter_sample_keys, base_params):
    lookup = {}
    for sample_index, sample_row in sample_df.iterrows():
        overrides = {}
        for key in parameter_sample_keys:
            value = sample_row[key] if key in sample_row else base_params[key]
            overrides[key] = int(round(float(value))) if key == "start_point" else float(value)
        if "diffusion_sensitive" in overrides and "diffusion_resistant" not in overrides:
            overrides["diffusion_resistant"] = float(overrides["diffusion_sensitive"])
        lookup[int(sample_index)] = {"parameter_sample_index": int(sample_index), "parameter_sample_label": str(sample_row.get("sample_label", f"parameter_sample_{sample_index:03d}")), "parameter_sample_source": str(sample_row.get("sample_source", "bootstrap_uncertainty")), "parameter_sample_path": str(sample_row.get("source_path", "")), "parameter_overrides": overrides}
    return lookup


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_rows_from_time_series(paths, treatment, base_params, config, parameter_sample_lookup):
    status = load_series_status(paths)
    if status.size == 0:
        return []
    rows = []
    for run_index in np.flatnonzero(status):
        parameter_sample = None if parameter_sample_lookup is None else parameter_sample_lookup[int(run_index)]
        params = copy.deepcopy(base_params)
        if parameter_sample is not None:
            apply_parameter_overrides(params, parameter_sample["parameter_overrides"])
        series = load_time_series(paths, int(run_index))
        rows.append(build_run_row(treatment, int(run_index), params, series, config, "cached_time_series", parameter_sample))
    return rows


def sort_rows(rows, treatments):
    order = {treatment["id"]: index for index, treatment in enumerate(treatments)}
    return sorted(rows, key=lambda row: (order.get(str(row["treatment_id"]), len(order)), int(row["run_index"])))


def build_missing_tasks(treatment, available_indices, base_params, config, paths, endpoint_paths, parameter_sample_lookup):
    tasks = []
    for run_index in range(int(config["num_runs"])):
        if run_index in available_indices:
            continue
        parameter_sample = None if parameter_sample_lookup is None else parameter_sample_lookup[run_index]
        tasks.append({"project_root": str(get_project_root()), "treatment": treatment, "run_index": run_index, "base_params": base_params, "config": config, "series_paths": paths, "endpoint_paths": endpoint_paths, "endpoint_fields": list(config["endpoint_array_fields"]) if bool(config["save_endpoint_arrays"]) else [], "parameter_sample": parameter_sample})
    return tasks


def run_tasks(tasks, max_workers, progress_interval):
    if not tasks:
        return []
    resolved_workers = max(1, min(resolve_max_workers(max_workers), len(tasks)))
    rows = []
    completed = 0
    report_every = max(1, int(progress_interval))
    if resolved_workers == 1:
        for task in tasks:
            rows.append(simulate_run_task(task))
            completed += 1
            if completed % report_every == 0 or completed == len(tasks):
                print(f"  completed {completed}/{len(tasks)} simulated runs")
        return rows
    with cf.ProcessPoolExecutor(max_workers=resolved_workers) as executor:
        futures = [executor.submit(simulate_run_task, task) for task in tasks]
        for future in cf.as_completed(futures):
            rows.append(future.result())
            completed += 1
            if completed % report_every == 0 or completed == len(tasks):
                print(f"  completed {completed}/{len(tasks)} simulated runs")
    return rows


def run_treatment(treatment, base_params, config, output_stem, parameter_sample_lookup):
    total_time = int(base_params["total_time"])
    paths = ensure_scalar_store(output_stem, treatment, int(config["num_runs"]), total_time, config)
    endpoint_paths = None
    if bool(config["save_endpoint_arrays"]):
        endpoint_paths = ensure_endpoint_store(output_stem, treatment, int(config["num_runs"]), int(base_params["image_size"]), list(config["endpoint_array_fields"]))
    cached_rows = load_rows_from_time_series(paths, treatment, base_params, config, parameter_sample_lookup) if bool(config["resume_from_outputs"]) else []
    available_indices = {int(row["run_index"]) for row in cached_rows}
    tasks = build_missing_tasks(treatment, available_indices, base_params, config, paths, endpoint_paths, parameter_sample_lookup)
    print(f"{treatment['label']}: {len(cached_rows)} run(s) already available, simulating {len(tasks)} missing run(s).")
    simulated_rows = run_tasks(tasks, config["max_workers"], config["progress_interval"])
    rows = sort_rows(cached_rows + simulated_rows, [treatment])
    save_csv(paths["values_csv"], rows)
    print(f"Saved run values: {paths['values_csv']}")
    return rows


def finite_values(rows, field):
    values = np.asarray([float(row[field]) for row in rows if row.get(field) not in ("", None)], dtype=float)
    return values[np.isfinite(values)]


def calculate_distribution_metrics(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    row = {"n_runs": int(values.size), "mean": np.nan, "std_population": np.nan, "std_sample": np.nan, "cv_population": np.nan, "cv_sample": np.nan, "median": np.nan, "q25": np.nan, "q75": np.nan, "min": np.nan, "max": np.nan, "raw_moment_1": np.nan, "raw_moment_2": np.nan, "raw_moment_3": np.nan, "raw_moment_4": np.nan, "central_moment_2": np.nan, "central_moment_3": np.nan, "central_moment_4": np.nan, "skewness": np.nan, "kurtosis_pearson": np.nan, "binder_raw": np.nan, "binder_centered": np.nan, "bimodality_coefficient": np.nan}
    if values.size == 0:
        return row
    mean = float(np.mean(values))
    centered = values - mean
    raw2 = float(np.mean(values**2))
    raw4 = float(np.mean(values**4))
    cm2 = float(np.mean(centered**2))
    cm3 = float(np.mean(centered**3))
    cm4 = float(np.mean(centered**4))
    std_population = float(np.sqrt(cm2))
    std_sample = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    skewness = float(cm3 / (cm2 ** 1.5)) if cm2 > 0 else np.nan
    kurtosis = float(cm4 / (cm2**2)) if cm2 > 0 else np.nan
    row.update({"mean": mean, "std_population": std_population, "std_sample": std_sample, "cv_population": float(std_population / mean) if mean != 0 else np.nan, "cv_sample": float(std_sample / mean) if mean != 0 else np.nan, "median": float(np.median(values)), "q25": float(np.percentile(values, 25)), "q75": float(np.percentile(values, 75)), "min": float(np.min(values)), "max": float(np.max(values)), "raw_moment_1": mean, "raw_moment_2": raw2, "raw_moment_3": float(np.mean(values**3)), "raw_moment_4": raw4, "central_moment_2": cm2, "central_moment_3": cm3, "central_moment_4": cm4, "skewness": skewness, "kurtosis_pearson": kurtosis, "binder_raw": float(1.0 - raw4 / (3.0 * raw2 * raw2)) if raw2 > 0 else np.nan, "binder_centered": float(1.0 - cm4 / (3.0 * cm2 * cm2)) if cm2 > 0 else np.nan, "bimodality_coefficient": float((skewness**2 + 1.0) / kurtosis) if np.isfinite(skewness) and np.isfinite(kurtosis) and kurtosis != 0 else np.nan})
    return row


def histogram_rows(treatment, measurement_mode, values, bins):
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    counts, _ = np.histogram(values, bins=edges)
    width = edges[1] - edges[0]
    density = counts / (max(1, len(values)) * width)
    rows = []
    for index, count in enumerate(counts):
        rows.append({"measurement_mode": measurement_mode, "treatment_id": treatment["id"], "treatment_label": treatment["label"], "bin_index": index, "bin_left": float(edges[index]), "bin_right": float(edges[index + 1]), "bin_center": float((edges[index] + edges[index + 1]) / 2.0), "count": int(count), "density": float(density[index])})
    return rows


def kde_rows(treatment, measurement_mode, values, grid_points):
    grid = np.linspace(0.0, 1.0, int(grid_points))
    density = np.zeros_like(grid)
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size > 1 and float(np.std(finite)) > 0:
        try:
            from scipy.stats import gaussian_kde
            density = gaussian_kde(finite)(grid)
        except Exception:
            density = np.zeros_like(grid)
    return [{"measurement_mode": measurement_mode, "treatment_id": treatment["id"], "treatment_label": treatment["label"], "grid_index": index, "resistant_fraction": float(grid[index]), "density": float(density[index])} for index in range(len(grid))]


def find_peak_indices(density, prominence_fraction):
    density = np.asarray(density, dtype=float)
    if density.size < 3 or np.nanmax(density) <= 0:
        return []
    prominence = float(np.nanmax(density)) * float(prominence_fraction)
    try:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(density, prominence=prominence)
        return [int(index) for index in peaks]
    except Exception:
        peaks = []
        for index in range(1, len(density) - 1):
            if density[index] > density[index - 1] and density[index] > density[index + 1] and density[index] >= prominence:
                peaks.append(index)
        return peaks


def peak_rows(treatment, measurement_mode, kde_data, prominence_fraction):
    density = np.asarray([row["density"] for row in kde_data], dtype=float)
    grid = np.asarray([row["resistant_fraction"] for row in kde_data], dtype=float)
    peak_indices = find_peak_indices(density, prominence_fraction)
    rows = []
    for peak_number, index in enumerate(peak_indices, start=1):
        rows.append({"measurement_mode": measurement_mode, "treatment_id": treatment["id"], "treatment_label": treatment["label"], "peak_number": peak_number, "peak_resistant_fraction": float(grid[index]), "peak_density": float(density[index]), "n_peaks": int(len(peak_indices)), "is_bimodal": bool(len(peak_indices) >= 2)})
    if not rows:
        rows.append({"measurement_mode": measurement_mode, "treatment_id": treatment["id"], "treatment_label": treatment["label"], "peak_number": "", "peak_resistant_fraction": np.nan, "peak_density": np.nan, "n_peaks": 0, "is_bimodal": False})
    return rows


def measurement_value_rows(rows, measurement_mode):
    field = f"{measurement_mode}_resistant_fraction"
    selected_rows = []
    for row in rows:
        selected = dict(row)
        selected["measurement_mode"] = measurement_mode
        selected["resistant_fraction"] = float(row[field])
        selected_rows.append(selected)
    return selected_rows


def plot_histogram(output_stem, treatment, measurement_mode, values, bins):
    color = treatment.get("color", "#4c72b0")
    fig, ax = plt.subplots(figsize=(8.4 / 2.15, 6.5 / 1.9), dpi=300)
    ax.hist(values, bins=np.linspace(0.0, 1.0, int(bins) + 1), color=color, edgecolor="white", linewidth=0.6, alpha=0.9)
    if len(values) > 0:
        ax.axvline(np.mean(values), color="black", linewidth=1.0, linestyle=":", label=f"Mean = {np.mean(values):.3f}")
        ax.axvline(np.median(values), color="black", linewidth=1.1, linestyle="--", label=f"Median = {np.median(values):.3f}")
    ax.set_title(f"{treatment['label']} {measurement_mode} (n={len(values)})")
    ax.set_xlabel("Resistant fraction")
    ax.set_ylabel("Run count")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    base = output_stem.with_name(f"{output_stem.name}_{measurement_mode}_{treatment['id']}_histogram")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", transparent=True)
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", transparent=True, dpi=300)
    plt.close(fig)
    return base.with_suffix(".pdf"), base.with_suffix(".png")


def save_analysis_for_mode(output_stem, treatments, rows, measurement_mode, config):
    selected_rows = measurement_value_rows(rows, measurement_mode)
    save_csv(output_stem.with_name(f"{output_stem.name}_{measurement_mode}_values.csv"), sort_rows(selected_rows, treatments))
    summary_rows = []
    histogram_data = []
    kde_data = []
    peaks_data = []
    for treatment in treatments:
        treatment_rows = [row for row in selected_rows if row["treatment_id"] == treatment["id"]]
        values = finite_values(treatment_rows, "resistant_fraction")
        metrics = calculate_distribution_metrics(values)
        treatment_kde = kde_rows(treatment, measurement_mode, values, int(config["kde_grid_points"]))
        treatment_peaks = peak_rows(treatment, measurement_mode, treatment_kde, float(config["bimodality_peak_prominence_fraction"]))
        summary = treatment_metadata(treatment, float(config["steps_per_hour"]))
        summary["measurement_mode"] = measurement_mode
        summary.update(metrics)
        summary["n_density_peaks"] = int(treatment_peaks[0]["n_peaks"])
        summary["is_bimodal"] = bool(treatment_peaks[0]["is_bimodal"])
        summary_rows.append(summary)
        histogram_data.extend(histogram_rows(treatment, measurement_mode, values, int(config["bins"])))
        kde_data.extend(treatment_kde)
        peaks_data.extend(treatment_peaks)
        plot_histogram(output_stem, treatment, measurement_mode, values, int(config["bins"]))
    save_csv(output_stem.with_name(f"{output_stem.name}_{measurement_mode}_summary.csv"), summary_rows)
    save_csv(output_stem.with_name(f"{output_stem.name}_{measurement_mode}_histogram_data.csv"), histogram_data)
    save_csv(output_stem.with_name(f"{output_stem.name}_{measurement_mode}_kde_data.csv"), kde_data)
    save_csv(output_stem.with_name(f"{output_stem.name}_{measurement_mode}_peaks.csv"), peaks_data)
    return summary_rows


def save_all_analysis(output_stem, treatments, rows, config):
    all_summary_rows = []
    for measurement_mode in config["analysis_measurement_modes"]:
        if measurement_mode not in {"endpoint", "ttp"}:
            raise ValueError("analysis_measurement_modes may only contain 'endpoint' and 'ttp'.")
        all_summary_rows.extend(save_analysis_for_mode(output_stem, treatments, rows, measurement_mode, config))
    save_csv(output_stem.with_name(f"{output_stem.name}_all_measurement_summary.csv"), all_summary_rows)
    return all_summary_rows


def main(config=None):
    configure_plot_style()
    project_root = configure_import_paths()
    config = merge_config(config)
    output_stem = normalize_output_stem(project_root, config["output_stem"])
    urc = load_uncertainty_common()
    params_path = resolve_path(project_root, config["params_path"])
    base_params = urc.load_params(str(params_path))
    if config["total_time"] is not None:
        base_params["total_time"] = int(config["total_time"])
    if int(config["num_runs"]) <= 0:
        raise ValueError("num_runs must be positive.")
    if resolve_max_workers(config["max_workers"]) <= 0:
        raise ValueError("max_workers must be positive.")
    validate_endpoint_fields(config["endpoint_array_fields"])
    treatments = build_treatments(config)
    parameter_sample_lookup = None
    if bool(config["sample_parameter_uncertainty"]):
        sample_df = build_parameter_samples(project_root, base_params, config)
        if len(sample_df) != int(config["num_runs"]):
            raise ValueError(f"Requested {config['num_runs']} parameter samples, but the selected source yielded {len(sample_df)}.")
        parameter_sample_lookup = prepare_parameter_sample_lookup(sample_df, list(config["parameter_sample_keys"]), base_params)
        sample_csv = output_stem.with_name(f"{output_stem.name}_parameter_samples.csv")
        save_csv(sample_csv, sample_df.to_dict(orient="records"))
        print(f"Saved parameter samples: {sample_csv}")
    all_rows = []
    for treatment in treatments:
        all_rows.extend(run_treatment(treatment, base_params, config, output_stem, parameter_sample_lookup))
    all_rows = sort_rows(all_rows, treatments)
    save_csv(output_stem.with_name(f"{output_stem.name}_run_values.csv"), all_rows)
    summary_rows = save_all_analysis(output_stem, treatments, all_rows, config)
    print(f"Saved run values: {output_stem.with_name(f'{output_stem.name}_run_values.csv')}")
    for row in summary_rows:
        print(f"{row['treatment_label']} {row['measurement_mode']}: n={row['n_runs']}, median={row['median']:.4f}, CV={row['cv_sample']:.4f}, skewness={row['skewness']:.4f}, Binder={row['binder_centered']:.4f}, bimodal={row['is_bimodal']}")
    if bool(config["show"]):
        plt.show()


if __name__ == "__main__":
    run_config = default_config()
    main(run_config)  # set runtime to 10000 and image_size to 500 in the params file for these simulations
