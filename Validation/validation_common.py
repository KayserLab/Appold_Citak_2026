import ast
import datetime as dt
import json
import logging
import os
import pathlib as pl
import sys

import matplotlib.path as mpl_path
import numpy as np
import pandas as pd
import scipy.optimize as opt
import yaml
from source import core as cr

class FitOutcome:
    def __init__(self, result, metadata):
        self.result = result
        self.metadata = metadata


def find_project_root(current_dir, marker_file):
    current_dir = pl.Path(current_dir).resolve()
    while current_dir != current_dir.parent:
        if (current_dir / marker_file).exists():
            return current_dir
        current_dir = current_dir.parent
    return None


def json_ready(value):
    if isinstance(value, pl.Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def load_params_yaml(file_path):
    with (find_project_root(os.getcwd(), "setup.py") / file_path).open("r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def create_run_directory(base_dir, prefix):
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def setup_logger(log_path):
    logger = logging.getLogger(f"validation_{log_path.stem}")
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


def save_json(path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2)


def summarize_vector(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
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


def build_parameter_summary(fold_df, parameter_columns, full_fit_values):
    rows = []
    for column in parameter_columns:
        summary = summarize_vector(fold_df[column].to_numpy(dtype=float))
        rows.append({"parameter": column,
                     "full_data_fit": float(full_fit_values[column]),
                     "loo_mean": summary["mean"],
                     "loo_std": summary["std"],
                     "loo_median": summary["median"],
                     "loo_min": summary["min"],
                     "loo_max": summary["max"],
                     "n_folds": summary["count"]})
    return pd.DataFrame(rows)


def summarize_fold_metrics(fold_df, metric_columns):
    summary = {}
    for column in metric_columns:
        summary[column] = summarize_vector(fold_df[column].to_numpy(dtype=float))
    return summary


def relu_model(t, t0, m):
    return np.maximum(0.0, m * (np.asarray(t, dtype=float) - float(t0)))


def fit_relu_to_points_active(x_h, y_um, p0=(70.0, 20.0), n_iter=5):
    x_h = np.asarray(x_h, dtype=float)
    y_um = np.asarray(y_um, dtype=float)
    bounds_lower = [0.0, 0.0]
    bounds_upper = [200.0, 200.0]

    t0, m = p0
    for _ in range(int(n_iter)):
        active_mask = x_h >= t0
        if np.count_nonzero(active_mask) < 5:
            break
        popt, _ = opt.curve_fit(relu_model, x_h[active_mask], y_um[active_mask], p0=[t0, m], bounds=(bounds_lower, bounds_upper), maxfev=20000)
        t0, m = popt
    return float(t0), float(m)


def add_unique_identifier(df, df_id):
    df = df.copy()
    df["unique_particle"] = int(df_id) * 10000 + df["particle"]
    return df


def extract_initial_distances(df, min_duration_frames, scale_factor):
    work_df = df[["unique_particle", "frame", "distance_to_edge"]].copy()
    work_df["init_dist_um_tmp"] = work_df["distance_to_edge"] * float(scale_factor)

    counts = work_df.groupby("unique_particle")["frame"].count()
    good_ids = counts[counts >= int(min_duration_frames)].index
    kept = work_df[work_df["unique_particle"].isin(good_ids)]
    first_rows = (kept.sort_values(["unique_particle", "frame"]).groupby("unique_particle").first().reset_index())
    output = first_rows.rename(columns={"frame": "init_frame",
                                        "init_dist_um_tmp": "init_dist_um"})
    return output[["unique_particle", "init_frame", "init_dist_um"]]


def _extract_continuous_well_name(path):
    prefix = "colony_data_"
    suffix = "_with_clonearea.csv"
    if not path.name.startswith(prefix) or not path.name.endswith(suffix):
        raise ValueError(f"Could not parse well name from {path.name}.")
    return path.name[len(prefix) : -len(suffix)]


def _compute_mutation_clone_count(data_dir, well, contour_frame_index=63, max_first_frame=230):
    clones = pd.read_csv(data_dir / f"clone_data_fusion_resolved_{well}.csv")
    colony = pd.read_csv(data_dir / f"colony_data_{well}_with_clonearea_with_extrapolation_to_final.csv")

    frame_data = colony.iloc[contour_frame_index]
    contour = np.array(ast.literal_eval(frame_data["colony_contour"]))
    contour_xy = contour[:, [1, 0]]
    contour_path = mpl_path.Path(contour_xy)

    first_frames = clones.groupby("particle").first().reset_index()
    filtered_first_frames = first_frames[first_frames["frame"] <= max_first_frame]
    particle_positions = filtered_first_frames[["x", "y"]].values
    inside_mask = contour_path.contains_points(particle_positions)
    filtered_first_frames = filtered_first_frames[inside_mask]
    return float(len(filtered_first_frames))


def load_mutation_rate_dataset():
    data_dir = find_project_root(os.getcwd(), "setup.py") / "data" / "exp_data" / "Continuous_therapy"
    base_paths = sorted(data_dir.glob("colony_data_*_with_clonearea.csv"))

    dataset = []
    for path in base_paths:
        well = _extract_continuous_well_name(path)
        dataset.append({"label": well,
                        "clone_count": _compute_mutation_clone_count(data_dir, well)})
    return dataset


def load_regrowth_scatter_dataset(wells, *, min_duration_frames=10, scale_factor=8.648, min_distance_um=100.0, frames_per_hour=2.0):
    data_dir = find_project_root(os.getcwd(), "setup.py") / "data" / "exp_data" / "Continuous_therapy"

    dfs = []
    for idx, well in enumerate(wells):
        df = pd.read_csv(data_dir / f"clone_data_fusion_resolved_{well}.csv")
        df = add_unique_identifier(df, idx)
        df["well_label"] = well
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    init_df = extract_initial_distances(combined, min_duration_frames=min_duration_frames, scale_factor=scale_factor)
    labeled = init_df.merge(combined[["unique_particle", "well_label"]].drop_duplicates(), on="unique_particle", how="left")
    labeled = labeled[labeled["init_dist_um"] > float(min_distance_um)].copy()
    labeled["init_time_h"] = labeled["init_frame"] / float(frames_per_hour)
    return labeled.reset_index(drop=True)


def fit_regrowth_calibration_from_continuous_therapy(wells, min_duration_frames=10, scale_factor=8.648, min_distance_um=100.0, frames_per_hour=2.0, p0=(70.0, 20.0), n_iter=5):
    scatter_df = load_regrowth_scatter_dataset(wells=wells, min_duration_frames=min_duration_frames, scale_factor=scale_factor, min_distance_um=min_distance_um, frames_per_hour=frames_per_hour)

    t0_h, slope_um_per_h = fit_relu_to_points_active(scatter_df["init_time_h"].to_numpy(dtype=float), scatter_df["init_dist_um"].to_numpy(dtype=float), p0=p0, n_iter=n_iter)
    regrowth_dict = {"wells": wells,
                     "t0_h": float(t0_h),
                     "slope_um_per_h": float(slope_um_per_h),
                     "n_points": int(len(scatter_df)),
                     "scale_factor": float(scale_factor),
                     "frames_per_hour": float(frames_per_hour),
                     "min_distance_um": float(min_distance_um),
                     "min_duration_frames": int(min_duration_frames)}
    return regrowth_dict


def evaluate_regrowth_calibration_on_well(held_out_well, regrowth_calibration, min_duration_frames=10, min_distance_um=100.0):
    held_out_points = load_regrowth_scatter_dataset(wells=[held_out_well], min_duration_frames=min_duration_frames, min_distance_um=min_distance_um)

    if held_out_points.empty:
        return {"test_regrowth_points": 0,
                "test_regrowth_time_mae_h": np.nan,
                "test_regrowth_time_rmse_h": np.nan,
                "test_regrowth_time_r2": np.nan}

    predicted_times_h = (held_out_points["init_dist_um"].to_numpy(dtype=float) / float(regrowth_calibration["slope_um_per_h"]) + float(regrowth_calibration["t0_h"]))

    observed_times_h = held_out_points["init_time_h"].to_numpy(dtype=float)
    residuals_h = predicted_times_h - observed_times_h
    mae_h = float(np.mean(np.abs(residuals_h)))
    rmse_h = float(np.sqrt(np.mean(residuals_h**2)))
    total_variance = float(np.sum((observed_times_h - np.mean(observed_times_h)) ** 2))
    r2 = float(1.0 - np.sum(residuals_h**2) / total_variance)

    return {"test_regrowth_points": int(len(held_out_points)),
            "test_regrowth_time_mae_h": mae_h,
            "test_regrowth_time_rmse_h": rmse_h,
            "test_regrowth_time_r2": r2}


def regrowth_offsets_to_detection_frames(offsets_in_sim_pixels, start_point, regrowth_calibration, sim_pixel_size_um=13.76, sim_steps_per_experimental_frame=10.0):
    offsets = np.asarray(offsets_in_sim_pixels, dtype=float)
    offset_um = offsets * float(sim_pixel_size_um) * float(regrowth_calibration["scale_factor"])
    detection_time_h = offset_um / float(regrowth_calibration["slope_um_per_h"]) + float(regrowth_calibration["t0_h"])
    detection_frame = detection_time_h * float(regrowth_calibration["frames_per_hour"])
    return detection_frame * float(sim_steps_per_experimental_frame) + float(start_point)


def build_effective_simulation_params(params_yaml, start_point_override=None, simulation_param_overrides=None):
    effective_params = dict(params_yaml)
    if simulation_param_overrides is not None:
        effective_params.update(simulation_param_overrides)
    if start_point_override is not None:
        effective_params["start_point"] = int(start_point_override)
    return effective_params


def build_dispersion_parameter_overrides(dispersion_params):
    dispersion_params = np.asarray(dispersion_params, dtype=float)
    return {"diffusion_sensitive": float(dispersion_params[0]),
            "diffusion_resistant": float(dispersion_params[0]),
            "uptake_rate": float(dispersion_params[1]),
            "diffusion_nutrients": float(dispersion_params[2])}


def simulate_mutation_counts(mutation_rate, params_yaml, replicas, seed_offset=0, start_point_override=None, simulation_param_overrides=None):
    contour_max_frame = 230
    effective_params = build_effective_simulation_params(params_yaml, start_point_override=start_point_override, simulation_param_overrides=simulation_param_overrides)

    start_point = int(effective_params["start_point"])
    total_time = contour_max_frame * 10 + start_point
    treatment_start = int(effective_params["treatment_start"]) + start_point
    counts = []

    for seed in range(replicas):
        sim = cr.DiffusionModel2D()
        sim.params.update(effective_params)
        sim.random_seed = int(seed + seed_offset)
        sim.set_random_seed()
        sim.params["total_time"] = total_time
        sim.params["start_point"] = start_point
        sim.params["mutation_rate"] = float(mutation_rate)
        sim.params["mutations_active"] = True
        sim.treatment_times = np.zeros(total_time, dtype=bool)
        sim.treatment_times[treatment_start:] = True

        nutrients, sensitive, resistant = sim.get_initial_state()
        for timer in range(1, total_time):
            nutrients, sensitive, resistant = sim.update(timer, nutrients, sensitive, resistant)
        counts.append(float(sim.mutation_count))

    return np.array(counts, dtype=float)


def mutation_rate_objective(x, exp_mean, params_yaml, sim_replicas, start_point_override, simulation_param_overrides):
    mutation_rate = float(np.atleast_1d(x)[0])
    if not np.isfinite(mutation_rate) or mutation_rate < 0.0:
        return np.inf

    sim_counts = simulate_mutation_counts(mutation_rate=mutation_rate, params_yaml=params_yaml, replicas=sim_replicas, start_point_override=start_point_override, simulation_param_overrides=simulation_param_overrides)
    sim_mean = float(np.mean(sim_counts))
    error = float((exp_mean - sim_mean) ** 2)
    return error


def fit_mutation_rate_dataset(clone_counts, initial_guess, params_yaml, maxiter, sim_replicas, start_point_override=None, simulation_param_overrides=None):
    clone_counts = np.asarray(clone_counts, dtype=float)
    exp_mean = np.mean(clone_counts)
    exp_sem = np.std(clone_counts, ddof=1) / np.sqrt(clone_counts.size)
    initial_guess = max(0.0, float(initial_guess))

    result = opt.minimize(mutation_rate_objective, np.array([initial_guess], dtype=float), args=(exp_mean, params_yaml, sim_replicas, start_point_override, simulation_param_overrides), method="Nelder-Mead", bounds=opt.Bounds([0.0], [np.inf]), options={"disp": False, "maxiter": int(maxiter)})

    fitted_mutation_rate = max(0.0, float(np.atleast_1d(result.x)[0]))
    fitted_sim_counts = simulate_mutation_counts(mutation_rate=fitted_mutation_rate, params_yaml=params_yaml, replicas=sim_replicas, start_point_override=start_point_override, simulation_param_overrides=simulation_param_overrides)
    fitted_sim_mean = np.mean(fitted_sim_counts)
    standardized_mean_error = (fitted_sim_mean - exp_mean)/ exp_sem if exp_sem > 0 else np.nan

    return FitOutcome(result=result, metadata={"exp_clone_mean": exp_mean,
                                               "exp_clone_sem": exp_sem,
                                               "fitted_sim_clone_mean": fitted_sim_mean,
                                               "fitted_clone_mean_error_exp_sem": standardized_mean_error,
                                               "num_wells": int(clone_counts.size),
                                               "start_point_used": (int(start_point_override) if start_point_override is not None else int(params_yaml["start_point"]))})


def evaluate_mutation_rate_prediction(mutation_rate, held_out_clone_count, params_yaml, prediction_replicas, seed_offset=10000, start_point_override=None, simulation_param_overrides=None):
    predicted_counts = simulate_mutation_counts(mutation_rate=mutation_rate, params_yaml=params_yaml, replicas=prediction_replicas, seed_offset=seed_offset, start_point_override=start_point_override, simulation_param_overrides=simulation_param_overrides)

    predicted_mean = np.mean(predicted_counts)
    predicted_std = np.std(predicted_counts, ddof=1)
    error = predicted_mean - float(held_out_clone_count)

    return {"test_predicted_clone_mean": predicted_mean,
            "test_predicted_clone_std": predicted_std,
            "test_error": float(error),
            "test_abs_error": float(abs(error)),
            "test_squared_error": float(error**2),
            "test_relative_abs_error": (float(abs(error) / abs(held_out_clone_count)) if held_out_clone_count != 0 else np.nan),
            "test_standardized_residual": (float(error / predicted_std) if predicted_std > 0 else np.nan)}


def load_no_treatment_area_dataset():
    data_dir = find_project_root(os.getcwd(), "setup.py") / "data" / "exp_data" / "No_treatment_control"
    area_dataset = []

    for path in sorted(data_dir.glob("*clonearea.csv")):
        data_csv = pd.read_csv(path)
        area_dataset.append({"label": path.name,
                             "area": data_csv["colony_area"].to_numpy(dtype=float)[:150]})
    return area_dataset


def configure_dispersion_sim(sim, initial_guess):
    sim.params["diffusion_sensitive"] = float(initial_guess[0])
    sim.params["diffusion_resistant"] = float(initial_guess[0])
    sim.params["uptake_rate"] = float(initial_guess[1])
    sim.params["diffusion_nutrients"] = float(initial_guess[2])
    sim.params["mutations_active"] = False


def _area_from_state(sensitive, resistant, mutation_scaling):
    sen_thresholded = np.where(sensitive > (1.0 / mutation_scaling), 1, 0)
    res_thresholded = np.where(resistant > (1.0 / mutation_scaling), 1, 0)
    total_array = sen_thresholded + res_thresholded
    return float(np.count_nonzero(total_array)) * (1376 / 100) ** 2


def simulate_area_history_streaming(initial_guess, simulation_param_overrides=None):
    """
    Simulate the dispersion model while retaining only the scalar colony area trace.

    This preserves the exact update order and thresholding logic from the old
    full-history implementation, but avoids storing every 200x200 state snapshot.
    """
    sim = cr.DiffusionModel2D()
    treatment_times = np.zeros(1980, dtype=bool)
    configure_dispersion_sim(sim, np.asarray(initial_guess, dtype=float))
    if simulation_param_overrides is not None:
        sim.params.update(simulation_param_overrides)
    sim.params["total_time"] = int(len(treatment_times))
    sim.treatment_times = np.asarray(treatment_times, dtype=bool)
    sim.random_seed = 1
    sim.set_random_seed()

    nutrients, sensitive, resistant = sim.get_initial_state()
    mutation_scaling = float(sim.params["mutation_scaling"])
    area_history = np.empty(len(treatment_times), dtype=float)
    area_history[0] = _area_from_state(sensitive, resistant, mutation_scaling)

    for timer in range(1, len(treatment_times)):
        nutrients, sensitive, resistant = sim.update(timer, nutrients, sensitive, resistant)
        area_history[timer] = _area_from_state(sensitive, resistant, mutation_scaling)

    return area_history, mutation_scaling


def simulate_state_at_step(initial_guess, treatment_times, snapshot_step, simulation_param_overrides=None):
    if snapshot_step < 0:
        raise ValueError("snapshot_step must be non-negative.")
    if snapshot_step >= len(treatment_times):
        raise ValueError(
            f"snapshot_step={snapshot_step} is outside the treatment_times array of length {len(treatment_times)}.")

    sim = cr.DiffusionModel2D()
    configure_dispersion_sim(sim, np.asarray(initial_guess, dtype=float))
    if simulation_param_overrides is not None:
        sim.params.update(simulation_param_overrides)
    sim.params["total_time"] = int(len(treatment_times))
    sim.treatment_times = np.asarray(treatment_times, dtype=bool)
    sim.random_seed = 1
    sim.set_random_seed()

    nutrients, sensitive, resistant = sim.get_initial_state()
    if snapshot_step == 0:
        return (nutrients.copy(), sensitive.copy(), resistant.copy(), float(sim.params["mutation_scaling"]), float(sim.treatment_efficacy), bool(sim.prev_treatment))

    for timer in range(1, snapshot_step + 1):
        nutrients, sensitive, resistant = sim.update(timer, nutrients, sensitive, resistant)

    return (nutrients.copy(), sensitive.copy(), resistant.copy(), float(sim.params["mutation_scaling"]), float(sim.treatment_efficacy), bool(sim.prev_treatment))


def get_start_point(area_exp, area_sim):
    for idx in range(len(area_sim)):
        if area_sim[idx] >= area_exp[0]:
            return idx
    return 0


def get_nutrient_data(start_point, regrowth_calibration):
    position_of_mutant_in_sim_pixel = [2, 4, 6, 8]
    sim_step_of_growth = regrowth_offsets_to_detection_frames(position_of_mutant_in_sim_pixel, start_point=start_point, regrowth_calibration=regrowth_calibration)
    return position_of_mutant_in_sim_pixel, sim_step_of_growth


def run_sim_for_nutrient_diffusion(initial_guess, start_point, params_yaml, regrowth_calibration, simulation_param_overrides=None):
    first_start = int(params_yaml["treatment_start"]) + start_point
    pretreatment = np.zeros(first_start + 1, dtype=bool)
    pretreatment[first_start:] = True

    nut_start, sen_start, _, _, treatment_efficacy, prev_treatment = simulate_state_at_step(initial_guess, pretreatment, snapshot_step=first_start, simulation_param_overrides=simulation_param_overrides)

    sim = cr.DiffusionModel2D()
    configure_dispersion_sim(sim, np.asarray(initial_guess, dtype=float))
    if simulation_param_overrides is not None:
        sim.params.update(simulation_param_overrides)
    sim.params["total_time"] = int(2400 + start_point)
    sim.treatment_times = np.ones(3500, dtype=bool)
    sim.treatment_efficacy = float(treatment_efficacy)
    sim.prev_treatment = bool(prev_treatment)
    sim.random_seed = 1
    sim.set_random_seed()

    positions, exp_times = get_nutrient_data(start_point, regrowth_calibration)
    index = np.where(sen_start[100, :] >= 1.0 / sim.params["mutation_scaling"])[0]
    if len(index) == 0:
        return exp_times, np.array([0.0], dtype=float)

    res_start = np.zeros_like(sen_start)
    res_start[100, index[0] + int(positions[0])] = 1.0 / sim.params["mutation_scaling"]
    res_start[100, index[-1] - int(positions[1])] = 1.0 / sim.params["mutation_scaling"]
    res_start[index[0] + int(positions[2]), 100] = 1.0 / sim.params["mutation_scaling"]
    res_start[index[-1] - int(positions[3]), 100] = 1.0 / sim.params["mutation_scaling"]

    pos_to_check = [(100, index[0] + int(positions[0]) - 1), (100, index[-1] - int(positions[1]) + 1), (index[0] + int(positions[2]) - 1, 100), (index[-1] - int(positions[3]) + 1, 100)]
    triggered = [False, False, False, False]
    sim_times = []

    for timer in range(1, 3500):
        nut_start, sen_start, res_start = sim.update(timer, nut_start, sen_start, res_start)
        for pos_idx, (row, col) in enumerate(pos_to_check):
            if not triggered[pos_idx] and res_start[row, col] > 1.0 / sim.params["mutation_scaling"]:
                triggered[pos_idx] = True
                sim_times.append(timer + int(params_yaml["treatment_start"]) + start_point)
        if all(triggered):
            break

    if len(sim_times) < len(exp_times):
        return exp_times, np.array([0.0], dtype=float)

    return exp_times, np.array(sim_times, dtype=float)


def dispersion_error_function(area_exp, area_sim, exp_times, sim_times):
    if area_exp.shape[1] * 10 != area_sim.shape[1]:
        return 2.0  # Experimental and simulated area lengths do not match. Returning penalty value 2.
    if sim_times[0] == 0:
        return 2.0  # No resistant cells found in the nutrient timing simulation. Returning penalty value 2.

    area_nrmse_list = []
    for idx in range(len(area_exp)):
        area_mse = np.mean((area_sim[idx, ::10] - area_exp[idx]) ** 2)
        area_rmse = np.sqrt(area_mse)
        area_nrmse = area_rmse / np.mean(area_exp[idx])
        area_nrmse_list.append(area_nrmse)

    radius_nrmse = float(np.mean(area_nrmse_list))
    times_mse = np.mean((sim_times - exp_times) ** 2)
    times_rmse = np.sqrt(times_mse)
    times_nrmse = float(times_rmse / np.mean(exp_times))
    return radius_nrmse + times_nrmse


def evaluate_dispersion_objective(initial_guess, area_observations, params_yaml, regrowth_calibration, simulation_param_overrides=None):
    if np.any(np.asarray(initial_guess, dtype=float) <= 0):
        return np.inf, {"start_point": np.nan}

    area_sim_full, _ = simulate_area_history_streaming(initial_guess, simulation_param_overrides=simulation_param_overrides)
    start_point = get_start_point(area_observations[0], area_sim_full)
    exp_times, sim_times = run_sim_for_nutrient_diffusion(initial_guess, start_point, params_yaml, regrowth_calibration, simulation_param_overrides=simulation_param_overrides)

    area_exp_array = np.stack(area_observations)
    area_sim_list = []
    for area_exp in area_observations:
        area_slice = area_sim_full[start_point : (len(area_exp) * 10 + start_point)]
        area_sim_list.append(area_slice)

    if any(len(area_slice) != len(area_observations[0]) * 10 for area_slice in area_sim_list):
        return 2.0, {"start_point": start_point, "exp_times": exp_times, "sim_times": sim_times}

    area_sim_array = np.stack(area_sim_list)
    error = dispersion_error_function(area_exp_array, area_sim_array, exp_times, sim_times)
    return error, {"start_point": start_point, "exp_times": exp_times, "sim_times": sim_times}


def fit_dispersion_dataset(area_observations, initial_guess, params_yaml, regrowth_calibration, maxiter, simulation_param_overrides=None):
    def objective(x):
        error, _ = evaluate_dispersion_objective(np.asarray(x, dtype=float), area_observations, params_yaml, regrowth_calibration, simulation_param_overrides=simulation_param_overrides)
        return float(error)

    result = opt.minimize(objective, np.asarray(initial_guess, dtype=float), method="Nelder-Mead", options={"disp": False, "maxiter": int(maxiter)})

    final_error, final_metadata = evaluate_dispersion_objective(np.asarray(result.x, dtype=float), area_observations, params_yaml, regrowth_calibration, simulation_param_overrides=simulation_param_overrides)

    result.fun = final_error
    return FitOutcome(result=result, metadata=final_metadata)


def evaluate_dispersion_prediction(initial_guess, held_out_area, start_point, params_yaml, regrowth_calibration):
    area_sim_full, _ = simulate_area_history_streaming(initial_guess)
    area_slice = area_sim_full[start_point : (len(held_out_area) * 10 + start_point)]

    metrics = {"test_area_mae": np.nan,
               "test_area_rmse": np.nan,
               "test_area_nrmse": np.nan,
               "test_area_r2": np.nan,
               "shared_nutrient_rmse": np.nan,
               "shared_nutrient_nrmse": np.nan}

    if len(area_slice) != len(held_out_area) * 10:
        return metrics

    predicted_area = area_slice[::10]
    residuals = predicted_area - held_out_area
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    metrics["test_area_mae"] = mae
    metrics["test_area_rmse"] = rmse
    metrics["test_area_nrmse"] = float(rmse / np.mean(held_out_area))

    total_variance = float(np.sum((held_out_area - np.mean(held_out_area)) ** 2))
    if total_variance > 0:
        metrics["test_area_r2"] = float(1.0 - (np.sum(residuals**2) / total_variance))

    exp_times, sim_times = run_sim_for_nutrient_diffusion(initial_guess, start_point, params_yaml, regrowth_calibration)
    if sim_times[0] != 0 and len(sim_times) == len(exp_times):
        nutrient_rmse = float(np.sqrt(np.mean((sim_times - exp_times) ** 2)))
        metrics["shared_nutrient_rmse"] = nutrient_rmse
        metrics["shared_nutrient_nrmse"] = float(nutrient_rmse / np.mean(exp_times))

    return metrics
