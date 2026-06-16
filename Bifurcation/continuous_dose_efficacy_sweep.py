import copy
import functools as ft
import multiprocessing as mp
import os
import sys
import time
import numpy as np
import skimage.segmentation as seg
import torch
import tqdm
import yaml
from source import core as cr


def find_project_root(current_dir, marker_file):
    current_dir = os.path.abspath(current_dir)
    while current_dir != os.path.dirname(current_dir):
        if marker_file in os.listdir(current_dir):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return None


def get_project_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return find_project_root(current_dir, "requirements.txt") or os.getcwd()


def configure_project_import_path():
    project_root = get_project_root()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root


configure_project_import_path()


def get_continuous_sweep_config():
    return {"save_folder": os.path.join("data", "sweeps", "continuous_dose_sweep_rebuttal"),
            "full_trajectory_save_folder": os.path.join("data", "sim_data", "continuous_sweep_rebuttal"),
            "full_trajectory_replicate": 0,
            "target_efficacy_range": (0.0, 1.0, 0.01),
            "ttp_threshold": 71.0,
            "num_replicates": 20,
            "duration": None,
            "num_cpus": 24,
            "params_path": "params.yaml",
            "job_id": 0,
            "num_jobs": 1,
            "save_full_trajectories": True}


def get_nutrient_thresholds():
    return 1 / (np.exp(2) + 1), 1 / (np.exp(-2) + 1)


def resolve_path(project_root, path):
    if os.path.isabs(path):
        return path
    return os.path.join(project_root, path)


def load_params(params_path):
    with open(params_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_target_efficacy_axis(range_values):
    efficacy_min, efficacy_max, efficacy_step = [float(value) for value in range_values]

    if efficacy_step <= 0:
        raise ValueError("Target-efficacy step must be positive.")
    if efficacy_min < 0.0 or efficacy_max > 1.0:
        raise ValueError("Target efficacies must stay within [0, 1].")
    if efficacy_max < efficacy_min:
        raise ValueError("Target-efficacy max must be >= min.")

    num_steps = (efficacy_max - efficacy_min) / efficacy_step
    if not np.isclose(num_steps, round(num_steps)):
        raise ValueError("Target-efficacy range must be divisible by its step.")

    axis_values = (efficacy_min + np.arange(int(round(num_steps)) + 1, dtype=float) * efficacy_step)
    axis_values = np.round(axis_values, decimals=8)
    return axis_values.astype(np.float32)


def build_sweep_params(target_efficacy_values, num_replicates):
    sweep_params = []
    seed = 0

    for target_efficacy in target_efficacy_values:
        for replicate in range(int(num_replicates)):
            sweep_params.append((float(target_efficacy), int(replicate), int(seed)))
            seed += 1

    return sweep_params


def calc_initial_area(params):
    return (
        (eval(params["sim_pixel_to_exp_pixel_factor"]) ** 2) * (8.648**2) / 1e6
    )


def calc_px_to_mm(params):
    return float(eval(params["sim_pixel_to_exp_pixel_factor"])) * 8.648 / 1e3


def calc_nutrient_layer_positions(nutrients):
    center_row = int(nutrients.shape[0] // 2)
    scan_width = min(center_row, int(nutrients.shape[1]))
    threshold_low, threshold_high = get_nutrient_thresholds()

    layer_thickness_low = scan_width
    layer_thickness_high = scan_width

    for j in range(scan_width):
        if nutrients[center_row, j] >= threshold_low:
            layer_thickness_low -= 1
        if nutrients[center_row, j] >= threshold_high:
            layer_thickness_high -= 1

    return float(layer_thickness_low), float(layer_thickness_high)


def calc_nutrient_layer_distance_mm(nutrients, params):
    layer_low, layer_high = calc_nutrient_layer_positions(nutrients)
    return float((layer_low - layer_high) * calc_px_to_mm(params))


def calc_front_velocity(previous_sensitive, previous_resistant, sensitive, resistant, params):
    threshold = 1 / params["mutation_scaling"]
    mutation_scaling = params["mutation_scaling"]
    sensitive_diff = (sensitive - previous_sensitive) * mutation_scaling
    resistant_diff = (resistant - previous_resistant) * mutation_scaling

    sen_thresholded = np.where(previous_sensitive >= threshold, 1, 0)
    res_thresholded = np.where(previous_resistant >= threshold, 1, 0)
    total_array = sen_thresholded + res_thresholded

    total_boundary = seg.find_boundaries(total_array > 0, mode="inner")
    sensitive_boundary = seg.find_boundaries(sen_thresholded > 0, mode="inner")
    resistant_boundary = seg.find_boundaries(res_thresholded > 0, mode="inner")

    sensitive_front = total_boundary & sensitive_boundary
    resistant_front = total_boundary & resistant_boundary
    sensitive_velocity = sensitive_diff[sensitive_front].mean() if sensitive_front.any() else 0.0
    resistant_velocity = resistant_diff[resistant_front].mean() if resistant_front.any() else 0.0

    return float(sensitive_velocity), float(resistant_velocity)


def get_continuous_treatment_start(params):
    start_point = int(params["start_point"])
    treatment_start = int(params["treatment_start"]) + start_point
    return max(0, min(int(params["total_time"]), treatment_start))


def build_continuous_treatment_schedule(params):
    total_time = int(params["total_time"])
    treatment_start = get_continuous_treatment_start(params)
    treatment_times = np.zeros(total_time, dtype=np.bool_)
    treatment_times[treatment_start:] = True
    return treatment_times, treatment_start


def format_target_efficacy_value(target_efficacy):
    return f"{float(target_efficacy):.8f}"


def format_target_efficacy_label(target_efficacy):
    return format_target_efficacy_value(target_efficacy).replace(".", "")


def get_full_trajectory_save_path(full_trajectory_save_folder, target_efficacy):
    return os.path.join(
        full_trajectory_save_folder,
        f"continuous_dose_efficacy_{format_target_efficacy_label(target_efficacy)}",
    )


def init_full_trajectory_arrays(save_path, params):
    os.makedirs(save_path, exist_ok=True)

    total_time = int(params["total_time"])
    image_size = int(params["image_size"])
    state_shape = (total_time, image_size, image_size)

    return {
        "nutrients": np.lib.format.open_memmap(
            os.path.join(save_path, "nutrients.npy"),
            mode="w+",
            dtype=np.float32,
            shape=state_shape,
        ),
        "sensitive": np.lib.format.open_memmap(
            os.path.join(save_path, "sensitive.npy"),
            mode="w+",
            dtype=np.float32,
            shape=state_shape,
        ),
        "resistant": np.lib.format.open_memmap(
            os.path.join(save_path, "resistant.npy"),
            mode="w+",
            dtype=np.float32,
            shape=state_shape,
        ),
    }


def save_full_trajectory_metadata(save_path, params, treatment_times, treatment_efficacy):
    project_root = get_project_root()
    np.save(
        os.path.join(save_path, "treatment_times.npy"),
        np.asarray(treatment_times, dtype=np.bool_),
    )
    np.save(
        os.path.join(save_path, "treatment_efficacy.npy"),
        np.asarray(treatment_efficacy, dtype=np.float32),
    )

    saved_params = copy.deepcopy(params)
    saved_params["save_in_core"] = True
    saved_params["save_results"] = os.path.relpath(save_path, project_root)
    torch.save(saved_params, os.path.join(save_path, "params.pth"))


class CappedContinuousDoseModel(cr.DiffusionModel2D):
    def __init__(self, target_treatment_efficacy, params=None):
        super().__init__()

        if params is not None:
            self.params = copy.deepcopy(params)

        self.target_treatment_efficacy = float(
            np.clip(target_treatment_efficacy, 0.0, 1.0)
        )
        self.treatment_efficacy = 0.0
        self.treatment_temp = 0
        self.save_treat_efficacy = [0.0]
        self.save_size = [calc_initial_area(self.params)]
        self.save_ratio = [0.0]
        self.save_sensitive_front_velocity = []
        self.save_resistant_front_velocity = []
        self.prev_treatment = False
        self.extra_steps_remaining = 0
        self.lag_steps_remaining = 0

    def _clamp_treatment_efficacy(self):
        if self.treatment_efficacy > self.target_treatment_efficacy:
            self.treatment_efficacy = self.target_treatment_efficacy
        elif self.treatment_efficacy < 0.0:
            self.treatment_efficacy = 0.0

    def update(self, timer, nutrients, sensitive, resistant):
        # Keep the capped continuous-dose logic local to this script so
        # source/core.py remains unchanged.
        delta_t = self.params["delta_t"]
        treatment_delay = self.params["treatment_delay"]
        release_delay = self.params["release_delay"]
        lag_steps = self.params["lag_steps"]

        current_treatment = bool(self.treatment_times[timer])

        if self.prev_treatment and not current_treatment:
            self.extra_steps_remaining = self.params["overshoot_steps"]
            self.lag_steps_remaining = lag_steps

        if current_treatment:
            self.extra_steps_remaining = 0
            self.lag_steps_remaining = 0
            self.treatment_efficacy += delta_t / treatment_delay
        elif self.extra_steps_remaining > 0:
            self.treatment_efficacy += delta_t / treatment_delay
            self.extra_steps_remaining -= 1
            self.lag_steps_remaining -= 1
        elif self.lag_steps_remaining > 0:
            self.lag_steps_remaining -= 1
        else:
            self.treatment_efficacy -= delta_t / release_delay

        self._clamp_treatment_efficacy()

        self.save_treat_efficacy.append(self.treatment_efficacy)
        self.prev_treatment = current_treatment

        update_sensitive = (
            self.params["sensitive_growth_rate"]
            * nutrients
            * sensitive
            * (1 - self.treatment_efficacy)
        )
        update_resistant = (
            self.params["resistant_growth_rate"] * nutrients * resistant
        )

        update_sensitive_thresholded = np.where(
            (sensitive - self.params["density_threshold"]) > 0,
            update_sensitive,
            0,
        )
        update_resistant_thresholded = np.where(
            (resistant - self.params["density_threshold"]) > 0,
            update_resistant,
            0,
        )
        expansion_sensitive = self.params["diffusion_sensitive"] * self.apply_laplacian(
            update_sensitive_thresholded,
            mode="wrap",
        )
        expansion_resistant = self.params["diffusion_resistant"] * self.apply_laplacian(
            update_resistant_thresholded,
            mode="wrap",
        )

        growth_sensitive = update_sensitive + expansion_sensitive
        growth_resistant = update_resistant + expansion_resistant

        nutrient_depletion_total = -self.params["uptake_rate"] * (
            update_sensitive + update_resistant
        )
        for _ in range(self.params["nutrient_diffusion_steps"]):
            depletion_nutrients = self.params["diffusion_nutrients"] * self.apply_laplacian(
                nutrients,
                mode="constant",
            )
            depletion_nutrients = depletion_nutrients + nutrient_depletion_total
            nutrients += depletion_nutrients * (
                self.params["delta_t"] / self.params["nutrient_diffusion_steps"]
            )

        sensitive += growth_sensitive * self.params["delta_t"]
        resistant += growth_resistant * self.params["delta_t"]

        if self.params["mutations_active"]:
            unscaled_mutation_array = self.rng.poisson(
                update_sensitive * float(self.params["mutation_rate"]),
                size=sensitive.shape,
            )
            self.mutation_count += unscaled_mutation_array.sum()
            mutation_array = unscaled_mutation_array / float(
                self.params["mutation_scaling"]
            )
            resistant += mutation_array
            sensitive -= mutation_array

        np.clip(nutrients, 0, None, out=nutrients)
        np.clip(sensitive, 0, None, out=sensitive)
        np.clip(resistant, 0, None, out=resistant)

        return nutrients, sensitive, resistant

    def run_simulation(self, save_without_asking=False, stop_at_fullstop=False, stop_with_size=False, full_trajectory_save_path=None):
        if self.params["save_in_core"] or self.params["return_all"]:
            raise ValueError(
                "CappedContinuousDoseModel.run_simulation only supports the lightweight "
                "sweep mode used in this script."
            )

        self.set_random_seed()
        nutrients, sensitive, resistant = self.get_initial_state()
        full_trajectory_arrays = None
        if full_trajectory_save_path is not None:
            full_trajectory_arrays = init_full_trajectory_arrays(
                full_trajectory_save_path,
                self.params,
            )
            full_trajectory_arrays["nutrients"][0] = nutrients
            full_trajectory_arrays["sensitive"][0] = sensitive
            full_trajectory_arrays["resistant"][0] = resistant

        ttp_nutrient_layer_distance = np.nan
        endpoint_nutrient_layer_distance = np.nan
        ttp_step = None
        ttp_threshold = get_continuous_sweep_config()["ttp_threshold"]

        counter = 0
        for i in tqdm.tqdm(range(1, self.params["total_time"])):
            previous_sensitive = sensitive.copy()
            previous_resistant = resistant.copy()
            nutrients, sensitive, resistant = self.update(i, nutrients, sensitive, resistant)

            if self.params["set_mut_pos"] and not self.params["mutations_active"]:
                if i == self.params["mutation_pos_time"]:
                    sensitive[self.params["mutation_position"][0], self.params["mutation_position"][1]] -= 1 / self.params["mutation_scaling"]
                    resistant[self.params["mutation_position"][0], self.params["mutation_position"][1]] += 1 / self.params["mutation_scaling"]

                    sensitive[self.params["mutation_position"][1], self.params["mutation_position"][0]] -= 1 / self.params["mutation_scaling"]
                    resistant[self.params["mutation_position"][1], self.params["mutation_position"][0]] += 1 / self.params["mutation_scaling"]

                    sensitive[-self.params["mutation_position"][0], -self.params["mutation_position"][1]] -= 1 / self.params["mutation_scaling"]
                    resistant[-self.params["mutation_position"][0], -self.params["mutation_position"][1]] += 1 / self.params["mutation_scaling"]

                    sensitive[-self.params["mutation_position"][1], -self.params["mutation_position"][0]] -= 1 / self.params["mutation_scaling"]
                    resistant[-self.params["mutation_position"][1], -self.params["mutation_position"][0]] += 1 / self.params["mutation_scaling"]

            sensitive_front_velocity, resistant_front_velocity = calc_front_velocity(previous_sensitive, previous_resistant, sensitive, resistant, self.params)
            self.save_sensitive_front_velocity.append(sensitive_front_velocity)
            self.save_resistant_front_velocity.append(resistant_front_velocity)

            sen_thresholded = np.where(
                sensitive > (1 / self.params["mutation_scaling"]),
                1,
                0,
            )
            res_thresholded = np.where(
                resistant > (1 / self.params["mutation_scaling"]),
                1,
                0,
            )

            total_array = sen_thresholded + res_thresholded
            total_count = np.count_nonzero(total_array)
            size = (
                total_count
                * (eval(self.params["sim_pixel_to_exp_pixel_factor"]) ** 2)
                * (8.648**2)
            ) / 1e6
            self.save_size.append(size)

            sen_thresholded_ratio = np.where(
                sensitive > (1 / self.params["mutation_scaling"]),
                sensitive,
                0,
            )
            res_thresholded_ratio = np.where(
                resistant > (1 / self.params["mutation_scaling"]),
                resistant,
                0,
            )
            res_ratio = np.where(res_thresholded_ratio > sen_thresholded_ratio, 1, 0)
            self.save_ratio.append(
                np.count_nonzero(res_ratio) / total_count if total_count > 0 else 0
            )

            if total_count < 1:
                print(f"Total count is: {total_count} at timestep: {i} with treat: {self.treatment_times}")

            if size >= ttp_threshold and stop_with_size and counter == 0:
                ttp_nutrient_layer_distance = calc_nutrient_layer_distance_mm(
                    nutrients,
                    self.params,
                )
                ttp_step = int(i)
                counter += 1

            if stop_at_fullstop and self.treatment_efficacy >= self.target_treatment_efficacy:
                break

            if full_trajectory_arrays is not None:
                full_trajectory_arrays["nutrients"][i] = nutrients
                full_trajectory_arrays["sensitive"][i] = sensitive
                full_trajectory_arrays["resistant"][i] = resistant

        endpoint_nutrient_layer_distance = calc_nutrient_layer_distance_mm(
            nutrients,
            self.params,
        )

        if ttp_step is None:
            ttp_step = len(self.save_size) - 1
            ttp_nutrient_layer_distance = endpoint_nutrient_layer_distance

        if full_trajectory_arrays is not None:
            for array in full_trajectory_arrays.values():
                array.flush()
            save_full_trajectory_metadata(
                full_trajectory_save_path,
                self.params,
                self.treatment_times,
                self.save_treat_efficacy,
            )

        return (
            np.array(self.treatment_times),
            np.array(self.save_treat_efficacy, dtype=np.float32),
            np.array(self.save_size, dtype=np.float32),
            np.array(self.save_ratio, dtype=np.float32),
            np.array(self.save_sensitive_front_velocity, dtype=np.float32),
            np.array(self.save_resistant_front_velocity, dtype=np.float32),
            float(ttp_nutrient_layer_distance),
            float(endpoint_nutrient_layer_distance),
            int(ttp_step),
        )


def init_memmaps(save_folder, num_sim, total_time):
    os.makedirs(save_folder, exist_ok=True)

    efficacy = np.memmap(
        os.path.join(save_folder, "efficacy.dat"),
        dtype=np.float32,
        mode="w+",
        shape=(num_sim, total_time),
    )
    efficacy[:] = np.nan
    efficacy.flush()

    treatment_times = np.memmap(
        os.path.join(save_folder, "treatment_times.dat"),
        dtype=np.bool_,
        mode="w+",
        shape=(num_sim, total_time),
    )
    treatment_times[:] = False
    treatment_times.flush()

    size = np.memmap(
        os.path.join(save_folder, "size.dat"),
        dtype=np.float32,
        mode="w+",
        shape=(num_sim, total_time),
    )
    size[:] = np.nan
    size.flush()

    ratio = np.memmap(
        os.path.join(save_folder, "ratio.dat"),
        dtype=np.float32,
        mode="w+",
        shape=(num_sim, total_time),
    )
    ratio[:] = np.nan
    ratio.flush()

    sensitive_front_velocity = np.memmap(
        os.path.join(save_folder, "sensitive_front_velocity.dat"),
        dtype=np.float32,
        mode="w+",
        shape=(num_sim, total_time - 1),
    )
    sensitive_front_velocity[:] = np.nan
    sensitive_front_velocity.flush()

    resistant_front_velocity = np.memmap(
        os.path.join(save_folder, "resistant_front_velocity.dat"),
        dtype=np.float32,
        mode="w+",
        shape=(num_sim, total_time - 1),
    )
    resistant_front_velocity[:] = np.nan
    resistant_front_velocity.flush()

    nutrient_layer_distance_ttp = np.memmap(
        os.path.join(save_folder, "nutrient_layer_distance_ttp.dat"),
        dtype=np.float32,
        mode="w+",
        shape=(num_sim,),
    )
    nutrient_layer_distance_ttp[:] = np.nan
    nutrient_layer_distance_ttp.flush()

    nutrient_layer_distance_endpoint = np.memmap(
        os.path.join(save_folder, "nutrient_layer_distance_endpoint.dat"),
        dtype=np.float32,
        mode="w+",
        shape=(num_sim,),
    )
    nutrient_layer_distance_endpoint[:] = np.nan
    nutrient_layer_distance_endpoint.flush()

    status = np.memmap(
        os.path.join(save_folder, "status.dat"),
        dtype=np.bool_,
        mode="w+",
        shape=(num_sim,),
    )
    status[:] = False
    status.flush()


def write_metadata(save_folder, params_path, params, target_efficacy_values, num_replicates, sweep_params, *, save_full_trajectories, full_trajectory_save_folder, full_trajectory_replicate):
    if save_full_trajectories:
        os.makedirs(full_trajectory_save_folder, exist_ok=True)
    project_root = get_project_root()
    threshold_low, threshold_high = get_nutrient_thresholds()
    ttp_threshold = get_continuous_sweep_config()["ttp_threshold"]

    np.save(
        os.path.join(save_folder, "params_list.npy"),
        np.asarray(sweep_params, dtype=np.float32),
    )
    np.savez(
        os.path.join(save_folder, "sweep_axes.npz"),
        target_efficacy=np.asarray(target_efficacy_values, dtype=np.float32),
    )
    torch.save(sweep_params, os.path.join(save_folder, "params_list.pth"))
    torch.save(params, os.path.join(save_folder, "params.pth"))

    metadata = {
        "params_path": params_path,
        "total_time": int(params["total_time"]),
        "num_replicates": int(num_replicates),
        "mutations_active": bool(params["mutations_active"]),
        "mutation_rate": float(params["mutation_rate"]),
        "start_point": int(params["start_point"]),
        "start_point_used_for_schedule": int(params["start_point"]),
        "treatment_start": int(params["treatment_start"]),
        "continuous_treatment_start_step": int(get_continuous_treatment_start(params)),
        "treatment_delay": float(params["treatment_delay"]),
        "ttp_size_threshold": float(ttp_threshold),
        "nutrient_layer_threshold_low": float(threshold_low),
        "nutrient_layer_threshold_high": float(threshold_high),
        "nutrient_layer_distance_unit": "mm",
        "nutrient_layer_distance_definition": (
            "Distance is lambda_low - lambda_high along the center row, using the "
            "same nutrient thresholds and first-half-row scan as the kymograph scripts."
        ),
        "front_velocity_unit": "cell-equivalents/front-pixel/step",
        "front_velocity_definition": (
            "Average mutation-scaled step-to-step density change on total-boundary "
            "pixels that are also sensitive or resistant boundary pixels, matching "
            "Figure_3/panel_g_j/front_velocity.py. The saved files contain the raw "
            "per-step values; multiply by steps_per_hour for 1/h."
        ),
        "sweep_axis": "target continuous treatment efficacy cap",
        "target_efficacy_min": float(target_efficacy_values[0]),
        "target_efficacy_max": float(target_efficacy_values[-1]),
        "target_efficacy_step": (
            float(target_efficacy_values[1] - target_efficacy_values[0])
            if len(target_efficacy_values) > 1
            else 0.0
        ),
        "schedule_description": (
            "Treatment is OFF before treatment_start + start_point and ON afterwards. "
            "Efficacy ramps with treatment_delay until it reaches the swept target "
            "efficacy cap, then stays constant for the rest of the simulation."
        ),
        "full_trajectories_saved": bool(save_full_trajectories),
        "full_trajectory_replicate": (
            int(full_trajectory_replicate) if save_full_trajectories else None
        ),
        "full_trajectory_root": (
            os.path.relpath(full_trajectory_save_folder, project_root)
            if save_full_trajectories
            else None
        ),
        "full_trajectory_directory_pattern": (
            f"{os.path.relpath(full_trajectory_save_folder, project_root)}/"
            "continuous_dose_efficacy_{target_efficacy_no_dot}"
            if save_full_trajectories
            else None
        ),
        "full_trajectory_naming_example": (
            f"continuous_dose_efficacy_{format_target_efficacy_label(target_efficacy_values[0])}"
            if save_full_trajectories
            else None
        ),
        "full_trajectory_files": (
            [
                "nutrients.npy",
                "sensitive.npy",
                "resistant.npy",
                "treatment_times.npy",
                "treatment_efficacy.npy",
                "params.pth",
            ]
            if save_full_trajectories
            else []
        ),
        "saved_outputs": [
            "efficacy.dat",
            "treatment_times.dat",
            "size.dat",
            "ratio.dat",
            "sensitive_front_velocity.dat",
            "resistant_front_velocity.dat",
            "nutrient_layer_distance_ttp.dat",
            "nutrient_layer_distance_endpoint.dat",
            (
                os.path.relpath(full_trajectory_save_folder, project_root)
                if save_full_trajectories
                else None
            ),
        ],
    }
    metadata["saved_outputs"] = [
        output for output in metadata["saved_outputs"] if output is not None
    ]
    with open(os.path.join(save_folder, "metadata.yaml"), "w", encoding="utf-8") as file:
        yaml.safe_dump(metadata, file, sort_keys=False)


def worker(item, params=None, save_folder=None, num_sim=None, save_full_trajectories=False, full_trajectory_save_folder=None, full_trajectory_replicate=0):
    if full_trajectory_save_folder is None:
        full_trajectory_save_folder = get_continuous_sweep_config()["full_trajectory_save_folder"]

    idx, sweep_params = item
    target_efficacy, replicate, seed = sweep_params

    sim = CappedContinuousDoseModel(target_efficacy, params=params)
    sim.random_seed = int(seed)
    sim.params["continuous_target_efficacy"] = float(target_efficacy)
    sim.params["replicate"] = int(replicate)
    sim.params["save_folder"] = save_folder

    sim.treatment_times, treatment_start = build_continuous_treatment_schedule(sim.params)
    sim.params["treatment_on_duration"] = max(
        0,
        int(sim.params["total_time"]) - int(treatment_start),
    )
    full_trajectory_save_path = None
    if save_full_trajectories and int(replicate) == int(full_trajectory_replicate):
        full_trajectory_save_path = get_full_trajectory_save_path(
            full_trajectory_save_folder,
            target_efficacy,
        )

    (
        treatment_times,
        treatment_efficacy,
        sizes,
        ratios,
        sensitive_front_velocity,
        resistant_front_velocity,
        nutrient_layer_distance_ttp,
        nutrient_layer_distance_endpoint,
        _,
    ) = sim.run_simulation(
        save_without_asking=True,
        stop_with_size=True,
        full_trajectory_save_path=full_trajectory_save_path,
    )

    total_time = int(sim.params["total_time"])

    efficacy_mmap = np.memmap(
        os.path.join(save_folder, "efficacy.dat"),
        dtype=np.float32,
        mode="r+",
        shape=(num_sim, total_time),
    )
    efficacy_mmap[idx] = treatment_efficacy
    efficacy_mmap.flush()

    treat_times_mmap = np.memmap(
        os.path.join(save_folder, "treatment_times.dat"),
        dtype=np.bool_,
        mode="r+",
        shape=(num_sim, total_time),
    )
    treat_times_mmap[idx] = treatment_times
    treat_times_mmap.flush()

    size_mmap = np.memmap(
        os.path.join(save_folder, "size.dat"),
        dtype=np.float32,
        mode="r+",
        shape=(num_sim, total_time),
    )
    size_mmap[idx] = sizes
    size_mmap.flush()

    ratio_mmap = np.memmap(
        os.path.join(save_folder, "ratio.dat"),
        dtype=np.float32,
        mode="r+",
        shape=(num_sim, total_time),
    )
    ratio_mmap[idx] = ratios
    ratio_mmap.flush()

    sensitive_front_velocity_mmap = np.memmap(
        os.path.join(save_folder, "sensitive_front_velocity.dat"),
        dtype=np.float32,
        mode="r+",
        shape=(num_sim, total_time - 1),
    )
    sensitive_front_velocity_mmap[idx] = sensitive_front_velocity
    sensitive_front_velocity_mmap.flush()

    resistant_front_velocity_mmap = np.memmap(
        os.path.join(save_folder, "resistant_front_velocity.dat"),
        dtype=np.float32,
        mode="r+",
        shape=(num_sim, total_time - 1),
    )
    resistant_front_velocity_mmap[idx] = resistant_front_velocity
    resistant_front_velocity_mmap.flush()

    nutrient_layer_distance_ttp_mmap = np.memmap(
        os.path.join(save_folder, "nutrient_layer_distance_ttp.dat"),
        dtype=np.float32,
        mode="r+",
        shape=(num_sim,),
    )
    nutrient_layer_distance_ttp_mmap[idx] = np.float32(nutrient_layer_distance_ttp)
    nutrient_layer_distance_ttp_mmap.flush()

    nutrient_layer_distance_endpoint_mmap = np.memmap(
        os.path.join(save_folder, "nutrient_layer_distance_endpoint.dat"),
        dtype=np.float32,
        mode="r+",
        shape=(num_sim,),
    )
    nutrient_layer_distance_endpoint_mmap[idx] = np.float32(
        nutrient_layer_distance_endpoint
    )
    nutrient_layer_distance_endpoint_mmap.flush()

    status = np.memmap(
        os.path.join(save_folder, "status.dat"),
        dtype=np.bool_,
        mode="r+",
        shape=(num_sim,),
    )
    status[idx] = True
    status.flush()

    return idx


def run_continuous_dose_efficacy_sweep(target_efficacy_range=None, num_replicates=None, save_folder=None, num_cpus=None, params_path=None, duration=None, job_id=0, num_jobs=1, save_full_trajectories=True, full_trajectory_save_folder=None, full_trajectory_replicate=0):
    project_root = get_project_root()
    config = get_continuous_sweep_config()

    if target_efficacy_range is None:
        target_efficacy_range = config["target_efficacy_range"]
    if save_folder is None:
        save_folder = config["save_folder"]
    if full_trajectory_save_folder is None:
        full_trajectory_save_folder = config["full_trajectory_save_folder"]

    if params_path is None:
        params_path = os.path.join(project_root, "params.yaml")

    params_path = resolve_path(project_root, params_path)
    save_folder = resolve_path(project_root, save_folder)
    full_trajectory_save_folder = resolve_path(project_root, full_trajectory_save_folder)

    params = load_params(params_path)
    if duration is not None:
        if int(duration) <= 1:
            raise ValueError("duration must be greater than 1.")
        params["total_time"] = int(duration)
    else:
        params["total_time"] = int(params["total_time"])

    if int(params["total_time"]) <= 1:
        raise ValueError("total_time must be greater than 1.")

    if num_replicates is None:
        num_replicates = int(params["num_replicas"])
    if int(num_replicates) < 1:
        raise ValueError("num_replicates must be at least 1.")
    if int(num_jobs) < 1:
        raise ValueError("num_jobs must be at least 1.")
    if save_full_trajectories:
        if not 0 <= int(full_trajectory_replicate) < int(num_replicates):
            raise ValueError("full_trajectory_replicate must be within [0, num_replicates).")

    target_efficacy_values = build_target_efficacy_axis(target_efficacy_range)
    params_list = build_sweep_params(target_efficacy_values, num_replicates)
    num_sim = len(params_list)
    total_time = int(params["total_time"])

    print(f"Number of continuous-dose simulations: {num_sim}")
    if num_replicates > 1 and not bool(params["mutations_active"]):
        print("Running repeated deterministic replicates because mutations_active=False.")

    if job_id == 0:
        init_memmaps(save_folder, num_sim, total_time)
        write_metadata(
            save_folder,
            params_path,
            params,
            target_efficacy_values,
            num_replicates,
            params_list,
            save_full_trajectories=save_full_trajectories,
            full_trajectory_save_folder=full_trajectory_save_folder,
            full_trajectory_replicate=full_trajectory_replicate,
        )
    else:
        check_init = (
            os.path.exists(os.path.join(save_folder, "efficacy.dat"))
            and os.path.exists(os.path.join(save_folder, "treatment_times.dat"))
            and os.path.exists(os.path.join(save_folder, "size.dat"))
            and os.path.exists(os.path.join(save_folder, "ratio.dat"))
            and os.path.exists(os.path.join(save_folder, "sensitive_front_velocity.dat"))
            and os.path.exists(os.path.join(save_folder, "resistant_front_velocity.dat"))
            and os.path.exists(
                os.path.join(save_folder, "nutrient_layer_distance_ttp.dat")
            )
            and os.path.exists(
                os.path.join(save_folder, "nutrient_layer_distance_endpoint.dat")
            )
            and os.path.exists(os.path.join(save_folder, "status.dat"))
        )
        while not check_init:
            time.sleep(1)
            check_init = (
                os.path.exists(os.path.join(save_folder, "efficacy.dat"))
                and os.path.exists(os.path.join(save_folder, "treatment_times.dat"))
                and os.path.exists(os.path.join(save_folder, "size.dat"))
                and os.path.exists(os.path.join(save_folder, "ratio.dat"))
                and os.path.exists(os.path.join(save_folder, "sensitive_front_velocity.dat"))
                and os.path.exists(os.path.join(save_folder, "resistant_front_velocity.dat"))
                and os.path.exists(
                    os.path.join(save_folder, "nutrient_layer_distance_ttp.dat")
                )
                and os.path.exists(
                    os.path.join(save_folder, "nutrient_layer_distance_endpoint.dat")
                )
                and os.path.exists(os.path.join(save_folder, "status.dat"))
            )

    status = np.memmap(
        os.path.join(save_folder, "status.dat"),
        dtype=np.bool_,
        mode="r+",
        shape=(num_sim,),
    )
    undone = np.nonzero(status == False)[0]
    missing_idxs = [i for i in undone if i % int(num_jobs) == int(job_id)]
    jobs = [(i, params_list[i]) for i in missing_idxs]

    default_cpus = max(1, mp.cpu_count() - 1)
    if num_cpus is None:
        num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", default_cpus))
    num_cpus = max(1, int(num_cpus))

    worker_with_args = ft.partial(
        worker,
        params=params,
        save_folder=save_folder,
        num_sim=num_sim,
        save_full_trajectories=save_full_trajectories,
        full_trajectory_save_folder=full_trajectory_save_folder,
        full_trajectory_replicate=full_trajectory_replicate,
    )

    with mp.Pool(processes=num_cpus) as pool:
        for _ in tqdm.tqdm(pool.imap(worker_with_args, jobs), total=len(jobs)):
            pass


def main():
    config = get_continuous_sweep_config()

    run_continuous_dose_efficacy_sweep(target_efficacy_range=config["target_efficacy_range"], num_replicates=config["num_replicates"], save_folder=config["save_folder"], num_cpus=config["num_cpus"], params_path=config["params_path"], duration=config["duration"], job_id=config["job_id"], num_jobs=config["num_jobs"], save_full_trajectories=config["save_full_trajectories"], full_trajectory_save_folder=config["full_trajectory_save_folder"], full_trajectory_replicate=config["full_trajectory_replicate"])


if __name__ == "__main__":
    main()
