import copy
import os
import scipy.ndimage as ndi
import sys

import numpy as np
import yaml


def spatial_treatment_defaults():
    return {"inverse_gradient_decay_length": 12.0, 
            "gradient_mode": "exponential", 
            "gradient_floor": 0.0}


def find_project_root(current_dir, marker_file):
    current_dir = os.path.abspath(current_dir)
    while current_dir != os.path.dirname(current_dir):
        if marker_file in os.listdir(current_dir):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return None


def current_dir():
    return os.path.dirname(os.path.abspath(__file__))


def project_root():
    return find_project_root(current_dir(), "requirements.txt") or os.getcwd()


if project_root() not in sys.path:
    sys.path.insert(0, project_root())

from source import core as cr


def resolve_path(project_root, path):
    if os.path.isabs(path):
        return path
    return os.path.join(project_root, path)


def load_params(params_path):
    with open(params_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def sim_pixel_to_exp_pixel_factor(params):
    return float(eval(params["sim_pixel_to_exp_pixel_factor"]))


def calc_initial_area(params):
    pixel_factor = sim_pixel_to_exp_pixel_factor(params)
    return ((pixel_factor**2) * (8.648**2)) / 1e6


def build_treatment_schedule(treat_on, treat_off, params):
    first_start = int(params["treatment_start"])
    total_time = int(params["total_time"])
    treat_on = int(treat_on)
    treat_off = int(treat_off)

    if treat_on < 0 or treat_off < 0:
        raise ValueError("treat_on and treat_off must be >= 0.")

    treatment_times = np.zeros(total_time, dtype=np.bool_)
    treatment_length = treat_on

    if treat_off == 0:
        treatment_starts = [first_start]
        treatment_length = total_time - first_start
        if treat_on == 0:
            treatment_starts = []
    elif treat_on == 0:
        treatment_starts = []
    else:
        treatment_starts = [start for start in range(first_start, total_time, treat_off + treat_on)]

    if treatment_starts:
        treatment_ends = np.asarray(treatment_starts, dtype=np.int32) + treatment_length
    else:
        treatment_ends = np.asarray([], dtype=np.int32)

    for start in treatment_starts:
        treatment_times[start : (start + treatment_length)] = True

    return treatment_times, treatment_starts, treatment_ends


def colony_presence_threshold(params, occupancy_threshold=None):
    if occupancy_threshold is None:
        return float(1.0 / float(params["mutation_scaling"]))

    occupancy_threshold = float(occupancy_threshold)
    if occupancy_threshold < 0.0:
        raise ValueError("occupancy_threshold must be >= 0.")
    return occupancy_threshold


def compute_front_profile(occupied_mask, inverse_decay_length, mode=None, floor=None):
    if mode is None:
        mode = spatial_treatment_defaults()["gradient_mode"]
    if floor is None:
        floor = spatial_treatment_defaults()["gradient_floor"]

    inverse_decay_length = float(inverse_decay_length)
    if inverse_decay_length < 0.0:
        raise ValueError("inverse_decay_length must be positive.")

    mode = str(mode).lower()
    if mode not in {"exponential", "linear"}:
        raise ValueError("mode must be 'exponential' or 'linear'.")

    floor = float(np.clip(floor, 0.0, 1.0))
    occupied_mask = np.asarray(occupied_mask, dtype=bool)

    if not np.any(occupied_mask):
        return np.zeros(occupied_mask.shape, dtype=np.float32)

    distance_to_front = ndi.distance_transform_edt(occupied_mask)
    inward_distance = np.maximum(distance_to_front - 1.0, 0.0)

    if mode == "exponential":
        profile = np.exp(-inward_distance * inverse_decay_length)
    else:
        profile = np.clip(1.0 - (inward_distance * inverse_decay_length), 0.0, 1.0)

    if floor > 0.0:
        profile = floor + ((1.0 - floor) * profile)

    profile = np.where(occupied_mask, profile, 0.0)
    return profile.astype(np.float32, copy=False)


class SpatialGradientTreatmentModel(cr.DiffusionModel2D):
    def __init__(self, treat_on, treat_off, params=None, inverse_gradient_decay_length=None, gradient_mode=None, gradient_floor=None, occupancy_threshold=None):
        super().__init__()

        if params is not None:
            self.params = copy.deepcopy(params)

        defaults = spatial_treatment_defaults()
        if inverse_gradient_decay_length is None:
            inverse_gradient_decay_length = defaults["inverse_gradient_decay_length"]
        if gradient_mode is None:
            gradient_mode = defaults["gradient_mode"]
        if gradient_floor is None:
            gradient_floor = defaults["gradient_floor"]

        self.treat_on = int(treat_on)
        self.treat_off = int(treat_off)
        self.inverse_gradient_decay_length = float(inverse_gradient_decay_length)
        self.gradient_mode = str(gradient_mode).lower()
        self.gradient_floor = float(np.clip(gradient_floor, 0.0, 1.0))
        self.occupancy_threshold = colony_presence_threshold(self.params, occupancy_threshold=occupancy_threshold)

        if self.inverse_gradient_decay_length < 0.0:
            raise ValueError("inverse_gradient_decay_length must be positive.")
        if self.gradient_mode not in {"exponential", "linear"}:
            raise ValueError("gradient_mode must be either 'exponential' or 'linear'.")

        self.treatment_efficacy = 0.0
        self.treatment_temp = 0
        self.save_treat_efficacy = [0.0]
        self.save_mean_local_efficacy = [0.0]
        self.save_size = [calc_initial_area(self.params)]
        self.save_ratio = [0.0]
        self.prev_treatment = False
        self.extra_steps_remaining = 0
        self.lag_steps_remaining = 0

        self.treatment_times, self.treatment_starts, self.treatment_ends = build_treatment_schedule(self.treat_on, self.treat_off, self.params)

        self.params["treatment_on_duration"] = int(self.treat_on)
        self.params["treatment_off_duration"] = int(self.treat_off)
        self.params["spatial_treatment_mode"] = "front_decay_gradient"
        self.params["spatial_treatment_gradient_mode"] = self.gradient_mode
        self.params["spatial_treatment_inverse_decay_length"] = self.inverse_gradient_decay_length
        self.params["spatial_treatment_gradient_floor"] = self.gradient_floor
        self.params["spatial_treatment_occupancy_threshold"] = self.occupancy_threshold

    def _update_scalar_treatment_efficacy(self, timer):
        delta_t = float(self.params["delta_t"])
        treatment_delay = float(self.params["treatment_delay"])
        release_delay = float(self.params["release_delay"])
        lag_steps = int(self.params["lag_steps"])

        current_treatment = bool(self.treatment_times[timer])

        if self.prev_treatment and not current_treatment:
            self.extra_steps_remaining = int(self.params["overshoot_steps"])
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

        self.treatment_efficacy = float(np.clip(self.treatment_efficacy, 0.0, 1.0))
        self.save_treat_efficacy.append(self.treatment_efficacy)
        self.prev_treatment = current_treatment

    def _build_local_treatment_efficacy(self, sensitive, resistant):
        if self.treatment_efficacy <= 0.0:
            self.save_mean_local_efficacy.append(0.0)
            return np.zeros(sensitive.shape, dtype=np.float32)

        occupied_mask = (sensitive + resistant) > self.occupancy_threshold
        if not np.any(occupied_mask):
            self.save_mean_local_efficacy.append(0.0)
            return np.zeros(sensitive.shape, dtype=np.float32)

        front_profile = compute_front_profile(occupied_mask, self.inverse_gradient_decay_length, mode=self.gradient_mode, floor=self.gradient_floor)
        local_treatment_efficacy = np.clip(self.treatment_efficacy * front_profile, 0.0, 1.0).astype(np.float32, copy=False)

        self.save_mean_local_efficacy.append(float(np.mean(local_treatment_efficacy[occupied_mask])))
        return local_treatment_efficacy

    def update(self, timer, nutrients, sensitive, resistant):
        self._update_scalar_treatment_efficacy(timer)
        local_treatment_efficacy = self._build_local_treatment_efficacy(sensitive, resistant)

        update_sensitive = self.params["sensitive_growth_rate"] * nutrients * sensitive * (1.0 - local_treatment_efficacy)
        update_resistant = self.params["resistant_growth_rate"] * nutrients * resistant

        update_sensitive_thresholded = np.where((sensitive - self.params["density_threshold"]) > 0, update_sensitive, 0.0)
        update_resistant_thresholded = np.where((resistant - self.params["density_threshold"]) > 0, update_resistant, 0.0)
        expansion_sensitive = self.params["diffusion_sensitive"] * self.apply_laplacian(update_sensitive_thresholded, mode="wrap")
        expansion_resistant = self.params["diffusion_resistant"] * self.apply_laplacian(update_resistant_thresholded, mode="wrap")

        growth_sensitive = update_sensitive + expansion_sensitive
        growth_resistant = update_resistant + expansion_resistant

        nutrient_depletion_total = -self.params["uptake_rate"] * (update_sensitive + update_resistant)
        for _ in range(int(self.params["nutrient_diffusion_steps"])):
            depletion_nutrients = self.params["diffusion_nutrients"] * self.apply_laplacian(nutrients, mode="constant")
            depletion_nutrients = depletion_nutrients + nutrient_depletion_total
            nutrients += depletion_nutrients * (self.params["delta_t"] / self.params["nutrient_diffusion_steps"])

        sensitive += growth_sensitive * self.params["delta_t"]
        resistant += growth_resistant * self.params["delta_t"]

        if self.params["mutations_active"]:
            unscaled_mutation_array = self.rng.poisson(update_sensitive * float(self.params["mutation_rate"]), size=sensitive.shape)
            self.mutation_count += unscaled_mutation_array.sum()
            mutation_array = unscaled_mutation_array / float(self.params["mutation_scaling"])
            resistant += mutation_array
            sensitive -= mutation_array

        np.clip(nutrients, 0.0, None, out=nutrients)
        np.clip(sensitive, 0.0, None, out=sensitive)
        np.clip(resistant, 0.0, None, out=resistant)

        return nutrients, sensitive, resistant

    def run_simulation(self):
        if self.params["save_in_core"] or self.params["return_all"]:
            raise ValueError("SpatialGradientTreatmentModel.run_simulation only supports the lightweight trace-saving mode used by the spatial sweep.")

        self.set_random_seed()
        nutrients, sensitive, resistant = self.get_initial_state()

        for i in range(1, int(self.params["total_time"])):
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

            sen_thresholded = np.where(sensitive > (1.0 / self.params["mutation_scaling"]), 1, 0)
            res_thresholded = np.where(resistant > (1.0 / self.params["mutation_scaling"]), 1, 0)

            total_array = sen_thresholded + res_thresholded
            total_count = np.count_nonzero(total_array)
            pixel_factor = sim_pixel_to_exp_pixel_factor(self.params)
            size = (total_count * (pixel_factor**2) * (8.648**2)) / 1e6
            self.save_size.append(size)

            sen_thresholded_ratio = np.where(sensitive > (1.0 / self.params["mutation_scaling"]), sensitive, 0.0)
            res_thresholded_ratio = np.where(resistant > (1.0 / self.params["mutation_scaling"]), resistant, 0.0)
            res_ratio = np.where(res_thresholded_ratio > sen_thresholded_ratio, 1, 0)
            self.save_ratio.append(np.count_nonzero(res_ratio) / total_count if total_count > 0 else 0.0)

        return (np.array(self.treatment_times, dtype=np.bool_), np.array(self.save_treat_efficacy, dtype=np.float32), np.array(self.save_mean_local_efficacy, dtype=np.float32), np.array(self.save_size, dtype=np.float32), np.array(self.save_ratio, dtype=np.float32))


def run_spatial_treatment_simulation(treat_on, treat_off, params=None, random_seed=None, inverse_gradient_decay_length=None, gradient_mode=None, gradient_floor=None, occupancy_threshold=None):
    sim = SpatialGradientTreatmentModel(treat_on, treat_off, params=params, inverse_gradient_decay_length=inverse_gradient_decay_length, gradient_mode=gradient_mode, gradient_floor=gradient_floor, occupancy_threshold=occupancy_threshold)

    if random_seed is not None:
        sim.random_seed = int(random_seed)

    return sim.run_simulation()
