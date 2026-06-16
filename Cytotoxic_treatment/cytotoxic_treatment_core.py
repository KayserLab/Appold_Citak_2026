import copy
import os
import sys
import numpy as np
import yaml
from source import core as cr


def find_project_root(current_dir, marker_file):
    current_dir = os.path.abspath(current_dir)
    while current_dir != os.path.dirname(current_dir):
        if marker_file in os.listdir(current_dir):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return None


def resolve_path(project_root, path):
    if os.path.isabs(path):
        return path
    return os.path.join(project_root, path)


def load_params(params_path):
    with open(params_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def calc_initial_area(params):
    return (eval(params["sim_pixel_to_exp_pixel_factor"]) ** 2) * (8.648**2) / 1e6


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


class CytotoxicTreatmentModel(cr.DiffusionModel2D):
    def __init__(self, treat_on, treat_off, params=None, max_sensitive_death_rate=None):
        super().__init__()

        if params is not None:
            self.params = copy.deepcopy(params)

        self.treat_on = int(treat_on)
        self.treat_off = int(treat_off)
        self.max_sensitive_death_rate = float(max_sensitive_death_rate)

        if self.max_sensitive_death_rate <= 0.0:
            raise ValueError("max_sensitive_death_rate must be positive.")

        self.treatment_temp = 0
        self.cytotoxic_death_rate = 0.0
        self.save_death_rate = [0.0]
        self.save_normalized_death_rate = [0.0]
        self.save_size = [calc_initial_area(self.params)]
        self.save_ratio = [0.0]
        self.prev_treatment = False
        self.extra_steps_remaining = 0
        self.lag_steps_remaining = 0

        self.treatment_times, self.treatment_starts, self.treatment_ends = build_treatment_schedule(self.treat_on, self.treat_off, self.params)

        self.params["treatment_on_duration"] = int(self.treat_on)
        self.params["treatment_off_duration"] = int(self.treat_off)
        self.params["cytotoxic_treatment_mode"] = "homogeneous_sensitive_kill"
        self.params["max_sensitive_death_rate"] = self.max_sensitive_death_rate

    def _update_cytotoxic_death_rate(self, timer):
        delta_t = float(self.params["delta_t"])
        treatment_delay = float(self.params["treatment_delay"])
        release_delay = float(self.params["release_delay"])
        lag_steps = int(self.params["lag_steps"])

        current_treatment = bool(self.treatment_times[timer])
        ramp_up = delta_t * self.max_sensitive_death_rate / treatment_delay
        ramp_down = delta_t * self.max_sensitive_death_rate / release_delay

        if self.prev_treatment and not current_treatment:
            self.extra_steps_remaining = int(self.params["overshoot_steps"])
            self.lag_steps_remaining = lag_steps

        if current_treatment:
            self.extra_steps_remaining = 0
            self.lag_steps_remaining = 0
            self.cytotoxic_death_rate += ramp_up
        elif self.extra_steps_remaining > 0:
            self.cytotoxic_death_rate += ramp_up
            self.extra_steps_remaining -= 1
            self.lag_steps_remaining -= 1
        elif self.lag_steps_remaining > 0:
            self.lag_steps_remaining -= 1
        else:
            self.cytotoxic_death_rate -= ramp_down

        self.cytotoxic_death_rate = float(np.clip(self.cytotoxic_death_rate, 0.0, self.max_sensitive_death_rate))
        self.save_death_rate.append(self.cytotoxic_death_rate)
        self.save_normalized_death_rate.append(self.cytotoxic_death_rate / self.max_sensitive_death_rate)
        self.prev_treatment = current_treatment

    def update(self, timer, nutrients, sensitive, resistant):
        self._update_cytotoxic_death_rate(timer)

        update_sensitive = (self.params["sensitive_growth_rate"] * nutrients * sensitive)
        update_resistant = (self.params["resistant_growth_rate"] * nutrients * resistant)

        update_sensitive_thresholded = np.where((sensitive - self.params["density_threshold"]) > 0, update_sensitive, 0.0)
        update_resistant_thresholded = np.where((resistant - self.params["density_threshold"]) > 0, update_resistant, 0.0)
        expansion_sensitive = self.params["diffusion_sensitive"] * self.apply_laplacian(update_sensitive_thresholded, mode="wrap")
        expansion_resistant = self.params["diffusion_resistant"] * self.apply_laplacian(update_resistant_thresholded, mode="wrap")

        growth_sensitive = update_sensitive + expansion_sensitive
        growth_resistant = update_resistant + expansion_resistant

        death_sensitive = self.cytotoxic_death_rate * sensitive

        nutrient_depletion_total = -self.params["uptake_rate"] * (update_sensitive + update_resistant)
        
        for _ in range(int(self.params["nutrient_diffusion_steps"])):
            depletion_nutrients = self.params["diffusion_nutrients"] * self.apply_laplacian(nutrients, mode="constant")
            depletion_nutrients = depletion_nutrients + nutrient_depletion_total
            nutrients += depletion_nutrients * (self.params["delta_t"] / self.params["nutrient_diffusion_steps"])

        sensitive += (growth_sensitive - death_sensitive) * self.params["delta_t"]
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
            raise ValueError("CytotoxicTreatmentModel.run_simulation only supports the lightweight trace-saving mode used by the cytotoxic sweep.")

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
            size = (total_count * (eval(self.params["sim_pixel_to_exp_pixel_factor"]) ** 2) * (8.648**2)) / 1e6
            self.save_size.append(size)

            sen_thresholded_ratio = np.where(sensitive > (1.0 / self.params["mutation_scaling"]), sensitive, 0.0)
            res_thresholded_ratio = np.where(resistant > (1.0 / self.params["mutation_scaling"]), resistant, 0.0)
            res_ratio = np.where(res_thresholded_ratio > sen_thresholded_ratio, 1, 0)
            self.save_ratio.append(np.count_nonzero(res_ratio) / total_count if total_count > 0 else 0.0)

        return (np.array(self.treatment_times, dtype=np.bool_), np.array(self.save_death_rate, dtype=np.float32), np.array(self.save_normalized_death_rate, dtype=np.float32), np.array(self.save_size, dtype=np.float32), np.array(self.save_ratio, dtype=np.float32))


def run_cytotoxic_treatment_simulation(treat_on, treat_off, params=None, random_seed=None, max_sensitive_death_rate=None):
    sim = CytotoxicTreatmentModel(treat_on, treat_off, params=params, max_sensitive_death_rate=max_sensitive_death_rate)

    if random_seed is not None:
        sim.random_seed = int(random_seed)

    return sim.run_simulation()
