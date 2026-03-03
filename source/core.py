import os
import numpy as np
import tqdm
from scipy.ndimage import gaussian_filter, convolve
import yaml
import torch


class DiffusionModel2D:
    def __init__(self):
        path = os.path.join(find_project_root(os.getcwd(), 'requirements.txt'), 'params.yaml')
        with open(path, 'r') as file:
            self.params = yaml.safe_load(file)

        self.treatment_times = None
        self.treatment_efficacy = 0 
        self.treatment_temp = 0
        self.save_treat_efficacy = [0]
        self.save_size = [1*(eval(self.params['sim_pixel_to_exp_pixel_factor'])**2) * (8.648 ** 2) / 1e6]
        self.save_ratio = [0]
        self.random_seed = 1
        self.rng = None
        self.mutation_count = 0
        diag = 1/np.sqrt(2)
        center = -4-(4*diag)
        self.kernel = np.array([[diag, 1, diag],
                                [1, center, 1],
                                [diag, 1, diag]])

        self.prev_treatment = False
        self.extra_steps_remaining = 0
        self.lag_steps_remaining = 0

    def set_random_seed(self):
        self.rng = np.random.default_rng(self.random_seed)

    def apply_laplacian(self, mat, mode):
        if mode == 'wrap':
            return convolve(mat, self.kernel, mode='wrap')
        else:
            return convolve(mat, self.kernel, mode='constant', cval=1.0)

    def update(self, timer, nutrients, sensitive, resistant):
        delta_t = self.params['delta_t']
        treatment_delay = self.params['treatment_delay']
        release_delay = self.params['release_delay']

        lag_steps = self.params.get('lag_steps', 220)

        current_treatment = bool(self.treatment_times[timer])

        if self.prev_treatment and not current_treatment:
            self.extra_steps_remaining = 30
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

        if self.treatment_efficacy > 1.0:
            self.treatment_efficacy = 1.0
        elif self.treatment_efficacy < 0.0:
            self.treatment_efficacy = 0.0

        self.save_treat_efficacy.append(self.treatment_efficacy)
        self.prev_treatment = current_treatment

        update_sensitive = self.params['sensitive_growth_rate'] * nutrients * sensitive * (1 - self.treatment_efficacy)
        update_resistant = self.params['resistant_growth_rate'] * nutrients * resistant

        update_sensitive_thresholded = np.where((sensitive - self.params['density_threshold']) > 0, update_sensitive, 0)
        update_resistant_thresholded = np.where((resistant - self.params['density_threshold']) > 0, update_resistant, 0)
        expansion_sensitive = self.params['diffusion_sensitive'] * self.apply_laplacian(update_sensitive_thresholded, mode='wrap')
        expansion_resistant = self.params['diffusion_resistant'] * self.apply_laplacian(update_resistant_thresholded, mode='wrap')

        growth_sensitive = update_sensitive + expansion_sensitive
        growth_resistant = update_resistant + expansion_resistant

        nutrient_depletion_total = - self.params['uptake_rate'] * (update_sensitive + update_resistant)
        for i in range(self.params['nutrient_diffusion_steps']):
            depletion_nutrients = self.params['diffusion_nutrients'] * self.apply_laplacian(nutrients, mode='constant')
            depletion_nutrients = depletion_nutrients + nutrient_depletion_total
            nutrients += depletion_nutrients * (self.params['delta_t']/self.params['nutrient_diffusion_steps'])

        sensitive += growth_sensitive * self.params['delta_t']
        resistant += growth_resistant * self.params['delta_t']

        if self.params['mutations_active']:
            unscaled_mutation_array = self.rng.poisson(update_sensitive * float(self.params['mutation_rate']), size=sensitive.shape)
            self.mutation_count += unscaled_mutation_array.sum()
            mutation_array = unscaled_mutation_array/float(self.params['mutation_scaling'])
            resistant += mutation_array
            sensitive -= mutation_array

        np.clip(nutrients, 0, None, out=nutrients)
        np.clip(sensitive, 0, None, out=sensitive)
        np.clip(resistant, 0, None, out=resistant)

        return nutrients, sensitive, resistant

    def get_initial_state(self):
        image_size = self.params['image_size']
        nutrients = np.ones((image_size, image_size))
        sensitive = np.zeros((image_size, image_size))
        resistant = np.zeros((image_size, image_size))

        if self.params['gaussian']:
            x = np.linspace(-image_size // 2, image_size // 2, image_size)
            y = np.linspace(-image_size // 2, image_size // 2, image_size)
            grid_x, grid_y = np.meshgrid(x, y)
            sigma = self.params['gaussian_width'] / (2 * np.sqrt(2 * np.log(2)))
            sensitive = np.exp(-(grid_x ** 2 + grid_y ** 2) / (2 * sigma ** 2))
            sensitive = gaussian_filter(sensitive, sigma=sigma)
        else:
            sensitive[image_size // 2, image_size // 2] = 1/self.params['mutation_scaling']

        return nutrients, sensitive, resistant

    def run_simulation(self, save_without_asking=False, stop_at_fullstop=False, stop_with_size=False):
        self.set_random_seed()
        n_array, s_array, r_array, nut_save, sen_save, res_save = [], [], [], None, None, None
        nutrients, sensitive, resistant = self.get_initial_state()

        if self.params['save_in_core'] or self.params['return_all']:
            n_array.append(np.copy(nutrients))
            s_array.append(np.copy(sensitive))
            r_array.append(np.copy(resistant))

        counter = 0
        for i in tqdm.tqdm(range(1, self.params['total_time'])):
            nutrients, sensitive, resistant = self.update(i, nutrients, sensitive, resistant)

            if self.params['set_mut_pos'] and not self.params['mutations_active']:
                if i == self.params['mutation_pos_time']:
                    sensitive[self.params['mutation_position'][0], self.params['mutation_position'][1]] -= 1 / self.params['mutation_scaling']
                    resistant[self.params['mutation_position'][0], self.params['mutation_position'][1]] += 1 / self.params['mutation_scaling']

                    sensitive[self.params['mutation_position'][1], self.params['mutation_position'][0]] -= 1 / self.params['mutation_scaling']
                    resistant[self.params['mutation_position'][1], self.params['mutation_position'][0]] += 1 / self.params['mutation_scaling']

                    sensitive[-self.params['mutation_position'][0], -self.params['mutation_position'][1]] -= 1 / self.params['mutation_scaling']
                    resistant[-self.params['mutation_position'][0], -self.params['mutation_position'][1]] += 1 / self.params['mutation_scaling']

                    sensitive[-self.params['mutation_position'][1], -self.params['mutation_position'][0]] -= 1 / self.params['mutation_scaling']
                    resistant[-self.params['mutation_position'][1], -self.params['mutation_position'][0]] += 1 / self.params['mutation_scaling']

            sen_thresholded = np.where(sensitive > (1 / self.params['mutation_scaling']), 1, 0)
            res_thresholded = np.where(resistant > (1 / self.params['mutation_scaling']), 1, 0)

            total_array = sen_thresholded + res_thresholded
            total_count = np.count_nonzero(total_array)
            size = (total_count * (eval(self.params['sim_pixel_to_exp_pixel_factor'])**2) * (8.648 ** 2)) / 1e6
            self.save_size.append(size)

            sen_thresholded_ratio = np.where(sensitive > (1 / self.params['mutation_scaling']), sensitive, 0)
            res_thresholded_ratio = np.where(resistant > (1 / self.params['mutation_scaling']), resistant, 0)
            res_ratio = np.where(res_thresholded_ratio > sen_thresholded_ratio, 1, 0)
            self.save_ratio.append(np.count_nonzero(res_ratio) / total_count if total_count > 0 else 0)

            if total_count < 1:
                print(f'Total count is: {total_count} at timestep: {i} with treat: {self.treatment_times}')

            if size >= 71 and stop_with_size and counter == 0:
                nut_save = nutrients.copy()
                sen_save = sensitive.copy()
                res_save = resistant.copy()
                counter += 1

            if stop_at_fullstop and self.treatment_efficacy >= 1.0:
                break

            if self.params['save_in_core'] or self.params['return_all']:
                n_array.append(np.copy(nutrients))
                s_array.append(np.copy(sensitive))
                r_array.append(np.copy(resistant))

        if self.params['save_in_core']:
            save_path = os.path.join(find_project_root(os.getcwd(), 'requirements.txt'), self.params['save_results'])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            else:
                if not save_without_asking:
                    print(f"Directory {save_path} already exists. \nShould I overwrite it? (y/n)")
                    respone_flag = True
                    while respone_flag:
                        response = input()
                        if response.lower() == 'y':
                            print("Overwriting...")
                            respone_flag = False
                        elif response.lower() == 'n':
                            print("Exiting...")
                            respone_flag = False
                            exit()
                        else:
                            print("Invalid response. Please enter 'y' or 'n'")
                            respone_flag = True
                else:
                    print(f"Overwriting {save_path}...")

            np.save(f'{save_path}/nutrients.npy', np.array(n_array, dtype=np.float32))
            np.save(f'{save_path}/sensitive.npy', np.array(s_array, dtype=np.float32))
            np.save(f'{save_path}/resistant.npy', np.array(r_array, dtype=np.float32))
            np.save(f'{save_path}/treatment_times.npy', self.treatment_times)
            torch.save(self.params, f'{save_path}/params.pth')
            np.save(f'{save_path}/treatment_efficacy.npy', np.array(self.save_treat_efficacy, dtype=np.float32))
        else:
            if self.params['return_all']:
                return np.array(n_array, dtype=np.float32), np.array(s_array, dtype=np.float32), np.array(r_array, dtype=np.float32), np.array(self.treatment_times), np.array(self.save_treat_efficacy, dtype=np.float32)
            else:
                return np.array(nut_save, dtype=np.float32), np.array(sen_save, dtype=np.float32), np.array(res_save, dtype=np.float32), np.array(self.treatment_times), np.array(self.save_treat_efficacy, dtype=np.float32), np.array(self.save_size, dtype=np.float32), np.array(self.save_ratio, dtype=np.float32)


def find_project_root(current_dir, marker_file):
    current_dir = os.path.abspath(current_dir)
    while current_dir != os.path.dirname(current_dir):
        if marker_file in os.listdir(current_dir):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return None
