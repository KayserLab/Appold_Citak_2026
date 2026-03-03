import pandas as pd
import numpy as np
import ast
from matplotlib.path import Path
from source import core as cr
import os
import scipy.optimize as opt
import torch
import logging
import yaml


class FitMutationRate:
    def __init__(self):
        self.sim_params = self.get_params()
        self.params = {'contour_frame_index': 63,
                       'max_first_frame': 230,
                       'treatment_start': self.sim_params['start_point'] + 360} # check for start_point and adjust if needed (start_point + treatment_start)

    def get_params(self):
        path = os.path.join(self.find_project_root(os.getcwd(), 'requirements.txt'), 'params.yaml')
        with open(path, 'r') as file:
            params = yaml.safe_load(file)
        return params

    def find_project_root(self, current_dir, marker_file):
        current_dir = os.path.abspath(current_dir)
        while current_dir != os.path.dirname(current_dir):
            if marker_file in os.listdir(current_dir):
                return current_dir
            current_dir = os.path.dirname(current_dir)
        return None

    def get_mutation_number_experiment(self):
        path = '../fit_data/mutation_rate'
        wells = [x.split('_')[2] for x in os.listdir(path) if x.startswith('colony')]
        num_clones = []
        for well in wells:
            if well == 'P1':
                well = 'P1_'
            clones_1 = pd.read_csv(f'{path}/clone_data_fusion_resolved_{well}.csv')
            colony_1 = pd.read_csv(f'{path}/colony_data_{well}_with_clonearea_with_extrapolation_to_final.csv')

            frame_index = self.params["contour_frame_index"]
            frame_data = colony_1.iloc[frame_index]

            contour_str = frame_data['colony_contour']
            contours = ast.literal_eval(contour_str)
            contour = np.array(contours)
            contour_xy = contour[:, [1, 0]]
            contour_path = Path(contour_xy)

            first_frames = clones_1.groupby('particle').first().reset_index()
            max_first_frame = self.params["max_first_frame"]
            filtered_first_frames = first_frames[first_frames['frame'] <= max_first_frame]

            particle_positions = filtered_first_frames[['x', 'y']].values
            inside_mask = contour_path.contains_points(particle_positions)
            filtered_first_frames = filtered_first_frames[inside_mask]
            num_clones.append(len(filtered_first_frames))
        return np.mean(num_clones, dtype=np.float64), np.std(num_clones, dtype=np.float64, ddof=1)/ np.sqrt(len(num_clones))

    def get_mutation_number_simulation(self, mutation_rate, replicas=15):
        num_clones = []
        for i in range(replicas):
            sim = cr.DiffusionModel2D()
            sim.random_seed = i

            sim.params['total_time'] = self.params['max_first_frame'] * 10 + self.sim_params['start_point']
            sim.params['mutation_rate'] = mutation_rate

            sim.treatment_times = np.zeros(sim.params['total_time'])
            sim.treatment_times[self.params['treatment_start']:] = True

            _ = sim.run_simulation(save_without_asking=False, stop_at_fullstop=False)
            num_clones.append(sim.mutation_count)
        return np.mean(num_clones, dtype=np.float64), np.std(num_clones, dtype=np.float64, ddof=1)/ np.sqrt(len(num_clones))

    def minimization_function(self, mutation_rate, logger):
        rate = float(mutation_rate[0])
        mut_count_exp, mut_sem_exp = self.get_mutation_number_experiment()
        mut_count_sim, mut_sem_sim = self.get_mutation_number_simulation(mutation_rate=rate)
        if mut_sem_exp == 0:
            mut_sem_exp = 1.0
            print("Standard error of mean is zero, setting to 1.0 to avoid division by zero.")
        error = (mut_count_exp - mut_count_sim) ** 2
        logger.info(f"MSE: {error}, Mutation Count Exp: {mut_count_exp}, Mutation Count Sim: {mut_count_sim}")
        return error

    def fit_mutation_rate(self, initial_guess):
        initial_rate = np.array(initial_guess)

        log_dir = '../logs_fitting'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_list = [i for i in os.listdir(log_dir) if i.startswith('fit_mutation_rate_run_')]
        log_file = f'{log_dir}/fit_mutation_rate_run_{len(log_list)}.log'
        logger = setup_logger(log_file)
        logger.info(f"Initial guess: {initial_rate}")

        def callback_func(xk):
            logger.info(f"Current parameters: {xk}")
            print(xk)

        result = opt.minimize(self.minimization_function, initial_rate, args=(logger,), method='Nelder-Mead', callback=callback_func,
                              options={'disp': True, 'maxiter': 2100})
        # result = opt.least_squares(self.minimization_function, initial_rate, method='trf', bounds=(0, 20))
        optimized_params = result.x

        print(result)
        print("Optimized parameters:", optimized_params)
        fit_list = [i for i in os.listdir('../fit_results') if i.startswith('fit_mutation_rate')]
        torch.save(result, f'../fit_results/fit_mutation_rate_{len(fit_list)}_test.pth')

def setup_logger(log_file):
    logger = logging.getLogger(f'fitting_logger')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

if __name__ == '__main__':
    initial_guess = 0.75  # Initial guess for mutation rate
    fit_mutation = FitMutationRate()
    fit_mutation.fit_mutation_rate(initial_guess)