import os
import pandas as pd
import numpy as np
from source import core as cr
import scipy.optimize as opt
import tqdm
import torch
import logging
import yaml
from pathlib import Path


def get_exp_data(path):
    pulse_csv = pd.read_csv(os.path.join(cr.find_project_root(os.getcwd(), 'requirements.txt'), f'source/fit/fit_data/no_treatment_csv/{path}'))
    area = pulse_csv['colony_area'][:150]
    return area

def get_nutrient_data(start_point):
    position_of_mutant_in_sim_pixel = [3, 7, 11, 15]
    frame_of_growth = [62.7138815475517,  122.85735392335089, 183.00082629915008, 243.1442986749493]  # Calculate from experimental data (use calculate_res_regrowth_vals.py)(parameter_checking fit to regrowth curve)
    sim_step_of_growth = (np.array(frame_of_growth)) * 10 + start_point
    return position_of_mutant_in_sim_pixel, sim_step_of_growth

def run_simulation(initial_guess):
    sim = cr.DiffusionModel2D()
    sim.treatment_times = np.zeros(1980, dtype=bool)
    sim.params['diffusion_sensitive'] = initial_guess[0]
    sim.params['diffusion_resistant'] = initial_guess[0]
    sim.params['uptake_rate'] = initial_guess[1]
    sim.params['diffusion_nutrients'] = initial_guess[2]
    sim.params['image_size'] = 200
    sim.params['mutations_active'] = False
    sim.set_random_seed()

    n_array, s_array, r_array = [], [], []
    nutrients, sensitive, resistant = sim.get_initial_state()

    n_array.append(np.copy(nutrients))
    s_array.append(np.copy(sensitive))
    r_array.append(np.copy(resistant))

    for i in tqdm.tqdm(range(1, 1980)):
        nutrients, sensitive, resistant = sim.update(i, nutrients, sensitive, resistant)
        n_array.append(np.copy(nutrients))
        s_array.append(np.copy(sensitive))
        r_array.append(np.copy(resistant))

    return n_array, s_array, r_array, sim.params['mutation_scaling']

def update_params(start_point):
    path = os.path.join(find_project_root(os.getcwd(), 'requirements.txt'), 'params.yaml')
    with open(path, 'r') as file:
        params = yaml.safe_load(file)

    params['start_point'] = start_point
    path = Path(path)
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w") as f:
        yaml.safe_dump(params, f, sort_keys=False)
    os.replace(tmp_path, path)

def find_project_root(current_dir, marker_file):
    current_dir = os.path.abspath(current_dir)
    while current_dir != os.path.dirname(current_dir):
        if marker_file in os.listdir(current_dir):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return None

def run_sim_for_nut_diff(initial_guess, logger):
    sim1 = cr.DiffusionModel2D()
    start_point = sim1.params['start_point']
    sim1.params['diffusion_sensitive'] = initial_guess[0]
    sim1.params['diffusion_resistant'] = initial_guess[0]
    sim1.params['uptake_rate'] = initial_guess[1]
    sim1.params['diffusion_nutrients'] = initial_guess[2]
    sim1.params['mutations_active'] = False
    sim1.params['image_size'] = 200
    sim1.params['save_in_core'] = True
    first_start = 360 + start_point
    time = 390 + start_point
    sim1.treatment_times = np.zeros(time)
    sim1.treatment_times[first_start:] = True
    sim1.params['total_time'] = time
    sim1.params['save_results'] = os.path.join(cr.find_project_root(os.getcwd(), 'requirements.txt'), 'source/fit/fit_data/sim_data/nutrient_diffusion')
    sim1.run_simulation(save_without_asking=True)

    sim = cr.DiffusionModel2D()
    sim.params['diffusion_sensitive'] = initial_guess[0]
    sim.params['diffusion_resistant'] = initial_guess[0]
    sim.params['uptake_rate'] = initial_guess[1]
    sim.params['diffusion_nutrients'] = initial_guess[2]
    sim.params['total_time'] = 2000 + start_point
    sim.params['treatment_efficacy'] = np.load(os.path.join(cr.find_project_root(os.getcwd(), 'requirements.txt'), 'source/fit/fit_data/sim_data/nutrient_diffusion/treatment_efficacy.npy'))[360 + start_point]
    sim.params['image_size'] = 200
    sim.params['mutations_active'] = False
    sim.treatment_times = np.ones(3500, dtype=bool)
    sim.set_random_seed()
    sensitive = np.load(os.path.join(cr.find_project_root(os.getcwd(), 'requirements.txt'), 'source/fit/fit_data/sim_data/nutrient_diffusion/sensitive.npy'))
    nutrients = np.load(os.path.join(cr.find_project_root(os.getcwd(), 'requirements.txt'), 'source/fit/fit_data/sim_data/nutrient_diffusion/nutrients.npy'))

    sen_start = sensitive[360 + start_point]
    nut_start = nutrients[360 + start_point]
    pos, exp_times = get_nutrient_data(start_point)
    index = np.where(sen_start[100, :] >= 1/sim.params['mutation_scaling'])[0]
    if len(index) == 0:
        logger.info("No sensitive cells found at the specified position.")
        print("No sensitive cells found at the specified position.")
        return pos, exp_times, [0]

    res_start = np.zeros_like(sen_start)
    res_start[100, index[0] + int(pos[0])] = 1/sim.params['mutation_scaling']
    res_start[100, index[-1] - int(pos[1])] =  1/sim.params['mutation_scaling']
    res_start[index[0] + int(pos[2]), 100] =  1/sim.params['mutation_scaling']
    res_start[index[-1] - int(pos[3]), 100] =  1/sim.params['mutation_scaling']
    pos_to_check_1 = [100, index[0] + int(pos[0]) - 1]
    pos_to_check_2 = [100, index[-1] - int(pos[1]) + 1]
    pos_to_check_3 = [index[0] + int(pos[2]) - 1, 100]
    pos_to_check_4 = [index[-1] - int(pos[3]) + 1, 100]
    check_1 = False
    check_2 = False
    check_3 = False
    check_4 = False

    sim_times = []
    for i in tqdm.tqdm(range(1, 3500)):
        nut_start, sen_start, res_start = np.copy(sim.update(i, nut_start, sen_start, res_start))

        # Check if the resistant cells are over the threshold in the positions to check
        if np.sum(res_start[pos_to_check_1[0], pos_to_check_1[1]]) > 1/sim.params['mutation_scaling'] and not check_1:
            check_1 = True
            sim_times.append(i + 360 + start_point)

        if np.sum(res_start[pos_to_check_2[0], pos_to_check_2[1]]) > 1/sim.params['mutation_scaling'] and not check_2:
            check_2 = True
            sim_times.append(i + 360 + start_point)

        if np.sum(res_start[pos_to_check_3[0], pos_to_check_3[1]]) > 1/sim.params['mutation_scaling'] and not check_3:
            check_3 = True
            sim_times.append(i + 360 + start_point)

        if np.sum(res_start[pos_to_check_4[0], pos_to_check_4[1]]) > 1/sim.params['mutation_scaling'] and not check_4:
            check_4 = True
            sim_times.append(i + 360 + start_point)

    if len(sim_times) < len(exp_times):
        for _ in range(len(exp_times) - len(sim_times)):
            sim_times = [0]

    return pos, exp_times, sim_times

def extract_area(sensitive_array, resistant_array, mutation_scaling):
    area = []
    for current_dt in range(sensitive_array.shape[0]):
        sen_thresholded = np.where(sensitive_array[current_dt] > (1 / mutation_scaling), 1, 0)
        res_thresholded = np.where(resistant_array[current_dt] > (1 / mutation_scaling), 1, 0)

        total_array = sen_thresholded + res_thresholded
        total_count = np.count_nonzero(total_array)
        area.append(total_count)
    return np.array(area)*(1376/100)**2

def get_start_point(area_exp, area_sim):
    for i in range(len(area_sim)):
        if area_sim[i] >= area_exp[0]:
            return i
    return 0

def minimization_function(initial_guess, logger):
    area_sim_list = []
    area_exp_list = []
    experiment_replicas = [path for path in os.listdir(os.path.join(cr.find_project_root(os.getcwd(), 'requirements.txt'), 'source/fit/fit_data/no_treatment_csv/For_Manuscript')) if path.endswith('clonearea.csv')]
    positions, exp_times, sim_times = run_sim_for_nut_diff(initial_guess, logger)
    for i, path in enumerate(experiment_replicas):
        area_exp = get_exp_data(path)
        area_exp_list.append(area_exp)
        nutrient_array, sensitive_array, resistant_array, mutation_scaling = run_simulation(initial_guess)
        area_sim = extract_area(np.array(sensitive_array), np.array(resistant_array), mutation_scaling)
        if i == 0:
            start_point = get_start_point(area_exp, area_sim)
            update_params(start_point)
        logger.info(f"Start point for {path}: {start_point}")
        area_sim_list.append(area_sim[start_point:(len(area_exp)*10 + start_point)])
    value_to_min = error_function(np.array(area_exp_list), np.array(area_sim_list), np.array(exp_times), np.array(sim_times), logger)
    print(f"Results: {value_to_min}")
    return value_to_min

def error_function(area_exp, area_sim, exp_times, sim_times, logger):
    if len(area_exp[0])*10 != len(area_sim[0]):
        print(len(area_exp[0]) * 10, len(area_sim[0]))
        logger.info("Experimental and simulated data lengths do not match. (Most likely due to overflows in the simulation due to wrong parameters.) Return 2 as a value to minimize.")
        print("Experimental and simulated data lengths do not match. (Most likely due to overflows in the simulation due to wrong parameters.) Return 2 as a value to minimize.")
        return 2
    if sim_times[0] == 0:
        logger.info("No resistant cells found in the simulation. Return 2 to punish fitting.")
        print("No resistant cells found in the simulation. Return 2 to punish fitting.")
        return 2
    area_nrmse_list = []
    for i in range(len(area_exp)):
        area_mse_part = np.mean((area_sim[i, ::10] - area_exp[i]) ** 2)
        area_rmse_part = np.sqrt(area_mse_part)
        area_nrmse_part = area_rmse_part / np.mean(area_exp[i, ::10])
        area_nrmse_list.append(area_nrmse_part)
    radius_nrmse = np.mean(area_nrmse_list)

    times_mse = np.mean((sim_times - exp_times)**2)
    times_rmse = np.sqrt(times_mse)
    times_nrmse = times_rmse / np.mean(exp_times)
    logger.info(f"Radius NRMSE: {radius_nrmse}, Times NRMSE: {times_nrmse}")
    total_value_to_min = radius_nrmse + times_nrmse
    return total_value_to_min

def fit_simulation(initial_guess):
    initial_guess = initial_guess

    log_dir = '../logs_fitting'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_list = [i for i in os.listdir(log_dir) if i.startswith('fit_dispersion_and_nutrients_run_')]
    log_file = f'{log_dir}/fit_dispersion_and_nutrients_run_{len(log_list)}.log'
    logger = setup_logger(log_file)
    logger.info(f"Initial guess: {initial_guess}")

    def callback_func(xk):
        logger.info(f"Current parameters: {xk}")
        print(xk)

    result = opt.minimize(minimization_function, np.array(initial_guess), args=(logger,), method='Nelder-Mead', callback=callback_func,
                          options={'disp': True, 'maxiter': 2100})
    optimized_params = result.x

    print(result)
    print("Optimized parameters:", optimized_params)
    fit_list = [i for i in os.listdir('../fit_results') if i.startswith('fit_dispersion_and_nutrients')]

    torch.save(result, f'../fit_results/fit_dispersion_and_nutrients_{len(fit_list)}.pth')
    logger.info(f"Optimized parameters saved to ../fit_results/fit_dispersion_and_nutrients_{len(fit_list)}.pth")

def setup_logger(log_file):
    logger = logging.getLogger(f'fitting_logger')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def main():
    # dispersion_sensitive/resistant, uptake_rate, diffusion_nutrients
    initial_guess = [0.38293199, 1.90624767, 0.09785365]
    fit_simulation(initial_guess)

if __name__ == '__main__':
    main()
