import numpy as np
from source import core as cr
import scipy.optimize as opt
import torch
import logging
import os


def check_edge(img, x, y,):
    for i in range(x - 1, x + 2):
        for j in range(y - 1, y + 2):
            if i < 0 or j < 0 or i >= img.shape[0] or j >= img.shape[1]:
                return True
            if img[i, j] == 0:
                return True
    return False

def remove_edge(img, index_list, removal_layers=1):
    img0 = img.copy()
    for _ in range(removal_layers):
        img_temp = img0.copy()
        for x, y in index_list:
            if img_temp[x, y] == True:
                if check_edge(img_temp, x, y):
                    img0[x, y] = 0
    return img0

def has_breakout(sensitive, resistant):
    if not np.any(resistant):
        return False

    inverted_sen = ~sensitive
    res_out = np.where(inverted_sen, resistant, False)
    if np.sum(res_out) > 0:
        return True
    else:
        sensitive = remove_edge(sensitive, np.argwhere(sensitive))
        inverted_sen = ~sensitive
        res_out = np.where(inverted_sen, resistant, False)
        if np.sum(res_out) > 0:
            return True
        else:
            return False

def get_start_point(initial_guess):
    sim = cr.DiffusionModel2D()
    sim.params['mutation_scaling'] = initial_guess
    sim.params['mutations_active'] = True
    sim.params['image_size'] = 200
    sim.params['return_all'] = True
    sim.params['save_in_core'] = False
    sim.random_seed = 0
    sim.params['total_time'] = 450
    sim.treatment_times = np.zeros(sim.params['total_time'])
    _, sensitive, resistant, _, _ = sim.run_simulation()

    counts = []
    for i in range(len(sensitive)):
        sen_thresholded = np.where(sensitive[i] > (1 / initial_guess), 1, 0)
        res_thresholded = np.where(sensitive[i] > (1 / initial_guess), 1, 0)

        total_array = sen_thresholded + res_thresholded
        total_count = np.count_nonzero(total_array)
        counts.append(total_count)
    area_sim = np.array(counts) * (1376 / 100) ** 2
    area_exp = 8020.307692307692

    for i in range(len(area_sim)):
        if area_sim[i] >= area_exp:
            return i
    return None


def run_sim(initial_guess, replicates, treatment_duration, experiment_duration, logger):
    start_point = 351
    logger.info(f"Determined start point: {start_point}")
    breakout_probabilitys = []

    for j in range(len(replicates)):
        sim_replicates = replicates[j] * 2
        breakouts = []

        for i in range(sim_replicates):
            sim = cr.DiffusionModel2D()
            sim.params['mutation_scaling'] = initial_guess
            sim.params['mutations_active'] = True
            sim.params['image_size'] = 200
            sim.params['return_all'] = False
            sim.params['save_in_core'] = False
            sim.params['total_time'] = experiment_duration[j] + start_point
            sim.random_seed = i
            sim.set_random_seed()
            sim.treatment_times = np.zeros(sim.params['total_time'])
            treat_start = 360 + start_point
            sim.treatment_times[treat_start:treat_start+treatment_duration[j]] = True
            _, sensitive, resistant, _, _ = sim.run_simulation()

            sensitive, resistant = np.where(sensitive > 1/initial_guess, True, False), np.where(resistant > 1/initial_guess, True, False)

            breakout = has_breakout(sensitive, resistant)
            if breakout:
                breakouts.append(1)

        sim_breakout_probability = np.sum(breakouts)  / sim_replicates
        breakout_probabilitys.append(sim_breakout_probability)
    return breakout_probabilitys

def minimization_function(initial_guess, breakouts, replicates, treatment_duration, experiment_duration, logger):
    sim_breakout_probability = run_sim(initial_guess, replicates, treatment_duration, experiment_duration, logger)
    exp_breakout_probability = np.array(breakouts)/np.array(replicates)
    value_to_min = error_function(np.array(sim_breakout_probability), exp_breakout_probability)
    logger.info(f"MSE: {value_to_min}")
    return value_to_min

def error_function(sim_breakout_probability, exp_breakout_probability):
    err = np.mean((sim_breakout_probability - exp_breakout_probability)**2)
    return err

def fit_simulation(initial_guess):
    initial_guess = initial_guess
    breakouts =           np.array([   4,    4,    5,    8,   10,   12,   13,   12,   10,   10]) # np.array([   4,    6,    5,    7,   10,   13,   13,   12,   11,   10])
    replicates =          np.array([  10,   12,   10,   15,   11,   13,   13,   12,   10,   10])
    treatment_duration =  np.array([  40,   80,  120,  160,  200,  240,  280,  320,  360,  400])
    experiment_duration = np.array([1360, 1360, 1360, 1360, 1360, 1360, 1360, 1360, 1360, 1360])

    # breakouts =             [  13,   4,   9,   14,    6,   13,    2,   11,   1]
    # replicates =            [  15,  16,  13,   14,    9,   15,    7,   12,  12]
    # treatment_duration =    [ 200, 180, 170,  280,  230,  210,  200,  400,  40]
    # experiment_duration =   [1000, 900, 930, 1200, 1060, 1040, 1030, 1240, 880] # treat_start + treat_duration + 400 + 80 (most of the time)

    log_dir = '../logs_fitting'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_list = [i for i in os.listdir(log_dir) if i.startswith('fit_mutation_scaling_run_')]
    log_file = f'{log_dir}/fit_mutation_scaling_run_{len(log_list)}.log'
    logger = setup_logger(log_file)
    logger.info(f"Initial guess: {initial_guess}")

    def callback_func(xk):
        logger.info(f"Current parameters: {xk}")
        print(xk)

    result = opt.minimize(minimization_function, np.array(initial_guess), args=(breakouts, replicates, treatment_duration, experiment_duration, logger),
                          method='Nelder-Mead', callback=callback_func, options={'disp': True, 'maxiter': 2100})
    optimized_params = result.x

    print(result)
    print("Optimized parameters:", optimized_params)
    fit_list = [i for i in os.listdir('../fit_results') if i.startswith('fit_mutation_scaling')]
    torch.save(result, f'../fit_results/fit_mutation_scaling_{len(fit_list)}.pth')

def setup_logger(log_file):
    logger = logging.getLogger(f'fitting_logger')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def main():
    fit_simulation(initial_guess=1000)

if __name__ == '__main__':
    main()
