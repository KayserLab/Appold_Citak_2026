import time
import numpy as np
from source import core as cr
import os
import multiprocessing as mp
import yaml
import torch
import tqdm
from functools import partial


def init_memmaps(params, num_sim):
    img_size = params['image_size']
    os.makedirs(params['save_folder'], exist_ok=True)
    shapes = {'nutrients': (num_sim, img_size, img_size), 'sensitive': (num_sim, img_size, img_size),
              'resistant': (num_sim, img_size, img_size), 'efficacy': (num_sim, params['total_time']),
              'treatment_times': (num_sim, params['total_time'])}
    dtypes = {'nutrients': np.float32, 'sensitive': np.float32, 'resistant': np.float32, 'efficacy': np.float32, 'treatment_times': np.bool_}

    for key in shapes:
        fname = os.path.join(params['save_folder'], f'{key}.dat')
        np.memmap(fname, dtype=dtypes[key], mode='w+', shape=shapes[key])

    status_path = os.path.join(params['save_folder'], 'status.dat')
    if not os.path.exists(status_path):
        np.memmap(status_path, mode='w+', dtype=np.bool_, shape=(num_sim,))

def worker(item, num_sim=None):
    idx, sweep_params = item

    sim = cr.DiffusionModel2D()
    treatment_on = sweep_params[0]
    treatment_off = sweep_params[1]
    replica = sweep_params[2]
    mutation_rate = sweep_params[4]
    sim.random_seed = sweep_params[3]

    first_start = sim.params['treatment_start'] + sim.params['start_point']
    time = sim.params['total_time']
    sim.params['mutation_rate'] = mutation_rate

    sim.treatment_times = np.zeros(time)
    treatment_length = treatment_on
    if treatment_off == 0:
        treatment_starts = [first_start]
        treatment_length = time - first_start
        if treatment_on == 0:
            treatment_starts = []
    elif treatment_on == 0:
        treatment_starts = []
    else:
        treatment_starts = [d for d in range(first_start, time, treatment_off + treatment_on)]

    for i in range(len(treatment_starts)):
        sim.treatment_times[treatment_starts[i]:(treatment_starts[i] + treatment_length)] = True

    nutrients, sensitive, resistant, treatment_times, treatment_efficacy = sim.run_simulation(save_without_asking=True, stop_with_size=True)

    nutrient_mmap = np.memmap(os.path.join(sim.params['save_folder'], 'nutrients.dat'), dtype=np.float32, mode='r+',
                              shape=(num_sim, sim.params['image_size'], sim.params['image_size']))
    sensitive_mmap = np.memmap(os.path.join(sim.params['save_folder'], 'sensitive.dat'), dtype=np.float32, mode='r+',
                               shape=(num_sim, sim.params['image_size'], sim.params['image_size']))
    resistant_mmap = np.memmap(os.path.join(sim.params['save_folder'], 'resistant.dat'), dtype=np.float32, mode='r+',
                               shape=(num_sim, sim.params['image_size'], sim.params['image_size']))
    efficacy_mmap = np.memmap(os.path.join(sim.params['save_folder'], 'efficacy.dat'), dtype=np.float32, mode='r+',
                              shape=(num_sim, sim.params['total_time']))
    treat_times_mmap = np.memmap(os.path.join(sim.params['save_folder'], 'treatment_times.dat'), dtype=np.bool_, mode='r+',
                                 shape=(num_sim, sim.params['total_time']))

    nutrient_mmap[idx] = nutrients
    sensitive_mmap[idx] = sensitive
    resistant_mmap[idx] = resistant
    efficacy_mmap[idx] = treatment_efficacy
    treat_times_mmap[idx] = treatment_times

    status = np.memmap(os.path.join(sim.params['save_folder'], 'status.dat'), dtype=np.bool_, mode='r+', shape=(num_sim,))
    status[idx] = True

    return idx


def build_sweep_params(params):
    num_treatment_on_steps = int((params['treatment_on_max'] - params['treatment_on_min']) / params['treatment_on_step']) + 1
    treatment_on_durations = np.linspace(params['treatment_on_min'], params['treatment_on_max'], num_treatment_on_steps, dtype=np.int16)

    num_treatment_off_steps = int((params['treatment_off_max'] - params['treatment_off_min']) / params['treatment_off_step']) + 1
    treatment_off_durations = np.linspace(params['treatment_off_min'], params['treatment_off_max'], num_treatment_off_steps, dtype=np.int16)

    num_mutation_rate_steps = int((params['mutation_rate_max'] - params['mutation_rate_min']) / params['mutation_rate_step']) + 1
    mutation_rates = np.linspace(params['mutation_rate_min'], params['mutation_rate_max'], num_mutation_rate_steps)
    print(treatment_on_durations)
    print(treatment_off_durations)
    print(mutation_rates)

    rand_iterations = params['num_replicas']
    random_seeds = [i for i in range(len(treatment_on_durations)*len(treatment_off_durations)*rand_iterations*len(mutation_rates))]
    counter = 0

    sweep_params = []
    for treatment_on in treatment_on_durations:
        for treatment_off in treatment_off_durations:
            for mutation_rate in mutation_rates:
                for j, seed in enumerate(random_seeds[counter:counter+rand_iterations]):
                    sweep_params.append((treatment_on, treatment_off, j, seed, mutation_rate))
                    counter += 1

    return sweep_params


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', type=int, default=0)
    parser.add_argument('--num-jobs', type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    with open('../params.yaml', 'r') as file:
        params = yaml.safe_load(file)['simulation_params']

    params_list = build_sweep_params(params)
    num_sim = len(params_list)
    print(f'Number of simulations: {num_sim}')

    if args.job_id == 0:
        init_memmaps(params, num_sim)
    else:
        check_init = os.path.exists(os.path.join(params['save_folder'], 'nutrients.dat')) and \
                      os.path.exists(os.path.join(params['save_folder'], 'resistant.dat')) and \
                      os.path.exists(os.path.join(params['save_folder'], 'sensitive.dat')) and \
                      os.path.exists(os.path.join(params['save_folder'], 'efficacy.dat')) and \
                      os.path.exists(os.path.join(params['save_folder'], 'treatment_times.dat')) and \
                      os.path.exists(os.path.join(params['save_folder'], 'status.dat'))

        while not check_init:
            time.sleep(1)
            check_init = os.path.exists(os.path.join(params['save_folder'], 'nutrients.dat')) and \
                         os.path.exists(os.path.join(params['save_folder'], 'resistant.dat')) and \
                         os.path.exists(os.path.join(params['save_folder'], 'sensitive.dat')) and \
                         os.path.exists(os.path.join(params['save_folder'], 'efficacy.dat')) and \
                         os.path.exists(os.path.join(params['save_folder'], 'treatment_times.dat')) and \
                         os.path.exists(os.path.join(params['save_folder'], 'status.dat'))

    status = np.memmap(os.path.join(params['save_folder'], 'status.dat'), dtype=np.bool_, mode='r+', shape=(num_sim,))
    undone = np.nonzero(status == False)[0]
    missing_idxs = [i for i in undone if i % args.num_jobs == args.job_id]
    jobs = [(i, params_list[i]) for i in missing_idxs]

    num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', mp.cpu_count() - 4))

    worker_with_num_sim = partial(worker, num_sim=num_sim)
    with mp.Pool(processes=num_cpus) as pool:
        for _ in tqdm.tqdm(pool.imap(worker_with_num_sim, jobs), total=len(jobs)):
            pass

    torch.save(params_list, f'{params["save_folder"]}/params_list.pth')
    torch.save(params, f'{params["save_folder"]}/params.pth')


if __name__ == '__main__':
    main()
