import pathlib as pl
import sys


def add_project_root_to_path():
    project_root = pl.Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


add_project_root_to_path()

import numpy as np
import tqdm
from source import core as cr
from Validation import validation_common as vc
import matplotlib.pyplot as plt


plt.rcParams.update({'font.size': 7,
                     'pdf.fonttype': 42,
                     'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial'],
                     'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Arial',
                     'mathtext.it': 'Arial:italic',
                     'mathtext.bf': 'Arial:bold',
                     })


def get_full_data_regrowth_calibration():
    continuous_dataset = vc.load_mutation_rate_dataset()
    continuous_wells = [str(entry["label"]) for entry in continuous_dataset]
    return vc.fit_regrowth_calibration_from_continuous_therapy(wells=continuous_wells, min_duration_frames=10, min_distance_um=100.0)


def get_initial_guess(params_yaml):
    return np.array([params_yaml["diffusion_sensitive"], params_yaml["uptake_rate"], params_yaml["diffusion_nutrients"]], dtype=float)


def get_nutrient_data(start_point, regrowth_calibration):
    position_of_mutant_in_sim_pixel = [2, 4, 6, 8]
    sim_step_of_growth = vc.regrowth_offsets_to_detection_frames(position_of_mutant_in_sim_pixel, start_point, regrowth_calibration)
    return position_of_mutant_in_sim_pixel, sim_step_of_growth


def get_pretreatment_state(initial_guess, params_yaml, start_point):
    first_start = int(params_yaml["treatment_start"]) + int(start_point)
    pretreatment = np.zeros(first_start + 1, dtype=bool)
    pretreatment[first_start:] = True
    return vc.simulate_state_at_step(initial_guess, pretreatment, first_start)


def build_continuation_sim(initial_guess, start_point, treatment_efficacy, prev_treatment):
    sim = cr.DiffusionModel2D()
    vc.configure_dispersion_sim(sim, initial_guess)
    sim.params["total_time"] = int(2400 + start_point)
    sim.treatment_times = np.ones(3500, dtype=bool)
    sim.treatment_efficacy = float(treatment_efficacy)
    sim.prev_treatment = bool(prev_treatment)
    sim.random_seed = 1
    sim.set_random_seed()
    return sim


def seed_resistant_cells(sen_start, positions, mutation_scaling):
    index = np.where(sen_start[100, :] >= 1.0 / mutation_scaling)[0]
    if len(index) == 0:
        raise ValueError("No sensitive cells found at the nutrient-diffusion start state.")

    res_start = np.zeros_like(sen_start)
    res_start[100, index[0] + int(positions[0])] = 1.0 / mutation_scaling
    res_start[100, index[-1] - int(positions[1])] = 1.0 / mutation_scaling
    res_start[index[0] + int(positions[2]), 100] = 1.0 / mutation_scaling
    res_start[index[-1] - int(positions[3]), 100] = 1.0 / mutation_scaling

    pos_to_check = [(100, index[0] + int(positions[0]) - 1), (100, index[-1] - int(positions[1]) + 1), (index[0] + int(positions[2]) - 1, 100), (index[-1] - int(positions[3]) + 1, 100)]
    return res_start, pos_to_check


def plot_seed_state(sen_start, res_start):
    res_max = np.max(res_start)
    sen_max = np.max(sen_start)
    red = res_start / res_max if res_max > 0 else res_start
    green = sen_start / sen_max if sen_max > 0 else sen_start
    plt.imshow(np.stack([red, green, np.zeros_like(sen_start)], axis=-1), interpolation='none')
    plt.show()


def run_resistant_timing(sim, nut_start, sen_start, res_start, pos_to_check, params_yaml, start_point):
    triggered = [False, False, False, False]
    sim_times = []
    threshold = 1.0 / sim.params["mutation_scaling"]

    for timer in tqdm.tqdm(range(1, 3500)):
        nut_start, sen_start, res_start = sim.update(timer, nut_start, sen_start, res_start)
        for pos_idx, (row, col) in enumerate(pos_to_check):
            if not triggered[pos_idx] and res_start[row, col] > threshold:
                triggered[pos_idx] = True
                sim_times.append(timer + int(params_yaml["treatment_start"]) + int(start_point))
        if all(triggered):
            break

    return np.array(sim_times, dtype=float)


def plot_timing_comparison(positions, exp_times, sim_times):
    plt.figure(figsize=(1.5, 1.8), dpi=300)
    plt.plot(np.array(positions) * (1376 / 100) * 8.648 / 1e3, exp_times / 20, 'bo', label='Experiment', markersize=5)
    plt.plot(np.array(positions[:len(sim_times)]) * (1376 / 100) * 8.648 / 1e3, sim_times / 20, 'ro', label='Simulation', markersize=5)
    plt.ylabel('Time (h)')
    plt.xlabel('Position (mm)')
    plt.xlim(0, 1.1)
    plt.legend()
    plt.savefig('SI_Figures/plots/growth_delay_comparison_test.pdf', bbox_inches='tight', transparent=True)
    plt.show()


def main():
    params_yaml = vc.load_params_yaml('params.yaml')
    start_point = int(params_yaml["start_point"])
    initial_guess = get_initial_guess(params_yaml)
    regrowth_calibration = get_full_data_regrowth_calibration()

    nut_start, sen_start, _, mutation_scaling, treatment_efficacy, prev_treatment = get_pretreatment_state(initial_guess, params_yaml, start_point)
    positions, exp_times = get_nutrient_data(start_point, regrowth_calibration)
    res_start, pos_to_check = seed_resistant_cells(sen_start, positions, mutation_scaling)
    sim = build_continuation_sim(initial_guess, start_point, treatment_efficacy, prev_treatment)

    print(f"Full-data regrowth calibration: t0_h={regrowth_calibration['t0_h']}, slope_um_per_h={regrowth_calibration['slope_um_per_h']}")
    print(f"Target frames from calibration: {((exp_times - start_point) / 10).tolist()}")

    plot_seed_state(sen_start, res_start)
    sim_times = run_resistant_timing(sim, nut_start, sen_start, res_start, pos_to_check, params_yaml, start_point)
    print(f"Simulation timing steps: {sim_times.tolist()}")
    plot_timing_comparison(positions, exp_times, sim_times)


if __name__ == '__main__':
    main()
