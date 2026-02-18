import numpy as np
from source import core as cr


def main(treat_on, treat_off, save_dir=None, random_seed=None, pulse=False, pulse_duration=280):
    sim = cr.DiffusionModel2D()
    sim.random_seed = 1
    if random_seed is not None:
        sim.random_seed = random_seed
    sim.set_random_seed()

    sim.params['save_in_core'] = True

    if save_dir is not None:
        sim.params['save_results'] = save_dir

    time = sim.params['total_time']
    start_point = 0 if sim.params['gaussian'] else sim.params['start_point']
    first_start = sim.params['treatment_start'] + start_point
    sim.params['treatment_on_duration'] = pulse_duration if pulse else treat_on

    sim.treatment_times = np.zeros(time)
    treatment_length = treat_on
    if treat_off == 0:
        treatment_starts = [first_start]
        treatment_length = time - first_start
        if treat_on == 0:
            treatment_starts = []
    elif treat_on == 0:
        treatment_starts = []
    else:
        treatment_starts = [d for d in range(first_start, time, treat_off + treat_on)]

    for i in range(len(treatment_starts)):
        sim.treatment_times[treatment_starts[i]:(treatment_starts[i] + treatment_length)] = True

    if pulse:
        sim.treatment_times[first_start:(first_start + pulse_duration)] = True

    _ = sim.run_simulation(save_without_asking=False, stop_at_fullstop=False)


if __name__ == "__main__":
    treatment_on = 130
    treatment_off = 360
    main(treatment_on, treatment_off)