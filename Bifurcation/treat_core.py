import os
import numpy as np
import yaml


def get_params():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(find_project_root(current_dir, "requirements.txt"), "params.yaml")
    with open(path, "r") as file:
        params = yaml.safe_load(file)
    return params


def find_project_root(current_dir, marker_file):
    current_dir = os.path.abspath(current_dir)
    while current_dir != os.path.dirname(current_dir):  # Stop at the root of the file system
        if marker_file in os.listdir(current_dir):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return None


def build_treatment_schedule(treat_on, treat_off, params):
    first_start = int(params["treatment_start"])
    total_time = int(params["total_time"])
    treat_on = int(treat_on)
    treat_off = int(treat_off)

    treatment_times = np.zeros(total_time, dtype=bool)
    treatment_length = treat_on

    if treat_off == 0:
        treatment_starts = [first_start]
        treatment_length = total_time - first_start
        if treat_on == 0:
            treatment_starts = []
    elif treat_on == 0:
        treatment_starts = []
    else:
        treatment_starts = [d for d in range(first_start, total_time, treat_off + treat_on)]

    if treatment_starts:
        treatment_ends = np.asarray(treatment_starts, dtype=int) + treatment_length
    else:
        treatment_ends = np.asarray([], dtype=int)

    for start in treatment_starts:
        treatment_times[start:(start + treatment_length)] = True

    return treatment_times, treatment_starts, treatment_ends


def _clamp_efficacy(treatment_efficacy):
    if treatment_efficacy > 1.0:
        return 1.0
    if treatment_efficacy < 0.0:
        return 0.0
    return treatment_efficacy


def _apply_treatment_step(treatment_efficacy, current_treatment, prev_treatment, extra_steps_remaining, lag_steps_remaining, params, noise=False, rng=None):
    delta_t = params["delta_t"]
    treatment_delay = params["treatment_delay"]
    release_delay = params["release_delay"]

    cycle_start_pre_noise = None
    cycle_start_post_noise = None

    if prev_treatment and not current_treatment:
        extra_steps_remaining = int(params["overshoot_steps"])
        lag_steps_remaining = int(params["lag_steps"])

    elif current_treatment and not prev_treatment:
        extra_steps_remaining = 0
        lag_steps_remaining = 0
        cycle_start_pre_noise = treatment_efficacy

        if noise:
            treatment_efficacy += rng.normal(0.0, 0.1)
            cycle_start_post_noise = treatment_efficacy

        treatment_efficacy += delta_t / treatment_delay

    elif current_treatment:
        extra_steps_remaining = 0
        lag_steps_remaining = 0
        treatment_efficacy += delta_t / treatment_delay

    elif extra_steps_remaining > 0:
        treatment_efficacy += delta_t / treatment_delay
        extra_steps_remaining -= 1
        lag_steps_remaining -= 1

    elif lag_steps_remaining > 0:
        lag_steps_remaining -= 1

    else:
        treatment_efficacy -= delta_t / release_delay

    treatment_efficacy = _clamp_efficacy(treatment_efficacy)
    prev_treatment = current_treatment

    return (
        treatment_efficacy,
        prev_treatment,
        extra_steps_remaining,
        lag_steps_remaining,
        cycle_start_pre_noise,
        cycle_start_post_noise,
    )


def _simulate_treatment_efficacy(treat_on, treat_off, params, noise=False, random_seed=None):
    treatment_times, treatment_starts, treatment_ends = build_treatment_schedule(treat_on, treat_off, params)

    treatment_efficacy = 0.0
    results = []

    prev_treatment = False
    extra_steps_remaining = 0
    lag_steps_remaining = 0

    cycle_start_pre_noise_vals = []
    cycle_start_post_noise_vals = []
    rng = np.random.default_rng(random_seed) if noise else None

    for i in range(1, int(params["total_time"])):
        (
            treatment_efficacy,
            prev_treatment,
            extra_steps_remaining,
            lag_steps_remaining,
            cycle_start_pre_noise,
            cycle_start_post_noise,
        ) = _apply_treatment_step(
            treatment_efficacy,
            bool(treatment_times[i]),
            prev_treatment,
            extra_steps_remaining,
            lag_steps_remaining,
            params,
            noise=noise,
            rng=rng,
        )

        if cycle_start_pre_noise is not None:
            cycle_start_pre_noise_vals.append(cycle_start_pre_noise)
        if cycle_start_post_noise is not None:
            cycle_start_post_noise_vals.append(cycle_start_post_noise)

        results.append(treatment_efficacy)

    return (
        np.asarray(results, dtype=float),
        treatment_starts,
        treatment_ends,
        cycle_start_pre_noise_vals,
        cycle_start_post_noise_vals,
    )


def calc_treatment_efficacy(treat_on, treat_off, params, noise, random_seed=None):
    efficacy, treatment_starts, treatment_ends, cycle_start_pre_noise_vals, cycle_start_post_noise_vals = _simulate_treatment_efficacy(
        treat_on,
        treat_off,
        params,
        noise=noise,
        random_seed=random_seed,
    )

    if noise:
        return 1 - efficacy, treatment_starts, treatment_ends, cycle_start_pre_noise_vals, cycle_start_post_noise_vals
    return 1 - efficacy, treatment_starts, treatment_ends


def _find_last_full_cycle_bounds(treat_on, treat_off, params, treatment_starts):
    cycle_length = int(treat_on) + int(treat_off)
    total_time = int(params["total_time"])

    if cycle_length <= 0 or not treatment_starts:
        return None, None

    full_cycle_starts = [start for start in treatment_starts if start + cycle_length <= total_time]
    if not full_cycle_starts:
        return None, None

    cycle_start = full_cycle_starts[-1]
    cycle_end = cycle_start + cycle_length
    return cycle_start, cycle_end


def calc_last_cycle_metrics(treat_on, treat_off, params, noise=False, random_seed=None, post_noise=False):
    efficacy, treatment_starts, _, cycle_start_pre_noise_vals, cycle_start_post_noise_vals = _simulate_treatment_efficacy(
        treat_on,
        treat_off,
        params,
        noise=noise,
        random_seed=random_seed,
    )

    if treatment_starts:
        cycle_values = cycle_start_post_noise_vals if noise and post_noise else cycle_start_pre_noise_vals
        if cycle_values:
            last_cycle_start_efficacy = float(cycle_values[-1])
        else:
            last_cycle_start_efficacy = float("nan")
    else:
        last_cycle_start_efficacy = float("nan")

    cycle_start, cycle_end = _find_last_full_cycle_bounds(treat_on, treat_off, params, treatment_starts)
    if cycle_start is None:
        last_cycle_mean_efficacy = float("nan")
    else:
        slice_start = max(0, cycle_start - 1)
        slice_end = max(slice_start, cycle_end - 1)
        if slice_end > slice_start:
            last_cycle_mean_efficacy = float(np.mean(efficacy[slice_start:slice_end]))
        else:
            last_cycle_mean_efficacy = float("nan")

    return last_cycle_start_efficacy, last_cycle_mean_efficacy


def calc_last_cycle_start_efficacy(treat_on, treat_off, params, noise=False, random_seed=None, post_noise=False):
    last_cycle_start_efficacy, _ = calc_last_cycle_metrics(
        treat_on,
        treat_off,
        params,
        noise=noise,
        random_seed=random_seed,
        post_noise=post_noise,
    )
    return last_cycle_start_efficacy


def calc_last_cycle_mean_efficacy(treat_on, treat_off, params, noise=False, random_seed=None, post_noise=False):
    _, last_cycle_mean_efficacy = calc_last_cycle_metrics(
        treat_on,
        treat_off,
        params,
        noise=noise,
        random_seed=random_seed,
        post_noise=post_noise,
    )
    return last_cycle_mean_efficacy


def advance_one_cycle(initial_efficacy, treat_on, treat_off, params, noise=False, random_seed=None):
    treat_on = int(treat_on)
    treat_off = int(treat_off)
    cycle_length = treat_on + treat_off

    if treat_on <= 0 or treat_off < 0 or cycle_length <= 0:
        return float("nan"), float("nan"), np.asarray([], dtype=float)

    treatment_efficacy = _clamp_efficacy(float(initial_efficacy))
    prev_treatment = False
    extra_steps_remaining = 0
    lag_steps_remaining = 0
    rng = np.random.default_rng(random_seed) if noise else None
    cycle_trace = []

    local_schedule = np.concatenate(
        (
            np.ones(treat_on, dtype=bool),
            np.zeros(treat_off, dtype=bool),
        )
    )

    for current_treatment in local_schedule:
        (
            treatment_efficacy,
            prev_treatment,
            extra_steps_remaining,
            lag_steps_remaining,
            _,
            _,
        ) = _apply_treatment_step(
            treatment_efficacy,
            bool(current_treatment),
            prev_treatment,
            extra_steps_remaining,
            lag_steps_remaining,
            params,
            noise=noise,
            rng=rng,
        )
        cycle_trace.append(treatment_efficacy)

    return (
        float(treatment_efficacy),
        float(np.mean(cycle_trace)) if cycle_trace else float("nan"),
        np.asarray(cycle_trace, dtype=float),
    )


def iterate_cycle_map(initial_efficacy, treat_on, treat_off, params, num_cycles):
    cycle_start_values = [float(initial_efficacy)]
    cycle_mean_values = []

    current_efficacy = float(initial_efficacy)
    for _ in range(int(num_cycles)):
        next_efficacy, cycle_mean, _ = advance_one_cycle(
            current_efficacy,
            treat_on,
            treat_off,
            params,
            noise=False,
        )
        cycle_start_values.append(next_efficacy)
        cycle_mean_values.append(cycle_mean)
        current_efficacy = next_efficacy

    return np.asarray(cycle_start_values, dtype=float), np.asarray(cycle_mean_values, dtype=float)
