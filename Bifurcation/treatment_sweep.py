import functools as ft
import multiprocessing as mp
import os
import time

import numpy as np
import tqdm
import yaml

try:
    from . import treat_core
except ImportError:
    import treat_core


def get_treatment_sweep_config():
    return {
        "default_save_folder": os.path.join("data", "sweeps", "treatment_only_sweep"),
        "duration": 300000,
        "tau_on_range": (0, 240, 1),
        "tau_off_range": (360, 360, 5),
        "num_replicates": 1,
        "save_folder": os.path.join("data", "sweeps", "treatment_only_sweep_rebuttal"),
        "noise": False,
        "post_noise": False,
        "num_cpus": 32,
        "params_path": "params.yaml",
    }


def resolve_path(project_root, path):
    if os.path.isabs(path):
        return path
    return os.path.join(project_root, path)


def load_params(params_path):
    with open(params_path, "r") as file:
        return yaml.safe_load(file)


def build_duration_axis(range_values, axis_name):
    axis_min, axis_max, axis_step = [float(value) for value in range_values]
    if axis_step <= 0:
        raise ValueError(f"{axis_name} step must be positive.")
    if axis_max < axis_min:
        raise ValueError(f"{axis_name} max must be >= min.")
    num_steps = (axis_max - axis_min) / axis_step
    if not np.isclose(num_steps, round(num_steps)):
        raise ValueError(f"{axis_name} range must be divisible by its step.")

    axis_values = (
        axis_min + np.arange(int(round(num_steps)) + 1, dtype=float) * axis_step
    )
    rounded_axis_values = np.round(axis_values)

    if not np.all(np.isclose(axis_values, rounded_axis_values)):
        raise ValueError(
            f"{axis_name} must align with integer simulation steps. "
            "For 0.5 h increments at 20 steps/h, use a step of 10."
        )

    return rounded_axis_values.astype(np.int32)


def build_sweep_params(tau_on_values, tau_off_values, num_replicates):
    sweep_params = []
    seed = 0

    for tau_on in tau_on_values:
        for tau_off in tau_off_values:
            for replicate in range(num_replicates):
                sweep_params.append(
                    (int(tau_on), int(tau_off), int(replicate), int(seed))
                )
                seed += 1

    return sweep_params


def init_memmaps(save_folder, num_sim):
    os.makedirs(save_folder, exist_ok=True)

    efficacy = np.memmap(
        os.path.join(save_folder, "last_cycle_start_efficacy.dat"),
        dtype=np.float32,
        mode="w+",
        shape=(num_sim,),
    )
    efficacy[:] = np.nan
    efficacy.flush()

    mean_efficacy = np.memmap(
        os.path.join(save_folder, "last_cycle_mean_efficacy.dat"),
        dtype=np.float32,
        mode="w+",
        shape=(num_sim,),
    )
    mean_efficacy[:] = np.nan
    mean_efficacy.flush()

    status = np.memmap(
        os.path.join(save_folder, "status.dat"),
        dtype=np.bool_,
        mode="w+",
        shape=(num_sim,),
    )
    status[:] = False
    status.flush()


def write_metadata(save_folder, params_path, params, duration, tau_on_values, tau_off_values, num_replicates, noise, post_noise, sweep_params):
    np.save(
        os.path.join(save_folder, "params_list.npy"),
        np.asarray(sweep_params, dtype=np.int32),
    )
    np.savez(
        os.path.join(save_folder, "sweep_axes.npz"),
        tau_on=tau_on_values,
        tau_off=tau_off_values,
    )

    metadata = {
        "params_path": params_path,
        "duration": int(duration),
        "treatment_start": int(params["treatment_start"]),
        "num_replicates": int(num_replicates),
        "noise": bool(noise),
        "post_noise": bool(post_noise),
        "save_metrics": [
            "last cycle-start treatment efficacy",
            "mean treatment efficacy over the last full tau_on + tau_off cycle",
        ],
        "cycle_definition": "one full cycle is tau_on + tau_off",
        "tau_on_min": int(tau_on_values[0]),
        "tau_on_max": int(tau_on_values[-1]),
        "tau_on_step": (
            int(tau_on_values[1] - tau_on_values[0]) if len(tau_on_values) > 1 else 0
        ),
        "tau_off_min": int(tau_off_values[0]),
        "tau_off_max": int(tau_off_values[-1]),
        "tau_off_step": (
            int(tau_off_values[1] - tau_off_values[0]) if len(tau_off_values) > 1 else 0
        ),
    }
    with open(os.path.join(save_folder, "metadata.yaml"), "w") as file:
        yaml.safe_dump(metadata, file, sort_keys=False)


def worker(item, params=None, save_folder=None, num_sim=None, noise=False, post_noise=False):
    idx, sweep_params = item
    tau_on, tau_off, _, seed = sweep_params

    last_cycle_start_efficacy, last_cycle_mean_efficacy = (
        treat_core.calc_last_cycle_metrics(
            tau_on,
            tau_off,
            params,
            noise=noise,
            random_seed=seed if noise else None,
            post_noise=post_noise,
        )
    )

    efficacy = np.memmap(
        os.path.join(save_folder, "last_cycle_start_efficacy.dat"),
        dtype=np.float32,
        mode="r+",
        shape=(num_sim,),
    )
    efficacy[idx] = np.float32(last_cycle_start_efficacy)
    efficacy.flush()

    mean_efficacy = np.memmap(
        os.path.join(save_folder, "last_cycle_mean_efficacy.dat"),
        dtype=np.float32,
        mode="r+",
        shape=(num_sim,),
    )
    mean_efficacy[idx] = np.float32(last_cycle_mean_efficacy)
    mean_efficacy.flush()

    status = np.memmap(
        os.path.join(save_folder, "status.dat"),
        dtype=np.bool_,
        mode="r+",
        shape=(num_sim,),
    )
    status[idx] = True
    status.flush()

    return idx


def run_treatment_sweep(duration, tau_on_range, tau_off_range, num_replicates, save_folder=None, noise=False, post_noise=False, num_cpus=None, params_path=None, job_id=0, num_jobs=1):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = (
        treat_core.find_project_root(current_dir, "requirements.txt") or os.getcwd()
    )
    config = get_treatment_sweep_config()

    if post_noise and not noise:
        raise ValueError("post_noise requires noise=True.")
    if num_replicates < 1:
        raise ValueError("num_replicates must be at least 1.")
    if duration <= 1:
        raise ValueError("duration must be greater than 1.")
    if num_replicates > 1 and not noise:
        print("Running repeated deterministic replicates because noise=False.")

    if params_path is None:
        params_path = os.path.join(project_root, "params.yaml")
    if save_folder is None:
        save_folder = config["default_save_folder"]

    params_path = resolve_path(project_root, params_path)
    save_folder = resolve_path(project_root, save_folder)

    params = load_params(params_path)
    params["total_time"] = int(duration)

    tau_on_values = build_duration_axis(tau_on_range, "tau_on")
    tau_off_values = build_duration_axis(tau_off_range, "tau_off")
    params_list = build_sweep_params(tau_on_values, tau_off_values, num_replicates)
    num_sim = len(params_list)
    print(f"Number of treatment-only simulations: {num_sim}")

    if job_id == 0:
        init_memmaps(save_folder, num_sim)
        write_metadata(
            save_folder,
            params_path,
            params,
            duration,
            tau_on_values,
            tau_off_values,
            num_replicates,
            noise,
            post_noise,
            params_list,
        )
    else:
        check_init = (
            os.path.exists(os.path.join(save_folder, "last_cycle_start_efficacy.dat"))
            and os.path.exists(
                os.path.join(save_folder, "last_cycle_mean_efficacy.dat")
            )
            and os.path.exists(os.path.join(save_folder, "status.dat"))
        )
        while not check_init:
            time.sleep(1)
            check_init = (
                os.path.exists(
                    os.path.join(save_folder, "last_cycle_start_efficacy.dat")
                )
                and os.path.exists(
                    os.path.join(save_folder, "last_cycle_mean_efficacy.dat")
                )
                and os.path.exists(os.path.join(save_folder, "status.dat"))
            )

    status = np.memmap(
        os.path.join(save_folder, "status.dat"),
        dtype=np.bool_,
        mode="r+",
        shape=(num_sim,),
    )
    undone = np.nonzero(status == False)[0]
    missing_idxs = [i for i in undone if i % num_jobs == job_id]
    jobs = [(i, params_list[i]) for i in missing_idxs]

    default_cpus = max(1, mp.cpu_count() - 1)
    num_cpus = num_cpus or int(os.environ.get("SLURM_CPUS_PER_TASK", default_cpus))

    worker_with_args = ft.partial(
        worker,
        params=params,
        save_folder=save_folder,
        num_sim=num_sim,
        noise=noise,
        post_noise=post_noise,
    )

    with mp.Pool(processes=num_cpus) as pool:
        for _ in tqdm.tqdm(pool.imap(worker_with_args, jobs), total=len(jobs)):
            pass


def main():
    config = get_treatment_sweep_config()

    run_treatment_sweep(
        duration=config["duration"],
        tau_on_range=config["tau_on_range"],
        tau_off_range=config["tau_off_range"],
        num_replicates=config["num_replicates"],
        save_folder=config["save_folder"],
        noise=config["noise"],
        post_noise=config["post_noise"],
        num_cpus=config["num_cpus"],
        params_path=config["params_path"],
    )


if __name__ == "__main__":
    main()
