import copy
import functools as ft
import multiprocessing as mp
import os
import sys
import time
import warnings
import spatial_treatment_core as stc
import numpy as np
import tqdm
import yaml
import matplotlib as mpl
import matplotlib.pyplot as plt


def current_dir():
    return os.path.dirname(os.path.abspath(__file__))


if current_dir() not in sys.path:
    sys.path.insert(0, current_dir())


def spatial_sweep_defaults():
    return {"save_folder": os.path.join("data", "sweeps", "spatial_treatment_0_gradient"), 
            "output_folder": os.path.join("Spatial_treatment", "plots"), 
            "ttp_threshold": 71.0, 
            "steps_per_hour": 20.0, 
            "ratio_metric": "ttp"}


def load_spatial_sweep_config(config_path=None):
    config = spatial_sweep_defaults()
    if config_path is None:
        config_path = os.path.join(current_dir(), "spatial_treatment_config.yaml")
    with open(config_path, "r", encoding="utf-8") as file:
        loaded_config = yaml.safe_load(file) or {}
    config.update(loaded_config)
    return config


def setup_plotting_params():
    plt.rcParams.update({"font.size": 7,
                         "pdf.fonttype": 42,
                         "font.family": "sans-serif",
                         "font.sans-serif": ["Arial"],
                         "mathtext.fontset": "custom",
                         "mathtext.rm": "Arial",
                         "mathtext.it": "Arial:italic",
                         "mathtext.bf": "Arial:bold"})
    plt.rcParams["axes.labelsize"] = 7
    plt.rcParams["xtick.labelsize"] = 6
    plt.rcParams["ytick.labelsize"] = 6


def build_duration_axis(range_values, axis_name):
    axis_min, axis_max, axis_step = [float(value) for value in range_values]
    if axis_step <= 0:
        raise ValueError(f"{axis_name} step must be positive.")
    if axis_max < axis_min:
        raise ValueError(f"{axis_name} max must be >= min.")

    num_steps = (axis_max - axis_min) / axis_step
    if not np.isclose(num_steps, round(num_steps)):
        raise ValueError(f"{axis_name} range must be divisible by its step.")

    axis_values = (axis_min + np.arange(int(round(num_steps)) + 1, dtype=float) * axis_step)
    rounded_axis_values = np.round(axis_values)
    if not np.all(np.isclose(axis_values, rounded_axis_values)):
        raise ValueError(f"{axis_name} must align with integer simulation steps. For 0.5 h increments at 20 steps/h, use a step of 10.")

    return rounded_axis_values.astype(np.int32)


def build_float_axis(range_values, axis_name):
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
    return np.round(axis_values, decimals=8).astype(np.float32)


def build_mutation_rate_values(params, mutation_rate_values=None):
    if mutation_rate_values is not None:
        mutation_rate_values = np.asarray(mutation_rate_values, dtype=np.float32)
        if mutation_rate_values.ndim == 0:
            mutation_rate_values = mutation_rate_values.reshape(1)
        if np.any(mutation_rate_values < 0.0):
            raise ValueError("mutation_rate_values must be >= 0.")
        return mutation_rate_values

    return np.asarray([float(params["mutation_rate"])], dtype=np.float32)


def build_sweep_params(mutation_rate_values, tau_on_values, tau_off_values, num_replicates):
    sweep_params = []
    seed = 0

    for mutation_rate in mutation_rate_values:
        for tau_on in tau_on_values:
            for tau_off in tau_off_values:
                for replicate in range(int(num_replicates)):
                    sweep_params.append((float(mutation_rate), int(tau_on), int(tau_off), int(replicate), int(seed)))
                    seed += 1

    return sweep_params


def init_memmaps(save_folder, num_sim, total_time):
    os.makedirs(save_folder, exist_ok=True)

    efficacy = np.memmap(os.path.join(save_folder, "efficacy.dat"), dtype=np.float32, mode="w+", shape=(num_sim, total_time))
    efficacy[:] = np.nan
    efficacy.flush()

    mean_local_efficacy = np.memmap(os.path.join(save_folder, "mean_local_efficacy.dat"), dtype=np.float32, mode="w+", shape=(num_sim, total_time))
    mean_local_efficacy[:] = np.nan
    mean_local_efficacy.flush()

    treatment_times = np.memmap(os.path.join(save_folder, "treatment_times.dat"), dtype=np.bool_, mode="w+", shape=(num_sim, total_time))
    treatment_times[:] = False
    treatment_times.flush()

    size = np.memmap(os.path.join(save_folder, "size.dat"), dtype=np.float32, mode="w+", shape=(num_sim, total_time))
    size[:] = np.nan
    size.flush()

    ratio = np.memmap(os.path.join(save_folder, "ratio.dat"), dtype=np.float32, mode="w+", shape=(num_sim, total_time))
    ratio[:] = np.nan
    ratio.flush()

    status = np.memmap(os.path.join(save_folder, "status.dat"), dtype=np.bool_, mode="w+", shape=(num_sim,))
    status[:] = False
    status.flush()


def write_metadata(save_folder, params_path, params, mutation_rate_values, tau_on_values, tau_off_values, num_replicates, sweep_params, inverse_gradient_decay_length, gradient_mode, gradient_floor, occupancy_threshold):
    np.save(os.path.join(save_folder, "params_list.npy"), np.asarray(sweep_params, dtype=np.float64))
    np.savez(os.path.join(save_folder, "sweep_axes.npz"), mutation_rate=np.asarray(mutation_rate_values, dtype=np.float32), tau_on=np.asarray(tau_on_values, dtype=np.int32), tau_off=np.asarray(tau_off_values, dtype=np.int32))

    with open(os.path.join(save_folder, "params_snapshot.yaml"), "w", encoding="utf-8") as file:
        yaml.safe_dump(params, file, sort_keys=False)

    metadata = {"params_path": params_path,
                "total_time": int(params["total_time"]),
                "start_point": int(params["start_point"]),
                "treatment_start": int(params["treatment_start"]),
                "num_replicates": int(num_replicates),
                "mutations_active": bool(params["mutations_active"]),
                "sweep_axes": ["mutation_rate", "tau_on", "tau_off"],
                "tau_on_min": int(tau_on_values[0]),
                "tau_on_max": int(tau_on_values[-1]),
                "tau_on_step": int(tau_on_values[1] - tau_on_values[0]) if len(tau_on_values) > 1 else 0,
                "tau_off_min": int(tau_off_values[0]),
                "tau_off_max": int(tau_off_values[-1]),
                "tau_off_step": int(tau_off_values[1] - tau_off_values[0]) if len(tau_off_values) > 1 else 0,
                "mutation_rate_min": float(mutation_rate_values[0]),
                "mutation_rate_max": float(mutation_rate_values[-1]),
                "mutation_rate_step": float(mutation_rate_values[1] - mutation_rate_values[0]) if len(mutation_rate_values) > 1 else 0.0,
                "spatial_treatment_mode": "front_decay_gradient",
                "inverse_gradient_decay_length_px": float(inverse_gradient_decay_length),
                "gradient_mode": str(gradient_mode),
                "gradient_floor": float(gradient_floor),
                "occupancy_threshold": float(occupancy_threshold) if occupancy_threshold is not None else float(1.0 / float(params["mutation_scaling"])),
                "ttp_size_threshold": float(spatial_sweep_defaults()["ttp_threshold"]),
                "saved_outputs": ["efficacy.dat",
                                  "mean_local_efficacy.dat",
                                  "treatment_times.dat",
                                  "size.dat",
                                  "ratio.dat",
                                  "status.dat"]}
    
    with open(os.path.join(save_folder, "metadata.yaml"), "w", encoding="utf-8") as file:
        yaml.safe_dump(metadata, file, sort_keys=False)


def worker(item, params, save_folder, num_sim, inverse_gradient_decay_length, gradient_mode, gradient_floor, occupancy_threshold):
    idx, sweep_params = item
    mutation_rate, tau_on, tau_off, replicate, seed = sweep_params

    local_params = copy.deepcopy(params)
    local_params["mutation_rate"] = float(mutation_rate)
    local_params["replicate"] = int(replicate)
    local_params["save_folder"] = save_folder

    treatment_times, treatment_efficacy, mean_local_efficacy, sizes, ratios = stc.run_spatial_treatment_simulation(tau_on, tau_off, params=local_params, random_seed=seed, inverse_gradient_decay_length=inverse_gradient_decay_length, gradient_mode=gradient_mode, gradient_floor=gradient_floor, occupancy_threshold=occupancy_threshold)

    total_time = int(local_params["total_time"])

    efficacy_mmap = np.memmap(os.path.join(save_folder, "efficacy.dat"), dtype=np.float32, mode="r+", shape=(num_sim, total_time))
    efficacy_mmap[idx] = treatment_efficacy
    efficacy_mmap.flush()

    mean_local_efficacy_mmap = np.memmap(os.path.join(save_folder, "mean_local_efficacy.dat"), dtype=np.float32, mode="r+", shape=(num_sim, total_time))
    mean_local_efficacy_mmap[idx] = mean_local_efficacy
    mean_local_efficacy_mmap.flush()

    treatment_times_mmap = np.memmap(os.path.join(save_folder, "treatment_times.dat"), dtype=np.bool_, mode="r+", shape=(num_sim, total_time))
    treatment_times_mmap[idx] = treatment_times
    treatment_times_mmap.flush()

    size_mmap = np.memmap(os.path.join(save_folder, "size.dat"), dtype=np.float32, mode="r+", shape=(num_sim, total_time))
    size_mmap[idx] = sizes
    size_mmap.flush()

    ratio_mmap = np.memmap(os.path.join(save_folder, "ratio.dat"), dtype=np.float32, mode="r+", shape=(num_sim, total_time))
    ratio_mmap[idx] = ratios
    ratio_mmap.flush()

    status = np.memmap(os.path.join(save_folder, "status.dat"), dtype=np.bool_, mode="r+", shape=(num_sim,))
    status[idx] = True
    status.flush()

    return idx


def run_spatial_treatment_sweep(mutation_rate_values=None, tau_on_range=None, tau_off_range=None, num_replicates=None, save_folder=None, num_cpus=None, params_path=None, duration=None, inverse_gradient_decay_length=None, gradient_mode=None, gradient_floor=None, occupancy_threshold=None, job_id=0, num_jobs=1):
    if params_path is None:
        params_path = os.path.join(stc.project_root(), "params.yaml")
    if save_folder is None:
        save_folder = spatial_sweep_defaults()["save_folder"]
    treatment_defaults = stc.spatial_treatment_defaults()
    if inverse_gradient_decay_length is None:
        inverse_gradient_decay_length = treatment_defaults["inverse_gradient_decay_length"]
    if gradient_mode is None:
        gradient_mode = treatment_defaults["gradient_mode"]
    if gradient_floor is None:
        gradient_floor = treatment_defaults["gradient_floor"]

    params_path = stc.resolve_path(stc.project_root(), params_path)
    save_folder = stc.resolve_path(stc.project_root(), save_folder)
    params = stc.load_params(params_path)

    if duration is not None:
        if int(duration) <= 1:
            raise ValueError("duration must be greater than 1.")
        params["total_time"] = int(duration)
    else:
        params["total_time"] = int(params["total_time"])

    if tau_on_range is None:
        tau_on_range = (params["treatment_on_min"], params["treatment_on_max"], params["treatment_on_step"])
    if tau_off_range is None:
        tau_off_range = (params["treatment_off_min"], params["treatment_off_max"], params["treatment_off_step"])

    tau_on_values = build_duration_axis(tau_on_range, "tau_on")
    tau_off_values = build_duration_axis(tau_off_range, "tau_off")

    if num_replicates is None:
        num_replicates = int(params["num_replicas"])
    num_replicates = int(num_replicates)
    if num_replicates < 1:
        raise ValueError("num_replicates must be at least 1.")
    if int(num_jobs) < 1:
        raise ValueError("num_jobs must be at least 1.")

    mutation_rate_values = build_mutation_rate_values(params, mutation_rate_values=mutation_rate_values)
    sweep_params = build_sweep_params(mutation_rate_values, tau_on_values, tau_off_values, num_replicates)
    num_sim = len(sweep_params)
    total_time = int(params["total_time"])

    print(f"Number of spatial-treatment simulations: {num_sim}")
    if num_replicates > 1 and not bool(params["mutations_active"]):
        print("Running repeated deterministic replicates because mutations_active=False.")

    if int(job_id) == 0:
        init_memmaps(save_folder, num_sim, total_time)
        write_metadata(save_folder, params_path, params, mutation_rate_values, tau_on_values, tau_off_values, num_replicates, sweep_params, inverse_gradient_decay_length, gradient_mode, gradient_floor, occupancy_threshold)
    else:
        while not (os.path.exists(os.path.join(save_folder, "efficacy.dat")) and os.path.exists(os.path.join(save_folder, "mean_local_efficacy.dat")) and os.path.exists(os.path.join(save_folder, "treatment_times.dat")) and os.path.exists(os.path.join(save_folder, "size.dat")) and os.path.exists(os.path.join(save_folder, "ratio.dat")) and os.path.exists(os.path.join(save_folder, "status.dat"))):
            time.sleep(1)

    status = np.memmap(os.path.join(save_folder, "status.dat"), dtype=np.bool_, mode="r+", shape=(num_sim,))
    undone = np.nonzero(status == False)[0]
    missing_idxs = [idx for idx in undone if idx % int(num_jobs) == int(job_id)]
    jobs = [(idx, sweep_params[idx]) for idx in missing_idxs]

    default_cpus = max(1, mp.cpu_count() - 1)
    if num_cpus is None:
        num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", default_cpus))
    num_cpus = max(1, int(num_cpus))

    worker_with_args = ft.partial(worker, params=params, save_folder=save_folder, num_sim=num_sim, inverse_gradient_decay_length=inverse_gradient_decay_length, gradient_mode=gradient_mode, gradient_floor=gradient_floor, occupancy_threshold=occupancy_threshold)

    if jobs:
        with mp.Pool(processes=num_cpus) as pool:
            for _ in tqdm.tqdm(pool.imap(worker_with_args, jobs), total=len(jobs)):
                pass

    return save_folder


def _float_key(value):
    return round(float(value), 8)


def compute_ttp_and_ratio(size_trace, ratio_trace, start_point, steps_per_hour, ttp_threshold):
    progression_indices = np.flatnonzero(np.asarray(size_trace) >= float(ttp_threshold))
    if progression_indices.size > 0:
        ttp_index = int(progression_indices[0])
    else:
        ttp_index = len(size_trace) - 1

    ttp_hours = (ttp_index - float(start_point)) / float(steps_per_hour)
    ratio_at_ttp = float(ratio_trace[ttp_index])
    ratio_endpoint = float(ratio_trace[-1])
    return ttp_hours, ratio_at_ttp, ratio_endpoint


def load_spatial_treatment_metrics(save_folder, steps_per_hour=None, ttp_threshold=None):
    if steps_per_hour is None:
        steps_per_hour = spatial_sweep_defaults()["steps_per_hour"]
    if ttp_threshold is None:
        ttp_threshold = spatial_sweep_defaults()["ttp_threshold"]

    params_list = np.load(os.path.join(save_folder, "params_list.npy"))
    sweep_axes = np.load(os.path.join(save_folder, "sweep_axes.npz"))

    with open(os.path.join(save_folder, "metadata.yaml"), "r", encoding="utf-8") as file:
        metadata = yaml.safe_load(file)

    mutation_rate_values = np.asarray(sweep_axes["mutation_rate"], dtype=np.float32)
    tau_on_values = np.asarray(sweep_axes["tau_on"], dtype=np.int32)
    tau_off_values = np.asarray(sweep_axes["tau_off"], dtype=np.int32)
    num_replicates = int(metadata["num_replicates"])
    total_time = int(metadata["total_time"])
    start_point = int(metadata["start_point"])
    num_sim = len(params_list)

    size = np.memmap(os.path.join(save_folder, "size.dat"), dtype=np.float32, mode="r", shape=(num_sim, total_time))
    ratio = np.memmap(os.path.join(save_folder, "ratio.dat"), dtype=np.float32, mode="r", shape=(num_sim, total_time))
    status = np.memmap(os.path.join(save_folder, "status.dat"), dtype=np.bool_, mode="r", shape=(num_sim,))

    mutation_index = {_float_key(value): idx for idx, value in enumerate(mutation_rate_values)}
    tau_on_index = {int(value): idx for idx, value in enumerate(tau_on_values)}
    tau_off_index = {int(value): idx for idx, value in enumerate(tau_off_values)}

    grid_shape = (len(mutation_rate_values), len(tau_on_values), len(tau_off_values), num_replicates)
    ttp_grid = np.full(grid_shape, np.nan, dtype=np.float32)
    ratio_ttp_grid = np.full_like(ttp_grid, np.nan)
    ratio_endpoint_grid = np.full_like(ttp_grid, np.nan)

    for idx, row in enumerate(params_list):
        mutation_rate, tau_on, tau_off, replicate, _ = row
        if not status[idx]:
            continue

        mutation_idx = mutation_index[_float_key(mutation_rate)]
        tau_on_idx = tau_on_index[int(round(tau_on))]
        tau_off_idx = tau_off_index[int(round(tau_off))]
        replicate_idx = int(round(replicate))

        ttp_hours, ratio_at_ttp, ratio_endpoint = compute_ttp_and_ratio(size[idx], ratio[idx], start_point=start_point, steps_per_hour=steps_per_hour, ttp_threshold=ttp_threshold)
        ttp_grid[mutation_idx, tau_on_idx, tau_off_idx, replicate_idx] = ttp_hours
        ratio_ttp_grid[mutation_idx, tau_on_idx, tau_off_idx, replicate_idx] = ratio_at_ttp
        ratio_endpoint_grid[mutation_idx, tau_on_idx, tau_off_idx, replicate_idx] = ratio_endpoint

    if not np.all(status):
        missing = int(np.count_nonzero(~status))
        print(f"Warning: {missing} spatial-treatment sweep entries are incomplete and will be plotted as NaN.")

    return mutation_rate_values, tau_on_values, tau_off_values, ttp_grid, ratio_ttp_grid, ratio_endpoint_grid, metadata


def summarize_metric_grid(metric_grid):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(metric_grid, axis=-1)
        q1 = np.nanpercentile(metric_grid, 25, axis=-1)
        q3 = np.nanpercentile(metric_grid, 75, axis=-1)

    return median, q1, q3


def format_hour_value(value):
    if np.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:.1f}".rstrip("0").rstrip(".")


def build_tick_positions(axis_values, steps_per_hour, max_ticks=6):
    if len(axis_values) <= max_ticks:
        positions = np.arange(len(axis_values), dtype=int)
    else:
        positions = np.unique(np.round(np.linspace(0, len(axis_values) - 1, max_ticks)).astype(int))

    labels = [format_hour_value(float(axis_values[position]) / float(steps_per_hour)) for position in positions]
    return positions, labels


def compute_max_line(ttp_plane):
    max_line = np.full(ttp_plane.shape[1], np.nan, dtype=np.float32)
    for col_idx in range(ttp_plane.shape[1]):
        column = ttp_plane[:, col_idx]
        if np.all(np.isnan(column)):
            continue
        max_line[col_idx] = float(np.nanargmax(column))
    return max_line


def format_filename_value(value):
    return format(value, ".6g").replace("-", "m").replace(".", "p")


def plot_spatial_treatment_phase_space(save_folder, output_folder=None, show=False, ratio_metric=None, steps_per_hour=None, ttp_threshold=None):
    if output_folder is None:
        output_folder = spatial_sweep_defaults()["output_folder"]
    if ratio_metric is None:
        ratio_metric = spatial_sweep_defaults()["ratio_metric"]
    if steps_per_hour is None:
        steps_per_hour = spatial_sweep_defaults()["steps_per_hour"]
    if ttp_threshold is None:
        ttp_threshold = spatial_sweep_defaults()["ttp_threshold"]

    mutation_rate_values, tau_on_values, tau_off_values, ttp_grid, ratio_ttp_grid, ratio_endpoint_grid, metadata = load_spatial_treatment_metrics(save_folder, steps_per_hour=steps_per_hour, ttp_threshold=ttp_threshold)
    setup_plotting_params()

    if ratio_metric == "ttp":
        ratio_grid = ratio_ttp_grid
        ratio_label = "Resistant fraction"
    elif ratio_metric == "endpoint":
        ratio_grid = ratio_endpoint_grid
        ratio_label = "Final resistant fraction"
    else:
        raise ValueError("ratio_metric must be either 'ttp' or 'endpoint'.")

    os.makedirs(output_folder, exist_ok=True)

    ttp_cmap = mpl.colors.LinearSegmentedColormap.from_list("black_green_white",[(0.0, (0.0, 0.0, 0.0)),
                                                                                 (0.35, (0.0, 128 / 255, 0.0)),
                                                                                 (1.0, (1.0, 1.0, 1.0))], N=256)
    ratio_cmap = mpl.colors.LinearSegmentedColormap.from_list("blue_yellow",[(65 / 255, 105 / 255, 225 / 255), 
                                                                             (218 / 255, 165 / 255, 32 / 255)], N=256)

    x_positions, x_labels = build_tick_positions(tau_off_values, steps_per_hour)
    y_positions, y_labels = build_tick_positions(tau_on_values, steps_per_hour)

    output_paths = []
    for mutation_idx, mutation_rate in enumerate(mutation_rate_values):
        ttp_median, _, _ = summarize_metric_grid(ttp_grid[mutation_idx])
        ratio_median, _, _ = summarize_metric_grid(ratio_grid[mutation_idx])

        if np.all(np.isnan(ttp_median)) or np.all(np.isnan(ratio_median)):
            print(f"Skipping mutation rate {mutation_rate} because all plotted values are NaN.")
            continue

        max_line = compute_max_line(ttp_median)
        valid_max_line = np.isfinite(max_line)

        fig, axes = plt.subplots(1, 2, figsize=(3.72, 1.24), dpi=300)

        im0 = axes[0].imshow(ttp_median, interpolation="none", cmap=ttp_cmap, origin="lower", vmin=float(75), vmax=float(215))
        if np.any(valid_max_line):
            axes[0].plot(np.nonzero(valid_max_line)[0][16:], max_line[valid_max_line][16:], color="black", linestyle=":", linewidth=1.0, label=r"$\tau^*$")
            axes[0].legend(loc="upper left", frameon=False, fontsize=6)

        axes[0].set_xlabel(r"$\tau_{\mathrm{off}}$ (h)")
        axes[0].set_ylabel(r"$\tau_{\mathrm{on}}$ (h)")
        axes[0].set_xticks(x_positions)
        axes[0].set_xticklabels(x_labels)
        axes[0].set_yticks(y_positions)
        axes[0].set_yticklabels(y_labels)
        cbar0 = plt.colorbar(im0, ax=axes[0], pad=0.035, shrink=0.56)
        cbar0.set_label("TTP (h)", rotation=270, labelpad=10)

        im1 = axes[1].imshow(ratio_median, interpolation="none", cmap=ratio_cmap, origin="lower", vmin=0.0, vmax=1.0)
        if np.any(valid_max_line):
            axes[1].plot(np.nonzero(valid_max_line)[0][16:], max_line[valid_max_line][16:], color="white", linestyle=":", linewidth=1.0)

        axes[1].set_xlabel(r"$\tau_{\mathrm{off}}$ (h)")
        axes[1].set_ylabel(r"$\tau_{\mathrm{on}}$ (h)")
        axes[1].set_xticks(x_positions)
        axes[1].set_xticklabels(x_labels)
        axes[1].set_yticks(y_positions)
        axes[1].set_yticklabels(y_labels)
        cbar1 = plt.colorbar(im1, ax=axes[1], pad=0.035, shrink=0.56)
        cbar1.set_label(ratio_label, rotation=270, labelpad=10)

        # fig.suptitle(f"Spatial front-gradient treatment\n $\\mu$ = {float(mutation_rate):.4g}, decay = {float(metadata['inverse_gradient_decay_length_px']):.2f} px", y=1.02, fontsize=8)
        plt.tight_layout()

        base_name = (f"spatial_treatment_04_phase_space_mu_{format_filename_value(float(mutation_rate))}")
        output_path = os.path.join(output_folder, f"{base_name}.pdf")
        fig.savefig(output_path, transparent=True, bbox_inches="tight")
        output_paths.append(output_path)

        if show:
            plt.show()
        else:
            plt.close(fig)

    return output_paths


def wait_for_spatial_treatment_completion(save_folder, poll_seconds=1.0):
    params_list_path = os.path.join(save_folder, "params_list.npy")
    status_path = os.path.join(save_folder, "status.dat")

    while not (os.path.exists(params_list_path) and os.path.exists(status_path)):
        time.sleep(float(poll_seconds))

    num_sim = len(np.load(params_list_path, mmap_mode="r"))
    while True:
        status = np.memmap(status_path, dtype=np.bool_, mode="r", shape=(num_sim,))
        if np.all(status):
            return
        time.sleep(float(poll_seconds))


def main():
    config = load_spatial_sweep_config()
    job_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    num_jobs = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    # save_folder = run_spatial_treatment_sweep(mutation_rate_values=config["mutation_rate_values"], tau_on_range=config["tau_on_range"], tau_off_range=config["tau_off_range"], num_replicates=config["num_replicates"], save_folder=config["save_folder"], num_cpus=config["num_cpus"], params_path=config["params_path"], duration=config["duration"], inverse_gradient_decay_length=config["inverse_gradient_decay_length"], gradient_mode=config["gradient_mode"], gradient_floor=config["gradient_floor"], occupancy_threshold=config["occupancy_threshold"], job_id=job_id, num_jobs=num_jobs)

    # if int(num_jobs) > 1:
    #     if int(job_id) != 0:
    #         return
    #       wait_for_spatial_treatment_completion(save_folder, poll_seconds=config["wait_poll_seconds"])
    
    save_folder = stc.resolve_path(stc.project_root(), 'data/sweeps/spatial_treatment_04_gradient')

    output_folder = stc.resolve_path(stc.project_root(), config["output_folder"])
    plot_spatial_treatment_phase_space(save_folder, output_folder=output_folder, show=config["show_plot"], ratio_metric=config["ratio_metric"], steps_per_hour=config["steps_per_hour"], ttp_threshold=config["ttp_threshold"])


if __name__ == "__main__":
    main()
