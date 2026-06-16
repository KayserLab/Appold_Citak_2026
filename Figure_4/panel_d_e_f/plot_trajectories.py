import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib as mpl
import skimage.segmentation as seg
from scipy.ndimage import binary_fill_holes, distance_transform_edt

plt.rcParams.update({'font.size': 7,
                     'pdf.fonttype': 42,
                     'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial'],
                     'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Arial',
                     'mathtext.it': 'Arial:italic',
                     'mathtext.bf': 'Arial:bold'})
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6


def calc_treatment_efficacy(treat_on, treat_off, params):
    first_start = params['treatment_start']

    treatment_times = np.zeros(params['total_time'])
    treatment_length = treat_on
    treatment_ends = []
    if treat_off == 0:
        treatment_starts = [first_start]
        treatment_length = params['total_time'] - first_start
        treatment_ends = [params['total_time']]
        if treat_on == 0:
            treatment_starts = []
            treatment_ends = []
    elif treat_on == 0:
        treatment_starts = []
    else:
        treatment_starts = [d for d in range(first_start, params['total_time'], treat_off + treat_on)]
        treatment_ends = np.array(treatment_starts) + treat_on

    for i in range(len(treatment_starts)):
        treatment_times[treatment_starts[i]:(treatment_starts[i] + treatment_length)] = True

    return treatment_starts, treatment_ends


def get_front_ring_mask(total_mask, ring_width_px):
    if ring_width_px <= 0 or not total_mask.any():
        return np.zeros_like(total_mask, dtype=bool)

    filled_mask = binary_fill_holes(total_mask.astype(bool))
    padded_mask = np.pad(filled_mask, 1, mode='constant', constant_values=False)
    distance_to_outside = distance_transform_edt(padded_mask)
    front_ring_mask = (distance_to_outside > 0) & (distance_to_outside <= ring_width_px)
    return front_ring_mask[1:-1, 1:-1] & total_mask


def get_treatment_schedule(treatment):
    schedule_map = {
        'continuous_dose': (10, 0),
        'no_treatment': (0, 0),
        'met_4_18': (4, 18),
        'met_6_5_18': (6.5, 18),
        'met_9_18': (9, 18),
        'met_4_5_18': (4.5, 18),
        'met_5_18': (5, 18),
        'met_5_5_18': (5.5, 18),
        'met_6_18': (6, 18),
        'met_6_25_18': (6.25, 18),
        'met_6_75_18': (6.75, 18),
        'met_7_18': (7, 18),
        'met_7_5_18': (7.5, 18),
        'met_8_18': (8, 18),
        'met_8_5_18': (8.5, 18),
        'met_4_15_5': (4, 15.5),
        'met_4_20_5': (4, 20.5),
        'met_6_5_15_5': (6.5, 15.5),
        'met_6_5_20_5': (6.5, 20.5),
        'met_9_15_5': (9, 15.5),
        'met_9_20_5': (9, 20.5),
    }
    if treatment in schedule_map:
        return schedule_map[treatment]

    parts = treatment.split('_')
    if len(parts) == 3 and parts[0] == 'met':
        return float(parts[1]), float(parts[2])

    raise ValueError(f'Unsupported treatment schedule: {treatment}')


def load_sim_data(path, ring_width_px=4):
    sen = np.load(f'{path}/sensitive.npy')
    res = np.load(f'{path}/resistant.npy')
    params = torch.load(f'{path}/params.pth')

    sen_diff = np.diff(sen * params['mutation_scaling'], axis=0)
    res_diff = np.diff(res * params['mutation_scaling'], axis=0)

    counts = []
    counts_res = []
    counts_res_front_ring = []
    res_total_count = []
    counts_sen = []
    sen_avg_front_den = []
    res_avg_front_den = []
    for i in range(len(sen) - 1):
        sen_thresholded = np.where(sen[i] >= (1 / params['mutation_scaling']), 1, 0)
        res_thresholded = np.where(res[i] >= (1 / params['mutation_scaling']), 1, 0)

        total_array = sen_thresholded + res_thresholded
        total_count = np.count_nonzero(total_array)
        counts.append(total_count)
        counts_sen.append(np.count_nonzero(sen_thresholded))

        sen_thresholded_ratio = np.where(sen[i] > (1 / params['mutation_scaling']), sen[i], 0)
        res_thresholded_ratio = np.where(res[i] > (1 / params['mutation_scaling']), res[i], 0)
        res_ratio = np.where(res_thresholded_ratio > sen_thresholded_ratio, 1., 0.)

        tot_mask = total_array > 0
        # front_ring_mask = get_front_ring_mask(tot_mask, ring_width_px)
        # front_ring_count = np.count_nonzero(front_ring_mask)
        sen_mask = sen_thresholded > 0
        res_mask = res_thresholded > 0

        tot_b = seg.find_boundaries(tot_mask, mode="inner")
        sen_b = seg.find_boundaries(sen_mask, mode="inner")
        res_b = seg.find_boundaries(res_mask, mode="inner")

        front_sen = tot_b & sen_b
        front_res = tot_b & res_b

        sen_avg_front_den.append(sen_diff[i][front_sen].mean() if front_sen.any() else 0)
        res_avg_front_den.append(res_diff[i][front_res].mean() if front_res.any() else 0)

        if total_count >= 1:
            counts_res.append(np.count_nonzero(res_ratio) / total_count)
            res_total_count.append(np.count_nonzero(res_ratio))
        else:
            counts_res.append(0)
            res_total_count.append(0)

        # if front_ring_count >= 1:
        #     counts_res_front_ring.append(np.count_nonzero(res_ratio[front_ring_mask]) / front_ring_count)
        # else:
        #     counts_res_front_ring.append(0)

    scale_factor = (eval(params['sim_pixel_to_exp_pixel_factor'])) ** 2
    return (np.array(counts) * scale_factor,
            np.array(counts_res),
            np.array(res_total_count) * scale_factor,
            np.array(counts_sen) * scale_factor,
            np.array(sen_avg_front_den),
            np.array(res_avg_front_den))


def get_sim_data(replicate, treatment, ring_width_px=4):
    sim_colony_area_temp = []
    sim_clone_area_temp = []
    sim_clone_tot_temp = []
    sim_colony_growth_rate_temp = []
    sim_clone_growth_rate_temp = []
    sim_sen_front_growth_rate_temp = []
    sim_res_front_growth_rate_temp = []
    # sim_front_ring_ratio_temp = []
    for i in range(replicate):
        sim_colony, sim_clone, sim_clone_tot, sim_sen_tot, sen_avg_front, res_avg_front = load_sim_data(
            f'data/sim_data/{treatment}/{treatment}_{i}',
            ring_width_px=ring_width_px,
        )
        sim_colony_area_temp.append(sim_colony)
        sim_clone_area_temp.append(sim_clone)
        sim_clone_tot_temp.append(sim_clone_tot)
        sim_colony_growth_rate_temp.append(np.gradient(rolling_average(sim_sen_tot, window_size=51), 1 / 20))
        sim_clone_growth_rate_temp.append(np.gradient(rolling_average(sim_clone_tot, window_size=51), 1 / 20))
        sim_sen_front_growth_rate_temp.append(rolling_average(sen_avg_front, window_size=51))
        sim_res_front_growth_rate_temp.append(rolling_average(res_avg_front, window_size=51))
        # sim_front_ring_ratio_temp.append(sim_front_ring_ratio)

    sim_colony_area = np.median(np.array(sim_colony_area_temp), axis=0)
    sim_colony_area_iqr = np.percentile(np.array(sim_colony_area_temp), [25, 75], axis=0)
    sim_clone_area = np.median(np.array(sim_clone_area_temp), axis=0)
    sim_clone_area_iqr = np.percentile(np.array(sim_clone_area_temp), [25, 75], axis=0)
    sim_clone_tot_area = np.median(np.array(sim_clone_tot_temp), axis=0)
    sim_clone_tot_area_iqr = np.percentile(np.array(sim_clone_tot_temp), [25, 75], axis=0)
    sim_colony_growth_rate = np.median(np.array(sim_colony_growth_rate_temp), axis=0)
    sim_clone_growth_rate = np.median(np.array(sim_clone_growth_rate_temp), axis=0)
    sim_colony_growth_rate_iqr = np.percentile(np.array(sim_colony_growth_rate_temp), [25, 75], axis=0)
    sim_clone_growth_rate_iqr = np.percentile(np.array(sim_clone_growth_rate_temp), [25, 75], axis=0)
    sim_sen_front_growth_rate = np.median(np.array(sim_sen_front_growth_rate_temp), axis=0)
    sim_res_front_growth_rate = np.median(np.array(sim_res_front_growth_rate_temp), axis=0)
    sim_sen_front_growth_rate_iqr = np.percentile(np.array(sim_sen_front_growth_rate_temp), [25, 75], axis=0)
    sim_res_front_growth_rate_iqr = np.percentile(np.array(sim_res_front_growth_rate_temp), [25, 75], axis=0)
    # sim_front_ring_ratio = np.median(np.array(sim_front_ring_ratio_temp), axis=0)
    # sim_front_ring_ratio_iqr = np.percentile(np.array(sim_front_ring_ratio_temp), [25, 75], axis=0)

    return (sim_colony_area, sim_colony_area_iqr, sim_clone_area, sim_clone_area_iqr, sim_clone_tot_area, sim_clone_tot_area_iqr, sim_colony_growth_rate,
            sim_colony_growth_rate_iqr, sim_clone_growth_rate, sim_clone_growth_rate_iqr, sim_sen_front_growth_rate, sim_sen_front_growth_rate_iqr,
            sim_res_front_growth_rate, sim_res_front_growth_rate_iqr)


def rolling_median(data, window_size):
    padded_data = np.pad(data, (window_size // 2, window_size - window_size // 2 - 1), mode='edge')
    rolling_medians = np.array([np.median(padded_data[i:i + window_size]) for i in range(len(data))])
    return rolling_medians


def rolling_average(data, window_size):
    left = window_size // 2
    right = window_size - left - 1
    padded_data = np.pad(data, (left, right), mode="edge")
    kernel = np.ones(window_size, dtype=float) / window_size

    return np.convolve(padded_data, kernel, mode="valid")

def plot_sim(ax, x_sim, sim_area, sim_iqr, color, start_point, clone, treat_starts_test, treat_ends_test, linestyle='solid', label=None):
    scaling_factor = 8.648 ** 2 / 1e6
    if clone:
        scaling_factor = 1
    ax.plot(x_sim, rolling_average(sim_area * scaling_factor, window_size=51)[start_point:3001 + start_point], color=color, linestyle=linestyle, label=label)
    ax.fill_between(x_sim,
                    sim_iqr[0][start_point:3001 + start_point] * scaling_factor,
                    sim_iqr[1][start_point:3001 + start_point] * scaling_factor,
                    color=color, alpha=0.25, lw=0)
    for i in range(len(treat_starts_test)):
        ax.axvspan(treat_starts_test[i] / 20, treat_ends_test[i] / 20, color='#bfbfbf', alpha=1, lw=0, zorder=0)


def plot_comparison(replicate, treatment, color, ring_width_px=4):
    fig1, ax1 = plt.subplots(figsize=(8.4 / 3, 6.5 / 6.5), dpi=300)
    fig2, ax2 = plt.subplots(figsize=(8.4 / 3, 6.5 / 6.5), dpi=300)
    fig3, ax3 = plt.subplots(figsize=(8.4 / 3, 6.5 / 6.5), dpi=300)
    fig4, ax4 = plt.subplots(figsize=(8.4 / 3, 6.5 / 6.5), dpi=300)
    # fig5, ax5 = plt.subplots(figsize=(8.4 / 3, 6.5 / 6.5), dpi=300)

    params = torch.load(f'data/sim_data/{treatment}/{treatment}_0/params.pth')
    treat_on, treat_off = get_treatment_schedule(treatment)
    treat_starts_test, treat_ends_test = calc_treatment_efficacy(int(treat_on * 20), int(treat_off * 20), params)
    (sim_colony_area, sim_colony_area_iqr, sim_clone_area, sim_clone_area_iqr, sim_clone_tot_area, sim_clone_tot_area_iqr,
     sen_growth_rate, sen_growth_rate_iqr, res_growth_rate, res_growth_rate_iqr, sim_sen_front_growth_rate,
     sim_sen_front_growth_rate_iqr, sim_res_front_growth_rate, sim_res_front_growth_rate_iqr) = get_sim_data(replicate, treatment, ring_width_px=ring_width_px)

    start_point = params['start_point']
    x_sim = np.linspace(0, 3000, 3001) / 20
    if len(np.argwhere(sim_colony_area * 8.648 ** 2 / 1e6 >= 71)) > 0:
        print(np.argwhere(sim_colony_area * 8.648 ** 2 / 1e6 >= 71)[0])
    else:
        print('No limit reached')

    plot_sim(ax1, x_sim, sim_colony_area, sim_colony_area_iqr, color, start_point, clone=False, treat_starts_test=treat_starts_test, treat_ends_test=treat_ends_test, label='Total')
    plot_sim(ax1, x_sim, sim_clone_tot_area, sim_clone_tot_area_iqr, color, start_point, clone=False, treat_starts_test=treat_starts_test, treat_ends_test=treat_ends_test, linestyle=':', label='Resistant')
    plot_sim(ax2, x_sim, sim_clone_area, sim_clone_area_iqr, color, start_point, clone=True, treat_starts_test=treat_starts_test, treat_ends_test=treat_ends_test, label='Resistant fraction')
    plot_sim(ax3, x_sim, sen_growth_rate, sen_growth_rate_iqr, color, start_point, clone=False, treat_starts_test=treat_starts_test, treat_ends_test=treat_ends_test, label='Sensitive', linestyle='--')
    plot_sim(ax3, x_sim, res_growth_rate, res_growth_rate_iqr, color, start_point, clone=False, treat_starts_test=treat_starts_test, treat_ends_test=treat_ends_test, linestyle=':', label='Resistant')
    plot_sim(ax4, x_sim, sim_sen_front_growth_rate*20, sim_sen_front_growth_rate_iqr*20, color, start_point, clone=True, treat_starts_test=treat_starts_test, treat_ends_test=treat_ends_test, label='Sensitive front', linestyle='--')
    plot_sim(ax4, x_sim, sim_res_front_growth_rate*20, sim_res_front_growth_rate_iqr*20, color, start_point, clone=True, treat_starts_test=treat_starts_test, treat_ends_test=treat_ends_test, label='Resistant front', linestyle=':')
    # plot_sim(ax5, x_sim, sim_front_ring_ratio, sim_front_ring_ratio_iqr, color, start_point, clone=True, treat_starts_test=treat_starts_test, treat_ends_test=treat_ends_test, label=f'Resistant fraction ({ring_width_px}px front ring)')

    ax1.set_xlim(0, 150)
    ax1.set_ylim(0, 71)
    ax2.set_xlim(0, 150)
    ax2.set_ylim(0, 1)
    ax3.set_xlim(0, 150)
    ax3.set_ylim(0, 2)
    ax4.set_xlim(0, 150)
    ax4.set_ylim(0, 0.03 * params['mutation_scaling'])
    # ax5.set_xlim(0, 150)
    # ax5.set_ylim(0, 1)

    ax4.axhline(3.32795, color='black', linestyle='--', lw=0.5, label=r'$\gamma^*_\mathrm{front}$')

    if treatment == 'met_4_18':
        ax1.set_ylabel(r'Area, A (mm²)')
        ax2.set_ylabel(r'Resistant/Total')
        ax3.set_ylabel(r'dA/dt (mm²/h)')
        ax4.set_ylabel(r'$\gamma_\mathrm{front}$ (1/h)')
        # ax5.set_ylabel(r'Res./Total in front ring')

        leg_handles1 = ax1.legend(loc='upper left', frameon=False)
        for handle in leg_handles1.legend_handles:
            handle.set_color('black')

        leg_handles2 = ax2.legend(loc='upper left', ncol=2, frameon=False)
        for handle in leg_handles2.legend_handles:
            handle.set_color('black')

        leg_handles3 = ax3.legend(loc='upper left', ncol=2, frameon=False)
        for handle in leg_handles3.legend_handles:
            handle.set_color('black')

        leg_handles4 = ax4.legend(loc='upper left', ncol=2, frameon=False)
        for handle in leg_handles4.legend_handles:
            handle.set_color('black')

        # leg_handles5 = ax5.legend(loc='upper left', ncol=2, frameon=False)
        # for handle in leg_handles5.legend_handles:
        #     handle.set_color('black')
    else:
        ax1.set_yticklabels([])
        ax2.set_yticklabels([])
        ax3.set_yticklabels([])
        ax4.set_yticklabels([])
        # ax5.set_yticklabels([])

    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])
    ax4.set_xticklabels([])
    # ax5.set_xticklabels([])

    fig1.savefig(fr'Figure_4/panel_d_e_f/{treatment}_colony_area.pdf', bbox_inches='tight', transparent=True)
    fig2.savefig(fr'Figure_4/panel_d_e_f/{treatment}_clone_area.pdf', bbox_inches='tight', transparent=True)
    fig3.savefig(fr'Figure_4/panel_d_e_f/{treatment}_growth_rate.pdf', bbox_inches='tight', transparent=True)
    fig4.savefig(fr'Figure_4/panel_d_e_f/{treatment}_front_growth_rate.pdf', bbox_inches='tight', transparent=True)
    # fig5.savefig(fr'Figure_4/panel_d_e_f/{treatment}_front_ring_ratio_{ring_width_px}px.pdf', bbox_inches='tight', transparent=True)


if __name__ == "__main__":
    replicates = 20
    front_ring_width_px = 4
    treatments = ['met_4_18', 'met_6_5_18', 'met_9_18', 'continuous_dose', 'no_treatment']
    colors_nums = [4, 8, 12, 16, 0]
    for i, treatment in enumerate(treatments):
        color = mpl.colormaps.get_cmap('tab20b').colors[colors_nums[i]]
        plot_comparison(replicates, treatment, color, ring_width_px=front_ring_width_px)
