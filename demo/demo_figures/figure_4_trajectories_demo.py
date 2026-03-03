import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib as mpl
import skimage.segmentation as seg

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
        if treat_on == 0:
            treatment_starts = []
    elif treat_on == 0:
        treatment_starts = []
    else:
        treatment_starts = [d for d in range(first_start, params['total_time'], treat_off + treat_on)]
        treatment_ends = np.array(treatment_starts) + treat_on

    for i in range(len(treatment_starts)):
        treatment_times[treatment_starts[i]:(treatment_starts[i] + treatment_length)] = True

    return treatment_starts, treatment_ends

def get_sim_data(treatment):
    sim_colony_area = np.load(f'demo/demo_data/{treatment}_colony_area.npy')
    sim_colony_area_iqr = np.load(f'demo/demo_data/{treatment}_colony_area_iqr.npy')
    sim_clone_area = np.load(f'demo/demo_data/{treatment}_clone_area.npy')
    sim_clone_area_iqr = np.load(f'demo/demo_data/{treatment}_clone_area_iqr.npy')
    sim_clone_tot_area = np.load(f'demo/demo_data/{treatment}_clone_tot_area.npy')
    sim_clone_tot_area_iqr = np.load(f'demo/demo_data/{treatment}_clone_tot_area_iqr.npy')
    sim_sen_front_growth_rate = np.load(f'demo/demo_data/{treatment}_sen_front_growth_rate.npy')
    sim_res_front_growth_rate = np.load(f'demo/demo_data/{treatment}_res_front_growth_rate.npy')
    sim_sen_front_growth_rate_iqr = np.load(f'demo/demo_data/{treatment}_sen_front_growth_rate_iqr.npy')
    sim_res_front_growth_rate_iqr = np.load(f'demo/demo_data/{treatment}_res_front_growth_rate_iqr.npy')

    return (sim_colony_area, sim_colony_area_iqr, sim_clone_area, sim_clone_area_iqr, sim_clone_tot_area, sim_clone_tot_area_iqr,
            sim_sen_front_growth_rate, sim_sen_front_growth_rate_iqr, sim_res_front_growth_rate, sim_res_front_growth_rate_iqr)

def rolling_median(data, window_size):
    padded_data = np.pad(data, (window_size // 2, window_size - window_size // 2 - 1), mode='edge')
    rolling_medians = np.array([np.median(padded_data[i:i + window_size]) for i in range(len(data))])
    return rolling_medians

def rolling_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def plot_sim(ax, x_sim, sim_area, sim_iqr, color, start_point, clone, treat_starts_test, treat_ends_test, linestyle='solid', label=None):
    scaling_factor = 8.648 ** 2 / 1e6
    if clone:
        scaling_factor = 1
    ax.plot(x_sim, rolling_average(sim_area*scaling_factor, window_size=51)[start_point:3001 + start_point], color=color, linestyle=linestyle, label=label)
    ax.fill_between(x_sim, sim_iqr[0][start_point:3001 + start_point]*scaling_factor, sim_iqr[1][start_point:3001 + start_point]*scaling_factor, color=color, alpha=0.25, lw=0)
    for i in range(len(treat_starts_test)):
        ax.axvspan(treat_starts_test[i] / 20, treat_ends_test[i] / 20, color='#bfbfbf', alpha=1, lw=0, zorder=0)

def plot_comparison(treatment, color):
    fig1, ax1 = plt.subplots(figsize=(8.4/3, 6.5/6), dpi=300)
    fig2, ax2 = plt.subplots(figsize=(8.4/3, 6.5/6), dpi=300)
    fig3, ax3 = plt.subplots(figsize=(8.4/3, 6.5/6), dpi=300)

    params = torch.load(f'demo/demo_data/sweep_params.pth')
    treat_on = int(treatment.split('_')[1])
    if treatment == 'met_6_5_18':
        treat_on = 6.5

    treat_starts_test, treat_ends_test = calc_treatment_efficacy(int(treat_on * 20), int(18 * 20), params)
    sim_colony_area, sim_colony_area_iqr, sim_clone_area, sim_clone_area_iqr, sim_clone_tot_area, sim_clone_tot_area_iqr, sim_sen_front_growth_rate, sim_sen_front_growth_rate_iqr, sim_res_front_growth_rate, sim_res_front_growth_rate_iqr = get_sim_data(treatment)

    start_point = params['start_point']

    x_sim = np.linspace(0, 3000, 3001)/20
    plot_sim(ax1, x_sim, sim_colony_area, sim_colony_area_iqr, color, start_point, clone=False, treat_starts_test=treat_starts_test, treat_ends_test=treat_ends_test, label='Total')
    plot_sim(ax1, x_sim, sim_clone_tot_area, sim_clone_tot_area_iqr, color, start_point, clone=False, treat_starts_test=treat_starts_test, treat_ends_test=treat_ends_test, linestyle=':', label='Resistant')
    plot_sim(ax2, x_sim, sim_clone_area, sim_clone_area_iqr, color, start_point, clone=True, treat_starts_test=treat_starts_test, treat_ends_test=treat_ends_test, label='Resistant fraction')
    plot_sim(ax3, x_sim, sim_sen_front_growth_rate, sim_sen_front_growth_rate_iqr, color, start_point, clone=True, treat_starts_test=treat_starts_test, treat_ends_test=treat_ends_test, label='Sensitive front', linestyle='--')
    plot_sim(ax3, x_sim, sim_res_front_growth_rate, sim_res_front_growth_rate_iqr, color, start_point, clone=True, treat_starts_test=treat_starts_test, treat_ends_test=treat_ends_test, label='Resistant front', linestyle=':')

    ax1.set_xlim(0, 150)
    ax1.set_ylim(0, 71)

    ax2.set_xlim(0, 150)
    ax2.set_ylim(0, 1)

    ax3.set_xlim(0, 150)
    ax3.set_ylim(0, 0.0015*params['mutation_scaling'])

    ax1.set_ylabel(r'Area, A (mm²)')
    ax2.set_ylabel(r'Resistant/Total')
    ax3.set_ylabel(r'gamma_front (1/h)')

    leg_handles1 = ax1.legend(loc='upper left', frameon=False)
    for handle in leg_handles1.legend_handles:
        handle.set_color('black')

    leg_handles2 = ax2.legend(loc='upper left', frameon=False)
    for handle in leg_handles2.legend_handles:
        handle.set_color('black')

    leg_handles3 = ax3.legend(loc='upper left', frameon=False)
    for handle in leg_handles3.legend_handles:
        handle.set_color('black')

    fig1.savefig(fr'demo/demo_figures/{treatment}_colony_area.pdf', bbox_inches='tight', transparent=True)
    fig2.savefig(fr'demo/demo_figures/{treatment}_clone_area.pdf', bbox_inches='tight', transparent=True)
    fig3.savefig(fr'demo/demo_figures/{treatment}_front_growth_rate.pdf', bbox_inches='tight', transparent=True)

    fig1.show()
    fig2.show()
    fig3.show()


if __name__ == "__main__":
    treatments = ['met_4_18', 'met_6_5_18', 'met_9_18']
    colors_nums = [4, 8, 12]
    for i, treatment in enumerate(treatments):
        color = mpl.colormaps.get_cmap('tab20b').colors[colors_nums[i]]
        plot_comparison(treatment, color)