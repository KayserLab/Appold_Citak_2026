import matplotlib.pyplot as plt
import numpy as np
import torch
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


def load_sim_data(path):
    sen = np.load(f'{path}/sensitive.npy')
    res = np.load(f'{path}/resistant.npy')
    params = torch.load(f'{path}/params.pth')

    sen_diff = np.diff(sen*params['mutation_scaling'], axis=0)
    res_diff = np.diff(res*params['mutation_scaling'], axis=0)

    counts = []
    counts_res = []
    res_total_count = []
    counts_sen = []
    sen_avg_front_den = []
    res_avg_front_den = []
    for i in range(len(sen)-1):
        sen_thresholded = np.where(sen[i] >= (1 / params['mutation_scaling']), 1, 0)
        res_thresholded = np.where(res[i] >= (1 / params['mutation_scaling']), 1, 0)

        total_array = sen_thresholded + res_thresholded

        tot_mask = total_array > 0
        sen_mask = sen_thresholded > 0
        res_mask = res_thresholded > 0

        # boundary pixels (boolean images)
        tot_b = seg.find_boundaries(tot_mask, mode="inner")  # or "outer"
        sen_b = seg.find_boundaries(sen_mask, mode="inner")
        res_b = seg.find_boundaries(res_mask, mode="inner")

        # intersection of contours
        front_sen = tot_b & sen_b
        front_res = tot_b & res_b

        # average original values on that front
        sen_avg_front_den.append(sen_diff[i][front_sen].mean() if front_sen.any() else 0)
        res_avg_front_den.append(res_diff[i][front_res].mean() if front_res.any() else 0)

    return (np.array(counts)*(eval(params['sim_pixel_to_exp_pixel_factor']))**2, np.array(counts_res), np.array(res_total_count)*(eval(params['sim_pixel_to_exp_pixel_factor']))**2, np.array(counts_sen)*(eval(params['sim_pixel_to_exp_pixel_factor']))**2,
            np.array(sen_avg_front_den), np.array(res_avg_front_den))

def get_sim_data(replicate, treatment):
    sim_sen_front_growth_rate_temp = []
    sim_res_front_growth_rate_temp = []
    for i in range(replicate):
        sim_colony, sim_clone, sim_clone_tot, sim_sen_tot, sen_avg_front, res_avg_front = load_sim_data(f'../../data/sim_data/{treatment}/{treatment}_{i}')
        sim_sen_front_growth_rate_temp.append(rolling_average(sen_avg_front, window_size=51))
        sim_res_front_growth_rate_temp.append(rolling_average(res_avg_front, window_size=51))

    sim_sen_front_growth_rate = np.median(np.array(sim_sen_front_growth_rate_temp), axis=0)
    sim_res_front_growth_rate = np.median(np.array(sim_res_front_growth_rate_temp), axis=0)
    sim_sen_front_growth_rate_iqr = np.percentile(np.array(sim_sen_front_growth_rate_temp), [25, 75], axis=0)
    sim_res_front_growth_rate_iqr = np.percentile(np.array(sim_res_front_growth_rate_temp), [25, 75], axis=0)

    return sim_sen_front_growth_rate, sim_sen_front_growth_rate_iqr, sim_res_front_growth_rate, sim_res_front_growth_rate_iqr

def rolling_median(data, window_size):
    padded_data = np.pad(data, (window_size // 2, window_size - window_size // 2 - 1), mode='edge')
    rolling_medians = np.array([np.median(padded_data[i:i + window_size]) for i in range(len(data))])
    return rolling_medians

def rolling_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def plot_sim(ax, x_sim, sim_area, sim_iqr, color, start_point, clone, treatment, linestyle='solid', label=None):
    scaling_factor = 8.648 ** 2 / 1e6
    if clone:
        scaling_factor = 1
    ax.plot(x_sim, rolling_average(sim_area*scaling_factor, window_size=51)[start_point:3001 + start_point], color=color, linestyle=linestyle, label=label)
    ax.fill_between(x_sim, sim_iqr[0][start_point:3001 + start_point]*scaling_factor, sim_iqr[1][start_point:3001 + start_point]*scaling_factor, color=color, alpha=0.25, lw=0)
    if treatment == 'continuous_dose':
        ax.axvspan(18, 300, color='#bfbfbf', alpha=1, lw=0, zorder=0)
    elif treatment == 'pulse':
        ax.axvspan(18, 32, color='#bfbfbf', alpha=1, lw=0, zorder=0)

def plot_comparison(replicate, treatment, color):
    fig, ax = plt.subplots(figsize=(8.35/3, 6.5/6), dpi=300) # (8.125/2, 7.1/5)

    params = torch.load(f'../../data/sim_data/{treatment}/{treatment}_0/params.pth')
    sim_sen_front_growth_rate, sim_sen_front_growth_rate_iqr, sim_res_front_growth_rate, sim_res_front_growth_rate_iqr = get_sim_data(replicate, treatment)

    start_point = params['start_point']

    x_sim = np.linspace(0, 3000, 3001)/20

    plot_sim(ax, x_sim, sim_sen_front_growth_rate, sim_sen_front_growth_rate_iqr, color, start_point, clone=True, treatment=treatment, label='Sensitive front', linestyle='--')
    plot_sim(ax, x_sim, sim_res_front_growth_rate, sim_res_front_growth_rate_iqr, color, start_point, clone=True, treatment=treatment, label='Resistant front', linestyle=':')

    ax.set_xlim(0, 150)
    ax.set_ylim(0, 0.0015*params['mutation_scaling'])

    ax.set_ylabel(r'gamma_front (1/h)')
    ax.legend(loc='upper left', frameon=False)

    ax.set_xticklabels([])
    # ax.set_xlabel('Time (h)')

    fig.savefig(fr'{treatment}_front_growth_rate_SI.pdf', bbox_inches='tight', transparent=True)
    fig.show()


if __name__ == "__main__":
    replicates = 20
    treatments = ['continuous_dose', 'pulse']
    for i, treatment in enumerate(treatments):
        color = 'black'
        plot_comparison(replicates, treatment, color)
