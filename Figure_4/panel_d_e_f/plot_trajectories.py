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
    first_start = params['treatment_start'] # + params['start_point']

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
        total_count = np.count_nonzero(total_array)
        counts.append(total_count)
        counts_sen.append(np.count_nonzero(sen_thresholded))

        sen_thresholded_ratio = np.where(sen[i] > (1 / params['mutation_scaling']), sen[i], 0)
        res_thresholded_ratio = np.where(res[i] > (1 / params['mutation_scaling']), res[i], 0)
        res_ratio = np.where(res_thresholded_ratio > sen_thresholded_ratio, 1., 0.)

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

        # tot_arr = np.where(total_array > 0, 1, 0)
        # tot_arr_contour = measure.find_contours(tot_arr, 0.5)
        # sen_arr_contour = measure.find_contours(sen_thresholded, 0.5)
        # res_arr_contour = measure.find_contours(res_thresholded, 0.5)
        #
        # # remove contour points from tot_arr_contour that are not in sen_arr_contour
        # tot_arr_contour_filtered_sen = []
        # for contour in tot_arr_contour:
        #     filtered_contour = []
        #     for point in contour:
        #         if any(np.all(point == sen_point) for sen_contour in sen_arr_contour for sen_point in sen_contour):
        #             filtered_contour.append(point)
        #     tot_arr_contour_filtered_sen.append(np.array(filtered_contour))
        #
        # # remove contour points from tot_arr_contour that are not in res_arr_contour
        # tot_arr_contour_filtered_res = []
        # for contour in tot_arr_contour:
        #     filtered_contour = []
        #     for point in contour:
        #         if any(np.all(point == res_point) for res_contour in res_arr_contour for res_point in res_contour):
        #             filtered_contour.append(point)
        #     tot_arr_contour_filtered_res.append(np.array(filtered_contour))
        #
        # rows_sen = tot_arr_contour_filtered_sen[0][:, 0].astype(int)
        # cols_sen = tot_arr_contour_filtered_sen[0][:, 1].astype(int)
        # sen_front_values = sen[i][rows_sen, cols_sen]
        # sen_avg_front_den.append(np.mean(sen_front_values))
        #
        # if len(tot_arr_contour_filtered_res[0]) > 0:
        #     rows_res = tot_arr_contour_filtered_res[0][:, 0].astype(int)
        #     cols_res = tot_arr_contour_filtered_res[0][:, 1].astype(int)
        #     res_front_values = res[i][rows_res, cols_res]
        #     res_avg_front_den.append(np.mean(res_front_values))
        # else:
        #     res_avg_front_den.append(0)

        if total_count >= 1:
            counts_res.append(np.count_nonzero(res_ratio)/total_count)
            res_total_count.append(np.count_nonzero(res_ratio))
        else:
            counts_res.append(0)
            res_total_count.append(0)

    return (np.array(counts)*(eval(params['sim_pixel_to_exp_pixel_factor']))**2, np.array(counts_res), np.array(res_total_count)*(eval(params['sim_pixel_to_exp_pixel_factor']))**2, np.array(counts_sen)*(eval(params['sim_pixel_to_exp_pixel_factor']))**2,
            np.array(sen_avg_front_den), np.array(res_avg_front_den))

def get_sim_data(replicate, treatment):
    sim_colony_area_temp = []
    sim_clone_area_temp = []
    sim_clone_tot_temp = []
    sim_colony_growth_rate_temp = []
    sim_clone_growth_rate_temp = []
    sim_sen_front_growth_rate_temp = []
    sim_res_front_growth_rate_temp = []
    for i in range(replicate):
        sim_colony, sim_clone, sim_clone_tot, sim_sen_tot, sen_avg_front, res_avg_front = load_sim_data(f'../../data/sim_data/{treatment}/{treatment}_{i}')
        sim_colony_area_temp.append(sim_colony)
        sim_clone_area_temp.append(sim_clone)
        sim_clone_tot_temp.append(sim_clone_tot)
        sim_colony_growth_rate_temp.append(np.gradient(rolling_average(sim_sen_tot, window_size=51), 1/20))
        sim_clone_growth_rate_temp.append(np.gradient(rolling_average(sim_clone_tot, window_size=51), 1/20))
        sim_sen_front_growth_rate_temp.append(rolling_average(sen_avg_front, window_size=51))
        sim_res_front_growth_rate_temp.append(rolling_average(res_avg_front, window_size=51))

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

    return (sim_colony_area, sim_colony_area_iqr, sim_clone_area, sim_clone_area_iqr, sim_clone_tot_area, sim_clone_tot_area_iqr, sim_colony_growth_rate,
            sim_colony_growth_rate_iqr, sim_clone_growth_rate, sim_clone_growth_rate_iqr, sim_sen_front_growth_rate, sim_sen_front_growth_rate_iqr, sim_res_front_growth_rate, sim_res_front_growth_rate_iqr)

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

def plot_comparison(replicate, treatment, color):
    fig1, ax1 = plt.subplots(figsize=(8.4/3, 6.5/6), dpi=300)
    fig2, ax2 = plt.subplots(figsize=(8.4/3, 6.5/6), dpi=300)
    fig3, ax3 = plt.subplots(figsize=(8.4/3, 6.5/6), dpi=300)
    fig4, ax4 = plt.subplots(figsize=(8.4/3, 6.5/6), dpi=300)

    params = torch.load(f'../../data/sim_data/{treatment}/{treatment}_0/params.pth')
    # params = torch.load(f'../../Figure_3/sim_data/{treatment}_0/params.pth')
    # print(treatment.split('_')[1])
    treat_on = int(treatment.split('_')[1])
    if treatment == 'met_6_5_18':
        treat_on = 6.5

    treat_starts_test, treat_ends_test = calc_treatment_efficacy(int(treat_on * 20), int(18 * 20), params)
    sim_colony_area, sim_colony_area_iqr, sim_clone_area, sim_clone_area_iqr, sim_clone_tot_area, sim_clone_tot_area_iqr, sen_growth_rate, sen_growth_rate_iqr, res_growth_rate, res_growth_rate_iqr, sim_sen_front_growth_rate, sim_sen_front_growth_rate_iqr, sim_res_front_growth_rate, sim_res_front_growth_rate_iqr = get_sim_data(replicate, treatment)

    start_point = params['start_point']

    x_sim = np.linspace(0, 3000, 3001)/20
    if len(np.argwhere(sim_colony_area * 8.648 ** 2 / 1e6 >= 71)) > 0:
        print(np.argwhere(sim_colony_area * 8.648 ** 2 / 1e6 >= 71)[0])
    else:
        print('No limit reached')
    plot_sim(ax1, x_sim, sim_colony_area, sim_colony_area_iqr, color, start_point, clone=False, treat_starts_test=treat_starts_test, treat_ends_test=treat_ends_test, label='Total')
    plot_sim(ax1, x_sim, sim_clone_tot_area, sim_clone_tot_area_iqr, color, start_point, clone=False, treat_starts_test=treat_starts_test, treat_ends_test=treat_ends_test, linestyle=':', label='Resistant')
    plot_sim(ax2, x_sim, sim_clone_area, sim_clone_area_iqr, color, start_point, clone=True, treat_starts_test=treat_starts_test, treat_ends_test=treat_ends_test, label='Resistant fraction')

    plot_sim(ax3, x_sim, sen_growth_rate, sen_growth_rate_iqr, color, start_point, clone=False, treat_starts_test=treat_starts_test, treat_ends_test=treat_ends_test, label='Sensitive', linestyle='--')
    plot_sim(ax3, x_sim, res_growth_rate, res_growth_rate_iqr, color, start_point, clone=False, treat_starts_test=treat_starts_test, treat_ends_test=treat_ends_test, linestyle=':', label='Resistant')

    plot_sim(ax4, x_sim, sim_sen_front_growth_rate, sim_sen_front_growth_rate_iqr, color, start_point, clone=True, treat_starts_test=treat_starts_test, treat_ends_test=treat_ends_test, label='Sensitive front', linestyle='--')
    plot_sim(ax4, x_sim, sim_res_front_growth_rate, sim_res_front_growth_rate_iqr, color, start_point, clone=True, treat_starts_test=treat_starts_test, treat_ends_test=treat_ends_test, label='Resistant front', linestyle=':')

    ax1.set_xlim(0, 150)
    ax1.set_ylim(0, 71)

    ax2.set_xlim(0, 150)
    ax2.set_ylim(0, 1)

    ax3.set_xlim(0, 150)
    ax3.set_ylim(0, 2)

    ax4.set_xlim(0, 150)
    ax4.set_ylim(0, 0.0015*params['mutation_scaling'])

    # ax2.legend(loc='upper left', frameon=False)
    # ax3.legend(loc='upper left', frameon=False)

    # ax1.set_ylabel(r'Area (mm²)')
    # ax2.set_ylabel(r'Resistant/Total')
    # ax3.set_ylabel(r'Growth rate (mm²/h)')
    # ax1.set_xlabel('Time (h)')
    # ax2.set_xlabel('Time (h)')
    # ax3.set_xlabel('Time (h)')

    if treatment == 'met_4_18':
        ax1.set_ylabel(r'Area, A (mm²)')
        ax2.set_ylabel(r'Resistant/Total')
        ax3.set_ylabel(r'dA/dt (mm²/h)')
        ax4.set_ylabel(r'gamma_front (1/h)')

        leg_handles1 = ax1.legend(loc='upper left', frameon=False)
        for handle in leg_handles1.legend_handles:
            handle.set_color('black')

        leg_handles2 = ax2.legend(loc='upper left', frameon=False)
        for handle in leg_handles2.legend_handles:
            handle.set_color('black')

        leg_handles3 = ax3.legend(loc='upper left', frameon=False)
        for handle in leg_handles3.legend_handles:
            handle.set_color('black')

        leg_handles4 = ax4.legend(loc='upper left', frameon=False)
        for handle in leg_handles4.legend_handles:
            handle.set_color('black')

    else:
        ax1.set_yticklabels([])
        ax2.set_yticklabels([])
        ax3.set_yticklabels([])
        ax4.set_yticklabels([])

    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])
    ax4.set_xticklabels([])

    fig1.savefig(fr'{treatment}_colony_area.pdf', bbox_inches='tight', transparent=True)
    fig2.savefig(fr'{treatment}_clone_area.pdf', bbox_inches='tight', transparent=True)
    fig3.savefig(fr'{treatment}_growth_rate.pdf', bbox_inches='tight', transparent=True)
    fig4.savefig(fr'{treatment}_front_growth_rate.pdf', bbox_inches='tight', transparent=True)

    fig1.show()
    fig2.show()
    fig3.show()
    fig4.show()


if __name__ == "__main__":
    replicates = 20
    treatments = ['met_4_18', 'met_6_5_18', 'met_9_18']
    colors_nums = [4, 8, 12]
    for i, treatment in enumerate(treatments):
        color = mpl.colormaps.get_cmap('tab20b').colors[colors_nums[i]]
        plot_comparison(replicates, treatment, color)
