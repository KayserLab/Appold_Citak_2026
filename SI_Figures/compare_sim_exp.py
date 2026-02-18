import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.transforms as mtransforms
from matplotlib.patches import ConnectionPatch
import source.run_core as rc
import torch
import matplotlib as mpl


plt.rcParams.update({'font.size': 7,
                     'pdf.fonttype': 42,
                     'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial'],
                     'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Arial',
                     'mathtext.it': 'Arial:italic',
                     'mathtext.bf': 'Arial:bold',
                     })
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6

def get_data(path):
    data_list = [i for i in os.listdir(path) if i.startswith('colony') and i.endswith('clonearea.csv')]
    data = []
    for i in data_list:
        data.append(pd.read_csv(f'{path}/{i}'))
    return data

def average_over_area(exp_data, length):
    temp_colony = []
    temp_clone = []
    for data in exp_data:
        colony_area = data['colony_area']
        # temp_colony.append(colony_area)
        clone_area = data['total_clone_area']
        print(len(colony_area))
        # clone_area_addition = data['extrapolated_clone_area'] - data['total_clone_area']
        temp_colony.append(colony_area[:length])
        temp_clone.append(clone_area[:length]/(colony_area[:length]))
    return np.median(np.array(temp_colony), axis=0), np.percentile(np.array(temp_colony), [25, 75], axis=0), np.median(np.array(temp_clone), axis=0), np.percentile(np.array(temp_clone), [25, 75], axis=0)

def load_sim_data(path):
    sen = np.load(f'{path}/sensitive.npy')
    res = np.load(f'{path}/resistant.npy')
    params = torch.load(f'{path}/params.pth')

    counts = []
    counts_res = []
    for i in range(len(sen)):
        sen_thresholded = np.where(sen[i] >= (1 / params['mutation_scaling']), 1, 0)
        res_thresholded = np.where(res[i] >= (1 / params['mutation_scaling']), 1, 0)

        total_array = sen_thresholded + res_thresholded
        total_count = np.count_nonzero(total_array)
        counts.append(total_count)

        sen_thresholded_ratio = np.where(sen[i] > (1 / params['mutation_scaling']), sen[i], 0)
        res_thresholded_ratio = np.where(res[i] > (1 / params['mutation_scaling']), res[i], 0)
        res_ratio = np.where(res_thresholded_ratio > sen_thresholded_ratio, 1., 0.)

        if total_count >= 1:
            counts_res.append(np.count_nonzero(res_ratio)/total_count)
        else:
            counts_res.append(0)

    return np.array(counts)*(eval(params['sim_pixel_to_exp_pixel_factor']))**2, np.array(counts_res)

def get_start_point(area_exp, area_sim):
    for i in range(len(area_sim)):
        if area_sim[i] >= area_exp[0]:
            print(area_sim[i], area_exp[0])
            return i
    return 0

def create_data(replicates, treatments):
    for j in range(len(replicates)):
        if treatments[j] == 'continuous_dose':
            treat_on, treat_off = 10, 0
            pulse, pulse_duration = False, None
        elif treatments[j] == 'no_treatment':
            treat_on, treat_off = 0, 0
            pulse, pulse_duration = False, None
        elif treatments[j] == 'pulse':
            treat_on, treat_off = 0, 0
            pulse, pulse_duration = True, 280
        else:
            treat_on, treat_off = None, None
            pulse, pulse_duration = None, None
            print('Treatment not found')
        for i in range(replicates[j]):
            rc.main(treat_on, treat_off, save_dir=f'data/sim_data/{treatments[j]}_{i}', random_seed=i, pulse=pulse, pulse_duration=pulse_duration)

def plot_exp(ax, x_exp, exp_colony_area, exp_colony_iqr, color, clone):
    scaling_factor = 8.648 ** 2 / 1e6
    if clone:
        scaling_factor = 1
    ax.plot(x_exp, exp_colony_area * scaling_factor, color=color)
    ax.fill_between(x_exp, exp_colony_iqr[0] * scaling_factor, exp_colony_iqr[1] * scaling_factor, color=color, alpha=0.25)

def plot_sim(ax, x_sim, sim_colony_area, sim_colony_iqr, color, start_point, clone):
    scaling_factor = 8.648 ** 2 / 1e6
    if clone:
        scaling_factor = 1
    ax.plot(x_sim, sim_colony_area[start_point:3240 + start_point] * scaling_factor, color=color, linestyle='--')
    ax.fill_between(x_sim, sim_colony_iqr[0][start_point:3240 + start_point] * scaling_factor,
                    sim_colony_iqr[1][start_point:3240 + start_point] * scaling_factor, color=color, alpha=0.25)

def add_bar_label(ax, y_mid, label, color):
    ax.hlines(y_mid, -3, 0, color=color, lw=0.8, clip_on=False)
    ax.text(-5, y_mid, label, va='center', ha='right', fontsize=7, color=color)

def plot_treat_bars(fig, ax):
    bar_ax = ax.inset_axes([0, -0.385, 1, 0.3])  # [x0, y0, width, height] in relative coords of ax[1]
    bar_ax.set_xlim(ax.get_xlim())
    bar_ax.set_ylim(0, 1)
    bar_ax.axis('off')  # remove frame, ticks, labels

    # Format: (x_start, width)
    metronomic_periods = [[18, 4], [40, 4], [62, 4], [84, 4], [106, 4], [128, 4], [150, 4], [172, 4], [194, 4]]  # example treatment intervals
    continuous_periods = [[18, 6.5], [42.5, 6.5], [67, 6.5], [91.5, 6.5], [116, 6.5], [140.5, 6.5], [165, 6.5]]
    nt_periods = [[18, 9], [45, 9], [72, 9], [99, 9], [126, 9], [153, 9], [180, 9]]

    BAR_H = 0.2
    y_pulse = 0.27
    y_cont = 0.57
    y_nt = 0.87

    # draw bars (now same thickness, no clipping)
    bar_ax.broken_barh(metronomic_periods, (y_pulse - BAR_H / 2, BAR_H), facecolors=mpl.colormaps.get_cmap('tab20b').colors[4], edgecolor='none')
    bar_ax.broken_barh(continuous_periods, (y_cont - BAR_H / 2, BAR_H), facecolors=mpl.colormaps.get_cmap('tab20b').colors[8], edgecolor='none')
    bar_ax.broken_barh(nt_periods, (y_nt - BAR_H / 2, BAR_H), facecolors=mpl.colormaps.get_cmap('tab20b').colors[12], edgecolor='none')

    # optional labels on the left
    add_bar_label(bar_ax, y_nt, "4/18", mpl.colormaps.get_cmap('tab20b').colors[4])
    add_bar_label(bar_ax, y_cont, "6.5/18", mpl.colormaps.get_cmap('tab20b').colors[8])
    add_bar_label(bar_ax, y_pulse, "9/18", mpl.colormaps.get_cmap('tab20b').colors[12])

    # continuous_periods = [[18, 300]]
    # nt_periods = []

    # BAR_H = 0.2
    # y_cont = 0.43
    # y_nt = 0.76

    # draw bars (now same thickness, no clipping)
    # bar_ax.broken_barh(continuous_periods, (y_cont - BAR_H / 2, BAR_H), facecolors=mpl.colormaps.get_cmap('tab20b').colors[16], edgecolor='none')
    # bar_ax.broken_barh(nt_periods, (y_nt - BAR_H / 2, BAR_H), facecolors=mpl.colormaps.get_cmap('tab20b').colors[0], edgecolor='none')

    # optional labels on the left
    # add_bar_label(bar_ax, y_nt, "NT", mpl.colormaps.get_cmap('tab20b').colors[0])
    # add_bar_label(bar_ax, y_cont, "CT", mpl.colormaps.get_cmap('tab20b').colors[16])

    tick_ax = ax.inset_axes([0, -0.385, 1, 0.0001])  # below the bar

    # Make sure both axes share identical x-limits
    tick_ax.set_xlim(ax.get_xlim())

    # Copy tick locator/formatter so positions & labels match perfectly
    tick_ax.xaxis.set_major_locator(ax.xaxis.get_major_locator())
    tick_ax.xaxis.set_major_formatter(ax.xaxis.get_major_formatter())

    # Force tick calculation to be up-to-date
    fig.canvas.draw()

    # Remove labels from the bottom plot, keep tick marks
    ax.set_xticklabels([])

    # Hide y of tick_ax and spines except bottom
    tick_ax.tick_params(axis='y', left=False, labelleft=False)
    for sp in ("left", "right", "top"):
        tick_ax.spines[sp].set_visible(False)

    # Build blended transforms: x in data coords, y in axes coords
    bt1 = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)  # ax[1]: y=0 is bottom of ax[1]
    bt2 = mtransforms.blended_transform_factory(tick_ax.transData, tick_ax.transAxes)  # tick_ax: y=1 is top of tick_ax

    # Draw dotted connectors from bottom of ax[1] down to top of tick_ax
    xr = ax.get_xlim()[1]
    ticks_conn = [t for t in ax.get_xticks() if t < xr - 1e-9]

    for tick in ticks_conn:
        con = ConnectionPatch((tick, 0), (tick, 1), coordsA=bt1, coordsB=bt2, linestyle=":", color="0.5", lw=1, clip_on=False, zorder=0, alpha=0.5)
        ax.figure.add_artist(con)

    tick_ax.set_xlabel("Time (h)", fontsize=7, labelpad=2)
    tick_ax.spines['bottom'].set_visible(False)

def get_sim_data(replicate, treatment):
    sim_colony_area_temp = []
    sim_clone_area_temp = []
    for i in range(replicate):
        sim_colony, sim_clone = load_sim_data(f'../../data/sim_data/{treatment}/{treatment}_{i}')
        sim_colony_area_temp.append(sim_colony)
        sim_clone_area_temp.append(sim_clone)

    sim_colony_area = np.median(np.array(sim_colony_area_temp), axis=0)
    sim_colony_area_iqr = np.percentile(np.array(sim_colony_area_temp), [25, 75], axis=0)
    sim_clone_area = np.median(np.array(sim_clone_area_temp), axis=0)
    sim_clone_area_iqr = np.percentile(np.array(sim_clone_area_temp), [25, 75], axis=0)

    return sim_colony_area, sim_colony_area_iqr, sim_clone_area, sim_clone_area_iqr

def plot_comparison(replicates, treatments):
    fig1, ax1 = plt.subplots(figsize=(6.6/3, 6.6/5), dpi=300)
    fig2, ax2 = plt.subplots(figsize=(6.6/3, 6.6/5), dpi=300)
    ax1.plot([0,1], [10000, 10000], 'black', label='Experiment')
    ax1.plot([0, 1], [10000, 10000], 'black', linestyle=':', label='Simulation')
    ax2.plot([0, 1], [10000, 10000], 'black', label='Experiment')
    ax2.plot([0, 1], [10000, 10000], 'black', linestyle=':', label='Simulation')

    lengths = [293, 329, 333]
    colors = [mpl.colormaps.get_cmap('tab20b').colors[4], mpl.colormaps.get_cmap('tab20b').colors[8], mpl.colormaps.get_cmap('tab20b').colors[12]]
    # lengths = [174, 335]
    # colors = [mpl.colormaps.get_cmap('tab20b').colors[0], mpl.colormaps.get_cmap('tab20b').colors[16]]

    for j in range(len(treatments)):
        print(treatments[j])
        params = torch.load(f'../../data/sim_data/{treatments[j]}/{treatments[j]}_0/params.pth')
        path = f'../../data/exp_data/{treatments[j]}_csv'
        if treatments[j] == 'pulse':
            path = f'../../data/exp_data/{treatments[j]}_csv/For_Manuscript'
        exp_data = get_data(path)
        exp_colony_area, exp_colony_iqr, exp_clone_area, exp_clone_iqr = average_over_area(exp_data, length=lengths[j] + 1)
        sim_colony_area, sim_colony_area_iqr, sim_clone_area, sim_clone_area_iqr = get_sim_data(replicates[j], treatments[j])

        start_point = params['start_point']

        x_sim = np.linspace(0, 3239, 3240)/20
        x_exp = np.linspace(0, lengths[j], lengths[j])/2

        plot_exp(ax1, x_exp, exp_colony_area, exp_colony_iqr, colors[j], clone=False)
        plot_sim(ax1, x_sim, sim_colony_area, sim_colony_area_iqr, colors[j], start_point, clone=False)

        plot_exp(ax2, x_exp, exp_clone_area, exp_clone_iqr, colors[j], clone=True)
        plot_sim(ax2, x_sim, sim_clone_area, sim_clone_area_iqr, colors[j], start_point, clone=True)

        ax1.set_xlim(0, 284/2)
        ax1.set_ylim(0, 71)

        ax2.set_xlim(0, 284/2)
        ax2.set_ylim(0, 1)

        ax1.legend(loc='lower right', frameon=False)
        ax2.legend(loc='upper left', frameon=False)
        ax1.set_ylabel(r'Total area (mm²)', fontsize=7)
        ax2.set_ylabel(r'Resistant fraction', fontsize=7)

    plot_treat_bars(fig1, ax1)
    plot_treat_bars(fig2, ax2)

    fig1.savefig('compare_colony_area.pdf', bbox_inches='tight', transparent=True)
    fig2.savefig('compare_clone_area.pdf', bbox_inches='tight', transparent=True)
    fig1.show()
    fig2.show()


if __name__ == "__main__":
    replicates = [20, 20, 20]
    treatments = ['met_4_18', 'met_6_5_18', 'met_9_18']
    # replicates = [20, 20]
    # treatments = ['no_treatment', 'continuous_dose']
    # create_data(replicates, treatments)
    plot_comparison(replicates, treatments)

