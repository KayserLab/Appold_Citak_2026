import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import matplotlib as mpl


plt.rcParams.update({'font.size': 6,
                     'pdf.fonttype': 42,
                     'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial'],
                     'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Arial',
                     'mathtext.it': 'Arial:italic',
                     'mathtext.bf': 'Arial:bold',
                     'axes.titlesize': 6,
                     'axes.labelsize': 6,
                     'axes.linewidth': 0.5,
                     'xtick.major.size': 2,
                     'ytick.major.size': 2,
                     'xtick.minor.size': 2,
                     'ytick.minor.size': 2,
                     'xtick.major.width': 0.5,
                     'ytick.major.width': 0.5,
                     'xtick.minor.width': 0.5,
                     'ytick.minor.width': 0.5,
                     'xtick.direction': 'out',
                     'ytick.direction': 'out',
                     'xtick.labelsize': 5,
                     'ytick.labelsize': 5,
                     'axes.labelpad': 1,
                     'xtick.major.pad': 2,
                     'ytick.major.pad': 2,
                     })
plt.rcParams['axes.labelsize'] = 6
plt.rcParams['xtick.labelsize'] = 5
plt.rcParams['ytick.labelsize'] = 5

def get_params():
    path = os.path.join(find_project_root(os.getcwd(), 'requirements.txt'), 'params.yaml')
    with open(path, 'r') as file:
        params = yaml.safe_load(file) # ['simulation_params']
    return params

def find_project_root(current_dir, marker_file):
    current_dir = os.path.abspath(current_dir)
    while current_dir != os.path.dirname(current_dir):  # Stop at the root of the file system
        if marker_file in os.listdir(current_dir):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return None

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

    return np.array(treatment_starts)/10, np.array(treatment_ends)/10


def get_data(path):
    exp_data_list = [i for i in os.listdir(path) if i.startswith('colony') and i.endswith('clonearea.csv')]
    exp_data = []
    for i in exp_data_list:
        exp_data.append(pd.read_csv(f'{path}/{i}'))
    return exp_data


def median_over_area_derivative(exp_data):
    dt = 0.5
    temp_sen = []
    temp_res = []
    temp_total = []
    for exp in exp_data:
        area_derivative_sen = np.gradient(exp['colony_area']*(8.648**2)/1e6 - exp['total_clone_area']*(8.648**2)/1e6, dt)
        area_derivative_total = np.gradient(exp['colony_area']*(8.648**2)/1e6, dt)
        area_derivative_res = np.gradient(exp['total_clone_area']*(8.648**2)/1e6, dt)
        temp_res.append(area_derivative_res)
        temp_sen.append(area_derivative_sen)
        temp_total.append(area_derivative_total)
    sen_growth_speed = np.median(np.array(temp_sen), axis=0)
    sen_iqr = np.percentile(np.array(temp_sen), [25, 75], axis=0)
    res_growth_speed = np.median(np.array(temp_res), axis=0)
    res_iqr = np.percentile(np.array(temp_res), [25, 75], axis=0)
    total_growth_speed = np.median(np.array(temp_total), axis=0)
    return rolling_median(sen_growth_speed, window_size=9), rolling_median(res_growth_speed, window_size=9), sen_iqr, res_iqr, rolling_median(total_growth_speed, window_size=9)

def rolling_median(data, window_size):
    padded_data = np.pad(data, (window_size // 2, window_size - window_size // 2 - 1), mode='edge')
    rolling_medians = np.array([np.median(padded_data[i:i + window_size]) for i in range(len(data))])
    return rolling_medians


def main(path, t_on, t_off, save_name, color):
    treatment_starts, treatment_ends = calc_treatment_efficacy(t_on, t_off, get_params())
    exp_data = get_data(path)
    median_area_derivative_sen, median_area_derivative_res, sen_iqr, res_iqr, total = median_over_area_derivative(exp_data)

    x = np.linspace(0, len(median_area_derivative_sen)/2, len(median_area_derivative_sen))
    fig_w = 1.8
    fig_h = 1.3
    ax_w = 1.25
    ax_h = 0.85
    left = 0.35
    bottom = 0.30

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([
        left / fig_w,
        bottom / fig_h,
        ax_w / fig_w,
        ax_h / fig_h])

    ax.plot(x, median_area_derivative_sen, linewidth=1, color=color, zorder=20, label='Sensitive', linestyle='--')
    ax.fill_between(x, rolling_median(sen_iqr[0], window_size=9), rolling_median(sen_iqr[1], window_size=9), color=color, alpha=0.3, lw=0, zorder=10)
    for i in range(len(treatment_starts)):
        ax.axvspan(treatment_starts[i]/2, treatment_ends[i]/2, color='#bfbfbf', lw=0)

    # ax.plot(treat_effic[::20], color='red', linestyle='--', label='Treatment Efficacy', linewidth=0.6)

    print(treatment_starts, treatment_ends)
    ax.set_ylim(0, 1.4)
    ax.set_xlim(0, 150)
    ax.set_xticks([0,25,50,75,100,125,150])

    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Area growth (mm²/h)')
    med_x_pos = [treatment_starts[i]/2 + (treatment_starts[i+1]/2 - treatment_starts[i]/2)/2 for i in range(len(treatment_starts)-1)]
    med_y_int = [np.sum(median_area_derivative_sen[int(treatment_starts[i]):int(treatment_starts[i+1])])/(treatment_starts[i+1] - treatment_starts[i]) for i in range(len(treatment_starts)-1)]

    ax.plot(x, median_area_derivative_res, lw=1, color=color, zorder=20, linestyle=':', label='Resistant')
    # ax.plot(x, total, lw=1, color='black', zorder=20, label='Total')
    ax.fill_between(x, rolling_median(res_iqr[0], window_size=9), rolling_median(res_iqr[1], window_size=9), color=color, alpha=0.3, zorder=10,
                    lw=0)

    ax.plot(med_x_pos[:-2], med_y_int[:-2], marker='o', color='green', markersize=1.5, linestyle='-.', label='Sensitive cycle average', zorder=30, lw=1)
    ax.legend(frameon=False, fontsize=5)#, loc='upper left')
    # plt.tight_layout()
    plt.savefig(f'{save_name}.pdf', transparent=True)
    plt.show()

    # fig, ax = plt.subplots(figsize=(1.8, 1.3), dpi=300)
    # ax.plot(x, median_area_derivative_res, lw=1, color=color, zorder=20, linestyle=':')
    # ax.fill_between(x, rolling_median(res_iqr[0], window_size=9), rolling_median(res_iqr[1], window_size=9), color=color, alpha=0.3, zorder=10, lw=0)
    # for i in range(len(treatment_starts)):
    #     ax.axvspan(treatment_starts[i]/2, 300, color='#bfbfbf', lw=0)

    # ax.plot(treat_effic[::20], color='red', linestyle='--', label='Treatment Efficacy', lw=0.6)

    # ax.set_ylim(-0.13, 0.7)
    # ax.set_xlim(0, 140)
    # ax.set_xticks([0,25,50,75,100,125])
    #
    # ax.set_xlabel('Time (h)')
    # ax.set_ylabel('Resistant growth (mm²/h)')

    # plt.tight_layout()
    # plt.savefig(f'plots/{save_name}_res.pdf', bbox_inches='tight', transparent=True)
    # plt.show()


if __name__ == "__main__":
    path = r'../../data/exp_data/met_6_5_18_csv'
    save_name = 'growth_speeds_6_5_18'
    treat_on = 130
    treat_off = 360

    # no_treatment = 0
    # 4_18 = 4
    # 6_18 = 8
    # 9_18 = 12
    # continuous = 16
    color = mpl.colormaps.get_cmap('tab20b').colors[8]
    main(path, treat_on, treat_off, save_name, color)