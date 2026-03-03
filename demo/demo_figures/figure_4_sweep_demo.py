import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib as mpl
import os
import yaml


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

def find_project_root(current_dir, marker_file):
    current_dir = os.path.abspath(current_dir)
    while current_dir != os.path.dirname(current_dir):
        if marker_file in os.listdir(current_dir):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return None

def get_params():
    path = os.path.join(find_project_root(os.getcwd(), 'requirements.txt'), 'params.yaml')
    with open(path, 'r') as file:
        params = yaml.safe_load(file)['simulation_params']
    return params

def build_sweep_indices(params):
    num_treatment_on_steps = int((params['treatment_on_max'] - params['treatment_on_min']) / params['treatment_on_step']) + 1
    treat_on = np.linspace(params['treatment_on_min'], params['treatment_on_max'], num_treatment_on_steps, dtype=np.int16)

    num_treatment_off_steps = int((params['treatment_off_max'] - params['treatment_off_min']) / params['treatment_off_step']) + 1
    treat_off = np.linspace(params['treatment_off_min'], params['treatment_off_max'], num_treatment_off_steps, dtype=np.int16)

    num_mutation_rate_steps = int((params['mutation_rate_max'] - params['mutation_rate_min']) / params['mutation_rate_step']) + 1
    mutation_rates = np.linspace(params['mutation_rate_min'], params['mutation_rate_max'], num_mutation_rate_steps)
    replicas = params['num_replicas']
    return treat_on, treat_off, mutation_rates, replicas

def plot_existing_data(folder):
    mean_total, mean_resistant = np.load(f'demo/demo_data/{folder}_size_array.npy'), np.load(f'demo/demo_data/{folder}_ratio_array.npy')
    params = torch.load(f'demo/demo_data/sweep_params.pth', map_location='cpu', weights_only=False)
    treat_on, treat_off, mutation_rates, replicas = build_sweep_indices(params)
    treat_on_len, treat_off_len, mutation_rates_len = len(treat_on), len(treat_off), len(mutation_rates)

    cmap_dots = mpl.colormaps.get_cmap('tab20b')
    steps_x = 40 // params['treatment_off_step']
    steps_y = 40 // params['treatment_on_step']
    x_ticks = [i // 20 if i % 20 == 0 else i for i in treat_off]
    y_ticks = [i // 20 if i % 20 == 0 else i for i in treat_on]

    for mutation_rate in range(mutation_rates_len):
        colors = [
            (0, (0, 0, 0)),  # black
            (0.35, (0/255, 128/255, 0/255)), # green
            (1, (1, 1, 1))]  # white

        cmap = mpl.colors.LinearSegmentedColormap.from_list("black_green_white", colors, N=256)
        max_line = np.argmax(mean_total[mutation_rate], axis=0)

        fig, axs = plt.subplots(figsize=(2.7, 1.8))
        im0 = axs.imshow(mean_total[mutation_rate][:41, :81], interpolation='none', cmap=cmap, origin='lower', vmin=mean_total[mutation_rate].min(), vmax=mean_total[mutation_rate].max())
        axs.plot(range(len(max_line))[24:81],max_line[24:81], color='black', linestyle=':', linewidth=1, zorder=0, label=r'$\tau*$')
        axs.set_xlabel(r'$\tau_{\mathrm{off}}$ (h)')
        axs.set_xticks(np.arange(0, treat_off_len, 2*steps_x))
        axs.set_xticklabels(x_ticks[::2*steps_x])
        axs.set_yticks(np.arange(0, treat_on_len, 2*steps_y))
        axs.set_yticklabels(y_ticks[::2*steps_y])
        axs.set_ylabel(r'$\tau_{\mathrm{on}}$ (h)')
        axs.text(3, 34, 'overtreatment', color='white', fontdict=None, zorder=5)
        axs.text(44, 3, 'undertreatment', color='white', fontdict=None)
        axs.vlines(36, 0, 40, color='c', linestyle='--', zorder=0, linewidth=1)

        cbar0 = plt.colorbar(im0, ax=axs, pad=0.04, shrink=0.55)
        cbar0.set_label('TTP (h)', color='black', rotation=270, labelpad=10)

        axs.plot(36, 8, 'o', color=cmap_dots.colors[4], markersize=4, zorder=20, markeredgecolor='white', markeredgewidth=0.4)
        axs.plot(36, 13, 'o', color=cmap_dots.colors[8], markersize=4, zorder=2, markeredgecolor='white', markeredgewidth=0.4)
        axs.plot(36, 18, 'o', color=cmap_dots.colors[12], markersize=4, zorder=2, markeredgecolor='white', markeredgewidth=0.4)

        plt.tight_layout()
        leg = plt.legend(loc='upper right', frameon=False, bbox_to_anchor=(1, 1.05))
        for text in leg.get_texts():
            text.set_color('white')
        plt.savefig(fr'demo/demo_figures/{mutation_rates[mutation_rate]}_{folder}_median.pdf', transparent=True, bbox_inches='tight')
        plt.show()
        plt.close()

        colors = [(65/255, 105/255, 225/255), (218/255, 165/255, 32/255)]
        cmap = mpl.colors.LinearSegmentedColormap.from_list("blue_yellow", colors, N=256)

        fig, axs = plt.subplots(figsize=(2.7, 1.8))
        im1 = axs.imshow(mean_resistant[mutation_rate][:41, :81], cmap=cmap, origin='lower', interpolation='none', vmin=0, vmax=1)
        axs.plot(range(len(max_line))[24:81],max_line[24:81], color='white', linestyle=':', linewidth=1, zorder=0)
        axs.set_xticks(np.arange(0, treat_off_len, 2*steps_x))
        axs.set_xticklabels(x_ticks[::2*steps_x])
        axs.set_yticks(np.arange(0, treat_on_len, 2*steps_y))
        axs.set_yticklabels(y_ticks[::2*steps_y])

        axs.set_xlabel(r'$\tau_{\mathrm{off}}$ (h)')
        axs.set_ylabel(r'$\tau_{\mathrm{on}}$ (h)')
        axs.text(3, 34, 'overtreatment', color='black', fontdict=None)
        axs.text(44, 3, 'undertreatment', color='black', fontdict=None)
        axs.vlines(36, 0, 40, color='m', linestyle='--', zorder=0, linewidth=1)

        cbar1 = plt.colorbar(im1, ax=axs, pad=0.04, shrink=0.55)
        cbar1.set_label('Resistant fraction', color='black', rotation=270, labelpad=10)

        axs.plot(36, 8, 'o', color=cmap_dots.colors[4], markersize=4, zorder=20, markeredgecolor='white', markeredgewidth=0.4)
        axs.plot(36, 13, 'o', color=cmap_dots.colors[8], markersize=4, zorder=2, markeredgecolor='white', markeredgewidth=0.4)
        axs.plot(36, 18, 'o', color=cmap_dots.colors[12], markersize=4, zorder=2, markeredgecolor='white', markeredgewidth=0.4)

        plt.tight_layout()
        plt.savefig(fr'demo/demo_figures/{mutation_rates[mutation_rate]}_{folder}_ratio.pdf', transparent=True, bbox_inches='tight')
        plt.show()
        plt.close()


def main():
    folder = 'sweep'
    plot_existing_data(folder)


if __name__ == "__main__":
    main()
