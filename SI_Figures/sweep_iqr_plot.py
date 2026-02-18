import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch

def build_sweep_indices(params):
    num_treatment_on_steps = int((params['treatment_on_max'] - params['treatment_on_min']) / params['treatment_on_step']) + 1
    treat_on = np.linspace(params['treatment_on_min'], params['treatment_on_max'], num_treatment_on_steps, dtype=np.int16)

    num_treatment_off_steps = int((params['treatment_off_max'] - params['treatment_off_min']) / params['treatment_off_step']) + 1
    treat_off = np.linspace(params['treatment_off_min'], params['treatment_off_max'], num_treatment_off_steps, dtype=np.int16)

    num_mutation_rate_steps = int((params['mutation_rate_max'] - params['mutation_rate_min']) / params['mutation_rate_step']) + 1
    mutation_rates = np.linspace(params['mutation_rate_min'], params['mutation_rate_max'], num_mutation_rate_steps)
    replicas = params['num_replicas']
    return treat_on, treat_off, mutation_rates, replicas

plt.rcParams.update({'font.size': 6,
                     'pdf.fonttype': 42,
                     'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial'],
                     'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Arial',
                     'mathtext.it': 'Arial:italic',
                     'mathtext.bf': 'Arial:bold',
                     })

def duty_cycle_func(x):
    a = 1/(1+(800/260))
    return -(a*x)/(a-1)

plt.rcParams['axes.labelsize'] = 7
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6

folder = 'last_sweep_v0'

# mean_total = np.load(f'sweep_arrays/sweep_total.npy')[0]
q1_total = np.load(f'../data/sweep_arrays/{folder}_size_q1_array.npy')[0]
q3_total = np.load(f'../data/sweep_arrays/{folder}_size_q3_array.npy')[0]
iqr_total = q3_total - q1_total
params = torch.load(f'../data/sweeps/{folder}/params.pth', map_location='cpu', weights_only=False)
treat_on, treat_off, mutation_rates, replicas = build_sweep_indices(params)
treat_on_len, treat_off_len, mutation_rates_len = len(treat_on), len(treat_off), len(mutation_rates)
steps_x = 40 // params['treatment_off_step']
steps_y = 40 // params['treatment_on_step']
x_ticks = [i // 20 if i % 20 == 0 else i for i in treat_off]
y_ticks = [i // 20 if i % 20 == 0 else i for i in treat_on]

mean_total = np.load(f'../data/sweep_arrays/{folder}_size_array.npy')
max_line = np.argmax(mean_total[0], axis=0)

colors = [
    (0, (0, 0, 0)),  # black
    (0.35, (0/255, 128/255, 0/255)), # green
    (1, (1, 1, 1))]  # white

cmap = mpl.colors.LinearSegmentedColormap.from_list("black_green_white", colors, N=256)

fig, axs = plt.subplots(figsize=(7.1*0.526, 2.6))
# axs.plot(min_total[1], min_total[0], 'o', color='cyan', markersize=3, zorder=20)
im0 = axs.imshow(iqr_total, interpolation='none', cmap=cmap, origin='lower', vmin=iqr_total.min(), vmax=iqr_total.max())
axs.plot(range(len(max_line))[24:],max_line[24:], color='white', linestyle=':', linewidth=1, zorder=0, label=r'$\tau*$')
# axs[0].set_title('Total Population Size', fontsize=6)
axs.set_xlabel(r'$\tau_{\mathrm{off}}$ (h)')
# axs.vlines(36, 0, 60, color='c', linestyle='--', zorder=0, linewidth=1)
# axs.vlines(24, 0, 60, color='gray', linestyle='--', zorder=0, linewidth=1)
# axs.vlines(48, 0, 60, color='m', linestyle='--', zorder=0, linewidth=1)
# x_dut = [0, 80]
# y_dut = [0, 26]
# axs.plot(x_dut, y_dut, linewidth=1, color='red')
# x_duty = np.linspace(0.5, 100, 200)
# axs.plot(x_duty, duty_cycle_func(x_duty), linewidth=1, color='red')
axs.set_xticks(np.arange(0, treat_off_len, 2*steps_x))
axs.set_xticklabels(x_ticks[::2*steps_x])
axs.set_yticks(np.arange(0, treat_on_len, 2*steps_y))
axs.set_yticklabels(y_ticks[::2*steps_y])
axs.set_ylabel(r'$\tau_{\mathrm{on}}$ (h)')
# axs[0].plot([52], [26], 'o', color='C2', markersize=3, zorder=5)
# axs.text(6, 52, 'overtreatment', color='white', fontdict=None, zorder=5)
# axs.text(74, 52, 'metronomic', color='white', fontdict=None)
# axs.text(60, 4, 'undertreatment', color='white', fontdict=None)
cbar0 = plt.colorbar(im0, ax=axs, pad=0.04, shrink=0.7)
cbar0.set_label(r'IQR TTP (h)', color='black', rotation=270, labelpad=10)

cmap_dots = mpl.colormaps.get_cmap('tab20b')
# axs.plot(36, 8, 'o', color=cmap_dots.colors[4], markersize=3, zorder=20)
# axs.plot(36, 13, 'o', color=cmap_dots.colors[8], markersize=3, zorder=2)
# axs.plot(0, 18, 'o', color=cmap_dots.colors[16], markersize=3, zorder=20)
# axs.plot(36, 18, 'o', color=cmap_dots.colors[12], markersize=3, zorder=2)
# axs.plot(36, 0, 'o', color=cmap_dots.colors[0], markersize=3, zorder=2)

leg = plt.legend(loc='upper right', frameon=False)
for text in leg.get_texts():
    text.set_color('white')
plt.tight_layout()
plt.savefig(fr'{mutation_rates[0]}_iqr_{folder}_SI.pdf', transparent=True)
plt.show()