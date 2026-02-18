import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import lineStyles

plt.rcParams.update({'font.size': 6,
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

folder = 'last_sweep_v0'

median_total = np.load(f'../../data/sweep_arrays/{folder}_size_array.npy')
q1_total = np.load(f'../../data/sweep_arrays/{folder}_size_q1_array.npy')
q3_total = np.load(f'../../data/sweep_arrays/{folder}_size_q3_array.npy')
ratio_total = np.load(f'../../data/sweep_arrays/{folder}_ratio_array.npy')
ratio_q1_total = np.load(f'../../data/sweep_arrays/{folder}_ratio_q1_array.npy')
ratio_q3_total = np.load(f'../../data/sweep_arrays/{folder}_ratio_q3_array.npy')
print(median_total.shape)
x = np.arange(median_total[0].shape[0])/2

fig, ax1 = plt.subplots(figsize=(2.1, 1.35))

# plt.axvline(4, color=mpl.colormaps.get_cmap('tab20b').colors[4], linestyle='--', linewidth=0.75, alpha=0.8)
# plt.axvline(6.5, color=mpl.colormaps.get_cmap('tab20b').colors[8], linestyle='--', linewidth=0.75, alpha=0.8)
# plt.axvline(9, color=mpl.colormaps.get_cmap('tab20b').colors[12], linestyle='--', linewidth=0.75, alpha=0.8)

ax1.set_ylabel('TTP (h)', color='c')
ax1.set_xlabel(r'$\tau_{\mathrm{on}}$ (h)')

# ax1.plot(x, median_total[0][:,24], 'gray', label=r'$\tau_{\mathrm{off}}$ = 12 h')
# ax1.fill_between(x, q1_total[0][:,24], q3_total[0][:,24], alpha=0.5, color='gray', linewidth=0)

ax1.plot(x[:41], median_total[0][:,36][:41], 'c') # , label=r'$\tau_{\mathrm{off}}$ = 18 h')
print(np.argmax(median_total[0][:,36]))
# ax1.axvline(np.argmax(median_total[0][:,36])/2, color='black', linestyle='--', linewidth=0.8)
# ax1.axvline(6.5, color='black', linestyle='--', linewidth=0.8)
# ax1.axvline(9, color='black', linestyle='--', linewidth=0.8)
ax1.fill_between(x[:41], q1_total[0][:,36][:41], q3_total[0][:,36][:41], alpha=0.5, color='c', linewidth=0)

# ax1.plot(x, median_total[0][:,48], 'm', label=r'$\tau_{\mathrm{off}}$ = 24 h')
# ax1.fill_between(x, q1_total[0][:,48], q3_total[0][:,48], alpha=0.5, color='m', linewidth=0)

ax2 = ax1.twinx()

ax2.set_ylabel('Resistant fraction', rotation=270, labelpad=10, color='m')
ax2.set_ylim(-0.05, 1.05)
ax2.plot(x[:41], ratio_total[0][:,36][:41], 'mx', markersize=3)
ax2.fill_between(x[:41], ratio_q1_total[0][:,36][:41], ratio_q3_total[0][:,36][:41], alpha=0.5, color='c', linewidth=0)


# ax2.plot(x, ratio_total[0][:,24], 'gray', marker='x', linestyle='None', markersize=3)
# ax2.fill_between(x, ratio_q1_total[0][:,24], ratio_q3_total[0][:,24], alpha=0.5, color='gray', linewidth=0)

# ax2.plot(x, ratio_total[0][:,48], 'mx', markersize=3)
# ax2.fill_between(x, ratio_q1_total[0][:,48], ratio_q3_total[0][:,48], alpha=0.5, color='m', linewidth=0)

# dots_y = [median_total[0][0, 36], median_total[0][12, 36], median_total[0][14, 36], median_total[0][18, 36]]
# dots_x = [0, 6, 7, 9]
# color = [0, 4, 8, 12]
#
# for i in range(len(dots_x)):
#     plt.plot(dots_x[i], dots_y[i], 'o', color=mpl.colormaps.get_cmap('tab20b').colors[color[i]], markersize=4)

# plt.title('Treatment off duration 18h')
# ax1.legend(frameon=False, loc='center right')
plt.tight_layout()
plt.savefig(fr'sweep_profile_{folder}.pdf', transparent=True, bbox_inches='tight')
plt.show()
