import numpy as np
import matplotlib.pyplot as plt

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

ax1.set_ylabel('TTP (h)', color='c')
ax1.set_xlabel(r'$\tau_{\mathrm{on}}$ (h)')

ax1.plot(x[:41], median_total[0][:,36][:41], 'c')
print(np.argmax(median_total[0][:,36]))
ax1.fill_between(x[:41], q1_total[0][:,36][:41], q3_total[0][:,36][:41], alpha=0.5, color='c', linewidth=0)

ax2 = ax1.twinx()

ax2.set_ylabel('Resistant fraction', rotation=270, labelpad=10, color='m')
ax2.set_ylim(-0.05, 1.05)
ax2.plot(x[:41], ratio_total[0][:,36][:41], 'mx', markersize=3)
ax2.fill_between(x[:41], ratio_q1_total[0][:,36][:41], ratio_q3_total[0][:,36][:41], alpha=0.5, color='c', linewidth=0)

plt.tight_layout()
plt.savefig(fr'sweep_profile_{folder}.pdf', transparent=True, bbox_inches='tight')
plt.show()
