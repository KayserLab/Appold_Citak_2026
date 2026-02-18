import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import torch

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

def load_sim_data(path):
    sen = np.load(f'{path}/sensitive.npy')[1200]
    res = np.load(f'{path}/resistant.npy')[1200]
    nut = np.load(f'{path}/nutrients.npy')[1200]
    params = torch.load(f'{path}/params.pth')

    radii_sen = []
    radii_res = []
    nut_front = []
    sen_thresholded = np.where(sen > (1 / params['mutation_scaling']), 1, 0)
    res_thresholded = np.where(res > (1 / params['mutation_scaling']), 1, 0)

    for j in range(sen_thresholded.shape[0]):
        if sen_thresholded[100, j] > 0:
            radii_sen.append(100 - j)
            break
    else:
        radii_sen.append(0)

    for j in range(res_thresholded.shape[0]):
        if res_thresholded[100, j] > 0:
            radii_res.append(100 - j)
            break
    else:
        radii_res.append(0)

    for j in range(nut.shape[0]):
        if 0.01 >= nut[100, j]:
            nut_front.append(100 - j)
            break
    else:
        nut_front.append(0)

    return np.array(radii_sen).item(), np.array(nut_front).item()

def calc_nut_int(nut_ar, radius):
    nut_int = np.sum(nut_ar[100, 100 - int(radius):100])
    return nut_int

def calc_effective_growth_layer(nut_ar, nut_int, radius):
    eff_growth_layer = nut_int/(nut_ar[100, 100 - int(radius)])
    return eff_growth_layer

print('To generate this data run the SI_Figures/no_mutation_kymo.py script with ct treatment!')
path = '../../data/sim_data/ct_no_mut'  # to generate this data run the SI_Figures/no_mutation_kymo.py script with ct treatment
sen = np.load(f'{path}/sensitive.npy')[1200]
res = np.load(f'{path}/resistant.npy')[1200]
nut = np.load(f'{path}/nutrients.npy')[1200]
params = torch.load(f'{path}/params.pth')

formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((3,3))

px_to_mm = 13.76*8.648/1e3
radius, nut_front = load_sim_data(path)

nut_threshold_low = 1/(np.exp(2)+1)
nut_threshold_high = 1/(np.exp(-2)+1)

layer_thickness_low = 100
layer_thickness_high = 100

for j in range(len(nut[100,:100])):
    if nut[100, j] >= nut_threshold_low:
        layer_thickness_low -= 1
    if nut[100, j] >= nut_threshold_high:
        layer_thickness_high -= 1
growth_layer_low = layer_thickness_low
growth_layer_high = layer_thickness_high

x = np.linspace(0, 99*px_to_mm, 100)
fig, ax = plt.subplots(figsize=(6.29921*(1.5/5), 6.29921*(2.3/5)*(3/5)), dpi=300)
ax1 = ax.twinx()
ax1.tick_params(axis='y', labelcolor='#e34234')
ax.tick_params(axis='y', labelcolor='royalblue', pad=2)
ax.plot(x, sen[100][:100][::-1]*params['mutation_scaling'], label='Sensitive', color='royalblue')
ax1.plot(x, nut[100][:100][::-1], label='Nutrients', color='#e34234')
ax.axvline(x=(radius)*px_to_mm, color='black', linestyle='--', lw=1.5, label='Colony front')
ax.axvline(x=(growth_layer_low)*px_to_mm, color='#e34234', linestyle=':', lw=1.5, label='Effective growth layer', zorder=20)
ax.axvline(x=(growth_layer_high)*px_to_mm, color='#e34234', linestyle='-.', lw=1.5, label='Effective growth layer', zorder=20)
ax.fill_between(x, 0, sen[100][:100][::-1]*params['mutation_scaling'], color='royalblue', alpha=0.3, lw=0)
ax1.axhline(1/(np.exp(2)+1), color='#e34234', lw=0.8, label='Nutrient threshold low', alpha=0.5)
ax1.axhline(1/(np.exp(-2)+1), color='#e34234', lw=0.8, label='Nutrient threshold high', alpha=0.5)

ax.set_xlim(0, 23*px_to_mm)
ax.set_ylim(0, 1*params['mutation_scaling'])
ax1.set_ylim(0, 1)
ax1.set_ylabel('Nutrient concentration', rotation=270, labelpad=8, color='#e34234')
ax.set_ylabel('Cell density, 1/deme', color='royalblue', labelpad=1)
ax.set_xlabel('Distance to Center (mm)')
plt.tight_layout()
plt.savefig(r'nutrient_profile.pdf', dpi=300, transparent=True)
plt.show()
