import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap
  
"""
Run the demo.py script to generate the necessary data before running this script!
"""

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
    sen = np.load(f'{path}/sensitive.npy')[351:]
    res = np.load(f'{path}/resistant.npy')[351:]
    nut = np.load(f'{path}/nutrients.npy')[351:]
    params = torch.load(f'{path}/params.pth')

    radii_sen = []
    radii_res = []
    nut_front = []
    for i in range(len(sen)):
        sen_thresholded = np.where(sen[i] > (1 / params['mutation_scaling']), 1, 0)
        res_thresholded = np.where(res[i] > (1 / params['mutation_scaling']), 1, 0)

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

        for j in range(nut[i].shape[0]):
            if 0.01 >= nut[i][100, j]:
                nut_front.append(100 - j)
                break
        else:
            nut_front.append(0)
        
    colony_front = []
    for i in range(len(sen)):
        if radii_sen[i] > radii_res[i]:
            colony_front.append(radii_sen[i])
        else:
            colony_front.append(radii_res[i])
   
    return np.array(colony_front), nut[:,100,:], np.array(nut_front)

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

def rolling_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def generate_kymograph_from_sim_data(path):
    sensitive = np.load(f'{path}/sensitive.npy')[351:]
    resistant = np.load(f'{path}/resistant.npy')[351:]
    params = torch.load(f'{path}/params.pth')

    kymograph = []
    for i in range(len(sensitive)):
        if i % 20 != 0:
            continue
        sen = sensitive[i]
        res = resistant[i]
        def normalize_array(x, gamma, vmin=None, vmax=None):
            if vmin is None or vmax in None:
                vmin = np.nanmin(x)
                vmax = np.nanmax(x)
            x = (x - vmin) / (vmax - vmin + 1e-16)
            x = np.clip(x ** (1 / gamma), 0, 1)
            return x

        def mono_colormap(color_hex, name='mono'):
            return LinearSegmentedColormap.from_list(name, [(0, 'black'), (1, color_hex)], N=256)

        def apply_cmap(x, cmap):
            rgba = cmap(x)
            return rgba[..., :3]

        sen_norm = normalize_array(sen, 10)
        res_norm = normalize_array(res, 10)

        goldenrod = '#DAA520'
        royalblue = '#4169E1'

        sen_cmap = mono_colormap(royalblue, 'sensitive')
        res_cmap = mono_colormap(goldenrod, 'resistant')

        sen_rgb = apply_cmap(sen_norm, sen_cmap)
        res_rgb = apply_cmap(res_norm, res_cmap)

        alpha_sen, alpha_res = sen_norm[..., None], res_norm[..., None]
        rgb_add = res_rgb * alpha_res + sen_rgb * alpha_sen * (1 - alpha_res)

        alpha = np.where(sen > 1 / params['mutation_scaling'], 1, 0)
        alpha = np.where(res > 1 / params['mutation_scaling'], 1, alpha)
        rgba = np.dstack([rgb_add, alpha])

        row = rgba[100, :100]
        kymograph.append(row[::-1])
    kymograph = np.array(kymograph)
    return kymograph

path = 'demo/demo_data/met_6_5_18/met_6_5_18_0'
rad_sen, nut, nut_front = load_sim_data(path)
kymograph = generate_kymograph_from_sim_data(path)

nut_threshold_low = 1/(np.exp(2)+1)
nut_threshold_high = 1/(np.exp(-2)+1)

growth_layer_low = []
growth_layer_high = []
for i in range(rad_sen.shape[0]):
    layer_thickness_low = 100
    layer_thickness_high = 100
    for j in range(len(nut[i, :100])):
        if nut[i, j] >= nut_threshold_low:
            layer_thickness_low -= 1
        if nut[i, j] >= nut_threshold_high:
            layer_thickness_high -= 1
    growth_layer_low.append(layer_thickness_low)
    growth_layer_high.append(layer_thickness_high)
growth_layer_low = np.array(growth_layer_low)
growth_layer_high = np.array(growth_layer_high)

d = np.linspace(0, len(growth_layer_low)/20, len(growth_layer_low))

params = torch.load(f'{path}/params.pth')

px_to_mm = 13.76 * 8.648 / 1e3
Ny = kymograph.shape[1]

treat_starts_test, treat_ends_test = calc_treatment_efficacy(int(6.5*20), int(18*20), params)

plt.figure(figsize=(7.3/2, 7.1/3.5))
for i in range(len(treat_starts_test)):
        plt.axvspan(treat_starts_test[i] / 20, treat_ends_test[i] / 20, color='#bfbfbf', alpha=1, lw=0, zorder=0)
plt.imshow(kymograph.transpose(1,0,2), origin='lower', aspect='auto',  extent=[0, len(rad_sen)/20, 0, Ny * px_to_mm])
plt.ylim(0, 50*px_to_mm)
plt.xlim(0, 150)
plt.plot(d, rolling_average(growth_layer_low * px_to_mm, window_size=90), label=r'$\lambda_\mathrm{low}$', color='#e34234', linestyle=':')
plt.plot(d, rolling_average(growth_layer_high * px_to_mm, window_size=90), label=r'$\lambda_\mathrm{high}$', color='#e34234', linestyle='-.')
plt.xlabel('Time (h)')
plt.ylabel('Radial position (mm)')
plt.legend(frameon=False, loc='upper left', ncol=2)
plt.tight_layout()
plt.savefig(r'demo/demo_figures/kymo_demo.pdf', dpi=300, transparent=True)
plt.show()