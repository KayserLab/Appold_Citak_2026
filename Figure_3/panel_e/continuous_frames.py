import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import source.run_core as rc
import matplotlib as mpl


plt.rcParams.update({'font.size': 7,
                     'pdf.fonttype': 42,
                     'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial'],
                     'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Arial',
                     'mathtext.it': 'Arial:italic',
                     'mathtext.bf': 'Arial:bold'})

def create_simulation_data():
    rc.main(10, 0, save_dir=f'Figure_3/panel_e/continuous')

def load_data():
    path = '../../data/sim_data/continuous_dose/continuous_dose_0'
    sensitive = np.load(f'{path}/sensitive.npy')
    resistant = np.load(f'{path}/resistant.npy')
    nutrients = np.load(f'{path}/nutrients.npy')
    treatment_schedule = np.load(f'{path}/treatment_times.npy')
    treatment_efficacy = np.load(f'{path}/treatment_efficacy.npy')
    params = torch.load(f'{path}/params.pth', weights_only=False)
    sensitive = np.where((sensitive - 1/params['mutation_scaling']) > 0, sensitive, 0) / np.max(sensitive)
    resistant = np.where((resistant - 1/params['mutation_scaling']) > 0, resistant, 0) / np.max(resistant)
    nutrients = nutrients / np.max(nutrients)
    return nutrients, sensitive, resistant, treatment_schedule, treatment_efficacy, params


def plot_frame(save_name, frame):
    nutrients, sensitive, resistant, treatment_schedule, treatment_efficacy, params = load_data()

    sen = sensitive[frame]
    res = resistant[frame]

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

    sen_norm = normalize_array(sen, 5)
    res_norm = normalize_array(res, 2.5)
    gamma_sen = 5
    gamma_res = 2.5

    goldenrod = '#DAA520'
    royalblue = '#4169E1'

    sen_cmap = mono_colormap(royalblue, 'sensitive')
    res_cmap = mono_colormap(goldenrod, 'resistant')

    sen_rgb = apply_cmap(sen_norm, sen_cmap)
    res_rgb = apply_cmap(res_norm, res_cmap)

    alpha_sen, alpha_res = sen_norm[..., None], res_norm[..., None]
    rgb_add = res_rgb * alpha_res + sen_rgb * alpha_sen * (1 - alpha_res)

    fig, ax = plt.subplots(figsize=(2,2), dpi=300)
    alpha = np.where(sensitive[frame] > 1/params['mutation_scaling'], 1, 0)
    alpha = np.where(resistant[frame] > 1/params['mutation_scaling'], 1, alpha)
    rgba = np.dstack([rgb_add, alpha])
    ax.imshow(rgba, interpolation='none')
    if frame == 2251:
        ax.hlines(100, 0, 100, color='black', linewidth=0.6)
    ax.axis('off')
    plt.savefig(f'plots_continuous/{save_name}_{frame}_cells.pdf', dpi=300, transparent=True, bbox_inches='tight')
    plt.close()

    colors = [(0, (1, 1, 1)),  # white
              (1, (227 / 255, 66 / 255, 52 / 255))]  # vermilion red
    cmap = mpl.colors.LinearSegmentedColormap.from_list("black_green_white", colors, N=256)

    fig, ax = plt.subplots(figsize=(6,6), dpi=300)
    ax.imshow(nutrients[frame], interpolation='none', cmap=cmap)
    if frame == 651:
        scale_bar_length = 1000/(13.76 * 8.648)  # mm = 1000µm / (sim_to_exp_px * exp_px_to_µm)
        ax.hlines(190, 10, 10 + scale_bar_length*2, color='black', linewidth=3)
    ax.axis('off')
    plt.savefig(fr'plots_continuous/{save_name}_{frame}_nutrients.pdf', dpi=300, transparent=True, bbox_inches='tight')
    plt.close()

    # Colorbars
    nut = nutrients[frame]
    norm_res = Normalize(vmin=res.min(), vmax=res.max())
    norm_sen = Normalize(vmin=sen.min(), vmax=sen.max())
    norm_nut = Normalize(vmin=nut.min(), vmax=nut.max())

    cmap_res = LinearSegmentedColormap.from_list('resist', [(0, 0, 0), (218/255, 165/255, 32/255)])
    cmap_sen = LinearSegmentedColormap.from_list('sens', [(0, 0, 0), (65/255, 105/255, 225/255)])
    cmap_nut = plt.get_cmap(cmap)

    sm_res = ScalarMappable(norm=norm_res, cmap=cmap_res)
    sm_sen = ScalarMappable(norm=norm_sen, cmap=cmap_sen)
    sm_nut = ScalarMappable(norm=norm_nut, cmap=cmap_nut)

    for sm, label, fname in [(sm_res, fr'Resistant Density ($\gamma$ = {gamma_res})', fr'plot_cbars/{save_name}_cbar_resistant.pdf'),
                             (sm_sen, fr'Sensitive Density ($\gamma$ = {gamma_sen})', fr'plot_cbars/{save_name}_cbar_sensitive.pdf'),
                             (sm_nut, 'Nutrient Concentration', fr'plot_cbars/{save_name}_cbar_nutrients.pdf')]:

        fig = plt.figure(figsize=(2, 0.15), dpi=300)
        cax = fig.add_axes([0.05, 0.25, 0.9, 0.5])
        cb = plt.colorbar(sm, cax=cax, orientation='horizontal')
        cb.set_label(label, rotation=0, ha='center', va='bottom', labelpad=-15, fontsize=7)
        cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        cax.tick_params(bottom=True, top=False, labelbottom=True, labeltop=False, labelsize=6, width=0.6, length=3, direction='inout', pad=1)
        cax.spines['top'].set_visible(False)
        cax.spines['right'].set_visible(False)
        cax.spines['left'].set_visible(False)
        fig.savefig(fname, bbox_inches='tight', transparent=True)
        plt.close(fig)


def main():
    # continuous
    # create_simulation_data()  # only use if you want to test something (you will need to adjust the paths in plot_frame as well)
    for i in [351 + 300, 351 + 700, 351 + 1100, 351 + 1500, 351 + 1900]:
        plot_frame('continuous', frame=i)


if __name__ == "__main__":
    main()