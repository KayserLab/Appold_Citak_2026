import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import to_rgba

from matplotlib.collections import LineCollection
from skimage import measure


def load_data(path):
    sensitive = np.load(f'{path}/sensitive.npy')
    resistant = np.load(f'{path}/resistant.npy')
    nutrients = np.load(f'{path}/nutrients.npy')
    treatment_schedule = np.load(f'{path}/treatment_times.npy')
    treatment_efficacy = np.load(f'{path}/treatment_efficacy.npy')
    params = torch.load(f'{path}/params.pth', weights_only=False)
    sensitive = np.where((sensitive - 1 / params['mutation_scaling']) > 0, sensitive, 0) / np.max(sensitive)
    resistant = np.where((resistant - 1 / params['mutation_scaling']) > 0, resistant, 0) / np.max(resistant)
    nutrients = nutrients / np.max(nutrients)
    return nutrients[351::2], sensitive[351::2], resistant[351::2], treatment_schedule[351::2], treatment_efficacy[351::2]


def animate_simulation(path, save_name, fps, save_path, nut_threshold, plot_nutes=False):
    nutrients, sensitive, resistant, treatment_schedule, treatment_efficacy = load_data(path)

    def normalize_array(x, gamma, vmin=None, vmax=None):
        if vmin is None or vmax in None:
            vmin = np.nanmin(x)
            vmax = np.nanmax(x)
        x = (x - vmin) / (vmax - vmin + 1e-16)
        x = np.clip(x**(1/gamma), 0, 1)
        return x

    def mono_colormap(color_hex, name='mono'):
        return LinearSegmentedColormap.from_list(name, [(0, 'black') , (1, color_hex)], N=256)

    def apply_cmap(x, cmap):
        rgba = cmap(x)
        return rgba[..., :3]

    def make_boundary_mask(shape, width=2):
        h, w = shape
        m = np.zeros((h, w), dtype=bool)
        m[:, :width] = True
        m[:, -width:] = True
        m[:width, :] = True
        m[-width:, :] = True
        return m

    def overlay_rgba_from_mask(mask, color="#bfbfbf", alpha=1.0):
        rgba = np.zeros((*mask.shape, 4), dtype=float)
        r, g, b, _ = to_rgba(color)
        rgba[mask, 0] = r
        rgba[mask, 1] = g
        rgba[mask, 2] = b
        rgba[mask, 3] = alpha
        return rgba

    def update(frame):
        time_text.set_text(f'Time: {frame / 10:.1f} h')

        sen = sensitive[frame]
        res = resistant[frame]
        nut = nutrients[frame]

        sen_norm = normalize_array(sen, 5)
        res_norm = normalize_array(res, 2.5)

        goldenrod = '#DAA520'
        royalblue = '#4169E1'

        sen_cmap = mono_colormap(royalblue, 'sensitive')
        res_cmap = mono_colormap(goldenrod, 'resistant')

        sen_rgb = apply_cmap(sen_norm, sen_cmap)
        res_rgb = apply_cmap(res_norm, res_cmap)

        alpha_sen, alpha_res = sen_norm[..., None], res_norm[..., None]
        rgb_add = res_rgb * alpha_res + sen_rgb * alpha_sen * (1 - alpha_res)

        alpha = np.where(sensitive[frame] > 1 / 2184.07516852, 1, 0)
        alpha = np.where(resistant[frame] > 1 / 2184.07516852, 1, alpha)
        temp_img = np.dstack([rgb_add, alpha])

        temp_img[:, :2, :] = 1
        temp_img[:, -2:, :] = 1
        temp_img[:2, :, :] = 1
        temp_img[-2:, :, :] = 1

        temp_img[:, :2, :3] = (256 - 65 * treatment_schedule[frame-1])/256
        temp_img[:, -2:, :3] = (256 - 65 * treatment_schedule[frame-1])/256
        temp_img[:2, :, :3] = (256 - 65 * treatment_schedule[frame-1])/256
        temp_img[-2:, :, :3] = (256 - 65 * treatment_schedule[frame-1])/256

        img.set_array(temp_img)

        lc.set_segments(contours_to_segments(nut, nut_threshold))

        return img,

    def update_nuts(frame):
        time_text.set_text(f'Time: {frame/10:.1f} h')
        time_text_nut.set_text(f'Time: {frame/10:.1f} h')

        sen = sensitive[frame]
        res = resistant[frame]

        sen_norm = normalize_array(sen, 5)
        res_norm = normalize_array(res, 2.5)

        goldenrod = '#DAA520'
        royalblue = '#4169E1'

        sen_cmap = mono_colormap(royalblue, 'sensitive')
        res_cmap = mono_colormap(goldenrod, 'resistant')

        sen_rgb = apply_cmap(sen_norm, sen_cmap)
        res_rgb = apply_cmap(res_norm, res_cmap)

        alpha_sen, alpha_res = sen_norm[..., None], res_norm[..., None]
        rgb_add = res_rgb * alpha_res + sen_rgb * alpha_sen * (1 - alpha_res)

        alpha = np.where(sensitive[frame] > 1 / 2184.07516852, 1, 0)
        alpha = np.where(resistant[frame] > 1 / 2184.07516852, 1, alpha)
        temp_img = np.dstack([rgb_add, alpha])

        img.set_array(temp_img)

        nut_temp = nutrients[frame]
        nuts.set_data(nut_temp)

        a = np.clip(treatment_schedule[frame - 1], 0, 1)
        boundary_img0.set_data(overlay_rgba_from_mask(boundary_mask, "#bfbfbf", alpha=a))
        boundary_img1.set_data(overlay_rgba_from_mask(boundary_mask, "#bfbfbf", alpha=a))

        return img, nuts, boundary_img0, boundary_img1

    if  plot_nutes:
        fig, ax = plt.subplots(figsize=(12,6), dpi=100, nrows=1, ncols=2)

        sen = sensitive[0]
        res = resistant[0]

        sen_norm = normalize_array(sen, 5)
        res_norm = normalize_array(res, 2.5)

        goldenrod = '#DAA520'
        royalblue = '#4169E1'

        sen_cmap = mono_colormap(royalblue, 'sensitive')
        res_cmap = mono_colormap(goldenrod, 'resistant')

        sen_rgb = apply_cmap(sen_norm, sen_cmap)
        res_rgb = apply_cmap(res_norm, res_cmap)

        alpha_sen, alpha_res = sen_norm[..., None], res_norm[..., None]
        rgb_add = res_rgb * alpha_res + sen_rgb * alpha_sen * (1 - alpha_res)

        alpha = np.where(sensitive[0] > 1 / 2184.07516852, 1, 0)
        alpha = np.where(resistant[0] > 1 / 2184.07516852, 1, alpha)
        rgb = np.dstack([rgb_add, alpha])
        img = ax[0].imshow(rgb, interpolation='none')
        colors = [(0, (1, 1, 1)),  # white
                  (1, (227 / 255, 66 / 255, 52 / 255))]  # vermilion red
        cmap = LinearSegmentedColormap.from_list("black_green_white", colors, N=256)

        nuts = ax[1].imshow(nutrients[0], cmap=cmap, interpolation='none', vmin=0, vmax=1)

        boundary_mask = make_boundary_mask(sensitive[0].shape, width=2)
        boundary_rgba0 = overlay_rgba_from_mask(boundary_mask, color="#bfbfbf", alpha=1.0)

        boundary_img0 = ax[0].imshow(boundary_rgba0, interpolation='none')
        boundary_img1 = ax[1].imshow(boundary_rgba0, interpolation='none')

        nuts.set_clim(vmin=0, vmax=1)
        ax[0].axis('off')
        ax[1].axis('off')

        time_text = ax[0].text(4, 4, 'Time: 0.0 h', color='black', fontsize=12, verticalalignment='top')
        time_text_nut = ax[1].text(4, 4, 'Time: 0.0 h', color='black', fontsize=12, verticalalignment='top')

        one_mm_to_px = 2 * 1e3 / (13.76 * 8.648)
        ax[0].hlines(y=img.get_array().shape[0] - 10, xmin=10, xmax=10 + one_mm_to_px, colors='black', linewidth=4)
        ax[0].text(2 + one_mm_to_px, img.get_array().shape[0] - 12, f'2 mm', color='black', fontsize=7, ha='center')
        ax[1].hlines(y=img.get_array().shape[0] - 10, xmin=10, xmax=10 + one_mm_to_px, colors='black', linewidth=4)
        ax[1].text(2 + one_mm_to_px, img.get_array().shape[0] - 12, f'2 mm', color='black', fontsize=7, ha='center')

        ani = animation.FuncAnimation(fig, update_nuts, frames=len(sensitive), interval=100, blit=True)
    else:
        fig, ax = plt.subplots(figsize=(6,6), dpi=100)
        sen = sensitive[0]
        res = resistant[0]
        nut = nutrients[0]

        sen_norm = normalize_array(sen, 5)
        res_norm = normalize_array(res, 2.5)

        goldenrod = '#DAA520'
        royalblue = '#4169E1'

        sen_cmap = mono_colormap(royalblue, 'sensitive')
        res_cmap = mono_colormap(goldenrod, 'resistant')

        sen_rgb = apply_cmap(sen_norm, sen_cmap)
        res_rgb = apply_cmap(res_norm, res_cmap)

        alpha_sen, alpha_res = sen_norm[..., None], res_norm[..., None]
        rgb_add = res_rgb * alpha_res + sen_rgb * alpha_sen * (1 - alpha_res)

        alpha = np.where(sensitive[0] > 1 / 2184.07516852, 1, 0)
        alpha = np.where(resistant[0] > 1 / 2184.07516852, 1, alpha)
        rgb = np.dstack([rgb_add, alpha])
        img = ax.imshow(rgb, interpolation='none')
        ax.axis('off')

        time_text = ax.text(4, 4, 'Time: 0.0 h', color='black', fontsize=12, verticalalignment='top')

        one_mm_to_px = 2 * 1e3 / (13.76 * 8.648)
        ax.hlines(y=img.get_array().shape[0] - 10, xmin=10, xmax=10 + one_mm_to_px, colors='black', linewidth=4)
        ax.text(2 + one_mm_to_px, img.get_array().shape[0] - 12, f'2 mm', color='black', fontsize=7, ha='center')

        lc = LineCollection([], colors='red', linewidths=1.75, linestyles='solid')
        ax.add_collection(lc)

        def contours_to_segments(z, level):
            contours = measure.find_contours(z, level=level)
            return [np.column_stack((c[:, 1], c[:, 0])) for c in contours]

        lc.set_segments(contours_to_segments(nut, nut_threshold))

        ani = animation.FuncAnimation(fig, update, frames=len(sensitive), interval=100, blit=True)

    video_writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
    fig.set_tight_layout(True)
    ani.save(f'{save_path}/{save_name}.mp4', writer=video_writer)


def main():
    nut_threshold = 1/(np.exp(2)+1)
    path = 'data/demo/met_6_5_18/met_6_5_18_0'
    save_path = 'videos'
    animate_simulation(path, 'met_6_5_18_video', 90, save_path, nut_threshold, plot_nutes=True)
    # path = 'results/pulse_set_mut_pos'
    # save_path = 'results'
    # animate_simulation(path, 'pulse_set_mut_pos_video_new', 90, save_path, nut_threshold, plot_nutes=False)


if __name__ == "__main__":
    main()
