import numpy as np
import torch
import os
import yaml


def find_project_root(current_dir, marker_file):
    current_dir = os.path.abspath(current_dir)
    while current_dir != os.path.dirname(current_dir):  # Stop at the root of the file system
        if marker_file in os.listdir(current_dir):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return None

def get_params():
    path = os.path.join(find_project_root(os.getcwd(), 'requirements.txt'), 'params.yaml')
    with open(path, 'r') as file:
        params = yaml.safe_load(file) #['simulation_params']
    return params

def check_edge(img, x, y,):
    for i in range(x - 1, x + 2):
        for j in range(y - 1, y + 2):
            if i < 0 or j < 0 or i >= img.shape[0] or j >= img.shape[1]:
                return True
            if img[i, j] == 0:
                return True
    return False

def remove_edge(img, index_list, removal_layers=1):
    img0 = img.copy()
    for _ in range(removal_layers):
        img_temp = img0.copy()
        for x, y in index_list:
            if img_temp[x, y] == True:
                if check_edge(img_temp, x, y):
                    img0[x, y] = 0
    return img0

def sigmoid(x):
    y = 5.74744036e+02 / (1 + np.exp(6.50299401*(x-3.21720167e-01))) + -3.71899104
    return y

def has_breakout(sensitive, resistant, treat_effic, scaling):

    needed_num_mutations = int(sigmoid(treat_effic))

    if not np.any(resistant):
        return False

    inverted_sen = ~sensitive
    res_out = np.where(inverted_sen, resistant, False)
    res_out = np.where(res_out >= (needed_num_mutations / scaling), resistant, False)
    if np.sum(res_out) > 0:
        return True
    else:
        sensitive = remove_edge(sensitive, np.argwhere(sensitive))
        inverted_sen = ~sensitive
        res_out = np.where(inverted_sen, resistant, False)
        res_out = np.where(res_out >= (needed_num_mutations / scaling), resistant, False)
        if np.sum(res_out) > 0:
            return True
        else:
            return False

def build_sweep_indices(params):
    num_treatment_on_steps = int((params['treatment_on_max'] - params['treatment_on_min']) / params['treatment_on_step']) + 1
    treat_on = np.linspace(params['treatment_on_min'], params['treatment_on_max'], num_treatment_on_steps, dtype=np.int16)

    num_treatment_off_steps = int((params['treatment_off_max'] - params['treatment_off_min']) / params['treatment_off_step']) + 1
    treat_off = np.linspace(params['treatment_off_min'], params['treatment_off_max'], num_treatment_off_steps, dtype=np.int16)

    num_mutation_rate_steps = int((params['mutation_rate_max'] - params['mutation_rate_min']) / params['mutation_rate_step']) + 1
    mutation_rates = np.linspace(params['mutation_rate_min'], params['mutation_rate_max'], num_mutation_rate_steps)
    replicas = params['num_replicas']
    return treat_on, treat_off, mutation_rates, replicas

def create_array(folder):
    params = get_params()
    treat_on, treat_off, mutation_rates, replicas = build_sweep_indices(params)
    treat_on_len, treat_off_len, mutation_rates_len = len(treat_on), len(treat_off), len(mutation_rates)
    img_size = params['image_size']

    bytes_per_cond = img_size * img_size * np.dtype(np.float32).itemsize * replicas
    bytes_per_treat_effic = params['total_time'] * np.dtype(np.float32).itemsize * replicas

    mean_total = np.zeros((mutation_rates_len, treat_on_len, treat_off_len), dtype=float)
    var_total  = np.zeros((mutation_rates_len, treat_on_len, treat_off_len), dtype=float)
    median_total = np.zeros((mutation_rates_len, treat_on_len, treat_off_len), dtype=float)
    q1_total = np.zeros((mutation_rates_len, treat_on_len, treat_off_len), dtype=float)
    q3_total = np.zeros((mutation_rates_len, treat_on_len, treat_off_len), dtype=float)

    mean_total_area = np.zeros((mutation_rates_len, treat_on_len, treat_off_len), dtype=float)
    var_total_area = np.zeros((mutation_rates_len, treat_on_len, treat_off_len), dtype=float)
    median_total_area = np.zeros((mutation_rates_len, treat_on_len, treat_off_len), dtype=float)
    q1_total_area = np.zeros((mutation_rates_len, treat_on_len, treat_off_len), dtype=float)
    q3_total_area = np.zeros((mutation_rates_len, treat_on_len, treat_off_len), dtype=float)

    mean_resistant = np.zeros_like(mean_total)
    median_resistant = np.zeros_like(mean_total)
    breakouts_array = np.zeros_like(mean_total)
    ratios_array_mean = np.zeros_like(mean_total)
    ratios_array_median = np.zeros_like(mean_total)
    ratios_array_mean_area = np.zeros_like(mean_total)
    ratios_array_median_area = np.zeros_like(mean_total)
    ratio_median_res_total = np.zeros_like(mean_total)
    ratio_test_array = np.zeros_like(mean_total)

    size_array = np.zeros_like(mean_total)
    size_q1_array = np.zeros_like(mean_total)
    size_q3_array = np.zeros_like(mean_total)
    ratio_array = np.zeros_like(mean_total)
    ratio_q1_array = np.zeros_like(mean_total)
    ratio_q3_array = np.zeros_like(mean_total)

    ratio_endpoint_array = np.zeros_like(mean_total)
    ratio_endpoint_q1_array = np.zeros_like(mean_total)
    ratio_endpoint_q3_array = np.zeros_like(mean_total)

    for on in range(treat_on_len):
        for off in range(treat_off_len):
            for mut in range(mutation_rates_len):
                offset_bytes = (((on * treat_off_len) + off) * mutation_rates_len + mut) * bytes_per_cond
                offset_bytes_treat = (((on * treat_off_len) + off) * mutation_rates_len + mut) * bytes_per_treat_effic

                # sen = np.memmap(f'../data/{folder}/sensitive.dat', dtype=np.float32, mode='r', offset=offset_bytes, shape=(replicas, img_size, img_size))
                # res = np.memmap(f'../data/{folder}/resistant.dat', dtype=np.float32, mode='r', offset=offset_bytes, shape=(replicas, img_size, img_size))
                # treat_effic = np.memmap(f'../data/{folder}/efficacy.dat', dtype=np.float32, mode='r', offset=offset_bytes_treat, shape=(replicas, params['total_time']))
                size = np.memmap(f'{folder}/size.dat', dtype=np.float32, mode='r', offset=offset_bytes_treat, shape=(replicas, params['total_time']))
                ratio = np.memmap(f'{folder}/ratio.dat', dtype=np.float32, mode='r', offset=offset_bytes_treat, shape=(replicas, params['total_time']))

                # breakouts = []
                # for i in range(replicas):
                #     sensitive, resistant = np.where(sen[i] > 1 / params['mutation_scaling'], True, False), np.where(res[i] > 1 / params['mutation_scaling'], True, False)
                #
                #     breakout = has_breakout(sensitive, resistant, treat_effic[i][-1], params['mutation_scaling'])
                #     if breakout:
                #         breakouts.append(1)

                # sen_thresholded = np.where(sen > (1 / params['mutation_scaling']), sen / (1 / params['mutation_scaling']), 0)
                # res_thresholded = np.where(res > (1/params['mutation_scaling']), res/(1/params['mutation_scaling']), 0)
                # sen_thresholded_area = np.where(sen > (1 / params['mutation_scaling']), 1, 0)
                # res_thresholded_area = np.where(res > (1/params['mutation_scaling']), 1, 0)

                # sen_thresholded_ratio = np.where(sen > (1 / params['mutation_scaling']), sen, 0)
                # res_thresholded_ratio = np.where(res > (1 / params['mutation_scaling']), res, 0)
                # sen_ratio = np.where(sen_thresholded_ratio > res_thresholded_ratio, 1., 0.)
                # res_ratio = np.where(res_thresholded_ratio > sen_thresholded_ratio, 1., 0.)
                # sen_ratio_count_mean = np.mean(sen_ratio.sum(axis=(1,2))*(eval(params['sim_pixel_to_exp_pixel_factor'])**2)*8.648**2/1e6)
                # res_ratio_count_mean = np.mean(res_ratio.sum(axis=(1,2))*(eval(params['sim_pixel_to_exp_pixel_factor'])**2)*8.648**2/1e6)
                # sen_ratio_median = np.median(sen_ratio.sum(axis=(1,2))*(eval(params['sim_pixel_to_exp_pixel_factor'])**2)*8.648**2/1e6)
                # res_count_area = np.count_nonzero(res_ratio, axis=(1,2))*(eval(params['sim_pixel_to_exp_pixel_factor'])**2)*8.648**2/1e6
                # res_ratio_median = np.median(res_count_area)

                # total_array_area = np.stack([sen_thresholded_area, res_thresholded_area], axis=0)
                # total_array_area = np.sum(total_array_area, axis=0)
                # total_count_area = np.count_nonzero(total_array_area, axis=(1,2))*(eval(params['sim_pixel_to_exp_pixel_factor'])**2)*8.648**2/1e6
                # total_count_area_mean = total_count_area.mean()
                # total_count_area_median = np.median(total_count_area)
                # total_count_area_iqr = np.percentile(total_count_area, [25, 75])
                # total_count_area_var = total_count_area.var()
                #
                # total_array = np.stack([sen_thresholded, res_thresholded], axis=0)
                # total_array = np.sum(total_array, axis=0)
                # total_count = total_array.sum(axis=(1,2))
                # total_count_mean = total_count.mean()
                # total_count_median = np.median(total_count)
                # total_count_iqr = np.percentile(total_count, [25, 75])
                # total_count_var = total_count.var()

                # resistant_count_mean = res_thresholded.sum(axis=(1,2)).mean()
                # sensitive_count_mean = sen_thresholded.sum(axis=(1,2)).mean()
                # resistant_count_median = np.median(res_thresholded.sum(axis=(1, 2)))
                # sensitive_count_median = np.median(sen_thresholded.sum(axis=(1, 2)))
                #
                # mean_total[mut, on, off] = total_count_mean
                # var_total[mut, on, off] = total_count_var
                # median_total[mut, on, off] = total_count_median
                # q1_total[mut, on, off] = total_count_iqr[0]
                # q3_total[mut, on, off] = total_count_iqr[1]
                #
                # mean_total_area[mut, on, off] = total_count_area_mean
                # var_total_area[mut, on, off] = total_count_area_var
                # median_total_area[mut, on, off] = total_count_area_median
                # q1_total_area[mut, on, off] = total_count_area_iqr[0]
                # q3_total_area[mut, on, off] = total_count_area_iqr[1]
                #
                # mean_resistant[mut, on, off] = resistant_count_mean
                # median_resistant[mut, on, off] = resistant_count_median
                # breakouts_array[mut, on, off] = np.sum(breakouts)
                #
                # ratios_array_mean[mut, on, off] = resistant_count_mean / sensitive_count_mean
                # ratios_array_median[mut, on, off] = resistant_count_median / sensitive_count_median
                # ratios_array_mean_area[mut, on, off] = res_ratio_count_mean / sen_ratio_count_mean
                # ratios_array_median_area[mut, on, off] = res_ratio_median / sen_ratio_median
                # ratio_median_res_total[mut, on, off] = res_ratio_median / total_count_area_median

                ttp_size = []
                ttp_ratio = []
                ttp_ratio_endpoint = []

                for j, lst in enumerate(size):
                    for i, val in enumerate(lst):
                        if val >= 71:
                            ttp_size.append((i - params['start_point'])/20)
                            ttp_ratio.append(ratio[j, i])
                            ttp_ratio_endpoint.append(ratio[j, -1])
                            break
                    else:
                        ttp_size.append((len(lst) - 1 - params['start_point'])/20)
                        ttp_ratio.append(ratio[j, len(lst) - 1])
                        ttp_ratio_endpoint.append(ratio[j, -1])

                size_iqr = np.percentile(ttp_size, q=[25, 75])
                size_median = np.median(ttp_size)
                size_array[mut, on, off] = size_median
                size_q1_array[mut, on, off] = size_iqr[0]
                size_q3_array[mut, on, off] = size_iqr[1]

                ratio_iqr = np.percentile(ttp_ratio, q=[25, 75])
                ratio_median = np.median(ttp_ratio)
                ratio_array[mut, on, off] = ratio_median
                ratio_q1_array[mut, on, off] = ratio_iqr[0]
                ratio_q3_array[mut, on, off] = ratio_iqr[1]

                ratio_endpoint_iqr = np.percentile(ttp_ratio_endpoint, q=[25, 75])
                ratio_endpoint_median = np.median(ttp_ratio_endpoint)
                ratio_endpoint_array[mut, on, off] = ratio_endpoint_median
                ratio_endpoint_q1_array[mut, on, off] = ratio_endpoint_iqr[0]
                ratio_endpoint_q3_array[mut, on, off] = ratio_endpoint_iqr[1]


    return size_array, ratio_array, ratio_q1_array, ratio_q3_array, size_q1_array, size_q3_array, ratio_endpoint_array, ratio_endpoint_q1_array, ratio_endpoint_q3_array
    # return mean_total, mean_resistant, breakouts_array, ratios_array_mean, var_total, median_total, q1_total, q3_total, mean_total_area, var_total_area, median_total_area, q1_total_area, q3_total_area, ratios_array_median, ratios_array_mean_area, ratios_array_median_area


def main():
    folder = '../../data/sweeps/last_sweep_v0'

    size_array, ratio_array, ratio_q1_array, ratio_q3_array, size_q1_array, size_q3_array, ratio_endpoint_array,  ratio_endpoint_q1_array,  ratio_endpoint_q3_array = create_array(folder)
    np.save(f'../../data/sweep_arrays/last_sweep_v0_size_array.npy', size_array)
    np.save(f'../../data/sweep_arrays/last_sweep_v0_ratio_array.npy', ratio_array)
    np.save(f'../../data/sweep_arrays/last_sweep_v0_size_q1_array.npy', size_q1_array)
    np.save(f'../../data/sweep_arrays/last_sweep_v0_size_q3_array.npy', size_q3_array)
    np.save(f'../../data/sweep_arrays/last_sweep_v0_ratio_q1_array.npy', ratio_q1_array)
    np.save(f'../../data/sweep_arrays/last_sweep_v0_ratio_q3_array.npy', ratio_q3_array)
    np.save(f'../../data/sweep_arrays/last_sweep_v0_ratio_endpoint_array.npy', ratio_endpoint_array)
    np.save(f'../../data/sweep_arrays/last_sweep_v0_ratio_endpoint_q1_array.npy', ratio_endpoint_q1_array)
    np.save(f'../../data/sweep_arrays/last_sweep_v0_ratio_endpoint_q3_array.npy', ratio_endpoint_q3_array)
    # (sweep_array_total_mean, sweep_array_resistant, sweep_array_breakouts, sweep_array_rations, sweep_array_total_var, sweep_array_total_median,
    #  sweep_array_total_q1, sweep_array_total_q3, sweep_array_mean_area, sweep_array_var_area, sweep_array_median_area, sweep_array_q1_area,
    #  sweep_array_q3_area, sweep_array_ratio_median, sweep_array_ratio_mean_area, sweep_array_ratio_median_area) = create_array(folder)
    # np.save(f'../sweep_arrays/sweep_total.npy', sweep_array_total_mean)
    # np.save(f'../sweep_arrays/sweep_resistant.npy', sweep_array_resistant)
    # np.save(f'../sweep_arrays/sweep_breakouts.npy', sweep_array_breakouts)
    # np.save(f'../sweep_arrays/sweep_ratios.npy', sweep_array_rations)
    # np.save(f'../sweep_arrays/sweep_total_var.npy', sweep_array_total_var)
    # np.save(f'../sweep_arrays/sweep_total_median.npy', sweep_array_total_median)
    # np.save(f'../sweep_arrays/sweep_total_q1.npy', sweep_array_total_q1)
    # np.save(f'../sweep_arrays/sweep_total_q3.npy', sweep_array_total_q3)
    # np.save(f'../sweep_arrays/sweep_total_mean_area.npy', sweep_array_mean_area)
    # np.save(f'../sweep_arrays/sweep_total_var_area.npy', sweep_array_var_area)
    # np.save(f'../sweep_arrays/sweep_total_median_area.npy', sweep_array_median_area)
    # np.save(f'../sweep_arrays/sweep_total_q1_area.npy', sweep_array_q1_area)
    # np.save(f'../sweep_arrays/sweep_total_q3_area.npy', sweep_array_q3_area)
    # np.save(f'../sweep_arrays/sweep_total_ratio_median.npy', sweep_array_ratio_median)
    # np.save(f'../sweep_arrays/sweep_total_ratio_mean_area.npy', sweep_array_ratio_mean_area)
    # np.save(f'../sweep_arrays/sweep_total_ratio_median_area.npy', sweep_array_ratio_median_area)
    print("Sweep arrays created and saved successfully.")

if __name__ == "__main__":
    main()
