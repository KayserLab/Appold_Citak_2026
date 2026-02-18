import numpy as np
from source import core as cr
import matplotlib.pyplot as plt
from scipy.stats import norm, beta
import scipy


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
    y = 5.85696873e+02 / (1 + np.exp(6.00627550 * (x + 7.93420507e-02))) - 1.38418863
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

def get_breakout_statistics(durations, replicates):
    summed_breakouts = []
    for j, duration in enumerate(durations):
        breakouts = []
        for i in range(replicates[j]):
            sim = cr.DiffusionModel2D()
            sim.random_seed = i
            sim.params['save_in_core'] = False
            time = sim.params['total_time']
            start_point = 0 if sim.params['gaussian'] else sim.params['start_point']
            first_start = sim.params['treatment_start'] + start_point

            sim.treatment_times = np.zeros(time)
            sim.treatment_times[first_start:(first_start + duration)] = True

            _, sen, res, _, treat_effic = sim.run_simulation(save_without_asking=False, stop_at_fullstop=False)

            sensitive, resistant = np.where(sen > 1 / sim.params['mutation_scaling'], True, False), np.where(res > 1 / sim.params['mutation_scaling'], True, False)

            breakout = has_breakout(sensitive, resistant, treat_effic[-1], sim.params['mutation_scaling'])
            if breakout:
                breakouts.append(1)
        summed_breakouts.append(np.sum(breakouts))
    np.save('breakout_statistics.npy', np.array(summed_breakouts))

def calc_errorbars(k_arr, n_arr, confidence_level, method="wilson"):
    lower_bounds = []
    upper_bounds = []
    probability = []
    if method == "clopper-pearson":
        method = "exact"
    for i in range(len(k_arr)):
        if n_arr[i] == 0:
            raise ValueError("Number of trials must be greater than zero.")
        if k_arr[i] < 0 or k_arr[i] > n_arr[i]:
            raise ValueError("Number of successes must be between 0 and number of trials.")
        distribution = scipy.stats.binomtest(k_arr[i], n_arr[i])
        ci = distribution.proportion_ci(method=method, confidence_level=confidence_level)
        lower_bounds.append(distribution.statistic - ci[0])
        upper_bounds.append(ci[1] - k_arr[i]/n_arr[i])
        probability.append(distribution.statistic)
    return probability, (np.array(lower_bounds), np.array(upper_bounds))


def plot_breakout_probability(durations, replicates):
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

    # Data taken from pulse duration experiment
    breakouts_exp =             np.array([  4,  4,   5,    8,   10,   12,   13,   12,   10,  10])  # np.array([  4,  4,   5,    8,   10,   13,   13,   12,   11,  10])
    replicates_exp =            np.array([ 10, 12,  10,   15,   11,   14,   13,   12,   11,  10])
    treatment_duration_exp =    np.array([ 40, 80, 120,  160,  200,  240,  280,  320,  360, 400])

    breakouts_sim = np.load('breakout_statistics.npy') # np.array([np.int64(21), np.int64(29), np.int64(25), np.int64(27), np.int64(17), np.int64(25), np.int64(32), np.int64(49), np.int64(48), np.int64(50)])
    replicates_sim = np.array([75, 75, 75, 75, 75, 75, 75, 75, 75, 75]) # np.array([replicates for _ in durations])
    treatment_duration_sim = np.array(durations)

    p_hat_exp, yerr_exp = calc_errorbars(breakouts_exp, replicates_exp, confidence_level=0.95, method="clopper-pearson")
    p_hat_sim, yerr_sim = calc_errorbars(breakouts_sim, replicates_sim, confidence_level=0.95, method="clopper-pearson")

    plt.figure(figsize=(2.5, 1.8), dpi=300)
    plt.errorbar(treatment_duration_exp/20, p_hat_exp, label='Experiment', marker='o', linestyle='None', color='blue', yerr=yerr_exp, capsize=3.5, capthick=0.5, markersize=3.5)
    plt.errorbar(treatment_duration_sim/20, p_hat_sim, label='Simulation', marker='o', linestyle='None', color='red', yerr=yerr_sim, capsize=3.5, capthick=0.5, markersize=3.5)
    plt.xlabel('Treatment duration (h)')
    plt.ylabel('Breakout probability')
    plt.ylim(-0.05, 1.05)
    plt.xlim(1, 21)
    plt.xticks(np.arange(0, 22, 2))
    # plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(frameon=False, loc='lower right')
    plt.savefig('breakouts_per_pulse.pdf', bbox_inches='tight', transparent=True)
    plt.show()

def main(durations):
    replicates = np.array([10, 12, 10, 15, 11, 13, 13, 12, 10, 10])  # experiment replicate number
    replicates_sim = np.array([75, 75, 75, 75, 75, 75, 75, 75, 75, 75])
    get_breakout_statistics(durations, replicates_sim)
    plot_breakout_probability(durations, replicates)

if __name__ == "__main__":
    duration_array = [40, 80, 120, 160, 200, 240, 280, 320, 360, 400]
    main(duration_array)
