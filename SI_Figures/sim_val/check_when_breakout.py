from source import core as cr
import numpy as np
import os
import matplotlib.pyplot as plt
import tqdm
from scipy import ndimage
import scipy.optimize as opt


def run_sim_start():
    sim1 = cr.DiffusionModel2D()
    sim1.params['mutations_active'] = False
    sim1.params['image_size'] = 200
    sim1.params['save_in_core'] = True
    time = 750 + sim1.params['start_point']
    sim1.treatment_times = np.zeros(time)
    sim1.params['total_time'] = time
    sim1.params['save_results'] = 'SI_Figures/data/sim_data/breakouts'
    sim1.run_simulation(save_without_asking=True)

def run_sim():
    for j in range(223, 800, 1):
        sim = cr.DiffusionModel2D()
        sim.params['total_time'] = 3683
        sim.params['image_size'] = 200
        sim.params['mutations_active'] = False
        sim.treatment_efficacy = 0.0
        sim.treatment_times = np.zeros(sim.params['total_time'], dtype=bool)
        sim.set_random_seed()
        sim.prev_treatment = True
        sensitive = np.load('../../data/sim_data/breakouts/sensitive.npy')[-30]
        resistant = np.load('../../data/sim_data/breakouts/resistant.npy')[-30]
        nutrients = np.load('../../data/sim_data/breakouts/nutrients.npy')[-30]
        resistant[84, 84] = j / sim.params['mutation_scaling']
        resistant[100, 121] = j / sim.params['mutation_scaling']
        print(sensitive[84, 83] * sim.params['mutation_scaling'], sensitive[100, 122]*sim.params['mutation_scaling'])
        print(sensitive[83, 83] * sim.params['mutation_scaling'], sensitive[84, 82] * sim.params['mutation_scaling'])
        print(sensitive[85, 83] * sim.params['mutation_scaling'], sensitive[84, 84] * sim.params['mutation_scaling'])
        # plt.figure(dpi=300)
        # plt.imshow(np.stack([resistant/np.max(resistant), sensitive/np.max(sensitive), np.zeros_like(sensitive)], axis=-1))
        # plt.imshow(sensitive >= 1 / sim.params['mutation_scaling'], alpha=0.3, cmap='gray')
        # plt.show()
        for i in tqdm.tqdm(range(1, sim.params['total_time'] - 720)):
            nutrients, sensitive, resistant = np.copy(sim.update(i, nutrients, sensitive, resistant))
        fig, ax = plt.subplots(nrows=1, ncols=2, dpi=300)
        ax[0].imshow(np.stack([resistant/np.max(resistant), sensitive/np.max(sensitive), np.zeros_like(sensitive)], axis=-1))
        plt.title(f'{j} mutated cells')
        ax[1].imshow(resistant/np.max(resistant))
        sensitive = np.where(sensitive > 1 / sim.params['mutation_scaling'], True, False)
        eroded = ndimage.binary_erosion(sensitive)
        outline = sensitive ^ eroded
        resistant = np.where(resistant > 1 / sim.params['mutation_scaling'], True, False)
        res_eroded = ndimage.binary_erosion(resistant)
        res_outline = resistant ^ res_eroded
        ax[1].contour(res_outline, colors='blue', linewidths=0.5)
        ax[1].contour(outline, colors='red', linewidths=0.5)
        plt.show()

def plot_and_fit():
    def sigmoid(x, L ,x0, k, b):
        y = L / (1 + np.exp(-k*(x-x0))) + b
        return y
    treat_effics = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    # break_num = [498, 491, 468, 434, 393, 347, 300, 254, 210, 171, 135, 105, 79, 59, 42, 30, 20, 13, 8, 5, 3]
    break_num = [224, 182, 146, 117, 91, 71, 54, 41, 30, 21, 15, 11, 7, 5, 3, 2, 2, 1, 1, 1, 1]

    def sigmoid_test(x):
        y = 5.85696873e+02 / (1 + np.exp(6.00627550 * (x + 7.93420507e-02))) - 1.38418863
        return y

    p0 = [max(break_num), np.median(treat_effics),1,min(break_num)] # this is an mandatory initial guess
    popt = opt.curve_fit(sigmoid, treat_effics, break_num, p0)
    print(f'Fitted parameters: {popt[0]}')
    x_fit = np.linspace(0, 1, 100)
    y_fit = sigmoid(x_fit, *popt[0])

    plt.plot(treat_effics, break_num, 'o-', label='Data', linewidth=2)
    plt.plot(x_fit, y_fit, '--', label='Fitted Sigmoid', linewidth=2)
    plt.plot(x_fit, sigmoid_test(x_fit), ':', label='Old Sigmoid', linewidth=2)
    plt.xlabel('Treatment Efficacy')
    plt.ylabel('Number of Resistant Cells at Breakout')
    plt.title('Breakout Analysis')
    plt.legend()
    plt.savefig('breakout.pdf')
    plt.show()

def main():
    # run_sim_start()
    # run_sim()
    plot_and_fit()


if __name__ == "__main__":
    main()
