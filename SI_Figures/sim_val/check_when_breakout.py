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
    sim1.params['save_in_core'] = True
    time = 750 + sim1.params['start_point']
    sim1.treatment_times = np.zeros(time)
    sim1.params['total_time'] = time
    sim1.params['save_results'] = 'data/sim_data/breakouts'
    sim1.run_simulation(save_without_asking=True)

def run_sim():
    for j in range(238, 800, 1):
        sim = cr.DiffusionModel2D()
        sim.params['mutations_active'] = False
        sim.treatment_efficacy = 0.05
        sim.treatment_times = np.zeros(sim.params['total_time'], dtype=bool)
        sim.set_random_seed()
        sim.prev_treatment = True
        sensitive = np.load('data/sim_data/breakouts/sensitive.npy')[-30]
        resistant = np.load('data/sim_data/breakouts/resistant.npy')[-30]
        nutrients = np.load('data/sim_data/breakouts/nutrients.npy')[-30]
        resistant[83, 83] = j / sim.params['mutation_scaling']
        resistant[100, 123] = j / sim.params['mutation_scaling']
        # print(sensitive[83, 82] * sim.params['mutation_scaling'], sensitive[100, 122]*sim.params['mutation_scaling'])
        # print(sensitive[82, 82] * sim.params['mutation_scaling'], sensitive[84, 82] * sim.params['mutation_scaling'])
        # print(sensitive[84, 82] * sim.params['mutation_scaling'], sensitive[83, 83] * sim.params['mutation_scaling'])
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
    break_num_diagonal = [156, 131, 109, 89, 72, 58, 46, 36, 28, 21, 16, 12, 9, 6, 4, 3, 2, 2, 1, 1, 1]  # diagonal
    break_num_horizontal = [287, 240, 201, 167, 137, 111, 90, 71, 56, 43, 33, 25, 18, 13, 9, 7, 5, 3, 3, 2, 2]  # horizontal

    def sigmoid_diagonal(x):
        y = 3.87401421e+02 / (1 + np.exp(5.33988821 * (x + 7.21574279e-02))) - 1.25396977
        return y

    def sigmoid_horizontal(x):
        y = 7.78241196e+02 / (1 + np.exp(4.94935493 * (x + 1.06728521e-01))) - 3.38920380
        return y

    p0 = [max(break_num_diagonal), np.median(treat_effics),1,min(break_num_diagonal)] # this is an mandatory initial guess
    popt = opt.curve_fit(sigmoid, treat_effics, break_num_diagonal, p0)
    print(f'Fitted parameters: {popt[0]}')
    p0_1 = [max(break_num_horizontal), np.median(treat_effics),1,min(break_num_horizontal)] # this is an mandatory initial guess
    popt_1 = opt.curve_fit(sigmoid, treat_effics, break_num_horizontal, p0_1)
    print(f'Fitted parameters: {popt_1[0]}')
    x_fit = np.linspace(0, 1, 100)
    y_fit_1 = sigmoid(x_fit, *popt[0])
    y_fit_2 = sigmoid(x_fit, *popt_1[0])

    plt.plot(treat_effics, break_num_diagonal, 'o-', label='Data', linewidth=2)
    plt.plot(treat_effics, break_num_horizontal, 's-', label='Data', linewidth=2)
    plt.plot(x_fit, y_fit_1, '--', label='Fitted Sigmoid (diagonal)', linewidth=2)
    plt.plot(x_fit, y_fit_2, ':', label='Fitted Sigmoid (horizontal)', linewidth=2)
    plt.plot(x_fit, sigmoid_diagonal(x_fit), ':', label='Old Sigmoid (diagonal)', linewidth=2)
    plt.plot(x_fit, sigmoid_horizontal(x_fit), '--', label='Old Sigmoid (horizontal)', linewidth=2)
    plt.xlabel('Treatment Efficacy')
    plt.ylabel('Number of Resistant Cells at Breakout')
    plt.title('Breakout Analysis')
    plt.legend()
    plt.savefig('SI_Figures/plots/breakout.pdf')
    plt.show()

def main():
    # run_sim_start()
    # run_sim()
    plot_and_fit()


if __name__ == "__main__":
    main()
