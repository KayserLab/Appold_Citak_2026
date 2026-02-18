import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

plt.rcParams.update({'font.size': 7,
                     'pdf.fonttype': 42,
                     'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial'],
                     'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Arial',
                     'mathtext.it': 'Arial:italic',
                     'mathtext.bf': 'Arial:bold',
                     })

nutrients = np.load('../../data/sim_data/no_treatment/no_treatment_0/nutrients.npy')
resistant = np.load('../../data/sim_data/no_treatment/no_treatment_0/resistant.npy')
sensitive = np.load('../../data/sim_data/no_treatment/no_treatment_0/sensitive.npy')
params = torch.load('../../data/sim_data/no_treatment/no_treatment_0/params.pth', weights_only=False)
pulse_csv = pd.read_csv('../../source/fit_data/no_treatment_csv/For_manuscript/colony_data_P1__with_clonearea.csv')
radius = pulse_csv['colony_radius'][:160]
print(pulse_csv['colony_area'][0])

# fig, ax = plt.subplots(nrows=1, ncols=2)
# # nutrients = np.where(nutrients <= 0.005, 0, nutrients)
# sensitive_adapted = np.where(sensitive <= 0.05, 0, sensitive)
# sen_diff = sensitive[1200, 100, :] - sensitive[1199, 100, :]
# sen_diff_max = np.max(sen_diff)
# # get fwhm of sen diff
# half_max = sen_diff_max / 2
# index = np.where(sen_diff >= half_max)
# print(index)
# ax[0].plot(nutrients[1200, 100, :] - nutrients[1199, 100, :], label='Nutrients')
# ax[1].plot(sensitive_adapted[1200, 100, :] - sensitive_adapted[1199, 100, :], label='Sensitive')
# ax[1].vlines(x=16, ymin=0, ymax=0.00025, color='r', linestyle='--', label='FWHM')
# ax[1].vlines(x=22, ymin=0, ymax=0.00025, color='r', linestyle='--')
# plt.show()
#
#
# x = np.linspace(-100 // 2, 100 // 2, 100)
# y = np.linspace(-100 // 2, 100 // 2, 100)
# grid_x, grid_y = np.meshgrid(x, y)
# width = 2.2
# sigma = width / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to standard deviation
# initial_gaussian = np.exp(-(grid_x ** 2 + grid_y ** 2) / (2 * sigma ** 2))
# initial_gaussian = gaussian_filter(initial_gaussian, sigma=sigma)
# col_area = np.where(initial_gaussian <= 0.05, 0, 1)
# fig, ax = plt.subplots(nrows=1, ncols=2)
# ax[0].imshow(initial_gaussian, cmap='hot')
# ax[1].imshow(col_area, cmap='hot')
# plt.show()
# print(np.count_nonzero(col_area)*(1376/100)**2)

pos = []
for current_dt in range(nutrients.shape[0]):
    # line_nut = nutrients[i, 150, :]
    line_sen = sensitive[current_dt, 100, :]
    max_sen = np.max(line_sen)
    # get point closest to half max
    # half_max = max_sen / 1000
    # index = np.where(line_sen >= half_max/(current_dt**8 + 1))
    index = np.where(line_sen >= 1/params['mutation_scaling'])
    if len(index[0]) > 0:
        # print(index[0])
        # print(line_sen[index[0]])
        pos.append(index[0][-1])
    else:
        # print('No index found')
        # pos.append(0)
        continue

def linear_func(x, a, b):
    return a * x + b

# print(np.unique(pos, return_counts=True))
x = np.linspace(0, len(radius)/2, len(radius))
# popt, pcov = opt.curve_fit(linear_func, np.arange(len(pos)), pos, p0=[0.006, 163])
# print(pos[480]-100)

len_rad_arr = len(np.array(pos[params['start_point']:1999]))
x_sim = np.linspace(0, len_rad_arr/20, len_rad_arr)

plt.figure(figsize=(2.5,1.8), dpi=300)
plt.ylim(0,6)
plt.plot(x_sim, (np.array(pos[params['start_point']:1999]) - 100)*(1376/100) * 8.648 /1e3, label='Simulation', color='red')
plt.plot(x[:-7], radius[:-7] * 8.648 /1e3, label='Experiment', color='blue')
plt.ylabel('Radius (mm)')
plt.xlabel('Time (h)')
# plt.plot(np.arange(len(pos)), popt[0] * np.arange(len(pos)) + popt[1], label='Fitted Line')
plt.legend(fontsize=7)
# plt.tight_layout()
plt.savefig('fitted_front_velocity.pdf', bbox_inches='tight', transparent=True)
plt.show()

# print((pos[-1]-pos[0])/3289)
# print(pos[0], pos[-1])
# print(popt[0])
    # plt.figure()
    # plt.plot(line_nut)
    # plt.plot(line_res)
    # plt.plot(line_sen)
    # plt.title('Nutrient Concentration')
    # plt.show()
