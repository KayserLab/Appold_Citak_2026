import numpy as np
import tqdm
from source import core as cr
import os
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 7,
                     'pdf.fonttype': 42,
                     'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial'],
                     'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Arial',
                     'mathtext.it': 'Arial:italic',
                     'mathtext.bf': 'Arial:bold',
                     })

def get_nutrient_data(start_point):
    position_of_mutant_in_sim_pixel = [2,4,6,8] # [3, 7, 11, 15]
    frame_of_growth = [47.6780134536019, 77.7497496415015, 107.8214858294011, 137.8932220173007] # [62.7138815475517,  122.85735392335089, 183.00082629915008, 243.1442986749493]
    sim_step_of_growth = (np.array(frame_of_growth)+(start_point/10)) * 10
    return position_of_mutant_in_sim_pixel, sim_step_of_growth

sim1 = cr.DiffusionModel2D()
sim1.params['mutations_active'] = False
sim1.params['image_size'] = 200
sim1.params['save_in_core'] = True
start_point = sim1.params['start_point']
first_start = 360 + start_point
time = 750 + start_point
sim1.treatment_times = np.zeros(time)
sim1.treatment_times[first_start:] = True
sim1.params['total_time'] = time
sim1.params['save_results'] = os.path.join(cr.find_project_root(os.getcwd(), 'requirements.txt'), '../../source/fit/fit_data/sim_data/nutrient_diffusion')
sim1.run_simulation(save_without_asking=True)

sim = cr.DiffusionModel2D()
sim.params['treatment_efficacy'] = np.load(os.path.join(cr.find_project_root(os.getcwd(), 'requirements.txt'), '../../source/fit/fit_data/sim_data/nutrient_diffusion/treatment_efficacy.npy'))[360 + start_point]
sim.params['image_size'] = 200
sim.params['mutations_active'] = False
sim.treatment_times = np.ones(3001, dtype=bool)
sim.set_random_seed()
sensitive = np.load(os.path.join(cr.find_project_root(os.getcwd(), 'requirements.txt'), '../../source/fit/fit_data/sim_data/nutrient_diffusion/sensitive.npy'))
nutrients = np.load(os.path.join(cr.find_project_root(os.getcwd(), 'requirements.txt'), '../../source/fit/fit_data/sim_data/nutrient_diffusion/nutrients.npy'))

sen_start = sensitive[360 + start_point]
nut_start = nutrients[360 + start_point]
pos, exp_times = get_nutrient_data(start_point)
index = np.where(sen_start[100, :] >= 1 / sim.params['mutation_scaling'])[0]

res_start = np.zeros_like(sen_start)
res_start[100, index[0] + int(pos[0])] = 1 / sim.params['mutation_scaling']
res_start[100, index[-1] - int(pos[1])] = 1 / sim.params['mutation_scaling']
res_start[index[0] + int(pos[2]), 100] = 1 / sim.params['mutation_scaling']
res_start[index[-1] - int(pos[3]), 100] = 1 / sim.params['mutation_scaling']
sen_start[100, index[0] + int(pos[0])] -= 1 / sim.params['mutation_scaling']
sen_start[100, index[-1] - int(pos[1])] -= 1 / sim.params['mutation_scaling']
sen_start[index[0] + int(pos[2]), 100] -= 1 / sim.params['mutation_scaling']
sen_start[index[-1] - int(pos[3]), 100] -= 1 / sim.params['mutation_scaling']
pos_to_check_1 = [100, index[0] + int(pos[0]) - 1]
pos_to_check_2 = [100, index[-1] - int(pos[1]) + 1]
pos_to_check_3 = [index[0] + int(pos[2]) - 1, 100]
pos_to_check_4 = [index[-1] - int(pos[3]) + 1, 100]
check_1 = False
check_2 = False
check_3 = False
check_4 = False

plt.imshow(np.stack([res_start/np.max(res_start), sen_start/np.max(sen_start), np.zeros_like(sen_start)], axis=-1), interpolation='none')
plt.show()

sim_times = []
for i in tqdm.tqdm(range(1, 2700)):
    nut_start, sen_start, res_start = np.copy(sim.update(i, nut_start, sen_start, res_start))

    # Check if the resistant cells are over the threshold in the positions to check
    if np.sum(res_start[pos_to_check_1[0], pos_to_check_1[1]]) > 1 / sim.params['mutation_scaling'] and not check_1:
        check_1 = True
        sim_times.append(i + 360 + start_point)

    if np.sum(res_start[pos_to_check_2[0], pos_to_check_2[1]]) > 1 / sim.params['mutation_scaling'] and not check_2:
        check_2 = True
        sim_times.append(i + 360 + start_point)

    if np.sum(res_start[pos_to_check_3[0], pos_to_check_3[1]]) > 1 / sim.params['mutation_scaling'] and not check_3:
        check_3 = True
        sim_times.append(i + 360 + start_point)

    if np.sum(res_start[pos_to_check_4[0], pos_to_check_4[1]]) > 1 / sim.params['mutation_scaling'] and not check_4:
        check_4 = True
        sim_times.append(i + 360 + start_point)

print(sim_times)

plt.rcParams.update({'font.size': 7,
                     'pdf.fonttype': 42,
                     'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial'],
                     'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Arial',
                     'mathtext.it': 'Arial:italic',
                     'mathtext.bf': 'Arial:bold',
                     })

sim_times = np.array(sim_times)
# popt_sim = np.polyfit(pos[:len(sim_times)], sim_times, 1, cov=False)
# popt_data = np.polyfit(pos, exp_times, 1, cov=False)
# print(popt_sim)
plt.figure(figsize=(1.5, 1.8), dpi=300)
plt.plot(np.array(pos)*(1376/100) * 8.648 /1e3, exp_times/20, 'bo', label='Experiment', markersize=5)
plt.plot(np.array(pos[:len(sim_times)])*(1376/100) * 8.648 /1e3, sim_times/20, 'ro', label='Simulation', markersize=5)
# plt.plot(pos, popt_sim[0] * np.array(pos) + popt_sim[1], label='Fitted Line')

plt.ylabel('Time (h)')
plt.xlabel('Position (mm)')
plt.xlim(0,2)
# plt.plot(pos, popt_data[0] * np.array(pos) + popt_data[1], label='Fitted Line')
plt.legend()
plt.savefig('growth_delay_comparison_test.pdf', bbox_inches='tight', transparent=True)
plt.show()
