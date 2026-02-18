import numpy as np
import matplotlib.pyplot as plt
import yaml
import os


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

def get_params():
    path = os.path.join(find_project_root(os.getcwd(), 'requirements.txt'), 'params.yaml')
    with open(path, 'r') as file:
        params = yaml.safe_load(file) # ['simulation_params']
    return params

def find_project_root(current_dir, marker_file):
    current_dir = os.path.abspath(current_dir)
    while current_dir != os.path.dirname(current_dir):  # Stop at the root of the file system
        if marker_file in os.listdir(current_dir):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return None

def treat_func(x):
    return -np.exp(5*(-x))+1

def calc_treatment_efficacy(treat_on, treat_off, params):
    first_start = params['treatment_start'] # + params['start_point']

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

    delta_t = params['delta_t']
    treatment_delay = params['treatment_delay']
    release_delay = params['release_delay']

    # number of steps in the lag phase after treatment ends
    lag_on_steps = params.get('lag_on_steps', 220)

    treatment_efficacy = 0.0
    results = []

    # state variables
    prev_treatment = False
    extra_steps_remaining = 0
    lag_steps_remaining = 0

    for i in range(1, params['total_time']):
        current_treatment = bool(treatment_times[i])

        if prev_treatment and not current_treatment:
            extra_steps_remaining = 30
            print(extra_steps_remaining)
            lag_steps_remaining = lag_on_steps

        elif current_treatment:
            # Treatment ON overrides any extra/lag
            extra_steps_remaining = 0
            lag_steps_remaining = 0

            treatment_efficacy += delta_t / treatment_delay

        elif extra_steps_remaining > 0:
            # Residual treatment phase: still behaving as if treatment is ON
            treatment_efficacy += delta_t / treatment_delay
            extra_steps_remaining -= 1
            lag_steps_remaining -= 1

        elif lag_steps_remaining > 0:
            # Lag phase after treatment: efficacy held constant
            lag_steps_remaining -= 1


        else:
            # True OFF / release phase: efficacy decays linearly
            treatment_efficacy -= delta_t / release_delay

        # Clamp to [0, 1]
        if treatment_efficacy > 1.0:
            treatment_efficacy = 1.0
        elif treatment_efficacy < 0.0:
            treatment_efficacy = 0.0

        results.append(treatment_efficacy)
        prev_treatment = current_treatment

    # as before: return 1 - efficacy as the "growth factor" or similar
    return 1 - np.array(results), treatment_starts, treatment_ends

def plot_treat_effic(treat_on_duration, treat_off_duration):
    params = get_params()
    treat_effic_point_test, treat_starts_test, treat_ends_test = calc_treatment_efficacy(int(treat_on_duration*20), int(treat_off_duration*20), params)
    plt.figure(figsize=(2, 1.5), dpi = 300)
    print(np.max(treat_effic_point_test), np.min(treat_effic_point_test))
    x = np.linspace(0, len(treat_effic_point_test)/20, len(treat_effic_point_test))
    plt.plot(x, treat_effic_point_test, color='black')
    for i in range(len(treat_starts_test)):
        plt.axvspan(treat_starts_test[i]/20, treat_ends_test[i]/20, color='#bfbfbf', alpha=1, lw=0, zorder=0)
    plt.xlim(0, len(treat_effic_point_test)/20)
    plt.xlabel('Time (h)')
    plt.ylabel(r'Effective growth rate, $\xi$')
    plt.title(f'{treat_on_duration}/{treat_off_duration}', fontsize=7)
    # plt.axvline(18, color='red', linestyle='--', linewidth=0.8)
    # plt.axvline(30, color='gray', linestyle='--', linewidth=0.8)
    # plt.axvline(32, color='red', linestyle='--', linewidth=0.8)
    # plt.axvline(48, color='gray', linestyle='--', linewidth=0.8)
    # plt.axvline(60, color='gray', linestyle='--', linewidth=0.8)
    plt.axhline(1, color='gray', linestyle='--', linewidth=0.8)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    # plt.xlim(350, 370)
    # plt.axvline(360, color='red', linestyle='--', linewidth=0.8)
    # plt.axvline(590, color='red', linestyle='--', linewidth=0.8)
    plt.savefig(f'treat_effic_{treat_on_duration}_{treat_off_duration}.pdf', transparent=True)
    plt.show()


treat_on_duration = 4 # in hours
treat_off_duration = 18  # in hours
plot_treat_effic(treat_on_duration, treat_off_duration)
