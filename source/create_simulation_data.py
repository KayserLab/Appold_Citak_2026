import source.run_core as rc


def create_data(replicates, treatments):
    for j in range(len(replicates)):
        if treatments[j] == 'continuous_dose':
            treat_on, treat_off = 10, 0
            pulse, pulse_duration = False, None
        elif treatments[j] == 'no_treatment':
            treat_on, treat_off = 0, 0
            pulse, pulse_duration = False, None
        elif treatments[j] == 'pulse':
            treat_on, treat_off = 0, 0
            pulse, pulse_duration = True, 280
        elif treatments[j] == 'met_4_18':
            treat_on, treat_off = 80, 360
            pulse, pulse_duration = False, None
        elif treatments[j] == 'met_6_5_18':
            treat_on, treat_off = 130, 360
            pulse, pulse_duration = False, None
        elif treatments[j] == 'met_9_18':
            treat_on, treat_off = 180, 360
            pulse, pulse_duration = False, None
        else:
            treat_on, treat_off = None, None
            pulse, pulse_duration = None, None
            print('Treatment not found')
        for i in range(replicates[j]):
            rc.main(treat_on, treat_off, save_dir=f'data/sim_data/{treatments[j]}/{treatments[j]}_{i}', random_seed=i, pulse=pulse, pulse_duration=pulse_duration)


if __name__ == "__main__":
    replicates = [20, 20, 20, 20, 20, 20]
    treatments = ['met_4_18', 'met_6_5_18', 'met_9_18', 'no_treatment', 'continuous_dose', 'pulse']
    create_data(replicates, treatments)
