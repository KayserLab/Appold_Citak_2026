import source.run_core as rc


def create_data(replicates, treatments):
    for j in range(len(replicates)):
        if treatments[j] == 'met_6_5_18':
            treat_on, treat_off = 130, 360
            pulse, pulse_duration = False, None
        else:
            treat_on, treat_off = None, None
            pulse, pulse_duration = None, None
            print('Treatment not found')
        for i in range(replicates[j]):
            rc.main(treat_on, treat_off, save_dir=f'demo/demo_data/{treatments[j]}/{treatments[j]}_{i}', random_seed=i, pulse=pulse, pulse_duration=pulse_duration)


if __name__ == "__main__":
    replicates = [2]
    treatments = ['met_6_5_18']
    create_data(replicates, treatments)
