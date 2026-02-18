import matplotlib.pyplot as plt
import numpy as np
import source.run_core as rc

def main(treat_on, treat_off):
    rc.main(treat_on, treat_off, save_dir=f'data/sim_data/treat_effic_pulse_400', pulse=True, pulse_duration=400)

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

    treat_effic = np.load(f'../../data/sim_data/treat_effic_pulse_400/treatment_efficacy.npy')[351:3483]
    treat_effic = -treat_effic + 1
    x = np.array(range(len(treat_effic)))

    fig, ax = plt.subplots(figsize=(6.29921*(1.3/5), 6.29921*(2.3/5)*(3/5)), dpi=300)
    ax.plot(x/20, treat_effic, linewidth=1.5, color='black')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel(r'Effective growth rate, $\xi$')
    ax.set_ylim(-0.01, 1.12)
    ax.set_xlim(0, 100)
    ax.set_yticks([0,1])
    ax.axvspan(18, 38, color='#bfbfbf', alpha=1, label='Treatment', lw=0)
    ax.text(25, 1.02, r'$\tau_{on}$', fontsize=7, color='gray', fontdict=None)
    plt.tight_layout()
    plt.savefig(r'treatment_efficacy.pdf', dpi=300, transparent=True)
    plt.show()

if __name__ == '__main__':
    treat_on = 0
    treat_off = 0
    main(treat_on, treat_off)
