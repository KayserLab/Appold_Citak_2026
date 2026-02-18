import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 7,
                     'pdf.fonttype': 42,
                     'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial'],
                     'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Arial',
                     'mathtext.it': 'Arial:italic',
                     'mathtext.bf': 'Arial:bold'})
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6

def sigmoid(x, x0, k):
    """Sigmoid function for smooth transitions."""
    return 1 / (1 + np.exp(-k * (x - x0)))

x = np.linspace(-4, 4, 5000)

plt.figure(figsize = (4, 2), dpi = 300)
plt.plot(x, sigmoid(x, 0, 1))
plt.plot(x, 0.25*x+0.5, 'C0--')
plt.ylim(0.0, 1.0)
plt.axvline(x=2, color='gray', linestyle='--')
plt.axvline(x=-2, color='gray', linestyle='--')
plt.axhline(y=1/(1+np.exp(2)), color='#e34234', alpha=0.8, label=r'$\mathrm{N}_{\mathrm{low}}$')
plt.axhline(y=1/(1+np.exp(-2)), color='#e34234', alpha=0.8, label=r'$\mathrm{N}_{\mathrm{high}}$')
plt.plot(0, 0.5, 'ro', markersize=4)
plt.legend(frameon=False)
plt.savefig('nut_thres_explain.pdf', bbox_inches='tight')
plt.show()