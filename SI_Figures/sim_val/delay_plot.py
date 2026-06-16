import numpy as np
import matplotlib.pyplot as plt
import pathlib as pl
import sys


PROJECT_ROOT = pl.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.fit.parameter_fitting import delay_parameter_estimation as dpe

plt.rcParams.update({'font.size': 7,
                     'pdf.fonttype': 42,
                     'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial'],
                     'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Arial',
                     'mathtext.it': 'Arial:italic',
                     'mathtext.bf': 'Arial:bold',
                     })

def format_estimate(name, value, stderr):
    print(f'{name}: {value:.2f} +/- {stderr:.2f}')

def format_estimate_with_legacy(name, value, stderr, legacy_value):
    print(f'{name}: {value:.2f} +/- {stderr:.2f} (legacy rounded: {legacy_value:.0f})')

analysis = dpe.analyze_delay_parameters()
delay_estimates = analysis["parameter_estimates"]
pulse_derivative = analysis["data"]["pulse_derivative"]
met_7_derivative = analysis["data"]["met_7_derivative"]
nt_dev = analysis["data"]["nt_derivative"]
treatment_times = list(analysis["treatment_times"])
popt4 = analysis["fits"]["pulse_constant"]
popt0 = analysis["fits"]["pulse_line"]
popt1 = analysis["fits"]["met_7_constant"]
popt2 = analysis["fits"]["release_line"]
popt3 = analysis["fits"]["control_line"]
xfit = analysis["four_segment_xfit"]
yfit = analysis["four_segment_yfit"]

format_estimate_with_legacy(
    'Treat On Delay',
    delay_estimates["treatment_delay"]["estimate"],
    delay_estimates["treatment_delay"]["stderr"],
    delay_estimates["treatment_delay"]["legacy_rounded"],
)
format_estimate_with_legacy(
    'Treat Off Delay',
    delay_estimates["release_delay"]["estimate"],
    delay_estimates["release_delay"]["stderr"],
    delay_estimates["release_delay"]["legacy_rounded"],
)
format_estimate(
    'Overshoot',
    delay_estimates["overshoot_steps"]["estimate"],
    delay_estimates["overshoot_steps"]["stderr"],
)
format_estimate(
    'Lag Phase',
    delay_estimates["lag_steps"]["estimate"],
    delay_estimates["lag_steps"]["stderr"],
)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7.6, 2.4), dpi=300)
ax[0].plot(pulse_derivative, color='C0', marker='.', lw=2, markersize=8)
ax[0].plot(nt_dev, color='C7', marker='.', lw=2, markersize=8)
# ax[0].plot(np.arange(len(pulse_derivative))+39, -0.0021930446024563703 * np.arange(len(pulse_derivative)) + 0.0357882352941176, color='C1', label='Linear fit', linewidth=2)
ax[0].plot(np.arange(len(pulse_derivative)), np.polyval(popt0["coef"], np.arange(len(pulse_derivative))), color='C1', label='Linear fit', linewidth=2)
ax[0].plot(np.arange(len(pulse_derivative)), np.polyval(popt2["coef"], np.arange(len(pulse_derivative))), color='C2', linewidth=2)
ax[0].plot(np.arange(len(pulse_derivative)), np.polyval(popt3["coef"], np.arange(len(pulse_derivative))), color='C3', linewidth=2)
ax[0].plot(np.arange(len(pulse_derivative)), np.polyval(popt4["coef"], np.arange(len(pulse_derivative))), color='C4', linewidth=2)
# ax[0].plot(x_treat, [1 for _ in range(len(x_treat))] * popt_treat[0], color='orange', label='Max speed', linewidth=2)
# ax[0].vlines(x=crossing_point_treat, ymin=0, ymax=5, color='green', linestyle='--', label='Half speed', linewidth=2)
ax[0].set_xlim(treatment_times[0] - 10, treatment_times[1] + 80)
# ax[0].axvspan(treatment_times[0], treatment_times[0] + (crossing_point_treat - treatment_times[0]) * 2, color='gray', alpha=0.3, label='Delay Period')
ax[1].plot(met_7_derivative, marker='.', color='C0', lw=2, markersize=8)
ax[1].plot(np.arange(len(met_7_derivative)), np.polyval(popt1["coef"], np.arange(len(met_7_derivative))), color='C5', label='Constant fit', linewidth=2)
ax[1].plot(np.arange(len(met_7_derivative)), np.polyval(popt0["coef"], np.arange(len(met_7_derivative))), color='C1', linewidth=2)
ax[1].plot(np.arange(len(met_7_derivative)) - 33, np.polyval(popt2["coef"], np.arange(len(met_7_derivative))), color='C2', linewidth=2)
# ax[1].plot(x_rel, [1 for _ in range(len(x_rel))] * popt_rel[0], color='orange', label='Max speed', linewidth=2)
# ax[1].vlines(x=crossing_point, ymin=0, ymax=5, color='green', linestyle='--', label='Half speed', linewidth=2)
# ax[1].axvspan(treatment_times[1], treatment_times[1] + (crossing_point - treatment_times[1]) * 2, color='gray', alpha=0.3, label='Delay Period')
ax[1].set_xlim(treatment_times[0] - 10, treatment_times[1] + 25) #treatment_times[0] - 10, 102)
ax[1].plot(xfit, yfit, '-', lw=2, color='C6', label='4-segment fit')

ax[0].axvspan(treatment_times[0], treatment_times[0] + 28, color='#bfbfbf', lw=0)
ax[1].axvspan(treatment_times[0], treatment_times[0] + 14, color='#bfbfbf', lw=0)

ax[0].set_title('Pulse Experiment', fontsize=7)
ax[1].set_title('7/18 Treatment', fontsize=7)
# ax[0].legend()
ax[0].set_xlabel('Time (h)')
ax[1].set_xlabel('Time (h)')
ax[0].set_ylabel('Radial velocity (mm/h)')
ax[0].set_ylim(-0, 0.08)
ax[1].set_ylim(0, 0.08)
# ax[1].set_yticklabels([])

# plt.tight_layout()
plt.savefig('SI_Figures/plots/delays.pdf', bbox_inches='tight', transparent=True)
plt.show()
