import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.optimize as opt
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

def get_data(folder_path, length=None):
    path_list = os.listdir(
        fr'../../data/exp_data/{folder_path}/Sus_Kymos')
    path_list_filtered = [i for i in path_list if i.endswith('.csv')]

    rad_list = []
    for path in path_list_filtered:
        data = pd.read_csv(
            fr'../../data/exp_data/{folder_path}\Sus_Kymos\{path}')
        rad_list.append(rolling_median((data['max_distance_mm'].values[:length]), window_size=9))

    rad_mean = np.mean(rad_list, axis=0)
    rad_dev = np.gradient(rad_mean, 0.5)
    return rad_mean, rad_dev

def rolling_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def median_over_radius_derivative(exp_data):
    dt = 0.5
    temp = []
    for exp in exp_data:
        radius_derivative = np.gradient(exp['colony_radius'], dt)
        temp.append(radius_derivative[:295])
    return np.median(np.array(temp), axis=0) #rolling_median(np.median(np.array(temp), axis=0), window_size=9)

def rolling_median(data, window_size):
    half_window = window_size // 2
    padded_data = np.pad(data, (half_window, half_window), mode='edge')
    smoothed_data = np.array([np.median(padded_data[i:i + window_size]) for i in range(len(data))])
    return smoothed_data

def get_data_nt(path):
    exp_data_list = [i for i in os.listdir(path) if i.startswith('colony') and i.endswith('clonearea.csv')]
    exp_data = []
    for i in exp_data_list:
        exp_data.append(pd.read_csv(f'{path}/{i}'))
    return exp_data

def median_over_radius_derivative_nt(exp_data):
    dt = 0.5
    temp = []
    radius = []
    for exp in exp_data:
        radius_derivative = np.gradient(exp['colony_radius']*8.648/1e3, dt)
        radius.append(exp['colony_radius'][:295]*8.648/1e3)
        temp.append(radius_derivative[:295])
    return rolling_median(np.median(np.array(temp), axis=0), window_size=9), rolling_median(np.median(np.array(radius), axis=0), window_size=9)


nt = get_data_nt(r'../../data/exp_data/no_treatment_csv/For_Manuscript')
nt_dev, nt_radius = median_over_radius_derivative_nt(nt)

pulse_mean, pulse_derivative = get_data('20241210_pulse', length=259)
# under_mean, under_derivative = get_data('20250909_metr_undertreat', length=302)
under_mean, under_derivative = get_data('20251007_metr_7_18', length=302)
treatment_times = [36, 64]  # images every 30 mins (image number)


popt4 = np.polyfit(np.arange(len(pulse_derivative[70:85]))+70, pulse_derivative[70:85], 0)

popt0 = np.polyfit(np.arange(len(pulse_derivative[36:65]))+36, pulse_derivative[36:65], 1)
print(f'Treat On Delay: {np.round((popt4[0]-popt0[1])/popt0[0]-36) * 10}')

popt1 = np.polyfit(np.arange(len(under_derivative[60:75]))+60, under_derivative[60:75], 0)

popt2 = np.polyfit(np.arange(len(pulse_derivative[103:115]))+103, pulse_derivative[103:115], 1)
popt3 = np.polyfit(np.arange(len(nt_dev[90:125]))+90, nt_dev[90:125], 1)
print(f'Treat Off Delay {np.round((popt3[1]-popt2[1])/(popt2[0]-popt3[0]) - (popt4[0] - popt2[1])/popt2[0]) * 10}')
# print(f'Lag Phase Duration: {np.round((((popt1[0]-popt2[1])/popt2[0]) - 41 - 50)*10)}')

def fit_linear_ls(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    m, q = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, q

def predict_4seg(x, a, b, c, y0, m1, q1, m3):
    x = np.asarray(x, dtype=float)
    yhat = np.empty_like(x, dtype=float)

    y2 = m1 * b + q1

    m0 = x < a
    m1mask = (x >= a) & (x < b)
    m2mask = (x >= b) & (x < c)
    m3mask = x >= c

    yhat[m0] = y0
    yhat[m1mask] = m1 * x[m1mask] + q1
    yhat[m2mask] = y2
    yhat[m3mask] = y2 + m3 * (x[m3mask] - c)
    return yhat

def grid_fit_4seg(x, y, m3, linear_fit_end=None, min_len=(2, 3, 3, 2)):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xs = np.unique(x)

    min0, min1, min2, min3len = min_len
    best = {"sse": np.inf}

    # candidate indices for a,b,c
    for ia in range(min0, len(xs) - (min1 + min2 + min3len)):
        a = xs[ia]

        # baseline y0 from x<a (mean is robust enough; could use median)
        y0 = float(np.mean(y[x < a]))

        for ib in range(ia + min1, len(xs) - (min2 + min3len)):
            b = xs[ib]

            for ic in range(ib + min2, len(xs) - min3len + 1):
                c = xs[ic] if ic < len(xs) else xs[-1] + 1  # safety

                # choose data range to fit the linear segment
                lin_mask = (x >= a) & (x < b)
                if linear_fit_end is not None:
                    lin_mask = (x >= a) & (x < min(b, linear_fit_end))

                # must have enough points to fit
                if np.sum(lin_mask) < 2:
                    continue

                m1, q1 = fit_linear_ls(x[lin_mask], y[lin_mask])

                yhat = predict_4seg(x, a, b, c, y0, m1, q1, m3)
                sse = float(np.sum((y - yhat) ** 2))

                if sse < best["sse"]:
                    best = {
                        "a": float(a), "b": float(b), "c": float(c),
                        "y0": y0, "m1": float(m1), "q1": float(q1),
                        "sse": sse
                    }
    return best

xdata = np.arange(36, np.argmax(under_derivative[60:110])+60, dtype=float)
ydata = under_derivative[36:np.argmax(under_derivative[60:110])+60].astype(float)

m3 = popt2[0]  # known slope of the final segment

best = grid_fit_4seg(xdata, ydata, m3=m3, linear_fit_end=55)    # try 0.0 first; if you want fewer jumps try e.g. 10 or 100

print(f'Overshoot: {(best["b"] - treatment_times[0] - 14)*10}')
print(f'Lag Phase: {(best["c"] - treatment_times[0] - 14)*10}')

# print(f'Overshoot: {(b_best - 40)/2}, Lag Phase: {(c_best - 40)/2}')
# evaluate fit
xfit = np.linspace(xdata.min(), xdata.max(), 500)
yfit = predict_4seg(
    xfit,
    best["a"], best["b"], best["c"],
    best["y0"],
    best["m1"], best["q1"], m3=m3)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7.6, 2.4), dpi=300)
ax[0].plot(pulse_derivative, color='C0', marker='.', lw=2, markersize=8)
ax[0].plot(nt_dev, color='C7', marker='.', lw=2, markersize=8)
# ax[0].plot(np.arange(len(pulse_derivative))+39, -0.0021930446024563703 * np.arange(len(pulse_derivative)) + 0.0357882352941176, color='C1', label='Linear fit', linewidth=2)
ax[0].plot(np.arange(len(pulse_derivative)), np.polyval(popt0, np.arange(len(pulse_derivative))), color='C1', label='Linear fit', linewidth=2)
ax[0].plot(np.arange(len(pulse_derivative)), np.polyval(popt2, np.arange(len(pulse_derivative))), color='C2', linewidth=2)
ax[0].plot(np.arange(len(pulse_derivative)), np.polyval(popt3, np.arange(len(pulse_derivative))), color='C3', linewidth=2)
ax[0].plot(np.arange(len(pulse_derivative)), np.polyval(popt4, np.arange(len(pulse_derivative))), color='C4', linewidth=2)
# ax[0].plot(x_treat, [1 for _ in range(len(x_treat))] * popt_treat[0], color='orange', label='Max speed', linewidth=2)
# ax[0].vlines(x=crossing_point_treat, ymin=0, ymax=5, color='green', linestyle='--', label='Half speed', linewidth=2)
ax[0].set_xlim(treatment_times[0] - 10, treatment_times[1] + 80)
# ax[0].axvspan(treatment_times[0], treatment_times[0] + (crossing_point_treat - treatment_times[0]) * 2, color='gray', alpha=0.3, label='Delay Period')
ax[1].plot(under_derivative, marker='.', color='C0', lw=2, markersize=8)
ax[1].plot(np.arange(len(under_derivative)), np.polyval(popt1, np.arange(len(under_derivative))), color='C5', label='Constant fit', linewidth=2)
ax[1].plot(np.arange(len(under_derivative)), np.polyval(popt0, np.arange(len(under_derivative))), color='C1', linewidth=2)
ax[1].plot(np.arange(len(under_derivative)) - 33, np.polyval(popt2, np.arange(len(under_derivative))), color='C2', linewidth=2)
# ax[1].plot(x_rel, [1 for _ in range(len(x_rel))] * popt_rel[0], color='orange', label='Max speed', linewidth=2)
# ax[1].vlines(x=crossing_point, ymin=0, ymax=5, color='green', linestyle='--', label='Half speed', linewidth=2)
# ax[1].axvspan(treatment_times[1], treatment_times[1] + (crossing_point - treatment_times[1]) * 2, color='gray', alpha=0.3, label='Delay Period')
ax[1].set_xlim(treatment_times[0] - 10, treatment_times[1] + 25) #treatment_times[0] - 10, 102)

xxx = np.arange(len(under_derivative))
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
plt.savefig('delays.pdf', bbox_inches='tight', transparent=True)
plt.show()
