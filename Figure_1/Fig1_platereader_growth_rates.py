import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Function to load the data from the second sheet
def load_data(filepath):
    df = pd.read_excel(filepath, sheet_name=1)  # Load second sheet

    # Generate the time column starting at 10 minutes and incrementing by 10 minutes for each row
    df['Time'] = np.arange(10, 10 * (len(df) + 1), 10)

    return df


# Function to filter the data for the exponential phase
def filter_exponential_phase(df, min_y=0.02, max_y=0.6):
    filtered_df = df[(df > min_y) & (df < max_y)]
    return filtered_df.dropna()


# Function to apply the logarithmic transformation
def apply_log_transformation(df):
    return np.log(df)


# Function to perform linear fitting on log-transformed data
def linear_fit(time, log_data):
    # Fit the data using a linear model: log(OD) = m*time + b
    def model(t, m, b):
        return m * t + b

    popt, _ = curve_fit(model, time, log_data)
    return popt  # Return slope (m) and intercept (b)


# Function to calculate doubling time from growth speed
def calculate_doubling_time(growth_speed):
    if growth_speed == 0:
        return np.inf  # Return infinity if the growth speed is 0 (no growth)
    return np.log(2) / growth_speed


# Function to process and calculate growth speed and doubling time for the given sets
# Function to plot the linear fit on log-transformed data
def plot_linear_fit(time, log_data, popt, title):
    plt.figure(figsize=(10, 6))
    plt.plot(time, log_data, 'o-', label='Log-transformed data')
    plt.plot(time, popt[0] * time + popt[1], 'r--', label=f'Fit: m={popt[0]:.4f}, b={popt[1]:.4f}')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Log(OD)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def process_and_calculate(filepath, well_groups, plot_fits=False):
    df = load_data(filepath)
    time = df['Time'].values  # Use the generated time column

    growth_speeds = {}
    doubling_times = {}

    for group_name, wells in well_groups.items():
        group_speeds = []
        group_doubling_times = []
        for well in wells:
            data = df[well].values
            filtered_data = filter_exponential_phase(df[well])

            # Check if there are enough data points
            if len(filtered_data) >= 10:
                time_filtered = df.loc[filtered_data.index, 'Time'].values
                log_data_filtered = apply_log_transformation(filtered_data)

                # Perform linear fit
                popt = linear_fit(time_filtered, log_data_filtered)

                # Extract the growth speed (slope of the linear fit)
                growth_speed = popt[0]
                group_speeds.append(growth_speed)

                # Calculate the doubling time
                doubling_time = calculate_doubling_time(growth_speed)
                group_doubling_times.append(doubling_time)

                # Plot the linear fit if the option is enabled
                if plot_fits:
                    plot_linear_fit(time_filtered, log_data_filtered, popt,
                                    f"{well} - Linear Fit on Log-Transformed Data")
            else:
                print(f"Skipping {well}: Not enough data points in the defined range.")
                group_speeds.append(0)  # Assign growth speed of 0 if no valid data
                group_doubling_times.append(np.inf)  # Assign infinity to doubling time if no valid data

        # Store the growth speeds and doubling times for the group
        growth_speeds[group_name] = group_speeds
        doubling_times[group_name] = group_doubling_times

    return growth_speeds, doubling_times


# Function to plot doubling times for both sets across different files
def plot_doubling_times_over_files(doubling_times_all, file_labels):
    plt.figure(figsize=(10, 6), dpi=300)

    for set_name in doubling_times_all.keys():
        means = [np.mean(doubling_times_all[set_name][i]) for i in range(len(file_labels))]
        stds = [np.std(doubling_times_all[set_name][i]) for i in range(len(file_labels))]
        plt.errorbar(file_labels, means, yerr=stds, label=f"{set_name} Doubling Time", marker='o', capsize=5,
                     linestyle='None')


    plt.xlabel('Condition')
    plt.ylabel('Doubling Time (minutes)')
    plt.title('Doubling Time Across Conditions')
    plt.legend()
    plt.grid(True)
    plt.show()


# Function to plot growth speeds for both sets across different files
def plot_growth_speeds_over_files(growth_speeds_all, file_labels):
    plt.figure(figsize=(10, 6), dpi=300)

    for set_name in growth_speeds_all.keys():
        means = [np.mean(growth_speeds_all[set_name][i]) for i in range(len(file_labels))]
        stds = [np.std(growth_speeds_all[set_name][i]) for i in range(len(file_labels))]
        plt.errorbar(file_labels, means, yerr=stds, label=f"{set_name} Growth Speed", marker='o', capsize=5,
                     linestyle='None')

    plt.ylim(bottom=0)
    plt.xlabel('Condition')
    plt.ylabel('Growth Speed (slope of log-transformed data)')
    plt.title('Growth Speed Across Conditions')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_combined_doubling_and_growth(doubling_times_all, growth_speeds_all, file_labels_combined):
    colors = {
        'yNA16': 'royalblue',
        'yNA16S': 'gold'
    }

    fig, axs = plt.subplots(2, 1, figsize=(10, 12), dpi=300, sharex=True)

    for set_name in ['yNA16', 'yNA16S']:
        # Average the first two conditions (30°C replicates)
        doubling_avg_30C = np.mean(doubling_times_all[set_name][:2], axis=0)
        doubling_std_30C = np.std(doubling_times_all[set_name][:2], axis=0)

        doubling_combined = [
            doubling_avg_30C,
            doubling_times_all[set_name][2]  # 35°C
        ]

        doubling_means = [np.mean(vals) for vals in doubling_combined]
        doubling_stds = [np.std(vals) for vals in doubling_combined]

        axs[0].errorbar(file_labels_combined, doubling_means, yerr=doubling_stds, label=set_name,
                        marker='o', capsize=5, color=colors[set_name], linestyle='-', linewidth=2)

        # Repeat for growth speeds
        growth_avg_30C = np.mean(growth_speeds_all[set_name][:2], axis=0)
        growth_std_30C = np.std(growth_speeds_all[set_name][:2], axis=0)

        growth_combined = [
            growth_avg_30C,
            growth_speeds_all[set_name][2]
        ]

        growth_means = [np.mean(vals) for vals in growth_combined]
        growth_stds = [np.std(vals) for vals in growth_combined]

        axs[1].errorbar(file_labels_combined, growth_means, yerr=growth_stds, label=set_name,
                        marker='o', capsize=5, color=colors[set_name], linestyle='-', linewidth=2)

    axs[0].set_title("Doubling Time Across Conditions")
    axs[0].set_ylabel("Doubling Time (min)")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].set_title("Growth Speed Across Conditions")
    axs[1].set_ylabel("Growth Speed (slope of log-transformed OD)")
    axs[1].set_xlabel("Condition")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.show()

def plot_nature_style_growth_speed(growth_speeds_all, file_labels_combined, save_path=None):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # Use Helvetica or fallback
    mpl.rcParams.update({
        'pdf.fonttype': 42,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial'],
        'font.size': 7,
        'axes.linewidth': 0.5,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.direction': 'out',
        'ytick.direction': 'out'
    })


    # Colors (colorblind-friendly options from ColorBrewer)
    colors = {
        'yNA16': '#4169E1',   # royalblue
        'yNA16S': '#DAA520'   # goldenrod
    }

    fig, ax = plt.subplots(figsize=(2, 1.25), dpi=600)  # ~89 mm wide, single column

    for set_name in ['yNA16', 'yNA16S']:
        # Average 30°C
        growth_avg_30C = np.mean(growth_speeds_all[set_name][:2], axis=0)
        growth_std_30C = np.std(growth_speeds_all[set_name][:2], axis=0)

        growth_35C = growth_speeds_all[set_name][2]
        growth_combined = [growth_avg_30C, growth_35C]

        means = [np.mean(g) for g in growth_combined]
        stds = [np.std(g) for g in growth_combined]

        ax.errorbar(
            file_labels_combined,
            means,
            yerr=stds,
            label=set_name,
            marker='o',
            color=colors[set_name],
            linestyle='-',
            linewidth=0.75,
            capsize=2,
            markersize=4
        )

    # Style adjustments
    ax.set_ylabel('Growth speed\n(slope of log(OD))')
    ax.set_xlabel('Temperature\n(°C)')
    ax.set_ylim(bottom=0)
    ax.tick_params(axis='both', which='both', length=3, width=0.5, labelsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc='lower left')
    ax.grid(False)

    plt.tight_layout()

    # Save high-resolution figure
    if save_path:
        fig.savefig(save_path, format='pdf', dpi=600, bbox_inches='tight')
    print("Using font:", plt.rcParams['font.sans-serif'])
    plt.show()


def plot_nature_style_divisions_per_hour_boxplot(doubling_times_all, file_labels_combined, save_path=None):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    mpl.rcParams.update({
        'pdf.fonttype': 42,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial'],
        'font.size': 7,
        'axes.linewidth': 0.5,
        'xtick.major.size': 0,
        'ytick.major.size': 3,
        'xtick.minor.size': 3,
        'xtick.direction': 'out',
        'ytick.direction': 'out'
    })

    # map friendly names → your dict keys
    key_map = {
        'Susceptible': 'yNA16',
        'Resistant':   'yNA16S'
    }

    colors = {
        'Susceptible': '#4169E1',
        'Resistant':   '#DAA520'
    }

    data = []
    positions = []
    box_colors = []
    offset = 0.2

    for i, temp_label in enumerate(file_labels_combined):
        center = i + 1
        for strain in ['Susceptible', 'Resistant']:
            orig_key = key_map[strain]
            # pull the two 30°C replicates or the single 35°C
            if temp_label.startswith("30"):
                dt = np.hstack((
                    doubling_times_all[orig_key][0],
                    doubling_times_all[orig_key][1]
                ))
            else:
                dt = np.array(doubling_times_all[orig_key][2])

            div_per_hr = 60.0 / dt
            data.append(div_per_hr)
            positions.append(center + ( -offset if strain=='Susceptible' else offset ))
            box_colors.append(colors[strain])

    fig, ax = plt.subplots(figsize=(2.5, 1.5), dpi=600)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.35,
        patch_artist=True,
        showfliers=False
    )

    for patch, c in zip(bp['boxes'], box_colors):
        patch.set_facecolor(c)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.4)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(0.6)
    for whisker in bp['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(0.5)
    for cap in bp['caps']:
        cap.set_color('black')
        cap.set_linewidth(0.5)

    ax.set_xlim(0.5, len(file_labels_combined) + 0.5)
    # Major ticks: group labels
    ax.set_xticks([1, 2])
    ax.set_xticklabels(file_labels_combined, fontsize=7)
    ax.xaxis.set_tick_params(which='major', pad=12)

    # Minor ticks: Sus/Res labels under each box
    ax.set_xticks(positions, minor=True)
    ax.set_xticklabels(['Sensitive', 'Resistant'] * len(file_labels_combined),
                       minor=True, fontsize=6)
    ax.xaxis.set_tick_params(which='minor', pad=3)

    ax.set_ylabel('Divisions per hour', fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', width=0.5, length=3, labelsize = 6, pad = 2)
    ax.yaxis.labelpad = 3
    #ax.set_ylim(bottom=0)
    x_left_lim, x_right_lim = ax.get_xlim()
    ax.axvspan((x_left_lim + x_right_lim) / 2, x_right_lim, facecolor="#bfbfbf", alpha=1, zorder=-1, linewidth=0)
    ax.axhline(
        y=0,
        color='0.5',  # or '#888888'
        linestyle=(0, (3, 3)),  # short dashes
        linewidth=0.6,
        alpha=0.6,
        zorder=0.5  # under the boxes
    )
    ax.grid(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, format='pdf', dpi=600, bbox_inches='tight')
    plt.show()


file_30C = r"S:/Members/Nico/Platereader/yNA16_s_excel_files/20230706_yNA16_S_30C.xlsx"  # Update with the correct path
file_35C = r"S:/Members/Nico/Platereader/yNA16_s_excel_files/20230705_yNA16_S_35C.xlsx"  # Update with the correct path
file_30C2 = r"S:/Members/Nico/Platereader/yNA16_s_excel_files/20230808_yNA16_S_30C.xlsx"  # Update with the correct path

# Define well groups based on your criteria
well_groups = {
    'yNA16': ['A1', 'B1', 'C1', 'A2', 'B2', 'C2', 'A3', 'B3', 'C3'],
    'yNA16S': ['A6', 'B6', 'C6', 'A7', 'B7', 'C7', 'A8', 'B8', 'C8']
}

# List of files and their labels
files = [file_30C, file_30C2, file_35C]
file_labels = ["30°C", "30°C2", "35°C"]

# Collect results for all files
doubling_times_all = {'yNA16': [], 'yNA16S': []}
growth_speeds_all = {'yNA16': [], 'yNA16S': []}

for filepath in files:
    speeds, doubling_times = process_and_calculate(filepath, well_groups, plot_fits=False)
    for set_name in well_groups.keys():
        doubling_times_all[set_name].append(doubling_times[set_name])
        growth_speeds_all[set_name].append(speeds[set_name])

# Plot doubling times across all conditions
#plot_doubling_times_over_files(doubling_times_all, file_labels)

# Plot growth speeds across all conditions
#plot_growth_speeds_over_files(growth_speeds_all, file_labels)

# Replace the original plotting calls with the combined version
file_labels_combined = ["30°C", "35°C"]
#plot_combined_doubling_and_growth(doubling_times_all, growth_speeds_all, file_labels_combined)
#plot_nature_style_growth_speed(growth_speeds_all, file_labels_combined,
#                              save_path=r"C:\Users\nappold\Desktop\Manuscript Figures\growth_speed_plot.pdf")


plot_nature_style_divisions_per_hour_boxplot(
    doubling_times_all,
    file_labels_combined,
    #save_path=r"C:\Users\nappold\Desktop\Manuscript Figures\division_rate_boxplot.pdf"
)