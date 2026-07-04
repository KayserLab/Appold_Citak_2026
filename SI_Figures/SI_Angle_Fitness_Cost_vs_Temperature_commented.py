from pathlib import Path
import csv
import math
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))

import matplotlib
import matplotlib as mpl
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_ROOT = SCRIPT_DIR / "Input_files"
OUTPUT_ROOT = SCRIPT_DIR / "Output_files"

# Place the expected angle CSV files in this folder.
ROOT_DIR = INPUT_ROOT / "Angle_measurements"

# Save all generated CSV tables and figures in this folder.
OUTPUT_DIR = OUTPUT_ROOT / "Angle_measurements"

# Set whether the script searches ROOT_DIR recursively for the expected files.
RECURSIVE_SEARCH = True

# List the angle CSV filenames that should be loaded.
EXPECTED_FILENAMES = [
    "30_pure_linear.csv",
    "31_pure_linear.csv",
    "32_pure_linear.csv",
    "35_pure_linear.csv",
]


# =============================================================================
# PLOT STYLE
# =============================================================================

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "axes.linewidth": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.frameon": True,
    "legend.fontsize": 6,
    "lines.linewidth": 1.0,
    "savefig.dpi": 300,
    "figure.dpi": 300,
    "axes.spines.top": True,
    "axes.spines.right": True,
})


# =============================================================================
# FILE HANDLING
# =============================================================================

def find_expected_files(root_dir: Path) -> list[Path]:
    root_dir = root_dir.resolve()

    if not root_dir.exists():
        raise FileNotFoundError(f"ROOT_DIR does not exist:\n{root_dir}")

    found_files: list[Path] = []
    missing_files: list[str] = []
    duplicate_messages: list[str] = []

    for filename in EXPECTED_FILENAMES:
        if RECURSIVE_SEARCH:
            matches = sorted(root_dir.rglob(filename))
        else:
            matches = sorted(root_dir.glob(filename))

        if len(matches) == 0:
            missing_files.append(filename)
        elif len(matches) > 1:
            duplicate_messages.append(
                f"{filename}:\n" + "\n".join(f"  - {match}" for match in matches)
            )
        else:
            found_files.append(matches[0])

    if missing_files or duplicate_messages:
        message_parts = []

        if missing_files:
            message_parts.append(
                "Missing expected files:\n"
                + "\n".join(f"  - {filename}" for filename in missing_files)
            )

        if duplicate_messages:
            message_parts.append(
                "Found duplicate matches. Please remove duplicates or disable recursive search:\n"
                + "\n\n".join(duplicate_messages)
            )

        raise RuntimeError("\n\n".join(message_parts))

    return found_files


def temperature_from_filename(path: Path) -> int:
    return int(path.name.split("_", 1)[0])


# =============================================================================
# FITNESS CALCULATIONS
# =============================================================================

def angle_to_fitness_cost_percent(angle_degrees: float) -> float:
    """Convert linear-sector opening angle phi to slower-strain fitness cost.

    Korolev et al. 2012, equation 8:
        tan(phi / 2) = sqrt(s * (2 + s))

    Solving for s gives:
        s = sec(phi / 2) - 1

    The slower strain's cost relative to the faster strain is:
        100 * (1 - cos(phi / 2))
    """
    phi = abs(angle_degrees)

    if phi >= 180:
        raise ValueError(f"Angle must be less than 180 degrees, got {angle_degrees}")

    return 100.0 * (1.0 - math.cos(math.radians(phi) / 2.0))


def angle_to_sensitive_resistant_ratio(angle_degrees: float) -> float:
    """Convert linear-sector opening angle phi to v_sensitive / v_resistant."""
    phi = abs(angle_degrees)

    if phi >= 180:
        raise ValueError(f"Angle must be less than 180 degrees, got {angle_degrees}")

    return math.cos(math.radians(phi) / 2.0)


# =============================================================================
# CSV IO
# =============================================================================

def read_angles(path: Path) -> list[float]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)

        if "angle" not in (reader.fieldnames or []):
            raise ValueError(f"{path} does not contain an 'angle' column")

        angles: list[float] = []

        for row in reader:
            value = row.get("angle", "")

            if value is None:
                continue

            value = value.strip()

            if value:
                angles.append(float(value))

    return angles


def write_combined(rows: list[dict[str, float | int | str]], output_dir: Path) -> None:
    output_path = output_dir / "combined_fitness_measurements.csv"

    with output_path.open("w", newline="") as handle:
        fieldnames = [
            "temperature_c",
            "source_file",
            "angle_degrees",
            "fitness_cost_percent",
            "v_sensitive_over_v_resistant",
        ]

        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict[str, float | int | str]], output_dir: Path) -> None:
    groups: dict[int, list[float]] = {}

    for row in rows:
        groups.setdefault(int(row["temperature_c"]), []).append(
            float(row["fitness_cost_percent"])
        )

    output_path = output_dir / "fitness_summary.csv"

    with output_path.open("w", newline="") as handle:
        fieldnames = [
            "temperature_c",
            "n",
            "mean_fitness_cost_percent",
            "median_fitness_cost_percent",
            "std_fitness_cost_percent",
            "min_fitness_cost_percent",
            "max_fitness_cost_percent",
        ]

        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for temperature in sorted(groups):
            values = np.array(groups[temperature], dtype=float)

            writer.writerow({
                "temperature_c": temperature,
                "n": values.size,
                "mean_fitness_cost_percent": np.mean(values),
                "median_fitness_cost_percent": np.median(values),
                "std_fitness_cost_percent": np.std(values, ddof=1) if values.size > 1 else 0.0,
                "min_fitness_cost_percent": np.min(values),
                "max_fitness_cost_percent": np.max(values),
            })

        all_values = np.array(
            [float(row["fitness_cost_percent"]) for row in rows],
            dtype=float,
        )

        writer.writerow({
            "temperature_c": "all",
            "n": all_values.size,
            "mean_fitness_cost_percent": np.mean(all_values),
            "median_fitness_cost_percent": np.median(all_values),
            "std_fitness_cost_percent": np.std(all_values, ddof=1) if all_values.size > 1 else 0.0,
            "min_fitness_cost_percent": np.min(all_values),
            "max_fitness_cost_percent": np.max(all_values),
        })


# =============================================================================
# PLOTTING
# =============================================================================

def add_histogram(ax, values: np.ndarray, title: str, bins: np.ndarray | int) -> None:
    mean = float(np.mean(values))
    median = float(np.median(values))

    ax.hist(
        values,
        bins=bins,
        color="#4f81bd",
        edgecolor="white",
        alpha=0.85,
        linewidth=0.5,
    )

    ax.axvline(
        mean,
        color="#c43c39",
        linewidth=1.0,
        label=f"mean = {mean:.3g}",
    )

    ax.axvline(
        median,
        color="#2e7d32",
        linewidth=1.0,
        linestyle="--",
        label=f"median = {median:.3g}",
    )

    ax.set_title(title)
    ax.set_xlabel("Fitness cost of slower type (% of faster type)")
    ax.set_ylabel("Count")
    ax.legend()


def plot_histograms(rows: list[dict[str, float | int | str]], output_dir: Path) -> None:
    grouped: dict[int, list[float]] = {}

    for row in rows:
        grouped.setdefault(int(row["temperature_c"]), []).append(
            float(row["fitness_cost_percent"])
        )

    all_values = np.array(
        [float(row["fitness_cost_percent"]) for row in rows],
        dtype=float,
    )

    upper_bin = max(1.0, float(np.max(all_values)) * 1.05)
    bins = np.linspace(0, upper_bin, 12)

    temperatures = sorted(grouped)
    n_groups = len(temperatures)
    n_cols = 2
    n_rows = int(np.ceil(n_groups / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.0, 4.0),
        constrained_layout=True,
        squeeze=False,
    )

    axes_flat = axes.ravel()

    for ax, temperature in zip(axes_flat, temperatures):
        values = np.array(grouped[temperature], dtype=float)
        add_histogram(ax, values, f"{temperature} °C (n={values.size})", bins=bins)

    for ax in axes_flat[n_groups:]:
        ax.set_visible(False)

    fig.suptitle("Fitness cost from linear-sector angles by temperature")

    fig.savefig(output_dir / "fitness_histograms_by_temperature.png")
    fig.savefig(output_dir / "fitness_histograms_by_temperature.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.6, 2.4), constrained_layout=True)
    add_histogram(ax, all_values, f"All temperatures (n={all_values.size})", bins=bins)
    fig.suptitle("All fitness cost measurements")

    fig.savefig(output_dir / "fitness_histogram_all.png")
    fig.savefig(output_dir / "fitness_histogram_all.pdf")
    plt.close(fig)


def temperature_background_gray(temperature: int, min_temp: int, max_temp: int) -> float:
    if max_temp == min_temp:
        return 1.0

    # Gradient from white (#ffffff) at min_temp
    # to #bfbfbf at max_temp, e.g. 35 °C
    gray_at_max = int("bf", 16) / 255
    fraction = (temperature - min_temp) / (max_temp - min_temp)

    return 1.0 + fraction * (gray_at_max - 1.0)


def plot_temperature_response(
    rows: list[dict[str, float | int | str]],
    output_dir: Path,
) -> None:
    grouped: dict[int, list[float]] = {}

    for row in rows:
        grouped.setdefault(int(row["temperature_c"]), []).append(
            float(row["v_sensitive_over_v_resistant"])
        )

    temperatures = sorted(grouped)
    values = [np.array(grouped[temperature], dtype=float) for temperature in temperatures]

    min_temp = min(temperatures)
    max_temp = max(temperatures)

    fig, ax = plt.subplots(figsize=(2.95, 1.8), constrained_layout=True)

    tick_temperatures = list(range(min_temp, max_temp + 1))

    for temperature in tick_temperatures:
        gray = temperature_background_gray(temperature, min_temp, max_temp)

        ax.axvspan(
            temperature - 0.5,
            temperature + 0.5,
            color=(gray, gray, gray),
            zorder=0,
            linewidth=0,
        )

    box = ax.boxplot(
        values,
        positions=temperatures,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.0},
        whiskerprops={"color": "black", "linewidth": 0.5},
        capprops={"color": "black", "linewidth": 0.5},
        boxprops={"color": "black", "linewidth": 0.5},
    )

    for patch in box["boxes"]:
        patch.set_facecolor("royalblue")
        patch.set_alpha(0.85)

    rng = np.random.default_rng(7)

    for temperature, y_values in zip(temperatures, values):
        jitter = rng.uniform(-0.12, 0.12, size=y_values.size)

        ax.scatter(
            np.full(y_values.size, temperature) + jitter,
            y_values,
            s=8,
            color="black",
            alpha=0.7,
            linewidths=0,
            zorder=3,
        )

    ax.set_xlim(min_temp - 0.5, max_temp + 0.5)
    ax.set_xticks(tick_temperatures)
    ax.set_xticklabels([str(temperature) for temperature in tick_temperatures])
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("v_sensitive / v_resistant")
    ax.set_title("Fitness cost as a function of temperature")

    fig.savefig(output_dir / "fitness_cost_vs_temperature.png")
    fig.savefig(output_dir / "fitness_cost_vs_temperature.pdf")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    root_dir = ROOT_DIR.resolve()
    output_dir = OUTPUT_DIR.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_files = find_expected_files(root_dir)

    print("Found input files:")
    for path in data_files:
        print(f"  - {path}")

    rows: list[dict[str, float | int | str]] = []

    for path in data_files:
        temperature = temperature_from_filename(path)

        try:
            source_file = str(path.relative_to(root_dir))
        except ValueError:
            source_file = path.name

        for angle in read_angles(path):
            rows.append({
                "temperature_c": temperature,
                "source_file": source_file,
                "angle_degrees": angle,
                "fitness_cost_percent": angle_to_fitness_cost_percent(angle),
                "v_sensitive_over_v_resistant": angle_to_sensitive_resistant_ratio(angle),
            })

    if not rows:
        raise RuntimeError("No angle measurements were found in the input files.")

    write_combined(rows, output_dir)
    write_summary(rows, output_dir)
    plot_histograms(rows, output_dir)
    plot_temperature_response(rows, output_dir)

    print("\nSaved outputs to:")
    print(f"  {output_dir}")
    print("\nCreated:")
    print("  - combined_fitness_measurements.csv")
    print("  - fitness_summary.csv")
    print("  - fitness_histograms_by_temperature.png")
    print("  - fitness_histograms_by_temperature.pdf")
    print("  - fitness_histogram_all.png")
    print("  - fitness_histogram_all.pdf")
    print("  - fitness_cost_vs_temperature.png")
    print("  - fitness_cost_vs_temperature.pdf")


if __name__ == "__main__":
    main()