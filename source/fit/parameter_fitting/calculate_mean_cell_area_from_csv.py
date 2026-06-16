from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

import yaml


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def load_area_values(csv_path, only_accepted=True):
    csv_path = Path(csv_path)
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"The CSV file {csv_path} has no header.")

        if "area_reported" in reader.fieldnames:
            area_column = "area_reported"
            default_unit = "reported_units"
        elif "area_px" in reader.fieldnames:
            area_column = "area_px"
            default_unit = "px^2"
        else:
            raise ValueError(f"The CSV file {csv_path} does not contain 'area_reported' or 'area_px'.")

        values: list[float] = []
        area_unit = default_unit

        for row in reader:
            if only_accepted and "accepted" in row and not parse_bool(row["accepted"]):
                continue

            if "area_unit" in row and row["area_unit"].strip():
                area_unit = row["area_unit"].strip()

            values.append(float(row[area_column]))

    if not values:
        raise RuntimeError("No area values were found in the CSV after applying the current filter.")

    return values, area_column, area_unit


def summarize_values(values):
    mean_value = sum(values) / len(values)
    median_value = statistics.median(values)
    std_value = statistics.stdev(values) if len(values) > 1 else 0.0
    sem_value = std_value / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"n": len(values),
            "mean": mean_value,
            "median": median_value,
            "std": std_value,
            "sem": sem_value}


def load_sim_pixel_area_um2(params_path, exp_pixel_size_um):
    params_path = Path(params_path)
    with params_path.open("r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)

    sim_pixel_to_exp_pixel_factor = float(eval(params["sim_pixel_to_exp_pixel_factor"], {"__builtins__": {}}, {}))
    return (sim_pixel_to_exp_pixel_factor * exp_pixel_size_um) ** 2


def calculate_scaling_factor(mean_area, area_sem, sim_pixel_area_um2):
    scaling_factor = sim_pixel_area_um2 / mean_area
    scaling_factor_error = sim_pixel_area_um2 * area_sem / (mean_area**2)
    return scaling_factor, scaling_factor_error


def main(csv_path, only_accepted=True, params_path="params.yaml", exp_pixel_size_um=8.648):
    values, area_column, area_unit = load_area_values(csv_path=csv_path, only_accepted=only_accepted)
    summary = summarize_values(values)
    sim_pixel_area_um2 = load_sim_pixel_area_um2(params_path=params_path, exp_pixel_size_um=exp_pixel_size_um)
    scaling_factor, scaling_factor_error = calculate_scaling_factor(mean_area=summary["mean"], area_sem=summary["sem"], sim_pixel_area_um2=sim_pixel_area_um2)

    print(f"CSV file: {Path(csv_path)}")
    print(f"Params file: {Path(params_path)}")
    print(f"Area column: {area_column}")
    print(f"Only accepted cells: {only_accepted}")
    print(f"Number of cells: {summary['n']}")
    print(f"Mean area: {summary['mean']:.6f} {area_unit}")
    print(f"Median area: {summary['median']:.6f} {area_unit}")
    print(f"Std: {summary['std']:.6f} {area_unit}")
    print(f"SEM: {summary['sem']:.6f} {area_unit}")
    print(f"Sim pixel area: {sim_pixel_area_um2:.6f} um^2")
    print(f"Scaling factor: {scaling_factor}")
    print(f"Scaling factor error: {scaling_factor_error}")
    print("Scaling factor error is propagated from the mean-area SEM.")


if __name__ == "__main__":
    csv_path = Path("data/Single_cell_resolution_yNA16_cell_area/Single_cell_resolution_yNA16_cell_measurements.csv")
    only_accepted = True
    params_path = Path("params.yaml")
    exp_pixel_size_um = 8.648

    main(csv_path=csv_path, only_accepted=only_accepted, params_path=params_path, exp_pixel_size_um=exp_pixel_size_um)
