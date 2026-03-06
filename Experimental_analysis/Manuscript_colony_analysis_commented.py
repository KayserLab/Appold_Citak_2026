#!/usr/bin/env python3
"""
Extract colony geometry over time from segmented TIFF time-lapse stacks and save as CSV.

Folder structure (next to this script):
- Input_files/
    - <input_relpath>/           (your TIFF frames)
- Output_files/
    - colony_data_<identifier>.csv

Assumptions about segmentation labels:
- Background: 3
- Colony labels: 1 and 2 (2 is merged into 1 for colony detection)

Per frame, the script computes:
- colony_area (px)
- colony_center (centroid; row/col)
- colony_radius (mean distance from contour to centroid; px)
- colony_contour (JSON-encoded list of contour points in (row, col))
"""

import os
import re
import json
import numpy as np
import pandas as pd

from skimage import io, measure
from scipy.spatial.distance import cdist
from tqdm import tqdm


def colony_parameters(array: np.ndarray) -> tuple[float, np.ndarray, tuple[float, float], float]:
    """
    Compute colony parameters from a segmented frame.

    Returns
    -------
    colony_area : float
        Area (px) of the largest connected component (colony).
    largest_contour : np.ndarray
        Longest contour of the colony mask (Nx2 array in (row, col)).
    centroid_rc : tuple[float, float]
        Colony centroid (row, col).
    colony_radius : float
        Mean distance from contour points to centroid (px).
    """
    array = np.where(array == 2, 1, array)
    labeled_colony = measure.label(array, background=3)

    components = measure.regionprops(labeled_colony)
    biggest_component = max(components, key=lambda x: x.area)
    largest_label = biggest_component.label

    colony_mask = labeled_colony == largest_label
    contours = measure.find_contours(colony_mask, level=0.5)
    largest_contour = max(contours, key=len)

    colony_radius = float(np.mean(cdist(largest_contour, [biggest_component.centroid])))

    return float(biggest_component.area), largest_contour, biggest_component.centroid, colony_radius


def get_number(filename: str) -> int:
    """
    Extract trailing frame number from filenames ending in '<number>.tiff' for sorting.
    """
    match = re.search(r"(\d+)\.tiff$", filename)
    return int(match.group(1)) if match else 0


def tiff_data_to_colony_size(folder_path: str, output_dir: str, identifiers: list[str]) -> pd.DataFrame:
    """
    For each identifier:
    - Load all TIFF frames matching the identifier
    - Compute colony contour/centroid/radius/area per frame
    - Save one CSV per identifier to `output_dir`
    """
    os.makedirs(output_dir, exist_ok=True)

    last_colony_df = pd.DataFrame()

    for identifier in tqdm(identifiers, desc="Position"):
        tiff_files = [f for f in os.listdir(folder_path) if f.endswith(".tiff") and identifier in f]
        if not tiff_files:
            print(f"[WARN] No TIFF files found for identifier: {identifier}")
            continue

        tiff_files.sort(key=get_number)

        images = []
        for file in tqdm(tiff_files, desc=f"Loading {identifier}", leave=False):
            image = io.imread(os.path.join(folder_path, file))
            image = np.pad(image, 1, constant_values=3)  # pad with background at borders
            images.append(image)

        combined_array = np.stack(images)  # (T, H, W)

        rows = []
        for t in tqdm(range(combined_array.shape[0]), desc=f"Frames {identifier}", leave=False):
            colony_area, contour, centroid, radius = colony_parameters(combined_array[t])

            rows.append(
                {
                    "colony_center": centroid,  # (row, col)
                    "colony_radius": float(radius),
                    "colony_contour": json.dumps(contour.tolist()),
                    "colony_area": float(colony_area),
                }
            )

        colony_df = pd.DataFrame(
            rows,
            columns=["colony_center", "colony_radius", "colony_contour", "colony_area"],
        )

        out_csv = os.path.join(output_dir, f"colony_data_{identifier}.csv")
        colony_df.to_csv(out_csv, index=False)

        last_colony_df = colony_df

    return last_colony_df


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    input_root = os.path.join(script_dir, "Input_files")
    output_root = os.path.join(script_dir, "Output_files")

    # Subfolder inside Input_files/ (set "" if TIFFs are directly in Input_files/)
    input_relpath = os.path.join("Segmented_Images",  "Your_Experiment_folder_folder")  # <-- CHANGE THIS to your folder containing TIFF stacks (can have subfolders if needed)  # <-- CHANGE THIS to your subfolder name or "" if none
    folder_path = os.path.join(input_root, input_relpath)

    identifiers = ["P2", "P3", "P4", "P5", "P6", "P7", "P8", "P10", "P11", "P12", "P14", "P15", "P16"] # <-- CHANGE THIS to your actual identifiers (e.g. positions) that are part of the TIFF filenames. P1_ is recommended to address P1

    tiff_data_to_colony_size(folder_path, output_root, identifiers)


if __name__ == "__main__":
    main()