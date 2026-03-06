#!/usr/bin/env python3
"""
Control script to get total area of the final colony frame, to proof that extrapolation works well.
Compute colony area (largest connected component) for segmented TIFFs in a folder and write one CSV.

Folder structure (next to this script):
- Input_files/
    - <input_relpath>/          (segmented TIFF frames)
- Output_files/
    - colony_areas.csv

Assumptions about segmentation labels:
- Colony pixels are 1 or 2 (2 is merged into 1 for colony detection)
- Background is 3

CSV columns:
    identifier     (string)
    filename       (original TIFF file name)
    frame          (number extracted from file name, if present)
    colony_area    (pixel area of selected connected component)
    colony_center  (centroid as JSON [row, col])
    colony_radius  (mean distance of contour points to centroid)
"""

import os
import re
import json
import numpy as np
import pandas as pd

from skimage import io, measure
from scipy.spatial.distance import cdist
from tqdm import tqdm


def colony_parameters(array: np.ndarray):
    """
    Return colony area, contour, centroid, and radius for the colony located near the image center.

    Selection logic:
      1) Merge labels 1 and 2 into one foreground.
      2) Label connected components with background=3.
      3) Among components whose centroid lies within 25% (relative) of the image center
         in both row and col, pick the largest by area.
      4) If none are near the center, fall back to the globally largest component.

    Returns
    -------
    area : float
        Area (px) of the selected connected component.
    contour : np.ndarray
        N×2 contour coordinates (row, col).
    centroid : tuple[float, float]
        (row, col) centroid.
    radius : float
        Mean distance of contour points to centroid (px).
    """
    array = np.where(array == 2, 1, array)

    labeled_colony = measure.label(array, background=3)
    components = measure.regionprops(labeled_colony)

    if not components:
        return 0.0, np.empty((0, 2)), (np.nan, np.nan), 0.0

    h, w = array.shape
    center_r, center_c = h / 2.0, w / 2.0
    max_rel_dist = 0.25

    central_components = []
    for comp in components:
        r, c = comp.centroid
        rel_dr = abs(r - center_r) / h
        rel_dc = abs(c - center_c) / w
        if rel_dr <= max_rel_dist and rel_dc <= max_rel_dist:
            central_components.append(comp)

    chosen = max(central_components, key=lambda x: x.area) if central_components else max(components, key=lambda x: x.area)

    mask = labeled_colony == chosen.label
    contours = measure.find_contours(mask, level=0.5)
    if not contours:
        return float(chosen.area), np.empty((0, 2)), chosen.centroid, 0.0

    contour = max(contours, key=len)
    centroid = chosen.centroid
    radius = float(np.mean(cdist(contour, [centroid])))

    return float(chosen.area), contour, centroid, radius


def get_number(filename: str) -> int:
    """
    Extract trailing frame number from filenames ending in '<number>.tiff' for sorting.
    """
    match = re.search(r"(\d+)\.tiff$", filename, flags=re.IGNORECASE)
    return int(match.group(1)) if match else 0


def compute_colony_areas(folder_path: str, output_dir: str, identifiers: list[str] | None) -> pd.DataFrame:
    """
    Compute colony areas for all TIFF files in `folder_path`, optionally grouped by `identifiers`,
    and save a single CSV to `output_dir`.
    """
    os.makedirs(output_dir, exist_ok=True)

    rows: list[dict] = []

    if identifiers is None:
        tiff_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".tiff")]
        tiff_files.sort(key=get_number)

        for fname in tqdm(tiff_files, desc="Processing TIFFs"):
            img = io.imread(os.path.join(folder_path, fname))
            img = np.pad(img, 1, constant_values=3)

            area, _contour, centroid, radius = colony_parameters(img)

            rows.append(
                {
                    "identifier": "all",
                    "filename": fname,
                    "frame": get_number(fname),
                    "colony_area": area,
                    "colony_center": json.dumps([float(centroid[0]), float(centroid[1])]),
                    "colony_radius": float(radius),
                }
            )

    else:
        for identifier in tqdm(identifiers, desc="Identifiers"):
            tiff_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".tiff") and identifier in f]
            tiff_files.sort(key=get_number)

            for fname in tqdm(tiff_files, desc=f"{identifier}", leave=False):
                img = io.imread(os.path.join(folder_path, fname))
                img = np.pad(img, 1, constant_values=3)

                area, _contour, centroid, radius = colony_parameters(img)

                rows.append(
                    {
                        "identifier": identifier,
                        "filename": fname,
                        "frame": get_number(fname),
                        "colony_area": area,
                        "colony_center": json.dumps([float(centroid[0]), float(centroid[1])]),
                        "colony_radius": float(radius),
                    }
                )

    df = pd.DataFrame(
        rows,
        columns=["identifier", "filename", "frame", "colony_area", "colony_center", "colony_radius"],
    )

    csv_path = os.path.join(output_dir, "colony_areas.csv")
    df.to_csv(csv_path, index=False)

    return df


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))

    input_root = os.path.join(script_dir, "Input_files")
    output_root = os.path.join(script_dir, "Output_files")

    input_relpath = "Segmented_TIFFs_finals"  # Adjust this to your actual subfolder name containing TIFFs
    folder_path = os.path.join(input_root, input_relpath)

    identifiers = ["P2", "P3", "P4", "P5", "P6", "P7", "P8", "P10", "P11", "P12", "P14", "P15", "P16"]

    compute_colony_areas(folder_path, output_root, identifiers=identifiers)


if __name__ == "__main__":
    main()