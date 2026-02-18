#!/usr/bin/env python3
"""
Compute colony area (largest connected component) for all TIFFs in a folder.

- Optionally uses a list of identifiers to group files (P1_, P2, ...).
- For each TIFF: load, pad, find largest *central* component, compute area.
- Prints per-image area to console.
- Writes a single CSV "colony_areas.csv" into the same folder.

Columns in CSV:
    identifier    (string, or "all" if no identifier used)
    filename      (original TIFF file name)
    frame         (number extracted from file name, if present)
    colony_area   (pixel area of selected connected component)
    colony_center (centroid as [row, col])
    colony_radius (mean distance of contour points to centroid)
"""

import os
import re
import json
import numpy as np
import pandas as pd
from skimage import io, measure
from scipy.spatial.distance import cdist
from tqdm import tqdm


# ────────────────────────── helpers ──────────────────────────

def colony_parameters(array: np.ndarray):
    """
    Find the colony in the CENTER of the image and return its area,
    contour, centroid, and radius.

    Logic:
      1. Merge labels 1 and 2 into one foreground.
      2. Label connected components with background = 3.
      3. Among components whose centroid lies near the image center,
         pick the largest by area.
      4. If none are near the center, fall back to the globally largest
         component.

    Parameters
    ----------
    array : np.ndarray
        Label / segmentation image where colony pixels are 1 or 2,
        background is 3 (as in your pipeline).

    Returns
    -------
    area : float
        Area (in pixels) of the selected connected component.
    contour : np.ndarray
        N×2 array of contour coordinates (row, col) of the component.
    centroid : tuple
        (row, col) coordinates of the component centroid.
    radius : float
        Mean distance of contour points to centroid (in pixels).
    """
    # Reduce 2s to 1s → all foreground is label 1
    array = np.where(array == 2, 1, array)

    # Label connected components; background is 3
    labeled_colony = measure.label(array, background=3)
    detected_components = measure.regionprops(labeled_colony)

    if not detected_components:
        # No component found – return zeros / empty
        return 0.0, np.empty((0, 2)), (np.nan, np.nan), 0.0

    # Image center (row, col)
    h, w = array.shape
    center_r = h / 2.0
    center_c = w / 2.0

    # We define a "central window" as some fraction of the image size.
    # Here: centroid must be within 25% of height/width from the center.
    max_rel_dist = 0.25  # tweak if you like

    central_components = []
    for comp in detected_components:
        r, c = comp.centroid
        rel_dr = abs(r - center_r) / h
        rel_dc = abs(c - center_c) / w
        if rel_dr <= max_rel_dist and rel_dc <= max_rel_dist:
            central_components.append(comp)

    if central_components:
        # Largest component among those near the center
        chosen = max(central_components, key=lambda x: x.area)
    else:
        # Fallback: just the global largest component
        chosen = max(detected_components, key=lambda x: x.area)

    largest_label = chosen.label
    biggest_component_mask = labeled_colony == largest_label

    # Find contour(s)
    contours = measure.find_contours(biggest_component_mask, level=0.5)
    if not contours:
        return chosen.area, np.empty((0, 2)), chosen.centroid, 0.0

    # Take the longest contour
    largest_contour = max(contours, key=len)

    centroid = chosen.centroid  # (row, col)
    radius = np.mean(cdist(largest_contour, [centroid]))

    return chosen.area, largest_contour, centroid, radius


def get_number(filename: str) -> int:
    """
    Extract an integer frame number from a filename like 'P1_colony_0034.tiff'.

    If no number is found, returns 0 so those files are sorted first.
    """
    match = re.search(r'(\d+)\.tiff$', filename)
    return int(match.group(1)) if match else 0


# ─────────────────────── main computation ───────────────────────

def compute_colony_areas(folder_path: str, identifiers=None) -> pd.DataFrame:
    """
    Compute colony areas for all TIFF files in folder_path.

    Parameters
    ----------
    folder_path : str
        Folder containing .tiff images.
    identifiers : list of str or None
        If provided, files must contain one of these substrings.
        If None, all .tiff files are processed.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with one row per TIFF image.
    """
    folder_path = folder_path.replace('\\', '/')

    # Collect rows here
    rows = []

    if identifiers is None:
        # No grouping: just take all .tiff files
        tiff_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.tiff')]
        tiff_files.sort(key=get_number)

        if not tiff_files:
            print("No TIFF files found in folder.")
            return pd.DataFrame()

        print(f"Found {len(tiff_files)} TIFF files in {folder_path}")
        iterator = tqdm(tiff_files, desc="Processing TIFFs")
        for fname in iterator:
            file_path = os.path.join(folder_path, fname)
            img = io.imread(file_path)

            # Pad image with background label 3 around the frame
            # → ensures the colony doesn't artificially connect to the border.
            img = np.pad(img, 1, constant_values=3)

            area, contour, centroid, radius = colony_parameters(img)

            frame = get_number(fname)
            print(f"{fname} (frame {frame}): colony area = {area}")

            rows.append({
                "identifier": "all",
                "filename": fname,
                "frame": frame,
                "colony_area": area,
                "colony_center": json.dumps(centroid),
                "colony_radius": radius
            })

    else:
        # Grouped by identifiers (P1_, P2, P3, ...)
        for identifier in tqdm(identifiers, desc="Identifiers"):
            tiff_files = [
                f for f in os.listdir(folder_path)
                if f.lower().endswith('.tiff') and identifier in f
            ]
            if not tiff_files:
                print(f"No TIFF files found for identifier: {identifier}")
                continue

            tiff_files.sort(key=get_number)
            print(f"Identifier {identifier}: {len(tiff_files)} TIFF files found.")

            for fname in tqdm(tiff_files, desc=f"Processing {identifier}", leave=False):
                file_path = os.path.join(folder_path, fname)
                img = io.imread(file_path)

                # Pad image with background=3 to ignore border collisions
                img = np.pad(img, 1, constant_values=3)

                area, contour, centroid, radius = colony_parameters(img)

                frame = get_number(fname)
                print(f"{identifier} | {fname} (frame {frame}): colony area = {area}")

                rows.append({
                    "identifier": identifier,
                    "filename": fname,
                    "frame": frame,
                    "colony_area": area,
                    "colony_center": json.dumps(centroid),
                    "colony_radius": radius
                })

    # Turn into DataFrame and save CSV
    df = pd.DataFrame(rows)
    csv_path = os.path.join(folder_path, "colony_areas.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved colony areas to: {csv_path}")

    return df


# ────────────────────────── run as script ──────────────────────────

if __name__ == "__main__":
    # EDIT HERE
    folder_path = r"D:\Image_Segmentation\9_18_finals"
    identifiers = ['P1_', 'P2', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', 'P14']

    # Option A: use identifiers (your current scheme)
    colony_df = compute_colony_areas(folder_path, identifiers=identifiers)

    # Option B: to ignore identifiers and just process ALL tiffs in the folder,
    # comment out the line above and uncomment this:
    # colony_df = compute_colony_areas(folder_path, identifiers=None)
