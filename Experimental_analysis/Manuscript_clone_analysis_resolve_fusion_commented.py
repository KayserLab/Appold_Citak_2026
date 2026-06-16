#!/usr/bin/env python3
"""
Extract clone detections from segmented TIFF time-lapse images, link them with trackpy,
optionally resolve fusion events by keeping the clone closer to the colony front, and
save per-identifier CSV outputs.

Folder structure (next to this script):
- Input_files/
    - <input_relpath>/            (your TIFF stacks; may contain subfolders if needed)
- Output_files/
    - (CSV outputs are written here)

Assumptions about segmentation labels:
- Clone pixels: value 1
- Other class (e.g. resistant): value 2 (treated as background for clone detection)
- Background: value 3

Outputs:
- Output_files/clone_data_fusion_resolved_<identifier>.csv
"""

import os
import re
import numpy as np
import pandas as pd
import trackpy as tp

from skimage import io, measure
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from tqdm import tqdm


def colony_parameters(array: np.ndarray):
    """
    Compute colony properties from a segmented frame.

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
    colony_mask : np.ndarray
        Boolean mask of the colony (True inside colony).
    """
    array = np.where(array == 2, 1, array)  # merge class 2 into 1 for colony detection
    labeled_colony = measure.label(array, background=3)
    detected_components = measure.regionprops(labeled_colony)

    biggest_component = max(detected_components, key=lambda x: x.area)
    largest_label = biggest_component.label

    colony_mask = labeled_colony == largest_label
    contours = measure.find_contours(colony_mask, level=0.5)
    largest_contour = max(contours, key=len)

    colony_radius = np.mean(cdist(largest_contour, [biggest_component.centroid]))

    return (
        biggest_component.area,
        largest_contour,
        biggest_component.centroid,
        colony_radius,
        colony_mask,
    )


def get_number(filename: str) -> int:
    """
    Extract trailing frame number from filenames ending in '<number>.tiff' for sorting.
    """
    match = re.search(r"(\d+)\.tiff$", filename)
    return int(match.group(1)) if match else 0


def get_clones(array: np.ndarray):
    """
    Detect connected components corresponding to clones (label 1).

    Notes
    -----
    - Values 2 are converted to 3 so they do not contribute to clone detection.
    - The returned regionprops are computed on a labeled image with background=3.
    """
    array = np.where(array == 2, 3, array)

    if 1 in array:
        labeled_clone = measure.label(array, background=3)
        clone_props = measure.regionprops(labeled_clone)
        return clone_props, labeled_clone

    return None, None


def reassign_fusions_by_front(
    linked_df: pd.DataFrame,
    spatial_threshold: float = 10,
    plot_fusions: bool = False,
):
    """
    Resolve potential fusion events after trackpy linking.

    When two distinct particles come within `spatial_threshold` in the same frame,
    keep the identity of the clone closer to the colony front (smaller distance_to_edge),
    and relabel the other track from that frame onward.

    Requirements
    ------------
    linked_df must contain: ['frame', 'particle', 'x', 'y', 'distance_to_edge'].
    """
    reassigned_df = linked_df.copy()
    reassignment_log = []

    for frame in reassigned_df["frame"].unique():
        frame_data = reassigned_df[reassigned_df["frame"] == frame]
        coords = frame_data[["x", "y"]].values
        particles = frame_data["particle"].values

        if len(coords) < 2:
            continue

        distances = cdist(coords, coords)

        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                pid1, pid2 = particles[i], particles[j]
                if pid1 == pid2:
                    continue

                if distances[i, j] < spatial_threshold:
                    d1_row = frame_data[frame_data["particle"] == pid1].iloc[0]
                    d2_row = frame_data[frame_data["particle"] == pid2].iloc[0]

                    # Keep the clone closer to the colony front (smaller distance_to_edge)
                    to_keep = pid1 if d1_row["distance_to_edge"] < d2_row["distance_to_edge"] else pid2
                    to_relabel = pid2 if to_keep == pid1 else pid1

                    mask = (reassigned_df["particle"] == to_relabel) & (reassigned_df["frame"] >= frame)
                    reassigned_df.loc[mask, "particle"] = to_keep

                    reassignment_log.append(
                        {
                            "frame": int(frame),
                            "kept_particle": int(to_keep),
                            "relabelled_particle": int(to_relabel),
                            "distance_between_centroids": float(distances[i, j]),
                            "front_distance_keep": float(min(d1_row["distance_to_edge"], d2_row["distance_to_edge"])),
                            "front_distance_relabelled": float(
                                max(d1_row["distance_to_edge"], d2_row["distance_to_edge"])
                            ),
                        }
                    )

                    if plot_fusions:
                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.plot(d1_row["x"], d1_row["y"], "o", label=f"keep {to_keep}")
                        ax.plot(d2_row["x"], d2_row["y"], "o", label=f"relabel {to_relabel}")
                        ax.set_title(f"Frame {frame}: {to_relabel} → {to_keep}")
                        ax.legend()
                        ax.set_aspect("equal", adjustable="box")
                        plt.tight_layout()
                        plt.show()

    reassigned_df = reassigned_df.sort_values(["particle", "frame"]).reset_index(drop=True)
    return reassigned_df, reassignment_log


def tiff_data_to_csv(folder_path: str, output_dir: str, identifiers: list[str], search_area: float):
    """
    For each identifier:
    - Load all TIFF frames matching the identifier
    - Detect colony contour/centroid/radius
    - Detect clones inside the colony and compute distances
    - Link detections over time using trackpy
    - Reassign fusions by keeping the clone closer to the colony front
    - Save one CSV per identifier
    """
    os.makedirs(output_dir, exist_ok=True)

    for identifier in tqdm(identifiers, desc="Position"):
        tiff_files = [f for f in os.listdir(folder_path) if f.endswith(".tiff") and identifier in f]
        tiff_files.sort(key=get_number)

        if not tiff_files:
            print(f"[WARN] No TIFF files found for identifier '{identifier}' in '{folder_path}'. Skipping.")
            continue

        images = []
        for file in tiff_files:
            img = io.imread(os.path.join(folder_path, file))
            img = np.pad(img, 1, constant_values=3)  # pad with background to avoid border artifacts
            images.append(img)

        combined_array = np.stack(images)  # (T, H, W)
        all_data = pd.DataFrame()

        for t in tqdm(range(combined_array.shape[0]), desc="Frames", leave=False):
            colony_area, contour, centroid, radius, colony_mask = colony_parameters(combined_array[t])

            masked_array = np.where(colony_mask, combined_array[t], 3)
            detected_clones, labeled_image = get_clones(masked_array)

            if detected_clones is None:
                continue

            for clone in detected_clones:
                if clone.area <= search_area:
                    continue

                label = clone.label
                clone_mask = labeled_image == label

                clone_contours = measure.find_contours(clone_mask, level=0.8)
                if not clone_contours:
                    continue
                clone_contour = max(clone_contours, key=len)

                distance_to_edge = cdist(clone_contour, contour)
                min_idx = np.unravel_index(np.argmin(distance_to_edge), distance_to_edge.shape)

                closest_point_clone = clone_contour[min_idx[0]]   # (row, col)
                closest_point_colony = contour[min_idx[1]]        # (row, col)

                distance_to_center = cdist(clone_contour, [centroid])

                data_row = {
                    "x": float(clone.centroid[1]),
                    "y": float(clone.centroid[0]),
                    "frame": int(t),
                    "size": float(clone.area),
                    "colony_center_x": float(centroid[1]),
                    "colony_center_y": float(centroid[0]),
                    "colony_radius": float(radius),
                    "colony_area": float(colony_area),
                    "distance_to_center": float(cdist([centroid], [clone.centroid])[0][0]),
                    "distance_to_edge": float(distance_to_edge.min()),
                    "max_distance_to_center": float(distance_to_center.max()),
                    "closest_point_clone_x": float(closest_point_clone[1]),
                    "closest_point_clone_y": float(closest_point_clone[0]),
                    "closest_point_colony_x": float(closest_point_colony[1]),
                    "closest_point_colony_y": float(closest_point_colony[0]),
                }

                all_data = pd.concat([all_data, pd.DataFrame([data_row])], ignore_index=True)

        if all_data.empty:
            print(f"[WARN] No clones above search_area found for '{identifier}'. Skipping linking.")
            continue

        # Trackpy linking
        tp.linking.Linker.MAX_SUB_NET_SIZE = 100
        search_range = 20
        memory = 3
        linked = tp.link_df(all_data, search_range, memory=memory)

        # Fusion reassignment (plots disabled for reviewer version)
        linked_after, _reassignment_log = reassign_fusions_by_front(
            linked,
            spatial_threshold=10,
            plot_fusions=False,
        )
        #!!!!
        linked_after = linked.sort_values(["particle", "frame"]).reset_index(drop=True)
        #The data in the manuscript is sorting by linked, which overrides the fusion correction.
        #The correct option is:
        #linked_after = linked_after.sort_values(["particle", "frame"]).reset_index(drop=True)
        #This does not change anything crucial, but is mentioned here for transparency and reproducibility.
        #!!!

        # Optional diagnostic plot: total clone area before/after reassignment
        if False:
            area_before = linked.groupby("frame")["size"].sum()
            area_after = linked_after.groupby("frame")["size"].sum()

            plt.figure(figsize=(6, 3))
            plt.plot(area_before, label="before reassignment", lw=2)
            plt.plot(area_after, label="after reassignment", lw=1)
            plt.legend()
            plt.xlabel("Frame")
            plt.ylabel("Total clone area [px²]")
            plt.tight_layout()
            plt.show()

        linked_after = linked_after.loc[:, ~linked_after.columns.str.contains("^Unnamed")]

        out_csv = os.path.join(output_dir, f"clone_data_fusion_resolved_{identifier}.csv")
        linked_after.to_csv(out_csv, index=False)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    input_root = os.path.join(script_dir, "Input_files")
    output_root = os.path.join(script_dir, "Output_files")

    # One input location for this example (may point to a subfolder inside Input_files/)
    input_relpath = os.path.join("Segmented_Images", "Your_Experiment_folder_folder")  # <-- CHANGE THIS to your folder containing TIFF stacks (can have subfolders if needed)
    folder_path = os.path.join(input_root, input_relpath)

    search_area = 20
    identifiers = ["P2", "P3", "P4", "P5", "P6", "P7", "P8", "P10", "P11", "P12", "P14", "P15", "P16"] # <-- CHANGE THIS to your actual identifiers (e.g. positions) that are part of the TIFF filenames. P1_ is recommended to address P1

    tiff_data_to_csv(folder_path, output_root, identifiers, search_area)


if __name__ == "__main__":
    main()