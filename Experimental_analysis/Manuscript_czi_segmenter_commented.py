#!/usr/bin/env python3
"""
CZI → Z-max projection → per-channel scaling → HDF5 export → ilastik segmentation.

This script:
1. Loads a CZI file.
2. Extracts a selected scene.
3. Computes a Z-maximum projection between start_z and stop_z (inclusive).
4. Scales each channel independently to 0–255 using integer floor division.
5. Stores each timepoint as one dataset in an HDF5 file.
6. Runs ilastik headless segmentation on all datasets.
"""

import os
import subprocess

import dask.array as da
import h5py
import numpy as np
from aicsimageio import AICSImage
from tqdm import tqdm


# ──────────────────────────────────────────────
# Scaling logic
# ──────────────────────────────────────────────
def map_range(x, in_min, in_max, out_min, out_max):
    """
    Map values from one range to another using integer floor division.
    """
    return (x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min


def dask_map_range(x):
    """Dask-compatible wrapper for map_range."""
    return map_range(x, x.min(), x.max(), 0, 255)


# ──────────────────────────────────────────────
# CZI Analyzer Class
# ──────────────────────────────────────────────
class czi_analyzer:
    """Analyzer for exporting CZI scenes to HDF5 and segmenting with ilastik."""

    def __init__(self, path, output_path, project_path):
        self.czi_path = path
        self.output_path = output_path
        self.project_path = project_path

        self.img = AICSImage(self.czi_path)
        self.scenes = self.img.scenes
        self.dims = self.img.dims

        self.identifier = None
        self.chosen_scene = None
        self.start_t = None
        self.stop_t = None

    def get_dataset(self, identifier, scene, start_t, stop_t, start_z, stop_z, write_h5=True):
        """
        Export selected scene/time window into HDF5.

        Parameters
        ----------
        identifier : str
            Prefix for output file.
        scene : str
            Scene name (e.g. "P2").
        start_t, stop_t : int
            Timepoint range (inclusive).
        start_z, stop_z : int
            Z range for projection (inclusive).
        """
        if scene not in self.scenes:
            raise ValueError(f"Scene must be one of {self.scenes}")

        if start_z > stop_z or stop_z > self.dims["Z"][0]:
            raise ValueError("Invalid Z range.")

        if start_t > stop_t or stop_t > self.dims["T"][0]:
            raise ValueError("Invalid T range.")

        self.identifier = identifier
        self.chosen_scene = scene
        self.start_t = start_t
        self.stop_t = stop_t

        self.img.set_scene(scene)

        if not write_h5:
            return

        os.makedirs(self.output_path, exist_ok=True)

        output_file = os.path.join(
            self.output_path,
            f"{identifier}_{scene}_{start_t}_{stop_t}.h5",
        )

        processed_image = h5py.File(output_file, "w", track_order=True)

        for t in tqdm(range(start_t, stop_t + 1)):
            read_file = self.img.get_image_dask_data("YXZC", T=t)

            # Z-max projection (unchanged)
            working_array = read_file[:, :, start_z]
            for z in range(start_z, stop_z):
                working_array = np.maximum(
                    working_array,
                    read_file[:, :, z + 1],
                )

            working_array = working_array.astype(np.uint32)

            # Per-channel scaling (unchanged)
            scaled_array_g = working_array[:, :, 0].map_blocks(dask_map_range).compute()
            scaled_array_r = working_array[:, :, 1].map_blocks(dask_map_range).compute()
            scaled_array_b = working_array[:, :, 2].map_blocks(dask_map_range).compute()

            # Channel stacking order preserved: (R, G, B) = (channel 1, channel 0, channel 2)
            scaled_array = da.dstack((
                scaled_array_r,
                scaled_array_g,
                scaled_array_b,
            ))

            processed_image.create_dataset(str(t), data=scaled_array)

        processed_image.close()

    def segment_dataset(self, h5_path=None):
        """Run ilastik headless segmentation."""
        if self.chosen_scene is None:
            raise ValueError("Call get_dataset() first.")

        command_list = [
            "C:/Program Files/ilastik-1.4.0.post1-gpu/ilastik.exe",
            "--headless",
            "--output_format=tiff",
            "--export_source=simple segmentation",
        ]

        output_flag = [f"--output_filename={self.output_path}/{{nickname}}.tiff"]
        project_flag = [f"--project={self.project_path}"]

        if h5_path is None:
            file_path = f"{self.output_path}/{self.identifier}_{self.chosen_scene}_{self.start_t}_{self.stop_t}.h5/"
        else:
            file_path = f"{h5_path}/"

        file_list = [file_path + str(x) for x in range(self.start_t, self.stop_t + 1)]
        complete_list = command_list + project_flag + output_flag + file_list

        subprocess.run(complete_list)


# ──────────────────────────────────────────────
# Main Execution
# ──────────────────────────────────────────────
def main():
    # Project-relative IO roots
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_root = os.path.join(script_dir, "Input_files")
    output_dir = os.path.join(script_dir, "Output_files")

    # Input CZI path inside Input_files/
    czi_relpath = os.path.join(
        "czi_files",
        "name_of_your_file.czi", # Change to your file name
    )
    czi_path = os.path.join(input_root, czi_relpath).replace("\\", "/")

    # Ilastik project path (stored inside Input_files/ for portability)
    project_relpath = os.path.join("Ilastik_projects", "General_optimized_project.ilp")
    project_path = os.path.join(input_root, project_relpath).replace("\\", "/")

    output_dir = output_dir.replace("\\", "/")
    os.makedirs(output_dir, exist_ok=True)

    identifier = "Test"
    start_frame = 0
    stop_frame = 51 # Adjust based on your CZI's T dimension (0-based indexing)
    z_start = 0
    z_stop = 14

    analyzer = czi_analyzer(czi_path, output_dir, project_path)

    analyzer.get_dataset(
        identifier,
        scene="P1_", # Adjust based on your CZI's scene names. Recommended --> Use underscore for P1
        start_t=start_frame,
        stop_t=stop_frame,
        start_z=z_start,
        stop_z=z_stop,
        write_h5=True,
    )

    analyzer.segment_dataset()


if __name__ == "__main__":
    main()