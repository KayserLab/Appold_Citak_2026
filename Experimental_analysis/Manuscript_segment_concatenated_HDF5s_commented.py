#!/usr/bin/env python3
"""
Run ilastik headless segmentation on HDF5 time series stacks.

This script reads one HDF5 file per scene (e.g. "P2_concatenated.h5") and calls ilastik
in headless mode to export the "simple segmentation" result as TIFF images.

Folder structure (next to this script):
- Input_files/
    - <input_relpath>/
        - P2_concatenated.h5
        - P3_concatenated.h5
        - ...
- Output_files/
    - (ilastik output TIFFs will be written here)

Assumptions:
- The HDF5 contains datasets keyed by integers as strings ("0", "1", "2", ...).
- Each dataset corresponds to one timepoint.
"""

from __future__ import annotations

import os
import subprocess

import h5py


def get_number_of_datasets(h5_path: str) -> int:
    """
    Count datasets in an HDF5 file.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file.

    Returns
    -------
    int
        Number of datasets in the file.
    """
    with h5py.File(h5_path, "r") as h5_file:
        return int(len(h5_file.keys()))


def segment_dataset(h5_filename: str, input_dir: str, output_dir: str, project_path: str,
                    start_t: int = 0, stop_t: int | None = None) -> None:
    """
    Run ilastik headless on all datasets in one HDF5, using relative paths to avoid WinError 206.

    h5_filename: e.g. "P2_concatenated.h5" (NOT a full path)
    input_dir: directory where the H5 files live (used as cwd)
    output_dir: where ilastik writes TIFFs
    project_path: .ilp project (can stay absolute)
    """
    os.makedirs(output_dir, exist_ok=True)

    h5_path_abs = os.path.join(input_dir, h5_filename)
    if stop_t is None:
        stop_t = get_number_of_datasets(h5_path_abs)

    ilastik_exe = r"C:\Program Files\ilastik-1.4.0.post1-gpu\ilastik.exe"

    # IMPORTANT: relative dataset paths, because we set cwd=input_dir
    file_list = [f"{h5_filename}/{t}" for t in range(start_t, stop_t)]

    cmd = [
        ilastik_exe,
        "--headless",
        "--output_format=tiff",
        "--export_source=simple segmentation",
        f"--project={project_path}",
        f"--output_filename={os.path.join(output_dir, '{nickname}.tiff')}",
        *file_list,
    ]

    subprocess.run(cmd, check=False, cwd=input_dir)


def main() -> None:
    scenes = ["P2", "P3", "P4", "P5", "P6", "P7", "P8", "P10", "P11", "P12", "P14", "P15", "P16"]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "Input_files", "HDF5_stack")
    output_dir = os.path.join(script_dir, "Output_files")

    project_path = r"D:\Final_ilastik_project\Optimized_project_13042025.ilp"

    for scene in scenes:
        h5_filename = f"{scene}_concatenated.h5"
        segment_dataset(
            h5_filename=h5_filename,
            input_dir=input_dir,
            output_dir=output_dir,
            project_path=project_path,
            start_t=0,
            stop_t=None,
        )

if __name__ == "__main__":
    main()