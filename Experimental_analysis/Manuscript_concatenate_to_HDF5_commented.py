#!/usr/bin/env python3
"""
Needed for experiments where imaging was done in multiple batches (e.g. 2–3 CZI files per scene).
Merge multiple CZI files into a single HDF5 per scene by concatenating timepoints.

For each CZI in `czi_paths`, the script:
- selects a given scene (e.g. "P2")
- iterates through all timepoints in that file
- loads data as YXZC
- computes a Z-max projection between start_z..stop_z
  (inclusive on start, exclusive on stop, exactly as in the original script)
- rescales each channel independently to 0..255 (integer floor division)
- writes each timepoint as one dataset into an HDF5 file with dataset keys "0", "1", "2", ...
  continuing across file boundaries (global index).

Folder structure (relative to this script):
- Input_files/
    - <your_czi_folder>/
        - file1.czi
        - file2.czi
        - ...
- Output_files/
    - P2_concatenated.h5
    - P3_concatenated.h5
    - ...
"""

import os

from aicsimageio import AICSImage
import dask.array as da
import h5py
import numpy as np
from tqdm import tqdm


def merge_czi_images_to_h5(
    czi_paths: list[str],
    scene: str,
    start_z: int,
    stop_z: int,
    output_h5: str,
) -> None:
    """
    Read multiple CZI files and write their Z-max-projected timepoints into a single HDF5 file.

    Parameters
    ----------
    czi_paths : list[str]
        List of CZI file paths. Timepoints from these files are appended in order.
    scene : str
        Scene name in the CZI, e.g. "P2".
    start_z : int
        First z-plane for projection (inclusive).
    stop_z : int
        Last z-plane for projection (exclusive). This matches the original implementation.
        The loop uses z+1 indexing, so the maximum plane accessed is `stop_z`.
    output_h5 : str
        Output path for the concatenated HDF5 file.
    """

    def dask_map_range(x):
        """
        Map min→0 and max→255 using integer floor division.
        This matches the original script's scaling logic.
        """
        return (x - x.min()) * 255 // (x.max() - x.min())

    with h5py.File(output_h5, "w", track_order=True) as out_h5:
        global_idx = 0

        for czi_path in czi_paths:
            img = AICSImage(czi_path)

            if scene not in img.scenes:
                raise ValueError(f"Scene '{scene}' not found in {czi_path}; valid: {img.scenes}")

            img.set_scene(scene)

            dims = img.dims
            n_t = dims["T"][0]

            if stop_z >= dims["Z"][0]:
                raise ValueError(f"stop_z={stop_z} out of range (max {dims['Z'][0] - 1}) for {czi_path}")

            for t in tqdm(range(n_t), desc=f"Processing {os.path.basename(czi_path)}"):
                read_file = img.get_image_dask_data("YXZC", T=t)

                # Z-max projection (logic unchanged from original script)
                working_array = read_file[:, :, start_z]
                for z in range(start_z, stop_z):
                    working_array = np.maximum(working_array, read_file[:, :, z + 1])

                working_array = working_array.astype(np.uint32)

                # Per-channel scaling (logic unchanged from original script)
                scaled_array_g = working_array[:, :, 0].map_blocks(dask_map_range).compute()
                scaled_array_r = working_array[:, :, 1].map_blocks(dask_map_range).compute()
                scaled_array_b = working_array[:, :, 2].map_blocks(dask_map_range).compute()

                # Channel stacking order unchanged: (R, G, B) = (channel 1, channel 0, channel 2)
                scaled_array = da.dstack((scaled_array_r, scaled_array_g, scaled_array_b))

                out_h5.create_dataset(str(global_idx), data=scaled_array)
                global_idx += 1

    print(f"Done: wrote {global_idx} total datasets to '{output_h5}'")


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_root = os.path.join(script_dir, "Input_files")
    output_root = os.path.join(script_dir, "Output_files")
    os.makedirs(output_root, exist_ok=True)

    # CZI files are referenced relative to Input_files/
    czi_relpaths = [
        os.path.join("czi_files", "Your_first_file.czi"),
        os.path.join("czi_files", "Your_second_file.czi"),
        # ... Add all files needed
    ]
    czi_files = [os.path.join(input_root, p).replace("\\", "/") for p in czi_relpaths]

    scenes = ["P2"] #Add all scenes needed, e.g. ["P2", "P3", "P4", ...]. P1_ is recommended for P1

    for scene in scenes:
        output_path = os.path.join(output_root, f"{scene}_concatenated.h5").replace("\\", "/")

        merge_czi_images_to_h5(
            czi_paths=czi_files,
            scene=scene,
            start_z=0,
            stop_z=14,
            output_h5=output_path,
        )


if __name__ == "__main__":
    main()