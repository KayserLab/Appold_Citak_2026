from aicsimageio import AICSImage
import h5py
import numpy as np
from skimage import io
import dask.array as da
from tqdm import tqdm

from aicsimageio import AICSImage
import h5py
import numpy as np
import dask.array as da
from tqdm import tqdm

def merge_czi_images_to_h5(
    czi_paths: list[str],
    scene: str,
    start_z: int,
    stop_z: int,
    output_h5: str
):
    """
    Reads multiple CZI files (each possibly with a different number of timepoints)
    and writes their Z-max‐projected images for every timepoint into one HDF5,
    naming datasets '0','1','2',… across file boundaries.
    """
    def dask_map_range(x):
        # map the min→0, max→255
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
                raise ValueError(f"stop_z={stop_z} out of range (max {dims['Z'][0]-1}) for {czi_path}")

            for t in tqdm(range(n_t), desc=f"Processing {czi_path}"):
                read_file = img.get_image_dask_data("YXZC", T=t)
                working_array = read_file[:, :, start_z]

                for z in range(start_z, stop_z):
                    working_array = np.maximum(working_array, read_file[:, :, z + 1])

                working_array = working_array.astype(np.uint32)
                scaled_array_g = working_array[:, :, 0].map_blocks(dask_map_range).compute()
                scaled_array_r = working_array[:, :, 1].map_blocks(dask_map_range).compute()
                scaled_array_b = working_array[:, :, 2].map_blocks(dask_map_range).compute()

                # Adjust separate channels and restack, otherwise it will adjust to the brightest channel
                # scaled_array_g = self.func(working_array[:, :, 0], np.min(working_array[:, :, 0]),
                #                            np.max(working_array[:, :, 0]), 0, 255)
                # scaled_array_r = self.func(working_array[:, :, 1], np.min(working_array[:, :, 1]),
                #                            np.max(working_array[:, :, 1]), 0, 255)
                # scaled_array_b = self.func(working_array[:, :, 2], np.min(working_array[:, :, 2]),
                #                            np.max(working_array[:, :, 2]), 0, 255)
                # scaled_array = np.dstack((scaled_array_r, scaled_array_g, scaled_array_b))

                scaled_array = da.dstack((scaled_array_r, scaled_array_g, scaled_array_b))

                # write to HDF5 under the next index
                out_h5.create_dataset(str(global_idx), data=scaled_array)
                global_idx += 1


    print(f"Done: wrote {global_idx} total datasets to '{output_h5}'")


czi_files = [
    r"W:\Nico\20251205_6_18\2025-12-05\20251205_10.5x_yNA16_0BED_30C_1.5x_objective_omniwell_24_hpi-01.czi",
    r"W:\Nico\20251205_6_18\2025-12-07\20251205_10.5x_yNA16_0BED_30C_1.5x_objective_omniwell_24_hpi-01.czi",
    r"W:\Nico\20251205_6_18\2025-12-07\20251205_10.5x_yNA16_0BED_30C_1.5x_objective_omniwell_24_hpi-02.czi",
    r"W:\Nico\20251205_6_18\2025-12-08\20251205_10.5x_yNA16_0BED_30C_1.5x_objective_omniwell_24_hpi-01.czi",
    r"W:\Nico\20251205_6_18\2025-12-09\20251205_10.5x_yNA16_0BED_30C_1.5x_objective_omniwell_24_hpi-01.czi",
    r"W:\Nico\20251205_6_18\2025-12-11\20251205_10.5x_yNA16_0BED_30C_1.5x_objective_omniwell_24_hpi-01.czi"

]

for szene in ["P2", "P3", "P4", "P5", "P6", "P7", "P8", "P10", "P11", "P12", "P14", "P15", "P16"]:
    merge_czi_images_to_h5(
        czi_paths=czi_files,
        scene=szene,
        start_z=0,
        stop_z=14,
        output_h5=fr"D:\Image_Segmentation\20251205_6_18\{szene}_concatenated.h5"
    )
