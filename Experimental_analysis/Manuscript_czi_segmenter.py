from aicsimageio import AICSImage
import h5py
import numpy as np
import subprocess
from skimage import io
import os
from tqdm import tqdm
import dask.array as da


def map_range(x, in_min, in_max, out_min, out_max):
    """
    Maps a value from one range to another
    :param x: Value to be mapped
    :param in_min: Minimum value of the input range
    :param in_max: Maximum value of the input range
    :param out_min: Minimum value of the output range
    :param out_max: Maximum value of the output range
    :return: Mapping of x from the input range to the output range
    """
    return (x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min

# Define the map_range function for Dask array
def dask_map_range(x):
    return map_range(x, x.min(), x.max(), 0, 255)


def check_file_exists(path):
    """
    Checks if a file already exists and asks the user if they want to overwrite it
    :param path:
    :return:
    """
    if os.path.exists(path):
        choice = input("File already exists. Do you want to overwrite it? (y/n): ")
        while choice not in ['y', 'n']:
            print("Invalid choice. Please enter 'y' or 'n'.")
            choice = input("File already exists. Do you want to overwrite it? (y/n): ")
        if choice.lower() == 'y':
            return True
        elif choice.lower() == 'n':
            return False
    else:
        return True


class czi_analyzer:
    """
    Class to analyze czi files

    :param path: Path to the czi file
    :param output_path: Path to the output folder
    :param project_path: Path to the project folder

    """

    def __init__(self, path, output_path, project_path):
        self.czi_path = path
        self.output_path = output_path
        self.project_path = project_path
        self.img = AICSImage(self.czi_path)
        self.scenes = self.img.scenes
        self.dims = self.img.dims
        self.func = np.vectorize(map_range)
        self.identifier = None
        self.chosen_scene = None
        self.start_t = None
        self.stop_t = None
        self.segmentation = None

    def __str__(self):
        """Controls what is returned if object is called as a string"""
        return 'I am a lazy read of your original czi file'

    def get_dataset(self, identifier: str, scene: str, start_t, stop_t, start_z, stop_z, write_h5 = True):
        """
        Get limits of the data to be analyzed and write the stacked image into a hdf5 file
        :param identifier: Set unique identifier for the file
        :param scene: Scene to be analyzed
        :param start_t: First time point to be analyzed
        :param stop_t: Last time point to be analyzed
        :param start_z: First z plane to be stacked
        :param stop_z: Last z plane to be stacked
        :return: hd5f file with the defined dimensions
        """

        if scene not in self.scenes:
            raise ValueError(
                f"Scene input has to be a string and part of the files scenes. Valid inputs are {self.scenes}")

        if start_z > stop_z or stop_z > self.dims['Z'][0]:
            raise ValueError(f"Z input is not in range of 0 - {self.dims['Z'][0] - 1}, or start is higher then stop")

        if start_t > stop_t or stop_t > self.dims['T'][0]:
            raise ValueError(f"T input is not in range of 0 - {self.dims['T'][0] - 1}, or start is higher then stop")
        self.identifier = identifier
        self.chosen_scene = scene
        self.start_t = start_t
        self.stop_t = stop_t
        self.img.set_scene(scene)


        if write_h5:
            #output_path = f"{self.output_path}{identifier}_{scene}_{start_t}_{stop_t}.h5"
            output_path = os.path.join(self.output_path, f"{identifier}_{scene}_{start_t}_{stop_t}.h5")
            if check_file_exists(output_path):
                # Code to write the file or perform other operations
                print(f"Writing file to {output_path}...")
                # Your code here

                processed_image = h5py.File(
                    f"{self.output_path}{identifier}{scene}_{start_t}_{stop_t}.h5", 'w', track_order=True)
                # ToDo: Resolve what happens if file is already open or not closed properly. Code just get stuck. Maybe use
                #  "with XXX as f:"

                for t in tqdm(range(start_t, stop_t + 1)):
                    read_file = self.img.get_image_dask_data("YXZC", T=t)
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
                    #scaled_array = np.dstack((scaled_array_r, scaled_array_g, scaled_array_b))

                    scaled_array = da.dstack((scaled_array_r, scaled_array_g, scaled_array_b))
                    processed_image.create_dataset(str(t), data=scaled_array)
                processed_image.close()

            else:
                print("Operation canceled.")
        else:
            print("Skipping writing file...")
    def segment_dataset(self, h5_path = None):
        if self.chosen_scene is None:
            raise ValueError("No chosen_scene detected. Might have to run get_dataset first!")

        command_list = [
            'C:/Program Files/ilastik-1.4.0.post1-gpu/ilastik.exe',
            '--headless',
            '--output_format=tiff',
            '--export_source=simple segmentation',
        ]
        if h5_path is None:
            output_path = [f'--output_filename={self.output_path}{{nickname}}.tiff']
            project_path = [f'--project={self.project_path}']
            file_path = f'{self.output_path}{self.identifier}{self.chosen_scene}_{self.start_t}_{self.stop_t}.h5/'
            file_list = [file_path + str(x) for x in range(self.start_t, self.stop_t + 1)]
            complete_list = command_list + project_path + output_path + file_list
        else:
            output_path = [f'--output_filename={self.output_path}{{nickname}}.tiff']
            project_path = [f'--project={self.project_path}']
            file_path = f'{h5_path}/'
            file_list = [file_path + str(x) for x in range(self.start_t, self.stop_t + 1)]
            complete_list = command_list + project_path + output_path + file_list

        subprocess.run(complete_list)

    def load_segmentation(self, image):  # ToDo add option for multible inputs
        self.segmentation = io.imread(
            "C:/Users/nappold/PycharmProjects/czi_clone_size_analysis/sample_pictures/P3_30_35-30.tiff")
        self.segmentation = np.pad(image, pad_width=1, mode='constant',
                                   constant_values=3)  # Add background frame around image


def main():
    positions_list =  ["P1","P2", "P3", "P4", "P5", "P6", "P7", "P8"]
    for i in positions_list:
        czi_path = r"W:\Nico\20240227_yNA16_planned_adaptive\2024-02-27\20240227_10.5x_yNA16_0BED_0.5x_objective_omniwell_adaptive_24hpi-01.czi"
        #Change backslashes to forward slashes
        czi_path = czi_path.replace('\\', '/')
        path_output = r"D:\Image_Segmentation\20240227_adaptivetherapy\optimized"
        #Add a forward slash to the end of the path
        if path_output[-1] != '/':
            path_output = path_output + '/'
        #Change backslashes to forward slashes
        path_output = path_output.replace('\\', '/')
        path_project = r"D:\Final_ilastik_project\Optimized_project_13042025.ilp"
        path_project = path_project.replace('\\', '/')
        file_identifier = "20240227_"
        position = i
        start_frame = 0     # should be the frame 24h after the start of the experiment
        stop_frame = 251    # last frame in czi file -1 (for 0 indexing)
        z_start = 0
        z_stop = 14

        analyzer = czi_analyzer(czi_path, path_output, path_project)
        analyzer.get_dataset(file_identifier, position, start_frame, stop_frame, z_start, z_stop, write_h5=True)
        analyzer.segment_dataset()
        print(analyzer.segmentation)

if __name__ == '__main__':
    '''
    czi_path = r"D:/Axiozoom/Nico/20240130yNA16_Adaptive_Therapy/2024-01-30/20240130_10x_yNA16_0.2BED_30C_1.5x_objective_omniwell_pretreatment30C_22.5hpi-01.czi"
    path_output = r"D:/Imaga_Segmentation/20240130adaptivetherapy/"
    path_project = "C:/Users/nappold/Documents/Ilastik/20240108_colony_detection.ilp"
    file_identifier = "Timelapse_"
    start_frame = 0
    stop_frame = 152
    z_start = 0
    z_stop = 3

    h5_files = [f for f in os.listdir(path_output) if f.endswith('.h5')]

    for h5_file in h5_files:
        #extract position from filename
        position = h5_file.split('_')[1]
        analyzer = czi_analyzer(czi_path, path_output, path_project)
        analyzer.get_dataset(file_identifier, position, start_frame, stop_frame, z_start, z_stop, write_h5=False)
        analyzer.segment_dataset(h5_path=os.path.join(path_output, h5_file))



    '''
    # for i in ('P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10'):
    #     czi_path = r"C:\Users\labadmin-guck\Documents\Nico_temp\czi_file\202402027_10.5x_yNA16_0BED_0.5x_objective_omniwell_adaptive_24hpi-01.czi"
    #     #Change backslashes to forward slashes
    #     czi_path = czi_path.replace('\\', '/')
    #     path_output = r"C:\Users\labadmin-guck\Documents\Nico_temp\Segmented_output/"
    #     #Change backslashes to forward slashes
    #     path_output = path_output.replace('\\', '/')
    #     path_project = "C:/Users/labadmin-guck/Documents/Ilastik_Projects/20240227_clone_detection.ilp"
    #     file_identifier = "20240227_adaptive_"
    #     position = i
    #     start_frame = 0
    #     stop_frame = 251
    #     z_start = 0
    #     z_stop = 14
    #
    #     analyzer = czi_analyzer(czi_path, path_output, path_project)
    #     analyzer.get_dataset(file_identifier, position, start_frame, stop_frame, z_start, z_stop)
    #     analyzer.segment_dataset()
    #     print(analyzer.segmentation)
    main()