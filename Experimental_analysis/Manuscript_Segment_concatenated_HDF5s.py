from os import path
import subprocess
import h5py

from czi_segmenter_auguste import identifier


def get_number_of_datasets(path):
    """
    Counts the number of datasets in an HDF5 file.

    Parameters:
    path (str): Path to the HDF5 file.

    Returns:
    int: Number of datasets in the file.
    """
    with h5py.File(path, "r") as h5_file:
        print(f"Number of datasets in {path}: {len(h5_file.keys())}")
        return int(len(h5_file.keys()))

def segment_dataset(h5_path: str ,output_path: str, project_path: str , identifier: str , chosen_scene: str , start_t=0, stop_t=None):
    if chosen_scene is None:
        raise ValueError("No chosen_scene detected.")

    command_list = [
        'C:/Program Files/ilastik-1.4.0.post1-gpu/ilastik.exe',
        '--headless',
        '--output_format=tiff',
        '--export_source=simple segmentation',
    ]

    output_path = [f'--output_filename={output_path}{{nickname}}.tiff']
    project_path = [f'--project={project_path}']
    file_path = f'{h5_path}/'
    file_list = [file_path + str(x) for x in range(start_t, stop_t)]
    complete_list = command_list + project_path + output_path + file_list
    print(f"Running command: {complete_list}")

    subprocess.run(complete_list)

chosen_scenes = ["P2", "P3", "P4", "P5", "P6", "P7", "P8", "P10", "P11", "P12", "P14", "P15", "P16"]
output_path = r"D:\Image_Segmentation\20251205_6_18"
output_path = output_path.replace('\\', '/')
if output_path[-1] != '/':
    output_path = output_path + '/'
project_path = r"D:\Final_ilastik_project\Optimized_project_13042025.ilp"
project_path = project_path.replace('\\', '/')
identifier = "20251205_"

for scene in chosen_scenes:
    segment_dataset(h5_path = fr"D:\Image_Segmentation\20251205_6_18\{scene}_concatenated.h5",
                    project_path = project_path,
                    output_path = output_path,
                    identifier = identifier,
                    chosen_scene = scene,
                    stop_t = get_number_of_datasets(fr"D:\Image_Segmentation\20251205_6_18\{scene}_concatenated.h5"))