import os
import re
import numpy as np
from skimage import io
from skimage import measure
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import trackpy as tp
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import json


def colony_parameters(array):
    """
    Find the largest connected component in a binary image
    :param array: Binary image
    :return: Binary image with only the largest connected component
    """
    # Reduce 2s to 1s
    array = np.where(array == 2, 1, array)

    # Find connected components in the image
    labeled_colony = measure.label(array, background=3)

    detected_components = measure.regionprops(labeled_colony)

    # Extract the biggest component
    biggest_component = max(detected_components, key=lambda x: x.area)
    largest_label = biggest_component.label

    # Step 3: Extract the contour of the largest component
    biggest_component_mask = labeled_colony == largest_label
    colony_contour = measure.find_contours(biggest_component_mask, level=0.5)

    # Note that find_contours might return multiple contours if there are holes in the component.
    # You can take the longest contour if that's suitable for your case:
    largest_contour = max(colony_contour, key=len)

    #get the radius of the contour
    colony_radius = np.mean(cdist(largest_contour, [biggest_component.centroid]))

    return biggest_component.area, largest_contour, biggest_component.centroid, colony_radius

def get_number(filename):
    """
    Extracts a numerical identifier from a filename. This function assumes the filename
    contains a specific pattern of numbers that can be uniquely identified as an identifier.

    :param filename: The name of the file from which to extract the number.
    :return: The extracted numerical identifier as an integer. If no number can be extracted,
             the function returns None. It is assumed that the caller will handle this
             gracefully.
    """
    match = re.search(r'(\d+)\.tiff$', filename)
    return int(match.group(1)) if match else 0

def tiff_data_to_colony_size(folder_path, identifiers):
    """
    Iterates through all TIFF files in the folder, extracts the data, and saves it as a CSV file.
    :param folder_path: Path to the folder containing the TIFF files.
    :param identifiers: List of identifiers to be used to filter the files.
    """
    # Iterate through each identifier
    for identifier in tqdm(identifiers, desc="Position"):

        tiff_files = [f for f in os.listdir(folder_path) if f.endswith('.tiff') and identifier in f]


        # Check if there are any TIFF files for the current identifier
        if not tiff_files:
            print(f"No TIFF files found for identifier: {identifier}")
            continue
        # Sort files based on the number extracted from the filename
        tiff_files.sort(key=get_number)

        # Initialize a list to hold image data
        images = []

        # Load each TIFF file using imageio
        for file in tqdm(tiff_files, desc="Processing files"):
            file_path = os.path.join(folder_path, file)
            image = io.imread(file_path)
            # pad image by adding a row of threes to top bottom left and right
            image = np.pad(image, 1, constant_values=3)

            images.append(image)

        # Combine all images into a single NumPy array
        combined_array = np.stack(images)

        colony_Data = pd.DataFrame(columns=['colony_center', 'colony_radius',
                                         'colony_contour'])

        for t in tqdm(range(combined_array.shape[0]), desc="Processing time points"):
            # Iterate through each time point and find the largest area
            largest_areas = colony_parameters(combined_array[t])[0]
            contour = colony_parameters(combined_array[t])[1]
            colony_contour_json = json.dumps(contour.tolist())
            centroid = colony_parameters(combined_array[t])[2]
            radius = colony_parameters(combined_array[t])[3]

            colony_Data = pd.concat([colony_Data, pd.DataFrame([[centroid, radius, colony_contour_json, largest_areas]], columns=['colony_center', 'colony_radius', 'colony_contour', 'colony_area'])])
            colony_Data.to_csv(folder_path + '/colony_data_' + identifier +
                          '.csv')

    return colony_Data

folder_path = r"D:\Image_Segmentation\20240227_adaptivetherapy\optimized"
folder_path = folder_path.replace('\\', '/')
identifiers = ["P1","P2", "P3", "P4", "P5", "P6", "P7", "P8"]

colony_Data = tiff_data_to_colony_size(folder_path, identifiers)


