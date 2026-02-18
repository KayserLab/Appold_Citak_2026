import os
import re
import numpy as np
from scipy.stats import alpha
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

    return biggest_component.area, largest_contour, biggest_component.centroid, colony_radius, biggest_component_mask


# Regular expression to find the number before ".tiff"
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


def get_clones(array):
    """Get all connected components of 1s in the array timeseries and get their centroid size
    Analyzes an image represented as an array to find all connected components that represent clones,
    calculating the centroid and size for each component. This function is particularly useful for
    biological or colony analysis where each clone is isolated and needs to be quantified in terms of
    its position and area.

    The function assumes the input array is a binary or labeled image where the background is represented
    by 0s and the clones or components of interest are represented by 1s or unique labels for different
    clones.

    Parameters:
    - array (numpy.ndarray): A 2D array representing a binary or labeled image. Clone components should be
      marked with 1s or unique labels in a labeled image.

    Returns:
    - list of dicts: A list where each element is a dictionary containing information about a single clone.
      Each dictionary includes keys such as 'centroid' (the central point of the clone) and 'size' (the
      number of pixels or area of the clone). The exact structure and content of this list may vary based
      on the implementation details, such as whether additional properties of clones are calculated (e.g.,
      perimeter, circularity).

    Example:
    Given a binary array representing an image with several distinct connected components (clones),
    `get_clones` will return a list of dictionaries, each representing a clone with its centroid and size.

    Note:
    This function requires the input array to be pre-processed if necessary, to ensure that it is suitable
    for connected component analysis. This might involve thresholding, noise reduction, or other image
    preprocessing steps not covered by this function.

    Raises:
    - ValueError: If the input array is not 2D or does not meet expected criteria for analysis.
    """

    # Reduce 2s to 3s to get rid of red cells
    array = np.where(array == 2, 3, array)
    # Check if there are any 1s (green pixels) in the array
    if 1 in array:
        # Find connected components in the image
        labeled_clone = measure.label(array, background=3)
        clone_props = measure.regionprops(labeled_clone)
    else:
        clone_props = None
        labeled_clone = None
    #Get Number of Pixels with value one
    green_pixels = np.sum(array == 1)
    print(green_pixels)

    return clone_props, labeled_clone




# Create function to iterate through all files in the folder extract the data and save it as a csv
def tiff_data_to_csv(folder_path, identifiers, search_area):
    """
    Iterates through all TIFF files in the folder, extracts the data, and saves it as a CSV file.
    :param folder_path: Path to the folder containing the TIFF files.
    :param identifiers: List of identifiers to be used to filter the files.
    :param search_area: Minimum area of a clone to be considered.
    """
    # Iterate through each identifier


    for identifier in tqdm(identifiers, desc="Position"):

        tiff_files = [f for f in os.listdir(folder_path) if f.endswith('.tiff') and identifier in f]

        # Skip identifier if no TIFF files found
        if not tiff_files:
            print(f"⚠️  No TIFF files found for identifier '{identifier}' in folder '{folder_path}'. Skipping.")
            continue

        # Sort files based on the number extracted from the filename
        tiff_files.sort(key=get_number)

        # Initialize a list to hold image data
        images = []

        # Load each TIFF file using imageio
        for file in tiff_files:
            file_path = os.path.join(folder_path, file)
            image = io.imread(file_path)
            #pad image by adding a row of threes to top bottom left and right
            image = np.pad(image, 1, constant_values=3)

            images.append(image)

        # Combine all images into a single NumPy array
        combined_array = np.stack(images)


        all_Data = pd.DataFrame()

        for t in tqdm(range(combined_array.shape[0]), desc ="Processing files"):
            # Iterate through each time point and find the largest area
            largest_areas = colony_parameters(combined_array[t])[0]
            print(largest_areas)
            contour = colony_parameters(combined_array[t])[1]
            #colony_contour_json = json.dumps(contour.tolist())
            centroid = colony_parameters(combined_array[t])[2]
            radius = colony_parameters(combined_array[t])[3]
            colony_mask = colony_parameters(combined_array[t])[4]
            masked_array = np.where(colony_mask == False, 3, combined_array[t])

            detected_clones = get_clones(masked_array)[0]
            labeled_image = get_clones(masked_array)[1]
            #plt.imshow(masked_array, cmap='gray')
            #plt.show()

            # Initialize counters
            total_clones_in_frame = 0
            clones_bigger_than_search_area = 0

            # Get the x and y coordinates, the centroid and the averaged radius of the contour for each clone
            if detected_clones is not None:

                for clone in detected_clones:
                    total_clones_in_frame += 1  # Increment total clones counter
                    if clone.area > search_area:
                        clones_bigger_than_search_area += 1  # Increment clones bigger than search area
                        # Get contour of the clone
                        label = clone.label
                        clone_mask = labeled_image == label
                        clone_contour = measure.find_contours(clone_mask, level=0.8)
                        clone_contour = max(clone_contour, key=len)

                        # Calculate the pairwise distances between the clone and colony contours
                        distance_to_edge = cdist(clone_contour, contour)

                        # Find the indices of the minimum distance
                        min_idx = np.unravel_index(np.argmin(distance_to_edge), distance_to_edge.shape)

                        # Extract the coordinates of the closest points
                        closest_point_clone = clone_contour[min_idx[0]]  # (row, col) format
                        closest_point_colony = contour[min_idx[1]]  # (row, col) format

                        # Calculate distance to colony center
                        distance_to_center = cdist(clone_contour, [centroid])

                        # Prepare data for the DataFrame
                        data_row = {
                            'x': clone.centroid[1],
                            'y': clone.centroid[0],
                            'frame': t,
                            'size': clone.area,
                            'colony_center_x': centroid[1],
                            'colony_center_y': centroid[0],
                            'colony_radius': radius,
                            'colony_area': largest_areas,
                            'distance_to_center': cdist([centroid], [clone.centroid])[0][0],
                            'distance_to_edge': distance_to_edge.min(),
                            'max_distance_to_center': distance_to_center.max(),
                            'closest_point_clone_x': closest_point_clone[1],
                            'closest_point_clone_y': closest_point_clone[0],
                            'closest_point_colony_x': closest_point_colony[1],
                            'closest_point_colony_y': closest_point_colony[0],
                        }

                        # Add the data to your DataFrame
                        all_Data = pd.concat([all_Data, pd.DataFrame([data_row])], ignore_index=True)

                    # Print the number of clones detected
                print(
                    f'Detected {total_clones_in_frame} clones in {identifier} frame {t} and Position {identifier}')
                print(
                    f'Detected {clones_bigger_than_search_area} clones bigger than {search_area} pixels in frame: {t}')
            else:
                print(f'No clones detected in frame: {t}')

        # Define parameters for trackpy
        tp.linking.Linker.MAX_SUB_NET_SIZE = 100  # increase limit from default 30
        search_range = 20 #was 25
        memory = 3
        # Link the clones
        linked = tp.link_df(all_Data, search_range, memory=memory)

        linked_before = linked.copy()
        # Resolve fusions by front
        linked_after, reassignment_log = reassign_fusions_by_front(
            linked,
            spatial_threshold=10,
            plot_fusions=ENABLE_PLOTTING,
            image_loader=image_loader
        )

        # Sort the data by particle and frame
        linked_after = linked.sort_values(by=['particle', 'frame'])

        print(f"Reassigned {len(reassignment_log)} potential fusion events.")
        for event in reassignment_log[:5]:
            print(event)

        area_before = linked_before.groupby('frame')['size'].sum()
        area_after = linked_after.groupby('frame')['size'].sum()

        plt.figure(figsize=(6, 3))
        plt.plot(area_before, label='before reassignment', alpha=1, lw=2)
        plt.plot(area_after, label='after reassignment', lw=1, alpha=1)
        plt.legend()
        plt.xlabel('Frame')
        plt.ylabel('Total clone area [px²]')
        plt.tight_layout()
        plt.show()

        linked = linked_after

        #drop ('^Unnamed') first column
        linked = linked.loc[:, ~linked.columns.str.contains('^Unnamed')]

        #Save Dataframe as csv
        linked.to_csv(folder_path + '/clone_data_fusion_resolved_' + identifier +
                      '.csv')

    return linked

def read_and_adjust_csv(file_path, particle_offset):
    """
    Reads a CSV file, adjusts the 'particle' column, and returns the DataFrame.
    :param file_path: Path to the CSV file.
    :param particle_offset: The offset to be added to the 'particle' column.
    :return: Adjusted DataFrame.
    """
    df = pd.read_csv(file_path)
    df['particle'] += particle_offset
    return df


# Function to concatenate all csv files into one dataframe
def concat_csv(csv_list):
    """
    Concatenates multiple CSV files into a single DataFrame.
    :param csv_list: List of CSV files to be concatenated.
    :return: Concatenated DataFrame.
    """
    concat_data = pd.DataFrame()
    max_particle_number = 0

    # Process each file
    for file in csv_list:
        df = read_and_adjust_csv(os.path.join(folder_path, file[0]), max_particle_number)
        max_particle_number = df['particle'].max() + 1
        concat_data = pd.concat([concat_data, df], ignore_index=True)

    return concat_data


def reassign_fusions_by_front(linked_df, spatial_threshold=10, plot_fusions=False, image_loader=None):
    """
    Reassigns track identities after trackpy linking so that when two clones come close
    (possible fusion), the *front-most* clone (smallest distance_to_edge) keeps the ID.
    No data are deleted; only particle labels are reassigned.

    Parameters
    ----------
    linked_df : pd.DataFrame
        DataFrame from trackpy.link_df(), must contain ['frame', 'particle', 'x', 'y', 'distance_to_edge'].
    spatial_threshold : float, optional
        Distance (in pixels or µm) below which two clones are considered candidates for fusion.
    plot_fusions : bool, optional
        If True, visualize reassignment events using image_loader.
    image_loader : callable, optional
        Function(frame:int) -> np.ndarray for debug plotting.

    Returns
    -------
    reassigned_df : pd.DataFrame
        DataFrame where particles have been relabeled so the front-most clone continues.
    reassignment_log : list of dict
        Log of reassignment events.
    """
    from scipy.spatial.distance import cdist
    import matplotlib.pyplot as plt
    import numpy as np

    reassigned_df = linked_df.copy()
    reassignment_log = []

    # loop through frames
    for frame in reassigned_df['frame'].unique():
        frame_data = reassigned_df[reassigned_df['frame'] == frame]
        coords = frame_data[['x', 'y']].values
        particles = frame_data['particle'].values

        if len(coords) < 2:
            continue

        distances = cdist(coords, coords)

        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                pid1, pid2 = particles[i], particles[j]
                if pid1 == pid2:
                    continue

                if distances[i, j] < spatial_threshold:
                    d1_row = frame_data[frame_data['particle'] == pid1].iloc[0]
                    d2_row = frame_data[frame_data['particle'] == pid2].iloc[0]

                    # Pick the one closer to the front
                    to_keep = pid1 if d1_row['distance_to_edge'] < d2_row['distance_to_edge'] else pid2
                    to_relabel = pid2 if to_keep == pid1 else pid1

                    # Reassign: from this frame onward, use front-most particle ID
                    mask = (reassigned_df['particle'] == to_relabel) & (reassigned_df['frame'] >= frame)
                    reassigned_df.loc[mask, 'particle'] = to_keep

                    reassignment_log.append({
                        "frame": frame,
                        "kept_particle": int(to_keep),
                        "relabelled_particle": int(to_relabel),
                        "distance_between_centroids": float(distances[i, j]),
                        "front_distance_keep": float(min(d1_row['distance_to_edge'], d2_row['distance_to_edge'])),
                        "front_distance_relabelled": float(max(d1_row['distance_to_edge'], d2_row['distance_to_edge'])),
                    })

                    if plot_fusions and image_loader is not None:
                        img = image_loader(frame)
                        if img is not None:
                            fig, ax = plt.subplots(figsize=(7, 7))
                            ax.imshow(img, cmap='gray', vmin=1, vmax=3)
                            ax.plot(d1_row['x'], d1_row['y'], 'o', color='limegreen', label=f'Keep {to_keep}')
                            ax.plot(d2_row['x'], d2_row['y'], 'o', color='crimson', label=f'Relabel {to_relabel}')
                            ax.set_title(f"Frame {frame} – Reassign {to_relabel} → {to_keep}")
                            ax.legend()
                            plt.axis("off")
                            plt.tight_layout()
                            plt.show()

    reassigned_df = reassigned_df.sort_values(['particle', 'frame']).reset_index(drop=True)
    return reassigned_df, reassignment_log


def image_loader(frame):
    if ENABLE_PLOTTING and 100 <= frame <= 350:
        path = f"D:/Image_Segmentation/20240917_yNA16_continous_dose/optimized/continous_dose_P1_0_334-{frame}.tiff"
        try:
            return io.imread(path)
        except FileNotFoundError:
            print(f"⚠️  Image for frame {frame} not found.")
            return None
    return None

# Path to your folder containing TIFF images
folder_paths = {#r"D:\Image_Segmentation\20240917_yNA16_continous_dose\optimized",
                #r"D:\Image_Segmentation\20251024_no_treatment",
                #r"D:\Image_Segmentation\20241210_pulse_treatment\optimized",
                #r"D:\Image_Segmentation\20250902_metronomic_overtreat",
                #r"D:\Image_Segmentation\20250909_metr_undertreat",
                #r"D:\Image_Segmentation\20250930_metronomic_6_18",
                #r"D:\Image_Segmentation\20250826_metronomic_11_21",
                #r"D:\Image_Segmentation\20251007_7_18",
                #r"D:\Image_Segmentation\20251114_9_18",
                #r"D:\Image_Segmentation\9_18_finals",
                #r"D:\Image_Segmentation\20251121_4_18",
                #r"D:\Image_Segmentation\20251121_4_18_finals",
                #r"D:\Image_Segmentation\20251205_6_18",
    #r"D:\Image_Segmentation\20251205_6_18_finals"
                #r"D:\Image_Segmentation\20240917_Continuous_final",
r"D:\Image_Segmentation\20240227_adaptivetherapy\optimized"

                }

for folder_path in folder_paths:
    folder_path = folder_path.replace('\\', '/')
    search_area = 20 #Was 20 for older files
    identifiers = ["P1","P2", "P3", "P4", "P5", "P6", "P7", "P8"] # List of identifiers to filter files
    #['P1_','P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', '10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19']
    ENABLE_PLOTTING = False  # Set to False to disable all fusion plots

    linked = tiff_data_to_csv(folder_path, identifiers, search_area)