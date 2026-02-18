import h5py
import numpy as np
import cv2
import os

def in_any_span(frame_idx, spans):
    """spans like [(start, end), ...], inclusive."""
    return any(start <= frame_idx <= end for start, end in spans)

def draw_scale_bar(img_bgr, um_per_px=8.648, bar_um=1000, margin=20, thickness=8,
                   color=(255, 255, 255), text="1 mm"):
    """
    Draw a 1 mm scale bar bottom-right. img_bgr is uint8 (H,W,3).
    """
    h, w = img_bgr.shape[:2]
    bar_px = int(round(bar_um / um_per_px))  # ~116 px for your numbers

    # bottom-right placement
    x2 = w - margin
    y2 = h - margin
    x1 = max(margin, x2 - bar_px)
    y1 = y2 - thickness

    # bar
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, -1)

    # optional label above the bar
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    text_thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, text_thickness)
    tx = x2 - tw
    ty = y1 - 8
    if ty - th < 0:
        ty = y2 + th + 8  # fallback if too close to top
    cv2.putText(img_bgr, text, (tx, ty), font, font_scale, color, text_thickness, cv2.LINE_AA)

    return img_bgr

def extract_and_stack_images(file_path):
    # Open the HDF5 file in read mode
    with h5py.File(file_path, 'r') as f:
        # Get all keys (image names)
        image_keys = list(f.keys())

        # Sort the keys by their numeric order
        sorted_keys = sorted(image_keys, key=lambda x: int(x))

        # Initialize a list to store images
        images = []

        # Loop through sorted keys and extract the images
        for key in sorted_keys:
            img = f[key][:]
            images.append(img)

        # Stack images into a 4D array (time, height, width, channels)
        time_series = np.stack(images, axis=0)

    return time_series


def create_video_from_stack(image_stack, output_path, fps=30):
    # Get dimensions of the images
    num_frames, height, width, channels = image_stack.shape

    # Ensure images are in the correct format (uint8) for video writing
    if image_stack.dtype != np.uint8:
        image_stack = (image_stack * 255).astype(np.uint8)  # Normalizing if required

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Using mp4v codec for .mp4 files
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write each frame to the video file
    for i in range(num_frames):
        video_writer.write(image_stack[i])

    # Release the video writer object
    video_writer.release()
    print(f"Video saved as {output_path}")

def draw_timestamp(img_bgr, frame_idx, minutes_per_frame=30,
                   origin=(20, 45),
                   font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1.0,
                   thickness=2,
                   text_color=(255, 255, 255),
                   bg_color=(0, 0, 0),
                   bg_alpha=0.45,
                   pad=8):
    """
    Draw timestamp as hh:mm (hours can exceed 99) at top-left with a semi-transparent background.
    """
    total_min = int(frame_idx * minutes_per_frame)
    hh = total_min // 60
    mm = total_min % 60
    text = f"{hh:02d}:{mm:02d}"

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin  # y is baseline for putText

    # background rectangle coords
    x1 = x - pad
    y1 = y - th - pad
    x2 = x + tw + pad
    y2 = y + baseline + pad

    # clamp to image bounds
    h, w = img_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    # draw semi-transparent background
    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, bg_alpha, img_bgr, 1 - bg_alpha, 0, img_bgr)

    # draw text
    cv2.putText(img_bgr, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    return img_bgr


def apply_color_map(grayscale_img, color):
    """
    Map a grayscale image to a specific color.

    grayscale_img: 2D array of shape (height, width), dtype uint8
    color: tuple of (B, G, R), values in [0, 255]

    Returns:
    colorized_img: 3D array of shape (height, width, 3), dtype uint8
    """
    # Prepare an empty color image
    colorized_img = np.zeros((grayscale_img.shape[0], grayscale_img.shape[1], 3), dtype=np.uint8)

    # Map the grayscale image to the specific color
    for i in range(3):
        # Multiply the grayscale image by the color component (scaled between 0 and 1)
        colorized_img[:, :, i] = (grayscale_img.astype(np.float32) * (color[i] / 255.0)).astype(np.uint8)

    return colorized_img

def create_colored_video_from_stack(
    image_stack, output_path, fps=30, include_brightfield=True,
    highlight_spans=None,
    um_per_px=8.648, scalebar_um=1000,
    border_thickness=15
):
    num_frames, height, width, channels = image_stack.shape
    if channels != 3:
        raise ValueError("The image stack should have exactly 3 channels.")

    image_stack = image_stack.astype(np.uint8)
    color_images = []

    # border colors in BGR
    BORDER_NORMAL = (0, 0, 0)                 # black
    BORDER_HIGHLIGHT = (191, 191, 191)        # #bfbfbf

    if highlight_spans is None:
        highlight_spans = []

    for i in range(num_frames):
        yellow_channel = image_stack[i, :, :, 1]
        blue_channel   = image_stack[i, :, :, 0]

        if include_brightfield:
            brightfield_channel = image_stack[i, :, :, 2]
        else:
            brightfield_channel = np.zeros_like(yellow_channel)

        # --- your coloring (using your apply_color_map approach) ---
        ROYALBLUE = (int(0.882*255), int(0.41*255), int(0.254*255))   # BGR
        GOLDENROD = (int(0.125*255), int(0.647*255), int(0.855*255))  # BGR

        blue_img = apply_color_map(blue_channel, ROYALBLUE)
        gold_img = apply_color_map(yellow_channel, GOLDENROD)

        if include_brightfield:
            brightfield_img = cv2.cvtColor(brightfield_channel, cv2.COLOR_GRAY2BGR)
        else:
            brightfield_img = np.zeros_like(blue_img)

        combined_img = cv2.addWeighted(gold_img, 1.0, blue_img, 1.0, 0)
        combined_img = cv2.addWeighted(combined_img, 1.0, brightfield_img, 1.0, 0)
        combined_img = np.clip(combined_img, 0, 255).astype(np.uint8)

        # --- timestamp top-left (hh:mm), 1 frame = 30 min ---
        draw_timestamp(
            combined_img,
            frame_idx=i,
            minutes_per_frame=30,
            origin=(20, 55),      # tweak placement
            font_scale=1.0,       # tweak size
            thickness=2
        )

        # --- border frame: black unless in highlight span ---
        highlight_spans_shifted = [(start - 1, end - 1) for start, end in highlight_spans]
        border_color = BORDER_HIGHLIGHT if in_any_span(i, highlight_spans_shifted) else BORDER_NORMAL
        cv2.rectangle(combined_img, (0, 0), (width - 1, height - 1), border_color, border_thickness)

        # --- scalebar bottom-right (1 mm) ---
        draw_scale_bar(
            combined_img,
            um_per_px=um_per_px,
            bar_um=scalebar_um,
            margin=20,
            thickness=8,
            color=(255, 255, 255),
            text="1 mm"
        )

        color_images.append(combined_img)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in color_images:
        video_writer.write(frame)
    video_writer.release()
    print(f"Colored video saved as {output_path}")



# Example usage:
file_path = r"D:\Image_Segmentation\20250909_metr_undertreat" # Replace with your HDF5 file path
file_path = file_path.replace('\\', '/')
if file_path[-1] != '/':
    folder_path = file_path + '/'
#Extract the filename of all hdf5 files in the folder
hdf5_files = [f for f in os.listdir(file_path) if f.endswith('.h5')]
for file in hdf5_files:
    file_path_joined = os.path.join(file_path, file)
    file_save_path = file_path_joined.replace('.h5', '.mp4')

    time_series = extract_and_stack_images(file_path_joined)
    highlight_spans = [[37, 41], [83, 87], [129, 133], [175, 179], [221, 225], [267, 271], [313, 317]]  # Example highlight spans

    create_colored_video_from_stack(
        time_series, file_save_path,
        fps=24,
        include_brightfield=False,
        highlight_spans=highlight_spans,
        um_per_px=8.648,
        scalebar_um=1000,  # 1 mm
        border_thickness=10
    )

# time_series = extract_and_stack_images(file_path)
# output_file = r'E:\Image_Segmentation\20240917_yNA16_continous_dose\20240820_continous_dose_P1_0_334.mp4'  # Output
# # video file name
# fps = 24  # Frames per second (you can adjust this)
# #create_video_from_stack(time_series, output_file, fps)
# create_colored_video_from_stack(time_series, output_file.replace('.mp4', '_colored.mp4'), fps,
#                                 include_brightfield=False)
#
# # Output the shape of the stacked time series
# print(f"Time series shape: {time_series.shape}")