#!/usr/bin/env python3
"""
Create MP4 videos from HDF5 image stacks.

This script expects a folder structure next to the script:
- Input_files/
    - <input_relpath>/           (contains one or more .h5 files)
- Output_files/
    - (videos will be written here)

Each HDF5 file is assumed to contain a time series of RGB images stored as datasets with
integer keys ("0", "1", "2", ...). Each dataset must have shape (H, W, 3) and dtype uint8.

For every HDF5 file found in the input folder, the script:
1) Loads all datasets and stacks them into a 4D array: (T, H, W, 3)
2) Creates a colored MP4 video with:
   - royalblue overlay for channel 0
   - goldenrod overlay for channel 1
   - optional brightfield channel from channel 2
   - a timestamp (hh:mm) assuming 30 minutes per frame
   - a border that turns grey during specified highlight spans
   - a 1 mm scale bar in the bottom-right corner
"""

import os
import h5py
import numpy as np
import cv2


def in_any_span(frame_idx: int, spans: list[tuple[int, int]]) -> bool:
    """
    Return True if frame_idx lies in any (start, end) span (inclusive).
    """
    return any(start <= frame_idx <= end for start, end in spans)


def draw_scale_bar(
    img_bgr: np.ndarray,
    um_per_px: float = 8.648,
    bar_um: float = 1000,
    margin: int = 20,
    thickness: int = 8,
    color: tuple[int, int, int] = (255, 255, 255),
    text: str = "1 mm",
) -> np.ndarray:
    """
    Draw a scale bar in the bottom-right corner.

    Parameters
    ----------
    img_bgr : np.ndarray
        Image in BGR order, dtype uint8, shape (H, W, 3).
    um_per_px : float
        Micrometers per pixel.
    bar_um : float
        Scale bar length in micrometers (1000 µm = 1 mm).
    margin : int
        Margin (px) from the image edge.
    thickness : int
        Bar thickness in pixels.
    color : tuple[int, int, int]
        BGR color for bar and text.
    text : str
        Label to draw next to the scale bar.
    """
    h, w = img_bgr.shape[:2]
    bar_px = int(round(bar_um / um_per_px))

    x2 = w - margin
    y2 = h - margin
    x1 = max(margin, x2 - bar_px)
    y1 = y2 - thickness

    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    text_thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)

    tx = x2 - tw
    ty = y1 - 8
    if ty - th < 0:
        ty = y2 + th + baseline + 8

    cv2.putText(img_bgr, text, (tx, ty), font, font_scale, color, text_thickness, cv2.LINE_AA)
    return img_bgr


def extract_and_stack_images(file_path: str) -> np.ndarray:
    """
    Load all datasets from an HDF5 file and stack them to shape (T, H, W, 3).
    Dataset keys are expected to be integers stored as strings ("0","1","2",...).
    """
    with h5py.File(file_path, "r") as f:
        keys = sorted(list(f.keys()), key=lambda x: int(x))
        images = [f[k][:] for k in keys]
    return np.stack(images, axis=0)


def create_video_from_stack(image_stack: np.ndarray, output_path: str, fps: int = 30) -> None:
    """
    Write a raw MP4 video from a stack of frames (T, H, W, 3), dtype uint8.
    """
    num_frames, height, width, channels = image_stack.shape
    if channels != 3:
        raise ValueError("Expected 3 channels (RGB/BGR) per frame.")

    if image_stack.dtype != np.uint8:
        image_stack = (image_stack * 255).astype(np.uint8)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(num_frames):
        video_writer.write(image_stack[i])

    video_writer.release()


def draw_timestamp(
    img_bgr: np.ndarray,
    frame_idx: int,
    minutes_per_frame: int = 30,
    origin: tuple[int, int] = (20, 45),
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 1.0,
    thickness: int = 2,
    text_color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (0, 0, 0),
    bg_alpha: float = 0.45,
    pad: int = 8,
) -> np.ndarray:
    """
    Draw timestamp as hh:mm at top-left with a semi-transparent background.
    """
    total_min = int(frame_idx * minutes_per_frame)
    hh = total_min // 60
    mm = total_min % 60
    text = f"{hh:02d}:{mm:02d}"

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin

    x1 = x - pad
    y1 = y - th - pad
    x2 = x + tw + pad
    y2 = y + baseline + pad

    h, w = img_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, bg_alpha, img_bgr, 1 - bg_alpha, 0, img_bgr)

    cv2.putText(img_bgr, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    return img_bgr


def apply_color_map(grayscale_img: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    """
    Map a grayscale image to a single BGR color.

    Parameters
    ----------
    grayscale_img : np.ndarray
        2D image, dtype uint8.
    color : tuple[int, int, int]
        Color in BGR order, range 0..255.

    Returns
    -------
    np.ndarray
        Colorized image (H, W, 3), dtype uint8.
    """
    colorized = np.zeros((grayscale_img.shape[0], grayscale_img.shape[1], 3), dtype=np.uint8)
    for i in range(3):
        colorized[:, :, i] = (grayscale_img.astype(np.float32) * (color[i] / 255.0)).astype(np.uint8)
    return colorized


def create_colored_video_from_stack(
    image_stack: np.ndarray,
    output_path: str,
    fps: int = 30,
    include_brightfield: bool = True,
    highlight_spans: list[list[int]] | None = None,
    um_per_px: float = 8.648,
    scalebar_um: float = 1000,
    border_thickness: int = 15,
) -> None:
    """
    Create a colored MP4 from an image stack.

    The stack is expected to have 3 channels:
    - channel 0: "blue" label map
    - channel 1: "yellow" label map
    - channel 2: brightfield (optional)
    """
    num_frames, height, width, channels = image_stack.shape
    if channels != 3:
        raise ValueError("The image stack must have exactly 3 channels per frame.")

    image_stack = image_stack.astype(np.uint8)

    BORDER_NORMAL = (0, 0, 0)          # black
    BORDER_HIGHLIGHT = (191, 191, 191) # grey (#bfbfbf)

    if highlight_spans is None:
        highlight_spans = []

    # Convert provided spans from 1-based to 0-based, as in the original code
    highlight_spans_shifted = [(start - 1, end - 1) for start, end in highlight_spans]

    # Colors in BGR (as in the original code)
    ROYALBLUE = (int(0.882 * 255), int(0.41 * 255), int(0.254 * 255))
    GOLDENROD = (int(0.125 * 255), int(0.647 * 255), int(0.855 * 255))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(num_frames):
        yellow_channel = image_stack[i, :, :, 1]
        blue_channel = image_stack[i, :, :, 0]

        if include_brightfield:
            brightfield_channel = image_stack[i, :, :, 2]
        else:
            brightfield_channel = np.zeros_like(yellow_channel)

        blue_img = apply_color_map(blue_channel, ROYALBLUE)
        gold_img = apply_color_map(yellow_channel, GOLDENROD)

        if include_brightfield:
            brightfield_img = cv2.cvtColor(brightfield_channel, cv2.COLOR_GRAY2BGR)
        else:
            brightfield_img = np.zeros_like(blue_img)

        combined = cv2.addWeighted(gold_img, 1.0, blue_img, 1.0, 0)
        combined = cv2.addWeighted(combined, 1.0, brightfield_img, 1.0, 0)
        combined = np.clip(combined, 0, 255).astype(np.uint8)

        draw_timestamp(
            combined,
            frame_idx=i,
            minutes_per_frame=30,
            origin=(20, 55),
            font_scale=1.0,
            thickness=2,
        )

        border_color = BORDER_HIGHLIGHT if in_any_span(i, highlight_spans_shifted) else BORDER_NORMAL
        cv2.rectangle(combined, (0, 0), (width - 1, height - 1), border_color, border_thickness)

        draw_scale_bar(
            combined,
            um_per_px=um_per_px,
            bar_um=scalebar_um,
            margin=20,
            thickness=8,
            color=(255, 255, 255),
            text="1 mm",
        )

        video_writer.write(combined)

    video_writer.release()


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))

    input_root = os.path.join(script_dir, "Input_files")
    output_root = os.path.join(script_dir, "Output_files")
    os.makedirs(output_root, exist_ok=True)

    # Folder inside Input_files/ containing the .h5 files
    input_relpath = "HDF5_stack"  # change this if your .h5 files are in a different subfolder
    input_dir = os.path.join(input_root, input_relpath)

    # Find all HDF5 files in the input directory
    hdf5_files = [f for f in os.listdir(input_dir) if f.endswith(".h5")]
    print("Input dir:", input_dir)
    print("Found HDF5 files:", hdf5_files)

    highlight_spans = [[37, 50], [86, 99], [135, 148], [184, 197], [233, 246], [282, 295], [331, 344]]

    for file in hdf5_files:
        h5_path = os.path.join(input_dir, file)
        mp4_name = os.path.splitext(file)[0] + ".mp4"
        mp4_path = os.path.join(output_root, mp4_name)

        time_series = extract_and_stack_images(h5_path)

        create_colored_video_from_stack(
            time_series,
            mp4_path,
            fps=24,
            include_brightfield=False,
            highlight_spans=highlight_spans,
            um_per_px=8.648,
            scalebar_um=1000,
            border_thickness=10,
        )


if __name__ == "__main__":
    main()