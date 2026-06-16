import json
import math
from types import SimpleNamespace
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import exposure, filters, measure, morphology, segmentation


def safe_optional_float(value):
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(converted):
        return None
    return converted


def robust_rescale(image, lower=1.0, upper=99.5):
    low, high = np.percentile(image, [lower, upper])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return exposure.rescale_intensity(image, in_range=(low, high), out_range=(0.0, 1.0)).astype(np.float32)


def focus_score(image):
    normalized = robust_rescale(image)
    return float(np.var(ndi.laplace(normalized)))


def resolve_scene(img, scene):
    if scene is None:
        return str(img.scenes[0])
    if scene in img.scenes:
        return str(scene)
    if isinstance(scene, int):
        scene_index = scene
        if 0 <= scene_index < len(img.scenes):
            return str(img.scenes[scene_index])
    raise ValueError(f"Scene '{scene}' not found. Available scenes: {list(img.scenes)}")


def data_array_to_numpy(data):
    array = data.data
    if hasattr(array, "compute"):
        array = array.compute()
    return np.asarray(array, dtype=np.float32)


def reduce_to_2d(data, z_mode, z_index):
    if "Z" in data.dims:
        if z_index is not None:
            if not (0 <= z_index < data.sizes["Z"]):
                raise ValueError(f"Requested z-index {z_index} is outside the stack range 0..{data.sizes['Z'] - 1}.")
            reduced = data.isel(Z=z_index)
            selection = f"z_index_{z_index}"
        elif z_mode == "best":
            scores = []
            for current_z in range(data.sizes["Z"]):
                scores.append(focus_score(data_array_to_numpy(data.isel(Z=current_z).transpose("Y", "X"))))
            best_z = int(np.argmax(scores))
            reduced = data.isel(Z=best_z)
            selection = f"best_focus_z_{best_z}"
        elif z_mode == "max":
            reduced = data.max(dim="Z")
            selection = "max_projection"
        else:
            reduced = data.mean(dim="Z")
            selection = "mean_projection"
    else:
        reduced = data
        selection = "single_plane"

    reduced = reduced.squeeze()
    if "Y" not in reduced.dims or "X" not in reduced.dims:
        raise ValueError(f"Could not reduce image to YX. Remaining dimensions: {reduced.dims}")
    reduced = reduced.transpose("Y", "X")
    array = data_array_to_numpy(reduced)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D image after reduction, got shape {array.shape}.")
    return array, selection


def auto_select_channel(data, channel_names, z_mode, z_index):
    scores = []
    for channel_index in range(data.sizes["C"]):
        preview, _ = reduce_to_2d(data.isel(C=channel_index), z_mode, z_index)
        normalized = robust_rescale(preview)
        score = float(np.var(filters.sobel(normalized)) + 0.25 * np.var(normalized))
        scores.append(score)
    best_index = int(np.argmax(scores))
    if channel_names:
        print(f"Auto-selected channel {best_index} ({channel_names[best_index]}) from {len(channel_names)} channels.")
    else:
        print(f"Auto-selected channel {best_index} from {len(scores)} channels.")
    return best_index


def resolve_channel(data, channel, channel_names, z_mode, z_index):
    if "C" not in data.dims:
        return None, None

    if channel == "auto":
        selected = auto_select_channel(data, channel_names, z_mode, z_index)
        return selected, (channel_names[selected] if channel_names and selected < len(channel_names) else None)

    if isinstance(channel, int):
        selected = channel
        if not (0 <= selected < data.sizes["C"]):
            raise ValueError(f"Requested channel {selected} is outside the range 0..{data.sizes['C'] - 1}.")
        return selected, (channel_names[selected] if channel_names and selected < len(channel_names) else None)

    if channel_names:
        channel_lookup = {name.lower(): index for index, name in enumerate(channel_names)}
        if str(channel).lower() in channel_lookup:
            selected = channel_lookup[str(channel).lower()]
            return selected, channel_names[selected]

    raise ValueError(f"Channel '{channel}' not found. Available channel names: {channel_names if channel_names else 'none'}")


def load_czi_image(czi_path, scene, time_index, channel, z_mode, z_index, pixel_size_override_um):
    try:
        from aicsimageio import AICSImage
    except ImportError as exc:
        raise ImportError("aicsimageio is required to read CZI files in this script.") from exc

    img = AICSImage(str(czi_path))
    selected_scene = resolve_scene(img, scene)
    img.set_scene(selected_scene)

    data = img.xarray_dask_data
    if "T" in data.dims:
        if not (0 <= time_index < data.sizes["T"]):
            raise ValueError(f"Requested time index {time_index} is outside the range 0..{data.sizes['T'] - 1}.")
        data = data.isel(T=time_index)
    elif time_index != 0:
        raise ValueError("The CZI file has no time dimension, so --time-index must be 0.")

    channel_names = [str(name) for name in img.channel_names] if getattr(img, "channel_names", None) else None
    channel_index, channel_name = resolve_channel(data, channel, channel_names, z_mode, z_index)
    if channel_index is not None:
        data = data.isel(C=channel_index)

    image_2d, z_selection = reduce_to_2d(data=data, z_mode=z_mode, z_index=z_index)

    pixel_size_y_um = safe_optional_float(getattr(img.physical_pixel_sizes, "Y", None))
    pixel_size_x_um = safe_optional_float(getattr(img.physical_pixel_sizes, "X", None))
    if pixel_size_override_um is not None:
        pixel_size_y_um = pixel_size_override_um
        pixel_size_x_um = pixel_size_override_um

    return SimpleNamespace(image=image_2d, scene=selected_scene, time_index=time_index, channel_index=channel_index, channel_name=channel_name, z_selection=z_selection, pixel_size_y_um=pixel_size_y_um, pixel_size_x_um=pixel_size_x_um)


def remove_small_objects_binary(binary, min_size):
    labeled, number_of_components = ndi.label(binary)
    if number_of_components == 0:
        return binary
    component_sizes = np.bincount(labeled.ravel())
    keep = component_sizes >= min_size
    keep[0] = False
    return keep[labeled]


def choose_central_component(binary):
    labeled = measure.label(binary)
    props = measure.regionprops(labeled)
    if not props:
        return np.zeros_like(binary, dtype=bool)

    center = np.array([(binary.shape[0] - 1) / 2.0, (binary.shape[1] - 1) / 2.0])
    center_row = int(round(center[0]))
    center_col = int(round(center[1]))
    center_label = labeled[center_row, center_col]
    if center_label > 0:
        return labeled == center_label

    chosen = min(props, key=lambda prop: np.sum((np.array(prop.centroid) - center) ** 2))
    return labeled == chosen.label


def detect_central_colony_band(image, expected_cell_width_px, segmentable_band_width_px):
    image_norm = robust_rescale(image)
    texture_sigma = max(1.5, expected_cell_width_px / 2.0)
    mean = ndi.gaussian_filter(image_norm, sigma=texture_sigma)
    mean_sq = ndi.gaussian_filter(image_norm**2, sigma=texture_sigma)
    texture = np.sqrt(np.clip(mean_sq - mean**2, 0.0, None))
    texture = robust_rescale(texture)

    colony_threshold = max(float(filters.threshold_otsu(texture)), float(np.quantile(texture, 0.90)))
    colony_candidates = texture > colony_threshold
    close_radius = max(3, int(round(expected_cell_width_px * 1.5)))
    colony_candidates = morphology.closing(colony_candidates, morphology.disk(close_radius))
    colony_candidates = ndi.binary_fill_holes(colony_candidates)
    colony_candidates = remove_small_objects_binary(colony_candidates, min_size=max(1000, int(np.pi * (expected_cell_width_px * 8.0) ** 2)))

    colony_mask = choose_central_component(colony_candidates)
    colony_mask = morphology.closing(colony_mask, morphology.disk(close_radius))
    colony_mask = ndi.binary_fill_holes(colony_mask)

    band_width_px = max(1, int(round(segmentable_band_width_px)))
    distance_inside_colony = ndi.distance_transform_edt(colony_mask)
    analysis_mask = colony_mask & (distance_inside_colony <= band_width_px)
    return colony_mask, analysis_mask, texture, colony_threshold


def segment_with_cellpose(image, analysis_mask, expected_cell_width_px, use_gpu, invert, flow_threshold, cellprob_threshold, min_size, augment, tile_overlap):
    try:
        from cellpose import models
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise ImportError("Cellpose is not installed in the active environment. Install it in the Python environment you use to run this script.") from exc
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise ImportError("PyTorch is required for Cellpose but is not installed in the active environment.") from exc

    if not np.any(analysis_mask):
        return robust_rescale(image), np.zeros_like(image, dtype=np.int32)

    if use_gpu and not torch.cuda.is_available():
        raise RuntimeError("cellpose_use_gpu=True, but CUDA is not available in the active Python environment. Install a CUDA-enabled PyTorch build and run the script from that environment.")

    rows, cols = np.where(analysis_mask)
    padding = max(20, int(round(expected_cell_width_px * 4.0)))
    row_start = max(0, int(rows.min()) - padding)
    row_end = min(image.shape[0], int(rows.max()) + 1 + padding)
    col_start = max(0, int(cols.min()) - padding)
    col_end = min(image.shape[1], int(cols.max()) + 1 + padding)

    image_norm = robust_rescale(image)
    image_crop = image_norm[row_start:row_end, col_start:col_end]
    mask_crop = analysis_mask[row_start:row_end, col_start:col_end]

    # Measurements used for mutation_scaling were generated with Cellpose v4.1.0 and its default pretrained Cellpose-SAM model, cpsam.
    model = models.CellposeModel(gpu=use_gpu)
    masks, _flows, _styles = model.eval(image_crop, channel_axis=None, invert=invert, diameter=expected_cell_width_px, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold, min_size=min_size, augment=augment, tile_overlap=tile_overlap, resample=True)

    cropped_labels = masks.astype(np.int32)
    cropped_labels[~mask_crop] = 0
    cropped_labels, _, _ = segmentation.relabel_sequential(cropped_labels)

    full_labels = np.zeros_like(image, dtype=np.int32)
    full_labels[row_start:row_end, col_start:col_end] = cropped_labels
    return image_norm, full_labels


def build_object_table(labels):
    image_height, image_width = labels.shape
    border_margin = 1
    records = []
    for prop in measure.regionprops(labels):
        min_row, min_col, max_row, max_col = prop.bbox
        touches_border = (min_row <= border_margin or min_col <= border_margin or max_row >= image_height - border_margin or max_col >= image_width - border_margin)
        records.append({"label": int(prop.label), "area_px": float(prop.area), "centroid_row": float(prop.centroid[0]), "centroid_col": float(prop.centroid[1]), "eccentricity": float(prop.eccentricity), "solidity": float(prop.solidity), "extent": float(prop.extent), "touches_border": bool(touches_border)})
    return pd.DataFrame.from_records(records)


def passes_size_and_shape(table, min_area_px, max_area_px):
    if table.empty:
        return pd.Series(dtype=bool)
    return ((table["area_px"] >= min_area_px) & (table["area_px"] <= max_area_px) & (table["solidity"] >= 0.75) & (table["eccentricity"] <= 0.95) & (table["extent"] >= 0.25) & (~table["touches_border"]))


def clip_rectangle(rectangle, image_shape):
    row_start, row_end, col_start, col_end = rectangle
    height, width = image_shape
    row_start = max(0, min(height, row_start))
    row_end = max(0, min(height, row_end))
    col_start = max(0, min(width, col_start))
    col_end = max(0, min(width, col_end))
    if row_end <= row_start or col_end <= col_start:
        raise ValueError(f"Rectangle {rectangle} collapses after clipping to the image bounds.")
    return row_start, row_end, col_start, col_end


def build_manual_exclusion_mask(rectangles, image_shape):
    mask = np.zeros(image_shape, dtype=bool)
    for rectangle in rectangles:
        row_start, row_end, col_start, col_end = clip_rectangle(rectangle, image_shape)
        mask[row_start:row_end, col_start:col_end] = True
    return mask


def build_tile_quality_mask(image_shape, labels, table, size_shape_mask, max_area_px, tile_size_px):
    if table.empty:
        return np.zeros(image_shape, dtype=bool), pd.DataFrame()

    accepted_table = table.loc[size_shape_mask].copy()
    oversized_labels = table.loc[table["area_px"] > max_area_px, "label"].to_numpy(dtype=np.int32)
    accepted_labels = accepted_table["label"].to_numpy(dtype=np.int32)

    foreground_mask = labels > 0
    oversized_mask = np.isin(labels, oversized_labels)
    accepted_mask = np.isin(labels, accepted_labels)

    global_median_area = (float(accepted_table["area_px"].median()) if not accepted_table.empty else math.nan)
    global_tile_mask = np.zeros(image_shape, dtype=bool)
    tile_records = []

    image_height, image_width = image_shape
    min_cells_for_local_area_check = 5

    for row_start in range(0, image_height, tile_size_px):
        for col_start in range(0, image_width, tile_size_px):
            row_end = min(image_height, row_start + tile_size_px)
            col_end = min(image_width, col_start + tile_size_px)
            tile_slice = np.s_[row_start:row_end, col_start:col_end]
            tile_area = float((row_end - row_start) * (col_end - col_start))

            accepted_in_tile = accepted_table.loc[(accepted_table["centroid_row"] >= row_start) & (accepted_table["centroid_row"] < row_end) & (accepted_table["centroid_col"] >= col_start) & (accepted_table["centroid_col"] < col_end)]

            foreground_fraction = float(np.mean(foreground_mask[tile_slice]))
            accepted_fraction = float(np.mean(accepted_mask[tile_slice]))
            oversized_fraction = float(np.mean(oversized_mask[tile_slice]))
            local_count = int(len(accepted_in_tile))
            local_median_area = (float(accepted_in_tile["area_px"].median()) if local_count > 0 else math.nan)

            is_good = True
            reasons = []

            if oversized_fraction > 0.12:
                is_good = False
                reasons.append("high_oversized_fraction")

            if (local_count >= min_cells_for_local_area_check and np.isfinite(global_median_area) and (local_median_area < global_median_area * 0.55 or local_median_area > global_median_area * 1.8)):
                is_good = False
                reasons.append("local_median_area_outlier")

            if foreground_fraction > 0.12 and local_count == 0:
                is_good = False
                reasons.append("foreground_without_valid_cells")

            if (foreground_fraction > 0.45 and local_count < min_cells_for_local_area_check):
                is_good = False
                reasons.append("dense_tile_with_too_few_cells")

            if is_good:
                global_tile_mask[tile_slice] = True

            tile_records.append({"row_start": row_start, "row_end": row_end, "col_start": col_start, "col_end": col_end, "tile_area_px": tile_area, "foreground_fraction": foreground_fraction, "accepted_fraction": accepted_fraction, "oversized_fraction": oversized_fraction, "n_accepted_cells": local_count, "median_area_px": local_median_area, "is_good": is_good, "reasons": ";".join(reasons) if reasons else "ok"})

    return global_tile_mask, pd.DataFrame.from_records(tile_records)


def summarize_measurements(values):
    if values.size == 0:
        raise RuntimeError("No cells remained after segmentation and bad-region filtering.")
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    sem = float(std / np.sqrt(values.size)) if values.size > 1 else 0.0
    return {"n": int(values.size),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": std,
            "sem": sem}


def save_overlay(output_path, raw_image, labels, accepted_labels, excluded_region_mask):
    figure, axis = plt.subplots(figsize=(10, 10))
    base_image = robust_rescale(raw_image)
    overlay_image = np.dstack([base_image, base_image, base_image])

    accepted_label_image = np.where(np.isin(labels, accepted_labels), labels, 0)
    rejected_label_image = np.where((labels > 0) & ~np.isin(labels, accepted_labels), labels, 0)
    accepted_boundaries = segmentation.find_boundaries(accepted_label_image, mode="inner")
    rejected_boundaries = segmentation.find_boundaries(rejected_label_image, mode="inner")

    overlay_image[rejected_boundaries] = (1.0, 0.75, 0.0)
    overlay_image[accepted_boundaries] = (0.0, 1.0, 0.4)
    axis.imshow(overlay_image)

    excluded_overlay = np.zeros((*raw_image.shape, 4), dtype=np.float32)
    excluded_overlay[excluded_region_mask] = (1.0, 0.0, 0.0, 0.22)
    axis.imshow(excluded_overlay)

    axis.set_title("Accepted cells (green), rejected segments (orange), excluded regions (red)")
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main(config):
    czi_path = config["czi_path"]
    output_dir = config["output_dir"]
    scene = config["scene"]
    time_index = config["time_index"]
    channel = config["channel"]
    z_mode = config["z_mode"]
    z_index = config["z_index"]
    pixel_size_xy_um = config["pixel_size_xy_um"]
    min_area = config["min_area"]
    max_area = config["max_area"]
    expected_cell_width_px = config["expected_cell_width_px"]
    segmentable_band_width_px = config["segmentable_band_width_px"]
    cellpose_use_gpu = config["cellpose_use_gpu"]
    cellpose_invert = config["cellpose_invert"]
    cellpose_flow_threshold = config["cellpose_flow_threshold"]
    cellpose_cellprob_threshold = config["cellpose_cellprob_threshold"]
    cellpose_min_size = config["cellpose_min_size"]
    cellpose_augment = config["cellpose_augment"]
    cellpose_tile_overlap = config["cellpose_tile_overlap"]
    tile_size_px = config["tile_size_px"]
    exclude_rectangles = config["exclude_rectangles"]

    czi_path = Path(czi_path)
    output_dir = Path(output_dir) if output_dir is not None else None

    if tile_size_px <= 0:
        raise ValueError("--tile-size-px must be a positive integer.")
    if z_mode not in {"best", "max", "mean"}:
        raise ValueError("z_mode must be one of: 'best', 'max', 'mean'.")
    if expected_cell_width_px <= 0:
        raise ValueError("expected_cell_width_px must be positive.")
    if cellpose_min_size < 0:
        raise ValueError("cellpose_min_size must be non-negative.")
    if not (0.0 <= cellpose_tile_overlap < 1.0):
        raise ValueError("cellpose_tile_overlap must be in the range [0, 1).")

    if output_dir is None:
        output_dir = czi_path.with_suffix("")
        output_dir = output_dir.parent / f"{output_dir.name}_cell_area"
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_czi_image(czi_path, scene, time_index, channel, z_mode, z_index, pixel_size_xy_um)

    if loaded.pixel_size_x_um is not None and loaded.pixel_size_y_um is not None:
        pixel_area = loaded.pixel_size_x_um * loaded.pixel_size_y_um
        area_unit = "um^2"
    else:
        pixel_area = 1.0
        area_unit = "px^2"
        print("Pixel size metadata is missing. Area outputs and thresholds will be interpreted in px^2. Pass pixel_size_xy_um to report areas in um^2.")

    min_area_px = max(min_area / pixel_area, 1.0)
    max_area_px = max(max_area / pixel_area, min_area_px + 1.0)
    if segmentable_band_width_px is None:
        segmentable_band_width_px = expected_cell_width_px * 5.0

    colony_mask, analysis_mask, _texture_map, colony_threshold = detect_central_colony_band(loaded.image, expected_cell_width_px, segmentable_band_width_px)
    if not np.any(colony_mask):
        raise RuntimeError("Failed to detect the central colony mask. Adjust expected_cell_width_px if needed.")
    if not np.any(analysis_mask):
        raise RuntimeError("The detected colony band is empty. Increase segmentable_band_width_px or adjust expected_cell_width_px.")

    _corrected, labels = segment_with_cellpose(loaded.image, analysis_mask, expected_cell_width_px, cellpose_use_gpu, cellpose_invert, cellpose_flow_threshold, cellpose_cellprob_threshold, cellpose_min_size, cellpose_augment, cellpose_tile_overlap)
    table = build_object_table(labels)
    selected_method = "cellpose"
    selected_polarity = "cellpose_invert" if cellpose_invert else "cellpose_noninvert"

    if table.empty:
        raise RuntimeError("Cellpose produced no objects. Try another channel or change cellpose_invert.")

    table["passes_size_shape"] = passes_size_and_shape(table, min_area_px, max_area_px)

    tile_quality_mask, tile_table = build_tile_quality_mask(loaded.image.shape, labels, table, table["passes_size_shape"], max_area_px, tile_size_px)

    manual_exclusion_mask = build_manual_exclusion_mask(exclude_rectangles, loaded.image.shape)

    if tile_table.empty:
        excluded_tile_mask = np.ones_like(loaded.image, dtype=bool)
    else:
        excluded_tile_mask = ~tile_quality_mask

    centroid_rows = np.clip(table["centroid_row"].round().astype(int), 0, loaded.image.shape[0] - 1)
    centroid_cols = np.clip(table["centroid_col"].round().astype(int), 0, loaded.image.shape[1] - 1)
    table["manual_excluded"] = manual_exclusion_mask[centroid_rows, centroid_cols]
    table["passes_tile_quality"] = tile_quality_mask[centroid_rows, centroid_cols]
    table["accepted"] = table["passes_size_shape"] & table["passes_tile_quality"] & (~table["manual_excluded"])
    table["reject_reason"] = "accepted"
    table.loc[~table["passes_size_shape"], "reject_reason"] = "size_or_shape_filter"
    table.loc[table["passes_size_shape"] & ~table["passes_tile_quality"], "reject_reason"] = "bad_tile"
    table.loc[table["manual_excluded"], "reject_reason"] = "manual_exclusion"

    table["area_reported"] = table["area_px"] * pixel_area
    table["area_unit"] = area_unit

    final_measurements = table.loc[table["accepted"]].copy()
    if final_measurements.empty:
        raise RuntimeError("No cells remained after filtering. Adjust channel, min_area/max_area, cellpose_invert, or tile_size_px.")

    accepted_labels = final_measurements["label"].to_numpy(dtype=np.int32)
    excluded_region_mask = (~analysis_mask) | excluded_tile_mask | manual_exclusion_mask

    summary = summarize_measurements(final_measurements["area_reported"].to_numpy(dtype=np.float64))
    summary_payload = {"czi_path": str(czi_path),
                       "scene": loaded.scene,
                       "time_index": loaded.time_index,
                       "channel_index": loaded.channel_index,
                       "channel_name": loaded.channel_name,
                       "z_selection": loaded.z_selection,
                       "cellpose_polarity": selected_polarity,
                       "segmentation_backend": "cellpose",
                       "cellpose_use_gpu": cellpose_use_gpu,
                       "cellpose_invert": cellpose_invert,
                       "cellpose_flow_threshold": cellpose_flow_threshold,
                       "cellpose_cellprob_threshold": cellpose_cellprob_threshold,
                       "cellpose_min_size": cellpose_min_size,
                       "cellpose_augment": cellpose_augment,
                       "cellpose_tile_overlap": cellpose_tile_overlap,
                       "segmentation_method": selected_method,
                       "colony_texture_threshold": colony_threshold,
                       "colony_area_px": int(np.sum(colony_mask)),
                       "analysis_band_area_px": int(np.sum(analysis_mask)),
                       "expected_cell_width_px": expected_cell_width_px,
                       "segmentable_band_width_px": float(segmentable_band_width_px),
                       "pixel_size_y_um": loaded.pixel_size_y_um,
                       "pixel_size_x_um": loaded.pixel_size_x_um,
                       "area_unit": area_unit,
                       "n_segmented_objects": int(len(table)),
                       "n_after_size_shape_filter": int(table["passes_size_shape"].sum()),
                       "n_after_tile_filter": int((table["passes_size_shape"] & table["passes_tile_quality"]).sum()),
                       "n_final_cells": summary["n"],
                       "mean_area": summary["mean"],
                       "median_area": summary["median"],
                       "std_area": summary["std"],
                       "sem_area": summary["sem"],
                       "excluded_image_fraction": float(np.mean(excluded_region_mask)),
                       "excluded_foreground_fraction": float(np.sum(excluded_region_mask & (labels > 0)) / max(1, np.sum(labels > 0)))}

    measurement_path = output_dir / f"{czi_path.stem}_cell_measurements.csv"
    tile_quality_path = output_dir / f"{czi_path.stem}_tile_quality.csv"
    summary_path = output_dir / f"{czi_path.stem}_summary.json"
    overlay_path = output_dir / f"{czi_path.stem}_overlay.png"

    table.to_csv(measurement_path, index=False)
    tile_table.to_csv(tile_quality_path, index=False)
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    save_overlay(overlay_path, loaded.image, labels, accepted_labels, excluded_region_mask)

    print(f"Processed file: {czi_path}")
    print(f"Scene: {loaded.scene}")
    if loaded.channel_index is not None:
        channel_label = loaded.channel_name if loaded.channel_name is not None else str(loaded.channel_index)
        print(f"Channel: {loaded.channel_index} ({channel_label})")
    print(f"Z selection: {loaded.z_selection}")
    print(f"Cellpose polarity: {selected_polarity}")
    print("Segmentation backend: cellpose")
    print(f"Cellpose GPU requested: {cellpose_use_gpu}")
    print("Cellpose settings: " f"invert={cellpose_invert}, " f"flow_threshold={cellpose_flow_threshold:.2f}, " f"cellprob_threshold={cellpose_cellprob_threshold:.2f}, " f"min_size={cellpose_min_size}, " f"augment={cellpose_augment}, " f"tile_overlap={cellpose_tile_overlap:.2f}")
    print(f"Segmentation method: {selected_method}")
    print(f"Expected cell width: {expected_cell_width_px:.2f} px")
    print(f"Analysis band width: {segmentable_band_width_px:.2f} px")
    print(f"Accepted cells: {summary['n']}")
    print(f"Mean cell area: {summary['mean']:.3f} {area_unit}")
    print(f"Std: {summary['std']:.3f} {area_unit}")
    print(f"SEM: {summary['sem']:.3f} {area_unit}")
    print(f"Excluded image fraction: {summary_payload['excluded_image_fraction']:.3f}")
    print(f"Saved measurements to: {measurement_path}")
    print(f"Saved tile quality table to: {tile_quality_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved overlay to: {overlay_path}")


if __name__ == "__main__":
    config = {"czi_path": Path("data/Single_cell_resolution_yNA16.czi"),
              "output_dir": None,
              "scene": None,
              "time_index": 0,
              "channel": "auto",
              "z_mode": "best",
              "z_index": None,
              "pixel_size_xy_um": None,
              "min_area": 5.0,
              "max_area": 40.0,
              "expected_cell_width_px": 6.5,
              "segmentable_band_width_px": 25,
              "cellpose_use_gpu": True,
              "cellpose_invert": True,
              "cellpose_flow_threshold": 0.8,
              "cellpose_cellprob_threshold": -2.0,
              "cellpose_min_size": 6,
              "cellpose_augment": True,
              "cellpose_tile_overlap": 0.25,
              "tile_size_px": 256,
              "exclude_rectangles": []}
    main(config)
