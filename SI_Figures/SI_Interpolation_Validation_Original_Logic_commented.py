#!/usr/bin/env python3
"""
Validate interpolation-to-final uncertainty using the original adjusted-colony logic.

What this script validates
--------------------------
This version follows the original interpolation logic:

    adjusted_colony_area = measured_colony_area - measured_clone_area + interpolated_clone_area

Therefore, the clone area is the only quantity that is linearly interpolated to
or from the final endpoint. Colony-area uncertainty is then derived from the
impact of replacing the measured clone area with the interpolated clone area.

For each colony:
    1. Load a fully observed timelapse CSV with measured values up to at least
       GROUND_TRUTH_FRAME, usually frame 250.
    2. Load the matching separate final CSV, identified by the P<id> in the file name.
    3. Use the final CSV clone area as the interpolation endpoint, exactly like
       the real interpolation-to-final workflow.
    4. Pretend that interpolation started earlier at frame 150, 155, 160, ..., 250.
    5. For each artificial start frame, calculate the interpolated clone area at
       GROUND_TRUTH_FRAME.
    6. Compare:

           clone error = interpolated_clone_area_at_GT - measured_clone_area_at_GT

       and derive the adjusted colony area as:

           adjusted_colony_area_at_GT = measured_colony_area_at_GT
                                       - measured_clone_area_at_GT
                                       + interpolated_clone_area_at_GT

       Then compare that adjusted colony area with the measured colony area at GT:

           colony error = adjusted_colony_area_at_GT - measured_colony_area_at_GT

       Algebraically, the colony error in mm² is identical to the clone interpolation
       error in mm², but the colony percent error is calculated relative to the total
       measured colony area, while the clone percent error is calculated relative to
       the measured clone area.

Input folder layout
-------------------
Place the input files next to this script in:

    Input_files/interpolation_validation/timelapse/
    Input_files/interpolation_validation/finals/

Timelapse CSVs should contain:
    - frame
    - colony_area
    - total_clone_area

Final CSVs should contain:
    - total_clone_area

Final CSVs may also contain colony_area, but this script does not use final
colony_area for the adjusted-colony validation.

Output
------
Results are written to OUTPUT_RELPATH.

Main outputs:
    - original_logic_interpolation_validation_per_start_errors.csv
    - original_logic_interpolation_validation_by_start_frame_summary.csv
    - original_logic_interpolation_validation_overall_summary.csv
    - original_logic_interpolation_validation_report_text.txt
    - PDF plots of adjusted colony-area error vs positive Δt before the ground-truth frame

Important
---------
The final endpoint needs a frame index. By default, this script follows the
extrapolation-to-final workflow and sets:

    final_frame_index = last timelapse frame + 1

If your separate final image corresponds to a different time point, set
FINAL_FRAME_INDEX_MODE = "manual" and choose MANUAL_FINAL_FRAME_INDEX below.
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ───────────────────────────── CONFIG ─────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_ROOT = SCRIPT_DIR / "Input_files"
OUTPUT_ROOT = SCRIPT_DIR / "Output_files"

# Input folders for reviewers.
# These paths are resolved relative to Input_files/.
TIMELAPSE_INPUT_RELPATH = "interpolation_validation/timelapse"
FINALS_INPUT_RELPATH = "interpolation_validation/finals"

# Output folder for generated CSV, TXT, and PDF files.
# This path is resolved relative to Output_files/.
OUTPUT_RELPATH = "interpolation_validation_original_logic"

# Pixel sizes.
SEG_SCALE_UM_PER_PX = 8.648       # timelapse data
FINAL_SCALE_UM_PER_PX = 14.424    # separate final CSV/image

# Validation setup.
GROUND_TRUTH_FRAME = 250
START_FRAMES = list(range(150, 251, 5))  # 150, 155, ..., 250

# Time conversion for plots.
# Acquisition uses 2 frames = 1 hour.
FRAMES_PER_HOUR = 2.0

# How to assign the final CSV to a frame index.
# "last_timelapse_plus_one" reproduces the extrapolation-to-final workflow:
#     final_frame_index = max(frame in timelapse CSV) + 1
#
# "manual" uses MANUAL_FINAL_FRAME_INDEX below.
FINAL_FRAME_INDEX_MODE = "last_timelapse_plus_one"
MANUAL_FINAL_FRAME_INDEX = None  # e.g. 318 if the final image corresponds to frame 318

# Start frame 250 is the same as the ground-truth frame.
# Its error is zero by construction and should usually NOT be included in the
# main summary, otherwise the average error is biased downward.
INCLUDE_EVAL_ANCHOR_START_IN_SUMMARY = False

# Column names.
TIMELAPSE_COLONY_COL = "colony_area"
TIMELAPSE_CLONE_COL = "total_clone_area"
FINAL_CLONE_COL = "total_clone_area"

# File filtering.
REQUIRE_CLONEAREA_IN_TIMELAPSE_FILENAME = True

# Optional border-touch check. For this validation set, colonies should ideally
# not touch the border before or at GROUND_TRUTH_FRAME.
CHECK_BORDER_TOUCH_IF_CONTOUR_EXISTS = True
SKIP_FILE_IF_TOUCHES_BEFORE_OR_AT_GROUND_TRUTH = True

IMG_WIDTH = 1376
IMG_HEIGHT = 1104
TOUCH_DETECTION_START_FRAME = 20
BORDER_TOL = 1e-6

# Frame matching.
ALLOW_NEAREST_FRAME = False
NEAREST_FRAME_MAX_DISTANCE = 1

# Plot settings.
SAVE_PLOTS = True
SHOW_PLOTS = False

PLOT_CONFIG = {
    "fixed_layout": None,
    "figsize": (3.6, 1.8),
    "dpi": 300,
    "line_width": 1.0,
}

# These names match the values in the 'measurement' column of the summary table.
MEASUREMENT_STYLES = {
    "adjusted_colony_area": {"color": "dimgray", "label": "Derived colony area"},
    "clone_area": {"color": "goldenrod", "label": "Clone area"},
}

# Plot only the derived colony-area statistics.
# The clone-area rows are still calculated and saved in the CSV summaries,
# but the clone-area plot series is commented out here.
PLOT_MEASUREMENTS = [
    "adjusted_colony_area",
    # "clone_area",
]


# ───────────────────────────── HELPERS ─────────────────────────────

def resolve_path(root: Path, path_like: str) -> Path:
    """Resolve absolute paths as-is and relative paths against root."""
    p = Path(path_like)
    if p.is_absolute():
        return p
    return root / p


def px2_to_mm2(px2, pixel_size_um: float):
    """Convert area from px² to mm²."""
    return np.asarray(px2, dtype=float) * (pixel_size_um * 1e-3) ** 2


def frame_to_hours(frame):
    """Convert frame number(s) to hours using 2 frames = 1 hour."""
    return np.asarray(frame, dtype=float) / FRAMES_PER_HOUR


def frame_delta_from_ground_truth_hours(frame, ground_truth_frame: int = GROUND_TRUTH_FRAME):
    """
    Convert frame number(s) to signed Δt relative to the ground-truth frame.

    Δt = 0 means the ground-truth frame.
    Negative Δt values are artificial interpolation starts before the ground truth.
    This signed value is kept for backwards compatibility in the raw output table.
    """
    return (np.asarray(frame, dtype=float) - float(ground_truth_frame)) / FRAMES_PER_HOUR


def frame_time_before_ground_truth_hours(frame, ground_truth_frame: int = GROUND_TRUTH_FRAME):
    """
    Convert frame number(s) to positive time before the ground-truth frame.

    0 h means the ground-truth frame. Earlier artificial interpolation start
    frames are shown as positive values.
    Example with 2 frames/h and GROUND_TRUTH_FRAME=250:
        frame 250 ->  0 h
        frame 200 -> 25 h
        frame 150 -> 50 h
    """
    return (float(ground_truth_frame) - np.asarray(frame, dtype=float)) / FRAMES_PER_HOUR


def get_pid(path: Path) -> Optional[int]:
    """Extract numeric P-id from filename, e.g. P11 -> 11."""
    match = re.search(r"P(\d{1,2})(?!\d)", path.name, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def get_pid_label(path: Path) -> Optional[str]:
    pid = get_pid(path)
    if pid is None:
        return None
    return f"P{pid}"


def setup_matplotlib() -> None:
    """Manuscript-style rcParams matching the P7 extrapolation script."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "axes.linewidth": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.frameon": True,
        "legend.fontsize": 6,
        "lines.linewidth": 1.0,
        "savefig.dpi": 300,
        "figure.dpi": 300,
        "axes.spines.top": True,
        "axes.spines.right": True,
    })


def fixed_axes(fig_w, fig_h, ax_w, ax_h, left=0.25, bottom=0.22):
    """Create a figure with one axes of fixed physical size."""
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([
        left / fig_w,
        bottom / fig_h,
        ax_w / fig_w,
        ax_h / fig_h,
    ])
    return fig, ax


def load_csv_with_frame(path: Path) -> pd.DataFrame:
    """Load a CSV, ensure a frame column if possible, sort by frame."""
    df = pd.read_csv(path)

    if "frame" not in df.columns:
        print(f"[warn] {path.name}: no 'frame' column found; using row index as frame.")
        df = df.reset_index().rename(columns={"index": "frame"})

    df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
    df = df.dropna(subset=["frame"]).copy()
    df["frame"] = df["frame"].astype(int)

    before = len(df)
    df = df.sort_values("frame").drop_duplicates(subset=["frame"], keep="last").reset_index(drop=True)
    after = len(df)
    if after < before:
        print(f"[warn] {path.name}: removed {before - after} duplicate frame rows; kept last occurrence.")

    return df


def find_existing_frame(frames: np.ndarray, requested_frame: int) -> Optional[int]:
    """Return requested frame, or nearest frame if enabled."""
    frames = np.asarray(frames, dtype=int)

    if requested_frame in set(frames):
        return int(requested_frame)

    if not ALLOW_NEAREST_FRAME or len(frames) == 0:
        return None

    idx = int(np.argmin(np.abs(frames - requested_frame)))
    nearest = int(frames[idx])
    if abs(nearest - requested_frame) <= NEAREST_FRAME_MAX_DISTANCE:
        return nearest

    return None


def detect_timelapse_files(folder: Path) -> List[Path]:
    """Find timelapse CSVs."""
    files = []
    for path in sorted(folder.glob("*.csv")):
        lower = path.name.lower()
        if REQUIRE_CLONEAREA_IN_TIMELAPSE_FILENAME and "clonearea" not in lower:
            continue
        files.append(path)

    def sort_key(path: Path):
        pid = get_pid(path)
        return (pid is None, pid if pid is not None else 9999, path.name)

    files = sorted(files, key=sort_key)
    print(f"[info] Found {len(files)} timelapse CSV files.")
    return files


def build_finals_map(folder: Path) -> Dict[int, Path]:
    """Build P-id -> final CSV path map."""
    mapping: Dict[int, Path] = {}

    for path in sorted(folder.glob("*.csv")):
        pid = get_pid(path)
        if pid is None:
            continue
        if pid in mapping:
            print(
                f"[warn] Multiple final CSVs found for P{pid}; "
                f"using the later one in sorted order: {path.name}"
            )
        mapping[pid] = path

    print(f"[info] Found {len(mapping)} final CSV files with P<id> in name.")
    return mapping


def parse_contour(contour_value):
    """Parse a contour stored as list-like object or string."""
    if isinstance(contour_value, str):
        contour_value = ast.literal_eval(contour_value)

    arr = np.asarray(contour_value, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2 or arr.size == 0:
        return None
    return arr


def first_touch_frame(
    df: pd.DataFrame,
    width: int = IMG_WIDTH,
    height: int = IMG_HEIGHT,
    start_frame: int = TOUCH_DETECTION_START_FRAME,
    tol: float = BORDER_TOL,
) -> Optional[int]:
    """Return first frame where colony_contour touches the image border."""
    if "colony_contour" not in df.columns:
        return None

    border_x_min, border_y_min = 0.5, 0.5
    border_x_max, border_y_max = width + 0.5, height + 0.5

    for _, row in df.iterrows():
        frame = int(row["frame"])
        if frame < start_frame:
            continue

        try:
            arr = parse_contour(row["colony_contour"])
        except Exception:
            continue

        if arr is None:
            continue

        y = arr[:, 0]
        x = arr[:, 1]

        touches = (
            (y <= border_y_min + tol).any()
            or (y >= border_y_max - tol).any()
            or (x <= border_x_min + tol).any()
            or (x >= border_x_max - tol).any()
        )
        if touches:
            return frame

    return None


def get_final_frame_index(timelapse_frames: np.ndarray) -> int:
    """Return the frame index assigned to the separate final endpoint."""
    if FINAL_FRAME_INDEX_MODE == "last_timelapse_plus_one":
        return int(np.nanmax(timelapse_frames)) + 1

    if FINAL_FRAME_INDEX_MODE == "manual":
        if MANUAL_FINAL_FRAME_INDEX is None:
            raise ValueError("FINAL_FRAME_INDEX_MODE is 'manual', but MANUAL_FINAL_FRAME_INDEX is None.")
        return int(MANUAL_FINAL_FRAME_INDEX)

    raise ValueError(f"Unknown FINAL_FRAME_INDEX_MODE: {FINAL_FRAME_INDEX_MODE}")


def read_final_clone_value_mm2(final_path: Path) -> Optional[float]:
    """Read final endpoint clone area from final CSV and convert it to mm²."""
    final_df = pd.read_csv(final_path)

    if FINAL_CLONE_COL not in final_df.columns:
        print(f"[warn] {final_path.name}: missing final clone column '{FINAL_CLONE_COL}'.")
        return None

    final_px2 = pd.to_numeric(final_df[FINAL_CLONE_COL], errors="coerce").dropna()
    if final_px2.empty:
        print(f"[warn] {final_path.name}: final clone column '{FINAL_CLONE_COL}' contains no numeric values.")
        return None

    # Same convention as the extrapolation-to-final workflow: use the last row by default.
    return float(px2_to_mm2(final_px2.iloc[-1], FINAL_SCALE_UM_PER_PX))


def error_metrics(estimated: float, measured: float) -> Dict[str, float]:
    """Calculate signed, absolute, percent, and symmetric percent errors."""
    signed_error = estimated - measured
    abs_error = abs(signed_error)

    if measured != 0:
        percent_error = 100.0 * signed_error / measured
        abs_percent_error = abs(percent_error)
    else:
        percent_error = np.nan
        abs_percent_error = np.nan

    denom = abs(estimated) + abs(measured)
    if denom != 0:
        symmetric_percent_error = 200.0 * signed_error / denom
        symmetric_abs_percent_error = abs(symmetric_percent_error)
    else:
        symmetric_percent_error = np.nan
        symmetric_abs_percent_error = np.nan

    return {
        "signed_error_mm2": signed_error,
        "abs_error_mm2": abs_error,
        "percent_error": percent_error,
        "abs_percent_error": abs_percent_error,
        "symmetric_percent_error": symmetric_percent_error,
        "symmetric_abs_percent_error": symmetric_abs_percent_error,
    }


# ───────────────────────────── VALIDATION ─────────────────────────────

def validate_file(timelapse_path: Path, final_path: Path) -> List[dict]:
    """Validate one timelapse/final pair using the original adjusted-colony logic."""
    rows: List[dict] = []

    df = load_csv_with_frame(timelapse_path)
    frames = df["frame"].to_numpy(dtype=int)

    required_cols = {TIMELAPSE_COLONY_COL, TIMELAPSE_CLONE_COL}
    missing = required_cols.difference(df.columns)
    if missing:
        print(f"[skip] {timelapse_path.name}: missing required timelapse columns: {sorted(missing)}")
        return rows

    if CHECK_BORDER_TOUCH_IF_CONTOUR_EXISTS and "colony_contour" in df.columns:
        touch_frame = first_touch_frame(df)
        if touch_frame is not None:
            print(f"[touch] {timelapse_path.name}: first border touch at frame {touch_frame}")
            if SKIP_FILE_IF_TOUCHES_BEFORE_OR_AT_GROUND_TRUTH and touch_frame <= GROUND_TRUTH_FRAME:
                print(
                    f"[skip] {timelapse_path.name}: touches border before/at "
                    f"ground-truth frame {GROUND_TRUTH_FRAME}."
                )
                return rows
        else:
            print(f"[ok] {timelapse_path.name}: no border touch detected.")

    eval_frame = find_existing_frame(frames, GROUND_TRUTH_FRAME)
    if eval_frame is None:
        print(f"[skip] {timelapse_path.name}: ground-truth frame {GROUND_TRUTH_FRAME} not found.")
        return rows

    final_frame_index = get_final_frame_index(frames)
    if final_frame_index <= eval_frame:
        print(
            f"[warn] {timelapse_path.name}: final_frame_index={final_frame_index} "
            f"is not after eval_frame={eval_frame}. Results may not be meaningful."
        )

    final_clone_mm2 = read_final_clone_value_mm2(final_path)
    if final_clone_mm2 is None or not np.isfinite(final_clone_mm2):
        print(f"[skip] {timelapse_path.name}: no usable final clone endpoint in {final_path.name}.")
        return rows

    pid_label = get_pid_label(timelapse_path)

    print(
        f"[info] {timelapse_path.name}: using final {final_path.name}; "
        f"eval frame = {eval_frame}; final frame index = {final_frame_index}"
    )

    frame_to_idx = {int(frame): idx for idx, frame in enumerate(frames)}
    idx_eval = frame_to_idx[eval_frame]

    colony_values_px2 = pd.to_numeric(df[TIMELAPSE_COLONY_COL], errors="coerce").to_numpy(dtype=float)
    clone_values_px2 = pd.to_numeric(df[TIMELAPSE_CLONE_COL], errors="coerce").to_numpy(dtype=float)

    colony_values_mm2 = px2_to_mm2(colony_values_px2, SEG_SCALE_UM_PER_PX)
    clone_values_mm2 = px2_to_mm2(clone_values_px2, SEG_SCALE_UM_PER_PX)

    measured_colony_eval_mm2 = float(colony_values_mm2[idx_eval])
    measured_clone_eval_mm2 = float(clone_values_mm2[idx_eval])

    if not np.isfinite(measured_colony_eval_mm2) or not np.isfinite(measured_clone_eval_mm2):
        print(f"[skip] {timelapse_path.name}: non-finite measured values at ground-truth frame.")
        return rows

    for start_frame_requested in START_FRAMES:
        start_frame = find_existing_frame(frames, start_frame_requested)
        if start_frame is None:
            print(f"[skip] {timelapse_path.name}: start frame {start_frame_requested} not found.")
            continue

        if start_frame > eval_frame:
            print(
                f"[skip] {timelapse_path.name}: start frame {start_frame} is after "
                f"ground-truth frame {eval_frame}."
            )
            continue

        if start_frame >= final_frame_index:
            print(
                f"[skip] {timelapse_path.name}: start frame {start_frame} is at/after "
                f"final frame index {final_frame_index}."
            )
            continue

        idx_start = frame_to_idx[start_frame]
        start_clone_mm2 = float(clone_values_mm2[idx_start])
        measured_colony_start_mm2 = float(colony_values_mm2[idx_start])

        if not np.isfinite(start_clone_mm2) or not np.isfinite(measured_colony_start_mm2):
            continue

        t_eval = (eval_frame - start_frame) / (final_frame_index - start_frame)

        # This is the only linear interpolation step.
        interpolated_clone_eval_mm2 = start_clone_mm2 + t_eval * (final_clone_mm2 - start_clone_mm2)

        # Original corrected-colony logic.
        adjusted_colony_eval_mm2 = (
            measured_colony_eval_mm2
            - measured_clone_eval_mm2
            + interpolated_clone_eval_mm2
        )

        evaluation_is_start_anchor = bool(start_frame == eval_frame)
        include_in_main_summary = not (
            evaluation_is_start_anchor and not INCLUDE_EVAL_ANCHOR_START_IN_SUMMARY
        )

        status = "ok"
        if evaluation_is_start_anchor:
            status = "evaluation_frame_is_start_anchor_zero_error_by_definition"

        common = {
            "source_file": timelapse_path.name,
            "final_file": final_path.name,
            "pid": pid_label,
            "start_frame_requested": start_frame_requested,
            "start_frame_used": start_frame,
            "ground_truth_frame_requested": GROUND_TRUTH_FRAME,
            "ground_truth_frame_used": eval_frame,
            "start_delta_from_ground_truth_h": float(frame_delta_from_ground_truth_hours(start_frame, eval_frame)),
            "start_time_before_ground_truth_h": float(frame_time_before_ground_truth_hours(start_frame, eval_frame)),
            "final_frame_index": final_frame_index,
            "t_at_ground_truth_frame": t_eval,
            "evaluation_is_start_anchor": evaluation_is_start_anchor,
            "include_in_main_summary": include_in_main_summary,
            "measured_colony_area_at_ground_truth_mm2": measured_colony_eval_mm2,
            "measured_clone_area_at_ground_truth_mm2": measured_clone_eval_mm2,
            "start_clone_area_mm2": start_clone_mm2,
            "final_clone_endpoint_area_mm2": final_clone_mm2,
            "interpolated_clone_area_at_ground_truth_mm2": interpolated_clone_eval_mm2,
            "adjusted_colony_area_at_ground_truth_mm2": adjusted_colony_eval_mm2,
            "status": status,
        }

        # Clone-area validation row.
        clone_metrics = error_metrics(
            estimated=interpolated_clone_eval_mm2,
            measured=measured_clone_eval_mm2,
        )
        rows.append({
            **common,
            "measurement": "clone_area",
            "error_logic": "direct_error_of_interpolated_clone_area",
            "measured_area_mm2": measured_clone_eval_mm2,
            "estimated_area_mm2": interpolated_clone_eval_mm2,
            "start_area_mm2": start_clone_mm2,
            "final_endpoint_area_mm2": final_clone_mm2,
            **clone_metrics,
        })

        # Derived adjusted-colony validation row.
        colony_metrics = error_metrics(
            estimated=adjusted_colony_eval_mm2,
            measured=measured_colony_eval_mm2,
        )
        rows.append({
            **common,
            "measurement": "adjusted_colony_area",
            "error_logic": "measured_colony_minus_measured_clone_plus_interpolated_clone",
            "measured_area_mm2": measured_colony_eval_mm2,
            "estimated_area_mm2": adjusted_colony_eval_mm2,
            # There is no direct colony interpolation. These fields are included only
            # to make the long-format table compatible with the summary functions.
            "start_area_mm2": measured_colony_start_mm2,
            "final_endpoint_area_mm2": np.nan,
            **colony_metrics,
        })

    return rows


def summarize_results(per_start: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize across files by artificial start frame and overall."""
    if per_start.empty:
        return pd.DataFrame(), pd.DataFrame()

    data = per_start[per_start["include_in_main_summary"] == True].copy()
    data = data[data["status"].isin(["ok"])].copy()

    if data.empty:
        return pd.DataFrame(), pd.DataFrame()

    metrics = [
        "signed_error_mm2",
        "abs_error_mm2",
        "percent_error",
        "abs_percent_error",
        "symmetric_percent_error",
        "symmetric_abs_percent_error",
    ]

    by_start_rows = []
    for (measurement, start_frame), group in data.groupby(["measurement", "start_frame_requested"], sort=True):
        row = {
            "measurement": measurement,
            "start_frame_requested": start_frame,
            "ground_truth_frame": int(group["ground_truth_frame_used"].iloc[0]),
            "start_delta_from_ground_truth_h": float(frame_delta_from_ground_truth_hours(start_frame, int(group["ground_truth_frame_used"].iloc[0]))),
            "start_time_before_ground_truth_h": float(frame_time_before_ground_truth_hours(start_frame, int(group["ground_truth_frame_used"].iloc[0]))),
            "n_files": int(group["source_file"].nunique()),
            "n_values": int(len(group)),
            "mean_t_at_ground_truth_frame": group["t_at_ground_truth_frame"].mean(),
        }

        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_sd"] = np.nan
                row[f"{metric}_sem"] = np.nan
                row[f"{metric}_median"] = np.nan
                row[f"{metric}_q25"] = np.nan
                row[f"{metric}_q75"] = np.nan
            else:
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_sd"] = values.std(ddof=1) if len(values) > 1 else np.nan
                row[f"{metric}_sem"] = values.sem(ddof=1) if len(values) > 1 else np.nan
                row[f"{metric}_median"] = values.median()
                row[f"{metric}_q25"] = values.quantile(0.25)
                row[f"{metric}_q75"] = values.quantile(0.75)

        by_start_rows.append(row)

    overall_rows = []
    for measurement, group in data.groupby("measurement", sort=True):
        row = {
            "measurement": measurement,
            "ground_truth_frame": int(group["ground_truth_frame_used"].iloc[0]),
            "start_frames_included": ", ".join(str(x) for x in sorted(group["start_frame_requested"].unique())),
            "start_deltas_from_ground_truth_h_included": ", ".join(
                f"{frame_delta_from_ground_truth_hours(x, int(group['ground_truth_frame_used'].iloc[0])):g}"
                for x in sorted(group["start_frame_requested"].unique())
            ),
            "start_times_before_ground_truth_h_included": ", ".join(
                f"{frame_time_before_ground_truth_hours(x, int(group['ground_truth_frame_used'].iloc[0])):g}"
                for x in sorted(group["start_frame_requested"].unique(), reverse=True)
            ),
            "n_files": int(group["source_file"].nunique()),
            "n_values": int(len(group)),
        }

        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_sd"] = np.nan
                row[f"{metric}_sem"] = np.nan
                row[f"{metric}_median"] = np.nan
                row[f"{metric}_q25"] = np.nan
                row[f"{metric}_q75"] = np.nan
            else:
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_sd"] = values.std(ddof=1) if len(values) > 1 else np.nan
                row[f"{metric}_sem"] = values.sem(ddof=1) if len(values) > 1 else np.nan
                row[f"{metric}_median"] = values.median()
                row[f"{metric}_q25"] = values.quantile(0.25)
                row[f"{metric}_q75"] = values.quantile(0.75)

        overall_rows.append(row)

    return pd.DataFrame(by_start_rows), pd.DataFrame(overall_rows)


# ───────────────────────────── OUTPUT ─────────────────────────────

def plot_metric_by_start(
    by_start: pd.DataFrame,
    out_dir: Path,
    metric: str,
    ylabel: str,
    filename: str,
) -> None:
    """
    Plot the median value for each artificial interpolation start frame as positive Δt before ground truth.

    Styling:
    - no circular markers
    - colored connecting line
    - colored IQR range from q25 to q75
    - black horizontal center line at the median value
    """
    if by_start.empty or not SAVE_PLOTS:
        return

    plot_data = by_start[by_start["measurement"].isin(PLOT_MEASUREMENTS)].copy()
    if plot_data.empty:
        return

    setup_matplotlib()

    center_col = f"{metric}_median"
    q25_col = f"{metric}_q25"
    q75_col = f"{metric}_q75"

    dpi = int(PLOT_CONFIG.get("dpi", 300))
    layout = PLOT_CONFIG.get("fixed_layout", None)
    if layout:
        fig, ax = fixed_axes(
            fig_w=layout["fig_w"],
            fig_h=layout["fig_h"],
            ax_w=layout["ax_w"],
            ax_h=layout["ax_h"],
            left=layout.get("left", 0.25),
            bottom=layout.get("bottom", 0.22),
        )
    else:
        fig, ax = plt.subplots(
            figsize=PLOT_CONFIG.get("figsize", (3.6, 3.2)),
            dpi=dpi,
        )
    fig.set_dpi(dpi)
    line_width = float(PLOT_CONFIG.get("line_width", 1.0))

    plotted_measurements = []
    for measurement, group in plot_data.groupby("measurement", sort=True):
        ground_truth_frame = int(group["ground_truth_frame"].iloc[0])
        group = group.copy()
        group["plot_time_before_ground_truth_h"] = frame_time_before_ground_truth_hours(
            group["start_frame_requested"].to_numpy(dtype=float),
            ground_truth_frame,
        )
        group = group.sort_values("plot_time_before_ground_truth_h")
        plotted_measurements.append(measurement)

        x = group["plot_time_before_ground_truth_h"].to_numpy(dtype=float)
        y = group[center_col].to_numpy(dtype=float)
        q25 = group[q25_col].to_numpy(dtype=float)
        q75 = group[q75_col].to_numpy(dtype=float)

        # Keep the visual cap widths equivalent to the previous frame-based plot:
        # 1.5 frames = 0.75 h and 2 frames = 1 h.
        iqr_cap_half_width_h = 1.5 / FRAMES_PER_HOUR
        center_line_half_width_h = 2.0 / FRAMES_PER_HOUR

        style = MEASUREMENT_STYLES.get(measurement, {})
        color = style.get("color", None)
        label = style.get("label", measurement)

        # Colored connecting line, no round markers.
        ax.plot(x, y, linewidth=line_width, color=color, label=label)

        # Colored IQR range with colored caps.
        for xi, yi, lo, hi in zip(x, y, q25, q75):
            if np.isfinite(lo) and np.isfinite(hi):
                ax.vlines(xi, lo, hi, color=color, linewidth=line_width)
                ax.hlines(
                    [lo, hi],
                    xi - iqr_cap_half_width_h,
                    xi + iqr_cap_half_width_h,
                    color=color,
                    linewidth=line_width,
                )

            # Black center line at the median.
            if np.isfinite(yi):
                ax.hlines(
                    yi,
                    xi - center_line_half_width_h,
                    xi + center_line_half_width_h,
                    color="black",
                    linewidth=line_width,
                )

    ax.axhline(0, linewidth=0.5, linestyle=":")
    ax.axvline(0, linewidth=0.5, linestyle=":")
    ax.set_xlim(left=0)
    ax.set_xlabel("Time before ground truth, Δt (h)")
    ax.set_ylabel(ylabel)
    if metric != "signed_error_mm2":
        _, ymax = ax.get_ylim()
        ax.set_ylim(bottom=0, top=ymax)
    if len(plotted_measurements) > 1:
        ax.legend(
            frameon=True,
            fontsize=6,
            handlelength=1.8,
            borderpad=0.3,
        )

    ax.tick_params(width=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    if not layout:
        fig.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, bbox_inches="tight", transparent=True)

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    print(f"[plot saved] {out_path}")


def write_report(out_dir: Path, per_start: pd.DataFrame, by_start: pd.DataFrame, overall: pd.DataFrame) -> None:
    report_path = out_dir / "original_logic_interpolation_validation_report_text.txt"

    lines = []
    lines.append("Original-logic interpolation-to-final validation report")
    lines.append("=" * 55)
    lines.append("")
    lines.append(f"Ground-truth comparison frame: {GROUND_TRUTH_FRAME}")
    lines.append(f"Artificial interpolation start frames: {START_FRAMES}")
    start_deltas_signed = [float(frame_delta_from_ground_truth_hours(f)) for f in START_FRAMES]
    start_times_before_gt = [float(frame_time_before_ground_truth_hours(f)) for f in START_FRAMES]
    lines.append(f"Artificial interpolation signed Δt values from ground truth (h): {start_deltas_signed}")
    lines.append(f"Artificial interpolation times before ground truth (h): {start_times_before_gt}")
    lines.append(f"Final frame index mode: {FINAL_FRAME_INDEX_MODE}")
    if FINAL_FRAME_INDEX_MODE == "manual":
        lines.append(f"Manual final frame index: {MANUAL_FINAL_FRAME_INDEX}")
    lines.append(f"Timelapse scale: {SEG_SCALE_UM_PER_PX} µm/px")
    lines.append(f"Final-image scale: {FINAL_SCALE_UM_PER_PX} µm/px")
    lines.append("")
    lines.append("Logic:")
    lines.append(
        "For each start frame, the script interpolated only clone area from the measured "
        "timelapse clone area at that start frame to the separate final CSV clone endpoint. "
        f"It then evaluated that interpolated clone area at frame {GROUND_TRUTH_FRAME}."
    )
    lines.append(
        "The derived colony area was calculated as: measured colony area at the ground-truth "
        "frame - measured clone area at the ground-truth frame + interpolated clone area at "
        "the ground-truth frame."
    )
    lines.append(
        "Thus, the adjusted colony-area error in mm² is algebraically identical to the clone "
        "interpolation error in mm². The percent errors differ because clone percent error is "
        "normalized to measured clone area, whereas adjusted colony percent error is normalized "
        "to total measured colony area."
    )
    lines.append("")

    if per_start.empty:
        lines.append("No valid results were generated.")
    else:
        included = per_start[per_start["include_in_main_summary"] == True]
        lines.append(f"Raw validation rows: {len(per_start)}")
        lines.append(f"Rows included in main summaries: {len(included)}")
        if not included.empty:
            lines.append(f"Files included: {included['source_file'].nunique()}")
        lines.append("")

    if not by_start.empty:
        lines.append("By-start-frame summary")
        lines.append("-" * 22)
        for _, row in by_start.sort_values(["measurement", "start_frame_requested"]).iterrows():
            measurement = row["measurement"]
            start = row["start_frame_requested"]
            delta_t = row.get("start_time_before_ground_truth_h", np.nan)
            n = row["n_files"]
            mae = row.get("abs_error_mm2_median", np.nan)
            mae_q25 = row.get("abs_error_mm2_q25", np.nan)
            mae_q75 = row.get("abs_error_mm2_q75", np.nan)
            mape = row.get("abs_percent_error_median", np.nan)
            mape_q25 = row.get("abs_percent_error_q25", np.nan)
            mape_q75 = row.get("abs_percent_error_q75", np.nan)

            lines.append(
                f"{measurement}, start {start} (time before GT Δt={delta_t:g} h), n={n}: "
                f"median absolute error = {mae:.6g} mm²"
                + (
                    f" [IQR {mae_q25:.6g}–{mae_q75:.6g}]"
                    if np.isfinite(mae_q25) and np.isfinite(mae_q75)
                    else ""
                )
                + f"; median absolute percent error = {mape:.4g}%"
                + (
                    f" [IQR {mape_q25:.4g}–{mape_q75:.4g}]"
                    if np.isfinite(mape_q25) and np.isfinite(mape_q75)
                    else ""
                )
            )
        lines.append("")

    if not overall.empty:
        lines.append("Overall summary")
        lines.append("-" * 15)
        for _, row in overall.sort_values("measurement").iterrows():
            measurement = row["measurement"]
            n_files = row["n_files"]
            n_values = row["n_values"]
            mae = row.get("abs_error_mm2_median", np.nan)
            mae_q25 = row.get("abs_error_mm2_q25", np.nan)
            mae_q75 = row.get("abs_error_mm2_q75", np.nan)
            mape = row.get("abs_percent_error_median", np.nan)
            mape_q25 = row.get("abs_percent_error_q25", np.nan)
            mape_q75 = row.get("abs_percent_error_q75", np.nan)

            lines.append(
                f"{measurement}, n_files={n_files}, n_values={n_values}: "
                f"median absolute error = {mae:.6g} mm²"
                + (
                    f" [IQR {mae_q25:.6g}–{mae_q75:.6g}]"
                    if np.isfinite(mae_q25) and np.isfinite(mae_q75)
                    else ""
                )
                + f"; median absolute percent error = {mape:.4g}%"
                + (
                    f" [IQR {mape_q25:.4g}–{mape_q75:.4g}]"
                    if np.isfinite(mape_q25) and np.isfinite(mape_q75)
                    else ""
                )
                + "."
            )
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {report_path}")


# ───────────────────────────── MAIN ─────────────────────────────

def main() -> None:
    timelapse_dir = resolve_path(INPUT_ROOT, TIMELAPSE_INPUT_RELPATH)
    finals_dir = resolve_path(INPUT_ROOT, FINALS_INPUT_RELPATH)
    out_dir = resolve_path(OUTPUT_ROOT, OUTPUT_RELPATH)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not timelapse_dir.is_dir():
        print(f"[error] Timelapse input folder not found: {timelapse_dir}")
        return

    if not finals_dir.is_dir():
        print(f"[error] Finals input folder not found: {finals_dir}")
        return

    timelapse_files = detect_timelapse_files(timelapse_dir)
    finals_map = build_finals_map(finals_dir)

    if not timelapse_files:
        print("[error] No timelapse files found.")
        return

    if not finals_map:
        print("[error] No final CSVs found.")
        return

    all_rows: List[dict] = []

    for tl_path in timelapse_files:
        pid = get_pid(tl_path)
        if pid is None:
            print(f"[skip] {tl_path.name}: no P<id> found in filename.")
            continue

        final_path = finals_map.get(pid)
        if final_path is None:
            print(f"[skip] {tl_path.name}: no matching final CSV found for P{pid}.")
            continue

        print(f"\n[processing] {tl_path.name}  ->  {final_path.name}")
        try:
            rows = validate_file(tl_path, final_path)
            all_rows.extend(rows)
        except Exception as exc:
            print(f"[error] Failed processing {tl_path.name}: {exc}")

    per_start = pd.DataFrame(all_rows)

    per_start_path = out_dir / "original_logic_interpolation_validation_per_start_errors.csv"
    by_start_path = out_dir / "original_logic_interpolation_validation_by_start_frame_summary.csv"
    overall_path = out_dir / "original_logic_interpolation_validation_overall_summary.csv"

    if per_start.empty:
        print("[warn] No validation rows generated.")
        return

    per_start.to_csv(per_start_path, index=False)
    print(f"[saved] {per_start_path}")

    by_start, overall = summarize_results(per_start)

    if not by_start.empty:
        by_start.to_csv(by_start_path, index=False)
        print(f"[saved] {by_start_path}")

    if not overall.empty:
        overall.to_csv(overall_path, index=False)
        print(f"[saved] {overall_path}")

    if SAVE_PLOTS and not by_start.empty:
        ground_truth_hour = float(frame_to_hours(GROUND_TRUTH_FRAME))

        plot_metric_by_start(
            by_start=by_start,
            out_dir=out_dir,
            metric="abs_error_mm2",
            ylabel=f"Median absolute error at {ground_truth_hour:g} h (mm²)",
            filename="original_logic_colony_only_median_abs_error_mm2_by_time_before_ground_truth.pdf",
        )
        plot_metric_by_start(
            by_start=by_start,
            out_dir=out_dir,
            metric="abs_percent_error",
            ylabel=f"Median absolute error at {ground_truth_hour:g} h (%)",
            filename="original_logic_colony_only_median_abs_percent_error_by_time_before_ground_truth.pdf",
        )
        plot_metric_by_start(
            by_start=by_start,
            out_dir=out_dir,
            metric="signed_error_mm2",
            ylabel=f"Median signed error at {ground_truth_hour:g} h (mm²)",
            filename="original_logic_colony_only_median_signed_error_mm2_by_time_before_ground_truth.pdf",
        )

    write_report(out_dir, per_start, by_start, overall)

    print("\nAll done!")
    print(f"Results folder: {out_dir}")


if __name__ == "__main__":
    main()
