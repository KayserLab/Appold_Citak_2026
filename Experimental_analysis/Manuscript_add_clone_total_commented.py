#!/usr/bin/env python3
"""
Add per-frame total clone area to the corresponding colony CSVs.

This script scans a folder for clone- and colony-level CSV files that share a
logical identifier of the form "P<digits>" (optionally followed by an underscore
in the filename). For each identifier, it computes the total clone area per
frame by summing clone `size` values and merges that into the colony CSV as
`total_clone_area`.

Expected filename patterns (examples):
    clone_data_fusion_resolved_P1_.csv
    clone_data_fusion_resolved_P11_.csv
    colony_data_P1.csv
    colony_data_P11.csv

Merge logic:
    - Determine logical id from filename: "P<digits>_?" -> "P<digits>"
      (e.g., P1 and P1_ both map to logical id "P1")
    - For each logical id:
        - choose the first clone CSV containing "clone" in the name
        - choose the first colony CSV containing "colony" in the name
        - group clone rows by `frame` and sum `size`
        - merge onto colony table by `frame` (left join)
        - missing totals are filled with 0
    - Write either:
        - overwrite original colony file, or
        - a new file with suffix "_with_clonearea"
"""

import os
import re

import pandas as pd


# ───────────────────────────── CONFIG ─────────────────────────────
# Input and output roots relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_ROOT = os.path.join(SCRIPT_DIR, "Input_files")
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "Output_files")

# Folder inside Input_files/ containing the experiment data
input_relpath = "files_for_extrapolation"

# If True: overwrite colony files. If False: write *_with_clonearea.csv
OVERWRITE_EXISTING = False

# ───────────────────────────── Helpers ─────────────────────────────
def scan_folder_for_pairs(folder: str):
    """
    Scan a folder for CSV files and build mappings:
        logical_id -> clone_csv_path
        logical_id -> colony_csv_path

    The logical_id is always like "P1", "P2", "P11" (no trailing underscore).
    If multiple clone/colony CSVs exist for the same logical_id, the first match
    is kept and later matches are ignored (with a warning).
    """
    clone_files = {}
    colony_files = {}

    # Match P<number> optionally followed by underscore, anywhere in the name.
    id_pattern = re.compile(r"P(\d{1,2})_?", re.IGNORECASE)

    for fname in os.listdir(folder):
        if not fname.lower().endswith(".csv"):
            continue

        m = id_pattern.search(fname)
        if not m:
            continue

        num_str = m.group(1)        # e.g. "1", "2", "11"
        logical_id = f"P{num_str}"  # e.g. "P1", "P2", "P11"
        full_path = os.path.join(folder, fname)
        lower = fname.lower()

        if "clone" in lower:
            if logical_id not in clone_files:
                clone_files[logical_id] = full_path
            else:
                print(
                    f"[warn] Multiple clone files for {logical_id}, keeping first: "
                    f"{os.path.basename(clone_files[logical_id])}"
                )
        elif "colony" in lower:
            if logical_id not in colony_files:
                colony_files[logical_id] = full_path
            else:
                print(
                    f"[warn] Multiple colony files for {logical_id}, keeping first: "
                    f"{os.path.basename(colony_files[logical_id])}"
                )

    all_ids = sorted(
        set(clone_files.keys()) | set(colony_files.keys()),
        key=lambda x: int(x[1:]),  # sort by numeric part
    )

    if not all_ids:
        print("[warn] No IDs found (no clone/colony CSVs with 'P<digits>' in name).")
        return [], {}, {}

    print("[info] Found IDs:", ", ".join(all_ids))
    for lid in all_ids:
        print(
            f"  {lid}: "
            f"clone={os.path.basename(clone_files.get(lid, 'MISSING'))}, "
            f"colony={os.path.basename(colony_files.get(lid, 'MISSING'))}"
        )

    return all_ids, clone_files, colony_files


def add_total_clone_area(clone_path: str, colony_path: str, out_path: str):
    """
    Compute total clone area per frame from clone_path and merge into colony_path.

    Requirements (clone CSV):
        - must contain columns: 'frame', 'particle', 'size'

    Requirements (colony CSV):
        - should contain a 'frame' column; if absent, row index is treated as frame number.

    Output:
        - writes merged CSV to out_path
    """
    print("\nProcessing logical pair:")
    print(f"  clone : {clone_path}")
    print(f"  colony: {colony_path}")

    clone_df = pd.read_csv(clone_path)
    required_cols = {"frame", "particle", "size"}
    if not required_cols.issubset(clone_df.columns):
        raise ValueError(
            f"{os.path.basename(clone_path)} missing required columns "
            f"{required_cols}; found {set(clone_df.columns)}"
        )

    # Total clone area per frame
    clone_sum = clone_df.groupby("frame")["size"].sum().rename("total_clone_area")

    colony_df = pd.read_csv(colony_path)

    # Ensure colony has a 'frame' column to join on.
    # If it doesn't exist, assume row index == frame number (0..N-1).
    if "frame" not in colony_df.columns:
        colony_df = colony_df.reset_index().rename(columns={"index": "frame"})

    merged = pd.merge(colony_df, clone_sum, on="frame", how="left")
    merged["total_clone_area"] = merged["total_clone_area"].fillna(0)

    merged.to_csv(out_path, index=False)
    print(f"[saved] {out_path}")


def iter_csv_folders(root: str):
    """
    Yield folders under `root` (including root itself) that contain at least one CSV file.

    This supports a layout where Input_files may contain nested experiment subfolders.
    """
    for current_dir, _subdirs, files in os.walk(root):
        if any(f.lower().endswith(".csv") for f in files):
            yield current_dir


# ───────────────────────────── Main ─────────────────────────────
def main():
    if not os.path.isdir(INPUT_ROOT):
        print(f"[error] Input folder not found: {INPUT_ROOT}")
        print("Create it next to this script and place your CSVs inside (subfolders allowed).")
        return

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Process each input subfolder independently (mirrors typical experiment folder layout).
    any_processed = False
    for in_dir in iter_csv_folders(INPUT_ROOT):
        rel = os.path.relpath(in_dir, INPUT_ROOT)
        out_dir = OUTPUT_ROOT
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n[info] Scanning input folder: {in_dir}")
        ids, clone_files, colony_files = scan_folder_for_pairs(in_dir)
        if not ids:
            continue

        any_processed = True

        for lid in ids:
            clone_path = clone_files.get(lid)
            colony_path = colony_files.get(lid)

            if not clone_path or not colony_path:
                print(f"[warn] Skipping {lid}: clone or colony file missing.")
                continue

            colony_name = os.path.basename(colony_path)

            if OVERWRITE_EXISTING:
                out_path = os.path.join(out_dir, colony_name)
            else:
                base, ext = os.path.splitext(colony_name)
                out_path = os.path.join(out_dir, f"{base}_with_clonearea{ext}")

            add_total_clone_area(clone_path, colony_path, out_path)

    if not any_processed:
        print(f"[warn] No CSV files found under: {INPUT_ROOT}")
        return

    print("\nAll done!")


if __name__ == "__main__":
    main()