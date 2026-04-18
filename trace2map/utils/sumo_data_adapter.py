"""
sumo_data_adapter.py
Convert dataset_collector SUMO output into the format expected by
the trace2map preprocess_dataset / generate_sampled_scenes pipeline.

Input layout (dataset_collector/output/<scene>/):
    trajectories.csv  - columns: scenario, veh_id, seed, t, x, y, vx, vy,
                                  lane_id, vehicle_type
                        coordinates are already in local frame
                        (x = travel direction, y = lateral-left positive)
    map_meta.json     - bounds, pixels_per_meter, coordinate_system, ...

Output layout (data/sumo_data/):
    scenario_metadata.csv          - compatible with preprocess_dataset
    <scene_name>/
        veh_1.csv                  - columns: loc_x(m), loc_y(m)
        veh_2.csv
        ...

Because SUMO trajectories are already in local frame (no yaw rotation
needed), we set wz_start_x=0, wz_start_y=0, wz_start_yaw=0 in the
generated metadata so that preprocess_dataset's rotation matrix
degenerates to the identity and the coordinates pass through unchanged.
"""

from __future__ import annotations

import json
import os
import re

import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_num_lanes(scene_name: str) -> int:
    """Extract total lane count from scene directory name.

    Examples:
        '4L_close_1'     -> 4
        '4L_close_1_2_3' -> 4
    """
    match = re.search(r'(\d+)L', scene_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    print(f"[sumo_adapter] WARNING: cannot parse lane count from '{scene_name}', defaulting to 4")
    return 4


def _build_metadata_row(scene_name: str, map_meta: dict) -> dict:
    """Build one row for scenario_metadata.csv from map_meta.json content.

    All fields that preprocess_dataset actually uses are set correctly.
    wz_start_x / wz_start_y / wz_start_yaw are set to 0 because the
    SUMO trajectories are already in local frame.
    """
    bounds = map_meta.get("bounds", {})
    xmin = bounds.get("xmin", 0.0)
    xmax = bounds.get("xmax", 0.0)
    ymin = bounds.get("ymin", 0.0)
    ymax = bounds.get("ymax", 0.0)

    wz_length_m = float(xmax - xmin)
    road_width_m = float(ymax - ymin)
    num_lanes = _parse_num_lanes(scene_name)

    return {
        "scenario_name": scene_name,
        "map_name": "sumo",
        "num_lanes": num_lanes,
        "closed_lanes": "",           # not used by CVAE preprocessing
        "wz_length_m": round(wz_length_m, 4),
        "road_width_m": round(road_width_m, 4),
        "wz_start_x": 0.0,           # local frame origin = road start
        "wz_start_y": 0.0,
        "wz_start_z": 0.0,
        "wz_start_yaw": 0.0,          # identity rotation
        "wz_start_pitch": 0.0,
        "wz_start_roll": 0.0,
    }


def _split_and_save_vehicles(
    traj_csv_path: str,
    scene_out_dir: str,
) -> int:
    """Read a scene's trajectories.csv, split by vehicle, write one CSV per vehicle.

    Each output CSV has columns: loc_x(m), loc_y(m)
    (column names expected by data_utils.preprocess_dataset / rasterize_csv_files)

    Returns:
        Number of vehicle CSV files written.
    """
    df = pd.read_csv(traj_csv_path)

    required = {"veh_id", "x", "y"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"trajectories.csv at '{traj_csv_path}' is missing columns: "
            f"{required - set(df.columns)}"
        )

    os.makedirs(scene_out_dir, exist_ok=True)

    veh_ids = sorted(df["veh_id"].unique())
    for veh_id in veh_ids:
        veh_df = df[df["veh_id"] == veh_id][["x", "y"]].copy()
        veh_df = veh_df.rename(columns={"x": "loc_x(m)", "y": "loc_y(m)"})
        out_path = os.path.join(scene_out_dir, f"veh_{int(veh_id)}.csv")
        veh_df.to_csv(out_path, index=False)

    return len(veh_ids)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_sumo_to_cvae_format(
    collector_output_dir: str,
    sumo_data_dir: str,
) -> tuple[str, str]:
    """Convert all SUMO dataset_collector outputs to CVAE-compatible format.

    Parameters
    ----------
    collector_output_dir : str
        Path to dataset_collector/output/ (contains one subdirectory per scene).
    sumo_data_dir : str
        Destination root (e.g. 'data/sumo_data').
        Created if it does not exist.

    Returns
    -------
    (sumo_data_dir, metadata_path) : tuple[str, str]
        Paths to the converted data root and the generated metadata CSV.
    """
    collector_output_dir = os.path.abspath(collector_output_dir)
    sumo_data_dir = os.path.abspath(sumo_data_dir)

    if not os.path.isdir(collector_output_dir):
        raise FileNotFoundError(
            f"collector_output_dir not found: '{collector_output_dir}'"
        )

    os.makedirs(sumo_data_dir, exist_ok=True)
    metadata_path = os.path.join(sumo_data_dir, "scenario_metadata.csv")

    print("\n" + "=" * 60)
    print("[sumo_adapter] Converting SUMO outputs -> CVAE format")
    print(f"  Source : {collector_output_dir}")
    print(f"  Dest   : {sumo_data_dir}")
    print("=" * 60)

    metadata_rows = []
    total_vehicles = 0
    processed_scenes = 0

    for scene_name in sorted(os.listdir(collector_output_dir)):
        scene_dir = os.path.join(collector_output_dir, scene_name)
        if not os.path.isdir(scene_dir):
            continue

        traj_csv = os.path.join(scene_dir, "trajectories.csv")
        map_meta_json = os.path.join(scene_dir, "map_meta.json")

        if not os.path.isfile(traj_csv):
            print(f"[sumo_adapter] SKIP '{scene_name}': no trajectories.csv")
            continue

        # Load map_meta (required for metadata generation)
        if not os.path.isfile(map_meta_json):
            print(f"[sumo_adapter] SKIP '{scene_name}': no map_meta.json")
            continue

        with open(map_meta_json, "r", encoding="utf-8") as f:
            map_meta = json.load(f)

        # 1. Split trajectories into per-vehicle CSVs
        scene_out_dir = os.path.join(sumo_data_dir, scene_name)
        try:
            n_vehs = _split_and_save_vehicles(traj_csv, scene_out_dir)
        except Exception as exc:
            print(f"[sumo_adapter] ERROR processing '{scene_name}': {exc}")
            continue

        # 2. Build metadata row
        meta_row = _build_metadata_row(scene_name, map_meta)
        metadata_rows.append(meta_row)

        total_vehicles += n_vehs
        processed_scenes += 1
        print(f"[sumo_adapter]   {scene_name}: {n_vehs} vehicles written")

    if not metadata_rows:
        print("[sumo_adapter] WARNING: no scenes were converted.")
        return sumo_data_dir, metadata_path

    # 3. Save scenario_metadata.csv
    meta_df = pd.DataFrame(metadata_rows)
    meta_cols = [
        "scenario_name", "map_name", "num_lanes", "closed_lanes",
        "wz_length_m", "road_width_m",
        "wz_start_x", "wz_start_y", "wz_start_z",
        "wz_start_yaw", "wz_start_pitch", "wz_start_roll",
    ]
    meta_df = meta_df[meta_cols]
    meta_df.to_csv(metadata_path, index=False)

    print("\n" + "=" * 60)
    print(f"[sumo_adapter] Done. {processed_scenes} scenes, {total_vehicles} vehicles.")
    print(f"  Metadata : {metadata_path}")
    print(f"  Data root: {sumo_data_dir}")
    print("=" * 60 + "\n")

    return sumo_data_dir, metadata_path
