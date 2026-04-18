# Multi-Scene Conditional CVAE - Data Processing Utilities

import numpy as np
import pandas as pd
import torch
import cv2
import os
import json
import re
from tqdm import tqdm

FT_TO_M = 0.3048
GRID_H = 512
GRID_W = 128


# ==========================================
# ==========================================

def compute_pca_alignment(x_world, y_world):
    """
    Map-free alignment: centroid shift + PCA principal direction rotation.

    Args:
        x_world: world X coordinates (N,)
        y_world: world Y coordinates (N,)

    Returns:
        alignment_info: dict {
            'centroid_x': float,
            'centroid_y': float,
            'rotation_angle_rad': float
        }
    """
    cx = np.mean(x_world)
    cy = np.mean(y_world)

    x_centered = x_world - cx
    y_centered = y_world - cy

    cov_matrix = np.cov(np.vstack([x_centered, y_centered]))
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    principal_idx = np.argmax(eigenvalues)
    principal_direction = eigenvectors[:, principal_idx]

    angle_rad = np.arctan2(principal_direction[1], principal_direction[0])

    return {
        'centroid_x': float(cx),
        'centroid_y': float(cy),
        'rotation_angle_rad': float(angle_rad)
    }


def apply_alignment(x_world, y_world, alignment_info):
    """
    Apply alignment transform: centroid shift + rotation.

    Args:
        x_world, y_world: world coordinates (N,)
        alignment_info: dict returned by compute_pca_alignment()

    Returns:
        x_local, y_local: aligned local coordinates (N,)
    """
    cx = alignment_info['centroid_x']
    cy = alignment_info['centroid_y']
    angle = alignment_info['rotation_angle_rad']

    dx = x_world - cx
    dy = y_world - cy

    cos_theta = np.cos(-angle)
    sin_theta = np.sin(-angle)

    x_local = dx * cos_theta - dy * sin_theta
    y_local = dx * sin_theta + dy * cos_theta

    return x_local, y_local


def compute_workzone_bounds(wz_length_m, road_width_m,
                            longitudinal_padding=10.0, lateral_padding=10.0):
    """
    Compute bounds based on workzone length and road width.

    Lateral (y) coordinate assumption: y=0 at right road edge, y increases
    leftward up to road_width_m. Suitable for SUMO-style coordinate systems
    (origin: rightmost lane edge, y lateral toward left).

    Args:
        wz_length_m: workzone length (meters)
        road_width_m: road width (meters)
        longitudinal_padding: longitudinal extension distance (meters)
        lateral_padding: lateral extension distance (meters)

    Returns:
        bounds: dict {'xmin', 'xmax', 'ymin', 'ymax'}
    """
    x_min = -longitudinal_padding
    x_max = wz_length_m + longitudinal_padding
    y_min = 0.0 - lateral_padding
    y_max = road_width_m + lateral_padding

    return {
        'xmin': float(x_min),
        'xmax': float(x_max),
        'ymin': float(y_min),
        'ymax': float(y_max)
    }


def compute_global_bounds(all_aligned_points_list, longitudinal_padding=10.0, lateral_padding=10.0, percentile_clip=True):
    """
    Compute global bounds over all aligned point clouds from training scenarios.

    Args:
        all_aligned_points_list: list of (x_local, y_local) tuples
        longitudinal_padding: longitudinal (x-direction) boundary extension (meters)
        lateral_padding: lateral (y-direction) boundary extension (meters)
        percentile_clip: whether to use percentile clipping for outlier removal

    Returns:
        global_bounds: dict {
            'xmin': float, 'xmax': float,
            'ymin': float, 'ymax': float
        }
    """
    all_x = np.concatenate([pts[0] for pts in all_aligned_points_list])
    all_y = np.concatenate([pts[1] for pts in all_aligned_points_list])

    if percentile_clip:
        xmin = np.percentile(all_x, 1) - longitudinal_padding
        xmax = np.percentile(all_x, 99) + longitudinal_padding
        ymin = np.percentile(all_y, 1) - lateral_padding
        ymax = np.percentile(all_y, 99) + lateral_padding
    else:
        xmin = np.min(all_x) - longitudinal_padding
        xmax = np.max(all_x) + longitudinal_padding
        ymin = np.min(all_y) - lateral_padding
        ymax = np.max(all_y) + lateral_padding

    return {
        'xmin': float(xmin),
        'xmax': float(xmax),
        'ymin': float(ymin),
        'ymax': float(ymax)
    }


# ==========================================
# ==========================================

def rasterize_trajectories_v2(x_local, y_local, global_bounds):
    """
    Rasterize using global bounds (replaces old version that relied on meta_row).

    Args:
        x_local, y_local: aligned local coordinates (N,)
        global_bounds: dict with keys ['xmin', 'xmax', 'ymin', 'ymax']

    Returns:
        heatmap: (GRID_H, GRID_W) normalized heatmap
    """
    xmin = global_bounds['xmin']
    xmax = global_bounds['xmax']
    ymin = global_bounds['ymin']
    ymax = global_bounds['ymax']

    x_norm = (x_local - xmin) / (xmax - xmin)
    y_norm = (y_local - ymin) / (ymax - ymin)

    mask = (x_norm >= 0) & (x_norm < 1) & (y_norm >= 0) & (y_norm < 1)
    x_valid = x_norm[mask]
    y_valid = y_norm[mask]

    img_r = ((1.0 - x_valid) * (GRID_H - 1)).astype(np.int32)
    img_c = (y_valid * (GRID_W - 1)).astype(np.int32)

    heatmap = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    np.add.at(heatmap, (img_r, img_c), 1.0)

    heatmap = cv2.GaussianBlur(heatmap, (9, 9), 2.0)

    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap


# ==========================================
# ==========================================

def transform_to_local(df_traj, meta_row):
    """
    [DEPRECATED] Old version: coordinate transform using metadata WZ start point.
    Use compute_pca_alignment + apply_alignment instead.
    (Kept only for backward compatibility.)
    """
    ref_x = meta_row['wz_start_x']
    ref_y = meta_row['wz_start_y']
    ref_yaw_deg = meta_row['wz_start_yaw']
    ref_yaw_rad = np.deg2rad(ref_yaw_deg)

    dx = df_traj['loc_x(m)'].values - ref_x
    dy = df_traj['loc_y(m)'].values - ref_y

    cos_theta = np.cos(-ref_yaw_rad)
    sin_theta = np.sin(-ref_yaw_rad)

    x_local = dx * cos_theta - dy * sin_theta
    y_local = dx * sin_theta + dy * cos_theta

    return x_local, y_local


def rasterize_trajectories(x_local, y_local, meta_row):
    """
    Rasterize local coordinate points into a 512x128 heatmap.
    """
    wz_len = meta_row['wz_length_m']
    road_width = meta_row['road_width_m']

    x_min = -ROI_FRONT_BACK_EXT_M
    x_max = wz_len + ROI_FRONT_BACK_EXT_M

    y_min = -(road_width / 2.0) - ROI_SIDE_EXT_M
    y_max = (road_width / 2.0) + ROI_SIDE_EXT_M

    x_norm = (x_local - x_min) / (x_max - x_min)
    y_norm = (y_local - y_min) / (y_max - y_min)

    mask = (x_norm >= 0) & (x_norm < 1) & (y_norm >= 0) & (y_norm < 1)
    x_valid = x_norm[mask]
    y_valid = y_norm[mask]

    img_r = ((1.0 - x_valid) * (GRID_H - 1)).astype(np.int32)
    img_c = (y_valid * (GRID_W - 1)).astype(np.int32)

    heatmap = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    np.add.at(heatmap, (img_r, img_c), 1.0)

    heatmap = cv2.GaussianBlur(heatmap, (9, 9), 2.0)

    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap


# ==========================================
# ==========================================

def parse_num_lanes_from_scenario_name(scenario_name):
    """
    Parse number of lanes from scenario name.
    e.g.: 'Town04_4L_Close_1_2_3' -> 4
          'Town03_2L_Close_1' -> 2
    """
    match = re.search(r'(\d+)L', scenario_name)
    if match:
        return int(match.group(1))
    else:
        print(f"Warning: Cannot parse lane count from scenario name '{scenario_name}', defaulting to 4")
        return 4


def preprocess_dataset(raw_data_dir, output_dir, scenarios_to_process=None, metadata_path=None,
                       longitudinal_padding=10.0, lateral_padding=10.0):
    """
    [Refactored] Multi-scenario preprocessing: supports automatic scenario scanning
    from the dataset directory.

    Key improvements:
    1. Restored dependency on metadata.csv for workzone information
    2. Infer scenario_name from CSV or directory name
    3. Parse num_lanes from scenario_name
    4. Workzone start point alignment + workzone-based global bounds

    Args:
        raw_data_dir: dataset root directory
        output_dir: output directory (saves .pt and global_bounds.json)
        scenarios_to_process: optional, specify which scenarios to process; None processes all
        metadata_path: path to scenario_metadata.csv file (required)
        longitudinal_padding: longitudinal (x-direction) boundary extension (meters)
        lateral_padding: lateral (y-direction) boundary extension (meters)

    Returns:
        count: number of successfully processed scenarios
    """
    print(f"\n{'='*70}")
    print(f"[Multi-scenario preprocessing] Scanning dataset: {raw_data_dir}")
    print(f"{'='*70}")

    if metadata_path is None:
        possible_paths = [
            os.path.join(os.path.dirname(raw_data_dir), 'scenario_metadata.csv'),
            os.path.join(raw_data_dir, 'scenario_metadata.csv'),
            'scenario_metadata.csv',
            'data/scenario_metadata.csv'
        ]
        metadata_path = None
        for path in possible_paths:
            if os.path.exists(path):
                metadata_path = path
                break

        if metadata_path is None:
            raise FileNotFoundError(
                "Cannot find scenario_metadata.csv!\n"
                "Please specify the metadata_path argument or ensure the file is in one of:\n" +
                "\n".join(f"  - {p}" for p in possible_paths)
            )

    print(f"\n[0/6] Loading metadata file: {metadata_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata_df = pd.read_csv(metadata_path)
    print(f"      Loaded {len(metadata_df)} metadata records")

    metadata_dict = {}
    for _, row in metadata_df.iterrows():
        sc_name = row['scenario_name']
        metadata_dict[sc_name] = {
            'wz_length_m': float(row['wz_length_m']),
            'road_width_m': float(row['road_width_m']),
            'wz_start_x': float(row['wz_start_x']),
            'wz_start_y': float(row['wz_start_y']),
            'wz_start_yaw': float(row['wz_start_yaw']),
            'num_lanes': int(row['num_lanes'])
        }

    print(f"\n[1/6] Scanning CSV files...")
    traj_files = []
    for root, _, files in os.walk(raw_data_dir):
        for f in files:
            if f.endswith('.csv') and 'metadata' not in f.lower():
                traj_files.append(os.path.join(root, f))

    print(f"      Found {len(traj_files)} trajectory files")

    if len(traj_files) == 0:
        print(f"\nError: No CSV files found in '{raw_data_dir}'!")
        return 0

    print(f"\n[2/6] Aggregating data by scenario...")
    scenario_data = {}

    for fpath in tqdm(traj_files, desc="      Reading CSVs"):
        try:
            df = pd.read_csv(fpath)

            if 'scenario_name' in df.columns:
                sc_name = df['scenario_name'].iloc[0]
            else:
                parent_dir = os.path.basename(os.path.dirname(fpath))
                sc_name = parent_dir

            x = df['loc_x(m)'].values
            y = df['loc_y(m)'].values

            if sc_name not in scenario_data:
                scenario_data[sc_name] = []
            scenario_data[sc_name].append((x, y))

        except Exception as e:
            print(f"\n      Warning: Failed to read file {fpath}: {e}")
            continue

    print(f"      Found {len(scenario_data)} scenarios")

    if scenarios_to_process is not None:
        print(f"\n      [Filter] Processing only {len(scenarios_to_process)} specified scenarios")
        filtered_data = {k: v for k, v in scenario_data.items() if k in scenarios_to_process}
        not_found = set(scenarios_to_process) - set(scenario_data.keys())
        if not_found:
            print(f"      Warning: The following scenarios were not found: {not_found}")
        scenario_data = filtered_data

    if not scenario_data:
        print("\nError: No scenarios to process!")
        return 0

    missing_metadata = []
    for sc_name in scenario_data.keys():
        if sc_name not in metadata_dict:
            missing_metadata.append(sc_name)

    if missing_metadata:
        print(f"\nWarning: The following scenarios are missing metadata: {missing_metadata}")
        print("      These scenarios will be skipped")
        scenario_data = {k: v for k, v in scenario_data.items() if k in metadata_dict}

    if not scenario_data:
        print("\nError: No scenarios to process (all scenarios are missing metadata)!")
        return 0

    print(f"\n[3/6] Coordinate transform using workzone start point...")
    scenario_metadata_info = {}
    all_aligned_points = []
    scenario_num_lanes = {}

    for sc_name, data_list in tqdm(scenario_data.items(), desc="      Transforming scenarios"):
        meta = metadata_dict[sc_name]
        scenario_metadata_info[sc_name] = meta

        x_all = np.concatenate([d[0] for d in data_list])
        y_all = np.concatenate([d[1] for d in data_list])

        ref_x = meta['wz_start_x']
        ref_y = meta['wz_start_y']
        ref_yaw_deg = meta['wz_start_yaw']
        ref_yaw_rad = np.deg2rad(ref_yaw_deg)

        dx = x_all - ref_x
        dy = y_all - ref_y

        cos_theta = np.cos(-ref_yaw_rad)
        sin_theta = np.sin(-ref_yaw_rad)

        x_local = dx * cos_theta - dy * sin_theta
        y_local = dx * sin_theta + dy * cos_theta

        all_aligned_points.append((x_local, y_local))
        scenario_num_lanes[sc_name] = meta['num_lanes']

    print(f"\n[4/6] Computing global bounds (based on workzone info)...")

    scenario_bounds = []
    for sc_name in scenario_data.keys():
        meta = scenario_metadata_info[sc_name]
        bounds = compute_workzone_bounds(
            meta['wz_length_m'],
            meta['road_width_m'],
            longitudinal_padding=longitudinal_padding,
            lateral_padding=lateral_padding
        )
        scenario_bounds.append(bounds)

    xmin_list = [b['xmin'] for b in scenario_bounds]
    xmax_list = [b['xmax'] for b in scenario_bounds]
    ymin_list = [b['ymin'] for b in scenario_bounds]
    ymax_list = [b['ymax'] for b in scenario_bounds]

    global_bounds = {
        'xmin': float(np.min(xmin_list)),
        'xmax': float(np.max(xmax_list)),
        'ymin': float(np.min(ymin_list)),
        'ymax': float(np.max(ymax_list))
    }

    print(f"      Global bounds: xmin={global_bounds['xmin']:.2f}m, xmax={global_bounds['xmax']:.2f}m")
    print(f"                     ymin={global_bounds['ymin']:.2f}m, ymax={global_bounds['ymax']:.2f}m")
    print(f"      Computed from workzone info of {len(scenario_bounds)} scenarios")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    bounds_path = os.path.join(output_dir, 'global_bounds.json')
    with open(bounds_path, 'w') as f:
        json.dump(global_bounds, f, indent=2)
    print(f"      Saved global bounds to: {bounds_path}")

    print(f"\n[5/6] Rasterizing and saving...")
    count = 0

    for sc_name, data_list in tqdm(scenario_data.items(), desc="      Generating heatmaps"):
        x_all = np.concatenate([d[0] for d in data_list])
        y_all = np.concatenate([d[1] for d in data_list])

        meta = scenario_metadata_info[sc_name]
        ref_x = meta['wz_start_x']
        ref_y = meta['wz_start_y']
        ref_yaw_deg = meta['wz_start_yaw']
        ref_yaw_rad = np.deg2rad(ref_yaw_deg)

        dx = x_all - ref_x
        dy = y_all - ref_y

        cos_theta = np.cos(-ref_yaw_rad)
        sin_theta = np.sin(-ref_yaw_rad)

        x_local = dx * cos_theta - dy * sin_theta
        y_local = dx * sin_theta + dy * cos_theta

        heatmap = rasterize_trajectories_v2(x_local, y_local, global_bounds)

        data_pkg = {
            'heatmap': torch.from_numpy(heatmap).unsqueeze(0).float(),  # [1, 512, 128]
            'num_lanes': scenario_num_lanes[sc_name],
            'scenario_name': sc_name,
            'alignment_info': {
                'wz_start_x': ref_x,
                'wz_start_y': ref_y,
                'wz_start_yaw': ref_yaw_deg
            }
        }

        safe_name = "".join([c if c.isalnum() or c in ('-', '_') else '_' for c in sc_name])
        torch.save(data_pkg, os.path.join(output_dir, f"{safe_name}_GT.pt"))
        count += 1

    print(f"\n{'='*70}")
    print(f"Preprocessing complete! Successfully processed {count} scenarios")
    print(f"  Output directory: {output_dir}")
    print(f"  Scenario list: {list(scenario_data.keys())[:5]}{'...' if len(scenario_data) > 5 else ''}")
    print(f"{'='*70}\n")

    return count


def rasterize_csv_files(csv_file_paths, alignment_info, global_bounds):
    """
    Read multiple CSV files and rasterize into an obs_heatmap.

    Args:
        csv_file_paths: list[str] list of CSV file paths
        alignment_info: dict alignment parameters (wz_start_x, wz_start_y, wz_start_yaw)
        global_bounds: dict global bounds (xmin, xmax, ymin, ymax)

    Returns:
        obs_heatmap: np.ndarray [512, 128] rasterized heatmap
    """
    all_x = []
    all_y = []

    for csv_path in csv_file_paths:
        try:
            df = pd.read_csv(csv_path)
            x = df['loc_x(m)'].values
            y = df['loc_y(m)'].values
            all_x.append(x)
            all_y.append(y)
        except Exception as e:
            print(f"Warning: Failed to read CSV file {csv_path}: {e}")
            continue

    if not all_x:
        return np.zeros((GRID_H, GRID_W), dtype=np.float32)

    x_all = np.concatenate(all_x)
    y_all = np.concatenate(all_y)

    ref_x = alignment_info['wz_start_x']
    ref_y = alignment_info['wz_start_y']
    ref_yaw_deg = alignment_info['wz_start_yaw']
    ref_yaw_rad = np.deg2rad(ref_yaw_deg)

    dx = x_all - ref_x
    dy = y_all - ref_y

    cos_theta = np.cos(-ref_yaw_rad)
    sin_theta = np.sin(-ref_yaw_rad)

    x_local = dx * cos_theta - dy * sin_theta
    y_local = dx * sin_theta + dy * cos_theta

    obs_heatmap = rasterize_trajectories_v2(x_local, y_local, global_bounds)

    return obs_heatmap
