"""
Generate CVAE drivable-area heatmaps aligned to the map_mask pixel grid.

Supports two modes:
  --heatmap_mode fixed    : one heatmap per scenario from first N vehicles
  --heatmap_mode dynamic  : epoch-based heatmaps updated as vehicles complete

Usage:
    python integration/preprocessing/heatmap_generator.py \
        --scenarios_dir dataset_collector/output \
        --cvae_ckpt "trace2map/cvae_best.pth" \
        --heatmap_mode fixed \
        --output_dir integration/data/heatmaps
"""
import sys
import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import map_coordinates

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Add CVAE project to path for model & utils imports
_CVAE_DIR = os.path.join(_PROJECT_ROOT, 'trace2map')
if _CVAE_DIR not in sys.path:
    sys.path.insert(0, _CVAE_DIR)


# -----------------------------------------------------------------------
# Coordinate alignment helpers
# -----------------------------------------------------------------------
# CVAE heatmap grid: [512, 128] covering global_bounds
# Map mask grid: [W_px, H_px] at pixels_per_meter (e.g., 5001 x 76)
# Both are in local frame but at different resolutions and extents.

def load_global_bounds():
    """Load the CVAE training global bounds."""
    candidates = [
        os.path.join(_CVAE_DIR, 'data', 'sumo_data', 'global_bounds.json'),
        os.path.join(_CVAE_DIR, 'processed_data', 'global_bounds.json'),
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p, 'r') as f:
                return json.load(f)
    raise FileNotFoundError(
        "Cannot find global_bounds.json.  Searched:\n  " +
        "\n  ".join(candidates))


def compute_heatmap_homography(global_bounds):
    """
    Compute homography that maps scene (x, y) to heatmap pixel (x_dim, y_dim).

    CVAE convention (rasterize_trajectories_v2):
      row = (1 - x_norm) * 511  ->  x_dim = 512 * (xmax - x) / (xmax - xmin)
      col = y_norm * 127       ->  y_dim = 128 * (y - ymin) / (ymax - ymin)

    Returns 3x3 numpy array for GeometricMap.to_map_points().
    """
    xmin = global_bounds['xmin']
    xmax = global_bounds['xmax']
    ymin = global_bounds['ymin']
    ymax = global_bounds['ymax']
    x_range = xmax - xmin
    y_range = ymax - ymin
    H = np.array([
        [-512.0 / x_range, 0.0, 512.0 * xmax / x_range],
        [0.0, 128.0 / y_range, -128.0 * ymin / y_range],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return H


def resample_heatmap_to_map_grid(heatmap_512x128, global_bounds, map_meta):
    """
    Resample a CVAE heatmap [512, 128] to the map_mask pixel grid [W_px, H_px].

    The CVAE heatmap covers global_bounds (xmin..xmax, ymin..ymax) at 512x128.
    The map_mask covers map_meta bounds at pixels_per_meter resolution.

    CVAE rasterization convention (from rasterize_trajectories_v2):
      row_idx = (1 - (x - xmin) / (xmax - xmin)) * 512   (x is inverted -> row 0 = xmax)
      col_idx = (y - ymin) / (ymax - ymin) * 128

    Map mask convention:
      pixel_x = scene_x * pixels_per_meter   (axis 0 = x)
      pixel_y = scene_y * pixels_per_meter   (axis 1 = y)

    Returns [W_px, H_px] float32 array in [0, 1].
    """
    ppm = map_meta['pixels_per_meter']
    W_px = map_meta['W_px']
    H_px = map_meta['H_px']
    map_xmin = map_meta['bounds']['xmin']
    map_ymin = map_meta['bounds']['ymin']

    gb_xmin = global_bounds['xmin']
    gb_xmax = global_bounds['xmax']
    gb_ymin = global_bounds['ymin']
    gb_ymax = global_bounds['ymax']

    # For each pixel (ix, iy) in the map_mask grid, find the scene coordinate
    # and then find the corresponding location in the CVAE heatmap.
    ix = np.arange(W_px, dtype=np.float64)
    iy = np.arange(H_px, dtype=np.float64)

    # Scene coordinates of each map pixel
    scene_x = map_xmin + ix / ppm  # shape [W_px]
    scene_y = map_ymin + iy / ppm  # shape [H_px]

    # CVAE heatmap coordinates (continuous)
    # row = (1 - (scene_x - gb_xmin) / (gb_xmax - gb_xmin)) * 512
    hm_row = (1.0 - (scene_x - gb_xmin) / (gb_xmax - gb_xmin)) * 512.0  # [W_px]
    # col = (scene_y - gb_ymin) / (gb_ymax - gb_ymin) * 128
    hm_col = (scene_y - gb_ymin) / (gb_ymax - gb_ymin) * 128.0  # [H_px]

    # Build meshgrid: [W_px, H_px] -> coordinates in the [512, 128] heatmap
    hm_row_2d, hm_col_2d = np.meshgrid(hm_row, hm_col, indexing='ij')

    # Bilinear interpolation from CVAE heatmap
    coords = np.array([hm_row_2d.ravel(), hm_col_2d.ravel()])
    resampled = map_coordinates(heatmap_512x128.astype(np.float64),
                                coords, order=1, mode='constant', cval=0.0)
    resampled = resampled.reshape(W_px, H_px).astype(np.float32)

    # Clip to [0, 1]
    np.clip(resampled, 0.0, 1.0, out=resampled)
    return resampled


def parse_num_lanes(scenario_name):
    """Extract lane count from scenario name like '4L_close_2_3'."""
    match = re.search(r'(\d+)L', scenario_name)
    return int(match.group(1)) if match else 4


# -----------------------------------------------------------------------
# Fixed heatmap generation
# -----------------------------------------------------------------------
def generate_fixed_heatmap(scenario_dir, model, device, global_bounds,
                          drop_first_n=100, fixed_n_traj=20):
    """
    Generate a single per-scenario heatmap from the first N vehicles' trajectories.

    fixed_n_traj: use the first N trajectories (veh_id 1..N), must be < drop_first_n.

    Returns: heatmap_float [512, 128] in [0, 1].
    """
    from utils.data_utils import rasterize_trajectories_v2

    scenario_name = os.path.basename(scenario_dir)
    num_lanes = parse_num_lanes(scenario_name)

    # Load trajectory data
    traj_path = os.path.join(scenario_dir, 'trajectories.csv')
    df = pd.read_csv(traj_path)

    # Use first fixed_n_traj vehicles for the observation heatmap (from front to back)
    df_obs = df[df['veh_id'] <= fixed_n_traj].copy()
    if df_obs.empty:
        print(f"  WARNING: No vehicles with veh_id <= {fixed_n_traj} in {scenario_name}")
        return np.zeros((512, 128), dtype=np.float32)

    x_local = df_obs['x'].values.astype(np.float64)
    y_local = df_obs['y'].values.astype(np.float64)

    # Rasterize into CVAE input grid
    obs_heatmap = rasterize_trajectories_v2(x_local, y_local, global_bounds)
    # obs_heatmap: [512, 128] float32 in [0, 1]

    # Build observation mask (nonzero -> 1)
    obs_mask = (obs_heatmap > 0.01).astype(np.float32)

    # Stack -> [1, 2, 512, 128]
    obs_hm_t = torch.from_numpy(obs_heatmap).unsqueeze(0).unsqueeze(0).float()
    obs_mk_t = torch.from_numpy(obs_mask).unsqueeze(0).unsqueeze(0).float()
    stacked_input = torch.cat([obs_hm_t, obs_mk_t], dim=1).to(device)

    num_lanes_t = torch.tensor([num_lanes], dtype=torch.long, device=device)

    with torch.no_grad():
        recon, _, _ = model(stacked_input, num_lanes_t)
        # recon: [1, 1, 512, 128]

    heatmap = recon[0, 0].cpu().numpy()  # [512, 128] float32 in [0,1]
    return heatmap


# -----------------------------------------------------------------------
# Dynamic heatmap generation
# -----------------------------------------------------------------------
def generate_dynamic_heatmaps(scenario_dir, model, device, global_bounds,
                              drop_first_n=100, epoch_size=50):
    """
    Generate epoch-based heatmaps for dynamic Mode C.

    For each "epoch" of vehicles completing, generate a new heatmap from
    the last `drop_first_n` completed vehicles.

    Returns: list of (frame_threshold, heatmap_float [512, 128])
    """
    from utils.data_utils import rasterize_trajectories_v2

    scenario_name = os.path.basename(scenario_dir)
    num_lanes = parse_num_lanes(scenario_name)

    traj_path = os.path.join(scenario_dir, 'trajectories.csv')
    df = pd.read_csv(traj_path)

    dt = 0.1
    t_min = df['t'].min()
    df['frame'] = np.round((df['t'].values - t_min) / dt).astype(int)

    # Get completion time for each vehicle
    veh_completion = df.groupby('veh_id')['frame'].max().sort_values()
    all_veh_ids = veh_completion.index.values
    all_completion_frames = veh_completion.values

    # Skip the first drop_first_n vehicles (they seed the initial heatmap)
    # Generate epoch heatmaps every epoch_size vehicles after that
    epochs = []
    n_total = len(all_veh_ids)

    # Initial epoch: first drop_first_n vehicles
    if drop_first_n > 0 and drop_first_n < n_total:
        seed_ids = set(all_veh_ids[:drop_first_n])
        seed_df = df[df['veh_id'].isin(seed_ids)]
        frame_thresh = int(all_completion_frames[drop_first_n - 1])
        hm = _rasterize_and_infer(seed_df, model, device, global_bounds, num_lanes)
        epochs.append((frame_thresh, hm))

    # Subsequent epochs
    for start_idx in range(drop_first_n, n_total, epoch_size):
        end_idx = min(start_idx + epoch_size, n_total)
        # Use last drop_first_n completed vehicles as observation
        obs_start = max(0, end_idx - drop_first_n)
        obs_ids = set(all_veh_ids[obs_start:end_idx])
        obs_df = df[df['veh_id'].isin(obs_ids)]
        frame_thresh = int(all_completion_frames[end_idx - 1])
        hm = _rasterize_and_infer(obs_df, model, device, global_bounds, num_lanes)
        epochs.append((frame_thresh, hm))

    return epochs


def _rasterize_and_infer(obs_df, model, device, global_bounds, num_lanes):
    """Helper: rasterize trajectories and run CVAE inference."""
    from utils.data_utils import rasterize_trajectories_v2

    x_local = obs_df['x'].values.astype(np.float64)
    y_local = obs_df['y'].values.astype(np.float64)

    obs_heatmap = rasterize_trajectories_v2(x_local, y_local, global_bounds)
    obs_mask = (obs_heatmap > 0.01).astype(np.float32)

    obs_hm_t = torch.from_numpy(obs_heatmap).unsqueeze(0).unsqueeze(0).float()
    obs_mk_t = torch.from_numpy(obs_mask).unsqueeze(0).unsqueeze(0).float()
    stacked_input = torch.cat([obs_hm_t, obs_mk_t], dim=1).to(device)
    num_lanes_t = torch.tensor([num_lanes], dtype=torch.long, device=device)

    with torch.no_grad():
        recon, _, _ = model(stacked_input, num_lanes_t)
    return recon[0, 0].cpu().numpy()


# -----------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Generate CVAE heatmaps for Trajectron++')
    parser.add_argument('--scenarios_dir', type=str,
                        default=os.path.join(_PROJECT_ROOT, 'dataset_collector', 'output'))
    parser.add_argument('--cvae_ckpt', type=str,
                        default=os.path.join(_CVAE_DIR, 'cvae_best.pth'))
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Where to save heatmaps. Default: same as --scenarios_dir (writes into each scenario folder for build_environment)')
    parser.add_argument('--heatmap_mode', type=str, default='fixed',
                        choices=['fixed', 'dynamic'])
    parser.add_argument('--output_format', type=str, default='aligned',
                        choices=['aligned', 'native'],
                        help='aligned: resample to map_mask grid; native: keep 512x128 CVAE grid')
    parser.add_argument('--drop_first_n', type=int, default=100,
                        help='Number of vehicles used for observation heatmap')
    parser.add_argument('--epoch_size', type=int, default=50,
                        help='Dynamic mode: vehicles per epoch')
    parser.add_argument('--fixed_heatmap_n_traj', type=int, default=20,
                        help='Fixed mode: number of trajectories (from front to back) for heatmap. Must be < drop_first_n.')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    if args.heatmap_mode == 'fixed' and args.fixed_heatmap_n_traj >= args.drop_first_n:
        parser.error('fixed_heatmap_n_traj must be < drop_first_n')

    if args.output_dir is None:
        args.output_dir = args.scenarios_dir

    os.makedirs(args.output_dir, exist_ok=True)

    # Load CVAE model
    from models.cvae import CVAE
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = CVAE(latent_dim=128, num_lane_classes=10)
    state_dict = torch.load(args.cvae_ckpt, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Loaded CVAE from {args.cvae_ckpt} on {device}")

    # Load global bounds
    global_bounds = load_global_bounds()
    print(f"Global bounds: x=[{global_bounds['xmin']}, {global_bounds['xmax']}], "
          f"y=[{global_bounds['ymin']}, {global_bounds['ymax']}]")

    # Discover scenarios
    scenario_dirs = sorted([
        os.path.join(args.scenarios_dir, d)
        for d in os.listdir(args.scenarios_dir)
        if os.path.isdir(os.path.join(args.scenarios_dir, d))
    ])
    print(f"Found {len(scenario_dirs)} scenarios")

    for scenario_dir in scenario_dirs:
        scenario_name = os.path.basename(scenario_dir)
        traj_path = os.path.join(scenario_dir, 'trajectories.csv')
        if not os.path.exists(traj_path):
            print(f"  [SKIP] {scenario_name}: no trajectories.csv")
            continue

        # Load map metadata for resampling
        meta_path = os.path.join(scenario_dir, 'map_meta.json')
        if not os.path.exists(meta_path):
            print(f"  [SKIP] {scenario_name}: no map_meta.json")
            continue
        with open(meta_path, 'r', encoding='utf-8') as f:
            map_meta = json.load(f)

        out_scenario_dir = os.path.join(args.output_dir, scenario_name)
        os.makedirs(out_scenario_dir, exist_ok=True)

        if args.heatmap_mode == 'fixed':
            print(f"  {scenario_name}: generating fixed heatmap ...")
            hm_cvae = generate_fixed_heatmap(
                scenario_dir, model, device, global_bounds,
                drop_first_n=args.drop_first_n,
                fixed_n_traj=args.fixed_heatmap_n_traj)

            if args.output_format == 'aligned':
                # Resample to map_mask grid
                hm_aligned = resample_heatmap_to_map_grid(
                    hm_cvae, global_bounds, map_meta)  # [W_px, H_px] float32
                hm_uint8 = (hm_aligned * 255).clip(0, 255).astype(np.uint8)
                hm_out = hm_uint8[np.newaxis]  # [1, W_px, H_px]
                out_path = os.path.join(out_scenario_dir, 'heatmap_aligned.npy')
                np.save(out_path, hm_out)
                print(f"    Saved {hm_out.shape} to {out_path}")
            else:
                # Native: keep [1, 512, 128], uint8 for Trajectron++ GeometricMap
                hm_uint8 = (hm_cvae * 255).clip(0, 255).astype(np.uint8)
                hm_out = hm_uint8[np.newaxis]  # [1, 512, 128]
                out_path = os.path.join(out_scenario_dir, 'heatmap_native.npy')
                np.save(out_path, hm_out)
                heatmap_meta = {
                    'global_bounds': global_bounds,
                    'homography': compute_heatmap_homography(global_bounds).tolist(),
                    'shape': [512, 128],
                    'dtype': 'uint8',
                }
                meta_path_out = os.path.join(out_scenario_dir, 'heatmap_meta.json')
                with open(meta_path_out, 'w') as f:
                    json.dump(heatmap_meta, f, indent=2)
                print(f"    Saved {hm_out.shape} to {out_path} + heatmap_meta.json")

        elif args.heatmap_mode == 'dynamic':
            print(f"  {scenario_name}: generating dynamic heatmaps ...")
            epochs = generate_dynamic_heatmaps(
                scenario_dir, model, device, global_bounds,
                args.drop_first_n, args.epoch_size)

            for k, (frame_thresh, hm_cvae) in enumerate(epochs):
                if args.output_format == 'aligned':
                    hm_aligned = resample_heatmap_to_map_grid(
                        hm_cvae, global_bounds, map_meta)
                    hm_uint8 = (hm_aligned * 255).clip(0, 255).astype(np.uint8)
                    hm_out = hm_uint8[np.newaxis]
                    out_path = os.path.join(
                        out_scenario_dir, f'heatmap_aligned_epoch_{k}.npy')
                else:
                    hm_uint8 = (hm_cvae * 255).clip(0, 255).astype(np.uint8)
                    hm_out = hm_uint8[np.newaxis]  # [1, 512, 128]
                    out_path = os.path.join(
                        out_scenario_dir, f'heatmap_native_epoch_{k}.npy')
                np.save(out_path, hm_out)

            if args.output_format == 'native':
                heatmap_meta = {
                    'global_bounds': global_bounds,
                    'homography': compute_heatmap_homography(global_bounds).tolist(),
                    'shape': [512, 128],
                    'dtype': 'uint8',
                }
                meta_path_out = os.path.join(out_scenario_dir, 'heatmap_meta.json')
                with open(meta_path_out, 'w') as f:
                    json.dump(heatmap_meta, f, indent=2)

            # Save epoch index for lookup
            index = [{'epoch': k, 'frame_threshold': int(ft)}
                     for k, (ft, _) in enumerate(epochs)]
            idx_path = os.path.join(out_scenario_dir, 'heatmap_epochs.json')
            with open(idx_path, 'w') as f:
                json.dump(index, f, indent=2)
            print(f"    Saved {len(epochs)} epoch heatmaps + index")

    print("\nDone.")


if __name__ == '__main__':
    main()
