"""
Convert SUMO dataset_collector output to Trajectron++ Environment/Scene/Node .pkl files.

Usage:
    python integration/preprocessing/build_environment.py \
        --map_mode hdmap \
        --output_dir integration/data \
        --scenarios_dir dataset_collector/output
"""
import sys
import os
import copy
import argparse
import json
import numpy as np
import pandas as pd
import dill
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Path setup: allow importing from Trajectron++ and integration packages
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_TRAJECTRON_DIR = os.path.join(_PROJECT_ROOT, 'trajectron_plus_plus', 'trajectron')
if _TRAJECTRON_DIR not in sys.path:
    sys.path.insert(0, _TRAJECTRON_DIR)

from environment import Environment, Scene, Node, GeometricMap, derivative_of
from environment.data_structures import DoubleHeaderNumpyArray

# ---------------------------------------------------------------------------
# Data column definition (exact nuScenes convention)
# ---------------------------------------------------------------------------
data_columns_vehicle = pd.MultiIndex.from_product(
    [['position', 'velocity', 'acceleration', 'heading'], ['x', 'y']]
)
data_columns_vehicle = data_columns_vehicle.append(
    pd.MultiIndex.from_tuples([('heading', '\u00b0'), ('heading', 'd\u00b0')])
)
data_columns_vehicle = data_columns_vehicle.append(
    pd.MultiIndex.from_product([['velocity', 'acceleration'], ['norm']])
)

data_columns_cone = pd.MultiIndex.from_product([['position'], ['x', 'y']])

# ---------------------------------------------------------------------------
# Standardization (matches nuScenes process_data.py)
# ---------------------------------------------------------------------------
standardization_vehicle = {
    'VEHICLE': {
        'position': {
            'x': {'mean': 0, 'std': 80},
            'y': {'mean': 0, 'std': 80}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 15},
            'y': {'mean': 0, 'std': 15},
            'norm': {'mean': 0, 'std': 15}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 4},
            'y': {'mean': 0, 'std': 4},
            'norm': {'mean': 0, 'std': 4}
        },
        'heading': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1},
            '\u00b0': {'mean': 0, 'std': np.pi},
            'd\u00b0': {'mean': 0, 'std': 1}
        }
    }
}

standardization_cone = {
    'CONE': {
        'position': {
            'x': {'mean': 0, 'std': 80},
            'y': {'mean': 0, 'std': 80}
        }
    }
}


# ---------------------------------------------------------------------------
# Per-vehicle node construction
# ---------------------------------------------------------------------------
def build_vehicle_node(veh_id, veh_df, dt, env):
    """Build a Trajectron++ Node from a single vehicle's trajectory DataFrame."""
    veh_df = veh_df.sort_values('frame')

    x = veh_df['x'].values.astype(np.float64)
    y = veh_df['y'].values.astype(np.float64)

    # Check for continuity (no gaps in frame index)
    frames = veh_df['frame'].values
    if len(frames) < 2:
        return None
    diffs = np.diff(frames)
    if not np.all(diffs == 1):
        # Vehicle has gaps -- skip it
        return None

    # Derive velocity from position (more consistent than CSV values)
    vx = derivative_of(x, dt)
    vy = derivative_of(y, dt)
    ax = derivative_of(vx, dt)
    ay = derivative_of(vy, dt)

    # Heading from velocity
    heading = np.arctan2(vy, vx)

    # Heading unit vector (for data augmentation support)
    v = np.stack((vx, vy), axis=-1)
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    heading_v = np.divide(v, v_norm, out=np.zeros_like(v), where=(v_norm > 1.))
    heading_x = heading_v[:, 0]
    heading_y = heading_v[:, 1]

    d_heading = derivative_of(heading, dt, radian=True)

    data_dict = {
        ('position', 'x'): x,
        ('position', 'y'): y,
        ('velocity', 'x'): vx,
        ('velocity', 'y'): vy,
        ('velocity', 'norm'): np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1),
        ('acceleration', 'x'): ax,
        ('acceleration', 'y'): ay,
        ('acceleration', 'norm'): np.linalg.norm(np.stack((ax, ay), axis=-1), axis=-1),
        ('heading', 'x'): heading_x,
        ('heading', 'y'): heading_y,
        ('heading', '\u00b0'): heading,
        ('heading', 'd\u00b0'): d_heading,
    }
    node_data = pd.DataFrame(data_dict, columns=data_columns_vehicle)
    node = Node(
        node_type=env.NodeType.VEHICLE,
        node_id=str(veh_id),
        data=node_data,
    )
    node.first_timestep = int(frames[0])
    return node


# ---------------------------------------------------------------------------
# Clip node to time window (for sliding-window scene splitting)
# ---------------------------------------------------------------------------
def clip_node_to_window(node, window_start, window_end, env):
    """
    Clip a Node to a time window. Returns a new Node with data and timesteps
    in window-local coordinates, or None if the node has no data in the window.

    :param node: Original Node (VEHICLE or CONE)
    :param window_start: Start frame of window (global, inclusive)
    :param window_end: End frame of window (global, inclusive)
    :param env: Environment (for NodeType)
    :return: New Node or None
    """
    clip_start = max(0, window_start - node.first_timestep)
    clip_end = min(node.timesteps - 1, window_end - node.first_timestep)
    if clip_start > clip_end:
        return None

    clipped_data = node.data.data[clip_start:clip_end + 1]
    new_first_timestep = node.first_timestep + clip_start - window_start

    clipped_dha = DoubleHeaderNumpyArray(clipped_data.copy(), node.data.header)
    new_node = Node(
        node_type=node.type,
        node_id=node.id,
        data=clipped_dha,
        is_robot=node.is_robot,
    )
    new_node.first_timestep = new_first_timestep
    return new_node


# ---------------------------------------------------------------------------
# Load map data for a scenario
# ---------------------------------------------------------------------------
def load_map(scenario_dir, map_mode, heatmap_dir=None):
    """Return (map_data_uint8, homography) based on map_mode."""
    meta_path = os.path.join(scenario_dir, 'map_meta.json')
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    homography = np.array(meta['homography'], dtype=np.float64)

    if map_mode == 'hdmap':
        map_data = np.load(os.path.join(scenario_dir, 'map_mask.npy'))  # [3, W, H] uint8
        return map_data, homography

    elif map_mode == 'hdmap_plain':
        # 3-channel plain HD map (no workzone info) for Mode D
        map_data = np.load(os.path.join(scenario_dir, 'map_mask_plain.npy'))
        return map_data, homography

    elif map_mode == 'cones_raster':
        # 4-channel: HD map + cone Gaussian layer
        from integration.preprocessing.cone_rasterizer import rasterize_cones
        map_mask = np.load(os.path.join(scenario_dir, 'map_mask.npy'))
        cones_path = os.path.join(scenario_dir, 'cones.csv')
        cone_channel = rasterize_cones(cones_path, map_mask.shape[1], map_mask.shape[2],
                                       meta['pixels_per_meter'])
        map_data = np.concatenate([map_mask, cone_channel[np.newaxis]], axis=0)
        return map_data, homography

    elif map_mode == 'cones_raster_plain':
        # 4-channel: plain HD map (no workzone info) + cone Gaussian layer for Mode E
        from integration.preprocessing.cone_rasterizer import rasterize_cones
        map_mask = np.load(os.path.join(scenario_dir, 'map_mask_plain.npy'))
        cones_path = os.path.join(scenario_dir, 'cones.csv')
        cone_channel = rasterize_cones(cones_path, map_mask.shape[1], map_mask.shape[2],
                                       meta['pixels_per_meter'])
        map_data = np.concatenate([map_mask, cone_channel[np.newaxis]], axis=0)
        return map_data, homography

    elif map_mode in ('cones_pointnet', 'cones_scenegraph'):
        # 3-channel HD map (cones handled elsewhere)
        map_data = np.load(os.path.join(scenario_dir, 'map_mask.npy'))
        return map_data, homography

    elif map_mode == 'heatmap':
        # 1-channel CVAE heatmap (already aligned to map_mask grid)
        scenario_name = os.path.basename(scenario_dir)
        heatmap_path = os.path.join(heatmap_dir or scenario_dir, scenario_name,
                                    'heatmap_aligned.npy')
        if not os.path.exists(heatmap_path):
            # Fallback: look in scenario_dir directly
            heatmap_path = os.path.join(scenario_dir, 'heatmap_aligned.npy')
        heatmap = np.load(heatmap_path)  # [1, W, H] uint8
        return heatmap, homography

    elif map_mode == 'heatmap_native':
        # 1-channel CVAE heatmap at native [1, 512, 128] resolution.
        # Homography comes from heatmap_meta.json, NOT map_meta.json.
        scenario_name = os.path.basename(scenario_dir)
        heatmap_path = os.path.join(heatmap_dir or scenario_dir, scenario_name,
                                    'heatmap_native.npy')
        if not os.path.exists(heatmap_path):
            heatmap_path = os.path.join(scenario_dir, 'heatmap_native.npy')

        heatmap_meta_path = os.path.join(heatmap_dir or scenario_dir, scenario_name,
                                         'heatmap_meta.json')
        if not os.path.exists(heatmap_meta_path):
            heatmap_meta_path = os.path.join(scenario_dir, 'heatmap_meta.json')

        heatmap = np.load(heatmap_path)  # [1, 512, 128] uint8
        with open(heatmap_meta_path, 'r', encoding='utf-8') as f:
            heatmap_meta = json.load(f)
        native_homography = np.array(heatmap_meta['homography'], dtype=np.float64)
        return heatmap, native_homography

    else:
        raise ValueError(f"Unknown map_mode: {map_mode}")


# ---------------------------------------------------------------------------
# Build a single scenario
# ---------------------------------------------------------------------------
def process_scenario(scenario_dir, env, map_mode, drop_first_n=100,
                     heatmap_dir=None, use_window=True, window_size=800,
                     window_stride=400, min_nodes_per_window=2):
    """
    Process one scenario directory into (train_scenes, eval_scenes).

    When use_window=True: returns lists of windowed scenes (many per scenario).
    When use_window=False: returns lists with one scene each (legacy behavior).

    Returns ([], []) if the scenario has insufficient data.
    """
    scenario_name = os.path.basename(scenario_dir)
    dt = 0.1

    # ------------------------------------------------------------------
    # 1) Load trajectories
    # ------------------------------------------------------------------
    traj_path = os.path.join(scenario_dir, 'trajectories.csv')
    if not os.path.exists(traj_path):
        print(f"  [SKIP] No trajectories.csv in {scenario_name}")
        return [], []
    df = pd.read_csv(traj_path)
    if df.empty:
        return [], []

    # ------------------------------------------------------------------
    # 2) Convert timestamps to frame indices
    # ------------------------------------------------------------------
    t_min = df['t'].min()
    df['frame'] = np.round((df['t'].values - t_min) / dt).astype(int)

    # ------------------------------------------------------------------
    # 3) Drop first N vehicles (reserved for heatmap generation)
    # ------------------------------------------------------------------
    df = df[df['veh_id'] > drop_first_n].copy()
    if df.empty:
        print(f"  [SKIP] No vehicles left after dropping first {drop_first_n} in {scenario_name}")
        return [], []

    unique_veh_ids = sorted(df['veh_id'].unique())

    # ------------------------------------------------------------------
    # 4) Train/test split by veh_id (80/20)
    # ------------------------------------------------------------------
    if len(unique_veh_ids) < 5:
        print(f"  [SKIP] Too few vehicles ({len(unique_veh_ids)}) in {scenario_name}")
        return [], []

    train_ids, eval_ids = train_test_split(
        unique_veh_ids, test_size=0.2, random_state=42
    )
    train_ids_set = set(train_ids)
    eval_ids_set = set(eval_ids)

    # ------------------------------------------------------------------
    # 5) Load map
    # ------------------------------------------------------------------
    map_data, homography = load_map(scenario_dir, map_mode, heatmap_dir)
    geo_map = GeometricMap(data=map_data, homography=homography,
                           description=f'{scenario_name}_{map_mode}')

    # Load cones for Mode B-C
    cones_path = os.path.join(scenario_dir, 'cones.csv')
    cones_array = None
    if os.path.exists(cones_path):
        cones_df = pd.read_csv(cones_path)
        cones_array = cones_df[['x', 'y']].values.astype(np.float64)

    # ------------------------------------------------------------------
    # 6) Build vehicle nodes (shared for both modes)
    # ------------------------------------------------------------------
    train_nodes = []
    eval_nodes = []
    for veh_id in tqdm(unique_veh_ids, desc=f'  {scenario_name}', leave=False, ncols=80):
        veh_df = df[df['veh_id'] == veh_id]
        node = build_vehicle_node(veh_id, veh_df, dt, env)
        if node is None:
            continue

        if veh_id in train_ids_set:
            train_nodes.append(node)
        elif veh_id in eval_ids_set:
            eval_nodes.append(node)

    max_frame = df['frame'].max()
    total_timesteps = max_frame + 1

    if use_window:
        return _process_scenario_windowed(
            scenario_name=scenario_name,
            dt=dt,
            map_data=map_data,
            geo_map=geo_map,
            cones_array=cones_array,
            train_nodes=train_nodes,
            eval_nodes=eval_nodes,
            max_frame=max_frame,
            window_size=window_size,
            window_stride=window_stride,
            min_nodes_per_window=min_nodes_per_window,
            map_mode=map_mode,
            env=env,
        )
    else:
        return _process_scenario_legacy(
            scenario_name=scenario_name,
            dt=dt,
            total_timesteps=total_timesteps,
            geo_map=geo_map,
            cones_array=cones_array,
            train_nodes=train_nodes,
            eval_nodes=eval_nodes,
            map_data=map_data,
            map_mode=map_mode,
            env=env,
        )


def _process_scenario_legacy(scenario_name, dt, total_timesteps, geo_map,
                            cones_array, train_nodes, eval_nodes, map_data,
                            map_mode, env):
    """Legacy: one train scene and one eval scene per scenario."""
    train_scene = Scene(timesteps=total_timesteps, dt=dt, name=f'{scenario_name}_train')
    train_scene.map = {'VEHICLE': geo_map}
    train_scene.cones = cones_array

    eval_scene = Scene(timesteps=total_timesteps, dt=dt, name=f'{scenario_name}_eval')
    eval_scene.map = {'VEHICLE': geo_map}
    eval_scene.cones = cones_array

    for node in train_nodes:
        train_scene.nodes.append(node)
    for node in eval_nodes:
        eval_scene.nodes.append(node)

    if map_mode == 'cones_scenegraph' and cones_array is not None:
        for i, (cx, cy) in enumerate(cones_array):
            cone_data_arr = np.column_stack([
                np.full(total_timesteps, cx),
                np.full(total_timesteps, cy),
            ])
            cone_df = pd.DataFrame(cone_data_arr, columns=data_columns_cone)
            cone_node = Node(
                node_type=env.NodeType.CONE,
                node_id=f'cone_{i}',
                data=cone_df,
            )
            cone_node.first_timestep = 0
            train_scene.nodes.append(cone_node)
            eval_cone = Node(
                node_type=env.NodeType.CONE,
                node_id=f'cone_{i}',
                data=cone_df.copy(),
            )
            eval_cone.first_timestep = 0
            eval_scene.nodes.append(eval_cone)

    if len(train_nodes) == 0:
        return [], []

    print(f"  {scenario_name}: {len(train_nodes)} train / {len(eval_nodes)} eval vehicles, "
          f"{total_timesteps} timesteps, map {map_data.shape}")
    return [train_scene], [eval_scene]


def _process_scenario_windowed(scenario_name, dt, map_data, geo_map, cones_array,
                               train_nodes, eval_nodes, max_frame,
                               window_size, window_stride, min_nodes_per_window,
                               map_mode, env):
    """Windowed: multiple scenes per scenario via sliding time windows."""
    train_scenes = []
    eval_scenes = []

    for window_start in range(0, max_frame - window_size + 2, window_stride):
        window_end = window_start + window_size - 1

        # Train window: filter nodes overlapping [window_start, window_end]
        train_window_nodes = []
        for node in train_nodes:
            if node.first_timestep <= window_end and node.last_timestep >= window_start:
                clipped = clip_node_to_window(node, window_start, window_end, env)
                if clipped is not None:
                    train_window_nodes.append(clipped)

        # Eval window
        eval_window_nodes = []
        for node in eval_nodes:
            if node.first_timestep <= window_end and node.last_timestep >= window_start:
                clipped = clip_node_to_window(node, window_start, window_end, env)
                if clipped is not None:
                    eval_window_nodes.append(clipped)

        # Skip windows with too few nodes
        if len(train_window_nodes) >= min_nodes_per_window:
            scene = Scene(timesteps=window_size, dt=dt,
                         name=f'{scenario_name}_train_w{window_start}')
            scene.map = {'VEHICLE': geo_map}
            scene.cones = cones_array
            for n in train_window_nodes:
                scene.nodes.append(n)
            if map_mode == 'cones_scenegraph' and cones_array is not None:
                for i, (cx, cy) in enumerate(cones_array):
                    cone_data_arr = np.column_stack([
                        np.full(window_size, cx),
                        np.full(window_size, cy),
                    ])
                    cone_df = pd.DataFrame(cone_data_arr, columns=data_columns_cone)
                    cone_node = Node(
                        node_type=env.NodeType.CONE,
                        node_id=f'cone_{i}',
                        data=cone_df,
                    )
                    cone_node.first_timestep = 0
                    scene.nodes.append(cone_node)
            train_scenes.append(scene)

        if len(eval_window_nodes) >= min_nodes_per_window:
            scene = Scene(timesteps=window_size, dt=dt,
                         name=f'{scenario_name}_eval_w{window_start}')
            scene.map = {'VEHICLE': geo_map}
            scene.cones = cones_array
            for n in eval_window_nodes:
                scene.nodes.append(n)
            if map_mode == 'cones_scenegraph' and cones_array is not None:
                for i, (cx, cy) in enumerate(cones_array):
                    cone_data_arr = np.column_stack([
                        np.full(window_size, cx),
                        np.full(window_size, cy),
                    ])
                    cone_df = pd.DataFrame(cone_data_arr, columns=data_columns_cone)
                    cone_node = Node(
                        node_type=env.NodeType.CONE,
                        node_id=f'cone_{i}',
                        data=cone_df,
                    )
                    cone_node.first_timestep = 0
                    scene.nodes.append(cone_node)
            eval_scenes.append(scene)

    avg_train = sum(len(s.nodes) for s in train_scenes) / len(train_scenes) if train_scenes else 0
    avg_eval = sum(len(s.nodes) for s in eval_scenes) / len(eval_scenes) if eval_scenes else 0
    print(f"  {scenario_name}: {len(train_scenes)} train / {len(eval_scenes)} eval windows, "
          f"~{avg_train:.0f}/~{avg_eval:.0f} nodes per window, map {map_data.shape}")

    return train_scenes, eval_scenes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Build Trajectron++ environment from SUMO data')
    parser.add_argument('--scenarios_dir', type=str,
                        default=os.path.join(_PROJECT_ROOT, 'dataset_collector', 'output'),
                        help='Directory containing scenario subdirectories')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(_PROJECT_ROOT, 'integration', 'data'),
                        help='Output directory for .pkl files')
    parser.add_argument('--map_mode', type=str, default='heatmap_native',
                        choices=['hdmap', 'hdmap_plain',
                                 'cones_raster', 'cones_raster_plain',
                                 'cones_pointnet', 'cones_scenegraph',
                                 'heatmap', 'heatmap_native'],
                        help='Map mode: hdmap | hdmap_plain | cones_raster | cones_raster_plain | '
                             'cones_pointnet | cones_scenegraph | heatmap | heatmap_native')
    parser.add_argument('--heatmap_dir', type=str, default=None,
                        help='Directory containing heatmaps (for --map_mode heatmap or heatmap_native)')
    parser.add_argument('--drop_first_n', type=int, default=100,
                        help='Drop first N vehicles per scenario (reserved for heatmap)')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Fraction for eval split')
    parser.add_argument('--window_size', type=int, default=800,
                        help='Sliding window size in frames (default 800 = 80s at 0.1s dt)')
    parser.add_argument('--window_stride', type=int, default=800,
                        help='Sliding window stride in frames (default 400 = 40s)')
    parser.add_argument('--no_window', action='store_true',
                        help='Disable windowing; use legacy single-scene-per-split (for small datasets)')
    parser.add_argument('--min_nodes_per_window', type=int, default=2,
                        help='Minimum nodes per window to keep it (default 2)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Build Environment
    # ------------------------------------------------------------------
    if args.map_mode == 'cones_scenegraph':
        node_type_list = ['VEHICLE', 'CONE']
        standardization = {**standardization_vehicle, **standardization_cone}
    else:
        node_type_list = ['VEHICLE']
        standardization = dict(standardization_vehicle)

    # Create a temporary env to get NodeType enum
    env = Environment(node_type_list=node_type_list,
                      standardization=standardization)

    # Attention radius
    attention_radius = dict()
    attention_radius[(env.NodeType.VEHICLE, env.NodeType.VEHICLE)] = 30.0
    if args.map_mode == 'cones_scenegraph':
        attention_radius[(env.NodeType.VEHICLE, env.NodeType.CONE)] = 15.0
        attention_radius[(env.NodeType.CONE, env.NodeType.VEHICLE)] = 15.0
        attention_radius[(env.NodeType.CONE, env.NodeType.CONE)] = 0.0

    # ------------------------------------------------------------------
    # Discover scenarios
    # ------------------------------------------------------------------
    scenario_dirs = sorted([
        os.path.join(args.scenarios_dir, d)
        for d in os.listdir(args.scenarios_dir)
        if os.path.isdir(os.path.join(args.scenarios_dir, d))
    ])
    print(f"Found {len(scenario_dirs)} scenarios in {args.scenarios_dir}")
    if not args.no_window:
        print(f"Using sliding windows: size={args.window_size}, stride={args.window_stride}")
    else:
        print("Using legacy mode (one scene per split per scenario)")

    # ------------------------------------------------------------------
    # Process all scenarios
    # ------------------------------------------------------------------
    train_scenes = []
    eval_scenes = []
    use_window = not args.no_window
    for scenario_dir in scenario_dirs:
        train_scenes_from_scenario, eval_scenes_from_scenario = process_scenario(
            scenario_dir, env, args.map_mode,
            drop_first_n=args.drop_first_n,
            heatmap_dir=args.heatmap_dir,
            use_window=use_window,
            window_size=args.window_size,
            window_stride=args.window_stride,
            min_nodes_per_window=args.min_nodes_per_window,
        )
        train_scenes.extend(train_scenes_from_scenario)
        eval_scenes.extend(eval_scenes_from_scenario)

    print(f"\nTotal: {len(train_scenes)} train scenes, {len(eval_scenes)} eval scenes")

    # ------------------------------------------------------------------
    # Build and save environments
    # ------------------------------------------------------------------
    mode_suffix = {
        'hdmap': 'hdmap',
        'hdmap_plain': 'hdmap_plain',
        'cones_raster': 'cones_raster',
        'cones_raster_plain': 'cones_raster_plain',
        'cones_pointnet': 'cones_pointnet',
        'cones_scenegraph': 'cones_scenegraph',
        'heatmap': 'heatmap',
        'heatmap_native': 'heatmap_native',
    }[args.map_mode]

    # Train environment
    train_env = Environment(node_type_list=node_type_list,
                            standardization=standardization,
                            scenes=train_scenes,
                            attention_radius=attention_radius)
    train_path = os.path.join(args.output_dir, f'train_env_{mode_suffix}.pkl')
    with open(train_path, 'wb') as f:
        dill.dump(train_env, f, protocol=dill.HIGHEST_PROTOCOL)
    print(f"Saved train environment to {train_path}")

    # Eval environment
    eval_env = Environment(node_type_list=node_type_list,
                           standardization=standardization,
                           scenes=eval_scenes,
                           attention_radius=attention_radius)
    eval_path = os.path.join(args.output_dir, f'eval_env_{mode_suffix}.pkl')
    with open(eval_path, 'wb') as f:
        dill.dump(eval_env, f, protocol=dill.HIGHEST_PROTOCOL)
    print(f"Saved eval environment to {eval_path}")


if __name__ == '__main__':
    main()
