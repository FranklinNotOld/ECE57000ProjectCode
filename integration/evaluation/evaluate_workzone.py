"""
Evaluation pipeline for work zone trajectory prediction experiments.

Computes standard metrics (ADE, FDE, NLL) plus work zone specific metrics:
  - Out-of-bounds (OOB) rate: % predicted points in non-drivable area
  - Cone collision rate: % predicted points within cone_radius of any cone

Usage:
    python integration/evaluation/evaluate_workzone.py \
        --modes A B-A B-B B-C C \
        --data_dir integration/data \
        --model_dir experiments/logs \
        --output_dir results
"""
import sys
import os
import json
import argparse
import numpy as np
import dill
import torch
from tqdm import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_TRAJECTRON_DIR = os.path.join(_PROJECT_ROOT, 'trajectron_plus_plus', 'trajectron')
if _TRAJECTRON_DIR not in sys.path:
    sys.path.insert(0, _TRAJECTRON_DIR)

from model.trajectron import Trajectron
from model.model_registrar import ModelRegistrar
from model.dataset import EnvironmentDataset, collate
from utils import prediction_output_to_trajectories
from scipy.spatial import cKDTree


# -----------------------------------------------------------------------
# Mode -> file mapping (same as train_workzone.py)
# -----------------------------------------------------------------------
MODE_TABLE = {
    'A': {'eval_pkl': 'eval_env_hdmap.pkl', 'config': 'workzone_base.json'},
    'B-A': {'eval_pkl': 'eval_env_cones_raster.pkl', 'config': 'workzone_cones_raster.json'},
    'B-B': {'eval_pkl': 'eval_env_cones_pointnet.pkl', 'config': 'workzone_cones_pointnet.json'},
    'B-C': {'eval_pkl': 'eval_env_cones_scenegraph.pkl', 'config': 'workzone_cones_scenegraph.json'},
    'C': {'eval_pkl': 'eval_env_heatmap.pkl', 'config': 'workzone_heatmap.json'},
}


# -----------------------------------------------------------------------
# Custom metrics
# -----------------------------------------------------------------------
def compute_oob_rate(predictions, scene_map, node_type_str='VEHICLE'):
    """
    Out-of-bounds rate: fraction of predicted positions landing on
    non-drivable pixels (drivable_area channel == 0).

    predictions: dict[timestep] -> dict[node] -> [n_samples, 1, ph, 2]
    scene_map: dict with node_type keys -> GeometricMap
    """
    total_points = 0
    oob_points = 0

    # Get the map for VEHICLE
    geo_map = None
    for key, val in scene_map.items():
        if str(key) == node_type_str:
            geo_map = val
            break
    if geo_map is None:
        return 0.0

    # Channel 0 = drivable_area (255 = drivable, 0 = non-drivable)
    drivable = geo_map.data[0]  # [W_px, H_px]

    for t_dict in predictions.values():
        for node, pred_arr in t_dict.items():
            # pred_arr: [n_samples, 1, ph, 2] or [1, n_samples, ph, 2]
            pts = pred_arr.reshape(-1, 2)  # [N, 2]
            total_points += len(pts)

            # Convert scene coords to pixel coords
            px_pts = geo_map.to_map_points(pts)  # [N, 2]
            px_x = np.round(px_pts[:, 0]).astype(int)
            px_y = np.round(px_pts[:, 1]).astype(int)

            # Clip to image bounds
            valid = ((px_x >= 0) & (px_x < drivable.shape[0]) &
                     (px_y >= 0) & (px_y < drivable.shape[1]))

            for i in range(len(pts)):
                if valid[i]:
                    if drivable[px_x[i], px_y[i]] == 0:
                        oob_points += 1
                else:
                    # Out of map bounds = out of bounds
                    oob_points += 1

    return oob_points / max(total_points, 1)


def compute_cone_collision_rate(predictions, cones_xy, cone_radius=1.5):
    """
    Cone collision rate: fraction of predicted positions within
    `cone_radius` meters of any cone.

    predictions: dict[timestep] -> dict[node] -> [n_samples, 1, ph, 2]
    cones_xy: np.ndarray [N_cones, 2]
    """
    if cones_xy is None or len(cones_xy) == 0:
        return 0.0

    tree = cKDTree(cones_xy)
    total_points = 0
    collision_points = 0

    for t_dict in predictions.values():
        for node, pred_arr in t_dict.items():
            pts = pred_arr.reshape(-1, 2)
            total_points += len(pts)

            # Query: how many predicted points are within cone_radius of any cone
            dists, _ = tree.query(pts, k=1)
            collision_points += np.sum(dists <= cone_radius)

    return collision_points / max(total_points, 1)


# -----------------------------------------------------------------------
# Standard metrics (ADE/FDE)
# -----------------------------------------------------------------------
def compute_ade_fde(prediction_output_dict, dt, max_hl, ph, best_of_k=True):
    """Compute ADE and FDE from prediction output dict."""
    (prediction_dict, _, futures_dict) = prediction_output_to_trajectories(
        prediction_output_dict, dt, max_hl, ph, prune_ph_to_future=False)

    ade_list = []
    fde_list = []

    for t in prediction_dict.keys():
        for node in prediction_dict[t].keys():
            pred = prediction_dict[t][node]  # [n_samples, 1, ph, 2] or similar
            gt = futures_dict[t][node]  # [ph, 2]

            error = np.linalg.norm(pred - gt, axis=-1)  # [n_samples, 1, ph]
            ade_per_sample = np.mean(error, axis=-1).flatten()  # [n_samples]
            fde_per_sample = error[..., -1].flatten()

            if best_of_k:
                ade_list.append(np.min(ade_per_sample))
                fde_list.append(np.min(fde_per_sample))
            else:
                ade_list.extend(ade_per_sample.tolist())
                fde_list.extend(fde_per_sample.tolist())

    return {
        'ade_mean': float(np.mean(ade_list)) if ade_list else 0.0,
        'ade_median': float(np.median(ade_list)) if ade_list else 0.0,
        'fde_mean': float(np.mean(fde_list)) if fde_list else 0.0,
        'fde_median': float(np.median(fde_list)) if fde_list else 0.0,
        'n_samples': len(ade_list),
    }


# -----------------------------------------------------------------------
# Evaluate a single mode
# -----------------------------------------------------------------------
def evaluate_mode(mode, data_dir, model_dir, num_samples=20, device='cpu'):
    """
    Evaluate a trained mode and return metrics dict.
    """
    mode_info = MODE_TABLE[mode]

    # Load config
    config_path = os.path.join(_PROJECT_ROOT, 'integration', 'config', mode_info['config'])
    with open(config_path, 'r', encoding='utf-8') as f:
        hyperparams = json.load(f)

    # Add required runtime hyperparams
    hyperparams['dynamic_edges'] = 'yes'
    hyperparams['edge_state_combine_method'] = 'sum'
    hyperparams['edge_influence_combine_method'] = 'attention'
    hyperparams['edge_addition_filter'] = [0.25, 0.5, 0.75, 1.0]
    hyperparams['edge_removal_filter'] = [1.0, 0.0]
    hyperparams['offline_scene_graph'] = 'yes'
    hyperparams['incl_robot_node'] = False
    hyperparams['edge_encoding'] = True
    hyperparams['use_map_encoding'] = True
    hyperparams['augment'] = False
    if 'override_attention_radius' not in hyperparams:
        hyperparams['override_attention_radius'] = []
    hyperparams['node_freq_mult_train'] = False
    hyperparams['node_freq_mult_eval'] = False
    hyperparams['scene_freq_mult_train'] = False
    hyperparams['scene_freq_mult_eval'] = False
    hyperparams['scene_freq_mult_viz'] = False

    # Load eval environment
    eval_pkl_path = os.path.join(data_dir, mode_info['eval_pkl'])
    if not os.path.exists(eval_pkl_path):
        print(f"  [SKIP] Eval data not found: {eval_pkl_path}")
        return None

    with open(eval_pkl_path, 'rb') as f:
        eval_env = dill.load(f, encoding='latin1')

    # Load model
    device = torch.device(device)
    model_registrar = ModelRegistrar(model_dir, device)
    model_registrar.load_models(iter_num='best')

    trajectron = Trajectron(model_registrar, hyperparams, None, device)
    trajectron.set_environment(eval_env)

    # Compute scene graphs offline
    for scene in eval_env.scenes:
        scene.calculate_scene_graph(eval_env.attention_radius,
                                    hyperparams['edge_addition_filter'],
                                    hyperparams['edge_removal_filter'])

    # Evaluate
    max_hl = hyperparams['maximum_history_length']
    ph = hyperparams['prediction_horizon']

    all_predictions = {}
    print(f"  Evaluating {len(eval_env.scenes)} scenes ...")
    with torch.no_grad():
        for scene in tqdm(eval_env.scenes, desc=f'  Mode {mode}', ncols=80):
            timesteps = scene.sample_timesteps(scene.timesteps)
            predictions = trajectron.predict(scene, timesteps, ph,
                                             num_samples=num_samples,
                                             min_future_timesteps=ph,
                                             full_dist=False)
            # Merge
            for ts, node_preds in predictions.items():
                if ts not in all_predictions:
                    all_predictions[ts] = {}
                all_predictions[ts].update(node_preds)

    # Standard metrics
    metrics = compute_ade_fde(all_predictions, eval_env.scenes[0].dt,
                              max_hl, ph, best_of_k=True)

    # Work zone metrics
    # Aggregate across scenes
    oob_rates = []
    cone_coll_rates = []
    for scene in eval_env.scenes:
        # Filter predictions for this scene's nodes
        scene_preds = {}
        scene_node_ids = {n.id for n in scene.nodes}
        for ts, node_preds in all_predictions.items():
            for node, pred in node_preds.items():
                if node.id in scene_node_ids:
                    if ts not in scene_preds:
                        scene_preds[ts] = {}
                    scene_preds[ts][node] = pred

        if scene_preds:
            oob = compute_oob_rate(scene_preds, scene.map)
            oob_rates.append(oob)

            cones = getattr(scene, 'cones', None)
            cc = compute_cone_collision_rate(scene_preds, cones)
            cone_coll_rates.append(cc)

    metrics['oob_rate'] = float(np.mean(oob_rates)) if oob_rates else 0.0
    metrics['cone_collision_rate'] = float(np.mean(cone_coll_rates)) if cone_coll_rates else 0.0

    return metrics


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Evaluate work zone modes')
    parser.add_argument('--modes', nargs='+', default=['A'],
                        choices=list(MODE_TABLE.keys()))
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(_PROJECT_ROOT, 'integration', 'data'))
    parser.add_argument('--model_dir', type=str,
                        default=os.path.join(_PROJECT_ROOT, 'experiments', 'logs'))
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(_PROJECT_ROOT, 'results'))
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    for mode in args.modes:
        print(f"\n=== Evaluating Mode {mode} ===")
        # Find model directory for this mode
        mode_model_dir = os.path.join(args.model_dir, f'mode_{mode}')
        if not os.path.exists(mode_model_dir):
            # Try finding the latest matching directory
            candidates = [d for d in os.listdir(args.model_dir)
                          if d.endswith(f'_mode_{mode}')]
            if candidates:
                mode_model_dir = os.path.join(args.model_dir, sorted(candidates)[-1])
            else:
                print(f"  [SKIP] No model directory found for mode {mode}")
                continue

        metrics = evaluate_mode(mode, args.data_dir, mode_model_dir,
                                args.num_samples, args.device)
        if metrics is not None:
            results[mode] = metrics
            print(f"  ADE: {metrics['ade_mean']:.4f} | FDE: {metrics['fde_mean']:.4f}")
            print(f"  OOB: {metrics['oob_rate']:.4f} | Cone collision: {metrics['cone_collision_rate']:.4f}")

    # Save results
    out_path = os.path.join(args.output_dir, 'metrics.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print comparison table
    if results:
        print(f"\n{'Mode':<6} {'ADE':>8} {'FDE':>8} {'OOB%':>8} {'Cone%':>8}")
        print('-' * 42)
        for mode, m in results.items():
            print(f"{mode:<6} {m['ade_mean']:8.4f} {m['fde_mean']:8.4f} "
                  f"{m['oob_rate']*100:7.2f}% {m['cone_collision_rate']*100:7.2f}%")


if __name__ == '__main__':
    main()
