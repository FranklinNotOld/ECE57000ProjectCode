"""
Visualize trajectory predictions on map with cones and work zone overlay.

Usage:
    python integration/evaluation/visualize_predictions.py \
        --mode A \
        --data_dir integration/data \
        --model_dir experiments/logs/models_xxx_mode_A \
        --output_dir results/viz \
        --num_scenes 5
"""
import sys
import os
import json
import argparse
import numpy as np
import dill
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_TRAJECTRON_DIR = os.path.join(_PROJECT_ROOT, 'trajectron_plus_plus', 'trajectron')
if _TRAJECTRON_DIR not in sys.path:
    sys.path.insert(0, _TRAJECTRON_DIR)

from model.trajectron import Trajectron
from model.model_registrar import ModelRegistrar
from utils import prediction_output_to_trajectories


MODE_TABLE = {
    'A': {'eval_pkl': 'eval_env_hdmap.pkl', 'config': 'workzone_base.json'},
    'B-A': {'eval_pkl': 'eval_env_cones_raster.pkl', 'config': 'workzone_cones_raster.json'},
    'B-B': {'eval_pkl': 'eval_env_cones_pointnet.pkl', 'config': 'workzone_cones_pointnet.json'},
    'B-C': {'eval_pkl': 'eval_env_cones_scenegraph.pkl', 'config': 'workzone_cones_scenegraph.json'},
    'C': {'eval_pkl': 'eval_env_heatmap.pkl', 'config': 'workzone_heatmap.json'},
}


def plot_scene_predictions(scene, predictions_dict, max_hl, ph, dt,
                           num_samples_to_show=5, ax=None):
    """
    Plot a single scene with background map, cones, history, ground truth,
    and predicted trajectories.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(20, 4))
    else:
        fig = ax.figure

    # Draw background map
    geo_map = None
    for key, val in scene.map.items():
        if str(key) == 'VEHICLE':
            geo_map = val
            break

    if geo_map is not None:
        # Drivable area as light gray background
        map_img = geo_map.data[0].T  # [H, W] transpose for imshow
        extent = [0, map_img.shape[1] / 5.0, 0, map_img.shape[0] / 5.0]
        ax.imshow(map_img, origin='lower', extent=extent,
                  cmap='gray', alpha=0.3, vmin=0, vmax=255)

    # Draw cones
    cones = getattr(scene, 'cones', None)
    if cones is not None and len(cones) > 0:
        ax.scatter(cones[:, 0], cones[:, 1], c='orange', s=15,
                   marker='^', zorder=5, label='Cones')

    # Process predictions
    (prediction_dict, histories_dict, futures_dict) = \
        prediction_output_to_trajectories(predictions_dict, dt, max_hl, ph,
                                          prune_ph_to_future=False)

    # Pick random timestep with predictions
    available_ts = sorted(prediction_dict.keys())
    if not available_ts:
        return fig

    # Show several nodes at a random timestep
    t = np.random.choice(available_ts)

    has_history = False
    has_gt = False
    has_pred = False

    for node in prediction_dict[t].keys():
        pred = prediction_dict[t][node]  # [n_samples, 1, ph, 2]
        gt = futures_dict[t][node]  # [ph, 2]
        hist = histories_dict[t][node]  # [hl, 2]

        # History (blue)
        if not has_history:
            ax.plot(hist[:, 0], hist[:, 1], 'b-', linewidth=1.5,
                    alpha=0.7, label='History')
            has_history = True
        else:
            ax.plot(hist[:, 0], hist[:, 1], 'b-', linewidth=1.5, alpha=0.7)

        # Ground truth (green)
        if not has_gt:
            ax.plot(gt[:, 0], gt[:, 1], 'g-', linewidth=2.0,
                    alpha=0.8, label='Ground Truth')
            has_gt = True
        else:
            ax.plot(gt[:, 0], gt[:, 1], 'g-', linewidth=2.0, alpha=0.8)

        # Predictions (red, multiple samples)
        n_show = min(num_samples_to_show, pred.shape[0])
        for s in range(n_show):
            traj = pred[s, 0]  # [ph, 2]
            if s == 0 and not has_pred:
                ax.plot(traj[:, 0], traj[:, 1], 'r-', linewidth=0.8,
                        alpha=0.4, label='Predictions')
                has_pred = True
            else:
                ax.plot(traj[:, 0], traj[:, 1], 'r-', linewidth=0.8, alpha=0.4)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'{scene.name}  t={t}')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_aspect('equal')
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize predictions')
    parser.add_argument('--mode', type=str, required=True,
                        choices=list(MODE_TABLE.keys()))
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(_PROJECT_ROOT, 'integration', 'data'))
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to the trained model directory')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(_PROJECT_ROOT, 'results', 'viz'))
    parser.add_argument('--num_scenes', type=int, default=5)
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    mode_info = MODE_TABLE[args.mode]

    # Load config
    config_path = os.path.join(_PROJECT_ROOT, 'integration', 'config', mode_info['config'])
    with open(config_path, 'r', encoding='utf-8') as f:
        hyperparams = json.load(f)

    # Runtime hyperparams
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
    hyperparams['override_attention_radius'] = []
    hyperparams['node_freq_mult_train'] = False
    hyperparams['node_freq_mult_eval'] = False
    hyperparams['scene_freq_mult_train'] = False
    hyperparams['scene_freq_mult_eval'] = False
    hyperparams['scene_freq_mult_viz'] = False

    # Load eval environment
    eval_pkl_path = os.path.join(args.data_dir, mode_info['eval_pkl'])
    with open(eval_pkl_path, 'rb') as f:
        eval_env = dill.load(f, encoding='latin1')

    # Load model
    device = torch.device(args.device)
    model_registrar = ModelRegistrar(args.model_dir, device)
    model_registrar.load_models(iter_num='best')

    trajectron = Trajectron(model_registrar, hyperparams, None, device)
    trajectron.set_environment(eval_env)

    max_hl = hyperparams['maximum_history_length']
    ph = hyperparams['prediction_horizon']

    # Compute scene graphs
    for scene in eval_env.scenes:
        scene.calculate_scene_graph(eval_env.attention_radius,
                                    hyperparams['edge_addition_filter'],
                                    hyperparams['edge_removal_filter'])

    # Select scenes to visualize
    scenes_to_viz = eval_env.scenes[:args.num_scenes]
    print(f"Visualizing {len(scenes_to_viz)} scenes for mode {args.mode}")

    with torch.no_grad():
        for idx, scene in enumerate(scenes_to_viz):
            timesteps = scene.sample_timesteps(min(256, scene.timesteps))
            predictions = trajectron.predict(scene, timesteps, ph,
                                             num_samples=args.num_samples,
                                             min_future_timesteps=ph,
                                             full_dist=False)

            fig = plot_scene_predictions(scene, predictions, max_hl, ph,
                                         scene.dt)
            out_path = os.path.join(args.output_dir,
                                    f'mode_{args.mode}_{scene.name}.png')
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved {out_path}")


if __name__ == '__main__':
    main()
