"""
Master training entry point for work zone trajectory prediction experiments.

Maps high-level mode selection (A / B-A / B-B / B-C / C) to the correct
config file, data pkl, and Trajectron++ arguments, then launches training.

Usage:
    python integration/training/train_workzone.py \
        --mode A \
        --data_dir integration/data \
        --log_dir experiments/logs \
        --train_epochs 100 \
        --eval_every 5
"""
import sys
import os
import argparse
import subprocess

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# ---------------------------------------------------------------------------
# Short abbreviations for auto-tagging TensorBoard runs.
# Only parameters explicitly passed on the CLI get appended to the log_tag.
# Keys = argparse dest names; values = short tag prefixes.
# ---------------------------------------------------------------------------
PARAM_TAG_ABBREV = {
    'learning_rate': 'lr',
    'dropout':       'do',
    'batch_size':    'bs',
    'kl_weight':     'kl',
}

# ---------------------------------------------------------------------------
# Mode -> config / data mapping
# ---------------------------------------------------------------------------
MODE_TABLE = {
    'A': {
        'config': 'workzone_base.json',
        'train_pkl': 'train_env_hdmap.pkl',
        'eval_pkl': 'eval_env_hdmap.pkl',
        'log_tag': '_mode_A',
    },
    'B-A': {
        'config': 'workzone_cones_raster.json',
        'train_pkl': 'train_env_cones_raster.pkl',
        'eval_pkl': 'eval_env_cones_raster.pkl',
        'log_tag': '_mode_B-A',
    },
    'B-B': {
        'config': 'workzone_cones_pointnet.json',
        'train_pkl': 'train_env_cones_pointnet.pkl',
        'eval_pkl': 'eval_env_cones_pointnet.pkl',
        'log_tag': '_mode_B-B',
    },
    'B-C': {
        'config': 'workzone_cones_scenegraph.json',
        'train_pkl': 'train_env_cones_scenegraph.pkl',
        'eval_pkl': 'eval_env_cones_scenegraph.pkl',
        'log_tag': '_mode_B-C',
    },
    'C': {
        'config': 'workzone_heatmap.json',
        'train_pkl': 'train_env_heatmap.pkl',
        'eval_pkl': 'eval_env_heatmap.pkl',
        'log_tag': '_mode_C',
    },
    'C-native': {
        'config': 'workzone_heatmap_native.json',
        'train_pkl': 'train_env_heatmap_native.pkl',
        'eval_pkl': 'eval_env_heatmap_native.pkl',
        'log_tag': '_mode_C_native',
    },
    'D': {
        'config': 'workzone_base.json',
        'train_pkl': 'train_env_hdmap_plain.pkl',
        'eval_pkl': 'eval_env_hdmap_plain.pkl',
        'log_tag': '_mode_D',
    },
    'E': {
        'config': 'workzone_cones_raster.json',
        'train_pkl': 'train_env_cones_raster_plain.pkl',
        'eval_pkl': 'eval_env_cones_raster_plain.pkl',
        'log_tag': '_mode_E',
    },
}


def main():
    parser = argparse.ArgumentParser(
        description='Work zone Trajectron++ training wrapper')
    parser.add_argument('--mode', type=str, default='C',
                        choices=list(MODE_TABLE.keys()),
                        help='Training mode: A | B-A | B-B | B-C | C | C-native | D | E')
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(_PROJECT_ROOT, 'integration', 'data'),
                        help='Directory containing train/eval .pkl files')
    parser.add_argument('--log_dir', type=str,
                        default=os.path.join(_PROJECT_ROOT, 'experiments', 'logs'),
                        help='TensorBoard / model log directory')
    parser.add_argument('--train_epochs', type=int, default=20)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--vis_every', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--eval_batch_size', type=int, default=1024)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--eval_device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--preprocess_workers', type=int, default=0)
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='batches to prefetch per worker when preprocess_workers>0; use 1 to reduce memory')
    parser.add_argument('--amp', action='store_true',
                        help='enable automatic mixed precision (AMP) training')
    parser.add_argument('--offline_scene_graph', type=str, default='yes')
    parser.add_argument('--dynamic_edges', type=str, default='no')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--python', type=str, default=sys.executable,
                        help='Python interpreter to use')
    parser.add_argument('--eval_kde', type=str, default='no', choices=['yes', 'no'],
                        help='whether to compute KDE during eval (yes/no)')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='override learning rate from config json')
    parser.add_argument('--dropout', type=float, default=None,
                        help='override map_encoder dropout from config json')
    parser.add_argument('--kl_weight', type=float, default=None,
                        help='override kl_weight from config json')
    parser.add_argument('--extra_tag', type=str, default=None,
                        help='custom suffix appended to TensorBoard run name '
                             '(for parameters without auto-detection)')
    args = parser.parse_args()

    # --- Build concise auto-tag from explicitly overridden hyperparameters ---
    # Detect which taggable args were explicitly passed on the command line
    _cli_flags = set(sys.argv)
    auto_tag_parts = []
    for arg_name, abbrev in PARAM_TAG_ABBREV.items():
        # Check if any option string for this dest appeared in sys.argv
        if f'--{arg_name}' in _cli_flags:
            val = getattr(args, arg_name)
            if val is not None:
                fmt = f'{val:g}' if isinstance(val, float) else str(val)
                auto_tag_parts.append(f'_{abbrev}{fmt}')
    if args.extra_tag:
        auto_tag_parts.append(f'_{args.extra_tag}')
    auto_tag_suffix = ''.join(auto_tag_parts)

    mode_info = MODE_TABLE[args.mode]

    # Resolve paths
    config_path = os.path.join(_PROJECT_ROOT, 'integration', 'config',
                               mode_info['config'])
    train_script = os.path.join(_PROJECT_ROOT, 'trajectron_plus_plus', 'trajectron',
                                'train.py')

    # Verify files exist
    for label, path in [('Config', config_path),
                        ('Train script', train_script)]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found: {path}")
            sys.exit(1)

    train_pkl = os.path.join(args.data_dir, mode_info['train_pkl'])
    eval_pkl = os.path.join(args.data_dir, mode_info['eval_pkl'])
    for label, path in [('Train data', train_pkl), ('Eval data', eval_pkl)]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found: {path}")
            print(f"  Run build_environment.py first with the appropriate --map_mode")
            sys.exit(1)

    # Build Trajectron++ CLI arguments
    cmd = [
        args.python, train_script,
        '--conf', config_path,
        '--data_dir', args.data_dir,
        '--train_data_dict', mode_info['train_pkl'],
        '--eval_data_dict', mode_info['eval_pkl'],
        '--log_dir', args.log_dir,
        '--log_tag', mode_info['log_tag'] + auto_tag_suffix,
        '--train_epochs', str(args.train_epochs),
        '--batch_size', str(args.batch_size),
        '--eval_batch_size', str(args.eval_batch_size),
        '--device', args.device,
        '--seed', str(args.seed),
        '--preprocess_workers', str(args.preprocess_workers),
        '--prefetch_factor', str(args.prefetch_factor),
        '--offline_scene_graph', args.offline_scene_graph,
        '--dynamic_edges', args.dynamic_edges,
        '--map_encoding',  # Always enable map encoding
        '--save_every', str(args.save_every),
    ]

    if args.eval_every is not None:
        cmd.extend(['--eval_every', str(args.eval_every)])
    if args.vis_every is not None:
        cmd.extend(['--vis_every', str(args.vis_every)])
    if args.eval_device is not None:
        cmd.extend(['--eval_device', args.eval_device])
    if args.augment:
        cmd.append('--augment')
    if args.amp:
        cmd.append('--amp')
    cmd.extend(['--eval_kde', args.eval_kde])
    if args.learning_rate is not None:
        cmd.extend(['--learning_rate', str(args.learning_rate)])
    if args.dropout is not None:
        cmd.extend(['--dropout', str(args.dropout)])
    if args.kl_weight is not None:
        cmd.extend(['--kl_weight', str(args.kl_weight)])

    print(f"=== Work Zone Training: Mode {args.mode} ===")
    print(f"  Config:     {config_path}")
    print(f"  Train data: {train_pkl}")
    print(f"  Eval data:  {eval_pkl}")
    print(f"  Log dir:    {args.log_dir}")
    print(f"  Epochs:     {args.train_epochs}")
    print(f"  Command:    {' '.join(cmd)}")
    print()

    # Launch training as subprocess (keeps Trajectron++'s argument_parser happy)
    result = subprocess.run(cmd, cwd=os.path.dirname(train_script))
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
