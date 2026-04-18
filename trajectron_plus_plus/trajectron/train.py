import torch
from torch import nn, optim, utils
import numpy as np
import os
import time
import dill
import json
import random
import pathlib
import warnings
import contextlib
from tqdm import tqdm
import visualization
import evaluation
import matplotlib.pyplot as plt
from argument_parser import args
from model.trajectron import Trajectron
from model.model_registrar import ModelRegistrar
from model.model_utils import cyclical_lr
from collections import defaultdict
from model.dataset import (EnvironmentDataset, collate, make_collate_fn,
                            collate_with_metadata, make_collate_fn_with_metadata)
from environment.map import GeometricMap
from tensorboardX import SummaryWriter
# torch.autograd.set_detect_anomaly(True)

if not torch.cuda.is_available() or args.device == 'cpu':
    args.device = torch.device('cpu')
else:
    if torch.cuda.device_count() == 1:
        # If you have CUDA_VISIBLE_DEVICES set, which you should,
        # then this will prevent leftover flag arguments from
        # messing with the device allocation.
        args.device = 'cuda:0'

    args.device = torch.device(args.device)

if args.eval_device is None:
    args.eval_device = torch.device('cpu')

# This is needed for memory pinning using a DataLoader (otherwise memory is pinned to cuda:0 by default)
torch.cuda.set_device(args.device)

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def _chunked_predict(trajectron_model, scene, timesteps, ph,
                     chunk_size=50, **predict_kwargs):
    """Call predict in timestep chunks to avoid GPU OOM, then merge results."""
    if len(timesteps) <= chunk_size:
        return trajectron_model.predict(scene, timesteps, ph, **predict_kwargs)

    merged = {}
    for start in range(0, len(timesteps), chunk_size):
        chunk_ts = timesteps[start:start + chunk_size]
        chunk_pred = trajectron_model.predict(scene, chunk_ts, ph, **predict_kwargs)
        for ts, nodes_dict in chunk_pred.items():
            if ts not in merged:
                merged[ts] = {}
            merged[ts].update(nodes_dict)
    return merged


def _clear_map_cuda_caches(scenes):
    """Clear GeometricMap CUDA caches before spawning DataLoader workers.

    When the main process caches map tensors on CUDA (e.g. during predict()),
    those tensors get shared via CUDA IPC when workers are spawned.  Workers
    cannot re-serialise IPC-received CUDA tensors, causing RuntimeError.
    Clearing the caches forces workers to build fresh (non-IPC) tensors.
    """
    for scene in scenes:
        if scene.map is not None:
            for nt_map in scene.map.values():
                if hasattr(nt_map, 'clear_cuda_cache'):
                    nt_map.clear_cuda_cache()


def _get_viz_map(scene):
    """Get map for visualization: prefer VISUALIZATION (nuScenes), else first available (e.g. VEHICLE)."""
    if scene.map is None:
        return None
    return scene.map.get('VISUALIZATION') or next(iter(scene.map.values()), None)


def main():
    # Load hyperparameters from json
    if not os.path.exists(args.conf):
        print('Config json not found!')
    with open(args.conf, 'r', encoding='utf-8') as conf_json:
        hyperparams = json.load(conf_json)

    map_chunk_size = hyperparams.get('map_chunk_size', None)
    if map_chunk_size is not None:
        GeometricMap._map_chunk_size = map_chunk_size

    # Add hyperparams from arguments
    hyperparams['dynamic_edges'] = args.dynamic_edges
    hyperparams['edge_state_combine_method'] = args.edge_state_combine_method
    hyperparams['edge_influence_combine_method'] = args.edge_influence_combine_method
    hyperparams['edge_addition_filter'] = args.edge_addition_filter
    hyperparams['edge_removal_filter'] = args.edge_removal_filter
    hyperparams['batch_size'] = args.batch_size
    hyperparams['k_eval'] = args.k_eval
    hyperparams['offline_scene_graph'] = args.offline_scene_graph
    hyperparams['incl_robot_node'] = args.incl_robot_node
    hyperparams['node_freq_mult_train'] = args.node_freq_mult_train
    hyperparams['node_freq_mult_eval'] = args.node_freq_mult_eval
    hyperparams['scene_freq_mult_train'] = args.scene_freq_mult_train
    hyperparams['scene_freq_mult_eval'] = args.scene_freq_mult_eval
    hyperparams['scene_freq_mult_viz'] = args.scene_freq_mult_viz
    hyperparams['edge_encoding'] = not args.no_edge_encoding
    hyperparams['use_map_encoding'] = args.map_encoding
    hyperparams['augment'] = args.augment
    hyperparams['override_attention_radius'] = args.override_attention_radius

    if args.learning_rate is not None:
        hyperparams['learning_rate'] = args.learning_rate
    if args.dropout is not None:
        for nt in hyperparams.get('map_encoder', {}):
            hyperparams['map_encoder'][nt]['dropout'] = args.dropout
    if args.kl_weight is not None:
        hyperparams['kl_weight'] = args.kl_weight

    print('-----------------------')
    print('| TRAINING PARAMETERS |')
    print('-----------------------')
    print('| batch_size: %d' % args.batch_size)
    print('| device: %s' % args.device)
    print('| eval_device: %s' % args.eval_device)
    print('| Offline Scene Graph Calculation: %s' % args.offline_scene_graph)
    print('| EE state_combine_method: %s' % args.edge_state_combine_method)
    print('| EIE scheme: %s' % args.edge_influence_combine_method)
    print('| dynamic_edges: %s' % args.dynamic_edges)
    print('| robot node: %s' % args.incl_robot_node)
    print('| edge_addition_filter: %s' % args.edge_addition_filter)
    print('| edge_removal_filter: %s' % args.edge_removal_filter)
    print('| MHL: %s' % hyperparams['minimum_history_length'])
    print('| PH: %s' % hyperparams['prediction_horizon'])
    use_amp = (args.device.type == torch.device('cuda').type and args.amp)
    if use_amp:
        print('| AMP: enabled')
    else:
        print('| AMP: disabled')
    print('-----------------------')

    log_writer = None
    model_dir = None
    if not args.debug:
        # Create the log and model directiory if they're not present.
        model_dir = os.path.join(args.log_dir,
                                 'models_' + time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime()) + args.log_tag)
        pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)

        # Save config to model directory
        with open(os.path.join(model_dir, 'config.json'), 'w') as conf_json:
            json.dump(hyperparams, conf_json)

        log_writer = SummaryWriter(log_dir=model_dir)

    # Load training and evaluation environments and scenes
    train_scenes = []
    train_data_path = os.path.join(args.data_dir, args.train_data_dict)
    with open(train_data_path, 'rb') as f:
        train_env = dill.load(f, encoding='latin1')

    for attention_radius_override in args.override_attention_radius:
        node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
        train_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

    if train_env.robot_type is None and hyperparams['incl_robot_node']:
        train_env.robot_type = train_env.NodeType[0]  # TODO: Make more general, allow the user to specify?
        for scene in train_env.scenes:
            scene.add_robot_from_nodes(train_env.robot_type)

    train_scenes = train_env.scenes
    train_scenes_sample_probs = train_env.scenes_freq_mult_prop if args.scene_freq_mult_train else None

    train_dataset = EnvironmentDataset(train_env,
                                       hyperparams['state'],
                                       hyperparams['pred_state'],
                                       scene_freq_mult=hyperparams['scene_freq_mult_train'],
                                       node_freq_mult=hyperparams['node_freq_mult_train'],
                                       hyperparams=hyperparams,
                                       min_history_timesteps=hyperparams['minimum_history_length'],
                                       min_future_timesteps=hyperparams['prediction_horizon'],
                                       return_robot=not args.incl_robot_node)
    train_data_loader = dict()
    for node_type_data_set in train_dataset:
        if len(node_type_data_set) == 0:
            continue

        collate_fn = make_collate_fn(train_env, node_type_data_set.node_type) if args.preprocess_workers > 0 else collate
        # pin_memory=False when use_map_encoding or num_workers>0: collate returns map on CUDA, cannot pin
        dl_kwargs = dict(
            collate_fn=collate_fn,
            pin_memory=False if (args.device == 'cpu' or hyperparams.get('use_map_encoding', False) or args.preprocess_workers > 0) else True,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.preprocess_workers,
        )
        if args.preprocess_workers > 0:
            dl_kwargs['prefetch_factor'] = args.prefetch_factor
            # spawn required: collate uses CUDA for map cropping; fork cannot re-init CUDA in subprocess
            dl_kwargs['multiprocessing_context'] = 'spawn'
        node_type_dataloader = utils.data.DataLoader(node_type_data_set, **dl_kwargs)
        train_data_loader[node_type_data_set.node_type] = node_type_dataloader

    print(f"Loaded training data from {train_data_path}")

    eval_scenes = []
    eval_scenes_sample_probs = None
    if args.eval_every is not None:
        eval_data_path = os.path.join(args.data_dir, args.eval_data_dict)
        with open(eval_data_path, 'rb') as f:
            eval_env = dill.load(f, encoding='latin1')

        for attention_radius_override in args.override_attention_radius:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            eval_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

        if eval_env.robot_type is None and hyperparams['incl_robot_node']:
            eval_env.robot_type = eval_env.NodeType[0]  # TODO: Make more general, allow the user to specify?
            for scene in eval_env.scenes:
                scene.add_robot_from_nodes(eval_env.robot_type)

        eval_scenes = eval_env.scenes
        eval_scenes_sample_probs = eval_env.scenes_freq_mult_prop if args.scene_freq_mult_eval else None

        eval_dataset = EnvironmentDataset(eval_env,
                                          hyperparams['state'],
                                          hyperparams['pred_state'],
                                          scene_freq_mult=hyperparams['scene_freq_mult_eval'],
                                          node_freq_mult=hyperparams['node_freq_mult_eval'],
                                          hyperparams=hyperparams,
                                          min_history_timesteps=hyperparams['minimum_history_length'],
                                          min_future_timesteps=hyperparams['prediction_horizon'],
                                          return_robot=not args.incl_robot_node)
        eval_data_loader = dict()
        for node_type_data_set in eval_dataset:
            if len(node_type_data_set) == 0:
                continue

            collate_fn = make_collate_fn(eval_env, node_type_data_set.node_type) if args.preprocess_workers > 0 else collate
            # pin_memory=False when use_map_encoding or num_workers>0: collate returns map on CUDA, cannot pin
            eval_dl_kwargs = dict(
                collate_fn=collate_fn,
                pin_memory=False if (args.eval_device == 'cpu' or hyperparams.get('use_map_encoding', False) or args.preprocess_workers > 0) else True,
                batch_size=args.eval_batch_size,
                shuffle=False,
                num_workers=args.preprocess_workers,
            )
            if args.preprocess_workers > 0:
                eval_dl_kwargs['prefetch_factor'] = args.prefetch_factor
                eval_dl_kwargs['multiprocessing_context'] = 'spawn'
            node_type_dataloader = utils.data.DataLoader(node_type_data_set, **eval_dl_kwargs)
            eval_data_loader[node_type_data_set.node_type] = node_type_dataloader

        # DataLoader for MM Evaluation (metadata-aware for per-node predictions)
        eval_pred_data_loader = dict()
        for node_type_data_set in eval_dataset:
            if len(node_type_data_set) == 0:
                continue
            pred_collate_fn = (make_collate_fn_with_metadata(eval_env, node_type_data_set.node_type)
                               if args.preprocess_workers > 0 else collate_with_metadata)
            pred_dl_kwargs = dict(
                collate_fn=pred_collate_fn,
                pin_memory=False,
                batch_size=args.eval_batch_size,
                shuffle=False,
                num_workers=args.preprocess_workers,
            )
            if args.preprocess_workers > 0:
                pred_dl_kwargs['prefetch_factor'] = args.prefetch_factor
                pred_dl_kwargs['multiprocessing_context'] = 'spawn'
            eval_pred_data_loader[node_type_data_set.node_type] = utils.data.DataLoader(
                node_type_data_set, **pred_dl_kwargs)

        print(f"Loaded evaluation data from {eval_data_path}")

    # Offline Calculate Scene Graph
    if hyperparams['offline_scene_graph'] == 'yes':
        print(f"Offline calculating scene graphs")
        for scene in tqdm(train_scenes, desc="Train scene graphs", ncols=80, unit="scene"):
            scene.calculate_scene_graph(train_env.attention_radius,
                                        hyperparams['edge_addition_filter'],
                                        hyperparams['edge_removal_filter'])

        for scene in tqdm(eval_scenes, desc="Eval scene graphs", ncols=80, unit="scene"):
            scene.calculate_scene_graph(eval_env.attention_radius,
                                        hyperparams['edge_addition_filter'],
                                        hyperparams['edge_removal_filter'])

    model_registrar = ModelRegistrar(model_dir, args.device)

    trajectron = Trajectron(model_registrar,
                            hyperparams,
                            log_writer,
                            args.device)

    trajectron.set_environment(train_env)
    trajectron.set_annealing_params()
    print('Created Training Model.')

    eval_trajectron = None
    if args.eval_every is not None or args.vis_every is not None:
        eval_trajectron = Trajectron(model_registrar,
                                     hyperparams,
                                     log_writer,
                                     args.eval_device)
        eval_trajectron.set_environment(eval_env)
        eval_trajectron.set_annealing_params()
    print('Created Evaluation Model.')

    optimizer = dict()
    lr_scheduler = dict()
    for node_type in train_env.NodeType:
        if node_type not in hyperparams['pred_state']:
            continue
        optimizer[node_type] = optim.Adam([{'params': model_registrar.get_all_but_name_match('map_encoder').parameters()},
                                           {'params': model_registrar.get_name_match('map_encoder').parameters(), 'lr':0.0008}], lr=hyperparams['learning_rate'])
        # Set Learning Rate
        if hyperparams['learning_rate_style'] == 'const':
            lr_scheduler[node_type] = optim.lr_scheduler.ExponentialLR(optimizer[node_type], gamma=1.0)
        elif hyperparams['learning_rate_style'] == 'exp':
            lr_scheduler[node_type] = optim.lr_scheduler.ExponentialLR(optimizer[node_type],
                                                                       gamma=hyperparams['learning_decay_rate'])

    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    #################################
    #           TRAINING            #
    #################################
    curr_iter_node_type = {node_type: 0 for node_type in train_data_loader.keys()}
    global_step = 0
    for epoch in range(1, args.train_epochs + 1):
        model_registrar.to(args.device)
        train_dataset.augment = args.augment
        for node_type, data_loader in train_data_loader.items():
            curr_iter = curr_iter_node_type[node_type]
            pbar = tqdm(data_loader, ncols=80)
            for batch in pbar:
                trajectron.set_curr_iter(curr_iter)
                trajectron.step_annealers(node_type)
                optimizer[node_type].zero_grad()
                with torch.cuda.amp.autocast() if use_amp else contextlib.nullcontext():
                    train_loss = trajectron.train_loss(batch, node_type)
                pbar.set_description(f"Epoch {epoch}, {node_type} L: {train_loss.item():.2f}")
                if use_amp:
                    scaler.scale(train_loss).backward()
                    if hyperparams['grad_clip'] is not None:
                        scaler.unscale_(optimizer[node_type])
                        nn.utils.clip_grad_value_(model_registrar.parameters(), hyperparams['grad_clip'])
                    scaler.step(optimizer[node_type])
                    scaler.update()
                else:
                    train_loss.backward()
                    if hyperparams['grad_clip'] is not None:
                        nn.utils.clip_grad_value_(model_registrar.parameters(), hyperparams['grad_clip'])
                    optimizer[node_type].step()

                # Stepping forward the learning rate scheduler and annealers.
                lr_scheduler[node_type].step()

                if log_writer is not None:
                    log_writer.add_scalar(f"{node_type}/train/loss", train_loss.item(), global_step)
                global_step += 1

                curr_iter += 1
            curr_iter_node_type[node_type] = curr_iter
        train_dataset.augment = False
        if args.eval_every is not None or args.vis_every is not None:
            eval_trajectron.set_curr_iter(epoch)

        #################################
        #           EVALUATION          #
        #################################
        if args.eval_every is not None and not args.debug and epoch % args.eval_every == 0 and epoch > 0:
            max_hl = hyperparams['maximum_history_length']
            ph = hyperparams['prediction_horizon']
            model_registrar.to(args.eval_device)
            # Clear CUDA caches before eval DataLoader workers (epoch 2+)
            if args.preprocess_workers > 0:
                _clear_map_cuda_caches(eval_scenes)
            with torch.no_grad():
                # Calculate evaluation loss
                for node_type, data_loader in eval_data_loader.items():
                    eval_loss = []
                    print(f"Starting Evaluation @ epoch {epoch} for node type: {node_type}")
                    pbar = tqdm(data_loader, ncols=80)
                    for batch in pbar:
                        eval_loss_node_type = eval_trajectron.eval_loss(batch, node_type)
                        pbar.set_description(f"Epoch {epoch}, {node_type} L: {eval_loss_node_type.item():.2f}")
                        eval_loss.append({node_type: {'nll': [eval_loss_node_type]}})
                        del batch

                    evaluation.log_batch_errors(eval_loss,
                                                log_writer,
                                                f"{node_type}/eval_loss",
                                                epoch)

                # Predict batch timesteps for evaluation dataset evaluation
                eval_batch_errors = []
                for scene in tqdm(eval_scenes, desc='Sample Evaluation', ncols=80):
                    timesteps = scene.sample_timesteps(args.eval_batch_size)

                    predictions = _chunked_predict(eval_trajectron, scene,
                                                   timesteps, ph,
                                                   num_samples=args.k_eval,
                                                   min_future_timesteps=ph,
                                                   full_dist=False)

                    eval_batch_errors.append(evaluation.compute_batch_statistics(predictions,
                                                                                 scene.dt,
                                                                                 max_hl=max_hl,
                                                                                 ph=ph,
                                                                                 node_type_enum=eval_env.NodeType,
                                                                                 map=scene.map,
                                                                                 kde=(args.eval_kde == 'yes')))

                evaluation.log_batch_errors(eval_batch_errors,
                                            log_writer,
                                            "eval",
                                            epoch)

                # Predict maximum likelihood via DataLoader-based batched inference
                # Toggle metadata on for prediction DataLoaders
                for nds in eval_dataset:
                    nds.return_metadata = True

                # Sample Evaluation above ran predict() in the main process,
                # which re-cached CUDA tensors on GeometricMap.  Clear again
                # before spawning MM Eval workers.
                if args.preprocess_workers > 0:
                    _clear_map_cuda_caches(eval_scenes)

                scene_preds_ml = defaultdict(dict)  # scene -> {ts: {node: pred}}
                for node_type, data_loader in eval_pred_data_loader.items():
                    pbar = tqdm(data_loader, desc=f'MM Eval {node_type}', ncols=80)
                    for batch_with_meta in pbar:
                        batch_data, meta_list = batch_with_meta
                        predictions_np = eval_trajectron.predict_from_batch(
                            batch_data, node_type, ph,
                            num_samples=1, z_mode=True, gmm_mode=True, full_dist=False)
                        for i, (scene, t, node) in enumerate(meta_list):
                            pred_i = np.transpose(predictions_np[:, [i]], (1, 0, 2, 3))
                            if t not in scene_preds_ml[scene]:
                                scene_preds_ml[scene][t] = {}
                            scene_preds_ml[scene][t][node] = pred_i

                for nds in eval_dataset:
                    nds.return_metadata = False

                eval_batch_errors_ml = []
                for scene, pred_dict in scene_preds_ml.items():
                    eval_batch_errors_ml.append(
                        evaluation.compute_batch_statistics(pred_dict,
                                                            scene.dt,
                                                            max_hl=max_hl,
                                                            ph=ph,
                                                            map=scene.map,
                                                            node_type_enum=eval_env.NodeType,
                                                            kde=False))

                evaluation.log_batch_errors(eval_batch_errors_ml,
                                            log_writer,
                                            "eval/ml",
                                            epoch)

        if args.save_every is not None and args.debug is False and epoch % args.save_every == 0:
            model_registrar.save_models(epoch)


if __name__ == '__main__':
    main()
