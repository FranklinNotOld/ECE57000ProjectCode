import torch
import numpy as np
import collections.abc
from torch.utils.data._utils.collate import default_collate
import dill
container_abcs = collections.abc


class _CollateFnWithScenes:
    """Picklable collate_fn wrapper for num_workers > 0. Lambda cannot be pickled on Windows spawn."""

    def __init__(self, scenes_by_name, node_type):
        self.scenes_by_name = scenes_by_name
        self.node_type = node_type

    def __call__(self, batch):
        return collate(batch, scenes_by_name=self.scenes_by_name, node_type=self.node_type)


def make_collate_fn(env, node_type):
    """
    Create collate_fn that resolves scene_id->scene_map when num_workers > 0.
    Workers return scene_name instead of scene_map to avoid OOM during serialization.
    """
    scenes_by_name = {s.name: s for s in env.scenes}
    return _CollateFnWithScenes(scenes_by_name, node_type)


def collate_with_metadata(batch, scenes_by_name=None, node_type=None):
    """Collate batch of (data_tuple, metadata) pairs from return_metadata=True datasets.

    Separates metadata from data, delegates data collation to existing collate(),
    and returns (collated_data, metadata_list).
    """
    if len(batch) == 0:
        return batch
    data_list = [item[0] for item in batch]
    meta_list = [item[1] for item in batch]
    collated_data = collate(data_list, scenes_by_name=scenes_by_name, node_type=node_type)
    return collated_data, meta_list


class _CollateFnWithScenesAndMeta:
    """Picklable collate_fn wrapper for metadata-aware collation with num_workers > 0."""

    def __init__(self, scenes_by_name, node_type):
        self.scenes_by_name = scenes_by_name
        self.node_type = node_type

    def __call__(self, batch):
        return collate_with_metadata(batch, scenes_by_name=self.scenes_by_name, node_type=self.node_type)


def make_collate_fn_with_metadata(env, node_type):
    """Create metadata-aware collate_fn for prediction evaluation DataLoaders."""
    scenes_by_name = {s.name: s for s in env.scenes}
    return _CollateFnWithScenesAndMeta(scenes_by_name, node_type)


def restore(data):
    """
    In case we dilled some structures to share between multiple process this function will restore them.
    If the data input are not bytes we assume it was not dilled in the first place

    :param data: Possibly dilled data structure
    :return: Un-dilled data structure
    """
    if type(data) is bytes:
        return dill.loads(data)
    return data


def collate(batch, scenes_by_name=None, node_type=None):
    """
    Collate batch. When num_workers > 0, workers return (scene_name, map_point, heading_angle, patch_size)
    instead of (scene_map, ...) to avoid OOM. Pass scenes_by_name and node_type to resolve maps in main process.
    """
    if len(batch) == 0:
        return batch
    elem = batch[0]
    if elem is None:
        return None
    elif isinstance(elem, container_abcs.Sequence):
        if len(elem) == 4:  # map_tuple: (scene_map|scene_name, map_point, heading_angle, patch_size)
            first_el, scene_pts, heading_angle, patch_size = zip(*batch)
            if heading_angle[0] is None:
                heading_angle = None
            else:
                heading_angle = torch.Tensor(heading_angle)
            scene_pts_t = torch.tensor(np.array(scene_pts), dtype=torch.float)
            # Worker returns scene_name (str); main process has scene_map (Map object)
            if isinstance(first_el[0], str) and scenes_by_name is not None and node_type is not None:
                scene_maps = [scenes_by_name[name].map[node_type] for name in first_el]
            else:
                scene_maps = list(first_el)
            _map_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            map_result = scene_maps[0].get_cropped_maps_from_scene_map_batch(
                scene_maps, scene_pts=scene_pts_t, patch_size=patch_size[0], rotation=heading_angle,
                device=_map_device)
            return map_result
        # Mode B-B: cone_tuple is a 2-tuple of (cone_rel [K,2], mask [K])
        if len(elem) == 2 and isinstance(elem[0], torch.Tensor) and elem[0].dim() == 2:
            cone_rel_list, mask_list = zip(*batch)
            return (torch.stack(cone_rel_list, dim=0),
                    torch.stack(mask_list, dim=0))
        transposed = zip(*batch)
        return [collate(samples, scenes_by_name=scenes_by_name, node_type=node_type) for samples in transposed]
    elif isinstance(elem, container_abcs.Mapping):
        # We have to dill the neighbors structures. Otherwise each tensor is put into
        # shared memory separately -> slow, file pointer overhead
        # we only do this in multiprocessing
        neighbor_dict = {key: [d[key] for d in batch] for key in elem}
        return dill.dumps(neighbor_dict) if torch.utils.data.get_worker_info() else neighbor_dict
    return default_collate(batch)


def get_relative_robot_traj(env, state, node_traj, robot_traj, node_type, robot_type):
    # TODO: We will have to make this more generic if robot_type != node_type
    # Make Robot State relative to node
    _, std = env.get_standardize_params(state[robot_type], node_type=robot_type)
    std[0:2] = env.attention_radius[(node_type, robot_type)]
    robot_traj_st = env.standardize(robot_traj,
                                    state[robot_type],
                                    node_type=robot_type,
                                    mean=node_traj,
                                    std=std)
    robot_traj_st_t = torch.tensor(robot_traj_st, dtype=torch.float)

    return robot_traj_st_t


def get_node_timestep_data(env, scene, t, node, state, pred_state,
                           edge_types, max_ht, max_ft, hyperparams,
                           scene_graph=None):
    """
    Pre-processes the data for a single batch element: node state over time for a specific time in a specific scene
    as well as the neighbour data for it.

    :param env: Environment
    :param scene: Scene
    :param t: Timestep in scene
    :param node: Node
    :param state: Specification of the node state
    :param pred_state: Specification of the prediction state
    :param edge_types: List of all Edge Types for which neighbours are pre-processed
    :param max_ht: Maximum history timesteps
    :param max_ft: Maximum future timesteps (prediction horizon)
    :param hyperparams: Model hyperparameters
    :param scene_graph: If scene graph was already computed for this scene and time you can pass it here
    :return: Batch Element
    """

    # Node
    timestep_range_x = np.array([t - max_ht, t])
    timestep_range_y = np.array([t + 1, t + max_ft])

    x = node.get(timestep_range_x, state[node.type])
    y = node.get(timestep_range_y, pred_state[node.type])
    first_history_index = (max_ht - node.history_points_at(t)).clip(0)

    _, std = env.get_standardize_params(state[node.type], node.type)
    std[0:2] = env.attention_radius[(node.type, node.type)]
    rel_state = np.zeros_like(x[0])
    rel_state[0:2] = np.array(x)[-1, 0:2]
    x_st = env.standardize(x, state[node.type], node.type, mean=rel_state, std=std)
    if list(pred_state[node.type].keys())[0] == 'position':  # If we predict position we do it relative to current pos
        y_st = env.standardize(y, pred_state[node.type], node.type, mean=rel_state[0:2])
    else:
        y_st = env.standardize(y, pred_state[node.type], node.type)

    x_t = torch.tensor(x, dtype=torch.float)
    y_t = torch.tensor(y, dtype=torch.float)
    x_st_t = torch.tensor(x_st, dtype=torch.float)
    y_st_t = torch.tensor(y_st, dtype=torch.float)

    # Neighbors
    neighbors_data_st = None
    neighbors_edge_value = None
    if hyperparams['edge_encoding']:
        # Scene Graph
        scene_graph = scene.get_scene_graph(t,
                                            env.attention_radius,
                                            hyperparams['edge_addition_filter'],
                                            hyperparams['edge_removal_filter']) if scene_graph is None else scene_graph

        neighbors_data_st = dict()
        neighbors_edge_value = dict()
        for edge_type in edge_types:
            neighbors_data_st[edge_type] = list()
            # We get all nodes which are connected to the current node for the current timestep
            connected_nodes = scene_graph.get_neighbors(node, edge_type[1])

            if hyperparams['dynamic_edges'] == 'yes':
                # We get the edge masks for the current node at the current timestep
                edge_masks = torch.tensor(scene_graph.get_edge_scaling(node), dtype=torch.float)
                neighbors_edge_value[edge_type] = edge_masks
            else:
                # When dynamic_edges is 'no', encode_edge does not use edge values; use placeholder
                neighbors_edge_value[edge_type] = []

            for connected_node in connected_nodes:
                neighbor_state_np = connected_node.get(np.array([t - max_ht, t]),
                                                       state[connected_node.type],
                                                       padding=0.0)

                # Make State relative to node where neighbor and node have same state
                _, std = env.get_standardize_params(state[connected_node.type], node_type=connected_node.type)
                std[0:2] = env.attention_radius[edge_type]
                equal_dims = np.min((neighbor_state_np.shape[-1], x.shape[-1]))
                rel_state = np.zeros_like(neighbor_state_np)
                rel_state[:, ..., :equal_dims] = x[-1, ..., :equal_dims]
                neighbor_state_np_st = env.standardize(neighbor_state_np,
                                                       state[connected_node.type],
                                                       node_type=connected_node.type,
                                                       mean=rel_state,
                                                       std=std)

                neighbor_state = torch.tensor(neighbor_state_np_st, dtype=torch.float)
                neighbors_data_st[edge_type].append(neighbor_state)

    # Robot
    robot_traj_st_t = None
    if hyperparams['incl_robot_node']:
        timestep_range_r = np.array([t, t + max_ft])
        if scene.non_aug_scene is not None:
            robot = scene.get_node_by_id(scene.non_aug_scene.robot.id)
        else:
            robot = scene.robot
        robot_type = robot.type
        robot_traj = robot.get(timestep_range_r, state[robot_type], padding=0.0)
        node_state = np.zeros_like(robot_traj[0])
        node_state[:x.shape[1]] = x[-1]
        robot_traj_st_t = get_relative_robot_traj(env, state, node_state, robot_traj, node.type, robot_type)

    # Map
    map_tuple = None
    if hyperparams['use_map_encoding']:
        if node.type in hyperparams['map_encoder']:
            if node.non_aug_node is not None:
                x = node.non_aug_node.get(np.array([t]), state[node.type])
            me_hyp = hyperparams['map_encoder'][node.type]
            if 'heading_state_index' in me_hyp:
                heading_state_index = me_hyp['heading_state_index']
                # We have to rotate the map in the opposit direction of the agent to match them
                if type(heading_state_index) is list:  # infer from velocity or heading vector
                    heading_angle = -np.arctan2(x[-1, heading_state_index[1]],
                                                x[-1, heading_state_index[0]]) * 180 / np.pi
                else:
                    heading_angle = -x[-1, heading_state_index] * 180 / np.pi
            else:
                heading_angle = None

            scene_map = scene.map[node.type]
            map_point = x[-1, :2]
            patch_size = hyperparams['map_encoder'][node.type]['patch_size']
            # When in DataLoader worker: return scene_id instead of scene_map to avoid
            # serializing full map when passing batch to main process (causes OOM).
            if torch.utils.data.get_worker_info() is not None:
                base_scene = getattr(scene, 'non_aug_scene', None) or scene
                map_tuple = (base_scene.name, map_point, heading_angle, patch_size)
            else:
                map_tuple = (scene_map, map_point, heading_angle, patch_size)

    # Cone data for Mode B-B (PointNet cone encoder)
    cone_tuple = None
    cone_cfg = hyperparams.get('cone_encoder')
    if cone_cfg and cone_cfg.get('enabled', False):
        K = cone_cfg['k_nearest']
        node_pos_np = np.array(x_t[-1, :2])  # current position [2]
        if hasattr(scene, 'cones') and scene.cones is not None and len(scene.cones) > 0:
            dists = np.linalg.norm(scene.cones - node_pos_np, axis=1)
            nearest_idx = np.argsort(dists)[:K]
            cone_rel = scene.cones[nearest_idx] - node_pos_np  # [<=K, 2]
            n_actual = len(cone_rel)
            # Pad to exactly K
            if n_actual < K:
                pad = np.zeros((K - n_actual, 2), dtype=np.float64)
                cone_rel = np.concatenate([cone_rel, pad], axis=0)
            mask = np.zeros(K, dtype=bool)
            mask[:n_actual] = True
        else:
            cone_rel = np.zeros((K, 2), dtype=np.float64)
            mask = np.zeros(K, dtype=bool)
        cone_tuple = (torch.tensor(cone_rel, dtype=torch.float),
                      torch.tensor(mask, dtype=torch.bool))

    return (first_history_index, x_t, y_t, x_st_t, y_st_t, neighbors_data_st,
            neighbors_edge_value, robot_traj_st_t, map_tuple, cone_tuple)


def get_timesteps_data(env, scene, t, node_type, state, pred_state,
                       edge_types, min_ht, max_ht, min_ft, max_ft, hyperparams):
    """
    Puts together the inputs for ALL nodes in a given scene and timestep in it.

    :param env: Environment
    :param scene: Scene
    :param t: Timestep in scene
    :param node_type: Node Type of nodes for which the data shall be pre-processed
    :param state: Specification of the node state
    :param pred_state: Specification of the prediction state
    :param edge_types: List of all Edge Types for which neighbors are pre-processed
    :param max_ht: Maximum history timesteps
    :param max_ft: Maximum future timesteps (prediction horizon)
    :param hyperparams: Model hyperparameters
    :return:
    """
    nodes_per_ts = scene.present_nodes(t,
                                       type=node_type,
                                       min_history_timesteps=min_ht,
                                       min_future_timesteps=max_ft,
                                       return_robot=not hyperparams['incl_robot_node'])
    batch = list()
    nodes = list()
    out_timesteps = list()
    for timestep in nodes_per_ts.keys():
            scene_graph = scene.get_scene_graph(timestep,
                                                env.attention_radius,
                                                hyperparams['edge_addition_filter'],
                                                hyperparams['edge_removal_filter'])
            present_nodes = nodes_per_ts[timestep]
            for node in present_nodes:
                nodes.append(node)
                out_timesteps.append(timestep)
                batch.append(get_node_timestep_data(env, scene, timestep, node, state, pred_state,
                                                    edge_types, max_ht, max_ft, hyperparams,
                                                    scene_graph=scene_graph))
    if len(out_timesteps) == 0:
        return None
    return collate(batch), nodes, out_timesteps
