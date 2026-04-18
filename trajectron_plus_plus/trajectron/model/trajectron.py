import torch
import numpy as np
from model.mgcvae import MultimodalGenerativeCVAE
from model.dataset import get_timesteps_data, restore


class Trajectron(object):
    def __init__(self, model_registrar,
                 hyperparams, log_writer,
                 device):
        super(Trajectron, self).__init__()
        self.hyperparams = hyperparams
        self.log_writer = log_writer
        self.device = device
        self.curr_iter = 0

        self.model_registrar = model_registrar
        self.node_models_dict = dict()
        self.nodes = set()

        self.env = None

        self.min_ht = self.hyperparams['minimum_history_length']
        self.max_ht = self.hyperparams['maximum_history_length']
        self.ph = self.hyperparams['prediction_horizon']
        self.state = self.hyperparams['state']
        self.state_length = dict()
        for state_type in self.state.keys():
            self.state_length[state_type] = int(
                np.sum([len(entity_dims) for entity_dims in self.state[state_type].values()])
            )
        self.pred_state = self.hyperparams['pred_state']

    def set_environment(self, env):
        self.env = env

        self.node_models_dict.clear()
        edge_types = env.get_edge_types()

        for node_type in env.NodeType:
            # Only add a Model for NodeTypes we want to predict
            if node_type in self.pred_state.keys():
                self.node_models_dict[node_type] = MultimodalGenerativeCVAE(env,
                                                                            node_type,
                                                                            self.model_registrar,
                                                                            self.hyperparams,
                                                                            self.device,
                                                                            edge_types,
                                                                            log_writer=self.log_writer)

    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter
        for node_str, model in self.node_models_dict.items():
            model.set_curr_iter(curr_iter)

    def set_annealing_params(self):
        for node_str, model in self.node_models_dict.items():
            model.set_annealing_params()

    def step_annealers(self, node_type=None):
        if node_type is None:
            for node_type in self.node_models_dict:
                self.node_models_dict[node_type].step_annealers()
        else:
            self.node_models_dict[node_type].step_annealers()

    @staticmethod
    def _unpack_batch(batch):
        """Unpack batch tuple, handling both 9-element (vanilla) and 10-element (with cones) formats."""
        if len(batch) == 10:
            (first_history_index, x_t, y_t, x_st_t, y_st_t,
             neighbors_data_st, neighbors_edge_value,
             robot_traj_st_t, map_data, cone_data) = batch
        else:
            (first_history_index, x_t, y_t, x_st_t, y_st_t,
             neighbors_data_st, neighbors_edge_value,
             robot_traj_st_t, map_data) = batch
            cone_data = None
        return (first_history_index, x_t, y_t, x_st_t, y_st_t,
                neighbors_data_st, neighbors_edge_value,
                robot_traj_st_t, map_data, cone_data)

    def _move_cones_to_device(self, cone_data):
        """Move cone tuple to device if present."""
        if cone_data is None:
            return None
        if isinstance(cone_data, (tuple, list)) and len(cone_data) == 2:
            return (cone_data[0].to(self.device), cone_data[1].to(self.device))
        return None

    def train_loss(self, batch, node_type):
        (first_history_index, x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st, neighbors_edge_value,
         robot_traj_st_t, map, cone_data) = self._unpack_batch(batch)

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)
        cones = self._move_cones_to_device(cone_data)

        # Run forward pass
        model = self.node_models_dict[node_type]
        loss = model.train_loss(inputs=x,
                                inputs_st=x_st_t,
                                first_history_indices=first_history_index,
                                labels=y,
                                labels_st=y_st_t,
                                neighbors=restore(neighbors_data_st),
                                neighbors_edge_value=restore(neighbors_edge_value),
                                robot=robot_traj_st_t,
                                map=map,
                                prediction_horizon=self.ph,
                                cones=cones)

        return loss

    def eval_loss(self, batch, node_type):
        (first_history_index, x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st, neighbors_edge_value,
         robot_traj_st_t, map, cone_data) = self._unpack_batch(batch)

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)
        cones = self._move_cones_to_device(cone_data)

        # Run forward pass
        model = self.node_models_dict[node_type]
        nll = model.eval_loss(inputs=x,
                              inputs_st=x_st_t,
                              first_history_indices=first_history_index,
                              labels=y,
                              labels_st=y_st_t,
                              neighbors=restore(neighbors_data_st),
                              neighbors_edge_value=restore(neighbors_edge_value),
                              robot=robot_traj_st_t,
                              map=map,
                              prediction_horizon=self.ph,
                              cones=cones)

        return nll.cpu().detach().numpy()

    def predict_from_batch(self, batch, node_type, ph,
                           num_samples=1, z_mode=False, gmm_mode=False,
                           full_dist=True, all_z_sep=False):
        """Run prediction on a pre-collated batch from DataLoader.

        Unlike predict(), this does not call get_timesteps_data or build a
        predictions_dict.  The caller is responsible for metadata mapping.

        Returns:
            predictions_np: numpy array [num_samples, batch_size, ph, pred_dim]
        """
        (first_history_index, x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st, neighbors_edge_value,
         robot_traj_st_t, map_data, cone_data) = self._unpack_batch(batch)

        x = x_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map_data) == torch.Tensor:
            map_data = map_data.to(self.device)
        cones = self._move_cones_to_device(cone_data)

        model = self.node_models_dict[node_type]
        predictions = model.predict(inputs=x,
                                    inputs_st=x_st_t,
                                    first_history_indices=first_history_index,
                                    neighbors=restore(neighbors_data_st),
                                    neighbors_edge_value=restore(neighbors_edge_value),
                                    robot=robot_traj_st_t,
                                    map=map_data,
                                    prediction_horizon=ph,
                                    num_samples=num_samples,
                                    z_mode=z_mode,
                                    gmm_mode=gmm_mode,
                                    full_dist=full_dist,
                                    all_z_sep=all_z_sep,
                                    cones=cones)
        return predictions.cpu().detach().numpy()

    def predict(self,
                scene,
                timesteps,
                ph,
                num_samples=1,
                min_future_timesteps=0,
                min_history_timesteps=1,
                z_mode=False,
                gmm_mode=False,
                full_dist=True,
                all_z_sep=False):

        predictions_dict = {}
        for node_type in self.env.NodeType:
            if node_type not in self.pred_state:
                continue

            model = self.node_models_dict[node_type]

            # Get Input data for node type and given timesteps
            batch = get_timesteps_data(env=self.env, scene=scene, t=timesteps, node_type=node_type, state=self.state,
                                       pred_state=self.pred_state, edge_types=model.edge_types,
                                       min_ht=min_history_timesteps, max_ht=self.max_ht, min_ft=min_future_timesteps,
                                       max_ft=min_future_timesteps, hyperparams=self.hyperparams)
            # There are no nodes of type present for timestep
            if batch is None:
                continue
            batch_data, nodes, timesteps_o = batch
            (first_history_index, x_t, y_t, x_st_t, y_st_t,
             neighbors_data_st, neighbors_edge_value,
             robot_traj_st_t, map, cone_data) = self._unpack_batch(batch_data)

            x = x_t.to(self.device)
            x_st_t = x_st_t.to(self.device)
            if robot_traj_st_t is not None:
                robot_traj_st_t = robot_traj_st_t.to(self.device)
            if type(map) == torch.Tensor:
                map = map.to(self.device)
            cones = self._move_cones_to_device(cone_data)

            # Run forward pass
            predictions = model.predict(inputs=x,
                                        inputs_st=x_st_t,
                                        first_history_indices=first_history_index,
                                        neighbors=neighbors_data_st,
                                        neighbors_edge_value=neighbors_edge_value,
                                        robot=robot_traj_st_t,
                                        map=map,
                                        prediction_horizon=ph,
                                        num_samples=num_samples,
                                        z_mode=z_mode,
                                        gmm_mode=gmm_mode,
                                        full_dist=full_dist,
                                        all_z_sep=all_z_sep,
                                        cones=cones)

            predictions_np = predictions.cpu().detach().numpy()

            # Assign predictions to node
            for i, ts in enumerate(timesteps_o):
                if ts not in predictions_dict.keys():
                    predictions_dict[ts] = dict()
                predictions_dict[ts][nodes[i]] = np.transpose(predictions_np[:, [i]], (1, 0, 2, 3))

        return predictions_dict
