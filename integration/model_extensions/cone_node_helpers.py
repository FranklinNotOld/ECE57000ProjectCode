"""
Utility functions for Mode B-C: CONE as a static node type in the scene graph.

Mode B-C requires NO patches to Trajectron++ code because:
  1. CONE is defined as a node type in the Environment (node_type_list=['VEHICLE', 'CONE'])
  2. CONE is in 'state' and 'dynamic' config but NOT in 'pred_state'
     -> NodeTypeDataset skips CONE (no training samples generated)
     -> Trajectron only creates models for VEHICLE
  3. Scene graph automatically handles VEHICLE->CONE edges via attention_radius
  4. Edge encoder handles variable neighbor_state_length per edge type

The only integration point is in build_environment.py which creates CONE Node
objects for each traffic cone with constant position across all timesteps.

This module provides optional helpers for working with CONE nodes.
"""
import numpy as np
import pandas as pd


def count_cone_neighbors(scene, node, timestep, attention_radius):
    """Count the number of CONE nodes within attention radius of a given node at a timestep."""
    if not hasattr(scene, 'cones') or scene.cones is None:
        return 0
    node_pos = node.get(np.array([timestep, timestep]), {'position': ['x', 'y']})
    if node_pos is None:
        return 0
    pos = node_pos[-1, :2]
    dists = np.linalg.norm(scene.cones - pos, axis=1)
    cone_type = None
    for nt in scene.env.NodeType:
        if str(nt) == 'CONE':
            cone_type = nt
            break
    if cone_type is None:
        return 0
    radius = attention_radius.get((node.type, cone_type), 15.0)
    return int(np.sum(dists <= radius))
