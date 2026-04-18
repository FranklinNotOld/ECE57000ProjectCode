# ============================================================================
#  CODE RELEASE STATUS
#  Checkpoint-free inference wrapper backed by cached outputs.
#  This is fully functional and can reproduce all downstream results.
# ============================================================================
"""Checkpoint-free inference wrapper for Trace2Map.

Since the Trace2Map training code and pretrained weights are withheld
pending publication, this class loads pre-computed reconstructed heatmaps
from ``cached_outputs/`` instead of running the model forward pass.

This is sufficient to reproduce all downstream trajectory-prediction
results reported in the course project report.
"""
import os
import json
import numpy as np
from pathlib import Path


class Trace2MapInference:
    """Checkpoint-free inference backed by cached heatmap outputs.

    Parameters
    ----------
    cache_dir : str
        Path to the cached outputs directory (e.g., ``cached_outputs/sumo``).

    Examples
    --------
    >>> infer = Trace2MapInference("cached_outputs/sumo")
    >>> heatmap = infer.reconstruct("4L_close_1")
    >>> print(heatmap.shape)   # (N_scenes, 512, 128)
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self._index = self._load_index()

    def _load_index(self) -> dict:
        """Build an index mapping scene_id -> cached heatmap file path."""
        index = {}
        metadata_path = self.cache_dir / "metadata.json"

        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                meta = json.load(f)
            for entry in meta.get("scenes", []):
                scene_id = entry["scene_id"]
                heatmap_file = self.cache_dir / entry["heatmap_file"]
                if heatmap_file.exists():
                    index[scene_id] = str(heatmap_file)
        else:
            for npy_file in sorted(self.cache_dir.rglob("*.npy")):
                scene_id = npy_file.parent.name
                index[scene_id] = str(npy_file)

            for npz_file in sorted(self.cache_dir.rglob("*.npz")):
                scene_id = npz_file.parent.name
                index[scene_id] = str(npz_file)

        return index

    @property
    def available_scenes(self) -> list:
        """List all scene IDs with cached heatmaps."""
        return sorted(self._index.keys())

    def reconstruct(self, scene_id: str) -> np.ndarray:
        """Return the cached reconstructed heatmap for a given scene.

        Parameters
        ----------
        scene_id : str
            Scene identifier (e.g., ``"4L_close_1"``).

        Returns
        -------
        heatmap : np.ndarray of shape (H, W) or (N, H, W)
            Reconstructed drivable-area heatmap in [0, 1].
        """
        if scene_id not in self._index:
            available = ", ".join(self.available_scenes) or "(none)"
            raise KeyError(
                f"Scene '{scene_id}' not found in cache. "
                f"Available scenes: {available}"
            )

        path = self._index[scene_id]
        if path.endswith(".npz"):
            data = np.load(path)
            return data[data.files[0]]
        else:
            return np.load(path)

    def reconstruct_all(self) -> dict:
        """Load all cached heatmaps.

        Returns
        -------
        dict mapping scene_id -> np.ndarray
        """
        return {sid: self.reconstruct(sid) for sid in self.available_scenes}
