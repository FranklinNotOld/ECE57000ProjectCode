"""
PointNet-style encoder for traffic cone positions (Mode B-B).

Encodes the K nearest cone positions (relative to agent) into a fixed-dim
vector that is concatenated with the Trajectron++ CVAE input embedding.
"""
import torch
import torch.nn as nn


class ConePointNetEncoder(nn.Module):
    """
    Simple PointNet encoder for K nearest cone positions.

    Input:  cone_rel [B, K, 2]  -- relative (dx, dy) of K nearest cones
            mask     [B, K]     -- True where cone is valid, False for padding
    Output: [B, output_dim]
    """
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, cone_rel, mask):
        """
        Parameters
        ----------
        cone_rel : [B, K, 2]  relative cone positions
        mask     : [B, K]     True = valid cone, False = padding

        Returns
        -------
        [B, output_dim]
        """
        h = self.mlp(cone_rel)  # [B, K, hidden_dim]
        # Zero out invalid cones, then max-pool over valid ones only.
        # Avoids -inf which causes NaN gradients when all cones are masked.
        mask_exp = mask.unsqueeze(-1)          # [B, K, 1]
        h = h * mask_exp.float()               # zero out padding
        # For rows with at least one valid cone, max-pool picks the real max;
        # for all-masked rows, every entry is 0 so max returns 0 (safe).
        h, _ = h.max(dim=1)                    # [B, hidden_dim]
        return self.fc_out(h)                   # [B, output_dim]
