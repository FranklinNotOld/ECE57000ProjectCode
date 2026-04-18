# ============================================================================
#  CODE RELEASE STATUS
#  This file defines the public interface of the MutEx CVAE variant.
#  The full implementation is WITHHELD pending publication.
#  See README for details.
# ============================================================================
"""
Mutual-Exclusive (MutEx) CVAE variant.

Extends the base CVAE with separate decoder heads for drivable-area
and work-zone region reconstruction with mutual exclusivity constraints.
"""
import torch
import torch.nn as nn


class MutExCVAE(nn.Module):
    """MutEx CVAE with separate drivable-area and work-zone decoders.

    Inputs
    ------
    obs_heatmap : Tensor (B, 1, 512, 128)
    num_lanes : Tensor (B, 1)

    Outputs
    -------
    recon_drivable : Tensor (B, 1, 512, 128)
    recon_workzone : Tensor (B, 1, 512, 128)
    mu, logvar : Tensor (B, D_z) each
    """

    def __init__(self, latent_dim=128, num_lanes_dim=1):
        super().__init__()
        self.latent_dim = latent_dim
        raise NotImplementedError(
            "MutExCVAE implementation is withheld pending publication. "
            "See README for details."
        )

    def forward(self, obs_heatmap, num_lanes):
        """See class docstring for I/O specification."""
        raise NotImplementedError

    def encode(self, obs_heatmap, num_lanes):
        """Encode to latent posterior."""
        raise NotImplementedError

    def decode(self, z, num_lanes):
        """Decode to dual-head outputs."""
        raise NotImplementedError

    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        raise NotImplementedError
