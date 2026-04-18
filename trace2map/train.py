# ============================================================================
#  CODE RELEASE STATUS
#  This file defines the training entry point for Trace2Map.
#  The full training pipeline, including data augmentation, loss schedule,
#  and optimizer configuration, is WITHHELD pending publication of the
#  associated manuscript.
#
#  To reproduce the downstream trajectory-prediction results reported in
#  the paper, use the pre-computed reconstructed heatmaps under
#  `cached_outputs/` together with `scripts/run_prediction.sh`.
# ============================================================================
"""Training entry point for Trace2Map CVAE (withheld).

This script is intentionally stubbed. The full training pipeline,
including data augmentation, loss schedule, and optimizer configuration,
is withheld pending publication.

To reproduce the downstream trajectory-prediction results reported in
the course project report, use the pre-computed reconstructed heatmaps
under ``cached_outputs/`` together with the evaluation scripts.

Supported CVAE variants (selectable via TRAINING_MODE):
  - ORIGINAL:    Base CVAE architecture
  - MUTEX:       Mutual-exclusive dual-head decoder
  - ATTENTION:   Attention-augmented decoder
  - CONTRASTIVE: SimCLR contrastive learning objective

Supported data sources (selectable via DATA_SOURCE):
  - sumo:  SUMO simulation data
  - human: CARLA manual driving data
"""
import sys


def main():
    sys.exit(
        "Trace2Map training is not included in this release. "
        "See README for details."
    )


if __name__ == "__main__":
    main()
