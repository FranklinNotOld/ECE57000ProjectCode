# ============================================================================
#  CODE RELEASE STATUS
#  This file defines the domain adaptation pipeline for Trace2Map.
#  The full implementation is WITHHELD pending publication.
# ============================================================================
"""Domain adaptation / fine-tuning pipeline for Trace2Map (withheld).

This script is intentionally stubbed. The domain adaptation pipeline,
including fine-tuning strategies (Z_ONLY, FINETUNE_DECODER) and
loss configurations, is withheld pending publication.

Supported model types:
  - ORIGINAL, MUTEX, ATTENTION, CONTRASTIVE

Adaptation modes:
  - Z_ONLY:           Optimize only the latent code z
  - FINETUNE_DECODER: Fine-tune decoder weights on the target domain
"""
import sys


def main():
    sys.exit(
        "Trace2Map domain adaptation is not included in this release. "
        "See README for details."
    )


if __name__ == "__main__":
    main()
