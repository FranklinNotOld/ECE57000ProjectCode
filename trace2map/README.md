# Trace2Map Core Module

## Code Release Status

This directory contains the **core Trace2Map CVAE model** for drivable-area
reconstruction. Because the associated manuscript is currently under review,
the following are intentionally withheld:

- **Model architectures** (`models/`): Only class signatures and I/O specs
  are provided. Full implementations will be released upon publication.
- **Training pipeline** (`train.py`): Stubbed.
- **Domain adaptation** (`adapt.py`): Stubbed.
- **Custom loss functions** (`utils/losses.py`): Stubbed.

## What IS available

- **Data preprocessing** (`run_preprocess.py`, `utils/data_utils.py`,
  `utils/sumo_data_adapter.py`): Fully functional.
- **Inference wrapper** (`inference.py`): Loads pre-computed cached
  heatmaps for downstream evaluation without requiring model weights.

## CVAE Variants

| Variant | File | Description |
|---------|------|-------------|
| ORIGINAL | `models/cvae.py` | Base CVAE architecture |
| MUTEX | `models/cvae_mutex.py` | Dual-head mutual-exclusive decoder |
| ATTENTION | `models/cvae_attention.py` | Attention-augmented decoder |
| CONTRASTIVE | `models/cvae_contrastive.py` | SimCLR contrastive objective |
