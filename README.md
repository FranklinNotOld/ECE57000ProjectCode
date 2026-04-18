# Trace2Map: Crowdsourced Drivable Area Reconstruction via Trajectory-Conditioned CVAE for Work Zone Safety

Course project submission for CE 57000 (Purdue University).
Accompanies the manuscript *"Trace2Map: Crowdsourced Drivable Area Reconstruction
via Trajectory-Conditioned Generative Model for Work Zone Safety"* (currently under review).

---

## !! IMPORTANT: Code Release Status

> **This course-project submission is derived from an unpublished research
> project.** Because the associated manuscript is currently under review,
> **some core components are intentionally withheld** from this release.

### Withheld Components

| Component | Status | Reason |
|---|---|---|
| Trace2Map CVAE architecture (full impl.) | Stubbed (signatures only) | Core novelty under review |
| Training loop and novel loss terms | Stubbed | Core novelty under review |
| Domain adaptation pipeline | Stubbed | Core novelty under review |
| Pretrained checkpoints (`.pth`) | Not released | Training code withheld |
| WZ-DAM dataset | To be released with paper | Dataset contribution |

### Fully Released Components

| Component | Status |
|---|---|
| SUMO work zone simulation pipeline (`sumo/`) | Full |
| Dataset collection and export (`dataset_collector/`) | Full |
| Data preprocessing and rasterization (`trace2map/run_preprocess.py`, `trace2map/utils/`) | Full |
| Trajectron++ downstream integration (forked, see `trajectron_plus_plus/NOTICE`) | Full |
| Integration layer: preprocessing, training orchestration, evaluation (`integration/`) | Full |
| Evaluation metrics (IoU, F1, SSIM, ADE, FDE, OOB rate, cone collision rate) | Full |
| Pre-computed reconstructed heatmaps (`cached_outputs/`) | Full |
| Cone encoding modes (raster / PointNet / scene-graph) | Full |

---

## Installation

```bash
conda create -n trace2map python=3.9
conda activate trace2map
pip install -r requirements.txt
```

Key dependencies: PyTorch, NumPy, pandas, OpenCV, dill, scipy, matplotlib, tensorboardX.

SUMO (Simulation of Urban Mobility) is required for running traffic simulations.

---

## Repository Layout

```
trace2map-submission/
├── README.md
├── LICENSE
├── requirements.txt
├── run_export.py                       # SUMO data export launcher
├── sumo/                               # SUMO work zone simulation
│   ├── main.py                         # Simulation entry point
│   ├── config.py                       # WorkzoneConfig dataclass
│   ├── network_builder.py              # .net.xml generation
│   ├── geometry.py                     # Work zone geometry helpers
│   └── ...
├── dataset_collector/                  # SUMO -> training data pipeline
│   ├── exporter.py                     # Master export pipeline
│   ├── trajectory_processor.py         # FCD CSV -> trajectories
│   ├── map_rasterizer.py              # HD map rasterization
│   └── ...
├── trace2map/                          # Core CVAE module (partially withheld)
│   ├── models/                         # CVAE variants (stubbed)
│   ├── utils/                          # Data utils (full) + losses (stubbed)
│   ├── inference.py                    # Checkpoint-free cached inference
│   ├── train.py                        # Training (stubbed)
│   └── run_preprocess.py              # Multi-scenario preprocessing (full)
├── trajectron_plus_plus/               # Trajectory prediction framework
│   ├── NOTICE                          # Fork attribution
│   └── trajectron/                     # Core library
├── integration/                        # Bridges CVAE + Trajectron++
│   ├── config/                         # Mode-specific configs (A/B-A/B-B/B-C/C)
│   ├── preprocessing/                  # Environment building, heatmap generation
│   ├── model_extensions/               # Cone encoders (PointNet, scene-graph)
│   ├── training/                       # Training orchestration
│   └── evaluation/                     # Metrics + visualization
├── cached_outputs/                     # Pre-computed heatmaps for reproduction
└── scripts/                            # Utility scripts
```

---

## Prediction Modes

| Mode | Description | Map Input |
|------|-------------|-----------|
| A    | Baseline (HD map only) | 3-channel binary raster |
| B-A  | Cones as raster channel | 4-channel raster (HD map + cone channel) |
| B-B  | Cones via PointNet encoder | HD map + K-nearest cone embedding |
| B-C  | Cones as scene-graph nodes | HD map + CONE node type in graph |
| C    | CVAE heatmap | HD map + Trace2Map reconstructed heatmap |

---

## Reproducing Results

All results can be reproduced using cached heatmap outputs:

```python
from trace2map.inference import Trace2MapInference

infer = Trace2MapInference("cached_outputs/sumo")
heatmap = infer.reconstruct("4L_close_1")
```

---

## Dataset

The WZ-DAM dataset will be released together with the manuscript upon acceptance.
For this submission:
- SUMO simulation configs are provided in `sumo/scenarios.json` (14+ scenarios).
- Cached Trace2Map outputs in `cached_outputs/` are sufficient for reproducing
  all reported numbers.

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

The Trajectron++ component is a modified fork of the
[original repository](https://github.com/StanfordASL/Trajectron-plus-plus)
by Stanford ASL, also under the MIT License.

---

## LLM Usage Acknowledgement

Portions of this README, docstrings, and code comments were drafted with
assistance from large language models (Claude, Anthropic) and subsequently
reviewed and edited by the authors. No LLM-generated code was incorporated
without human verification.
