#!/usr/bin/env python3
"""
Convenience wrapper to launch the Dataset-Collector export pipeline.

Usage:
    python run_export.py --from-output sumo/_output
    python run_export.py --config sumo/scenarios.json --scenario 4L_close_2_3
    python run_export.py --config sumo/scenarios.json --scenario all
    python run_export.py --config sumo/scenarios.json --scenario 4L_close_2_3 --run-sim --no-gui
"""

import importlib.util
import sys
from pathlib import Path

# Add Dataset-Collector/ and Sumo/ to sys.path
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "dataset_collector"))
sys.path.insert(0, str(_ROOT / "sumo"))

if __name__ == "__main__":
    # Load the CLI module from Dataset-Collector/__main__.py
    spec = importlib.util.spec_from_file_location(
        "dc_main", str(_ROOT / "dataset_collector" / "__main__.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()
