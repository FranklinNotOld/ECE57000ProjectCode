"""
main.py — CLI entry point for SUMO workzone simulation.

Usage examples:
    # Single scenario with defaults
    python main.py

    # Custom parameters
    python main.py --num-lanes 4 --closed-lanes "1,2" --veh-per-hour 2000 --gui

    # Run all scenarios from a JSON file
    python main.py --config scenarios.json

    # Run a specific scenario from a JSON file
    python main.py --config scenarios.json --scenario-index 2

    # Headless mode (no GUI)
    python main.py --no-gui
"""

from __future__ import annotations

import argparse
import os
import sys

from config import WorkzoneConfig, load_scenarios
from network_builder import build_network
from route_generator import generate_routes
from visual_builder import build_additional
from simulation_runner import run_simulation
from fcd_converter import convert_fcd


def run_single_scenario(cfg: WorkzoneConfig) -> None:
    """Execute the full pipeline for one scenario."""
    print(f"\n{'='*60}")
    print(f"  Scenario: {cfg.scenario_name}")
    print(f"  Lanes: {cfg.num_lanes}  |  Closed: {cfg.closed_lanes}  |  Open: {cfg.num_open_lanes}")
    print(f"  Road: {cfg.total_road_length}m  |  WZ start: {cfg.workzone_start}m  |  WZ length: {cfg.workzone_length}m")
    if cfg.extend_front_taper_first_segment:
        print(f"  Front taper: first segment extended upstream by {cfg.front_taper_first_segment_extension}m")
    print(f"  Traffic: {cfg.veh_per_hour} veh/hr  |  Duration: {cfg.sim_duration}s")
    print(f"{'='*60}\n")

    cfg.validate()

    # 1. Build network
    build_network(cfg)

    # 2. Generate routes
    generate_routes(cfg)

    # 3. Generate visual elements
    build_additional(cfg)

    # 4. Run simulation
    run_simulation(cfg)

    # 5. Convert FCD to CSV
    if os.path.isfile(cfg.fcd_file):
        convert_fcd(cfg.fcd_file, cfg.csv_file)
    else:
        print(f"[WARN]    FCD file not found: {cfg.fcd_file}")

    print(f"\n[DONE]    Outputs in: {cfg.output_dir}/\n")


def build_config_from_args(args: argparse.Namespace) -> WorkzoneConfig:
    """Construct a WorkzoneConfig from CLI arguments."""
    overrides = {}
    # Map CLI flags to config fields
    mapping = {
        "scenario_name": "scenario_name",
        "total_road_length": "total_road_length",
        "num_lanes": "num_lanes",
        "lane_width": "lane_width",
        "speed_limit": "speed_limit",
        "closed_lanes": "closed_lanes",
        "workzone_start": "workzone_start",
        "workzone_length": "workzone_length",
        "taper_front_ratio": "taper_front_ratio",
        "taper_hold_ratio": "taper_hold_ratio",
        "taper_rear_ratio": "taper_rear_ratio",
        "workzone_speed_limit": "workzone_speed_limit",
        "veh_per_hour": "veh_per_hour",
        "car_fraction": "car_fraction",
        "sim_duration": "sim_duration",
        "depart_model": "depart_model",
        "seed": "seed",
        "output_dir": "output_dir",
        "step_length": "step_length",
        "lc_strategic": "lc_strategic",
        "lc_duration": "lc_duration",
        "lateral_resolution": "lateral_resolution",
        "time_scale": "time_scale",
        "num_exits": "num_exits",
        "exit_weights": "exit_weights",
        "extend_front_taper_first": "extend_front_taper_first_segment",
        "front_taper_first_segment_extension": "front_taper_first_segment_extension",
        "enable_front_taper_f1_lane_change_restriction": "enable_front_taper_f1_lane_change_restriction",
        "workzone_color": "workzone_color",
        "closed_lane_color": "closed_lane_color",
        "reassign_arrival_lane_in_taper": "reassign_arrival_lane_in_taper",
    }
    for cli_key, cfg_key in mapping.items():
        val = getattr(args, cli_key, None)
        if val is not None:
            overrides[cfg_key] = val

    # Handle --gui / --no-gui
    if args.no_gui:
        overrides["gui"] = False
    elif args.gui:
        overrides["gui"] = True

    # Handle --no-front-taper-f1-lane-change-restriction
    if getattr(args, "no_front_taper_f1_lane_change_restriction", False):
        overrides["enable_front_taper_f1_lane_change_restriction"] = False

    # Handle --no-draw-closed-lane-polygon
    if getattr(args, "no_draw_closed_lane_polygon", False):
        overrides["draw_closed_lane_polygon"] = False

    return WorkzoneConfig(**overrides)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SUMO Workzone Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Config file mode
    parser.add_argument("--config", type=str, default=None,
                        help="Path to scenarios.json for batch mode")
    parser.add_argument("--scenario-index", type=int, default=None,
                        help="Run only this scenario index from the config file (0-based)")

    # Direct parameter mode
    parser.add_argument("--scenario-name", type=str, default=None)
    parser.add_argument("--total-road-length", type=float, default=None)
    parser.add_argument("--num-lanes", type=int, default=None)
    parser.add_argument("--lane-width", type=float, default=None)
    parser.add_argument("--speed-limit", type=float, default=None)
    parser.add_argument("--closed-lanes", type=str, default=None)
    parser.add_argument("--workzone-start", type=float, default=None)
    parser.add_argument("--workzone-length", type=float, default=None)
    parser.add_argument("--taper-front-ratio", type=float, default=None)
    parser.add_argument("--taper-hold-ratio", type=float, default=None)
    parser.add_argument("--taper-rear-ratio", type=float, default=None)
    parser.add_argument("--workzone-speed-limit", type=float, default=None)
    parser.add_argument("--veh-per-hour", type=int, default=None)
    parser.add_argument("--car-fraction", type=float, default=None)
    parser.add_argument("--sim-duration", type=float, default=None)
    parser.add_argument("--depart-model", type=str, default=None,
                        choices=["poisson", "uniform"],
                        help="Departure model: poisson (random) or uniform (evenly spaced)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--step-length", type=float, default=None)
    parser.add_argument("--lc-strategic", type=float, default=None)
    parser.add_argument("--lc-duration", type=float, default=None)
    parser.add_argument("--lateral-resolution", type=float, default=None,
                        help="Sublane lateral resolution (m), e.g. 0.75; omit to disable sublane model")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--time-scale", type=float, default=None,
                        help="Simulation time / real time ratio (default 1.0 = real-time)")
    parser.add_argument("--num-exits", type=int, default=None,
                        help="Number of downstream exit edges (default: same as num-lanes)")
    parser.add_argument("--exit-weights", type=str, default=None,
                        help='Comma-separated exit weights, e.g. "0.1,0.2,0.3,0.4" (default: uniform)')
    parser.add_argument("--extend-front-taper-first", action="store_true", default=None,
                        help="Extend first third of front taper upstream")
    parser.add_argument("--front-taper-first-extension", type=float, default=None,
                        help="Extension distance (m) for first front taper segment (default: 50)")
    parser.add_argument("--no-front-taper-f1-lane-change-restriction", action="store_true", default=False,
                        help="Disable lane change restriction on first front taper segment (allow lane changes)")
    parser.add_argument("--workzone-color", type=str, default=None,
                        help="Workzone polygon color (SUMO RGBA, e.g. '1,0.55,0,0.30')")
    parser.add_argument("--closed-lane-color", type=str, default=None,
                        help="Closed lane polygon color (SUMO RGBA, e.g. '0.5,0.5,0.5,1')")
    parser.add_argument("--no-draw-closed-lane-polygon", action="store_true", default=False,
                        help="Skip drawing closed lane polygon (rely on SUMO lane rendering)")
    parser.add_argument("--reassign-arrival-lane-in-taper", action="store_true", default=None,
                        help="Randomly reassign arrival lane when vehicle first enters workzone transition area")

    gui_group = parser.add_mutually_exclusive_group()
    gui_group.add_argument("--gui", action="store_true", default=False,
                           help="Launch sumo-gui (default)")
    gui_group.add_argument("--no-gui", action="store_true", default=False,
                           help="Run headless sumo")

    args = parser.parse_args()

    if args.config:
        # Batch mode — load scenarios from JSON
        scenarios = load_scenarios(args.config)
        if not scenarios:
            print(f"[ERROR] No scenarios found in {args.config}")
            sys.exit(1)

        if args.scenario_index is not None:
            if args.scenario_index < 0 or args.scenario_index >= len(scenarios):
                print(f"[ERROR] scenario-index {args.scenario_index} out of range [0, {len(scenarios)-1}]")
                sys.exit(1)
            scenarios = [scenarios[args.scenario_index]]

        print(f"[Main]    Loaded {len(scenarios)} scenario(s) from {args.config}")

        # Apply CLI overrides to all scenarios
        for sc in scenarios:
            if args.no_gui:
                sc.gui = False
            elif args.gui:
                sc.gui = True
            if args.lateral_resolution is not None:
                sc.lateral_resolution = args.lateral_resolution
            if args.extend_front_taper_first is not None:
                sc.extend_front_taper_first_segment = args.extend_front_taper_first
            if args.front_taper_first_extension is not None:
                sc.front_taper_first_segment_extension = args.front_taper_first_extension
            if getattr(args, "no_front_taper_f1_lane_change_restriction", False):
                sc.enable_front_taper_f1_lane_change_restriction = False
            if args.workzone_color is not None:
                sc.workzone_color = args.workzone_color
            if args.closed_lane_color is not None:
                sc.closed_lane_color = args.closed_lane_color
            if getattr(args, "no_draw_closed_lane_polygon", False):
                sc.draw_closed_lane_polygon = False
            if getattr(args, "reassign_arrival_lane_in_taper", False):
                sc.reassign_arrival_lane_in_taper = True

        for i, sc in enumerate(scenarios):
            print(f"\n>>> Running scenario {i+1}/{len(scenarios)}: {sc.scenario_name}")
            # Give each scenario its own output subdirectory
            sc.output_dir = os.path.join("_output", sc.scenario_name)
            run_single_scenario(sc)
    else:
        # Single-scenario mode from CLI arguments
        cfg = build_config_from_args(args)
        run_single_scenario(cfg)


if __name__ == "__main__":
    main()
