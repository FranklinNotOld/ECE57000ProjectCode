"""
config.py — Hyperparameter dataclass for SUMO workzone simulation.

All configurable parameters for road geometry, workzone layout,
traffic generation, and simulation control.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional


@dataclass
class WorkzoneConfig:
    """All hyperparameters for a single SUMO workzone scenario."""

    # --- Scenario identification ---
    scenario_name: str = "4L_close_2_3_4"      # "default"

    # --- (1) Road parameters ---
    total_road_length: float = 1000.0       # metres
    num_lanes: int = 4                      # single-direction lanes
    lane_width: float = 3.75               # metres per lane
    speed_limit: float = 33.33             # m/s  (~120 km/h)
    closed_lanes: str = "2,3,4"                # comma-separated, 1-indexed, 1 = leftmost
    workzone_start: float = 300.0          # metres from road start to front taper begin

    # --- (2) Workzone parameters ---
    workzone_length: float = 400.0         # total: front taper + hold + rear taper
    taper_front_ratio: float = 0.625        # fraction of workzone_length
    taper_hold_ratio: float = 0.3125
    taper_rear_ratio: float = 0.0625
    workzone_speed_limit: float = 13.89    # m/s  (~50 km/h)
    enable_front_taper_speed_reduction: bool = False
    extend_front_taper_first_segment: bool = True
    front_taper_first_segment_extension: float = 150.0
    enable_front_taper_f1_lane_change_restriction: bool = False
    shape_resolution: float = 2.0          # metres between shape points in taper

    # --- Visualization colors (SUMO RGBA: r,g,b,a in 0-1 range) ---
    workzone_color: str = "1,0.55,0,1"         # opaque overlay polygon for the tapered workzone area (no transparency)
    closed_lane_color: str = "0,0,0,1"         # base color for all lanes (black, alpha=1 fully opaque)
    draw_closed_lane_polygon: bool = True       # True = draw black polygon over closed lanes to cover SUMO's gray permission-based rendering; False = rely on gui_settings (may not work for disallow lanes)

    # --- (3) Traffic parameters ---
    veh_per_hour: int = 750              # total flow rate
    car_fraction: float = 0.85             # rest are trucks
    sim_duration: float = 3600.0           # seconds
    depart_model: str = "poisson"          # "poisson" = random (period=exp), "uniform" = evenly spaced (vehsPerHour)
    reassign_arrival_lane_in_taper: bool = True

    # Car vehicle type
    car_max_speed: float = 33.33
    car_accel: float = 2.6
    car_decel: float = 4.5
    car_sigma: float = 0.5                # driver imperfection [0, 1]
    car_length: float = 5.0
    car_min_gap: float = 2.5

    # Truck vehicle type
    truck_max_speed: float = 25.0
    truck_accel: float = 1.3
    truck_decel: float = 4.0
    truck_sigma: float = 0.5
    truck_length: float = 12.0
    truck_min_gap: float = 3.0

    # Lane-change behaviour
    lc_strategic: float = 6.0             # higher → earlier mandatory merge
    lc_cooperative: float = 1.0
    lc_speed_gain: float = 0.5
    lc_duration: float = 3.0              # seconds for one lane-change manoeuvre
    lc_keep_right: float = 1.0             # 0 = disable keep-right preference (helps vehicles change back to 1,2 lanes after workzone)
    lateral_resolution: Optional[float] = None

    # --- (4) Simulation settings ---
    step_length: float = 0.1              # simulation timestep (s)
    seed: int = 42
    gui: bool = True                      # sumo-gui vs headless sumo
    output_dir: str = "_output"
    time_scale: float = 1
    time_to_teleport: int = 1
    time_to_teleport_remove: bool = True
    time_to_teleport_highways: Optional[int] = None

    # --- (5) Downstream exit diversion---
    num_exits: Optional[int] = None      # default None = use num_lanes
    exit_weights: Optional[str] = "0.25,0.25,0.25,0.25"   # comma-separated, e.g. "0.1,0.2,0.3,0.4"; None = uniform

    # ------------------------------------------------------------------ #
    #  Derived / computed properties
    # ------------------------------------------------------------------ #

    @property
    def exit_count(self) -> int:
        """Number of downstream exit edges (defaults to num_lanes)."""
        if self.num_exits is not None:
            return self.num_exits
        return self.num_lanes

    @property
    def exit_weights_list(self) -> List[float]:
        """Normalised exit weights, length exit_count. Uniform if exit_weights is None."""
        N = self.exit_count
        if self.exit_weights is None or not self.exit_weights.strip():
            return [1.0 / N] * N
        parts = [x.strip() for x in self.exit_weights.split(",") if x.strip()]
        if len(parts) != N:
            raise ValueError(
                f"exit_weights has {len(parts)} values but exit_count is {N}. "
                f"Provide exactly {N} comma-separated weights."
            )
        weights = [float(w) for w in parts]
        if any(w < 0 for w in weights):
            raise ValueError("exit_weights must be non-negative")
        total = sum(weights)
        if total <= 0:
            raise ValueError("exit_weights must sum to a positive value")
        return [w / total for w in weights]

    @property
    def closed_lane_indices(self) -> List[int]:
        """0-indexed sorted list of lanes to close (our convention: 0 = leftmost)."""
        return sorted(int(x) - 1 for x in self.closed_lanes.split(",") if x.strip())

    @property
    def open_lane_indices(self) -> List[int]:
        """0-indexed sorted list of lanes that remain open."""
        closed = set(self.closed_lane_indices)
        return [i for i in range(self.num_lanes) if i not in closed]

    @property
    def num_open_lanes(self) -> int:
        return len(self.open_lane_indices)

    @property
    def front_taper_length(self) -> float:
        return self.workzone_length * self.taper_front_ratio

    @property
    def hold_length(self) -> float:
        return self.workzone_length * self.taper_hold_ratio

    @property
    def rear_taper_length(self) -> float:
        return self.workzone_length * self.taper_rear_ratio

    @property
    def total_road_width(self) -> float:
        return self.num_lanes * self.lane_width

    @property
    def workzone_end(self) -> float:
        return self.workzone_start + self.workzone_length

    # ------------------------------------------------------------------ #
    #  Output file helpers
    # ------------------------------------------------------------------ #

    def out_path(self, filename: str) -> str:
        return str(Path(self.output_dir) / filename)

    @property
    def net_file(self) -> str:
        return self.out_path("workzone.net.xml")

    @property
    def route_file(self) -> str:
        return self.out_path("workzone.rou.xml")

    @property
    def additional_file(self) -> str:
        return self.out_path("workzone.add.xml")

    @property
    def sumocfg_file(self) -> str:
        return self.out_path("workzone.sumocfg")

    @property
    def fcd_file(self) -> str:
        return self.out_path("fcd_output.xml")

    @property
    def csv_file(self) -> str:
        return self.out_path("trajectories.csv")

    @property
    def gui_settings_file(self) -> str:
        """Path to sumo-gui view settings file (hide exit edges)."""
        return self.out_path("gui_settings.xml")

    # ------------------------------------------------------------------ #
    #  Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        """Raise ValueError on obviously wrong settings."""
        if self.num_lanes < 1:
            raise ValueError("num_lanes must be >= 1")
        if self.num_open_lanes < 1:
            raise ValueError(
                f"At least one lane must remain open. "
                f"closed_lanes='{self.closed_lanes}' closes all {self.num_lanes} lanes."
            )
        bad = [i for i in self.closed_lane_indices if i < 0 or i >= self.num_lanes]
        if bad:
            raise ValueError(
                f"Invalid lane numbers in closed_lanes='{self.closed_lanes}'. "
                f"Valid range: 1..{self.num_lanes}."
            )
        if not math.isclose(self.taper_front_ratio + self.taper_hold_ratio + self.taper_rear_ratio, 1.0, abs_tol=1e-6):
            raise ValueError("taper_front_ratio + taper_hold_ratio + taper_rear_ratio must equal 1.0")
        wz_end = self.workzone_start + self.workzone_length
        if wz_end > self.total_road_length:
            raise ValueError(
                f"Workzone extends beyond road: workzone_start({self.workzone_start}) + "
                f"workzone_length({self.workzone_length}) = {wz_end} > total_road_length({self.total_road_length})"
            )
        capacity = 1800 * self.num_open_lanes
        if self.veh_per_hour > capacity:
            print(
                f"[WARNING] veh_per_hour={self.veh_per_hour} exceeds estimated bottleneck "
                f"capacity ~{capacity} veh/hr for {self.num_open_lanes} open lanes. "
                f"Expect heavy congestion / gridlock."
            )
        if self.num_exits is not None:
            if self.num_exits < 1:
                raise ValueError("num_exits must be >= 1")
            if self.num_exits != self.num_lanes:
                raise ValueError(
                    f"num_exits ({self.num_exits}) must equal num_lanes ({self.num_lanes}) "
                    f"for 1:1 lane-to-exit topology."
                )
        # Trigger exit_weights_list to validate
        _ = self.exit_weights_list
        if self.extend_front_taper_first_segment:
            if self.workzone_start - self.front_taper_first_segment_extension < 0:
                raise ValueError(
                    f"workzone_start({self.workzone_start}) - "
                    f"front_taper_first_segment_extension({self.front_taper_first_segment_extension}) "
                    f"must be >= 0"
                )

    # ------------------------------------------------------------------ #
    #  Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorkzoneConfig":
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


def load_scenarios(path: str) -> List[WorkzoneConfig]:
    """Load a scenario-list JSON (global_settings + scenario_list) → list of WorkzoneConfig."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    global_settings = data.get("global_settings", {})
    configs: List[WorkzoneConfig] = []

    for sc in data.get("scenario_list", []):
        merged = {**global_settings, **sc}
        # Map legacy key 'name' → 'scenario_name'
        if "name" in merged and "scenario_name" not in merged:
            merged["scenario_name"] = merged.pop("name")
        configs.append(WorkzoneConfig.from_dict(merged))

    return configs
