"""
route_generator.py — Generate SUMO .rou.xml with vehicle types and traffic flows.
Single route to e_downstream; vehicles leave at end of 4-lane exit section.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

from config import WorkzoneConfig


def _vph_per_exit(vph_total: int, weights: list) -> list:
    """Split vph_total across exits by weights; return list of ints that sum to vph_total."""
    n = len(weights)
    if n == 0 or vph_total == 0:
        return [0] * n
    base = [int(vph_total * w) for w in weights]
    remainder = vph_total - sum(base)
    # Assign remainder one-by-one to exits (deterministic)
    i = 0
    while remainder > 0 and i < n:
        base[i] += 1
        remainder -= 1
        i += 1
    while remainder < 0 and i < n:
        if base[i] > 0:
            base[i] -= 1
            remainder += 1
        i += 1
    return base


def _indent(elem: ET.Element, level: int = 0) -> None:
    indent_str = "\n" + "    " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent_str + "    "
        for i, child in enumerate(elem):
            _indent(child, level + 1)
            if i < len(elem) - 1:
                child.tail = indent_str + "    "
            else:
                child.tail = indent_str
    if not elem.tail or not elem.tail.strip():
        elem.tail = indent_str


def generate_routes(cfg: WorkzoneConfig) -> str:
    """Generate .rou.xml.  Returns the file path."""
    os.makedirs(cfg.output_dir, exist_ok=True)
    path = cfg.route_file

    root = ET.Element("routes")

    # ---- vTypeDistribution: mixed car/truck per car_fraction ----
    car_prob = min(1.0, max(0.0, cfg.car_fraction))
    truck_prob = 1.0 - car_prob

    vtype_dist = ET.SubElement(root, "vTypeDistribution", id="mixed_types")
    car_vt = ET.SubElement(vtype_dist, "vType",
                           id="car",
                           probability=f"{car_prob:.4f}",
                           vClass="passenger",
                           accel=f"{cfg.car_accel:.2f}",
                           decel=f"{cfg.car_decel:.2f}",
                           sigma=f"{cfg.car_sigma:.2f}",
                           length=f"{cfg.car_length:.1f}",
                           minGap=f"{cfg.car_min_gap:.1f}",
                           maxSpeed=f"{cfg.car_max_speed:.2f}",
                           color="1,0.8,0",
                           lcStrategic=f"{cfg.lc_strategic:.1f}",
                           lcCooperative=f"{cfg.lc_cooperative:.1f}",
                           lcSpeedGain=f"{cfg.lc_speed_gain:.1f}",
                           lcKeepRight=f"{cfg.lc_keep_right:.1f}")
    truck_vt = ET.SubElement(vtype_dist, "vType",
                             id="truck",
                             probability=f"{truck_prob:.4f}",
                             vClass="truck",
                             accel=f"{cfg.truck_accel:.2f}",
                             decel=f"{cfg.truck_decel:.2f}",
                             sigma=f"{cfg.truck_sigma:.2f}",
                             length=f"{cfg.truck_length:.1f}",
                             minGap=f"{cfg.truck_min_gap:.1f}",
                             maxSpeed=f"{cfg.truck_max_speed:.2f}",
                             guiShape="truck",
                             color="0.5,0.3,0.1",
                             lcStrategic=f"{cfg.lc_strategic:.1f}",
                             lcCooperative=f"{cfg.lc_cooperative:.1f}",
                             lcSpeedGain=f"{cfg.lc_speed_gain:.1f}",
                             lcKeepRight=f"{cfg.lc_keep_right:.1f}")

    # ---- Single route to e_downstream (vehicles leave at end of 4-lane exit) ----
    N = cfg.num_lanes
    extend_f1 = cfg.extend_front_taper_first_segment and cfg.front_taper_first_segment_extension > 0
    f1_edges = "e_pre e_taper_f1_ext e_taper_f1 " if extend_f1 else "e_pre e_taper_f1 "
    route_edges = f1_edges + "e_taper_f2 e_taper_f3 e_work e_taper_r e_post e_downstream"
    ET.SubElement(root, "route",
                  id="main_route",
                  edges=route_edges.strip())

    # ---- Traffic flows: per-lane mixed flow ----
    vph_per_lane = cfg.veh_per_hour / N
    use_poisson = getattr(cfg, "depart_model", "poisson").lower() == "poisson"

    for lane in range(N):
        if vph_per_lane <= 0:
            continue
        flow_attrs = {
            "id": f"mixed_flow_L{lane}",
            "type": "mixed_types",
            "route": "main_route",
            "begin": "0",
            "end": f"{cfg.sim_duration:.0f}",
            "departLane": str(lane),
            "departSpeed": "desired",
            "departPos": "base",
            "arrivalLane": "random",
        }
        if use_poisson:
            insertions_per_sec = vph_per_lane / 3600.0
            flow_attrs["period"] = f"exp({insertions_per_sec:.6f})"
        else:
            flow_attrs["vehsPerHour"] = str(int(round(vph_per_lane)))
        ET.SubElement(root, "flow", **flow_attrs)

    _indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    print(f"[Routes]  Written → {path}")
    return path
