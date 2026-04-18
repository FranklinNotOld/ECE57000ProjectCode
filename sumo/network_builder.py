"""
network_builder.py — Programmatic SUMO road-network generation with workzone.

Generates .nod.xml, .edg.xml, .con.xml and invokes netconvert to produce .net.xml.

Network topology (8 edges when extend, else 7 edges):

    n0 ──e_pre── n1 ──[e_taper_f1_ext]── n1a ──e_taper_f1── n2a ──e_taper_f2── n2b ──e_taper_f3── n2
        ──e_work── n3 ──e_taper_r── n4 ──e_post── n5
    (e_taper_f1_ext only when extend_front_taper_first_segment; else n1→n2a is e_taper_f1)

    All edges have N lanes; road geometry stays straight. All lanes have 1-to-1
    connections at all junctions. Lane closure and earlier merging are enforced
    by progressively slowing only the CLOSED lanes across e_taper_f1/2/3 and then
    applying DISALLOW on the closed lanes in the last front-taper segment
    (e_taper_f3) and in e_work. cfg.closed_lanes is 1-indexed (1=leftmost);
    SUMO lane index 0=rightmost, so sumo_idx = N - 1 - orig_idx (orig_idx 0-indexed leftmost).
"""

from __future__ import annotations

import os
import subprocess
import xml.etree.ElementTree as ET
from typing import List, Tuple

from config import WorkzoneConfig
from geometry import compute_disallow_last_two_thirds_lanes


# ------------------------------------------------------------------ #
#  Internal helpers
# ------------------------------------------------------------------ #

def _indent(elem: ET.Element, level: int = 0) -> None:
    """Pretty-print indent for ElementTree (Python < 3.9 compat)."""
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


# ------------------------------------------------------------------ #
#  Node positions
# ------------------------------------------------------------------ #

# Length of each exit edge (metres); exits are downstream of n5
EXIT_EDGE_LENGTH = 20.0


def _node_positions(cfg: WorkzoneConfig) -> List[Tuple[str, float, float]]:
    """Return [(node_id, x, y), …] for main nodes n0..n6 (n6 = dead_end, vehicles leave at end of e_downstream).
    When extend_front_taper_first_segment: n1a at workzone_start separates extension from main taper segment."""
    ws = cfg.workzone_start
    fl = cfg.front_taper_length
    hl = cfg.hold_length
    ext = cfg.front_taper_first_segment_extension if cfg.extend_front_taper_first_segment else 0.0
    node_y = 0.0
    nodes = [
        ("n0", 0.0, node_y),
        ("n1", ws - ext, node_y),
        ("n2a", ws + fl * (1.0 / 3.0), node_y),
        ("n2b", ws + fl * (2.0 / 3.0), node_y),
        ("n2", ws + fl, node_y),
        ("n3", ws + fl + hl, node_y),
        ("n4", ws + cfg.workzone_length, node_y),
        ("n5", cfg.total_road_length, node_y),
        ("n6", cfg.total_road_length + EXIT_EDGE_LENGTH, node_y),
    ]
    if ext > 0:
        # Insert n1a at workzone_start (end of extension, start of main taper)
        idx = 2  # after n1
        nodes.insert(idx, ("n1a", ws, node_y))
    return nodes


# ------------------------------------------------------------------ #
#  XML writers
# ------------------------------------------------------------------ #

def _write_types(cfg: WorkzoneConfig, path: str) -> None:
    """Write minimal .typ.xml so netconvert accepts type='exitHidden'."""
    root = ET.Element("types")
    ET.SubElement(
        root, "type",
        id="exitHidden",
        numLanes="1",
        speed=f"{cfg.speed_limit:.2f}",
    )
    _indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _write_nodes(cfg: WorkzoneConfig, path: str) -> None:
    root = ET.Element("nodes")
    # n0 and n6 are dead_end (vehicles leave at end of e_downstream); rest are priority.
    for nid, x, y in _node_positions(cfg):
        if nid in ("n0", "n6"):
            ntype = "dead_end"
        else:
            ntype = "priority"
        ET.SubElement(root, "node", id=nid, x=f"{x:.4f}", y=f"{y:.4f}", type=ntype)
    _indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _write_edges(cfg: WorkzoneConfig, path: str) -> None:
    N = cfg.num_lanes
    closed_set = set(cfg.closed_lane_indices)  # 0-indexed leftmost
    # SUMO lane indices that are closed (0=rightmost in SUMO)
    closed_sumo: List[int] = [N - 1 - c for c in closed_set]
    # Lanes needing disallow on e_taper_f2 (middle-isolated blocks)
    last_two_thirds_orig = compute_disallow_last_two_thirds_lanes(cfg)
    closed_sumo_last_two_thirds: set[int] = {N - 1 - c for c in last_two_thirds_orig}

    # Lane change restrictions for closed_lanes="2,3" only (lane 2→left, lane 3→right)
    is_lane23_workzone = sorted(cfg.closed_lane_indices) == [1, 2]
    if is_lane23_workzone:
        sumo_idx_lane2 = N - 1 - 1
        sumo_idx_lane3 = N - 1 - 2

    root = ET.Element("edges")

    # --- e_pre  (N lanes, straight) ---
    ET.SubElement(root, "edge",
                  id="e_pre", **{"from": "n0"}, to="n1",
                  numLanes=str(N), speed=f"{cfg.speed_limit:.2f}",
                  spreadType="right")

    # Common taper speeds
    taper_speed = (cfg.speed_limit + cfg.workzone_speed_limit) / 2.0
    taper_speed_str = f"{taper_speed:.2f}"
    if cfg.enable_front_taper_speed_reduction:
        s1 = 0.85 * cfg.speed_limit
        s2 = 0.70 * cfg.speed_limit
        s3 = 0.55 * cfg.speed_limit
        s1_str = f"{s1:.2f}"
        s2_str = f"{s2:.2f}"
        s3_str = f"{s3:.2f}"
    else:
        s1_str = s2_str = s3_str = f"{cfg.speed_limit:.2f}"

    # --- e_taper_f1: extension (n1→n1a) + main (n1a→n2a) when extend; else single edge (n1→n2a) ---
    # Extension part: no lane-change restriction (vehicles may merge before taper).
    # Main part (from workzone_start): lane-change restriction for lane23 workzone.
    extend_f1 = cfg.extend_front_taper_first_segment and cfg.front_taper_first_segment_extension > 0
    f1_from, f1_to = ("n1a", "n2a") if extend_f1 else ("n1", "n2a")

    if extend_f1:
        e_taper_f1_ext = ET.SubElement(root, "edge",
                                      id="e_taper_f1_ext", **{"from": "n1"}, to="n1a",
                                      numLanes=str(N), speed=f"{cfg.speed_limit:.2f}",
                                      spreadType="right")
        for lane_idx in closed_sumo:
            lane_attrs: dict = {"index": str(lane_idx), "speed": s1_str}
            if lane_idx in closed_sumo_last_two_thirds:
                lane_attrs["disallow"] = "passenger truck"
            # No lane-change restriction in extension segment
            ET.SubElement(e_taper_f1_ext, "lane", **lane_attrs)

    e_taper_f1 = ET.SubElement(root, "edge",
                               id="e_taper_f1", **{"from": f1_from}, to=f1_to,
                               numLanes=str(N), speed=f"{cfg.speed_limit:.2f}",
                               spreadType="right")
    for lane_idx in closed_sumo:
        lane_attrs: dict = {"index": str(lane_idx), "speed": s1_str}
        if lane_idx in closed_sumo_last_two_thirds:
            lane_attrs["disallow"] = "passenger truck"
        if is_lane23_workzone:
            if lane_idx == sumo_idx_lane2:
                lane_attrs["changeRight"] = "pedestrian"  # prohibit passenger/truck
            elif lane_idx == sumo_idx_lane3:
                lane_attrs["changeLeft"] = "pedestrian"  # prohibit passenger/truck
        ET.SubElement(e_taper_f1, "lane", **lane_attrs)

    # --- e_taper_f2 (N lanes, straight; closed lanes slower; middle-isolated also disallow) ---
    e_taper_f2 = ET.SubElement(root, "edge",
                               id="e_taper_f2", **{"from": "n2a"}, to="n2b",
                               numLanes=str(N), speed=f"{cfg.speed_limit:.2f}",
                               spreadType="right")
    for lane_idx in closed_sumo:
        lane_attrs: dict = {"index": str(lane_idx), "speed": s2_str}
        if lane_idx in closed_sumo_last_two_thirds or (not is_lane23_workzone):
            lane_attrs["disallow"] = "passenger truck"
        if is_lane23_workzone:
            if lane_idx == sumo_idx_lane2:
                lane_attrs["changeRight"] = "pedestrian"  # prohibit passenger/truck
            elif lane_idx == sumo_idx_lane3:
                lane_attrs["changeLeft"] = "pedestrian"  # prohibit passenger/truck
        ET.SubElement(e_taper_f2, "lane", **lane_attrs)

    # --- e_taper_f3 (N lanes, straight; closed lanes slowest + disallow) ---
    e_taper_f3 = ET.SubElement(root, "edge",
                               id="e_taper_f3", **{"from": "n2b"}, to="n2",
                               numLanes=str(N), speed=f"{cfg.speed_limit:.2f}",
                               spreadType="right")
    for lane_idx in closed_sumo:
        lane_attrs: dict = {
            "index": str(lane_idx),
            "speed": s3_str,
        }
        lane_attrs["disallow"] = "passenger truck"
        if is_lane23_workzone:
            if lane_idx == sumo_idx_lane2:
                lane_attrs["changeRight"] = "pedestrian"  # prohibit passenger/truck
            elif lane_idx == sumo_idx_lane3:
                lane_attrs["changeLeft"] = "pedestrian"  # prohibit passenger/truck
        ET.SubElement(e_taper_f3, "lane", **lane_attrs)

    # --- e_work  (N lanes, straight; closed lanes have disallow) ---
    # Edge-level speed still applies to all lanes; to avoid globally slowing
    # open lanes inside the workzone, keep cfg.speed_limit at edge level and
    # rely on lane sub-elements only for closed-lane restrictions.
    e_work = ET.SubElement(root, "edge",
                           id="e_work", **{"from": "n2"}, to="n3",
                           numLanes=str(N), speed=f"{cfg.speed_limit:.2f}",
                           spreadType="right")
    # Add lane sub-elements with disallow for closed lanes
    for lane_idx in range(N):
        if lane_idx in closed_sumo:
            ET.SubElement(e_work, "lane", index=str(lane_idx), disallow="passenger truck")
        # Open lanes: no disallow (default allow)

    # --- e_taper_r  (N lanes, straight; rear taper open) ---
    e_taper_r = ET.SubElement(root, "edge",
                              id="e_taper_r", **{"from": "n3"}, to="n4",
                              numLanes=str(N), speed=taper_speed_str,
                              spreadType="right")
    if is_lane23_workzone and not cfg.reassign_arrival_lane_in_taper:
        ET.SubElement(e_taper_r, "lane", index=str(sumo_idx_lane2), changeRight="pedestrian")
        ET.SubElement(e_taper_r, "lane", index=str(sumo_idx_lane3), changeLeft="pedestrian")

    # --- e_post  (N lanes, straight) ---
    ET.SubElement(root, "edge",
                  id="e_post", **{"from": "n4"}, to="n5",
                  numLanes=str(N), speed=f"{cfg.speed_limit:.2f}",
                  spreadType="right")

    # --- e_downstream (N lanes): exit section; vehicles leave at n6 (dead_end)
    ET.SubElement(root, "edge",
                  id="e_downstream", **{"from": "n5"}, to="n6",
                  numLanes=str(N), speed=f"{cfg.speed_limit:.2f}",
                  spreadType="right")

    _indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _write_connections(cfg: WorkzoneConfig, path: str) -> None:
    """Write explicit lane connections. All edges have N lanes. All lanes get
    1-to-1 connections at all junctions. Lane closure is enforced via disallow
    on the closed lanes in the last front-taper segment (e_taper_f3) and in
    e_work, encouraging earlier merges in the front taper.

    Lane index mapping: cfg uses 0-indexed leftmost (0=left). SUMO uses 0=rightmost.
    sumo_idx = N - 1 - orig_idx. closed_lanes string is 1-indexed (1=leftmost).
    """
    N = cfg.num_lanes

    root = ET.Element("connections")

    extend_f1 = cfg.extend_front_taper_first_segment and cfg.front_taper_first_segment_extension > 0
    first_f1_edge = "e_taper_f1_ext" if extend_f1 else "e_taper_f1"

    # n1: e_pre (N) → first front-taper edge (N), all lanes 1-to-1
    for lane in range(N):
        ET.SubElement(root, "connection",
                      **{"from": "e_pre"}, to=first_f1_edge,
                      fromLane=str(lane), toLane=str(lane))

    if extend_f1:
        # n1a: e_taper_f1_ext (N) → e_taper_f1 (N), all lanes 1-to-1
        for lane in range(N):
            ET.SubElement(root, "connection",
                         **{"from": "e_taper_f1_ext"}, to="e_taper_f1",
                         fromLane=str(lane), toLane=str(lane))

    # n2a: e_taper_f1 (N) → e_taper_f2 (N), all lanes 1-to-1
    for lane in range(N):
        ET.SubElement(root, "connection",
                      **{"from": "e_taper_f1"}, to="e_taper_f2",
                      fromLane=str(lane), toLane=str(lane))

    # n2b: e_taper_f2 (N) → e_taper_f3 (N), all lanes 1-to-1
    for lane in range(N):
        ET.SubElement(root, "connection",
                      **{"from": "e_taper_f2"}, to="e_taper_f3",
                      fromLane=str(lane), toLane=str(lane))

    # n2: e_taper_f3 (N) → e_work (N), all lanes 1-to-1
    for lane in range(N):
        ET.SubElement(root, "connection",
                      **{"from": "e_taper_f3"}, to="e_work",
                      fromLane=str(lane), toLane=str(lane))

    # n3: e_work (N) → e_taper_r (N), all lanes 1-to-1
    for lane in range(N):
        ET.SubElement(root, "connection",
                      **{"from": "e_work"}, to="e_taper_r",
                      fromLane=str(lane), toLane=str(lane))

    # n4: e_taper_r (N) → e_post (N), all lanes 1-to-1
    for lane in range(N):
        ET.SubElement(root, "connection",
                      **{"from": "e_taper_r"}, to="e_post",
                      fromLane=str(lane), toLane=str(lane))

    # n5: e_post (N) → e_downstream (N), all lanes 1-to-1
    for lane in range(N):
        ET.SubElement(root, "connection",
                      **{"from": "e_post"}, to="e_downstream",
                      fromLane=str(lane), toLane=str(lane))

    _indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


# ------------------------------------------------------------------ #
#  Public API
# ------------------------------------------------------------------ #

def build_network(cfg: WorkzoneConfig) -> str:
    """Generate the full SUMO network file.  Returns path to .net.xml."""
    os.makedirs(cfg.output_dir, exist_ok=True)

    nod_path = cfg.out_path("workzone.nod.xml")
    edg_path = cfg.out_path("workzone.edg.xml")
    con_path = cfg.out_path("workzone.con.xml")
    typ_path = cfg.out_path("workzone.typ.xml")
    net_path = cfg.net_file

    print(f"[Network] Writing types  → {typ_path}")
    _write_types(cfg, typ_path)

    if cfg.extend_front_taper_first_segment:
        print(f"[Network] Front taper first segment extended upstream by {cfg.front_taper_first_segment_extension}m")
    print(f"[Network] Writing nodes  → {nod_path}")
    _write_nodes(cfg, nod_path)

    print(f"[Network] Writing edges  → {edg_path}")
    _write_edges(cfg, edg_path)

    print(f"[Network] Writing connections → {con_path}")
    _write_connections(cfg, con_path)

    print(f"[Network] Running netconvert …")
    cmd = [
        "netconvert",
        "--node-files", nod_path,
        "--edge-files", edg_path,
        "--connection-files", con_path,
        "--type-files", typ_path,
        "--output-file", net_path,
        "--default.lanewidth", f"{cfg.lane_width:.2f}",
        "--default.spreadtype", "right",
        "--no-turnarounds", "true",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[Network] netconvert STDERR:")
        print(result.stderr)
        raise RuntimeError(f"netconvert failed (exit {result.returncode}). See stderr above.")

    print(f"[Network] Network written → {net_path}")
    return net_path
