"""
visual_builder.py — Generate SUMO .add.xml for workzone visualisation.

Creates:
  - Semi-transparent polygon highlighting the workzone area
  - POI markers along cone taper curves (every 3 m)

Coordinate convention: Polygons and cones are computed in a fixed coordinate system with
y in [0, total_road_width], where y=0 is the right edge and y=total_road_width is the left edge.
These coordinates are then automatically shifted to match the actual road geometry in the generated
workzone.net.xml by parsing the network file and inferring the real y-bounds. This ensures visual
elements align correctly regardless of whether SUMO outputs right-spread or centered coordinates.

This module does NOT use compute_lane_targets or lane-packing logic; it uses fixed lane geometry
based on lane_width and num_lanes.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple

from config import WorkzoneConfig
from geometry import (
    generate_cone_positions,
    _group_contiguous,
    cos_lerp,
)


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


def _infer_road_y_shift(cfg: WorkzoneConfig, ref_edge_id: str = "e_pre") -> float:
    """Infer the y-coordinate shift needed to align visual elements with the actual
    road geometry in the generated network file.
    
    Our visual coordinates assume y in [0, total_road_width]. This function finds
    where the actual road is in the network and calculates the offset.
    
    Returns:
        y_shift: The y offset to add to all visual element coordinates.
    """
    try:
        if not os.path.exists(cfg.net_file):
            print(f"[Visual] Warning: {cfg.net_file} not found. Using y_shift=0.")
            return 0.0
        
        tree = ET.parse(cfg.net_file)
        root = tree.getroot()
        
        # Find the reference edge - skip internal edges  
        edge_elem = None
        for edge in root.findall(".//edge"):
            edge_id = edge.get("id", "")
            if ":" in edge_id or edge.get("function") == "internal":
                continue
            if edge_id == ref_edge_id:
                edge_elem = edge
                break
        
        if edge_elem is None:
            print(f"[Visual] Warning: edge '{ref_edge_id}' not found. Using y_shift=0.")
            return 0.0
        
        # Collect y-coordinates from all lane centerline points
        y_coords = []
        lanes = edge_elem.findall("lane")
        
        for lane in lanes:
            shape_str = lane.get("shape", "")
            if not shape_str:
                continue
            # Parse "x1,y1 x2,y2 ..." and collect ALL y values
            for point_str in shape_str.strip().split():
                parts = point_str.split(",")
                if len(parts) < 2:
                    continue
                try:
                    y = float(parts[1])
                    y_coords.append(y)
                except ValueError:
                    # Skip malformed coordinates but continue parsing others
                    continue
        
        if not y_coords:
            print(f"[Visual] Warning: No lane shapes found. Using y_shift=0.")
            return 0.0
        
        # Calculate actual road bounds from all sampled lane centerline points
        min_center_y = min(y_coords)
        max_center_y = max(y_coords)
        actual_road_bottom = min_center_y - cfg.lane_width / 2.0
        actual_road_top = max_center_y + cfg.lane_width / 2.0
        actual_road_width = actual_road_top - actual_road_bottom
        
        print(f"[Visual] Found {len(y_coords)} lane sample points, centers at y=[{min_center_y:.2f}, {max_center_y:.2f}]")
        print(f"[Visual] Actual road y-range: [{actual_road_bottom:.2f}, {actual_road_top:.2f}], width={actual_road_width:.2f}m")
        print(f"[Visual] Expected road width: {cfg.total_road_width:.2f}m")
        
        # Our visual coords assume [0, total_road_width]
        # Actual network has [actual_road_bottom, actual_road_top]
        # So we shift our 0 to actual_road_bottom
        y_shift = actual_road_bottom
        
        # Safety check: if shift is unreasonably large, something went wrong
        if abs(y_shift) > 100:
            print(f"[Visual] WARNING: y_shift={y_shift:.2f} seems too large! Using 0 instead.")
            print(f"[Visual] This might indicate a problem with coordinate system interpretation.")
            return 0.0
        
        # Verify the width matches expectations (within 10% tolerance)
        width_diff = abs(actual_road_width - cfg.total_road_width)
        if width_diff > cfg.total_road_width * 0.1:
            print(f"[Visual] WARNING: Road width mismatch! Expected {cfg.total_road_width:.2f}m, got {actual_road_width:.2f}m")
            print(f"[Visual] Proceeding with inferred y_shift despite width mismatch; please check lane_width/num_lanes and coordinate system.")
        
        print(f"[Visual] Applying y_shift = {y_shift:.2f}m to align visual elements")
        return y_shift
        
    except Exception as e:
        print(f"[Visual] Error parsing {cfg.net_file}: {e}. Using y_shift=0.")
        import traceback
        traceback.print_exc()
        return 0.0


def _taper_curve_points(
    cfg: WorkzoneConfig,
    y_normal: float,
    y_work: float,
    resolution: float,
) -> List[Tuple[float, float]]:
    """Generate (x, y) points along one taper boundary curve through the
    full work zone (front taper + hold + rear taper).

    Uses the same parameterisation as geometry._add_cone_line().
    """
    wz_start = cfg.workzone_start
    total = cfg.workzone_length
    front_r = cfg.taper_front_ratio
    hold_r = cfg.taper_hold_ratio
    rear_r = cfg.taper_rear_ratio

    n_pts = max(int(total / resolution) + 1, 2)
    points: List[Tuple[float, float]] = []
    for k in range(n_pts):
        t = k / max(n_pts - 1, 1)
        x = wz_start + t * total

        if t < front_r:
            y = cos_lerp(y_normal, y_work, t / front_r)
        elif t < front_r + hold_r:
            y = y_work
        else:
            y = cos_lerp(y_work, y_normal,
                         (t - front_r - hold_r) / max(rear_r, 1e-6))

        points.append((round(x, 4), round(y, 4)))
    return points


def _y_center_lane(cfg: WorkzoneConfig, lane_idx: int) -> float:
    """Lane center Y for lane_idx (0=leftmost). spreadType=right: y=0 at right edge."""
    N = cfg.num_lanes
    w = cfg.lane_width
    return w * (N - 1 - lane_idx) + w / 2.0


def _generate_wz_polygon_shapes(cfg: WorkzoneConfig, y_shift: float = 0.0) -> List[str]:
    """Generate SUMO polygon shape strings for the closed-lane area.

    Uses fixed lane geometry (no lane targets). y in [0, total_road_width],
    then shifted by y_shift to align with actual network coordinates.
    One polygon per contiguous block; taper curves: front hold rear.
    
    Args:
        cfg: Workzone configuration
        y_shift: Y-coordinate offset to add to all points (from _infer_road_y_shift)
    """
    closed = sorted(cfg.closed_lane_indices)
    if not closed:
        return []

    N = cfg.num_lanes
    w = cfg.lane_width
    total_w = cfg.total_road_width
    blocks = _group_contiguous(closed)
    resolution = cfg.shape_resolution
    x_start = cfg.workzone_start
    x_end = cfg.workzone_start + cfg.workzone_length
    shapes: List[str] = []

    for block in blocks:
        left_anchored = block[0] == 0
        right_anchored = block[-1] == N - 1

        if left_anchored and right_anchored:
            # Full closure — rectangle inside road bounds
            y_lo = 0.0 + y_shift
            y_hi = total_w + y_shift
            shape_str = (f"{x_start},{y_lo} {x_end},{y_lo} "
                         f"{x_end},{y_hi} {x_start},{y_hi}")
            shapes.append(shape_str)
            continue

        if left_anchored:
            # Closed on the left. Anchor at left road edge (y = total_road_width).
            y_anchor = total_w
            first_open = block[-1] + 1
            y_boundary = _y_center_lane(cfg, first_open) + w / 2.0
            y_boundary = max(0.0, min(total_w, y_boundary))

            curve = _taper_curve_points(cfg, y_anchor, y_boundary, resolution)
            pts = curve + [(x_end, y_anchor), (x_start, y_anchor)]

        elif right_anchored:
            # Closed on the right. Anchor at right road edge (y = 0).
            y_anchor = 0.0
            last_open = block[0] - 1
            y_boundary = _y_center_lane(cfg, last_open) - w / 2.0
            y_boundary = max(0.0, min(total_w, y_boundary))

            curve = _taper_curve_points(cfg, y_anchor, y_boundary, resolution)
            pts = curve + [(x_end, y_anchor), (x_start, y_anchor)]

        else:
            # Middle block: double-curve polygon (center → left/right taper, like CARLA V2)
            y_left = _y_center_lane(cfg, block[0] - 1) - w / 2.0
            y_right = _y_center_lane(cfg, block[-1] + 1) + w / 2.0
            y_left = max(0.0, min(total_w, y_left))
            y_right = max(0.0, min(total_w, y_right))
            y_center = (y_left + y_right) / 2.0
            left_curve = _taper_curve_points(cfg, y_center, y_left, resolution)
            right_curve = _taper_curve_points(cfg, y_center, y_right, resolution)
            pts = left_curve + list(reversed(right_curve))

        # Apply y_shift to all points before building shape string
        shape_str = " ".join(f"{x},{y + y_shift}" for x, y in pts)
        shapes.append(shape_str)

    return shapes


def _generate_closed_lane_rect_shape(cfg: WorkzoneConfig, y_shift: float = 0.0) -> Optional[str]:
    """Generate a rectangular polygon shape covering the full width of closed lanes
    from e_taper_f1 start to e_taper_r end. This ensures complete coverage where
    disallow is applied, avoiding gray lane visibility in taper areas.
    """
    closed = sorted(cfg.closed_lane_indices)
    if not closed:
        return None

    w = cfg.lane_width
    leftmost = min(closed)
    rightmost = max(closed)
    y_lo = _y_center_lane(cfg, rightmost) - w / 2.0
    y_hi = _y_center_lane(cfg, leftmost) + w / 2.0
    y_lo = max(0.0, min(cfg.total_road_width, y_lo))
    y_hi = max(0.0, min(cfg.total_road_width, y_hi))

    if cfg.extend_front_taper_first_segment:
        x_start = cfg.workzone_start - cfg.front_taper_first_segment_extension
    else:
        x_start = cfg.workzone_start
    x_end = cfg.workzone_start + cfg.workzone_length

    y_lo_shifted = y_lo + y_shift
    y_hi_shifted = y_hi + y_shift
    shape_str = (f"{x_start},{y_lo_shifted} {x_end},{y_lo_shifted} "
                 f"{x_end},{y_hi_shifted} {x_start},{y_hi_shifted}")
    return shape_str


def build_additional(cfg: WorkzoneConfig) -> str:
    """Generate .add.xml with workzone visual elements.  Returns file path."""
    os.makedirs(cfg.output_dir, exist_ok=True)
    path = cfg.additional_file

    root = ET.Element("additional")

    # Infer the y-coordinate shift from the actual generated network
    y_shift = _infer_road_y_shift(cfg)

    wz_shapes = _generate_wz_polygon_shapes(cfg, y_shift=y_shift)

    # ------------------------------------------------------------------ #
    #  Closed lane area polygon(s) (base layer, road-like color)
    #  Only drawn when draw_closed_lane_polygon=True
    # ------------------------------------------------------------------ #
    if cfg.draw_closed_lane_polygon:
        for idx, shape in enumerate(wz_shapes):
            poly_id = "closed_lane" if len(wz_shapes) == 1 else f"closed_lane_{idx}"
            ET.SubElement(root, "poly",
                          id=poly_id,
                          type="closed_lane",
                          color=cfg.closed_lane_color,
                          fill="1",
                          layer="5",
                          shape=shape)
        # Rectangular polygon covering full closed-lane width from e_taper_f1 to e_taper_r end
        # (defense-in-depth: ensures complete coverage where disallow is applied)
        rect_shape = _generate_closed_lane_rect_shape(cfg, y_shift=y_shift)
        if rect_shape is not None:
            ET.SubElement(root, "poly",
                          id="closed_lane_rect",
                          type="closed_lane",
                          color=cfg.closed_lane_color,
                          fill="1",
                          layer="5",
                          shape=rect_shape)

    # ------------------------------------------------------------------ #
    #  Workzone area polygon(s) (overlay layer, semi-transparent highlight)
    # ------------------------------------------------------------------ #
    for idx, shape in enumerate(wz_shapes):
        poly_id = "wz_area" if len(wz_shapes) == 1 else f"wz_area_{idx}"
        ET.SubElement(root, "poly",
                      id=poly_id,
                      type="workzone",
                      color=cfg.workzone_color,
                      fill="1",
                      layer="10",
                      shape=shape)

    # ------------------------------------------------------------------ #
    #  Cone POIs along the taper boundaries (same boundary logic as polygons)
    # ------------------------------------------------------------------ #
    cones = generate_cone_positions(cfg, spacing=3.0)
    for i, (cx, cy) in enumerate(cones):
        ET.SubElement(root, "poi",
                      id=f"cone_{i}",
                      type="cone",
                      color="1,0.55,0",
                      x=f"{cx:.3f}",
                      y=f"{cy + y_shift:.3f}",
                      width="0.25",
                      height="0.25",
                      layer="20")

    _indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    print(f"[Visual]  Written → {path}  ({len(cones)} cone POIs, "
          f"{len(wz_shapes)} wz polygon(s))")
    return path
