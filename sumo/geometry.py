"""
geometry.py — Workzone geometry: cosine taper curves and lane-target calculations.

Ported from CARLA workzone_helpers.py (cos_lerp, setup_workzone lane-boundary logic).
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from config import WorkzoneConfig


# ------------------------------------------------------------------ #
#  Core interpolation
# ------------------------------------------------------------------ #

def cos_lerp(v0: float, v1: float, t: float) -> float:
    """Cosine interpolation (smooth-step).

    Identical to CARLA workzone_helpers.py:143-146:
        w = 0.5 * (1 - cos(pi * t))
        result = v0 * (1 - w) + v1 * w
    """
    t = max(0.0, min(1.0, t))
    w = 0.5 * (1.0 - math.cos(math.pi * t))
    return v0 * (1.0 - w) + v1 * w


# ------------------------------------------------------------------ #
#  Contiguous-block grouping  (CARLA workzone_helpers.py:180-186)
# ------------------------------------------------------------------ #

def _group_contiguous(indices: List[int]) -> List[List[int]]:
    """Group sorted indices into contiguous blocks.

    Example: [0, 1, 3] → [[0, 1], [3]]
    """
    if not indices:
        return []
    blocks: List[List[int]] = [[indices[0]]]
    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1] + 1:
            blocks[-1].append(indices[i])
        else:
            blocks.append([indices[i]])
    return blocks


def compute_disallow_last_two_thirds_lanes(cfg: WorkzoneConfig) -> set[int]:
    """Return lanes (0-indexed, leftmost=0) that need disallow on the front-taper
    last 2/3 (e_taper_f2 + e_taper_f3). Middle-isolated blocks (single lane with
    open lanes on both sides) use last 2/3; edge-anchored and middle-contiguous
    use last 1/3 only.
    """
    closed = sorted(cfg.closed_lane_indices)
    if not closed:
        return set()
    N = cfg.num_lanes
    blocks = _group_contiguous(closed)
    result: set[int] = set()
    for block in blocks:
        left_anchored = block[0] == 0
        right_anchored = block[-1] == N - 1
        if left_anchored or right_anchored:
            continue  # last 1/3 only
        if len(block) == 1:
            result.update(block)  # middle isolated → last 2/3
        # len(block) > 1: middle contiguous → last 1/3 only
    return result


# ------------------------------------------------------------------ #
#  Preserve-side lane-target calculator
# ------------------------------------------------------------------ #

def compute_lane_targets(cfg: WorkzoneConfig) -> Dict[int, Tuple[float, float]]:
    """Compute (y_normal, y_work) for every *open* lane.

    Uses the *preserve-side packing* strategy that mirrors CARLA's
    directional taper behaviour (workzone_helpers.py:195-214):

      - LEFT-ANCHORED closure  → open lanes pack LEFT
      - RIGHT-ANCHORED closure → open lanes pack RIGHT
      - MIDDLE closure         → lanes left of gap stay, lanes right shift left
      - Multiple blocks        → each block's neighbours shift inward

    Y coordinate convention (matching SUMO spreadType="right"):
        y = 0  at the *right* road edge; y grows leftward.
        Lane i (0 = leftmost) has normal centre at:
            y_normal = lane_width * (num_lanes - 1 - i) + lane_width / 2
    """
    N = cfg.num_lanes
    w = cfg.lane_width
    closed_set = set(cfg.closed_lane_indices)
    open_lanes = cfg.open_lane_indices          # sorted, 0 = leftmost

    def _y_normal(lane_idx: int) -> float:
        """Normal centre-Y for a lane (right-spread coords)."""
        return w * (N - 1 - lane_idx) + w / 2.0

    blocks = _group_contiguous(sorted(closed_set))

    # Start with each open lane at its normal position; we will shift some.
    targets: Dict[int, float] = {i: _y_normal(i) for i in open_lanes}

    if not blocks:
        # Nothing closed — no shift needed.
        return {i: (_y_normal(i), _y_normal(i)) for i in open_lanes}

    if len(blocks) == 1:
        block = blocks[0]
        left_anchored = block[0] == 0
        right_anchored = block[-1] == N - 1

        if left_anchored and right_anchored:
            # Full closure — should have been caught by validate(), but handle.
            return {}

        if left_anchored and not right_anchored:
            # Closed lanes are on the LEFT. Open lanes pack LEFT (high Y → low Y
            # in our right-spread coords is actually … let's think carefully).
            #
            # In right-spread coords bigger Y = more to the LEFT.
            # "Pack left" means open lanes move to occupy positions starting from
            # the leftmost physical position (highest Y).
            M = len(open_lanes)
            for j, orig in enumerate(open_lanes):
                # j-th open lane (sorted left→right) gets position of the j-th
                # lane from the left in an M-lane road that is left-aligned.
                targets[orig] = w * (N - 1 - j) + w / 2.0

        elif right_anchored and not left_anchored:
            # Closed lanes on the RIGHT. Open lanes pack RIGHT (smallest Y values).
            M = len(open_lanes)
            for j, orig in enumerate(open_lanes):
                # j-th open lane occupies position counting from right edge.
                # The rightmost open lane (last in sorted list) gets y = w/2.
                targets[orig] = w * (M - 1 - j) + w / 2.0

        else:
            # MIDDLE closure: lanes left of block stay, lanes right of block shift left.
            gap_width = len(block) * w
            right_of_block = [i for i in open_lanes if i > block[-1]]
            for i in right_of_block:
                targets[i] = _y_normal(i) + gap_width   # shift toward higher Y = leftward
    else:
        # Multiple non-contiguous blocks — process each independently.
        # Walk left→right through blocks; accumulate gap width.
        # Each open lane to the right of a block shifts leftward (higher Y in
        # right-spread coords) by the cumulative gap seen so far.
        targets = {i: _y_normal(i) for i in open_lanes}
        cum = 0.0
        for block in blocks:
            cum += len(block) * w
            for i in open_lanes:
                if i > block[-1]:
                    targets[i] = _y_normal(i) + cum

    return {i: (_y_normal(i), targets[i]) for i in open_lanes}


# ------------------------------------------------------------------ #
#  Shape-point generation for taper edges
# ------------------------------------------------------------------ #

def generate_taper_shapes(
    cfg: WorkzoneConfig,
    targets: Dict[int, Tuple[float, float]],
    is_front: bool,
) -> List[Tuple[int, List[Tuple[float, float]]]]:
    """Generate per-lane shape polylines for a taper edge.

    Returns a list of (sumo_lane_index, [(x, y), ...]) sorted by sumo_lane_index.
    sumo_lane_index is 0 = rightmost in SUMO convention.

    For the front taper the interpolation goes  y_normal → y_work.
    For the rear  taper the interpolation goes  y_work   → y_normal.
    """
    if is_front:
        taper_len = cfg.front_taper_length
        x_start = cfg.workzone_start
    else:
        taper_len = cfg.rear_taper_length
        x_start = cfg.workzone_start + cfg.front_taper_length + cfg.hold_length

    n_pts = max(int(taper_len / cfg.shape_resolution) + 1, 2)
    open_lanes = cfg.open_lane_indices
    M = len(open_lanes)

    result: List[Tuple[int, List[Tuple[float, float]]]] = []

    for j, orig_idx in enumerate(open_lanes):
        y_normal, y_work = targets[orig_idx]

        if is_front:
            y_from, y_to = y_normal, y_work
        else:
            y_from, y_to = y_work, y_normal

        pts: List[Tuple[float, float]] = []
        for k in range(n_pts):
            t = k / max(n_pts - 1, 1)
            x = x_start + t * taper_len
            y = cos_lerp(y_from, y_to, t)
            pts.append((round(x, 4), round(y, 4)))

        # Convert to SUMO lane index (0 = rightmost among the M open lanes)
        sumo_lane = M - 1 - j
        result.append((sumo_lane, pts))

    result.sort(key=lambda pair: pair[0])
    return result


def generate_work_shapes(
    cfg: WorkzoneConfig,
    targets: Dict[int, Tuple[float, float]],
) -> List[Tuple[int, List[Tuple[float, float]]]]:
    """Generate per-lane shape for the constant work-area edge (straight at y_work)."""
    x_start = cfg.workzone_start + cfg.front_taper_length
    x_end = x_start + cfg.hold_length
    open_lanes = cfg.open_lane_indices
    M = len(open_lanes)

    result: List[Tuple[int, List[Tuple[float, float]]]] = []
    for j, orig_idx in enumerate(open_lanes):
        _, y_work = targets[orig_idx]
        sumo_lane = M - 1 - j
        pts = [(round(x_start, 4), round(y_work, 4)),
               (round(x_end, 4), round(y_work, 4))]
        result.append((sumo_lane, pts))

    result.sort(key=lambda pair: pair[0])
    return result


def _y_center_lane(cfg: WorkzoneConfig, lane_idx: int) -> float:
    """Lane center Y for lane_idx (0=leftmost). spreadType=right: y=0 at right edge."""
    N = cfg.num_lanes
    w = cfg.lane_width
    return w * (N - 1 - lane_idx) + w / 2.0


def generate_cone_positions(
    cfg: WorkzoneConfig,
    spacing: float = 3.0,
) -> List[Tuple[float, float]]:
    """Generate (x, y) positions for visual cone markers along taper boundaries.

    Uses the same fixed boundary logic as visual_builder: y in [0, total_road_width].
    Left-anchored: curve y_anchor -> y_boundary -> y_anchor; right-anchored similarly.
    """
    closed = sorted(cfg.closed_lane_indices)
    if not closed:
        return []

    N = cfg.num_lanes
    w = cfg.lane_width
    total_w = cfg.total_road_width
    blocks = _group_contiguous(closed)
    positions: List[Tuple[float, float]] = []

    for block in blocks:
        left_anchored = block[0] == 0
        right_anchored = block[-1] == N - 1

        if left_anchored and right_anchored:
            continue  # full closure, no cones needed

        if left_anchored:
            y_anchor = total_w
            first_open = block[-1] + 1
            y_boundary = _y_center_lane(cfg, first_open) + w / 2.0
            y_boundary = max(0.0, min(total_w, y_boundary))
            _add_cone_line(positions, cfg, y_anchor, y_boundary, spacing)
        elif right_anchored:
            y_anchor = 0.0
            last_open = block[0] - 1
            y_boundary = _y_center_lane(cfg, last_open) - w / 2.0
            y_boundary = max(0.0, min(total_w, y_boundary))
            _add_cone_line(positions, cfg, y_anchor, y_boundary, spacing)
        else:
            # Middle block: two taper curves from center to left/right (like CARLA V2)
            y_left = _y_center_lane(cfg, block[0] - 1) - w / 2.0
            y_right = _y_center_lane(cfg, block[-1] + 1) + w / 2.0
            y_left = max(0.0, min(total_w, y_left))
            y_right = max(0.0, min(total_w, y_right))
            y_center = (y_left + y_right) / 2.0
            _add_cone_line(positions, cfg, y_center, y_left, spacing)
            _add_cone_line(positions, cfg, y_center, y_right, spacing)

    return positions


def _add_cone_line(
    out: List[Tuple[float, float]],
    cfg: WorkzoneConfig,
    y_normal: float,
    y_work: float,
    spacing: float,
) -> None:
    """Append cone (x, y) positions for one boundary through the full workzone."""
    wz_start = cfg.workzone_start
    front_end = wz_start + cfg.front_taper_length
    hold_end = front_end + cfg.hold_length
    wz_end = wz_start + cfg.workzone_length

    s = 0.0
    total = cfg.workzone_length
    while s <= total + 1e-6:
        x = wz_start + s
        t = s / max(total, 1e-6)
        front_r = cfg.taper_front_ratio
        hold_r = cfg.taper_hold_ratio
        rear_r = cfg.taper_rear_ratio

        if t < front_r:
            y = cos_lerp(y_normal, y_work, t / front_r)
        elif t < front_r + hold_r:
            y = y_work
        else:
            y = cos_lerp(y_work, y_normal, (t - front_r - hold_r) / max(rear_r, 1e-6))

        out.append((round(x, 3), round(y, 3)))
        s += spacing
