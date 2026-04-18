"""
fcd_converter.py — Convert SUMO FCD XML output to CSV.

FCD XML structure:
    <fcd-export>
        <timestep time="0.10">
            <vehicle id="car_flow.0" x="5.0" y="3.75" angle="90.0"
                     speed="12.5" pos="5.0" lane="e_pre_2" type="car" …/>
            …
        </timestep>
        …
    </fcd-export>

Output CSV columns:
    timestamp_s, vehicle_id, x, y, speed, angle, lane_id, edge_id, type
"""

from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET


CSV_HEADER = [
    "timestamp_s",
    "vehicle_id",
    "x",
    "y",
    "speed",
    "angle",
    "lane_id",
    "edge_id",
    "type",
]

# Regex for fallback parsing of truncated XML (e.g. simulation interrupted)
_RE_TIMESTEP = re.compile(r'<timestep\s+time="([^"]+)"')


def _extract_vehicle_attrs(line: str) -> dict | None:
    """Extract vehicle attributes from a complete <vehicle .../> line. Returns None if incomplete."""
    if "<vehicle" not in line or "/>" not in line:
        return None
    attrs = {}
    for name in ("id", "x", "y", "angle", "type", "speed", "lane"):
        m = re.search(rf'{name}="([^"]*)"', line)
        if not m:
            return None
        attrs[name] = m.group(1)
    return attrs


def _convert_fcd_fallback(fcd_xml_path: str, csv_output_path: str) -> int:
    """Fallback: parse truncated FCD XML line-by-line when standard parser fails."""
    rows = 0
    current_time = "0"
    with open(fcd_xml_path, "r", encoding="utf-8") as fin:
        with open(csv_output_path, "w", newline="", encoding="utf-8") as fout:
            writer = csv.writer(fout)
            writer.writerow(CSV_HEADER)
            for line in fin:
                ts = _RE_TIMESTEP.search(line)
                if ts:
                    current_time = ts.group(1)
                attrs = _extract_vehicle_attrs(line)
                if attrs:
                    lane = attrs["lane"]
                    parts = lane.rsplit("_", 1)
                    edge_id = parts[0] if len(parts) == 2 and parts[1].isdigit() else lane
                    writer.writerow([
                        current_time, attrs["id"], attrs["x"], attrs["y"],
                        attrs["speed"], attrs["angle"], lane, edge_id, attrs["type"]
                    ])
                    rows += 1
    return rows


def _convert_fcd_standard(fcd_xml_path: str, csv_output_path: str) -> int:
    """Standard iterparse-based conversion."""
    rows = 0
    with open(csv_output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

        # Use iterparse to avoid loading the entire XML into memory
        context = ET.iterparse(fcd_xml_path, events=("start", "end"))
        current_time: str = "0"

        for event, elem in context:
            if event == "start" and elem.tag == "timestep":
                current_time = elem.get("time", "0")
            elif event == "end" and elem.tag == "vehicle":
                lane = elem.get("lane", "")
                # Extract edge id from lane id (e.g., "e_pre_2" → "e_pre")
                parts = lane.rsplit("_", 1)
                edge_id = parts[0] if len(parts) == 2 and parts[1].isdigit() else lane

                writer.writerow([
                    current_time,
                    elem.get("id", ""),
                    elem.get("x", ""),
                    elem.get("y", ""),
                    elem.get("speed", ""),
                    elem.get("angle", ""),
                    lane,
                    edge_id,
                    elem.get("type", ""),
                ])
                rows += 1
                elem.clear()  # free memory
            elif event == "end" and elem.tag == "timestep":
                elem.clear()

    return rows


def convert_fcd(fcd_xml_path: str, csv_output_path: str) -> int:
    """Parse FCD XML and write CSV.  Returns the number of data rows written.
    Falls back to line-by-line parsing if XML is truncated (e.g. simulation interrupted).
    """
    try:
        rows = _convert_fcd_standard(fcd_xml_path, csv_output_path)
    except ET.ParseError as e:
        print(f"[FCD]     XML parse error (truncated file?): {e}")
        print(f"[FCD]     Using fallback parser to extract valid data...")
        rows = _convert_fcd_fallback(fcd_xml_path, csv_output_path)
    print(f"[FCD]     Converted {rows} records → {csv_output_path}")
    return rows
