"""
simulation_runner.py — Build .sumocfg and run SUMO simulation via TraCI.
"""

from __future__ import annotations

import os
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from config import WorkzoneConfig


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


def _write_sumocfg(cfg: WorkzoneConfig) -> str:
    """Generate .sumocfg.  Returns the file path."""
    path = cfg.sumocfg_file

    # Use relative names (files sit in the same output_dir)
    net_name = os.path.basename(cfg.net_file)
    rou_name = os.path.basename(cfg.route_file)
    add_name = os.path.basename(cfg.additional_file)
    fcd_name = os.path.basename(cfg.fcd_file)

    root = ET.Element("configuration")

    inp = ET.SubElement(root, "input")
    ET.SubElement(inp, "net-file", value=net_name)
    ET.SubElement(inp, "route-files", value=rou_name)
    ET.SubElement(inp, "additional-files", value=add_name)

    time_el = ET.SubElement(root, "time")
    ET.SubElement(time_el, "begin", value="0")
    ET.SubElement(time_el, "end", value=f"{cfg.sim_duration:.0f}")
    ET.SubElement(time_el, "step-length", value=f"{cfg.step_length}")

    proc = ET.SubElement(root, "processing")
    if cfg.lateral_resolution is not None:
        ET.SubElement(proc, "lateral-resolution", value=f"{cfg.lateral_resolution}")
    else:
        ET.SubElement(proc, "lanechange.duration", value=f"{cfg.lc_duration}")
    ET.SubElement(proc, "device.fcd.period", value=f"{cfg.step_length}")
    ET.SubElement(proc, "time-to-teleport", value=str(cfg.time_to_teleport))
    ET.SubElement(proc, "time-to-teleport.remove", value=str(cfg.time_to_teleport_remove).lower())

    output_el = ET.SubElement(root, "output")
    ET.SubElement(output_el, "fcd-output", value=fcd_name)

    rand = ET.SubElement(root, "random_number")
    ET.SubElement(rand, "seed", value=str(cfg.seed))

    _indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    print(f"[Config]  Written → {path}")
    return path


def write_gui_settings(cfg: WorkzoneConfig) -> str:
    """Write gui_settings.xml that forces uniform lane coloring.

    This overrides SUMO's default behavior of rendering disallow-ed lanes
    in a different color (whitish/light gray), making closed lanes match
    normal road surface color.
    """
    path = cfg.gui_settings_file

    # Parse closed_lane_color to get RGBA values for the scheme
    parts = [float(p.strip()) for p in cfg.closed_lane_color.split(",")]
    # Normalize to 0-1 range if needed
    if any(v > 1.0 for v in parts):
        parts = [v / 255.0 for v in parts]
    r, g, b = parts[0], parts[1], parts[2]

    # Convert to 0-255 integer format for SUMO gui settings (RGB only; alpha not used)
    r255, g255, b255 = int(r * 255), int(g * 255), int(b * 255)

    root = ET.Element("viewsettings")
    ET.SubElement(root, "delay", value="0")

    scheme = ET.SubElement(root, "scheme", name="custom_workzone")

    # Use laneEdgeMode="0" which is "uniform" mode — paints all lanes the same color
    # laneShowBorders="1" keeps lane boundary lines visible
    edges = ET.SubElement(scheme, "edges",
                          laneEdgeMode="0",
                          laneShowBorders="1")

    # The "uniform" colorScheme entry defines the single color for all lanes.
    # SUMO viewsettings typically expects RGB (3 values) or #RRGGBB; 4-value RGBA may not be recognized.
    color_scheme = ET.SubElement(edges, "colorScheme", name="uniform")
    ET.SubElement(color_scheme, "entry",
                  color=f"{r255},{g255},{b255}",
                  name="")

    _indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    print(f"[GUI]     Written → {path}  (lane color: rgb={r255},{g255},{b255})")
    return path


def _ensure_sumo_home() -> None:
    if "SUMO_HOME" not in os.environ:
        raise EnvironmentError(
            "SUMO_HOME environment variable is not set. "
            "Please install SUMO and set SUMO_HOME to the installation directory."
        )
    tools_dir = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)


def run_simulation(cfg: WorkzoneConfig) -> None:
    """Launch SUMO and step through the simulation via TraCI."""
    _ensure_sumo_home()
    _write_sumocfg(cfg)

    import traci  # imported here so the rest of the code can load without SUMO installed
    import time

    binary = "sumo-gui" if cfg.gui else "sumo"
    sumo_cmd = [binary, "-c", cfg.sumocfg_file]
    # Write warnings/errors to fixed log files for Dataset-Collector abnormal filtering
    log_path = str(Path(cfg.output_dir) / "sumo.log")
    err_path = str(Path(cfg.output_dir) / "sumo_error.log")
    sumo_cmd.extend(["--log", log_path, "--error-log", err_path])
    sumo_cmd.extend(["--time-to-teleport", str(cfg.time_to_teleport)])
    sumo_cmd.extend(["--time-to-teleport.remove", str(cfg.time_to_teleport_remove).lower()])
    if cfg.time_to_teleport_highways is not None:
        sumo_cmd.extend(["--time-to-teleport.highways", str(cfg.time_to_teleport_highways)])
    if cfg.gui:
        write_gui_settings(cfg)
        sumo_cmd.extend(["--gui-settings-file", cfg.gui_settings_file])

    print(f"[Sim]     Launching {binary} (time_scale={cfg.time_scale:.3g}) …")
    traci.start(sumo_cmd)
    random.seed(cfg.seed)

    if cfg.gui:
        # Activate custom scheme as early as possible (before first step)
        try:
            view_id = getattr(traci.gui, "DEFAULT_VIEW", "View #0")
            traci.gui.setSchema(view_id, "custom_workzone")
            print(f"[Sim]     Activated GUI scheme 'custom_workzone'")
        except Exception as e:
            print(f"[Sim]     Could not set GUI scheme: {e}")
        # Step once to ensure GUI is initialized, then apply lane colors as fallback
        traci.simulationStep()
        _override_closed_lane_colors(cfg, traci)

    step = 0
    already_reassigned: set[str] = set()
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            wall_start = time.time()

            traci.simulationStep()
            step += 1

            if cfg.reassign_arrival_lane_in_taper:
                transition_edges = {"e_taper_r"}
                for veh_id in traci.vehicle.getIDList():
                    if veh_id in already_reassigned:
                        continue
                    try:
                        edge = traci.vehicle.getRoadID(veh_id)
                        if edge in transition_edges:
                            target_lane = random.randint(0, cfg.num_lanes - 1)
                            traci.vehicle.changeLane(veh_id, target_lane, 999.0)
                            already_reassigned.add(veh_id)
                    except traci.exceptions.TraCIException:
                        pass

            if cfg.time_scale > 0:
                if cfg.time_scale >= 1.0:
                    target_real_dt = cfg.step_length * cfg.time_scale
                    elapsed = time.time() - wall_start
                    sleep_s = target_real_dt - elapsed
                    if sleep_s > 0:
                        time.sleep(sleep_s)

            if step % 1000 == 0:
                t = traci.simulation.getTime()
                n = traci.vehicle.getIDCount()
                print(f"[Sim]     step={step}  time={t:.1f}s  vehicles={n}")

            # Stop if we've exceeded the configured duration
            if traci.simulation.getTime() >= cfg.sim_duration:
                break
    finally:
        traci.close()

    print(f"[Sim]     Simulation complete — {step} steps executed.")
    print(f"[Sim]     FCD output → {cfg.fcd_file}")


def _parse_color_to_rgba255(color_str: str) -> tuple:
    """Parse SUMO color string 'r,g,b,a' (0-1 or 0-255) → tuple of ints (0-255)."""
    parts = [float(p.strip()) for p in color_str.split(",")]
    # If all values <= 1.0, interpret as 0-1 float range
    if all(v <= 1.0 for v in parts):
        parts = [int(v * 255) for v in parts]
    else:
        parts = [int(v) for v in parts]
    while len(parts) < 4:
        parts.append(255)
    return tuple(parts[:4])


def _override_closed_lane_colors(cfg: WorkzoneConfig, traci_module) -> None:
    """Force ALL lanes to display as solid black (no gray, no transparency)."""
    ref_color = _parse_color_to_rgba255(cfg.closed_lane_color)

    # Method 1: Try traci.lane.setColor on all lanes
    if hasattr(traci_module.lane, "setColor"):
        lane_ids = traci_module.lane.getIDList()
        count = 0
        for lane_id in lane_ids:
            try:
                traci_module.lane.setColor(lane_id, ref_color)
                count += 1
            except Exception:
                pass
        if count > 0:
            print(f"[Sim]     Set color of {count} lanes to {ref_color} (opaque)")

    # Method 2: Try setting the GUI scheme to our custom one
    try:
        view_id = getattr(traci_module.gui, "DEFAULT_VIEW", "View #0")
        traci_module.gui.setSchema(view_id, "custom_workzone")
        print(f"[Sim]     Activated GUI scheme 'custom_workzone'")
    except Exception as e:
        print(f"[Sim]     Could not set GUI scheme: {e}")
