# -*- coding: utf-8 -*-
"""
Enhanced Anomaly Injection Module
Supports all 5 anomaly types with configurable parameters
"""
import copy
import numpy as np
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def inject_all_anomalies(net, measurements, cim_devices=None, svg_devices=None,
                         switches=None, seed=42):
    """
    Inject all 5 types of anomalies into the network.
    
    Returns:
        dict with keys: net, measurements, cim_devices, svg_devices, switches,
                        ground_truth, injection_details
    """
    rng = np.random.default_rng(seed)
    ground_truth = []
    details = []
    
    net_out = copy.deepcopy(net)
    meas_out = copy.deepcopy(measurements)
    cim_out = copy.deepcopy(cim_devices) if cim_devices else []
    svg_out = copy.deepcopy(svg_devices) if svg_devices else []
    sw_out = copy.deepcopy(switches) if switches else []
    
    # ---- Type 1: Model mismatch (CIM/SVG device count difference) ----
    if cim_out and svg_out:
        # Remove 1-2 SVG devices
        n_remove = min(2, max(1, len(svg_out) // 10))
        removed = []
        for _ in range(n_remove):
            if svg_out:
                removed.append(svg_out.pop())
        if removed:
            ground_truth.append({"type": "图模不符", "location": "svg_devices"})
            details.append(f"Removed {len(removed)} SVG devices")
    
    # ---- Type 2: Topology interruption (line outage) ----
    # Prefer bridge lines (whose removal disconnects the graph)
    if len(net_out.line) > 2:
        import networkx as nx
        from utils.graph_utils import build_graph_from_pandapower
        bridge_lines = []
        for idx in net_out.line.index:
            if not bool(net_out.line.at[idx, "in_service"]):
                continue
            net_test = copy.deepcopy(net_out)
            net_test.line.at[idx, "in_service"] = False
            try:
                G_test = build_graph_from_pandapower(net_test)
                if G_test.number_of_nodes() > 0 and not nx.is_connected(G_test):
                    bridge_lines.append(idx)
            except:
                pass
        if bridge_lines:
            line_idx = int(rng.choice(bridge_lines))
        else:
            # Fallback: pick any in-service line
            candidates = [i for i in net_out.line.index 
                          if bool(net_out.line.at[i, "in_service"])]
            line_idx = int(rng.choice(candidates)) if candidates else int(net_out.line.index[0])
        net_out.line.at[line_idx, "in_service"] = False
        ground_truth.append({"type": "拓扑中断", "location": f"line_{line_idx}"})
        details.append(f"Line {line_idx} set out of service (bridge={line_idx in bridge_lines})")
    
    # ---- Type 3: Virtual/faulty connection (modify line endpoint) ----
    if len(net_out.line) > 5:
        candidates = [i for i in net_out.line.index if i in net_out.line.index]
        if candidates:
            line_idx = int(rng.choice(candidates))
            all_buses = list(net_out.bus.index)
            old_to = int(net_out.line.at[line_idx, "to_bus"])
            new_buses = [b for b in all_buses if b != old_to 
                         and b != int(net_out.line.at[line_idx, "from_bus"])]
            if new_buses:
                new_to = int(rng.choice(new_buses))
                net_out.line.at[line_idx, "to_bus"] = new_to
                ground_truth.append({"type": "虚拟接/错接", "location": f"line_{line_idx}"})
                details.append(f"Line {line_idx} endpoint {old_to}->{new_to}")
    
    # ---- Type 4: Telemetry-topology mismatch (large voltage offset) ----
    bus_voltages = meas_out.get("bus_voltages", [])
    if len(bus_voltages) > 3:
        # Pick 1-2 buses and apply large offset
        n_inject = min(2, len(bus_voltages) // 5 + 1)
        injected_indices = rng.choice(len(bus_voltages), size=n_inject, replace=False)
        for idx in injected_indices:
            old_v = bus_voltages[idx]["vm_pu"]
            # Shift to below 0.90 threshold
            bus_voltages[idx]["vm_pu"] = 0.70 + rng.uniform(-0.05, 0.05)
            ground_truth.append({"type": "遥测!=拓扑", "location": f"bus_{idx}"})
            details.append(f"Bus {idx} voltage {old_v:.3f}->{bus_voltages[idx]['vm_pu']:.3f}")
    
    # ---- Type 5: Switch state mismatch ----
    if len(net_out.switch) > 0:
        sw_idx = int(rng.choice(net_out.switch.index))
        current = bool(net_out.switch.at[sw_idx, "closed"])
        net_out.switch.at[sw_idx, "closed"] = not current
        ground_truth.append({"type": "遥信!=遥测", "location": f"switch_{sw_idx}"})
        details.append(f"Switch {sw_idx} closed {current}->{not current}")
    
    logger.info(f"Injected {len(ground_truth)} anomalies: {details}")
    
    return {
        "net": net_out,
        "measurements": meas_out,
        "cim_devices": cim_out,
        "svg_devices": svg_out,
        "switches": sw_out,
        "ground_truth": ground_truth,
        "injection_details": details,
    }

# ===== P1-3: SMOTE + Synthetic Fault Data Augmentation =====

def inject_hif_fault(net, measurements, rng=None):
    """Inject High Impedance Fault: subtle voltage sag + reactive power increase.
    Returns modified (net, measurements, ground_truth_entry).
    """
    if rng is None:
        rng = np.random.default_rng()
    net_out = copy.deepcopy(net)
    meas_out = copy.deepcopy(measurements)

    if len(net_out.line) < 2:
        return net_out, meas_out, None

    # Pick a random line
    line_idx = int(rng.choice(net_out.line.index))
    fb = int(net_out.line.at[line_idx, "from_bus"])
    tb = int(net_out.line.at[line_idx, "to_bus"])

    # Modify bus voltage (subtle sag 0.88-0.95)
    for bv in meas_out.get("bus_voltages", []):
        if bv["bus"] in (fb, tb):
            old_v = bv["vm_pu"]
            bv["vm_pu"] = 0.88 + rng.uniform(0, 0.07)
            break

    # Increase reactive power on the line
    for lp in meas_out.get("line_powers", []):
        if int(lp.get("line", -1)) == line_idx:
            lp["q_mvar"] = abs(lp.get("q_mvar", 0.0)) * (1.5 + rng.uniform(0, 0.5))
            lp["i_ka"] = lp.get("i_ka", 0.05) * (1.1 + rng.uniform(0, 0.3))
            break

    return net_out, meas_out, {"type": "hif_fault", "location": f"line_{line_idx}"}


def inject_line_break(net, measurements, rng=None):
    """Inject line break: near-zero current but voltage difference present.
    Returns modified (net, measurements, ground_truth_entry).
    """
    if rng is None:
        rng = np.random.default_rng()
    net_out = copy.deepcopy(net)
    meas_out = copy.deepcopy(measurements)

    if len(net_out.line) < 3:
        return net_out, meas_out, None

    # Pick a non-bridge line (don't disconnect the network)
    candidates = [i for i in net_out.line.index
                  if bool(net_out.line.at[i, "in_service"])]
    if not candidates:
        return net_out, meas_out, None

    line_idx = int(rng.choice(candidates))

    # Set current near zero
    for lp in meas_out.get("line_powers", []):
        if int(lp.get("line", -1)) == line_idx:
            lp["p_mw"] = rng.uniform(-0.001, 0.001)
            lp["q_mvar"] = rng.uniform(-0.001, 0.001)
            lp["i_ka"] = rng.uniform(0.00001, 0.0005)
            break

    # Create voltage difference (one end normal, other end low)
    fb = int(net_out.line.at[line_idx, "from_bus"])
    tb = int(net_out.line.at[line_idx, "to_bus"])
    for bv in meas_out.get("bus_voltages", []):
        if bv["bus"] == tb:
            bv["vm_pu"] = 0.70 + rng.uniform(0, 0.10)  # significant drop
            break

    return net_out, meas_out, {"type": "line_break", "location": f"line_{line_idx}"}


def smote_augment_samples(samples, target_per_class=None, rng=None):
    """SMOTE-style augmentation for graph training samples.
    
    For each class with fewer than target_per_class samples,
    synthesize new samples by interpolating node features.
    
    Args:
        samples: list of dicts with node_features, edge_index, labels
        target_per_class: target samples per class (default: max class count)
        rng: random generator
    Returns:
        augmented samples list
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Group by majority label
    from collections import defaultdict
    class_groups = defaultdict(list)
    for s in samples:
        labels = s.get("labels", [])
        if len(labels) == 0:
            continue
        majority = max(set(labels), key=list(labels).count)
        class_groups[majority].append(s)

    if not class_groups:
        return samples

    max_count = max(len(v) for v in class_groups.values())
    if target_per_class is None:
        target_per_class = max_count

    augmented = list(samples)
    for cls, cls_samples in class_groups.items():
        n_needed = target_per_class - len(cls_samples)
        if n_needed <= 0:
            continue
        for _ in range(n_needed):
            # Pick two random samples from this class
            idx1, idx2 = rng.choice(len(cls_samples), size=2, replace=True)
            s1, s2 = cls_samples[idx1], cls_samples[idx2]
            # Interpolate node features
            alpha = rng.uniform(0.2, 0.8)
            nf1 = np.array(s1["node_features"])
            nf2 = np.array(s2["node_features"])
            min_n = min(len(nf1), len(nf2))
            new_nf = (alpha * nf1[:min_n] + (1 - alpha) * nf2[:min_n]).tolist()
            # Use shorter sample's edge_index and labels
            new_sample = {
                "node_features": new_nf,
                "edge_index": s1["edge_index"],
                "labels": s1["labels"][:min_n] if len(s1["labels"]) >= min_n else s2["labels"][:min_n],
            }
            augmented.append(new_sample)

    rng.shuffle(augmented)
    logger.info(f"SMOTE augmentation: {len(samples)} -> {len(augmented)} samples")
    return augmented





# ============================================================
# v16新增: 3种调度实际异常类型注入方法 (types 26-28)
# ============================================================

def inject_bus_section_mismatch(net, measurements, rng=None):
    """注入母线分段开关状态与拓扑不一致异常。
    
    模拟: 母线分段开关实际打开但拓扑显示闭合(或反之)，
    导致拓扑模型与实际运行方式不一致。
    """
    if rng is None:
        rng = np.random.default_rng()
    net_out = copy.deepcopy(net)
    meas_out = copy.deepcopy(measurements) if isinstance(measurements, dict) else {}
    
    # Find buses with multiple connections (section buses)
    bus_degrees = {}
    for idx in net_out.line.index:
        fb = int(net_out.line.at[idx, "from_bus"])
        tb = int(net_out.line.at[idx, "to_bus"])
        bus_degrees[fb] = bus_degrees.get(fb, 0) + 1
        bus_degrees[tb] = bus_degrees.get(tb, 0) + 1
    
    # Pick a high-degree bus as section bus
    section_buses = [b for b, d in bus_degrees.items() if d >= 3]
    if not section_buses:
        section_buses = list(bus_degrees.keys())[:3]
    
    target_bus = int(rng.choice(section_buses)) if section_buses else 0
    
    # Flip switch state for lines connected to this bus
    for idx in net_out.line.index:
        fb = int(net_out.line.at[idx, "from_bus"])
        tb = int(net_out.line.at[idx, "to_bus"])
        if fb == target_bus or tb == target_bus:
            # Toggle in_service to simulate mismatch
            net_out.line.at[idx, "in_service"] = not bool(net_out.line.at[idx, "in_service"])
            break
    
    return net_out, meas_out, {"type": "bus_section_mismatch", "location": f"bus_{target_bus}"}


def inject_bypass_operation(net, measurements, rng=None):
    """注入旁路代路操作后拓扑未更新异常。
    
    模拟: 某线路通过旁路供电，但拓扑模型仍显示原线路运行，
    导致拓扑与实际运行方式不一致。旁路线路阻抗极低。
    """
    if rng is None:
        rng = np.random.default_rng()
    net_out = copy.deepcopy(net)
    meas_out = copy.deepcopy(measurements) if isinstance(measurements, dict) else {}
    
    # Pick a line to bypass
    in_service_lines = [i for i in net_out.line.index if bool(net_out.line.at[i, "in_service"])]
    if not in_service_lines:
        return net_out, meas_out, None
    
    line_idx = int(rng.choice(in_service_lines))
    
    # Simulate bypass: reduce impedance to 5% of original (bypass is very low impedance)
    if "r_ohm_per_km" in net_out.line.columns:
        net_out.line.at[line_idx, "r_ohm_per_km"] *= 0.05
    if "x_ohm_per_km" in net_out.line.columns:
        net_out.line.at[line_idx, "x_ohm_per_km"] *= 0.05
    
    return net_out, meas_out, {"type": "bypass_operation", "location": f"line_{line_idx}"}


def inject_load_transfer_residual(net, measurements, rng=None):
    """注入负荷转供后拓扑残留异常。
    
    模拟: 负荷已从一条馈线转供到另一条，但拓扑模型中仍保留
    原馈线的连接关系，导致拓扑冗余/矛盾。
    """
    if rng is None:
        rng = np.random.default_rng()
    net_out = copy.deepcopy(net)
    meas_out = copy.deepcopy(measurements) if isinstance(measurements, dict) else {}
    
    # Find two feeders (groups of connected components)
    lines = list(net_out.line.index)
    if len(lines) < 4:
        return net_out, meas_out, None
    
    # Pick a line and create a "residual" connection
    line_idx = int(rng.choice(lines))
    fb = int(net_out.line.at[line_idx, "from_bus"])
    tb = int(net_out.line.at[line_idx, "to_bus"])
    
    # Simulate residual: mark line as out of service but keep measurements
    net_out.line.at[line_idx, "in_service"] = False
    
    # Keep old measurements (residual)
    if isinstance(meas_out, dict) and "line_powers" in meas_out:
        # Don't remove the line from measurements - this is the residual
        pass
    
    # Add a phantom connection to another bus
    all_buses = list(net_out.bus.index)
    other_buses = [b for b in all_buses if b != fb and b != tb]
    if other_buses:
        phantom_bus = int(rng.choice(other_buses))
        # Modify measurement to show power flowing to phantom bus
        if isinstance(meas_out, dict) and "bus_voltages" in meas_out:
            for bv in meas_out["bus_voltages"]:
                if int(bv.get("bus", -1)) == phantom_bus:
                    bv["v_pu"] = float(bv.get("v_pu", 1.0)) * 0.95  # slight voltage drop
                    break
    
    return net_out, meas_out, {"type": "load_transfer_residual", "location": f"line_{line_idx}_to_bus_{phantom_bus if other_buses else '?'}"}


def inject_all_anomalies_v2(net, measurements, cim_devices=None, svg_devices=None,
                            switches=None, seed=42, include_new_types=True):
    """Enhanced anomaly injection: original 5 types + HIF + line break.
    
    Args:
        include_new_types: if True, also inject HIF and line break faults
    """
    rng = np.random.default_rng(seed)
    
    # Run original 5 types
    result = inject_all_anomalies(net, measurements, cim_devices, svg_devices, switches, seed)
    
    if not include_new_types:
        return result
    
    net_out = result["net"]
    meas_out = result["measurements"]
    ground_truth = result["ground_truth"]
    details = result["injection_details"]
    
    # Type 6: HIF
    net_out, meas_out, gt_hif = inject_hif_fault(net_out, meas_out, rng)
    if gt_hif:
        ground_truth.append(gt_hif)
        details.append(f"HIF on {gt_hif['location']}")
    
    # Type 7: Line break
    net_out, meas_out, gt_lb = inject_line_break(net_out, meas_out, rng)
    if gt_lb:
        ground_truth.append(gt_lb)
        details.append(f"Line break on {gt_lb['location']}")
    
    result["net"] = net_out
    result["measurements"] = meas_out
    result["ground_truth"] = ground_truth
    result["injection_details"] = details
    
    logger.info(f"Enhanced injection: {len(ground_truth)} anomalies (incl. HIF + line break)")
    return result
