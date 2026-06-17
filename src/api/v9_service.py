# -*- coding: utf-8 -*-
"""
v9 Detection Service - bridges benchmark_v9_expanded.py with FastAPI
Provides 25-type anomaly detection pipeline for the API layer.
"""
import copy
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import networkx as nx
import pandapower as pp

logger = logging.getLogger(__name__)

# Ensure project root in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_preprocessing.scada_simulator import SCADASimulator, load_pandapower_network
from data_preprocessing.anomaly_injector import inject_all_anomalies
from utils.graph_utils import build_graph_from_pandapower
from anomaly_detection.rule_engine import run_rule_engine
from anomaly_detection.topology_cleanup import cleanup_topology
from anomaly_detection.bad_data_detection import classify_bad_data, apply_bad_data_plan
from correction_engine.n1_security import n1_contingency_check
from correction_engine.operation_ticket import generate_operation_ticket, format_ticket_text

# Import v9 detection functions from benchmark
_benchmark_path = PROJECT_ROOT / "tests"
if str(_benchmark_path) not in sys.path:
    sys.path.insert(0, str(_benchmark_path))

# Lazy-loaded benchmark module
_bench = None

def _get_bench():
    """Lazy-load benchmark module with all v9 detection functions."""
    global _bench
    if _bench is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "benchmark_v9_expanded",
            str(PROJECT_ROOT / "tests" / "benchmark_v9_expanded.py")
        )
        _bench = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(_bench)
            logger.info("v9 benchmark module loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load v9 benchmark: {e}")
            raise
    return _bench


# 25 anomaly type list (matching benchmark)
ANOMALY_TYPE_LIST = [
    "topo_interrupt", "virtual_faulty", "model_mismatch",
    "telemetry_mismatch", "signal_mismatch",
    "measurement_outlier", "stale_data", "parameter_error",
    "load_shift", "reverse_power_flow", "communication_loss",
    "voltage_collapse", "ghost_topology", "duplicate_measurement",
    "protection_misconfig",
    "trafo_tap_fault", "grounding_fault", "clock_drift",
    "harmonic_pollution", "impedance_degradation", "dg_intermittent",
    "measurement_bias", "branch_contingency", "topo_obfuscation",
    "voltage_regulation",
]

# Type name mapping for display
TYPE_DISPLAY = {
    "topo_interrupt": "拓扑中断", "virtual_faulty": "虚接错接",
    "model_mismatch": "图模不符", "telemetry_mismatch": "遥测矛盾",
    "signal_mismatch": "遥信矛盾", "measurement_outlier": "量测异常",
    "stale_data": "陈旧数据", "parameter_error": "参数错误",
    "load_shift": "负荷突变", "reverse_power_flow": "反向潮流",
    "communication_loss": "通信中断", "voltage_collapse": "电压崩溃",
    "ghost_topology": "幽灵拓扑", "duplicate_measurement": "重复量测",
    "protection_misconfig": "保护误配", "trafo_tap_fault": "分接头故障",
    "grounding_fault": "接地故障", "clock_drift": "时钟漂移",
    "harmonic_pollution": "谐波污染", "impedance_degradation": "阻抗退化",
    "dg_intermittent": "DG间歇", "measurement_bias": "量测偏差",
    "branch_contingency": "支路停运", "topo_obfuscation": "拓扑混淆",
    "voltage_regulation": "电压调节",
}


def run_v9_detect(
    network_name: str = "case33bw",
    inject_anomalies: bool = True,
    anomaly_count: int = 3,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Run the full v9 28-type detection pipeline on a PandaPower network.
    Returns detection results in API-compatible format.
    """
    bench = _get_bench()
    rng = np.random.RandomState(random_seed)

    # 1. Load network
    net = load_pandapower_network(network_name)
    net_orig = copy.deepcopy(net)

    # 2. Build graph
    try:
        G = build_graph_from_pandapower(net)
    except Exception:
        G = nx.Graph()
        for idx in net.bus.index:
            G.add_node(int(idx))
        for idx in net.line.index:
            fb = int(net.line.at[idx, "from_bus"])
            tb = int(net.line.at[idx, "to_bus"])
            G.add_edge(fb, tb)

    # 3. Generate SCADA measurements
    sim = SCADASimulator(net, seed=random_seed)
    try:
        pp.runpp(net, numba=False)
    except Exception:
        pass
    meas = sim.generate_measurements()

    # 4. Inject anomalies if requested
    gt = []
    net_inj = copy.deepcopy(net)
    if inject_anomalies:
        try:
            injected = inject_all_anomalies(net_inj, n_anomalies=anomaly_count, seed=random_seed)
            if isinstance(injected, list):
                gt = injected
            elif isinstance(injected, dict) and "anomalies" in injected:
                gt = injected["anomalies"]
            else:
                gt = [{"type": "unknown", "location": "auto"}] * anomaly_count
        except Exception as e:
            logger.debug(f"Anomaly injection: {e}")
            gt = [{"type": "injected", "location": "auto"}] * anomaly_count
        # Re-generate measurements after injection
        try:
            pp.runpp(net_inj, numba=False)
        except Exception:
            pass
        sim = SCADASimulator(net_inj, seed=random_seed)
        meas = sim.generate_measurements()

    # 4.5 Topology cleanup (5-step pipeline)
    try:
        net_inj = cleanup_topology(net_inj)
        # Rebuild graph after cleanup
        G = build_graph_from_pandapower(net_inj)
    except Exception as e:
        logger.debug(f"Topology cleanup: {e}")

    # 4.6 Bad data detection (two-pass + paired immunity)
    try:
        bd_plan = classify_bad_data(meas, threshold=3.0)
        if bd_plan.get("bad_indices"):
            meas = apply_bad_data_plan(meas, bd_plan)
            logger.debug(f"Bad data removed: {len(bd_plan['bad_indices'])} measurements")
    except Exception as e:
        logger.debug(f"Bad data detection: {e}")



    # 5. Run detection layers
    detected = []

    # Layer 1: Rule engine
    try:
        from data_preprocessing.cim_parser import parse_cim_rdf
        from data_preprocessing.svg_parser import parse_svg
        cim_devices = [{"uri": f"#_bus_{i}", "name": f"Bus_{i}", "type": "ConductingEquipment", "subtype": "BusbarSection"} for i in net_inj.bus.index]
        svg_devices = [{"id": f"Bus_{i}", "type": "Bus", "x": float(i*100), "y": 300.0} for i in net_inj.bus.index]
        rule_out = run_rule_engine(G, cim_devices=cim_devices, svg_devices=svg_devices, measurements=meas)
        for d in rule_out:
            if "layer" not in d:
                d["layer"] = "Rule"
        detected.extend(rule_out[:20])
    except Exception as e:
        logger.debug(f"Rule engine: {e}")

    # Layer 2: State estimation
    try:
        from anomaly_detection.state_estimator import StateEstimator
        se = StateEstimator()
        se_out = se.detect(net_inj, meas)
        for d in se_out:
            d["layer"] = "SE"
        detected.extend(se_out[:10])
    except Exception as e:
        logger.debug(f"SE: {e}")

    # Layer 3: GNN
    try:
        gnn = bench.get_gnn()
        if gnn and G:
            gnn_out = gnn.detect({"graph": G, "measurements": meas})
            if gnn_out:
                detected.extend(gnn_out[:5])
    except Exception as e:
        logger.debug(f"GNN: {e}")

    # Layer 4+5: v8/v9 specialized detectors
    det_funcs = [
        ("OUT", bench.detect_measurement_outlier, "measurement_outlier"),
        ("STALE", bench.detect_stale_data, "stale_data"),
        ("PARAM", bench.detect_parameter_error, "parameter_error"),
        ("LOAD", bench.detect_load_shift, "load_shift"),
        ("RPF", bench.detect_reverse_power_flow, "reverse_power_flow"),
        ("COMM", bench.detect_communication_loss, "communication_loss"),
        ("VC", bench.detect_voltage_collapse, "voltage_collapse"),
        ("GHOST", bench.detect_ghost_topology, "ghost_topology"),
        ("DUP", bench.detect_duplicate_measurement, "duplicate_measurement"),
        ("PROT", bench.detect_protection_misconfig, "protection_misconfig"),
        ("TRAFO_TAP", bench.detect_trafo_tap_fault, "trafo_tap_fault"),
        ("GROUNDING", bench.detect_grounding_fault, "grounding_fault"),
        ("CLOCK_DRIFT", bench.detect_clock_drift, "clock_drift"),
        ("HARMONIC", bench.detect_harmonic_pollution, "harmonic_pollution"),
        ("IMPEDANCE", bench.detect_impedance_degradation, "impedance_degradation"),
        ("DG_INTER", bench.detect_dg_intermittent, "dg_intermittent"),
        ("MEAS_BIAS", bench.detect_measurement_bias, "measurement_bias"),
        ("CONTINGENCY", bench.detect_branch_contingency, "branch_contingency"),
        ("TOPO_OBF", bench.detect_topo_obfuscation, "topo_obfuscation"),
        ("VREG", bench.detect_voltage_regulation, "voltage_regulation"),
    ]

    for layer_name, func, type_name in det_funcs:
        try:
            out = func(net_inj, meas)
            for d in out:
                if "layer" not in d:
                    d["layer"] = layer_name
            detected.extend(out)
        except Exception as e:
            logger.debug(f"{layer_name}: {e}")

    # 6. Apply filters (matching benchmark)
    n_buses = len(net_inj.bus)
    detected = bench.ensemble_vote(detected, min_layers=2)
    detected = bench.smart_filter(detected, n_buses)
    detected = bench.filter_detections(detected, max_per_type=6, n_buses=n_buses)

    # 7. Format results
    anomalies = []
    for d in detected:
        anomalies.append({
            "type": d.get("type", "unknown"),
            "type_display": TYPE_DISPLAY.get(d.get("type", ""), d.get("type", "")),
            "location": str(d.get("location", "")),
            "confidence": round(float(d.get("confidence", 0)), 3),
            "layer": d.get("layer", "unknown"),
            "details": d.get("details", ""),
        })

    return {
        "success": True,
        "network_name": network_name,
        "n_buses": n_buses,
        "n_lines": len(net_inj.line),
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
        "injected_count": len(gt),
        "injected_types": [g["type"] for g in gt],
        "detection_layers": list(set(a["layer"] for a in anomalies)),
        "summary": {
            "total_detected": len(anomalies),
            "per_type": {t: sum(1 for a in anomalies if a["type"] == t) for t in set(a["type"] for a in anomalies)} if anomalies else {},
            "avg_confidence": round(np.mean([a["confidence"] for a in anomalies]), 3) if anomalies else 0,
        }
    }


def get_network_topology(network_name: str = "case33bw") -> Dict[str, Any]:
    """Get network topology data for frontend visualization."""
    net = load_pandapower_network(network_name)
    try:
        pp.runpp(net, numba=False)
    except Exception:
        pass

    buses = []
    for idx in net.bus.index:
        vm = 1.0
        try:
            vm = float(net.res_bus.at[idx, "vm_pu"]) if hasattr(net, "res_bus") and idx in net.res_bus.index else 1.0
        except Exception:
            pass
        buses.append({
            "id": f"bus_{idx}",
            "index": int(idx),
            "name": str(net.bus.at[idx, "name"]) if "name" in net.bus.columns else f"Bus_{idx}",
            "vn_kv": float(net.bus.at[idx, "vn_kv"]),
            "vm_pu": vm,
            "type": "source" if any(idx == int(net.ext_grid.at[g, "bus"]) for g in net.ext_grid.index) else "load" if any(idx == int(net.load.at[l, "bus"]) for l in net.load.index) else "bus",
        })

    links = []
    for idx in net.line.index:
        links.append({
            "id": f"line_{idx}",
            "source": f"bus_{int(net.line.at[idx, 'from_bus'])}",
            "target": f"bus_{int(net.line.at[idx, 'to_bus'])}",
            "in_service": bool(net.line.at[idx, "in_service"]),
            "type": "line",
        })
    for idx in net.trafo.index:
        links.append({
            "id": f"trafo_{idx}",
            "source": f"bus_{int(net.trafo.at[idx, 'hv_bus'])}",
            "target": f"bus_{int(net.trafo.at[idx, 'lv_bus'])}",
            "in_service": bool(net.trafo.at[idx, "in_service"]),
            "type": "trafo",
        })

    return {
        "network_name": network_name,
        "buses": buses,
        "links": links,
        "n_buses": len(buses),
        "n_links": len(links),
    }


def list_networks() -> List[Dict[str, Any]]:
    """List all available PandaPower networks with metadata."""
    networks = [
        ("case4gs", 4, 4, "小型测试"),
        ("case5", 5, 6, "小型测试"),
        ("case6ww", 6, 11, "小型测试"),
        ("case9", 9, 9, "IEEE 9节点"),
        ("case14", 14, 15, "IEEE 14节点"),
        ("case_ieee30", 30, 34, "IEEE 30节点"),
        ("case33bw", 33, 37, "IEEE 33节点配电网"),
        ("case39", 39, 35, "IEEE 39节点"),
        ("case57", 57, 63, "IEEE 57节点"),
        ("case89pegase", 89, 160, "PEGASE 89"),
        ("case118", 118, 173, "IEEE 118节点"),
        ("case300", 300, 283, "IEEE 300节点"),
        ("case1354pegase", 1354, 1751, "PEGASE 1354"),
        ("case1888rte", 1888, 1976, "RTE 1888"),
        ("case2869pegase", 2869, 4051, "PEGASE 2869"),
        ("case3120sp", 3120, 3487, "SP 3120"),
        ("case6470rte", 6470, 7426, "RTE 6470"),
        ("case9241pegase", 9241, 13797, "PEGASE 9241"),
    ]
    return [{"name": n, "buses": b, "lines": l, "desc": d} for n, b, l, d in networks]
