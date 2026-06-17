# -*- coding: utf-8 -*-
"""
业务逻辑层
封装异常检测、修正方案生成、数据上传等核心流程
所有对底层模块的调用集中在此层，路由层不直接操作底层模块
"""
from __future__ import annotations
import copy
import logging
import importlib
from typing import Dict, List, Optional, Any

import pandapower as pp
import networkx as nx

from config import THRESHOLDS, ANOMALY_TYPES, PANDAPOWER_NETWORKS
from data_preprocessing.cim_parser import parse_cim_rdf, parse_cim_directory
from data_preprocessing.svg_parser import parse_svg
from data_preprocessing.scada_simulator import SCADASimulator, load_pandapower_network
from anomaly_detection.detector import AnomalyDetector
from correction_engine.corrector import generate_corrections
from utils.graph_utils import build_graph_from_pandapower, build_graph_from_cim, find_sources

logger = logging.getLogger(__name__)

AVAILABLE_NETWORKS: Dict[str, bool] = {}


def _check_network_available(name: str) -> bool:
    """检查 PandaPower 网络是否可加载"""
    if name in AVAILABLE_NETWORKS:
        return AVAILABLE_NETWORKS[name]
    try:
        load_pandapower_network(name)
        AVAILABLE_NETWORKS[name] = True
        return True
    except Exception:
        AVAILABLE_NETWORKS[name] = False
        return False


def _build_synthetic_cim_svg(net) -> tuple:
    """
    从 PandaPower 网络构建合成 CIM / SVG / 开关 数据（测试用）。
    与 run_mvp.py 中 build_synthetic_cim_svg 逻辑一致。
    Returns: (cim_devices, svg_devices, switches)
    """
    cim_devices, svg_devices, switches = [], [], []

    for idx in net.bus.index:
        name, uri = f"Bus_{idx}", f"#_bus_{idx}"
        cim_devices.append({"uri": uri, "name": name,
                            "type": "ConductingEquipment", "subtype": "BusbarSection"})
        svg_devices.append({"id": name, "type": "Bus",
                            "x": float(idx * 100), "y": 300.0, "label": name})

    for idx in net.line.index:
        name, uri = f"Line_{idx}", f"#_line_{idx}"
        cim_devices.append({"uri": uri, "name": name,
                            "type": "ConductingEquipment", "subtype": "ACLineSegment"})
        svg_devices.append({"id": name, "type": "ACLineSegment",
                            "x": float(idx * 100 + 50), "y": 200.0, "label": name})

    for idx in net.trafo.index:
        name, uri = f"Trafo_{idx}", f"#_trafo_{idx}"
        cim_devices.append({"uri": uri, "name": name,
                            "type": "ConductingEquipment", "subtype": "PowerTransformer"})
        svg_devices.append({"id": name, "type": "Transformer",
                            "x": float(idx * 100 + 50), "y": 400.0, "label": name})

    if hasattr(net, "switch") and len(net.switch) > 0:
        for idx in net.switch.index:
            name, uri = f"Switch_{idx}", f"#_switch_{idx}"
            closed = bool(net.switch.at[idx, "closed"])
            cim_devices.append({"uri": uri, "name": name,
                                "type": "ConductingEquipment", "subtype": "Breaker"})
            switches.append({"uri": uri, "name": name, "subtype": "Breaker",
                             "normal_open": not closed, "open_pos": not closed})
            svg_devices.append({"id": name, "type": "Breaker",
                                "x": float(idx * 100), "y": 250.0, "label": name})
    return cim_devices, svg_devices, switches


def run_detect(
    network_name: str = "example_simple",
    use_rule_engine: bool = True,
    use_state_estimator: bool = True,
    inject_anomalies: bool = False,
    anomaly_count: int = 3,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    执行一次完整的异常检测流程。
    流程: 加载网络 -> 构建CIM/SVG -> SCADA仿真 -> [注入异常] -> 拓扑图 -> 检测
    """
    logger.info(f"[detect] network={network_name}, rule={use_rule_engine}, SE={use_state_estimator}")

    # 1. 加载网络
    net = load_pandapower_network(network_name)
    # 2. 构建 CIM/SVG
    cim_devices, svg_devices, switches = _build_synthetic_cim_svg(net)
    # 3. SCADA 仿真
    simulator = SCADASimulator(net, seed=random_seed)
    measurements = simulator.generate_measurements()

    inj_net = copy.deepcopy(net)
    inj_meas = copy.deepcopy(measurements)
    inj_svg = copy.deepcopy(svg_devices)
    inj_sw = copy.deepcopy(switches)

    # 4. [可选] 注入异常
    if inject_anomalies:
        logger.info(f"[detect] injecting {anomaly_count} anomalies")
        inj_net, inj_meas, inj_svg, inj_sw = _inject_anomalies(
            simulator, net, measurements, svg_devices, switches,
            anomaly_count, random_seed)

    # 5. 构建 NetworkX 拓扑图
    graph = build_graph_from_pandapower(inj_net)

    # 6. 运行检测器
    network_data = {
        "graph": graph, "cim_devices": cim_devices,
        "svg_devices": inj_svg, "measurements": inj_meas,
        "switches": inj_sw, "net": inj_net, "scada_data": inj_meas,
    }
    detector = AnomalyDetector(use_rule_engine=use_rule_engine,
                               use_state_estimator=use_state_estimator)
    detected = detector.detect_all(network_data)
    summary = detector.get_summary()
    logger.info(f"[detect] done: {len(detected)} anomalies")

    return {"success": True, "anomaly_count": len(detected),
            "anomalies": detected, "summary": summary}


def run_correct(
    network_name: str = "example_simple",
    use_rule_engine: bool = True,
    use_state_estimator: bool = True,
    inject_anomalies: bool = True,
    anomaly_count: int = 3,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """完整检测 + 修正流程"""
    detection = run_detect(
        network_name=network_name, use_rule_engine=use_rule_engine,
        use_state_estimator=use_state_estimator,
        inject_anomalies=inject_anomalies,
        anomaly_count=anomaly_count, random_seed=random_seed)

    graph = build_graph_from_pandapower(load_pandapower_network(network_name))
    network_data = {"graph": graph}
    corr_result = generate_corrections(detection["anomalies"], network_data)
    logger.info(f"[correct] {corr_result['summary']['total']} corrections")

    return {"success": True, "detection": detection,
            "correction_count": corr_result["summary"]["total"],
            "corrections": corr_result["corrections"],
            "correction_summary": corr_result["summary"]}


def upload_data(cim_path: Optional[str] = None,
                svg_path: Optional[str] = None) -> Dict[str, Any]:
    """上传并解析 CIM/SVG 数据文件"""
    import os
    result = {"success": True, "message": "",
              "cim_devices": 0, "svg_devices": 0, "switches": 0, "nodes": 0}
    parts = []

    if cim_path:
        if os.path.isdir(cim_path):
            cim_data = parse_cim_directory(cim_path)
        else:
            cim_data = parse_cim_rdf(cim_path)
        result["cim_devices"] = len(cim_data.get("devices", []))
        result["switches"] = len(cim_data.get("switches", []))
        result["nodes"] = len(cim_data.get("nodes", []))
        parts.append(f"CIM: {result['cim_devices']} devices, "
                     f"{result['switches']} switches, {result['nodes']} nodes")

    if svg_path:
        svg_data = parse_svg(svg_path)
        result["svg_devices"] = (len(svg_data.get("devices", []))
                                 + len(svg_data.get("switches", [])))
        parts.append(f"SVG: {result['svg_devices']} elements")

    result["message"] = "Upload OK. " + "; ".join(parts) if parts else "No file uploaded."
    return result


def list_available_networks() -> Dict[str, str]:
    """返回所有可用的 PandaPower 测试网络"""
    available = {}
    for name in PANDAPOWER_NETWORKS:
        try:
            load_pandapower_network(name)
            available[name] = PANDAPOWER_NETWORKS[name]
        except Exception:
            pass
    return available


def get_health() -> Dict[str, Any]:
    """服务健康检查: 尝试导入各模块并返回可用性"""
    modules = {}
    checks = [
        ("anomaly_detection", "anomaly_detection.detector"),
        ("correction_engine", "correction_engine.corrector"),
        ("cim_parser", "data_preprocessing.cim_parser"),
        ("svg_parser", "data_preprocessing.svg_parser"),
        ("scada_simulator", "data_preprocessing.scada_simulator"),
        ("graph_utils", "utils.graph_utils"),
        ("metrics", "utils.metrics"),
        ("pandapower", "pandapower"),
    ]
    for label, module_path in checks:
        try:
            importlib.import_module(module_path)
            modules[label] = True
        except Exception:
            modules[label] = False

    all_ok = all(modules.values())
    return {"status": "ok" if all_ok else "degraded",
            "version": "1.0.0", "modules": modules}


def _inject_anomalies(simulator, net, measurements, svg_devices,
                      switches, count, seed) -> tuple:
    """注入合成异常（与 run_mvp.py 逻辑一致）"""
    import numpy as np
    rng = np.random.default_rng(seed)
    inj_net = copy.deepcopy(net)
    inj_meas = copy.deepcopy(measurements)
    inj_svg = copy.deepcopy(svg_devices)
    inj_sw = copy.deepcopy(switches)

    active_lines = [i for i in inj_net.line.index if bool(inj_net.line.at[i, "in_service"])]
    if active_lines:
        line_idx = int(rng.choice(active_lines))
        inj_net.line.at[line_idx, "in_service"] = False
        logger.info(f"[inject] topo interrupt: line {line_idx}")

    bus_voltages = inj_meas.get("bus_voltages", [])
    if bus_voltages:
        idx = int(rng.integers(0, len(bus_voltages)))
        bus_voltages[idx]["vm_pu"] += 0.15
        logger.info(f"[inject] measurement error: bus {bus_voltages[idx]['bus']}")

    if inj_svg and len(inj_svg) > 2:
        remove_idx = int(rng.integers(0, len(inj_svg)))
        removed = inj_svg.pop(remove_idx)
        logger.info(f"[inject] model mismatch: SVG remove {removed.get('id', '?')}")

    if inj_sw:
        sw_idx = int(rng.integers(0, len(inj_sw)))
        inj_sw[sw_idx]["normal_open"] = not inj_sw[sw_idx].get("normal_open", False)
        logger.info(f"[inject] signal mismatch: switch {inj_sw[sw_idx].get('name', '?')}")

    return inj_net, inj_meas, inj_svg, inj_sw
