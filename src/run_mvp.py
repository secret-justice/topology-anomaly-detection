# -*- coding: utf-8 -*-
"""
MVP主入口脚本
完整流程: 加载网络 -> 生成SCADA -> 注入异常 -> 规则引擎 -> 状态估计
         -> 合并检测 -> 修正方案 -> 评估 -> 可视化 -> 输出报告
"""
import sys
from pathlib import Path

# 确保项目路径在 sys.path 中
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

import copy
import json
import logging
from datetime import datetime

import pandapower as pp
import pandapower.networks as pn

from config import OUTPUT_ROOT, ANOMALY_TYPES
from data_preprocessing.scada_simulator import SCADASimulator
from utils.graph_utils import build_graph_from_pandapower, find_sources
from utils.metrics import evaluate_detection_results, evaluate_by_type
from anomaly_detection.detector import AnomalyDetector
from correction_engine.corrector import generate_corrections
from visualization.topo_visualizer import save_topology_image

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MVP")


# ============================================================
# 辅助函数
# ============================================================
def create_test_network():
    """
    创建测试网络。优先 IEEE 33 节点(case33bw)，回退到 IEEE 13，再回退到 example_simple。
    """
    for name, loader in [("case33bw", pn.case33bw), ("case_ieee30", pn.case_ieee30), ("example_simple", pn.example_simple)]:
        try:
            net = loader()
            logger.info(f"加载 {name}: 母线={len(net.bus)}, 线路={len(net.line)}, "
                        f"变压器={len(net.trafo)}, 开关={len(net.switch)}")
            return net
        except Exception as e:
            logger.warning(f"{name} 加载失败: {e}")
    raise RuntimeError("无法加载任何测试网络")


def build_synthetic_cim_svg(net):
    """
    从PandaPower网络构建合成 CIM / SVG / 开关 数据（MVP测试用）。
    实际项目中应由 cim_parser / svg_parser 提供。

    Returns:
        (cim_devices, svg_devices, switches)
    """
    cim_devices = []
    svg_devices = []
    switches    = []

    # 母线
    for idx in net.bus.index:
        name = f"Bus_{idx}"
        uri  = f"#_bus_{idx}"
        cim_devices.append({"uri": uri, "name": name,
                            "type": "ConductingEquipment",
                            "subtype": "BusbarSection"})
        svg_devices.append({"id": name, "type": "Bus",
                            "x": float(idx * 100), "y": 300.0,
                            "label": name})

    # 线路
    for idx in net.line.index:
        name = f"Line_{idx}"
        uri  = f"#_line_{idx}"
        cim_devices.append({"uri": uri, "name": name,
                            "type": "ConductingEquipment",
                            "subtype": "ACLineSegment"})
        svg_devices.append({"id": name, "type": "ACLineSegment",
                            "x": float(idx * 100 + 50), "y": 200.0,
                            "label": name})

    # 变压器
    for idx in net.trafo.index:
        name = f"Trafo_{idx}"
        uri  = f"#_trafo_{idx}"
        cim_devices.append({"uri": uri, "name": name,
                            "type": "ConductingEquipment",
                            "subtype": "PowerTransformer"})
        svg_devices.append({"id": name, "type": "Transformer",
                            "x": float(idx * 100 + 50), "y": 400.0,
                            "label": name})

    # 开关
    if hasattr(net, "switch") and len(net.switch) > 0:
        for idx in net.switch.index:
            name   = f"Switch_{idx}"
            uri    = f"#_switch_{idx}"
            closed = bool(net.switch.at[idx, "closed"])
            cim_devices.append({"uri": uri, "name": name,
                                "type": "ConductingEquipment",
                                "subtype": "Breaker"})
            switches.append({"uri": uri, "name": name,
                             "subtype": "Breaker",
                             "normal_open": not closed,
                             "open_pos": not closed})
            svg_devices.append({"id": name, "type": "Breaker",
                                "x": float(idx * 100), "y": 250.0,
                                "label": name})

    return cim_devices, svg_devices, switches


def inject_synthetic_anomalies(net, measurements,
                               cim_devices, svg_devices, switches):
    """
    Inject 5 anomaly types with measurement regeneration after topology changes.
    """
    import pandapower as pp
    gt = []
    inj_net   = copy.deepcopy(net)
    inj_meas  = copy.deepcopy(measurements)
    inj_svg   = copy.deepcopy(svg_devices)
    inj_sw    = copy.deepcopy(switches)
    aid = 0

    # ---- Anomaly 1: Topology interrupt ----
    if len(inj_net.line) > 2:
        tl = int(inj_net.line.index[2])
        old_to = int(inj_net.line.at[tl, "to_bus"])
        inj_net.line.at[tl, "in_service"] = False
        try:
            pp.runpp(inj_net, calculate_voltage_angles=True)
            sim_new = SCADASimulator(inj_net, seed=42)
            inj_meas = sim_new.generate_measurements()
        except Exception as e:
            logger.warning("topo interrupt PF failed: {}".format(e))
        gt.append({"type": ANOMALY_TYPES["TOPO_INTERRUPT"],
                   "location": "bus_{}".format(old_to),
                   "anomaly_id": aid}); aid += 1
        logger.info("inject 1: topo interrupt line {} -> bus_{}".format(tl, old_to))

    # ---- Anomaly 2: Virtual/faulty connection ----
    if len(inj_net.line) > 5:
        fl = int(inj_net.line.index[5])
        old_to_bus = int(inj_net.line.at[fl, "to_bus"])
        all_buses = list(inj_net.bus.index)
        candidates = [b for b in all_buses
                      if b != old_to_bus and b != int(inj_net.line.at[fl, "from_bus"])]
        if candidates:
            new_to_bus = candidates[len(candidates) // 2]
            inj_net.line.at[fl, "to_bus"] = new_to_bus
            try:
                pp.runpp(inj_net, calculate_voltage_angles=True)
                sim_new = SCADASimulator(inj_net, seed=42)
                inj_meas = sim_new.generate_measurements()
            except Exception as e:
                logger.warning("virtual faulty PF failed: {}".format(e))
            gt.append({"type": ANOMALY_TYPES["VIRTUAL_FAULTY"],
                       "location": "line_{}".format(fl),
                       "anomaly_id": aid}); aid += 1
            logger.info("inject 2: virtual faulty line {} bus_{}->bus_{}".format(fl, old_to_bus, new_to_bus))

    # ---- Anomaly 3: Measurement error ----
    if len(inj_meas.get("bus_voltages", [])) > 2:
        bv = inj_meas["bus_voltages"][2]
        old_vm = bv["vm_pu"]
        bv["vm_pu"] = 0.82
        gt.append({"type": ANOMALY_TYPES["TELE_TOPO_MISMATCH"],
                   "location": "bus_{}".format(bv["bus"]),
                   "anomaly_id": aid}); aid += 1
        logger.info("inject 3: meas error bus {} V={:.4f}->0.82".format(bv["bus"], old_vm))

    # ---- Anomaly 4: Model mismatch ----
    if len(inj_svg) > 5:
        removed = inj_svg.pop(5)
        gt.append({"type": ANOMALY_TYPES["MODEL_MISMATCH"],
                   "location": removed["id"],
                   "anomaly_id": aid}); aid += 1
        logger.info("inject 4: model mismatch removed {}".format(removed["id"]))

    # ---- Anomaly 5: Signal mismatch ----
    if hasattr(inj_net, "switch") and len(inj_net.switch) > 0:
        sw_idx = int(inj_net.switch.index[0])
        old_closed = bool(inj_net.switch.at[sw_idx, "closed"])
        inj_net.switch.at[sw_idx, "closed"] = not old_closed
        gt.append({"type": ANOMALY_TYPES["TELE_SIGNAL_MISMATCH"],
                   "location": "switch_{}".format(sw_idx),
                   "anomaly_id": aid}); aid += 1
        logger.info("inject 5: signal mismatch switch {}".format(sw_idx))
    else:
        if len(inj_meas.get("line_powers", [])) > 3:
            lp = inj_meas["line_powers"][3]
            old_p = lp["p_mw"]
            lp["p_mw"] = old_p * 3.0
            gt.append({"type": ANOMALY_TYPES["TELE_SIGNAL_MISMATCH"],
                       "location": "line_{}".format(lp["line"]),
                       "anomaly_id": aid}); aid += 1
            logger.info("inject 5(alt): signal mismatch line {} P*3".format(lp["line"]))

    return inj_net, inj_meas, inj_svg, inj_sw, gt


# ============================================================
# 主流程
# ============================================================
def run_mvp():
    """运行完整 MVP 流程（10步）"""
    print("=" * 70)
    print("  配电网图模拓扑智能识别与修正 — MVP 演示")
    print("=" * 70)

    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {"timestamp": ts, "steps": {}}

    # ---- 步骤1: 创建测试网络 ----
    logger.info("=" * 50)
    logger.info("步骤1/10: 创建测试网络")
    net = create_test_network()
    pp.runpp(net)
    logger.info(f"潮流收敛: {net.converged}")
    report["steps"]["network"] = {
        "buses": len(net.bus), "lines": len(net.line),
        "trafos": len(net.trafo), "converged": net.converged,
    }

    # ---- 步骤2: 生成 SCADA 仿真数据 ----
    logger.info("步骤2/10: 生成SCADA仿真数据")
    simulator   = SCADASimulator(net)
    measurements = simulator.generate_measurements()
    logger.info(f"量测: 电压={len(measurements['bus_voltages'])}, "
                f"线路功率={len(measurements['line_powers'])}, "
                f"变压器功率={len(measurements['trafo_powers'])}")
    report["steps"]["scada"] = {
        "bus_voltages":  len(measurements["bus_voltages"]),
        "line_powers":   len(measurements["line_powers"]),
        "trafo_powers":  len(measurements["trafo_powers"]),
    }

    # ---- 步骤3: 构建 CIM/SVG 并注入异常 ----
    logger.info("步骤3/10: 构建CIM/SVG并注入合成异常")
    cim_devices, svg_devices, switches = build_synthetic_cim_svg(net)
    (inj_net, inj_meas, inj_svg, inj_sw,
     ground_truth) = inject_synthetic_anomalies(
        net, measurements, cim_devices, svg_devices, switches)
    logger.info(f"注入 {len(ground_truth)} 个合成异常")
    report["steps"]["anomalies_injected"] = len(ground_truth)
    report["ground_truth"] = ground_truth

    # ---- 步骤4: 构建 NetworkX 拓扑图 ----
    logger.info("步骤4/10: 构建NetworkX拓扑图")
    graph   = build_graph_from_pandapower(inj_net)
    sources = find_sources(graph)
    logger.info(f"拓扑图: 节点={graph.number_of_nodes()}, "
                f"边={graph.number_of_edges()}, 电源={sources}")

    # ---- 步骤5: 运行规则引擎 + 状态估计 ----
    logger.info("步骤5/10: 运行异常检测 (规则引擎 + 状态估计)")
    network_data = {
        "graph":        graph,
        "cim_devices":  cim_devices,
        "svg_devices":  inj_svg,
        "measurements": inj_meas,
        "switches":     inj_sw,
        "net":          inj_net,
        "scada_data":   inj_meas,
    }
    detector = AnomalyDetector(use_rule_engine=True,
                               use_state_estimator=True)
    detected = detector.detect_all(network_data)
    logger.info(f"检测到 {len(detected)} 个异常")
    report["steps"]["detection"] = detector.get_summary()
    report["detected_anomalies"] = [
        {k: (str(v) if not isinstance(v, (int, float, bool)) else v)
         for k, v in a.items()}
        for a in detected
    ]
    # ---- 步骤5b: 增强检测器（三层集成+设备定位） ----
    try:
        from anomaly_detection.enhanced_detector import EnhancedDetector
        enhanced = EnhancedDetector(enable_rule=True, enable_se=True)
        enhanced_result = enhanced.detect(network_data)
        enhanced_count = enhanced_result["summary"]["total"]
        logger.info(f"增强检测器: {enhanced_count} 个异常 (耗时 {enhanced_result['summary'].get('processing_time_ms', '?')}ms)")
        report["steps"]["enhanced_detection"] = enhanced_result["summary"]
    except Exception as e:
        logger.warning(f"增强检测器跳过: {e}")
        report["steps"]["enhanced_detection"] = {"error": str(e)}


    # ---- 步骤6: 生成修正方案 ----
    logger.info("步骤6/10: 生成修正方案")
    corrections = generate_corrections(detected, network_data)
    logger.info(f"生成 {corrections['summary']['total']} 个修正方案")
    report["steps"]["corrections"] = corrections["summary"]

    # ---- 步骤7: 评估检测效果 ----
    logger.info("步骤7/10: 评估检测效果")
    eval_result = evaluate_detection_results(
        predictions=detected, ground_truth=ground_truth,
        match_field="location")
    # 类型级评估（检查异常类型覆盖）
    type_eval = evaluate_by_type(detected, ground_truth)
    logger.info(f"类型级评估: 覆盖率(Recall)={type_eval['type_recall']:.1%}, "
                f"精确类型数={len(type_eval['detected_types'])}/{len(type_eval['per_type'])}")
    for t, info in type_eval["per_type"].items():
        status = "已检测" if info["detected"] else "未检测"
        logger.info(f"  {t}: {status} (检测到{info['detected_count']}个)")
    
    report["steps"]["evaluation"] = {
        "type_recall": type_eval["type_recall"],
        "type_precision": type_eval["type_precision"],
        "detected_types": type_eval["detected_types"],
        "missed_types": type_eval["missed_types"],
        "extra_types": type_eval["extra_types"],
        "per_type": type_eval["per_type"],
    }

    # ---- 步骤8: 可视化 — 正常拓扑 ----
    logger.info("步骤8/10: 生成拓扑可视化")
    normal_path  = str(OUTPUT_ROOT / f"topology_normal_{ts}.png")
    anomaly_path = str(OUTPUT_ROOT / f"topology_anomalies_{ts}.png")
    save_topology_image(graph, normal_path,
                        title="配电网正常拓扑图")
    save_topology_image(graph, anomaly_path, anomalies=detected,
                        title="异常检测结果 (红=异常)")
    logger.info(f"拓扑图: {normal_path}")
    logger.info(f"异常图: {anomaly_path}")
    report["steps"]["visualization"] = {
        "normal_topology":  normal_path,
        "anomaly_topology": anomaly_path,
    }

    # ---- 步骤9: 保存 JSON 报告 ----
    logger.info("步骤9/10: 保存报告")
    report_path     = str(OUTPUT_ROOT / f"mvp_report_{ts}.json")
    corrections_path = str(OUTPUT_ROOT / f"corrections_{ts}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    with open(corrections_path, "w", encoding="utf-8") as f:
        json.dump(corrections, f, ensure_ascii=False, indent=2, default=str)

    # ---- 步骤10: 打印摘要 ----
    print("\n" + "=" * 70)
    print("  MVP 运行完成 — 结果摘要")
    print("=" * 70)
    print(f"  网络规模: {len(net.bus)} 母线, {len(net.line)} 线路, "
          f"{len(net.trafo)} 变压器")
    print(f"  注入异常: {len(ground_truth)} 个")
    print(f"  检测结果: {len(detected)} 个异常")
    print(f"  评估指标:")
    print(f"    准确率(Precision) = {eval_result['precision']:.3f}")
    print(f"    召回率(Recall)    = {eval_result['recall']:.3f}")
    print(f"    F1分数            = {eval_result['f1']:.3f}")
    print(f"    TP={eval_result['tp']}, FP={eval_result['fp']}, "
          f"FN={eval_result['fn']}")
    print(f"  修正方案: {corrections['summary']['total']} 个")
    print()
    print(f"  输出文件:")
    print(f"    报告:     {report_path}")
    print(f"    修正方案: {corrections_path}")
    print(f"    正常拓扑: {normal_path}")
    print(f"    异常拓扑: {anomaly_path}")
    print("=" * 70)

    return report, corrections, eval_result


if __name__ == "__main__":
    run_mvp()

