# -*- coding: utf-8 -*-
"""
第一层检测: 规则引擎
基于图论和物理约束的确定性检测规则
- 连通性分析（BFS/DFS检测不可达节点）
- 辐射状校验（检测环路）
- CIM-SVG设备ID一致性比对
- 节点度异常检测
- KVL/KCL 粗校验
"""
import networkx as nx
import numpy as np
from typing import Dict, List, Set, Tuple
import logging

logger = logging.getLogger(__name__)



def detect_model_mismatch(cim_devices, svg_devices):
    """图模不符检测：比较CIM和SVG设备列表的一致性"""
    anomalies = []
    if not cim_devices or not svg_devices:
        return anomalies
    
    cim_ids = {d.get("uri", d.get("id", "")) for d in cim_devices}
    svg_ids = {d.get("id", "") for d in svg_devices}
    
    # CIM有但SVG没有
    only_cim = cim_ids - svg_ids
    if only_cim:
        anomalies.append({
            "type": "model_mismatch",
            "location": f"CIM独有{len(only_cim)}个设备",
            "confidence": 0.85,
            "severity": "medium",
            "details": f"CIM中有{len(only_cim)}个设备在SVG中缺失",
            "layer": "Rule",
        })
    
    # SVG有但CIM没有
    only_svg = svg_ids - cim_ids
    if only_svg:
        anomalies.append({
            "type": "model_mismatch",
            "location": f"SVG独有{len(only_svg)}个设备",
            "confidence": 0.85,
            "severity": "medium",
            "details": f"SVG中有{len(only_svg)}个设备在CIM中缺失",
            "layer": "Rule",
        })
    
    return anomalies


# v16: 类型特定置信度阈值
TYPE_CONFIDENCE_THRESHOLDS = {
    # 硬故障(断线/短路): 低阈值，宁可多报不可漏报
    "topo_interrupt": 0.3,
    "grounding_fault": 0.3,
    "voltage_collapse": 0.3,
    "branch_contingency": 0.3,
    # 软异常(参数偏差): 高阈值，减少误报
    "parameter_error": 0.7,
    "measurement_bias": 0.7,
    "impedance_degradation": 0.6,
    "harmonic_pollution": 0.6,
    # 中等: 默认阈值
    "default": 0.5,
}

def get_confidence_threshold(anomaly_type):
    """获取异常类型的特定置信度阈值"""
    return TYPE_CONFIDENCE_THRESHOLDS.get(anomaly_type, 0.5)

class RuleEngine:
    """规则引擎：第一层异常检测"""

    def __init__(self, thresholds: Dict = None):
        self.thresholds = thresholds or {
            "voltage_pu_max": 1.10,
            "voltage_pu_min": 0.90,
            "connectivity_degree_max": 20,
        }
        self.anomalies: List[Dict] = []

    def run_all_checks(self, G: nx.Graph,
                       cim_data=None,
                       svg_data=None,
                       measurements: Dict = None,
                       alignment: Dict = None) -> list:
        """
        运行全部规则检查

        Args:
            G: NetworkX拓扑图（节点=母线，边=线路/开关）
            cim_data: CIM解析结果
            svg_data: SVG解析结果
            measurements: SCADA量测数据
            alignment: CIM-SVG对齐结果

        Returns:
            anomalies: 异常列表 [{type, severity, description, evidence}]
        """
        self.anomalies = []

        # 1. 连通性分析
        self._check_connectivity(G)

        # 2. 辐射状校验（配电网应为树状结构）
        self._check_radial(G)

        # 3. 节点度异常
        self._check_node_degree(G)

        # 4. CIM-SVG ID一致性比对
        if alignment:
            self._check_id_consistency(alignment)

        # 5. 电压量测合理性（KVL粗校验）
        if measurements:
            self._check_voltage_plausibility(measurements)

        # 6. 功率平衡校验（KCL粗校验）
        if measurements:
            self._check_power_balance(G, measurements)

        # 7. 遥信!=遥测检测
        if measurements:
            self._check_signal_mismatch(G, measurements)

        # 8. 图模不符检测(CIM-SVG设备一致性)
        if cim_data and svg_data:
            mm_anomalies = detect_model_mismatch(cim_data, svg_data)
            self.anomalies.extend(mm_anomalies)
        

        # 9. Virtual/faulty connection detection
        self._check_virtual_faulty_connection(G, measurements)
        
        

        # 10. High Impedance Fault (HIF) detection
        if measurements:
            self._check_hif_fault(G, measurements)

        # 11. Line break / open conductor detection
        if measurements:
            self._check_line_break(G, measurements)

        # Normalize all type names to English
        TYPE_EN = {
            "\u56fe\u6a21\u4e0d\u7b26": "model_mismatch", "\u865a\u62df\u63a5/\u9519\u63a5": "virtual_faulty",
            "\u9065\u4fe1!=\u9065\u6d4b": "signal_mismatch", "\u62d3\u6251\u4e2d\u65ad": "topo_interrupt",
            "\u9065\u6d4b!=\u62d3\u6251": "telemetry_mismatch", "\u8fde\u901a\u6027\u5f02\u5e38": "topo_interrupt",
            "\u7535\u6c14\u5b64\u5c9b": "topo_interrupt",
        }
        for a in self.anomalies:
            t = a.get("type", "")
            if t in TYPE_EN:
                a["type"] = TYPE_EN[t]
            if a.get("layer") == "\u89c4\u5219\u5f15\u64ce":
                a["layer"] = "Rule"
        
        logger.info(f"规则引擎检测完成: 共发现 {len(self.anomalies)} 个异常")
        return self.anomalies

    def _check_connectivity(self, G: nx.Graph):
        """连通性分析：检测不可达节点（拓扑中断）
        v22: 只报告有明确断线证据的隔离组件，限制每个网络最多3条检测。"""
        if G.number_of_nodes() == 0:
            return

        components = list(nx.connected_components(G))
        if len(components) <= 1:
            return

        main_comp = max(components, key=len)
        node_to_comp = {}
        for i, comp in enumerate(components):
            for n in comp:
                node_to_comp[n] = i
        
        # Only report bridge lines that are explicitly out-of-service
        reported_lines = set()
        n_reported = 0
        max_reports = 3  # v22: cap per network to reduce FP
        
        for comp in components:
            if comp == main_comp or n_reported >= max_reports:
                continue
            if len(comp) < 2:  # v22: skip single-node components (likely noise)
                continue
            sample_nodes = list(comp)[:5]
            
            # Find out-of-service bridge edges connecting this component
            found_bridge = False
            for u, v, data in G.edges(data=True):
                u_comp = node_to_comp.get(u)
                v_comp = node_to_comp.get(v)
                if u_comp is None or v_comp is None or u_comp == v_comp:
                    continue
                if not ({u_comp, v_comp} & {node_to_comp.get(n) for n in comp}):
                    continue
                in_svc = data.get("in_service", data.get("in_svc", True))
                if not in_svc:  # Only report explicitly out-of-service lines
                    line_id = data.get("device", data.get("element_idx", f"edge_{u}_{v}"))
                    if line_id not in reported_lines:
                        reported_lines.add(line_id)
                        self.anomalies.append({
                            "type": "topo_interrupt",
                            "severity": "critical",
                            "description": f"Line {line_id} disconnected: {len(comp)} nodes isolated",
                            "evidence": {
                                "component_size": len(comp),
                                "sample_nodes": sample_nodes,
                                "bridge_line": line_id,
                            },
                            "rule": "connectivity_bridge_line",
                            "confidence": 0.95,
                        })
                        n_reported += 1
                        found_bridge = True
            
            # v22: Only report isolated component if no bridge found AND component is significant
            if not found_bridge and len(comp) >= 3:
                self.anomalies.append({
                    "type": "topo_interrupt",
                    "severity": "critical",
                    "description": f"Isolated component: {len(comp)} nodes",
                    "evidence": {"component_size": len(comp), "sample_nodes": sample_nodes},
                    "rule": "connectivity_check",
                    "confidence": 0.85,
                })
                n_reported += 1
        
        if n_reported > 0:
            logger.warning(f"连通性异常: {len(components)} 个电气岛, {n_reported} 个报告")


    def _check_radial(self, G: nx.Graph):
        """辐射状校验：配电网应为树（无环），有环则可能是虚接/错接"""
        if G.number_of_nodes() == 0:
            return

        is_tree = nx.is_tree(G)
        if not is_tree:
            # Skip full cycle enumeration for large networks (too slow, too many false positives)
            if G.number_of_nodes() > 200:
                self.anomalies.append({
                    "type": "virtual_faulty",
                    "severity": "medium",
                    "description": f"网络不是辐射状结构(大规模网络,跳过环路枚举)",
                    "evidence": {"is_tree": False, "n_nodes": G.number_of_nodes()},
                    "rule": "radial_check",
                })
                return
            cycles = list(nx.cycle_basis(G))
            self.anomalies.append({
                "type": "\u865a\u63a5/\u9519\u63a5",
                "severity": "\u4e2d\u7b49",
                "description": f"\u7f51\u7edc\u4e0d\u662f\u8f90\u5c04\u72b6\u7ed3\u6784\uff0c\u5b58\u5728 {len(cycles)} \u4e2a\u73af\u8def",
                "evidence": {
                    "is_tree": False,
                    "n_cycles": len(cycles),
                    "cycles_sample": [c[:10] for c in cycles[:5]],
                },
                "rule": "radial_check",
            })
            logger.warning(f"\u8f90\u5c04\u72b6\u5f02\u5e38: \u53d1\u73b0 {len(cycles)} \u4e2a\u73af\u8def")

    def _check_node_degree(self, G: nx.Graph):
        """节点度异常检测：度数过高可能表示错接"""
        deg_max = self.thresholds.get("connectivity_degree_max", 20)
        for node, deg in G.degree():
            if deg > deg_max:
                self.anomalies.append({
                    "type": "\u865a\u63a5/\u9519\u63a5",
                    "severity": "\u9884\u8b66",
                    "description": f"\u8282\u70b9 {node} \u5ea6\u6570={deg}\uff0c\u8d85\u8fc7\u9608\u503c {deg_max}",
                    "evidence": {"node": str(node), "degree": deg,
                                 "neighbors": list(G.neighbors(node))[:10]},
                    "rule": "node_degree_check",
                })

    def _check_id_consistency(self, alignment: Dict):
        """CIM-SVG设备ID一致性比对（图模不符）"""
        cim_only = alignment.get("cim_only", [])
        svg_only = alignment.get("svg_only", [])

        if cim_only:
            self.anomalies.append({
                "type": "\u56fe\u6a21\u4e0d\u7b26",
                "severity": "\u4e2d\u7b49",
                "description": f"CIM\u4e2d\u6709 {len(cim_only)} \u4e2a\u8bbe\u5907\u5728SVG\u4e2d\u7f3a\u5931",
                "evidence": {"missing_in_svg": cim_only[:20]},
                "rule": "id_consistency_cim_to_svg",
            })

        if svg_only:
            self.anomalies.append({
                "type": "\u56fe\u6a21\u4e0d\u7b26",
                "severity": "\u4e2d\u7b49",
                "description": f"SVG\u4e2d\u6709 {len(svg_only)} \u4e2a\u8bbe\u5907\u5728CIM\u4e2d\u7f3a\u5931",
                "evidence": {"missing_in_cim": svg_only[:20]},
                "rule": "id_consistency_svg_to_cim",
            })

    def _check_voltage_plausibility(self, measurements: Dict):
        """电压量测合理性检查（KVL粗校验）"""
        v_max = self.thresholds.get("voltage_pu_max", 1.10)
        v_min = self.thresholds.get("voltage_pu_min", 0.90)
        violations = []

        for bv in measurements.get("bus_voltages", []):
            vm = bv["vm_pu"]
            if vm > v_max or vm < v_min:
                violations.append({
                    "bus": bv["bus"], "vm_pu": vm,
                    "bound": f"[{v_min}, {v_max}]",
                })

        if violations:
            self.anomalies.append({
                "type": "\u9065\u6d4b!=\u62d3\u6251",
                "severity": "\u9884\u8b66",
                "description": f"{len(violations)} \u6761\u6bcd\u7ebf\u7535\u538b\u8d8a\u9650",
                "evidence": {"violations": violations[:20]},
                "rule": "voltage_plausibility",
            })

    def _check_power_balance(self, G: nx.Graph, measurements: Dict):
        """功率平衡粗校验（KCL简略版）"""
        bus_inject = {}
        for lp in measurements.get("line_powers", []):
            fb, tb = lp["from_bus"], lp["to_bus"]
            p = lp["p_mw"]
            side = lp["side"]
            if side == "from":
                bus_inject.setdefault(fb, 0.0)
                bus_inject[fb] -= p
                bus_inject.setdefault(tb, 0.0)
                bus_inject[tb] += p
            elif side == "to":
                bus_inject.setdefault(tb, 0.0)
                bus_inject[tb] -= p
                bus_inject.setdefault(fb, 0.0)
                bus_inject[fb] += p

        balance_violations = []
        for bus_idx, p_inject in bus_inject.items():
            if abs(p_inject) > 10.0:
                balance_violations.append({
                    "bus": bus_idx, "net_injection_mw": round(p_inject, 4),
                })

        if balance_violations:
            self.anomalies.append({
                "type": "\u9065\u6d4b!=\u62d3\u6251",
                "severity": "\u9884\u8b66",
                "description": f"{len(balance_violations)} \u4e2a\u8282\u70b9\u529f\u7387\u4e0d\u5e73\u8861",
                "evidence": {"violations": balance_violations[:20]},
                "rule": "power_balance_kcl",
            })

    def _check_signal_mismatch(self, G: nx.Graph, measurements: Dict):
        """遥信!=遥测检测：功率突变 + 电压/功率矛盾"""
        if not measurements:
            return

        line_powers = measurements.get("line_powers", [])
        if len(line_powers) < 2:
            return

        # Compute mean absolute power per line
        line_p_mean = {}
        for lp in line_powers:
            lid = lp.get("line", -1)
            p = abs(lp.get("p_mw", 0))
            line_p_mean.setdefault(lid, []).append(p)

        # Detect lines with power >> mean (potential signal mismatch)
        all_p = [abs(lp.get("p_mw", 0)) for lp in line_powers]
        if not all_p:
            return
        mean_p = np.mean(all_p)
        std_p = np.std(all_p) if np.std(all_p) > 0 else mean_p * 0.1
        threshold = max(mean_p + 4 * std_p, mean_p * 3.0)

        for lp in line_powers:
            p = abs(lp.get("p_mw", 0))
            if p > threshold and p > 1.0:  # ignore tiny networks
                self.anomalies.append({
                    "type": "signal_mismatch",
                    "severity": "medium",
                    "description": f"线路 {lp.get('line','?')} 功率突变 ({p:.2f} MW), "
                                   f"均值={mean_p:.2f}, 可能遥信/遥测矛盾",
                    "evidence": {"line": lp.get("line"), "p_mw": round(p, 4),
                                 "mean_p_mw": round(mean_p, 4),
                                 "threshold": round(threshold, 4)},
                    "rule": "signal_mismatch_power_spike",
                })
                break  # report first occurrence

        # Check voltage vs power consistency
        bus_voltages = measurements.get("bus_voltages", [])
        if bus_voltages:
            low_v = [bv for bv in bus_voltages if bv.get("vm_pu", 1.0) < 0.88]
            if low_v and any(abs(lp.get("p_mw", 0)) < mean_p * 0.1 for lp in line_powers):
                # Low voltage but very low power -> possible signal mismatch
                self.anomalies.append({
                    "type": "signal_mismatch",
                    "severity": "warning",
                    "description": f"电压偏低({len(low_v)}条母线<0.88pu)但功率正常, "
                                   f"可能遥信状态与实际不符",
                    "evidence": {"low_voltage_buses": [bv["bus"] for bv in low_v[:5]]},
                    "rule": "signal_mismatch_voltage_power",
                })

    def _check_virtual_faulty_connection(self, G, measurements=None):
        """Virtual/faulty connection detection: degree anomaly + extreme voltage + ring"""
        if G.number_of_nodes() < 3:
            return

        import numpy as np

        # Check 1: Degree anomaly
        degrees = [d for _, d in G.degree()]
        if len(degrees) > 5:
            mean_d = np.mean(degrees)
            std_d = np.std(degrees) if np.std(degrees) > 0 else 1.0
            for node, deg in G.degree():
                if deg > mean_d + 4 * std_d and deg > 8:  # v22: tightened from 3sigma/5 to 4sigma/8
                    self.anomalies.append({
                        "type": "virtual_faulty",
                        "severity": "medium",
                        "description": f"节点 {node} 度数异常 ({deg}), 可能错接",
                        "evidence": {"node": node, "degree": deg,
                                     "mean": round(mean_d, 2), "std": round(std_d, 2)},
                        "rule": "virtual_connection_degree",
                    })

        # Check 2: Voltage deviation suggests wrong/virtual connection
        if measurements:
            bus_voltages = measurements.get('bus_voltages', [])
            if len(bus_voltages) >= 3:
                v_vals = [bv['vm_pu'] for bv in bus_voltages]
                v_mean = np.mean(v_vals)
                v_std = max(np.std(v_vals), 0.01)
                for bv in bus_voltages:
                    vm = bv['vm_pu']
                    deviation = abs(vm - v_mean)
                    if deviation > 5 * v_std and (vm < 0.85 or vm > 1.15):  # v22: tightened thresholds
                        self.anomalies.append({
                            "type": "virtual_faulty",
                            "severity": "medium",
                            "description": f"节点 {bv['bus']} 电压偏离 ({vm:.3f}), 可能虚接/错接",
                            "evidence": {"bus": bv["bus"], "vm_pu": vm, "deviation": round(deviation, 4),
                                         "mean": round(v_mean, 4), "std": round(v_std, 4)},
                            "rule": "virtual_connection_voltage",
                        })

        # Check 3: Ring detection in radial network (v22: only for small/medium networks)
        import networkx as nx
        if not nx.is_tree(G) and len(G.nodes) <= 500:
            cycles = nx.cycle_basis(G)
            if cycles:
                cycle = cycles[0]
                if len(cycle) >= 5:  # v22: increased from 4 to reduce FP
                    self.anomalies.append({
                        "type": "virtual_faulty",
                        "severity": "medium",
                        "description": f"发现环路 ({len(cycle)}节点), 配电网应为辐射状",
                        "evidence": {"cycle_nodes": cycle[:10], "cycle_length": len(cycle)},
                        "rule": "virtual_connection_ring",
                    })


    def _check_hif_fault(self, G, measurements):
        """High Impedance Fault (HIF) detection - 5 criteria joint assessment.
        
        HIF characteristics:
        - Current increase small (1.1-1.7x normal)
        - Voltage sag 0.85-0.95 pu
        - Harmonic distortion (odd harmonics increase)
        - Negative sequence current (unbalanced)
        - Zero sequence voltage rise
        
        Rule-based detection using available SCADA measurements.
        """
        if not measurements:
            return
        
        bus_voltages = {bv["bus"]: bv for bv in measurements.get("bus_voltages", [])}
        line_powers = measurements.get("line_powers", [])
        
        # Thresholds (configurable)
        v_sag_low = self.thresholds.get("hif_v_sag_low", 0.85)
        v_sag_high = self.thresholds.get("hif_v_sag_high", 0.95)
        q_p_ratio = self.thresholds.get("hif_q_p_ratio", 0.5)
        loss_factor = self.thresholds.get("hif_loss_factor", 2.0)
        
        for lp in line_powers:
            line_idx = lp.get("line", lp.get("element_idx", -1))
            p_mw = abs(lp.get("p_mw", 0.0))
            q_mvar = abs(lp.get("q_mvar", 0.0))
            i_ka = lp.get("i_ka", 0.0)
            fb = int(lp.get("from_bus", -1))
            tb = int(lp.get("to_bus", -1))
            
            vf = bus_voltages.get(fb, {}).get("vm_pu", 1.0)
            vt = bus_voltages.get(tb, {}).get("vm_pu", 1.0)
            
            score = 0
            details = []
            
            # Criterion 1: Voltage sag at one or both ends
            if (v_sag_low <= vf <= v_sag_high) or (v_sag_low <= vt <= v_sag_high):
                score += 1
                details.append(f"voltage_sag: Vf={vf:.3f} Vt={vt:.3f}")
            
            # Criterion 2: Reactive power anomaly (Q/P ratio high)
            if p_mw > 0.01 and q_mvar / p_mw > q_p_ratio:
                score += 1
                details.append(f"Q/P={q_mvar/max(p_mw,0.001):.2f}")
            
            # Criterion 3: Voltage imbalance (from vs to bus)
            if abs(vf - vt) > 0.03 and abs(vf - vt) < 0.15:
                score += 1
                details.append(f"V_diff={abs(vf-vt):.3f}")
            
            # Criterion 4: Line loss anomaly (P*I relationship)
            if i_ka > 0.01 and p_mw > 0:
                expected_loss = i_ka ** 2 * 0.5  # rough estimate
                if p_mw * 0.05 > expected_loss * loss_factor:
                    score += 1
                    details.append("loss_anomaly")
            
            # Criterion 5: Non-zero but reduced current (not short circuit, not open)
            if 0.005 < i_ka < 0.5:
                score += 1
                details.append(f"current_regime: I={i_ka:.4f}kA")
            
            if score >= 4:
                self.anomalies.append({
                    "type": "hif_fault",
                    "severity": "critical",
                    "description": f"HIF suspected on line {line_idx} (bus {fb}-{tb}), score={score}/5",
                    "evidence": {
                        "line_idx": line_idx,
                        "from_bus": fb,
                        "to_bus": tb,
                        "hif_score": score,
                        "criteria": details,
                        "v_from": vf,
                        "v_to": vt,
                        "p_mw": p_mw,
                        "q_mvar": q_mvar,
                        "i_ka": i_ka,
                    },
                    "rule": "hif_detection",
                })
                logger.warning(f"HIF suspected: line {line_idx}, score {score}/5")

    def _check_line_break(self, G, measurements):
        """Line break / open conductor detection.
        
        Characteristics:
        - Line current near zero but line is in_service
        - Voltage difference between ends
        - Zero power flow but load demand exists on downstream side
        """
        if not measurements:
            return
        
        bus_voltages = {bv["bus"]: bv for bv in measurements.get("bus_voltages", [])}
        line_powers = measurements.get("line_powers", [])
        
        i_threshold = self.thresholds.get("line_break_i_ka", 0.0005)
        v_diff_threshold = self.thresholds.get("line_break_v_diff", 0.08)
        
        for lp in line_powers:
            line_idx = lp.get("line", lp.get("element_idx", -1))
            i_ka = abs(lp.get("i_ka", 0.0))
            p_mw = abs(lp.get("p_mw", 0.0))
            fb = int(lp.get("from_bus", -1))
            tb = int(lp.get("to_bus", -1))
            
            vf = bus_voltages.get(fb, {}).get("vm_pu", 1.0)
            vt = bus_voltages.get(tb, {}).get("vm_pu", 1.0)
            
            # Both buses have voltage but current is near zero
            if (i_ka < i_threshold and 
                vf > 0.80 and vt > 0.80 and 
                abs(vf - vt) > v_diff_threshold):
                
                # Check if downstream load exists
                has_downstream_load = False
                for n in G.neighbors(f"bus_{tb}") if f"bus_{tb}" in G else []:
                    ndata = G.nodes.get(n, {})
                    if ndata.get("is_load", False):
                        has_downstream_load = True
                        break
                
                confidence = 0.7 if has_downstream_load else 0.5
                if abs(vf - vt) > 0.10:
                    confidence = min(confidence + 0.15, 0.95)
                
                self.anomalies.append({
                    "type": "line_break",
                    "severity": "critical",
                    "description": f"Line break suspected: line {line_idx} (bus {fb}-{tb}), I={i_ka:.4f}kA, dV={abs(vf-vt):.3f}pu",
                    "evidence": {
                        "line_idx": line_idx,
                        "from_bus": fb,
                        "to_bus": tb,
                        "i_ka": i_ka,
                        "v_from": vf,
                        "v_to": vt,
                        "v_diff": abs(vf - vt),
                        "has_downstream_load": has_downstream_load,
                    },
                    "confidence": confidence,
                    "rule": "line_break_detection",
                })
                logger.warning(f"Line break suspected: line {line_idx}, I={i_ka:.4f}kA, dV={abs(vf-vt):.3f}")

def build_nx_graph_from_pandapower(net) -> nx.Graph:
    """从PandaPower网络构建NetworkX图"""
    G = nx.Graph()

    for idx in net.bus.index:
        G.add_node(f"bus_{idx}", vn_kv=float(net.bus.at[idx, "vn_kv"]),
                   name=str(net.bus.at[idx, "name"]))

    for idx in net.line.index:
        if net.line.at[idx, "in_service"]:
            G.add_edge(f"bus_{net.line.at[idx, 'from_bus']}",
                       f"bus_{net.line.at[idx, 'to_bus']}",
                       type="line", index=int(idx),
                       length_km=float(net.line.at[idx, "length_km"]))

    for idx in net.trafo.index:
        if net.trafo.at[idx, "in_service"]:
            G.add_edge(f"bus_{net.trafo.at[idx, 'hv_bus']}",
                       f"bus_{net.trafo.at[idx, 'lv_bus']}",
                       type="transformer", index=int(idx))

    for idx in net.switch.index:
        if net.switch.at[idx, "closed"]:
            et = net.switch.at[idx, "et"]
            bus_n = f"bus_{net.switch.at[idx, 'bus']}"
            elem_n = f"bus_{net.switch.at[idx, 'element']}"
            if et == "b":
                G.add_edge(bus_n, elem_n, type="switch", index=int(idx))

    return G


def run_rule_engine(graph, cim_devices, svg_devices, measurements, switches, thresholds=None):
    """运行规则引擎检测（函数式接口）"""
    engine = RuleEngine(thresholds=thresholds)
    return engine.run_all_checks(graph, cim_data=cim_devices, svg_data=svg_devices, measurements=measurements)