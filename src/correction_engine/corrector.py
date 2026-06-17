# -*- coding: utf-8 -*-
"""
修正方案生成器
基于最小修改原则，针对5类异常生成修正建议
"""
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def generate_corrections(anomalies: List[Dict], network_data: Dict) -> Dict:
    """
    基于异常列表生成修正方案。

    原则:
      1. 最小修改 — 优先翻转开关状态而非重构拓扑
      2. 高置信度优先
      3. 安全约束 — 不做危险操作

    Args:
        anomalies:    异常列表 [{type, location, confidence, layer, details}]
        network_data: 网络数据字典

    Returns:
        {corrections: [...], summary: {...}}
    """
    corrections: List[Dict] = []
    stats: Dict[str, int] = {"total": 0}

    for anomaly in sorted(anomalies, key=lambda x: x.get("confidence", 0),
                          reverse=True):
        atype     = anomaly.get("type", "")
        confidence = anomaly.get("confidence", 0)
        corr = None

        # ---- 类型1: 图模不符 ----
        if "图模不符" in atype or "MODEL_MISMATCH" in atype:
            corr = _correct_model_mismatch(anomaly)

        # ---- 类型2: 拓扑中断 ----
        elif "拓扑中断" in atype or "TOPO_INTERRUPT" in atype:
            corr = _correct_topology_interrupt(anomaly)

        # ---- 类型3: 虚接/错接 ----
        elif "虚接" in atype or "错接" in atype or "VIRTUAL_FAULTY" in atype:
            corr = _correct_virtual_faulty(anomaly)

        # ---- 类型4: 遥测!=拓扑 ----
        elif "遥测" in atype or "TELE_TOPO_MISMATCH" in atype:
            corr = _correct_measurement_error(anomaly)

        # ---- 类型5: 遥信!=遥测 ----
        elif "遥信" in atype or "TELE_SIGNAL_MISMATCH" in atype:
            corr = _correct_switch_mismatch(anomaly)

        # ---- 状态估计层 ----
        elif "不良数据" in atype:
            corr = _correct_bad_measurement(anomaly)
        elif "拓扑错误" in atype:
            corr = _correct_topology_error(anomaly)
        elif "chi2" in atype.lower():
            corr = _correct_chi2_failure(anomaly)

        if corr:
            corrections.append(corr)
            stats["total"] += 1
            key = corr.get("anomaly_type", "其他")
            stats[key] = stats.get(key, 0) + 1

    logger.info(f"生成 {stats['total']} 个修正方案")
    return {"corrections": corrections, "summary": stats}


# ============================================================
# 各类异常的修正子函数
# ============================================================

def _correct_model_mismatch(anomaly: Dict) -> Dict:
    return {
        "anomaly_type": "图模不符",
        "action":       "sync_model",
        "priority":     "高" if anomaly.get("confidence", 0) > 0.8 else "中",
        "target":       str(anomaly.get("location", "")),
        "description":  f"建议同步CIM/SVG模型: {anomaly.get('details', '')}",
        "steps": [
            "1. 检查缺失设备是否存在于CIM文件中",
            "2. 检查SVG图形是否遗漏了设备图元",
            "3. 补全缺失的模型/图形数据",
            "4. 验证CIM-SVG设备数量一致",
        ],
        "confidence": anomaly.get("confidence", 0),
    }


def _correct_topology_interrupt(anomaly: Dict) -> Dict:
    location = str(anomaly.get("location", ""))
    if "孤立" in anomaly.get("details", ""):
        steps = [
            f"1. 检查节点 {location} 的CIM终端定义",
            "2. 确认连接线路/开关是否正确录入",
            "3. 如有缺失，补充连接关系",
        ]
    else:
        steps = [
            "1. 检查路径上的开关状态（是否有误断开的开关）",
            "2. 检查线路in_service状态",
            "3. 如开关状态错误，修正遥信数据",
            "4. 如线路确实断开，评估是否需要合闸恢复",
        ]
    return {
        "anomaly_type": "拓扑中断",
        "action":       "check_switch_or_line",
        "priority":     "高",
        "target":       location,
        "description":  f"节点 {location} 不可达: {anomaly.get('details', '')}",
        "steps":        steps,
        "confidence":   anomaly.get("confidence", 0),
    }


def _correct_virtual_faulty(anomaly: Dict) -> Dict:
    return {
        "anomaly_type": "虚接/错接",
        "action":       "check_connection",
        "priority":     "高",
        "target":       str(anomaly.get("location", "")),
        "description":  f"检测到连接异常: {anomaly.get('details', '')}",
        "steps": [
            "1. 核实设备端子与连接节点的对应关系",
            "2. 检查是否有多余或错误的连接",
            "3. 修正CIM模型中的Terminal/ConnectivityNode映射",
        ],
        "confidence": anomaly.get("confidence", 0),
    }


def _correct_measurement_error(anomaly: Dict) -> Dict:
    return {
        "anomaly_type": "遥测异常",
        "action":       "calibrate_sensor",
        "priority":     "中",
        "target":       str(anomaly.get("location", "")),
        "description":  f"量测数据异常: {anomaly.get('details', '')}",
        "steps": [
            "1. 检查对应传感器/互感器是否正常",
            "2. 对比其他冗余量测",
            "3. 如确认传感器故障，标记为不可用",
            "4. 使用状态估计替代值",
        ],
        "confidence": anomaly.get("confidence", 0),
    }


def _correct_switch_mismatch(anomaly: Dict) -> Dict:
    return {
        "anomaly_type": "遥信异常",
        "action":       "verify_switch_state",
        "priority":     "高",
        "target":       str(anomaly.get("location", "")),
        "description":  f"开关状态不一致: {anomaly.get('details', '')}",
        "steps": [
            "1. 现场核实开关实际位置",
            "2. 检查遥信采集回路是否正常",
            "3. 如确认遥信误报，更新SCADA数据库",
            "4. 如开关确实异常，派检修处理",
        ],
        "confidence": anomaly.get("confidence", 0),
    }


def _correct_bad_measurement(anomaly: Dict) -> Dict:
    return {
        "anomaly_type": "不良数据",
        "action":       "remove_or_correct",
        "priority":     "中",
        "target":       str(anomaly.get("location", "")),
        "description":  f"状态估计识别的不良数据: {anomaly.get('details', '')}",
        "steps": [
            "1. 检查量测设备是否故障",
            "2. 与相邻量测交叉验证",
            "3. 如确认为不良数据，从状态估计中剔除",
            "4. 使用估计值替代",
        ],
        "confidence": anomaly.get("confidence", 0),
    }


def _correct_topology_error(anomaly: Dict) -> Dict:
    target = str(anomaly.get("location", ""))
    if "switch" in target:
        steps = [
            "1. 核实开关实际位置",
            "2. 如确认状态错误，更新拓扑模型中开关状态",
            "3. 重新运行潮流计算验证",
        ]
    else:
        steps = [
            "1. 检查相关线路的连接关系",
            "2. 核实是否存在误断开的开关",
            "3. 修正拓扑模型后重新估计",
        ]
    return {
        "anomaly_type": "拓扑错误",
        "action":       "verify_topology",
        "priority":     "高" if "switch" in target else "中",
        "target":       target,
        "description":  f"拓扑错误: {anomaly.get('details', '')}",
        "steps":        steps,
        "confidence":   anomaly.get("confidence", 0),
    }


def _correct_chi2_failure(anomaly: Dict) -> Dict:
    return {
        "anomaly_type": "全局检验失败",
        "action":       "systematic_check",
        "priority":     "高",
        "target":       "system",
        "description":  f"chi2全局检验未通过: {anomaly.get('details', '')}",
        "steps": [
            "1. 系统性检查所有量测数据质量",
            "2. 检查是否存在未建模的拓扑变化",
            "3. 检查是否存在拓扑错误",
            "4. 逐步剔除可疑量测直到chi2检验通过",
        ],
        "confidence": anomaly.get("confidence", 0),
    }

# ============================================================
# Auto Topology Corrector (P0-4)
# ============================================================
class AutoTopologyCorrector:
    """Automatic topology reconstruction with minimal-change principle.
    
    Key capability: competitors only DETECT, we also CORRECT.
    Each correction is verified via PandaPower power flow before applying.
    """

    def __init__(self, net=None, max_switch_changes=5):
        self.net = net
        self.max_switch_changes = max_switch_changes
        self.correction_log = []

    def auto_correct(self, anomalies, network_data):
        """Auto-correct entry point. Returns corrections with verification."""
        import copy
        results = []
        switch_changes = 0

        for anomaly in sorted(anomalies, key=lambda x: x.get("confidence", 0), reverse=True):
            if switch_changes >= self.max_switch_changes:
                break
            atype = anomaly.get("type", "")
            r = None
            if atype == "topo_interrupt":
                r = self._fix_topology_interrupt(anomaly, network_data)
            elif atype == "virtual_faulty":
                r = self._fix_virtual_faulty(anomaly, network_data)
            elif atype == "signal_mismatch":
                r = self._fix_switch_state(anomaly, network_data)
            elif atype == "hif_fault":
                r = self._isolate_faulty_section(anomaly, network_data)
            elif atype == "line_break":
                r = self._reroute_power(anomaly, network_data)
            if r and r.get("verified"):
                results.append(r)
                switch_changes += r.get("switch_changes", 0)

        return {"corrections": results, "total_switch_changes": switch_changes}

    def _verify_correction(self, net_test):
        """Verify correction via power flow. Returns (passed, details)."""
        import copy
        import pandapower as pp
        import networkx as nx
        try:
            G = nx.Graph()
            for idx in net_test.line.index:
                if net_test.line.at[idx, "in_service"]:
                    G.add_edge(int(net_test.line.at[idx, "from_bus"]),
                               int(net_test.line.at[idx, "to_bus"]))
            if not nx.is_connected(G) and G.number_of_nodes() > 1:
                return False, {"reason": "disconnected"}
            pp.runpp(net_test)
            vmin = float(net_test.res_bus.vm_pu.min())
            vmax = float(net_test.res_bus.vm_pu.max())
            if vmin < 0.85 or vmax > 1.15:
                return False, {"reason": f"voltage {vmin:.3f}-{vmax:.3f} out of range"}
            return True, {"vmin": vmin, "vmax": vmax}
        except Exception as e:
            return False, {"reason": str(e)[:50]}

    def _fix_topology_interrupt(self, anomaly, network_data):
        """Fix topology interrupt: find open switches that can reconnect islands."""
        import copy
        import pandapower as pp
        if self.net is None:
            return None
        evidence = anomaly.get("evidence", {})
        bridge_lines = evidence.get("bridge_lines", [])
        
        net = self.net
        if not hasattr(net, "switch") or len(net.switch) == 0:
            return None

        # Find open switches that could reconnect isolated components
        candidates = []
        for sw_idx in net.switch.index:
            if not net.switch.at[sw_idx, "closed"]:
                candidates.append(sw_idx)

        for sw_idx in candidates:
            net_test = copy.deepcopy(net)
            net_test.switch.at[sw_idx, "closed"] = True
            passed, details = self._verify_correction(net_test)
            if passed:
                self.net.switch.at[sw_idx, "closed"] = True
                return {
                    "action": "close_switch",
                    "switch_idx": int(sw_idx),
                    "verified": True,
                    "verification": details,
                    "switch_changes": 1,
                    "anomaly_type": "topo_interrupt",
                }
        return None

    def _fix_virtual_faulty(self, anomaly, network_data):
        """Fix virtual/faulty connection: open a switch in a loop to restore radial."""
        import copy
        import pandapower as pp
        import networkx as nx
        if self.net is None:
            return None
        net = self.net
        G = nx.Graph()
        for idx in net.line.index:
            if net.line.at[idx, "in_service"]:
                G.add_edge(int(net.line.at[idx, "from_bus"]),
                           int(net.line.at[idx, "to_bus"]))
        if nx.is_tree(G):
            return None
        cycles = nx.cycle_basis(G)
        if not cycles:
            return None
        # Find switches in cycles that can be opened
        if hasattr(net, "switch"):
            for sw_idx in net.switch.index:
                if net.switch.at[sw_idx, "closed"]:
                    bus = int(net.switch.at[sw_idx, "bus"])
                    elem = int(net.switch.at[sw_idx, "element"])
                    # Check if this switch is in a cycle
                    for cycle in cycles:
                        if bus in cycle:
                            net_test = copy.deepcopy(net)
                            net_test.switch.at[sw_idx, "closed"] = False
                            passed, details = self._verify_correction(net_test)
                            if passed:
                                self.net.switch.at[sw_idx, "closed"] = False
                                return {
                                    "action": "open_switch",
                                    "switch_idx": int(sw_idx),
                                    "verified": True,
                                    "verification": details,
                                    "switch_changes": 1,
                                    "anomaly_type": "virtual_faulty",
                                }
                            break
        return None

    def _fix_switch_state(self, anomaly, network_data):
        """Fix switch state based on KCL residual."""
        import copy
        if self.net is None:
            return None
        evidence = anomaly.get("evidence", {})
        switch_idx = evidence.get("switch_idx")
        if switch_idx is None:
            return None
        net = self.net
        if switch_idx not in net.switch.index:
            return None
        current = bool(net.switch.at[switch_idx, "closed"])
        net_test = copy.deepcopy(net)
        net_test.switch.at[switch_idx, "closed"] = not current
        passed, details = self._verify_correction(net_test)
        if passed:
            self.net.switch.at[switch_idx, "closed"] = not current
            return {
                "action": "toggle_switch",
                "switch_idx": int(switch_idx),
                "old_state": "closed" if current else "open",
                "new_state": "open" if current else "closed",
                "verified": True,
                "verification": details,
                "switch_changes": 1,
                "anomaly_type": "signal_mismatch",
            }
        return None

    def _isolate_faulty_section(self, anomaly, network_data):
        """Isolate HIF fault section by opening switches at fault location."""
        import copy
        if self.net is None:
            return None
        evidence = anomaly.get("evidence", {})
        from_bus = evidence.get("from_bus", -1)
        to_bus = evidence.get("to_bus", -1)
        net = self.net
        if not hasattr(net, "switch"):
            return None
        # Find switches at fault location
        for sw_idx in net.switch.index:
            sw_bus = int(net.switch.at[sw_idx, "bus"])
            sw_elem = int(net.switch.at[sw_idx, "element"])
            if sw_bus in (from_bus, to_bus) and net.switch.at[sw_idx, "closed"]:
                net_test = copy.deepcopy(net)
                net_test.switch.at[sw_idx, "closed"] = False
                passed, details = self._verify_correction(net_test)
                if passed:
                    self.net.switch.at[sw_idx, "closed"] = False
                    return {
                        "action": "isolate_hif",
                        "switch_idx": int(sw_idx),
                        "verified": True,
                        "verification": details,
                        "switch_changes": 1,
                        "anomaly_type": "hif_fault",
                    }
        return None

    def _reroute_power(self, anomaly, network_data):
        """Reroute power around line break via tie switches."""
        import copy
        import pandapower as pp
        if self.net is None:
            return None
        evidence = anomaly.get("evidence", {})
        from_bus = evidence.get("from_bus", -1)
        to_bus = evidence.get("to_bus", -1)
        net = self.net
        # First isolate the broken line
        for line_idx in net.line.index:
            fb = int(net.line.at[line_idx, "from_bus"])
            tb = int(net.line.at[line_idx, "to_bus"])
            if (fb == from_bus and tb == to_bus) or (fb == to_bus and tb == from_bus):
                net_test = copy.deepcopy(net)
                net_test.line.at[line_idx, "in_service"] = False
                # Find and close a tie switch for rerouting
                if hasattr(net, "switch"):
                    for sw_idx in net.switch.index:
                        if not net.switch.at[sw_idx, "closed"]:
                            net_test2 = copy.deepcopy(net_test)
                            net_test2.switch.at[sw_idx, "closed"] = True
                            passed, details = self._verify_correction(net_test2)
                            if passed:
                                self.net.line.at[line_idx, "in_service"] = False
                                self.net.switch.at[sw_idx, "closed"] = True
                                return {
                                    "action": "reroute_power",
                                    "line_idx": int(line_idx),
                                    "tie_switch_idx": int(sw_idx),
                                    "verified": True,
                                    "verification": details,
                                    "switch_changes": 1,
                                    "anomaly_type": "line_break",
                                }
        return None


def auto_correct_topology(anomalies, network_data):
    """Convenience function for auto topology correction."""
    net = network_data.get("net")
    if net is None:
        return {"corrections": [], "total_switch_changes": 0, "note": "no net available"}
    corrector = AutoTopologyCorrector(net)
    return corrector.auto_correct(anomalies, network_data)
