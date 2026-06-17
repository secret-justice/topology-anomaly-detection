# -*- coding: utf-8 -*-
"""
设备级异常定位器
将检测结果从类型级精确到设备级（母线/线路/开关/变压器）

核心思路:
1. 规则引擎：直接定位到具体设备（已有）
2. 状态估计：通过残差分析定位到具体量测→对应设备
3. GNN：通过注意力权重定位到具体节点/边
"""
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DeviceLocalizer:
    """设备级异常定位器"""

    def __init__(self, graph: nx.Graph = None, net=None):
        self.graph = graph
        self.net = net

    def localize(self, anomalies: List[Dict], network_data: Dict) -> List[Dict]:
        """
        将异常列表中的每个异常精确定位到设备级

        Args:
            anomalies: 检测到的异常列表
            network_data: 网络数据字典

        Returns:
            增强后的异常列表（含设备级信息）
        """
        enhanced = []
        for a in anomalies:
            ea = dict(a)  # copy
            device_info = self._localize_single(a, network_data)
            ea.update(device_info)
            enhanced.append(ea)
        return enhanced

    def _localize_single(self, anomaly: Dict, network_data: Dict) -> Dict:
        """定位单个异常到设备级"""
        atype = anomaly.get("type", "")
        location = anomaly.get("location", "")
        layer = anomaly.get("layer", "")

        info = {
            "device_type": "unknown",
            "device_name": "",
            "device_id": "",
            "connected_devices": [],
            "electrical_context": {},
        }

        # 1. 拓扑中断 → 定位到具体母线和关联线路
        if atype in ("拓扑中断",):
            info.update(self._localize_topology_interrupt(anomaly, network_data))

        # 2. 虚接/错接 → 定位到具体线路和连接节点
        elif atype in ("虚接/错接",):
            info.update(self._localize_virtual_faulty(anomaly, network_data))

        # 3. 遥测!=拓扑 → 定位到具体量测设备
        elif atype in ("遥测!=拓扑", "不良数据", "chi2检验未通过"):
            info.update(self._localize_measurement_error(anomaly, network_data))

        # 4. 图模不符 → 定位到具体缺失/多余的设备
        elif atype in ("图模不符",):
            info.update(self._localize_model_mismatch(anomaly, network_data))

        # 5. 遥信!=遥测 → 定位到具体开关
        elif atype in ("遥信!=遥测",):
            info.update(self._localize_signal_mismatch(anomaly, network_data))

        return info

    def _localize_topology_interrupt(self, anomaly: Dict, data: Dict) -> Dict:
        """拓扑中断定位: 找到断开的具体设备和关联元件"""
        graph = data.get("graph")
        if not graph:
            return {}

        # 找到所有不可达节点
        sources = [n for n in graph.nodes() if graph.nodes[n].get("is_source", False)]
        if not sources:
            sources = [n for n in graph.nodes() if "slack" in str(n).lower() or "source" in str(n).lower()]
        if not sources and graph.nodes():
            sources = [list(graph.nodes())[0]]

        reachable = set()
        for s in sources:
            if s in graph:
                reachable |= set(nx.bfs_tree(graph, s).nodes())

        unreachable = set(graph.nodes()) - reachable
        if not unreachable:
            return {"device_type": "bus", "device_name": str(anomaly.get("location", ""))}

        # 找到断开的边（连接可达和不可达节点的边）
        bridge_edges = []
        for u, v in graph.edges():
            if (u in reachable and v in unreachable) or (u in unreachable and v in reachable):
                bridge_edges.append((u, v))

        return {
            "device_type": "bus",
            "device_name": str(list(unreachable)[:3]),
            "unreachable_count": len(unreachable),
            "bridge_edges": bridge_edges[:5],
            "connected_devices": list(unreachable)[:10],
            "electrical_context": {
                "impact": "负荷失电" if unreachable else "无影响",
                "affected_buses": len(unreachable),
            },
        }

    def _localize_virtual_faulty(self, anomaly: Dict, data: Dict) -> Dict:
        """虚接/错接定位: 找到异常连接的具体线路"""
        graph = data.get("graph")
        if not graph:
            return {}

        # 检查度异常的节点
        high_degree_nodes = []
        for node, deg in graph.degree():
            if deg > 5:
                high_degree_nodes.append({"node": node, "degree": deg})

        # 检查环路
        cycles = []
        if not nx.is_tree(graph):
            try:
                cycles = nx.cycle_basis(graph)
            except:
                pass

        target = str(anomaly.get("location", ""))
        return {
            "device_type": "line",
            "device_name": target,
            "high_degree_nodes": high_degree_nodes[:3],
            "cycles_found": len(cycles),
            "electrical_context": {
                "issue": "连接关系异常",
                "suggestion": "检查线路两端连接是否正确",
            },
        }

    def _localize_measurement_error(self, anomaly: Dict, data: Dict) -> Dict:
        """量测异常定位: 找到具体哪个量测设备有问题"""
        location = str(anomaly.get("location", ""))

        # 从location解析设备信息
        device_type = "bus"
        device_id = ""
        if "bus_" in location:
            device_type = "bus"
            device_id = location.replace("bus_", "")
        elif "line_" in location:
            device_type = "line"
            device_id = location.replace("line_", "")
        elif "trafo_" in location:
            device_type = "transformer"
            device_id = location.replace("trafo_", "")

        # 从量测数据中获取该设备的量测值
        measurements = data.get("measurements", {})
        meas_value = None
        if device_type == "bus":
            for bv in measurements.get("bus_voltages", []):
                if str(bv.get("bus", "")) == device_id:
                    meas_value = bv
                    break

        return {
            "device_type": device_type,
            "device_name": "{}_{}".format(device_type, device_id),
            "device_id": device_id,
            "measurement": meas_value,
            "electrical_context": {
                "issue": "量测值与估计值偏差过大",
                "value": meas_value.get("vm_pu", "") if meas_value else "",
            },
        }

    def _localize_model_mismatch(self, anomaly: Dict, data: Dict) -> Dict:
        """图模不符定位: 找到具体缺失/多余的设备"""
        location = str(anomaly.get("location", ""))
        details = anomaly.get("details", "")

        return {
            "device_type": "mixed",
            "device_name": location,
            "electrical_context": {
                "issue": "CIM与SVG设备不一致",
                "details": details[:100],
            },
        }

    def _localize_signal_mismatch(self, anomaly: Dict, data: Dict) -> Dict:
        """遥信异常定位: 找到具体开关"""
        location = str(anomaly.get("location", ""))

        device_type = "switch"
        if "switch_" in location:
            device_type = "switch"
        elif "line_" in location:
            device_type = "line"

        return {
            "device_type": device_type,
            "device_name": location,
            "electrical_context": {
                "issue": "遥信状态与量测推断不一致",
                "suggestion": "检查开关辅助接点和通信状态",
            },
        }
