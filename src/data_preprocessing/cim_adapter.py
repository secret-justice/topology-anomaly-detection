# -*- coding: utf-8 -*-
"""
CIM数据适配器
解析CIM XML/RDF格式，提取拓扑和量测信息，转换为PandaPower网络对象

遵循IEC 61970/61968标准，支持CIM16和CIM100命名空间
依赖已有 cim_parser.py 的 RDF 解析能力
"""
import os
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import pandapower as pp
import networkx as nx

from data_preprocessing.cim_parser import (
    parse_cim_rdf, parse_cim_directory, get_device_connectivity,
)

logger = logging.getLogger(__name__)

# CIM设备子类 -> PandaPower元素类型
_CIM_TO_PP_TYPE = {
    "ACLineSegment":          "line",
    "PowerTransformer":       "trafo",
    "EnergyConsumer":         "load",
    "EnergySource":           "ext_grid",
    "SynchronousMachine":     "gen",
    "LinearShuntCompensator": "shunt",
    "PowerElectronicsConnection": "sgen",
    "Breaker":                "switch",
    "Disconnector":           "switch",
    "LoadBreakSwitch":        "switch",
    "Fuse":                   "switch",
    "Recloser":               "switch",
}

# 默认电气参数（当CIM未提供时使用）
_VOLTAGE_DEFAULT_PARAMS = {
    (0, 1):   {"r_ohm_per_km": 0.50, "x_ohm_per_km": 0.10, "max_i_ka": 0.20},
    (1, 10):  {"r_ohm_per_km": 0.30, "x_ohm_per_km": 0.15, "max_i_ka": 0.40},
    (10, 35): {"r_ohm_per_km": 0.15, "x_ohm_per_km": 0.12, "max_i_ka": 0.60},
    (35, 110):{"r_ohm_per_km": 0.05, "x_ohm_per_km": 0.10, "max_i_ka": 1.50},
    (110,220):{"r_ohm_per_km": 0.02, "x_ohm_per_km": 0.15, "max_i_ka": 3.00},
}

def _get_voltage_default(vn_kv, param_name):
    for (vmin, vmax), params in _VOLTAGE_DEFAULT_PARAMS.items():
        if vmin <= vn_kv < vmax:
            return params.get(param_name, 0.5)
    return 0.5

_DEFAULT_LINE_PARAMS = {
    "r_ohm_per_km": 0.5,
    "x_ohm_per_km": 0.4,
    "c_nf_per_km": 10.0,
    "max_i_ka": 0.4,
    "type": "cs",
    "std_type": None,
}
_DEFAULT_TRAFO_PARAMS = {
    "sn_mva": 25.0,
    "vn_hv_kv": 110.0,
    "vn_lv_kv": 20.0,
    "vk_percent": 10.0,
    "vkr_percent": 0.5,
    "pfe_kw": 10.0,
    "i0_percent": 0.1,
    "shift_degree": 0.0,
}


class CIMAdapter:
    """
    CIM数据适配器

    将CIM XML/RDF文件解析并转换为PandaPower网络对象。
    支持:
      - 解析单个或目录级CIM文件
      - 提取母线、线路、变压器、开关、负荷、发电机
      - 提取量测信息（电压、功率）
      - 转换为可运行潮流的PandaPower net
    """

    def __init__(self, default_vn_kv: float = 20.0):
        """
        Args:
            default_vn_kv: 当CIM未提供额定电压时的默认值(kV)
        """
        self.default_vn_kv = default_vn_kv
        self.cim_data: Optional[Dict] = None
        self.node_bus_map: Dict[str, int] = {}   # connectivity_node_uri -> pp bus idx
        self.device_pp_map: Dict[str, str] = {}  # cim_device_uri -> pp element desc
        self.warnings: List[str] = []
        self.version_report: Dict = {}
        self._default_param_count = 0
        self._valid_param_count = 0

    # ================================================================
    # 公开接口
    # ================================================================

    def load_cim(self, path: str) -> Dict:
        """
        加载CIM数据（单文件或目录）

        Args:
            path: CIM XML/RDF 文件路径或目录路径

        Returns:
            cim_data 字典 (devices, terminals, nodes, switches, connections)
        """
        if os.path.isdir(path):
            self.cim_data = parse_cim_directory(path)
        else:
            self.cim_data = parse_cim_rdf(path)

        logger.info(
            f"CIM加载完成: 设备={len(self.cim_data['devices'])}, "
            f"节点={len(self.cim_data['nodes'])}, "
            f"开关={len(self.cim_data['switches'])}"
        )
        self.version_report = self._detect_cim_version(self.cim_data)
        logger.info("CIM version: %s", self.version_report.get("detected_version", "Unknown"))
        return self.cim_data

    def _detect_cim_version(self, cim_data):
        """Auto-detect CIM version from RDF namespaces."""
        raw = str(cim_data.get("namespaces", ""))
        if "cim17" in raw.lower():
            version = "CIM17"
        elif "cim16" in raw.lower():
            version = "CIM16"
        elif "CIM100" in raw:
            version = "CIM100"
        elif "entsoe" in raw.lower():
            version = "CGMES3.0" if "2019" in raw else "CGMES2.4"
        else:
            version = "CIM16_or_later"
        return {"detected_version": version, "supported": True,
                "param_warnings": list(self.warnings)}

    def _validate_line_params(self, r_ohm, x_ohm, length_km, max_i_ka, vn_kv):
        """Validate line parameters for reasonableness."""
        w = []
        rx = r_ohm / max(x_ohm, 1e-6)
        if rx > 5.0: w.append(f"R/X={rx:.2f} high")
        if rx < 0.01: w.append(f"R/X={rx:.4f} low")
        if r_ohm < 0: w.append(f"R={r_ohm} negative")
        if x_ohm < 0: w.append(f"X={x_ohm} negative")
        if max_i_ka > 10.0: w.append(f"I={max_i_ka}kA high")
        if max_i_ka < 0.001: w.append(f"I={max_i_ka}kA low")
        if length_km > 100: w.append(f"L={length_km}km long")
        return w

    def to_pandapower(self, cim_data: Optional[Dict] = None) -> pp.pandapowerNet:
        """
        将CIM数据转换为PandaPower网络

        Args:
            cim_data: 可选，直接传入cim_data字典。若为None则使用self.cim_data

        Returns:
            PandaPower网络对象（已添加元素但未运行潮流）
        """
        data = cim_data or self.cim_data
        if data is None:
            raise ValueError("请先调用 load_cim() 或传入 cim_data")

        self.cim_data = data
        self.warnings = []
        self.node_bus_map = {}
        self.device_pp_map = {}

        net = pp.create_empty_network(name="CIM_imported")

        # Step 1: 从连接节点创建母线
        self._create_buses(net, data)

        # Step 2: 从设备分类创建PandaPower元素
        self._create_lines(net, data)
        self._create_transformers(net, data)
        self._create_loads(net, data)
        self._create_generators(net, data)
        self._create_switches(net, data)
        self._create_shunts(net, data)
        self._create_sgens(net, data)

        logger.info(
            f"PandaPower网络创建完成: "
            f"母线={len(net.bus)}, 线路={len(net.line)}, "
            f"变压器={len(net.trafo)}, 负荷={len(net.load)}, "
            f"发电机={len(net.gen)}, 开关={len(net.switch)}"
        )
        if self.warnings:
            logger.warning(f"转换过程中有 {len(self.warnings)} 条警告")

        return net

    def extract_measurements(self, cim_data: Optional[Dict] = None) -> Dict:
        """
        从CIM数据中提取量测信息

        Returns:
            {bus_voltages: [...], line_powers: [...], metadata: {...}}
        """
        data = cim_data or self.cim_data
        if data is None:
            return {"bus_voltages": [], "line_powers": [], "metadata": {}}

        bus_voltages = []
        line_powers = []

        # 尝试从CIM-Analog/Measurement对象提取量测
        for dev in data.get("devices", []):
            subtype = dev.get("subtype", "")
            uri = dev.get("uri", "")
            name = dev.get("name", "")

            # 模拟量测对象 (CIM Analog)
            if subtype in ("Analog", "Measurement"):
                meas_type = dev.get("measurement_type", "")
                value = dev.get("value", None)
                if value is not None:
                    if "voltage" in meas_type.lower() or "vm" in name.lower():
                        bus_voltages.append({
                            "bus": self.node_bus_map.get(uri, -1),
                            "vm_pu": float(value),
                            "sigma": 0.005,
                        })
                    elif "power" in meas_type.lower() or "p_" in name.lower():
                        line_powers.append({
                            "line": -1,
                            "p_mw": float(value),
                            "sigma": 0.05,
                        })

        # 如果CIM中没有显式量测，返回空（需外部SCADA补充）
        if not bus_voltages and not line_powers:
            self.warnings.append("CIM中未找到显式量测数据，需外部SCADA补充")

        return {
            "bus_voltages": bus_voltages,
            "line_powers": line_powers,
            "metadata": {
                "source": "cim",
                "n_devices": len(data.get("devices", [])),
                "n_measurements": len(bus_voltages) + len(line_powers),
            },
        }

    def get_topology_graph(self, cim_data: Optional[Dict] = None) -> nx.Graph:
        """
        从CIM数据构建拓扑图

        Returns:
            NetworkX无向图
        """
        from utils.graph_utils import build_graph_from_cim
        data = cim_data or self.cim_data
        if data is None:
            raise ValueError("请先加载CIM数据")
        return build_graph_from_cim(data)

    def get_alignment_info(self) -> Dict:
        """返回CIM设备与PandaPower元素的对齐信息"""
        return {
            "node_bus_map": dict(self.node_bus_map),
            "device_pp_map": dict(self.device_pp_map),
            "warnings": list(self.warnings),
        }

    # ================================================================
    # 内部方法: PandaPower元素创建
    # ================================================================

    def _create_buses(self, net, data):
        """从CIM ConnectivityNode创建母线"""
        nodes = data.get("nodes", [])
        vn_map = self._infer_bus_voltages(data)

        for i, node in enumerate(nodes):
            uri = node["uri"]
            name = node.get("name", f"Node_{i}")
            vn_kv = vn_map.get(uri, self.default_vn_kv)
            bus_idx = pp.create_bus(net, vn_kv=vn_kv, name=name)
            self.node_bus_map[uri] = bus_idx

        logger.info(f"创建母线: {len(self.node_bus_map)}")

    def _create_lines(self, net, data):
        """从ACLineSegment创建线路"""
        dev_nodes = self._build_device_node_map(data)

        for dev in data.get("devices", []):
            if dev["subtype"] != "ACLineSegment":
                continue
            uri = dev["uri"]
            name = dev.get("name", "")
            nodes_list = dev_nodes.get(uri, [])
            if len(nodes_list) < 2:
                self.warnings.append(f"线路 {name} 连接节点不足2个，跳过")
                continue

            from_bus = self.node_bus_map.get(nodes_list[0])
            to_bus = self.node_bus_map.get(nodes_list[1])
            if from_bus is None or to_bus is None:
                self.warnings.append(f"线路 {name} 无法映射到母线，跳过")
                continue

            params = self._extract_line_params(dev)
            try:
                pp.create_line_from_parameters(
                    net, from_bus=from_bus, to_bus=to_bus,
                    length_km=params.get("length_km", 1.0),
                    r_ohm_per_km=params.get("r_ohm_per_km", _get_voltage_default(vn_kv, "r_ohm_per_km")),
                    x_ohm_per_km=params.get("x_ohm_per_km", _get_voltage_default(vn_kv, "x_ohm_per_km")),
                    c_nf_per_km=params.get("c_nf_per_km", _DEFAULT_LINE_PARAMS["c_nf_per_km"]),
                    max_i_ka=params.get("max_i_ka", _get_voltage_default(vn_kv, "max_i_ka")),
                    name=name,
                )
                self.device_pp_map[uri] = f"line_{len(net.line)-1}"
            except Exception as e:
                self.warnings.append(f"创建线路 {name} 失败: {e}")

        logger.info(f"创建线路: {len(net.line)}")

    def _create_transformers(self, net, data):
        """从PowerTransformer创建变压器"""
        dev_nodes = self._build_device_node_map(data)
        trafo_end_map = self._build_trafo_end_map(data)

        for dev in data.get("devices", []):
            if dev["subtype"] != "PowerTransformer":
                continue
            uri = dev["uri"]
            name = dev.get("name", "")

            ends = trafo_end_map.get(uri, [])
            if len(ends) < 2:
                nodes_list = dev_nodes.get(uri, [])
                if len(nodes_list) >= 2:
                    hv_bus = self.node_bus_map.get(nodes_list[0])
                    lv_bus = self.node_bus_map.get(nodes_list[1])
                else:
                    self.warnings.append(f"变压器 {name} 绕组信息不足，跳过")
                    continue
            else:
                ends_sorted = sorted(ends, key=lambda e: e.get("sequence", 99))
                hv_node = ends_sorted[0].get("node")
                lv_node = ends_sorted[-1].get("node")
                hv_bus = self.node_bus_map.get(hv_node) if hv_node else None
                lv_bus = self.node_bus_map.get(lv_node) if lv_node else None

            if hv_bus is None or lv_bus is None:
                self.warnings.append(f"变压器 {name} 无法映射到母线，跳过")
                continue

            params = self._extract_trafo_params(dev, ends)
            try:
                pp.create_transformer_from_parameters(
                    net, hv_bus=hv_bus, lv_bus=lv_bus,
                    sn_mva=params.get("sn_mva", _DEFAULT_TRAFO_PARAMS["sn_mva"]),
                    vn_hv_kv=params.get("vn_hv_kv", _DEFAULT_TRAFO_PARAMS["vn_hv_kv"]),
                    vn_lv_kv=params.get("vn_lv_kv", _DEFAULT_TRAFO_PARAMS["vn_lv_kv"]),
                    vk_percent=params.get("vk_percent", _DEFAULT_TRAFO_PARAMS["vk_percent"]),
                    vkr_percent=params.get("vkr_percent", _DEFAULT_TRAFO_PARAMS["vkr_percent"]),
                    pfe_kw=params.get("pfe_kw", _DEFAULT_TRAFO_PARAMS["pfe_kw"]),
                    i0_percent=params.get("i0_percent", _DEFAULT_TRAFO_PARAMS["i0_percent"]),
                    shift_degree=params.get("shift_degree", _DEFAULT_TRAFO_PARAMS["shift_degree"]),
                    name=name,
                )
                self.device_pp_map[uri] = f"trafo_{len(net.trafo)-1}"
            except Exception as e:
                self.warnings.append(f"创建变压器 {name} 失败: {e}")

        logger.info(f"创建变压器: {len(net.trafo)}")

    def _create_loads(self, net, data):
        """从EnergyConsumer创建负荷"""
        dev_nodes = self._build_device_node_map(data)

        for dev in data.get("devices", []):
            if dev["subtype"] != "EnergyConsumer":
                continue
            uri = dev["uri"]
            name = dev.get("name", "")
            nodes_list = dev_nodes.get(uri, [])
            if not nodes_list:
                self.warnings.append(f"负荷 {name} 无连接节点，跳过")
                continue

            bus = self.node_bus_map.get(nodes_list[0])
            if bus is None:
                continue

            p_mw = dev.get("p_mw", 0.1)
            q_mvar = dev.get("q_mvar", 0.05)
            try:
                pp.create_load(net, bus=bus, p_mw=p_mw, q_mvar=q_mvar, name=name)
                self.device_pp_map[uri] = f"load_{len(net.load)-1}"
            except Exception as e:
                self.warnings.append(f"创建负荷 {name} 失败: {e}")

        logger.info(f"创建负荷: {len(net.load)}")

    def _create_generators(self, net, data):
        """从EnergySource/SynchronousMachine创建发电机/外部电网"""
        dev_nodes = self._build_device_node_map(data)

        for dev in data.get("devices", []):
            uri = dev["uri"]
            name = dev.get("name", "")
            subtype = dev["subtype"]
            nodes_list = dev_nodes.get(uri, [])
            if not nodes_list:
                continue

            bus = self.node_bus_map.get(nodes_list[0])
            if bus is None:
                continue

            if subtype == "EnergySource":
                try:
                    pp.create_ext_grid(net, bus=bus, vm_pu=1.0, name=name)
                    self.device_pp_map[uri] = f"ext_grid_{len(net.ext_grid)-1}"
                except Exception as e:
                    self.warnings.append(f"创建外部电网 {name} 失败: {e}")

            elif subtype == "SynchronousMachine":
                p_mw = dev.get("p_mw", 1.0)
                try:
                    pp.create_gen(net, bus=bus, p_mw=p_mw, vm_pu=1.0, name=name)
                    self.device_pp_map[uri] = f"gen_{len(net.gen)-1}"
                except Exception as e:
                    self.warnings.append(f"创建发电机 {name} 失败: {e}")

        logger.info(f"创建发电机/外部电网: ext_grid={len(net.ext_grid)}, gen={len(net.gen)}")

    def _create_switches(self, net, data):
        """从CIM开关设备创建PandaPower开关"""
        dev_nodes = self._build_device_node_map(data)

        for sw in data.get("switches", []):
            uri = sw["uri"]
            name = sw.get("name", "")
            normal_open = sw.get("normal_open", False)
            nodes_list = dev_nodes.get(uri, [])
            if len(nodes_list) < 2:
                continue

            bus = self.node_bus_map.get(nodes_list[0])
            element = self.node_bus_map.get(nodes_list[1])
            if bus is None or element is None:
                continue

            try:
                pp.create_switch(
                    net, bus=bus, element=element, et="b",
                    closed=not normal_open, name=name,
                )
                self.device_pp_map[uri] = f"switch_{len(net.switch)-1}"
            except Exception as e:
                self.warnings.append(f"创建开关 {name} 失败: {e}")

        logger.info(f"创建开关: {len(net.switch)}")

    def _create_shunts(self, net, data):
        """从LinearShuntCompensator创建并联补偿器"""
        dev_nodes = self._build_device_node_map(data)

        for dev in data.get("devices", []):
            if dev["subtype"] != "LinearShuntCompensator":
                continue
            uri = dev["uri"]
            name = dev.get("name", "")
            nodes_list = dev_nodes.get(uri, [])
            if not nodes_list:
                continue
            bus = self.node_bus_map.get(nodes_list[0])
            if bus is None:
                continue

            q_mvar = dev.get("q_mvar", 0.5)
            try:
                pp.create_shunt(net, bus=bus, q_mvar=q_mvar, p_mw=0.0, name=name)
                self.device_pp_map[uri] = f"shunt_{len(net.shunt)-1}"
            except Exception as e:
                self.warnings.append(f"创建并联补偿器 {name} 失败: {e}")

    def _create_sgens(self, net, data):
        """从PowerElectronicsConnection创建静态发电机"""
        dev_nodes = self._build_device_node_map(data)

        for dev in data.get("devices", []):
            if dev["subtype"] != "PowerElectronicsConnection":
                continue
            uri = dev["uri"]
            name = dev.get("name", "")
            nodes_list = dev_nodes.get(uri, [])
            if not nodes_list:
                continue
            bus = self.node_bus_map.get(nodes_list[0])
            if bus is None:
                continue

            p_mw = dev.get("p_mw", 0.5)
            try:
                pp.create_sgen(net, bus=bus, p_mw=p_mw, name=name)
                self.device_pp_map[uri] = f"sgen_{len(net.sgen)-1}"
            except Exception as e:
                self.warnings.append(f"创建静态发电机 {name} 失败: {e}")

    # ================================================================
    # 辅助方法
    # ================================================================

    def _build_device_node_map(self, data):
        """构建设备URI -> 连接节点URI列表 映射"""
        dev_nodes = defaultdict(list)
        for conn in data.get("connections", []):
            dev = conn["device"]
            node = conn["node"]
            if node not in dev_nodes[dev]:
                dev_nodes[dev].append(node)
        return dict(dev_nodes)

    def _build_trafo_end_map(self, data):
        """构建变压器 -> 绕组端信息 映射"""
        trafo_ends = defaultdict(list)
        for dev in data.get("devices", []):
            if dev["subtype"] == "PowerTransformerEnd":
                parent = dev.get("transformer_uri", dev.get("parent_uri", ""))
                if parent:
                    trafo_ends[parent].append({
                        "uri": dev["uri"],
                        "name": dev.get("name", ""),
                        "sequence": dev.get("sequence", 1),
                        "rated_s_mva": dev.get("rated_s_mva", None),
                        "rated_u_kv": dev.get("rated_u_kv", None),
                        "node": None,
                    })

        for end_list in trafo_ends.values():
            for end_info in end_list:
                end_uri = end_info["uri"]
                for conn in data.get("connections", []):
                    if conn["device"] == end_uri:
                        end_info["node"] = conn["node"]
                        break

        return dict(trafo_ends)

    def _infer_bus_voltages(self, data):
        """从设备属性推断各节点的额定电压"""
        vn_map = {}
        dev_nodes = self._build_device_node_map(data)

        for dev in data.get("devices", []):
            rated_u = dev.get("rated_u_kv", dev.get("base_voltage", None))
            if rated_u is None:
                continue
            nodes_list = dev_nodes.get(dev["uri"], [])
            for node_uri in nodes_list:
                if node_uri not in vn_map:
                    vn_map[node_uri] = float(rated_u)

        return vn_map

    def _extract_line_params(self, dev):
        """从CIM设备属性提取线路参数"""
        params = {}
        for key in ("length_km", "r_ohm_per_km", "x_ohm_per_km",
                     "c_nf_per_km", "max_i_ka"):
            val = dev.get(key)
            if val is not None:
                params[key] = float(val)
        return params

    def _extract_trafo_params(self, dev, ends):
        """从CIM设备属性和绕组信息提取变压器参数"""
        params = {}
        if ends:
            hv_end = ends[0] if ends else {}
            lv_end = ends[-1] if len(ends) > 1 else ends[0] if ends else {}
            if hv_end.get("rated_s_mva"):
                params["sn_mva"] = float(hv_end["rated_s_mva"])
            if hv_end.get("rated_u_kv"):
                params["vn_hv_kv"] = float(hv_end["rated_u_kv"])
            if lv_end.get("rated_u_kv"):
                params["vn_lv_kv"] = float(lv_end["rated_u_kv"])

        for key in ("sn_mva", "vn_hv_kv", "vn_lv_kv", "vk_percent",
                     "vkr_percent", "pfe_kw", "i0_percent", "shift_degree"):
            val = dev.get(key)
            if val is not None:
                params[key] = float(val)
        return params
