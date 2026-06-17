# -*- coding: utf-8 -*-
"""
多源数据对齐器
将 CIM、SVG、SCADA 三源数据的设备ID进行映射和对齐
实现 CIM-SVG 设备一一对应，SCADA量测挂载到拓扑元素上
"""
from typing import Dict, List, Set, Tuple, Optional
from difflib import SequenceMatcher
import re
import logging

logger = logging.getLogger(__name__)


class DataAligner:
    """多源数据对齐器"""

    def __init__(self, fuzzy_threshold: float = 0.85):
        """
        Args:
            fuzzy_threshold: 模糊匹配阈值（0~1），用于名称相似度匹配
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.id_mapping = {}      # {cim_uri: svg_id}
        self.unmatched_cim = set()
        self.unmatched_svg = set()

    def align_cim_svg(self, cim_devices: List[Dict],
                      svg_devices: List[Dict]) -> Dict:
        """
        对齐CIM设备与SVG设备

        对齐策略:
        1. 精确ID匹配（去掉URI前缀后比对）
        2. 名称精确匹配
        3. 模糊名称匹配（SequenceMatcher）

        返回:
          mapping    - {cim_uri: svg_id} 已匹配映射
          cim_only   - CIM中有但SVG中缺失的设备
          svg_only   - SVG中有但CIM中缺失的设备
          fuzzy      - 模糊匹配的设备 [(cim_uri, svg_id, score)]
        """
        mapping = {}
        fuzzy_matches = []
        cim_matched = set()
        svg_matched = set()

        # 构建索引
        cim_name_map = {}  # name -> cim_device
        for dev in cim_devices:
            cim_name_map[dev["name"]] = dev

        svg_name_map = {}
        for dev in svg_devices:
            svg_name_map[dev["id"]] = dev

        # --- 阶段1: 精确匹配 ---
        for cim_dev in cim_devices:
            cim_short = _extract_short_id(cim_dev["uri"])
            cim_name = cim_dev["name"]

            for svg_dev in svg_devices:
                svg_id = svg_dev["id"]

                # URI短ID == SVG ID
                if cim_short and cim_short == svg_id:
                    mapping[cim_dev["uri"]] = svg_id
                    cim_matched.add(cim_dev["uri"])
                    svg_matched.add(svg_id)
                    break

                # CIM name == SVG label 或 SVG id
                if cim_name and (cim_name == svg_id or
                                 cim_name == svg_dev.get("label", "")):
                    mapping[cim_dev["uri"]] = svg_id
                    cim_matched.add(cim_dev["uri"])
                    svg_matched.add(svg_id)
                    break

        # --- 阶段2: 模糊匹配 ---
        for cim_dev in cim_devices:
            if cim_dev["uri"] in cim_matched:
                continue
            best_score = 0.0
            best_svg = None
            for svg_dev in svg_devices:
                if svg_dev["id"] in svg_matched:
                    continue
                score = self._similarity(cim_dev["name"], svg_dev["id"])
                if score > best_score:
                    best_score = score
                    best_svg = svg_dev["id"]

            if best_svg and best_score >= self.fuzzy_threshold:
                fuzzy_matches.append((cim_dev["uri"], best_svg, best_score))
                mapping[cim_dev["uri"]] = best_svg
                cim_matched.add(cim_dev["uri"])
                svg_matched.add(best_svg)

        # 未匹配的
        cim_only = [d["uri"] for d in cim_devices if d["uri"] not in cim_matched]
        svg_only = [d["id"] for d in svg_devices if d["id"] not in svg_matched]

        self.id_mapping = mapping
        self.unmatched_cim = set(cim_only)
        self.unmatched_svg = set(svg_only)

        logger.info(f"CIM-SVG对齐: 匹配={len(mapping)}, "
                    f"CIM独有={len(cim_only)}, SVG独有={len(svg_only)}, "
                    f"模糊匹配={len(fuzzy_matches)}")

        return {
            "mapping": mapping,
            "cim_only": cim_only,
            "svg_only": svg_only,
            "fuzzy": fuzzy_matches,
        }

    def align_scada_to_topology(self, measurements: Dict,
                                 bus_mapping: Dict[int, str]) -> Dict:
        """
        将SCADA量测挂载到拓扑元素

        Args:
            measurements: SCADASimulator.generate_measurements() 的输出
            bus_mapping: {pandapower_bus_idx: cim_node_uri}

        返回:
          bus_meas  - [{node_uri, vm_pu, ...}]
          line_meas - [{line_uri, p_mw, ...}]
        """
        bus_meas = []
        for bv in measurements.get("bus_voltages", []):
            node_uri = bus_mapping.get(bv["bus"], f"bus_{bv['bus']}")
            bus_meas.append({**bv, "node_uri": node_uri})

        line_meas = []
        for lp in measurements.get("line_powers", []):
            line_uri = f"line_{lp['line']}"
            line_meas.append({**lp, "line_uri": line_uri})

        return {"bus_meas": bus_meas, "line_meas": line_meas}

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """计算两个字符串的相似度"""
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _extract_short_id(uri: str) -> str:
    """从URI中提取短ID（#后或最后一段）"""
    if "#" in uri:
        return uri.split("#")[-1]
    return uri.split("/")[-1]


def build_bus_id_map(net) -> Dict[int, str]:
    """
    为PandaPower网络构建 bus_idx -> 假想CIM节点URI 映射
    实际项目中应从CIM文件解析获得
    """
    return {int(idx): f"#_bus_{idx}" for idx in net.bus.index}
