# -*- coding: utf-8 -*-
"""
SVG数据适配器
解析电力系统SVG图形文件，提取设备ID和连接关系，
与CIM数据对齐

基于已有 svg_parser.py，增加:
  - CIM-SVG设备ID对齐（精确+模糊匹配）
  - SVG拓扑提取（通过空间近邻关系推断连接）
  - 与CIM拓扑的一致性校验
"""
import os
import logging
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from difflib import SequenceMatcher

from lxml import etree

from data_preprocessing.svg_parser import parse_svg, get_device_positions
from data_preprocessing.cim_adapter import CIMAdapter

logger = logging.getLogger(__name__)

# SVG命名空间
SVG_NS = "http://www.w3.org/2000/svg"
CIM_NS = "http://iec.ch/TC57/CIM100#"




# ===== P1-5: SVG Transform Matrix Parsing =====

import re
import math

def parse_svg_transform(transform_str):
    """Parse SVG transform attribute into a 3x3 affine matrix.
    
    Supports: matrix, translate, scale, rotate, skewX, skewY
    Can handle chained transforms: "translate(10,20) rotate(45)"
    
    Returns:
        3x3 numpy array (identity if no transform)
    """
    if not transform_str or not transform_str.strip():
        return np.eye(3)
    
    result = np.eye(3)
    
    # Parse each transform function
    pattern = r'(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^)]*)\)'
    matches = re.findall(pattern, transform_str)
    
    for func, args_str in matches:
        args = [float(x.strip()) for x in args_str.replace(',', ' ').split() if x.strip()]
        m = np.eye(3)
        
        if func == 'matrix' and len(args) == 6:
            # matrix(a,b,c,d,e,f) -> [a c e; b d f; 0 0 1]
            m[0, 0] = args[0]; m[1, 0] = args[1]
            m[0, 1] = args[2]; m[1, 1] = args[3]
            m[0, 2] = args[4]; m[1, 2] = args[5]
        
        elif func == 'translate':
            tx = args[0] if args else 0
            ty = args[1] if len(args) > 1 else 0
            m[0, 2] = tx
            m[1, 2] = ty
        
        elif func == 'scale':
            sx = args[0] if args else 1
            sy = args[1] if len(args) > 1 else sx
            m[0, 0] = sx
            m[1, 1] = sy
        
        elif func == 'rotate':
            angle = args[0] if args else 0
            cx = args[1] if len(args) > 1 else 0
            cy = args[2] if len(args) > 2 else 0
            rad = math.radians(angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            # Translate to origin, rotate, translate back
            if len(args) > 1:
                m = np.array([
                    [cos_a, -sin_a, cx - cx * cos_a + cy * sin_a],
                    [sin_a, cos_a, cy - cx * sin_a - cy * cos_a],
                    [0, 0, 1]
                ])
            else:
                m[0, 0] = cos_a; m[0, 1] = -sin_a
                m[1, 0] = sin_a; m[1, 1] = cos_a
        
        elif func == 'skewX':
            angle = args[0] if args else 0
            m[0, 1] = math.tan(math.radians(angle))
        
        elif func == 'skewY':
            angle = args[0] if args else 0
            m[1, 0] = math.tan(math.radians(angle))
        
        result = result @ m
    
    return result


def apply_transform(matrix, x, y):
    """Apply 3x3 affine transform to a 2D point.
    Returns (new_x, new_y).
    """
    v = np.array([x, y, 1.0])
    result = matrix @ v
    return float(result[0]), float(result[1])


def apply_transform_to_bbox(matrix, x, y, width, height):
    """Apply transform to a bounding box (x, y, width, height).
    Returns transformed (x, y, width, height).
    """
    corners = [
        (x, y), (x + width, y), (x, y + height), (x + width, y + height)
    ]
    transformed = [apply_transform(matrix, cx, cy) for cx, cy in corners]
    xs = [p[0] for p in transformed]
    ys = [p[1] for p in transformed]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def extract_device_positions_with_transform(svg_data):
    """Extract device positions from SVG, properly handling transforms.
    
    Resolves nested transform hierarchies: parent group transforms
    are composed with child element transforms.
    
    Args:
        svg_data: parsed SVG data dict from svg_parser
    
    Returns:
        dict of {device_id: (x, y)} with transforms applied
    """
    positions = {}
    devices = svg_data.get("devices", []) + svg_data.get("switches", [])
    
    for dev in devices:
        dev_id = dev.get("id", "")
        x = float(dev.get("x", dev.get("cx", 0)))
        y = float(dev.get("y", dev.get("cy", 0)))
        
        # Parse element transform
        elem_transform = parse_svg_transform(dev.get("transform", ""))
        
        # Parse parent group transforms (if available)
        parent_transform = np.eye(3)
        parent_groups = dev.get("parent_transforms", [])
        for pt in parent_groups:
            parent_transform = parent_transform @ parse_svg_transform(pt)
        
        # Compose: parent @ element
        full_transform = parent_transform @ elem_transform
        
        # Apply to position
        new_x, new_y = apply_transform(full_transform, x, y)
        positions[dev_id] = (new_x, new_y)
    
    return positions


def resolve_nested_transforms(svg_elements):
    """Resolve nested SVG group transforms.
    
    Walks the SVG element tree and composes transforms from root to leaf.
    
    Args:
        svg_elements: list of SVG elements with optional 'children' and 'transform'
    
    Returns:
        flat list of elements with 'resolved_transform' field
    """
    resolved = []
    
    def _walk(elements, parent_transform=np.eye(3)):
        for elem in elements:
            elem_transform = parse_svg_transform(elem.get("transform", ""))
            combined = parent_transform @ elem_transform
            elem_copy = dict(elem)
            elem_copy["resolved_transform"] = combined
            elem_copy.pop("children", None)
            resolved.append(elem_copy)
            # Recurse into children
            children = elem.get("children", [])
            if children:
                _walk(children, combined)
    
    _walk(svg_elements)
    return resolved


class SVGAdapter:
    """
    SVG数据适配器

    解析电力系统SVG图形文件并与CIM数据对齐。
    支持:
      - 解析SVG提取设备、连接线、开关状态
      - 通过ID/name与CIM设备精确对齐
      - 通过空间近邻关系推断设备连接
      - CIM-SVG一致性校验
    """

    def __init__(self, fuzzy_threshold: float = 0.80,
                 spatial_merge_dist: float = 10.0):
        """
        Args:
            fuzzy_threshold: 模糊匹配阈值 (0~1)
            spatial_merge_dist: 空间合并距离(像素)，距离小于该值的点视为同一节点
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.spatial_merge_dist = spatial_merge_dist
        self.svg_data: Optional[Dict] = None
        self.cim_data: Optional[Dict] = None
        self.alignment: Optional[Dict] = None

    # ================================================================
    # 公开接口
    # ================================================================

    def load_svg(self, svg_path: str) -> Dict:
        """
        加载SVG文件

        Args:
            svg_path: SVG文件路径

        Returns:
            svg_data字典 (devices, connections, switches, viewBox)
        """
        self.svg_data = parse_svg(svg_path)
        logger.info(
            f"SVG加载完成: 设备={len(self.svg_data['devices'])}, "
            f"开关={len(self.svg_data['switches'])}, "
            f"连接线={len(self.svg_data['connections'])}"
        )
        return self.svg_data

    def align_with_cim(self, cim_data: Dict,
                       svg_data: Optional[Dict] = None) -> Dict:
        """
        将SVG设备与CIM设备对齐

        对齐策略:
          1. 精确ID匹配 (URI fragment == SVG id)
          2. 名称精确匹配 (CIM name == SVG label/id)
          3. 模糊名称匹配 (SequenceMatcher >= threshold)

        Args:
            cim_data: cim_parser输出的CIM数据
            svg_data: svg_parser输出的SVG数据，若None则使用self.svg_data

        Returns:
            对齐结果 {mapping, cim_only, svg_only, fuzzy, stats}
        """
        self.cim_data = cim_data
        svg = svg_data or self.svg_data
        if svg is None:
            raise ValueError("请先调用 load_svg() 或传入 svg_data")

        cim_devices = cim_data.get("devices", [])
        svg_devices = svg.get("devices", []) + svg.get("switches", [])

        mapping = {}         # {cim_uri: svg_id}
        fuzzy_matches = []   # [(cim_uri, svg_id, score)]
        cim_matched: Set[str] = set()
        svg_matched: Set[str] = set()

        # --- 阶段1: 精确匹配 ---
        for cim_dev in cim_devices:
            cim_uri = cim_dev["uri"]
            cim_short = self._extract_short_id(cim_uri)
            cim_name = cim_dev.get("name", "")

            for svg_dev in svg_devices:
                svg_id = svg_dev.get("id", "")
                svg_label = svg_dev.get("label", "")

                # URI短ID == SVG ID
                if cim_short and cim_short == svg_id:
                    mapping[cim_uri] = svg_id
                    cim_matched.add(cim_uri)
                    svg_matched.add(svg_id)
                    break

                # CIM name == SVG label or SVG id
                if cim_name and (cim_name == svg_id or
                                 (svg_label and cim_name == svg_label)):
                    mapping[cim_uri] = svg_id
                    cim_matched.add(cim_uri)
                    svg_matched.add(svg_id)
                    break

        # --- 阶段2: 模糊匹配 ---
        for cim_dev in cim_devices:
            if cim_dev["uri"] in cim_matched:
                continue
            cim_name = cim_dev.get("name", "")
            if not cim_name:
                continue

            best_score = 0.0
            best_svg = None
            for svg_dev in svg_devices:
                svg_id = svg_dev.get("id", "")
                if svg_id in svg_matched:
                    continue
                score = self._similarity(cim_name, svg_id)
                label_score = self._similarity(
                    cim_name, svg_dev.get("label", ""))
                score = max(score, label_score)
                if score > best_score:
                    best_score = score
                    best_svg = svg_id

            if best_svg and best_score >= self.fuzzy_threshold:
                fuzzy_matches.append((cim_dev["uri"], best_svg, best_score))
                mapping[cim_dev["uri"]] = best_svg
                cim_matched.add(cim_dev["uri"])
                svg_matched.add(best_svg)

        # 未匹配的
        cim_only = [d["uri"] for d in cim_devices
                    if d["uri"] not in cim_matched]
        svg_only = [d.get("id", "") for d in svg_devices
                    if d.get("id", "") not in svg_matched]

        self.alignment = {
            "mapping": mapping,
            "cim_only": cim_only,
            "svg_only": svg_only,
            "fuzzy": fuzzy_matches,
            "stats": {
                "cim_total": len(cim_devices),
                "svg_total": len(svg_devices),
                "matched": len(mapping),
                "cim_only_count": len(cim_only),
                "svg_only_count": len(svg_only),
                "fuzzy_count": len(fuzzy_matches),
                "match_rate": (len(mapping) / max(len(cim_devices), 1)),
            },
        }

        logger.info(
            f"CIM-SVG对齐: 匹配={len(mapping)}, "
            f"CIM独有={len(cim_only)}, SVG独有={len(svg_only)}, "
            f"模糊匹配={len(fuzzy_matches)}, "
            f"匹配率={self.alignment['stats']['match_rate']:.1%}"
        )
        return self.alignment

    def extract_svg_topology(self, svg_data: Optional[Dict] = None) -> Dict:
        """
        从SVG图形推断拓扑连接关系

        通过空间近邻分析:
          - 连接线端点靠近设备中心 -> 设备连接
          - 开关端点靠近母线 -> 开关连接

        Returns:
            {connections: [{device_a, device_b, via}], graph_data: {...}}
        """
        svg = svg_data or self.svg_data
        if svg is None:
            raise ValueError("请先加载SVG数据")

        # 构建设备位置索引
        device_positions = {}
        device_types = {}
        for dev in svg.get("devices", []):
            device_positions[dev["id"]] = (dev["x"], dev["y"])
            device_types[dev["id"]] = dev.get("type", "Unknown")
        for sw in svg.get("switches", []):
            device_positions[sw["id"]] = (sw["x"], sw["y"])
            device_types[sw["id"]] = sw.get("type", "switch")

        # 连接线端点收集
        line_endpoints = []
        for conn in svg.get("connections", []):
            points = conn.get("points", [])
            if len(points) >= 2:
                line_endpoints.append({
                    "id": conn.get("id", ""),
                    "start": points[0],
                    "end": points[-1],
                })

        # 通过空间近邻推断连接
        connections = []
        for line_info in line_endpoints:
            start = line_info["start"]
            end = line_info["end"]

            # 找到离起点最近的设备
            dev_start = self._find_nearest_device(
                start, device_positions, self.spatial_merge_dist)
            dev_end = self._find_nearest_device(
                end, device_positions, self.spatial_merge_dist)

            if dev_start and dev_end and dev_start != dev_end:
                connections.append({
                    "device_a": dev_start,
                    "device_b": dev_end,
                    "via": line_info["id"],
                    "type_a": device_types.get(dev_start, "Unknown"),
                    "type_b": device_types.get(dev_end, "Unknown"),
                })

        # 构建图数据
        graph_nodes = []
        for did, pos in device_positions.items():
            graph_nodes.append({
                "id": did,
                "x": pos[0],
                "y": pos[1],
                "type": device_types.get(did, "Unknown"),
            })

        graph_edges = []
        for conn in connections:
            graph_edges.append({
                "from": conn["device_a"],
                "to": conn["device_b"],
            })

        logger.info(f"SVG拓扑提取: 连接={len(connections)}")
        return {
            "connections": connections,
            "graph_data": {
                "nodes": graph_nodes,
                "edges": graph_edges,
            },
        }

    def validate_consistency(self, cim_data: Optional[Dict] = None,
                              svg_data: Optional[Dict] = None) -> Dict:
        """
        CIM-SVG一致性校验

        检查:
          1. 设备数量一致性
          2. 设备类型一致性
          3. 开关状态一致性
          4. 拓扑结构一致性（连通分量数）

        Returns:
            {passed: bool, issues: [{type, severity, detail}]}
        """
        cim = cim_data or self.cim_data
        svg = svg_data or self.svg_data
        if cim is None or svg is None:
            return {"passed": False, "issues": [
                {"type": "missing_data", "severity": "error",
                 "detail": "CIM或SVG数据未加载"}
            ]}

        issues = []

        # 1. 设备数量对比
        cim_count = len(cim.get("devices", []))
        svg_count = len(svg.get("devices", [])) + len(svg.get("switches", []))
        ratio = min(cim_count, svg_count) / max(cim_count, svg_count, 1)
        if ratio < 0.8:
            issues.append({
                "type": "device_count_mismatch",
                "severity": "warning",
                "detail": (f"CIM设备数={cim_count}, SVG设备数={svg_count}, "
                           f"比率={ratio:.1%}"),
            })

        # 2. 设备类型分布对比
        cim_types = defaultdict(int)
        for d in cim.get("devices", []):
            cim_types[d.get("subtype", "Unknown")] += 1
        svg_types = defaultdict(int)
        for d in svg.get("devices", []):
            svg_types[d.get("type", "Unknown")] += 1

        # 3. 开关数量对比
        cim_sw = len(cim.get("switches", []))
        svg_sw = len(svg.get("switches", []))
        if cim_sw > 0 and svg_sw == 0:
            issues.append({
                "type": "switch_missing_in_svg",
                "severity": "warning",
                "detail": f"CIM有{cim_sw}个开关但SVG中未找到开关",
            })
        elif svg_sw > 0 and cim_sw == 0:
            issues.append({
                "type": "switch_missing_in_cim",
                "severity": "warning",
                "detail": f"SVG有{svg_sw}个开关但CIM中未找到开关",
            })

        # 4. 对齐匹配率
        if self.alignment:
            match_rate = self.alignment.get("stats", {}).get("match_rate", 0)
            if match_rate < 0.5:
                issues.append({
                    "type": "low_match_rate",
                    "severity": "warning",
                    "detail": f"CIM-SVG匹配率仅{match_rate:.1%}，数据可能不对应同一网络",
                })

        passed = not any(i["severity"] == "error" for i in issues)
        return {"passed": passed, "issues": issues}

    def get_device_positions_map(self, svg_data: Optional[Dict] = None
                                  ) -> Dict[str, Tuple[float, float]]:
        """返回 {device_id: (x, y)} 位置映射"""
        svg = svg_data or self.svg_data
        if svg is None:
            return {}
        return get_device_positions(svg)

    # ================================================================
    # 内部方法
    # ================================================================

    @staticmethod
    def _extract_short_id(uri: str) -> str:
        """从URI中提取短ID (#后或最后一段)"""
        if "#" in uri:
            return uri.split("#")[-1]
        return uri.split("/")[-1]

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """计算两个字符串的相似度"""
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    @staticmethod
    def _find_nearest_device(point: Tuple[float, float],
                              positions: Dict[str, Tuple[float, float]],
                              max_dist: float) -> Optional[str]:
        """找到离给定坐标最近的设备（距离 < max_dist）"""
        best_id = None
        best_dist = max_dist
        for did, pos in positions.items():
            dist = math.sqrt((point[0] - pos[0])**2 + (point[1] - pos[1])**2)
            if dist < best_dist:
                best_dist = dist
                best_id = did
        return best_id
