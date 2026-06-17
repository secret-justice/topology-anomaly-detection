# -*- coding: utf-8 -*-
"""
SVG 解析器
从电力系统SVG图形文件中提取设备位置、连接线和开关状态
SVG中设备通常以 <g id="设备ID"> 标注，位置在 transform 属性中
"""
from lxml import etree
from typing import Dict, List, Tuple, Optional
import re
import logging

logger = logging.getLogger(__name__)

# SVG命名空间
SVG_NS = {"svg": "http://www.w3.org/2000/svg"}


def parse_svg(svg_path: str) -> Dict:
    """
    解析电力系统SVG文件，提取设备图元和位置信息

    返回字典:
      devices      - [{id, type, x, y, label}]
      connections  - [{from_id, to_id, points}]
      switches     - [{id, x, y, state}]  state: open/closed
      viewBox      - (width, height) 画布大小
    """
    tree = etree.parse(svg_path)
    root = tree.getroot()

    devices = []
    connections = []
    switches = []

    # 提取 viewBox
    vb = root.get("viewBox", "0 0 1000 1000")
    vb_parts = vb.split()
    canvas = (float(vb_parts[2]), float(vb_parts[3])) if len(vb_parts) >= 3 else (1000, 1000)

    # 遍历所有 <g> 元素提取设备
    for g_elem in root.iter("{http://www.w3.org/2000/svg}g"):
        elem_id = g_elem.get("id", "")
        if not elem_id:
            continue

        # 提取位置（从 transform 或 子元素）
        x, y = _extract_position(g_elem)

        # 判断设备类型（通过 class 或 id 前缀）
        dev_type = _classify_svg_element(g_elem, elem_id)

        if dev_type in ("breaker_open", "breaker_closed", "switch_open", "switch_closed"):
            switches.append({
                "id": elem_id, "x": x, "y": y,
                "state": "open" if "open" in dev_type else "closed",
                "type": dev_type,
            })
        else:
            devices.append({
                "id": elem_id, "type": dev_type, "x": x, "y": y,
                "label": _extract_label(g_elem),
            })

    # 提取连接线（<line> 和 <path>）
    for line_elem in root.iter("{http://www.w3.org/2000/svg}line"):
        x1 = float(line_elem.get("x1", 0))
        y1 = float(line_elem.get("y1", 0))
        x2 = float(line_elem.get("x2", 0))
        y2 = float(line_elem.get("y2", 0))
        lid = line_elem.get("id", "")
        connections.append({
            "id": lid, "from_id": None, "to_id": None,
            "points": [(x1, y1), (x2, y2)],
        })

    for path_elem in root.iter("{http://www.w3.org/2000/svg}path"):
        pid = path_elem.get("id", "")
        d_attr = path_elem.get("d", "")
        points = _parse_svg_path_d(d_attr)
        if points:
            connections.append({
                "id": pid, "from_id": None, "to_id": None, "points": points,
            })

    logger.info(f"SVG解析完成: 设备={len(devices)}, 开关={len(switches)}, "
                f"连接线={len(connections)}")

    return {
        "devices": devices,
        "connections": connections,
        "switches": switches,
        "viewBox": canvas,
    }


def _extract_position(elem) -> Tuple[float, float]:
    """从 transform 或 子 circle/rect 提取坐标"""
    transform = elem.get("transform", "")
    # 尝试解析 translate(x, y)
    m = re.search(r"translate\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)", transform)
    if m:
        return float(m.group(1)), float(m.group(2))

    # 尝试子元素 <circle cx cy> 或 <rect x y>
    for child in elem:
        tag = etree.QName(child).localname if hasattr(child, "tag") else ""
        if tag == "circle":
            cx = float(child.get("cx", 0))
            cy = float(child.get("cy", 0))
            return cx, cy
        if tag == "rect":
            rx = float(child.get("x", 0))
            ry = float(child.get("y", 0))
            return rx, ry
    return 0.0, 0.0


def _classify_svg_element(elem, elem_id: str) -> str:
    """根据CSS类名或id前缀判断设备类型"""
    cls = elem.get("class", "").lower()
    eid = elem_id.lower()

    if "breaker" in cls or "breaker" in eid:
        return "breaker_closed" if "open" not in cls else "breaker_open"
    if "switch" in cls or "disconnector" in eid:
        return "switch_closed" if "open" not in cls else "switch_open"
    if "line" in cls or "acline" in eid:
        return "ACLineSegment"
    if "transformer" in cls or "trafo" in eid:
        return "Transformer"
    if "load" in cls or "consumer" in eid:
        return "Load"
    if "source" in cls or "gen" in eid:
        return "Source"
    return "Unknown"


def _extract_label(elem) -> str:
    """提取设备文本标签"""
    for child in elem.iter("{http://www.w3.org/2000/svg}text"):
        if child.text:
            return child.text.strip()
    return ""


def _parse_svg_path_d(d_attr: str) -> List[Tuple[float, float]]:
    """从 SVG path 的 d 属性提取折线端点"""
    points = []
    nums = re.findall(r"[-\d.]+", d_attr)
    nums = [float(n) for n in nums]
    # 简单取前几对坐标
    for i in range(0, len(nums) - 1, 2):
        points.append((nums[i], nums[i + 1]))
    return points


def get_device_positions(svg_data: Dict) -> Dict[str, Tuple[float, float]]:
    """返回 {device_id: (x, y)} 映射"""
    pos = {}
    for dev in svg_data["devices"]:
        pos[dev["id"]] = (dev["x"], dev["y"])
    for sw in svg_data["switches"]:
        pos[sw["id"]] = (sw["x"], sw["y"])
    return pos
