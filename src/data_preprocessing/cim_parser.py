# -*- coding: utf-8 -*-
"""
CIM/RDF 解析器
基于 rdflib 解析 IEC 61970 CIM 模型，提取设备、终端、拓扑节点
支持 CIM16 和 CIM100 两种命名空间

参考: 07_参考文献/CIM_RDF解析示例代码.py
"""
from rdflib import Graph, Namespace, RDF
from collections import defaultdict
from typing import Dict, List
import logging
import re

logger = logging.getLogger(__name__)

# 支持的CIM命名空间
CIM_NAMESPACES = {
    "cim16":   "http://iec.ch/TC57/2013/CIM-schema-cim16#",
    "cim100":  "http://iec.ch/TC57/CIM100#",
}


def _detect_cim_namespace(filepath: str) -> str:
    """从XML文件头部快速检测CIM命名空间"""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        head = f.read(4096)
    for uri in CIM_NAMESPACES.values():
        if uri in head:
            return uri
    m = re.search(r'xmlns:cim="([^"]+)"', head)
    if m:
        return m.group(1)
    return CIM_NAMESPACES["cim100"]


def parse_cim_rdf(filepath: str) -> Dict:
    """
    解析单个CIM/RDF文件，提取配电网拓扑结构。
    自动检测CIM命名空间（CIM16/CIM100）。

    返回字典:
      devices     - 设备列表 [{uri, name, type, subtype}]
      terminals   - 端子列表 [{uri, name, conducting_eq, connectivity_node, seq}]
      nodes       - 连接节点 [{uri, name}]
      switches    - 开关 [{uri, name, subtype, normal_open, open_pos}]
      connections - 连接关系 [{device, terminal, node}]
      cim_namespace - 使用的CIM命名空间
    """
    cim_uri = _detect_cim_namespace(filepath)
    logger.info(f"CIM命名空间: {cim_uri}")

    g = Graph()
    g.parse(filepath, format="xml")
    CIM = Namespace(cim_uri)

    is_cim100 = "CIM100" in cim_uri

    # ---- 属性名映射（CIM16 vs CIM100）----
    PROP_NAME        = CIM.name if not is_cim100 else CIM["IdentifiedObject.name"]
    PROP_COND_EQ     = CIM.ConductingEquipment if not is_cim100 else CIM["Terminal.ConductingEquipment"]
    PROP_CONN_NODE   = CIM.ConnectivityNode if not is_cim100 else CIM["Terminal.ConnectivityNode"]
    PROP_SEQ_NUM     = CIM.sequenceNumber if not is_cim100 else CIM["ACDCTerminal.sequenceNumber"]
    PROP_NORMAL_OPEN = CIM.normalOpen if not is_cim100 else CIM["Switch.normalOpen"]

    # ---- 设备子类 ----
    # CIM100中ConductingEquipment是父类，rdf:type只记录具体子类
    # 必须逐个子类查询
    DEVICE_SUBTYPES = {
        CIM.Breaker:                "Breaker",
        CIM.Disconnector:           "Disconnector",
        CIM.LoadBreakSwitch:        "LoadBreakSwitch",
        CIM.Fuse:                   "Fuse",
        CIM.Recloser:               "Recloser",
        CIM.ACLineSegment:          "ACLineSegment",
        CIM.PowerTransformer:       "PowerTransformer",
        CIM.EnergyConsumer:         "EnergyConsumer",
        CIM.EnergySource:           "EnergySource",
        CIM.SynchronousMachine:     "SynchronousMachine",
        CIM.LinearShuntCompensator: "LinearShuntCompensator",
        CIM.RatioTapChanger:        "RatioTapChanger",
        CIM.PowerTransformerEnd:    "PowerTransformerEnd",
        CIM.SeriesCompensator:      "SeriesCompensator",
        CIM.PowerElectronicsConnection: "PowerElectronicsConnection",
    }

    SWITCH_SUBTYPES = {
        "Breaker", "Disconnector", "LoadBreakSwitch", "Fuse", "Recloser",
    }

    devices = []
    terminals = []
    nodes = []
    switches = []
    device_map = {}
    seen_uris = set()

    # ---- 导电设备（逐子类查询）----
    for rdf_cls, stype in DEVICE_SUBTYPES.items():
        for subj in g.subjects(RDF.type, rdf_cls):
            uri = str(subj)
            if uri in seen_uris:
                continue
            seen_uris.add(uri)
            name = str(g.value(subj, PROP_NAME) or "")
            dev = {"uri": uri, "name": name, "type": "ConductingEquipment", "subtype": stype}
            devices.append(dev)
            device_map[uri] = dev

            if stype in SWITCH_SUBTYPES:
                normal_open = g.value(subj, PROP_NORMAL_OPEN)
                no_val = None
                if normal_open is not None:
                    s = str(normal_open).lower()
                    no_val = s in ("true", "1")
                switches.append({
                    "uri": uri, "name": name, "subtype": stype,
                    "normal_open": no_val, "open_pos": None,
                })

    # ---- 端子 ----
    for subj in g.subjects(RDF.type, CIM.Terminal):
        uri = str(subj)
        name = str(g.value(subj, PROP_NAME) or "")
        ce = g.value(subj, PROP_COND_EQ)
        cn = g.value(subj, PROP_CONN_NODE)
        seq = g.value(subj, PROP_SEQ_NUM)
        terminals.append({
            "uri": uri, "name": name,
            "conducting_eq": str(ce) if ce else None,
            "connectivity_node": str(cn) if cn else None,
            "sequence_number": int(seq) if seq else None,
        })

    # ---- 连接节点 ----
    for subj in g.subjects(RDF.type, CIM.ConnectivityNode):
        uri = str(subj)
        name = str(g.value(subj, PROP_NAME) or "")
        nodes.append({"uri": uri, "name": name})

    # ---- URI规范化（处理rdf:ID vs rdf:resource引用）----
    # rdflib将rdf:ID="xxx"解析为 file:///path#xxx
    # rdf:resource="#xxx"也解析为 file:///path#xxx
    # 但如果文件路径含中文，URI会URL编码，需要规范化
    def _normalize_uri(u: str) -> str:
        """提取URI的fragment部分用于匹配"""
        if '#' in u:
            return '#' + u.split('#', 1)[1]
        return u

    # 构建设备URI的规范化索引
    norm_device_map = {}
    for uri in device_map:
        norm_device_map[_normalize_uri(uri)] = device_map[uri]

    # ---- 构建连接关系 ----
    connections = []
    for t in terminals:
        ce_raw = t["conducting_eq"]
        cn_raw = t["connectivity_node"]
        if ce_raw and cn_raw:
            # 尝试直接匹配，回退到fragment匹配
            if ce_raw in device_map:
                dev_uri = ce_raw
            else:
                norm = _normalize_uri(ce_raw)
                dev = norm_device_map.get(norm)
                dev_uri = dev["uri"] if dev else ce_raw
            connections.append({
                "device": dev_uri,
                "terminal": t["uri"],
                "node": cn_raw,
            })

    logger.info(f"CIM解析完成: 设备={len(devices)}, 端子={len(terminals)}, "
                f"节点={len(nodes)}, 开关={len(switches)}")

    return {
        "devices": devices,
        "terminals": terminals,
        "nodes": nodes,
        "switches": switches,
        "connections": connections,
        "cim_namespace": cim_uri,
    }


def parse_cim_directory(dirpath: str) -> Dict:
    """解析目录下所有RDF/XML文件并合并结果"""
    import os
    merged = {"devices": [], "terminals": [], "nodes": [],
              "switches": [], "connections": []}
    for root, _, files in os.walk(dirpath):
        for fname in sorted(files):
            if fname.lower().endswith((".rdf", ".xml", ".owl")):
                fpath = os.path.join(root, fname)
                try:
                    data = parse_cim_rdf(fpath)
                    for key in ("devices", "terminals", "nodes", "switches", "connections"):
                        merged[key].extend(data[key])
                except Exception as e:
                    logger.warning(f"跳过 {fname}: {e}")
    return merged


def get_device_connectivity(cim_data: Dict) -> Dict[str, List[str]]:
    """从CIM数据构建 设备->邻接节点 映射"""
    adj = defaultdict(list)
    for conn in cim_data["connections"]:
        adj[conn["device"]].append(conn["node"])
    return dict(adj)