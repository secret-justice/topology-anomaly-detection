# -*- coding: utf-8 -*-
"""
图论工具模块
从CIM数据或PandaPower网络构建NetworkX图，提供图论分析辅助函数
"""
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def build_graph_from_cim(cim_data: Dict) -> nx.Graph:
    """
    从CIM数据构建NetworkX无向图

    节点 = 连接节点(connectivity_node)
    边   = 导电设备(conducting_equipment)，通过terminal连接两个节点

    Args:
        cim_data: cim_parser.parse_cim_rdf() 的返回值
                  包含 devices, terminals, nodes, switches, connections

    Returns:
        NetworkX无向图，节点属性含name，边属性含device信息
    """
    G = nx.Graph()

    # 添加所有连接节点作为图节点
    node_uris = set()
    for node in cim_data.get("nodes", []):
        uri = node["uri"]
        G.add_node(uri, name=node.get("name", ""), type="connectivity_node")
        node_uris.add(uri)

    # 按设备分组终端，找出每个设备连接的所有节点
    device_nodes: Dict[str, List[str]] = {}
    for conn in cim_data.get("connections", []):
        dev = conn["device"]
        node = conn["node"]
        device_nodes.setdefault(dev, [])
        if node not in device_nodes[dev]:
            device_nodes[dev].append(node)

    # 设备类型索引
    device_type_map = {d["uri"]: d.get("subtype", "Unknown")
                       for d in cim_data.get("devices", [])}
    device_name_map = {d["uri"]: d.get("name", "")
                       for d in cim_data.get("devices", [])}

    # 为连接两个不同节点的设备添加边
    for dev_uri, nodes_list in device_nodes.items():
        dev_type = device_type_map.get(dev_uri, "Unknown")
        dev_name = device_name_map.get(dev_uri, "")
        for i in range(len(nodes_list)):
            for j in range(i + 1, len(nodes_list)):
                n1, n2 = nodes_list[i], nodes_list[j]
                if n1 in G and n2 in G:
                    G.add_edge(n1, n2,
                               device=dev_uri,
                               device_type=dev_type,
                               device_name=dev_name)

    # 标记开关设备
    for sw in cim_data.get("switches", []):
        for u, v, data in G.edges(data=True):
            if data.get("device") == sw["uri"]:
                data["is_switch"] = True
                data["switch_type"] = sw.get("subtype", "")
                data["normal_open"] = sw.get("normal_open", False)
                break

    logger.info(f"图构建完成: 节点={G.number_of_nodes()}, 边={G.number_of_edges()}")
    return G


def build_graph_from_pandapower(net) -> nx.Graph:
    """
    从PandaPower网络构建NetworkX图（用于MVP测试）

    Args:
        net: PandaPower网络对象

    Returns:
        NetworkX无向图
    """
    G = nx.Graph()

    # 找出ext_grid所在的母线，标记为松弛节点
    slack_buses = set()
    if hasattr(net, "ext_grid"):
        for eg_idx in net.ext_grid.index:
            slack_buses.add(int(net.ext_grid.at[eg_idx, "bus"]))

    # 添加母线节点
    for idx in net.bus.index:
        bus_int = int(idx)
        node_type = "slack" if bus_int in slack_buses else "bus"
        G.add_node(f"bus_{bus_int}",
                   name=str(net.bus.at[idx, "name"]) if "name" in net.bus.columns else f"Bus {bus_int}",
                   type=node_type,
                   bus_idx=bus_int)

    # 添加在运线路作为边
    for idx in net.line.index:
        from_bus = int(net.line.at[idx, "from_bus"])
        to_bus = int(net.line.at[idx, "to_bus"])
        in_service = bool(net.line.at[idx, "in_service"])
        if not in_service:
            continue  # 跳过停运线路
        G.add_edge(f"bus_{from_bus}", f"bus_{to_bus}",
                   device=f"line_{idx}",
                   device_type="ACLineSegment",
                   device_name=str(net.line.at[idx, "name"]) if "name" in net.line.columns else f"Line {idx}",
                   in_service=in_service,
                   element_type="line",
                   element_idx=int(idx))

    # 添加在运变压器作为边
    for idx in net.trafo.index:
        hv_bus = int(net.trafo.at[idx, "hv_bus"])
        lv_bus = int(net.trafo.at[idx, "lv_bus"])
        in_service = bool(net.trafo.at[idx, "in_service"])
        if not in_service:
            continue
        G.add_edge(f"bus_{hv_bus}", f"bus_{lv_bus}",
                   device=f"trafo_{idx}",
                   device_type="Transformer",
                   device_name=str(net.trafo.at[idx, "name"]) if "name" in net.trafo.columns else f"Trafo {idx}",
                   in_service=in_service,
                   element_type="trafo",
                   element_idx=int(idx))

    # 添加开关
    if hasattr(net, "switch") and len(net.switch) > 0:
        for idx in net.switch.index:
            et = str(net.switch.at[idx, "et"])
            bus = int(net.switch.at[idx, "bus"])
            element = int(net.switch.at[idx, "element"])
            closed = bool(net.switch.at[idx, "closed"])
            if et == "b":
                G.add_edge(f"bus_{bus}", f"bus_{element}",
                           device=f"switch_{idx}",
                           device_type="Switch",
                           is_switch=True,
                           closed=closed,
                           element_type="switch",
                           element_idx=int(idx))

    logger.info(f"PandaPower图构建完成: 节点={G.number_of_nodes()}, 边={G.number_of_edges()}")
    return G


def find_sources(graph: nx.Graph) -> List[str]:
    """
    找到图中的电源节点

    策略:
    1. 查找 type 为 slack / EnergySource / Source 的节点
    2. 若无，返回度最大的节点（当作松弛节点）

    Args:
        graph: NetworkX图

    Returns:
        电源节点列表
    """
    sources = []
    source_types = {"slack", "energysource", "source", "synchronousmachine"}

    for node, data in graph.nodes(data=True):
        if data.get("type", "").lower() in source_types:
            sources.append(node)

    if sources:
        return sources

    for u, v, data in graph.edges(data=True):
        dev_type = data.get("device_type", "").lower()
        if dev_type in ("synchronousmachine", "energysource"):
            sources.extend([u, v])

    if sources:
        return list(set(sources))

    if graph.number_of_nodes() > 0:
        sources.append(max(graph.nodes(), key=lambda n: graph.degree(n)))

    return sources


def get_substations(graph: nx.Graph) -> List[str]:
    """
    找到图中的变电站节点（连接变压器的母线）

    Args:
        graph: NetworkX图

    Returns:
        变电站节点列表
    """
    substations = []
    for u, v, data in graph.edges(data=True):
        if data.get("device_type") == "Transformer":
            substations.extend([u, v])
    for node, data in graph.nodes(data=True):
        if "substation" in data.get("type", "").lower() or \
           "substation" in data.get("name", "").lower():
            substations.append(node)
    return list(set(substations))


def get_isolated_nodes(graph: nx.Graph) -> List[str]:
    """返回所有孤立节点（度为0）"""
    return [n for n in graph.nodes() if graph.degree(n) == 0]


def get_leaf_nodes(graph: nx.Graph) -> List[str]:
    """返回所有叶子节点（度为1）"""
    return [n for n in graph.nodes() if graph.degree(n) == 1]
