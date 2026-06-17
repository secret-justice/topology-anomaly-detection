# -*- coding: utf-8 -*-
"""
v16: 拓扑清理管线 — 检测前预处理
来源: gridstate/topology.py
5步清理: SLACK统一 → 节点类型修正 → 孤立支路 → 断开连通分量 → 孤立节点
"""
import numpy as np
import logging
from collections import deque

logger = logging.getLogger(__name__)


def cleanup_topology(net):
    """
    对PandaPower网络执行5步拓扑清理
    
    Args:
        net: pandapower network
        
    Returns:
        dict: 清理统计 {"orphan_branches": N, "disconnected": N, "isolated": N, ...}
    """
    stats = {"orphan_branches": 0, "disconnected": 0, "isolated": 0, "slack_fixed": 0}
    
    # Step 1: 多SLACK→单SLACK
    stats["slack_fixed"] = _refine_slack_to_one(net)
    
    # Step 2: 有发电机节点→PV (PandaPower中自动处理)
    
    # Step 3: 禁用引用失效节点的支路
    stats["orphan_branches"] = _disable_orphan_branches(net)
    
    # Step 4: 禁用与主网断开的连通分量
    stats["disconnected"] = _disable_disconnected_components(net)
    
    # Step 5: 禁用无活动支路的孤立节点
    stats["isolated"] = _disable_isolated_nodes(net)
    
    total = sum(stats.values())
    if total > 0:
        logger.info(f"Topology cleanup: {stats}")
    
    return stats


def _refine_slack_to_one(net):
    """确保只有一个ext_grid(SLACK)"""
    if len(net.ext_grid) <= 1:
        return 0
    # 保留第一个，其余转为PV节点
    keep_idx = net.ext_grid.index[0]
    remove_idx = net.ext_grid.index[1:]
    for idx in remove_idx:
        bus = net.ext_grid.at[idx, "bus"]
        # 如果该bus没有其他generator，添加一个static generator
        if bus not in net.gen.bus.values:
            from pandapower.create import create_gen
            create_gen(net, bus=bus, p_mw=0, vm_pu=net.ext_grid.at[idx, "vm_pu"],
                      name=f"converted_slack_{idx}")
    net.ext_grid = net.ext_grid.loc[[keep_idx]]
    return len(remove_idx)


def _disable_orphan_branches(net):
    """禁用引用失效节点的支路"""
    active_buses = set(net.bus[net.bus.in_service].index)
    count = 0
    
    for tbl_name in ["line", "trafo"]:
        tbl = getattr(net, tbl_name)
        for idx in tbl.index:
            if not tbl.at[idx, "in_service"]:
                continue
            fb = int(tbl.at[idx, "from_bus"]) if "from_bus" in tbl.columns else int(tbl.at[idx, "hv_bus"])
            tb = int(tbl.at[idx, "to_bus"]) if "to_bus" in tbl.columns else int(tbl.at[idx, "lv_bus"])
            if fb not in active_buses or tb not in active_buses:
                tbl.at[idx, "in_service"] = False
                count += 1
    
    return count


def _disable_disconnected_components(net):
    """禁用与主网断开的连通分量"""
    import networkx as nx
    G = nx.Graph()
    for idx in net.bus.index:
        G.add_node(int(idx))
    for idx in net.line.index:
        if net.line.at[idx, "in_service"]:
            fb = int(net.line.at[idx, "from_bus"])
            tb = int(net.line.at[idx, "to_bus"])
            G.add_edge(fb, tb)
    for idx in net.trafo.index:
        if net.trafo.at[idx, "in_service"]:
            hb = int(net.trafo.at[idx, "hv_bus"])
            lb = int(net.trafo.at[idx, "lv_bus"])
            G.add_edge(hb, lb)
    
    if G.number_of_nodes() == 0:
        return 0
    
    # 找到SLACK所在的连通分量
    slack_buses = set(net.ext_grid.bus.values.astype(int))
    if not slack_buses:
        return 0
    
    slack_bus = list(slack_buses)[0]
    if slack_bus not in G:
        return 0
    
    main_component = nx.node_connected_component(G, slack_bus)
    count = 0
    for comp in nx.connected_components(G):
        if comp != main_component:
            for bus in comp:
                if bus in net.bus.index:
                    net.bus.at[bus, "in_service"] = False
                    count += 1
    return count


def _disable_isolated_nodes(net):
    """禁用无活动支路的孤立节点"""
    connected_buses = set()
    for idx in net.line.index:
        if net.line.at[idx, "in_service"]:
            connected_buses.add(int(net.line.at[idx, "from_bus"]))
            connected_buses.add(int(net.line.at[idx, "to_bus"]))
    for idx in net.trafo.index:
        if net.trafo.at[idx, "in_service"]:
            connected_buses.add(int(net.trafo.at[idx, "hv_bus"]))
            connected_buses.add(int(net.trafo.at[idx, "lv_bus"]))
    # SLACK节点总是connected
    for idx in net.ext_grid.index:
        connected_buses.add(int(net.ext_grid.at[idx, "bus"]))
    
    count = 0
    for idx in net.bus.index:
        if int(idx) not in connected_buses and net.bus.at[idx, "in_service"]:
            net.bus.at[idx, "in_service"] = False
            count += 1
    return count
