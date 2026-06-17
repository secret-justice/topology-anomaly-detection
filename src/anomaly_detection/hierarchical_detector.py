# -*- coding: utf-8 -*-
"""
大规模网络分层检测模块
=====================
将大规模配电网拓扑图分层聚合、分区检测、关键节点优先分析，
解决单次检测在大网络上 OOM 或超时的问题。

核心策略:
  1. 图粗化 (Graph Coarsening) —— 连通分量 / 社区 / 电压等级 多级聚合
  2. 分区独立检测 —— 每个分区调用 base_detector_fn，互不干扰
  3. 关键节点优先 —— 按度数 & 介数中心性排序，优先检测高影响节点
  4. 边界合并 —— 跨分区边界的异常做去重与冲突消解

接口:
  HierarchicalDetector(max_partition_size=50).detect(graph, base_detector_fn, network_data)
"""
import copy
import math
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ============================================================
# 辅助: 图粗化 (Graph Coarsening)
# ============================================================
def _coarsen_by_connected_components(graph: nx.Graph) -> List[nx.Graph]:
    """
    按连通分量拆分图。

    Returns:
        子图列表，每个子图保留原始节点/边属性
    """
    components = []
    for cc in nx.connected_components(graph):
        sub = graph.subgraph(cc).copy()
        components.append(sub)
    return components


def _coarsen_by_voltage_level(graph: nx.Graph,
                              net=None) -> List[Tuple[str, nx.Graph]]:
    """
    按电压等级分区。

    如果节点带有 bus_idx 属性且 PandaPower net 可用，按 vn_kv 分组；
    同时保留开关/变压器跨区边信息。

    Returns:
        [(voltage_label, subgraph), ...]
    """
    if net is None:
        # 无法获取电压等级，退化为连通分量
        logger.warning("net 不可用，按连通分量分区")
        return [("default", g) for g in _coarsen_by_connected_components(graph)]

    # 构建 bus_idx -> vn_kv 映射
    bus_kv: Dict[int, float] = {}
    for idx in net.bus.index:
        bus_kv[int(idx)] = float(net.bus.at[idx, "vn_kv"])

    # 按 vn_kv 分组节点
    kv_groups: Dict[float, List[str]] = defaultdict(list)
    ungrouped: List[str] = []
    for node, data in graph.nodes(data=True):
        bidx = data.get("bus_idx")
        if bidx is not None and bidx in bus_kv:
            kv = bus_kv[bidx]
            kv_groups[kv].append(node)
        else:
            ungrouped.append(node)

    # 构建子图（含跨区边）
    result: List[Tuple[str, nx.Graph]] = []
    for kv, nodes in kv_groups.items():
        label = f"{kv:.1f}kV"
        sub = graph.subgraph(nodes).copy()
        result.append((label, sub))

    if ungrouped:
        sub = graph.subgraph(ungrouped).copy()
        result.append(("ungrouped", sub))

    return result


def _coarsen_by_community(graph: nx.Graph,
                          max_partition_size: int = 50) -> List[nx.Graph]:
    """
    基于社区检测 (Louvain / greedy modularity) 将大分区进一步拆分。

    当分区节点数超过 max_partition_size 时调用。
    """
    if graph.number_of_nodes() <= max_partition_size:
        return [graph]

    # 尝试 Louvain（需要 python-louvain 或 networkx>=2.8 内置）
    try:
        communities = nx.community.louvain_communities(graph, seed=42)
    except AttributeError:
        # 退回 greedy modularity
        try:
            communities = nx.community.greedy_modularity_communities(graph)
        except Exception:
            # 最终退回连通分量
            logger.warning("社区检测不可用，退化为连通分量")
            return _coarsen_by_connected_components(graph)

    subgraphs = []
    for comm in communities:
        if len(comm) == 0:
            continue
        sub = graph.subgraph(comm).copy()
        subgraphs.append(sub)

    return subgraphs


# ============================================================
# 辅助: 关键节点排序
# ============================================================
def _rank_critical_nodes(graph: nx.Graph,
                         top_k: Optional[int] = None) -> List[str]:
    """
    按节点重要性排序。

    综合指标 = 0.6 × 归一化度数 + 0.4 × 归一化介数中心性
    度数高 = 连接多，介数高 = 关键路径枢纽。

    Args:
        graph: 子图
        top_k: 返回前 k 个；None 表示全部

    Returns:
        按重要性降序排列的节点列表
    """
    if graph.number_of_nodes() == 0:
        return []

    # 度数
    degrees = dict(graph.degree())
    max_deg = max(degrees.values()) if degrees else 1

    # 介数中心性（大图采样加速）
    n = graph.number_of_nodes()
    if n > 500:
        betweenness = nx.betweenness_centrality(graph, k=min(100, n))
    else:
        betweenness = nx.betweenness_centrality(graph)

    max_btw = max(betweenness.values()) if betweenness else 1.0

    # 综合评分
    scores: Dict[str, float] = {}
    for node in graph.nodes():
        d_norm = degrees.get(node, 0) / max_deg if max_deg > 0 else 0
        b_norm = betweenness.get(node, 0) / max_btw if max_btw > 0 else 0
        scores[node] = 0.6 * d_norm + 0.4 * b_norm

    ranked = sorted(scores, key=scores.get, reverse=True)
    if top_k is not None:
        ranked = ranked[:top_k]
    return ranked


# ============================================================
# 辅助: 边界异常去重与冲突消解
# ============================================================
def _merge_cross_partition_anomalies(
    anomalies_per_partition: List[List[Dict]],
    partition_labels: List[str]
) -> List[Dict]:
    """
    合并各分区检测结果，去重并标记跨分区异常。

    去重规则:
      - 同一 (type, location) 的异常只保留 confidence 最高的
      - 跨分区边界异常标记 origin_partitions
    """
    # 按 (type, location) 分组
    grouped: Dict[Tuple, List[Dict]] = defaultdict(list)
    for idx, anomalies in enumerate(anomalies_per_partition):
        label = partition_labels[idx] if idx < len(partition_labels) else f"P{idx}"
        for a in anomalies:
            key = (a.get("type", ""), a.get("location", ""))
            a_copy = dict(a)
            a_copy["_partition"] = label
            grouped[key].append(a_copy)

    merged: List[Dict] = []
    for key, items in grouped.items():
        # 取 confidence 最高的作为代表
        best = max(items, key=lambda x: x.get("confidence", 0))
        partitions = list(set(it["_partition"] for it in items))
        if len(partitions) > 1:
            best["cross_partition"] = True
            best["origin_partitions"] = partitions
        else:
            best["cross_partition"] = False
        # 移除内部标记
        best.pop("_partition", None)
        merged.append(best)

    return merged


# ============================================================
# 主接口: HierarchicalDetector
# ============================================================
class HierarchicalDetector:
    """
    大规模网络分层检测器。

    工作流程:
      1. 判断图规模是否需要分区（节点数 > max_partition_size）
      2. 图粗化: 按连通分量 → 按电压等级 → 按社区检测 多级拆分
      3. 关键节点优先: 每个分区内按节点重要性排序，优先检测
      4. 分区独立检测: 每个子图调用 base_detector_fn
      5. 合并结果 + 跨分区边界异常去重

    使用方式:
        detector = HierarchicalDetector(max_partition_size=50)
        anomalies = detector.detect(graph, base_detector_fn, network_data)

    Args:
        max_partition_size: 单分区最大节点数，超过则进一步社区拆分
    """

    def __init__(self, max_partition_size: int = 50):
        self.max_partition_size = max_partition_size
        self._partition_info: List[Dict] = []    # 分区元信息
        self._critical_nodes: List[str] = []     # 全局关键节点
        self._coarsen_strategy: str = ""          # 实际采用的分区策略

    # ----------------------------------------------------------
    # 核心接口
    # ----------------------------------------------------------
    def detect(
        self,
        graph: nx.Graph,
        base_detector_fn: Callable[[nx.Graph, Dict], List[Dict]],
        network_data: Dict,
    ) -> List[Dict]:
        """
        分层检测大网络。

        Args:
            graph:            NetworkX 拓扑图（全网）
            base_detector_fn: 基础检测函数，签名 (subgraph, sub_network_data) -> [anomalies]
                              其中 sub_network_data 是裁剪后的网络数据子集
            network_data:     完整网络数据字典

        Returns:
            标准化异常列表 [{type, location, confidence, layer, details, ...}, ...]
        """
        n_nodes = graph.number_of_nodes()
        logger.info(f"[分层检测] 图规模: 节点={n_nodes}, 边={graph.number_of_edges()}")

        # 1. 检查是否需要分区
        if n_nodes > 5000:
            streamer = StreamingHierarchicalDetector(self.max_partition_size)
            return streamer.detect_streaming(graph, base_detector_fn, network_data)

        if n_nodes <= self.max_partition_size:
            logger.info("[分层检测] 小图，直接全图检测")
            self._coarsen_strategy = "none"
            return base_detector_fn(graph, network_data)

        # 2. 图分区
        partitions = self._partition_graph(graph, network_data)
        logger.info(f"[分层检测] 分区策略={self._coarsen_strategy}, 分区数={len(partitions)}")

        # 3. 关键节点排序（全局）
        self._critical_nodes = _rank_critical_nodes(graph, top_k=min(50, n_nodes))
        logger.info(f"[分层检测] 关键节点 Top-5: {self._critical_nodes[:5]}")

        # 4. 每个分区独立检测
        anomalies_per_partition: List[List[Dict]] = []
        labels: List[str] = []
        for i, (label, subgraph) in enumerate(partitions):
            labels.append(label)
            sub_data = self._subset_network_data(network_data, subgraph)
            logger.info(
                f"[分层检测] 分区 {i+1}/{len(partitions)} [{label}]: "
                f"节点={subgraph.number_of_nodes()}, 边={subgraph.number_of_edges()}"
            )
            try:
                sub_anomalies = base_detector_fn(subgraph, sub_data)
            except Exception as e:
                logger.error(f"[分层检测] 分区 {label} 检测失败: {e}")
                sub_anomalies = []
            anomalies_per_partition.append(sub_anomalies)

            self._partition_info.append({
                "label": label,
                "n_nodes": subgraph.number_of_nodes(),
                "n_edges": subgraph.number_of_edges(),
                "n_anomalies": len(sub_anomalies),
            })

        # 5. 合并结果 + 边界检查
        merged = _merge_cross_partition_anomalies(anomalies_per_partition, labels)

        # 标记是否关联关键节点
        critical_set = set(self._critical_nodes[:20])
        for a in merged:
            loc = a.get("location", "")
            a["near_critical_node"] = loc in critical_set or any(
                loc in str(n) for n in critical_set
            )

        logger.info(
            f"[分层检测] 完成: 共 {sum(len(ap) for ap in anomalies_per_partition)} 个原始异常, "
            f"合并去重后 {len(merged)} 个"
        )
        return merged

    # ----------------------------------------------------------
    # 图分区策略
    # ----------------------------------------------------------
    def _partition_graph(
        self, graph: nx.Graph, network_data: Dict
    ) -> List[Tuple[str, nx.Graph]]:
        """
        多级图分区:
          1. 先按连通分量拆分
          2. 对含电压等级信息的分量，进一步按 vn_kv 拆分
          3. 对仍超过阈值的分区，做社区检测
        """
        # 第一步: 连通分量
        components = _coarsen_by_connected_components(graph)
        self._coarsen_strategy = "connected_components"

        if len(components) > 1:
            logger.info(f"[图粗化] 连通分量数={len(components)}")

        # 第二步: 尝试按电压等级细分
        net = network_data.get("net")
        if net is not None and len(components) == 1:
            kv_partitions = _coarsen_by_voltage_level(graph, net)
            if len(kv_partitions) > 1:
                components = [g for _, g in kv_partitions]
                self._coarsen_strategy = "voltage_level"
                logger.info(f"[图粗化] 电压等级分区数={len(kv_partitions)}")

        # 第三步: 社区检测进一步拆分超大分区
        final: List[Tuple[str, nx.Graph]] = []
        part_idx = 0
        for comp in components:
            if comp.number_of_nodes() <= self.max_partition_size:
                final.append((f"P{part_idx}", comp))
                part_idx += 1
            else:
                subs = _coarsen_by_community(comp, self.max_partition_size)
                if len(subs) > 1:
                    self._coarsen_strategy += "+community"
                for sub in subs:
                    final.append((f"P{part_idx}", sub))
                    part_idx += 1

        return final

    # ----------------------------------------------------------
    # 网络数据子集提取
    # ----------------------------------------------------------
    def _subset_network_data(
        self, network_data: Dict, subgraph: nx.Graph
    ) -> Dict:
        """
        从全网数据中提取子图对应的数据子集。

        保留: graph, net, scada_data (全局)，但过滤 cim_devices / measurements / switches
        """
        sub_data = dict(network_data)
        sub_data["graph"] = subgraph

        # 收集子图中的节点集合（用于过滤）
        node_set: Set[str] = set(subgraph.nodes())

        # 过滤 cim_devices: 保留与子图节点关联的设备
        if "cim_devices" in network_data:
            sub_devices = []
            for dev in network_data["cim_devices"]:
                dev_uri = dev.get("uri", "")
                # 检查是否有边连接到子图节点
                for u, v, edata in subgraph.edges(data=True):
                    if edata.get("device") == dev_uri:
                        sub_devices.append(dev)
                        break
            sub_data["cim_devices"] = sub_devices

        # 过滤 switches: 保留子图中带 is_switch 标记的边对应的开关
        if "switches" in network_data:
            sub_switch_indices: Set[int] = set()
            for u, v, edata in subgraph.edges(data=True):
                if edata.get("is_switch"):
                    eidx = edata.get("element_idx")
                    if eidx is not None:
                        sub_switch_indices.add(int(eidx))
            sub_data["switches"] = [
                sw for sw in network_data["switches"]
                if sw.get("idx") in sub_switch_indices
                or sw.get("element_idx") in sub_switch_indices
            ]

        # measurements 和 scada_data 保留全局（状态估计需要完整数据）
        return sub_data

    # ----------------------------------------------------------
    # 辅助查询
    # ----------------------------------------------------------
    def get_partition_info(self) -> List[Dict]:
        """返回上一次 detect 调用的分区元信息"""
        return self._partition_info

    def get_critical_nodes(self) -> List[str]:
        """返回上一次 detect 调用的关键节点列表"""
        return self._critical_nodes

    def get_coarsen_strategy(self) -> str:
        """返回实际采用的图粗化策略"""
        return self._coarsen_strategy


# ============================================================
# 便捷函数
# ============================================================

# ===== P1-4: Large-Scale Partitioning for 5000+ Nodes =====

def _coarsen_by_metis_style(graph, max_partition_size=50, num_partitions=None):
    "METIS-style partitioning using spectral or edge-cut heuristics."
    import networkx as nx
    n = graph.number_of_nodes()
    if n <= max_partition_size:
        return [graph]
    if num_partitions is None:
        num_partitions = max(2, (n + max_partition_size - 1) // max_partition_size)
    try:
        import networkx.algorithms.community as nx_comm
        communities = nx_comm.greedy_modularity_communities(graph, k=num_partitions)
        partitions = []
        for comm in communities:
            if len(comm) > 0:
                subg = graph.subgraph(comm).copy()
                if subg.number_of_nodes() > 0:
                    partitions.append(subg)
        if len(partitions) >= 2:
            return _refine_partitions(partitions, max_partition_size)
    except Exception:
        pass
    return _recursive_bisect(graph, max_partition_size)


def _recursive_bisect(graph, max_size, depth=0, max_depth=10):
    "Recursive bisection using minimum edge cut."
    import networkx as nx
    if graph.number_of_nodes() <= max_size or depth >= max_depth:
        return [graph]
    try:
        cut_value, partition = nx.stoer_wagner(graph)
        part_a, part_b = partition
        if len(part_a) == 0 or len(part_b) == 0:
            return [graph]
        sub_a = graph.subgraph(part_a).copy()
        sub_b = graph.subgraph(part_b).copy()
        result = []
        result.extend(_recursive_bisect(sub_a, max_size, depth + 1, max_depth))
        result.extend(_recursive_bisect(sub_b, max_size, depth + 1, max_depth))
        return result
    except Exception:
        nodes = list(graph.nodes())
        np.random.shuffle(nodes)
        mid = len(nodes) // 2
        result = []
        sub_a = graph.subgraph(nodes[:mid]).copy()
        sub_b = graph.subgraph(nodes[mid:]).copy()
        if sub_a.number_of_nodes() > 0:
            result.extend(_recursive_bisect(sub_a, max_size, depth + 1, max_depth))
        if sub_b.number_of_nodes() > 0:
            result.extend(_recursive_bisect(sub_b, max_size, depth + 1, max_depth))
        return result


def _refine_partitions(partitions, max_size):
    "Refine partitions: split oversized, merge tiny ones."
    import networkx as nx
    refined = []
    for part in partitions:
        if part.number_of_nodes() > max_size * 2:
            refined.extend(_recursive_bisect(part, max_size))
        else:
            refined.append(part)
    merged = []
    buffer_g = nx.Graph()
    for part in refined:
        if part.number_of_nodes() < max_size // 4:
            buffer_g = nx.compose(buffer_g, part)
            if buffer_g.number_of_nodes() >= max_size // 2:
                merged.append(buffer_g)
                buffer_g = nx.Graph()
        else:
            if buffer_g.number_of_nodes() > 0:
                merged.append(buffer_g)
                buffer_g = nx.Graph()
            merged.append(part)
    if buffer_g.number_of_nodes() > 0:
        if merged:
            smallest = min(merged, key=lambda g: g.number_of_nodes())
            idx = merged.index(smallest)
            merged[idx] = nx.compose(smallest, buffer_g)
        else:
            merged.append(buffer_g)
    return merged


class StreamingHierarchicalDetector:
    "Streaming detection for very large networks (>5000 nodes)."

    def __init__(self, max_partition_size=200, batch_size=10, overlap_nodes=5):
        self.max_partition_size = max_partition_size
        self.batch_size = batch_size
        self.overlap_nodes = overlap_nodes

    def detect_streaming(self, graph, base_detector_fn, network_data=None):
        "Stream detection over partitions with boundary overlap."
        import networkx as nx
        n = graph.number_of_nodes()
        partitions = _coarsen_by_metis_style(graph, self.max_partition_size)
        overlapped = self._add_boundary_overlap(graph, partitions)
        all_anomalies = []
        for batch_start in range(0, len(overlapped), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(overlapped))
            for i, (subg, core_nodes) in enumerate(overlapped[batch_start:batch_end]):
                try:
                    sub_data = dict(network_data) if network_data else {}
                    sub_data["_partition_idx"] = batch_start + i
                    sub_anomalies = base_detector_fn(subg, sub_data)
                    all_anomalies.extend(sub_anomalies)
                except Exception as e:
                    logger.warning("Partition %d failed: %s", batch_start + i, e)
        return self._deduplicate(all_anomalies)

    def _add_boundary_overlap(self, graph, partitions):
        result = []
        for part in partitions:
            core = set(part.nodes())
            boundary = set()
            for node in core:
                for nb in graph.neighbors(node):
                    if nb not in core:
                        boundary.add(nb)
            extended = core | boundary
            result.append((graph.subgraph(extended).copy(), core))
        return result

    def _deduplicate(self, anomalies):
        seen = set()
        unique = []
        for a in anomalies:
            key = (a.get("type", ""), str(a.get("description", ""))[:50])
            if key not in seen:
                seen.add(key)
                unique.append(a)
        return unique

def run_hierarchical_detection(
    graph: nx.Graph,
    base_detector_fn: Callable[[nx.Graph, Dict], List[Dict]],
    network_data: Dict,
    max_partition_size: int = 50,
) -> List[Dict]:
    """
    便捷入口: 创建 HierarchicalDetector 并执行检测。

    Args:
        graph:                NetworkX 拓扑图
        base_detector_fn:     基础检测函数 (subgraph, sub_data) -> [anomalies]
        network_data:         网络数据字典
        max_partition_size:   单分区最大节点数

    Returns:
        合并去重后的异常列表
    """
    detector = HierarchicalDetector(max_partition_size=max_partition_size)
    return detector.detect(graph, base_detector_fn, network_data)
