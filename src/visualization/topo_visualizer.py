# -*- coding: utf-8 -*-
"""
拓扑可视化模块
使用 matplotlib + networkx 绘制配电网拓扑图，异常节点/边用红色标注
"""
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # 非交互式后端，适合服务器/脚本
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# 中文字体配置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def visualize_topology(graph: nx.Graph,
                       anomalies: Optional[List[Dict]] = None,
                       title: str = "配电网拓扑图",
                       figsize: Tuple[int, int] = (14, 10),
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    绘制拓扑图，异常节点/边用红色标注。

    Args:
        graph:     NetworkX拓扑图
        anomalies: 异常列表（可选）
        title:     图标题
        figsize:   图大小
        save_path: 保存路径（可选）

    Returns:
        matplotlib Figure 对象
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "空图", ha="center", va="center", fontsize=20)
        ax.set_title(title, fontsize=16)
        return fig

    # 布局
    try:
        pos = nx.spring_layout(graph, k=2.0, iterations=80, seed=42)
    except Exception:
        pos = nx.circular_layout(graph)

    # 收集异常信息
    anomaly_nodes: set = set()
    anomaly_edges: set = set()
    if anomalies:
        for a in anomalies:
            loc = a.get("location", "")
            if isinstance(loc, str) and loc in graph.nodes():
                anomaly_nodes.add(loc)
            # 按device名匹配边
            for u, v, data in graph.edges(data=True):
                if data.get("device") == str(loc):
                    anomaly_edges.add((u, v))

    # 节点颜色
    node_colors = [
        "#FF3333" if n in anomaly_nodes else "#4CAF50"
        for n in graph.nodes()
    ]

    # 边颜色与宽度
    edge_colors = []
    edge_widths = []
    for u, v, data in graph.edges(data=True):
        if (u, v) in anomaly_edges or (v, u) in anomaly_edges:
            edge_colors.append("#FF3333")
            edge_widths.append(3.0)
        elif not data.get("in_service", True):
            edge_colors.append("#999999")
            edge_widths.append(1.5)
        elif data.get("is_switch"):
            edge_colors.append("#2196F3")
            edge_widths.append(2.0)
        else:
            edge_colors.append("#333333")
            edge_widths.append(1.5)

    # 绘制
    nx.draw_networkx_edges(graph, pos, ax=ax,
                           edge_color=edge_colors, width=edge_widths, alpha=0.7)
    nx.draw_networkx_nodes(graph, pos, ax=ax,
                           node_color=node_colors, node_size=300,
                           edgecolors="black", linewidths=1)

    # 节点标签 (短名)
    labels = {}
    for node, data in graph.nodes(data=True):
        name = data.get("name", "")
        labels[node] = name if name else (
            str(node).split("_")[-1] if "_" in str(node) else str(node))
    nx.draw_networkx_labels(graph, pos, labels, ax=ax, font_size=8)

    # 图例
    legend_elements = [
        mpatches.Patch(color="#4CAF50", label="正常节点"),
        mpatches.Patch(color="#FF3333", label="异常节点"),
        plt.Line2D([0], [0], color="#333333", linewidth=1.5, label="正常线路"),
        plt.Line2D([0], [0], color="#FF3333", linewidth=3,   label="异常线路"),
        plt.Line2D([0], [0], color="#2196F3", linewidth=2,   label="开关"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    # 标题
    ax.set_title(title, fontsize=16, fontweight="bold")

    info = f"节点: {graph.number_of_nodes()}  边: {graph.number_of_edges()}"
    if anomalies:
        info += f"  异常: {len(anomalies)}"
    ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=9,
            va="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.axis("off")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"拓扑图已保存: {save_path}")

    return fig


def save_topology_image(graph: nx.Graph,
                        filepath: str,
                        anomalies: Optional[List[Dict]] = None,
                        title: str = "配电网拓扑图") -> str:
    """
    保存拓扑图为PNG文件。

    Args:
        graph:   NetworkX拓扑图
        filepath: 保存路径 (PNG)
        anomalies: 异常列表（可选）
        title:   图标题

    Returns:
        保存的文件绝对路径
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    fig = visualize_topology(graph, anomalies=anomalies,
                             title=title, save_path=filepath)
    plt.close(fig)
    return str(Path(filepath).resolve())
