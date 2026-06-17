# -*- coding: utf-8 -*-
"""
第三层: GNN自适应增强检测 (v3 - 支持6维边特征)
基于GraphSAGE的拓扑异常检测网络
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import os

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx


class SimpleSAGEConv(nn.Module):
    """纯PyTorch GraphSAGE卷积层"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(2 * in_channels, out_channels)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])
        degree = torch.zeros(N, dtype=torch.float, device=x.device)
        degree.index_add_(0, dst, torch.ones(dst.size(0), device=x.device))
        degree = degree.clamp(min=1.0).unsqueeze(1)
        agg = agg / degree
        return self.linear(torch.cat([x, agg], dim=1))


class SimpleEdgeSAGEConv(nn.Module):
    """支持边特征的GraphSAGE卷积层 (纯PyTorch)

    公式: h_v = Linear_self(h_v) + Linear_neigh(AGG({h_u * w_e}))
    其中 w_e = sigmoid(MLP(edge_attr)) 通过注意力权重调制邻居聚合
    """
    def __init__(self, in_channels: int, out_channels: int,
                 edge_channels: int = 6, dropout: float = 0.3):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_channels, edge_channels * 2),
            nn.ReLU(),
            nn.Linear(edge_channels * 2, 1),
            nn.Sigmoid(),
        )
        self.linear_neigh = nn.Linear(in_channels, out_channels)
        self.linear_self = nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.linear_neigh.weight)
        nn.init.zeros_(self.linear_neigh.bias)
        nn.init.xavier_uniform_(self.linear_self.weight)
        nn.init.zeros_(self.linear_self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        N = x.size(0)

        attn_weight = self.edge_mlp(edge_attr).squeeze(-1)  # [num_edges]
        weighted_src = attn_weight.unsqueeze(-1) * x[src]   # [num_edges, in_ch]

        agg = torch.zeros(N, x.size(1), device=x.device)
        agg.index_add_(0, dst, weighted_src)

        deg = torch.zeros(N, dtype=torch.float, device=x.device)
        deg.index_add_(0, dst, attn_weight.detach())
        deg = deg.clamp(min=1e-6).unsqueeze(1)
        agg = agg / deg

        out = self.linear_self(x) + self.linear_neigh(agg)
        return self.dropout(F.relu(out))


class GraphStatsMLP(nn.Module):
    """Graph-level statistics MLP classifier (v22).
    
    Input: [batch, 72] where 72 = 12 features * 6 stats (mean/std/min/max/range/median)
    Architecture: 72 -> 256 -> 128 -> 64 -> num_classes (4 layers, nn.Sequential)
    """
    def __init__(self, in_dim=72, hid=256, num_classes=13):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.BatchNorm1d(hid), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hid, hid // 2), nn.BatchNorm1d(hid // 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hid // 2, hid // 4), nn.BatchNorm1d(hid // 4), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hid // 4, num_classes),
        )
    def forward(self, x):
        return self.net(x)


class TopologyGNN(nn.Module):
    """GraphSAGE拓扑异常检测网络"""
    def __init__(self, in_channels=8, hidden_channels=128, out_channels=6, dropout=0.3):
        super().__init__()
        self.conv1 = SimpleSAGEConv(in_channels, hidden_channels)
        self.conv2 = SimpleSAGEConv(hidden_channels, hidden_channels // 2)
        self.node_classifier = nn.Linear(hidden_channels // 2, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = self.dropout(h)
        h = F.relu(self.conv2(h, edge_index))
        h = self.dropout(h)
        return self.node_classifier(h)


class EdgeTopologyGNN(nn.Module):
    """支持边特征的GraphSAGE拓扑异常检测网络

    架构: 2层SimpleEdgeSAGEConv + edge attention + node_classifier
    与TopologyGNN接口兼容，额外接收edge_attr参数
    """
    def __init__(self, in_channels: int = 8, edge_channels: int = 6,
                 hidden_channels: int = 128, out_channels: int = 6,
                 dropout: float = 0.3):
        super().__init__()
        self.conv1 = SimpleEdgeSAGEConv(in_channels, hidden_channels,
                                        edge_channels=edge_channels,
                                        dropout=dropout)
        self.conv2 = SimpleEdgeSAGEConv(hidden_channels, hidden_channels // 2,
                                        edge_channels=edge_channels,
                                        dropout=dropout)
        self.node_classifier = nn.Linear(hidden_channels // 2, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.edge_attn = nn.Sequential(
            nn.Linear(edge_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is not None:
            h = self.conv1(x, edge_index, edge_attr)
            h = self.conv2(h, edge_index, edge_attr)
            return self.node_classifier(h)
        else:
            h = F.relu(self.conv1.linear_self(x))
            h = self.dropout(h)
            h = F.relu(self.conv2.linear_self(h))
            h = self.dropout(h)
            return self.node_classifier(h)




class NodeGNN(nn.Module):
    """Node-level GNN: 2-layer message passing + node classifier (v25)."""
    def __init__(self, in_dim=12, hid=64, num_classes=16):
        super().__init__()
        self.conv1 = nn.Linear(in_dim * 2, hid)
        self.conv2 = nn.Linear(hid * 2, hid)
        self.classifier = nn.Linear(hid, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])
        deg = torch.zeros(N, dtype=torch.float, device=x.device)
        deg.index_add_(0, dst, torch.ones(dst.size(0), device=x.device))
        deg = deg.clamp(min=1.0).unsqueeze(1)
        agg = agg / deg
        h = F.relu(self.conv1(torch.cat([x, agg], dim=1)))
        h = self.dropout(h)
        agg2 = torch.zeros_like(h)
        agg2.index_add_(0, dst, h[src])
        agg2 = agg2 / deg
        h = F.relu(self.conv2(torch.cat([h, agg2], dim=1)))
        h = self.dropout(h)
        return self.classifier(h)


ANOMALY_LABELS = [
    "normal",
    "topo_interrupt", "virtual_faulty", "model_mismatch",
    "telemetry_mismatch", "signal_mismatch",
    "measurement_outlier", "stale_data", "parameter_error",
    "load_shift", "reverse_power_flow", "communication_loss",
    "voltage_collapse", "ghost_topology", "duplicate_measurement",
    "protection_misconfig",
    "trafo_tap_fault", "grounding_fault", "clock_drift",
    "harmonic_pollution", "impedance_degradation", "dg_intermittent",
    "measurement_bias", "branch_contingency", "topo_obfuscation",
    "voltage_regulation",
    # v16新增: 3种调度实际异常类型
    "bus_section_mismatch",    # 母线分段开关状态与拓扑不一致
    "bypass_operation",        # 旁路代路操作后拓扑未更新
    "load_transfer_residual",  # 负荷转供后拓扑残留
]




def build_edge_features_from_graph(graph, network_data=None):
    """Build 6-dim edge features from NetworkX graph.
    [0] type (0=line, 1=trafo, 2=switch)
    [1] R (p.u., normalized)
    [2] X (p.u., normalized)
    [3] P (MW, normalized by 10)
    [4] Q (MVAr, normalized by 10)
    [5] switch_state (0=open, 1=closed)
    """
    import networkx as nx
    line_powers = {}
    if network_data:
        for lp in network_data.get("measurements", {}).get("line_powers", []):
            key = (int(lp.get("from_bus", -1)), int(lp.get("to_bus", -1)))
            line_powers[key] = lp

    raw_features = []
    for u, v, data in graph.edges(data=True):
        etype = data.get("type", data.get("element_type", "line"))
        type_code = 0.0 if "line" in str(etype) else (1.0 if "trafo" in str(etype) else 2.0)
        r_pu = float(data.get("r_pu", data.get("r_ohm_per_km", 0.01)))
        x_pu = float(data.get("x_pu", data.get("x_ohm_per_km", 0.01)))
        raw_features.append((type_code, r_pu, x_pu, u, v, data))

    if not raw_features:
        return None

    max_r = max(abs(f[1]) for f in raw_features) or 1.0
    max_x = max(abs(f[2]) for f in raw_features) or 1.0

    edge_features = []
    for type_code, r_pu, x_pu, u, v, data in raw_features:
        r_norm = min(r_pu / max_r, 1.0)
        x_norm = min(x_pu / max_x, 1.0)
        bus_u = int(str(u).replace("bus_", "")) if "bus_" in str(u) else int(u) if str(u).isdigit() else 0
        bus_v = int(str(v).replace("bus_", "")) if "bus_" in str(v) else int(v) if str(v).isdigit() else 0
        lp = line_powers.get((bus_u, bus_v), line_powers.get((bus_v, bus_u), {}))
        p_mw = lp.get("p_mw", 0.0) / 10.0
        q_mvar = lp.get("q_mvar", 0.0) / 10.0
        closed = 1.0 if data.get("closed", data.get("in_service", True)) else 0.0
        edge_features.append([type_code, r_norm, x_norm, p_mw, q_mvar, closed])

    return torch.tensor(edge_features, dtype=torch.float)



# ===== Heterogeneous Power Graph (P1-1) =====

class HeteroGraphConv(nn.Module):
    """Heterogeneous graph convolution: different weights per edge type.
    Edge types: line(0), trafo(1), switch(2)
    """
    def __init__(self, in_channels, out_channels, num_edge_types=3):
        super().__init__()
        self.num_edge_types = num_edge_types
        # One linear per edge type + one for self-loop
        self.linears = nn.ModuleList([
            nn.Linear(in_channels, out_channels) for _ in range(num_edge_types + 1)
        ])
        for lin in self.linears:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x, edge_index, edge_type):
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        # Self-loop
        out = self.linears[-1](x)
        # Per-type aggregation
        for t in range(self.num_edge_types):
            mask = (edge_type == t)
            if mask.sum() == 0:
                continue
            src_t = src[mask]
            dst_t = dst[mask]
            msg = self.linears[t](x[src_t])
            agg = torch.zeros_like(out)
            agg.index_add_(0, dst_t, msg)
            deg = torch.zeros(N, dtype=torch.float, device=x.device)
            deg.index_add_(0, dst_t, torch.ones(dst_t.size(0), device=x.device))
            deg = deg.clamp(min=1.0).unsqueeze(1)
            out = out + agg / deg
        return out


class HeteroPowerGNN(nn.Module):
    """Heterogeneous GNN: 4 node types (bus/line/switch/trafo), 3 edge types.
    Uses HeteroGraphConv for type-aware message passing.
    """
    def __init__(self, in_channels=8, hidden_channels=128, out_channels=6,
                 num_edge_types=3, dropout=0.3):
        super().__init__()
        self.node_type_embed = nn.Embedding(4, 8)  # bus=0, line=1, switch=2, trafo=3
        actual_in = in_channels + 8  # features + type embedding
        self.conv1 = HeteroGraphConv(actual_in, hidden_channels, num_edge_types)
        self.conv2 = HeteroGraphConv(hidden_channels, hidden_channels // 2, num_edge_types)
        self.classifier = nn.Linear(hidden_channels // 2, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type=None, node_type=None):
        if node_type is not None:
            type_emb = self.node_type_embed(node_type)
            x = torch.cat([x, type_emb], dim=-1)
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)
        h = F.relu(self.conv1(x, edge_index, edge_type))
        h = self.dropout(h)
        h = F.relu(self.conv2(h, edge_index, edge_type))
        h = self.dropout(h)
        return self.classifier(h)


# ===== Graph Transformer Layer (P1-2) =====

class GraphTransformerLayer(nn.Module):
    """Graph Transformer: multi-head attention over graph neighborhoods.
    Combines structural (degree) + electrical (voltage/power) features.
    """
    def __init__(self, in_channels, out_channels, num_heads=4, dropout=0.1):
        super().__init__()
        assert out_channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        self.q_proj = nn.Linear(in_channels, out_channels)
        self.k_proj = nn.Linear(in_channels, out_channels)
        self.v_proj = nn.Linear(in_channels, out_channels)
        self.out_proj = nn.Linear(out_channels, out_channels)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels * 2, out_channels),
            nn.Dropout(dropout),
        )
        self.attn_dropout = nn.Dropout(dropout)
        # Laplacian positional encoding bias
        self.pe_bias = nn.Linear(2, num_heads, bias=False)

    def forward(self, x, edge_index):
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        # QKV
        Q = self.q_proj(x).view(N, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(N, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(N, self.num_heads, self.head_dim)
        # Attention scores per edge
        q_src = Q[dst]  # [E, H, D]
        k_src = K[src]  # [E, H, D]
        attn = (q_src * k_src).sum(dim=-1) / (self.head_dim ** 0.5)  # [E, H]
        # Sparse softmax per destination node
        attn = torch.softmax(
            self._scatter_softmax(attn, dst, N), dim=0
        )  # [E, H]
        attn = self.attn_dropout(attn)
        # Aggregate values
        v_src = V[src]  # [E, H, D]
        msg = attn.unsqueeze(-1) * v_src  # [E, H, D]
        out = torch.zeros(N, self.num_heads, self.head_dim, device=x.device)
        out.index_add_(0, dst, msg)
        out = out.view(N, -1)
        out = self.out_proj(out)
        # Residual + norm
        x_proj = self.norm1(out + x if x.size(-1) == out.size(-1) else out)
        out2 = self.ffn(x_proj)
        return self.norm2(out2 + x_proj)

    def _scatter_softmax(self, attn, dst, N):
        """Sparse softmax: normalize per destination node."""
        max_val = torch.full((N, self.num_heads), -1e9, device=attn.device)
        max_val.scatter_reduce_(0, dst.unsqueeze(1).expand_as(attn), attn, reduce="amax")
        attn = attn - max_val[dst]
        attn_exp = torch.exp(attn)
        sum_exp = torch.zeros(N, self.num_heads, device=attn.device)
        sum_exp.scatter_add_(0, dst.unsqueeze(1).expand_as(attn_exp), attn_exp)
        return attn_exp / (sum_exp[dst] + 1e-10)


class GraphTransformerGNN(nn.Module):
    """Graph Transformer for power grid topology anomaly detection.
    2-layer GraphTransformer + classification head.
    """
    def __init__(self, in_channels=8, hidden_channels=128, out_channels=6,
                 num_heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.gt1 = GraphTransformerLayer(hidden_channels, hidden_channels, num_heads, dropout)
        self.gt2 = GraphTransformerLayer(hidden_channels, hidden_channels, num_heads, dropout)
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        h = F.relu(self.input_proj(x))
        h = self.gt1(h, edge_index)
        h = self.dropout(h)
        h = self.gt2(h, edge_index)
        h = self.dropout(h)
        return self.classifier(h)


def build_hetero_graph_data(graph, network_data=None):
    """Build heterogeneous graph: node_type (4 types) + edge_type (3 types).
    Returns (node_features, edge_index, node_type, edge_type).
    """
    node_list = list(graph.nodes())
    node_idx = {n: i for i, n in enumerate(node_list)}
    node_type_map = {"bus": 0, "line": 1, "switch": 2, "trafo": 3}

    # Determine node types from graph attributes
    node_types = []
    for n in node_list:
        ndata = graph.nodes[n]
        ntype = ndata.get("type", "bus")
        node_types.append(node_type_map.get(ntype, 0))
    node_type = torch.tensor(node_types, dtype=torch.long)

    # Build edges with type
    edges_src, edges_dst, edge_types = [], [], []
    etype_map = {"line": 0, "trafo": 1, "switch": 2}
    for u, v, data in graph.edges(data=True):
        ui, vi = node_idx.get(u), node_idx.get(v)
        if ui is None or vi is None:
            continue
        etype = etype_map.get(data.get("type", "line"), 0)
        edges_src.extend([ui, vi])
        edges_dst.extend([vi, ui])
        edge_types.extend([etype, etype])

    if not edges_src:
        return None, None, node_type, None

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    return None, edge_index, node_type, edge_type



# ===== P2-1: Bayesian GNN with MC Dropout Uncertainty =====

class BayesianGNNWrapper:
    """Wraps any GNN model to provide uncertainty estimates via MC Dropout.
    
    During inference, runs N forward passes with dropout enabled.
    Returns mean prediction + uncertainty (std) per node.
    """
    def __init__(self, model, n_samples=10, dropout=0.3):
        self.model = model
        self.n_samples = n_samples
        self.dropout = dropout
    
    def predict_with_uncertainty(self, x, edge_index, edge_attr=None):
        """Run MC Dropout inference.
        
        Returns:
            mean_pred: [N, C] mean softmax predictions
            uncertainty: [N] per-node uncertainty (entropy of prediction std)
            confidence: [N] confidence = 1 - normalized_entropy
        """
        import torch
        import torch.nn.functional as F
        
        self.model.train()  # Enable dropout
        
        predictions = []
        for _ in range(self.n_samples):
            with torch.no_grad():
                if edge_attr is not None:
                    try:
                        out = self.model(x, edge_index, edge_attr)
                    except TypeError:
                        out = self.model(x, edge_index)
                else:
                    out = self.model(x, edge_index)
                probs = F.softmax(out, dim=1)
                predictions.append(probs)
        
        self.model.eval()
        
        stacked = torch.stack(predictions)  # [S, N, C]
        mean_pred = stacked.mean(dim=0)     # [N, C]
        
        # Uncertainty: std across samples, averaged over classes
        pred_std = stacked.std(dim=0)       # [N, C]
        uncertainty = pred_std.mean(dim=1)  # [N]
        
        # Confidence: 1 - normalized entropy
        entropy = -(mean_pred * torch.log(mean_pred + 1e-10)).sum(dim=1)
        max_entropy = torch.log(torch.tensor(float(mean_pred.size(1))))
        confidence = 1.0 - entropy / max_entropy
        
        return mean_pred, uncertainty, confidence
    
    def detect_with_uncertainty(self, network_data, threshold=0.5, uncertainty_threshold=0.3):
        """Detect anomalies with uncertainty-aware filtering.
        
        Only report detections where:
        1. Prediction confidence > threshold
        2. Uncertainty < uncertainty_threshold (model is sure)
        """
        graph = network_data.get("graph")
        if not graph:
            return []
        
        from anomaly_detection.gnn_detector import GNNDetector
        detector = GNNDetector()
        data = detector._build_data(graph, network_data)
        if data is None:
            return []
        
        if len(data) == 3:
            x, edge_index, edge_attr = data
        else:
            x, edge_index, edge_attr = data[0], data[1], None
        
        mean_pred, uncertainty, confidence = self.predict_with_uncertainty(
            x, edge_index, edge_attr
        )
        
        # Use the module-level ANOMALY_LABELS (29 types) defined at top of file
        from anomaly_detection.gnn_detector import ANOMALY_LABELS as _FULL_LABELS
        _ANOMALY_LABELS = _FULL_LABELS
        
        results = []
        node_list = list(graph.nodes())
        preds = mean_pred.argmax(dim=1)
        
        for i, node in enumerate(node_list):
            if preds[i] == 0:
                continue
            if confidence[i] < threshold:
                continue
            if uncertainty[i] > uncertainty_threshold:
                continue
            
            label = _ANOMALY_LABELS[preds[i]] if preds[i] < len(_ANOMALY_LABELS) else "unknown"
            results.append({
                "type": label,
                "location": str(node),
                "confidence": float(confidence[i]),
                "uncertainty": float(uncertainty[i]),
                "layer": "GNN_Bayesian",
                "details": f"MC-Dropout: conf={confidence[i]:.3f}, unc={uncertainty[i]:.3f}",
            })
        
        return results

class GNNDetector:
    """GNN异常检测器(纯PyTorch)"""

    def __init__(self, model_path: str = None, device: str = "cpu"):
        self.device = device
        self.model = None
        self.is_ready = False
        self.has_edge_features = False
        self._stats_model = False
        self._node_gnn = False
        self._label_map = {}
        self._inv_map = {}
        self._X_mean = None
        self._X_std = None
        self._load_model(model_path)

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    def _adapt_topo_weights(self, topo_ckpt: dict) -> dict:
        """将TopoGNN权重映射到EdgeTopologyGNN (兼容旧checkpoint)"""
        new_sd = {}
        for k, v in topo_ckpt.items():
            if k.startswith("conv1.linear."):
                suffix = k[len("conv1.linear."):]
                new_sd[f"conv1.linear_self.{suffix}"] = v
                new_sd[f"conv1.linear_neigh.{suffix}"] = v
            elif k.startswith("conv2.linear."):
                suffix = k[len("conv2.linear."):]
                new_sd[f"conv2.linear_self.{suffix}"] = v
                new_sd[f"conv2.linear_neigh.{suffix}"] = v
            else:
                new_sd[k] = v
        return new_sd

    def _load_model(self, path: str):
        if not path or not os.path.exists(path):
            logger.info("GNN模型文件不存在: %s, 使用默认EdgeTopologyGNN", path)
            self.model = EdgeTopologyGNN(in_channels=8, edge_channels=6)
            self.has_edge_features = True
            self.model.eval()
            return
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            sd = ckpt.get("model_state_dict", {})
            
            # Check for GAT model (v17)
            model_type = ckpt.get("model_type", "")
            if model_type == "GAT":
                from anomaly_detection.gnn_trainer import GATNet
                num_classes = ckpt.get("num_classes", 29)
                hidden_channels = ckpt.get("hidden_channels", 64)
                self.model = GATNet(in_channels=8, hidden_channels=hidden_channels, 
                                   num_classes=num_classes, heads=4)
                self.model.load_state_dict(sd)
                self.has_edge_features = False
                self.is_ready = True
                logger.info("GAT模型加载成功: %s (classes=%d)", path, num_classes)
            elif model_type == "TAG":
                from anomaly_detection.gnn_trainer import TAGNet
                num_classes = ckpt.get("num_classes", 29)
                hidden_channels = ckpt.get("hidden_channels", 64)
                self.model = TAGNet(in_channels=8, hidden_channels=hidden_channels,
                                   num_classes=num_classes, K=3)
                self.model.load_state_dict(sd)
                self.has_edge_features = False
                self.is_ready = True
                logger.info("TAG模型加载成功: %s (classes=%d)", path, num_classes)
            elif model_type == "NodeGNN":
                input_dim = ckpt.get("input_dim", 12)
                hidden_dim = ckpt.get("hidden_dim", 64)
                num_classes = ckpt.get("num_classes", 16)
                self.model = NodeGNN(in_dim=input_dim, hid=hidden_dim, num_classes=num_classes)
                self.model.load_state_dict(sd)
                self.has_edge_features = False
                self.is_ready = True
                self._stats_model = False
                self._node_gnn = True
                logger.info("NodeGNN loaded: %s (classes=%d, acc=%.1f%%)", path, num_classes, ckpt.get("val_acc", 0)*100)
            elif model_type in ("GraphStatsMLP", "GraphStatsMLP_v22"):
                # v20b/v22: Graph-level statistics MLP
                input_dim = ckpt.get("input_dim", 72)
                num_classes = ckpt.get("num_classes", 12)
                self.model = GraphStatsMLP(in_dim=input_dim, num_classes=num_classes)
                self.model.load_state_dict(sd)
                self.has_edge_features = False
                self.is_ready = True
                self._stats_model = True
                self._label_map = ckpt.get("label_map", {})
                self._inv_map = ckpt.get("inv_map", {})
                self._X_mean = torch.tensor(ckpt.get("X_mean", [0]*input_dim), dtype=torch.float)
                self._X_std = torch.tensor(ckpt.get("X_std", [1]*input_dim), dtype=torch.float)
                logger.info("GraphStatsMLP加载成功: %s (classes=%d, input=%d)", path, num_classes, input_dim)
            else:
                # Legacy model format
                has_edge_ckpt = "edge_channels" in ckpt or any(
                    k.startswith("conv1.edge_mlp") for k in sd)
                in_ch = ckpt.get("in_channels", 8)
                hid = ckpt.get("hidden_channels", 128)
                out = ckpt.get("out_channels", 6)

                if has_edge_ckpt:
                    edge_ch = ckpt.get("edge_channels", 6)
                    self.model = EdgeTopologyGNN(in_channels=in_ch,
                                                 edge_channels=edge_ch,
                                                 hidden_channels=hid,
                                                 out_channels=out)
                    try:
                        self.model.load_state_dict(sd, strict=True)
                    except RuntimeError:
                        self.model.load_state_dict(
                            self._adapt_topo_weights(sd), strict=False)
                    self.has_edge_features = True
                    logger.info("EdgeTopologyGNN加载成功: %s", path)
                else:
                    self.model = TopologyGNN(in_channels=in_ch,
                                             hidden_channels=hid,
                                             out_channels=out)
                    self.model.load_state_dict(sd)
                    self.has_edge_features = False
                    logger.info("TopologyGNN加载成功: %s", path)
                self.is_ready = True

            self.model.eval()
        except Exception as e:
            logger.error("GNN模型加载失败: %s", e)
            self.model = EdgeTopologyGNN(in_channels=8, edge_channels=6)
            self.has_edge_features = True
            self.model.eval()

    # ------------------------------------------------------------------
    # 检测
    # ------------------------------------------------------------------

    def detect(self, network_data: Dict) -> List[Dict]:
        if not self.is_ready or self.model is None:
            return []
        graph = network_data.get("graph")
        if not graph or graph.number_of_nodes() == 0:
            return []
        
        # v25: Node-level GNN path
        if self._node_gnn:
            return self._detect_node_gnn(graph, network_data)
        # v20b: GraphStatsMLP path
        if self._stats_model:
            return self._detect_stats(graph, network_data)
        
        try:
            data = self._build_data(graph, network_data)
            if data is None:
                return []
            x, edge_index, edge_attr = data
            with torch.no_grad():
                if edge_attr is not None and self.has_edge_features:
                    out = self.model(x, edge_index, edge_attr)
                else:
                    out = self.model(x, edge_index)
                probs = F.softmax(out, dim=1)
                preds = probs.argmax(dim=1)
            return self._parse(graph, preds, probs)
        except Exception as e:
            logger.error("GNN检测失败: %s", e)
            import traceback
            traceback.print_exc()
            return []

    # ------------------------------------------------------------------
    # 辅助构建
    # ------------------------------------------------------------------

    def _build_edge_features(
        self,
        graph,
        network_data: Dict,
        node_idx: dict,
    ) -> Optional[torch.Tensor]:
        """构建6维边特征

        特征定义:
          [0] 线路类型 (0=line, 1=trafo, 2=switch)
          [1] 电阻 R  (p.u., 除以0.1归一化到[0,1])
          [2] 电抗 X  (p.u., 除以0.1归一化到[0,1])
          [3] 有功潮流 P (MW, /max归一化)
          [4] 无功潮流 Q (MVAr, /max归一化)
          [5] 开关状态 (0=open, 1=closed)

        Returns: Tensor [num_directed_edges, 6] 或 None
        """
        measurements = network_data.get("measurements", {})
        bus_voltages = {bv["bus"]: bv for bv in measurements.get("bus_voltages", [])}

        line_info = {}
        for lp in measurements.get("line_powers", []):
            fb = int(lp.get("from_bus", -1))
            tb = int(lp.get("to_bus", -1))
            if fb < 0 or tb < 0:
                continue
            key = (min(fb, tb), max(fb, tb))
            line_info[key] = lp

        # 构建阻抗/开关参数 (优先从线路参数表获取)
        line_params = {}
        for lp in network_data.get("line_params", []):
            fb = int(lp.get("from_bus", -1))
            tb = int(lp.get("to_bus", -1))
            if fb < 0 or tb < 0:
                continue
            key = (min(fb, tb), max(fb, tb))
            line_params[key] = lp

        if not line_info and not line_params:
            logger.debug("无线路信息, 边特征为空")
            return None

        # 归一化潮流值
        all_p = [abs(lp.get("p_mw", 0.0)) for lp in line_info.values()]
        all_q = [abs(lp.get("q_mvar", 0.0)) for lp in line_info.values()]
        max_p = max(all_p) if all_p else 1.0
        max_q = max(all_q) if all_q else 1.0
        max_p = max(max_p, 1e-6)
        max_q = max(max_q, 1e-6)

        # 类型映射
        TYPE_MAP = {"line": 0, "trafo": 1, "transformer": 1, "switch": 2}

        features_ordered = []
        for u, v in graph.edges():
            ui = node_idx.get(u)
            vi = node_idx.get(v)
            if ui is None or vi is None:
                continue
            uid = int(str(u).replace("bus_", ""))
            vid = int(str(v).replace("bus_", ""))
            key = (min(uid, vid), max(uid, vid))
            lp = line_info.get(key, {})
            lparam = line_params.get(key, {})
            etype = str(lp.get("type", lparam.get("type", "line"))).lower()
            feat = [
                float(TYPE_MAP.get(etype, 0)),
                float(lp.get("r_pu", lparam.get("r_pu", 0.0))) / 0.1,
                float(lp.get("x_pu", lparam.get("x_pu", 0.01))) / 0.1,
                float(lp.get("p_mw", 0.0)) / max_p,
                float(lp.get("q_mvar", 0.0)) / max_q,
                float(lp.get("in_service", lparam.get("in_service", 1))),
            ]
            features_ordered.extend([feat, feat])  # u->v, v->u

        if not features_ordered:
            return None

        return torch.tensor(features_ordered, dtype=torch.float)

    def _detect_stats(self, graph, network_data) -> List[Dict]:
        """v24: Graph-level statistics detection using MLP (24-dim features)."""
        try:
            measurements = network_data.get("measurements", {})
            bus_voltages = {bv["bus"]: bv for bv in measurements.get("bus_voltages", [])}
            
            nodes = list(graph.nodes())
            n = len(nodes)
            if n == 0:
                return []
            
            bus_v = {}
            for bv in measurements.get("bus_voltages", []):
                bus_id = int(str(bv.get("bus", "")).replace("bus_", ""))
                bus_v[bus_id] = float(bv.get("vm_pu", 1.0))
            
            p_inject = {}
            q_inject = {}
            for lp in measurements.get("line_powers", []):
                fb = int(str(lp.get("from_bus", "")).replace("bus_", ""))
                tb = int(str(lp.get("to_bus", "")).replace("bus_", ""))
                p = float(lp.get("p_mw", 0.0))
                q = float(lp.get("q_mvar", 0.0))
                if not (p != p) and not (p == float("inf")):  # not NaN, not Inf
                    p_inject[fb] = p_inject.get(fb, 0.0) - p
                    p_inject[tb] = p_inject.get(tb, 0.0) + p
                if not (q != q) and not (q == float("inf")):
                    q_inject[fb] = q_inject.get(fb, 0.0) - q
                    q_inject[tb] = q_inject.get(tb, 0.0) + q
            
            for d in [p_inject, q_inject]:
                for k in list(d.keys()):
                    v = d[k]
                    if v != v or v == float("inf") or v == float("-inf"):
                        d[k] = 0.0
            
            max_p = max(abs(v) for v in p_inject.values()) if p_inject else 1.0
            max_p = max(max_p, 1e-6)
            max_q = max(abs(v) for v in q_inject.values()) if q_inject else 1.0
            max_q = max(max_q, 1e-6)
            
            src_buses = set()
            for node in nodes:
                node_id = int(str(node).replace("bus_", ""))
                bv = bus_voltages.get(node, bus_voltages.get(node_id, {}))
                if bv.get("is_source", False) or bv.get("bus_type", "") == "slack":
                    src_buses.add(node)
            
            comps = {}
            for ci, comp in enumerate(nx.connected_components(graph)):
                for nd in comp:
                    comps[nd] = ci
            max_comp = max(comps.values()) if comps else 1
            
            features = []
            for node in nodes:
                node_id = int(str(node).replace("bus_", ""))
                k = graph.degree(node)
                vm = bus_v.get(node_id, bus_v.get(node, 1.0))
                p = p_inject.get(node_id, p_inject.get(node, 0.0)) / max_p
                q = q_inject.get(node_id, q_inject.get(node, 0.0)) / max_q
                
                neighbors = list(graph.neighbors(node))
                n_edges = sum(1 for i, a in enumerate(neighbors) for b in neighbors[i+1:] if graph.has_edge(a, b))
                cl = (2.0 * n_edges) / (k * (k - 1)) if k > 1 else 0.0
                ed = (2.0 * k) / max(n - 1, 1)
                nv_v = [bus_v.get(int(str(nb).replace("bus_","")), 1.0) for nb in neighbors]
                nv_std = (sum((vv - vm)**2 for vv in nv_v) / max(len(nv_v), 1)) ** 0.5
                sc = sum(1 for nb in neighbors if (p > 0 and p_inject.get(int(str(nb).replace("bus_","")), 0)/max_p < 0) or (p < 0 and p_inject.get(int(str(nb).replace("bus_","")), 0)/max_p > 0)) / max(k, 1)
                bt = 0.0 if node in src_buses else (0.5 if p > 0.01 else (0.75 if p < -0.01 else 1.0))
                nb_degrees = [graph.degree(nb) for nb in neighbors]
                nb_deg_mean = sum(nb_degrees) / max(len(nb_degrees), 1)
                v_grad = nv_std / max(abs(vm), 1e-6)
                p_abs = abs(p_inject.get(node_id, 0.0))
                q_abs = abs(q_inject.get(node_id, 0.0))
                pf = p_abs / max((p_abs**2 + q_abs**2)**0.5, 1e-6)
                comp_ratio = sum(1 for c in comps.values() if c == comps.get(node, 0)) / max(n, 1)
                nv_range = max(nv_v) - min(nv_v) if nv_v else 0.0
                
                feat = [k/10.0, vm, float(node in src_buses), p, q, vm-1.0, bt,
                        comps.get(node, 0)/max(max_comp, 1), cl, nv_std, sc, ed,
                        0.0, 1.0/n, nb_deg_mean/10.0, v_grad, pf, comp_ratio,
                        float(k == 1), float(k >= 4), abs(p)*vm, nv_range, 0.0,
                        q/max(abs(p), 0.01)]
                features.append(feat)
            
            x = torch.tensor(features, dtype=torch.float32)
            x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
            
            # Graph-level statistics: 24 features * 6 stats = 144
            mean = x.mean(dim=0)
            std = x.std(dim=0)
            min_v = x.min(dim=0)[0]
            max_v = x.max(dim=0)[0]
            range_v = max_v - min_v
            median = x.median(dim=0)[0]
            stats = torch.cat([mean, std, min_v, max_v, range_v, median]).unsqueeze(0)
            
            # Normalize
            if self._X_mean is not None and self._X_std is not None:
                stats = (stats - self._X_mean) / self._X_std.clamp(min=1e-6)
            
            # Predict
            with torch.no_grad():
                self.model.eval()
                out = self.model(stats)
                probs = F.softmax(out, dim=1)
                pred_idx = probs.argmax(dim=1).item()
                conf = probs[0, pred_idx].item()
            
            # Map back to original label
            pred_label = self._inv_map.get(pred_idx, pred_idx) if self._inv_map else pred_idx
            
            # Return top-3 predictions
            results = []
            top_probs, top_indices = probs[0].topk(min(3, probs.shape[1]))
            for k_idx in range(len(top_indices)):
                p_idx = top_indices[k_idx].item()
                p_conf = top_probs[k_idx].item()
                p_label = self._inv_map.get(p_idx, p_idx) if self._inv_map else p_idx
                if p_label == 0 or p_conf < 0.15:
                    continue
                from anomaly_detection.gnn_trainer import ANOMALY_LABELS
                results.append({
                    "type": ANOMALY_LABELS.get(p_label, str(p_label)),
                    "location": "network",
                    "confidence": float(p_conf),
                    "layer": "GNN",
                    "details": f"GNN-stats: type={ANOMALY_LABELS.get(p_label, '?')}, conf={p_conf:.2%}",
                })
            return results
        except Exception as e:
            logger.error("GNN stats detection failed: %s", e)
            return []


    def _detect_node_gnn(self, graph, network_data) -> List[Dict]:
        """v25: Node-level GNN detection."""
        try:
            import math
            measurements = network_data.get("measurements", {})
            bus_voltages = {bv["bus"]: bv for bv in measurements.get("bus_voltages", [])}
            
            nodes = list(graph.nodes())
            n = len(nodes)
            if n == 0:
                return []
            
            node_map = {nd: i for i, nd in enumerate(nodes)}
            
            # Build 12-dim node features
            bus_v = {}
            for bv in measurements.get("bus_voltages", []):
                bus_id = int(str(bv.get("bus", "")).replace("bus_", ""))
                bus_v[bus_id] = float(bv.get("vm_pu", 1.0))
            
            p_inject = {}
            for lp in measurements.get("line_powers", []):
                fb = int(str(lp.get("from_bus", "")).replace("bus_", ""))
                tb = int(str(lp.get("to_bus", "")).replace("bus_", ""))
                p = float(lp.get("p_mw", 0.0))
                if not (p != p) and not (p == float("inf")):
                    p_inject[fb] = p_inject.get(fb, 0.0) - p
                    p_inject[tb] = p_inject.get(tb, 0.0) + p
            for k in list(p_inject.keys()):
                v = p_inject[k]
                if v != v or v == float("inf") or v == float("-inf"):
                    p_inject[k] = 0.0
            
            max_p = max(abs(v) for v in p_inject.values()) if p_inject else 1.0
            max_p = max(max_p, 1e-6)
            
            src_buses = set()
            for node in nodes:
                node_id = int(str(node).replace("bus_", ""))
                bv = bus_voltages.get(node, bus_voltages.get(node_id, {}))
                if bv.get("is_source", False) or bv.get("bus_type", "") == "slack":
                    src_buses.add(node)
            
            comps = {}
            for ci, comp in enumerate(nx.connected_components(graph)):
                for nd in comp:
                    comps[nd] = ci
            max_comp = max(comps.values()) if comps else 1
            
            features = []
            for node in nodes:
                node_id = int(str(node).replace("bus_", ""))
                k = graph.degree(node)
                vm = bus_v.get(node_id, bus_v.get(node, 1.0))
                p = p_inject.get(node_id, p_inject.get(node, 0.0)) / max_p
                nbs = list(graph.neighbors(node))
                n_edges = sum(1 for i, a in enumerate(nbs) for b in nbs[i+1:] if graph.has_edge(a, b))
                cl = (2.0 * n_edges) / (k * (k - 1)) if k > 1 else 0.0
                ed = (2.0 * k) / max(n - 1, 1)
                nv_v = [bus_v.get(int(str(nb).replace("bus_","")), 1.0) for nb in nbs]
                nv_std = (sum((vv - vm)**2 for vv in nv_v) / max(len(nv_v), 1)) ** 0.5
                bt = 0.0 if node in src_buses else (0.5 if p > 0.01 else (0.75 if p < -0.01 else 1.0))
                
                feat = [k/10.0, vm, float(node in src_buses), p, vm-1.0, bt,
                        comps.get(node, 0)/max(max_comp, 1), cl, nv_std, ed,
                        float(k == 1), float(k >= 4)]
                features.append(feat)
            
            x = torch.tensor(features, dtype=torch.float32)
            x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
            
            # Build edge_index
            edge_index = []
            for u, v in graph.edges():
                ui, vi = node_map.get(u), node_map.get(v)
                if ui is not None and vi is not None:
                    edge_index.append([ui, vi])
                    edge_index.append([vi, ui])
            if not edge_index:
                return []
            ei = torch.tensor(edge_index, dtype=torch.long).t()
            
            # Predict
            with torch.no_grad():
                self.model.eval()
                out = self.model(x, ei)
                probs = F.softmax(out, dim=1)
            
            # Aggregate: find nodes with high anomaly probability
            results = []
            anomaly_cols = [c for c in range(1, probs.shape[1])]  # exclude normal (col 0)
            
            # Count per-anomaly-type node predictions
            from collections import Counter
            type_counts = Counter()
            type_confs = {}
            for i in range(len(nodes)):
                for c in anomaly_cols:
                    conf = probs[i, c].item()
                    if conf > 0.3:
                        type_counts[c] += 1
                        if c not in type_confs or conf > type_confs[c]:
                            type_confs[c] = conf
            
            # Report top anomaly types
            ANOMALY_LABELS = [
                "normal", "topo_interrupt", "virtual_faulty", "model_mismatch",
                "telemetry_mismatch", "signal_mismatch", "measurement_outlier",
                "stale_data", "parameter_error", "load_shift", "reverse_power_flow",
                "communication_loss", "voltage_collapse", "ghost_topology",
                "duplicate_measurement", "protection_misconfig",
            ]
            
            for type_idx, count in type_counts.most_common(3):
                conf = type_confs.get(type_idx, 0.5)
                label = ANOMALY_LABELS[type_idx] if type_idx < len(ANOMALY_LABELS) else str(type_idx)
                results.append({
                    "type": label,
                    "location": "network",
                    "confidence": float(conf),
                    "layer": "GNN-node",
                    "details": f"NodeGNN: {count} nodes flagged as {label}, max_conf={conf:.2%}",
                })
            
            return results
        except Exception as e:
            logger.error("NodeGNN detection failed: %s", e)
            return []


    def _build_data(self, graph, network_data):
        """Build node features + edge features.

        Returns:
            (x, edge_index, edge_attr) 或 None
            edge_attr: Tensor [num_edges, 6] 或 None
        """
        node_list = list(graph.nodes())
        n_nodes = len(node_list)
        node_idx = {n: i for i, n in enumerate(node_list)}
        measurements = network_data.get("measurements", {})
        bus_voltages = {bv["bus"]: bv for bv in measurements.get("bus_voltages", [])}

        # Compute power injection per bus from line powers
        p_inject = {}
        q_inject = {}
        for lp in measurements.get("line_powers", []):
            fb, tb = int(lp.get("from_bus", -1)), int(lp.get("to_bus", -1))
            p = lp.get("p_mw", 0.0)
            side = lp.get("side", "from")
            if side == "from":
                p_inject[fb] = p_inject.get(fb, 0.0) - p
                p_inject[tb] = p_inject.get(tb, 0.0) + p
            else:
                p_inject[tb] = p_inject.get(tb, 0.0) - p
                p_inject[fb] = p_inject.get(fb, 0.0) + p

        # Identify source buses from graph metadata
        source_buses = set()
        for node in node_list:
            if graph.nodes[node].get("is_source", False):
                bus_id = int(str(node).replace("bus_", "")) if "bus_" in str(node) else node
                source_buses.add(bus_id)

        # Compute connected components
        import networkx as nx
        components = {}
        for comp_id, comp in enumerate(nx.connected_components(graph)):
            for node in comp:
                components[node] = comp_id
        max_comp = max(components.values()) if components else 1

        # Normalize p_inject
        max_p = max(abs(v) for v in p_inject.values()) if p_inject else 1.0
        max_p = max(max_p, 1e-6)

        features = []
        for node in node_list:
            deg = graph.degree(node)
            bus_id = int(str(node).replace("bus_", "")) if "bus_" in str(node) else node
            # 使用bus_id(int)查找，因为measurements中bus字段是整数
            bv = bus_voltages.get(bus_id, bus_voltages.get(node, {}))
            vm = bv.get("vm_pu", 1.0)

            p = p_inject.get(bus_id, 0.0) / max_p

            # Determine bus type
            if bus_id in source_buses:
                bus_type = 0.0  # slack
            elif p > 0.01:
                bus_type = 0.5  # gen (positive injection)
            elif p < -0.01:
                bus_type = 0.75  # load (negative injection)
            else:
                bus_type = 1.0  # bus

            # v20: 扩展到12维特征, 新增4维拓扑结构特征
            # 计算局部聚类系数
            neighbors = list(graph.neighbors(node))
            n_edges_between = 0
            for ni, nj in [(a, b) for i_a, a in enumerate(neighbors) for b in neighbors[i_a+1:]]:
                if graph.has_edge(ni, nj):
                    n_edges_between += 1
            k = len(neighbors)
            clustering = (2.0 * n_edges_between) / (k * (k - 1)) if k > 1 else 0.0
            
            # 邻居电压标准差 (电压异常的空间聚集性)
            neighbor_voltages = []
            for nb in neighbors:
                nb_id = int(str(nb).replace("bus_", "")) if "bus_" in str(nb) else nb
                nb_bv = bus_voltages.get(nb_id, bus_voltages.get(nb, {}))
                neighbor_voltages.append(nb_bv.get("vm_pu", 1.0))
            nv_std = (sum((v - vm)**2 for v in neighbor_voltages) / max(len(neighbor_voltages), 1)) ** 0.5
            
            # 注入功率符号变化数 (异常区域边界检测)
            sign_changes = 0
            for nb in neighbors:
                nb_id = int(str(nb).replace("bus_", "")) if "bus_" in str(nb) else nb
                nb_p = p_inject.get(nb_id, 0.0) / max_p
                if (p > 0 and nb_p < 0) or (p < 0 and nb_p > 0):
                    sign_changes += 1
            sign_changes_norm = sign_changes / max(k, 1)
            
            # 边密度 (节点在局部拓扑中的连接程度)
            edge_density = (2.0 * graph.degree(node)) / max(n_nodes - 1, 1)
            
            feat = [
                deg / 10.0,                          # [0] degree
                vm,                                   # [1] voltage
                1.0 if bus_id in source_buses else 0.0,  # [2] is_source
                p,                                    # [3] p_inject (normalized)
                0.0,                                  # [4] q_inject (placeholder)
                vm - 1.0,                             # [5] voltage deviation
                bus_type,                             # [6] bus_type
                components.get(node, 0) / max(max_comp, 1),  # [7] component_id
                clustering,                           # [8] clustering_coefficient
                nv_std,                               # [9] neighbor_voltage_std
                sign_changes_norm,                    # [10] p_inject_sign_changes
                edge_density,                         # [11] edge_density
            ]
            features.append(feat)

        x = torch.tensor(features, dtype=torch.float)
        edges = []
        for u, v in graph.edges():
            if u in node_idx and v in node_idx:
                edges.append([node_idx[u], node_idx[v]])
                edges.append([node_idx[v], node_idx[u]])
        if not edges:
            return None
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # 构建边特征 (6维)
        edge_attr = self._build_edge_features(graph, network_data, node_idx)

        # Build edge features (must match directed edge count)
        edge_attr_undir = build_edge_features_from_graph(graph, network_data)
        if edge_attr_undir is not None:
            # Duplicate for bidirectional edges
            n_undir = edge_attr_undir.size(0)
            n_dir = edge_index.size(1)
            if n_dir == 2 * n_undir:
                edge_attr = torch.cat([edge_attr_undir, edge_attr_undir], dim=0)
            elif n_dir == n_undir:
                edge_attr = edge_attr_undir
            else:
                edge_attr = None
            if edge_attr is not None:
                return x, edge_index, edge_attr
        return x, edge_index, None

    def _parse(self, graph, preds, probs) -> List[Dict]:
        """v20: 改进解析逻辑 - 多标签支持 + 自适应阈值
        
        改进:
        - 每个节点可返回top-2预测 (如果第二高概率也超过阈值)
        - 对高置信度异常(>0.7)降低阈值要求
        - 过滤正常类(0)的误检
        """
        node_list = list(graph.nodes())
        anomalies = []
        for node_i, (pred, prob_vec) in enumerate(zip(preds.tolist(), probs.tolist())):
            prob = prob_vec
            max_prob = max(prob)
            
            # 主预测
            if pred == 0:
                continue
            
            # 自适应阈值: 高置信度(>0.5)直接通过, 否则需要>0.2
            threshold = 0.2 if max_prob > 0.5 else 0.25
            if max_prob < threshold:
                continue
                
            anomalies.append({
                "type": ANOMALY_LABELS[pred] if pred < len(ANOMALY_LABELS) else "未知",
                "location": str(node_list[node_i]),
                "confidence": float(max_prob),
                "layer": "GNN",
                "details": "GNN: type={}, conf={:.2%}".format(
                    ANOMALY_LABELS[pred] if pred < len(ANOMALY_LABELS) else "?", max_prob),
            })
        return anomalies

