# -*- coding: utf-8 -*-
"""
GNN训练数据生成器与训练脚本
纯PyTorch实现GraphSAGE，不依赖torch_geometric

功能:
1. SimpleSAGEConv - 纯PyTorch的GraphSAGE卷积层
2. TrainingDataGenerator - 基于PandaPower的5类异常注入数据生成
3. GNNTrainer - 完整训练循环(训练/验证/保存最佳模型)
4. main() - 使用case33bw训练并保存模型

异常类型:
  0 = 正常
  1 = 拓扑中断(断线)
  2 = 虚接错接(改连接)
  3 = 遥测矛盾(改量测)
  4 = 图模不符(删设备)
  5 = 遥信矛盾(改开关)

节点特征(8维):
  [0] 归一化度数 (degree/10)
  [1] 电压幅值 (vm_pu)
  [2] 是否电源节点 (0/1)
  [3] 有功注入 (p_mw, 归一化)
  [4] 无功注入 (q_mvar, 归一化)
  [5] 电压偏差 (vm_pu - 1.0)
  [6] 节点类型编码 (bus/slack/load/gen)
  [7] 连通分量ID (归一化)

参考:
  - gnn_detector.py 中的 TopologyGNN 架构
  - config.py 中的 ANOMALY_TYPES 定义
"""

import copy
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

logger = logging.getLogger(__name__)

# ============================================================
# 常量定义
# ============================================================
NUM_ANOMALY_CLASSES = 29  # 0=正常 + 28类异常 (v16: +3调度实际类型)
NODE_FEATURE_DIM = 12    # 节点特征维度 (v20: 8->12, 新增4维拓扑特征)
DEFAULT_HIDDEN_DIM = 128  # v20: 64->128, 更深网络
DEFAULT_DROPOUT = 0.3


# Physics-informed loss (v16)
try:
    from anomaly_detection.physics_loss import physics_informed_loss
    HAS_PHYSICS_LOSS = True
except ImportError:
    HAS_PHYSICS_LOSS = False

# 异常类型名称映射
ANOMALY_LABELS = {
    0: "正常",
    1: "topo_interrupt", 2: "virtual_faulty", 3: "model_mismatch",
    4: "telemetry_mismatch", 5: "signal_mismatch",
    6: "measurement_outlier", 7: "stale_data", 8: "parameter_error",
    9: "load_shift", 10: "reverse_power_flow", 11: "communication_loss",
    12: "voltage_collapse", 13: "ghost_topology", 14: "duplicate_measurement",
    15: "protection_misconfig",
    16: "trafo_tap_fault", 17: "grounding_fault", 18: "clock_drift",
    19: "harmonic_pollution", 20: "impedance_degradation", 21: "dg_intermittent",
    22: "measurement_bias", 23: "branch_contingency", 24: "topo_obfuscation",
    25: "voltage_regulation",
    26: "bus_section_mismatch",    # 母线分段开关状态与拓扑不一致
    27: "bypass_operation",        # 旁路代路操作后拓扑未更新
    28: "load_transfer_residual",  # 负荷转供后拓扑残留
}

# 项目根目录
PROJECT_ROOT = Path(r"E:\项目大全\电力拓扑图修正")
OUTPUT_DIR = PROJECT_ROOT / "02_算法代码" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. SimpleSAGEConv - 纯PyTorch GraphSAGE卷积层
# ============================================================
class SimpleSAGEConv(nn.Module):
    """
    纯PyTorch实现的GraphSAGE卷积层

    GraphSAGE核心公式:
      h_N = AGGREGATE({h_u, u in N(v)})  -- 邻居聚合(均值)
      h_v = CONCAT(h_v, h_N)             -- 拼接自身与邻居
      h_v = W * h_v + b                  -- 线性变换
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = nn.Linear(2 * in_channels, out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        # 邻居均值聚合
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])
        degree = torch.zeros(N, dtype=torch.float, device=x.device)
        degree.index_add_(0, dst, torch.ones(dst.size(0), device=x.device))
        degree = degree.clamp(min=1.0)
        agg = agg / degree.unsqueeze(1)
        # 拼接自身 + 邻居
        combined = torch.cat([x, agg], dim=1)
        return self.linear(combined)


# ============================================================
# 2. GraphSAGE网络模型
# ============================================================

class PhysicsConstrainedLoss(nn.Module):
    """Physics-Informed loss: CE + lambda_kcl * KCL_violation + lambda_kvl * KVL_violation.
    
    KCL: sum of power injections at each bus should balance.
    KVL: voltage drops around loops should sum to zero (for radial, check branch drops).
    """
    def __init__(self, class_weight=None, lambda_kcl=0.1, lambda_kvl=0.05):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weight)
        self.lambda_kcl = lambda_kcl
        self.lambda_kvl = lambda_kvl

    def forward(self, pred, target, edge_index=None, node_features=None):
        data_loss = self.ce_loss(pred, target)
        
        if edge_index is None or node_features is None:
            return data_loss
        
        # KCL violation: power balance at each node
        # node_features[:, 3] = p_inject (normalized), node_features[:, 4] = q_inject
        # Sum of neighbor power flows should match injection
        src, dst = edge_index[0], edge_index[1]
        n = node_features.size(0)
        
        # Compute predicted anomaly scores as "power correction factors"
        pred_probs = F.softmax(pred, dim=1)
        # Normal nodes should have balanced power
        anomaly_score = 1.0 - pred_probs[:, 0]  # probability of being anomalous
        
        # KCL: for normal nodes, power injection should be near zero
        p_inject = node_features[:, 3]  # normalized p_inject
        kcl_violation = (p_inject * anomaly_score).pow(2).mean()
        
        # KVL: voltage deviation should be small for normal nodes
        v_dev = node_features[:, 5]  # vm_pu - 1.0
        kvl_violation = (v_dev * anomaly_score).pow(2).mean()
        
        total = data_loss + self.lambda_kcl * kcl_violation + self.lambda_kvl * kvl_violation
        return total



# ============================================================
# 2b. GATConv - 图注意力卷积层 (多头注意力)
# 来源: EmergentGNN (8-head attention + physics constraint)
# ============================================================
class GATConvLayer(nn.Module):
    """Graph Attention Network卷积层 (纯PyTorch实现)"""
    def __init__(self, in_channels: int, out_channels: int, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.heads = heads
        self.head_dim = out_channels // heads
        assert out_channels % heads == 0, "out_channels must be divisible by heads"
        
        # 每个注意力头的参数
        self.W = nn.Linear(in_channels, out_channels, bias=False)
        self.a_src = nn.Parameter(torch.zeros(heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.zeros(heads, self.head_dim))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        h = self.W(x).view(N, self.heads, self.head_dim)  # [N, heads, head_dim]
        
        src, dst = edge_index[0], edge_index[1]
        
        # 注意力系数
        attn_src = (h * self.a_src).sum(dim=-1)  # [N, heads]
        attn_dst = (h * self.a_dst).sum(dim=-1)  # [N, heads]
        
        attn = self.leaky_relu(attn_src[src] + attn_dst[dst])  # [E, heads]
        
        # Softmax per destination node
        attn_max = torch.zeros(N, self.heads, device=x.device)
        attn_max.scatter_reduce_(0, dst.unsqueeze(1).expand_as(attn), attn, reduce="amax")
        attn = torch.exp(attn - attn_max[dst])
        
        attn_sum = torch.zeros(N, self.heads, device=x.device)
        attn_sum.scatter_add_(0, dst.unsqueeze(1).expand_as(attn), attn)
        attn = attn / (attn_sum[dst] + 1e-10)
        attn = self.dropout(attn)
        
        # 聚合
        out = torch.zeros(N, self.heads, self.head_dim, device=x.device)
        msg = h[src] * attn.unsqueeze(-1)  # [E, heads, head_dim]
        out.scatter_add_(0, dst.unsqueeze(1).unsqueeze(2).expand_as(msg), msg)
        
        return out.view(N, -1) + self.bias


class TopologyAdaptiveConv(nn.Module):
    """TAGConv - 拓扑自适应图卷积 (K-hop邻居)"""
    def __init__(self, in_channels: int, out_channels: int, K: int = 3):
        super().__init__()
        self.K = K
        self.linears = nn.ModuleList([nn.Linear(in_channels, out_channels, bias=False) for _ in range(K + 1)])
        self.bias = nn.Parameter(torch.zeros(out_channels))
        for lin in self.linears:
            nn.init.xavier_uniform_(lin.weight)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        
        # 构建归一化邻接
        deg = torch.zeros(N, device=x.device)
        deg.scatter_add_(0, dst, torch.ones(src.size(0), device=x.device))
        deg_inv = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
        norm = deg_inv[src] * deg_inv[dst]
        
        out = self.linears[0](x)
        x_k = x
        for k in range(1, self.K + 1):
            # 消息传递
            msg = x_k[src] * norm.unsqueeze(-1)
            x_k = torch.zeros_like(x_k)
            x_k.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)
            out = out + self.linears[k](x_k)
        
        return out + self.bias


class GATNet(nn.Module):
    """GAT网络 v20 - 3层注意力 + 残差连接 + BatchNorm
    
    改进:
    - 3层GAT (原2层): 接收野从2跳扩展到3跳
    - 128 hidden (原64): 更大容量处理29类异常
    - 残差连接: 缓解深层梯度消失
    - BatchNorm: 稳定训练
    """
    def __init__(self, in_channels: int, hidden_channels: int = 128, num_classes: int = 29, heads: int = 4):
        super().__init__()
        self.conv1 = GATConvLayer(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConvLayer(hidden_channels, hidden_channels, heads=heads)
        self.conv3 = GATConvLayer(hidden_channels, hidden_channels // 2, heads=heads)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels // 2)
        self.classifier = nn.Linear(hidden_channels // 2, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x, edge_index):
        # Layer 1
        h1 = F.relu(self.bn1(self.conv1(x, edge_index)))
        h1 = self.dropout(h1)
        # Layer 2 + residual
        h2 = F.relu(self.bn2(self.conv2(h1, edge_index)))
        h2 = self.dropout(h2) + h1  # 残差连接
        # Layer 3
        h3 = F.relu(self.bn3(self.conv3(h2, edge_index)))
        h3 = self.dropout(h3)
        return self.classifier(h3)


class TAGNet(nn.Module):
    """TAG网络 - 拓扑自适应卷积"""
    def __init__(self, in_channels: int, hidden_channels: int = 64, num_classes: int = 29, K: int = 3):
        super().__init__()
        self.conv1 = TopologyAdaptiveConv(in_channels, hidden_channels, K=K)
        self.conv2 = TopologyAdaptiveConv(hidden_channels, hidden_channels // 2, K=K)
        self.classifier = nn.Linear(hidden_channels // 2, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        return self.classifier(x)


class GraphSAGENet(nn.Module):
    """
    纯PyTorch GraphSAGE拓扑异常检测网络

    架构与TopologyGNN一致:
      SimpleSAGEConv(in, 64) -> ReLU -> Dropout
      SimpleSAGEConv(64, 32) -> ReLU -> Dropout
      Linear(32, 6) -> 节点分类
    """

    def __init__(self, in_channels=NODE_FEATURE_DIM,
                 hidden_channels=DEFAULT_HIDDEN_DIM,
                 out_channels=NUM_ANOMALY_CLASSES,
                 dropout=DEFAULT_DROPOUT):
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


# ============================================================
# 3. TrainingDataGenerator - 训练数据生成器
# ============================================================
class TrainingDataGenerator:
    """
    基于PandaPower网络的训练数据生成器

    通过注入5类异常生成带标签的图数据:
      1. 拓扑中断(断线): 将线路设为停运
      2. 虚接错接(改连接): 修改线路的连接母线
      3. 遥测矛盾(改量测): 添加电压/功率量测噪声
      4. 图模不符(删设备): 删除网络中的设备
      5. 遥信矛盾(改开关): 随机改变开关状态

    每个样本输出:
      - node_features: [N, 8] 节点特征矩阵
      - edge_index: [2, E] 边索引(双向)
      - labels: [N] 节点标签(0-5)
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        logger.info("TrainingDataGenerator初始化, seed=%d", seed)

    def generate_from_pandapower(self, net, n_samples: int = 500) -> List[Dict]:
        """
        从PandaPower网络生成训练样本

        Args:
            net: PandaPower网络对象
            n_samples: 生成样本数

        Returns:
            样本列表，每个样本包含:
              - node_features: np.ndarray [N, 8]
              - edge_index: np.ndarray [2, E]
              - labels: np.ndarray [N]
              - anomaly_types: list 注入的异常类型名
        """
        logger.info("开始生成训练数据: %d个样本, %d条母线",
                     n_samples, len(net.bus))
        samples = []
        n_buses = len(net.bus)

        for i in range(n_samples):
            try:
                sample = self._generate_one_sample(net, n_buses, i)
                samples.append(sample)
            except Exception as e:
                logger.warning("样本%d生成失败: %s", i, e)
                samples.append(self._generate_normal_sample(net, n_buses))

        logger.info("训练数据生成完成: %d个样本", len(samples))
        return samples

    def _generate_one_sample(self, net, n_buses: int, sample_id: int) -> Dict:
        """生成单个训练样本"""
        inj_net = copy.deepcopy(net)
        labels = np.zeros(n_buses, dtype=np.int64)
        anomaly_names = []

        # 70%概率注入1-2种异常
        if self.rng.random() < 0.7:
            n_anomalies = int(self.rng.integers(1, 3))
            chosen = self.rng.choice(28, size=n_anomalies, replace=False) + 1
            for anomaly_type in chosen:
                self._inject_anomaly(inj_net, int(anomaly_type), labels, n_buses)
                anomaly_names.append(ANOMALY_LABELS[int(anomaly_type)])

        features = self._extract_features(inj_net, n_buses)
        edge_index = self._build_edge_index(inj_net)

        return {
            "node_features": features,
            "edge_index": edge_index,
            "labels": labels,
            "anomaly_types": anomaly_names,
        }

    def _generate_normal_sample(self, net, n_buses: int) -> Dict:
        """生成正常样本(兜底)"""
        features = self._extract_features(net, n_buses)
        edge_index = self._build_edge_index(net)
        return {
            "node_features": features,
            "edge_index": edge_index,
            "labels": np.zeros(n_buses, dtype=np.int64),
            "anomaly_types": [],
        }

    def _inject_anomaly(self, net, anomaly_type: int, labels: np.ndarray,
                        n_buses: int):
        """向网络注入指定类型的异常 (28种: 原25 + v16新增3种)"""
        _map = {
            1: lambda: self._inject_topology_interrupt(net, labels),
            2: lambda: self._inject_virtual_faulty(net, labels, n_buses),
            3: lambda: self._inject_telemetry_mismatch(net, labels, n_buses),
            4: lambda: self._inject_model_mismatch(net, labels),
            5: lambda: self._inject_signal_mismatch(net, labels, n_buses),
            6: lambda: self._inject_measurement_outlier(net, labels, n_buses),
            7: lambda: self._inject_stale_data(net, labels, n_buses),
            8: lambda: self._inject_parameter_error(net, labels, n_buses),
            9: lambda: self._inject_load_shift(net, labels, n_buses),
            10: lambda: self._inject_reverse_power_flow(net, labels, n_buses),
            11: lambda: self._inject_communication_loss(net, labels, n_buses),
            12: lambda: self._inject_voltage_collapse(net, labels, n_buses),
            13: lambda: self._inject_ghost_topology(net, labels, n_buses),
            14: lambda: self._inject_duplicate_measurement(net, labels, n_buses),
            15: lambda: self._inject_protection_misconfig(net, labels, n_buses),
            16: lambda: self._inject_trafo_tap_fault(net, labels, n_buses),
            17: lambda: self._inject_grounding_fault(net, labels, n_buses),
            18: lambda: self._inject_clock_drift(net, labels, n_buses),
            19: lambda: self._inject_harmonic_pollution(net, labels, n_buses),
            20: lambda: self._inject_impedance_degradation(net, labels, n_buses),
            21: lambda: self._inject_dg_intermittent(net, labels, n_buses),
            22: lambda: self._inject_measurement_bias(net, labels, n_buses),
            23: lambda: self._inject_branch_contingency(net, labels, n_buses),
            24: lambda: self._inject_topo_obfuscation(net, labels, n_buses),
            25: lambda: self._inject_voltage_regulation(net, labels, n_buses),
            # v16新增: 3种调度实际异常类型 (26-28)
            26: lambda: self._inject_bus_section_mismatch(net, labels, n_buses),
            27: lambda: self._inject_bypass_operation(net, labels, n_buses),
            28: lambda: self._inject_load_transfer_residual(net, labels, n_buses),
        }
        fn = _map.get(anomaly_type)
        if fn:
            fn()

    def _inject_topology_interrupt(self, net, labels: np.ndarray):
        """注入拓扑中断: 断开一条线路"""
        if len(net.line) < 3:
            return
        candidates = net.line.index[1:]
        if len(candidates) == 0:
            return
        idx = int(self.rng.choice(candidates))
        net.line.at[idx, "in_service"] = False
        fb = int(net.line.at[idx, "from_bus"])
        tb = int(net.line.at[idx, "to_bus"])
        if fb < len(labels):
            labels[fb] = 1
        if tb < len(labels):
            labels[tb] = 1
        logger.debug("注入拓扑中断: line_%d (%d->%d)", idx, fb, tb)

    def _inject_virtual_faulty(self, net, labels: np.ndarray, n_buses: int):
        """注入虚接错接: 修改线路的连接目标"""
        if len(net.line) < 3:
            return
        candidates = net.line.index[2:]
        if len(candidates) == 0:
            return
        idx = int(self.rng.choice(candidates))
        old_to = int(net.line.at[idx, "to_bus"])
        all_buses = list(net.bus.index)
        valid_buses = [int(b) for b in all_buses
                       if int(b) != old_to
                       and int(b) != int(net.line.at[idx, "from_bus"])]
        if not valid_buses:
            return
        new_to = int(self.rng.choice(valid_buses))
        net.line.at[idx, "to_bus"] = new_to
        fb = int(net.line.at[idx, "from_bus"])
        if fb < len(labels):
            labels[fb] = 2
        if new_to < len(labels):
            labels[new_to] = 2
        logger.debug("注入虚接错接: line_%d to_bus %d->%d", idx, old_to, new_to)

    def _inject_telemetry_mismatch(self, net, labels: np.ndarray, n_buses: int):
        """注入遥测矛盾: 在随机母线上制造量测偏差"""
        n_affected = max(1, int(n_buses * 0.1))
        affected = self.rng.choice(n_buses, size=n_affected, replace=False)
        for bus_idx in affected:
            if int(bus_idx) < len(labels):
                labels[int(bus_idx)] = 3
        if len(net.load) > 0:
            for ld_idx in net.load.index:
                if self.rng.random() < 0.3:
                    noise = 1.0 + self.rng.uniform(0.3, 0.5) * self.rng.choice([-1, 1])
                    net.load.at[ld_idx, "p_mw"] *= noise
        logger.debug("注入遥测矛盾: %d条母线受影响", n_affected)

    def _inject_model_mismatch(self, net, labels: np.ndarray):
        """注入图模不符: 删除设备(设为停运)"""
        if len(net.line) < 4:
            return
        candidates = net.line.index[2:]
        if len(candidates) == 0:
            return
        idx = int(self.rng.choice(candidates))
        net.line.at[idx, "in_service"] = False
        fb = int(net.line.at[idx, "from_bus"])
        tb = int(net.line.at[idx, "to_bus"])
        if fb < len(labels):
            labels[fb] = 4
        if tb < len(labels):
            labels[tb] = 4
        logger.debug("注入图模不符: line_%d停运", idx)

    def _inject_signal_mismatch(self, net, labels: np.ndarray, n_buses: int):
        """注入遥信矛盾: 随机改变开关状态"""
        if hasattr(net, "switch") and len(net.switch) > 0:
            for sw_idx in net.switch.index:
                if self.rng.random() < 0.5:
                    current = bool(net.switch.at[sw_idx, "closed"])
                    net.switch.at[sw_idx, "closed"] = not current
                    bus = int(net.switch.at[sw_idx, "bus"])
                    if bus < len(labels):
                        labels[bus] = 5
                    logger.debug("注入遥信矛盾: switch_%d %s->%s",
                                 sw_idx, current, not current)
        else:
            n_affected = max(1, int(n_buses * 0.05))
            affected = self.rng.choice(n_buses, size=n_affected, replace=False)
            for bus_idx in affected:
                if int(bus_idx) < len(labels):
                    labels[int(bus_idx)] = 5


    # === v8 anomaly injection methods (types 6-15) ===

    def _inject_measurement_outlier(self, net, labels, n_buses):
        """注入量测异常: 随机母线电压偏移"""
        n_aff = max(1, int(n_buses * 0.08))
        affected = self.rng.choice(n_buses, size=n_aff, replace=False)
        for b in affected:
            if int(b) < len(labels):
                labels[int(b)] = 6
        if len(net.load) > 0:
            for ld in net.load.index:
                if self.rng.random() < 0.25:
                    net.load.at[ld, "p_mw"] *= self.rng.uniform(1.4, 2.0)

    def _inject_stale_data(self, net, labels, n_buses):
        """注入陈旧数据: 部分量测冻结"""
        n_aff = max(1, int(n_buses * 0.1))
        affected = self.rng.choice(n_buses, size=n_aff, replace=False)
        for b in affected:
            if int(b) < len(labels):
                labels[int(b)] = 7
        if len(net.load) > 0:
            for ld in net.load.index:
                if self.rng.random() < 0.3:
                    net.load.at[ld, "p_mw"] = 0.0

    def _inject_parameter_error(self, net, labels, n_buses):
        """注入参数错误: 线路阻抗偏差"""
        if len(net.line) < 2:
            return
        n_aff = max(1, min(3, len(net.line)))
        affected = self.rng.choice(net.line.index, size=n_aff, replace=False)
        for idx in affected:
            fb = int(net.line.at[idx, "from_bus"])
            if fb < len(labels):
                labels[fb] = 8
            net.line.at[idx, "r_ohm_per_km"] *= self.rng.uniform(2.0, 5.0)
            net.line.at[idx, "x_ohm_per_km"] *= self.rng.uniform(0.3, 0.7)

    def _inject_load_shift(self, net, labels, n_buses):
        """注入负荷突变"""
        n_aff = max(1, int(n_buses * 0.15))
        affected = self.rng.choice(n_buses, size=n_aff, replace=False)
        for b in affected:
            if int(b) < len(labels):
                labels[int(b)] = 9
        if len(net.load) > 0:
            for ld in net.load.index:
                if self.rng.random() < 0.4:
                    net.load.at[ld, "p_mw"] *= self.rng.uniform(1.5, 3.0)

    def _inject_reverse_power_flow(self, net, labels, n_buses):
        """注入反向潮流: 负荷变发电"""
        if len(net.load) < 1:
            return
        ld = int(self.rng.choice(net.load.index))
        bus = int(net.load.at[ld, "bus"])
        if bus < len(labels):
            labels[bus] = 10
        net.load.at[ld, "p_mw"] = -abs(net.load.at[ld, "p_mw"]) * 1.5
        net.load.at[ld, "q_mvar"] = -abs(net.load.at[ld, "q_mvar"])

    def _inject_communication_loss(self, net, labels, n_buses):
        """注入通信中断: 部分量测丢失"""
        n_aff = max(1, int(n_buses * 0.12))
        affected = self.rng.choice(n_buses, size=n_aff, replace=False)
        for b in affected:
            if int(b) < len(labels):
                labels[int(b)] = 11

    def _inject_voltage_collapse(self, net, labels, n_buses):
        """注入电压崩溃: 电压严重偏低"""
        n_aff = max(1, int(n_buses * 0.15))
        affected = self.rng.choice(n_buses, size=n_aff, replace=False)
        for b in affected:
            if int(b) < len(labels):
                labels[int(b)] = 12
        if len(net.load) > 0:
            for ld in net.load.index:
                if self.rng.random() < 0.5:
                    net.load.at[ld, "p_mw"] *= self.rng.uniform(2.0, 4.0)

    def _inject_ghost_topology(self, net, labels, n_buses):
        """注入幽灵拓扑: 添加虚假线路"""
        all_buses = list(net.bus.index)
        if len(all_buses) < 3:
            return
        fb = int(self.rng.choice(all_buses))
        tb_candidates = [int(b) for b in all_buses if int(b) != fb]
        if not tb_candidates:
            return
        tb = int(self.rng.choice(tb_candidates))
        if fb < len(labels):
            labels[fb] = 13
        if tb < len(labels):
            labels[tb] = 13
        try:
            pp.create_line_from_parameters(
                net, from_bus=fb, to_bus=tb, length_km=0.1,
                r_ohm_per_km=0.1, x_ohm_per_km=0.1,
                c_nf_per_km=0, max_i_ka=0.1, name="ghost_line"
            )
        except Exception:
            pass

    def _inject_duplicate_measurement(self, net, labels, n_buses):
        """注入重复量测"""
        n_aff = max(1, int(n_buses * 0.1))
        affected = self.rng.choice(n_buses, size=n_aff, replace=False)
        for b in affected:
            if int(b) < len(labels):
                labels[int(b)] = 14
        if len(net.load) > 0:
            for ld in net.load.index:
                if self.rng.random() < 0.2:
                    net.load.at[ld, "p_mw"] *= self.rng.uniform(0.5, 0.8)

    def _inject_protection_misconfig(self, net, labels, n_buses):
        """注入保护误配"""
        if hasattr(net, "switch") and len(net.switch) > 0:
            for sw in net.switch.index:
                if self.rng.random() < 0.4:
                    net.switch.at[sw, "closed"] = not bool(net.switch.at[sw, "closed"])
                    bus = int(net.switch.at[sw, "bus"])
                    if bus < len(labels):
                        labels[bus] = 15
        else:
            n_aff = max(1, int(n_buses * 0.08))
            affected = self.rng.choice(n_buses, size=n_aff, replace=False)
            for b in affected:
                if int(b) < len(labels):
                    labels[int(b)] = 15

    # === v9 anomaly injection methods (types 16-25) ===

    def _inject_trafo_tap_fault(self, net, labels, n_buses):
        """注入变压器分接头故障"""
        if len(net.trafo) > 0:
            for t in net.trafo.index:
                if self.rng.random() < 0.5:
                    net.trafo.at[t, "tap_pos"] = int(self.rng.integers(-5, 6))
                    bus = int(net.trafo.at[t, "hv_bus"])
                    if bus < len(labels):
                        labels[bus] = 16
        else:
            n_aff = max(1, int(n_buses * 0.05))
            for b in self.rng.choice(n_buses, size=n_aff, replace=False):
                if int(b) < len(labels):
                    labels[int(b)] = 16

    def _inject_grounding_fault(self, net, labels, n_buses):
        """注入接地故障"""
        n_aff = max(1, int(n_buses * 0.08))
        affected = self.rng.choice(n_buses, size=n_aff, replace=False)
        for b in affected:
            if int(b) < len(labels):
                labels[int(b)] = 17
        if len(net.shunt) > 0:
            for s in net.shunt.index:
                if self.rng.random() < 0.4:
                    net.shunt.at[s, "q_mvar"] *= self.rng.uniform(3.0, 6.0)

    def _inject_clock_drift(self, net, labels, n_buses):
        """注入时钟漂移"""
        n_aff = max(1, int(n_buses * 0.1))
        affected = self.rng.choice(n_buses, size=n_aff, replace=False)
        for b in affected:
            if int(b) < len(labels):
                labels[int(b)] = 18
        if len(net.load) > 0:
            for ld in net.load.index:
                if self.rng.random() < 0.3:
                    net.load.at[ld, "p_mw"] *= self.rng.uniform(0.7, 1.3)

    def _inject_harmonic_pollution(self, net, labels, n_buses):
        """注入谐波污染"""
        n_aff = max(1, int(n_buses * 0.1))
        affected = self.rng.choice(n_buses, size=n_aff, replace=False)
        for b in affected:
            if int(b) < len(labels):
                labels[int(b)] = 19
        if len(net.load) > 0:
            for ld in net.load.index:
                if self.rng.random() < 0.3:
                    net.load.at[ld, "q_mvar"] *= self.rng.uniform(1.5, 3.0)

    def _inject_impedance_degradation(self, net, labels, n_buses):
        """注入阻抗退化"""
        if len(net.line) < 2:
            return
        n_aff = max(1, min(3, len(net.line)))
        affected = self.rng.choice(net.line.index, size=n_aff, replace=False)
        for idx in affected:
            fb = int(net.line.at[idx, "from_bus"])
            if fb < len(labels):
                labels[fb] = 20
            net.line.at[idx, "r_ohm_per_km"] *= self.rng.uniform(1.5, 3.0)

    def _inject_dg_intermittent(self, net, labels, n_buses):
        """注入分布式电源间歇性"""
        if len(net.sgen) > 0:
            for s in net.sgen.index:
                if self.rng.random() < 0.5:
                    net.sgen.at[s, "p_mw"] *= self.rng.uniform(0.1, 0.5)
                    bus = int(net.sgen.at[s, "bus"])
                    if bus < len(labels):
                        labels[bus] = 21
        elif len(net.gen) > 0:
            g = int(self.rng.choice(net.gen.index))
            net.gen.at[g, "p_mw"] *= self.rng.uniform(0.1, 0.5)
            bus = int(net.gen.at[g, "bus"])
            if bus < len(labels):
                labels[bus] = 21
        else:
            n_aff = max(1, int(n_buses * 0.05))
            for b in self.rng.choice(n_buses, size=n_aff, replace=False):
                if int(b) < len(labels):
                    labels[int(b)] = 21

    def _inject_measurement_bias(self, net, labels, n_buses):
        """注入量测偏差"""
        n_aff = max(1, int(n_buses * 0.1))
        affected = self.rng.choice(n_buses, size=n_aff, replace=False)
        for b in affected:
            if int(b) < len(labels):
                labels[int(b)] = 22
        if len(net.load) > 0:
            for ld in net.load.index:
                if self.rng.random() < 0.3:
                    bias = self.rng.uniform(0.15, 0.3) * self.rng.choice([-1, 1])
                    net.load.at[ld, "p_mw"] *= (1.0 + bias)

    def _inject_branch_contingency(self, net, labels, n_buses):
        """注入支路停运"""
        if len(net.line) < 3:
            return
        n_out = max(1, min(2, len(net.line) - 1))
        affected = self.rng.choice(net.line.index[1:], size=n_out, replace=False)
        for idx in affected:
            net.line.at[idx, "in_service"] = False
            fb = int(net.line.at[idx, "from_bus"])
            tb = int(net.line.at[idx, "to_bus"])
            if fb < len(labels):
                labels[fb] = 23
            if tb < len(labels):
                labels[tb] = 23

    def _inject_topo_obfuscation(self, net, labels, n_buses):
        """注入拓扑混淆: 修改开关+添加虚假连接"""
        if hasattr(net, "switch") and len(net.switch) > 0:
            for sw in net.switch.index:
                if self.rng.random() < 0.4:
                    net.switch.at[sw, "closed"] = not bool(net.switch.at[sw, "closed"])
                    bus = int(net.switch.at[sw, "bus"])
                    if bus < len(labels):
                        labels[bus] = 24
        n_aff = max(1, int(n_buses * 0.06))
        for b in self.rng.choice(n_buses, size=n_aff, replace=False):
            if int(b) < len(labels):
                labels[int(b)] = 24

    def _inject_voltage_regulation(self, net, labels, n_buses):
        """注入电压调节异常"""
        n_aff = max(1, int(n_buses * 0.1))
        affected = self.rng.choice(n_buses, size=n_aff, replace=False)
        for b in affected:
            if int(b) < len(labels):
                labels[int(b)] = 25
        if len(net.trafo) > 0:
            for t in net.trafo.index:
                if self.rng.random() < 0.3:
                    net.trafo.at[t, "tap_pos"] = int(self.rng.integers(-8, 8))
        if len(net.load) > 0:
            for ld in net.load.index:
                if self.rng.random() < 0.2:
                    net.load.at[ld, "p_mw"] *= self.rng.uniform(1.2, 2.0)


    def _inject_bus_section_mismatch(self, net, labels: np.ndarray, n_buses: int):
        """注入母线分段开关状态与拓扑不一致异常 (v16新增)
        
        模拟: 母线分段开关实际打开但拓扑显示闭合(或反之)
        """
        # Find buses with multiple connections (section buses)
        bus_degrees = {}
        for idx in net.line.index:
            fb = int(net.line.at[idx, "from_bus"])
            tb = int(net.line.at[idx, "to_bus"])
            bus_degrees[fb] = bus_degrees.get(fb, 0) + 1
            bus_degrees[tb] = bus_degrees.get(tb, 0) + 1
        
        # Pick a high-degree bus as section bus
        section_buses = [b for b, d in bus_degrees.items() if d >= 3]
        if not section_buses:
            section_buses = list(bus_degrees.keys())[:3]
        
        if section_buses:
            target_bus = int(self.rng.choice(section_buses))
            # Flip switch state for lines connected to this bus
            for idx in net.line.index:
                fb = int(net.line.at[idx, "from_bus"])
                tb = int(net.line.at[idx, "to_bus"])
                if fb == target_bus or tb == target_bus:
                    net.line.at[idx, "in_service"] = not bool(net.line.at[idx, "in_service"])
                    break
            labels[target_bus] = 26

    def _inject_bypass_operation(self, net, labels: np.ndarray, n_buses: int):
        """注入旁路代路操作后拓扑未更新异常 (v16新增)
        
        模拟: 某线路通过旁路供电，但拓扑模型仍显示原线路运行
        """
        in_service_lines = [i for i in net.line.index if bool(net.line.at[i, "in_service"])]
        if not in_service_lines:
            return
        
        line_idx = int(self.rng.choice(in_service_lines))
        # Simulate bypass: reduce impedance to 5% (bypass is very low impedance)
        if "r_ohm_per_km" in net.line.columns:
            net.line.at[line_idx, "r_ohm_per_km"] *= 0.05
        if "x_ohm_per_km" in net.line.columns:
            net.line.at[line_idx, "x_ohm_per_km"] *= 0.05
        
        # Mark affected buses
        fb = int(net.line.at[line_idx, "from_bus"])
        tb = int(net.line.at[line_idx, "to_bus"])
        labels[fb] = 27
        labels[tb] = 27

    def _inject_load_transfer_residual(self, net, labels: np.ndarray, n_buses: int):
        """注入负荷转供后拓扑残留异常 (v16新增)
        
        模拟: 负荷已从一条馈线转供到另一条，但拓扑模型中仍保留原馈线的连接关系
        """
        lines = list(net.line.index)
        if len(lines) < 4:
            return
        
        line_idx = int(self.rng.choice(lines))
        # Mark line as out of service but keep measurements (residual)
        net.line.at[line_idx, "in_service"] = False
        
        # Mark affected buses
        fb = int(net.line.at[line_idx, "from_bus"])
        tb = int(net.line.at[line_idx, "to_bus"])
        labels[fb] = 28
        labels[tb] = 28


    def _extract_features(self, net, n_buses: int) -> np.ndarray:
        """从PandaPower网络提取8维节点特征"""
        features = np.zeros((n_buses, NODE_FEATURE_DIM), dtype=np.float32)

        # 尝试运行潮流计算
        pf_success = False
        try:
            import pandapower as pp
            pp.runpp(net, numba=False)
            pf_success = True
        except Exception as e:
            logger.debug("潮流计算失败(使用静态特征): %s", e)

        # 松弛节点(电源)
        slack_buses = set()
        if hasattr(net, "ext_grid"):
            for eg_idx in net.ext_grid.index:
                slack_buses.add(int(net.ext_grid.at[eg_idx, "bus"]))

        # 发电机节点
        gen_buses = set()
        if hasattr(net, "gen"):
            for g_idx in net.gen.index:
                gen_buses.add(int(net.gen.at[g_idx, "bus"]))

        # 度数
        degree = np.zeros(n_buses, dtype=np.float32)
        for line_idx in net.line.index:
            fb = int(net.line.at[line_idx, "from_bus"])
            tb = int(net.line.at[line_idx, "to_bus"])
            if bool(net.line.at[line_idx, "in_service"]):
                if fb < n_buses:
                    degree[fb] += 1
                if tb < n_buses:
                    degree[tb] += 1
        for trafo_idx in net.trafo.index:
            hv = int(net.trafo.at[trafo_idx, "hv_bus"])
            lv = int(net.trafo.at[trafo_idx, "lv_bus"])
            if bool(net.trafo.at[trafo_idx, "in_service"]):
                if hv < n_buses:
                    degree[hv] += 1
                if lv < n_buses:
                    degree[lv] += 1

        # 有功/无功注入
        p_inject = np.zeros(n_buses, dtype=np.float32)
        q_inject = np.zeros(n_buses, dtype=np.float32)
        if hasattr(net, "load"):
            for ld_idx in net.load.index:
                bus = int(net.load.at[ld_idx, "bus"])
                if bus < n_buses and bool(net.load.at[ld_idx, "in_service"]):
                    p_inject[bus] -= float(net.load.at[ld_idx, "p_mw"])
                    try:
                        q_inject[bus] -= float(net.load.at[ld_idx, "q_mvar"])
                    except (KeyError, ValueError):
                        pass
        if hasattr(net, "gen"):
            for g_idx in net.gen.index:
                bus = int(net.gen.at[g_idx, "bus"])
                if bus < n_buses and bool(net.gen.at[g_idx, "in_service"]):
                    p_inject[bus] += float(net.gen.at[g_idx, "p_mw"])
                    try:
                        q_inject[bus] += float(net.gen.at[g_idx, "q_mvar"])
                    except (KeyError, ValueError):
                        pass

        # 连通分量
        component_id = self._get_component_ids(net, n_buses)

        # 电压(潮流不收敛时res_bus可能含NaN，需过滤)
        vm_pu = np.ones(n_buses, dtype=np.float32)
        if pf_success and hasattr(net, "res_bus") and len(net.res_bus) > 0:
            for bus_idx in net.res_bus.index:
                bi = int(bus_idx)
                if bi < n_buses:
                    val = float(net.res_bus.at[bus_idx, "vm_pu"])
                    if np.isfinite(val) and 0.5 < val < 1.5:
                        vm_pu[bi] = val

        # 归一化
        max_p = max(np.max(np.abs(p_inject)), 1.0)
        max_q = max(np.max(np.abs(q_inject)), 1.0)

        for i in range(n_buses):
            features[i, 0] = degree[i] / 10.0                  # 归一化度数
            features[i, 1] = vm_pu[i]                           # 电压幅值
            features[i, 2] = 1.0 if (i in slack_buses or i in gen_buses) else 0.0
            features[i, 3] = p_inject[i] / max_p                # 有功注入
            features[i, 4] = q_inject[i] / max_q                # 无功注入
            features[i, 5] = vm_pu[i] - 1.0                     # 电压偏差
            if i in slack_buses:
                features[i, 6] = 0.25                           # slack
            elif i in gen_buses:
                features[i, 6] = 0.5                            # gen
            elif p_inject[i] < -0.01:
                features[i, 6] = 0.75                           # load
            else:
                features[i, 6] = 1.0                            # bus
            features[i, 7] = component_id[i] / max(max(component_id), 1.0)

        return features

    def _get_component_ids(self, net, n_buses: int) -> np.ndarray:
        """并查集计算连通分量ID"""
        parent = list(range(n_buses))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for line_idx in net.line.index:
            if bool(net.line.at[line_idx, "in_service"]):
                fb = int(net.line.at[line_idx, "from_bus"])
                tb = int(net.line.at[line_idx, "to_bus"])
                if fb < n_buses and tb < n_buses:
                    union(fb, tb)
        for trafo_idx in net.trafo.index:
            if bool(net.trafo.at[trafo_idx, "in_service"]):
                hv = int(net.trafo.at[trafo_idx, "hv_bus"])
                lv = int(net.trafo.at[trafo_idx, "lv_bus"])
                if hv < n_buses and lv < n_buses:
                    union(hv, lv)

        return np.array([find(i) for i in range(n_buses)], dtype=np.float32)

    def _build_edge_index(self, net) -> np.ndarray:
        """构建边索引 [2, E]，双向边"""
        n_buses = len(net.bus)
        edges = []
        for line_idx in net.line.index:
            if bool(net.line.at[line_idx, "in_service"]):
                fb = int(net.line.at[line_idx, "from_bus"])
                tb = int(net.line.at[line_idx, "to_bus"])
                if fb < n_buses and tb < n_buses:
                    edges.append([fb, tb])
                    edges.append([tb, fb])
        for trafo_idx in net.trafo.index:
            if bool(net.trafo.at[trafo_idx, "in_service"]):
                hv = int(net.trafo.at[trafo_idx, "hv_bus"])
                lv = int(net.trafo.at[trafo_idx, "lv_bus"])
                if hv < n_buses and lv < n_buses:
                    edges.append([hv, lv])
                    edges.append([lv, hv])
        if hasattr(net, "switch") and len(net.switch) > 0:
            for sw_idx in net.switch.index:
                et = str(net.switch.at[sw_idx, "et"])
                if et == "b" and bool(net.switch.at[sw_idx, "closed"]):
                    bus = int(net.switch.at[sw_idx, "bus"])
                    element = int(net.switch.at[sw_idx, "element"])
                    if bus < n_buses and element < n_buses:
                        edges.append([bus, element])
                        edges.append([element, bus])
        if not edges:
            edges = [[0, 0]]
        return np.array(edges, dtype=np.int64).T


# ============================================================
# 4. GNNTrainer - 训练器
# ============================================================
class GNNTrainer:
    """
    GNN训练器

    功能:
      - 训练/验证损失跟踪
      - 学习率调度
      - 早停机制
      - 保存最佳模型
    """

    def __init__(self, model: nn.Module = None, device: str = None,
                 output_dir: Path = OUTPUT_DIR):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if model is None:
            model = GraphSAGENet()
        self.model = model.to(self.device)

        logger.info("GNNTrainer初始化: device=%s, 参数量=%d",
                     self.device, sum(p.numel() for p in self.model.parameters()))

    def train(self, train_data: List[Dict], val_data: List[Dict],
              epochs: int = 50, lr: float = 0.001,
              patience: int = 10, save_best: bool = True, **kwargs) -> Dict:
        """
        训练循环

        Args:
            train_data: 训练样本列表
            val_data: 验证样本列表
            epochs: 训练轮数
            lr: 学习率
            patience: 早停耐心值
            save_best: 是否保存最佳模型

        Returns:
            训练历史字典
        """
        optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                      patience=5, min_lr=1e-6)

        # 类别权重(逆频率)
        class_weights = self._compute_class_weights(train_data)
        use_physics = kwargs.get('use_physics', True)
        if use_physics:
            criterion = PhysicsConstrainedLoss(class_weight=class_weights.to(self.device))
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))

        history = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "best_val_loss": float("inf"),
            "best_epoch": -1,
        }
        no_improve = 0

        logger.info("开始训练: epochs=%d, lr=%f, train=%d, val=%d",
                     epochs, lr, len(train_data), len(val_data))

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # --- 训练阶段 ---
            self.model.train()
            train_loss = 0.0
            n_train_nodes = 0

            for sample in train_data:
                x = torch.tensor(sample["node_features"],
                                 dtype=torch.float).to(self.device)
                edge_index = torch.tensor(sample["edge_index"],
                                          dtype=torch.long).to(self.device)
                y = torch.tensor(sample["labels"],
                                 dtype=torch.long).to(self.device)

                optimizer.zero_grad()
                out = self.model(x, edge_index)
                if hasattr(criterion, 'lambda_kcl') and edge_index is not None:
                    loss = criterion(out, y, edge_index, x)
                else:
                    loss = criterion(out, y)
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning("Loss NaN/Inf at batch, skipping")
                    optimizer.zero_grad()
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item() * x.size(0)
                n_train_nodes += x.size(0)

            train_loss /= max(n_train_nodes, 1)

            # --- 验证阶段 ---
            val_loss, val_acc = self._validate(val_data, criterion)

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if val_loss < history["best_val_loss"]:
                history["best_val_loss"] = val_loss
                history["best_epoch"] = epoch
                no_improve = 0
                if save_best:
                    self._save_model("gnn_model_best.pt")
            else:
                no_improve += 1

            elapsed = time.time() - t0
            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | "
                    "val_acc=%.4f | lr=%.2e | %.1fs",
                    epoch, epochs, train_loss, val_loss, val_acc,
                    current_lr, elapsed)

            if no_improve >= patience:
                logger.info("早停触发: %d轮无改善", patience)
                break

        self._save_model("gnn_model_final.pt")
        logger.info("训练完成: 最佳轮次=%d, 最佳val_loss=%.4f",
                     history["best_epoch"], history["best_val_loss"])
        return history

    def _validate(self, val_data: List[Dict],
                  criterion: nn.Module) -> Tuple[float, float]:
        """验证阶段"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for sample in val_data:
                x = torch.tensor(sample["node_features"],
                                 dtype=torch.float).to(self.device)
                edge_index = torch.tensor(sample["edge_index"],
                                          dtype=torch.long).to(self.device)
                y = torch.tensor(sample["labels"],
                                 dtype=torch.long).to(self.device)

                out = self.model(x, edge_index)
                loss = criterion(out, y)

                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss += loss.item() * x.size(0)
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += x.size(0)

        val_loss /= max(total, 1)
        val_acc = correct / max(total, 1)
        return val_loss, val_acc

    def _compute_class_weights(self, train_data: List[Dict]) -> torch.Tensor:
        """计算类别权重(sqrt逆频率, 防止极端权重导致NaN)"""
        counts = np.zeros(NUM_ANOMALY_CLASSES, dtype=np.float64)
        for sample in train_data:
            for label in sample["labels"]:
                counts[int(label)] += 1
        counts = np.maximum(counts, 1.0)
        # 使用sqrt逆频率避免极端权重
        weights = 1.0 / np.sqrt(counts)
        weights = weights / weights.sum() * NUM_ANOMALY_CLASSES
        weights = np.clip(weights, 0.1, 10.0)  # 限制权重范围
        return torch.tensor(weights, dtype=torch.float)

    def _save_model(self, filename: str):
        """保存模型权重"""
        path = self.output_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "in_channels": NODE_FEATURE_DIM,
            "hidden_channels": DEFAULT_HIDDEN_DIM,
            "out_channels": NUM_ANOMALY_CLASSES,
        }, path)
        logger.info("模型已保存: %s", path)

    def load_model(self, filename: str = "gnn_model_best.pt") -> bool:
        """加载模型权重"""
        path = self.output_dir / filename
        if not path.exists():
            logger.warning("模型文件不存在: %s", path)
            return False
        checkpoint = torch.load(path, map_location=self.device,
                                weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        logger.info("模型已加载: %s", path)
        return True


# ============================================================
# 5. 辅助函数
# ============================================================
def load_pandapower_network(network_name: str = "case33bw"):
    """加载PandaPower内置网络"""
    import pandapower.networks as pn
    network_map = {
        "case33bw": pn.case33bw,
        "case34bw": pn.case34bw if hasattr(pn, "case34bw") else None,
        "case118": pn.case118,
        "case_ieee30": pn.case_ieee30,
        "example_simple": pn.example_simple,
        "example_multivoltage": pn.example_multivoltage,
        "cigre_mv": pn.create_cigre_network_mv,
        "cigre_lv": pn.create_cigre_network_lv,
        "kerber_dorfnetz": pn.create_kerber_dorfnetz,
        "kerber_landnetz_freileitung": pn.create_kerber_landnetz_freileitung_1,
        "kerber_landnetz_kabel": pn.create_kerber_landnetz_kabel_1,
        "kerber_vorstadtnetz_kabel": pn.create_kerber_vorstadtnetz_kabel_1,
    }
    if network_name not in network_map:
        raise ValueError("不支持的网络: {}, 可选: {}".format(
            network_name, list(network_map.keys())))
    factory = network_map[network_name]
    if factory is None:
        raise ValueError("网络 {} 在当前PandaPower版本中不可用".format(network_name))
    net = factory()
    logger.info("加载网络 %s: %d条母线, %d条线路, %d个负载",
                 network_name, len(net.bus), len(net.line), len(net.load))
    return net


def split_dataset(samples: List[Dict], val_ratio: float = 0.2,
                  seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """划分训练集和验证集"""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(samples))
    n_val = max(1, int(len(samples) * val_ratio))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    train_data = [samples[i] for i in train_indices]
    val_data = [samples[i] for i in val_indices]
    logger.info("数据划分: 训练=%d, 验证=%d", len(train_data), len(val_data))
    return train_data, val_data


# ============================================================
# 6. main() - 入口函数
# ============================================================
def main():
    """
    主入口: 使用case33bw网络训练GNN模型

    流程:
      1. 加载PandaPower网络(case33bw)
      2. 生成训练数据(500样本)
      3. 划分训练/验证集
      4. 创建模型并训练
      5. 保存最佳模型到output/gnn_model_best.pt
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("GNN拓扑异常检测训练 - 开始")
    logger.info("=" * 60)

    # 1. 加载网络
    logger.info("[1/4] 加载PandaPower网络...")
    net = load_pandapower_network("case33bw")

    # 2. 生成训练数据
    logger.info("[2/4] 生成训练数据(500样本)...")
    generator = TrainingDataGenerator(seed=42)
    samples = generator.generate_from_pandapower(net, n_samples=500)

    # 统计标签分布
    all_labels = np.concatenate([s["labels"] for s in samples])
    unique, counts = np.unique(all_labels, return_counts=True)
    logger.info("标签分布:")
    for label, count in zip(unique, counts):
        logger.info("  %s: %d (%.1f%%)",
                     ANOMALY_LABELS.get(int(label), "未知({})".format(label)),
                     count, count / len(all_labels) * 100)

    # 3. 划分数据集
    logger.info("[3/4] 划分训练/验证集...")
    train_data, val_data = split_dataset(samples, val_ratio=0.2)

    # 4. 训练
    logger.info("[4/4] 开始训练...")
    model = GraphSAGENet(
        in_channels=NODE_FEATURE_DIM,
        hidden_channels=DEFAULT_HIDDEN_DIM,
        out_channels=NUM_ANOMALY_CLASSES,
        dropout=DEFAULT_DROPOUT,
    )

    trainer = GNNTrainer(model=model)
    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=50,
        lr=0.001,
        patience=10,
    )

    # 输出结果摘要
    logger.info("=" * 60)
    logger.info("训练完成!")
    logger.info("  最佳轮次: %d", history["best_epoch"])
    logger.info("  最佳验证损失: %.4f", history["best_val_loss"])
    best_idx = history["best_epoch"] - 1
    if 0 <= best_idx < len(history["val_acc"]):
        logger.info("  最佳验证准确率: %.4f", history["val_acc"][best_idx])
    logger.info("  模型已保存: %s", OUTPUT_DIR / "gnn_model_best.pt")
    logger.info("=" * 60)

    return history


if __name__ == "__main__":
    main()

# ===== P2-4: Contrastive Learning Pre-training =====

class ContrastivePretrainer:
    """Contrastive learning for graph-level representation pre-training.
    
    Uses SimCLR-style contrastive loss on graph augmentations:
    1. Node feature masking (random dropout of features)
    2. Edge dropping (random removal of edges)
    3. Subgraph sampling
    
    Pre-trained representations improve downstream anomaly detection.
    """
    
    def __init__(self, model, projector_dim=64, temperature=0.1):
        self.model = model
        self.temperature = temperature
        # Projection head for contrastive loss
        hidden_dim = 64  # default
        for name, param in model.named_parameters():
            if "node_classifier" in name and "weight" in name:
                hidden_dim = param.shape[1]
                break
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, projector_dim),
            nn.ReLU(),
            nn.Linear(projector_dim, projector_dim),
        )
    
    def augment(self, x, edge_index, mask_ratio=0.1, drop_ratio=0.1):
        """Augment graph: feature masking + edge dropping."""
        x_aug = x.clone()
        # Feature masking
        mask = torch.rand(x.size(1)) > mask_ratio
        x_aug = x_aug * mask.unsqueeze(0)
        
        # Edge dropping
        n_edges = edge_index.size(1)
        keep = torch.rand(n_edges) > drop_ratio
        edge_aug = edge_index[:, keep]
        
        return x_aug, edge_aug
    
    def contrastive_loss(self, z1, z2):
        """NT-Xent contrastive loss."""
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        N = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # [2N, D]
        sim = torch.mm(z, z.t()) / self.temperature  # [2N, 2N]
        
        # Mask: positive pairs are (i, i+N) and (i+N, i)
        labels = torch.arange(N, device=z.device)
        labels = torch.cat([labels + N, labels])  # [2N]
        
        # Remove self-similarity
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, -1e9)
        
        loss = F.cross_entropy(sim, labels)
        return loss
    
    def pretrain(self, train_data, epochs=20, lr=0.001):
        """Pre-train with contrastive learning."""
        optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.projector.parameters()),
            lr=lr
        )
        
        history = {"loss": []}
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            n_batches = 0
            
            for sample in train_data:
                x = torch.tensor(sample["node_features"], dtype=torch.float)
                edge_index = torch.tensor(sample["edge_index"], dtype=torch.long)
                
                # Two augmented views
                x1, ei1 = self.augment(x, edge_index)
                x2, ei2 = self.augment(x, edge_index)
                
                # Forward (get embeddings before classifier)
                h1 = self._get_embeddings(x1, ei1)
                h2 = self._get_embeddings(x2, ei2)
                
                # Pool to graph-level
                z1 = self.projector(h1.mean(dim=0, keepdim=True))
                z2 = self.projector(h2.mean(dim=0, keepdim=True))
                
                loss = self.contrastive_loss(z1, z2)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / max(n_batches, 1)
            history["loss"].append(avg_loss)
            if epoch % 5 == 0:
                logger.info("Contrastive pretrain epoch %d: loss=%.4f", epoch, avg_loss)
        
        return history
    
    def _get_embeddings(self, x, edge_index):
        """Get hidden embeddings (before classifier)."""
        # Forward through conv layers only
        h = F.relu(self.model.conv1(x, edge_index))
        h = self.model.dropout(h) if hasattr(self.model, "dropout") else h
        h = F.relu(self.model.conv2(h, edge_index))
        return h
