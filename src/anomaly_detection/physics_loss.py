# -*- coding: utf-8 -*-
"""
v16: 物理约束损失函数 — KCL/KVL嵌入GNN训练
来源: PhysicsInformed-OPF/physical_loss.py + EmergentGNN GAT+PI
"""
import torch
import torch.nn as nn


def incidence_matrix(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """构建节点-支路关联矩阵 (sparse)"""
    device = edge_index.device
    num_edges = edge_index.size(1)
    row = torch.cat([edge_index[0], edge_index[1]], dim=0)
    col = torch.cat([torch.arange(num_edges, device=device)] * 2, dim=0)
    values = torch.cat([
        torch.ones(num_edges, device=device),
        -torch.ones(num_edges, device=device),
    ], dim=0)
    indices = torch.stack([row, col])
    return torch.sparse_coo_tensor(indices, values, (num_nodes, num_edges), device=device)


def kcl_residual(net_injection: torch.Tensor, edge_index: torch.Tensor,
                 edge_flows: torch.Tensor) -> torch.Tensor:
    """计算KCL残差: B @ flow - net_injection"""
    if edge_flows.dim() == 1:
        edge_flows = edge_flows.unsqueeze(-1)
    B = incidence_matrix(edge_index, net_injection.size(0))
    aggregated = torch.sparse.mm(B, edge_flows)
    return aggregated - net_injection


def physics_informed_loss(pred: torch.Tensor, edge_index: torch.Tensor,
                          edge_attr: torch.Tensor, lambda_kcl: float = 0.05) -> torch.Tensor:
    """
    物理约束损失: L = L_task + lambda * L_KCL
    
    Args:
        pred: GNN预测 [N, feat_dim] (含P_inj, Q_inj, V, theta)
        edge_index: 图拓扑 [2, E]
        edge_attr: 边特征 [E, edge_dim] (含R, X, P_flow, Q_flow)
        lambda_kcl: KCL惩罚权重
    """
    # 从预测中提取功率注入 (假设前2维是P_inj, Q_inj)
    p_inj = pred[:, 0]
    q_inj = pred[:, 1]
    
    # 从边特征中提取支路潮流 (假设第3,4维是P_flow, Q_flow)
    if edge_attr.size(1) >= 4:
        p_flow = edge_attr[:, 2]
        q_flow = edge_attr[:, 3]
    else:
        # 如果边特征不够，用简化估计
        src, dst = edge_index[0], edge_index[1]
        v_src = pred[src, 2] if pred.size(1) > 2 else torch.ones(src.size(0), device=pred.device)
        v_dst = pred[dst, 2] if pred.size(1) > 2 else torch.ones(dst.size(0), device=pred.device)
        r = edge_attr[:, 0].clamp(min=1e-6)
        p_flow = (v_src - v_dst) / r
    
    # KCL残差
    kcl_p = kcl_residual(p_inj, edge_index, p_flow)
    kcl_q = kcl_residual(q_inj, edge_index, q_flow)
    
    # 惩罚项
    loss_kcl = (kcl_p ** 2).mean() + (kcl_q ** 2).mean()
    
    return lambda_kcl * loss_kcl
