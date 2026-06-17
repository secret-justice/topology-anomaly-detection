# -*- coding: utf-8 -*-
"""
GNN Graph-Level Anomaly Detector v3.0
Graph Contrastive Learning (GraphCL) approach.

Architecture:
1. GraphSAGE encoder -> graph-level readout (mean + max pool)
2. Projection head for contrastive learning (InfoNCE)
3. Reconstruction decoder (auxiliary objective)
4. Mahalanobis distance from normal distribution = graph anomaly score
5. Node-level localization via embedding deviation

Key improvements over v2 (GAE):
- Graph-level detection instead of per-node (fixes fundamental mismatch)
- Contrastive learning handles class imbalance (95.8%% normal)
- Anomaly injection as hard negatives for better decision boundary
- Mahalanobis distance calibrated on normal graph distribution
"""
import numpy as np
try:
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging
import os
import networkx as nx
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================
# Graph Convolution Layer
# ============================================================

class SimpleSAGEConv(nn.Module):
    """Pure PyTorch GraphSAGE convolution."""
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


# ============================================================
# GraphCL Encoder with Graph-Level Readout
# ============================================================

class GraphCLEncoder(nn.Module):
    """GraphSAGE encoder with graph-level readout (mean + max pooling).
    Produces a fixed-size graph embedding regardless of graph size.
    """
    def __init__(self, in_dim: int, hidden_dim: int = 64, latent_dim: int = 32):
        super().__init__()
        self.conv1 = SimpleSAGEConv(in_dim, hidden_dim)
        self.conv2 = SimpleSAGEConv(hidden_dim, latent_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(latent_dim)
        self.dropout = nn.Dropout(0.2)
        self.latent_dim = latent_dim
        self.readout_dim = latent_dim * 2

    def forward(self, x, edge_index, batch=None):
        h = F.relu(self.bn1(self.conv1(x, edge_index)))
        h = self.dropout(h)
        node_emb = self.bn2(self.conv2(h, edge_index))
        if batch is None:
            g_mean = node_emb.mean(dim=0, keepdim=True)
            g_max = node_emb.max(dim=0, keepdim=True)[0]
        else:
            from torch_geometric.nn import global_mean_pool, global_max_pool
            g_mean = global_mean_pool(node_emb, batch)
            g_max = global_max_pool(node_emb, batch)
        graph_emb = torch.cat([g_mean, g_max], dim=1)
        return node_emb, graph_emb


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""
    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GraphDecoder(nn.Module):
    """Auxiliary decoder for reconstruction loss."""
    def __init__(self, graph_dim: int, hidden_dim: int = 64, out_dim: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(graph_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward_node_recon(self, graph_emb, num_nodes):
        h = F.relu(self.fc1(graph_emb))
        x_recon = self.fc2(h)
        if num_nodes is not None and x_recon.size(0) == 1:
            x_recon = x_recon.expand(num_nodes, -1)
        return x_recon


# ============================================================
# Graph Contrastive Model
# ============================================================

class GraphCLModel(nn.Module):
    """GraphCL-style model for graph-level anomaly detection."""
    def __init__(self, in_dim=8, hidden_dim=64, latent_dim=32):
        super().__init__()
        self.encoder = GraphCLEncoder(in_dim, hidden_dim, latent_dim)
        self.projector = ProjectionHead(self.encoder.readout_dim, hidden_dim, latent_dim)
        self.decoder = GraphDecoder(self.encoder.readout_dim, hidden_dim, in_dim)

    def forward(self, x, edge_index, batch=None):
        node_emb, graph_emb = self.encoder(x, edge_index, batch)
        proj = self.projector(graph_emb)
        return node_emb, graph_emb, proj

    def get_graph_embedding(self, x, edge_index):
        with torch.no_grad():
            _, graph_emb = self.encoder(x, edge_index)
        return graph_emb


# ============================================================
# Graph Augmentation for Contrastive Learning
# ============================================================

class GraphAugmentor:
    """GraphCL augmentation: feature masking + edge dropping + noise."""
    def __init__(self, mask_ratio=0.15, edge_drop_ratio=0.1, noise_std=0.05, seed=42):
        self.mask_ratio = mask_ratio
        self.edge_drop_ratio = edge_drop_ratio
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

    def augment(self, x, edge_index):
        x_aug = x.clone()
        edge_aug = edge_index.clone()
        if self.rng.random() < 0.7:
            mask = torch.rand(x.size(0), x.size(1)) > self.mask_ratio
            x_aug = x_aug * mask.float()
        if self.rng.random() < 0.7 and edge_aug.size(1) > 4:
            n_edges = edge_aug.size(1)
            keep = torch.rand(n_edges) > self.edge_drop_ratio
            if keep.sum() >= 2:
                edge_aug = edge_aug[:, keep]
        if self.rng.random() < 0.5:
            noise = torch.randn_like(x_aug) * self.noise_std
            x_aug = x_aug + noise
        return x_aug, edge_aug


# ============================================================
# Anomaly Injection for Hard Negative Mining
# ============================================================

class AnomalyInjector:
    """Inject realistic anomalies into normal graph data.
    Types: feature perturbation, edge removal, feature swap,
    voltage deviation, edge addition.
    """
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def inject_anomalies(self, x, edge_index, anomaly_ratio=0.15):
        N = x.size(0)
        n_anomalous = max(1, int(N * anomaly_ratio))
        labels = torch.zeros(N, dtype=torch.long)
        anom_nodes = torch.randperm(N)[:n_anomalous]
        labels[anom_nodes] = 1
        x_aug = x.clone()
        edge_aug = edge_index.clone()
        for node_idx in anom_nodes:
            anomaly_type = self.rng.integers(0, 5)
            if anomaly_type == 0:
                noise = torch.randn(x.size(1)) * 0.2
                x_aug[node_idx] = x[node_idx] + noise
            elif anomaly_type == 1:
                mask = (edge_aug[0] == node_idx) | (edge_aug[1] == node_idx)
                edge_indices = torch.where(mask)[0]
                if len(edge_indices) > 0:
                    n_remove = max(1, len(edge_indices) // 3)
                    remove_idx = edge_indices[torch.randperm(len(edge_indices))[:n_remove]]
                    keep_mask = torch.ones(edge_aug.size(1), dtype=torch.bool)
                    keep_mask[remove_idx] = False
                    edge_aug = edge_aug[:, keep_mask]
            elif anomaly_type == 2:
                swap_target = self.rng.integers(0, N)
                x_aug[node_idx], x_aug[swap_target] = x[swap_target].clone(), x[node_idx].clone()
            elif anomaly_type == 3:
                direction = self.rng.choice([-1, 1])
                x_aug[node_idx, 1] += direction * self.rng.uniform(0.1, 0.3)
                x_aug[node_idx, 5] = x_aug[node_idx, 1] - 1.0
            elif anomaly_type == 4:
                random_target = self.rng.integers(0, N)
                if random_target != node_idx:
                    new_edge = torch.tensor([[node_idx, random_target],
                                            [random_target, node_idx]], dtype=torch.long)
                    edge_aug = torch.cat([edge_aug, new_edge], dim=1)
        return x_aug, edge_aug, labels

    def inject_graph_level_anomaly(self, x, edge_index):
        x_aug = x.clone()
        edge_aug = edge_index.clone()
        N = x.size(0)
        anom_type = self.rng.integers(0, 3)
        if anom_type == 0:
            n_affected = max(2, N // 3)
            affected = torch.randperm(N)[:n_affected]
            for node_idx in affected:
                x_aug[node_idx, 1] += self.rng.choice([-1, 1]) * self.rng.uniform(0.05, 0.2)
                x_aug[node_idx, 5] = x_aug[node_idx, 1] - 1.0
        elif anom_type == 1:
            n_edges = edge_aug.size(1)
            n_remove = max(2, n_edges // 4)
            remove_idx = torch.randperm(n_edges)[:n_remove]
            keep_mask = torch.ones(n_edges, dtype=torch.bool)
            keep_mask[remove_idx] = False
            edge_aug = edge_aug[:, keep_mask]
        elif anom_type == 2:
            shift = torch.randn(x.size(1)) * 0.3
            x_aug = x_aug + shift.unsqueeze(0).expand_as(x_aug)
        return x_aug, edge_aug


# ============================================================
# GNN Anomaly Detector v3 (GraphCL)
# ============================================================

class GNNAnomalyDetectorV2:
    """Graph-level anomaly detection using contrastive learning.

    Compatible with benchmark_v7_breakthrough.py interface.
    Uses Mahalanobis distance from learned normal distribution.
    """
    def __init__(self, model_path=None, device="cpu",
                 in_dim=8, hidden_dim=64, latent_dim=32):
        self.device = device
        self.model = GraphCLModel(in_dim, hidden_dim, latent_dim)
        self.threshold = None
        self.is_ready = False
        self.in_dim = in_dim
        self._normal_embeddings = None
        self.normal_mean = None
        self.normal_cov_inv = None
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)

    def _load_model(self, path):
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.threshold = ckpt.get("threshold", 0.5)
            self.normal_mean = ckpt.get("normal_mean", None)
            self.normal_cov_inv = ckpt.get("normal_cov_inv", None)
            self.model.eval()
            self.is_ready = True
            logger.info("GNN v3 model loaded: %s (threshold=%.4f)", path, self.threshold)
        except Exception as e:
            logger.error("GNN v3 load failed: %s", e)

    def save_model(self, path):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "threshold": self.threshold,
            "in_dim": self.in_dim,
            "normal_mean": self.normal_mean,
            "normal_cov_inv": self.normal_cov_inv,
            "version": "v3_graphcl",
        }, path)
        logger.info("GNN v3 model saved: %s", path)

    # --- Contrastive Loss ---

    def _simple_infonce(self, z1, z2, temperature=0.5):
        B = z1.size(0)
        if B < 2:
            return torch.tensor(0.0, device=z1.device, requires_grad=True)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        pos_sim = (z1 * z2).sum(dim=1) / temperature
        neg_sim = torch.mm(z1, z2.t()) / temperature
        mask = torch.eye(B, dtype=torch.bool, device=z1.device)
        neg_sim.masked_fill_(mask, -1e9)
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(B, dtype=torch.long, device=z1.device)
        return F.cross_entropy(logits, labels)

    # --- Training ---

    def train(self, normal_graphs, n_epochs=150, lr=5e-4, val_split=0.2,
              anomaly_aug_ratio=0.4, patience=20, temperature=0.5,
              lambda_recon=0.1, lambda_anom=0.3):
        """Train with graph contrastive learning."""
        n_val = max(1, int(len(normal_graphs) * val_split))
        train_data = normal_graphs[:-n_val]
        val_data = normal_graphs[-n_val:]
        if len(train_data) < 2:
            train_data = normal_graphs
            val_data = normal_graphs[:max(1, len(normal_graphs) // 5)]

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=lr * 0.01)
        augmentor = GraphAugmentor()
        injector = AnomalyInjector()
        best_val_score = float("inf")
        best_state = None
        no_improve = 0
        logger.info("Training GraphCL: %d train / %d val", len(train_data), len(val_data))

        for epoch in range(n_epochs):
            self.model.train()
            total_loss = 0.0
            n_batches = 0
            perm = np.random.permutation(len(train_data))
            for idx in perm:
                sample = train_data[idx]
                x, edge_index = sample["x"], sample["edge_index"]
                if edge_index.size(1) == 0:
                    continue
                optimizer.zero_grad()
                x1, ei1 = augmentor.augment(x, edge_index)
                x2, ei2 = augmentor.augment(x, edge_index)
                _, _, proj1 = self.model(x1, ei1)
                _, _, proj2 = self.model(x2, ei2)
                loss_cl = self._simple_infonce(proj1, proj2, temperature)
                loss_anom = torch.tensor(0.0, device=self.device)
                if np.random.random() < anomaly_aug_ratio:
                    x_anom, ei_anom = injector.inject_graph_level_anomaly(x, edge_index)
                    _, g_emb_orig, _ = self.model(x, edge_index)
                    _, g_emb_anom, _ = self.model(x_anom, ei_anom)
                    g_n = F.normalize(g_emb_orig, dim=1)
                    g_a = F.normalize(g_emb_anom, dim=1)
                    cos_sim = (g_n * g_a).sum(dim=1)
                    loss_anom = F.relu(cos_sim - 0.3).mean()
                node_emb, graph_emb = self.model.encoder(x, edge_index)
                x_recon = self.model.decoder.forward_node_recon(graph_emb, x.size(0))
                loss_recon = F.mse_loss(x_recon, x)
                loss = loss_cl + lambda_anom * loss_anom + lambda_recon * loss_recon
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            scheduler.step()
            self.model.eval()
            val_scores = []
            with torch.no_grad():
                for v in val_data:
                    s = self._compute_graph_anomaly_score(v["x"], v["edge_index"])
                    val_scores.append(s)
            vm = np.mean(val_scores) if val_scores else 0
            vs = np.std(val_scores) if val_scores else 0
            val_metric = vm + 0.5 * vs
            if val_metric < best_val_score:
                best_val_score = val_metric
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if (epoch + 1) % 10 == 0:
                logger.info("Epoch %d: loss=%.4f val_mean=%.4f best=%.4f",
                            epoch + 1, total_loss / max(n_batches, 1), vm, best_val_score)
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        if best_state:
            self.model.load_state_dict(best_state)
        self._fit_normal_distribution(train_data)
        # Re-validate with Mahalanobis distances (must happen AFTER fitting distribution)
        self.model.eval()
        re_val_scores = []
        with torch.no_grad():
            for v in val_data:
                s = self._compute_graph_anomaly_score(v["x"], v["edge_index"])
                re_val_scores.append(s)
        logger.info("Post-fit val scores: mean=%.4f std=%.4f",
                    np.mean(re_val_scores), np.std(re_val_scores))
        self._calibrate_threshold(val_data)
        self.is_ready = True
        self.model.eval()
        logger.info("Training complete. Threshold=%.4f", self.threshold)

    def _fit_normal_distribution(self, train_data):
        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for s in train_data:
                _, ge = self.model.encoder(s["x"], s["edge_index"])
                embeddings.append(ge.squeeze(0).numpy())
        if len(embeddings) < 2:
            return
        arr = np.array(embeddings)
        self._normal_embeddings = arr
        self.normal_mean = torch.tensor(arr.mean(axis=0), dtype=torch.float)
        # Strong regularization for high-dimensional / few-sample case
        n_samples, n_dim = arr.shape
        reg = max(1e-2, 0.1 * np.trace(np.cov(arr.T)) / n_dim) if n_dim > 0 else 1e-2
        cov = np.cov(arr.T) + np.eye(arr.shape[1]) * reg
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)
        self.normal_cov_inv = torch.tensor(cov_inv, dtype=torch.float)
        logger.info("Normal distribution fitted: %d samples, dim=%d",
                    len(embeddings), arr.shape[1])

    def _compute_graph_anomaly_score(self, x, edge_index):
        """Compute graph anomaly score using PCA + cosine similarity.
        
        More robust than Mahalanobis distance for high-dim embeddings
        with few training samples (avoids ill-conditioned covariance).
        """
        _, graph_emb = self.model.encoder(x, edge_index)
        emb = graph_emb.detach().cpu().numpy().flatten()
        
        if self._normal_embeddings is not None and len(self._normal_embeddings) > 5:
            try:
                if HAS_SKLEARN:
                    n_comp = min(16, len(self._normal_embeddings) - 1, emb.shape[0])
                    pca = PCA(n_components=n_comp)
                    normal_pca = pca.fit_transform(self._normal_embeddings)
                    test_pca = pca.transform(emb.reshape(1, -1))
                    sims = cosine_similarity(test_pca, normal_pca)
                    return float(1.0 - np.max(sims))
                else:
                    centroid = np.mean(self._normal_embeddings, axis=0)
                    return float(np.linalg.norm(emb - centroid))
            except Exception:
                centroid = np.mean(self._normal_embeddings, axis=0)
                return float(np.linalg.norm(emb - centroid))
        
        if self.normal_mean is not None:
            mean_np = self.normal_mean.detach().cpu().numpy().flatten()
            score = float(np.linalg.norm(emb - mean_np))
        else:
            score = float(np.linalg.norm(emb))
        return score

    def _calibrate_threshold(self, val_data):
        scores = []
        self.model.eval()
        with torch.no_grad():
            for s in val_data:
                scores.append(self._compute_graph_anomaly_score(s["x"], s["edge_index"]))
        if scores:
            mu = np.mean(scores)
            sigma = np.std(scores)
            # Use max of p95 and mu+2*sigma for robust threshold
            p95 = float(np.percentile(scores, 95))
            self.threshold = max(p95, mu + 2.0 * sigma)
            # Ensure threshold is at least mu + some margin
            self.threshold = max(self.threshold, mu * 1.5)
        else:
            self.threshold = 2.0
        logger.info("Threshold: %.4f (mean=%.4f std=%.4f p95=%.4f n=%d)",
                    self.threshold, np.mean(scores), np.std(scores),
                    float(np.percentile(scores, 95)) if scores else 0, len(scores))

    # --- Inference ---

    def detect(self, network_data):
        """Detect anomalies. Interface: detect({"graph": G, "measurements": meas})."""
        if not self.is_ready:
            return []
        graph = network_data.get("graph")
        if not graph or graph.number_of_nodes() == 0:
            return []
        try:
            data = self._build_data(graph, network_data)
            if data is None:
                return []
            x, edge_index = data
            self.model.eval()
            graph_score = self._compute_graph_anomaly_score(x, edge_index)
            if graph_score <= (self.threshold or 0.5):
                return []
            return self._localize_anomalies(graph, x, edge_index, graph_score)
        except Exception as e:
            logger.error("GNN v3 detection failed: %s", e)
            return []

    def _build_data(self, graph, network_data):
        node_list = list(graph.nodes())
        node_idx = {n: i for i, n in enumerate(node_list)}
        measurements = network_data.get("measurements", {})
        bus_voltages = {bv["bus"]: bv for bv in measurements.get("bus_voltages", [])}
        p_inject = {}
        for lp in measurements.get("line_powers", []):
            fb = int(lp.get("from_bus", -1))
            tb = int(lp.get("to_bus", -1))
            p = lp.get("p_mw", 0.0)
            side = lp.get("side", "from")
            if side == "from":
                p_inject[fb] = p_inject.get(fb, 0.0) - p
                p_inject[tb] = p_inject.get(tb, 0.0) + p
            else:
                p_inject[tb] = p_inject.get(tb, 0.0) - p
                p_inject[fb] = p_inject.get(fb, 0.0) + p
        source_buses = set()
        for node in node_list:
            if graph.nodes[node].get("is_source", False):
                bid = int(str(node).replace("bus_", "")) if "bus_" in str(node) else node
                source_buses.add(bid)
        components = {}
        for cid, comp in enumerate(nx.connected_components(graph)):
            for nd in comp:
                components[nd] = cid
        max_comp = max(components.values()) if components else 1
        max_p = max(abs(v) for v in p_inject.values()) if p_inject else 1.0
        max_p = max(max_p, 1e-6)
        features = []
        for node in node_list:
            deg = graph.degree(node)
            bv = bus_voltages.get(node, {})
            vm = bv.get("vm_pu", 1.0)
            bid = int(str(node).replace("bus_", "")) if "bus_" in str(node) else node
            p = p_inject.get(bid, 0.0) / max_p
            if bid in source_buses:
                bt = 0.0
            elif p > 0.01:
                bt = 0.5
            elif p < -0.01:
                bt = 0.75
            else:
                bt = 1.0
            feat = [deg / 10.0, vm, 1.0 if bid in source_buses else 0.0,
                    p, 0.0, vm - 1.0, bt, components.get(node, 0) / max(max_comp, 1)]
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
        return x, edge_index

    def _localize_anomalies(self, graph, x, edge_index, graph_score):
        node_list = list(graph.nodes())
        detections = []
        self.model.eval()
        with torch.no_grad():
            node_emb, graph_emb = self.model.encoder(x, edge_index)
            graph_center = node_emb.mean(dim=0)
            deviation = torch.norm(node_emb - graph_center.unsqueeze(0), dim=1)
            dev_np = deviation.numpy()
            dev_max = dev_np.max()
            dev_norm = dev_np / dev_max if dev_max > 0 else dev_np
            x_recon = self.model.decoder.forward_node_recon(graph_emb, x.size(0))
            feat_error = (x - x_recon).abs().numpy()
        dev_mean = dev_np.mean()
        dev_std = dev_np.std()
        node_thresh = dev_mean + 1.0 * dev_std
        g_conf = min(0.5 + (graph_score - (self.threshold or 0.5)) / (self.threshold + 1e-6) * 0.3, 0.98)
        for i, (si, node) in enumerate(zip(dev_np, node_list)):
            if si > node_thresh and dev_norm[i] > 0.5:
                ve = feat_error[i][1]
                se = feat_error[i][0]
                if se > ve * 1.5 and se > 0.05:
                    at = "\u62d3\u6251\u4e2d\u65ad"
                elif ve > 0.1:
                    at = "\u865a\u63a5/\u9519\u63a5"
                else:
                    at = "\u9065\u6d4b!=\u62d3\u6251"
                nc = max(0.5, min(g_conf * dev_norm[i], 0.99))
                detections.append({
                    "type": at, "location": str(node), "confidence": float(nc),
                    "layer": "GNN_v2",
                    "details": "GraphCL: graph_score=%.4f, node_dev=%.4f, thresh=%.4f"
                               % (graph_score, si, self.threshold),
                })
        detections.sort(key=lambda d: d.get("confidence", 0), reverse=True)
        return detections[:5]
