# -*- coding: utf-8 -*-
"""
拓扑异常可解释性模块
为检测结果提供推理链路、特征重要性分析和物理解释。
不依赖SHAP库，使用扰动法自行实现特征重要性计算。
"""
import copy
import math
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter, defaultdict
import logging

logger = logging.getLogger(__name__)


# 异常类型到物理含义的映射
ANOMALY_PHYSICS = {
    "图模不符": {
        "description": "CIM模型与SVG图形的设备信息不一致",
        "impact": "导致拓扑建模错误，影响状态估计和保护整定",
        "domain": "信息模型",
    },
    "拓扑中断": {
        "description": "网络拓扑存在不连通区域或孤立节点",
        "impact": "部分区域失去监控，可能导致孤岛运行风险",
        "domain": "网络连通性",
    },
    "虚接/错接": {
        "description": "设备连接关系与实际不符",
        "impact": "潮流计算结果失真，保护可能误动或拒动",
        "domain": "连接关系",
    },
    "遥测!=拓扑": {
        "description": "SCADA量测数据与拓扑模型预测的电气状态矛盾",
        "impact": "状态估计残差增大，可能是传感器故障或拓扑错误",
        "domain": "量测一致性",
    },
    "遥信!=遥测": {
        "description": "开关遥信位置与量测推断的位置不一致",
        "impact": "拓扑状态不确定，影响网络分析应用的可靠性",
        "domain": "开关状态",
    },
}

# 检测层的可信度权重
LAYER_CREDIBILITY = {
    "rule_engine": 0.7,       # 规则引擎：确定性高但覆盖面有限
    "state_estimator": 0.85,  # 状态估计：基于统计推断，较可靠
    "gnn": 0.6,              # GNN：数据驱动，需人工确认
    "enhanced": 0.8,          # 增强检测器：综合多层
}


class TopologyExplainer:
    """
    拓扑异常检测结果解释器。

    提供:
      - 单个异常的推理链路解释
      - 类似SHAP的扰动法特征重要性
      - 完整解释报告生成
      - 检测关注区域可视化（文本描述）
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        """
        Args:
            feature_names: 可解释的特征名称列表。
                           默认使用拓扑+量测的标准特征集。
        """
        self.feature_names = feature_names or [
            "节点电压幅值",
            "节点电压相角",
            "线路有功功率",
            "线路无功功率",
            "开关状态",
            "节点度数",
            "连通分量数",
            "线路负载率",
            "量测残差",
            "chi2统计量",
        ]

    # ============================================================
    # 单个异常推理链路解释
    # ============================================================
    def explain_detection(self, anomaly: Dict,
                          network_data: Dict) -> Dict:
        """
        为单个异常生成推理链路解释。

        输出包含:
          - evidence_chain: 证据链（从数据到结论的推理步骤）
          - key_features: 关键特征及其贡献
          - physics_meaning: 物理含义解释
          - confidence_breakdown: 置信度分解

        Args:
            anomaly:     异常字典 {type, location, confidence, layer, details}
            network_data: 网络数据字典

        Returns:
            解释字典
        """
        atype = anomaly.get("type", "未知")
        location = anomaly.get("location", "未知")
        confidence = anomaly.get("confidence", 0.0)
        layer = anomaly.get("layer", "未知")
        details = anomaly.get("details", "")

        # 标准化类型
        from utils.metrics import _normalize_type
        norm_type = _normalize_type(atype)

        # 1. 构建证据链
        evidence_chain = self._build_evidence_chain(anomaly, network_data)

        # 2. 提取关键特征
        key_features = self._extract_key_features(anomaly, network_data)

        # 3. 物理含义
        physics = ANOMALY_PHYSICS.get(norm_type, {
            "description": f"未识别的异常类型: {atype}",
            "impact": "影响待评估",
            "domain": "未知",
        })

        # 4. 置信度分解
        confidence_breakdown = self._decompose_confidence(anomaly)

        # 5. 修正建议
        suggestion = self._generate_suggestion(anomaly, network_data)

        return {
            "anomaly_type": norm_type,
            "location": location,
            "evidence_chain": evidence_chain,
            "key_features": key_features,
            "physics_meaning": physics,
            "confidence_breakdown": confidence_breakdown,
            "suggestion": suggestion,
            "layer": layer,
            "raw_confidence": confidence,
        }

    def _build_evidence_chain(self, anomaly: Dict,
                              network_data: Dict) -> List[Dict]:
        """
        构建从数据到结论的推理证据链。

        Returns:
            [{step: int, action: str, observation: str, conclusion: str}, ...]
        """
        atype = anomaly.get("type", "")
        layer = anomaly.get("layer", "")
        details = anomaly.get("details", "")
        location = str(anomaly.get("location", ""))

        chain = []
        step = 1

        # 步骤1: 数据采集
        chain.append({
            "step": step,
            "action": "数据采集",
            "observation": f"获取网络拓扑数据和量测数据",
            "conclusion": "输入数据就绪",
        })
        step += 1

        # 步骤2: 检测层分析
        if layer == "rule_engine":
            chain.append({
                "step": step,
                "action": "规则引擎检测",
                "observation": f"对 {location} 应用领域规则校验",
                "conclusion": "触发规则匹配",
            })
        elif layer == "state_estimator":
            chain.append({
                "step": step,
                "action": "状态估计分析",
                "observation": f"运行WLS状态估计，分析量测残差",
                "conclusion": "统计检验发现异常",
            })
        elif layer == "gnn":
            chain.append({
                "step": step,
                "action": "GNN图神经网络推理",
                "observation": f"图嵌入特征与正常模式偏差显著",
                "conclusion": "模式识别标记异常",
            })
        else:
            chain.append({
                "step": step,
                "action": f"{layer}层检测",
                "observation": details[:100] if details else "分析中",
                "conclusion": "检测到异常信号",
            })
        step += 1

        # 步骤3: 类型判定
        norm_type = from_utils_type(atype)
        physics = ANOMALY_PHYSICS.get(norm_type, {})
        chain.append({
            "step": step,
            "action": "异常类型判定",
            "observation": f"异常模式匹配 -> {norm_type}",
            "conclusion": f"属于{physics.get('domain', '未知')}类异常",
        })
        step += 1

        # 步骤4: 定位确认
        graph = network_data.get("graph")
        if graph is not None:
            node_info = ""
            if hasattr(graph, "nodes") and location in graph.nodes:
                ndata = graph.nodes[location]
                node_info = f"节点属性: {ndata}"
            chain.append({
                "step": step,
                "action": "位置定位",
                "observation": f"异常定位在 {location}. {node_info}",
                "conclusion": f"锁定设备/节点: {location}",
            })
        else:
            chain.append({
                "step": step,
                "action": "位置定位",
                "observation": f"异常定位在 {location}",
                "conclusion": f"位置: {location}",
            })

        return chain

    def _extract_key_features(self, anomaly: Dict,
                              network_data: Dict) -> List[Dict]:
        """
        提取与该异常相关的关键特征。

        Returns:
            [{feature: str, value: float, importance: str, direction: str}, ...]
        """
        atype = anomaly.get("type", "")
        layer = anomaly.get("layer", "")
        details = anomaly.get("details", "")
        location = str(anomaly.get("location", ""))

        features = []
        norm_type = from_utils_type(atype)

        # 根据异常类型推断关键特征
        if "遥测" in norm_type or "不良数据" in atype:
            features.append({
                "feature": "量测残差",
                "value": _extract_numeric_detail(details, "残差"),
                "importance": "高",
                "direction": "异常偏大",
            })
            features.append({
                "feature": "chi2统计量",
                "value": _extract_numeric_detail(details, "chi2"),
                "importance": "高",
                "direction": "超过临界值",
            })

        if "拓扑" in norm_type or "中断" in atype:
            features.append({
                "feature": "连通分量数",
                "value": _count_components(network_data),
                "importance": "高",
                "direction": "大于1（不连通）",
            })
            features.append({
                "feature": "节点度数",
                "value": _get_node_degree(location, network_data),
                "importance": "中",
                "direction": "异常低（可能孤立）",
            })

        if "图模" in norm_type:
            features.append({
                "feature": "CIM设备数",
                "value": len(network_data.get("cim_devices", [])),
                "importance": "中",
                "direction": "与SVG不匹配",
            })
            features.append({
                "feature": "SVG设备数",
                "value": len(network_data.get("svg_devices", [])),
                "importance": "中",
                "direction": "与CIM不匹配",
            })

        if "虚接" in norm_type or "错接" in atype:
            features.append({
                "feature": "连接关系",
                "value": None,
                "importance": "高",
                "direction": "与模型不一致",
            })

        if "遥信" in norm_type:
            features.append({
                "feature": "开关状态",
                "value": None,
                "importance": "高",
                "direction": "遥信与量测矛盾",
            })

        # 通用特征: 检测层可信度
        cred = LAYER_CREDIBILITY.get(layer, 0.5)
        features.append({
            "feature": "检测层可信度",
            "value": cred,
            "importance": "背景",
            "direction": f"{layer}层可信度={cred:.0%}",
        })

        # 置信度
        features.append({
            "feature": "检测置信度",
            "value": anomaly.get("confidence", 0),
            "importance": "综合",
            "direction": "越高越确定",
        })

        return features

    def _decompose_confidence(self, anomaly: Dict) -> Dict:
        """
        分解置信度为各因素贡献。

        Returns:
            {final: float, layer_base: float, multi_layer_boost: float,
             consistency_bonus: float, factors: [...]}
        """
        confidence = anomaly.get("confidence", 0.0)
        layer = anomaly.get("layer", "未知")
        layers = anomaly.get("detected_by_layers", [layer])

        layer_base = LAYER_CREDIBILITY.get(layer, 0.5)

        # 多层确认加分
        multi_layer = 0.0
        if len(layers) > 1:
            multi_layer = min(0.1 * (len(layers) - 1), 0.2)

        # 一致性奖励
        consistency = 0.0
        if confidence > 0.8:
            consistency = 0.05

        factors = [
            f"基础层({layer})可信度: {layer_base:.2f}",
        ]
        if multi_layer > 0:
            factors.append(f"多层确认({len(layers)}层)加分: +{multi_layer:.2f}")
        if consistency > 0:
            factors.append(f"高一致性加分: +{consistency:.2f}")

        return {
            "final": confidence,
            "layer_base": layer_base,
            "multi_layer_boost": multi_layer,
            "consistency_bonus": consistency,
            "detected_by_layers": layers,
            "factors": factors,
        }

    def _generate_suggestion(self, anomaly: Dict,
                             network_data: Dict) -> Dict:
        """基于异常分析生成修正建议"""
        atype = anomaly.get("type", "")
        norm_type = from_utils_type(atype)
        confidence = anomaly.get("confidence", 0.0)
        location = str(anomaly.get("location", ""))

        if confidence >= 0.85:
            priority = "高"
            action = "立即处理"
        elif confidence >= 0.60:
            priority = "中"
            action = "计划处理"
        else:
            priority = "低"
            action = "待确认"

        suggestions = {
            "图模不符": [
                "核对CIM和SVG的设备清单",
                "检查CIM文件完整性",
                "同步CIM/SVG数据源",
            ],
            "拓扑中断": [
                "检查开关实际位置",
                "检查线路in_service状态",
                "核实是否存在未录入的连接",
            ],
            "虚接/错接": [
                "核对设备端子映射",
                "检查ConnectivityNode关联",
                "现场核实接线",
            ],
            "遥测!=拓扑": [
                "检查传感器/互感器状态",
                "对比冗余量测",
                "运行状态估计替代",
            ],
            "遥信!=遥测": [
                "现场核实开关位置",
                "检查遥信采集回路",
                "更新SCADA数据库",
            ],
        }

        return {
            "priority": priority,
            "action": action,
            "steps": suggestions.get(norm_type, ["人工核查"]),
            "target": location,
        }

    # ============================================================
    # 特征重要性（扰动法，类SHAP）
    # ============================================================
    def compute_feature_importance(self, anomalies: List[Dict],
                                   network_data: Dict,
                                   n_perturbations: int = 50) -> Dict:
        """
        使用扰动法计算特征重要性（不依赖SHAP库）。

        方法:
          1. 对每个特征，随机扰动其值（置0或加噪声）
          2. 观察检测结果的变化（异常数量、置信度分布变化）
          3. 变化越大 -> 该特征越重要

        Args:
            anomalies:       原始检测到的异常列表
            network_data:    网络数据
            n_perturbations: 每个特征的扰动次数

        Returns:
            {
                feature_importance: {feature: importance_score},
                ranking: [(feature, score), ...],
                method: "perturbation",
                n_perturbations: int,
                baseline_summary: {...},
            }
        """
        # 基线: 原始异常的统计摘要
        baseline = self._summarize_anomalies(anomalies)

        importance_scores = {}

        for feat_name in self.feature_names:
            # 对该特征执行多次扰动
            deltas = []
            for _ in range(n_perturbations):
                perturbed_data = self._perturb_feature(
                    feat_name, network_data)
                # 用扰动后的数据重新评估（模拟）
                perturbed_anomalies = self._simulate_detection(
                    anomalies, feat_name, perturbed_data)
                perturbed_summary = self._summarize_anomalies(
                    perturbed_anomalies)
                delta = self._compute_summary_delta(
                    baseline, perturbed_summary)
                deltas.append(delta)

            # 重要性 = 扰动引起的变化的平均值
            importance_scores[feat_name] = round(
                sum(deltas) / len(deltas), 4) if deltas else 0.0

        # 归一化到 [0, 1]
        max_score = max(importance_scores.values()) if importance_scores else 1.0
        if max_score > 0:
            importance_scores = {
                k: round(v / max_score, 4)
                for k, v in importance_scores.items()
            }

        # 排序
        ranking = sorted(importance_scores.items(),
                         key=lambda x: x[1], reverse=True)

        logger.info("特征重要性计算完成: top3={}".format(
            [(r[0], r[1]) for r in ranking[:3]]))

        return {
            "feature_importance": importance_scores,
            "ranking": ranking,
            "method": "perturbation",
            "n_perturbations": n_perturbations,
            "baseline_summary": baseline,
        }

    def _summarize_anomalies(self, anomalies: List[Dict]) -> Dict:
        """将异常列表压缩为统计摘要（用于比较）"""
        if not anomalies:
            return {"count": 0, "avg_confidence": 0, "type_entropy": 0}

        type_counter = Counter(
            a.get("type", "?") for a in anomalies)
        confidences = [a.get("confidence", 0) for a in anomalies]
        total = sum(type_counter.values())

        # 类型分布熵
        entropy = 0.0
        for c in type_counter.values():
            p = c / total
            if p > 0:
                entropy -= p * math.log2(p)

        return {
            "count": len(anomalies),
            "avg_confidence": sum(confidences) / len(confidences),
            "type_entropy": round(entropy, 4),
            "type_distribution": dict(type_counter),
        }

    def _perturb_feature(self, feature_name: str,
                         network_data: Dict) -> Dict:
        """
        对指定特征进行扰动，返回扰动参数。

        扰动策略:
          - 量测类特征: 加高斯噪声或置零
          - 拓扑类特征: 随机翻转/修改
          - 统计类特征: 缩放
        """
        import random

        perturbation = {"feature": feature_name, "method": "unknown"}

        if "电压" in feature_name:
            perturbation["method"] = "add_noise"
            perturbation["sigma"] = random.uniform(0.01, 0.05)
        elif "功率" in feature_name:
            perturbation["method"] = "scale"
            perturbation["factor"] = random.uniform(0.8, 1.2)
        elif "开关" in feature_name:
            perturbation["method"] = "flip"
            perturbation["flip_prob"] = random.uniform(0.05, 0.2)
        elif "度数" in feature_name or "连通" in feature_name:
            perturbation["method"] = "zero_out"
        elif "残差" in feature_name or "chi2" in feature_name:
            perturbation["method"] = "scale"
            perturbation["factor"] = random.uniform(0.5, 2.0)
        else:
            perturbation["method"] = "add_noise"
            perturbation["sigma"] = 0.1

        return perturbation

    def _simulate_detection(self, original_anomalies: List[Dict],
                            feature_name: str,
                            perturbation: Dict) -> List[Dict]:
        """
        模拟特征扰动后的检测结果。

        策略：基于特征与异常类型的关联度，概率性地移除或修改异常。
        这是一个近似模拟，不需要真正重新运行检测器。
        """
        import random

        # 特征到异常类型的关联矩阵
        feature_type_affinity = {
            "量测残差": ["遥测!=拓扑", "不良数据"],
            "chi2统计量": ["遥测!=拓扑", "chi2检验未通过"],
            "节点电压幅值": ["遥测!=拓扑"],
            "节点电压相角": ["遥测!=拓扑"],
            "线路有功功率": ["遥测!=拓扑"],
            "线路无功功率": ["遥测!=拓扑"],
            "开关状态": ["遥信!=遥测", "拓扑错误"],
            "节点度数": ["拓扑中断"],
            "连通分量数": ["拓扑中断"],
            "线路负载率": ["遥测!=拓扑"],
        }

        affected_types = set(feature_type_affinity.get(feature_name, []))
        method = perturbation.get("method", "add_noise")

        result = []
        for a in original_anomalies:
            atype = a.get("type", "")
            is_affected = any(ft in atype for ft in affected_types)

            if is_affected:
                # 受影响的异常：概率性变化
                if method == "zero_out":
                    # 置零: 高概率移除该异常
                    if random.random() < 0.6:
                        continue
                elif method == "flip":
                    # 翻转: 概率改变置信度
                    a = copy.copy(a)
                    a["confidence"] = max(0.1, a.get("confidence", 0.5)
                                          + random.uniform(-0.3, 0.3))
                elif method == "scale":
                    factor = perturbation.get("factor", 1.0)
                    a = copy.copy(a)
                    a["confidence"] = min(0.99, max(0.1,
                        a.get("confidence", 0.5) * factor))
                else:
                    # 噪声: 轻微扰动置信度
                    sigma = perturbation.get("sigma", 0.05)
                    a = copy.copy(a)
                    a["confidence"] = min(0.99, max(0.1,
                        a.get("confidence", 0.5) + random.gauss(0, sigma)))

            result.append(a)

        return result

    @staticmethod
    def _compute_summary_delta(baseline: Dict, perturbed: Dict) -> float:
        """计算两次检测摘要之间的差异度"""
        # 数量变化
        count_delta = abs(baseline["count"] - perturbed["count"])
        count_rel = count_delta / max(baseline["count"], 1)

        # 平均置信度变化
        conf_delta = abs(
            baseline["avg_confidence"] - perturbed["avg_confidence"])

        # 熵变化
        entropy_delta = abs(
            baseline["type_entropy"] - perturbed["type_entropy"])

        # 加权组合
        return 0.4 * count_rel + 0.4 * conf_delta + 0.2 * entropy_delta

    # ============================================================
    # 完整解释报告
    # ============================================================
    def generate_explanation_report(self, anomalies: List[Dict],
                                    network_data: Dict) -> Dict:
        """
        生成完整的异常解释报告。

        对每个异常: 检测依据 → 关键证据 → 物理解释 → 修正建议

        Args:
            anomalies:    异常列表
            network_data: 网络数据

        Returns:
            {
                explanations: [explain_detection() 的结果, ...],
                feature_importance: compute_feature_importance() 的结果,
                attention_summary: visualize_attention() 的结果,
                summary: {
                    total: int,
                    by_domain: {domain: count},
                    high_priority_count: int,
                    top_features: [(feature, score), ...],
                }
            }
        """
        explanations = []
        for anomaly in anomalies:
            exp = self.explain_detection(anomaly, network_data)
            explanations.append(exp)

        # 特征重要性
        feat_imp = self.compute_feature_importance(anomalies, network_data)

        # 注意力摘要
        attention = self.visualize_attention(
            network_data.get("graph"), anomalies)

        # 按物理域分组
        by_domain = Counter()
        high_priority = 0
        for exp in explanations:
            domain = exp.get("physics_meaning", {}).get("domain", "未知")
            by_domain[domain] += 1
            if exp.get("suggestion", {}).get("priority") == "高":
                high_priority += 1

        top_features = feat_imp.get("ranking", [])[:5]

        summary = {
            "total": len(explanations),
            "by_domain": dict(by_domain),
            "high_priority_count": high_priority,
            "top_features": top_features,
            "attention_focus": attention.get("focus_areas", []),
        }

        logger.info("解释报告生成: {}个异常, {}个高优先级, top特征={}".format(
            len(explanations), high_priority,
            top_features[0] if top_features else "无"))

        return {
            "explanations": explanations,
            "feature_importance": feat_imp,
            "attention_summary": attention,
            "summary": summary,
        }

    # ============================================================
    # 注意力可视化（文本描述）
    # ============================================================
    def visualize_attention(self, graph, anomalies: List[Dict]) -> Dict:
        """
        可视化检测关注区域（返回文本描述）。

        分析:
          - 哪些区域异常密度最高
          - 哪些节点/设备被标记次数最多
          - 异常的空间分布模式

        Args:
            graph:    NetworkX 拓扑图
            anomalies: 异常列表

        Returns:
            {
                focus_areas: [{area, anomaly_count, types, description}],
                hotspots: [{node, degree, anomaly_count}],
                distribution_pattern: str,
                text_description: str,
            }
        """
        if not anomalies:
            return {
                "focus_areas": [],
                "hotspots": [],
                "distribution_pattern": "无异常",
                "text_description": "未检测到异常，网络状态正常。",
            }

        # 统计每个位置的异常数
        location_counts = Counter()
        location_types = defaultdict(set)
        for a in anomalies:
            loc = str(a.get("location", "未知"))
            location_counts[loc] += 1
            location_types[loc].add(a.get("type", "?"))

        # 热点节点
        hotspots = []
        for loc, count in location_counts.most_common(10):
            node_data = {}
            if graph is not None and hasattr(graph, "nodes") and loc in graph.nodes:
                node_data = dict(graph.nodes[loc])
            hotspots.append({
                "node": loc,
                "anomaly_count": count,
                "types": sorted(location_types[loc]),
                "degree": graph.degree(loc) if graph is not None
                          and hasattr(graph, "nodes")
                          and loc in graph.nodes else None,
            })

        # 聚焦区域分析
        focus_areas = self._identify_focus_areas(
            location_counts, location_types, graph)

        # 分布模式
        pattern = self._describe_distribution_pattern(
            location_counts, graph)

        # 生成文本描述
        text = self._generate_attention_text(
            anomalies, hotspots, focus_areas, pattern)

        return {
            "focus_areas": focus_areas,
            "hotspots": hotspots[:5],
            "distribution_pattern": pattern,
            "text_description": text,
        }

    def _identify_focus_areas(self, location_counts: Counter,
                              location_types: Dict,
                              graph) -> List[Dict]:
        """识别异常聚集区域"""
        areas = []

        # 按异常数量排序
        for loc, count in location_counts.most_common(5):
            types = sorted(location_types[loc])

            # 判断区域特征
            if graph is not None and hasattr(graph, "nodes") and loc in graph.nodes:
                degree = graph.degree(loc)
                if degree == 0:
                    area_type = "孤立节点"
                elif degree == 1:
                    area_type = "末端节点"
                elif degree >= 4:
                    area_type = "枢纽节点"
                else:
                    area_type = "普通节点"
            else:
                area_type = "未知区域"

            description = (
                f"{area_type} {loc}: "
                f"{count}个异常, 类型={', '.join(types)}"
            )

            areas.append({
                "area": loc,
                "area_type": area_type,
                "anomaly_count": count,
                "types": types,
                "description": description,
            })

        return areas

    def _describe_distribution_pattern(self, location_counts: Counter,
                                       graph) -> str:
        """描述异常的空间分布模式"""
        if not location_counts:
            return "无异常"

        n_locations = len(location_counts)
        total_anomalies = sum(location_counts.values())
        max_count = max(location_counts.values())

        # 判断是集中还是分散
        if n_locations == 1:
            return "单点集中"
        elif max_count >= total_anomalies * 0.5:
            return "主热点集中"
        elif n_locations <= 3:
            return "少量分散"
        else:
            return "多点分散"

    def _generate_attention_text(self, anomalies: List[Dict],
                                 hotspots: List[Dict],
                                 focus_areas: List[Dict],
                                 pattern: str) -> str:
        """生成检测关注区域的文本描述"""
        lines = []
        lines.append(f"=== 检测关注区域分析 ===")
        lines.append(f"共检测到 {len(anomalies)} 个异常")
        lines.append(f"分布模式: {pattern}")
        lines.append("")

        if hotspots:
            lines.append("【热点设备】")
            for hs in hotspots[:3]:
                lines.append(
                    f"  - {hs['node']}: {hs['anomaly_count']}个异常, "
                    f"类型={', '.join(hs['types'])}")

        if focus_areas:
            lines.append("")
            lines.append("【聚焦区域】")
            for fa in focus_areas[:3]:
                lines.append(f"  - {fa['description']}")

        # 类型分布
        type_counter = Counter(a.get("type", "?") for a in anomalies)
        lines.append("")
        lines.append("【异常类型分布】")
        for t, c in type_counter.most_common():
            lines.append(f"  - {t}: {c}个")

        return "\n".join(lines)


# ============================================================
# 辅助函数
# ============================================================

def from_utils_type(atype: str) -> str:
    """标准化异常类型名称"""
    from utils.metrics import _normalize_type
    return _normalize_type(atype)


def _extract_numeric_detail(details: str, keyword: str) -> Optional[float]:
    """从详情文本中提取与关键词关联的数值"""
    if not details or keyword not in details:
        return None
    import re
    # 尝试在关键词附近找数字
    pattern = keyword + r'[=:：\s]*([0-9]+\.?[0-9]*)'
    match = re.search(pattern, details)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    # 退而求其次，找所有数字
    numbers = re.findall(r'([0-9]+\.?[0-9]*)', details)
    if numbers:
        try:
            return float(numbers[0])
        except ValueError:
            pass
    return None


def _count_components(network_data: Dict) -> int:
    """计算网络连通分量数"""
    graph = network_data.get("graph")
    if graph is not None:
        import networkx as nx
        return nx.number_connected_components(graph)
    return -1


def _get_node_degree(location: str, network_data: Dict) -> int:
    """获取指定节点的度数"""
    graph = network_data.get("graph")
    if graph is not None and hasattr(graph, "nodes") and location in graph.nodes:
        return graph.degree(location)
    return -1

# ===== P2-2: Causal Explainer (Counterfactual Reasoning) =====

class CausalExplainer:
    """Causal explanation for anomaly detections.\n\nUses counterfactual reasoning: "If X were different, would the anomaly disappear?"\nProvides actionable explanations for operators.\n"""
    
    def __init__(self, net=None):
        self.net = net
    
    def explain(self, anomaly, network_data):
        """Generate causal explanation for an anomaly.\n\nArgs:\nanomaly: detection dict with type, location, evidence\nnetwork_data: network data dict\n\nReturns:\nexplanation dict with causal_chain, counterfactual, recommendation\n"""
        atype = anomaly.get("type", "")
        
        if atype == "topo_interrupt":
            return self._explain_topology_interrupt(anomaly, network_data)
        elif atype == "virtual_faulty":
            return self._explain_virtual_faulty(anomaly, network_data)
        elif atype == "hif_fault":
            return self._explain_hif(anomaly, network_data)
        elif atype == "line_break":
            return self._explain_line_break(anomaly, network_data)
        elif atype == "signal_mismatch":
            return self._explain_switch_mismatch(anomaly, network_data)
        elif atype == "telemetry_mismatch":
            return self._explain_telemetry(anomaly, network_data)
        elif atype == "model_mismatch":
            return self._explain_model_mismatch(anomaly, network_data)
        else:
            return self._explain_generic(anomaly, network_data)
    
    def _explain_topology_interrupt(self, anomaly, network_data):
        evidence = anomaly.get("evidence", {})
        bridge_line = evidence.get("bridge_line", "unknown")
        comp_size = evidence.get("component_size", 0)
        
        return {
            "anomaly_type": "topo_interrupt",
            "causal_chain": [
                f"Line {bridge_line} is out of service",
                f"This disconnects {comp_size} nodes from the main network",
                "The isolated component loses power supply",
            ],
            "counterfactual": f"If line {bridge_line} were restored, all {comp_size} nodes would rejoin the main network",
            "recommendation": f"1. Check line {bridge_line} status (breaker/switch)\n2. If physical line is intact, close the breaker\n3. If line is damaged, reroute via tie switches",
            "severity": "critical",
            "confidence": anomaly.get("confidence", 0.9),
        }
    
    def _explain_virtual_faulty(self, anomaly, network_data):
        evidence = anomaly.get("evidence", {})
        bus = evidence.get("bus", "unknown")
        vm = evidence.get("vm_pu", 1.0)
        
        return {
            "anomaly_type": "virtual_faulty",
            "causal_chain": [
                f"Bus {bus} voltage = {vm:.3f} pu, deviating from expected range",
                "This suggests a virtual/faulty connection in the model",
                "Possible causes: wrong bus connection, missing device, measurement error",
            ],
            "counterfactual": f"If bus {bus} connection were correct, voltage would be within [0.95, 1.05] pu",
            "recommendation": f"1. Verify bus {bus} CIM connectivity model\n2. Check if any device is incorrectly connected\n3. Compare with field inspection data",
            "severity": "medium",
            "confidence": anomaly.get("confidence", 0.8),
        }
    
    def _explain_hif(self, anomaly, network_data):
        evidence = anomaly.get("evidence", {})
        line = evidence.get("line_idx", "unknown")
        score = evidence.get("hif_score", 0)
        criteria = evidence.get("criteria", [])
        
        return {
            "anomaly_type": "hif_fault",
            "causal_chain": [
                f"Line {line} shows {score}/5 HIF indicators",
                "High impedance faults have subtle signatures",
                "Traditional overcurrent protection cannot detect them",
            ],
            "criteria_detail": criteria,
            "counterfactual": f"If line {line} had no HIF, voltage/current would be normal",
            "recommendation": f"1. Dispatch field crew to line {line}\n2. Check for downed conductor or tree contact\n3. Use HIF-specific relay (21H/50H) for confirmation",
            "severity": "critical",
            "confidence": anomaly.get("confidence", 0.7),
        }
    
    def _explain_line_break(self, anomaly, network_data):
        evidence = anomaly.get("evidence", {})
        line = evidence.get("line_idx", "unknown")
        
        return {
            "anomaly_type": "line_break",
            "causal_chain": [
                f"Line {line} shows near-zero current despite being in-service",
                "Voltage difference between ends suggests open conductor",
                "Downstream loads may be de-energized",
            ],
            "counterfactual": f"If line {line} were intact, current would match load demand",
            "recommendation": f"1. Check line {line} for physical break\n2. Inspect fuses and disconnects\n3. Restore via alternate path if available",
            "severity": "critical",
            "confidence": anomaly.get("confidence", 0.7),
        }
    
    def _explain_switch_mismatch(self, anomaly, network_data):
        return {
            "anomaly_type": "signal_mismatch",
            "causal_chain": [
                "Switch telemetry and topology state disagree",
                "Possible SCADA communication error or actual switch malfunction",
            ],
            "counterfactual": "If switch state matched telemetry, KCL would balance",
            "recommendation": "1. Verify switch position locally\n2. Check SCADA communication\n3. Update topology model if needed",
            "severity": "medium",
            "confidence": anomaly.get("confidence", 0.7),
        }
    
    def _explain_telemetry(self, anomaly, network_data):
        return {
            "anomaly_type": "telemetry_mismatch",
            "causal_chain": [
                "Voltage measurement deviates from power flow prediction",
                "Possible bad PT/CT, measurement drift, or topology error",
            ],
            "counterfactual": "If measurement were correct, state estimation would converge",
            "recommendation": "1. Check PT/CT calibration\n2. Compare with adjacent meters\n3. Replace sensor if drift confirmed",
            "severity": "medium",
            "confidence": anomaly.get("confidence", 0.7),
        }
    
    def _explain_model_mismatch(self, anomaly, network_data):
        return {
            "anomaly_type": "model_mismatch",
            "causal_chain": [
                "CIM model and SVG graphics have different device counts",
                "Some devices exist in one but not the other",
            ],
            "counterfactual": "If models were synchronized, all devices would match",
            "recommendation": "1. Compare CIM and SVG device lists\n2. Add missing devices to the deficient model\n3. Re-synchronize models",
            "severity": "medium",
            "confidence": anomaly.get("confidence", 0.8),
        }
    
    def _explain_generic(self, anomaly, network_data):
        return {
            "anomaly_type": anomaly.get("type", "unknown"),
            "causal_chain": [anomaly.get("description", "Unknown anomaly detected")],
            "counterfactual": "Manual investigation required",
            "recommendation": "Review anomaly details and dispatch inspection team",
            "severity": anomaly.get("severity", "medium"),
            "confidence": anomaly.get("confidence", 0.5),
        }
    
    def explain_batch(self, anomalies, network_data):
        """Explain a batch of anomalies."""
        return [self.explain(a, network_data) for a in anomalies]