# -*- coding: utf-8 -*-
"""
检测协调器
串联规则引擎（第一层）和状态估计（第二层），合并检测结果
"""
import copy
from typing import Dict, List
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """异常检测协调器: 串联两层检测并合并去重"""

    def __init__(self, use_rule_engine: bool = True,
                 use_state_estimator: bool = True):
        """
        Args:
            use_rule_engine:      是否启用第一层规则引擎
            use_state_estimator:  是否启用第二层状态估计
        """
        self.use_rule_engine     = use_rule_engine
        self.use_state_estimator = use_state_estimator
        self.rule_results: List[Dict] = []
        self.se_results:   List[Dict] = []
        self.merged_results: List[Dict] = []

    def detect_all(self, network_data: Dict) -> List[Dict]:
        """
        综合运行所有检测层。

        Args:
            network_data: 网络数据字典，包含:
                graph        - NetworkX拓扑图
                cim_devices  - CIM设备列表
                svg_devices  - SVG设备列表
                measurements - SCADA量测数据
                switches     - 开关列表
                net          - PandaPower网络 (可选, 用于状态估计)
                scada_data   - 原始SCADA数据 (可选)

        Returns:
            标准化异常列表 [{type, location, confidence, layer, details}, ...]
        """
        all_anomalies: List[Dict] = []

        # ---- 第一层: 规则引擎 ----
        if self.use_rule_engine:
            from anomaly_detection.rule_engine import run_rule_engine
            rule_anomalies = run_rule_engine(
                graph=network_data["graph"],
                cim_devices=network_data.get("cim_devices", []),
                svg_devices=network_data.get("svg_devices", []),
                measurements=network_data.get("measurements", {}),
                switches=network_data.get("switches", []),
            )
            self.rule_results = rule_anomalies
            all_anomalies.extend(rule_anomalies)
            logger.info(f"[协调器] 规则引擎: {len(rule_anomalies)} 个异常")

        # ---- 第二层: 状态估计 ----
        if self.use_state_estimator and network_data.get("net") is not None:
            se_anomalies = self._run_state_estimation(network_data)
            self.se_results = se_anomalies
            all_anomalies.extend(se_anomalies)
            logger.info(f"[协调器] 状态估计: {len(se_anomalies)} 个异常")

        # ---- 合并与去重 ----
        self.merged_results = self._merge_anomalies(all_anomalies)
        logger.info(
            f"[协调器] 合并后: {len(self.merged_results)} 个异常 "
            f"(规则引擎={len(self.rule_results)}, "
            f"状态估计={len(self.se_results)})")

        return self.merged_results

    # ----------------------------------------------------------
    def _run_state_estimation(self, network_data: Dict) -> List[Dict]:
        """运行状态估计层"""
        from anomaly_detection.state_estimator import (
            add_measurements_to_network, run_wls_estimation,
            detect_bad_data, identify_topology_errors,
        )

        net = copy.deepcopy(network_data["net"])
        scada_data = network_data.get("scada_data",
                                      network_data.get("measurements", {}))
        anomalies: List[Dict] = []

        # 添加量测
        add_measurements_to_network(net, scada_data)

        # WLS
        success, results = run_wls_estimation(net)
        if not success:
            anomalies.append({
                "type": "状态估计失败",
                "location": "system",
                "confidence": 0.50,
                "layer": "state_estimator",
                "details": f"WLS未收敛: {results.get('error', '未知')}",
            })
            return anomalies

        # 不良数据检测
        bd = detect_bad_data(net)

        if not bd["chi2_passed"]:
            if bd.get("systematic_error", False):
                anomalies.append({
                    "type": "拓扑错误",
                    "location": "system",
                    "confidence": 0.85,
                    "layer": "state_estimator",
                    "details": (f"chi2统计量={bd['chi2_statistic']:.2f} > "
                                f"临界值={bd['chi2_critical']:.2f}, "
                                f"{bd.get('bad_measurement_ratio', 0):.0%}量测异常 "
                                f"-> 系统性拓扑错误"),
                })
            else:
                anomalies.append({
                    "type": "chi2检验未通过",
                    "location": "system",
                    "confidence": 0.90,
                    "layer": "state_estimator",
                    "details": (f"chi2统计量={bd['chi2_statistic']:.2f} > "
                                f"临界值={bd['chi2_critical']:.2f}"),
                })

        for idx in bd["bad_data_indices"]:
            nr = bd["normalized_residuals"].get(idx, 0)
            if idx in net.measurement.index:
                mt = net.measurement.at[idx, "measurement_type"]
                el = net.measurement.at[idx, "element"]
                et = net.measurement.at[idx, "element_type"]
                anomalies.append({
                    "type": "不良数据",
                    "location": f"{et}_{el}",
                    "confidence": min(0.60 + nr * 0.1, 0.99),
                    "layer": "state_estimator",
                    "details": (f"量测{idx}(类型={mt}, 元件={et}_{el}) "
                                f"标准化残差={nr:.2f}"),
                })

        # 拓扑错误辨识
        for te in identify_topology_errors(net, bd["bad_data_indices"]):
            if "switch_idx" in te:
                anomalies.append({
                    "type": "拓扑错误",
                    "location": f"switch_{te['switch_idx']}",
                    "confidence": te.get("confidence", 0.75),
                    "layer": "state_estimator",
                    "details": (f"开关{te['switch_idx']}状态可能有误: "
                                f"当前={te['current_state']}, "
                                f"建议={te['suggested_state']}"),
                })
            else:
                anomalies.append({
                    "type": "拓扑错误",
                    "location": te.get("line_idx", "?"),
                    "confidence": te.get("confidence", 0.60),
                    "layer": "state_estimator",
                    "details": te.get("details", ""),
                })

        return anomalies

    # ----------------------------------------------------------
    @staticmethod
    def _merge_anomalies(anomalies: List[Dict]) -> List[Dict]:
        """
        合并去重:
          - 同一 (type, location) 保留置信度最高
          - 多层同时命中 → 提升置信度
        """
        groups: Dict[tuple, List[Dict]] = {}
        for a in anomalies:
            key = (a.get("type", ""), str(a.get("location", "")))
            groups.setdefault(key, []).append(a)

        merged: List[Dict] = []
        for group in groups.values():
            if len(group) == 1:
                merged.append(group[0])
            else:
                best  = max(group, key=lambda x: x.get("confidence", 0))
                layers = list({a.get("layer", "") for a in group})
                best["confidence"]          = min(best.get("confidence", 0.5) * 1.1, 0.99)
                best["detected_by_layers"]  = layers
                best["details"] = (f"[多层确认: {','.join(layers)}] "
                                   f"{best.get('details', '')}")
                merged.append(best)

        merged.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        return merged

    def get_summary(self) -> Dict:
        """获取检测摘要"""
        type_counts  = Counter(a.get("type", "未知") for a in self.merged_results)
        layer_counts = Counter(a.get("layer", "未知") for a in self.merged_results)
        return {
            "total_anomalies":       len(self.merged_results),
            "by_type":               dict(type_counts),
            "by_layer":              dict(layer_counts),
            "rule_engine_count":     len(self.rule_results),
            "state_estimator_count": len(self.se_results),
        }
