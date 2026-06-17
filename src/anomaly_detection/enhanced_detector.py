# -*- coding: utf-8 -*-
"""
增强检测器 v3.0
集成: 规则引擎 + 状态估计(含鲁棒SE) + GNN + 分层检测 + 可解释性
"""
import copy
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class EnhancedDetector:
    """
    增强检测器 v3.0: 四层混合检测 + 鲁棒SE + 分层策略 + 可解释性

    使用方式:
        detector = EnhancedDetector(enable_gnn=True, enable_robust_se=True)
        result = detector.detect(network_data)
        explanation = detector.explain(result)
        corrections = detector.correct(network_data, result["anomalies"])
    """

    def __init__(self, enable_rule=True, enable_se=True, enable_gnn=False,
                 gnn_model_path=None, enable_robust_se=False,
                 enable_hierarchical=True, enable_explainer=True,
                 se_method="wls", max_partition_size=50):
        self.enable_rule = enable_rule
        self.enable_se = enable_se
        self.enable_gnn = enable_gnn
        self.enable_robust_se = enable_robust_se
        self.enable_hierarchical = enable_hierarchical
        self.enable_explainer = enable_explainer
        self.se_method = se_method  # "wls", "lav", "irls"
        self.max_partition_size = max_partition_size

        # 延迟加载模块
        self._detector = None
        self._localizer = None
        self._gnn = None
        self._verifier = None
        self._explainer = None
        self._hierarchical = None

        if enable_gnn:
            try:
                from anomaly_detection.gnn_detector import GNNDetector
                self._gnn = GNNDetector(model_path=gnn_model_path)
            except Exception as e:
                logger.warning("GNN初始化失败: {}, 回退到规则+SE".format(e))
                self.enable_gnn = False

        if enable_explainer:
            try:
                from anomaly_detection.explainer import TopologyExplainer
                self._explainer = TopologyExplainer()
            except Exception as e:
                logger.warning("解释器初始化失败: {}".format(e))
                self.enable_explainer = False

        if enable_hierarchical:
            try:
                from anomaly_detection.hierarchical_detector import HierarchicalDetector
                self._hierarchical = HierarchicalDetector(max_partition_size=max_partition_size)
            except Exception as e:
                logger.warning("分层检测器初始化失败: {}".format(e))
                self.enable_hierarchical = False

    def detect(self, network_data: Dict) -> Dict:
        """执行完整检测流程"""
        import time
        start_time = time.time()

        graph = network_data.get("graph")
        use_hierarchical = (
            self.enable_hierarchical
            and graph is not None
            and graph.number_of_nodes() > self.max_partition_size
        )

        # 1. 核心检测(规则+SE+GNN)
        if use_hierarchical:
            anomalies = self._detect_hierarchical(network_data)
        else:
            anomalies = self._detect_standard(network_data)

        # 2. 鲁棒SE增强(可选)
        robust_anomalies = []
        if self.enable_robust_se and network_data.get("net") is not None:
            robust_anomalies = self._run_robust_se(network_data)
            anomalies.extend(robust_anomalies)

        # 3. 设备级定位
        from anomaly_detection.device_localizer import DeviceLocalizer
        if self._localizer is None:
            self._localizer = DeviceLocalizer()
        localized = self._localizer.localize(anomalies, network_data)

        # 4. 去重合并
        merged = self._merge_detections(localized)

        elapsed = time.time() - start_time

        # 5. 构建摘要
        from collections import Counter
        type_counts = Counter(a.get("type", "?") for a in merged)
        layer_counts = Counter(a.get("layer", "?") for a in merged)

        result = {
            "anomalies": merged,
            "summary": {
                "total": len(merged),
                "by_type": dict(type_counts),
                "by_layer": dict(layer_counts),
                "processing_time_ms": round(elapsed * 1000, 1),
                "gnn_enabled": self.enable_gnn,
                "robust_se_enabled": self.enable_robust_se,
                "hierarchical_used": use_hierarchical,
                "se_method": self.se_method,
            },
            "layer_results": {
                "rule_engine": self._detector.rule_results if self._detector else [],
                "state_estimator": self._detector.se_results if self._detector else [],
                "gnn": [],
                "robust_se": robust_anomalies,
            },
            "network_data": network_data,  # 保留引用供解释器使用
        }

        logger.info("增强检测v3完成: {} 个异常, 耗时 {:.1f}ms".format(
            len(merged), elapsed * 1000))
        return result

    def explain(self, result: Dict) -> Dict:
        """为检测结果生成可解释性报告"""
        if not self.enable_explainer or self._explainer is None:
            return {"status": "explainer_disabled"}
        try:
            anomalies = result.get("anomalies", [])
            network_data = result.get("network_data", {})
            report = self._explainer.generate_explanation_report(anomalies, network_data)
            return report
        except Exception as e:
            logger.error("解释生成失败: {}".format(e))
            return {"status": "error", "message": str(e)}

    def correct(self, network_data: Dict, anomalies: List[Dict]) -> Dict:
        """执行自动修正 + 电气约束验证"""
        from correction_engine.corrector import generate_corrections
        from correction_engine.electrical_verifier import ElectricalVerifier

        corrections = generate_corrections(anomalies, network_data)
        verifier = ElectricalVerifier()
        net = network_data.get("net")

        if net is not None:
            import pandapower as pp
            for corr_item in corrections.get("corrections", []):
                try:
                    corrected_net = copy.deepcopy(net)
                    pp.runpp(corrected_net)
                    verification = verifier.verify_correction(net, corrected_net, corr_item)
                    corr_item["verification"] = verification
                except Exception as e:
                    corr_item["verification"] = {"passed": False, "error": str(e)[:50]}

        corrections["verification_summary"] = verifier.get_summary()
        return corrections

    def _detect_standard(self, network_data: Dict) -> List[Dict]:
        """标准三层检测"""
        from anomaly_detection.detector import AnomalyDetector
        if self._detector is None:
            self._detector = AnomalyDetector(
                use_rule_engine=self.enable_rule,
                use_state_estimator=self.enable_se)
        anomalies = self._detector.detect_all(network_data)

        if self.enable_gnn and self._gnn:
            gnn_anomalies = self._gnn.detect(network_data)
            anomalies.extend(gnn_anomalies)
        return anomalies

    def _detect_hierarchical(self, network_data: Dict) -> List[Dict]:
        """分层检测(大网络)"""
        graph = network_data.get("graph")
        if graph is None:
            return self._detect_standard(network_data)

        def base_detector(sub_network_data):
            return self._detect_standard(sub_network_data)

        anomalies = self._hierarchical.detect(graph, base_detector, network_data)
        logger.info("分层检测完成: {} 个分区, {} 个异常".format(
            self._hierarchical.get_partition_info().get("n_partitions", 0),
            len(anomalies)))
        return anomalies

    def _run_robust_se(self, network_data: Dict) -> List[Dict]:
        """运行鲁棒状态估计"""
        anomalies = []
        try:
            from anomaly_detection.robust_se import run_robust_estimation
            net = network_data.get("net")
            scada_data = network_data.get("scada_data", {})
            success, result = run_robust_estimation(net, scada_data, method=self.se_method)
            if success:
                for bd in result.get("bad_data", []):
                    anomalies.append({
                        "type": "telemetry_mismatch",
                        "layer": "robust_se_{}".format(self.se_method),
                        "confidence": bd.get("confidence", 0.8),
                        "location": bd.get("location", "unknown"),
                        "details": "鲁棒SE({}): {}".format(self.se_method, bd.get("detail", "")),
                    })
        except Exception as e:
            logger.warning("鲁棒SE失败: {}".format(e))
        return anomalies

    @staticmethod
    def _merge_detections(anomalies: List[Dict]) -> List[Dict]:
        """合并去重: 同一(type, location)保留置信度最高"""
        groups = {}
        for a in anomalies:
            key = (a.get("type", ""), str(a.get("device_name", a.get("location", ""))))
            groups.setdefault(key, []).append(a)

        merged = []
        for group in groups.values():
            if len(group) == 1:
                merged.append(group[0])
            else:
                best = max(group, key=lambda x: x.get("confidence", 0))
                layers = list({a.get("layer", "") for a in group})
                best["detected_by_layers"] = layers
                best["confidence"] = min(best.get("confidence", 0.5) * 1.1, 0.99)
                merged.append(best)

        merged.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        return merged


def run_enhanced_detection(network_data: Dict, enable_gnn: bool = False,
                           enable_robust_se: bool = False,
                           enable_hierarchical: bool = True,
                           se_method: str = "wls") -> Dict:
    """一键运行增强检测(便捷函数)"""
    detector = EnhancedDetector(
        enable_gnn=enable_gnn,
        enable_robust_se=enable_robust_se,
        enable_hierarchical=enable_hierarchical,
        se_method=se_method)
    return detector.detect(network_data)
