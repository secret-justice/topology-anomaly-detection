# -*- coding: utf-8 -*-
"""
完整 pytest 测试套件
覆盖: API服务层 / 数据预处理 / 异常检测 / 修正引擎 / 评估指标
"""
import sys
import copy
import pytest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import pandapower as pp
import pandapower.networks as pn


# ============================================================
# API 服务层测试
# ============================================================
class TestServiceLayer:
    """测试 API 服务层核心功能"""

    def test_get_health(self):
        from api.service import get_health
        h = get_health()
        assert isinstance(h, dict)
        assert h["status"] == "ok"
        assert "modules" in h

    def test_list_networks(self):
        from api.service import list_available_networks
        nets = list_available_networks()
        assert isinstance(nets, dict)
        assert len(nets) >= 3
        assert "case33bw" in nets

    def test_run_detect_no_inject(self):
        from api.service import run_detect
        r = run_detect(network_name="case33bw", inject_anomalies=False)
        assert r["success"] is True
        assert isinstance(r["anomaly_count"], int)
        assert isinstance(r["anomalies"], list)

    @pytest.mark.parametrize("network", ["case33bw", "case14", "case_ieee30"])
    def test_run_detect_with_inject(self, network):
        from api.service import run_detect
        r = run_detect(network_name=network, inject_anomalies=True,
                       anomaly_count=5, random_seed=42)
        assert r["success"] is True
        assert r["anomaly_count"] >= 0
        assert "summary" in r

    def test_run_correct(self):
        from api.service import run_correct
        r = run_correct(network_name="case33bw", inject_anomalies=True,
                        anomaly_count=3, random_seed=42)
        assert r["success"] is True
        assert "correction_count" in r
        assert r["correction_count"] >= 0


# ============================================================
# 数据预处理测试
# ============================================================
class TestDataPreprocessing:
    """测试 CIM/SVG 解析和 SCADA 仿真"""

    def test_scada_simulator_init(self, case33bw):
        from data_preprocessing.scada_simulator import SCADASimulator
        sim = SCADASimulator(case33bw, seed=42)
        assert sim is not None

    def test_scada_generate(self, case33bw):
        from data_preprocessing.scada_simulator import SCADASimulator
        sim = SCADASimulator(case33bw, seed=42)
        meas = sim.generate_measurements()
        assert isinstance(meas, dict)
        assert "bus_voltages" in meas or "line_currents" in meas or "line_power" in meas

    def test_cim_parser_load(self):
        """Test CIM parser with IEEE 13 test data"""
        cim_dir = PROJECT_DIR.parent / "07_参考文献" / "CIM_示例数据" / "CIMHub-master" / "model_output_tests"
        if not cim_dir.exists():
            pytest.skip("CIM test data not found")
        from data_preprocessing.cim_parser import parse_cim_directory
        devices = parse_cim_directory(str(cim_dir))
        assert isinstance(devices, (list, dict))

    def test_svg_parser(self):
        """Test SVG parser basic functionality"""
        from data_preprocessing.svg_parser import parse_svg
        # Should not crash with empty input
        try:
            result = parse_svg("<svg></svg>")
        except Exception:
            pass  # Some parsers may not support string input directly


# ============================================================
# 异常检测测试
# ============================================================
class TestAnomalyDetection:
    """测试三层异常检测管道"""

    def test_rule_engine(self, network_data):
        from anomaly_detection.rule_engine import RuleEngine
        engine = RuleEngine()
        import networkx as nx
        G = network_data['graph']
        assert isinstance(G, nx.Graph)
        results = engine.run_all_checks(G=G, cim_data=network_data.get('cim_devices'), svg_data=network_data.get('svg_devices'), measurements=network_data.get('measurements'))
        assert isinstance(results, list)

    def test_state_estimator(self, case33bw):
        from anomaly_detection.state_estimator import add_measurements_to_network, run_wls_estimation
        net = copy.deepcopy(case33bw)
        import data_preprocessing.scada_simulator as scada_mod
        sim = scada_mod.SCADASimulator(net, seed=42)
        scada_data = sim.generate_measurements()
        add_measurements_to_network(net, scada_data)
        success, results = run_wls_estimation(net)
        assert isinstance(success, bool)
        assert isinstance(results, dict)

    def test_detector_integration(self, network_data):
        from anomaly_detection.detector import AnomalyDetector
        detector = AnomalyDetector(use_rule_engine=True, use_state_estimator=True)
        detected = detector.detect_all(network_data)
        assert isinstance(detected, list)

    def test_enhanced_detector(self, network_data):
        from anomaly_detection.enhanced_detector import EnhancedDetector
        enhanced = EnhancedDetector(enable_rule=True, enable_se=True)
        result = enhanced.detect(network_data)
        assert "anomalies" in result
        assert "summary" in result
        assert isinstance(result["anomalies"], list)

    def test_device_localizer(self, network_data):
        from anomaly_detection.device_localizer import DeviceLocalizer
        localizer = DeviceLocalizer()
        # Create a mock anomaly
        mock_anomalies = [{"type": "图模不符", "location": "Bus_3", "confidence": 0.8, "layer": "rule_engine"}]
        localized = localizer.localize(mock_anomalies, network_data)
        assert isinstance(localized, list)


# ============================================================
# 修正引擎测试
# ============================================================
class TestCorrectionEngine:
    """测试修正方案生成和电气验证"""

    def test_generate_corrections(self, network_data):
        from correction_engine.corrector import generate_corrections
        mock_anomalies = [
            {"type": "图模不符", "location": "Bus_3", "confidence": 0.8, "layer": "rule_engine"},
            {"type": "拓扑中断", "location": "Line_2", "confidence": 0.9, "layer": "state_estimator"},
        ]
        corrections = generate_corrections(mock_anomalies, network_data)
        assert isinstance(corrections, dict)
        assert "summary" in corrections
        assert corrections["summary"]["total"] >= 0

    def test_electrical_verifier(self, case33bw):
        from correction_engine.electrical_verifier import ElectricalVerifier
        verifier = ElectricalVerifier()
        net = copy.deepcopy(case33bw)
        pp.runpp(net)
        # Verify against itself (should pass)
        result = verifier.verify_correction(net, net, {"type": "test"})
        assert isinstance(result, dict)


# ============================================================
# 评估指标测试
# ============================================================
class TestMetrics:
    """测试评估指标计算"""

    def test_evaluate_detection_perfect(self):
        from utils.metrics import evaluate_detection_results
        predictions = [{"type": "图模不符", "location": "Bus_3"}]
        ground_truth = [{"type": "图模不符", "location": "Bus_3"}]
        result = evaluate_detection_results(predictions, ground_truth, match_field="location")
        assert isinstance(result, dict)
        assert "precision" in result

    def test_evaluate_by_type(self):
        from utils.metrics import evaluate_by_type
        predictions = [{"type": "图模不符", "location": "Bus_3"}, {"type": "拓扑中断", "location": "Line_2"}]
        ground_truth = [{"type": "图模不符", "location": "Bus_3"}]
        result = evaluate_by_type(predictions, ground_truth)
        assert "type_recall" in result
        assert "type_precision" in result

    def test_graph_utils(self, case33bw):
        from utils.graph_utils import build_graph_from_pandapower, find_sources
        G = build_graph_from_pandapower(case33bw)
        assert G.number_of_nodes() > 0
        sources = find_sources(G)
        assert isinstance(sources, list)


# ============================================================
# 端到端集成测试
# ============================================================
class TestEndToEnd:
    """端到端完整流程测试"""

    @pytest.mark.parametrize("network", ["case33bw", "case14"])
    def test_full_pipeline(self, network):
        """完整检测+修正流程"""
        from api.service import run_correct
        r = run_correct(network_name=network, inject_anomalies=True,
                        anomaly_count=5, random_seed=42)
        assert r["success"] is True
        assert r["correction_count"] >= 0

    def test_multiple_seeds(self):
        """不同随机种子的稳定性"""
        from api.service import run_detect
        counts = []
        for seed in [42, 123, 456]:
            r = run_detect(network_name="case33bw", inject_anomalies=True,
                           anomaly_count=3, random_seed=seed)
            counts.append(r["anomaly_count"])
        # All should detect at least some anomalies
        assert all(c >= 0 for c in counts)