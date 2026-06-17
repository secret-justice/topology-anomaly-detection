# -*- coding: utf-8 -*-
"""
增强评估指标模块
提供定位精度、修正成功率、误修率、处理时间、严重程度分布、
类型覆盖率、完整报告生成及批量benchmark评估。
"""
import time
import statistics
from typing import Dict, List, Optional, Any
from collections import Counter
import logging

logger = logging.getLogger(__name__)

# 异常类型标准化（复用 metrics.py 的映射表）
from utils.metrics import _normalize_type, ANOMALY_TYPE_MAP, match_anomalies

# 5类标准异常类型（竞赛定义）
STANDARD_ANOMALY_TYPES = [
    "图模不符",
    "拓扑中断",
    "虚接/错接",
    "遥测!=拓扑",
    "遥信!=遥测",
]


# ============================================================
# 指标1: 定位精度
# ============================================================
def calculate_location_accuracy(predictions: List[Dict],
                                ground_truth: List[Dict],
                                tolerance: float = 1.0) -> Dict:
    """
    定位精度：在容差范围内正确匹配的位置数 / 检测到的异常总数。

    Args:
        predictions:  检测到的异常列表 [{type, location, ...}]
        ground_truth: 真实异常标签   [{type, location, ...}]
        tolerance:    位置匹配容差（数值型位置使用绝对差，字符串型使用精确匹配）

    Returns:
        {
            accuracy: float,          # 定位精度
            matched: int,             # 正确匹配数
            total_predicted: int,     # 检测总数
            total_truth: int,         # 真值总数
            match_details: list,      # 匹配详情 [(pred, gt), ...]
        }
    """
    if not predictions:
        return {
            "accuracy": 0.0,
            "matched": 0,
            "total_predicted": 0,
            "total_truth": len(ground_truth),
            "match_details": [],
        }

    matches, fps, fns = match_anomalies(predictions, ground_truth, tolerance)

    accuracy = len(matches) / len(predictions) if predictions else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "matched": len(matches),
        "total_predicted": len(predictions),
        "total_truth": len(ground_truth),
        "false_positives": len(fps),
        "false_negatives": len(fns),
        "match_details": matches,
    }


# ============================================================
# 指标2: 修正成功率
# ============================================================
def calculate_correction_rate(corrections: List[Dict],
                              ground_truth: List[Dict]) -> Dict:
    """
    修正成功率：成功修正的异常数 / 检测到的异常总数。

    "成功修正"定义：修正方案的目标(target)与真实异常的位置匹配。
    如果 corrections 包含 verification 字段，也参考验证结果。

    Args:
        corrections:  修正方案列表 [{target, anomaly_type, verification?, ...}]
        ground_truth: 真实异常列表 [{type, location, ...}]

    Returns:
        {
            correction_rate: float,
            successful: int,
            total_corrections: int,
            verified_pass: int,       # 电气验证通过数
            verified_fail: int,       # 电气验证失败数
            detail: list,
        }
    """
    if not corrections:
        return {
            "correction_rate": 0.0,
            "successful": 0,
            "total_corrections": 0,
            "verified_pass": 0,
            "verified_fail": 0,
            "detail": [],
        }

    # 构建真值位置索引（标准化后）
    gt_locations = set()
    for g in ground_truth:
        loc = str(g.get("location", ""))
        gt_locations.add(loc)

    successful = 0
    verified_pass = 0
    verified_fail = 0
    detail = []

    for corr in corrections:
        target = str(corr.get("target", ""))
        matched = target in gt_locations

        # 检查电气验证结果
        verification = corr.get("verification", {})
        v_passed = verification.get("passed", None)

        if v_passed is True:
            verified_pass += 1
        elif v_passed is False:
            verified_fail += 1

        if matched:
            successful += 1

        detail.append({
            "target": target,
            "matched_to_gt": matched,
            "verification_passed": v_passed,
            "anomaly_type": corr.get("anomaly_type", ""),
        })

    correction_rate = successful / len(corrections) if corrections else 0.0

    return {
        "correction_rate": round(correction_rate, 4),
        "successful": successful,
        "total_corrections": len(corrections),
        "verified_pass": verified_pass,
        "verified_fail": verified_fail,
        "detail": detail,
    }


# ============================================================
# 指标3: 误修率
# ============================================================
def calculate_false_correction_rate(corrections: List[Dict],
                                    ground_truth: List[Dict]) -> Dict:
    """
    误修率：错误修正数 / 总修正数。

    "错误修正"：修正目标不在真实异常位置中（即修正了本来没有问题的地方）。

    Args:
        corrections:  修正方案列表
        ground_truth: 真实异常列表

    Returns:
        {
            false_correction_rate: float,
            false_corrections: int,
            total_corrections: int,
            false_targets: list,      # 误修的目标列表
        }
    """
    if not corrections:
        return {
            "false_correction_rate": 0.0,
            "false_corrections": 0,
            "total_corrections": 0,
            "false_targets": [],
        }

    gt_locations = set()
    for g in ground_truth:
        gt_locations.add(str(g.get("location", "")))

    false_corrections = 0
    false_targets = []

    for corr in corrections:
        target = str(corr.get("target", ""))
        if target not in gt_locations:
            false_corrections += 1
            false_targets.append(target)

    fcr = false_corrections / len(corrections) if corrections else 0.0

    return {
        "false_correction_rate": round(fcr, 4),
        "false_corrections": false_corrections,
        "total_corrections": len(corrections),
        "false_targets": false_targets,
    }


# ============================================================
# 指标4: 处理时间统计
# ============================================================
def calculate_processing_time(result: Dict) -> Dict:
    """
    处理时间统计：从检测结果中提取各层耗时并汇总。

    支持两种输入格式：
    1. EnhancedDetector 输出: result["summary"]["processing_time_ms"]
    2. 自定义 timing dict:   result["timing"] = {layer: seconds, ...}

    Args:
        result: 检测/修正结果字典

    Returns:
        {
            total_ms: float,
            total_s: float,
            per_layer: dict,          # 各层耗时
            classification: str,      # "realtime"<100ms, "fast"<1s, "acceptable"<10s, "slow"
        }
    """
    total_ms = 0.0
    per_layer = {}

    # 格式1: summary.processing_time_ms
    summary = result.get("summary", {})
    if "processing_time_ms" in summary:
        total_ms = summary["processing_time_ms"]

    # 格式2: timing 字典
    timing = result.get("timing", {})
    if timing:
        for layer, t in timing.items():
            if isinstance(t, (int, float)):
                ms = t * 1000 if t < 100 else t  # 假设 <100 的是秒
                per_layer[layer] = round(ms, 2)
        if total_ms == 0:
            total_ms = sum(per_layer.values())

    # 格式3: 顶层 processing_time_ms
    if total_ms == 0 and "processing_time_ms" in result:
        total_ms = result["processing_time_ms"]

    # 分类
    if total_ms < 100:
        classification = "realtime"
    elif total_ms < 1000:
        classification = "fast"
    elif total_ms < 10000:
        classification = "acceptable"
    else:
        classification = "slow"

    return {
        "total_ms": round(total_ms, 2),
        "total_s": round(total_ms / 1000, 3),
        "per_layer": per_layer,
        "classification": classification,
    }


# ============================================================
# 指标5: 异常严重程度分布
# ============================================================
def calculate_severity_distribution(anomalies: List[Dict]) -> Dict:
    """
    异常严重程度分布：按置信度分桶统计。

    分桶规则（基于 confidence 字段）:
      - critical:   >= 0.90
      - high:       >= 0.75
      - medium:     >= 0.50
      - low:        <  0.50

    Args:
        anomalies: 异常列表 [{confidence, type, ...}]

    Returns:
        {
            distribution: {level: count},
            percentages: {level: float},
            total: int,
            avg_confidence: float,
            max_confidence: float,
            min_confidence: float,
            by_type: {type: {level: count}},
        }
    """
    if not anomalies:
        return {
            "distribution": {"critical": 0, "high": 0, "medium": 0, "low": 0},
            "percentages": {"critical": 0.0, "high": 0.0, "medium": 0.0, "low": 0.0},
            "total": 0,
            "avg_confidence": 0.0,
            "max_confidence": 0.0,
            "min_confidence": 0.0,
            "by_type": {},
        }

    def _severity_level(conf):
        if conf >= 0.90:
            return "critical"
        elif conf >= 0.75:
            return "high"
        elif conf >= 0.50:
            return "medium"
        else:
            return "low"

    distribution = Counter()
    by_type = {}
    confidences = []

    for a in anomalies:
        conf = a.get("confidence", 0.0)
        confidences.append(conf)
        level = _severity_level(conf)
        distribution[level] += 1

        atype = _normalize_type(a.get("type", "未知"))
        if atype not in by_type:
            by_type[atype] = Counter()
        by_type[atype][level] += 1

    total = len(anomalies)
    percentages = {k: round(v / total, 4) for k, v in distribution.items()}

    # 确保所有级别都有值
    for level in ["critical", "high", "medium", "low"]:
        distribution.setdefault(level, 0)
        percentages.setdefault(level, 0.0)

    return {
        "distribution": dict(distribution),
        "percentages": percentages,
        "total": total,
        "avg_confidence": round(statistics.mean(confidences), 4),
        "max_confidence": round(max(confidences), 4),
        "min_confidence": round(min(confidences), 4),
        "by_type": {k: dict(v) for k, v in by_type.items()},
    }


# ============================================================
# 指标6: 类型覆盖率
# ============================================================
def calculate_type_coverage(detected_types: List[str]) -> Dict:
    """
    类型覆盖率：检测到的异常类型数 / 5（标准异常类型总数）。

    Args:
        detected_types: 检测到的异常类型名称列表（原始名称）

    Returns:
        {
            coverage: float,          # 覆盖率 (0.0 ~ 1.0)
            detected_count: int,      # 覆盖的类型数
            total_types: int,         # 总类型数 (=5)
            detected: list,           # 覆盖的标准化类型名
            missing: list,            # 未覆盖的类型名
        }
    """
    # 标准化检测到的类型
    normalized = set()
    for t in detected_types:
        nt = _normalize_type(t)
        normalized.add(nt)

    # 计算覆盖
    covered = set()
    for std_type in STANDARD_ANOMALY_TYPES:
        if std_type in normalized:
            covered.add(std_type)

    coverage = len(covered) / len(STANDARD_ANOMALY_TYPES)
    missing = [t for t in STANDARD_ANOMALY_TYPES if t not in covered]

    return {
        "coverage": round(coverage, 4),
        "detected_count": len(covered),
        "total_types": len(STANDARD_ANOMALY_TYPES),
        "detected": sorted(covered),
        "missing": missing,
    }


# ============================================================
# 系统级评估摘要（保留原有接口）
# ============================================================
def evaluate_system_summary(predictions: List[Dict],
                            ground_truth: List[Dict]) -> Dict:
    """
    System-level evaluation summary for competition scoring.

    Evaluates at TYPE level (not location):
    - Did we detect each type of anomaly that was injected?
    - How many detections per type?

    Returns:
        {
            type_recall, type_precision,
            n_injected_types, n_detected_types, n_matched_types,
            per_type_detail, detection_density, system_health
        }
    """
    # Count injected types
    injected_types = Counter()
    for g in ground_truth:
        nt = _normalize_type(g.get("type", ""))
        injected_types[nt] += 1

    # Count detected types
    detected_types = Counter()
    for p in predictions:
        nt = _normalize_type(p.get("type", ""))
        detected_types[nt] += 1

    # Match types
    injected_set = set(injected_types.keys())
    detected_set = set(detected_types.keys())
    matched = injected_set & detected_set
    missed = injected_set - detected_set
    extra = detected_set - injected_set

    type_recall = len(matched) / len(injected_set) if injected_set else 0
    type_precision = len(matched) / len(detected_set) if detected_set else 0

    # Per-type detail
    per_type = {}
    for t in injected_set:
        per_type[t] = {
            "injected": True,
            "detected": t in detected_set,
            "injected_count": injected_types[t],
            "detected_count": detected_types.get(t, 0),
        }
    for t in extra:
        per_type[t] = {
            "injected": False,
            "detected": True,
            "injected_count": 0,
            "detected_count": detected_types[t],
            "note": "extra_detection"
        }

    # System health
    if type_recall >= 1.0 and type_precision >= 0.8:
        health = "excellent"
    elif type_recall >= 0.8:
        health = "good"
    elif type_recall >= 0.5:
        health = "partial"
    else:
        health = "poor"

    return {
        "type_recall": type_recall,
        "type_precision": type_precision,
        "n_injected_types": len(injected_set),
        "n_detected_types": len(detected_set),
        "n_matched_types": len(matched),
        "n_missed_types": len(missed),
        "missed_types": sorted(missed),
        "extra_types": sorted(extra),
        "per_type_detail": per_type,
        "detection_density": len(predictions) / max(len(ground_truth), 1),
        "system_health": health,
    }


# ============================================================
# 完整评估报告
# ============================================================
def generate_full_report(predictions: List[Dict],
                         ground_truth: List[Dict],
                         corrections: List[Dict],
                         timing: Dict) -> Dict:
    """
    生成完整评估报告，包含所有指标。

    Args:
        predictions:  检测到的异常列表
        ground_truth: 真实异常标签
        corrections:  修正方案列表
        timing:       处理时间 {"processing_time_ms": float} 或各层时间

    Returns:
        完整报告字典，包含:
          - location_accuracy: 定位精度
          - detection_metrics: P/R/F1（复用 metrics.py）
          - correction_rate: 修正成功率
          - false_correction_rate: 误修率
          - processing_time: 处理时间
          - severity_distribution: 严重程度分布
          - type_coverage: 类型覆盖率
          - system_summary: 系统级摘要
          - overall_score: 综合评分
    """
    report = {}

    # 1. 定位精度
    report["location_accuracy"] = calculate_location_accuracy(
        predictions, ground_truth)

    # 2. 检测 P/R/F1
    from utils.metrics import evaluate_detection_results, evaluate_by_type
    report["detection_metrics"] = evaluate_detection_results(
        predictions, ground_truth)
    report["per_type_metrics"] = evaluate_by_type(predictions, ground_truth)

    # 3. 修正成功率
    report["correction_rate"] = calculate_correction_rate(
        corrections, ground_truth)

    # 4. 误修率
    report["false_correction_rate"] = calculate_false_correction_rate(
        corrections, ground_truth)

    # 5. 处理时间
    result_for_timing = {"timing": timing} if timing else {}
    if "summary" in timing:
        result_for_timing["summary"] = timing["summary"]
    if "processing_time_ms" in timing:
        result_for_timing["processing_time_ms"] = timing["processing_time_ms"]
    report["processing_time"] = calculate_processing_time(result_for_timing)

    # 6. 严重程度分布
    report["severity_distribution"] = calculate_severity_distribution(
        predictions)

    # 7. 类型覆盖率
    detected_types = [p.get("type", "") for p in predictions]
    report["type_coverage"] = calculate_type_coverage(detected_types)

    # 8. 系统级摘要
    report["system_summary"] = evaluate_system_summary(
        predictions, ground_truth)

    # 9. 综合评分（加权）
    report["overall_score"] = _compute_overall_score(report)

    logger.info("完整评估报告生成: 综合评分={:.2f}".format(
        report["overall_score"]["score"]))
    return report


def _compute_overall_score(report: Dict) -> Dict:
    """
    计算综合评分（0~100分）。

    权重分配:
      - F1 分数:          30%
      - 定位精度:          20%
      - 修正成功率:        20%
      - 误修率(反向):      10%  (越低越好)
      - 类型覆盖率:        10%
      - 处理速度:          10%  (realtime=满分, slow=0分)

    Returns:
        {score: float, breakdown: {component: (weight, raw, weighted)}, grade: str}
    """
    weights = {
        "f1":               0.30,
        "location_accuracy": 0.20,
        "correction_rate":   0.20,
        "no_false_corr":     0.10,
        "type_coverage":     0.10,
        "speed":             0.10,
    }

    # 各项原始分
    f1 = report.get("detection_metrics", {}).get("f1", 0.0)
    loc_acc = report.get("location_accuracy", {}).get("accuracy", 0.0)
    corr_rate = report.get("correction_rate", {}).get("correction_rate", 0.0)
    fcr = report.get("false_correction_rate", {}).get("false_correction_rate", 1.0)
    no_false = 1.0 - fcr  # 反向：误修率越低越好
    type_cov = report.get("type_coverage", {}).get("coverage", 0.0)

    # 速度评分
    pt = report.get("processing_time", {}).get("total_ms", 10000)
    if pt < 100:
        speed = 1.0
    elif pt < 1000:
        speed = 0.8
    elif pt < 10000:
        speed = 0.5
    else:
        speed = 0.2

    raw_scores = {
        "f1":               f1,
        "location_accuracy": loc_acc,
        "correction_rate":   corr_rate,
        "no_false_corr":     max(0.0, no_false),
        "type_coverage":     type_cov,
        "speed":             speed,
    }

    breakdown = {}
    total = 0.0
    for key, weight in weights.items():
        raw = raw_scores.get(key, 0.0)
        weighted = raw * weight
        breakdown[key] = {
            "weight": weight,
            "raw": round(raw, 4),
            "weighted": round(weighted, 4),
        }
        total += weighted

    score = round(total * 100, 2)

    # 等级
    if score >= 90:
        grade = "A"
    elif score >= 80:
        grade = "B"
    elif score >= 70:
        grade = "C"
    elif score >= 60:
        grade = "D"
    else:
        grade = "F"

    return {
        "score": score,
        "grade": grade,
        "breakdown": breakdown,
    }


# ============================================================
# 批量评估 Benchmark 套件
# ============================================================
def evaluate_benchmark_suite(results: List[Dict]) -> Dict:
    """
    批量评估 benchmark 结果，生成汇总表。

    Args:
        results: 每个 benchmark 的结果列表，每项包含:
            {
                "name": str,
                "predictions": [...],
                "ground_truth": [...],
                "corrections": [...],
                "timing": {...},
            }

    Returns:
        {
            summary_table: list,
            aggregate: {avg_f1, avg_accuracy, ...},
            ranking: list,
        }
    """
    summary_table = []
    rankings = []

    for bench in results:
        name = bench.get("name", "unknown")
        predictions = bench.get("predictions", [])
        ground_truth = bench.get("ground_truth", [])
        corrections = bench.get("corrections", [])
        timing = bench.get("timing", {})

        # 逐项评估
        det = _quick_detection_metrics(predictions, ground_truth)
        loc = calculate_location_accuracy(predictions, ground_truth)
        corr = calculate_correction_rate(corrections, ground_truth)
        fcr = calculate_false_correction_rate(corrections, ground_truth)
        pt = calculate_processing_time({"timing": timing})
        tc = calculate_type_coverage([p.get("type", "") for p in predictions])

        row = {
            "name": name,
            "precision": det["precision"],
            "recall": det["recall"],
            "f1": det["f1"],
            "location_accuracy": loc["accuracy"],
            "correction_rate": corr["correction_rate"],
            "false_correction_rate": fcr["false_correction_rate"],
            "type_coverage": tc["coverage"],
            "processing_ms": pt["total_ms"],
            "n_predictions": len(predictions),
            "n_ground_truth": len(ground_truth),
            "n_corrections": len(corrections),
        }
        summary_table.append(row)

        # 综合评分
        overall = _compute_overall_score({
            "detection_metrics": det,
            "location_accuracy": loc,
            "correction_rate": corr,
            "false_correction_rate": fcr,
            "type_coverage": tc,
            "processing_time": pt,
        })
        rankings.append({"name": name, "score": overall["score"],
                         "grade": overall["grade"]})

    # 聚合统计
    n = len(summary_table)
    if n == 0:
        return {"summary_table": [], "aggregate": {}, "ranking": []}

    def _avg(key):
        vals = [r[key] for r in summary_table if key in r]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    passed = sum(1 for r in summary_table if r["f1"] >= 0.7)

    aggregate = {
        "avg_f1": _avg("f1"),
        "avg_precision": _avg("precision"),
        "avg_recall": _avg("recall"),
        "avg_accuracy": _avg("location_accuracy"),
        "avg_correction_rate": _avg("correction_rate"),
        "avg_false_correction_rate": _avg("false_correction_rate"),
        "avg_type_coverage": _avg("type_coverage"),
        "avg_processing_ms": _avg("processing_ms"),
        "total_tests": n,
        "passed_tests": passed,
        "pass_rate": round(passed / n, 4) if n else 0.0,
    }

    # 排序
    rankings.sort(key=lambda x: x["score"], reverse=True)

    logger.info("Benchmark套件评估: {} 个测试, 平均F1={:.3f}, 通过率={:.0%}".format(
        n, aggregate["avg_f1"], aggregate["pass_rate"]))

    return {
        "summary_table": summary_table,
        "aggregate": aggregate,
        "ranking": rankings,
    }


def _quick_detection_metrics(predictions: List[Dict],
                             ground_truth: List[Dict]) -> Dict:
    """快速计算检测 P/R/F1（类型级别，与 metrics.py 逻辑一致）"""
    pred_types = {_normalize_type(p.get("type", "")) for p in predictions}
    gt_types = {_normalize_type(g.get("type", "")) for g in ground_truth}

    tp = len(pred_types & gt_types)
    fp = len(pred_types - gt_types)
    fn = len(gt_types - pred_types)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }