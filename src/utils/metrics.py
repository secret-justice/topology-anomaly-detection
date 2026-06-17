# -*- coding: utf-8 -*-
"""
评估指标模块
计算检测准确率、召回率、F1分数等
"""
from typing import Dict, List, Tuple, Set
import logging

logger = logging.getLogger(__name__)



# ============================================================
# 异常类型映射表（检测结果 -> 标准类型）
# 解决检测器输出类型名与ground truth标签不一致的问题
# ============================================================
ANOMALY_TYPE_MAP = {
    # 检测器输出 -> 标准类型
    "不良数据": "遥测!=拓扑",
    "拓扑错误": "拓扑中断",
    "chi2检验未通过": "遥测!=拓扑",
    "虚接/错接": "虚接/错接",
    "遥测!=拓扑": "遥测!=拓扑",
    "遥信!=遥测": "遥信!=遥测",
    "拓扑中断": "拓扑中断",
    "图模不符": "图模不符",
}

def _normalize_type(t: str) -> str:
    """将异常类型标准化"""
    return ANOMALY_TYPE_MAP.get(t, t)

def calculate_precision(tp: int, fp: int) -> float:
    """准确率 Precision = TP / (TP + FP)"""
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def calculate_recall(tp: int, fn: int) -> float:
    """召回率 Recall = TP / (TP + FN)"""
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def calculate_f1(precision: float, recall: float) -> float:
    """F1分数 = 2 * P * R / (P + R)"""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_detection_results(predictions: List[Dict],
                               ground_truth: List[Dict],
                               match_field: str = "location") -> Dict:
    """
    综合评估检测结果

    Args:
        predictions:  检测到的异常列表 [{type, location, ...}]
        ground_truth: 真实异常标签   [{type, location, ...}]
        match_field:  用于匹配的字段名

    Returns:
        评估结果字典 {precision, recall, f1, tp, fp, fn,
                     true_positives, false_positives, false_negatives,
                     n_predictions, n_ground_truth}
    """
    pred_set: Set[tuple] = set()
    for p in predictions:
        ntype = _normalize_type(p.get("type", ""))
        pred_set.add((ntype, str(p.get(match_field, ""))))

    gt_set: Set[tuple] = set()
    for g in ground_truth:
        ntype = _normalize_type(g.get("type", ""))
        gt_set.add((ntype, str(g.get(match_field, ""))))

    tp_set = pred_set & gt_set
    fp_set = pred_set - gt_set
    fn_set = gt_set - pred_set

    tp, fp, fn = len(tp_set), len(fp_set), len(fn_set)
    precision = calculate_precision(tp, fp)
    recall    = calculate_recall(tp, fn)
    f1        = calculate_f1(precision, recall)

    # Also compute type-level matching (ignore location)
    pred_types_only = {_normalize_type(p.get("type", "")) for p in predictions}
    gt_types_only = {_normalize_type(g.get("type", "")) for g in ground_truth}
    type_tp = len(pred_types_only & gt_types_only)
    type_fp = len(pred_types_only - gt_types_only)
    type_fn = len(gt_types_only - pred_types_only)

    # Use type-level metrics as primary (location format varies)
    type_precision = type_tp / (type_tp + type_fp) if (type_tp + type_fp) > 0 else 0.0
    type_recall = type_tp / (type_tp + type_fn) if (type_tp + type_fn) > 0 else 0.0
    type_f1 = 2 * type_precision * type_recall / (type_precision + type_recall) if (type_precision + type_recall) > 0 else 0.0

    result = {
        "precision": type_precision,
        "recall": type_recall,
        "f1": type_f1,
        "tp": type_tp,
        "fp": type_fp,
        "fn": type_fn,
        "true_positives":  sorted(pred_types_only & gt_types_only),
        "false_positives": sorted(pred_types_only - gt_types_only),
        "false_negatives": sorted(gt_types_only - pred_types_only),
        "location_precision": precision,
        "location_recall": recall,
        "location_tp": tp,
        "n_predictions":   len(predictions),
        "n_ground_truth":  len(ground_truth),
    }

    logger.info("Type-level: P={:.3f}, R={:.3f}, F1={:.3f} (TP={}, FP={}, FN={})".format(
        type_precision, type_recall, type_f1, type_tp, type_fp, type_fn))
    logger.info("Location-level: P={:.3f}, R={:.3f} (TP={}, FP={}, FN={})".format(
        precision, recall, tp, fp, fn))
    return result





def evaluate_by_type(predictions, ground_truth):
    """
    按异常类型评估（不考虑位置匹配）。
    检查每种注入的异常类型是否被检测到。
    
    Returns:
        dict: {type: {detected: bool, count: int}}
    """
    pred_types = set()
    for p in predictions:
        pred_types.add(_normalize_type(p.get("type", "")))
    
    gt_types = set()
    type_results = {}
    for g in ground_truth:
        gt = _normalize_type(g.get("type", ""))
        gt_types.add(gt)
        type_results[gt] = {
            "injected": True,
            "detected": gt in pred_types,
            "detected_count": sum(1 for p in predictions 
                                  if _normalize_type(p.get("type","")) == gt),
        }
    
    detected_types = gt_types & pred_types
    missed_types = gt_types - pred_types
    extra_types = pred_types - gt_types
    
    return {
        "type_precision": len(detected_types) / len(pred_types) if pred_types else 0,
        "type_recall": len(detected_types) / len(gt_types) if gt_types else 0,
        "detected_types": sorted(detected_types),
        "missed_types": sorted(missed_types),
        "extra_types": sorted(extra_types),
        "per_type": type_results,
    }
def match_anomalies(predicted: List[Dict],
                    ground_truth: List[Dict],
                    tolerance: float = 0.0) -> Tuple[List[Tuple], List[Dict], List[Dict]]:
    """
    匹配预测异常与真实异常（支持容差匹配）

    Args:
        predicted:    预测异常列表
        ground_truth: 真实异常列表
        tolerance:    位置匹配容差（用于数值型位置）

    Returns:
        (matches, false_positives, false_negatives)
    """
    matched_gt: Set[int] = set()
    matches: List[Tuple] = []
    fps: List[Dict] = []

    for pred in predicted:
        found = False
        for i, gt in enumerate(ground_truth):
            if i in matched_gt:
                continue
            if _anomaly_matches(pred, gt, tolerance):
                matches.append((pred, gt))
                matched_gt.add(i)
                found = True
                break
        if not found:
            fps.append(pred)

    fns = [gt for i, gt in enumerate(ground_truth) if i not in matched_gt]
    return matches, fps, fns


def _anomaly_matches(pred: Dict, gt: Dict, tolerance: float) -> bool:
    """判断两个异常是否匹配"""
    if _normalize_type(pred.get("type", "")) != _normalize_type(gt.get("type", "")):
        return False
    pred_loc = pred.get("location")
    gt_loc   = gt.get("location")
    if isinstance(pred_loc, (int, float)) and isinstance(gt_loc, (int, float)):
        return abs(pred_loc - gt_loc) <= tolerance
    return str(pred_loc) == str(gt_loc)
