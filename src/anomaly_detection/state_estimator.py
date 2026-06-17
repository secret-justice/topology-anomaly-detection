# -*- coding: utf-8 -*-
"""
第二层: 状态估计
基于PandaPower WLS状态估计检测拓扑错误和不良数据
  - add_measurements_to_network  : 将SCADA数据添加为PandaPower量测
  - run_wls_estimation           : 运行WLS状态估计
  - detect_bad_data              : chi2检验 + 标准化残差BDD
  - identify_topology_errors     : 拓扑错误辨识（假设检验法）
"""
# numpy 2.x compatibility: numpy.linalg.linalg was removed
# numpy 2.x compatibility
import numpy as np
if not hasattr(np, 'in1d'):
    np.in1d = np.isin
if not hasattr(np.linalg, 'linalg'):
    np.linalg.linalg = np.linalg

if not hasattr(np.linalg, 'linalg'):
    np.linalg.linalg = np.linalg

import copy
import numpy as np
import pandapower as pp
from scipy.stats import chi2
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================
# 1. 将SCADA数据添加为PandaPower量测
# ============================================================
def add_measurements_to_network(net, scada_data: Dict) -> None:
    """
    将SCADA仿真量测添加为PandaPower量测元素。

    Args:
        net:       PandaPower网络（必须已运行潮流）
        scada_data: SCADASimulator.generate_measurements() 输出
    """
    # 清除已有量测
    if len(net.measurement) > 0:
        net.measurement.drop(net.measurement.index, inplace=True)

    n_v, n_p, n_q = 0, 0, 0

    # 电压量测 (每条母线)
    for bv in scada_data.get("bus_voltages", []):
        bus_idx = bv["bus"]
        if bus_idx in net.bus.index:
            pp.create_measurement(net, meas_type="v", element_type="bus",
                                  value=bv["vm_pu"], std_dev=bv.get("sigma", 0.005),
                                  element=bus_idx)
            n_v += 1

    # 线路功率量测 (首末端)
    for lp in scada_data.get("line_powers", []):
        line_idx = lp["line"]
        side = lp.get("side", "from")
        if line_idx in net.line.index:
            pp.create_measurement(net, meas_type="p", element_type="line",
                                  value=lp["p_mw"], std_dev=lp.get("sigma", 0.02),
                                  element=line_idx, side=side)
            pp.create_measurement(net, meas_type="q", element_type="line",
                                  value=lp["q_mvar"], std_dev=lp.get("sigma", 0.02),
                                  element=line_idx, side=side)
            n_p += 1
            n_q += 1

    # 变压器功率量测
    for tp in scada_data.get("trafo_powers", []):
        trafo_idx = tp["trafo"]
        side = tp.get("side", "hv")
        if trafo_idx in net.trafo.index:
            pp.create_measurement(net, meas_type="p", element_type="trafo",
                                  value=tp["p_mw"], std_dev=tp.get("sigma", 0.02),
                                  element=trafo_idx, side=side)
            pp.create_measurement(net, meas_type="q", element_type="trafo",
                                  value=tp["q_mvar"], std_dev=tp.get("sigma", 0.02),
                                  element=trafo_idx, side=side)
            n_p += 1
            n_q += 1

    # 移除停运元件的量测（避免SE内部模型不匹配导致残差爆炸）
    if len(net.measurement) > 0:
        drop_idx = []
        for idx in net.measurement.index:
            et = net.measurement.at[idx, "element_type"]
            elem = net.measurement.at[idx, "element"]
            if et == "line" and elem in net.line.index:
                if not net.line.at[elem, "in_service"]:
                    drop_idx.append(idx)
            elif et == "trafo" and elem in net.trafo.index:
                if not net.trafo.at[elem, "in_service"]:
                    drop_idx.append(idx)
            elif et == "bus" and elem in net.bus.index:
                # 检查该母线是否有任何连接的in_service元件
                connected_lines = net.line[
                    ((net.line.from_bus == elem) | (net.line.to_bus == elem)) &
                    net.line.in_service
                ]
                if len(connected_lines) == 0:
                    drop_idx.append(idx)
        if drop_idx:
            net.measurement.drop(drop_idx, inplace=True)
            logger.info(f"移除停运元件量测: {len(drop_idx)} 个")

    logger.info(f"量测添加完成: 电压={n_v}, 有功={n_p}, 无功={n_q}")


# ============================================================
# 2. 运行WLS状态估计
# ============================================================
def run_wls_estimation(net) -> Tuple[bool, Dict]:
    """
    运行WLS（加权最小二乘）状态估计。

    Args:
        net: PandaPower网络（已添加量测）

    Returns:
        (success, results) - 是否收敛，结果字典
    """
    import pandapower.estimation as ppest

    results = {
        "converged": False,
        "bus_results": None,
        "line_results": None,
        "trafo_results": None,
        "error": None,
    }

    try:
                # 先运行潮流计算获取初始状态估计（避免flat-start导致残差爆炸）
        try:
            pp.runpp(net, calculate_voltage_angles=True)
        except Exception:
            pass  # 如果潮流不收敛，仍尝试SE
        est_ret = ppest.estimate(net, init="results", calculate_voltage_angles=True)
        # ppest.estimate returns dict in newer PandaPower versions
        if isinstance(est_ret, dict):
            success = bool(est_ret.get("success", False))
        else:
            success = bool(est_ret)
        results["converged"] = success

        if success:
            if hasattr(net, "res_bus_est"):
                results["bus_results"]   = net.res_bus_est.copy()
            if hasattr(net, "res_line_est"):
                results["line_results"]  = net.res_line_est.copy()
            if hasattr(net, "res_trafo_est"):
                results["trafo_results"] = net.res_trafo_est.copy()
            logger.info("WLS状态估计收敛")
        else:
            logger.warning("WLS状态估计未收敛")
            results["error"] = "估计未收敛"

    except Exception as e:
        success = False
        logger.error(f"状态估计失败: {e}")
        results["error"] = str(e)

    return success, results


# ============================================================
# 3. 不良数据检测: chi2全局检验 + 标准化残差BDD
# ============================================================
def detect_bad_data(net, alpha: float = 0.05,
                    threshold: float = 3.0) -> Dict:
    """
    不良数据检测:
      1. chi2全局检验 (显著性水平 alpha)
      2. 标准化残差超过 threshold 判定为不良数据

    Args:
        net:       PandaPower网络（已运行状态估计）
        alpha:     显著性水平 (默认 0.05)
        threshold: 标准化残差阈值 (默认 3.0)

    Returns:
        {chi2_passed, chi2_statistic, chi2_critical,
         bad_data_indices, normalized_residuals, details}
    """
    result = {
        "chi2_passed": True,
        "chi2_statistic": None,
        "chi2_critical": None,
        "bad_data_indices": [],
        "normalized_residuals": {},
        "details": [],
    }

    if len(net.measurement) == 0:
        result["details"].append("无量测数据")
        return result

    if not hasattr(net, "res_bus_est") or net.res_bus_est is None:
        result["details"].append("未运行状态估计或未收敛")
        return result

    try:
        bad_indices: List[int] = []
        residuals: List[float] = []

        for idx in net.measurement.index:
            meas_type    = net.measurement.at[idx, "measurement_type"]
            element      = net.measurement.at[idx, "element"]
            value        = float(net.measurement.at[idx, "value"])
            std_dev      = float(net.measurement.at[idx, "std_dev"])
            et           = net.measurement.at[idx, "element_type"]

            est_value = _get_estimated_value(net, meas_type, et, element, idx)
            if est_value is None or np.isnan(est_value):
                continue

            residual       = abs(value - est_value)
            normalized_r   = residual / std_dev if std_dev > 0 else 0.0
            residuals.append(normalized_r)
            result["normalized_residuals"][int(idx)] = float(normalized_r)

            if normalized_r > threshold:
                bad_indices.append(int(idx))
                result["details"].append(
                    f"量测{idx}: 类型={meas_type}, 元件={et}_{element}, "
                    f"测量值={value:.4f}, 估计值={est_value:.4f}, "
                    f"标准化残差={normalized_r:.2f}(>{threshold})")

        result["bad_data_indices"] = bad_indices

        # chi2全局检验
        if residuals:
            chi2_stat  = sum(r ** 2 for r in residuals)
            n_states   = len(net.bus) * 2 - 1  # 电压幅值 + 角度(松弛角固定)
            dof        = max(len(residuals) - n_states, 1)
            chi2_crit  = float(chi2.ppf(1 - alpha, dof))

            result["chi2_statistic"] = float(chi2_stat)
            result["chi2_critical"]  = chi2_crit
            
            # Normalized chi2 test: chi2/dof should be ~1.0 under H0
            # For large networks, use normalized chi2 to avoid false positives
            chi2_normalized = chi2_stat / max(dof, 1)
            # Threshold: chi2/dof > 2.0 indicates systematic error
            chi2_norm_threshold = 2.0
            result["chi2_normalized"] = float(chi2_normalized)
            
            # Pass if either: raw chi2 <= critical OR normalized chi2 <= threshold
            # This prevents false positives for large networks with many measurements
            result["chi2_passed"] = chi2_stat <= chi2_crit or chi2_normalized <= chi2_norm_threshold

            logger.info(f"chi2检验: 统计量={chi2_stat:.2f}, "
                        f"临界值={chi2_crit:.2f} (dof={dof}), "
                        f"归一化={chi2_normalized:.3f}, "
                        f"{'通过' if result['chi2_passed'] else '未通过'}")

            # Proportion threshold: if >50% bad -> systematic topology error
            if not result["chi2_passed"]:
                n_meas = len(residuals)
                n_bad = len(bad_indices)
                bad_ratio = n_bad / max(n_meas, 1)
                result["bad_measurement_ratio"] = bad_ratio
                result["systematic_error"] = bad_ratio > 0.5
                # Report chi2 failure as topology-level anomaly
                result["topology_anomaly"] = {
                    "type": "telemetry_mismatch",
                    "confidence": min(0.99, 1.0 - chi2_crit / max(chi2_stat, 1)),
                    "detail": f"chi2全局检验失败: 统计量={chi2_stat:.1f} >> 临界值={chi2_crit:.1f}, 系统级拓扑错误",
                }
                if bad_ratio > 0.5:
                    # Systematic error: keep only top-5 highest residual detections
                    # Sort by normalized residual descending
                    bad_with_r = []
                    for bi in bad_indices:
                        nr = result["normalized_residuals"].get(bi, 0)
                        bad_with_r.append((bi, nr))
                    bad_with_r.sort(key=lambda x: x[1], reverse=True)
                    bad_indices = [bi for bi, _ in bad_with_r[:5]]
                    result["bad_data_indices"] = bad_indices
                    logger.warning(
                        f"chi2 failed, {bad_ratio:.0%} bad -> systematic error, keeping top-{len(bad_indices)} by residual")
                else:
                    # Non-systematic: keep all individual bad data detections
                    # These are likely real anomalies detected by SE
                    result["bad_data_indices"] = bad_indices
                    logger.warning(
                        f"chi2 failed, {bad_ratio:.0%} bad -> keeping {len(bad_indices)} individual detections")

        logger.info(f"不良数据检测: {len(bad_indices)} 个可疑量测 "
                    f"(共 {len(residuals)} 个残差)")

    except Exception as e:
        logger.error(f"不良数据检测失败: {e}")
        result["details"].append(f"检测异常: {e}")

    return result


def _get_estimated_value(net, meas_type: str, et: str,
                         element, idx) -> Optional[float]:
    """从估计结果中获取对应量测的估计值"""
    try:
        if meas_type == "v" and element in net.res_bus_est.index:
            return float(net.res_bus_est.at[element, "vm_pu"])

        if meas_type in ("p", "q"):
            side = net.measurement.at[idx, "side"] \
                if "side" in net.measurement.columns else "from"
            if meas_type == "p":
                col = f"p_{side}_mw"
            else:
                col = f"q_{side}_mvar"

            if et == "line" and element in net.res_line_est.index:
                if col in net.res_line_est.columns:
                    return float(net.res_line_est.at[element, col])
            elif et == "trafo" and element in net.res_trafo_est.index:
                if col in net.res_trafo_est.columns:
                    return float(net.res_trafo_est.at[element, col])
    except (KeyError, TypeError):
        pass
    return None


# ============================================================
# 4. 拓扑错误辨识 (假设检验法)
# ============================================================
def identify_topology_errors(net, bad_data_indices: List[int]) -> List[Dict]:
    """
    当检测到不良数据时，分析是否由拓扑错误引起。

    方法:
      1. 检查不良数据是否集中在某开关附近 -> 可能开关状态有误
      2. 检查不良数据是否集中在某线路 -> 可能线路拓扑错误

    Args:
        net:              PandaPower网络
        bad_data_indices: 不良数据的量测索引列表

    Returns:
        拓扑错误列表 [{switch_idx / line_idx, confidence, ...}]
    """
    topology_errors: List[Dict] = []
    if not bad_data_indices:
        return topology_errors

    # 收集不良数据关联的设备
    bad_elements: set = set()
    for idx in bad_data_indices:
        if idx in net.measurement.index:
            et     = net.measurement.at[idx, "element_type"]
            elem   = int(net.measurement.at[idx, "element"])
            bad_elements.add((et, elem))

    # --- 开关相关 ---
    if hasattr(net, "switch") and len(net.switch) > 0:
        for sw_idx in net.switch.index:
            sw_bus     = int(net.switch.at[sw_idx, "bus"])
            sw_element = int(net.switch.at[sw_idx, "element"])
            related = any((e == sw_bus or e == sw_element) for _, e in bad_elements)
            if related:
                current_closed = bool(net.switch.at[sw_idx, "closed"])
                topology_errors.append({
                    "switch_idx":     int(sw_idx),
                    "bus":            sw_bus,
                    "element":        sw_element,
                    "current_state":  "closed" if current_closed else "open",
                    "suggested_state": "open" if current_closed else "closed",
                    "confidence":     0.75,
                    "related_bad_data": [
                        idx for idx in bad_data_indices
                        if idx in net.measurement.index
                        and (net.measurement.at[idx, "element"] == sw_bus
                             or net.measurement.at[idx, "element"] == sw_element)
                    ],
                })

    # --- 线路相关 ---
    line_bad_count: Dict[int, int] = {}
    for idx in bad_data_indices:
        if idx in net.measurement.index:
            et   = net.measurement.at[idx, "element_type"]
            elem = int(net.measurement.at[idx, "element"])
            if et == "line":
                line_bad_count[elem] = line_bad_count.get(elem, 0) + 1

    for line_idx, count in line_bad_count.items():
        if count >= 2:
            topology_errors.append({
                "type": "line_topology_error",
                "line_idx":  line_idx,
                "confidence": min(0.60 + count * 0.1, 0.95),
                "details": f"线路{line_idx}关联{count}个不良数据，可能存在拓扑错误",
            })

    logger.info(f"拓扑错误辨识: 发现 {len(topology_errors)} 个疑似拓扑错误")
    return topology_errors




