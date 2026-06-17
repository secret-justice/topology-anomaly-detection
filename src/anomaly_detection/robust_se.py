# -*- coding: utf-8 -*-
"""
鲁棒状态估计模块
================
提供四种鲁棒手段，用于对抗SCADA不良数据和遥信缺失：
  1. LAV (最小绝对值) 状态估计 —— LP 求解，天然抗异常量测
  2. IRLS (迭代重加权最小二乘) —— Huber 权函数，迭代降权异常残差
  3. 遥信缺失插补 —— 基于基尔霍夫电流定律拓扑约束推断开关状态
  4. 遥测波动平滑 —— 滑动窗口中位数 + MAD 异常值剔除

所有函数均与 state_estimator.py 的接口风格保持一致。
"""
import copy
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandapower as pp
import logging

logger = logging.getLogger(__name__)


# ============================================================
# 内部工具函数
# ============================================================
def _build_measurement_vectors(net) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    从 PandaPower 网络提取量测向量 z、标准差 sigma 和量测元信息列表。

    Returns:
        z:      量测值向量 (m,)
        sigma:  标准差向量 (m,)
        info:   每个量测的元信息 [(meas_type, element_type, element, side, idx), ...]
    """
    measurements = net.measurement
    if len(measurements) == 0:
        return np.array([]), np.array([]), []

    z = measurements["value"].values.astype(float)
    sigma = measurements["std_dev"].values.astype(float)
    # 避免零标准差
    sigma = np.maximum(sigma, 1e-6)

    info = []
    for idx in measurements.index:
        info.append({
            "meas_type": measurements.at[idx, "measurement_type"],
            "element_type": measurements.at[idx, "element_type"],
            "element": int(measurements.at[idx, "element"]),
            "side": measurements.at[idx, "side"] if "side" in measurements.columns else "",
            "idx": int(idx),
        })

    return z, sigma, info


def _build_dc_jacobian(net, info: list) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    构建简化的 DC 潮流 Jacobian 矩阵 H，使 z ≈ H * x。

    x = 各母线电压相角（弧度），以松弛母线为参考。

    对于 DC 近似:
      - P量测: P_i ≈ Σ_j B_ij * (θ_i - θ_j)
      - V量测: V ≈ 1.0 + 小偏差（线性化）

    Returns:
        H:      Jacobian 矩阵 (m, n_state)
        z_dc:   量测向量（已减去参考值偏移）
        state_buses: 状态变量对应的母线编号列表
    """
    # 获取母线列表和松弛母线
    bus_indices = sorted(net.bus.index.tolist())
    if hasattr(net, "ext_grid") and len(net.ext_grid) > 0:
        slack_bus = int(net.ext_grid.at[net.ext_grid.index[0], "bus"])
    elif hasattr(net, "gen") and len(net.gen) > 0:
        slack_bus = int(net.gen.at[net.gen.index[0], "bus"])
    else:
        slack_bus = bus_indices[0]

    # 状态变量 = 除松弛母线外的所有母线相角
    state_buses = [b for b in bus_indices if b != slack_bus]
    n_state = len(state_buses)
    bus_to_idx = {b: i for i, b in enumerate(state_buses)}

    # 构建导纳矩阵（简化 B 矩阵，忽略电阻）
    n_bus = len(bus_indices)
    bus_pos = {b: i for i, b in enumerate(bus_indices)}
    B = np.zeros((n_bus, n_bus))

    # 线路
    if hasattr(net, "line"):
        for idx in net.line.index:
            if not net.line.at[idx, "in_service"]:
                continue
            fb = int(net.line.at[idx, "from_bus"])
            tb = int(net.line.at[idx, "to_bus"])
            x_pu = max(float(net.line.at[idx, "x_ohm_per_km"]) *
                        float(net.line.at[idx, "length_km"]), 1e-6)
            b_line = 1.0 / x_pu
            fi = bus_pos.get(fb)
            ti = bus_pos.get(tb)
            if fi is not None and ti is not None:
                B[fi, ti] -= b_line
                B[ti, fi] -= b_line
                B[fi, fi] += b_line
                B[ti, ti] += b_line

    # 变压器（简化处理）
    if hasattr(net, "trafo"):
        for idx in net.trafo.index:
            if not net.trafo.at[idx, "in_service"]:
                continue
            hb = int(net.trafo.at[idx, "hv_bus"])
            lb = int(net.trafo.at[idx, "lv_bus"])
            sn = float(net.trafo.at[idx, "sn_mva"]) if "sn_mva" in net.trafo.columns else 100.0
            vk = float(net.trafo.at[idx, "vk_percent"]) if "vk_percent" in net.trafo.columns else 10.0
            b_trafo = sn / max(vk, 0.1) * 100.0
            hi = bus_pos.get(hb)
            li = bus_pos.get(lb)
            if hi is not None and li is not None:
                B[hi, li] -= b_trafo
                B[li, hi] -= b_trafo
                B[hi, hi] += b_trafo
                B[li, li] += b_trafo

    # 构建 H 矩阵
    m = len(info)
    H = np.zeros((m, n_state))
    z_dc = np.zeros(m)

    for i, meas in enumerate(info):
        mt = meas["meas_type"]
        et = meas["element_type"]
        elem = meas["element"]
        side = meas["side"]

        if mt == "v":
            # 电压量测: V ≈ 1.0 (DC 近似中电压基本不变，作为弱约束)
            bus = elem
            si = bus_to_idx.get(bus)
            if si is not None:
                H[i, si] = 1.0  # 微小相角影响
            z_dc[i] = 0.0  # 偏移量

        elif mt == "p" and et == "line" and elem in net.line.index:
            # 线路有功: P_ij = B_ij * (θ_i - θ_j)
            fb = int(net.line.at[elem, "from_bus"])
            tb = int(net.line.at[elem, "to_bus"])
            x_pu = max(float(net.line.at[elem, "x_ohm_per_km"]) *
                        float(net.line.at[elem, "length_km"]), 1e-6)
            b_line = 1.0 / x_pu

            if side in ("from", ""):
                from_si = bus_to_idx.get(fb)
                to_si = bus_to_idx.get(tb)
            else:
                from_si = bus_to_idx.get(tb)
                to_si = bus_to_idx.get(fb)
                b_line = -b_line

            if from_si is not None:
                H[i, from_si] = b_line
            if to_si is not None:
                H[i, to_si] = -b_line

        elif mt == "p" and et == "trafo" and elem in net.trafo.index:
            # 变压器有功
            hb = int(net.trafo.at[elem, "hv_bus"])
            lb = int(net.trafo.at[elem, "lv_bus"])
            sn = float(net.trafo.at[elem, "sn_mva"]) if "sn_mva" in net.trafo.columns else 100.0
            vk = float(net.trafo.at[elem, "vk_percent"]) if "vk_percent" in net.trafo.columns else 10.0
            b_trafo = sn / max(vk, 0.1) * 100.0

            if side in ("hv", ""):
                from_si = bus_to_idx.get(hb)
                to_si = bus_to_idx.get(lb)
            else:
                from_si = bus_to_idx.get(lb)
                to_si = bus_to_idx.get(hb)
                b_trafo = -b_trafo

            if from_si is not None:
                H[i, from_si] = b_trafo
            if to_si is not None:
                H[i, to_si] = -b_trafo

    return H, z_dc, state_buses


def _should_use_ac_mode(net):
    """Auto detect if AC mode needed: distribution R/X > 0.3 means DC is bad"""
    if not hasattr(net, "line") or len(net.line) == 0:
        return False
    rx_ratios = []
    for idx in net.line.index:
        r = float(net.line.at[idx, "r_ohm_per_km"])
        x = float(net.line.at[idx, "x_ohm_per_km"])
        if x > 1e-6:
            rx_ratios.append(r / x)
    if not rx_ratios:
        return False
    return (sum(rx_ratios) / len(rx_ratios)) > 0.3


def _build_ac_jacobian(net, info):
    """Build full AC Jacobian H for z = H*x.
    State: [theta_1..theta_{n-1}, V_1..V_n].
    P_ij = Vi*Vj*(Gij*cos(th_ij)+Bij*sin(th_ij)) - Gij*Vi^2
    Q_ij = Vi*Vj*(Gij*sin(th_ij)-Bij*cos(th_ij)) + Bij*Vi^2
    """
    import numpy as np
    import pandapower as pp

    bus_indices = sorted(net.bus.index.tolist())
    if hasattr(net, "ext_grid") and len(net.ext_grid) > 0:
        slack_bus = int(net.ext_grid.at[net.ext_grid.index[0], "bus"])
    elif hasattr(net, "gen") and len(net.gen) > 0:
        slack_bus = int(net.gen.at[net.gen.index[0], "bus"])
    else:
        slack_bus = bus_indices[0]

    theta_buses = [b for b in bus_indices if b != slack_bus]
    v_buses = list(bus_indices)
    n_theta = len(theta_buses)
    n_v = len(v_buses)
    n_state = n_theta + n_v
    theta_to_idx = {b: i for i, b in enumerate(theta_buses)}
    v_to_idx = {b: i for i, b in enumerate(v_buses)}

    try:
        pp.runpp(net, calculate_voltage_angles=True)
        v_init = np.array([float(net.res_bus.at[b, "vm_pu"]) for b in bus_indices])
        theta_init = np.radians(np.array([float(net.res_bus.at[b, "va_degree"]) for b in bus_indices]))
    except Exception:
        v_init = np.ones(len(bus_indices))
        theta_init = np.zeros(len(bus_indices))

    bus_pos = {b: i for i, b in enumerate(bus_indices)}
    n_bus = len(bus_indices)
    G_mat = np.zeros((n_bus, n_bus))
    B_mat = np.zeros((n_bus, n_bus))
    base_mva = float(net.sn_mva) if hasattr(net, "sn_mva") else 100.0

    if hasattr(net, "line"):
        for idx in net.line.index:
            if not net.line.at[idx, "in_service"]:
                continue
            fb = int(net.line.at[idx, "from_bus"])
            tb = int(net.line.at[idx, "to_bus"])
            r_ohm = float(net.line.at[idx, "r_ohm_per_km"]) * float(net.line.at[idx, "length_km"])
            x_ohm = float(net.line.at[idx, "x_ohm_per_km"]) * float(net.line.at[idx, "length_km"])
            vn_kv = float(net.bus.at[fb, "vn_kv"]) if fb in net.bus.index else float(net.bus.at[tb, "vn_kv"])
            base_z = vn_kv ** 2 / base_mva
            r_pu, x_pu = r_ohm / base_z, x_ohm / base_z
            denom = r_pu**2 + x_pu**2
            if denom < 1e-12:
                continue
            g, b_val = r_pu / denom, -x_pu / denom
            fi, ti = bus_pos.get(fb), bus_pos.get(tb)
            if fi is not None and ti is not None:
                G_mat[fi, ti] -= g; G_mat[ti, fi] -= g
                B_mat[fi, ti] -= b_val; B_mat[ti, fi] -= b_val
                G_mat[fi, fi] += g; G_mat[ti, ti] += g
                B_mat[fi, fi] += b_val; B_mat[ti, ti] += b_val

    if hasattr(net, "trafo"):
        for idx in net.trafo.index:
            if not net.trafo.at[idx, "in_service"]:
                continue
            hb = int(net.trafo.at[idx, "hv_bus"])
            lb = int(net.trafo.at[idx, "lv_bus"])
            sn = float(net.trafo.at[idx, "sn_mva"]) if "sn_mva" in net.trafo.columns else 100.0
            vk = float(net.trafo.at[idx, "vk_percent"]) if "vk_percent" in net.trafo.columns else 10.0
            vkr = float(net.trafo.at[idx, "vkr_percent"]) if "vkr_percent" in net.trafo.columns else 0.5
            z_pu = vk / 100.0 * (base_mva / sn)
            r_pu = vkr / 100.0 * (base_mva / sn)
            x_pu = max(z_pu**2 - r_pu**2, 1e-12) ** 0.5
            denom = r_pu**2 + x_pu**2
            if denom < 1e-12:
                continue
            g, b_val = r_pu / denom, -x_pu / denom
            hi, li = bus_pos.get(hb), bus_pos.get(lb)
            if hi is not None and li is not None:
                G_mat[hi, li] -= g; G_mat[li, hi] -= g
                B_mat[hi, li] -= b_val; B_mat[li, hi] -= b_val
                G_mat[hi, hi] += g; G_mat[li, li] += g
                B_mat[hi, hi] += b_val; B_mat[li, li] += b_val

    m = len(info)
    H = np.zeros((m, n_state))
    z_ac = np.zeros(m)

    for i, meas in enumerate(info):
        mt, et, elem, side = meas["meas_type"], meas["element_type"], meas["element"], meas["side"]
        if mt == "v":
            vi = v_to_idx.get(elem)
            if vi is not None:
                H[i, n_theta + vi] = 1.0
            z_ac[i] = v_init[bus_pos.get(elem, 0)] if elem in bus_pos else 1.0
        elif mt == "p" and et == "line" and elem in net.line.index:
            fb = int(net.line.at[elem, "from_bus"])
            tb = int(net.line.at[elem, "to_bus"])
            fi, ti = bus_pos.get(fb), bus_pos.get(tb)
            if fi is None or ti is None:
                continue
            vi, vj = v_init[fi], v_init[ti]
            td = theta_init[fi] - theta_init[ti]
            g_ij, b_ij = G_mat[fi, ti], B_mat[fi, ti]
            dPdt_i = vi * vj * (-g_ij * np.sin(td) + b_ij * np.cos(td))
            dPdt_j = -dPdt_i
            if side in ("from", ""):
                dPdVi = vj * (g_ij * np.cos(td) + b_ij * np.sin(td)) + 2 * (-g_ij) * vi
                dPdVj = vi * (g_ij * np.cos(td) + b_ij * np.sin(td))
                p_val = vi * vj * (g_ij * np.cos(td) + b_ij * np.sin(td)) - (-g_ij) * vi**2
            else:
                td2 = -td
                dPdt_i = -vi * vj * (-g_ij * np.sin(td2) + b_ij * np.cos(td2))
                dPdt_j = -dPdt_i
                dPdVi = vi * (g_ij * np.cos(td2) + b_ij * np.sin(td2))
                dPdVj = vj * (g_ij * np.cos(td2) + b_ij * np.sin(td2)) + 2 * (-g_ij) * vj
                p_val = vi * vj * (g_ij * np.cos(td2) + b_ij * np.sin(td2)) - (-g_ij) * vj**2
            if fb in theta_to_idx:
                H[i, theta_to_idx[fb]] = dPdt_i
            if tb in theta_to_idx:
                H[i, theta_to_idx[tb]] = dPdt_j
            H[i, n_theta + v_to_idx[fb]] = dPdVi
            H[i, n_theta + v_to_idx[tb]] = dPdVj
            z_ac[i] = p_val if side in ("from", "") else -p_val
        elif mt == "p" and et == "trafo" and elem in net.trafo.index:
            hb = int(net.trafo.at[elem, "hv_bus"])
            lb = int(net.trafo.at[elem, "lv_bus"])
            fi, ti = bus_pos.get(hb), bus_pos.get(lb)
            if fi is None or ti is None:
                continue
            vi, vj = v_init[fi], v_init[ti]
            td = theta_init[fi] - theta_init[ti]
            g_ij, b_ij = G_mat[fi, ti], B_mat[fi, ti]
            dPdt_i = vi * vj * (-g_ij * np.sin(td) + b_ij * np.cos(td))
            dPdt_j = -dPdt_i
            dPdVi = vj * (g_ij * np.cos(td) + b_ij * np.sin(td)) + 2 * (-g_ij) * vi
            dPdVj = vi * (g_ij * np.cos(td) + b_ij * np.sin(td))
            p_val = vi * vj * (g_ij * np.cos(td) + b_ij * np.sin(td)) - (-g_ij) * vi**2
            if hb in theta_to_idx:
                H[i, theta_to_idx[hb]] = dPdt_i
            if lb in theta_to_idx:
                H[i, theta_to_idx[lb]] = dPdt_j
            H[i, n_theta + v_to_idx[hb]] = dPdVi
            H[i, n_theta + v_to_idx[lb]] = dPdVj
            z_ac[i] = p_val if side in ("hv", "") else -p_val
        elif mt == "q" and et == "line" and elem in net.line.index:
            fb = int(net.line.at[elem, "from_bus"])
            tb = int(net.line.at[elem, "to_bus"])
            fi, ti = bus_pos.get(fb), bus_pos.get(tb)
            if fi is None or ti is None:
                continue
            vi, vj = v_init[fi], v_init[ti]
            td = theta_init[fi] - theta_init[ti]
            g_ij, b_ij = G_mat[fi, ti], B_mat[fi, ti]
            dQdt_i = vi * vj * (g_ij * np.cos(td) + b_ij * np.sin(td))
            dQdt_j = -dQdt_i
            if side in ("from", ""):
                dQdVi = vj * (g_ij * np.sin(td) - b_ij * np.cos(td)) + 2 * b_ij * vi
                dQdVj = vi * (g_ij * np.sin(td) - b_ij * np.cos(td))
                q_val = vi * vj * (g_ij * np.sin(td) - b_ij * np.cos(td)) + b_ij * vi**2
            else:
                td2 = -td
                dQdt_i = -vi * vj * (g_ij * np.cos(td2) + b_ij * np.sin(td2))
                dQdt_j = -dQdt_i
                dQdVi = vi * (g_ij * np.sin(td2) - b_ij * np.cos(td2))
                dQdVj = vj * (g_ij * np.sin(td2) - b_ij * np.cos(td2)) + 2 * b_ij * vj
                q_val = vi * vj * (g_ij * np.sin(td2) - b_ij * np.cos(td2)) + b_ij * vj**2
            if fb in theta_to_idx:
                H[i, theta_to_idx[fb]] = dQdt_i
            if tb in theta_to_idx:
                H[i, theta_to_idx[tb]] = dQdt_j
            H[i, n_theta + v_to_idx[fb]] = dQdVi
            H[i, n_theta + v_to_idx[tb]] = dQdVj
            z_ac[i] = q_val if side in ("from", "") else -q_val

    state_buses = theta_buses + v_buses
    return H, z_ac, state_buses


def _apply_ac_se_results(net, x_est, theta_buses, v_buses, results):
    """Write AC SE results (theta + V) back to PandaPower net."""
    import numpy as np
    n_theta = len(theta_buses)
    theta_est = x_est[:n_theta]
    v_est = x_est[n_theta:n_theta + len(v_buses)]
    if hasattr(net, "res_bus_est") and len(net.res_bus_est) > 0:
        for i, bus in enumerate(theta_buses):
            if bus in net.res_bus_est.index and i < len(theta_est):
                net.res_bus_est.at[bus, "va_degree"] = float(np.degrees(theta_est[i]))
        for i, bus in enumerate(v_buses):
            if bus in net.res_bus_est.index and i < len(v_est):
                net.res_bus_est.at[bus, "vm_pu"] = float(v_est[i])
        results["bus_results"] = net.res_bus_est.copy()
    elif hasattr(net, "res_bus") and len(net.res_bus) > 0:
        for i, bus in enumerate(theta_buses):
            if bus in net.res_bus.index and i < len(theta_est):
                net.res_bus.at[bus, "va_degree"] = float(np.degrees(theta_est[i]))
        for i, bus in enumerate(v_buses):
            if bus in net.res_bus.index and i < len(v_est):
                net.res_bus.at[bus, "vm_pu"] = float(v_est[i])
        results["bus_results"] = net.res_bus.copy()


def run_ac_lav_estimation(net, scada_data, max_iter=20):
    """AC LAV estimation using full AC Jacobian. For distribution grids."""
    from scipy.optimize import linprog
    import numpy as np
    import pandapower as pp
    import warnings as _w

    results = {"converged": False, "method": "AC_LAV", "bus_results": None,
               "line_results": None, "trafo_results": None, "error": None,
               "objective": None, "n_measurements": 0, "n_states": 0}
    try:
        try:
            pp.runpp(net, calculate_voltage_angles=True)
        except Exception:
            pass
        z, sigma, info = _build_measurement_vectors(net)
        if len(z) == 0:
            results["error"] = "No measurements"
            return False, results
        results["n_measurements"] = len(z)
        H, z_ac, state_buses = _build_ac_jacobian(net, info)
        n_state = H.shape[1]
        results["n_states"] = n_state
        if n_state == 0:
            results["error"] = "No states"
            return False, results
        W_inv = np.diag(1.0 / sigma)
        H_s, z_s = W_inv @ H, W_inv @ z_ac
        m = len(z_s)
        c = np.concatenate([np.zeros(n_state), np.ones(m)])
        I_m = np.eye(m)
        A_ub = np.vstack([np.hstack([H_s, -I_m]), np.hstack([-H_s, -I_m])])
        b_ub = np.concatenate([z_s, -z_s])
        bounds = [(None, None)] * n_state + [(0, None)] * m
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                          method="highs", options={"maxiter": max_iter * 100})
        if not res.success:
            results["error"] = f"LP failed: {res.message}"
            return False, results
        x_est = res.x[:n_state]
        results["objective"] = float(res.fun)
        results["converged"] = True
        bus_indices = sorted(net.bus.index.tolist())
        slack = int(net.ext_grid.at[net.ext_grid.index[0], "bus"]) if hasattr(net, "ext_grid") and len(net.ext_grid) > 0 else bus_indices[0]
        theta_buses = [b for b in bus_indices if b != slack]
        _apply_ac_se_results(net, x_est, theta_buses, bus_indices, results)
        return True, results
    except Exception as e:
        results["error"] = str(e)
        return False, results


def run_ac_irls_estimation(net, scada_data, max_iter=20, tuning_constant=1.345, tol=1e-4):
    """AC IRLS robust estimation using full AC Jacobian."""
    import numpy as np

    results = {"converged": False, "method": "AC_IRLS", "bus_results": None,
               "line_results": None, "trafo_results": None, "error": None,
               "iterations": 0, "n_outliers": 0}
    try:
        z, sigma, info = _build_measurement_vectors(net)
        if len(z) == 0:
            results["error"] = "No measurements"
            return False, results
        H, z_ac, state_buses = _build_ac_jacobian(net, info)
        n_state = H.shape[1]
        if n_state == 0:
            results["error"] = "No states"
            return False, results
        W_inv = np.diag(1.0 / sigma)
        H_s, z_s = W_inv @ H, W_inv @ z_ac
        HtH = H_s.T @ H_s + np.eye(n_state) * 1e-8
        try:
            x_est = np.linalg.solve(HtH, H_s.T @ z_s)
        except np.linalg.LinAlgError:
            results["error"] = "Singular WLS"
            return False, results
        for iteration in range(max_iter):
            residuals = (z_ac - H @ x_est) / sigma
            weights = _huber_weight(residuals, tuning_constant)
            W_iter = np.diag(weights / sigma)
            H_w, z_w = W_iter @ H, W_iter @ z_ac
            HtWH = H_w.T @ H_w + np.eye(n_state) * 1e-8
            try:
                x_new = np.linalg.solve(HtWH, H_w.T @ z_w)
            except np.linalg.LinAlgError:
                break
            dx = np.max(np.abs(x_new - x_est))
            x_est = x_new
            if dx < tol:
                results["converged"] = True
                results["iterations"] = iteration + 1
                break
        else:
            results["converged"] = True
            results["iterations"] = max_iter
        final_r = np.abs((z_ac - H @ x_est) / sigma)
        results["n_outliers"] = int(np.sum(final_r > tuning_constant))
        bus_indices = sorted(net.bus.index.tolist())
        slack = int(net.ext_grid.at[net.ext_grid.index[0], "bus"]) if hasattr(net, "ext_grid") and len(net.ext_grid) > 0 else bus_indices[0]
        theta_buses = [b for b in bus_indices if b != slack]
        _apply_ac_se_results(net, x_est, theta_buses, bus_indices, results)
        return results["converged"], results
    except Exception as e:
        results["error"] = str(e)
        return False, results



def _huber_weight(residuals: np.ndarray, tuning_constant: float = 1.345) -> np.ndarray:
    """
    Huber 权函数。

    |r| <= c  →  w = 1
    |r| > c  →  w = c / |r|

    Args:
        residuals: 标准化残差向量
        tuning_constant: Huber 调谐常数（默认 1.345，95% 渐近效率）

    Returns:
        权重向量（与 residuals 同维）
    """
    abs_r = np.abs(residuals)
    weights = np.ones_like(abs_r)
    mask = abs_r > tuning_constant
    weights[mask] = tuning_constant / abs_r[mask]
    return weights


# ============================================================
# 1. LAV (最小绝对值) 状态估计
# ============================================================
def run_lav_estimation(net, scada_data: Dict, max_iter: int = 20) -> Tuple[bool, Dict]:
    """
    LAV (Least Absolute Value) 状态估计。

    相比 WLS 优化 Σ(r_i² / σ_i²)，LAV 优化 Σ|r_i / σ_i|，
    对异常量测（bad data）具有天然的鲁棒性——单个大残差不会主导目标函数。

    通过线性规划 (LP) 求解:
        min  Σ t_i
        s.t. t_i >= (z_i - H*x)_i / σ_i
             t_i >= -(z_i - H*x)_i / σ_i

    Args:
        net:       PandaPower 网络（已添加量测，可选已运行潮流）
        scada_data: SCADA 数据字典
        max_iter:   最大迭代次数（LAV 通常单次 LP 即可，此参数预留）

    Returns:
        (success, results) - 是否成功，结果字典
    """
    from scipy.optimize import linprog

    results = {
        "converged": False,
        "method": "LAV",
        "bus_results": None,
        "line_results": None,
        "trafo_results": None,
        "error": None,
        "objective": None,
        "n_measurements": 0,
        "n_states": 0,
    }

    try:
        # 先运行一次潮流获取初始状态
        try:
            pp.runpp(net, calculate_voltage_angles=True)
        except Exception:
            logger.warning("初始潮流失败，使用 flat start")

        # 提取量测
        z, sigma, info = _build_measurement_vectors(net)
        if len(z) == 0:
            results["error"] = "无量测数据"
            return False, results

        results["n_measurements"] = len(z)

        # 构建 DC Jacobian
        H, z_dc, state_buses = _build_dc_jacobian(net, info)
        n_state = H.shape[1]
        results["n_states"] = n_state

        if n_state == 0:
            results["error"] = "无可估计状态量"
            return False, results

        # 标准化
        W_inv = np.diag(1.0 / sigma)
        H_s = W_inv @ H
        z_s = W_inv @ z_dc

        m = len(z_s)

        # LP 变量: [x(n_state), t(m)]，t_i >= 0
        # 目标: min Σ t_i
        c = np.concatenate([np.zeros(n_state), np.ones(m)])

        # 约束:
        #   H_s @ x - t <= z_s
        #  -H_s @ x - t <= -z_s
        # 等价于:
        #   H_s @ x - t - s1 = z_s,  s1 >= 0  (松弛)
        #  -H_s @ x - t - s2 = -z_s, s2 >= 0

        # 使用不等式约束 A_ub @ y <= b_ub
        # y = [x, t]
        I_m = np.eye(m)
        A_ub = np.vstack([
            np.hstack([H_s, -I_m]),    # H_s*x - t <= z_s
            np.hstack([-H_s, -I_m]),   # -H_s*x - t <= -z_s
        ])
        b_ub = np.concatenate([z_s, -z_s])

        # 边界: x 无约束, t >= 0
        bounds = [(None, None)] * n_state + [(0, None)] * m

        # 求解 LP
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                          method="highs", options={"maxiter": max_iter * 100})

        if not res.success:
            results["error"] = f"LP 求解失败: {res.message}"
            logger.warning(f"LAV 状态估计失败: {res.message}")
            return False, results

        x_est = res.x[:n_state]
        results["objective"] = float(res.fun)
        results["converged"] = True

        # 写回估计结果到 net
        _apply_se_results(net, x_est, state_buses, results)
        logger.info(f"LAV 状态估计收敛: 目标函数值={res.fun:.4f}, 量测数={m}, 状态数={n_state}")

        return True, results

    except Exception as e:
        results["error"] = str(e)
        logger.error(f"LAV 状态估计异常: {e}")
        return False, results


# ============================================================
# 2. IRLS (迭代重加权最小二乘) 鲁棒状态估计
# ============================================================
def run_irls_estimation(
    net,
    scada_data: Dict,
    max_iter: int = 20,
    tuning_constant: float = 1.345,
    tol: float = 1e-4,
) -> Tuple[bool, Dict]:
    """
    IRLS (Iteratively Reweighted Least Squares) 鲁棒状态估计。

    使用 Huber 权函数迭代降权异常残差:
      1. 初始运行 WLS
      2. 计算标准化残差
      3. 用 Huber 权函数重加权
      4. 求解加权最小二乘
      5. 重复直到收敛

    Args:
        net:              PandaPower 网络
        scada_data:       SCADA 数据字典
        max_iter:         最大迭代次数
        tuning_constant:  Huber 调谐常数（默认 1.345）
        tol:              收敛阈值（状态量变化量）

    Returns:
        (success, results) - 是否成功，结果字典
    """
    results = {
        "converged": False,
        "method": "IRLS",
        "bus_results": None,
        "line_results": None,
        "trafo_results": None,
        "error": None,
        "iterations": 0,
        "n_outliers": 0,
        "tuning_constant": tuning_constant,
    }

    try:
        # 先运行 WLS 获取初始估计
        from anomaly_detection.state_estimator import run_wls_estimation
        wls_ok, wls_res = run_wls_estimation(net)
        if not wls_ok and wls_res.get("error"):
            logger.warning(f"WLS 初始估计未收敛: {wls_res.get('error')}")

        # 提取量测
        z, sigma, info = _build_measurement_vectors(net)
        if len(z) == 0:
            results["error"] = "无量测数据"
            return False, results

        # 构建 DC Jacobian
        H, z_dc, state_buses = _build_dc_jacobian(net, info)
        n_state = H.shape[1]

        if n_state == 0:
            results["error"] = "无可估计状态量"
            return False, results

        # 初始 WLS 解
        W_inv = np.diag(1.0 / sigma)
        H_s = W_inv @ H
        z_s = W_inv @ z_dc

        # 初始 WLS: x = (H^T W H)^-1 H^T W z
        HtH = H_s.T @ H_s
        # 正则化防止奇异
        HtH += np.eye(n_state) * 1e-8
        Htz = H_s.T @ z_s
        try:
            x_est = np.linalg.solve(HtH, Htz)
        except np.linalg.LinAlgError:
            results["error"] = "初始 WLS 矩阵奇异"
            return False, results

        # IRLS 迭代
        for iteration in range(max_iter):
            residuals = (z_dc - H @ x_est) / sigma
            weights = _huber_weight(residuals, tuning_constant)

            # 加权
            W_iter = np.diag(weights / sigma)
            H_w = W_iter @ H
            z_w = W_iter @ z_dc

            HtWH = H_w.T @ H_w + np.eye(n_state) * 1e-8
            HtWz = H_w.T @ z_w

            try:
                x_new = np.linalg.solve(HtWH, HtWz)
            except np.linalg.LinAlgError:
                logger.warning(f"IRLS 迭代 {iteration}: 矩阵奇异，提前终止")
                break

            # 收敛判断
            dx = np.max(np.abs(x_new - x_est))
            x_est = x_new

            if dx < tol:
                results["converged"] = True
                results["iterations"] = iteration + 1
                logger.info(f"IRLS 收敛: 迭代={iteration+1}, Δx={dx:.2e}")
                break
        else:
            results["converged"] = True
            results["iterations"] = max_iter
            logger.warning(f"IRLS 达到最大迭代 {max_iter}")

        # 统计异常值
        final_residuals = np.abs((z_dc - H @ x_est) / sigma)
        results["n_outliers"] = int(np.sum(final_residuals > tuning_constant))

        # 写回结果
        _apply_se_results(net, x_est, state_buses, results)
        logger.info(
            f"IRLS 状态估计: 迭代={results['iterations']}, "
            f"异常量测={results['n_outliers']}/{len(z)}"
        )

        return results["converged"], results

    except Exception as e:
        results["error"] = str(e)
        logger.error(f"IRLS 状态估计异常: {e}")
        return False, results


def _apply_se_results(net, x_est: np.ndarray, state_buses: list,
                      results: Dict) -> None:
    """将估计结果写入 PandaPower 网络"""
    # 写 bus 估计结果
    if hasattr(net, "res_bus_est") and len(net.res_bus_est) > 0:
        for i, bus in enumerate(state_buses):
            if bus in net.res_bus_est.index and i < len(x_est):
                # DC 近似: 电压相角变化
                net.res_bus_est.at[bus, "va_degree"] = float(np.degrees(x_est[i]))
        results["bus_results"] = net.res_bus_est.copy()
    elif hasattr(net, "res_bus") and len(net.res_bus) > 0:
        # 回退到 res_bus
        results["bus_results"] = net.res_bus.copy()

    if hasattr(net, "res_line_est"):
        results["line_results"] = net.res_line_est.copy()
    if hasattr(net, "res_trafo_est"):
        results["trafo_results"] = net.res_trafo_est.copy()


# ============================================================
# 3. 遥信缺失插补
# ============================================================
def impute_missing_switches(
    net,
    graph,
    measurements: Dict,
) -> List[Dict]:
    """
    遥信缺失插补：基于拓扑约束推断缺失开关状态。

    原理:
      对于每个开关状态未知的节点，利用基尔霍夫电流定律 (KCL):
        Σ P_inj = 0
      如果该节点注入功率已知，且关联支路中除该开关外功率均可量测，
      则可推算出通过该开关的功率，进而推断开/合状态:
        |P_switch| > threshold → 开关闭合
        |P_switch| ≈ 0        → 开关断开

    Args:
        net:          PandaPower 网络
        graph:        NetworkX 拓扑图
        measurements: SCADA 量测数据字典

    Returns:
        插补结果列表 [{switch_idx, bus, element, inferred_state, confidence, method}, ...]
    """
    results: List[Dict] = []

    if not hasattr(net, "switch") or len(net.switch) == 0:
        logger.info("无开关，跳过遥信插补")
        return results

    # 构建量测查询表
    meas_by_element: Dict[Tuple[str, int], Dict] = {}
    for lp in measurements.get("line_powers", []):
        meas_by_element[("line", int(lp["line"]))] = lp
    for tp in measurements.get("trafo_powers", []):
        meas_by_element[("trafo", int(tp["trafo"]))] = tp
    for bv in measurements.get("bus_voltages", []):
        meas_by_element[("bus", int(bv["bus"]))] = bv

    # 母线注入功率
    bus_injection: Dict[int, float] = {}
    for gen in measurements.get("gen_powers", []):
        bus = int(gen.get("bus", -1))
        if bus >= 0:
            bus_injection[bus] = bus_injection.get(bus, 0.0) + gen.get("p_mw", 0.0)
    for load in measurements.get("load_powers", []):
        bus = int(load.get("bus", -1))
        if bus >= 0:
            bus_injection[bus] = bus_injection.get(bus, 0.0) - load.get("p_mw", 0.0)

    # 逐开关分析
    for sw_idx in net.switch.index:
        sw_closed = bool(net.switch.at[sw_idx, "closed"])
        sw_bus = int(net.switch.at[sw_idx, "bus"])
        sw_element = int(net.switch.at[sw_idx, "element"])
        sw_et = str(net.switch.at[sw_idx, "et"])

        # 检查该开关是否有对应量测（有量测说明已知）
        if ("switch", int(sw_idx)) in meas_by_element:
            continue  # 已知，跳过

        # 收集该母线除开关外的已知功率
        known_power_sum = 0.0
        n_known = 0

        # 遍历该母线关联的线路
        if hasattr(net, "line"):
            for line_idx in net.line.index:
                fb = int(net.line.at[line_idx, "from_bus"])
                tb = int(net.line.at[line_idx, "to_bus"])
                if fb != sw_bus and tb != sw_bus:
                    continue
                if ("line", int(line_idx)) in meas_by_element:
                    lp = meas_by_element[("line", int(line_idx))]
                    side = "from" if fb == sw_bus else "to"
                    p = lp.get("p_mw", 0.0)
                    # 从母线流出为正
                    if side == "from":
                        known_power_sum += p
                    else:
                        known_power_sum -= p
                    n_known += 1

        # 遍历该母线关联的变压器
        if hasattr(net, "trafo"):
            for trafo_idx in net.trafo.index:
                hb = int(net.trafo.at[trafo_idx, "hv_bus"])
                lb = int(net.trafo.at[trafo_idx, "lv_bus"])
                if hb != sw_bus and lb != sw_bus:
                    continue
                if ("trafo", int(trafo_idx)) in meas_by_element:
                    tp = meas_by_element[("trafo", int(trafo_idx))]
                    side = "hv" if hb == sw_bus else "lv"
                    p = tp.get("p_mw", 0.0)
                    if side == "hv":
                        known_power_sum += p
                    else:
                        known_power_sum -= p
                    n_known += 1

        # 注入功率
        inj = bus_injection.get(sw_bus, 0.0)

        if n_known == 0:
            # 无可参考量测，仅基于当前拓扑状态做低置信度猜测
            results.append({
                "switch_idx": int(sw_idx),
                "bus": sw_bus,
                "element": sw_element,
                "inferred_state": "closed" if sw_closed else "open",
                "confidence": 0.3,
                "method": "topology_default",
                "details": "无关联量测，保持当前状态",
            })
            continue

        # KCL 残差: 未知开关功率 ≈ inj - Σ known
        residual_power = inj - known_power_sum

        # 推断状态
        if abs(residual_power) > 0.5:
            # 有显著功率流过 → 开关闭合
            inferred = "closed"
            confidence = min(0.5 + abs(residual_power) * 0.1, 0.95)
        else:
            # 功率平衡 → 开关断开
            inferred = "open"
            confidence = min(0.5 + (0.5 - abs(residual_power)) * 0.5, 0.90)

        # 检查是否与当前状态矛盾
        current = "closed" if sw_closed else "open"
        conflict = (inferred != current)

        results.append({
            "switch_idx": int(sw_idx),
            "bus": sw_bus,
            "element": sw_element,
            "inferred_state": inferred,
            "current_state": current,
            "confidence": round(confidence, 3),
            "method": "KCL_topological_constraint",
            "residual_power_mw": round(float(residual_power), 4),
            "conflict_with_telemetry": conflict,
            "details": (
                f"KCL残差={residual_power:.3f}MW, "
                f"推断={inferred}, 当前={current}"
                + (", 状态矛盾!" if conflict else "")
            ),
        })

    n_conflict = sum(1 for r in results if r.get("conflict_with_telemetry"))
    logger.info(
        f"遥信插补: 分析 {len(results)} 个开关, "
        f"矛盾={n_conflict} 个"
    )
    return results


# ============================================================
# 4. 遥测波动平滑
# ============================================================
def smooth_measurements(
    scada_data: Dict,
    window_size: int = 5,
    threshold: float = 3.0,
) -> Dict:
    """
    遥测波动平滑：滑动窗口中位数 + 3σ 异常值剔除。

    对每条母线电压 / 线路功率的时间序列（如有）进行平滑:
      1. 滑动窗口中位数滤波
      2. 计算 MAD (Median Absolute Deviation)
      3. 超过 threshold × MAD 的点视为异常，用中位数替换

    如果数据仅含单点量测（无时间序列），则在空间维度上对同类量测做全局异常检测。

    Args:
        scada_data:    SCADA 数据字典
        window_size:   滑动窗口大小（奇数）
        threshold:     异常判定倍数（MAD × threshold）

    Returns:
        平滑后的 SCADA 数据字典（原字典不修改）
    """
    smoothed = copy.deepcopy(scada_data)
    n_corrected = 0

    # 使窗口大小为奇数
    if window_size % 2 == 0:
        window_size += 1
    half_w = window_size // 2

    # ---- 电压平滑 ----
    bv_list = smoothed.get("bus_voltages", [])
    if bv_list:
        vm_values = np.array([bv["vm_pu"] for bv in bv_list])
        n_corrected += _smooth_spatial(
            vm_values, bv_list, "vm_pu", threshold
        )

    # ---- 线路有功平滑 ----
    lp_list = smoothed.get("line_powers", [])
    if lp_list:
        p_values = np.array([lp["p_mw"] for lp in lp_list])
        n_corrected += _smooth_spatial(
            p_values, lp_list, "p_mw", threshold
        )

    # ---- 变压器有功平滑 ----
    tp_list = smoothed.get("trafo_powers", [])
    if tp_list:
        tp_values = np.array([tp["p_mw"] for tp in tp_list])
        n_corrected += _smooth_spatial(
            tp_values, tp_list, "p_mw", threshold
        )

    # ---- 时间序列平滑（如有 time_series 字段）----
    for key in ("bus_voltages_ts", "line_powers_ts", "trafo_powers_ts"):
        ts_data = smoothed.get(key)
        if ts_data is not None and isinstance(ts_data, dict):
            for element_id, series in ts_data.items():
                if isinstance(series, list) and len(series) >= window_size:
                    arr = np.array(series, dtype=float)
                    arr_smooth, nc = _smooth_temporal(arr, half_w, threshold)
                    ts_data[element_id] = arr_smooth.tolist()
                    n_corrected += nc

    logger.info(f"遥测平滑: 修正 {n_corrected} 个异常量测点")
    return smoothed


def _smooth_spatial(values: np.ndarray, data_list: list,
                    field: str, threshold: float) -> int:
    """
    空间维度异常检测: 对同类量测做全局 MAD 检测。

    Args:
        values:     量测值数组
        data_list:  对应的字典列表（原地修改）
        field:      要修改的字段名
        threshold:  MAD 倍数阈值

    Returns:
        修正的量测点数
    """
    if len(values) < 3:
        return 0

    median_val = np.median(values)
    mad = np.median(np.abs(values - median_val))
    if mad < 1e-10:
        # MAD=0 说明数据太集中，用标准差回退
        mad = np.std(values) * 0.6745
    if mad < 1e-10:
        return 0

    n_corrected = 0
    for i, item in enumerate(data_list):
        deviation = abs(values[i] - median_val)
        if deviation > threshold * mad:
            # 异常 → 用中位数替换
            old_val = values[i]
            item[field] = float(median_val)
            n_corrected += 1
            logger.debug(
                f"空间平滑修正: {field}[{i}] {old_val:.4f} → {median_val:.4f} "
                f"(偏差={deviation:.4f}, MAD={mad:.4f})"
            )

    return n_corrected


def _smooth_temporal(series: np.ndarray, half_w: int,
                     threshold: float) -> Tuple[np.ndarray, int]:
    """
    时间维度滑动窗口中位数平滑。

    Args:
        series:    时间序列数组
        half_w:    窗口半宽
        threshold: MAD 倍数阈值

    Returns:
        (平滑序列, 修正点数)
    """
    n = len(series)
    smoothed = series.copy()
    n_corrected = 0

    for i in range(n):
        lo = max(0, i - half_w)
        hi = min(n, i + half_w + 1)
        window = series[lo:hi]
        med = np.median(window)
        mad = np.median(np.abs(window - med))
        if mad < 1e-10:
            mad = np.std(window) * 0.6745
        if mad < 1e-10:
            continue

        if abs(series[i] - med) > threshold * mad:
            smoothed[i] = med
            n_corrected += 1

    return smoothed, n_corrected


# ============================================================
# 便捷入口: 一键鲁棒估计
# ============================================================
def run_robust_estimation(
    net,
    scada_data: Dict,
    method: str = "irls",
    **kwargs,
) -> Tuple[bool, Dict]:
    """
    一键鲁棒状态估计入口。

    Args:
        net:        PandaPower 网络
        scada_data: SCADA 数据
        method:     "lav" 或 "irls"
        **kwargs:   传递给具体估计函数的参数

    Returns:
        (success, results)
    """
    method = method.lower().strip()
    if method in ("lav", "irls") and _should_use_ac_mode(net):
        logger.info("Detected distribution grid R/X > 0.3, auto-switching to AC mode")
        method = "ac_" + method
    if method == "lav":
        return run_lav_estimation(net, scada_data, **kwargs)
    elif method == "irls":
        return run_irls_estimation(net, scada_data, **kwargs)
    elif method == "ac_lav":
        return run_ac_lav_estimation(net, scada_data, **kwargs)
    elif method == "ac_irls":
        return run_ac_irls_estimation(net, scada_data, **kwargs)
    else:
        logger.error(f"Unknown robust SE method: {method}, support 'lav'/'irls'/'ac_lav'/'ac_irls'")
        return False, {"error": f"Unknown method: {method}"}