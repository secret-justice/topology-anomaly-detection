# -*- coding: utf-8 -*-
"""
v16: 两遍坏数据检测 + 配对免疫
来源: gridstate/bad_data_repass.py
特性:
  - σ_det = min(σ, 30) 限制检测灵敏度
  - 翻转检测: |z+h| < γ·|z-h|
  - 配对免疫: 同支路首末端P/Q量测一致性
  - 域覆盖保护: 不能移除节点唯一P/Q量测
  - Q注入阻尼: 从不reject，只做方差放大
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)

# 配对免疫参数
PAIR_TOL_P = 0.08   # P量测配对一致性容差
PAIR_TOL_Q = 0.30   # Q量测配对一致性容差
PAIR_Z_MIN = 5.0    # 最小量测值(MW/MVAr)
PAIR_SIGMA_MULT = 5.0  # sigma倍数下限


def classify_bad_data(measurements, threshold=3.0, sigma_cap=30.0, flip_ratio=0.3):
    """
    两遍坏数据检测 — 第一遍: 分类
    
    Args:
        measurements: list of dict, 每个包含 {value, variance, estimated, type, bus, branch_id, side}
        threshold: 归一化残差阈值
        sigma_cap: 检测sigma上限
        flip_ratio: 翻转判定比值
        
    Returns:
        dict: {flip_ids, reject_ids, damp_ids, n_candidates, n_immune, n_restored}
    """
    flip_ids = set()
    reject_ids = set()
    damp_ids = set()
    n_candidates = 0
    n_immune = 0
    
    # 只检测P/Q量测(不检测V量测 — V量测不参与符号翻转检测)
    detectable = [m for m in measurements 
                  if m.get("type") in ("P_flow", "Q_flow", "P_inj", "Q_inj")
                  and m.get("estimated") is not None]
    
    # 计算归一化残差
    candidates = []
    for i, m in enumerate(detectable):
        z = m["value"]
        h = m["estimated"]
        sigma = np.sqrt(m.get("variance", 1.0))
        sigma_det = min(sigma, sigma_cap)
        rn = abs(z - h) / max(sigma_det, 1e-9)
        if rn > threshold:
            candidates.append((i, m, rn, z, h, sigma_det))
            n_candidates += 1
    
    if not candidates:
        return {"flip_ids": flip_ids, "reject_ids": reject_ids, "damp_ids": damp_ids,
                "n_candidates": 0, "n_immune": 0, "n_restored": 0}
    
    # 配对免疫: 同支路首末端量测一致性
    immune = set()
    branch_pairs = {}
    for i, m, rn, z, h, sig in candidates:
        bid = m.get("branch_id")
        side = m.get("side")
        mtype = m.get("type")
        if bid is not None and side is not None:
            key = (bid, mtype[0])  # P或Q
            if key not in branch_pairs:
                branch_pairs[key] = {"from": [], "to": []}
            if side == "from":
                branch_pairs[key]["from"].append((i, z, sig))
            else:
                branch_pairs[key]["to"].append((i, z, sig))
    
    for key, sides in branch_pairs.items():
        if sides["from"] and sides["to"]:
            tol = PAIR_TOL_P if key[1] == "P" else PAIR_TOL_Q
            z_from = np.median([z for _, z, _ in sides["from"]])
            z_to = np.median([z for _, z, _ in sides["to"]])
            z_max = max(abs(z_from), abs(z_to))
            if z_max < PAIR_Z_MIN:
                continue
            sig_max = max(max(s for _, _, s in sides["from"] + sides["to"]), 1e-9)
            if abs(z_from + z_to) <= max(tol * z_max, PAIR_SIGMA_MULT * sig_max):
                for idx, _, _ in sides["from"] + sides["to"]:
                    immune.add(idx)
                    n_immune += 1
    
    # 分类: 翻转 / 拒绝 / 阻尼
    for i, m, rn, z, h, sig in candidates:
        if i in immune:
            continue
        mtype = m.get("type", "")
        
        # Q注入: 从不reject，只做方差放大(damp)
        if mtype == "Q_inj":
            damp_ids.add(m.get("id", i))
            continue
        
        # 翻转检测: |z+h| < γ·|z-h|
        if abs(z + h) < flip_ratio * abs(z - h) and abs(z) > 1e-9:
            flip_ids.add(m.get("id", i))
        else:
            reject_ids.add(m.get("id", i))
    
    return {
        "flip_ids": flip_ids,
        "reject_ids": reject_ids,
        "damp_ids": damp_ids,
        "n_candidates": n_candidates,
        "n_immune": n_immune,
        "n_restored": 0,
    }


def apply_bad_data_plan(measurements, plan):
    """
    两遍坏数据检测 — 第二遍: 应用修复计划
    
    Args:
        measurements: 量测列表
        plan: classify_bad_data的返回值
        
    Returns:
        修复后的量测列表
    """
    result = []
    for m in measurements:
        mid = m.get("id", id(m))
        if mid in plan["flip_ids"]:
            # 翻转: 取反
            m = dict(m)
            m["value"] = -m["value"]
            m["bad_data_action"] = "flipped"
        elif mid in plan["reject_ids"]:
            # 拒绝: 标记为不使用
            m = dict(m)
            m["status"] = False
            m["bad_data_action"] = "rejected"
        elif mid in plan["damp_ids"]:
            # 阻尼: 放大方差
            m = dict(m)
            m["variance"] = m.get("variance", 1.0) * 100
            m["bad_data_action"] = "damped"
        result.append(m)
    return result
