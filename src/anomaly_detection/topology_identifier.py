# -*- coding: utf-8 -*-
"""
第二层扩展: 拓扑错误辨识
当chi2检验未通过且排除不良数据后残差仍偏大，
说明拓扑模型本身有误。通过逐一假设支路拓扑参数错误，
重新估计并比较目标函数，目标函数显著下降的假设即为拓扑错误位置。
支持识别: 断路器状态错误 / 线路虚接 / 错接
适配 pandapower 3.x API
"""
import numpy as np
import pandapower as pp
import pandapower.estimation as ppest
import copy
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class TopologyIdentifier:
    """拓扑错误辨识器"""

    def __init__(self, residual_drop_ratio: float = 0.3,
                 max_candidates: int = 10):
        """
        Args:
            residual_drop_ratio: 目标函数下降比例阈值
                (baseline - new) / baseline > 此值则判定为拓扑错误
            max_candidates: 最多测试的候选支路数
        """
        self.residual_drop_ratio = residual_drop_ratio
        self.max_candidates = max_candidates

    def identify_topology_errors(self,
                                  net: pp.auxiliary.pandapowerNet,
                                  baseline_objective: float) -> List[Dict]:
        """
        拓扑错误辨识主流程

        Args:
            net: PandaPower网络（已注入量测）
            baseline_objective: 基线目标函数值（chi2检验的客观函数值）

        Returns:
            candidates: [{type, element, index, obj_before, obj_after,
                          drop_ratio, description}]
        """
        candidates = []
        hypotheses = self._generate_hypotheses(net)

        logger.info(f"拓扑错误辨识: 共 {len(hypotheses)} 个候选假设")

        for hyp in hypotheses[:self.max_candidates]:
            try:
                new_obj = self._test_hypothesis(net, hyp)
                if new_obj is not None and baseline_objective > 0:
                    drop = (baseline_objective - new_obj) / baseline_objective
                    if drop > self.residual_drop_ratio:
                        candidates.append({
                            "type": hyp["type"],
                            "element": hyp["element"],
                            "index": hyp["index"],
                            "obj_before": round(baseline_objective, 4),
                            "obj_after": round(new_obj, 4),
                            "drop_ratio": round(drop, 4),
                            "description": hyp["description"],
                        })
            except Exception as e:
                logger.debug(f"假设测试失败 [{hyp['description']}]: {e}")

        candidates.sort(key=lambda x: x["drop_ratio"], reverse=True)
        logger.info(f"拓扑错误辨识完成: {len(candidates)} 个候选")
        return candidates

    def _generate_hypotheses(self, net) -> List[Dict]:
        """
        生成候选假设集
        假设类型: 1) 开关状态翻转 2) 线路停运/投运
        """
        hypotheses = []

        # --- 开关状态翻转 ---
        for idx in net.switch.index:
            current_closed = bool(net.switch.at[idx, "closed"])
            hypotheses.append({
                "type": "开关状态翻转",
                "element": "switch",
                "index": int(idx),
                "action": "toggle_switch",
                "new_value": not current_closed,
                "description": (f"开关{idx} 翻转状态: "
                                f"{'闭合' if current_closed else '断开'}"
                                f"-> {'断开' if current_closed else '闭合'}"),
            })

        # --- 线路停运/投运 ---
        for idx in net.line.index:
            in_svc = bool(net.line.at[idx, "in_service"])
            hypotheses.append({
                "type": "线路状态翻转",
                "element": "line",
                "index": int(idx),
                "action": "toggle_line",
                "new_value": not in_svc,
                "description": (f"线路{idx} 翻转状态: "
                                f"{'投运' if in_svc else '停运'}"
                                f"-> {'停运' if in_svc else '投运'}"),
            })

        return hypotheses

    def _test_hypothesis(self, net, hypothesis: Dict) -> Optional[float]:
        """
        测试单个假设，返回测试后目标函数值

        Args:
            net: 原始网络（不会被修改）
            hypothesis: 假设字典

        Returns:
            new_obj: 新目标函数值，失败返回None
        """
        net_copy = copy.deepcopy(net)
        action = hypothesis["action"]
        idx = hypothesis["index"]

        if action == "toggle_switch":
            if idx in net_copy.switch.index:
                net_copy.switch.at[idx, "closed"] = hypothesis["new_value"]
        elif action == "toggle_line":
            if idx in net_copy.line.index:
                net_copy.line.at[idx, "in_service"] = hypothesis["new_value"]
        else:
            return None

        # 重新运行潮流 + 状态估计
        try:
            pp.runpp(net_copy, calculate_voltage_angles=True)
            result = ppest.estimate(
                net_copy, init="flat",
                calculate_voltage_angles=True,
            )
            if isinstance(result, dict) and not result.get("success", False):
                return None
            # 使用 objective_function_value（WLS的目标函数值）
            if isinstance(result, dict):
                return result.get("objective_function_value", None)
            return None
        except Exception:
            return None
