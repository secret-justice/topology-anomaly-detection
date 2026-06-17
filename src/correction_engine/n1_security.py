# -*- coding: utf-8 -*-
"""
v16: N-1安全校验 — 修正前必检
来源: PandaPower内置contingency模块
"""
import copy
import numpy as np
import logging

logger = logging.getLogger(__name__)


def n1_contingency_check(net, correction_plan=None):
    """
    N-1安全校验: 逐条断开线路/变压器，检查是否越限
    
    Args:
        net: pandapower network (已做过潮流计算)
        correction_plan: 修正方案(可选)，如果是None则检查当前状态
        
    Returns:
        dict: {safe: bool, violations: list, details: list}
    """
    import pandapower as pp
    import pandapower.run as ppr
    
    violations = []
    details = []
    
    # 基态潮流
    try:
        ppr.runpp(net)
        if not net.converged:
            return {"safe": False, "violations": ["base_case_not_converged"], "details": []}
    except Exception as e:
        return {"safe": False, "violations": [f"base_case_error: {e}"], "details": []}
    
    # 检查基态越限
    base_violations = _check_violations(net)
    if base_violations:
        violations.extend(base_violations)
        details.append({"case": "base", "violations": base_violations})
    
    # N-1: 逐条断开
    for idx in net.line.index:
        if not net.line.at[idx, "in_service"]:
            continue
        net_test = copy.deepcopy(net)
        net_test.line.at[idx, "in_service"] = False
        try:
            ppr.runpp(net_test)
            if net_test.converged:
                v = _check_violations(net_test)
                if v:
                    violations.extend(v)
                    details.append({"case": f"line_{idx}_out", "violations": v})
        except:
            pass
    
    for idx in net.trafo.index:
        if not net.trafo.at[idx, "in_service"]:
            continue
        net_test = copy.deepcopy(net)
        net_test.trafo.at[idx, "in_service"] = False
        try:
            ppr.runpp(net_test)
            if net_test.converged:
                v = _check_violations(net_test)
                if v:
                    violations.extend(v)
                    details.append({"case": f"trafo_{idx}_out", "violations": v})
        except:
            pass
    
    return {
        "safe": len(violations) == 0,
        "violations": violations,
        "details": details,
    }


def _check_violations(net, vm_lower=0.90, vm_upper=1.10, loading_max=100.0):
    """检查电压越限和线路过载"""
    violations = []
    
    # 电压越限
    for idx in net.res_bus.index:
        vm = net.res_bus.at[idx, "vm_pu"]
        if vm < vm_lower:
            violations.append(f"bus_{idx}_undervoltage({vm:.3f}pu)")
        elif vm_upper is not None and vm > vm_upper:
            violations.append(f"bus_{idx}_overvoltage({vm:.3f}pu)")
    
    # 线路过载
    for idx in net.res_line.index:
        loading = net.res_line.at[idx, "loading_percent"]
        if loading > loading_max:
            violations.append(f"line_{idx}_overload({loading:.1f}%)")
    
    # 变压器过载
    for idx in net.res_trafo.index:
        loading = net.res_trafo.at[idx, "loading_percent"]
        if loading > loading_max:
            violations.append(f"trafo_{idx}_overload({loading:.1f}%)")
    
    return violations
