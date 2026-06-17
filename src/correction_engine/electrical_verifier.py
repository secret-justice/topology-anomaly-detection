# -*- coding: utf-8 -*-
"""
电气约束验证器
修正后自动运行KCL/KVL校验 + 状态估计验证
确保修正方案不会引入新的错误
"""
import copy
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class ElectricalVerifier:
    """电气约束验证器: 每个修正方案执行6项检查"""

    def __init__(self, net=None, thresholds=None):
        self.net = net
        self.verification_results = []
        self.thresholds = thresholds or {"kcl_p_mw": 0.5, "kcl_q_mvar": 0.5, "line_loading_pct": 100.0}

    def verify_correction(self, original_net, corrected_net, correction: Dict) -> Dict:
        """验证单个修正方案的电气合理性"""
        result = {"correction_id": correction.get("id", ""), "passed": True, "checks": {}, "warnings": []}
        result["checks"]["connectivity"] = self._check_connectivity(corrected_net)
        result["checks"]["radial"] = self._check_radial(corrected_net)
        result["checks"]["powerflow"] = self._check_powerflow(corrected_net)
        result["checks"]["voltage"] = self._check_voltage_range(corrected_net)
        result["checks"]["kcl"] = self._check_kcl_balance(corrected_net)
        result["checks"]["minimal_change"] = self._check_minimal_change(original_net, corrected_net)
        result["checks"]["line_capacity"] = self._check_line_capacity(corrected_net)
        failed = [k for k, v in result["checks"].items() if not v.get("passed", True)]
        result["passed"] = len(failed) == 0
        if failed:
            result["warnings"] = ["验证失败: " + ", ".join(failed)]
        self.verification_results.append(result)
        return result

    def _check_connectivity(self, net) -> Dict:
        try:
            import networkx as nx
            G = nx.Graph()
            for idx in net.line.index:
                if net.line.at[idx, "in_service"]:
                    G.add_edge(int(net.line.at[idx, "from_bus"]), int(net.line.at[idx, "to_bus"]))
            if G.number_of_nodes() == 0:
                return {"passed": True, "detail": "no lines"}
            comps = list(nx.connected_components(G))
            return {"passed": len(comps) <= 1, "components": len(comps)}
        except Exception as e:
            return {"passed": False, "detail": str(e)[:50]}

    def _check_radial(self, net) -> Dict:
        try:
            import networkx as nx
            G = nx.Graph()
            for idx in net.line.index:
                if net.line.at[idx, "in_service"]:
                    G.add_edge(int(net.line.at[idx, "from_bus"]), int(net.line.at[idx, "to_bus"]))
            if G.number_of_nodes() == 0:
                return {"passed": True}
            is_tree = nx.is_tree(G)
            cycles = nx.cycle_basis(G) if not is_tree else []
            return {"passed": len(cycles) <= 1, "is_tree": is_tree, "cycles": len(cycles)}
        except Exception as e:
            return {"passed": False, "detail": str(e)[:50]}

    def _check_powerflow(self, net) -> Dict:
        try:
            import pandapower as pp
            nc = copy.deepcopy(net)
            pp.runpp(nc)
            return {"passed": nc._ppc.get("success", False)}
        except Exception as e:
            return {"passed": False, "detail": str(e)[:40]}

    def _check_voltage_range(self, net) -> Dict:
        try:
            import pandapower as pp
            nc = copy.deepcopy(net)
            pp.runpp(nc)
            vmin, vmax = float(nc.res_bus.vm_pu.min()), float(nc.res_bus.vm_pu.max())
            return {"passed": vmin >= 0.85 and vmax <= 1.15, "v_min": vmin, "v_max": vmax}
        except Exception as e:
            return {"passed": False, "detail": str(e)[:50]}

    def _check_kcl_balance(self, net) -> Dict:
        """Full KCL check: both P and Q balance at each bus, with configurable threshold."""
        try:
            import pandapower as pp
            nc = copy.deepcopy(net)
            pp.runpp(nc)
            violations = []
            p_threshold = self.thresholds.get("kcl_p_mw", 0.5)
            q_threshold = self.thresholds.get("kcl_q_mvar", 0.5)
            for idx in nc.bus.index:
                p_inj, q_inj = 0.0, 0.0
                # Loads (negative injection)
                for li in nc.load.index:
                    if nc.load.at[li, "bus"] == idx and nc.load.at[li, "in_service"]:
                        p_inj -= float(nc.load.at[li, "p_mw"])
                        q_mw = nc.load.at[li, "q_mvar"] if "q_mvar" in nc.load.columns else 0.0
                        q_inj -= float(q_mw)
                # Generators (positive injection)
                for gi in nc.gen.index:
                    if nc.gen.at[gi, "bus"] == idx and nc.gen.at[gi, "in_service"]:
                        p_inj += float(nc.gen.at[gi, "p_mw"])
                        q_mw = nc.gen.at[gi, "q_mvar"] if "q_mvar" in nc.gen.columns else 0.0
                        q_inj += float(q_mw)
                # Ext grid
                for ei in nc.ext_grid.index:
                    if nc.ext_grid.at[ei, "bus"] == idx:
                        # ext_grid absorbs remaining
                        pass
                # Line flows (from res_line)
                for li in nc.res_line.index:
                    fb = int(nc.line.at[li, "from_bus"])
                    tb = int(nc.line.at[li, "to_bus"])
                    pf = float(nc.res_line.at[li, "p_mw_from"])
                    qf = float(nc.res_line.at[li, "q_mvar_from"])
                    pt = float(nc.res_line.at[li, "p_mw_to"])
                    qt = float(nc.res_line.at[li, "q_mvar_to"])
                    if fb == idx:
                        p_inj -= pf
                        q_inj -= qf
                    if tb == idx:
                        p_inj -= pt
                        q_inj -= qt
                if abs(p_inj) > p_threshold or abs(q_inj) > q_threshold:
                    violations.append({
                        "bus": int(idx),
                        "p_inject": round(p_inj, 4),
                        "q_inject": round(q_inj, 4),
                    })
            return {"passed": len(violations) == 0, "violations": violations[:10]}
        except Exception as e:
            return {"passed": False, "detail": str(e)[:50]}

    def _check_minimal_change(self, original, corrected) -> Dict:
        try:
            changes = 0
            for idx in original.line.index:
                if idx in corrected.line.index:
                    if original.line.at[idx, "in_service"] != corrected.line.at[idx, "in_service"]:
                        changes += 1
                    if original.line.at[idx, "to_bus"] != corrected.line.at[idx, "to_bus"]:
                        changes += 1
            return {"passed": changes <= 5, "total_changes": changes}
        except Exception as e:
            return {"passed": False, "detail": str(e)[:50]}

    def _check_line_capacity(self, net) -> Dict:
        """Check line loading against thermal rating (max_i_ka)."""
        try:
            import pandapower as pp
            nc = copy.deepcopy(net)
            pp.runpp(nc)
            overloaded = []
            threshold = self.thresholds.get("line_loading_pct", 100.0)
            for idx in nc.res_line.index:
                loading = float(nc.res_line.at[idx, "loading_percent"])
                if loading > threshold:
                    fb = int(nc.line.at[idx, "from_bus"])
                    tb = int(nc.line.at[idx, "to_bus"])
                    max_i = float(nc.line.at[idx, "max_i_ka"]) if "max_i_ka" in nc.line.columns else 0.0
                    overloaded.append({
                        "line": int(idx),
                        "from_bus": fb,
                        "to_bus": tb,
                        "loading_pct": round(loading, 1),
                        "max_i_ka": max_i,
                    })
            return {"passed": len(overloaded) == 0, "overloaded": overloaded[:10]}
        except Exception as e:
            return {"passed": False, "detail": str(e)[:50]}



    def run_n1_security_check(self, net=None, max_contingencies=20):
        """N-1 security analysis: check if network survives single-element outage.
        
        Tests each line/trafo outage and checks:
        1. Network remains connected
        2. Voltage stays within [0.90, 1.10] pu
        3. Line loading < 100%
        
        Args:
            net: PandaPower network (uses self.net if None)
            max_contingencies: max number of contingencies to test
        
        Returns:
            dict with secure_count, insecure_count, violations
        """
        import copy
        import pandapower as pp
        import networkx as nx
        
        target = net or self.net
        if target is None:
            return {"error": "no network"}
        
        secure = 0
        insecure = 0
        violations = []
        
        # Test line outages
        line_indices = list(target.line.index)[:max_contingencies]
        for idx in line_indices:
            net_c = copy.deepcopy(target)
            net_c.line.at[idx, "in_service"] = False
            try:
                # Check connectivity
                G = nx.Graph()
                for li in net_c.line.index:
                    if net_c.line.at[li, "in_service"]:
                        G.add_edge(int(net_c.line.at[li, "from_bus"]),
                                   int(net_c.line.at[li, "to_bus"]))
                if G.number_of_nodes() > 1 and not nx.is_connected(G):
                    insecure += 1
                    violations.append({
                        "element": f"line_{idx}",
                        "type": "connectivity_loss",
                        "severity": "critical",
                    })
                    continue
                
                pp.runpp(net_c)
                vmin = float(net_c.res_bus.vm_pu.min())
                vmax = float(net_c.res_bus.vm_pu.max())
                max_loading = float(net_c.res_line.loading_percent.max()) if len(net_c.res_line) > 0 else 0
                
                if vmin < 0.90 or vmax > 1.10 or max_loading > 100:
                    insecure += 1
                    violations.append({
                        "element": f"line_{idx}",
                        "type": "voltage_violation" if (vmin < 0.90 or vmax > 1.10) else "thermal_violation",
                        "vmin": vmin, "vmax": vmax, "max_loading": max_loading,
                    })
                else:
                    secure += 1
            except Exception:
                insecure += 1
                violations.append({"element": f"line_{idx}", "type": "powerflow_failed"})
        
        total = secure + insecure
        result = {
            "secure_count": secure,
            "insecure_count": insecure,
            "total_tested": total,
            "security_ratio": secure / max(total, 1),
            "violations": violations[:20],
        }
        
        if hasattr(self, "verification_results"):
            self.verification_results.append({"check": "n1_security", **result})
        
        return result

    def get_summary(self) -> Dict:
        total = len(self.verification_results)
        passed = sum(1 for r in self.verification_results if r["passed"])
        return {"total": total, "passed": passed, "failed": total - passed, "pass_rate": passed / max(total, 1)}
