# -*- coding: utf-8 -*-
"""
SCADA 数据仿真器
基于 PandaPower 潮流计算生成仿真量测，并可注入5类拓扑异常
"""
import numpy as np
import pandapower as pp
import pandapower.estimation as ppest
import logging
from typing import Dict, List, Optional, Tuple
import copy

logger = logging.getLogger(__name__)


class SCADASimulator:
    """SCADA仿真器：生成带噪声的量测数据，支持注入异常"""

    def __init__(self, net: pp.auxiliary.pandapowerNet,
                 voltage_noise: float = 0.005,
                 power_noise: float = 0.02,
                 current_noise: float = 0.01,
                 seed: int = 42):
        """
        Args:
            net: PandaPower网络
            voltage_noise: 电压量测噪声标准差(p.u.)
            power_noise: 功率量测噪声标准差(p.u.)
            current_noise: 电流量测噪声标准差
            seed: 随机种子
        """
        self.net = net
        self.rng = np.random.default_rng(seed)
        self.noise = {
            "voltage": voltage_noise,
            "power": power_noise,
            "current": current_noise,
        }
        self._run_powerflow()

    def _run_powerflow(self):
        """运行潮流计算作为真值基准"""
        try:
            try:
                pp.runpp(self.net, calculate_voltage_angles=True)
            except Exception:
                pp.runpp(self.net, calculate_voltage_angles=False)
            self.pf_converged = True
            logger.info("潮流计算收敛")
        except Exception as e:
            logger.warning(f"潮流计算未收敛: {e}")
            self.pf_converged = False

    def generate_measurements(self) -> Dict:
        """
        生成仿真量测数据（带高斯噪声模拟SCADA精度）

        返回:
          bus_voltages  - [{bus, vm_pu, va_degree, sigma}]
          line_powers  - [{line, from_bus, to_bus, p_mw, q_mvar, side, sigma}]
          trafo_powers - [{trafo, hv_bus, lv_bus, p_mw, q_mvar, side, sigma}]
          metadata     - 网络基本信息
        """
        net = self.net
        bus_voltages = []
        line_powers = []
        trafo_powers = []

        # 电压量测（每条母线）
        for idx in net.bus.index:
            if idx in net.res_bus.index and not np.isnan(net.res_bus.vm_pu.at[idx]):
                vm_true = net.res_bus.vm_pu.at[idx]
                va_true = net.res_bus.va_degree.at[idx]
                vm_measured = vm_true + self.rng.normal(0, self.noise["voltage"])
                bus_voltages.append({
                    "bus": int(idx),
                    "vm_pu": float(vm_measured),
                    "vm_pu_true": float(vm_true),
                    "va_degree": float(va_true + self.rng.normal(0, 0.1)),
                    "sigma": self.noise["voltage"],
                })

        # 线路功率量测（首末端）
        for side in ("from", "to"):
            for idx in net.line.index:
                if idx in net.res_line.index:
                    p_col = f"p_{side}w" if f"p_{side}w" in net.res_line.columns else "p_mw"
                    q_col = f"q_{side}var" if f"q_{side}var" in net.res_line.columns else "q_mvar"
                    # PandaPower列名: p_from_mw, q_from_mvar, p_to_mw, q_to_mvar
                    p_key = f"p_{side}_mw"
                    q_key = f"q_{side}_mvar"
                    if p_key in net.res_line.columns:
                        p_true = net.res_line.at[idx, p_key]
                        q_true = net.res_line.at[idx, q_key]
                        p_sigma = max(abs(p_true) * self.noise["power"], 0.01)
                        q_sigma = max(abs(q_true) * self.noise["power"], 0.005)
                        p_m = p_true + self.rng.normal(0, p_sigma)
                        q_m = q_true + self.rng.normal(0, q_sigma)
                        line_powers.append({
                            "line": int(idx),
                            "from_bus": int(net.line.at[idx, "from_bus"]),
                            "to_bus": int(net.line.at[idx, "to_bus"]),
                            "p_mw": float(p_m),
                            "q_mvar": float(q_m),
                            "p_mw_true": float(p_true),
                            "q_mvar_true": float(q_true),
                            "side": side,
                            "sigma": float(p_sigma),
                        })

        # 变压器功率量测
        for side in ("hv", "lv"):
            for idx in net.trafo.index:
                if idx in net.res_trafo.index:
                    p_key = f"p_{side}_mw"
                    q_key = f"q_{side}_mvar"
                    if p_key in net.res_trafo.columns:
                        p_true = net.res_trafo.at[idx, p_key]
                        q_true = net.res_trafo.at[idx, q_key]
                        p_sigma = max(abs(p_true) * self.noise["power"], 0.01)
                        q_sigma = max(abs(q_true) * self.noise["power"], 0.005)
                        p_m = p_true + self.rng.normal(0, p_sigma)
                        q_m = q_true + self.rng.normal(0, q_sigma)
                        trafo_powers.append({
                            "trafo": int(idx),
                            "hv_bus": int(net.trafo.at[idx, "hv_bus"]),
                            "lv_bus": int(net.trafo.at[idx, "lv_bus"]),
                            "p_mw": float(p_m),
                            "q_mvar": float(q_m),
                            "p_mw_true": float(p_true),
                            "q_mvar_true": float(q_true),
                            "side": side,
                            "sigma": float(p_sigma),
                        })

        metadata = {
            "n_bus": len(net.bus),
            "n_line": len(net.line),
            "n_trafo": len(net.trafo),
            "n_load": len(net.load),
            "n_gen": len(net.gen),
            "n_switch": len(net.switch),
            "pf_converged": self.pf_converged,
        }

        logger.info(f"量测生成完成: 电压={len(bus_voltages)}, "
                    f"线路功率={len(line_powers)}, 变压器功率={len(trafo_powers)}")

        return {
            "bus_voltages": bus_voltages,
            "line_powers": line_powers,
            "trafo_powers": trafo_powers,
            "metadata": metadata,
        }

    def inject_topology_interrupt(self, line_idx: int) -> Dict:
        """
        注入异常类型2: 拓扑中断
        将指定线路设为 out_of_service
        """
        net_copy = copy.deepcopy(self.net)
        if line_idx in net_copy.line.index:
            net_copy.line.at[line_idx, "in_service"] = False
            logger.info(f"注入拓扑中断: 线路 {line_idx} 停运")
        return {"type": "拓扑中断", "element": "line", "index": line_idx,
                "net": net_copy}

    def inject_virtual_faulty_connection(self, line_idx: int,
                                         new_to_bus: int) -> Dict:
        """
        注入异常类型3: 虚接/错接
        修改线路末端母线（模拟连接到错误节点）
        """
        net_copy = copy.deepcopy(self.net)
        old_bus = int(net_copy.line.at[line_idx, "to_bus"])
        net_copy.line.at[line_idx, "to_bus"] = new_to_bus
        logger.info(f"注入虚接/错接: 线路{line_idx} 终端 {old_bus}->{new_to_bus}")
        return {"type": "虚接/错接", "element": "line", "index": line_idx,
                "old_to_bus": old_bus, "new_to_bus": new_to_bus,
                "net": net_copy}

    def inject_measurement_error(self, measurements: Dict,
                                  meas_type: str = "bus_voltages",
                                  index: int = 0,
                                  error_factor: float = 1.5) -> Dict:
        """
        注入异常类型4: 遥测!=拓扑
        对指定量测施加大幅偏差（模拟传感器故障）
        """
        meas_copy = copy.deepcopy(measurements)
        target_list = meas_copy[meas_type]
        if index < len(target_list):
            item = target_list[index]
            # 施加大偏差
            if "vm_pu" in item:
                item["vm_pu"] += error_factor
                logger.info(f"注入遥测异常: {meas_type}[{index}] "
                           f"电压偏移 {error_factor}")
            elif "p_mw" in item:
                item["p_mw"] *= error_factor
                logger.info(f"注入遥测异常: {meas_type}[{index}] "
                           f"功率偏移 ×{error_factor}")
        return {"type": "遥测!=拓扑", "injected_to": meas_type,
                "index": index, "measurements": meas_copy}

    def inject_switch_state_mismatch(self, switch_idx: int,
                                      actual_open: bool) -> Dict:
        """
        注入异常类型5: 遥信!=遥测
        开关遥信状态与实际电气状态矛盾
        遥信说闭合但实际量测显示断开（或反之）
        """
        net_copy = copy.deepcopy(self.net)
        if switch_idx in net_copy.switch.index:
            net_copy.switch.at[switch_idx, "closed"] = not actual_open
            logger.info(f"注入遥信!=遥测: 开关{switch_idx} "
                       f"遥信={'断开' if not actual_open else '闭合'} "
                       f"实际={'断开' if actual_open else '闭合'}")
        return {"type": "遥信!=遥测", "switch_idx": switch_idx,
                "reported_closed": not actual_open,
                "actual_open": actual_open, "net": net_copy}

    def inject_model_mismatch(self, bus_idx: int,
                               svg_device_count: int) -> Dict:
        """
        注入异常类型1: 图模不符
        模拟CIM有某设备但SVG中缺失（或反之）
        通过返回不一致的数据来模拟
        """
        logger.info(f"注入图模不符: 总线{bus_idx} 在SVG中设备数={svg_device_count}")
        return {"type": "图模不符", "bus": bus_idx,
                "cim_device_count": 1, "svg_device_count": svg_device_count}


def load_pandapower_network(network_name: str = "example_simple") -> pp.auxiliary.pandapowerNet:
    """
    加载PandaPower内置网络

    Args:
        network_name: 网络名称，见 config.PANDAPOWER_NETWORKS
    """
    import importlib
    module_name = f"pandapower.networks.{network_name}"
    try:
        mod = importlib.import_module(module_name)
        # 函数名通常与模块后缀同名
        func = getattr(mod, network_name)
        net = func()
        logger.info(f"加载网络 {network_name}: 母线={len(net.bus)}, "
                    f"线路={len(net.line)}, 变压器={len(net.trafo)}")
        return net
    except (ImportError, AttributeError):
        # 回退: 直接用 networks 模块的函数
        import pandapower.networks as pn
        func = getattr(pn, network_name, None)
        if func:
            return func()
        raise ValueError(f"未知网络: {network_name}")

# ===== P2-5: Three-Phase SCADA Simulation =====

class ThreePhaseSCADASimulator:
    """Three-phase unbalanced SCADA simulation.
    
    Generates per-phase measurements (Va, Vb, Vc, Pa, Pb, Pc, Qa, Qb, Qc)
    with realistic unbalance and noise.
    
    Key for distribution grid accuracy: most distribution systems are
    inherently unbalanced (single-phase laterals, uneven load allocation).
    """
    
    def __init__(self, net, seed=42, unbalance_ratio=0.05):
        """
        Args:
            net: PandaPower network
            seed: random seed
            unbalance_ratio: typical voltage unbalance (5% default)
        """
        self.net = net
        self.rng = np.random.default_rng(seed)
        self.unbalance = unbalance_ratio
    
    def generate_three_phase_measurements(self):
        """Generate three-phase measurements for all buses.
        
        Returns:
            dict with bus_voltages_3ph, line_powers_3ph
        """
        import pandapower as pp
        try:
            pp.runpp(self.net)
        except Exception:
            return {"bus_voltages_3ph": [], "line_powers_3ph": []}
        
        bus_voltages_3ph = []
        for idx in self.net.bus.index:
            vm = float(self.net.res_bus.at[idx, "vm_pu"])
            # Generate per-phase voltages with unbalance
            va = vm * (1 + self.rng.normal(0, self.unbalance))
            vb = vm * (1 + self.rng.normal(0, self.unbalance))
            vc = vm * (1 + self.rng.normal(0, self.unbalance))
            
            bus_voltages_3ph.append({
                "bus": int(idx),
                "vm_a_pu": va,
                "vm_b_pu": vb,
                "vm_c_pu": vc,
                "vm_pu": vm,  # positive sequence (average)
                "unbalance": max(abs(va - vb), abs(vb - vc), abs(va - vc)) / vm,
            })
        
        line_powers_3ph = []
        for idx in self.net.line.index:
            if not self.net.line.at[idx, "in_service"]:
                continue
            pf = float(self.net.res_line.at[idx, "p_from_mw"])
            qf = float(self.net.res_line.at[idx, "q_from_mvar"])
            # Split into 3 phases with unbalance
            pa = pf / 3 * (1 + self.rng.normal(0, self.unbalance))
            pb = pf / 3 * (1 + self.rng.normal(0, self.unbalance))
            pc = pf - pa - pb
            qa = qf / 3 * (1 + self.rng.normal(0, self.unbalance))
            qb = qf / 3 * (1 + self.rng.normal(0, self.unbalance))
            qc = qf - qa - qb
            
            line_powers_3ph.append({
                "line": int(idx),
                "from_bus": int(self.net.line.at[idx, "from_bus"]),
                "to_bus": int(self.net.line.at[idx, "to_bus"]),
                "p_a_mw": pa, "p_b_mw": pb, "p_c_mw": pc,
                "q_a_mvar": qa, "q_b_mvar": qb, "q_c_mvar": qc,
                "p_mw": pf, "q_mvar": qf,
            })
        
        return {
            "bus_voltages_3ph": bus_voltages_3ph,
            "line_powers_3ph": line_powers_3ph,
            "unbalance_summary": {
                "max_unbalance": max(b["unbalance"] for b in bus_voltages_3ph) if bus_voltages_3ph else 0,
                "avg_unbalance": np.mean([b["unbalance"] for b in bus_voltages_3ph]) if bus_voltages_3ph else 0,
            },
        }
