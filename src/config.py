# -*- coding: utf-8 -*-
"""
全局配置模块
配电网图模拓扑智能识别与修正 - MVP
"""
import os
from pathlib import Path

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = Path(r"E:\项目大全\电力拓扑图修正")
DATA_ROOT = PROJECT_ROOT / "07_参考文献"
OUTPUT_ROOT = PROJECT_ROOT / "02_算法代码" / "output"

# CIM示例数据目录
CIM_DATA_DIR = DATA_ROOT / "CIM_示例数据"
CIMHUB_DIR = CIM_DATA_DIR / "CIMHub-master"

# IEEE测试馈线数据
IEEE_FEEDER_DIR = DATA_ROOT / "IEEE_PES测试馈线数据"

# OpenDSS数据
OPENDSS_DIR = DATA_ROOT / "OpenDSS_IEEE测试馈线"

# PandaPower JSON模型
PP_EXAMPLE_JSON = DATA_ROOT / "example_simple_pandapower.json"
PP_MV_JSON = DATA_ROOT / "mv_oberrhein_pandapower.json"

# 输出目录（自动创建）
for d in [OUTPUT_ROOT]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# 异常检测阈值
# ============================================================
THRESHOLDS = {
    # WLS状态估计 - chi2全局检验
    "chi2_alpha": 0.05,                    # 显著性水平
    "chi2_confidence": 0.95,               # 置信度

    # 标准化残差
    "normalized_residual_threshold": 3.0,  # 超过此值判定为不良数据
    "residual_warning_threshold": 2.0,     # 预警阈值

    # 规则引擎
    "voltage_pu_max": 1.10,                # 电压标幺值上限
    "voltage_pu_min": 0.90,                # 电压标幺值下限
    "line_loading_max": 1.0,               # 线路负载率上限
    "connectivity_degree_max": 20,         # 节点最大度数（超过视为异常）
    "connectivity_degree_min": 0,          # 孤立节点

    # SCADA仿真噪声
    "scada_voltage_noise_sigma": 0.005,    # 电压量测噪声标准差(p.u.)
    "scada_power_noise_sigma": 0.05,  # 5% of measured value (MW)       # 功率量测噪声标准差(p.u.)
    "scada_current_noise_sigma": 0.01,     # 电流量测噪声标准差

    # 数据对齐
    "id_match_fuzzy_threshold": 0.85,      # 模糊匹配阈值
}

# ============================================================
# 异常类型定义
# ============================================================
ANOMALY_TYPES = {
    "MODEL_MISMATCH":    "图模不符",       # 类型1: CIM模型与SVG图形不一致
    "TOPO_INTERRUPT":    "拓扑中断",       # 类型2: 拓扑不连通/存在孤立节点
    "VIRTUAL_FAULTY":    "虚接/错接",      # 类型3: 连接关系错误
    "TELE_TOPO_MISMATCH":"遥测!=拓扑",     # 类型4: 量测与拓扑模型矛盾
    "TELE_SIGNAL_MISMATCH":"遥信!=遥测",   # 类型5: 遥信状态与量测不一致
}


# ============================================================
# PandaPower内置网络列表（可用于测试）
# ============================================================
PANDAPOWER_NETWORKS = {
    "case33bw":                    "pandapower.networks.case33bw",               # IEEE 33节点
    "case_ieee30":                 "pandapower.networks.case_ieee30",            # IEEE 30节点
    "case118":                     "pandapower.networks.case118",                # IEEE 118节点
    "example_simple":              "pandapower.networks.example_simple",          # 简单7节点
    "example_multivoltage":        "pandapower.networks.example_multivoltage",    # 多电压57节点
    "cigre_mv":                    "pandapower.networks.create_cigre_network_mv", # CIGRE中压
    "cigre_lv":                    "pandapower.networks.create_cigre_network_lv", # CIGRE低压
    "kerber_dorfnetz":             "pandapower.networks.create_kerber_dorfnetz",  # Kerber村庄116节点
    "kerber_landnetz_freileitung": "pandapower.networks.create_kerber_landnetz_freileitung_1",
    "kerber_landnetz_kabel":       "pandapower.networks.create_kerber_landnetz_kabel_1",
    "kerber_vorstadtnetz_kabel":   "pandapower.networks.create_kerber_vorstadtnetz_kabel_1",  # 294节点
    "dickert_lv":                  "pandapower.networks.create_dickert_lv_network",
}

# 默认测试网络
DEFAULT_TEST_NETWORK = "example_simple"
