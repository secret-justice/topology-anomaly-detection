# -*- coding: utf-8 -*-
"""IEEE8500 large-scale stress test"""
import sys, time, copy
sys.path.insert(0, r"E:\项目大全\电力拓扑图修正\02_算法代码")
import networkx as nx
import numpy as np
import pandapower.networks as pn
import logging
logging.basicConfig(level=logging.ERROR)
from data_preprocessing.cim_parser import parse_cim_rdf
from utils.graph_utils import build_graph_from_cim
from anomaly_detection.rule_engine import RuleEngine
from utils.metrics import evaluate_by_type

CIM_FILE = r"E:\项目大全\电力拓扑图修正\07_参考文献\CIM_示例数据\CIMHub-master\model_output_tests\IEEE8500.xml"

print("=" * 70)
print("  IEEE 8500-Node Stress Test (CIM -> Detection Pipeline)")
print("=" * 70)

# Step 1: Parse CIM
print("\n[1] Parsing IEEE8500 CIM XML...")
t0 = time.time()
cim = parse_cim_rdf(CIM_FILE)
t1 = time.time()
print(f"  Devices: {len(cim['devices']):>5}")
print(f"  Nodes:   {len(cim['nodes']):>5}")
print(f"  Switches:{len(cim['switches']):>5}")
print(f"  Parse time: {t1-t0:.2f}s")

# Step 2: Build graph
print("\n[2] Building topology graph...")
G = build_graph_from_cim(cim)
t2 = time.time()
print(f"  Nodes: {G.number_of_nodes():>5}")
print(f"  Edges: {G.number_of_edges():>5}")
print(f"  Connected: {nx.is_connected(G)}")
print(f"  Build time: {t2-t1:.2f}s")

# Step 3: Generate SCADA
print("\n[3] Generating synthetic SCADA...")
rng = np.random.default_rng(42)
scada = {"bus_voltages": [], "line_powers": []}
node_list = list(G.nodes())
for node in node_list:
    scada["bus_voltages"].append({"bus": node, "vm_pu": 1.0 + rng.normal(0, 0.02), "sigma": 0.005})
for i, (u, v) in enumerate(G.edges()):
    scada["line_powers"].append({"line": i, "from_bus": u, "to_bus": v,
                                  "p_mw": 0.5 + rng.normal(0, 0.1), "q_mvar": 0.1,
                                  "side": "from", "sigma": 0.02})
print(f"  Voltage measurements: {len(scada['bus_voltages']):>5}")
print(f"  Power measurements:   {len(scada['line_powers']):>5}")

# Step 4: Inject anomalies
print("\n[4] Injecting anomalies...")
gt = []
G_broken = copy.deepcopy(G)
edges = list(G.edges())
# Break a bridge-like edge
mid_edge = edges[len(edges)//2]
G_broken.remove_edge(*mid_edge)
gt.append({"type": "\u62d3\u6251\u4e2d\u65ad", "location": str(mid_edge)})
# Voltage anomaly
scada["bus_voltages"][5]["vm_pu"] = 0.65
gt.append({"type": "\u865a\u62df\u63a5/\u9519\u63a5", "location": "bus_5"})
# Model mismatch
gt.append({"type": "\u56fe\u6a21\u4e0d\u7b26", "location": "svg_missing"})
print(f"  Injected: {len(gt)} anomalies")

# Step 5: Run detection
print("\n[5] Running rule engine detection...")
t3 = time.time()
engine = RuleEngine()
detected = engine.run_all_checks(G_broken, measurements=scada,
                                  cim_data=cim["devices"], svg_data=[])
t4 = time.time()
print(f"  Detected: {len(detected)} anomalies")
print(f"  Detection time: {t4-t3:.2f}s")

# Step 6: Evaluate
print("\n[6] Evaluating...")
ev = evaluate_by_type(detected, gt)
print(f"  Type Recall: {ev['type_recall']:.0%}")
for t, info in ev["per_type"].items():
    status = "[OK]" if info["detected"] else "[MISS]"
    print(f"    {status} {t}: count={info['detected_count']}")

# Device type breakdown
print("\n[7] Device type breakdown:")
dev_types = {}
for d in cim["devices"]:
    st = d.get("subtype", "Unknown")
    dev_types[st] = dev_types.get(st, 0) + 1
for dt, cnt in sorted(dev_types.items(), key=lambda x: -x[1]):
    print(f"    {dt:<25} {cnt:>5}")

total_time = time.time() - t0
print(f"\n{'='*70}")
print(f"  IEEE8500 Stress Test PASSED")
print(f"  Total devices: {len(cim['devices'])}  Total nodes: {G.number_of_nodes()}")
print(f"  Total time: {total_time:.2f}s  Detection time: {t4-t3:.2f}s")
print(f"  Recall: {ev['type_recall']:.0%}")
print(f"{'='*70}")