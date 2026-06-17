# -*- coding: utf-8 -*-
"""
Comprehensive E2E CIM Pipeline Test
Tests all available IEEE CIM XML files through the full detection pipeline.
"""
import sys, time, json, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import networkx as nx
import logging
logging.basicConfig(level=logging.ERROR)

from data_preprocessing.cim_parser import parse_cim_rdf
from utils.graph_utils import build_graph_from_cim
from anomaly_detection.rule_engine import RuleEngine
from utils.metrics import evaluate_by_type

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CIM_DIR = Path(r"E:\项目大全\电力拓扑图修正\07_参考文献\CIM_示例数据\CIMHub-master\model_output_tests")

CIM_FILES = [
    "IEEE13.xml", "IEEE37.xml", "IEEE123.xml", "IEEE123_PV.xml",
    "IEEE8500.xml", "ACEP_PSIL.xml", "EPRI_DPV_J1.xml", "R2_12_47_2.xml",
    "Transactive.xml",
]


def test_cim_file(filepath, name):
    """Parse and analyze a single CIM file."""
    result = {"name": name, "status": "OK"}
    t0 = time.time()
    
    # Parse CIM
    try:
        cim = parse_cim_rdf(str(filepath))
    except Exception as e:
        result["status"] = f"CIM_PARSE_FAIL: {e}"
        return result
    
    result["devices"] = len(cim["devices"])
    result["terminals"] = len(cim["terminals"])
    result["nodes"] = len(cim["nodes"])
    result["switches"] = len(cim["switches"])
    result["namespace"] = cim.get("cim_namespace", "unknown")
    
    # Build graph
    try:
        G = build_graph_from_cim(cim)
    except Exception as e:
        result["status"] = f"GRAPH_FAIL: {e}"
        return result
    
    result["graph_nodes"] = G.number_of_nodes()
    result["graph_edges"] = G.number_of_edges()
    result["connected"] = nx.is_connected(G) if G.number_of_nodes() > 0 else False
    result["n_components"] = nx.number_connected_components(G) if G.number_of_nodes() > 0 else 0
    
    # Device type breakdown
    dev_types = {}
    for d in cim["devices"]:
        st = d.get("subtype", "Unknown")
        dev_types[st] = dev_types.get(st, 0) + 1
    result["device_types"] = dev_types
    
    # Synthetic SCADA for detection
    rng = np.random.default_rng(42)
    scada = {"bus_voltages": [], "line_powers": []}
    node_list = list(G.nodes())
    for node in node_list:
        vm = 1.0 + rng.normal(0, 0.02)
        scada["bus_voltages"].append({"bus": node, "vm_pu": vm, "sigma": 0.005})
    for i, (u, v) in enumerate(G.edges()):
        p = 0.5 + rng.normal(0, 0.1)
        scada["line_powers"].append({
            "line": i, "from_bus": u, "to_bus": v,
            "p_mw": p, "q_mvar": 0.1, "side": "from", "sigma": 0.02})
    
    # Inject topology anomaly
    gt = []
    edges = list(G.edges())
    if len(edges) > 5:
        G_broken = copy.deepcopy(G)
        broken = edges[len(edges)//2]
        G_broken.remove_edge(*broken)
        gt.append({"type": "拓扑中断", "location": str(broken)})
    else:
        G_broken = G
    
    # Voltage anomaly
    if len(scada["bus_voltages"]) > 3:
        scada["bus_voltages"][2]["vm_pu"] = 0.70
        gt.append({"type": "遥测!=拓扑", "location": "bus_2"})
    
    # Rule engine
    try:
        engine = RuleEngine()
        detected = engine.run_all_checks(G_broken, measurements=scada,
                                          cim_data=cim["devices"],
                                          svg_data=[])
    except Exception as e:
        result["status"] = f"DETECT_FAIL: {e}"
        return result
    
    result["detected_count"] = len(detected)
    result["injected_count"] = len(gt)
    
    # Evaluate
    try:
        ev = evaluate_by_type(detected, gt)
        result["type_recall"] = ev["type_recall"]
        result["per_type"] = {k: v["detected"] for k, v in ev["per_type"].items()}
    except:
        result["type_recall"] = 0
    
    result["time"] = time.time() - t0
    return result


def main():
    print("=" * 90)
    print("  E2E CIM Pipeline Test - All IEEE Test Feeders")
    print("=" * 90)
    
    all_results = []
    for fname in CIM_FILES:
        fpath = CIM_DIR / fname
        if not fpath.exists():
            print(f"  {fname:<25} [NOT FOUND]")
            continue
        
        sys.stdout.write(f"  {fname:<25} ... ")
        sys.stdout.flush()
        r = test_cim_file(fpath, fname)
        all_results.append(r)
        
        if r["status"] == "OK":
            print(f"[OK] dev={r['devices']:>3} node={r['graph_nodes']:>4} "
                  f"edge={r['graph_edges']:>4} conn={'Y' if r['connected'] else 'N'} "
                  f"recall={r['type_recall']:.0%} time={r['time']:.2f}s")
        else:
            print(f"[{r['status'][:30]}]")
    
    # Summary
    print("\n" + "=" * 90)
    ok = [r for r in all_results if r["status"] == "OK"]
    print(f"  PASSED: {len(ok)}/{len(all_results)}")
    if ok:
        print(f"\n  {'File':<25} {'Devices':>7} {'Nodes':>6} {'Edges':>6} "
              f"{'Conn':>5} {'Recall':>7} {'Time':>6}")
        print("  " + "-" * 70)
        for r in ok:
            print(f"  {r['name']:<25} {r['devices']:>7} {r['graph_nodes']:>6} "
                  f"{r['graph_edges']:>6} {'Y' if r['connected'] else 'N':>5} "
                  f"{r['type_recall']:.0%} {r['time']:>5.2f}s")
    
    # Device type summary across all files
    all_types = {}
    for r in ok:
        for dt, cnt in r.get("device_types", {}).items():
            all_types[dt] = all_types.get(dt, 0) + cnt
    if all_types:
        print(f"\n  Device Types Across All CIM Files:")
        for dt, cnt in sorted(all_types.items(), key=lambda x: -x[1]):
            print(f"    {dt:<25} {cnt:>5}")
    
    # Save
    with open(OUTPUT_DIR / "e2e_cim_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Results: {OUTPUT_DIR / 'e2e_cim_results.json'}")

if __name__ == "__main__":
    main()