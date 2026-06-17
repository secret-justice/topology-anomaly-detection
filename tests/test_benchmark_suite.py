# -*- coding: utf-8 -*-
"""
Benchmark Suite v2: Monte Carlo with bridge-aware injection + virtual connection detection
"""
import sys, time, json, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import networkx as nx
import pandapower.networks as pn
import logging
logging.basicConfig(level=logging.ERROR)
from data_preprocessing.scada_simulator import SCADASimulator
from data_preprocessing.anomaly_injector import inject_all_anomalies
from utils.graph_utils import build_graph_from_pandapower
from anomaly_detection.rule_engine import run_rule_engine
from anomaly_detection.state_estimator import add_measurements_to_network, run_wls_estimation, detect_bad_data
from utils.metrics import evaluate_by_type
from utils.metrics_enhanced import evaluate_system_summary
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
N_ITER = 10

def find_bridges(net):
    bridges = []
    G = build_graph_from_pandapower(net)
    for idx in net.line.index:
        if not bool(net.line.at[idx, "in_service"]): continue
        fb, tb = int(net.line.at[idx, "from_bus"]), int(net.line.at[idx, "to_bus"])
        u, v = f"bus_{fb}", f"bus_{tb}"
        if G.has_edge(u, v):
            Gt = copy.deepcopy(G); Gt.remove_edge(u, v)
            if Gt.number_of_nodes() > 0 and not nx.is_connected(Gt):
                bridges.append(idx)
    return bridges

def run_iter(seed):
    net = pn.case33bw()
    sim = SCADASimulator(net, seed=seed)
    if not sim.pf_converged:
        return {"seed": seed, "status": "PF_FAIL"}
    meas = sim.generate_measurements()
    cim = [{"uri": f"Bus_{i}", "name": f"Bus_{i}", "type": "CE"} for i in net.bus.index]
    svg = [{"id": f"Bus_{i}", "type": "Bus"} for i in net.bus.index]
    
    # Inject with bridge awareness
    rng = np.random.default_rng(seed)
    net_inj = copy.deepcopy(net)
    meas_inj = copy.deepcopy(meas)
    svg_inj = copy.deepcopy(svg)
    gt = []
    
    # Topology interrupt (bridge-aware)
    bridges = find_bridges(net)
    if bridges:
        li = int(rng.choice(bridges))
    else:
        li = int(rng.choice(net_inj.line.index))
    net_inj.line.at[li, "in_service"] = False
    gt.append({"type": "\u62d3\u6251\u4e2d\u65ad", "location": f"line_{li}"})
    
    # Telemetry error (extreme voltage -> virtual connection detection)
    bvs = meas_inj.get("bus_voltages", [])
    if len(bvs) > 2:
        bi = int(rng.integers(0, len(bvs)))
        bvs[bi]["vm_pu"] = 0.65  # extreme deviation
        gt.append({"type": "\u865a\u62df\u63a5/\u9519\u63a5", "location": f"bus_{bi}"})
    
    # Model mismatch
    if len(svg_inj) > 5:
        svg_inj.pop()
        gt.append({"type": "\u56fe\u6a21\u4e0d\u7b26", "location": "svg_missing"})
    
    # Rule engine
    G = build_graph_from_pandapower(net_inj)
    det = run_rule_engine(G, cim, svg_inj, meas_inj, [])
    
    # SE for small networks
    se_ok = False
    try:
        nse = copy.deepcopy(net_inj)
        add_measurements_to_network(nse, meas_inj)
        ok, _ = run_wls_estimation(nse)
        if ok:
            bd = detect_bad_data(nse)
            for idx in bd.get("bad_data_indices", []):
                if idx in nse.measurement.index:
                    mt = nse.measurement.at[idx, "measurement_type"]
                    el = nse.measurement.at[idx, "element"]
                    et = nse.measurement.at[idx, "element_type"]
                    det.append({"type": "\u4e0d\u826f\u6570\u636e", "location": f"{et}_{el}",
                                "confidence": 0.99, "layer": "state_estimator"})
            se_ok = True
    except:
        pass
    
    ev = evaluate_by_type(det, gt)
    return {
        "seed": seed, "status": "OK",
        "injected": len(gt), "detected": len(det),
        "type_recall": ev["type_recall"],
        "per_type": {k: {"det": v["detected"], "cnt": v["detected_count"]}
                     for k, v in ev["per_type"].items()},
        "se": se_ok,
    }

def main():
    print("=" * 80)
    print(f"  Benchmark Suite v2: case33bw x {N_ITER} iterations")
    print("=" * 80)
    all_r = []
    for i in range(N_ITER):
        seed = 42 + i * 17
        r = run_iter(seed)
        all_r.append(r)
        if r["status"] == "OK":
            pt = r.get("per_type", {})
            details = " ".join(f"{k[:3]}={'Y' if v.get('det') else 'N'}" for k, v in pt.items())
            print(f"  Iter {i+1:>2} seed={seed:>4} recall={r['type_recall']:.0%} "
                  f"det={r['detected']:>3} se={'Y' if r['se'] else 'N'} [{details}]")
        else:
            print(f"  Iter {i+1:>2} seed={seed:>4} [{r['status']}]")
    
    ok = [r for r in all_r if r["status"] == "OK"]
    if ok:
        recalls = [r["type_recall"] for r in ok]
        print(f"\n{'='*80}")
        print(f"  AGGREGATE: {len(ok)}/{N_ITER} OK")
        print(f"  Type Recall: {np.mean(recalls):.1%} +/- {np.std(recalls):.1%}")
        full = sum(1 for r in recalls if r >= 1.0)
        print(f"  100% recall: {full}/{len(ok)} iterations")
        type_stats = {}
        for r in ok:
            for t, info in r.get("per_type", {}).items():
                type_stats.setdefault(t, {"hit": 0, "total": 0})
                type_stats[t]["total"] += 1
                if info.get("det"): type_stats[t]["hit"] += 1
        print(f"\n  Per-type detection:")
        for t, s in sorted(type_stats.items()):
            rate = s["hit"] / max(s["total"], 1)
            tag = "[OK]" if rate >= 0.8 else "[WARN]" if rate >= 0.5 else "[MISS]"
            print(f"    {tag} {t}: {s['hit']}/{s['total']} ({rate:.0%})")
        print("=" * 80)
    
    with open(OUTPUT_DIR / "benchmark_suite_v2.json", "w", encoding="utf-8") as f:
        json.dump({"config": {"network": "case33bw", "iterations": N_ITER},
                    "results": all_r,
                    "aggregate": {"mean_recall": float(np.mean(recalls)) if ok else 0,
                                  "std_recall": float(np.std(recalls)) if ok else 0}},
                   f, ensure_ascii=False, indent=2, default=str)
    print(f"  Results: {OUTPUT_DIR / 'benchmark_suite_v2.json'}")

if __name__ == "__main__":
    main()