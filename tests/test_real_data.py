# -*- coding: utf-8 -*-
"""
Real data validation test framework
Supports CIM/SVG format data loading, detection pipeline, and two modes:
  - With ground truth: precision/recall/F1 evaluation
  - Without ground truth: anomaly count and confidence distribution
"""
import sys
import os
import time
import json
import logging
import copy
from pathlib import Path
from typing import Dict, List, Optional

# Setup project path
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandapower as pp
import pandapower.networks as pn
import networkx as nx

from data_preprocessing.cim_adapter import CIMAdapter
from data_preprocessing.svg_adapter import SVGAdapter
from data_preprocessing.scada_simulator import SCADASimulator
from data_preprocessing.anomaly_injector import inject_all_anomalies
from anomaly_detection.detector import AnomalyDetector
from anomaly_detection.rule_engine import RuleEngine, run_rule_engine
from utils.graph_utils import build_graph_from_pandapower, build_graph_from_cim, find_sources

logger = logging.getLogger(__name__)


class RealDataValidator:
    """
    Real data validation framework

    Two modes:
      1. With ground truth (gt): inject anomalies, detect, compute P/R/F1
      2. Without ground truth: load real data, detect anomalies, report counts/confidence
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = {}

    # ================================================================
    # Mode 1: With Ground Truth (from PandaPower + anomaly injection)
    # ================================================================

    def run_with_ground_truth(self, net=None, network_name="case33bw",
                               seed=42, n_trials=1) -> Dict:
        """
        Run detection with injected anomalies and compute precision/recall/F1

        Args:
            net: PandaPower network (if None, loads by network_name)
            network_name: network to load
            seed: random seed
            n_trials: number of repeated trials for statistical robustness

        Returns:
            {precision, recall, f1, per_type_metrics, timing, details}
        """
        if net is None:
            net = self._load_network(network_name)
        pp.runpp(net)

        all_trials = []
        for trial in range(n_trials):
            trial_seed = seed + trial
            result = self._run_single_gt_trial(net, trial_seed)
            all_trials.append(result)

        # Aggregate across trials
        agg = self._aggregate_trials(all_trials)
        self.results["ground_truth_mode"] = agg

        if self.verbose:
            self._print_gt_results(agg, network_name)

        return agg

    def _run_single_gt_trial(self, net, seed) -> Dict:
        """Run a single ground-truth trial"""
        t0 = time.time()

        # Build graph
        graph = build_graph_from_pandapower(net)

        # Create device lists (simulating CIM/SVG)
        cim_devices, svg_devices, switches = self._build_device_lists(net)

        # Generate SCADA measurements
        sim = SCADASimulator(net, seed=seed)
        measurements = sim.generate_measurements()

        # Inject anomalies
        injected = inject_all_anomalies(
            net, measurements, cim_devices, svg_devices, switches, seed=seed)

        net_inj = injected["net"]
        meas_inj = injected["measurements"]
        cim_inj = injected["cim_devices"]
        svg_inj = injected["svg_devices"]
        ground_truth = injected["ground_truth"]

        # Run detection on injected data
        graph_inj = build_graph_from_pandapower(net_inj)
        scada_sim_inj = SCADASimulator(net_inj, seed=seed+1000)
        meas_inj_full = scada_sim_inj.generate_measurements()

        anomalies = run_rule_engine(
            graph=graph_inj,
            cim_devices=cim_inj,
            svg_devices=svg_inj,
            measurements=meas_inj_full,
            switches=injected.get("switches", []),
        )

        # Match detected anomalies to ground truth
        matched, unmatched_gt, false_alarms = self._match_anomalies(
            anomalies, ground_truth)

        elapsed = time.time() - t0

        return {
            "ground_truth": ground_truth,
            "detected": anomalies,
            "matched": matched,
            "unmatched_gt": unmatched_gt,
            "false_alarms": false_alarms,
            "elapsed": elapsed,
            "n_gt": len(ground_truth),
            "n_detected": len(anomalies),
            "n_matched": len(matched),
        }

    def _match_anomalies(self, detected, ground_truth, match_threshold=0.5):
        """
        Match detected anomalies to ground truth entries

        Matching criteria:
          - Type similarity (exact or fuzzy match)
          - Location similarity (same element or nearby)

        Returns:
          matched: [(detected, gt_entry)]
          unmatched_gt: [gt entries not matched]
          false_alarms: [detected entries not matched to any gt]
        """
        TYPE_MAP = {
            "model_mismatch": "model_mismatch",
            "topo_interrupt": "topo_interrupt",
            "topo_interrupt": "topo_interrupt",
            "virtual_faulty": "virtual_faulty",
            "telemetry_mismatch": "telemetry_mismatch",
            "signal_mismatch": "signal_mismatch",
            "\u56fe\u6a21\u4e0d\u7b26": "model_mismatch",
            "\u865a\u62df\u63a5/\u9519\u63a5": "virtual_faulty",
            "\u9065\u4fe1!=\u9065\u6d4b": "signal_mismatch",
            "\u62d3\u6251\u4e2d\u65ad": "topo_interrupt",
            "\u9065\u6d4b!=\u62d3\u6251": "telemetry_mismatch",
        }

        matched = []
        used_gt = set()
        used_det = set()

        for i, gt in enumerate(ground_truth):
            gt_type = TYPE_MAP.get(gt.get("type", ""), gt.get("type", ""))
            gt_loc = gt.get("location", "")

            best_j = -1
            best_score = 0.0

            for j, det in enumerate(anomalies if False else detected):
                if j in used_det:
                    continue
                det_type = TYPE_MAP.get(det.get("type", ""), det.get("type", ""))
                det_loc = str(det.get("location", ""))

                score = 0.0
                if gt_type == det_type:
                    score += 0.5
                elif gt_type in det_type or det_type in gt_type:
                    score += 0.3

                if gt_loc and det_loc:
                    if gt_loc in det_loc or det_loc in gt_loc:
                        score += 0.5
                    elif self._location_overlap(gt_loc, det_loc):
                        score += 0.3

                if score > best_score:
                    best_score = score
                    best_j = j

            if best_j >= 0 and best_score >= match_threshold:
                matched.append((detected[best_j], gt))
                used_gt.add(i)
                used_det.add(best_j)

        unmatched_gt = [gt for i, gt in enumerate(ground_truth) if i not in used_gt]
        false_alarms = [det for j, det in enumerate(detected) if j not in used_det]

        return matched, unmatched_gt, false_alarms

    @staticmethod
    def _location_overlap(loc_a, loc_b):
        """Check if two location strings share common elements"""
        parts_a = set(loc_a.replace("_", " ").split())
        parts_b = set(loc_b.replace("_", " ").split())
        return len(parts_a & parts_b) > 0

    def _aggregate_trials(self, trials):
        """Aggregate results across multiple trials"""
        total_gt = sum(t["n_gt"] for t in trials)
        total_detected = sum(t["n_detected"] for t in trials)
        total_matched = sum(t["n_matched"] for t in trials)
        total_fa = sum(len(t["false_alarms"]) for t in trials)
        total_time = sum(t["elapsed"] for t in trials)

        precision = total_matched / max(total_detected, 1)
        recall = total_matched / max(total_gt, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        # Per-type breakdown
        per_type = {}
        for t in trials:
            for det, gt in t["matched"]:
                tt = gt.get("type", "unknown")
                per_type.setdefault(tt, {"tp": 0, "fp": 0, "fn": 0})
                per_type[tt]["tp"] += 1
            for fa in t["false_alarms"]:
                tt = fa.get("type", "unknown")
                per_type.setdefault(tt, {"tp": 0, "fp": 0, "fn": 0})
                per_type[tt]["fp"] += 1
            for gt in t["unmatched_gt"]:
                tt = gt.get("type", "unknown")
                per_type.setdefault(tt, {"tp": 0, "fp": 0, "fn": 0})
                per_type[tt]["fn"] += 1

        for tt, counts in per_type.items():
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            counts["precision"] = tp / max(tp + fp, 1)
            counts["recall"] = tp / max(tp + fn, 1)
            counts["f1"] = 2 * counts["precision"] * counts["recall"] / max(
                counts["precision"] + counts["recall"], 1e-9)

        return {
            "n_trials": len(trials),
            "total_ground_truth": total_gt,
            "total_detected": total_detected,
            "total_matched": total_matched,
            "total_false_alarms": total_fa,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "per_type": per_type,
            "avg_time_s": total_time / max(len(trials), 1),
            "trials": trials,
        }

    def _print_gt_results(self, agg, network_name):
        """Print ground truth evaluation results"""
        print("\n" + "=" * 70)
        print("  GROUND TRUTH VALIDATION: %s" % network_name)
        print("=" * 70)
        print("  Trials:           %d" % agg["n_trials"])
        print("  Total GT:         %d" % agg["total_ground_truth"])
        print("  Total Detected:   %d" % agg["total_detected"])
        print("  Total Matched:    %d" % agg["total_matched"])
        print("  False Alarms:     %d" % agg["total_false_alarms"])
        print("  -------------------------------")
        print("  Precision:        %.3f" % agg["precision"])
        print("  Recall:           %.3f" % agg["recall"])
        print("  F1 Score:         %.3f" % agg["f1"])
        print("  Avg Time:         %.3f s" % agg["avg_time_s"])

        if agg["per_type"]:
            print("\n  Per-type breakdown:")
            print("  %-25s %8s %8s %8s" % ("Type", "Prec", "Recall", "F1"))
            print("  " + "-" * 55)
            for tt, c in sorted(agg["per_type"].items()):
                print("  %-25s %8.3f %8.3f %8.3f" % (
                    tt, c["precision"], c["recall"], c["f1"]))
        print("=" * 70)

    # ================================================================
    # Mode 2: Without Ground Truth (real data)
    # ================================================================

    def run_without_ground_truth(self, cim_path=None, svg_path=None,
                                   net=None, measurements=None,
                                   network_name="real_data") -> Dict:
        """
        Run detection on real data without ground truth

        Can accept:
          - CIM file path (parsed via CIMAdapter)
          - SVG file path (parsed via SVGAdapter)
          - Pre-built net + measurements

        Returns:
          {anomalies, summary, timing, data_quality}
        """
        t0 = time.time()

        cim_data = None
        svg_data = None
        graph = None

        # Auto-load network if nothing provided
        if net is None and cim_path is None and svg_path is None:
            net = self._load_network(network_name)
            pp.runpp(net)

        # Load CIM if provided
        if cim_path and os.path.exists(cim_path):
            adapter = CIMAdapter()
            cim_data = adapter.load_cim(cim_path)
            graph = adapter.get_topology_graph()
            if net is None:
                try:
                    net = adapter.to_pandapower()
                    pp.runpp(net)
                except Exception as e:
                    logger.warning("CIM to PandaPower conversion failed: %s" % e)

        # Load SVG if provided
        if svg_path and os.path.exists(svg_path):
            svg_adapter = SVGAdapter()
            svg_data = svg_adapter.load_svg(svg_path)
            if cim_data:
                alignment = svg_adapter.align_with_cim(cim_data)
            else:
                alignment = None

        # If net is provided directly, build graph from it
        if net is not None and graph is None:
            graph = build_graph_from_pandapower(net)

        # Generate measurements if not provided
        if measurements is None and net is not None:
            sim = SCADASimulator(net, seed=42)
            measurements = sim.generate_measurements()

        # Build device lists
        if cim_data is None and net is not None:
            cim_devices, svg_devices, switches = self._build_device_lists(net)
        else:
            cim_devices = cim_data.get("devices", []) if cim_data else []
            svg_devices = svg_data.get("devices", []) if svg_data else []
            switches = cim_data.get("switches", []) if cim_data else []

        # Run detection
        anomalies = run_rule_engine(
            graph=graph,
            cim_devices=cim_devices,
            svg_devices=svg_devices,
            measurements=measurements or {},
            switches=switches,
        )

        elapsed = time.time() - t0

        # Data quality assessment
        data_quality = self._assess_data_quality(
            cim_data, svg_data, net, measurements)

        # Summary
        from collections import Counter
        type_counts = Counter(a.get("type", "unknown") for a in anomalies)
        conf_values = [a.get("confidence", 0) for a in anomalies]
        severity_counts = Counter(a.get("severity", "unknown") for a in anomalies)

        summary = {
            "total_anomalies": len(anomalies),
            "by_type": dict(type_counts),
            "by_severity": dict(severity_counts),
            "confidence_mean": float(np.mean(conf_values)) if conf_values else 0,
            "confidence_max": float(np.max(conf_values)) if conf_values else 0,
            "confidence_min": float(np.min(conf_values)) if conf_values else 0,
        }

        result = {
            "anomalies": anomalies,
            "summary": summary,
            "data_quality": data_quality,
            "elapsed": elapsed,
            "network_name": network_name,
        }

        self.results["no_gt_mode"] = result

        if self.verbose:
            self._print_no_gt_results(result)

        return result

    def _assess_data_quality(self, cim_data, svg_data, net, measurements):
        """Assess data quality for the loaded dataset"""
        quality = {"issues": [], "scores": {}}

        # Check power flow convergence
        if net is not None:
            try:
                converged = net.converged
                quality["scores"]["pf_converged"] = 1.0 if converged else 0.0
                if not converged:
                    quality["issues"].append("Power flow did not converge")
            except Exception:
                quality["scores"]["pf_converged"] = 0.0

        # Check CIM data completeness
        if cim_data:
            n_devices = len(cim_data.get("devices", []))
            n_nodes = len(cim_data.get("nodes", []))
            n_connections = len(cim_data.get("connections", []))
            quality["scores"]["cim_devices"] = n_devices
            quality["scores"]["cim_nodes"] = n_nodes
            quality["scores"]["cim_connections"] = n_connections
            if n_devices == 0:
                quality["issues"].append("No devices found in CIM data")
            if n_nodes == 0:
                quality["issues"].append("No connectivity nodes in CIM data")

        # Check measurement coverage
        if measurements:
            bv = measurements.get("bus_voltages", [])
            lp = measurements.get("line_powers", [])
            quality["scores"]["n_voltage_meas"] = len(bv)
            quality["scores"]["n_power_meas"] = len(lp)
            if net is not None and len(bv) < len(net.bus) * 0.5:
                quality["issues"].append(
                    "Voltage measurement coverage < 50%% (%d/%d)" % (len(bv), len(net.bus)))

        # Check graph connectivity
        if net is not None:
            graph = build_graph_from_pandapower(net)
            if graph.number_of_nodes() > 0:
                if nx.is_connected(graph):
                    quality["scores"]["connected"] = 1.0
                else:
                    n_comp = nx.number_connected_components(graph)
                    quality["scores"]["connected"] = 0.0
                    quality["issues"].append(
                        "Topology has %d disconnected components" % n_comp)

        quality["overall"] = "GOOD" if not quality["issues"] else "WARNINGS"
        return quality

    def _print_no_gt_results(self, result):
        """Print no-ground-truth results"""
        s = result["summary"]
        dq = result["data_quality"]

        print("\n" + "=" * 70)
        print("  REAL DATA VALIDATION: %s" % result["network_name"])
        print("=" * 70)
        print("  Anomalies found:    %d" % s["total_anomalies"])
        print("  Confidence range:   [%.3f, %.3f]" % (s["confidence_min"], s["confidence_max"]))
        print("  Confidence mean:    %.3f" % s["confidence_mean"])
        print("  Elapsed:            %.3f s" % result["elapsed"])

        if s["by_type"]:
            print("\n  By type:")
            for tt, cnt in sorted(s["by_type"].items(), key=lambda x: -x[1]):
                print("    %-30s %d" % (tt, cnt))

        if s["by_severity"]:
            print("\n  By severity:")
            for sev, cnt in sorted(s["by_severity"].items(), key=lambda x: -x[1]):
                print("    %-30s %d" % (sev, cnt))

        print("\n  Data quality: %s" % dq["overall"])
        for issue in dq["issues"]:
            print("    [!] %s" % issue)

        for key, val in sorted(dq["scores"].items()):
            print("    %s: %s" % (key, val))
        print("=" * 70)

    # ================================================================
    # CIM + SVG Round-trip Test
    # ================================================================

    def run_cim_roundtrip(self, network_name="case33bw", output_dir=None) -> Dict:
        """
        Full round-trip test: PandaPower -> CIM XML -> parse -> PandaPower -> compare

        Validates that the CIM adapter can:
          1. Generate CIM from PandaPower
          2. Parse the CIM back
          3. Convert to PandaPower
          4. The converted network is structurally equivalent
        """
        from data_preprocessing.generate_example_data import (
            generate_cim_from_pandapower, generate_svg_from_pandapower)

        t0 = time.time()
        net_orig = self._load_network(network_name)
        pp.runpp(net_orig)

        # Generate CIM
        if output_dir is None:
            output_dir = str(PROJECT_DIR / "output" / "roundtrip_test")
        os.makedirs(output_dir, exist_ok=True)

        cim_path = os.path.join(output_dir, "%s.xml" % network_name)
        generate_cim_from_pandapower(net_orig, cim_path, network_name=network_name)

        # Parse CIM back
        adapter = CIMAdapter()
        cim_data = adapter.load_cim(cim_path)
        net_parsed = adapter.to_pandapower()

        # Compare structures
        comparison = self._compare_networks(net_orig, net_parsed, cim_data)

        # Also test SVG generation
        svg_path = os.path.join(output_dir, "%s.svg" % network_name)
        generate_svg_from_pandapower(net_orig, svg_path, network_name=network_name)
        svg_adapter = SVGAdapter()
        svg_data = svg_adapter.load_svg(svg_path)
        alignment = svg_adapter.align_with_cim(cim_data)
        consistency = svg_adapter.validate_consistency()

        elapsed = time.time() - t0

        result = {
            "network": network_name,
            "cim_path": cim_path,
            "svg_path": svg_path,
            "comparison": comparison,
            "alignment_stats": alignment.get("stats", {}),
            "consistency": consistency,
            "elapsed": elapsed,
        }

        if self.verbose:
            self._print_roundtrip_results(result)

        return result

    def _compare_networks(self, net_orig, net_parsed, cim_data):
        """Compare original and parsed PandaPower networks"""
        comparison = {
            "bus_count_match": len(net_orig.bus) == len(net_parsed.bus),
            "line_count_match": len(net_orig.line) == len(net_parsed.bus),
            "orig_buses": len(net_orig.bus),
            "parsed_buses": len(net_parsed.bus),
            "orig_lines": len(net_orig.line),
            "parsed_lines": len(net_parsed.line),
            "orig_trafos": len(net_orig.trafo),
            "parsed_trafos": len(net_parsed.trafo),
            "orig_loads": len(net_orig.load),
            "parsed_loads": len(net_parsed.load),
            "cim_devices": len(cim_data.get("devices", [])),
            "cim_nodes": len(cim_data.get("nodes", [])),
            "cim_terminals": len(cim_data.get("terminals", [])),
            "cim_connections": len(cim_data.get("connections", [])),
            "cim_switches": len(cim_data.get("switches", [])),
        }

        # Structural equivalence
        orig_buses = set(int(i) for i in net_orig.bus.index)
        parsed_buses = set(int(i) for i in net_parsed.bus.index)
        comparison["bus_index_overlap"] = len(orig_buses & parsed_buses)
        comparison["bus_count_exact"] = len(net_orig.bus) == len(net_parsed.bus)
        comparison["line_count_exact"] = len(net_orig.line) == len(net_parsed.line)
        comparison["trafo_count_exact"] = len(net_orig.trafo) == len(net_parsed.trafo)
        comparison["load_count_exact"] = len(net_orig.load) == len(net_parsed.load)

        return comparison

    def _print_roundtrip_results(self, result):
        """Print round-trip test results"""
        c = result["comparison"]
        a = result["alignment_stats"]
        con = result["consistency"]

        print("\n" + "=" * 70)
        print("  CIM ROUND-TRIP TEST: %s" % result["network"])
        print("=" * 70)
        print("  Original -> Parsed element counts:")
        print("    Buses:   %d -> %d %s" % (
            c["orig_buses"], c["parsed_buses"],
            "[OK]" if c["bus_count_exact"] else "[MISMATCH]"))
        print("    Lines:   %d -> %d %s" % (
            c["orig_lines"], c["parsed_lines"],
            "[OK]" if c["line_count_exact"] else "[MISMATCH]"))
        print("    Trafos:  %d -> %d %s" % (
            c["orig_trafos"], c["parsed_trafos"],
            "[OK]" if c["trafo_count_exact"] else "[MISMATCH]"))
        print("    Loads:   %d -> %d %s" % (
            c["orig_loads"], c["parsed_loads"],
            "[OK]" if c["load_count_exact"] else "[MISMATCH]"))

        print("\n  CIM data structure:")
        print("    Devices:      %d" % c["cim_devices"])
        print("    Nodes:        %d" % c["cim_nodes"])
        print("    Terminals:    %d" % c["cim_terminals"])
        print("    Connections:  %d" % c["cim_connections"])
        print("    Switches:     %d" % c["cim_switches"])

        if a:
            print("\n  CIM-SVG alignment:")
            for k, v in sorted(a.items()):
                print("    %s: %s" % (k, v))

        if con:
            print("\n  Consistency check: %s" % ("PASS" if con.get("passed") else "ISSUES"))
            for issue in con.get("issues", []):
                print("    [%s] %s" % (issue.get("severity", ""), issue.get("detail", "")))

        print("  Elapsed: %.3f s" % result["elapsed"])
        print("=" * 70)

    # ================================================================
    # Helpers
    # ================================================================

    @staticmethod
    def _load_network(name):
        """Load PandaPower network by name"""
        import importlib
        try:
            mod = importlib.import_module("pandapower.networks.%s" % name)
            func = getattr(mod, name)
            return func()
        except (ImportError, AttributeError):
            func = getattr(pn, name)
            return func()

    @staticmethod
    def _build_device_lists(net):
        """Build CIM/SVG device lists from PandaPower for testing"""
        cim_devices = []
        svg_devices = []
        switches = []

        for idx in net.bus.index:
            nm = "Bus_%d" % idx
            cim_devices.append({
                "uri": "#_bus_%d" % idx, "name": nm,
                "type": "ConductingEquipment", "subtype": "BusbarSection"})
            svg_devices.append({
                "id": nm, "type": "Bus",
                "x": float(idx * 100), "y": 300.0, "label": nm})

        for idx in net.line.index:
            nm = "Line_%d" % idx
            cim_devices.append({
                "uri": "#_line_%d" % idx, "name": nm,
                "type": "ConductingEquipment", "subtype": "ACLineSegment"})

        for idx in net.trafo.index:
            nm = "Trafo_%d" % idx
            cim_devices.append({
                "uri": "#_trafo_%d" % idx, "name": nm,
                "type": "ConductingEquipment", "subtype": "PowerTransformer"})

        if hasattr(net, "switch"):
            for idx in net.switch.index:
                closed = bool(net.switch.at[idx, "closed"])
                switches.append({
                    "uri": "#_switch_%d" % idx, "name": "Switch_%d" % idx,
                    "subtype": "Breaker", "normal_open": not closed, "open_pos": None})

        return cim_devices, svg_devices, switches

    def export_results(self, output_path=None):
        """Export results to JSON"""
        if output_path is None:
            output_path = str(PROJECT_DIR / "output" / "validation_results.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        serializable = {}
        for key, val in self.results.items():
            if isinstance(val, dict):
                clean = {}
                for k, v in val.items():
                    if k == "trials":
                        continue
                    try:
                        json.dumps(v)
                        clean[k] = v
                    except (TypeError, ValueError):
                        clean[k] = str(v)
                serializable[key] = clean

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        print("Results exported to: %s" % output_path)
        return output_path


# ================================================================
# CLI entry point
# ================================================================
def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Real data validation framework for topology detection")
    parser.add_argument("--mode", choices=["gt", "nogt", "roundtrip", "full"],
                        default="full", help="Validation mode")
    parser.add_argument("--network", default="case33bw",
                        help="PandaPower network name")
    parser.add_argument("--cim", default=None, help="CIM XML file path")
    parser.add_argument("--svg", default=None, help="SVG file path")
    parser.add_argument("--trials", type=int, default=1,
                        help="Number of GT trials")
    parser.add_argument("--output", default=None,
                        help="Output JSON path")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    validator = RealDataValidator(verbose=not args.quiet)

    if args.mode in ("gt", "full"):
        print("\n>>> Running ground-truth validation...")
        gt_result = validator.run_with_ground_truth(
            network_name=args.network, n_trials=args.trials)

    if args.mode in ("nogt", "full"):
        print("\n>>> Running no-ground-truth validation...")
        nogt_result = validator.run_without_ground_truth(
            cim_path=args.cim, svg_path=args.svg,
            network_name=args.network)

    if args.mode in ("roundtrip", "full"):
        print("\n>>> Running CIM round-trip test...")
        rt_result = validator.run_cim_roundtrip(network_name=args.network)

    validator.export_results(args.output)
    print("\nAll validations complete.")


if __name__ == "__main__":
    main()
