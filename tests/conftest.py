# -*- coding: utf-8 -*-
"""pytest fixtures for the topology detection project"""
import sys
from pathlib import Path
import pytest

# Ensure project root is on sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import pandapower as pp
import pandapower.networks as pn
from data_preprocessing.scada_simulator import SCADASimulator
from utils.graph_utils import build_graph_from_pandapower, find_sources


@pytest.fixture(scope="session")
def case33bw():
    """Load IEEE 33-bus network (session-scoped)"""
    net = pn.case33bw()
    pp.runpp(net)
    return net


@pytest.fixture(scope="session")
def case14():
    """Load IEEE 14-bus network"""
    net = pn.case14()
    pp.runpp(net)
    return net


@pytest.fixture
def sample_measurements(case33bw):
    """Generate SCADA measurements for case33bw"""
    sim = SCADASimulator(case33bw, seed=42)
    return sim.generate_measurements()


@pytest.fixture
def network_data(case33bw, sample_measurements):
    """Full network data dict for detection pipeline"""
    graph = build_graph_from_pandapower(case33bw)
    cim_devices = []
    svg_devices = []
    switches = []
    for idx in case33bw.bus.index:
        name = f"Bus_{idx}"
        cim_devices.append({"uri": f"#_bus_{idx}", "name": name,
                            "type": "ConductingEquipment", "subtype": "BusbarSection"})
        svg_devices.append({"id": name, "type": "Bus",
                            "x": float(idx * 100), "y": 300.0, "label": name})
    for idx in case33bw.line.index:
        name = f"Line_{idx}"
        cim_devices.append({"uri": f"#_line_{idx}", "name": name,
                            "type": "ConductingEquipment", "subtype": "ACLineSegment"})
    return {
        "graph": graph,
        "cim_devices": cim_devices,
        "svg_devices": svg_devices,
        "measurements": sample_measurements,
        "switches": switches,
        "net": case33bw,
        "scada_data": sample_measurements,
    }