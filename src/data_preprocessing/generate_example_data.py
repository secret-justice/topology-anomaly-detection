# -*- coding: utf-8 -*-
"""
Example data generator
Generate CIM XML and SVG from PandaPower networks for adapter testing
"""
import os
import math
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

import pandapower as pp
import pandapower.networks as pn

logger = logging.getLogger(__name__)

CIM_URI = "http://iec.ch/TC57/CIM100#"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"


def generate_cim_from_pandapower(net, output_path, network_name="ExampleNetwork"):
    """Generate CIM XML from PandaPower network"""
    try:
        if not hasattr(net, "res_bus") or len(net.res_bus) == 0:
            pp.runpp(net)
    except Exception:
        pass

    # Register namespaces so ElementTree uses proper prefixes
    from xml.etree.ElementTree import register_namespace
    register_namespace("rdf", RDF_NS)
    register_namespace("cim", CIM_URI)
    register_namespace("rdfs", "http://www.w3.org/2000/01/rdf-schema#")
    rdf_root = Element("{%s}RDF" % RDF_NS)

    _counter = [0]
    def _nid():
        _counter[0] += 1
        return "_id_%06d" % _counter[0]

    def _res(rtype, oid, props):
        el = SubElement(rdf_root, "{%s}Description" % RDF_NS)
        el.set("{%s}about" % RDF_NS, "#" + oid)
        te = SubElement(el, "{%s}type" % RDF_NS)
        te.set("{%s}resource" % RDF_NS, CIM_URI + rtype)
        for k, v in props.items():
            if v is None:
                continue
            ch = SubElement(el, "{%s}%s" % (CIM_URI, k))
            if isinstance(v, str) and v.startswith("#"):
                ch.set("{%s}resource" % RDF_NS, v)
            else:
                ch.text = str(v)

    # ConnectivityNode
    bus_ids = {}
    for idx in net.bus.index:
        cid = _nid()
        bus_ids[int(idx)] = cid
        nm = str(net.bus.at[idx, "name"]) if "name" in net.bus.columns else "Bus_%d" % idx
        _res("ConnectivityNode", cid, {"IdentifiedObject.name": nm, "IdentifiedObject.mRID": cid})

    # ACLineSegment
    for idx in net.line.index:
        lid = _nid()
        nm = str(net.line.at[idx, "name"]) if "name" in net.line.columns else "Line_%d" % idx
        fb = int(net.line.at[idx, "from_bus"])
        tb = int(net.line.at[idx, "to_bus"])
        lg = float(net.line.at[idx, "length_km"])
        r = float(net.line.at[idx, "r_ohm_per_km"])
        x = float(net.line.at[idx, "x_ohm_per_km"])
        _res("ACLineSegment", lid, {
            "IdentifiedObject.name": nm, "IdentifiedObject.mRID": lid,
            "Conductor.length": str(lg), "ACLineSegment.r": str(r*lg), "ACLineSegment.x": str(x*lg),
        })
        t1 = _nid()
        _res("Terminal", t1, {"IdentifiedObject.mRID": t1, "ACDCTerminal.sequenceNumber": "1",
                               "Terminal.ConductingEquipment": "#"+lid, "Terminal.ConnectivityNode": "#"+bus_ids[fb]})
        t2 = _nid()
        _res("Terminal", t2, {"IdentifiedObject.mRID": t2, "ACDCTerminal.sequenceNumber": "2",
                               "Terminal.ConductingEquipment": "#"+lid, "Terminal.ConnectivityNode": "#"+bus_ids[tb]})

    # PowerTransformer + PowerTransformerEnd
    for idx in net.trafo.index:
        tid = _nid()
        nm = str(net.trafo.at[idx, "name"]) if "name" in net.trafo.columns else "Trafo_%d" % idx
        hv = int(net.trafo.at[idx, "hv_bus"])
        lv = int(net.trafo.at[idx, "lv_bus"])
        sn = float(net.trafo.at[idx, "sn_mva"])
        vnh = float(net.trafo.at[idx, "vn_hv_kv"])
        vnl = float(net.trafo.at[idx, "vn_lv_kv"])
        _res("PowerTransformer", tid, {"IdentifiedObject.name": nm, "IdentifiedObject.mRID": tid})
        eh = _nid()
        _res("PowerTransformerEnd", eh, {"IdentifiedObject.mRID": eh, "PowerTransformerEnd.PowerTransformer": "#"+tid,
                                          "PowerTransformerEnd.ratedS": str(sn), "PowerTransformerEnd.ratedU": str(vnh)})
        th = _nid()
        _res("Terminal", th, {"IdentifiedObject.mRID": th, "ACDCTerminal.sequenceNumber": "1",
                               "Terminal.ConductingEquipment": "#"+tid, "Terminal.ConnectivityNode": "#"+bus_ids[hv]})
        el_ = _nid()
        _res("PowerTransformerEnd", el_, {"IdentifiedObject.mRID": el_, "PowerTransformerEnd.PowerTransformer": "#"+tid,
                                           "PowerTransformerEnd.ratedS": str(sn), "PowerTransformerEnd.ratedU": str(vnl)})
        tl = _nid()
        _res("Terminal", tl, {"IdentifiedObject.mRID": tl, "ACDCTerminal.sequenceNumber": "2",
                               "Terminal.ConductingEquipment": "#"+tid, "Terminal.ConnectivityNode": "#"+bus_ids[lv]})

    # EnergyConsumer
    for idx in net.load.index:
        lid = _nid()
        nm = str(net.load.at[idx, "name"]) if "name" in net.load.columns else "Load_%d" % idx
        b = int(net.load.at[idx, "bus"])
        p = float(net.load.at[idx, "p_mw"])
        q = float(net.load.at[idx, "q_mvar"])
        _res("EnergyConsumer", lid, {"IdentifiedObject.name": nm, "IdentifiedObject.mRID": lid,
                                      "EnergyConsumer.p": str(p), "EnergyConsumer.q": str(q)})
        tt = _nid()
        _res("Terminal", tt, {"IdentifiedObject.mRID": tt, "ACDCTerminal.sequenceNumber": "1",
                               "Terminal.ConductingEquipment": "#"+lid, "Terminal.ConnectivityNode": "#"+bus_ids[b]})

    # EnergySource
    for idx in net.ext_grid.index:
        eid = _nid()
        b = int(net.ext_grid.at[idx, "bus"])
        nm = str(net.ext_grid.at[idx, "name"]) if "name" in net.ext_grid.columns else "ExtGrid_%d" % idx
        _res("EnergySource", eid, {"IdentifiedObject.name": nm, "IdentifiedObject.mRID": eid})
        tt = _nid()
        _res("Terminal", tt, {"IdentifiedObject.mRID": tt, "ACDCTerminal.sequenceNumber": "1",
                               "Terminal.ConductingEquipment": "#"+eid, "Terminal.ConnectivityNode": "#"+bus_ids[b]})

    # Switches
    for idx in net.switch.index:
        sid = _nid()
        b = int(net.switch.at[idx, "bus"])
        el_idx = int(net.switch.at[idx, "element"])
        closed = bool(net.switch.at[idx, "closed"])
        et = str(net.switch.at[idx, "et"])
        stype = "Breaker" if closed else "Disconnector"
        _res(stype, sid, {"IdentifiedObject.name": "Switch_%d" % idx, "IdentifiedObject.mRID": sid,
                           "Switch.normalOpen": "false" if closed else "true"})
        t1 = _nid()
        _res("Terminal", t1, {"IdentifiedObject.mRID": t1, "ACDCTerminal.sequenceNumber": "1",
                               "Terminal.ConductingEquipment": "#"+sid, "Terminal.ConnectivityNode": "#"+bus_ids[b]})
        if et == "b" and el_idx in bus_ids:
            t2 = _nid()
            _res("Terminal", t2, {"IdentifiedObject.mRID": t2, "ACDCTerminal.sequenceNumber": "2",
                                   "Terminal.ConductingEquipment": "#"+sid, "Terminal.ConnectivityNode": "#"+bus_ids[el_idx]})

    # SynchronousMachine
    for idx in net.gen.index:
        gid = _nid()
        b = int(net.gen.at[idx, "bus"])
        nm = str(net.gen.at[idx, "name"]) if "name" in net.gen.columns else "Gen_%d" % idx
        _res("SynchronousMachine", gid, {"IdentifiedObject.name": nm, "IdentifiedObject.mRID": gid,
                                          "RotatingMachine.p": str(float(net.gen.at[idx, "p_mw"]))})
        tt = _nid()
        _res("Terminal", tt, {"IdentifiedObject.mRID": tt, "ACDCTerminal.sequenceNumber": "1",
                               "Terminal.ConductingEquipment": "#"+gid, "Terminal.ConnectivityNode": "#"+bus_ids[b]})

    rough = tostring(rdf_root, encoding="unicode", method="xml")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ", encoding="utf-8")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(pretty)
    logger.info("CIM XML: %s" % output_path)
    return os.path.abspath(output_path)


def generate_svg_from_pandapower(net, output_path, network_name="ExampleNetwork",
                                  width=1200.0, height=800.0):
    """Generate SVG from PandaPower network"""
    n = len(net.bus)
    if n == 0:
        raise ValueError("No buses")
    sns = "http://www.w3.org/2000/svg"
    root = Element("{%s}svg" % sns)
    root.set("xmlns", sns)
    root.set("viewBox", "0 0 %d %d" % (int(width), int(height)))
    root.set("width", str(int(width)))
    root.set("height", str(int(height)))

    sty = SubElement(root, "style")
    sty.text = ".bus{fill:#2196F3;stroke:#1565C0;stroke-width:2}.line{stroke:#333;stroke-width:2;fill:none}.trafo{fill:#FF9800;stroke:#E65100;stroke-width:2}.load{fill:#F44336;stroke:#B71C1C;stroke-width:2}.source{fill:#4CAF50;stroke:#1B5E20;stroke-width:2}.sw_c{fill:#9C27B0;stroke:#4A148C;stroke-width:2}.sw_o{fill:#E0E0E0;stroke:#757575;stroke-width:2;stroke-dasharray:4,2}.label{font-size:10px;font-family:Arial;text-anchor:middle}.title{font-size:16px;font-family:Arial;font-weight:bold}"

    mx, my = 100.0, 100.0
    bs = (width - 2*mx) / max(n, 1)
    mid_y = height * 0.45
    bp = {}
    for i, idx in enumerate(net.bus.index):
        bp[int(idx)] = (mx + bs*(i+0.5), mid_y)

    # Title
    te = SubElement(root, "{%s}text" % sns)
    te.set("x", str(width/2)); te.set("y", "30"); te.set("class", "title"); te.text = network_name

    # Buses
    for idx, (bx, by) in bp.items():
        bw = 80
        nm = str(net.bus.at[idx, "name"]) if "name" in net.bus.columns else "Bus %d" % idx
        g = SubElement(root, "{%s}g" % sns)
        g.set("id", nm.replace(" ", "_"))
        g.set("transform", "translate(%.1f,%.1f)" % (bx-bw/2, by))
        ln = SubElement(g, "{%s}line" % sns)
        ln.set("x1","0"); ln.set("y1","0"); ln.set("x2",str(bw)); ln.set("y2","0"); ln.set("class","bus")
        tx = SubElement(g, "{%s}text" % sns)
        tx.set("x",str(bw/2)); tx.set("y","-10"); tx.set("class","label"); tx.text = nm

    # Lines
    for idx in net.line.index:
        if not bool(net.line.at[idx, "in_service"]): continue
        fb, tb = int(net.line.at[idx, "from_bus"]), int(net.line.at[idx, "to_bus"])
        if fb not in bp or tb not in bp: continue
        x1,y1 = bp[fb]; x2,y2 = bp[tb]
        nm = str(net.line.at[idx, "name"]) if "name" in net.line.columns else "Line_%d" % idx
        g = SubElement(root, "{%s}g" % sns); g.set("id", nm.replace(" ","_"))
        ln = SubElement(g, "{%s}line" % sns)
        ln.set("x1",str(x1)); ln.set("y1",str(y1+5)); ln.set("x2",str(x2)); ln.set("y2",str(y2+5)); ln.set("class","line")

    # Transformers
    for idx in net.trafo.index:
        hv, lv = int(net.trafo.at[idx, "hv_bus"]), int(net.trafo.at[idx, "lv_bus"])
        if hv not in bp or lv not in bp: continue
        x1,y1=bp[hv]; x2,y2=bp[lv]; cx,cy=(x1+x2)/2,(y1+y2)/2-20
        nm = str(net.trafo.at[idx, "name"]) if "name" in net.trafo.columns else "Trafo_%d" % idx
        g = SubElement(root, "{%s}g" % sns); g.set("id",nm.replace(" ","_")); g.set("transform","translate(%.1f,%.1f)"%(cx,cy))
        c1=SubElement(g,"{%s}circle"%sns); c1.set("cx","0");c1.set("cy","0");c1.set("r","15");c1.set("class","trafo");c1.set("fill","none")
        c2=SubElement(g,"{%s}circle"%sns); c2.set("cx","0");c2.set("cy","15");c2.set("r","15");c2.set("class","trafo");c2.set("fill","none")
        tx=SubElement(g,"{%s}text"%sns); tx.set("x","0");tx.set("y","-20");tx.set("class","label");tx.text=nm

    # Loads
    for idx in net.load.index:
        b = int(net.load.at[idx, "bus"])
        if b not in bp: continue
        bx,by=bp[b]; lx,ly=bx,by+80
        nm = str(net.load.at[idx, "name"]) if "name" in net.load.columns else "Load_%d" % idx
        g=SubElement(root,"{%s}g"%sns); g.set("id",nm.replace(" ","_")); g.set("transform","translate(%.1f,%.1f)"%(lx,ly))
        ar=SubElement(g,"{%s}polygon"%sns); ar.set("points","-8,-10 8,-10 0,10"); ar.set("class","load")
        cn=SubElement(g,"{%s}line"%sns); cn.set("x1","0");cn.set("y1","-10");cn.set("x2","0");cn.set("y2","%.1f"%(-(ly-by)+10));cn.set("class","line")
        tx=SubElement(g,"{%s}text"%sns); tx.set("x","0");tx.set("y","25");tx.set("class","label");tx.text=nm

    # External grids
    for idx in net.ext_grid.index:
        b = int(net.ext_grid.at[idx, "bus"])
        if b not in bp: continue
        bx,by=bp[b]; sx,sy=bx,by-80
        nm = str(net.ext_grid.at[idx, "name"]) if "name" in net.ext_grid.columns else "ExtGrid_%d" % idx
        g=SubElement(root,"{%s}g"%sns); g.set("id",nm.replace(" ","_")); g.set("transform","translate(%.1f,%.1f)"%(sx,sy))
        ci=SubElement(g,"{%s}circle"%sns); ci.set("cx","0");ci.set("cy","0");ci.set("r","12");ci.set("class","source")
        cn=SubElement(g,"{%s}line"%sns); cn.set("x1","0");cn.set("y1","12");cn.set("x2","0");cn.set("y2","%.1f"%(by-sy-12));cn.set("class","line")
        tx=SubElement(g,"{%s}text"%sns); tx.set("x","0");tx.set("y","-18");tx.set("class","label");tx.text=nm

    # Switches
    for idx in net.switch.index:
        et=str(net.switch.at[idx,"et"]); b=int(net.switch.at[idx,"bus"]); el=int(net.switch.at[idx,"element"])
        closed=bool(net.switch.at[idx,"closed"])
        if et!="b" or el not in bp or b not in bp: continue
        x1,y1=bp[b]; x2,y2=bp[el]; cx,cy=(x1+x2)/2,(y1+y2)/2+20
        nm="Switch_%d"%idx
        g=SubElement(root,"{%s}g"%sns); g.set("id",nm)
        g.set("class","breaker_closed" if closed else "breaker_open")
        g.set("transform","translate(%.1f,%.1f)"%(cx,cy))
        if closed:
            r=SubElement(g,"{%s}rect"%sns); r.set("x","-8");r.set("y","-4");r.set("width","16");r.set("height","8");r.set("class","sw_c")
        else:
            ln=SubElement(g,"{%s}line"%sns); ln.set("x1","-8");ln.set("y1","0");ln.set("x2","8");ln.set("y2","-8");ln.set("class","sw_o")
        tx=SubElement(g,"{%s}text"%sns); tx.set("x","0");tx.set("y","-12");tx.set("class","label");tx.text=nm

    rough = tostring(root, encoding="unicode", method="xml")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty)
    logger.info("SVG: %s" % output_path)
    return os.path.abspath(output_path)


def generate_example_dataset(network_name="case33bw", output_dir=None,
                              inject_anomalies=False):
    """Generate CIM + SVG example dataset from PandaPower network"""
    import importlib
    try:
        mod = importlib.import_module("pandapower.networks.%s" % network_name)
        func = getattr(mod, network_name)
    except (ImportError, AttributeError):
        func = getattr(pn, network_name)
    net = func()
    pp.runpp(net)

    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "example_data")
    os.makedirs(output_dir, exist_ok=True)

    cim_path = os.path.join(output_dir, "%s.xml" % network_name)
    generate_cim_from_pandapower(net, cim_path, network_name=network_name)

    svg_path = os.path.join(output_dir, "%s.svg" % network_name)
    svg_w = max(1200, len(net.bus) * 60)
    generate_svg_from_pandapower(net, svg_path, network_name=network_name, width=svg_w, height=800)

    from data_preprocessing.scada_simulator import SCADASimulator
    sim = SCADASimulator(net, seed=42)
    measurements = sim.generate_measurements()

    result = {"cim_path": cim_path, "svg_path": svg_path, "net": net,
              "measurements": measurements, "ground_truth": None}

    if inject_anomalies:
        from data_preprocessing.anomaly_injector import inject_all_anomalies
        from utils.graph_utils import build_graph_from_pandapower
        graph = build_graph_from_pandapower(net)
        cim_devs = []
        svg_devs = []
        for idx in net.bus.index:
            nm = "Bus_%d" % idx
            cim_devs.append({"uri": "#_bus_%d" % idx, "name": nm, "type": "ConductingEquipment", "subtype": "BusbarSection"})
            svg_devs.append({"id": nm, "type": "Bus", "x": float(idx*100), "y": 300.0, "label": nm})
        for idx in net.line.index:
            cim_devs.append({"uri": "#_line_%d" % idx, "name": "Line_%d" % idx, "type": "ConductingEquipment", "subtype": "ACLineSegment"})
        injected = inject_all_anomalies(net, measurements, cim_devs, svg_devs, seed=42)
        result["net_injected"] = injected["net"]
        result["measurements_injected"] = injected["measurements"]
        result["ground_truth"] = injected["ground_truth"]
        result["injection_details"] = injected["injection_details"]

    logger.info("Dataset generated: %s" % network_name)
    return result


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Generate CIM/SVG example data")
    parser.add_argument("--network", default="case33bw")
    parser.add_argument("--output", default=None)
    parser.add_argument("--inject-anomalies", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        for nn in ["example_simple", "case33bw", "case14", "case_ieee30", "cigre_mv"]:
            try:
                print("\n" + "="*60 + "\n  Generating: " + nn + "\n" + "="*60)
                r = generate_example_dataset(nn, output_dir=args.output, inject_anomalies=args.inject_anomalies)
                print("  CIM: " + r["cim_path"])
                print("  SVG: " + r["svg_path"])
            except Exception as e:
                print("  [ERROR] %s: %s" % (nn, e))
    else:
        r = generate_example_dataset(args.network, output_dir=args.output, inject_anomalies=args.inject_anomalies)
        print("\nCIM: " + r["cim_path"])
        print("SVG: " + r["svg_path"])
        if r.get("ground_truth"):
            print("Anomalies: %d" % len(r["ground_truth"]))
