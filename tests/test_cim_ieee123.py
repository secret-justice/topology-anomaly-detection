# -*- coding: utf-8 -*-
"""
测试CIM XML解析 + IEEE 123节点拓扑构建
验证CIM解析器能否正确读取IEEE 123节点馈线模型
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing.cim_parser import parse_cim_rdf
from utils.graph_utils import build_graph_from_cim
import networkx as nx

CIM_FILE = r"E:\项目大全\电力拓扑图修正\07_参考文献\CIM_示例数据\CIMHub-master\model_output_tests\IEEE123.xml"

def test_parse_ieee123():
    """解析IEEE123 CIM文件并验证"""
    print("=" * 60)
    print("  IEEE 123 Node CIM XML Parse Test")
    print("=" * 60)
    
    # Step 1: Parse CIM
    print("\n[1] Parsing CIM XML file...")
    cim_data = parse_cim_rdf(CIM_FILE)
    print(f"  CIM namespace: {cim_data.get('cim_namespace', 'N/A')}")
    print(f"  Devices: {len(cim_data['devices'])}")
    print(f"  Terminals: {len(cim_data['terminals'])}")
    print(f"  Connectivity Nodes: {len(cim_data['nodes'])}")
    print(f"  Switches: {len(cim_data['switches'])}")
    print(f"  Connections: {len(cim_data['connections'])}")
    
    # Step 2: Device type distribution
    print("\n[2] Device type distribution:")
    type_count = {}
    for dev in cim_data['devices']:
        st = dev.get('subtype', 'Unknown')
        type_count[st] = type_count.get(st, 0) + 1
    for t, c in sorted(type_count.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")
    
    # Step 3: Build topology graph
    print("\n[3] Building NetworkX topology graph...")
    G = build_graph_from_cim(cim_data)
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Connected components: {nx.number_connected_components(G)}")
    
    # Step 4: Switch status analysis
    print("\n[4] Switch status:")
    open_count = sum(1 for s in cim_data['switches'] if s.get('normal_open'))
    closed_count = len(cim_data['switches']) - open_count
    print(f"  Normally open: {open_count}")
    print(f"  Normally closed: {closed_count}")
    
    # Step 5: Topology validation
    print("\n[5] Topology validation:")
    if nx.is_connected(G):
        print("  [OK] Graph is connected")
    else:
        components = list(nx.connected_components(G))
        print(f"  [WARN] Graph not connected, {len(components)} components")
        for i, comp in enumerate(components[:5]):
            print(f"    Component {i}: {len(comp)} nodes")
    
    # Step 6: Degree distribution
    print("\n[6] Node degree distribution:")
    degrees = [d for _, d in G.degree()]
    from collections import Counter
    deg_dist = Counter(degrees)
    for deg in sorted(deg_dist.keys()):
        print(f"  Degree={deg}: {deg_dist[deg]} nodes")
    
    print(f"\n{'='*60}")
    print(f"  Test completed!")
    print(f"{'='*60}")
    
    return cim_data, G

if __name__ == "__main__":
    cim_data, G = test_parse_ieee123()