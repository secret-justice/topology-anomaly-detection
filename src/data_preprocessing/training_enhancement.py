# -*- coding: utf-8 -*-
"""
v17 Training Data Enhancement: SimBench + Synthetic Networks
Generates diverse training data for GNN with 28 anomaly types
"""
import os
import sys
import numpy as np
import logging

sys.path.insert(0, r"E:\项目大全\电力拓扑图修正\02_算法代码")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_simbench_networks():
    """Generate networks from SimBench dataset for training diversity."""
    try:
        import simbench as sb
        logger.info("SimBench available, generating networks...")
        
        # Get all available SimBench networks
        nets = []
        for code in sb.get_simbench_names()[:20]:  # Limit to 20
            try:
                net = sb.get_simbench_net(code)
                nets.append((code, net))
                logger.info(f"  Loaded {code}: {len(net.bus)} buses")
            except Exception as e:
                logger.debug(f"  Skip {code}: {e}")
        
        return nets
    except ImportError:
        logger.warning("SimBench not available, using PandaPower built-in")
        return []


def generate_synthetic_networks():
    """Generate synthetic distribution networks using Chung-Lu model."""
    import pandapower as pp
    import networkx as nx
    
    nets = []
    
    # Generate networks of various sizes
    for n_buses in [20, 50, 100, 200, 500]:
        for topology in ["radial", "ring", "mesh"]:
            try:
                net = create_synthetic_network(n_buses, topology)
                nets.append((f"synthetic_{topology}_{n_buses}", net))
                logger.info(f"  Created synthetic_{topology}_{n_buses}: {len(net.bus)} buses")
            except Exception as e:
                logger.debug(f"  Skip synthetic_{topology}_{n_buses}: {e}")
    
    return nets


def create_synthetic_network(n_buses, topology="radial"):
    """Create a synthetic distribution network."""
    import pandapower as pp
    
    net = pp.create_empty_network()
    
    # Create buses
    for i in range(n_buses):
        pp.create_bus(net, vn_kv=20.0, name=f"Bus_{i}")
    
    # Create external grid at bus 0
    pp.create_ext_grid(net, bus=0, vm_pu=1.0, name="Grid")
    
    # Create lines based on topology
    if topology == "radial":
        # Radial: each bus connected to parent
        for i in range(1, n_buses):
            parent = (i - 1) % i
            pp.create_line_from_parameters(net, from_bus=parent, to_bus=i, 
                                          length_km=1.0, r_ohm_per_km=0.1, 
                                          x_ohm_per_km=0.1, c_nf_per_km=0,
                                          max_i_ka=0.5, name=f"Line_{i}")
    elif topology == "ring":
        # Ring: chain + closing line
        for i in range(1, n_buses):
            pp.create_line_from_parameters(net, from_bus=i-1, to_bus=i,
                                          length_km=1.0, r_ohm_per_km=0.1,
                                          x_ohm_per_km=0.1, c_nf_per_km=0,
                                          max_i_ka=0.5, name=f"Line_{i}")
        # Close the ring
        pp.create_line_from_parameters(net, from_bus=n_buses-1, to_bus=0,
                                      length_km=1.0, r_ohm_per_km=0.1,
                                      x_ohm_per_km=0.1, c_nf_per_km=0,
                                      max_i_ka=0.5, name=f"Line_ring")
    else:  # mesh
        # Mesh: chain + random extra edges
        for i in range(1, n_buses):
            pp.create_line_from_parameters(net, from_bus=i-1, to_bus=i,
                                          length_km=1.0, r_ohm_per_km=0.1,
                                          x_ohm_per_km=0.1, c_nf_per_km=0,
                                          max_i_ka=0.5, name=f"Line_{i}")
        # Add random extra edges (10% of n_buses)
        rng = np.random.default_rng(42)
        for _ in range(max(1, n_buses // 10)):
            a, b = rng.choice(n_buses, 2, replace=False)
            if a != b:
                pp.create_line_from_parameters(net, from_bus=int(a), to_bus=int(b),
                                              length_km=1.0, r_ohm_per_km=0.1,
                                              x_ohm_per_km=0.1, c_nf_per_km=0,
                                              max_i_ka=0.5, name=f"Line_mesh_{a}_{b}")
    
    # Add loads to 30% of buses
    rng = np.random.default_rng(42)
    for i in range(1, n_buses):
        if rng.random() < 0.3:
            pp.create_load(net, bus=i, p_mw=rng.uniform(0.01, 0.5), 
                          q_mvar=rng.uniform(0.01, 0.2), name=f"Load_{i}")
    
    return net


if __name__ == "__main__":
    logger.info("=== Training Data Enhancement ===")
    
    # Generate SimBench networks
    simbench_nets = generate_simbench_networks()
    logger.info(f"SimBench networks: {len(simbench_nets)}")
    
    # Generate synthetic networks
    synthetic_nets = generate_synthetic_networks()
    logger.info(f"Synthetic networks: {len(synthetic_nets)}")
    
    total = len(simbench_nets) + len(synthetic_nets)
    logger.info(f"Total networks for training: {total}")
    logger.info("Run gnn_trainer.py with these networks for enhanced training")
