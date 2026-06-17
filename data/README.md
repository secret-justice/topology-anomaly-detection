# Data Directory

This directory contains benchmark results, trained models, and test network data used in the paper.

## Directory Structure

```
data/
├── networks/          # Test network models (PandaPower JSON format)
│   └── China_*.json   # 14 Chinese 10kV distribution networks
├── benchmarks/        # Multi-seed benchmark results (v4, v7, v8, v9)
├── experiments/       # Individual experiment results (RQ1-RQ9)
└── *.json             # Summary/validation files
```

## Test Networks

### Included in This Repository

**Chinese 10kV Distribution Networks** (14 models, 1.4 MB total):
Custom-built PandaPower models conforming to Chinese national standards:
- GB/T 50217-2018 (Cable engineering design)
- GB/T 12325-2008 (Power quality - voltage deviation)
- DL/T 621-1997 (Grounding system)

| Network | Buses | Lines | Loads | Category |
|---------|-------|-------|-------|----------|
| China_CBD_Commercial | 84 | 81 | 80 | Urban |
| China_Hospital_District | 34 | 31 | 30 | Urban |
| China_Industrial_Park | 58 | 49 | 54 | Industrial |
| China_Residential_High | 60 | 57 | 56 | Urban |
| China_Data_Center | 36 | 33 | 32 | Industrial |
| China_Township | 51 | 41 | 47 | Suburban |
| China_Rural_Overhead | 18 | 15 | 16 | Rural |
| China_University_Campus | 19 | 16 | 17 | Urban |
| China_New_Energy_Park | 53 | 49 | 49 | DER |
| China_Suburban_Mixed | 66 | 61 | 62 | Suburban |
| China_Large_110_10kV_DER | 57 | 52 | 53 | DER |
| China_10kV_CityUrban | 32 | 22 | 24 | Urban |
| China_10kV_CitySuburban | 32 | 22 | 24 | Suburban |
| China_10kV_Rural | 32 | 22 | 24 | Rural |

Loading example:
```python
import pandapower as pp
net = pp.from_json("data/networks/China_CBD_Commercial.json")
```

### External (Not Included)

The following networks are publicly available and must be obtained separately:

**PandaPower Built-in** (97 networks):
```python
pip install pandapower
import pandapower.networks as pn
net = pn.case33bw()  # IEEE 33-bus
net = pn.case118()   # IEEE 118-bus
```

**SimBench** (123 networks):
Download from [SimBench](https://simbench.de/en/)
Requires: `pip install simbench`

**IEEE PES Test Feeders**:
Available from [IEEE PES Test Feeder](https://ewh.ieee.org/soc/pes/dsacom/testfeeders/)

**CIGRE Test Networks**:
Available through PandaPower: `pn.cigre_mv()`, `pn.cigre_hv()`

## Benchmark Results

- `benchmark_v7_multiseed.json`: Main results (30 networks × 5 seeds)
- `benchmark_v8_expanded.json`: Extended benchmark (15 anomaly types)
- `hard_anomaly_results.json`: Layer-specific hard anomaly ablation
- `china_validation.json`: Chinese network validation (RQ9)

## Trained Models

- `gnn_model_best.pt`: Best GNN model (multi-network training)
- `gnn_binary_best.pt`: Binary classification GNN
- `gnn_gae_model.pt`: Graph autoencoder model
