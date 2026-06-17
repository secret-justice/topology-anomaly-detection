# Hybrid Intelligence for Topology Anomaly Detection and Correction in Power Distribution Networks

This repository contains the source code, trained models, and experimental data for the paper:

> **Hybrid Intelligence for Topology Anomaly Detection and Correction in Power Distribution Networks**
> Submitted to *Frontiers of Computer Science* (FCS), 2026

## Abstract

We propose a three-layer hybrid intelligence framework that integrates symbolic rule reasoning, physics-informed state estimation, and graph neural network (GNN)-based adaptive detection for comprehensive topology anomaly detection and correction in power distribution networks. The framework achieves 78.1% global anomaly recall (83.5% with improved GNN) across 30 test networks spanning 3 to 1,888 buses.

## Repository Structure

```
├── paper/                  # LaTeX source and figures
│   ├── main.tex           # Paper source
│   ├── ref.bib            # Bibliography (63 references)
│   ├── figures/           # All figures (PDF + PNG)
│   └── FCS_Highlights_3pages.pptx
├── src/                   # Source code
│   ├── anomaly_detection/ # Core detection engine
│   ├── correction_engine/ # Correction logic
│   ├── data_preprocessing/# Data pipeline
│   ├── api/               # FastAPI backend
│   ├── visualization/     # D3.js frontend
│   ├── utils/             # Utility functions
│   ├── config.py          # Configuration
│   └── run_mvp.py         # Main entry point
├── models/                # Trained GNN models
│   ├── gnn_model_best.pt  # Best GNN model
│   ├── gnn_binary_best.pt # Binary classifier
│   └── gnn_gae_model.pt   # Graph autoencoder
├── data/
│   ├── benchmarks/        # Benchmark results (JSON)
│   └── experiments/       # Paper experiment data
├── tests/                 # Test suite (pytest)
└── docs/                  # Documentation
```

## Quick Start

```bash
# Install dependencies
pip install -r src/requirements.txt

# Run the MVP
python src/run_mvp.py

# Run tests
pytest tests/ -v
```

## Key Results

| Metric | Value |
|--------|-------|
| Global anomaly recall | 78.1% (83.5% with improved GNN) |
| Network-level recall | 99.4% ± 0.5% |
| Per-type recall (TI/MTC/SMC) | ≥ 95% |
| Average processing time | 0.94 s |
| Test networks | 30 (3--1,888 buses) |

## Datasets Used

- [PandaPower](https://pandapower.readthedocs.io/) (46 networks)
- [SimBench](https://simbench.net/) (123 networks)
- [IEEE PES Test Feeders](https://site.ieee.org/pes-testfeeders/)
- [CIGRE Benchmark Systems](https://www.cigre.org/)
- Chinese distribution network models (14 models)

## Citation

```bibtex
@article{topology_anomaly_2026,
  title={Hybrid Intelligence for Topology Anomaly Detection and Correction in Power Distribution Networks},
  author={[Authors]},
  journal={Frontiers of Computer Science},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
