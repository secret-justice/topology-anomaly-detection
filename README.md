# Hybrid Intelligence for Topology Anomaly Detection and Correction in Power Distribution Networks

This repository contains the source code, trained models, and experimental data for the paper:

> **Hybrid Intelligence for Topology Anomaly Detection and Correction in Power Distribution Networks**
> Submitted to *Frontiers of Computer Science* (FCS), 2026

## Abstract

We propose a three-layer hybrid intelligence framework that integrates symbolic rule reasoning, physics-informed state estimation, and graph neural network (GNN)-based adaptive detection for comprehensive topology anomaly detection and correction in power distribution networks. The framework achieves 78.1% global anomaly recall (83.5% with improved GNN) across 30 test networks spanning 3 to 1,888 buses.

## Repository Structure

```
鈹溾攢鈹€ paper/                  # LaTeX source and figures
鈹?  鈹溾攢鈹€ main.tex           # Paper source
鈹?  鈹溾攢鈹€ ref.bib            # Bibliography (63 references)
鈹?  鈹溾攢鈹€ figures/           # All figures (PDF + PNG)
鈹?  鈹斺攢鈹€ FCS_Highlights_3pages.pptx
鈹溾攢鈹€ src/                   # Source code
鈹?  鈹溾攢鈹€ anomaly_detection/ # Core detection engine
鈹?  鈹溾攢鈹€ correction_engine/ # Correction logic
鈹?  鈹溾攢鈹€ data_preprocessing/# Data pipeline
鈹?  鈹溾攢鈹€ api/               # FastAPI backend
鈹?  鈹溾攢鈹€ visualization/     # D3.js frontend
鈹?  鈹溾攢鈹€ utils/             # Utility functions
鈹?  鈹溾攢鈹€ config.py          # Configuration
鈹?  鈹斺攢鈹€ run_mvp.py         # Main entry point
鈹溾攢鈹€ models/                # Trained GNN models
鈹?  鈹溾攢鈹€ gnn_model_best.pt  # Best GNN model
鈹?  鈹溾攢鈹€ gnn_binary_best.pt # Binary classifier
鈹?  鈹斺攢鈹€ gnn_gae_model.pt   # Graph autoencoder
鈹溾攢鈹€ data/
鈹?  鈹溾攢鈹€ benchmarks/        # Benchmark results (JSON)
鈹?  鈹斺攢鈹€ experiments/       # Paper experiment data
鈹溾攢鈹€ tests/                 # Test suite (pytest)
鈹斺攢鈹€ docs/                  # Documentation
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
| Network-level recall | 99.4% 卤 0.5% |
| Per-type recall (TI/MTC/SMC) | 鈮?95% |
| Average processing time | 0.94 s |
| Test networks | 30 (3--1,888 buses) |

## Datasets Used

- [PandaPower](https://pandapower.readthedocs.io/) (46 networks)
- [SimBench](https://simbench.net/) (123 networks)
- [IEEE PES Test Feeders](https://site.ieee.org/pes-testfeeders/)
- [CIGRE Benchmark Systems](https://www.cigre.org/)
- Chinese distribution network models (14 models)


## Data

### Included Datasets
- **Chinese 10kV Distribution Networks**: 14 models covering urban, suburban, rural, industrial, and DER-integrated configurations. See [`data/networks/`](data/networks/).
- **Benchmark Results**: Complete experimental results for RQ1-RQ9. See [`data/experiments/`](data/experiments/).
- **Trained Models**: GNN model weights (.pt format). See [`models/`](models/).

### External Datasets
PandaPower (97 built-in networks), SimBench (123 networks), IEEE PES test feeders, and CIGRE test networks are publicly available. See [`data/README.md`](data/README.md) for details.

## Citation

```bibtex
@article{topology_anomaly_2026,
  title={Hybrid Intelligence for Topology Anomaly Detection and Correction in Power Distribution Networks},
  author={Hanbo Wang},
  journal={Frontiers of Computer Science},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.


