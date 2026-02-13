# scClone2DR: Clone-level multi-modal prediction of tumour drug response

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

**scClone2DR** is a probabilistic multi-modal framework that predicts drug responses at the level of individual tumour clones by integrating single-cell DNA and RNA sequencing with ex-vivo drug-screening data.

## Features

- **Multi-modal Integration**: Combines scRNA-seq, scDNA-seq and drug-screening data
- **Probabilistic Modeling**: Uses Pyro for Bayesian inference
- **Subclone Analysis**: Models drug response at the clone level
- **Visualization Tools**: Comprehensive result analysis and visualization

## Installation

```bash
git clone https://github.com/ETH-NEXUS/scClone2DR_publication.git
cd scClone2DR_publication/package
pip install -e .
```


## Package Structure

```
scClone2DR/
├── datasets/           # Data loading and preprocessing
├── models/            # Model implementations
│   ├── scClone2DR/   # Main model
│   ├── factorization_machine/  # FM baseline
│   └── neural_network/         # NN baseline
├── trainers/          # Training utilities
├── resultanalysis/    # Analysis and visualization tools
└── utils.py          # Utility functions
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.10.0
- Pyro >= 1.8.0
- NumPy >= 1.20.0
- pandas >= 1.3.0
- h5py >= 3.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- tqdm >= 4.60.0
- scikit-learn >= 0.24.0
- scikit-fda>=0.8.1
- nbformat>=5.0.0
- plotly>=5.0.0

## Documentation

For detailed documentation and tutorials, visit:
- [Documentation](https://scClone2DR.readthedocs.io) (coming soon)
- [Tutorial Notebook](tutorial_scClone2DR.ipynb)

## Citation

If you use scClone2DR in your research, please cite:

```bibtex
@article{scClone2DR2026,
  title={Clone-level multi-modal prediction of tumour drug response},
  author={Quentin Duchemin , Daniel Trejo Banos, Anne Bertolini, Pedro F. Ferreira, Rudolf Schill,
Matthias Lienhard, Rebekka Wegmann, Tumor Profiler Consortium, Berend Snijder, Daniel Stekhoven,
Niko Beerenwinkel, Franziska Singer, Guillaume Obozinski and Jack Kuipers},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Quentin Duchemin - qduchemin9@gmail.com

Project Link: [https://github.com/ETH-NEXUS/scClone2DR_publication](https://github.com/ETH-NEXUS/scClone2DR_publication)

## Acknowledgments

This project was partially funded by PHRT and SDSC (grant numbers: 2021-802 and C21-19P).
