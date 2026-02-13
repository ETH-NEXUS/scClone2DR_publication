# scClone2DR Installation Guide

## Requirements

- Python >= 3.8
- pip or conda

## Installation Method

This is recommended if you want to modify the code or contribute to development.

```bash
# Clone the repository
git clone https://github.com/ETH-NEXUS/scClone2DR_publication.git
cd scClone2DR_publication/package

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

## Verify Installation

```python
import scClone2DR

# Try importing main components
from scClone2DR.models import scClone2DR
from scClone2DR.datasets import RealData
from scClone2DR.trainers import Trainer
```