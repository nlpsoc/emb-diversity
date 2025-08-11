# Diversity Measurement Package

A minimal Python package with a simple diversity function.

## Install (dev)
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
```
or 
```bash
conda create -n measure_diversity python=3.11 -y
conda activate measure_diversity
```
Then install the package and its dependencies:
```bash
pip install -r requirements-dev.txt
pip install -e . # Install in editable mode
```

## Usage
```python
from measure_diversity import calculate_diversity
result = calculate_diversity([[0,1], [0,0]])
print(result)  # Output: 5
```