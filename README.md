# FICTURE

*This repository is under active construction*

FICTURE: Scalable segmentation-free analysis of sub-micron resolution spatial transcriptomics

[preprint](https://biorxiv.org/cgi/content/short/2023.11.04.565621v1)

## General usage
```
ficture <command> <args>
```
Type `ficture` to see the list of available commands.

## Installation
Install after cloning this repository (until we deploy it on pypi):

Create a virtual environment and install dependencies
```
pip install -r requirements.txt
```

Under the root directory of this repository, run
```
python -m build
```

Install
```
pip install dist/ficture-0.0.1-py3-none-any.whl
```

## Example - Simulated data
See `./examples/simulation/simulation.md` for a complete example including simulating a high reoluation dataset, running FICTURE, and evaluating the result.
