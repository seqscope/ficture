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

Create a virtual environment and install dependencies
```
pip install -r requirements.txt
```
Install
```
pip install ficture
```

## Example - Simulated data
See `./examples/simulation.md` for a complete example including simulating a high reoluation dataset, running FICTURE, and evaluating the result.


## Example - Real data
See `./examples/realdata.md` for commands of processing an example data in `./examples/data/`. The data a subset from a public [Vizgen MERSCOPE mouse liver dataset](https://info.vizgen.com/mouse-liver-access) (mouse liver 1 slice 1).
