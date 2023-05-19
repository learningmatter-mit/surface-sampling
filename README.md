# Virtual Surface Site Reconstruction-Monte Carlo (VSSR-MC)

## Contents

- [Overview](#overview)
- [System requirements](#system-requirements)
- [Setup](#setup)
- [Demo](#demo)
- [Contribute](#contribute)


# Overview
This is the VSSR-MC algorithm for sampling surface reconstructions. VSSR-MC samples across both compositional and configuration spaces. It can interface with both a neural network potential (through ASE) or a classical potential (through ASE or LAMMPS). It is a key component of the Heuristic-Free Surface Reconstruction (HFSurfRecon) pipeline described in the following work:

"Machine learning-accelerated simulations enable heuristic-free surface reconstruction", by X. Du, J.K. Damewood, J.R. Lunger, R. Millan, B. Yildiz, L. Li, and R. Gómez-Bombarelli. https://doi.org/10.48550/arXiv.2305.07251

Please cite use if you find this work useful. Let us know in `issues` if you encounter any problems or have any questions.

# System requirements

## Hardware requirements
We tested out the code on a 12-core Intel(R) Core(TM) i9-7920X CPU @ 2.90GHz with 125 GB RAM.

For optimal performance, we recommend a computer with the following specs:

RAM: 16+ GB
CPU: 4+ cores, 3 GHz/core

To run with a neural network force field, a GPU is recommended. We ran on a single NVIDIA GeForce RTX 2080 Ti 11 GB GPU.

## Software requirements

The code has been tested till the XXX branch YYY revision.
### Operating system

This package has been tested on *Linux* Ubuntu 20.04.6 LTS. It should work on many *Linux* systems.
### Conda environment
The (Conda)(https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) is required. Either Miniconda or Anaconda should be installed.

Following that, the Python dependencies for the code can be installed with the following command
```
conda env create -f environment.yml
```

Installation should take only a few minutes


### Additional software

1. [LAMMPS](https://docs.lammps.org/Install.html) for classical force field optimization
2. [NFF](https://github.com/learningmatter-mit/NeuralForceField) for neural network force field

# Setup

Add the following to `~.bashrc` or equivalent
```
export SURFSAMPLINGDIR="/path/to/surface_sampling"
export PYTHONPATH="$SURFSAMPLINGDIR:$PYTHONPATH"

export LAMMPS_COMMAND="/path/to/lammps/src/lmp_serial"
export LAMMPS_POTENTIALS="/path/to/lammps/potentials/"
export ASE_LAMMPSRUN_COMMAND="$LAMMPS_COMMAND"
```

# Demo
Demos can be found in the `tutorials` folder
### Toy example Copper (Cu):
`tutorials/example.ipynb`

### GaN(0001) surface reconstruction with Tersoff potential
`tutorials/GaN.ipynb`
### SrTiO3(001) surface sampling with NFF
Likely slow... ?? maybe
`tutorials/SrTiO3.ipynb`

# Contribute
## Run the following to initialize Git `pre-commit` package:
`pre-commit install`
