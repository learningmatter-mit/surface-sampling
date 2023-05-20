# Virtual Surface Site Relaxation-Monte Carlo (VSSR-MC)

## Contents
- [Overview](#overview)
- [System requirements](#system-requirements)
- [Setup](#setup)
- [Demo](#demo)


# Overview

This is the VSSR-MC algorithm for sampling surface reconstructions. VSSR-MC samples across both compositional and configurational spaces. It can interface with both a neural network potential (through ASE) or a classical potential (through ASE or LAMMPS). It is a key component of the Heuristic-Free Surface Reconstruction (HFSurfRecon) pipeline described in the following work:

"Machine learning-accelerated simulations enable heuristic-free surface reconstruction", by X. Du, J.K. Damewood, J.R. Lunger, R. Millan, B. Yildiz, L. Li, and R. Gómez-Bombarelli. https://doi.org/10.48550/arXiv.2305.07251

Please cite use if you find this work useful. Let us know in `issues` if you encounter any problems or have any questions.

# System requirements

## Hardware requirements
We recommend a computer with the following specs:

RAM: 16+ GB
CPU: 4+ cores, 3 GHz/core

We tested out the code on machines with 6+ CPU cores @ 3.0+ GHz/core with 64+ GB of RAM. 

To run with a neural network force field, a GPU is recommended. We ran on a single NVIDIA GeForce RTX 2080 Ti 11 GB GPU.

## Software requirements
The code has been tested up to commit `19c518be8d822b1c8fd3e2723337876713ef1ff2` on the `master` branch.

### Operating system
This package has been tested on *Linux* Ubuntu 20.04.6 LTS but we expect it to be agnostic to the *Linux* system version.

### Conda environment
[Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) is required. Either Miniconda or Anaconda should be installed.

Following that, the Python dependencies for the code can be installed with the following command
```
conda env create -f environment.yml
```

Installation might take several minutes to resolve dependencies.

### Additional software
1. [LAMMPS](https://docs.lammps.org/Install.html) for classical force field optimization
2. [NFF](https://github.com/learningmatter-mit/NeuralForceField) for neural network force field

# Setup

Add the following to `~/.bashrc` or equivalent with appropriate paths and then `source ~/.bashrc`.
```
export SURFSAMPLINGDIR="/path/to/surface_sampling"
export PYTHONPATH="$SURFSAMPLINGDIR:$PYTHONPATH"

export LAMMPS_COMMAND="/path/to/lammps/src/lmp_serial"
export LAMMPS_POTENTIALS="/path/to/lammps/potentials/"
export ASE_LAMMPSRUN_COMMAND="$LAMMPS_COMMAND"

export NFFDIR="/path/to/NeuralForceField"
export PYTHONPATH=$NFFDIR:$PYTHONPATH
```

# Demo

A toy demo and other examples can be found in the `tutorials/` folder. More data/examples can be found in our Zenodo dataset (https://doi.org/10.5281/zenodo.7758174).

### Toy example of Cu(100):
A toy example to illustrate the use of VSSR-MC. It should only take about a minute to run. Refer to `tutorials/example.ipynb`.

### GaN(0001) surface sampling with Tersoff potential
We explicitly generate surface sites using `pymatgen`. This example could take 10 minutes or more to run. Refer to `tutorials/GaN.ipynb`.

### SrTiO3(001) surface sampling with machine learning potential
Demonstrates the integration of VSSR-MC with a neural network force field. This example could take 10 minutes or more to run. Refer to `tutorials/SrTiO3.ipynb`.
