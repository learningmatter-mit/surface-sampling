# Virtual Surface Site Relaxation-Monte Carlo (VSSR-MC)
[![Tests](https://github.com/learningmatter-mit/surface-sampling/actions/workflows/tests.yml/badge.svg)](https://github.com/learningmatter-mit/surface-sampling/actions/workflows/tests.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2305.07251-blue?logo=arXiv&logoColor=white&logoSize=auto)](https://arxiv.org/abs/2305.07251)
[![Zenodo](https://img.shields.io/badge/data-10.5281/zenodo.7758174-14b8a6?logo=zenodo&logoColor=white&logoSize=auto)](https://zenodo.org/doi/10.5281/zenodo.7758174)
[![arXiv](https://img.shields.io/badge/arXiv-2503.17870-blue?logo=arXiv&logoColor=white&logoSize=auto)](https://arxiv.org/abs/2503.17870)
[![Zenodo](https://img.shields.io/badge/data-10.5281/zenodo.15066441-14b8a6?logo=zenodo&logoColor=white&logoSize=auto)](https://zenodo.org/doi/10.5281/zenodo.15066441)

## Contents
- [Overview](#overview)
- [System requirements](#system-requirements)
- [Setup](#setup)
- [Demo](#demo)
- [Scripts](#scripts)
- [Citations](#citations)
- [Development & Bugs](#development--bugs)


# Overview
This is the VSSR-MC algorithm for sampling surface reconstructions. VSSR-MC samples across both compositional and configurational spaces. It can interface with both a neural network potential (through [ASE](https://wiki.fysik.dtu.dk/ase/)) or a classical potential (through ASE or [LAMMPS](https://www.lammps.org/)). It is a key component of the Automatic Surface Reconstruction (AutoSurfRecon) pipeline described in the following work: [Machine-learning-accelerated simulations to enable automatic surface reconstruction](https://doi.org/10.1038/s43588-023-00571-7). VSSR-MC can be used to sample either surfaces under gas/vacuum conditions as demonstrated in the [original work](https://doi.org/10.1038/s43588-023-00571-7) or under aqueous electrochemical conditions as described in this work: [Accelerating and enhancing thermodynamic simulations of electrochemical interfaces](https://doi.org/10.48550/arXiv.2503.17870).

![Cover image](site/static/vssr_cover_image.png)

# System requirements
We recommend a computer with the following specs:

- RAM: 16+ GB
- CPU: 4+ cores, 3 GHz/core

To run with a neural network force field, a GPU is recommended. We ran on a single NVIDIA GeForce RTX 2080 Ti 11 GB GPU. The code has been tested on *Linux* Ubuntu 20.04.6 LTS but we expect it to work on other *Linux* distributions.

# Setup
To start, run `git clone git@github.com:learningmatter-mit/surface-sampling.git` to your local directory or a workstation.

## Conda environment
We recommend creating a new [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) environment. Following that, the Python dependencies for the code can be installed. In the `surface-sampling` directory, run the following commands:
```bash
conda create -n vssr-mc python=3.11
conda activate vssr-mc
conda install -c conda-forge kimpy lammps openkim-models
pip install -e .
```
> If you're intending to contribute to the code, you can `pip install -e '.[dev]'` instead to also install the development dependencies.

To run with LAMMPS, add the following to `~/.bashrc` or equivalent with appropriate paths and then `source ~/.bashrc`. `conda` would have installed LAMMPS as a dependency.
```bash
export LAMMPS_COMMAND="/path/to/lammps/src/lmp"
export LAMMPS_POTENTIALS="/path/to/lammps/potentials/"
export ASE_LAMMPSRUN_COMMAND="$LAMMPS_COMMAND"
```

The `LAMMPS_COMMAND` should point to the LAMMPS executable, which can be found here: `/path/to/[vssr-mc-env]/bin/lmp`.
The `LAMMPS_POTENTIALS` directory should contain the LAMMPS potential files, which can found here: `/path/to/[surface-sampling-repo]/mcmc/potentials/`.
The `ASE_LAMMPSRUN_COMMAND` should point to the same LAMMPS executable. More information can be found here: [ASE LAMMPS](https://wiki.fysik.dtu.dk/ase/ase/calculators/lammpsrun.html).

If the `conda` installed LAMMPS does not work, you might have to install LAMMPS from source. More information can be found here: [LAMMPS](https://lammps.sandia.gov/doc/Build.html).

You might have to re-open/re-login to your terminal shell for the new settings to take effect.

# Demo
A toy demo and other examples can be found in the `tutorials/` folder.
```
tutorials/
├── example.ipynb
├── GaN_0001.ipynb
├── Si_111_5x5.ipynb
├── SrTiO3_001.ipynb
├── latent_space_clustering.ipynb
└── prepare_surface.ipynb
```
 More data/examples can be found in our Zenodo datasets: [1](https://doi.org/10.5281/zenodo.7758174) and [2](https://doi.org/10.5281/zenodo.15066440).

## Toy example of Cu(100)
A toy example to illustrate the use of VSSR-MC. It should only take about a few seconds to run. Refer to `tutorials/example.ipynb`.

## GaN(0001) surface sampling with Tersoff potential
This example could take a few minutes to run. Refer to `tutorials/GaN_0001.ipynb`.

## Si(111) 5x5 surface sampling with modified Stillinger–Weber potential
This example could take a few minutes to run. Refer to `tutorials/Si_111_5x5.ipynb`.

## SrTiO3(001) surface sampling with machine learning potential
Demonstrates the integration of VSSR-MC with a neural network force field. This example could take a few minutes to run. Refer to `tutorials/SrTiO3_001.ipynb`.

## Clustering MC-sampled surfaces in the latent space
Retrieves the neural network embeddings of VSSR-MC structures and performs clustering. This example should only take a minute to run. Refer to `tutorials/latent_space_clustering.ipynb`.

## Preparing surface from a bulk structure
This example demonstrates how to cut a surface from a bulk structure. Refer to `tutorials/prepare_surface.ipynb`.


# Scripts
Scripts can be found in the `scripts/` folder, including:
```
scripts/
├── sample_surface.py
├── sample_pourbaix_surface.py
├── clustering.py
└── create_surface_formation_entries.py
```

The arguments for the scripts can be found by running `python /path/to/script.py -h`.

## Example usage:
### Original VSSR-MC with PaiNN model trained on SrTiO3(001) surfaces
```bash
python scripts/sample_surface.py --run_name "SrTiO3_001_painn" \
--starting_structure_path "tutorials/data/SrTiO3_001/SrTiO3_001_2x2_pristine_slab.pkl" \
--model_type "PaiNN" --model_paths "tutorials/data/SrTiO3_001/nff/model01/best_model" \
"tutorials/data/SrTiO3_001/nff/model02/best_model" \
"tutorials/data/SrTiO3_001/nff/model03/best_model" \
--settings_path "scripts/configs/sample_config_painn.json"
```

### Pre-trained CHGNet model on SrTiO3(001) surfaces
```bash
python scripts/sample_surface.py --run_name "SrTiO3_001_chgnet" \
--starting_structure_path "tutorials/data/SrTiO3_001/SrTiO3_001_2x2_pristine_slab.pkl" \
--model_type "CHGNetNFF" --settings_path "scripts/configs/sample_config_chgnet.json"
```

### Pre-trained CHGNet model on LaMnO3(001) under pH-$U_\mathrm{SHE}$ conditions
```bash
python scripts/sample_pourbaix_surface.py --run_name LaMnO3_001_chgnet \
--starting_structure_path "tutorials/data/LaMnO3_001/LaMnO3_001_2x2x3_top_pristine.pkl" --model_type CHGNetNFF \
--phase_diagram_path "tutorials/data/LaMnO3_001/pourbaix/LaMnO_pd_dict.json" \
--pourbaix_diagram_path  "tutorials/data/LaMnO3_001/pourbaix/LaMnO_no_ternary_pbx_dict.json" \
--settings_path "scripts/configs/sample_pourbaix_config.json"
```

### Latent space clustering
```bash
python scripts/clustering.py --file_paths "tutorials/data/SrTiO3_001/SrTiO3_001_2x2_mcmc_structures_100.pkl" \
--save_folder "SrTiO3_001/clustering" --nff_model_type "PaiNN" \
--nff_paths "tutorials/data/SrTiO3_001/nff/model01/best_model" \
"tutorials/data/SrTiO3_001/nff/model02/best_model" \
"tutorials/data/SrTiO3_001/nff/model03/best_model" \
--clustering_metric "force_std" --cutoff_criterion "distance" \
--clustering_cutoff 0.2 --nff_device "cuda"
```

### Create surface surface formation entries for Pourbaix analysis
```bash
python scripts/create_surface_formation_entries.py --surface_name "LaMnO3_001_2x2" \
--file_paths "tutorials/data/LaMnO3_001/20241120-003720_AtomsBatch_surface_48.pkl" --model_type "CHGNetNFF" \
--model_paths "tutorials/data/LaMnO3_001/nff/finetuned/best_model" \
--phase_diagram_path "tutorials/data/LaMnO3_001/pourbaix/LaMnO_pd_dict.json" \
--pourbaix_diagram_path "tutorials/data/LaMnO3_001/pourbaix/LaMnO_no_ternary_pbx_dict.json" --correct_hydroxide_energy \
--input_job_id --elements "La" "Mn" "O" --device "cuda" --save_folder "tutorials/data/LaMnO3_001/pourbaix/"
```

# Citations
1. Original VSSR-MC work:
```bib
@article{duMachinelearningacceleratedSimulationsEnable2023,
  title = {Machine-Learning-Accelerated Simulations to Enable Automatic Surface Reconstruction},
  author = {Du, Xiaochen and Damewood, James K. and Lunger, Jaclyn R. and Millan, Reisel and Yildiz, Bilge and Li, Lin and {G{\'o}mez-Bombarelli}, Rafael},
  year = {2023},
  month = dec,
  journal = {Nature Computational Science},
  pages = {1--11},
  publisher = {Nature Publishing Group},
  issn = {2662-8457},
  doi = {10.1038/s43588-023-00571-7},
  urldate = {2023-12-07},
  keywords = {Computational methods,Computational science,Software,Surface chemistry}
}
```

2. VSSR-MC with aqueous electrochemical conditions:
```bib
@misc{duAcceleratingEnhancingThermodynamic2025,
  title = {Accelerating and Enhancing Thermodynamic Simulations of Electrochemical Interfaces},
  author = {Du, Xiaochen and Liu, Mengren and Peng, Jiayu and Chun, Hoje and Hoffman, Alexander and Yildiz, Bilge and Li, Lin and Bazant, Martin Z. and {G{\'o}mez-Bombarelli}, Rafael},
  year = {2025},
  month = mar,
  number = {arXiv:2503.17870},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2503.17870},
  keywords = {Computer Science - Computational Engineering Finance and Science,Computer Science - Machine Learning,Condensed Matter - Materials Science,Condensed Matter - Statistical Mechanics},
}
```

# Development & Bugs
VSSR-MC is under active development, if you encounter any bugs in installation and usage,
please open an [issue](https://github.com/learningmatter-mit/surface-sampling/issues). We appreciate your contributions!
