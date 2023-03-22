# Heuristic Free (HF) Surface Sampling

Use own conda env with required packages in `env.yml` or install a new conda env from that file using command
`conda env create -f environment.yml`

Requires [CatKit](https://github.com/SUNCAT-Center/CatKit):

Run this command inside your environment
`pip install git+https://github.com/SUNCAT-Center/CatKit.git`

LAMMPS install required.

Add the following to `~.bashrc` or equivalent
```
export SURFSAMPLINGDIR="/path/to/surface_sampling"
export PYTHONPATH="$SURFSAMPLINGDIR:$PYTHONPATH"

export LAMMPS_COMMAND="/path/to/lammps/src/lmp_serial"
export LAMMPS_POTENTIALS="/path/to/lammps/potentials/"
export ASE_LAMMPSRUN_COMMAND="$LAMMPS_COMMAND"
```

### Main script is:
`mcmc/mcmc.py`

### Example mcmc surface reconstruction for Copper (Cu) in:
`tutorials/example.ipynb`

### Run the following to initialize Git `pre-commit` package:
`pre-commit install`
