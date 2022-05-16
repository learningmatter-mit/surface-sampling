# surface_sampling

Use own conda env with required packages in `env.yml` or install a new conda env from that file using command
`conda env create -f environment.yml`

Requires [CatKit](https://github.com/SUNCAT-Center/CatKit): 

Run this command inside your environment
`pip install git+https://github.com/SUNCAT-Center/CatKit.git`

LAMMPS install required. Change `os.environ["LAMMPS_COMMAND"]` and `os.environ["LAMMPS_POTENTIALS"]` in `mcmc.py`.

Main script is:
`sgmc_surf/mcmc.py`

Example mcmc surface reconstruction for Copper (Cu) in:
`sgmc_surf/example.ipynb`
