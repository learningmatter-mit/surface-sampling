import os
import sys

import catkit
import numpy as np
from ase.calculators.eam import EAM
from ase.calculators.lammpslib import LAMMPSlib
from ase.calculators.lammpsrun import LAMMPS
from ase.io import read
from catkit.gen.adsorption import get_adsorption_sites
from mcmc import get_adsorption_coords, mcmc_run, slab_energy

sys.path.append("/home/dux/")
sys.path.append("/home/dux/surface_sampling/sgmc_surf")

from htvs.djangochem.pgmols.utils import surfaces

# Try LAMMPS lib
# From previous error: Pair style bop requires comm ghost cutoff at least 3x larger than 3.7
cmds = [
    "comm_modify cutoff 15",
    "pair_style bop save",
    "pair_coeff * * GaAs.bop.table Ga As",
]
# Check if units should be metal or what
# A: The .bop.table potential files provided with LAMMPS (see the potentials directory) are parameterized for metal units.
potential_file = os.path.join(os.environ["LAMMPS_POTENTIALS"], "GaAs.bop.table")

lammps_calc = LAMMPSlib(
    lmpcmds=cmds, atom_types={"Ga": 1, "As": 2}, log_file="test.log"
)

# GaAs from Materials Project
atoms = read("GaAs.cif")
slab, surface_atoms = surfaces.surface_from_bulk(
    atoms, [0, 0, 1], size=[6, 6], vacuum=10
)
slab.write("GaAs_001_4x4_pristine_slab.cif")

# slab.calc = lammps_calc
# print(f"pristine slab energy is {slab_energy(slab)}")

coords, connectivity, sym_idx = get_adsorption_sites(slab, symmetry_reduced=False)
# Pair style bop requires system dimension of at least 22.20

# try just some adsorption
some_ads_slab = read("ads_As_some_adsorbed_slab.cif")
some_ads_slab.calc = lammps_calc

# print(some_ads_slab)
# print(f"some adsorbed slab energy is {slab_energy(some_ads_slab)}")

# # get energy again without changing slab
# print(f"again: some adsorbed slab energy is {slab_energy(some_ads_slab)}")

# # delete something and try again
# del some_ads_slab[-1]
# print(f"adsorbed slab energy after deletion {slab_energy(some_ads_slab)}")

num_runs = 1
surface_name = "GaAs_001_4x4"

element = "As"
ads = catkit.gratoms.Gratoms(element)

lammps_calc = LAMMPSlib(
    lmpcmds=cmds, atom_types={"Ga": 1, "As": 2}, log_file="test.log"
)
history, energy_hist, frac_accept_hist, adsorption_count_hist = mcmc_run(
    num_runs=num_runs,
    temp=1,
    pot=0,
    alpha=0.99,
    slab=slab,
    calc=lammps_calc,
    surface_name=surface_name,
    element=element,
)
