import os
import pickle
import sys

import catkit
import numpy as np
import pytest
from ase.calculators.lammpsrun import LAMMPS
from ase.io import read
from catkit.gen.adsorption import get_adsorption_sites

from mcmc import get_adsorption_coords, mcmc_run

current_dir = os.path.dirname(__file__)

print(f"current folder is {current_dir}")

# some regression testing first
def test_Au_energy():
    required_energy = -79.03490823689619

    # create slab and get proper ads sites
    slab_pkl = open(
        os.path.join(current_dir, "resources/Au_110_2x2_pristine_slab.pkl"), "rb"
    )
    slab = pickle.load(slab_pkl)

    coords, connectivity, sym_idx = get_adsorption_sites(slab, symmetry_reduced=False)

    element = "Au"
    metal = catkit.gratoms.Gratoms(element)

    ads_coords = get_adsorption_coords(slab, metal, connectivity)

    proper_adsorbed = read(
        os.path.join(current_dir, "resources/Au_110_2x2_proper_adsorbed_slab.cif")
    )
    ads_positions = proper_adsorbed.get_positions()[len(slab) :]

    chem_pot = 0  # chem pot 0 to less complicate things
    num_ads_atoms = 4 + 2  # for canonical runs
    alpha = 0.9  # anneal
    temp = 1  # temp in terms of kbT
    num_sweeps = 50

    # use LAMMPS
    parameters = {"pair_style": "eam", "pair_coeff": ["* * Au_u3.eam"]}

    potential_file = os.path.join(os.environ["LAMMPS_POTENTIALS"], "Au_u3.eam")
    lammps_calc = LAMMPS(
        files=[potential_file],
        keep_tmp_files=False,
        keep_alive=False,
        tmp_dir="/home/dux/surface_sampling/tmp_files",
    )
    lammps_calc.set(**parameters)

    # call the main function
    history, energy_hist, frac_accept_hist, adsorption_count_hist = mcmc_run(
        num_sweeps=num_sweeps,
        temp=1,
        pot=0,
        alpha=alpha,
        slab=slab,
        calc=lammps_calc,
        element=element,
        canonical=True,
        num_ads_atoms=num_ads_atoms,
        ads_coords=ads_positions,
    )

    assert np.allclose(energy_hist[-1], required_energy)
