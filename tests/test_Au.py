"""Regression test for the energy of the Au(110) surface."""

import logging
import os
import pickle as pkl
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from mcmc import MCMC
from mcmc.calculators import LAMMPSRunSurfCalc
from mcmc.system import SurfaceSystem
from mcmc.utils import setup_logger

current_dir = Path(__file__).parent
logger = setup_logger("mcmc", current_dir / "mc.log", logging.INFO)


@pytest.mark.parametrize("required_energy", [-79.03490823689619])
def test_Au_energy(required_energy):
    """Test the energy of the Au(110) surface. Regression test."""
    surface_name = "Au(110)"
    run_folder = current_dir / surface_name
    run_folder.mkdir(parents=True, exist_ok=True)

    # create slab and get proper ads sites
    try:
        with open(current_dir / "data/Au_110_2x2_pristine_slab.pkl", "rb") as slab_pkl:
            slab = pkl.load(slab_pkl)
    except FileNotFoundError as e:
        print("Could not find the Au(110) slab file")
        raise e

    proper_adsorbed = read(current_dir / "data/Au_110_2x2_proper_adsorbed_slab.cif")
    ads_positions = proper_adsorbed.get_positions()[len(slab) :]

    num_ads_atoms = 4 + 2  # for canonical runs

    # define settings
    calc_settings = {"pair_style": "eam", "pair_coeff": ["* * Au_u3.eam"]}

    system_settings = {
        "surface_name": surface_name,
        "cutoff": 5.0,
    }

    sampling_settings = {
        "total_sweeps": 50,
        "sweep_size": len(ads_positions),
        "start_temp": 1.0,  # in terms of kbT
        "perform_annealing": True,
        "alpha": 0.99,  # slowly anneal
        "canonical": True,
        "num_ads_atoms": num_ads_atoms,
        "adsorbates": ["Au"],
        "run_folder": run_folder,
    }

    # set up the LAMMPS calculator
    potential_file = Path(os.environ["LAMMPS_POTENTIALS"]) / "Au_u3.eam"
    lammps_surf_calc = LAMMPSRunSurfCalc(
        files=[potential_file],
        keep_tmp_files=False,
        keep_alive=False,
        tmp_dir=Path.home() / "vssr_tmp_files",
    )
    lammps_surf_calc.set(**calc_settings)

    # initialize SurfaceSystem
    surface = SurfaceSystem(
        slab,
        ads_coords=ads_positions,
        calc=lammps_surf_calc,
        system_settings=system_settings,
        save_folder=run_folder,
    )

    # start MCMC
    mcmc = MCMC(**sampling_settings)
    results = mcmc.run(
        surface=surface,
        **sampling_settings,
    )

    assert np.allclose(
        np.min(results["energy_hist"]),
        required_energy,
    )
