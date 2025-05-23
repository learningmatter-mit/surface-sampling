"""Regression test for the energy of the Cu(100) surface."""

import logging
from pathlib import Path

import catkit
import numpy as np
import pytest
from ase.build import bulk

from mcmc import MCMC
from mcmc.calculators import LAMMPSRunSurfCalc
from mcmc.system import SurfaceSystem
from mcmc.utils import setup_logger

current_dir = Path(__file__).parent


@pytest.mark.parametrize("required_energy", [-25.2893])
def test_Cu_energy(required_energy):
    """Test the energy of the Cu(100) surface. Regression test."""
    surface_name = "Cu_100"
    run_folder = current_dir / surface_name
    run_folder.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("mcmc", run_folder / "mc.log", logging.INFO)  # noqa

    # Cu alat from https://www.copper.org/resources/properties/atomic_properties.html
    Cu_bulk = bulk("Cu", "fcc", a=3.6147)
    slab = catkit.build.surface(
        Cu_bulk,
        size=(2, 2, 2),
        miller=(1, 0, 0),
        termination=0,
        fixed=0,
        vacuum=15.0,
        orthogonal=False,
    )

    # define settings
    calc_settings = {"pair_style": "eam", "pair_coeff": ["* * Cu_u3.eam"]}

    system_settings = {
        "surface_name": surface_name,
        "cutoff": 5.0,
        "near_reduce": 0.01,
        "planar_distance": 1.5,
        "symm_reduce": True,
        "no_obtuse_hollow": True,
        "ads_site_type": "all",
    }

    sampling_settings = {
        "total_sweeps": 10,
        "sweep_size": 2,
        "start_temp": 1.0,  # in terms of kbT
        "perform_annealing": True,
        "alpha": 0.99,  # slowly anneal
        "adsorbates": ["Cu"],
        "run_folder": run_folder,
    }

    # set up the LAMMPS calculator
    potential_file = current_dir / "../mcmc/potentials" / "Cu_u3.eam"
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
