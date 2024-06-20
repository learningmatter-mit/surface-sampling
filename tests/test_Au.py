import os
import pickle as pkl

import numpy as np
from ase.io import read

from mcmc import MCMC
from mcmc.calculators import LAMMPSRunSurfCalc
from mcmc.system import SurfaceSystem

current_dir = os.path.dirname(__file__)


# some regression testing first
def test_Au_energy():
    required_energy = -79.03490823689619

    # create slab and get proper ads sites
    slab_pkl = open(
        os.path.join(current_dir, "data/Au_110_2x2_pristine_slab.pkl"), "rb"
    )

    slab = pkl.load(slab_pkl)

    proper_adsorbed = read(
        os.path.join(current_dir, "data/Au_110_2x2_proper_adsorbed_slab.cif")
    )
    ads_positions = proper_adsorbed.get_positions()[len(slab) :]

    element = "Au"

    num_ads_atoms = 4 + 2  # for canonical runs

    system_settings = {
        "surface_name": "Au(110)",
        "cutoff": 5.0,
        "num_ads_atoms": num_ads_atoms,
        "near_reduce": 0.01,
        "planar_distance": 1.5,
        "no_obtuse_hollow": True,
    }

    sampling_settings = {
        "alpha": 0.99,  # slowly anneal
        "temperature": 1.0,  # in terms of kbT
        "num_sweeps": 50,
        "sweep_size": len(ads_positions),
    }

    calc_settings = {
        "calc_name": "eam",
        "optimizer": "FIRE",
        "chem_pots": {"Au": 0.0},
        "relax_atoms": False,
        "relax_steps": 100,
        "run_dir": os.path.join(current_dir, "data/Au_110"),
    }

    # use LAMMPS
    parameters = {"pair_style": "eam", "pair_coeff": ["* * Au_u3.eam"]}

    # set up the LAMMPS calculator
    potential_file = os.path.join(os.environ["LAMMPS_POTENTIALS"], "Au_u3.eam")
    lammps_surf_calc = LAMMPSRunSurfCalc(
        files=[potential_file],
        keep_tmp_files=False,
        keep_alive=False,
        tmp_dir=os.path.join(os.path.expanduser("~"), "tmp_files"),
    )
    lammps_surf_calc.set(**parameters)

    # initialize SurfaceSystem
    surface = SurfaceSystem(
        slab,
        ads_coords=ads_positions,
        calc=lammps_surf_calc,
        system_settings=system_settings,
    )

    mcmc = MCMC(
        system_settings["surface_name"],
        calc=lammps_surf_calc,
        canonical=True,
        testing=False,
        element=element,
        adsorbates=list(calc_settings["chem_pots"].keys()),
        relax=calc_settings["relax_atoms"],
        optimizer=calc_settings["optimizer"],
        num_ads_atoms=system_settings["num_ads_atoms"],
    )  # no relaxation

    mcmc.mcmc_run(
        total_sweeps=sampling_settings["num_sweeps"],
        sweep_size=sampling_settings["sweep_size"],
        start_temp=sampling_settings["temperature"],
        pot=list(calc_settings["chem_pots"].values()),
        alpha=sampling_settings["alpha"],
        surface=surface,
    )

    assert np.allclose(np.min(mcmc.energy_hist), required_energy)
