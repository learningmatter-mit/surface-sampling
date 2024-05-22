import os

import numpy as np
from catkit import Gratoms
from catkit.gen.utils.connectivity import get_cutoff_neighbors
from catkit.gen.utils.graph import connectivity_to_edges

from mcmc import MCMC
from mcmc.calculators import LAMMPSRunSurfCalc
from mcmc.slab import get_adsorption_coords, initialize_slab
from mcmc.system import SurfaceSystem

current_dir = os.path.dirname(__file__)

# some regression testing first
def test_Cu_energy():
    required_energy = -36.068

    # Cu alat from https://www.copper.org/resources/properties/atomic_properties.html
    element = "Cu"
    Cu_alat = 3.6147
    slab = initialize_slab(Cu_alat, size=(2, 2, 2))

    system_settings = {
        "surface_name": "Cu(100)",
        "cutoff": 5.0,
        "lattice_param": {"Cu": Cu_alat},
        "near_reduce": 0.01,
        "planar_distance": 1.5,
        "no_obtuse_hollow": True,
    }

    sampling_settings = {
        "alpha": 0.99,  # slowly anneal
        "temperature": 1.0,  # in terms of kbT
        "num_sweeps": 50,
        "sweep_size": 16,
    }

    calc_settings = {
        "calc_name": "eam",
        "optimizer": "FIRE",
        "chem_pots": {"Cu": 0.0},
        "relax_atoms": False,
        "relax_steps": 100,
        "run_dir": os.path.join(current_dir, "data/Cu_100"),
    }

    # get ads positions
    connectivity = get_cutoff_neighbors(slab, cutoff=system_settings["cutoff"])
    pristine_slab = slab.copy()

    elem = Gratoms(element)
    ads_positions = get_adsorption_coords(pristine_slab, elem, connectivity, debug=True)

    # use LAMMPS
    parameters = {"pair_style": "eam", "pair_coeff": ["* * Cu_u3.eam"]}

    # set up the LAMMPS calculator
    potential_file = os.path.join(os.environ["LAMMPS_POTENTIALS"], "Cu_u3.eam")
    lammps_surf_calc = LAMMPSRunSurfCalc(
        files=[potential_file],
        keep_tmp_files=False,
        keep_alive=False,
        tmp_dir=os.path.join(os.path.expanduser("~"), "tmp_files"),
    )
    lammps_surf_calc.set(**parameters)

    # initialize SurfaceSystem
    surface = SurfaceSystem(
        slab, ads_positions, lammps_surf_calc, system_settings=system_settings
    )

    mcmc = MCMC(
        system_settings["surface_name"],
        calc=lammps_surf_calc,
        canonical=False,
        testing=False,
        element=element,
        adsorbates=list(calc_settings["chem_pots"].keys()),
        relax=calc_settings["relax_atoms"],
        optimizer=calc_settings["optimizer"],
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
