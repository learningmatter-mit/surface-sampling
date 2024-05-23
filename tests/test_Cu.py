import os

import numpy as np
from catkit import Gratoms
from catkit.gen.utils.connectivity import get_cutoff_neighbors
from catkit.gen.utils.graph import connectivity_to_edges
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.ase import AseAtomsAdaptor

from mcmc import MCMC
from mcmc.calculators import LAMMPSRunSurfCalc
from mcmc.slab import initialize_slab
from mcmc.system import SurfaceSystem

current_dir = os.path.dirname(__file__)

# some regression testing first
def test_Cu_energy():
    required_energy = -25.2893

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
        "num_sweeps": 10,
        "sweep_size": 2,
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
    pristine_slab = slab.copy()
    pristine_pmg_slab = AseAtomsAdaptor.get_structure(pristine_slab)
    site_finder = AdsorbateSiteFinder(pristine_pmg_slab)
    ads_positions = site_finder.find_adsorption_sites(
        put_inside=True,
        symm_reduce=True,
        near_reduce=system_settings["near_reduce"],
        distance=system_settings["planar_distance"],
        no_obtuse_hollow=system_settings["no_obtuse_hollow"],
    )["all"]
    print(f"adsorption coordinates are: {ads_positions[:5]}...")

    # ads_positions = get_adsorption_coords(pristine_slab, elem, connectivity, debug=True)

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
