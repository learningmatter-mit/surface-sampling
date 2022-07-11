import os

import numpy as np
from ase.calculators.lammpsrun import LAMMPS

from mcmc import mcmc_run
from mcmc.slab import initialize_slab


# some regression testing first
def test_Cu_energy():
    required_energy = -47.30809647

    # initialize some parameters first
    # Cu alat from https://www.copper.org/resources/properties/atomic_properties.html
    Cu_alat = 3.6147
    slab = initialize_slab(Cu_alat, size=(2, 2, 2))

    element = "Cu"
    chem_pot = 0  # chem pot 0 to less complicate things
    alpha = 0.99  # slowly anneal
    temp = 1  # temp in terms of kbT
    num_sweeps = 100

    # use LAMMPS
    parameters = {"pair_style": "eam", "pair_coeff": ["* * Cu_u3.eam"]}

    potential_file = os.path.join(os.environ["LAMMPS_POTENTIALS"], "Cu_u3.eam")
    lammps_calc = LAMMPS(
        files=[potential_file],
        keep_tmp_files=False,
        keep_alive=False,
    )
    lammps_calc.set(**parameters)

    # call the main function
    history, energy_hist, frac_accept_hist, adsorption_count_hist = mcmc_run(
        num_sweeps=num_sweeps,
        temp=temp,
        pot=chem_pot,
        alpha=alpha,
        slab=slab,
        calc=lammps_calc,
        element=element,
    )

    assert np.allclose(energy_hist[-1], required_energy)
