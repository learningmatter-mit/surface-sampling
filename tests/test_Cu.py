import os

import numpy as np
from ase.calculators.lammpsrun import LAMMPS

from mcmc import MCMC
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
    sweep_size = 16

    # use LAMMPS
    parameters = {"pair_style": "eam", "pair_coeff": ["* * Cu_u3.eam"]}

    potential_file = os.path.join(os.environ["LAMMPS_POTENTIALS"], "Cu_u3.eam")
    lammps_calc = LAMMPS(
        files=[potential_file],
        keep_tmp_files=False,
        keep_alive=False,
    )
    lammps_calc.set(**parameters)

    # initialize object
    Cu_mcmc = MCMC(
        calc=lammps_calc,
        element=element,
    )
    Cu_mcmc.mcmc_run(
        total_sweeps=num_sweeps,
        sweep_size=sweep_size,
        start_temp=temp,
        pot=chem_pot,
        alpha=alpha,
        slab=slab,
    )

    assert np.allclose(np.min(Cu_mcmc.energy_hist), required_energy)
