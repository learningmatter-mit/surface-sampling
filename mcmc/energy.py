import copy
import json
import logging
import os
from collections import Counter

import ase
import numpy as np
from ase.optimize import BFGS, FIRE
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.optimize.sciopt import SciPyFminCG
from lammps import (
    LMP_STYLE_ATOM,
    LMP_STYLE_GLOBAL,
    LMP_TYPE_SCALAR,
    LMP_TYPE_VECTOR,
    lammps,
)
from nff.io.ase import AtomsBatch
from nff.utils.constants import EV_TO_KCAL_MOL, HARTREE_TO_KCAL_MOL

from .dynamics import TrajectoryObserver
from .system import SurfaceSystem
from .utils.misc import get_atoms_batch

HARTREE_TO_EV = HARTREE_TO_KCAL_MOL / EV_TO_KCAL_MOL
# threshold for unrelaxed energy
ENERGY_THRESHOLD = 1000  # eV
MAX_FORCE_THRESHOLD = 1000  # eV/Angstrom

logger = logging.getLogger(__name__)


def optimize_slab(slab, optimizer="FIRE", save_traj=True, **kwargs):
    """Run relaxation for slab

    Parameters
    ----------
    slab : ase.Atoms
        Surface slab
    optimizer : str, optional
        Optimizer to use, by default "FIRE"

    Returns
    -------
    ase.Atoms
        Relaxed slab
    """
    if "LAMMPS" in optimizer:
        calc_slab, energy, _ = slab.calc.run_lammps_opt(slab, run_dir=slab.calc.run_dir)

        if isinstance(slab, AtomsBatch):
            calc_slab = get_atoms_batch(
                calc_slab,
                nff_cutoff=slab.cutoff,
                device=slab.device,
            )

        traj = None

    else:
        energy = None
        if "BFGSLineSearch" in optimizer:
            Optimizer = BFGSLineSearch
        elif "BFGS" in optimizer:
            Optimizer = BFGS
        elif "CG" in optimizer:
            Optimizer = SciPyFminCG
        else:
            Optimizer = FIRE
        if type(slab) is AtomsBatch:
            slab.update_nbr_list(update_atoms=True)
            calc_slab = copy.deepcopy(slab)
        else:
            calc_slab = slab.copy()
        calc_slab.calc = slab.calc
        if (
            kwargs.get("folder_name", None)
            and kwargs.get("iter", None)
            and kwargs.get("save", False)
        ):
            # TODO: remove
            # save every 10 steps
            iter = int(kwargs.get("iter"))
            # if iter % 10 == 0:
            # save only when told to
            # use BFGSLineSearch to ensure energy and forces go down
            dyn = Optimizer(
                calc_slab,
                trajectory=os.path.join(
                    kwargs["folder_name"],
                    f"final_slab_traj_{iter:04}.traj",
                ),
            )
        else:
            dyn = Optimizer(calc_slab)

        relax_steps = kwargs.get("relax_steps", 20)

        if save_traj:
            # add in hook to save the trajectory
            obs = TrajectoryObserver(calc_slab)
            # record_interval = int(relax_steps / 4)
            record_interval = kwargs.get("record_interval", 5)
            dyn.attach(obs, interval=record_interval)
            # self.relaxed_atoms = relax_atoms(self.relaxed_atoms, **kwargs)

        # default steps is 20 and max forces are 0.01
        # TODO set up a config file to change this
        dyn.run(steps=relax_steps, fmax=0.01)

        energy = float(calc_slab.get_potential_energy())

        if save_traj:
            traj = {
                "atoms": obs.atoms_history,
                "energies": obs.energies,
                "forces": obs.forces,
            }

        else:
            traj = None

    try:
        max_force = float(np.abs(calc_slab.calc.results["forces"]).max())
    except KeyError:
        max_force = 0.0

    if np.abs(energy) > ENERGY_THRESHOLD or max_force > MAX_FORCE_THRESHOLD:
        logger.info("encountered energy or force out of bounds")
        logger.info("energy %.3f", energy)
        logger.info("max force %.3f}", max_force)

        # set a high energy for mc acceptance criteria to reject
        energy = ENERGY_THRESHOLD
        energy_oob = True
    else:
        energy_oob = False

    return calc_slab, traj, energy, energy_oob
