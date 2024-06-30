"""Methods to optimize a slab using ASE optimizers."""

import logging

import numpy as np
from ase.optimize import BFGS, FIRE
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.optimize.sciopt import SciPyFminCG
from nff.io.ase import AtomsBatch

from .dynamics import TrajectoryObserver
from .utils.misc import get_atoms_batch

# threshold for unrelaxed energy
ENERGY_THRESHOLD = 1000  # eV
MAX_FORCE_THRESHOLD = 1000  # eV/Angstrom


def optimize_slab(
    slab,
    optimizer: str = "FIRE",
    save_traj: bool = True,
    logger: logging.Logger | None = None,
    **kwargs,
) -> tuple:
    """Run slab relaxation using ASE optimizers.

    Args:
        slab (ase.Atoms): Surface slab
        optimizer (str, optional): Optimizer to use, by default "FIRE"
        save_traj (bool, optional): Save trajectory, by default True
        logger (logging.Logger, optional): Logger object, by default None
        **kwargs: Additional keyword arguments

    Returns:
        tuple: slab, trajectory, energy, energy_oob
    """
    logger = logger or logging.getLogger(__name__)

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
        if isinstance(slab, AtomsBatch):
            slab.update_nbr_list(update_atoms=True)
        calc_slab = slab.copy()
        calc_slab.calc = slab.calc

        dyn = Optimizer(calc_slab)
        if save_traj:
            # hook to save the trajectory
            obs = TrajectoryObserver(calc_slab)
            record_interval = kwargs.get("record_interval", 5)
            dyn.attach(obs, interval=record_interval)

        # default steps is 20 and max forces are 0.01
        # TODO set up a config file to change this
        relax_steps = kwargs.get("relax_steps", 20)
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
