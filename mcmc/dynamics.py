"""Methods to optimize a slab using ASE optimizers."""

import logging
import pickle

import numpy as np
from ase import Atoms
from ase.optimize import BFGS, FIRE
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.optimize.sciopt import SciPyFminCG
from nff.io.ase import AtomsBatch

from mcmc.utils.misc import get_atoms_batch

# Threshold for unrelaxed energy
ENERGY_THRESHOLD = 1000  # eV
MAX_FORCE_THRESHOLD = 1000  # eV/Angstrom


class TrajectoryObserver:
    # Adapted from CHGNet (https://github.com/CederGroupHub/chgnet)
    """Trajectory observer is a hook in the relaxation process that saves the
    intermediate structures.
    """

    def __init__(self, atoms: Atoms) -> None:
        """Create a TrajectoryObserver from an Atoms object.

        Args:
            atoms (Atoms): the structure to observe.
        """
        self.atoms = atoms
        self.calc = atoms.calc
        self.energies: list[float] = []
        self.forces: list[np.ndarray] = []
        # self.stresses: list[np.ndarray] = []
        # self.magmoms: list[np.ndarray] = []
        self.atoms_history: list[Atoms] = []
        # self.cells: list[np.ndarray] = []

    def __call__(self):
        """The logic for saving the properties of an Atoms during the relaxation."""
        self.energies.append(self.compute_energy())
        self.forces.append(self.atoms.get_forces())
        # self.stresses.append(self.atoms.get_stress())
        # self.magmoms.append(self.atoms.get_magnetic_moments())

        self.atoms.calc = None
        self.atoms_history.append(self.atoms.copy())  # don't want to save the calculator
        self.atoms.calc = self.calc
        # self.cells.append(self.atoms.get_cell()[:])

    def __len__(self) -> int:
        """The number of steps in the trajectory."""
        return len(self.energies)

    def compute_energy(self) -> float:
        """Calculate the potential energy.

        Returns:
            energy (float): the potential energy.
        """
        return self.atoms.get_potential_energy()

    def save(self, filename: str) -> None:
        """Save the trajectory to file.

        Args:
            filename (str): filename to save the trajectory
        """
        out_pkl = {
            "energy": self.energies,
            "forces": self.forces,
            "atom_positions": self.atoms_history,
            "formula": self.atoms.get_chemical_formula(),
        }

        with open(filename, "wb") as file:
            pickle.dump(out_pkl, file)


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

        # Default steps is 20 and max forces are 0.01
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

        # Set a high energy for mc acceptance criteria to reject
        energy = ENERGY_THRESHOLD
        energy_oob = True
    else:
        energy_oob = False

    return calc_slab, traj, energy, energy_oob
