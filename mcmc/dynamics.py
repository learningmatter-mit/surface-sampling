import copy
import pickle

import numpy as np
from ase import Atoms


class TrajectoryObserver:
    # adapted from CHGNet
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
        self.atoms_history.append(
            self.atoms.copy()
        )  # don't want to save the calculator
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
