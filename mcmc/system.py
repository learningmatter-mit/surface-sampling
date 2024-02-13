import copy
import logging
import os
from typing import Dict, List

import ase
import numpy as np
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms

# refer to https://github.com/HojeChun/EpiKmc/blob/main/epikmc/system.py#L54

logger = logging.getLogger(__name__)
file_dir = os.path.dirname(__file__)

DEFAULT_SETTINGS = {
    "planar_distance": 1.5,
    "relax_atoms": False,
    "optimizer": "FIRE",
    "cutoff": 5.0,
    "calc_name": "kim",
    # "lattice_param": {"Ga": 5.653, "As": 5.653},
    "temperature": 1.0,  # in units of eV
    "chem_pots": {"Ga": 0.0, "As": 0.0},
    # "atomic_energy": {"Ga": -1.61455, "As": -1.98000},
    # "mass": {"Ga": 69.723, "As": 74.9216},
    "near_reduce": 0.01,
    "no_obtuse_hollow": True,
}

BULK_TAG = 0
SURFACE_TAG = 1
ADSORBATE_TAG = 2


class SurfaceSystem:
    # state here is changed to occ
    def __init__(
        self,
        atoms: ase.Atoms,
        ads_coords: List,
        calc: Calculator = None,
        occ: List = None,
        system_info: Dict = None,
    ):
        # TODO the procedure is to go from all_atoms to real_atoms and relaxed_atoms
        # but for now, we only have the real_atoms and relaxed_atoms to maintain compatibility
        # with ASE
        self.all_atoms = None
        # self.real_atoms = self.all_atoms.copy()
        self.system_info = system_info or DEFAULT_SETTINGS

        self.real_atoms = None
        self.num_pristine_atoms = 0
        self.calc = calc
        self.relax_traj = []
        self.relaxed_atoms = None
        self.relax_atoms = self.system_info.get(
            "relax_atoms", False
        )  # whether to relax surface
        self._states = {}
        self.constraints = []  # TODO
        self.surface_area = 0.0  # TODO

        self.surface_idx = []
        self.bulk_idx = []
        self.ads_idx = []
        self.ads_coords = []
        self.occ = []
        self.distance_matrix = []

        # TODO: give all virtual atoms 'X' identity, remove when exporting or calculating
        self.initialize(atoms, ads_coords, calc, occ)

    def save_state(self, key: str):
        self.real_atoms.calc = None
        if self.relax_atoms:
            self.relaxed_atoms.calc = None
        # TODO perhaps fix the trajectory saving
        self._states[key] = {
            "real_atoms": copy.deepcopy(self.real_atoms),
            "relaxed_atoms": copy.deepcopy(self.relaxed_atoms),
            "occupation": copy.deepcopy(self.occ),
        }
        self.real_atoms.calc = self.calc
        if self.relax_atoms:
            self.relaxed_atoms.calc = self.calc

    def restore_state(self, key: str):
        state = self._states.get(key, None)
        if state is None:
            raise ValueError(f"Cannot restore: No state available for key '{key}'.")

        self.real_atoms = state["real_atoms"]
        self.real_atoms.calc = self.calc
        if self.relax_atoms:
            self.relaxed_atoms = state["relaxed_atoms"]
        self.occ = state["occupation"]

    def initialize(
        self,
        atoms: ase.Atoms,
        ads_coords: List,
        calc: Calculator = None,
        occ: List = None,
    ):
        self.real_atoms = copy.deepcopy(atoms)
        self.ads_coords = ads_coords
        self.calc = calc
        self.real_atoms.calc = self.calc
        self.all_atoms = copy.deepcopy(self.real_atoms)
        self.initialize_virtual_atoms()
        if self.relax_atoms:
            self.relaxed_atoms, _ = self.relax_structure()

        if not (
            (isinstance(occ, list) and (len(self.occ) > 0))
            or isinstance(occ, np.ndarray)
        ):
            self.occ = np.zeros(len(self.ads_coords), dtype=int)
        else:
            assert len(occ) == len(self.ads_coords)
            self.occ = occ
        logger.info(f"initial state is {self.occ}")

        # calculate from real_atoms and occ
        self.num_pristine_atoms = len(self.real_atoms) - np.count_nonzero(self.occ)
        logger.info(f"number of pristine atoms is {self.num_pristine_atoms}")
        self.bulk_idx = np.where(self.real_atoms.get_tags() == BULK_TAG)[0]
        self.surface_idx = np.where(self.real_atoms.get_tags() == SURFACE_TAG)[0]
        logger.info(f"bulk indices are {self.bulk_idx}")
        logger.info(f"surface indices are {self.surface_idx}")

        # set constraints
        constraints = FixAtoms(indices=self.bulk_idx)
        self.real_atoms.set_constraint(constraints)
        if self.relax_atoms:
            self.relaxed_atoms.set_constraint(constraints)

    def initialize_virtual_atoms(self, virtual_atom_str: str = "X"):
        self.all_atoms = copy.deepcopy(self.real_atoms)
        for site_idx in range(len(self.ads_coords)):
            print(f"ads coords is {self.ads_coords[site_idx]}")
            virtual_adsorbate = ase.Atoms(
                virtual_atom_str, positions=[self.ads_coords[site_idx]]
            )
            self.all_atoms += virtual_adsorbate

    @property
    def adsorbate_idx(self):
        self.ads_idx = self.occ[self.occ.nonzero()[0]]
        return self.ads_idx

    def relax_structure(self, **kwargs):
        from .energy import optimize_slab

        relaxed_slab, energy, traj = optimize_slab(
            self.real_atoms, **self.system_info, **kwargs
        )
        self.relaxed_atoms = relaxed_slab
        # self.relax_traj.append(relaxed_structure) TODO
        return relaxed_slab, energy

    def get_relaxed_energy(self, recalculate=False, **kwargs):
        # TODO check if already relaxed
        if self.relaxed_atoms is None or recalculate:
            _, energy = self.relax_structure(**kwargs)
        else:
            energy = self.relaxed_atoms.get_potential_energy()
        return energy

    def get_unrelaxed_energy(self, **kwargs):
        # TODO the energy here should be equal to classical potential or DFT energy
        # write a helper function or method to return the correct energies
        return self.real_atoms.get_potential_energy()

    def get_potential_energy(self, **kwargs):
        if self.relax_atoms:
            return self.get_relaxed_energy(**kwargs)
        else:
            return self.get_unrelaxed_energy(**kwargs)

    def get_surface_energy(self, recalculate=False, **kwargs):
        """Calculate the surface energy of the system.

        Parameters
        ----------
        recalculate : bool
            If True, do relaxation again.
        kwargs : dict
            Additional keyword arguments to pass to the calculator.

        Returns
        -------
        float
            The surface energy of the system.
        """

        if self.calc is None:
            raise RuntimeError("SurfaceSystem object has no calculator.")

        if not hasattr(self.calc, "get_surface_energy"):
            raise AttributeError("Calculator object has no get_surface_energy method.")

        if self.relax_atoms:
            if self.relaxed_atoms is None or recalculate:
                _, raw_energy = self.relax_structure(**kwargs)
            return self.calc.get_surface_energy(atoms=self.relaxed_atoms, **kwargs)

        return self.calc.get_surface_energy(atoms=self.real_atoms, **kwargs)

    def get_forces(self, **kwargs):
        if self.relax_atoms:
            return self.relaxed_atoms.get_forces()
        else:
            return self.real_atoms.get_forces()

    def __len__(self):
        return len(self.real_atoms)
