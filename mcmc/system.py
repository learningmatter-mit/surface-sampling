import copy
import functools
import logging
import os
from typing import Dict, Iterable, List

import ase
import numpy as np
from ase.calculators.calculator import Calculator, PropertyNotImplementedError
from ase.constraints import FixAtoms
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.ase import AseAtomsAdaptor

logger = logging.getLogger(__name__)
file_dir = os.path.dirname(__file__)

DEFAULT_SETTINGS = {
    "planar_distance": 1.5,
    "relax_atoms": False,
    "optimizer": "FIRE",
    "cutoff": 5.0,
    "calc_name": "kim",
    "temperature": 1.0,  # in units of eV
    "chem_pots": {"Ga": 0.0, "As": 0.0},
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
        ads_coords: Iterable = (),
        ads_surface: bool = False,  # TODO
        calc: Calculator = None,
        occ: List = None,
        system_settings: Dict = None,
        calc_settings: Dict = None,
    ):
        """Initialize the SurfaceSystem object that encompasses a material surface and adsorption sites.

        Parameters
        ----------
        atoms : ase.Atoms
            The atoms object representing the surface.
        ads_coords : List
            The coordinates of the virtual adsorption sites.
        ads_surface : bool, optional
            Whether surface atoms should be included in adsorption sites, by default False
        calc : Calculator, optional
            ASE Calculator, by default None
        occ : List, optional
            The index of the adsorbed atom at each adsorption site, by default None
        system_settings : Dict, optional
            Settings for surface system, by default None
        calc_settings : Dict, optional
            Settings for calculator, by default None
        """
        # TODO the procedure is to go from all_atoms to real_atoms and relaxed_atoms
        # but for now, we only have the real_atoms and relaxed_atoms to maintain compatibility
        # with ASE
        self.all_atoms = None
        # self.real_atoms = self.all_atoms.copy()
        self.system_settings = system_settings or DEFAULT_SETTINGS
        self.calc_settings = calc_settings or (calc.parameters.copy() if calc else {})

        self.real_atoms = None
        self.num_pristine_atoms = 0
        self.calc = calc
        self.relax_traj = []
        self.relaxed_atoms = None
        self.relax_atoms = self.calc_settings.get(
            "relax_atoms", False
        )  # whether to relax surface
        self.results = {}
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
        """Save the state of the SurfaceSystem object.

        Parameters
        ----------
        key : str
            The key to save the state as.
        """
        self.real_atoms.calc = None
        if self.relax_atoms:
            self.relaxed_atoms.calc = None
        # TODO perhaps fix the trajectory saving
        self._states[key] = {
            "real_atoms": copy.deepcopy(self.real_atoms),
            "relaxed_atoms": copy.deepcopy(self.relaxed_atoms),
            "occupation": copy.deepcopy(self.occ),
            "results": copy.deepcopy(self.results),
        }
        self.real_atoms.calc = self.calc
        if self.relax_atoms:
            self.relaxed_atoms.calc = self.calc

    def restore_state(self, key: str):
        """Restore the state of the SurfaceSystem object.

        Parameters
        ----------
        key : str
            The key to restore the state from.

        Raises
        ------
        ValueError
            If no state is available for the given key.
        """
        state = self._states.get(key, None)
        if state is None:
            raise ValueError(f"Cannot restore: No state available for key '{key}'.")

        self.real_atoms = state["real_atoms"]
        self.real_atoms.calc = self.calc
        if self.relax_atoms:
            self.relaxed_atoms = state["relaxed_atoms"]
            self.relaxed_atoms.calc = self.calc
        self.occ = state["occupation"]
        self.results = state["results"]

    def initialize(
        self,
        atoms: ase.Atoms,
        ads_coords: List,
        calc: Calculator = None,
        occ: List = None,
    ):
        """Initialize the SurfaceSystem object.

        Parameters
        ----------
        atoms : ase.Atoms
            The atoms object representing the surface.
        ads_coords : List
            The coordinates of the virtual adsorption sites.
        calc : Calculator, optional
            The calculator object to use, by default None.
        occ : List, optional
            The index of the adsorbed atom at each adsorption site, by default None

        Returns
        -------
        None
        """
        self.real_atoms = copy.deepcopy(atoms)
        if not (
            (isinstance(ads_coords, list) and (len(ads_coords) > 0))
            or isinstance(ads_coords, np.ndarray)
        ):
            self.ads_coords = self.initialize_adsorption_sites()
        else:
            self.ads_coords = ads_coords
        self.calc = calc
        self.real_atoms.calc = self.calc
        self.all_atoms = copy.deepcopy(self.real_atoms)
        self.initialize_virtual_atoms()

        if not (
            (isinstance(occ, list) and (len(occ) > 0)) or isinstance(occ, np.ndarray)
        ):
            self.occ = np.zeros(len(self.ads_coords), dtype=int)
        else:
            assert len(occ) == len(self.ads_coords)
            self.occ = np.array(occ)
        logger.info("initial state is %s", self.occ)

        if not self.real_atoms.constraints:
            # calculate from real_atoms and occ
            self.num_pristine_atoms = len(self.real_atoms) - np.count_nonzero(self.occ)
            logger.info("number of pristine atoms is %s", self.num_pristine_atoms)
            self.bulk_idx = np.where(self.real_atoms.get_tags() == BULK_TAG)[0]
            self.surface_idx = np.where(self.real_atoms.get_tags() == SURFACE_TAG)[0]
            logger.info("bulk indices are %s", self.bulk_idx)
            logger.info("surface indices are %s", self.surface_idx)

            # set constraints
            constraints = FixAtoms(indices=self.bulk_idx)
            self.real_atoms.set_constraint(constraints)
        else:
            constraints = self.real_atoms.constraints
        logger.info("Real atoms have constraints %s", self.real_atoms.constraints)

        if self.relax_atoms:
            self.relaxed_atoms, _ = self.relax_structure()
            self.relaxed_atoms.set_constraint(constraints)
            logger.info(
                "Relaxed atoms have constraints %s", self.relaxed_atoms.constraints
            )

    def initialize_adsorption_sites(self):
        site_finder = AdsorbateSiteFinder(self.pymatgen_unrelaxed_structure)
        ads_positions = site_finder.find_adsorption_sites(
            put_inside=True,
            symm_reduce=False,
            near_reduce=self.system_settings["near_reduce"],
            distance=self.system_settings["planar_distance"],
            no_obtuse_hollow=self.system_settings["no_obtuse_hollow"],
        )["all"]
        logger.info("Generated adsorption coordinates are: %s...", ads_positions[:5])
        return ads_positions

    def initialize_virtual_atoms(self, virtual_atom_str: str = "X"):
        """Initialize virtual atoms on the surface.

        Parameters
        ----------
        virtual_atom_str : str
            The string representation of the virtual atom.

        Returns
        -------
        None
        """
        logger.info("initializing %s virtual atoms", len(self.ads_coords))
        self.all_atoms = copy.deepcopy(self.real_atoms)
        for site_idx in range(len(self.ads_coords)):
            virtual_adsorbate = ase.Atoms(
                virtual_atom_str, positions=[self.ads_coords[site_idx]]
            )
            self.all_atoms += virtual_adsorbate

    @property
    def adsorbate_idx(self):
        """Get the indices of the adsorbate atoms.

        Returns
        -------
        np.ndarray
            The indices of the adsorbate atoms.
        """
        self.ads_idx = self.occ[self.occ.nonzero()[0]]
        return self.ads_idx

    @property
    def pymatgen_unrelaxed_structure(self):
        return AseAtomsAdaptor.get_structure(self.real_atoms)

    def relax_structure(self, **kwargs):
        """Relax the surface structure.

        Returns
        -------
        Tuple[ase.Atoms, Union[float, List[float]]]
            The relaxed surface structure and the potential energy of the system.
        """
        # have to import here to avoid circular imports
        from .mcmc import optimize_slab

        relaxed_slab, energy, traj = optimize_slab(
            self.real_atoms, **self.calc_settings, **kwargs
        )
        self.relaxed_atoms = relaxed_slab
        self.relax_traj = traj
        return relaxed_slab, energy

    @staticmethod
    def update_results(_func=None, *, prop="surface_energy"):
        """Decorator to update the results dictionary with the property value."""

        def decorator_update_results(func):
            @functools.wraps(func)
            def wrapper_update_results(self, *args, **kwargs):
                val = func(self, *args, **kwargs)
                self.results[prop] = val
                return val

            return wrapper_update_results

        if _func is None:
            return decorator_update_results
        else:
            return decorator_update_results(_func)

    @update_results(prop="energy")
    def get_relaxed_energy(self, recalculate=False, **kwargs):
        """Get the relaxed potential energy of the system.

        Parameters
        ----------
        recalculate : bool, optional
            Re-relax surface, by default False

        Returns
        -------
        Union[float, List[float]
            The relaxed potential energy of the system.
        """
        # TODO check if already relaxed
        if self.relaxed_atoms is None or recalculate:
            _, energy = self.relax_structure(**kwargs)
        else:
            energy = self.relaxed_atoms.get_potential_energy()
        return energy

    @update_results(prop="energy")
    def get_unrelaxed_energy(self, **kwargs):
        """Get the unrelaxed potential energy of the system.

        Returns
        -------
        Union[float, List[float]]
            The unrelaxed potential energy of the system.
        """
        return self.real_atoms.get_potential_energy()

    @update_results(prop="energy")
    def get_potential_energy(self, **kwargs):
        """Get the potential energy of the system, relaxed or unrelaxed.

        Returns
        -------
        Union[float, List[float]]
            The relaxed or unrelaxed potential energy of the system.
        """
        if self.relax_atoms:
            return self.get_relaxed_energy(**kwargs)
        else:
            return self.get_unrelaxed_energy(**kwargs)

    @update_results(prop="surface_energy")
    def get_surface_energy(self, recalculate: bool = False, **kwargs):
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
        if "surface_energy" not in self.calc.implemented_properties:
            raise AttributeError("Calculator object has no get_surface_energy method.")

        if self.relax_atoms:
            if self.relaxed_atoms is None or recalculate:
                _, raw_energy = self.relax_structure(**kwargs)
            return self.calc.get_property(
                "surface_energy", atoms=self.relaxed_atoms, **kwargs
            )
        return self.calc.get_property("surface_energy", atoms=self.real_atoms, **kwargs)

    @update_results(prop="forces")
    def get_forces(self, **kwargs):
        """Get the forces acting on the atoms.

        Returns
        -------
        Union[np.ndarray, List]
            The forces acting on the atoms.
        """
        if self.relax_atoms:
            atoms = self.relaxed_atoms
        else:
            atoms = self.real_atoms
        try:
            return atoms.get_forces()
        except PropertyNotImplementedError:
            return np.zeros((len(atoms), 3))

    def __len__(self):
        return len(self.real_atoms)
