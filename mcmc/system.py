"""Module to represent a surface system with adsorption sites."""

import copy
import functools
import logging
from collections.abc import Iterable
from typing import Self

import ase
import numpy as np
from ase import io
from ase.calculators.calculator import Calculator, PropertyNotImplementedError
from ase.constraints import FixAtoms, FixConstraint
from ase.io.trajectory import TrajectoryWriter
from catkit.gen.utils import get_unique_coordinates
from nff.io.ase import AtomsBatch
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core import Structure

from mcmc.dynamics import optimize_slab
from mcmc.utils import SilenceLogger

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


class SurfaceSystem:
    """Class to represent surface atoms and adsorption sites."""

    def __init__(
        self,
        atoms: ase.Atoms,
        relaxed_atoms: ase.Atoms = None,
        calc: Calculator = None,
        ads_coords: list | None = None,
        occ: list | None = None,
        system_settings: dict | None = None,
        calc_settings: dict | None = None,
        distance_weight_matrix: np.ndarray = None,
        logger: logging.Logger | None = None,
        save_folder: str = ".",
    ) -> None:
        """Initialize the SurfaceSystem object that encompasses a material surface and adsorption
        sites.

        Args:
            atoms (ase.Atoms): The atoms object representing the surface.
            relaxed_atoms (ase.Atoms, optional): The relaxed atoms object. Defaults to None.
            calc (Calculator, optional): ASE-style Calculator. Defaults to None.
            ads_coords (list, optional): The coordinates of the virtual adsorption sites.
            occ (list, optional): Index of the adsorbed atom in the slab at each adsorption
                site. Empty sites are denoted by 0. Defaults to None.
            system_settings (Dict, optional): Settings for surface system. Defaults to None.
            calc_settings (Dict, optional): Settings for the calculator. Defaults to None.
            distance_weight_matrix (np.ndarray, optional): Distance weight matrix with size
                (n, n) where n is the number of ads sites. Defaults to None.
            logger (logging.Logger, optional): Logger object. Defaults to None.
            save_folder (str, optional): The default path to save the structures. Defaults to ".".

        Attributes:
            system_settings (dict): Settings for the surface system.
            calc_settings (dict): Settings for the calculator.
            surface_name (str): The name of the surface.
            surface_depth (int): Depth of the surface used to determine atoms that can relax.
            relax_atoms (bool): Whether to relax the surface.
            all_atoms (ase.Atoms): Atoms object representing the surface with virtual
                adsorption sites indicated by 'X'.
            real_atoms (ase.Atoms): Surface with physical atoms only.
            relaxed_atoms (ase.Atoms): Relaxed surface with physical atoms only.
            num_pristine_atoms (int): Number of pristine atoms in the surface.
            calc (Calculator): ASE-style Calculator.
            relax_traj (list): Trajectory of the relaxation.
            results (dict): Results of the calculations.
            _states (dict): Saved states of the SurfaceSystem object.
            constraints (list): Constraints on the surface.
            surface_area (float): Surface area in Angstrom^2. Unused.
            surface_idx (list): Indices of the surface atoms.
            bulk_idx (list): Indices of the bulk atoms.
            ads_idx (list): Indices of the adsorbate atoms.
            ads_coords (list): Coordinates of the virtual adsorption sites.
            occ (list): Index of the adsorbed atom in the slab at each adsorption site.
            distance_weight_matrix (np.ndarray): Distance weight matrix.
            save_folder (str): Default path to save the structures.

        Raises:
            ValueError: If no relaxed atoms are available
        """
        # TODO the procedure is to go from all_atoms (with X denoting an empty site) to real_atoms
        # and relaxed_atoms but for now, we only have the real_atoms and relaxed_atoms to maintain
        # compatibility with ASE
        self.system_settings = system_settings or DEFAULT_SETTINGS
        if calc_settings:
            self.calc_settings = calc_settings
        elif calc:
            self.calc_settings = calc.parameters.copy()
        else:
            self.calc_settings = {}
        self.surface_name = self.system_settings.get("surface_name", atoms.get_chemical_formula())
        self.surface_depth = self.system_settings.get("surface_depth", None)
        self.relax_atoms = self.calc_settings.get("relax_atoms", False)  # whether to relax surface

        # initialize attributes
        self.all_atoms = None
        self.real_atoms = None
        self.relaxed_atoms = None
        self.num_pristine_atoms = 0
        self.calc = None
        self.relax_traj = []
        self.results = {}
        self._states = {}
        self.constraints = []
        self.surface_area = 0.0  # unused
        self.surface_idx = []
        self.bulk_idx = []
        self.ads_idx = []
        self.ads_coords = []
        self.occ = []
        self.distance_weight_matrix = distance_weight_matrix
        self.logger = logger or logging.getLogger(__name__)
        self.save_folder = save_folder

        # TODO: give all virtual atoms 'X' identity, remove when exporting or calculating
        self.initialize(
            atoms, calc=calc, ads_coords=ads_coords, occ=occ, relaxed_atoms=relaxed_atoms
        )

    def save_state(self, key: str) -> None:
        """Save the state of the SurfaceSystem object.

        Args:
            key (str): The key to save the state as.
        """
        calc = self.unset_calc()
        self._states[key] = {
            "real_atoms": self.real_atoms.copy(),
            "relaxed_atoms": self.relaxed_atoms.copy() if self.relaxed_atoms else None,
            "occupation": self.occ.copy(),
            "results": self.results.copy(),
        }
        self.set_calc(calc)

    def restore_state(self, key: str) -> None:
        """Restore the state of the SurfaceSystem object.

        Args:
            key (str): The key to restore the state from.

        Raises:
            ValueError: If no state is available for the given key.
        """
        state = self._states.get(key, None)
        if state is None:
            raise ValueError(f"Cannot restore: No state available for key '{key}'.")

        self.real_atoms = state["real_atoms"]
        if self.relax_atoms:
            self.relaxed_atoms = state["relaxed_atoms"]
        self.set_calc(self.calc)
        self.occ = state["occupation"]
        self.results = state["results"]

    def initialize(
        self,
        atoms: ase.Atoms,
        calc: Calculator = None,
        ads_coords: list | None = None,
        occ: list | None = None,
        relaxed_atoms: ase.Atoms = None,
    ) -> None:
        """Initialize the SurfaceSystem object.

        Args:
            atoms (ase.Atoms): The atoms object representing the surface.
            ads_coords (list): The coordinates of the virtual adsorption sites.
            calc (Calculator, optional): The calculator object to use. Defaults to None.
            relaxed_atoms (ase.Atoms, optional): The relaxed atoms object. Defaults to None.
            occ (list, optional): The index of the adsorbed atom at each adsorption site.
                Defaults to None.
        """
        self.real_atoms = atoms.copy()
        if "ads_group" not in self.real_atoms.arrays:
            self.real_atoms.set_array("ads_group", np.zeros(len(self.real_atoms)), dtype=int)
        self.relaxed_atoms = relaxed_atoms
        self.set_calc(calc)
        self.ads_coords = (
            np.array(ads_coords)
            if isinstance(ads_coords, Iterable)
            else self.initialize_ads_positions()
        )
        self.all_atoms = self.real_atoms.copy()
        self.initialize_virtual_atoms()

        if not (isinstance(occ, Iterable) and (len(occ) > 0)):
            self.occ = np.zeros(len(self.ads_coords), dtype=int)
        else:
            assert len(occ) == len(self.ads_coords)
            self.occ = np.array(occ)
        self.logger.info("Initial state is %s", self.occ)

        self.num_pristine_atoms = len(self.real_atoms) - np.count_nonzero(self.occ)
        self.logger.info("Number of pristine atoms is %s", self.num_pristine_atoms)

        constraints = self.initialize_constraints()
        self.logger.info("Bulk indices are %s", self.bulk_idx)
        self.logger.info("Surface indices are %s", self.surface_idx)
        self.logger.info("Constraints are %s", constraints)

        if self.relax_atoms:
            if not isinstance(self.relaxed_atoms, ase.Atoms):
                self.logger.info("Relaxing initialized surface")
                self.relaxed_atoms, _ = self.relax_structure()
            self.relaxed_atoms.set_constraint(constraints)

    def initialize_ads_positions(self) -> np.ndarray:
        """Initialize the adsorption sites.

        Returns:
            np.ndarray: The coordinates of the virtual adsorption sites.
        """
        site_finder = AdsorbateSiteFinder(self.pymatgen_struct)
        self.logger.info("Initalizing adsorption sites with settings: %s", self.system_settings)
        ads_site_type = self.system_settings.get("ads_site_type", "all")
        ads_positions = site_finder.find_adsorption_sites(
            put_inside=True,
            symm_reduce=self.system_settings.get("symm_reduce", False),
            near_reduce=self.system_settings.get("near_reduce", 0.01),
            distance=self.system_settings.get("planar_distance", 2.0),
            no_obtuse_hollow=self.system_settings.get("no_obtuse_hollow", True),
        )[ads_site_type]
        self.logger.info("Generated adsorption coordinates are: %s...", ads_positions[:5])
        return ads_positions

    def initialize_virtual_atoms(self, virtual_atom_str: str = "X") -> None:
        """Initialize virtual atoms on the surface.

        Args:
            virtual_atom_str (str, optional): The string representation of the virtual atom.
                Defaults to "X".
        """
        self.logger.info("Initializing %s virtual atoms", len(self.ads_coords))
        self.all_atoms = self.real_atoms.copy()
        for _, ads_coord in enumerate(self.ads_coords):
            virtual_adsorbate = ase.Atoms(virtual_atom_str, positions=[ads_coord])
            self.all_atoms += virtual_adsorbate

    def initialize_constraints(self) -> FixConstraint:
        """Initialize constraints on the surface. If surface_depth is set, set tags according to
        Z coordinate. Surface will be tagged 1, with tag increasing layerwise downwards until the
        surface_depth is reached. All atoms with tags greater than surface_depth will be bulk and
        thus fixed.

        Returns:
            FixConstraint: The constraints on the surface.
        """
        get_unique_coordinates(self.real_atoms, tag=True)
        if self.surface_depth is not None:
            # clear existing constraints
            self.real_atoms.constraints = []
            # check valid surface_depth
            if self.surface_depth > max(self.real_atoms.get_tags()):
                self.logger.warning(
                    "Surface depth exceeds number of unique z-coordinates in system, all atoms will"
                    " be unconstrained."
                )
            # set constraints according to surface depth
            surface_mask = np.isin(
                self.real_atoms.get_tags(), list(range(1, self.surface_depth + 1))
            )
            self.surface_idx = np.where(surface_mask)[0]
            self.bulk_idx = np.where(~surface_mask)[0]
            constraints = FixAtoms(indices=self.bulk_idx)
            self.real_atoms.set_constraint(constraints)
        else:
            # extract constraints for application to relaxed slab
            constraints = self.real_atoms.constraints
            self.bulk_idx = [] if not constraints else constraints[0].todict()["kwargs"]["indices"]
            self.surface_idx = [i for i in range(len(self.real_atoms)) if i not in self.bulk_idx]
        return constraints

    @property
    def adsorbate_idx(self) -> np.ndarray:
        """Get the indices of the adsorbate atoms in the slab.

        Returns:
            np.ndarray: The indices of the adsorbate atoms.
        """
        self.ads_idx = self.occ[self.occ.nonzero()[0]]
        return self.ads_idx

    @property
    def num_adsorbates(self) -> int:
        """Get the number of adsorbate atoms in the slab.

        Returns:
            int: The number of adsorbate atoms.
        """
        return len(self.adsorbate_idx)

    @property
    def empty_occ_idx(self) -> list[int]:
        """Get the indices of the empty adsorption sites.

        Returns:
            list[int]: The indices of the empty adsorption sites.
        """
        return np.argwhere(self.occ == 0).flatten().tolist()

    @property
    def filled_occ_idx(self) -> list[int]:
        """Get the indices of the filled adsorption sites.

        Returns:
            list[int]: The indices of the filled adsorption sites.
        """
        return np.argwhere(self.occ != 0).flatten().tolist()

    @property
    def pymatgen_struct(self) -> Structure:
        """Get the pymatgen structure object.

        Returns:
            pymatgen.Structure: The pymatgen structure object.
        """
        return Structure.from_ase_atoms(self.real_atoms)

    def relax_structure(self, **kwargs) -> tuple[ase.Atoms, float | list[float]]:
        """Relax the surface structure.

        Args:
            **kwargs: Additional keyword arguments to pass to the calculator.

        Returns:
            Tuple[ase.Atoms, float | list[float]]: The relaxed surface structure and the potential
                energy of the system.
        """
        self.calc_settings.pop("logger", None)  # remove logger from calc_settings
        relaxed_slab, traj, energy, energy_oob = optimize_slab(
            self.real_atoms, **self.calc_settings, **kwargs
        )
        self.relaxed_atoms = relaxed_slab
        self.relax_traj = traj
        if energy_oob:
            self.save_structures(
                energy_oob=True,
            )
        return relaxed_slab, energy

    @staticmethod
    def update_results(_func=None, *, prop="surface_energy") -> callable:
        """Decorator to update the results dictionary with the property value.

        Args:
            _func: The function to decorate.
            prop (str): The property to update in the results dictionary.

        Returns:
            Callable: The decorated function or the decorator.
        """

        def decorator_update_results(func):
            @functools.wraps(func)
            def wrapper_update_results(self, *args, **kwargs):
                val = func(self, *args, **kwargs)
                self.results[prop] = val
                return val

            return wrapper_update_results

        if _func is None:
            return decorator_update_results
        return decorator_update_results(_func)

    @update_results(prop="energy")
    def get_relaxed_energy(self, recalculate=False, **kwargs) -> float:
        """Get the relaxed potential energy of the system.

        Args:
            recalculate (bool, optional): Re-relax surface. Defaults to False.
            **kwargs: Additional keyword arguments to pass to the calculator.

        Returns:
            float: The relaxed potential energy of the system.
        """
        # TODO check if already relaxed
        if self.relaxed_atoms is None or recalculate:
            _, energy = self.relax_structure(**kwargs)
        else:
            energy = self.relaxed_atoms.get_potential_energy()
        return energy

    @update_results(prop="energy")
    def get_unrelaxed_energy(self, **kwargs) -> float:
        """Get the unrelaxed potential energy of the system.

        Args:
            **kwargs: Additional keyword arguments to pass to the calculator.

        Returns:
            float: The unrelaxed potential energy of the system.
        """
        return self.real_atoms.get_potential_energy()

    @update_results(prop="energy")
    def get_potential_energy(self, **kwargs) -> float | list[float]:
        """Get the potential energy of the system, relaxed or unrelaxed.

        Args:
            **kwargs: Additional keyword arguments to pass to the calculator.

        Returns:
            float | list[float]: The relaxed or unrelaxed potential energy of the system.
        """
        if self.relax_atoms:
            return self.get_relaxed_energy(**kwargs)
        return self.get_unrelaxed_energy(**kwargs)

    @update_results(prop="surface_energy")
    def get_surface_energy(self, recalculate: bool = False, **kwargs) -> float:
        """Calculate the surface energy of the system.

        Args:
            recalculate (bool, optional): Re-relax surface. Defaults to False.
            **kwargs: Additional keyword arguments to pass to the calculator.

        Returns:
            float: The surface energy of the system.
        """
        if self.calc is None:
            raise RuntimeError("SurfaceSystem object has no calculator.")
        if "surface_energy" not in self.calc.implemented_properties:
            raise AttributeError("Calculator object has no get_surface_energy method.")

        if self.relax_atoms:
            if self.relaxed_atoms is None or recalculate:
                _, raw_energy = self.relax_structure(**kwargs)
            return self.calc.get_property("surface_energy", atoms=self.relaxed_atoms, **kwargs)
        return self.calc.get_property("surface_energy", atoms=self.real_atoms, **kwargs)

    @update_results(prop="forces")
    def get_forces(self, **kwargs) -> np.ndarray | list:
        """Get the forces acting on the atoms.

        Args:
            **kwargs: Additional keyword arguments to pass to the calculator.

        Returns:
            np.ndarray | list: The forces acting on the atoms.
        """
        atoms = self.relaxed_atoms if self.relaxed_atoms else self.real_atoms
        try:
            return atoms.get_forces()
        except PropertyNotImplementedError:
            return np.zeros((len(atoms), 3))

    def save_structures(
        self,
        sweep_num: int = 0,
        energy_oob: bool = False,
        save_folder: str | None = None,
    ) -> None:
        """Saves structures for easy viewing.

        Args:
            sweep_num (int, optional): The sweep number. Defaults to 0.
            energy_oob (bool, optional): Whether the energy is out of bounds. Defaults to False.
            save_folder (str, optional): The folder to save the structures in. Defaults to None.

        Raises:
            ValueError: If no relaxed atoms are available.
        """
        chemical_formula = self.real_atoms.get_chemical_formula()
        energy = float(
            self.get_surface_energy(recalculate=False)
        )  # correct structure would be restored
        self.logger.info("Optimized structure has surface energy = %.3f", energy)

        oob_str = "oob" if energy_oob else "inb"
        if not save_folder:
            save_folder = self.save_folder

        # save cifs for unrelaxed and relaxed slabs (if relaxed)
        io.write(
            f"{save_folder}/{oob_str}_unrelaxed_slab_sweep_{sweep_num:03}_energy_{energy:.3f}_{chemical_formula}.cif",
            self.real_atoms,
        )

        if self.relaxed_atoms:
            io.write(
                f"{save_folder}/{oob_str}_relaxed_slab_sweep_{sweep_num:03}_energy_{energy:.3f}_{chemical_formula}.cif",
                self.relaxed_atoms,
            )

        # save trajectories
        if self.relax_traj:
            atoms_list = self.relax_traj["atoms"]
            writer = TrajectoryWriter(
                f"{save_folder}/{oob_str}_slab_traj_{sweep_num:03}_energy_{energy:.3f}_{chemical_formula}.traj",
                mode="a",
            )
            for atoms in atoms_list:
                writer.write(atoms)

    def set_calc(self, calc: Calculator) -> None:
        """Set the calculator for the SurfaceSystem object.

        Args:
            calc (Calculator): The calculator object to set.
        """
        self.calc = calc
        self.real_atoms.calc = calc
        if self.relaxed_atoms:
            self.relaxed_atoms.calc = calc

    def unset_calc(self) -> Calculator:
        """Unset the calculator for the SurfaceSystem object and its atoms.

        Returns:
            Calculator: The calculator object that was unset.
        """
        calc = self.calc
        self.calc = None
        self.real_atoms.calc = None
        if self.relaxed_atoms:
            self.relaxed_atoms.calc = None

        return calc

    def copy(self, copy_calc=False) -> Self:
        """Create a copy of the SurfaceSystem object.

        Returns:
            SurfaceSystem: The copied SurfaceSystem object.
        """
        calc = self.unset_calc()

        with SilenceLogger():
            copy_obj = self.__class__(
                atoms=self.real_atoms,
                relaxed_atoms=self.relaxed_atoms,
                calc=None,
                ads_coords=self.ads_coords,
                occ=self.occ,
                system_settings=self.system_settings,
                calc_settings=self.calc_settings,
                distance_weight_matrix=self.distance_weight_matrix,
                save_folder=self.save_folder,
            )

        if copy_calc:
            copied_calc = copy.deepcopy(calc)
            copy_obj.set_calc(copied_calc)
        else:
            copy_obj.set_calc(calc)
        self.set_calc(calc)

        return copy_obj

    def todict(self) -> dict:
        """Return the SurfaceSystem object as a dictionary.

        Returns:
            dict: The SurfaceSystem object as a dictionary.
        """
        return {
            "real_atoms": self.real_atoms.todict(),
            "relaxed_atoms": self.relaxed_atoms.todict() if self.relaxed_atoms else None,
            "calc": None,  # not implemented
            "ads_coords": self.ads_coords,
            "occ": self.occ,
            "system_settings": self.system_settings,
            "calc_settings": self.calc_settings,
            "distance_weight_matrix": self.distance_weight_matrix,
            "save_folder": self.save_folder,
        }

    @classmethod
    def fromdict(cls, dct) -> Self:
        """Rebuild SurfaceSystem object from a dictionary representation (todict).

        Args:
            dct (dict): dictionary representation of the object.

        Returns:
            SurfaceSystem: The SurfaceSystem object created from dict.
        """
        real_atoms = AtomsBatch.fromdict(
            dct["real_atoms"]
        )  # what if ase.Atoms was the original object?
        relaxed_atoms = AtomsBatch.fromdict(dct["relaxed_atoms"]) if dct["relaxed_atoms"] else None
        calc = dct["calc"]
        ads_coords = dct["ads_coords"]
        occ = dct["occ"]
        system_settings = dct["system_settings"]
        calc_settings = dct["calc_settings"]
        distance_weight_matrix = dct["distance_weight_matrix"]
        save_folder = dct["save_folder"]

        return cls(
            real_atoms,
            relaxed_atoms=relaxed_atoms,
            calc=calc,
            ads_coords=ads_coords,
            occ=occ,
            system_settings=system_settings,
            calc_settings=calc_settings,
            distance_weight_matrix=distance_weight_matrix,
            save_folder=save_folder,
        )

    def __len__(self) -> int:
        """Return the number of atoms in the SurfaceSystem object.

        Returns:
            int: The number of atoms in the SurfaceSystem object.
        """
        return len(self.real_atoms)

    def __repr__(self) -> str:
        """Return the string representation of the SurfaceSystem object.

        Returns:
            str: The string representation of the SurfaceSystem object.
        """
        return (
            f"SurfaceSystem({self.real_atoms.get_chemical_formula()} with {len(self.ads_coords)}"
            " adsorption sites)"
        )
