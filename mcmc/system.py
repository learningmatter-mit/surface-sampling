import copy
import functools
import logging
import os
import pickle as pkl
from typing import Dict, List, Union

import ase
import numpy as np
from ase import io
from ase.calculators.calculator import Calculator, PropertyNotImplementedError
from ase.constraints import FixAtoms
from ase.io.trajectory import TrajectoryWriter
from catkit.gen.utils import get_unique_coordinates
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core import Structure
from typing_extensions import Self

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


class SurfaceSystem:
    def __init__(
        self,
        atoms: ase.Atoms,
        ads_coords: List,
        calc: Calculator = None,
        occ: List = None,
        surface_depth: int = None,
        system_settings: Dict = None,
        calc_settings: Dict = None,
        distance_weight_matrix: np.ndarray = None,
        default_io_path: str = ".",
    ) -> None:
        """Initialize the SurfaceSystem object that encompasses a material surface and adsorption sites.

        Args:
            atoms (ase.Atoms): The atoms object representing the surface.
            ads_coords (List): The coordinates of the virtual adsorption sites.
            calc (Calculator, optional): ASE-style Calculator. Defaults to None.
            occ (List, optional): The index of the adsorbed atom in the slab at each adsorption site. Defaults to None.
            surface_depth (int, optional): Number of slab layers to leave unconstrained, starting from highest z coord.
                A layer is defined as a unique z-coordinate, if left blank will retain constraints from input slab.
                Defaults to None.
            system_settings (Dict, optional): Settings for surface system. Defaults to None.
            calc_settings (Dict, optional): Settings for calculator. Defaults to None.
            distance_weight_matrix (np.ndarray, optional): The distance weight matrix with size (n, n) where n is the
                number of ads sites. Defaults to None.
            default_io_path (str, optional): The default path to save the structures. Defaults to ".".

        TODO: add Attributes once refactored
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
        # TODO: before relaxing atoms, save the current unrelaxed state
        # compare real_atoms with unrelaxed_atoms before deciding to relax
        self.results = {}
        self._states = {}
        self.constraints = []
        self.surface_area = 0.0  # TODO

        self.surface_idx = []
        self.bulk_idx = []
        self.ads_idx = []
        self.ads_coords = []
        self.occ = []
        self.distance_weight_matrix = distance_weight_matrix

        self.default_io_path = default_io_path

        # TODO: give all virtual atoms 'X' identity, remove when exporting or calculating
        self.initialize(atoms, ads_coords, calc, occ, surface_depth)

    def save_state(self, key: str) -> None:
        """Save the state of the SurfaceSystem object.

        Args:
            key (str): The key to save the state as.
        """
        self.real_atoms.calc = None
        if self.relax_atoms:
            self.relaxed_atoms.calc = None
        # TODO can add saving the trajectory
        self._states[key] = {
            "real_atoms": copy.deepcopy(self.real_atoms),
            "relaxed_atoms": copy.deepcopy(self.relaxed_atoms),
            "occupation": copy.deepcopy(self.occ),
            "results": copy.deepcopy(self.results),
        }
        self.real_atoms.calc = self.calc
        if self.relax_atoms:
            self.relaxed_atoms.calc = self.calc

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
        surface_depth: int = None,
    ) -> None:
        """Initialize the SurfaceSystem object.

        Args:
            atoms (ase.Atoms): The atoms object representing the surface.
            ads_coords (List): The coordinates of the virtual adsorption sites.
            calc (Calculator, optional): The calculator object to use. Defaults to None.
            occ (List, optional): The index of the adsorbed atom at each adsorption site. Defaults to None.
            surface_depth (int, optional): Number of slab layers to leave unconstrained, starting from highest z coord.
                A layer is defined as a unique z-coordinate, if left blank will retain constraints from input slab.
                Defaults to None.
        """
        self.real_atoms = copy.deepcopy(atoms)
        self.ads_coords = np.array(ads_coords)
        self.calc = calc
        self.surface_depth = surface_depth
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

        # calculate from real_atoms and occ
        self.num_pristine_atoms = len(self.real_atoms) - np.count_nonzero(self.occ)
        logger.info("number of pristine atoms is %s", self.num_pristine_atoms)

        # setting tags according to Z coordinate (surface will be tagged 1, with tag increasing layerwise downwards)
        # TODO can move to a helper function to set constraints
        get_unique_coordinates(self.real_atoms, tag=True)
        if surface_depth is not None:
            # clear existing constraints
            self.real_atoms.constraints = []
            # check valid surface_depth
            if surface_depth > max(self.real_atoms.get_tags()):
                logger.warning(
                    "Surface depth exceeds number of unique z-coordinates in system, all atoms will be unconstrained."
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
            self.bulk_idx = (
                [] if not constraints else constraints[0].todict()["kwargs"]["indices"]
            )
            self.surface_idx = [i for i in range(len(atoms)) if i not in self.bulk_idx]
        logger.info("bulk indices are %s", self.bulk_idx)
        logger.info("surface indices are %s", self.surface_idx)
        logger.info("constraints are %s", constraints)

        if self.relax_atoms:
            self.relaxed_atoms, _ = self.relax_structure()
            self.relaxed_atoms.set_constraint(constraints)

    def initialize_virtual_atoms(self, virtual_atom_str: str = "X") -> None:
        """Initialize virtual atoms on the surface.

        Args:
            virtual_atom_str (str, optional): The string representation of the virtual atom. Defaults to "X".
        """
        logger.info("initializing %s virtual atoms", len(self.ads_coords))
        self.all_atoms = copy.deepcopy(self.real_atoms)
        for site_idx in range(len(self.ads_coords)):
            virtual_adsorbate = ase.Atoms(
                virtual_atom_str, positions=[self.ads_coords[site_idx]]
            )
            self.all_atoms += virtual_adsorbate

    def initialize_ads_positions(self, ads_coords: List) -> None:
        # TODO: add ase positions here as an option if not too difficult
        # TODO: currently this method doesn't make sense
        """Initialize the adsorption sites.

        Args:
            ads_coords (List): The coordinates of the virtual adsorption sites.
        """
        site_finder = AdsorbateSiteFinder(self.pymatgen_struct)

        ads_positions = site_finder.find_adsorption_sites(
            put_inside=True,
            symm_reduce=False,
            near_reduce=self.system_settings["near_reduce"],
            distance=self.system_settings["planar_distance"],
            no_obtuse_hollow=self.system_settings["no_obtuse_hollow"],
        )[
            "all"
        ]  # TODO: make this better

    @property
    def adsorbate_idx(self) -> np.ndarray:
        """Get the indices of the adsorbate atoms in the slab.

        Returns:
            np.ndarray: The indices of the adsorbate atoms.
        """
        self.ads_idx = self.occ[self.occ.nonzero()[0]]
        return self.ads_idx

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
        return Structure.from_ase(self.real_atoms)

    def relax_structure(self, **kwargs) -> tuple[ase.Atoms, float]:
        """Relax the surface structure.

        Args:
            **kwargs: Additional keyword arguments to pass to the calculator.

        Returns:
            Tuple[ase.Atoms, Union[float, List[float]]]: The relaxed surface structure and the potential energy of the system.
        """
        # have to import here to avoid circular imports
        from .mcmc import optimize_slab

        relaxed_slab, traj, energy, energy_oob = optimize_slab(
            self.real_atoms, **self.calc_settings, **kwargs
        )
        self.relaxed_atoms = relaxed_slab
        self.relax_traj = traj
        if energy_oob:
            self.save_structures(
                energy_oob=True,
            )
            # TODO fix default_io_path
        return relaxed_slab, energy

    @staticmethod
    def update_results(_func=None, *, prop="surface_energy") -> callable:
        """Decorator to update the results dictionary with the property value.

        Args:
            _func: The function to decorate.
            prop (str): The property to update in the results dictionary.

        Returns:
            Union[Callable, Callable]: The decorated function or the decorator.
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
        else:
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
    def get_potential_energy(self, **kwargs) -> Union[float, List[float]]:
        """Get the potential energy of the system, relaxed or unrelaxed.

        Args:
            **kwargs: Additional keyword arguments to pass to the calculator.

        Returns:
            Union[float, List[float]]: The relaxed or unrelaxed potential energy of the system.
        """
        if self.relax_atoms:
            return self.get_relaxed_energy(**kwargs)
        else:
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
            return self.calc.get_property(
                "surface_energy", atoms=self.relaxed_atoms, **kwargs
            )
        return self.calc.get_property("surface_energy", atoms=self.real_atoms, **kwargs)

    @update_results(prop="forces")
    def get_forces(self, **kwargs) -> Union[np.ndarray, List]:
        """Get the forces acting on the atoms.

        Args:
            **kwargs: Additional keyword arguments to pass to the calculator.

        Returns:
            Union[np.ndarray, List]: The forces acting on the atoms.
        """
        if self.relax_atoms:
            atoms = self.relaxed_atoms
        else:
            atoms = self.real_atoms
        try:
            return atoms.get_forces()
        except PropertyNotImplementedError:
            return np.zeros((len(atoms), 3))

    def __len__(self) -> int:
        return len(self.real_atoms)

    def save_structures(
        self,
        sweep_num: int = 0,
        energy_oob: bool = False,
        save_folder: str = None,
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
        logger.info("optim structure has Energy = %.3f", energy)

        oob_str = "oob" if energy_oob else "inb"
        if not save_folder:
            save_folder = self.default_io_path

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

    def copy(self) -> Self:
        """Create a copy of the SurfaceSystem object.

        Returns:
            SurfaceSystem: The copied SurfaceSystem object.
        """
        copy_obj = self.copy_without_calc()
        copy_obj.set_calc(self.calc)

        return copy_obj

    def copy_without_calc(self) -> Self:
        """Create a copy of the SurfaceSystem object without the calculator.

        Returns:
            SurfaceSystem: The copied SurfaceSystem object without the calculator.
        """
        calc = self.unset_calc()
        copy_obj = copy.deepcopy(self)
        self.set_calc(calc)

        return copy_obj
