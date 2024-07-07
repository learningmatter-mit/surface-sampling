"""Module for ASE-style Calculators for surface energy calculations."""

import json
import logging
import os
from collections import Counter

import ase
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.lammpsrun import LAMMPS as LAMMPSRun
from lammps import (
    LMP_STYLE_ATOM,
    LMP_STYLE_GLOBAL,
    LMP_TYPE_SCALAR,
    LMP_TYPE_VECTOR,
    lammps,
)
from nff.io.ase import AtomsBatch
from nff.io.ase_calcs import EnsembleNFF
from nff.utils.constants import HARTREE_TO_EV
from tqdm import tqdm

logger = logging.getLogger(__name__)

ENERGY_THRESHOLD = 1000  # eV
MAX_FORCE_THRESHOLD = 1000  # eV/Angstrom


def get_results_single(atoms_batch: AtomsBatch, calc: Calculator) -> dict:
    """Calculate the results for a single AtomsBatch object.

    Args:
        atoms_batch (AtomsBatch): The AtomsBatch object to calculate the results for.
        calc (Calculator): The calculator to use.

    Returns:
        dict: The results of the calculation
    """
    atoms_batch.calc = calc
    calc.calculate(atoms_batch)

    return calc.results


def get_embeddings(atoms_batches: list[AtomsBatch], calc: Calculator) -> np.ndarray:
    """Calculate the embeddings for a list of AtomsBatch objects

    Args:
        atoms_batches (list[AtomsBatch]): List of AtomsBatch objects.
        calc (Calculator): NFF Calculator.

    Returns:
        np.ndarray: Latent space embeddings with each row corresponding to a structure
    """
    print(f"Calculating embeddings for {len(atoms_batches)} structures")
    embeddings = []
    for atoms_batch in tqdm(atoms_batches):
        embedding = get_embeddings_single(atoms_batch, calc)
        embeddings.append(embedding)
    return np.stack(embeddings)


def get_embeddings_single(
    atoms_batch: AtomsBatch,
    calc: Calculator,
    results_cache: dict | None = None,
    flatten: bool = True,
    flatten_axis: int = 0,
) -> np.ndarray:
    """Calculate the embeddings for a single AtomsBatch object

    Args:
        atoms_batch (AtomsBatch): AtomsBatch object.
        calc (Calculator): NFF Calculator.
        results_cache (dict): Cache for results.
        flatten (bool): Whether to flatten the embeddings.
        flatten_axis (int): Axis to flatten the embeddings.

    Returns:
        np.ndarray: Latent space embeddings
    """
    if results_cache is not None and "embedding" in results_cache:
        results = results_cache
    else:
        results = get_results_single(atoms_batch, calc)
    return (
        results["embedding"].mean(axis=flatten_axis).squeeze()
        if flatten
        else results["embedding"].squeeze()
    )


def get_std_devs(atoms_batches: list[AtomsBatch], calc: Calculator) -> np.ndarray:
    """Calculate the force standard deviations for a list of AtomsBatch objects

    Args:
    atoms_batches (List[AtomsBatch]): List of AtomsBatch objects
    calc (Calculator): NFF Calculator

    Returns:
        np.ndarray: Force standard deviations with each element corresponding to a structure
    """
    print(f"Calculating force standard deviations for {len(atoms_batches)} structures")
    force_stds = []
    for atoms_batch in tqdm(atoms_batches):
        force_std = get_std_devs_single(atoms_batch, calc)
        force_stds.append(force_std)

    return np.stack(force_stds)


def get_std_devs_single(atoms_batch: AtomsBatch, calc: Calculator) -> np.ndarray:
    """Calculate the force standard deviation for a single AtomsBatch object

    Args:
        atoms_batch (AtomsBatch): AtomsBatch object
        calc (Calculator): NFF Calculator

    Returns:
        np.ndarray: Force standard deviation
    """
    if len(calc.models) > 1:
        atoms_batch.calc = calc
        calc.calculate(atoms_batch)
        force_std = calc.results.get("forces_std", np.array([0.0])).mean()
    else:
        force_std = 0.0

    return force_std


# TODO define abstract base class for surface energy calcs
# use EnsembleNFF, NeuralFF classes for NFF
class EnsembleNFFSurface(EnsembleNFF):
    """Based on Ensemble Neural Force Field clas to calculate surface energy"""

    implemented_properties = (*EnsembleNFF.implemented_properties, "surface_energy")

    def __init__(self, *args, **kwargs):
        """Initialize the EnsembleNFFSurface class."""
        super().__init__(*args, **kwargs)
        self.chem_pots = {}
        self.offset_data = {}
        self.offset_units = kwargs.get("offset_units", "atomic")
        self.logger = kwargs.get("logger", logging.getLogger(__name__))

    def get_surface_energy(
        self,
        atoms: ase.Atoms = None,
        chem_pots: dict | None = None,
        offset_data: dict | None = None,
    ) -> float:
        """Get the surface energy of the system by subtracting the bulk energy and the chemical
        potential deviation from the bulk formula.

        Args:
            atoms (ase.Atoms): The atoms object to calculate the surface energy for.
            chem_pots (dict): The chemical potentials of the atoms in the system.
            offset_data (dict): The offset data for the system.

        Returns:
            float: The surface energy of the system.
        """
        if atoms is None:
            atoms = self.atoms

        if chem_pots is None and self.chem_pots is not None:
            chem_pots = self.chem_pots
        else:
            raise ValueError("chemical potentials are not set")

        if offset_data is None and self.offset_data is not None:
            offset_data = self.offset_data
        else:
            raise ValueError("offset data is not set")

        surface_energy = self.get_potential_energy(atoms=atoms)

        ads_count = Counter(atoms.get_chemical_symbols())

        bulk_energies = offset_data["bulk_energies"]
        stoics = offset_data["stoics"]
        ref_formula = offset_data["ref_formula"]
        ref_element = offset_data["ref_element"]

        # subtract the bulk energies
        bulk_ref_en = ads_count[ref_element] * bulk_energies[ref_formula]
        for ele in ads_count:
            if ele != ref_element:
                bulk_ref_en += (
                    ads_count[ele] - stoics[ele] / stoics[ref_element] * ads_count[ref_element]
                ) * bulk_energies[ele]

        if self.offset_units == "atomic":
            surface_energy -= bulk_ref_en * HARTREE_TO_EV
        else:
            surface_energy -= bulk_ref_en

        # TODO make this a separate function
        # subtract chemical potential deviation from bulk formula
        stoics = self.offset_data["stoics"]
        ref_element = self.offset_data["ref_element"]

        pot = 0
        for ele in ads_count:
            if ele != ref_element:
                pot += (
                    ads_count[ele] - stoics[ele] / stoics[ref_element] * ads_count[ref_element]
                ) * self.chem_pots[ele]

        surface_energy -= pot
        return surface_energy

    def set(self, **kwargs) -> dict:
        """Set parameters in key-value pairs. A dictionary containing the parameters that have been
        changed is returned. The special keyword 'parameters' can be used to read parameters from a
        file.

        Args:
            **kwargs: The parameters to set.

        Returns:
            dict: A dictionary containing the parameters that have been changed.
        """
        changed_parameters = EnsembleNFF.set(self, **kwargs)
        if "chem_pots" in self.parameters:
            self.chem_pots = self.parameters["chem_pots"]
            self.logger.info("chemical potentials: %s are set from parameters", self.chem_pots)
        if "offset_data" in self.parameters:
            self.offset_data = self.parameters["offset_data"]
            self.logger.info("offset data: %s is set from parameters", self.offset_data)
        return changed_parameters

    def calculate(
        self,
        atoms: ase.Atoms = None,
        properties: tuple = implemented_properties,
        system_changes: list = all_changes,
    ):
        """Caculate based on EnsembleNFF before add in surface energy calcs to results

        Args:
            atoms (ase.Atoms): The atoms object to calculate the properties for.
            properties (tuple): The properties to calculate.
            system_changes (list): The system changes to calculate.
        """
        if atoms is None:
            atoms = self.atoms

        EnsembleNFF.calculate(self, atoms, properties, system_changes)

        if "surface_energy" in properties:
            self.results["surface_energy"] = self.get_surface_energy(atoms=atoms)

        atoms.results.update(self.results)


# class NeuralFFSurface(NeuralFF):
#     pass


# use OpenKIM calc

# use ASE calc


class LAMMMPSCalc(Calculator):
    """Custom LAMMPSCalc class to calculate energies and forces to inteface with ASE."""

    name = "lammpscalc"
    implemented_properties = ("energy", "relaxed_energy", "forces", "per_atom_energies")
    # NOTE "energy" is the unrelaxed energy

    def __init__(self, *args, **kwargs):
        """Initialize the LAMMMPSCalc class."""
        super().__init__(*args, **kwargs)
        self.run_dir = os.getcwd()
        self.relax_steps = 100
        self.kim_potential = False
        self.logger = kwargs.get("logger", logging.getLogger(__name__))

    def run_lammps_calc(
        self,
        slab,
        run_dir="./",
        template_path="./lammps_opt_template.txt",
        lammps_config="./lammps_config.json",
        **kwargs,
    ) -> tuple:
        """Main function to run LAMMPS calculation. Can be used for both relaxation and static
        energy calculations.

        Args:
            slab (ase.Atoms): The slab to calculate the energy for.
            run_dir (str): The directory to run LAMMPS in.
            template_path (str): The path to the LAMMPS input template file.
            lammps_config (str): The path to the LAMMPS config file.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: The slab with the new atomic positions, the energy, and the per atom energies.
        """
        with open(template_path, "r", encoding="utf-8") as f:
            lammps_template = f.read()

        # config file is assumed to be stored in the folder you run lammps
        if isinstance(lammps_config, str):
            with open(lammps_config, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = lammps_config

        potential_file = config["potential_file"]
        atoms = config["atoms"]
        bulk_index = config["bulk_index"]

        # define necessary file locations
        lammps_data_file = f"{run_dir}/lammps.data"
        lammps_in_file = f"{run_dir}/lammps.in"
        lammps_out_file = f"{run_dir}/lammps.out"

        # write current surface into lammps.data
        slab.write(lammps_data_file, format="lammps-data", units="real", atom_style="atomic")
        steps = kwargs.get("relax_steps", 100)

        # write lammps.in file
        with open(lammps_in_file, "w", encoding="utf-8") as f:
            # if using KIM potential
            if kwargs.get("kim_potential", False):
                f.writelines(
                    lammps_template.format(lammps_data_file, bulk_index, steps, lammps_out_file)
                )
            else:
                f.writelines(
                    lammps_template.format(
                        lammps_data_file,
                        bulk_index,
                        potential_file,
                        *atoms,
                        steps,
                        lammps_out_file,
                    )
                )

        # run LAMMPS without too much output
        lmp = lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
        # lmp = lammps()
        self.logger.info(lmp.file(lammps_in_file))

        energy = lmp.extract_compute("thermo_pe", LMP_STYLE_GLOBAL, LMP_TYPE_SCALAR)
        if "opt" in lammps_template:
            pe_per_atom = []
        else:
            pe_per_atom = lmp.extract_compute("pe_per_atom", LMP_STYLE_ATOM, LMP_TYPE_VECTOR)
            pe_per_atom = np.ctypeslib.as_array(
                pe_per_atom, shape=(len(slab),)
            )  # convert to numpy array
        lmp.close()

        # Read from LAMMPS out
        new_slab = ase.io.read(lammps_out_file, format="lammps-data", style="atomic")

        atomic_numbers_dict = config["atomic_numbers_dict"]
        actual_atomic_numbers = [atomic_numbers_dict[str(x)] for x in new_slab.get_atomic_numbers()]

        new_slab.set_atomic_numbers(actual_atomic_numbers)
        new_slab.calc = slab.calc

        return energy, pe_per_atom, new_slab

    def run_lammps_opt(self, slab, run_dir="./", **kwargs) -> tuple:
        """Run LAMMPS relaxation calculation.

        Args:
            slab (ase.Atoms): The slab to calculate the energy for.
            run_dir (str): The directory to run LAMMPS in.
            **kwargs (dict): Additional keyword arguments.

        Returns:
            tuple: The slab with the new atomic positions, the energy, and the per atom energies.
        """
        energy, pe_per_atom, opt_slab = self.run_lammps_calc(
            slab,
            run_dir=run_dir,
            template_path=f"{run_dir}/lammps_opt_template.txt",
            lammps_config=f"{run_dir}/lammps_config.json",
            **kwargs,
        )
        self.logger.debug("slab energy in relaxation: %.3f", energy)
        return opt_slab, energy, pe_per_atom

    def run_lammps_energy(self, slab, run_dir="./", **kwargs) -> tuple:
        """Run LAMMPS static energy calculation.

        Args:
            slab (ase.Atoms): The slab to calculate the energy for.
            run_dir (str): The directory to run LAMMPS in.
            **kwargs (dict): Additional keyword arguments.

        Returns:
            tuple: The slab with the new atomic positions, the energy, and the per atom energies.
        """
        energy, pe_per_atom, _ = self.run_lammps_calc(
            slab,
            run_dir=run_dir,
            template_path=f"{run_dir}/lammps_energy_template.txt",
            lammps_config=f"{run_dir}/lammps_config.json",
            **kwargs,
        )
        self.logger.debug("slab energy in engrad: %.3f", energy)
        return slab, energy, pe_per_atom

    def set(self, **kwargs) -> dict:
        """Set parameters in key-value pairs. A dictionary containing the parameters that have been
        changed is returned. The special keyword 'parameters' can be used to read parameters from a
        file.

        Args:
            **kwargs: The parameters to set.

        Returns:
            dict: A dictionary containing the parameters that have been changed.
        """
        changed_parameters = Calculator.set(self, **kwargs)

        if "run_dir" in self.parameters:
            self.run_dir = self.parameters["run_dir"]
            self.logger.info("run directory: %s is set from parameters", self.run_dir)
        if "relax_steps" in self.parameters:
            self.relax_steps = self.parameters["relax_steps"]
            self.logger.info("relaxation steps: %s is set from parameters", self.relax_steps)
        if "kim_potential" in self.parameters:
            self.kim_potential = self.parameters["kim_potential"]
            self.logger.info("kim potential: %s is set from parameters", self.kim_potential)

        return changed_parameters

    def calculate(
        self,
        atoms: ase.Atoms = None,
        properties=implemented_properties,
        system_changes=all_changes,
    ) -> None:
        """Calculate the properties of the system including static and relaxed energies.

        Args:
            atoms (ase.Atoms): The atoms object to calculate the properties for.
            properties (tuple): The properties to calculate.
            system_changes (list): The system changes to calculate.
        """
        if atoms is None:
            atoms = self.atoms

        Calculator.calculate(self, atoms, properties, system_changes)

        if "energy" in properties:
            unrelaxed_results = self.run_lammps_energy(atoms, run_dir=self.run_dir)
            self.results["energy"] = unrelaxed_results[1]
            self.results["per_atom_energies"] = unrelaxed_results[2]

        if "relaxed_energy" in properties:
            relaxed_results = self.run_lammps_opt(atoms, run_dir=self.run_dir)
            self.results["relaxed_energy"] = relaxed_results[1]
            self.results["per_atom_energies"] = relaxed_results[2]


class LAMMPSSurfCalc(LAMMMPSCalc):
    """Custom LAMMPSSurfCalc class to calculate surface energy."""

    implemented_properties = (*LAMMMPSCalc.implemented_properties, "surface_energy")

    def __init__(self, *args, **kwargs):
        """Initialize the LAMMPSSurfCalc class."""
        super().__init__(*args, **kwargs)
        self.logger = kwargs.get("logger", logging.getLogger(__name__))

    def get_surface_energy(self, atoms: ase.Atoms = None) -> float:
        """Get the surface energy of the system. Currently the same as the potential energy.

        Args:
            atoms (ase.Atoms): The atoms object to calculate the surface energy for.

        Returns:
            float: The surface energy of the system.
        """
        if atoms is None:
            atoms = self.atoms

        return self.get_potential_energy(atoms=atoms)

    def set(self, **kwargs) -> dict:
        """Set parameters in key-value pairs. A dictionary containing the parameters that have been
        changed is returned. The special keyword 'parameters' can be used to read parameters from a
        file.

        Args:
            **kwargs: The parameters to set.

        Returns:
            dict: A dictionary containing the parameters that have been changed.
        """
        return LAMMMPSCalc.set(self, **kwargs)

    def calculate(
        self,
        atoms: ase.Atoms = None,
        properties=implemented_properties,
        system_changes=all_changes,
    ) -> None:
        """Caculate based on LAMMMPSCalc before add in surface energy calcs to results.

        Args:
            atoms (ase.Atoms): The atoms object to calculate the properties for.
            properties (tuple): The properties to calculate.
            system_changes (list): The system changes to calculate.
        """
        if atoms is None:
            atoms = self.atoms

        LAMMMPSCalc.calculate(self, atoms, properties, system_changes)

        if "surface_energy" in properties:
            self.results["surface_energy"] = self.get_surface_energy(atoms=atoms)


class LAMMPSRunSurfCalc(LAMMPSRun):
    """LAMMPSRunSurfCalc class based on ASE LAMMPSRun to calculate surface energy."""

    implemented_properties = (*LAMMPSRun.implemented_properties, "surface_energy")

    def __init__(self, *args, **kwargs):
        """Initialize the LAMMPSRunSurfCalc class."""
        super().__init__(*args, **kwargs)
        self.logger = kwargs.get("logger", logging.getLogger(__name__))

    def get_surface_energy(self, atoms: ase.Atoms = None) -> float:
        """Get the surface energy of the system. Currently the same as the potential energy.

        Args:
            atoms (ase.Atoms): The atoms object to calculate the surface energy for.

        Returns:
            float: The surface energy of the system.
        """
        if atoms is None:
            atoms = self.atoms

        return self.get_potential_energy(atoms=atoms)

    def set(self, **kwargs) -> dict:
        """Set parameters in key-value pairs. A dictionary containing the parameters that have been
        changed is returned. The special keyword 'parameters' can be used to read parameters from a
        file.

        Args:
            **kwargs: The parameters to set.

        Returns:
            dict: A dictionary containing the parameters that have been changed.
        """
        return LAMMPSRun.set(self, **kwargs)

    def calculate(
        self,
        atoms: ase.Atoms = None,
        properties=implemented_properties,
        system_changes=all_changes,
    ) -> None:
        """Caculate based on LAMMPSRun before add in surface energy calcs to results.

        Args:
            atoms (ase.Atoms): The atoms object to calculate the properties for.
            properties (tuple): The properties to calculate.
            system_changes (list): The system changes to calculate.
        """
        if atoms is None:
            atoms = self.atoms

        LAMMPSRun.calculate(self, atoms, properties, system_changes)

        if "surface_energy" in properties:
            self.results["surface_energy"] = self.get_surface_energy(atoms=atoms)
