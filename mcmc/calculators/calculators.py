"""Module for ASE-style Calculators for surface energy calculations."""

import json
import logging
import os
from collections import Counter

import ase
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.formula import Formula
from lammps import (
    LMP_STYLE_ATOM,
    LMP_STYLE_GLOBAL,
    LMP_TYPE_SCALAR,
    LMP_TYPE_VECTOR,
    lammps,
)
from nff.io.ase import AtomsBatch
from nff.io.ase_calcs import EnsembleNFF, NeuralFF
from nff.utils.constants import HARTREE_TO_EV
from tqdm import tqdm

from mcmc.pourbaix.atoms import PourbaixAtom

from .lammpsrun import LAMMPS as LAMMPSRun

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
    """Calculate the force standard deviations across multiple models for a list of AtomsBatch
        objects

    Args:
    atoms_batches (List[AtomsBatch]): List of AtomsBatch objects
    calc (Calculator): NFF Calculator

    Returns:
        np.ndarray: Force standard deviation with a single value for each structure
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


class NFFPourbaix(NeuralFF):
    """Calculate Pourbaix potential for surface or bulk systems using Neural Force Field.
    We calculate the energy difference based on consecutive desorption/adsorption reactions as in
    Rong and Kolpak, J. Phys. Chem. Lett., 2015.
    Each step consists of the following:
    1. G_ref -> G_new + A
    2. A + n_H2O H2O -> HxAOy^(z-) + n_H+ H+ + n_e e-
    Delta G_overall = Delta G_1 + Delta G_2
    The free energy change for the first step is given by:
    Delta G_1 = G_new + mu_A - G_ref
    where G_new is the energy of the new system, mu_A is the chemical potential of the element A,
    and G_ref is the energy of the reference system.
    The free energy change for the second step is given by:
    Delta G_2 = Delta G_SHE - n_e (e*U_SHE) - 2.3 n_H+ k_B T pH + k_B T ln(a_HxAOy^(z-))
    where Delta G_SHE is the energy change at standard hydrogen electrode potential,
    e is the electron charge, U_SHE is the standard hydrogen electrode potential,
    n_H+ is the number of protons, k_B is the Boltzmann constant, T is the temperature,
    pH is the pH, and a_HxAOy^(z-) is the activity of the species.

    Attributes:
        implemented_properties (list): List of implemented properties.
        chem_pots (dict): Dictionary of chemical potentials.
        reference_slab (dict): Dictionary of reference slabs.
        temp (float): Temperature in eV.
        phi (float): Electric potential.
        pH (float): pH value.
        pourbaix_atoms (dict): Dictionary of pourbaix atoms.

    Methods:
        get_delta_G2_individual: Get the standard free energy change for the second step of the
            Pourbaix reaction for a single atom.
        get_delta_G2: Get the standard free energy change for the second step of the
            Pourbaix reaction.
        get_delta_G1: Get the dissociation energy of all atoms.
        get_surface_energy: Get the surface energy of the system, which is equivalent to the
            Pourbaix potential.
        get_pourbaix_potential: Get the Pourbaix potential of the system.
        set: Set parameters.
        calculate: Calculate based on NeuralFF before adding surface energy calculations to results.
    """

    implemented_properties = (
        *NeuralFF.implemented_properties,
        "pourbaix_potential",
        "surface_energy",
    )

    def __init__(self, *args, **kwargs):
        """Initialize the NFFPourbaix class."""
        super().__init__(*args, **kwargs)
        self.chem_pots = {}
        self.reference_slab = {}
        self.temp = kwargs.get("temp", 0.0257)  # temperature in eV
        self.phi = kwargs.get("phi", 0)  # electric potential
        self.pH = kwargs.get("pH", 7)  # pH
        self.pourbaix_atoms = {}
        self.adsorbate_corrections = {}
        self.logger = kwargs.get("logger", logging.getLogger(__name__))

    def get_delta_G2_individual(self, atom: str | PourbaixAtom) -> float:
        """Get the free energy change for the second step of the Pourbaix reaction for a
        single atom.

        Args:
            atom (Union[str, PourbaixAtom]): The atom to calculate the free energy change for.

        Returns:
            float: The standard free energy change for the second step for a single atoms.
        """
        if isinstance(atom, str):
            atom = self.pourbaix_atoms[atom]
        # - n_e (e*U_SHE) - 2.3 n_H+ k_B T pH + k_B T ln(a_HxAOy^(z-))
        delta_G2_non_std = (
            -atom.num_e * self.phi
            - np.log(10) * atom.num_H * self.temp * self.pH
            + self.temp * np.log(atom.species_conc)
        )
        return atom.delta_G2_std + delta_G2_non_std

    def get_delta_G2(self, atoms: ase.Atoms = None) -> float:
        """Get the total free energy change for the second step of the Pourbaix reaction.

        Args:
            atoms (ase.Atoms, optional): The atoms object to calculate the free energy change for.
                Defaults to None.

        Returns:
            float: The total standard free energy change for the second step.
        """
        if atoms is None:
            atoms = self.atoms

        delta_G2 = 0
        for atom in atoms.get_chemical_symbols():
            delta_G2 += self.get_delta_G2_individual(atom)
        return delta_G2

    def get_delta_G1(self, atoms: ase.Atoms = None) -> float:
        """Get the dissociation energy of all atoms
        Args:
            atoms (ase.Atoms, optional): The atoms object to calculate the dissociation energy for.
                Defaults to None.

        Returns:
            float: The dissociation energy of all atoms.
        """
        if atoms is None:
            atoms = self.atoms

        atoms_count = Counter(atoms.get_chemical_symbols())
        sum_chem_pots = 0
        for atom, count in atoms_count.items():
            sum_chem_pots += count * self.pourbaix_atoms[atom].atom_std_state_energy
        slab_energy = self.get_potential_energy(atoms=atoms)
        # Add adsorbate corrections, e.g. OH ZPE-TS correction

        formula = Formula(atoms.get_chemical_formula())
        for adsorbate, correction in self.adsorbate_corrections.items():
            # Check for H2O
            if "O" in adsorbate and "H" in adsorbate:
                # Assume the extra H is from water so subtract H2O from the formula
                HO_diff = max(formula["H"] - formula["O"], 0)
                if HO_diff > 0:
                    logger.info("Correcting formula %s with HO diff %s", formula, HO_diff)
                    formula_dict_to_subtract = (Formula("H2O") * HO_diff).count()
                    formula_dict = formula.count()
                    formula_dict = {
                        formula: formula_dict[formula] - formula_dict_to_subtract.get(formula, 0)
                        for formula in formula_dict
                    }
                    formula = Formula.from_dict(formula_dict)
                    logger.info("Corrected formula %s", formula)
            div, _ = divmod(formula, adsorbate)
            slab_energy += div * correction
        return sum_chem_pots - slab_energy

    def get_surface_energy(self, atoms: ase.Atoms = None) -> float:
        """Get the surface energy of the system, which is equivalent to the Pourbaix potential.
        See get_pourbaix_potential for more details.

        Args:
            atoms (ase.Atoms, optional): The atoms object to calculate the surface energy for.
                Defaults to None.

        Returns:
            float: The surface energy of the system.
        """
        if atoms is None:
            atoms = self.atoms

        return self.get_pourbaix_potential(atoms=atoms)

    def get_pourbaix_potential(self, atoms: ase.Atoms = None) -> float:
        """Get the Pourbaix potential of the system, which is the negative of the sum of the free
        energy changes for the two steps of the Pourbaix dissolution reaction. This is also the
        "surface free energy" and the "Grand potential" in the Pourbaix diagram.

        Args:
            atoms (ase.Atoms, optional): The atoms object to calculate the Pourbaix potential for.
                Defaults to None.

        Returns:
            float: The Pourbaix potential of the system.
        """
        if atoms is None:
            atoms = self.atoms

        return -(self.get_delta_G1(atoms=atoms) + self.get_delta_G2(atoms=atoms))

    def set(self, **kwargs) -> dict:
        """Set parameters in key-value pairs. A dictionary containing the parameters that have been
        changed is returned. The special keyword 'parameters' can be used to read parameters from a
        file.

        Args:
            **kwargs: The parameters to set.

        Returns:
            dict: A dictionary containing the parameters that have been changed.
        """
        changed_params = NeuralFF.set(self, **kwargs)
        if "temperature" in self.parameters:
            self.temp = self.parameters["temperature"]
            self.logger.info("temperature: %.3f in kBT", self.temp)
        if "phi" in self.parameters:
            self.phi = self.parameters["phi"]
            self.logger.info("potential: %.3f is set from parameters", self.phi)
        if "pH" in self.parameters:
            self.pH = self.parameters["pH"]
            self.logger.info("pH: %.3f is set from parameters", self.pH)
        if "pourbaix_atoms" in self.parameters:
            self.pourbaix_atoms = self.parameters["pourbaix_atoms"]
            self.logger.info("Pourbaix atoms: %s are set from parameters", self.pourbaix_atoms)
        if "adsorbate_corrections" in self.parameters:
            self.adsorbate_corrections = self.parameters["adsorbate_corrections"]
            self.logger.info(
                "adsorbate corrections: %s are set from parameters", self.adsorbate_corrections
            )
        return changed_params

    def calculate(
        self,
        atoms: ase.Atoms = None,
        properties: tuple = implemented_properties,
        system_changes: list = all_changes,
    ):
        """Caculate based on NeuralFF before add in surface energy calcs to results
        Args:
            atoms: ase.Atoms
                The atoms object to calculate the properties for.
            properties: List
                The properties to calculate.
            system_changes: List
                The system changes to calculate.
        """
        if atoms is None:
            atoms = self.atoms

        NeuralFF.calculate(self, atoms, properties, system_changes)

        if "surface_energy" in properties:
            self.results["surface_energy"] = self.get_pourbaix_potential(atoms=atoms)

        atoms.results.update(self.results)


# TODO define abstract base class for surface energy calcs
# use EnsembleNFF, NeuralFF classes for NFF
class EnsembleNFFSurface(EnsembleNFF):
    """Based on Ensemble Neural Force Field class to calculate surface energy"""

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
        potential deviation from the bulk formula. Refer to Methods-Surface stability analysis
        section of Du, X. et al. Nat Comput Sci 1-11 (2023) doi:10.1038/s43588-023-00571-7
        for details.

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

        # Starting with the potential energy of the system (akin to DFT energy of the slab)
        surface_energy = self.get_potential_energy(atoms=atoms)

        ads_count = Counter(atoms.get_chemical_symbols())

        bulk_energies = offset_data["bulk_energies"]
        stoics = offset_data["stoics"]
        ref_formula = offset_data["ref_formula"]
        ref_element = offset_data["ref_element"]

        # Subtract the bulk energies
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

        # Subtract chemical potential deviation from bulk formula
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

        potential_file = config.get("potential_file")
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

        # For some ase versions, the retrieved atomic numbers are not the 'real' atomic numbers
        if not set(new_slab.get_atomic_numbers()) <= set(atomic_numbers_dict.values()):
            actual_atomic_numbers = [
                atomic_numbers_dict[str(x)] for x in new_slab.get_atomic_numbers()
            ]
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
