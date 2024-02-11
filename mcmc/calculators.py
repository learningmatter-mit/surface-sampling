import json
import logging
import os
from collections import Counter
from typing import Dict, List, Tuple

import ase
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from lammps import (
    LMP_STYLE_ATOM,
    LMP_STYLE_GLOBAL,
    LMP_TYPE_SCALAR,
    LMP_TYPE_VECTOR,
    lammps,
)
from nff.io.ase import EnsembleNFF, NeuralFF
from nff.utils.constants import EV_TO_KCAL_MOL, HARTREE_TO_KCAL_MOL

HARTREE_TO_EV = HARTREE_TO_KCAL_MOL / EV_TO_KCAL_MOL

logger = logging.getLogger(__name__)


# use EnsembleNFF, NeuralFF classes for NFF
class EnsembleNFFSurface(EnsembleNFF):
    implemented_properties = EnsembleNFF.implemented_properties + ["surface_energy"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chem_pots = {}
        self.offset_data = {}

    def get_surface_energy(
        self, atoms: ase.Atoms = None, chem_pots: Dict = None, offset_data: Dict = None
    ):
        """Get the surface energy of the system by subtracting the bulk energy and the chemical potential deviation from the bulk formula.

        Parameters
        ----------
        atoms : ase.Atoms
            The atoms object to calculate the surface energy for.
        chem_pots : Dict
            The chemical potentials of the atoms in the system.
        offset_data : Dict
            The offset data for the system.

        Returns
        -------
        float
            The surface energy of the system.

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
        # stoidict = offset_data["stoidict"]
        stoics = offset_data["stoics"]
        ref_formula = offset_data["ref_formula"]
        ref_element = offset_data["ref_element"]

        # subtract the bulk energies
        # TODO: move to surface energy calc
        bulk_ref_en = ads_count[ref_element] * bulk_energies[ref_formula]
        for ele, _ in ads_count.items():
            if ele != ref_element:
                bulk_ref_en += (
                    ads_count[ele]
                    - stoics[ele] / stoics[ref_element] * ads_count[ref_element]
                ) * bulk_energies[ele]

        surface_energy -= bulk_ref_en * HARTREE_TO_EV

        # TODO make this a separate function
        # subtract chemical potential deviation from bulk formula
        stoics = self.offset_data["stoics"]
        ref_element = self.offset_data["ref_element"]

        pot = 0
        for ele, _ in ads_count.items():
            if ele != ref_element:
                pot += (
                    ads_count[ele]
                    - stoics[ele] / stoics[ref_element] * ads_count[ref_element]
                ) * self.chem_pots[ele]

        surface_energy -= pot
        return surface_energy

    def set(self, **kwargs):
        """Set parameters like set(key1=value1, key2=value2, ...).

        A dictionary containing the parameters that have been changed
        is returned.

        The special keyword 'parameters' can be used to read
        parameters from a file."""
        EnsembleNFF.set(self, **kwargs)
        if "chem_pots" in self.parameters.keys():
            self.chem_pots = self.parameters["chem_pots"]
            print(f"chemical potentials: {self.chem_pots} are set from parameters")
        if "offset_data" in self.parameters.keys():
            self.offset_data = self.parameters["offset_data"]
            print(f"offset data: {self.offset_data} is set from parameters")


class NeuralFFSurface(NeuralFF):
    pass


# use OpenKIM calc

# use ASE calc


class LAMMPSSurfCalc(Calculator):
    pass


class LAMMMPSCalc(Calculator):
    name = "lammpscalc"
    implemented_properties = ["energy", "relaxed_energy", "forces", "per_atom_energies"]
    # NOTE "energy" is the unrelaxed energy

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.implemented_properties = LAMMMPSCalc.implemented_properties
        self.run_dir = os.getcwd()

    def run_lammps_calc(
        self,
        slab,
        run_dir=os.getcwd(),
        template_path=f"{os.getcwd()}/lammps_opt_template.txt",
        lammps_config=f"{os.getcwd()}/lammps_config.json",
        **kwargs,
    ):
        lammps_template = open(template_path, "r").read()

        # config file is assumed to be stored in the folder you run lammps
        if type(lammps_config) is str:
            with open(lammps_config, "r") as f:
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
        slab.write(
            lammps_data_file, format="lammps-data", units="real", atom_style="atomic"
        )
        steps = kwargs.get("relax_steps", 100)

        # write lammps.in file
        with open(lammps_in_file, "w") as f:
            # if using KIM potential
            if kwargs.get("kim_potential", False):
                f.writelines(
                    lammps_template.format(
                        lammps_data_file, bulk_index, steps, lammps_out_file
                    )
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
        # lmp = lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
        lmp = lammps()
        logger.debug(lmp.file(lammps_in_file))

        energy = lmp.extract_compute("thermo_pe", LMP_STYLE_GLOBAL, LMP_TYPE_SCALAR)
        if "opt" in lammps_template:
            pe_per_atom = []
        else:
            pe_per_atom = lmp.extract_compute(
                "pe_per_atom", LMP_STYLE_ATOM, LMP_TYPE_VECTOR
            )
            pe_per_atom = np.ctypeslib.as_array(
                pe_per_atom, shape=(len(slab),)
            )  # convert to numpy array
        lmp.close()

        # Read from LAMMPS out
        new_slab = ase.io.read(lammps_out_file, format="lammps-data", style="atomic")

        atomic_numbers_dict = config["atomic_numbers_dict"]
        actual_atomic_numbers = [
            atomic_numbers_dict[str(x)] for x in new_slab.get_atomic_numbers()
        ]

        new_slab.set_atomic_numbers(actual_atomic_numbers)
        new_slab.calc = slab.calc

        return energy, pe_per_atom, new_slab

    def run_lammps_opt(self, slab, run_dir=os.getcwd(), **kwargs):
        energy, pe_per_atom, opt_slab = self.run_lammps_calc(
            slab,
            run_dir=run_dir,
            template_path=f"{run_dir}/lammps_opt_template.txt",
            lammps_config=f"{run_dir}/lammps_config.json",
            **kwargs,
        )
        logger.debug(f"slab energy in relaxation: {energy}")
        return opt_slab, energy, pe_per_atom

    def run_lammps_energy(self, slab, run_dir=os.getcwd(), **kwargs):
        # import pdb; pdb.set_trace()
        energy, pe_per_atom, _ = self.run_lammps_calc(
            slab,
            run_dir=run_dir,
            template_path=f"{run_dir}/lammps_energy_template.txt",
            lammps_config=f"{run_dir}/lammps_config.json",
            **kwargs,
        )
        logger.debug(f"slab energy in engrad: {energy}")
        return slab, energy, pe_per_atom

    def set(self, **kwargs):
        """Set parameters like set(key1=value1, key2=value2, ...).

        A dictionary containing the parameters that have been changed
        is returned.

        The special keyword 'parameters' can be used to read
        parameters from a file."""
        Calculator.set(self, **kwargs)

        if "run_dir" in self.parameters.keys():
            self.run_dir = self.parameters["run_dir"]
            print(f"run directory: {self.run_dir} is set from parameters")

    def calculate(
        self,
        atoms: ase.Atoms = None,
        properties=implemented_properties,
        system_changes=all_changes,
    ):

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
