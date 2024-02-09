from collections import Counter
from typing import Dict, List, Tuple

import ase
from nff.io.ase import EnsembleNFF, NeuralFF
from nff.utils.constants import EV_TO_KCAL_MOL, HARTREE_TO_KCAL_MOL

HARTREE_TO_EV = HARTREE_TO_KCAL_MOL / EV_TO_KCAL_MOL


# use EnsembleNFF, NeuralFF classes for NFF
class EnsembleNFFSurface(EnsembleNFF):
    implemented_properties = EnsembleNFF.implemented_properties + ["surface_energy"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._surface = None
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
