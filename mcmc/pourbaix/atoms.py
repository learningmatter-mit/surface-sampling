from pathlib import Path
from typing import Union

import numpy as np
from ase import Atom
from monty.serialization import loadfn
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.analysis.pourbaix_diagram import (
    MultiEntry,
    OxygenPourbaixEntry,
    PourbaixEntry,
)
from pymatgen.core import Composition, Element
from pymatgen.entries.computed_entries import ComputedEntry

ELEMENTS_HO = {Element("H"), Element("O")}
SYMBOLS_HO = {elem.symbol for elem in ELEMENTS_HO}


class PourbaixAtom(Atom):
    """Atom object with support for Pourbaix calculations.

    Args:
        symbol (str): Chemical symbol of the atom.
        dominant_species (str): Dominant species of the atom.
        species_conc (float): Concentration of the species.
        num_e (int): Number of electrons.
        num_H (int): Number of protons.
        atom_std_state_energy (float): Atom standard state energy.
        delta_G2_std (float): Delta G2 standard.
        **kwargs: Additional keyword arguments.

    Attributes:
        dominant_species (str): Dominant species of the atom.
        species_conc (float): Concentration of the species.
        num_e (int): Number of electrons.
        num_H (int): Number of protons.
        atom_std_state_energy (float): Atom standard state energy.
        delta_G2_std (float): Delta G2 standard.
    """

    def __init__(
        self,
        symbol: Union[str, int],
        dominant_species: str,
        species_conc: float = 1e-6,
        num_e: int = 0,
        num_H: int = 0,
        atom_std_state_energy: float = 0,
        delta_G2_std: float = 0,
        **kwargs,
    ):
        super().__init__(symbol, **kwargs)
        self.dominant_species = dominant_species
        self.species_conc = species_conc
        self.num_e = num_e
        self.num_H = num_H
        self.atom_std_state_energy = atom_std_state_energy
        self.delta_G2_std = delta_G2_std

    @property
    def get_dominant_species(self):
        # TODO return based on pH and phi
        return self.dominant_species

    def get_electron_coefficient(self):
        return 0

    def get_proton_coefficient(self):
        return 0

    @classmethod
    def from_pourbaix_entry(
        cls,
        symbol: str,
        pbx_entry: PourbaixEntry,
        phase_diagram: PhaseDiagram,
        **kwargs,
    ) -> "PourbaixAtom":
        """Create a PourbaixAtom object from a PourbaixEntry and a PhaseDiagram.

        Args:
            symbol (str): Chemical symbol of the atom.
            pbx_entry (PourbaixEntry): pymatgen PourbaixEntry object.
            phase_diagram (PhaseDiagram): pymatgen PhaseDiagram object based on which the PourbaixEntry was created.

        Returns:
            PourbaixAtom: PourbaixAtom object.
        """
        return cls(
            symbol,
            dominant_species=pbx_entry.entry.reduced_formula,
            species_conc=pbx_entry.concentration,
            num_e=-pbx_entry.normalized_nPhi,
            num_H=-pbx_entry.normalized_npH,
            atom_std_state_energy=phase_diagram.get_reference_energy_per_atom(
                Composition(symbol)
            ),
            delta_G2_std=(pbx_entry.energy - pbx_entry.conc_term)
            * pbx_entry.normalization_factor,
            **kwargs,
        )

    def __repr__(self):
        s = (
            f"PourbaixAtom('{self.symbol}' species={self.dominant_species}, "
            f"num_e={self.num_e}, num_H={self.num_H}, "
            f"atom_std_state_energy={self.atom_std_state_energy:.3f}, "
            f"delta_G2_std={self.delta_G2_std:.3f})"
        )
        return s


def generate_pourbaix_atoms(
    phase_diagram_path: Union[Path, str],
    pourbaix_diagram_path: Union[Path, str],
    phi: float,
    pH: float,
    elements: list[str],
) -> dict[str, PourbaixAtom]:
    """Generate Pourbaix atoms representing the dominant species for the given elements at the given pH and phi.

    Args:
        phase_diagram_path (Union[Path, str]): path to the saved pymatgen PhaseDiagram
        pourbaix_diagram_path (Union[Path, str]): path to the saved pymatgen PourbaixDiagram
        phi (float): electrical potential
        pH (float): pH
        elements (list[str]): list of elements

    Returns:
        dict: dictionary of PourbaixAtom objects
    """
    phase_diagram = loadfn(phase_diagram_path)
    pourbaix_diagram = loadfn(pourbaix_diagram_path)

    pbx_multi_entry = pourbaix_diagram.get_stable_entry(pH, phi)
    assert isinstance(pbx_multi_entry, MultiEntry), "Expected a Pourbaix MultiEntry"
    sorted_pbx_entries = sorted(
        pbx_multi_entry.entry_list,  # there should be only 1 non-OH element per entry
        key=lambda entry: list(set(entry.composition.elements) - ELEMENTS_HO)
        .pop()
        .symbol,
    )
    sorted_symbols = sorted(set(elements) - SYMBOLS_HO)

    # H2O entries
    H2O_entry = [x for x in phase_diagram.stable_entries if x.reduced_formula == "H2O"][
        0
    ]
    H2O_pourbaix_entry = OxygenPourbaixEntry(
        ComputedEntry(
            H2O_entry.composition,
            phase_diagram.get_form_energy(H2O_entry),
            parameters=H2O_entry.parameters,
        )
    )

    sorted_symbols += ["O"]
    sorted_pbx_entries += [H2O_pourbaix_entry]

    pourbaix_atoms = {
        element: PourbaixAtom.from_pourbaix_entry(element, pbx_entry, phase_diagram)
        for element, pbx_entry in zip(sorted_symbols, sorted_pbx_entries)
    }

    return pourbaix_atoms
