"""Pourbaix atoms module for calculating the Pourbaix potential (energy)."""

from typing import Self

from ase import Atom
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.analysis.pourbaix_diagram import (
    HydrogenPourbaixEntry,
    IonEntry,
    MultiEntry,
    OxygenPourbaixEntry,
    PourbaixDiagram,
    PourbaixEntry,
)
from pymatgen.core import Composition, Element
from pymatgen.core.ion import Ion
from pymatgen.entries.computed_entries import ComputedEntry

ELEMENTS_HO = {Element("H"), Element("O")}
SYMBOLS_HO = {elem.symbol for elem in ELEMENTS_HO}


class PourbaixAtom(Atom):
    """Atom object with support for Pourbaix calculations."""

    def __init__(
        self,
        symbol: str | int,
        dominant_species: str,
        species_conc: float = 1e-6,
        num_e: int = 0,
        num_H: int = 0,
        atom_std_state_energy: float = 0,
        delta_G2_std: float = 0,
        **kwargs,
    ):
        """Create a PourbaixAtom object.

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
        super().__init__(symbol, **kwargs)
        self.dominant_species = dominant_species
        self.species_conc = species_conc
        self.num_e = num_e
        self.num_H = num_H
        self.atom_std_state_energy = atom_std_state_energy
        self.delta_G2_std = delta_G2_std

    @property
    def get_dominant_species(self):
        """Get the dominant species."""
        # TODO return based on pH and phi
        return self.dominant_species

    @classmethod
    def from_pourbaix_entry(
        cls,
        symbol: str,
        pbx_entry: PourbaixEntry,
        phase_diagram: PhaseDiagram,
        **kwargs,
    ) -> Self:
        """Create a PourbaixAtom object from a PourbaixEntry and a PhaseDiagram.

        Args:
            symbol (str): Chemical symbol of the atom.
            pbx_entry (PourbaixEntry): pymatgen PourbaixEntry object.
            phase_diagram (PhaseDiagram): pymatgen PhaseDiagram object based on which the
                PourbaixEntry was created.
            **kwargs: Additional keyword arguments.

        Returns:
            PourbaixAtom: PourbaixAtom object.
        """
        return cls(
            symbol,
            dominant_species=pbx_entry.entry.reduced_formula,
            species_conc=pbx_entry.concentration,
            num_e=-pbx_entry.normalized_nPhi,
            num_H=-pbx_entry.normalized_npH,
            atom_std_state_energy=phase_diagram.get_reference_energy_per_atom(Composition(symbol)),
            delta_G2_std=(pbx_entry.energy - pbx_entry.conc_term) * pbx_entry.normalization_factor,
            **kwargs,
        )

    def as_dict(self) -> dict[str, Self]:
        """Get MSONable dict."""
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "symbol": self.symbol,
            "dominant_species": self.dominant_species,
            "species_conc": self.species_conc,
            "num_e": self.num_e,
            "num_H": self.num_H,
            "atom_std_state_energy": self.atom_std_state_energy,
            "delta_G2_std": self.delta_G2_std,
        }

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Args:
            dct (dict): Dict representation.

        Returns:
            SurfacePourbaixDiagram
        """
        return cls(
            symbol=dct["symbol"],
            dominant_species=dct["dominant_species"],
            species_conc=dct["species_conc"],
            num_e=dct["num_e"],
            num_H=dct["num_H"],
            atom_std_state_energy=dct["atom_std_state_energy"],
            delta_G2_std=dct["delta_G2_std"],
        )

    def __repr__(self):
        """Representation of the object."""
        return (
            f"PourbaixAtom('{self.symbol}' species={self.dominant_species}, "
            f"num_e={self.num_e}, num_H={self.num_H}, "
            f"atom_std_state_energy={self.atom_std_state_energy:.3f}, "
            f"delta_G2_std={self.delta_G2_std:.3f})"
        )


def generate_pourbaix_atoms(
    phase_diagram: PhaseDiagram,
    pourbaix_diagram: PourbaixDiagram,
    phi: float,
    pH: float,
    elements: list[str],
) -> dict[str, PourbaixAtom]:
    """Generate Pourbaix atoms representing the dominant species for the given elements at the given
    pH and phi.

    Args:
        phase_diagram (Union[Path, str]): pymatgen PhaseDiagram
        pourbaix_diagram (Union[Path, str]): pymatgen PourbaixDiagram
        phi (float): electrical potential
        pH (float): pH
        elements (list[str]): list of elements

    Returns:
        dict: dictionary of PourbaixAtom objects
    """
    pbx_multi_entry = pourbaix_diagram.get_stable_entry(pH, phi)
    assert isinstance(pbx_multi_entry, MultiEntry), "Expected a Pourbaix MultiEntry"
    sorted_pbx_entries = sorted(
        pbx_multi_entry.entry_list,  # there should be only 1 non-OH element per entry
        key=lambda entry: list(set(entry.composition.elements) - ELEMENTS_HO).pop().symbol,
    )
    sorted_symbols = sorted(set(elements) - SYMBOLS_HO)

    # H2O entry
    # for the reaction: 1/2 O2 or O -> H2O - 2H+ - 2e-
    H2O_entry = next(x for x in phase_diagram.stable_entries if x.reduced_formula == "H2O")
    H2O_pourbaix_entry = OxygenPourbaixEntry(
        ComputedEntry(
            H2O_entry.composition,
            phase_diagram.get_form_energy(H2O_entry),
            parameters=H2O_entry.parameters,
        )
    )

    # H+ entry
    # for the reaction: 1/2 H2 or H -> H+ + e-
    H_ion_pourbaix_entry = HydrogenPourbaixEntry(IonEntry(Ion.from_formula("H[1+]"), 0.0))

    sorted_symbols += ["O", "H"]
    sorted_pbx_entries += [H2O_pourbaix_entry, H_ion_pourbaix_entry]

    return {
        element: PourbaixAtom.from_pourbaix_entry(element, pbx_entry, phase_diagram)
        for element, pbx_entry in zip(sorted_symbols, sorted_pbx_entries, strict=False)
    }
