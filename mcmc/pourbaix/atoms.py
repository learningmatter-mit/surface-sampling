from typing import Union

from ase import Atom


class PourbaixAtom(Atom):
    """Atom object with support for Pourbaix calculations.

    Parameters
    ----------
    PourbaixAtom : Atom
        Atom object from ASE.
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
