"""Materials Project style compatibility module for surface Pourbaix diagrams."""

from pymatgen.core import Element
from pymatgen.entries.compatibility import Compatibility
from pymatgen.entries.computed_entries import (
    CompositionEnergyAdjustment,
    ComputedEntry,
)


class SurfaceOHCompatibility(Compatibility):
    """Performs corrections for surface hydroxyl groups for surface Pourbaix diagrams."""

    def __init__(
        self,
        correction: float = 0.23,
    ) -> None:
        """Initialize the compatibility module.

        Args:
            correction: The energy correction to apply for surface hydroxyl groups in eV/group.
                Default is 0.23 eV from Rong and Kolpak, J. Phys. Chem. Lett., 2015.
        """
        self.correction = correction

    def get_adjustments(self, entry: ComputedEntry) -> list[CompositionEnergyAdjustment]:
        """Get the energy adjustments for a ComputedEntry or ComputedStructureEntry.

        Args:
            entry: A ComputedEntry or ComputedStructureEntry object.

        Returns:
            list[EnergyAdjustment]: A list of EnergyAdjustment to be applied to the Entry.
        """
        # apply energy adjustments
        adjustments: list[CompositionEnergyAdjustment] = []

        comp = entry.composition

        # Check for OH corrections
        # Assume the extra H is from water so subtract 1 O and 1 H from the composition
        HO_diff = max(comp["H"] - comp["O"], 0)
        if Element("O") in comp and Element("H") in comp:
            adjustments.append(
                CompositionEnergyAdjustment(
                    self.correction,
                    min(comp["O"], comp["H"]) - HO_diff,
                    # uncertainty_per_atom=self.comp_errors[ox_type],
                    name="Surface OH ZPE-TS correction",
                    cls=self.as_dict(),
                )
            )

        return adjustments
