"""Methods for generating slabs from bulk structures."""

from collections.abc import Iterable

import ase
import numpy as np
from catkit.gen.surface import SlabGenerator


def surface_from_bulk(
    bulk: ase.Atoms,
    miller_index: Iterable[int],
    layers: int = 5,
    fixed: int = 6,
    size: Iterable[int] = (1, 1),
    vacuum: float = 7.5,
    iterm: int = 0,
) -> tuple[ase.Atoms, list[bool]]:
    """Cut a surface from a bulk structure.

    Args:
        bulk (ase.Atoms): Bulk structure to cut surface from.
        miller_index (Iterable[int]): Miller indices for the surface, length 3.
        layers (int, optional): Number of layers in the slab, by default 5
        fixed (int, optional): Number of fixed layers, by default 6
        size (Iterable, optional): Size of the slab with respect to provided bulk, by default (1, 1)
        vacuum (float, optional): Vacuum space in Angstroms (in each direction), by default 7.5
        iterm (int, optional): Index of the slab termination, by default 0

    Returns:
        ase.Atoms: Surface cut from the crystal.
        list[bool]: list of surface atoms.
    """
    # bulk.set_initial_magnetic_moments(bulk.get_initial_magnetic_moments())
    # layers = 5
    gen = SlabGenerator(
        bulk,
        miller_index=miller_index,
        layers=layers,
        layer_type="angs",
        fixed=fixed,
        vacuum=vacuum,
        standardize_bulk=True,
        primitive=True,
        tol=0.2,
    )

    slab = gen.get_slab(iterm=iterm)
    slab = gen.set_size(slab, size)

    slab.center(vacuum=vacuum, axis=2)  # center slab in z direction

    # surface_atom_index = slab.get_surface_atoms().tolist()  # includes top surface atoms
    atom_list = np.arange(len(slab.numbers))
    z_values = slab.positions[:, 2]
    highest_z = np.max(z_values)
    surface_atoms = [highest_z - z_values[atom] < 1.2 for atom in atom_list]

    return slab, surface_atoms
