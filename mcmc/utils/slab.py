"""Methods for generating slabs from bulk structures."""

from collections.abc import Iterable

import ase
import numpy as np
from ase.build.tools import sort
from catkit.gen.surface import SlabGenerator
from pymatgen.core.structure import Structure
from pymatgen.core.surface import SlabGenerator as PMGSlabGenerator
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.transformations.standard_transformations import SupercellTransformation


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
        bulk: Bulk structure to cut surface from.
        miller_index: Miller indices for the surface, length 3.
        layers: Number of layers in the slab, by default 5
        fixed: Number of fixed layers, by default 6
        size: Size of the slab with respect to provided bulk, by default (1, 1)
        vacuum: Vacuum space in Angstroms (in each direction), by default 7.5
        iterm: Index of the slab termination, by default 0

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


def symmetrize_slab(slab: ase.Atoms, num_bottom_atoms: int, sort_z_axis: True) -> ase.Atoms:
    """Symmetrize a slab by copying the top half to the bottom half. Assume the slab is
    sorted in the z direction in increasing order.

    Args:
        slab: Surface slab.
        num_bottom_atoms: Number of atoms in the bottom layer.
        sort_z_axis: Whether to sort the slab in the z direction.

    Returns:
        ase.Symmetrized slab.
    """
    slab_copy = slab.copy()
    if sort_z_axis:
        # Sort according to z coordinate
        slab_copy = sort(slab_copy, tags=slab_copy.positions[:, 2])
    # Copy symmetric half except bottom atoms
    bottom_atoms_z_mean = slab_copy[:num_bottom_atoms].get_scaled_positions()[:, 2].mean()
    symmetric_bottom = slab_copy[num_bottom_atoms:].copy()
    dist_from_bottom = symmetric_bottom.get_scaled_positions()[:, 2] - bottom_atoms_z_mean

    symmetric_bottom.set_scaled_positions(
        np.hstack(
            [
                symmetric_bottom.get_scaled_positions()[:, :2],
                (bottom_atoms_z_mean - dist_from_bottom)[:, None],
            ]
        )
    )
    slab_copy += symmetric_bottom
    return slab_copy


class SupercellSurfaceGenerator:
    """Generate a supercell slab from a bulk structure."""

    def __init__(
        self, structure: Structure, miller_index: tuple[int], min_slab_size=10, min_vacuum_size=15
    ):
        """Initialize the supercell slab generator.

        Args:
            structure: Bulk structure to cut surface from.
            miller_index:  Miller indices for the surface, length 3.
            min_slab_size: Minimum slab size, by default 10.
            min_vacuum_size: Minimum vacuum size, by default 15.
        """
        self.primitive_structure = structure.get_primitive_structure()
        self.miller_index = miller_index
        self.slab_generator = PMGSlabGenerator(
            self.primitive_structure,
            (0, 0, 1),
            min_slab_size,
            min_vacuum_size,
            primitive=True,
            center_slab=True,
        )

    def get_supercell_lattice(
        self, new_a: float, new_b: float, new_c: float | None = None
    ) -> np.ndarray:
        """Get the supercell lattice for the slab.

        Args:
            new_a: New lattice vector a.
            new_b: New lattice vector b.
            new_c: New lattice vector c, by default None.

        Returns:
            Supercell lattice.
        """
        if new_c is None:
            new_c = 1
        # supercell_lattice = np.dot(lattice_matrix, supercell_lattice)
        return np.array([[new_a, 0, 0], [0, new_b, 0], [0, 0, new_c]])

    def get_primitive_slab(self, shift=0.5, repair_broken_bonds: dict | None = None) -> Structure:
        """Get the primitive slab structure.

        Args:
            shift: Shift for the slab, by default 0.5.
            repair_broken_bonds: Dictionary for repairing broken bonds, by default None.

        Returns:
            Primitive slab structure.
        """
        slab = self.slab_generator.get_slab(shift=shift)
        if repair_broken_bonds is not None:
            slab = self.slab_generator.repair_broken_bonds(slab, repair_broken_bonds)
        slab = slab.get_orthogonal_c_slab()
        for site in slab:
            site.to_unit_cell(in_place=True)
        return slab

    @property
    def hkl_to_hkil(self) -> tuple[int]:
        """Converts Miller indices (hkl) to hexagonal Miller-Bravais indices (hkil).

        Returns:
            Converted hexagonal Miller-Bravais indices (h, k, i, l).
        """
        # Calculate i = -(h + k)
        h, k, ell = self.miller_index
        i = -(h + k)
        return (h, k, i, ell)

    def generate_periodic_sites(self, structure: Structure) -> list[tuple[str, np.ndarray]]:
        """Generate periodic images by translating the atomic positions using lattice vectors and
            specific offsets.

        Args:
            structure: Structure object to generate periodic images from.

        Returns:
            List of translated atomic positions
        """
        periodic_sites = []
        offsets = [(0, 0), (1, 1), (1, -1), (-1, 1), (-1, -1), (0, 1), (1, 0), (0, -1), (-1, 0)]
        # Iterate over each translation offset
        for tx, ty in offsets:
            translation_vector = tx * structure.lattice.matrix[0] + ty * structure.lattice.matrix[1]
            # Translate all atoms in the structure by this vector
            for site in structure.sites:
                translated_cart_coords = site.coords + translation_vector
                periodic_sites.append((site.species, translated_cart_coords))
        return periodic_sites

    def filter_sites_in_box(
        self, cart_coords: list[float], lattice_vectors: list[float], epsilon: float = 1e-8
    ) -> tuple:
        """Filter the random sites that are inside the periodic box.

        Args:
            cart_coords: List of Cartesian coordinates of random sites (Nx3 array)
            lattice_vectors: Lattice vectors of the periodic box (3x3 array)
            epsilon: Small value to handle numerical precision when checking boundaries

        Returns:
            List of filtered sites inside the periodic box and their indices
        """
        # Initialize an empty list to store sites inside the box
        sites_inside_box = []
        filtered_indices = []
        # Loop through each random site
        for i, site in enumerate(cart_coords):
            # Convert the site from Cartesian coordinates to fractional coordinates
            fractional_coords = np.linalg.solve(lattice_vectors.T, site)
            # Check if all fractional coordinates are between 0 and 1 (within a small tolerance)
            if np.all(fractional_coords >= 0.0) and np.all(fractional_coords < 1.0):
                # If inside the box, add the site to the list
                sites_inside_box.append(site)
                filtered_indices.append(i)
        return sites_inside_box, filtered_indices

    def rotate_and_wrap_positions(self, periodic_sites, theta_deg, lattice_vectors):
        """Rotates the atomic positions by theta degrees in the xy-plane and wraps them back into
            the unit cell. Only atoms that end up in the (0, 0) unit cell are retained.

        Args:
            periodic_sites: List of atomic positions and species from periodic images
            theta_deg: Rotation angle in degrees
            lattice_vectors: Lattice vectors for the unit cell

        Returns:
            Rotated and wrapped Structure object, only keeping atoms within (0, 0) unit cell
        """
        # Convert the angle to radians
        theta_rad = np.radians(theta_deg)
        # Define the 2D rotation matrix
        rotation_matrix = np.array(
            [[np.cos(theta_rad), -np.sin(theta_rad)], [np.sin(theta_rad), np.cos(theta_rad)]]
        )
        # Initialize list for the rotated and wrapped sites
        rotated_cart_coords_list = []
        # Iterate over each site in the periodic images
        for _, cart_coords in periodic_sites:
            # Rotate the (x, y) coordinates
            xy_coords = np.dot(cart_coords[:2], rotation_matrix.T)
            rotated_cart_coords = np.array(
                [xy_coords[0], xy_coords[1], cart_coords[2]]
            )  # z remains unchanged
            rotated_cart_coords_list.append(rotated_cart_coords)
        # Filter the rotated sites that are inside the periodic box
        filtered_sites, filtered_indices = self.filter_sites_in_box(
            rotated_cart_coords_list, lattice_vectors
        )
        # Extract the species corresponding to the filtered sites
        filtered_species = [periodic_sites[i][0] for i in filtered_indices]
        # Create a new structure from the rotated and wrapped atomic positions in the (0, 0) offset
        return Structure(
            lattice_vectors, filtered_species, filtered_sites, coords_are_cartesian=True
        )

    def get_supercell_slab(
        self,
        new_a: float,
        new_b: float,
        new_c: float | None = None,
        rotation: float = 0.0,
        shift: float = 0.5,
        repair_broken_bonds: dict | None = None,
    ) -> Structure:
        """Generate a supercell slab from a bulk structure.

        Args:
            new_a: New lattice vector a.
            new_b: New lattice vector b.
            new_c: New lattice vector c, by default None.
            rotation: Rotation angle in degrees, by default 0.0.
            shift: Shift for the slab, by default 0.5.
            repair_broken_bonds: Dictionary for repairing broken bonds, by default None.

        Returns:
            Supercell slab structure.
        """
        slab = self.get_primitive_slab(shift=shift, repair_broken_bonds=repair_broken_bonds)
        supercell_lattice_scaler = self.get_supercell_lattice(new_a, new_b, new_c)
        supercell_transformer = SupercellTransformation(supercell_lattice_scaler)
        supercell = supercell_transformer.apply_transformation(slab.get_orthogonal_c_slab())
        supercell_lattice = np.dot(slab.lattice.matrix, supercell_lattice_scaler)
        periodic_sites = self.generate_periodic_sites(supercell)
        return self.rotate_and_wrap_positions(periodic_sites, rotation, supercell_lattice)

    @classmethod
    def save_slab(cls, slab: Structure, filename: str = "POSCAR") -> None:
        """Save the slab to a POSCAR file.

        Args:
            slab: Slab structure.
            filename: Filename for the POSCAR file, by default "POSCAR".
        """
        Poscar(slab).write_file(filename)
