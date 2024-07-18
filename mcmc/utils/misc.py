"""Miscellaneous utility functions for the MCMC workflow."""

import pickle as pkl
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from ase.atoms import Atoms
from nff.data import Dataset
from nff.io.ase import AtomsBatch
from scipy.spatial import distance
from scipy.special import softmax
from tqdm import tqdm


def get_atoms_batch(
    data: dict | Atoms,
    nff_cutoff: float,
    device: str = "cpu",
    **kwargs,
) -> AtomsBatch:
    """Generate AtomsBatch from atoms or dictionary.

    Args:
        data (Union[dict, Atoms]): Dictionary containing the properties of the atoms
        nff_cutoff (float): Neighbor cutoff for the NFF model
        device (str, optional): cpu or cuda device. Defaults to 'cpu'.
        **kwargs: Additional keyword arguments.

    Returns:
        AtomsBatch
    """
    if isinstance(data, Atoms):
        atoms_batch = AtomsBatch.from_atoms(
            data,
            cutoff=nff_cutoff,
            requires_large_offsets=False,
            dense_nbrs=False,
            directed=True,
            device=device,
            **kwargs,
        )
    else:
        pass
        # atoms_batch = AtomsBatch.from_dict(
        #     data,
        #     cutoff=nff_cutoff,
        #     requires_large_offsets=False,
        #     directed=True,
        #     device=device,
        #     **kwargs,
        # )

    return atoms_batch


def get_atoms_batches(
    data: Dataset | list[Atoms],
    nff_cutoff: float,
    device: str = "cpu",
    structures_per_batch: int = 32,
    **kwargs,
) -> list[AtomsBatch]:
    """Generate AtomsBatch

    Args:
        data (Union[Dataset, list[ase.Atoms]]): Dictionary or list of ase Atoms containing the
            properties of the atoms
        nff_cutoff (float): Neighbor cutoff for the NFF model
        device (str, optional): cpu or cuda device. Defaults to 'cpu'.
        structures_per_batch (int, optional): Number of structures per batch. Defaults to 32.
        **kwargs: Additional keyword arguments.

    Returns:
        list[AtomsBatch]: List of AtomsBatch objects.
    """
    print(f"Data has length {len(data)}")

    if isinstance(data, Dataset):
        atoms_batches = data.as_atoms_batches()
    else:
        atoms_batches = []
        # TODO: select structures_per_batch structures at a time
        for atoms in tqdm(data):
            atoms_batch = get_atoms_batch(atoms, nff_cutoff, device, **kwargs)
            atoms_batches.append(atoms_batch)

    return atoms_batches


def load_dataset_from_files(file_paths: list[Path | str]) -> list[Atoms]:
    """Load dataset from files. Dataset can be a list of ASE Atoms objects, an NFF Dataset
    or a list of file paths.

    Args:
        file_paths (list[Path]): List of file paths.

    Returns:
        list[Atoms]: List of ASE Atoms objects.
    """
    file_paths = [Path(file_name) for file_name in file_paths]

    dset = []
    for x in file_paths:
        if x.suffix == ".txt":
            # load file paths from a text file
            with open(x, "r", encoding="utf-8") as f:
                lines = f.readlines()
                return load_dataset_from_files([Path(line.strip()) for line in lines])
        elif x.suffix == ".pkl":
            with open(x, "rb") as f:
                dset.extend(pkl.load(f))
        else:  # .pth.tar
            data = Dataset.from_file(x)
            dset.extend(data)
    return dset


def filter_distances(slab: Atoms, ads: Iterable = ("O"), cutoff_distance: float = 1.5) -> bool:
    """This function filters out slabs that have atoms too close to each other based on a
    specified cutoff distance.

    Args:
        slab (Atoms): The slab structure to check for distances.
        ads (Iterable, optional): The adsorbate atom types in the slab to check for. Defaults to
            ("O").
        cutoff_distance (float, optional): The cutoff distance to check for. Defaults to 1.5.

    Returns:
        bool: True if the distances are greater than the cutoff distance, False otherwise.
    """
    # Checks distances of all adsorbates are greater than cutoff
    ads_arr = np.isin(slab.get_chemical_symbols(), ads)
    unique_dists = np.unique(np.triu(slab.get_all_distances(mic=True)[ads_arr][:, ads_arr]))
    # Get upper triangular matrix of ads dist
    return not any(unique_dists[(unique_dists > 0) & (unique_dists <= cutoff_distance)])


def randomize_structure(atoms, amplitude, displace_lattice=True) -> Atoms:
    """Randomly displaces the atomic coordinates (and lattice parameters)
    by a certain amplitude. Useful to generate slightly off-equilibrium
    configurations and starting points for MD simulations. The random
    amplitude is sampled from a uniform distribution.

    Same function as in pymatgen, but for ase.Atoms objects.

    Args:
        atoms (ase.Atoms): The input structure.
        amplitude (float): Max value of amplitude displacement in Angstroms.
        displace_lattice (bool): Whether to displace the lattice.

    Returns:
        ase.Atoms: The perturbed structure.
    """
    newcoords = atoms.get_positions() + np.random.uniform(
        -amplitude, amplitude, size=atoms.positions.shape
    )

    newlattice = np.array(atoms.get_cell())
    if displace_lattice:
        newlattice += np.random.uniform(-amplitude, amplitude, size=newlattice.shape)

    return Atoms(
        positions=newcoords,
        numbers=atoms.numbers,
        cell=newlattice,
        pbc=atoms.pbc,
    )


def compute_distance_weight_matrix(
    ads_coords: np.ndarray, distance_decay_factor: float
) -> np.ndarray:
    """Compute distance weight matrix using softmax.

    Args:
        ads_coords (np.ndarray): The coordinates of the adsorption sites.
        distance_decay_factor (float): Exponential decay factor.

    Returns:
        np.ndarray: The distance weight matrix.
    """
    # Compute pairwise distance matrix
    ads_coord_distances = distance.cdist(ads_coords, ads_coords, "euclidean")

    # Compute distance decay matrix using softmax
    distance_weight_matrix = softmax(-ads_coord_distances / distance_decay_factor, axis=1)

    assert np.allclose(np.sum(distance_weight_matrix, axis=1), 1.0)

    return distance_weight_matrix
