import ase
import numpy as np
import torch
from ase import Atoms
from nff.data import Dataset, concatenate_dict
from nff.io.ase import AtomsBatch
from nff.utils.constants import EV_TO_KCAL_MOL, HARTREE_TO_KCAL_MOL

HARTREE_TO_EV = HARTREE_TO_KCAL_MOL / EV_TO_KCAL_MOL


def make_uncertainty_dataset(atoms_list: list[Atoms], cutoff) -> Dataset:
    new_dict_list = []
    for atoms in atoms_list:
        atoms_batch = AtomsBatch.from_atoms(
            atoms,
            cutoff=cutoff,
            requires_large_offsets=False,
            device="cpu",
            dense_nbrs=False,
            directed=True,
        )
        new_dict_list.append(atoms_batch.get_batch())
    new_dicts = concatenate_dict(*new_dict_list)
    dataset = Dataset(new_dicts)
    return dataset


def shrink_uncertainty_dataset(dataset: Dataset, indices) -> Dataset:
    new_dict_list = []
    new_dict_list = []
    for idx in indices:
        data = dataset[idx]
        new_dict_list.append(data)
    new_dicts = concatenate_dict(*new_dict_list)
    dataset = Dataset(new_dicts)
    return dataset


def make_clustering_dataset(
    atoms_list: list[Atoms], center_atom_index_list: list[int], cutoff: float
) -> Dataset:
    new_dict_list = []
    for i, atoms in enumerate(atoms_list):
        atoms_batch = AtomsBatch.from_atoms(
            atoms,
            cutoff=cutoff,
            requires_large_offsets=False,
            device="cpu",
            dense_nbrs=False,
            directed=True,
        )
        batch = atoms_batch.get_batch()
        # print(i, center_atom_index_list[i], type(center_atom_index_list[i]))
        batch["center_idx"] = torch.tensor(center_atom_index_list[i], dtype=torch.long)
        new_dict_list.append(batch)
    new_dicts = concatenate_dict(*new_dict_list)
    dataset = Dataset(new_dicts)
    return dataset


def preprocess_traj(
    total_candidates: list[ase.Atoms], z_cutoff: int | None = None, z_threshold: float | None = None
) -> list[ase.Atoms]:
    new_total_candidates = []
    for atoms in total_candidates:
        sorted_atoms = sorted(atoms, key=lambda atom: atom.position[2])
        sorted_atoms = ase.Atoms(sorted_atoms, pbc=atoms.pbc, cell=atoms.cell)
        new_pos = []
        if z_cutoff is not None:
            if z_threshold is None:
                z_threshold = 0.1
            z_coords_with_indices = sorted(
                (z, i) for i, z in enumerate(atoms.get_positions()[:, 2])
            )
            z_coords, indices = group_layers_with_indices(z_coords_with_indices, z_threshold)
            removing_indices = []
            for i in range(z_cutoff):
                removing_indices += indices[i]
            # print(np.array(z_coords[0]))
            shift_val = np.mean(np.array(z_coords[z_cutoff])) - np.mean(np.array(z_coords[0]))
            reduced_atoms = atoms.copy()

            del reduced_atoms[removing_indices]
            for pos in reduced_atoms.get_positions():
                new_pos.append(pos + np.array([0, 0, -shift_val]))
            reduced_atoms.set_positions(new_pos)
            sorted_atoms = reduced_atoms.copy()
        new_total_candidates.append(sorted_atoms)
    return new_total_candidates


def group_layers_with_indices(z_coords, threshold):
    layers = []
    indices = []
    current_layer = [z_coords[0][0]]
    current_indices = [z_coords[0][1]]

    for z, i in z_coords[1:]:
        if z - current_layer[-1] <= threshold:
            current_layer.append(z)
            current_indices.append(i)
        else:
            layers.append(current_layer)
            indices.append(current_indices)
            current_layer = [z]
            current_indices = [i]
    layers.append(current_layer)  # Append the last layer
    indices.append(current_indices)  # Append the last indices

    return layers, indices
