import argparse
import datetime
import pickle as pkl
from pathlib import Path
from typing import Iterable, List, Union

import ase
import matplotlib.pyplot as plt
import numpy as np
from ase.visualize.plot import plot_atoms
from catkit.gen.surface import SlabGenerator
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Cut surfaces from structures.")
    parser.add_argument(
        "--file_paths",
        nargs="+",
        help="Full paths to NFF Dataset or ASE Atoms/NFF AtomsBatch",
        type=Path,
    )
    parser.add_argument(
        "--save_folder",
        type=Path,
        default="./",
        help="Folder to save cut surfaces.",
    )
    parser.add_argument(
        "--hkl",
        nargs="+",
        type=int,
        default=[0, 0, 1],
        help="Miller indices for the surface.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=6,
        help="Number of layers in the slab.",
    )
    parser.add_argument(
        "--fixed",
        type=int,
        default=4,
        help="Number of fixed layers.",
    )
    parser.add_argument(
        "--size",
        nargs="+",
        type=int,
        default=[1, 1],
        help="Size of the slab with respect to provided bulk.",
    )
    parser.add_argument(
        "--vacuum",
        type=int,
        default=10,
        help="Vacuum space in Angstroms (in each direction).",
    )

    return parser.parse_args()


def plot_surfaces(
    surfaces: List[ase.Atoms],
    fig_name: str = "cut_surfaces",
):
    """Plot the structures before and after perturbation.

    Parameters
    ----------
    surfaces : List[ase.Atoms]
        List of cut structures.
    fig_name : str, optional
        save name for figure, by default "cut_surfaces"
    """

    # plot 2 rows of surfaces
    fig, axes = plt.subplots(2, len(surfaces) // 2, figsize=(8, 8), dpi=200)
    for ax, atoms in zip(axes.ravel(), surfaces):
        ax.axis("off")
        composition = atoms.get_chemical_formula()
        ax.set_title(composition)
        plot_atoms(atoms, ax, radii=0.8, rotation=("-75x, 45y, 10z"))
    plt.tight_layout()
    plt.savefig(f"{fig_name}.png")


def surface_from_bulk(
    bulk: ase.Atoms,
    miller_index: Iterable[int],
    layers: int = 5,
    fixed: int = 6,
    size: Iterable = [1, 1],
    vacuum: float = 7.5,
    iterm: int = 0,
):
    """Cut a surface from a bulk structure.

    Parameters:
    ----------
    bulk : ase.Atoms
        Bulk structure to cut surface from.
    miller_index : Iterable[int]
        Miller indices for the surface, length 3.
    layers : int, optional
        Number of layers in the slab, by default 5
    fixed : int, optional
        Number of fixed layers, by default 6
    size : Iterable, optional
        Size of the slab with respect to provided bulk, by default [1, 1]
    vacuum : float, optional
        Vacuum space in Angstroms (in each direction), by default 7.5
    iterm : int, optional
        Index of the slab termination, by default 0

    Returns:
    ----------
    slab : ase.Atoms
        Surface cut from the crystal.
    surface_atoms : List[bool]
        List of surface atoms.
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

    slab.center(vacuum=vacuum, axis=2)

    surface_atom_index = slab.get_surface_atoms().tolist()  # includes top surface atoms
    atom_list = np.arange(len(slab.numbers))
    z_values = slab.positions[:, 2]
    highest_z = np.max(z_values)
    surface_atoms = [highest_z - z_values[atom] < 1.2 for atom in atom_list]

    return slab, surface_atoms


def main(
    file_names: List[str],
    hkl: List[int] = [0, 0, 1],
    layers: int = 6,
    fixed: int = 4,
    size: List[int] = [1, 1],
    vacuum: int = 10,
    save_folder: Union[Path, str] = "./",
):
    """Cut surfaces from provided bulk structures.

    Parameters
    ----------
    file_names : List[str]
        List of file names to load structures from.
    hkl : List[int], optional
        Miller indices for the surface, by default [0, 0, 1]
    layers : int, optional
        Number of layers in the slab, by default 6
    fixed : int, optional
        Number of fixed layers, by default 4
    size : List[int], optional
        Size of the slab with respect to provided bulk, by default [1, 1]
    vacuum : int, optional
        Vacuum space in Angstroms (in each direction), by default 10
    save_folder : Union[Path, str], optional
        Folder to save cut surfaces, by default "./"

    """
    start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    save_path = Path(save_folder)
    save_path.mkdir(
        parents=True, exist_ok=True
    )  # create save folder if it doesn't exist

    # TODO create new function to load structures from file
    file_paths = [Path(file_name) for file_name in file_names]
    print(f"There are a total of {len(file_paths)} input files")

    all_structures = []
    for full_path in file_paths:
        with open(full_path, "rb") as f:
            try:
                data = pkl.load(f)
                all_structures.extend(data)
            except EOFError:
                print(f"Could not load {full_path}")

    print(f"Total number of structures: {len(all_structures)}")

    all_slabs = []
    # use Jackie's command for now
    for bulk in tqdm(all_structures):
        slab, _ = surface_from_bulk(
            bulk, hkl, layers=layers, fixed=fixed, size=size, vacuum=vacuum
        )
        all_slabs.append(slab)

    # plot 10 sampled surfaces
    sampled_slabs = np.random.choice(len(all_slabs), 10, replace=False)
    print(f"Sampling surfaces at indices: {sampled_slabs}")
    plot_surfaces(
        [all_slabs[x] for x in sampled_slabs],
        fig_name=save_path / f"{start_time}_cut_surfaces",
    )

    # save cut surfaces
    with open(
        save_path
        / f"{start_time}_total_{len(all_slabs)}_cut_surfaces_hkl_{hkl}_layers_{layers}.pkl",
        "wb",
    ) as f:
        pkl.dump(all_slabs, f)

    print(
        f"Surface cuts complete. Saved to {start_time}_total_{len(all_slabs)}_cut_surfaces_hkl_{hkl}_layers_{layers}.pkl"
    )


if __name__ == "__main__":
    args = parse_args()
    main(
        args.file_paths,
        hkl=args.hkl,
        layers=args.layers,
        fixed=args.fixed,
        size=args.size,
        vacuum=args.vacuum,
        save_folder=args.save_folder,
    )