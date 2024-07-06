"""Cut surfaces from bulk structures."""

import argparse
import datetime
import pickle as pkl
from collections.abc import Iterable
from logging import getLevelNamesMapping
from pathlib import Path
from typing import Literal

import ase
import numpy as np
from catkit.gen.surface import SlabGenerator
from tqdm import tqdm

from mcmc.utils import setup_logger
from mcmc.utils.misc import load_dataset_from_files
from mcmc.utils.plot import plot_surfaces


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
    parser.add_argument(
        "--logging_level",
        type=str,
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Logging level",
    )

    return parser.parse_args()


# TODO move to surface.slab
def surface_from_bulk(
    bulk: ase.Atoms,
    miller_index: Iterable[int],
    layers: int = 5,
    fixed: int = 6,
    size: Iterable = [1, 1],
    vacuum: float = 7.5,
    iterm: int = 0,
) -> tuple[ase.Atoms, list[bool]]:
    """Cut a surface from a bulk structure.

    Args:
        bulk (ase.Atoms): Bulk structure to cut surface from.
        miller_index (Iterable[int]): Miller indices for the surface, length 3.
        layers (int, optional): Number of layers in the slab, by default 5
        fixed (int, optional): Number of fixed layers, by default 6
        size (Iterable, optional): Size of the slab with respect to provided bulk, by default [1, 1]
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


def main(
    file_names: list[str],
    hkl: Iterable[int] = (0, 0, 1),
    layers: int = 6,
    fixed: int = 4,
    size: Iterable[int] = (1, 1),
    vacuum: int = 10,
    save_folder: Path | str = "./",
    logging_level: Literal["debug", "info", "warning", "error", "critical"] = "info",
):
    """Cut surfaces from provided bulk structures.

    Args:
        file_names (list[str]): list of file names to load structures from.
        hkl (Iterable[int], optional): Miller indices for the surface, by default [0, 0, 1]
        layers (int, optional): Number of layers in the slab, by default 6
        fixed (int, optional): Number of fixed layers, by default 4
        size (Iterable[int], optional): Size of the slab with respect to provided bulk,
            by default [1, 1]
        vacuum (int, optional): Vacuum space in Angstroms (in each direction), by default 10
        save_folder (Path | str, optional): Folder to save cut surfaces, by default "./"
        logging_level (Literal["debug", "info", "warning", "error", "critical"], optional):
            Logging level, by default "info"
    """
    start_timestamp = datetime.now().isoformat(sep="-", timespec="milliseconds")

    # Initialize save folder
    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)
    file_base = f"{start_timestamp}_cut_surfaces"

    # Initialize logger
    logger = setup_logger(
        "cut_surfaces",
        save_path / "cut_surfaces.log",
        level=getLevelNamesMapping()[logging_level.upper()],
    )

    logger.info("There are a total of %d input files", len(file_names))
    all_structures = load_dataset_from_files(file_names)
    logger.info("Loaded %d structures", {len(all_structures)})

    all_slabs = []
    # Cut surfaces from all structures
    for bulk in tqdm(all_structures):
        slab, _ = surface_from_bulk(bulk, hkl, layers=layers, fixed=fixed, size=size, vacuum=vacuum)
        all_slabs.append(slab)

    # Plot 10 sampled surfaces
    sampled_slabs = np.random.choice(len(all_slabs), 10, replace=False)
    logger.info("Sampling surfaces at indices: %s", sampled_slabs)
    plot_surfaces(
        [all_slabs[x] for x in sampled_slabs],
        fig_name=file_base,
        save_folder=save_path,
    )

    # Save cut surfaces
    save_surface_path = (
        save_path / f"{file_base}_total_{len(all_slabs)}_hkl_{hkl}_layers_{layers}.pkl"
    )
    with open(
        save_surface_path,
        "wb",
    ) as f:
        pkl.dump(all_slabs, f)

    logger.info("Surface cuts complete. Saved to %s", save_surface_path)


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
        logging_level=args.logging_level,
    )
