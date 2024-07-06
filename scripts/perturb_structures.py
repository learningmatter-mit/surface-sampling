"""Perturb structure atomic positions and lattice and save them to a new file."""

import argparse
import datetime
import pickle as pkl
from logging import getLevelNamesMapping
from pathlib import Path
from typing import Literal

import numpy as np
from ase import Atoms
from tqdm import tqdm

from mcmc.utils import setup_logger
from mcmc.utils.misc import load_dataset_from_files
from mcmc.utils.plot import plot_surfaces


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Perturb structures and save them to a new file.")
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
        help="Folder to save perturbed structures.",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.3,
        help="Max value of amplitude displacement in Angstroms.",
    )
    parser.add_argument(
        "--displace_lattice",
        action="store_true",
        help="Whether to displace the lattice.",
    )
    parser.add_argument(
        "--logging_level",
        type=str,
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Logging level",
    )

    return parser.parse_args()


# def plot_structures(
#     starting_structures: List[ase.Atoms],
#     perturbed_structures: List[ase.Atoms],
#     fig_name: str = "structures",
# ):
#     """Plot the structures before and after perturbation.

#     Parameters
#     ----------
#     starting_structures : List[ase.Atoms]
#         List of starting structures.
#     perturbed_structures : List[ase.Atoms]
#         List of perturbed structures.
#     fig_name : str, optional
#         save name for figure, by default "structures"
#     """

#     fig, axes = plt.subplots(2, len(starting_structures), figsize=(8, 8), dpi=200)
#     for ax, atoms in zip(axes.ravel(), starting_structures + perturbed_structures):
#         ax.axis("off")
#         composition = atoms.get_chemical_formula()
#         ax.set_title(composition)
#         plot_atoms(atoms, ax, radii=0.8, rotation=("-75x, 45y, 10z"))
#     plt.tight_layout()
#     plt.savefig(f"{fig_name}.png")


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


def main(
    file_names: list[str],
    amplitude: float = 0.3,
    displace_lattice: bool = True,
    save_folder: Path | str = "./",
    logging_level: Literal["debug", "info", "warning", "error", "critical"] = "info",
) -> None:
    """Perturb structures and save them to a new file.

    Args:
        file_names (List[str]): List of file paths to load structures from.
        amplitude (float, optional): Max value of amplitude displacement in Angstroms.
            Defaults to 0.3.
        displace_lattice (bool, optional): Whether to displace the lattice.
            Defaults to True.
        save_folder (Union[Path, str], optional): Folder to save perturbed structures.
            Defaults to "./".
        logging_level (Literal["debug", "info", "warning", "error", "critical"], optional):
            Logging level. Defaults to "info".
    """
    start_timestamp = datetime.now().isoformat(sep="-", timespec="milliseconds")

    # Initialize save folder
    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)
    file_base = f"{start_timestamp}_perturbed_structures"

    # Initialize logger
    logger = setup_logger(
        "perturb_structures",
        save_path / "perturb_structures.log",
        level=getLevelNamesMapping()[logging_level.upper()],
    )

    logger.info("There are a total of %d input files", len(file_names))
    all_structures = load_dataset_from_files(file_names)
    logger.info("Loaded %d structures", {len(all_structures)})

    perturbed_structures = []
    # Perturb all structures
    for atoms in tqdm(all_structures):
        perturbed_atoms = randomize_structure(atoms, amplitude, displace_lattice=displace_lattice)
        perturbed_structures.append(perturbed_atoms)

    # Plot 5 sampled structures, before and after perturbation
    sampled_idx = np.random.choice(len(all_structures), 5, replace=False)
    logger.info("Sampling before and after structures at indices: %s", sampled_idx)
    plot_surfaces(
        [all_structures[x] for x in sampled_idx] + [perturbed_structures[x] for x in sampled_idx],
        fig_name=file_base,
        save_folder=save_path,
    )

    # Save perturbed structures
    perturbed_surface_path = (
        save_path / f"{file_base}_total_{len(perturbed_structures)}_amp_{amplitude}_structures.pkl"
    )
    with open(
        perturbed_surface_path,
        "wb",
    ) as f:
        pkl.dump(perturbed_structures, f)

    logger.info("Structure perturbation complete. Saved to %s", perturbed_surface_path)


if __name__ == "__main__":
    args = parse_args()
    main(
        args.file_paths,
        args.amplitude,
        args.displace_lattice,
        save_folder=args.save_folder,
        logging_level=args.logging_level,
    )
