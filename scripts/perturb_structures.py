"""Perturb structure atomic positions and lattice and save them to a new file."""

import argparse
import pickle as pkl
from datetime import datetime
from logging import getLevelNamesMapping
from pathlib import Path
from typing import Literal

import numpy as np
from tqdm import tqdm

from mcmc.utils import setup_logger
from mcmc.utils.misc import load_dataset_from_files, randomize_structure
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
    logger.info("Loaded %d structures", len(all_structures))

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
