"""Filter structures to select only those with a certain stoichiometric range."""

import argparse
import json
import pickle as pkl
from datetime import datetime
from logging import getLevelNamesMapping
from pathlib import Path
from typing import Literal

import ase

from mcmc.utils import setup_logger
from mcmc.utils.misc import load_dataset_from_files
from mcmc.utils.plot import plot_atom_type_histograms


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Filter structures to select only those with certain range for each type of atom."
        )
    )
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
    parser.add_argument("--atom_types", nargs="+", type=str, default=["Sr", "Ir", "O"])
    parser.add_argument(
        "--atom_ranges",
        type=json.loads,
        default={"Sr": [6, 10], "Ir": [6, 10], "O": [18, 30]},
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
    atom_types: list[str],
    atom_ranges: dict[str, tuple[int, int]],
    save_folder: Path | str = "./",
    logging_level: Literal["debug", "info", "warning", "error", "critical"] = "info",
) -> None:
    """Filter structures to select only those with certain range for each type of atom.

    Args:
        file_names (list[str]): list of file paths to load structures from.
        atom_types (list[str]): atom types to consider.
        atom_ranges (dict[str, tuple[int, int]]): dictionary with the range for each type of
            atom allowed.
        save_folder (Path | str, optional): folder to save filtered structures, by default "./"
        logging_level (Literal["debug", "info", "warning", "error", "critical"], optional):
            logging level, by default "info"
    """
    start_timestamp = datetime.now().isoformat(sep="-", timespec="milliseconds")

    # Initialize save folder
    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)
    file_base = f"{start_timestamp}_filter_stoics"

    # Initialize logger
    logger = setup_logger(
        "filter_stoics",
        save_path / "filter_stoics.log",
        level=getLevelNamesMapping()[logging_level.upper()],
    )

    logger.info("There are a total of %d input files", len(file_names))
    all_structures = load_dataset_from_files(file_names)
    # If all_structures are SurfaceSystems, take the relaxed_atoms
    all_structures = [s.relaxed_atoms for s in all_structures if hasattr(s, "relaxed_atoms")]
    logger.info("Loaded %d structures", len(all_structures))

    # Generate a stoichiometry dictionary for each structure
    all_stoic_dicts = [
        ase.formula.Formula(s.get_chemical_formula()).count() for s in all_structures
    ]

    # Before filtering
    plot_atom_type_histograms(
        all_stoic_dicts,
        atom_types,
        fig_name=f"{start_timestamp}_starting_stoic_hist",
        save_folder=save_path,
    )

    # Select only structures with certain range for each type of atom
    filtered_structures = []
    for s, d in zip(all_structures, all_stoic_dicts, strict=False):
        if all(
            atom_ranges[atom][0] <= d.get(atom, 0) <= atom_ranges[atom][1] for atom in atom_types
        ):
            filtered_structures.append(s)

    logger.info("Number of structures after filtering: %d", len(filtered_structures))

    # Save filtered structures
    save_surface_path = save_path / (
        f"{file_base}_total_{len(filtered_structures)}_{','.join(atom_types)}_"
        "filtered_structures.pkl"
    )
    with open(
        save_surface_path,
        "wb",
    ) as f:
        pkl.dump(filtered_structures, f)

    logger.info("Filtering structures complete. Saved to %s", save_surface_path)


if __name__ == "__main__":
    args = parse_args()
    main(
        args.file_paths,
        args.atom_types,
        args.atom_ranges,
        save_folder=args.save_folder,
        logging_level=args.logging_level,
    )
