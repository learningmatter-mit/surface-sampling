import argparse
import datetime
import json
import pickle as pkl
from pathlib import Path
from typing import Dict, List, Tuple, Union

import ase
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter structures to select only those with certain range for each type of atom."
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

    return parser.parse_args()


def plot_histograms(
    all_stoic_dicts: List[Dict[str, int]],
    atom_types: List[str],
    fig_name: str = "starting_stoic_hist",
):
    """Plot histogram of each atom type and the difference in number of Sr and Ir atoms.

    Parameters
    ----------
    all_stoic_dicts : List[Dict[str, int]]
        list of stoichiometry dictionaries for each structure.
    atom_types : List[str]
        list of atom types to consider.
    fig_name : str, optional
        save name for figure, by default "starting_stoic_hist"
    """

    delta_Sr_Ir = [
        d["Sr"] - d["Ir"] for d in all_stoic_dicts
    ]  # difference in number of Sr and Ir atoms

    n_atoms = {atom: [d[atom] for d in all_stoic_dicts] for atom in atom_types}

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), dpi=200)

    sns.histplot(delta_Sr_Ir, ax=ax[0], discrete=True, label="#Sr - #Ir")
    for atom in atom_types:
        sns.histplot(n_atoms[atom], ax=ax[1], discrete=True, label=f"#{atom}")
    ax[0].legend()
    ax[1].legend()

    plt.savefig(f"{fig_name}.png")


def main(
    file_paths: List[str],
    atom_types: List[str],
    atom_ranges: Dict[str, Tuple[int, int]],
    save_folder: Union[Path, str] = "./",
):
    """Filter structures to select only those with certain range for each type of atom.

    Parameters
    ----------
    file_paths : List[str]
        list of file paths to load structures from.
    atom_types : List[str]
        atom types to consider.
    atom_ranges : Dict[str, Tuple[int, int]]
        dictionary with the range for each type of atom allowed.
    save_folder : Union[Path, str], optional
        folder to save filtered structures, by default "./"
    """
    start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)

    all_structures = []
    for full_path in file_paths:
        with open(full_path, "rb") as f:
            try:
                data = pkl.load(f)
                all_structures.extend(data)
            except EOFError:
                print(f"Could not load {full_path}")

    print(f"Total number of structures before filtering: {len(all_structures)}")

    # generate a stoichiometry dictionary for each structure
    all_stoic_dicts = [
        ase.formula.Formula(s.get_chemical_formula()).count() for s in all_structures
    ]

    # before filtering
    plot_histograms(
        all_stoic_dicts,
        atom_types,
        fig_name=save_path / f"{start_time}_starting_stoic_hist",
    )

    # select only structures with certain range for each type of atom
    filtered_structures = [
        s
        for s, d in zip(all_structures, all_stoic_dicts)
        if all(
            [
                atom_ranges[atom][0] <= d[atom] <= atom_ranges[atom][1]
                for atom in atom_types
            ]
        )
    ]

    print(f"Number of structures after filtering: {len(filtered_structures)}")

    # save filtered structures
    with open(
        save_path
        / f"{start_time}_total_{len(filtered_structures)}_{','.join(atom_types)}_filtered_structures.pkl",
        "wb",
    ) as f:
        pkl.dump(filtered_structures, f)


if __name__ == "__main__":
    args = parse_args()
    main(
        args.file_paths, args.atom_types, args.atom_ranges, save_folder=args.save_folder
    )