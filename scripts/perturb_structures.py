import argparse
import datetime
import pickle as pkl
from pathlib import Path
from typing import List, Union

import ase
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.visualize.plot import plot_atoms
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Perturb structures and save them to a new file."
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

    return parser.parse_args()


def plot_structures(
    starting_structures: List[ase.Atoms],
    perturbed_structures: List[ase.Atoms],
    fig_name: str = "structures",
):
    """Plot the structures before and after perturbation.

    Parameters
    ----------
    starting_structures : List[ase.Atoms]
        List of starting structures.
    perturbed_structures : List[ase.Atoms]
        List of perturbed structures.
    fig_name : str, optional
        save name for figure, by default "structures"
    """

    fig, axes = plt.subplots(2, len(starting_structures), figsize=(8, 8), dpi=200)
    for ax, atoms in zip(axes.ravel(), starting_structures + perturbed_structures):
        ax.axis("off")
        composition = atoms.get_chemical_formula()
        ax.set_title(composition)
        plot_atoms(atoms, ax, radii=0.8, rotation=("-75x, 45y, 10z"))
    plt.tight_layout()
    plt.savefig(f"{fig_name}.png")


def randomize_structure(atoms, amplitude, displace_lattice=True):
    """Randomly displaces the atomic coordinates (and lattice parameters)
        by a certain amplitude. Useful to generate slightly off-equilibrium
        configurations and starting points for MD simulations. The random
        amplitude is sampled from a uniform distribution.

        Same function as in pymatgen, but for ase.Atoms objects.

    Parameters
    ----------
    atoms : ase.Atoms
        The input structure.
    amplitude : float
        Max value of amplitude displacement in Angstroms.
    displace_lattice : bool
        Whether to displace the lattice.

    Returns
    -------
    ase.Atoms
        The perturbed structure.
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
    file_paths: List[str],
    amplitude: float = 0.3,
    displace_lattice: bool = True,
    save_folder: Union[Path, str] = "./",
):
    """Perturb structures and save them to a new file.

    Parameters
    ----------
    file_paths : List[str]
        List of file paths to load structures from.
    amplitude : float, optional
        Max value of amplitude displacement in Angstroms, by default 0.3
    displace_lattice : bool, optional
        Whether to displace the lattice, by default True
    save_folder : Union[Path, str], optional
        Folder to save cut surfaces, by default "./"
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

    print(f"Total number of structures: {len(all_structures)}")

    perturbed_structures = []
    for atoms in tqdm(all_structures):
        perturbed_atoms = randomize_structure(
            atoms, amplitude, displace_lattice=displace_lattice
        )
        perturbed_structures.append(perturbed_atoms)

    # plot 5 sampled structures
    sampled_idx = np.random.choice(len(all_structures), 5, replace=False)
    print(f"Sampling structures at indices: {sampled_idx}")
    plot_structures(
        [all_structures[x] for x in sampled_idx],
        [perturbed_structures[x] for x in sampled_idx],
        fig_name=save_path / f"{start_time}_structures",
    )

    # save perturbed structures
    with open(
        save_path
        / f"{start_time}_total_{len(perturbed_structures)}_perturbed_amp_{amplitude}_structures.pkl",
        "wb",
    ) as f:
        pkl.dump(perturbed_structures, f)

    print(
        f"Structure perturbation complete. Saved to {start_time}_perturbed_structures.pkl"
    )


if __name__ == "__main__":
    args = parse_args()
    main(
        args.file_paths,
        args.amplitude,
        args.displace_lattice,
        save_folder=args.save_folder,
    )
