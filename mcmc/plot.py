from collections.abc import Iterable
from pathlib import Path

import ase
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms
from matplotlib.figure import Figure


def plot_summary_stats(
    energy_hist: Iterable,
    frac_accept_hist: Iterable,
    adsorption_count_hist: Iterable,
    num_sweeps: int,
    save_folder: str = ".",
) -> Figure:
    """Plot summary statistics of MCMC run.

    Args:
        energy_hist (Iterable): Energy history
        frac_accept_hist (Iterable): Fraction accepted history
        adsorption_count_hist (Iterable): Adsorption count history
        num_sweeps (int): Number of sweeps
        save_folder (str, optional): Folder to save the plot. Defaults to ".".

    Returns:
        Figure: The figure object.
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), dpi=200)
    runs = range(1, num_sweeps + 1)

    ax[0].plot(runs, energy_hist)
    ax[0].set_xlabel("Sweep #")
    ax[0].set_ylabel("Energy (eV)")
    ax[0].set_title("Energy (eV) vs Sweeps")

    ax[1].plot(runs, frac_accept_hist)
    ax[1].set_xlabel("Sweep #")
    ax[1].set_ylabel("Fraction accepted")
    ax[1].set_title("Fraction accepted vs Sweeps")

    ax[2].plot(runs, adsorption_count_hist)
    ax[2].set_xlabel("Sweep #")
    ax[2].set_ylabel("Adsorption count")
    ax[2].set_title("Adsorption count vs Sweeps")

    plt.tight_layout()
    plt.savefig(Path(save_folder) / "summary.png")
    return fig


def visualize_two_slabs(
    slab1: ase.Atoms,
    slab2: ase.Atoms,
    save_folder: str = ".",
) -> Figure:
    """Visualize two slabs side by side.

    Args:
        slab1 (ase.Atoms): First slab
        slab2 (ase.Atoms): Second slab
        save_folder (str, optional): Folder to save the plot. Defaults to ".".

    Returns:
        Figure: The figure object.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    labels = ["slab1", "slab2"]
    for i in range(2):
        ax[i].axis("off")
        ax[i].set_title(labels[i])
    plot_atoms(slab1, ax[0], radii=0.8, rotation=("90x, 15y, 90z"))
    plot_atoms(slab2, ax[1], radii=0.8, rotation=("90x, 15y, 90z"))

    plt.tight_layout()
    plt.savefig(Path(save_folder) / "slab_comparison.png")
    return fig
