import os

import matplotlib.pyplot as plt
import numpy as np
from ase.visualize.plot import plot_atoms


def plot_summary_stats(
    energy_hist, frac_accept_hist, adsorption_count_hist, num_sweeps, title
):
    runs = range(1, num_sweeps + 1)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(runs, energy_hist)
    ax[0].set_xlabel("Sweep #")
    ax[0].set_ylabel("Energy (eV)")
    ax[0].set_title("Energy (eV) vs Sweeps")

    ax[1].plot(runs, frac_accept_hist)
    ax[1].set_xlabel("Sweep #")
    ax[1].set_ylabel("Fraction accepted")
    ax[1].set_title("Fraction accepted vs Sweeps")

    ax[2].plot(runs, np.array(list(adsorption_count_hist.values())).T)
    ax[2].set_xlabel("Sweep #")
    ax[2].set_ylabel("Adsorption count")
    ax[2].legend(adsorption_count_hist.keys())
    ax[2].set_title("Adsorption count vs Sweeps")

    fig.tight_layout()
    fig.savefig(os.path.join(title, "summary.png"))


def visualize_two_slabs(slab1, slab2):
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    labels = ["slab1", "slab2"]
    for i in range(2):
        ax[i].axis("off")
        ax[i].set_title(labels[i])
    plot_atoms(slab1, ax[0], radii=0.8, rotation=("90x, 15y, 90z"))
    plot_atoms(slab2, ax[1], radii=0.8, rotation=("90x, 15y, 90z"))
