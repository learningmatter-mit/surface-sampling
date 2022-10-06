import os

import matplotlib.pyplot as plt
import numpy as np
from ase.visualize.plot import plot_atoms


def plot_summary_stats(
    energy_hist, frac_accept_hist, adsorption_count_hist, num_sweeps, title
):
    runs = range(1, num_sweeps + 1)

    # do the plots
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    ax[0, 0].plot(runs, energy_hist)
    ax[0, 0].set_xlabel("Iter")
    ax[0, 0].set_ylabel("Energy (E)")
    ax[0, 0].set_title("Energy (E) vs Sweeps")

    ax[0, 1].plot(runs, frac_accept_hist)
    ax[0, 1].set_xlabel("Iter")
    ax[0, 1].set_ylabel("Fraction accepted")
    ax[0, 1].set_title("Fraction accepted vs Sweeps")

    ax[1, 1].plot(runs, np.array(list(adsorption_count_hist.values())).T)
    ax[1, 1].set_xlabel("Iter")
    ax[1, 1].set_ylabel("Adsorption count")
    ax[1, 1].legend(adsorption_count_hist.keys())
    ax[1, 1].set_title("Adsorption count vs Iterations")

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
