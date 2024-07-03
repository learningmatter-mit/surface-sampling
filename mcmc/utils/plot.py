"""Plotting utilities for MCMC and clustering."""

from collections.abc import Iterable
from pathlib import Path

import ase
import matplotlib.pyplot as plt
import numpy as np
from ase.visualize.plot import plot_atoms
from matplotlib.figure import Figure
from scipy.special import softmax

DEFAULT_DPI = 200


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
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), dpi=DEFAULT_DPI)
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
    fig, ax = plt.subplots(1, 2, figsize=(10, 10), dpi=DEFAULT_DPI)
    labels = ["slab1", "slab2"]
    for i in range(2):
        ax[i].axis("off")
        ax[i].set_title(labels[i])
    plot_atoms(slab1, ax[0], radii=0.8, rotation=("90x, 15y, 90z"))
    plot_atoms(slab2, ax[1], radii=0.8, rotation=("90x, 15y, 90z"))

    plt.tight_layout()
    plt.savefig(Path(save_folder) / "slab_comparison.png")
    return fig


def plot_anneal_schedule(
    schedule: list,
    save_folder: Path | str = ".",
) -> Figure:
    """Plot the annealing schedule.

    Args:
        schedule (list): List of temperatures for each MC sweep.
        save_folder (Union[Path, str], optional): Folder to save the output in. Defaults to ".".

    Returns:
        Figure: Matplotlib figure object.
    """
    save_path = Path(save_folder)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=DEFAULT_DPI)
    ax.plot(schedule)
    ax.set_xlabel("Sweep number", fontsize=14)
    ax.set_ylabel("Temperature (kB T)", fontsize=14)
    ax.set_title("Annealing schedule", fontsize=16)
    plt.savefig(save_path / "anneal_schedule.png")
    return fig


def plot_clustering_results(
    points: np.ndarray,
    n_clusters: int,
    labels: np.ndarray,
    closest_points_indices: np.ndarray,
    save_folder: Path | str = ".",
) -> Figure:
    """Plot the clustering results.

    Args:
        points (np.ndarray): 2D numpy array of shape (n_points, 2) containing the points to cluster.
        n_clusters (int): Total number of calculated clusters.
        labels (np.ndarray): Numpy array of shape (n_points,) with the cluster number for each
            point.
        closest_points_indices (np.ndarray): Numpy array of shape (n_clusters,) with the index of
            the closest point to the centroid in each cluster.
        save_folder (str, optional): Folder to save the plot in. Defaults to ".".

    Returns:
        Figure: The figure object.
    """
    # Define colors
    colors = ["b", "g", "r", "c", "m", "y", "k"]

    # Create a larger plot
    fig, ax = plt.subplots(figsize=(10, 7), dpi=DEFAULT_DPI)

    # Create a scatter plot of all points, color-coded by cluster
    for i in range(1, n_clusters + 1):
        cluster_points = points[labels == i]
        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            color=colors[i % len(colors)],
            alpha=0.6,
            edgecolor="black",
            linewidth=1,
            s=100,
            label=f"Cluster {i}",
        )

    # Mark the closest points to the centroid in each cluster
    for i in range(n_clusters):
        closest_point = points[closest_points_indices[i]]
        ax.scatter(
            closest_point[0],
            closest_point[1],
            marker="*",
            color="black",
            edgecolor="black",
            linewidth=1,
            s=200,
        )

    ax.grid(True)
    ax.set_xlabel("Dimension 1", fontsize=14)
    ax.set_ylabel("Dimension 2", fontsize=14)
    ax.set_title("2D representation of points and clusters", fontsize=16)
    ax.legend(fontsize=12)
    plt.savefig(f"{save_folder}/clustering_results.png")

    return fig


def plot_distance_weight_matrix(
    distance_weight_matrix: np.ndarray, save_folder: str = "."
) -> Figure:
    """Plot distance weight matrix.

    Args:
        distance_weight_matrix (np.ndarray): Distance weight matrix.
        save_folder (str, optional): Folder to save the plot in. Defaults to ".".

    Returns:
        Figure: The figure object.
    """ ""
    # Define colors
    # colors = ["b", "g", "r", "c", "m", "y", "k"]

    # Create a larger plot
    fig, ax = plt.subplots(figsize=(10, 7), dpi=DEFAULT_DPI)

    # Display the distance weight matrix as an image
    img = ax.imshow(distance_weight_matrix, cmap="hot", interpolation="nearest")

    # Add a colorbar to the figure to show how colors correspond to values
    plt.colorbar(img, ax=ax)

    ax.set_xlabel("Dimension 1", fontsize=14)
    ax.set_xlabel("Dimension 2", fontsize=14)
    ax.set_title("Distance Weight Matrix")
    plt.savefig(f"{save_folder}/distance_weight_matrix.png")
    return fig


def plot_decay_curve(decay_factor: float, save_folder: str = ".") -> Figure:
    """Plot distance decay curve.

    Args:
        decay_factor (float): Exponential decay factor.
        save_folder (str, optional): Folder to save the plot in. Defaults to ".".

    Returns:
        Figure: The figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 7), dpi=DEFAULT_DPI)
    distances = np.linspace(0, 3 * decay_factor, 100)
    factor = softmax(-distances / decay_factor)
    ax.plot(distances, factor, color="blue", label="Decay Factor")

    ax.set_xlabel("Distance [Å]", fontsize=14)
    ax.set_ylabel("Probability density", fontsize=14)
    ax.set_title("Distance Decay Plot")
    ax.legend(fontsize=12)
    plt.savefig(f"{save_folder}/distance_weight_decay.png")
    return fig


def plot_specific_weights(
    coords: np.ndarray,
    weights: list | np.ndarray,
    site_idx: int,
    save_folder: Path | str = ".",
    run_iter: int = 0,
) -> Figure:
    """Plot weights of the adsorption sites on the lattice.

    Args:
        coords (np.ndarray): The coordinates of the adsorption sites.
        weights (Union[list, np.ndarray]): The weights of the adsorption sites.
        site_idx (int): The index of the site to plot.
        save_folder (Union[Path, str], optional): Folder to save the plot in. Defaults to ".".
        run_iter (int, optional): The iteration number. Defaults to 0.

    Returns:
        Figure: The figure object.
    """
    # Create a larger plot
    fig, ax = plt.subplots(figsize=(10, 7), dpi=DEFAULT_DPI)
    curr_site = coords[site_idx]

    # Create a scatter plot of all points, color-coded by weights
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=weights,
        alpha=0.6,
        edgecolor="black",
        linewidth=1,
        s=100,
    )
    ax.scatter(
        curr_site[0],
        curr_site[1],
        marker="*",
        color="black",
        edgecolor="black",
        linewidth=1,
        s=200,
    )
    # Add a colorbar to the figure to show how colors correspond to values
    plt.colorbar(scatter, ax=ax)

    ax.grid(True)
    ax.set_xlabel("Dimension 1", fontsize=14)
    ax.set_ylabel("Dimension 2", fontsize=14)
    ax.set_title("2D representation of adsorption sites color-coded by weights", fontsize=16)
    plt.savefig(f"{save_folder}/specific_weights_on_lattice_iter_{run_iter:06}.png")
    return fig
