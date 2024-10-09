"""Plotting utilities for MCMC and clustering."""

from collections.abc import Iterable
from pathlib import Path

import ase
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ase.visualize.plot import plot_atoms
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import dendrogram
from scipy.special import softmax

from mcmc.utils import plot_settings

plt.style.use("default")

DEFAULT_DPI = 200

LINEWIDTH = 2
FONTSIZE = 10
LABELSIZE = 18
ALPHA = 0.8
MARKERSIZE = 15 * 25
GRIDSIZE = 40

MAJOR_TICKLEN = 6
MINOR_TICKLEN = 3
TICKPADDING = 5

SECONDARY_CMAP = "inferno"

# Define custom settings in a dictionary
custom_settings = {
    "mathtext.default": "regular",
    "font.family": ("Avenir", "Arial", "Helvetica", "sans-serif"),
    "font.size": FONTSIZE,
    "lines.linewidth": 2,
    "axes.labelsize": 18,
    "axes.linewidth": 2,
    "axes.titlesize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "xtick.major.pad": 5,
    "ytick.major.pad": 5,
    # "xtick.length": 6,
    "xtick.major.width": 2,
    "ytick.major.width": 2,
    "xtick.minor.width": 2,
    "ytick.minor.width": 2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.fontsize": 18,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.format": "png",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
}

# Update Matplotlib's rcParams with custom settings
plt.rcParams.update(custom_settings)


def plot_energy_analysis(
    surf_df: pd.DataFrame,
    energy_label: str,
    x_axis_label: str,
    phi: float,
    y_lim: tuple[float],
    x_ticks: Iterable[float],
    special_df: pd.DataFrame | None = None,
    special_scatter_labels: list[str] | None = None,
    x_text_rel_loc: float = 0.8,
    save_folder: str = ".",
) -> Figure:
    """Plot energy analysis of the MCMC run. Plot the Grand potential energy against
    "collective variable" (e.g. number of La atoms - number of Mn atoms).


    Args:
        surf_df (pd.DataFrame): DataFrame containing the surface energies possibly at different
            external conditions such as chemical potential (mu), electrical potential (phi), pH,
            etc.
        energy_label (str): Column label for the energy (y-axis).
        x_axis_label (str): Column label for the x-axis.
        phi (float): The value of the electrical potential phi. TODO update to be more general
        y_lim (tuple[float]): The y-axis limits (min, max).
        x_ticks (Iterable[float]): The x-axis ticks.
        special_df (pd.DataFrame, optional): DataFrame containing special entries to plot.
            Defaults to None.
        special_scatter_labels (list[str], optional): Labels for the special scatter points.
            Defaults to None.
        x_text_rel_loc (float, optional): Relative location of the text label. Defaults to 0.8.
        save_folder (str, optional): Folder to save the plot. Defaults to ".".

    Returns:
        Figure: The figure object.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=DEFAULT_DPI)

    # 1) Plot all data
    data_scatter = ax.scatter(
        surf_df[x_axis_label],
        surf_df[energy_label],
        c=plot_settings.colors[1 + 2],
        s=MARKERSIZE,
        alpha=ALPHA,
        marker="_",
        linewidths=LINEWIDTH,
    )

    # 2) Plot minimum surf
    min_surf = surf_df.iloc[surf_df[energy_label].idxmin()]
    min_line = ax.axhline(
        y=min_surf[energy_label],
        alpha=ALPHA,
        c=plot_settings.colors[1],
        ls="--",
        lw=LINEWIDTH,
    )
    y_pos = (min_surf[energy_label] - y_lim[0]) / (y_lim[1] - y_lim[0])
    ax.text(
        x_text_rel_loc,
        y_pos,
        rf"{min_surf[energy_label]:.1f} eV",
        transform=ax.transAxes,
        va="center",
        fontsize=LABELSIZE,
        backgroundcolor="w",
    )

    # 3) Plot special entries
    if special_df is not None:
        special_scatters = [
            ax.scatter(
                term[x_axis_label],
                term[energy_label],
                s=MARKERSIZE,
                alpha=ALPHA,
                marker="X",
                edgecolors="k",
                linewidth=LINEWIDTH,
            )
            for _, term in special_df.iterrows()
        ]
    else:
        special_scatters = []

    ax.set_title(rf"$\varphi={float(phi):.2f}$", fontsize=FONTSIZE, pad=3 * TICKPADDING)
    ax.set_xlabel(r"$\Gamma^{La}_{Mn}$ [# La - # Mn]", fontsize=FONTSIZE)
    ax.set_ylabel(r"$\Omega_{pbx}$ [eV]", fontsize=FONTSIZE)
    ax.set_xticks(x_ticks)
    ax.set_ylim(y_lim)

    # set the thickness of the spines
    for ax_loc in ["bottom", "top", "left", "right"]:
        ax.spines[ax_loc].set_linewidth(LINEWIDTH)

    # increase tick size and make them point inwards
    ax.tick_params(
        axis="y",
        length=MAJOR_TICKLEN,
        width=LINEWIDTH,
        labelsize=LABELSIZE,
        pad=TICKPADDING,
        direction="in",
    )
    ax.tick_params(
        axis="x",
        length=MAJOR_TICKLEN,
        width=LINEWIDTH,
        labelsize=LABELSIZE,
        pad=TICKPADDING,
        direction="in",
    )

    handles = [data_scatter, min_line, *special_scatters]
    legends = [
        "VS3R-MC sample",
        r"Min. $\Omega_{pbx}$",
    ]

    leg = ax.legend(handles, legends, loc="best", fontsize=LABELSIZE, frameon=True)
    leg.get_frame().set_edgecolor("k")
    leg.get_frame().set_linewidth(LINEWIDTH)
    leg.get_frame().set_boxstyle("Square", pad=0)

    plt.tight_layout()
    plt.savefig(Path(save_folder) / "summary.png")
    return fig


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


def plot_surfaces(
    surfaces: list[ase.Atoms],
    fig_name: str = "cut_surfaces",
    save_folder: str = ".",
) -> Figure:
    """Plot cut surfaces.

    Args:
        surfaces (list[ase.Atoms]): List of cut structures.
        fig_name (str, optional): Save name for figure. Defaults to "cut_surfaces".
        save_folder (str, optional): Folder to save the plot. Defaults to ".".

    Returns:
        Figure: matplotlib figure object.
    """
    # Plot 2 rows of surfaces
    fig, axes = plt.subplots(2, len(surfaces) // 2, figsize=(8, 8), dpi=DEFAULT_DPI)
    for ax, atoms in zip(axes.ravel(), surfaces, strict=False):
        ax.axis("off")
        composition = atoms.get_chemical_formula()
        ax.set_title(composition)
        plot_atoms(atoms, ax, radii=0.8, rotation=("-75x, 45y, 10z"))

    plt.tight_layout()
    plt.savefig(Path(save_folder) / f"{fig_name}.png")
    return fig


def plot_atom_type_histograms(
    all_stoic_dicts: list[dict[str, int]],
    atom_types: list[str],
    fig_name: str = "starting_stoic_hist",
    save_folder: str = ".",
) -> Figure:
    """Plot histogram of each atom type and the difference in number of Sr and Ir atoms.

    Args:
        all_stoic_dicts (list[dict[str, int]]): List of stoichiometry dictionaries for each
            structure.
        atom_types (list[str]): List of atom types to consider.
        fig_name (str, optional): Save name for figure. Defaults to "starting_stoic_hist".
        save_folder (str, optional): Folder to save the plot. Defaults to ".".

    Returns:
        Figure: matplotlib figure object.
    """
    type1 = atom_types[0]
    type2 = atom_types[1]
    delta_atoms = [
        d[type1] - d[type2] for d in all_stoic_dicts
    ]  # difference in number of 1st and 2nd type atoms

    n_atoms = {atom: [d[atom] for d in all_stoic_dicts] for atom in atom_types}

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), dpi=DEFAULT_DPI)

    sns.histplot(delta_atoms, ax=ax[0], discrete=True, label=f"#{type1} - #{type2}")
    for atom in atom_types:
        sns.histplot(n_atoms[atom], ax=ax[1], discrete=True, label=f"#{atom}")
    ax[0].legend()
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(Path(save_folder) / f"{fig_name}.png")
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


# TODO: make a colorbar instead of a legend
def plot_clustering_results(
    points: np.ndarray,
    n_clusters: int,
    labels: np.ndarray,
    closest_points_indices: np.ndarray = None,
    save_prepend: str = "",
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
        save_prepend (str, optional): Prepend string for the saved plot. Defaults to "".
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
            s=20,
            label=f"Cluster {i}",
        )

    # Mark the closest points to the centroid in each cluster
    if closest_points_indices is not None:
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

    ax.grid(False)
    ax.set_xlabel("Dimension 1", fontsize=14)
    ax.set_ylabel("Dimension 2", fontsize=14)
    ax.set_title("2D representation of points and clusters", fontsize=16)
    ax.legend(fontsize=12)
    plt.savefig(Path(save_folder, save_prepend + "clustering_results.png"))

    return fig


def plot_dendrogram(
    Z: np.ndarray,
    save_prepend: str = "",
    save_folder: Path | str = ".",
) -> Figure:
    """Plot the dendrogram of the hierarchical clustering.

    Args:
        Z (np.ndarray): The linkage matrix.
        save_prepend (str, optional): Prepend string for the saved plot. Defaults to "".
        save_folder (Path | str, optional): Folder to save the plot in. Defaults to ".".

    Returns:
        Figure: The figure object.
    """
    fig, ax = plt.subplots(figsize=(25, 10), dpi=DEFAULT_DPI)
    dendrogram(Z, no_labels=True)
    ax.grid(False)
    # ax.set_ylabel("Dimension 2", fontsize=14)
    ax.set_title(
        "Dendrogram of structures clustered using the latent space embeddings", fontsize=16
    )
    plt.savefig(Path(save_folder, save_prepend + "dendrogram_Z.png"))

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
