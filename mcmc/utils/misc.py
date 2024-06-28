from pathlib import Path
from typing import Iterable, Union

import matplotlib.pyplot as plt
import numpy as np
from ase.atoms import Atoms
from matplotlib.figure import Figure
from nff.io.ase import AtomsBatch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import distance
from scipy.special import softmax

DPI = 200


def get_atoms_batch(
    data: Union[dict, Atoms],
    nff_cutoff: float,
    device: str = "cpu",
    **kwargs,
) -> AtomsBatch:
    """Generate AtomsBatch from atoms or dictionary.

    Args:
        data (Union[dict, Atoms]): Dictionary containing the properties of the atoms
        nff_cutoff (float): Neighbor cutoff for the NFF model
        device (str, optional): cpu or cuda device. Defaults to 'cpu'.

    Returns:
        AtomsBatch
    """
    if isinstance(data, Atoms):
        atoms_batch = AtomsBatch.from_atoms(
            data,
            cutoff=nff_cutoff,
            requires_large_offsets=False,
            dense_nbrs=False,
            directed=True,
            device=device,
            **kwargs,
        )
    else:
        pass
        # atoms_batch = AtomsBatch.from_dict(
        #     data,
        #     cutoff=nff_cutoff,
        #     requires_large_offsets=False,
        #     directed=True,
        #     device=device,
        #     **kwargs,
        # )

    return atoms_batch


def filter_distances(
    slab: Atoms, ads: Iterable = ("O"), cutoff_distance: float = 1.5
) -> bool:
    """This function filters out slabs that have atoms too close to each other based on a
    specified cutoff distance.

    Args:
        slab (Atoms): The slab structure to check for distances.
        ads (Iterable, optional): The adsorbate atom types in the slab to check for. Defaults to ("O").
        cutoff_distance (float, optional): The cutoff distance to check for. Defaults to 1.5.

    Returns:
        bool: True if the distances are greater than the cutoff distance, False otherwise.
    """
    # Checks distances of all adsorbates are greater than cutoff
    ads_arr = np.isin(slab.get_chemical_symbols(), ads)
    unique_dists = np.unique(
        np.triu(slab.get_all_distances(mic=True)[ads_arr][:, ads_arr])
    )
    # Get upper triangular matrix of ads dist
    if any(unique_dists[(unique_dists > 0) & (unique_dists <= cutoff_distance)]):
        return False  # Gail because atoms are too close
    return True


def get_cluster_centers(
    points: np.ndarray, n_clusters: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Peforms hierarchical clustering on a set of points and returns the centers of the resulting clusters.

    Args:
        points (np.ndarray): Numpy array of shape (n_points, n_dimensions) containing the points to cluster.
        n_clusters (int): The number of clusters to create.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the centers of the resulting clusters and the
        cluster labels shape for each point.
    """

    # Do hierarchical clustering
    Z = linkage(points, "ward")

    # Cut the tree to create k clusters
    labels = fcluster(Z, n_clusters, criterion="maxclust")

    centers = []
    for i in range(1, n_clusters + 1):
        # Get all points in cluster i
        cluster_points = points[labels == i]

        # Compute and store the center of the cluster
        center = np.mean(cluster_points, axis=0)
        centers.append(center)

    return np.array(centers), labels


# # Test
# points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
# n_clusters = 3
# centers, labels = get_cluster_centers(points, n_clusters)

# for i in range(n_clusters):
#     print(f"Center of cluster {i + 1}: {centers[i]}")
#     print(f"Points in cluster {i + 1}: {points[labels == i + 1]}")


def find_closest_points_indices(
    points: np.ndarray, centers: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """Finds the index of the point in each cluster that is closest to the center of the cluster.

    Args:
        points (np.ndarray): Numpy array of shape (n_points, n_dimensions) containing the points to cluster.
        centers (np.ndarray): Numpy array of shape (n_clusters, n_dimensions) containing the centers of the clusters.
        labels (np.ndarray): Numpy array of shape (n_points,) with the cluster number for each point.

    Returns:
        np.ndarray: Numpy array of shape (n_clusters,) with the index of the closest point to the centroid in each cluster.
    """
    closest_points_indices = []
    for i in range(1, len(centers) + 1):
        # Get indices of all points in cluster i
        cluster_indices = np.where(labels == i)[0]

        # Get all points in cluster i
        cluster_points = points[cluster_indices]

        # Calculate the distances from all points to the center point
        distances = np.linalg.norm(cluster_points - centers[i - 1], axis=1)

        # Find the index of the point with the smallest distance to the center point
        closest_point_index = cluster_indices[np.argmin(distances)]
        closest_points_indices.append(closest_point_index)

    return np.array(closest_points_indices)


def plot_clustering_results(
    points: np.ndarray,
    n_clusters: int,
    labels: np.ndarray,
    closest_points_indices: np.ndarray,
    save_folder: Union[Path, str] = ".",
) -> Figure:
    """Plot the clustering results.

    Args:
        points (np.ndarray): 2D numpy array of shape (n_points, 2) containing the points to cluster.
        n_clusters (int): Total number of calculated clusters.
        labels (np.ndarray): Numpy array of shape (n_points,) with the cluster number for each point.
        closest_points_indices (np.ndarray): Numpy array of shape (n_clusters,) with the index of the closest point
            to the centroid in each cluster.
        save_folder (str, optional): Folder to save the plot in. Defaults to ".".

    Returns:
        Figure: The figure object.
    """
    # Define colors
    colors = ["b", "g", "r", "c", "m", "y", "k"]

    # Create a larger plot
    fig, ax = plt.subplots(figsize=(10, 7), dpi=DPI)

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


# # Test
# points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
# n_clusters = 3
# centers, labels = get_cluster_centers(points, n_clusters)
# closest_points_indices = find_closest_points_indices(points, centers, labels)

# for i in range(n_clusters):
#     print(f"Center of cluster {i + 1}: {centers[i]}")
#     print(f"Index of closest point to center in cluster {i + 1}: {closest_points_indices[i]}")
#     print(f"Points in cluster {i + 1}: {points[labels == i + 1]}\n")


def compute_distance_weight_matrix(
    ads_coords: np.ndarray, distance_decay_factor: float
) -> np.ndarray:
    """Compute distance weight matrix using softmax.

    Args:
        ads_coords (np.ndarray): The coordinates of the adsorption sites.
        distance_decay_factor (float): Exponential decay factor.

    Returns:
        np.ndarray: The distance weight matrix.
    """

    # Compute pairwise distance matrix
    ads_coord_distances = distance.cdist(ads_coords, ads_coords, "euclidean")

    # Compute distance decay matrix using softmax
    distance_weight_matrix = softmax(
        -ads_coord_distances / distance_decay_factor, axis=1
    )

    assert np.allclose(np.sum(distance_weight_matrix, axis=1), 1.0)

    return distance_weight_matrix


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
    colors = ["b", "g", "r", "c", "m", "y", "k"]

    # Create a larger plot
    fig, ax = plt.subplots(figsize=(10, 7), dpi=DPI)

    # Display the distance weight matrix as an image
    img = ax.imshow(distance_weight_matrix, cmap="hot", interpolation="nearest")

    # Add a colorbar to the figure to show how colors correspond to values
    cb = plt.colorbar(img, ax=ax)

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
    fig, ax = plt.subplots(figsize=(10, 7), dpi=DPI)
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
    weights: Union[list, np.ndarray],
    site_idx: int,
    save_folder: Union[Path, str] = ".",
    run_iter: int = 0,
) -> Figure:
    """
    Plot weights of the adsorption sites on the lattice.

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
    fig, ax = plt.subplots(figsize=(10, 7), dpi=DPI)
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
    cb = plt.colorbar(scatter, ax=ax)

    ax.grid(True)
    ax.set_xlabel("Dimension 1", fontsize=14)
    ax.set_ylabel("Dimension 2", fontsize=14)
    ax.set_title(
        "2D representation of adsorption sites color-coded by weights", fontsize=16
    )
    plt.savefig(f"{save_folder}/specific_weights_on_lattice_iter_{run_iter:06}.png")
    return fig
