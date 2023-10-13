import matplotlib.pyplot as plt
import numpy as np
from ase.atoms import Atoms
from nff.io.ase import AtomsBatch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import distance
from scipy.special import softmax


def get_atoms_batch(slab: Atoms, neighbor_cutoff: float, nff_calc, device: str):
    return AtomsBatch(
        positions=slab.positions,
        numbers=slab.numbers,
        cell=slab.cell,
        pbc=True,
        cutoff=neighbor_cutoff,
        props={"energy": 0, "energy_grad": []},
        calculator=nff_calc,
        requires_large_offsets=True,
        directed=True,
        device=device,
    )


def filter_distances(slab, ads=["O"], cutoff_distance: float = 1.5):
    """This function filters out slabs that have atoms too close to each other based on a
    specified cutoff distance.

    Parameters
    ----------
    slab : ase.atoms.Atoms or catkit.gratoms.Gratoms or AtomsBatch
        The `slab` is the surface structure
    ads
        The ads parameter is a list of chemical symbols representing the atoms that are being checked
        for their distances from each other. By default, it is set to ["O"], which means that the function
        will check the distances of oxygen atoms from each other.
    cutoff_distance
        The minimum distance allowed between any two atoms in the slab. If any two adsorbate
        atoms are closer than this distance, the function will return False.

    Returns
    -------
        a boolean value. It returns True if all distances between the specified adsorbates in the given
    slab are greater than the specified cutoff distance, and False otherwise.

    """
    # Checks distances of all adsorbates are greater than cutoff
    ads_arr = np.isin(slab.get_chemical_symbols(), ads)
    unique_dists = np.unique(
        np.triu(slab.get_all_distances(mic=True)[ads_arr][:, ads_arr])
    )
    # get upper triangular matrix of ads dist
    if any(unique_dists[(unique_dists > 0) & (unique_dists <= cutoff_distance)]):
        return False  # fail because atoms are too close
    return True


def get_cluster_centers(points: np.ndarray, n_clusters: int):
    """
    This function performs hierarchical clustering on a set of points and returns the centers of the resulting clusters.

    Parameters
    ----------
    points : numpy.ndarray
        A numpy array of shape (n_points, n_dimensions) containing the points to cluster.
    n_clusters : int
        The number of clusters to create.

    Returns
    -------
    centers : numpy.ndarray
        A numpy array of shape (n_clusters, n_dimensions) containing the centers of the resulting clusters.
    labels : numpy.ndarray
        A numpy array of shape (n_points,) containing the cluster labels for each point.
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


def find_closest_points_indices(points, centers, labels):
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
    points,
    n_clusters,
    labels,
    closest_points_indices,
    save_folder=".",
):
    # Define colors
    colors = ["b", "g", "r", "c", "m", "y", "k"]

    # Create a larger plot
    plt.figure(figsize=(10, 7))

    # Create a scatter plot of all points, color-coded by cluster
    for i in range(1, n_clusters + 1):
        cluster_points = points[labels == i]
        plt.scatter(
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
        plt.scatter(
            closest_point[0],
            closest_point[1],
            marker="*",
            color="black",
            edgecolor="black",
            linewidth=1,
            s=200,
        )

    plt.grid(True)
    plt.xlabel("Dimension 1", fontsize=14)
    plt.ylabel("Dimension 2", fontsize=14)
    plt.title("2D representation of points and clusters", fontsize=16)
    plt.legend(fontsize=12)
    plt.savefig(f"{save_folder}/clustering_results.png")


# # Test
# points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
# n_clusters = 3
# centers, labels = get_cluster_centers(points, n_clusters)
# closest_points_indices = find_closest_points_indices(points, centers, labels)

# for i in range(n_clusters):
#     print(f"Center of cluster {i + 1}: {centers[i]}")
#     print(f"Index of closest point to center in cluster {i + 1}: {closest_points_indices[i]}")
#     print(f"Points in cluster {i + 1}: {points[labels == i + 1]}\n")


def compute_distance_weight_matrix(ads_coords, distance_decay_factor):
    # Compute pairwise distance matrix
    ads_coord_distances = distance.cdist(ads_coords, ads_coords, "euclidean")

    # Compute distance decay matrix using softmax
    distance_weight_matrix = softmax(
        -ads_coord_distances / distance_decay_factor, axis=1
    )

    assert np.allclose(np.sum(distance_weight_matrix, axis=1), 1.0)

    return distance_weight_matrix


def plot_distance_weight_matrix(distance_weight_matrix, save_folder="."):
    # Define colors
    colors = ["b", "g", "r", "c", "m", "y", "k"]

    # Create a larger plot
    plt.figure(figsize=(10, 7))

    # Display the distance weight matrix as an image
    plt.imshow(distance_weight_matrix, cmap="hot", interpolation="nearest")

    # Add a colorbar to the figure to show how colors correspond to values
    plt.colorbar()

    plt.xlabel("Dimension 1", fontsize=14)
    plt.ylabel("Dimension 2", fontsize=14)
    plt.title("Distance Weight Matrix")
    plt.savefig(f"{save_folder}/distance_weight_matrix.png")


def plot_decay_curve(decay_factor, save_folder="."):
    plt.figure(figsize=(10, 7))
    distances = np.linspace(0, 3 * decay_factor, 100)
    factor = softmax(-distances / decay_factor)
    plt.plot(distances, factor, color="blue", label="Decay Factor")

    plt.xlabel("Distance [Å]", fontsize=14)
    plt.ylabel("Probability density", fontsize=14)
    plt.title("Distance Decay Plot")
    plt.legend(fontsize=12)
    plt.savefig(f"{save_folder}/distance_weight_decay.png")


def plot_specific_weights(coords, weights, site_idx, save_folder=".", run_iter=0):
    # Create a larger plot
    plt.figure(figsize=(10, 7))
    curr_site = coords[site_idx]

    # Create a scatter plot of all points, color-coded by weights
    plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=weights,
        alpha=0.6,
        edgecolor="black",
        linewidth=1,
        s=100,
    )
    plt.scatter(
        curr_site[0],
        curr_site[1],
        marker="*",
        color="black",
        edgecolor="black",
        linewidth=1,
        s=200,
    )
    # Add a colorbar to the figure to show how colors correspond to values
    plt.colorbar()
    plt.grid(True)
    plt.xlabel("Dimension 1", fontsize=14)
    plt.ylabel("Dimension 2", fontsize=14)
    plt.title(
        "2D representation of adsorption sites color-coded by weights", fontsize=16
    )
    plt.savefig(f"{save_folder}/specific_weights_on_lattice_iter_{run_iter:06}.png")
    plt.close()
