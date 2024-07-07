"""Clustering utilities for MCMC simulations."""

import logging
import os
import pickle as pkl
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from ase import Atoms
from nff.data import Dataset
from nff.data.dataset import concatenate_dict
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA

from mcmc.utils.plot import plot_clustering_results, plot_dendrogram


def perform_clustering(
    embeddings: np.ndarray,
    clustering_cutoff: float,
    cutoff_criterion: Literal["distance", "maxclust"] = "distance",
    save_folder: Path | str = "./",
    save_prepend: str = "",
    logger: logging.Logger | None = None,
    **kwargs,
) -> np.ndarray:
    """Perform clustering on the embeddings using PCA and hierarchical clustering.
    Either distance or maxclust can be used as the cutoff criterion.

    Args:
        embeddings (np.ndarray): Latent space embeddings with each row corresponding to a structure
        clustering_cutoff (float): Either the distance or the maximum number of clusters
        cutoff_criterion (Literal['distance', 'maxclust'], optional): Either distance or maxclust,
            by default 'distance'
        save_folder (Union[Path, str], optional): Folder to save the plots, by default "./"
        save_prepend (str, optional): Save directory prefix, by default ""
        logger (logging.Logger, optional): Logger object, by default None
        **kwargs: Additional keyword arguments

    Returns:
        np.ndarray: Each element corresponds to the cluster number of the corresponding structure
    """
    logger = logger or logging.getLogger(__name__)

    # perform PCA
    X = np.stack(embeddings)
    pca = PCA(n_components=32, whiten=True).fit(X)
    X_r = pca.transform(X)

    # plot_pca(save_folder, save_prepend, X_r)
    logger.info("X_r has shape %s", X_r.shape)
    logger.info("X has shape %s", X.shape)

    logger.info("The first pca explained ratios are %s", pca.explained_variance_ratio_[:5])

    # Perform hierarchical clustering
    Z = linkage(X_r[:, :3], method="ward", metric="euclidean", optimal_ordering=True)

    # plot dendrogram
    plot_dendrogram(Z, save_prepend=save_prepend, save_folder=save_folder)

    # t sets the distance
    if cutoff_criterion == "distance":
        y = fcluster(Z, t=clustering_cutoff, criterion="distance", depth=2)
    else:
        y = fcluster(Z, t=clustering_cutoff, criterion="maxclust", depth=2)

    num_clusters = len(np.unique(y))

    logger.info("There are %s clusters", num_clusters)

    plot_clustering_results(
        X_r, num_clusters, y, save_prepend=save_prepend, save_folder=save_folder
    )

    return y


def select_data_and_save(
    atoms_batches: list[Atoms],
    y: np.ndarray,
    metric_values: np.ndarray,
    clustering_metric: Literal["force_std", "random", "energy"] = "force_std",
    save_folder: Path | str = "./",
    save_prepend: str = "",
    logger: logging.Logger | None = None,
) -> None:
    """Select one structure from each cluster according to the clustering metric and save them.

    Args:
        atoms_batches (List[Atoms]): List of Atoms objects
        y (np.ndarray): Each element corresponds to the cluster number of the corresponding
            structure metric_values (np.ndarray): Metric values for each structure
        clustering_metric (Literal['force_std', 'random', 'energy'], optional): Metric used to
            select the structure, by default 'force_std'
        metric_values (np.ndarray): Values for the selected clustering metric
        save_folder (Union[Path, str], optional): Folder to save the plots, by default "./"
        save_prepend (str, optional): Save directory prefix, by default ""
        logger (logging.Logger, optional): Logger object, by default None
    """
    logger = logger or logging.getLogger(__name__)

    # Find the maximum per cluster
    data = {"cluster": y, "metric_values": metric_values}

    clustering_df = pd.DataFrame(data).reset_index()

    logger.info("Before selection")
    logger.info(clustering_df.head())
    if clustering_metric in "random":
        # Select a random structure from each cluster
        sorted_df = (
            clustering_df.sort_values(["cluster", "metric_values"], ascending=[True, False])
            .groupby("cluster", as_index=False)
            .apply(lambda x: x.sample(1))
        )
    else:
        # Select the highest variance structure from each cluster
        sorted_df = (
            clustering_df.sort_values(["cluster", "metric_values"], ascending=[True, False])
            .groupby("cluster", as_index=False)
            .first()
        )

    logger.info("After selection")
    logger.info(sorted_df.head())

    logger.info(
        "Cluster: %s metric value: %s",
        y[sorted_df["index"].iloc[0]],
        metric_values[sorted_df["index"].iloc[0]],
    )

    selected_indices = sorted_df["index"].to_numpy()

    # save original atoms instead of atoms_batch
    selected_atoms = [atoms_batches[x] for x in selected_indices.tolist()]

    logger.info("Saving %d Atoms objects", len(selected_atoms))
    if len(selected_atoms) >= 1 and isinstance(selected_atoms[0], Atoms):
        clustered_atom_files = os.path.join(save_folder, save_prepend + "clustered.pkl")
        with open(clustered_atom_files, "wb") as f:
            pkl.dump(selected_atoms.copy(), f)
    else:
        clustered_atom_files = os.path.join(save_folder, save_prepend + "clustered.pth.tar")
        Dataset(concatenate_dict(*selected_atoms)).save(clustered_atom_files)

    logger.info("Saved to %s", clustered_atom_files)


def get_cluster_centers(points: np.ndarray, n_clusters: int) -> tuple[np.ndarray, np.ndarray]:
    """Peforms hierarchical clustering on a set of points and returns the centers of the resulting
    clusters.

    Args:
        points (np.ndarray): Numpy array of shape (n_points, n_dimensions) containing the points to
            cluster.
        n_clusters (int): The number of clusters to create.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the centers of the resulting clusters and
        the cluster labels shape for each point.
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
        points (np.ndarray): Numpy array of shape (n_points, n_dimensions) containing the points to
            cluster.
        centers (np.ndarray): Numpy array of shape (n_clusters, n_dimensions) containing the centers
            of the clusters.
        labels (np.ndarray): Numpy array of shape (n_points,) with the cluster number for each
            point.

    Returns:
        np.ndarray: Numpy array of shape (n_clusters,) with the index of the closest point to the
            centroid in each cluster.
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


# # Test
# points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
# n_clusters = 3
# centers, labels = get_cluster_centers(points, n_clusters)
# closest_points_indices = find_closest_points_indices(points, centers, labels)

# for i in range(n_clusters):
#     print(f"Center of cluster {i + 1}: {centers[i]}")
#     print(f"Index of closest point to center in cluster {i + 1}: {closest_points_indices[i]}")
#     print(f"Points in cluster {i + 1}: {points[labels == i + 1]}\n")
