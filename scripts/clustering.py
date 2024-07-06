"""Clustering of structures based on their latent space embeddings."""

import argparse
import datetime
import logging
import os
import pickle as pkl
from logging import getLevelNamesMapping
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from ase import Atoms
from nff.data import Dataset
from nff.data.dataset import concatenate_dict
from nff.io.ase_calcs import EnsembleNFF, NeuralFF
from nff.train.builders import load_model
from nff.utils.cuda import cuda_devices_sorted_by_free_mem
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA
from tqdm import tqdm

from mcmc.calculators import (
    get_results_single,
    get_std_devs_single,
)
from mcmc.utils import setup_logger
from mcmc.utils.misc import get_atoms_batch, load_dataset_from_files
from mcmc.utils.plot import plot_clustering_results, plot_dendrogram

np.set_printoptions(precision=3, suppress=True)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Cluster structures based on their latent space embeddings."
    )
    parser.add_argument(
        "--file_paths",
        nargs="+",
        help="Full paths to NFF Dataset, ASE Atoms/NFF AtomsBatch, or a text file of file paths.",
        type=Path,
    )
    parser.add_argument(
        "--save_folder",
        type=Path,
        default="./",
        help="Folder to save cut surfaces.",
    )
    parser.add_argument(
        "--nff_model_type",
        choices=("CHGNetNFF", "NffScaleMACE"),
        help="NFF model type",
        type=str,
        default="CHGNetNFF",
    )
    parser.add_argument(
        "--nff_paths", nargs="*", help="Full path to NFF model", type=str, default=[""]
    )
    parser.add_argument(
        "--nff_cutoff",
        help="NFF cutoff, should be consistent with NFF training cutoff",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--nff_device",
        help="NFF device, either cpu or cuda",
        type=str,
        default="cpu",
    )
    parser.add_argument(
        "--max_input_len",
        help="Maximum number of structures used in each clustering iteration",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--clustering_metric",
        help="Metric used to select structure from each cluster",
        choices=("force_std", "random", "energy"),
        type=str,
        default="force_std",
    )
    parser.add_argument(
        "--cutoff_criterion",
        choices=("distance", "maxclust"),
        help="Cutoff criterion, either distance or maxclust",
        type=str,
        default="distance",
    )
    parser.add_argument(
        "--clustering_cutoff",
        help=(
            "Clustering cutoff, either the cutoff distance between surfaces "
            "or the maximum number of clusters"
        ),
        type=float,
        default=200,
    )
    parser.add_argument(
        "--logging_level",
        type=str,
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Logging level",
    )

    return parser.parse_args()


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

    # plot_pca_clusters(X_r, max_index, y, save_folder)
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
    """Select one structure from each cluster according and save the corresponding Atoms objects

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
        f"Cluster: {y[sorted_df['index'].iloc[0]]} "
        f"metric value: {metric_values[sorted_df['index'].iloc[0]]}"
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


def main(
    file_names: list[str],
    nff_cutoff: float = 5.0,
    device: str = "cuda:0",
    model_type: str = "CHGNetNFF",
    clustering_cutoff: float = 0.2,
    cutoff_criterion: Literal["distance", "maxclust"] = "distance",
    clustering_metric: Literal["force_std", "random", "energy"] = "force_std",
    max_input_len: int = 1000,
    nff_paths: list[Path | str] | None = None,
    save_folder: Path | str = "./",
    logging_level: Literal["debug", "info", "warning", "error", "critical"] = "info",
) -> None:
    """Main function to perform clustering on a list of structures

    Args:
        file_names (list[str]) : List of file paths to load structures from
        nff_cutoff (float, optional) : Neighbor cutoff for the NFF model, by default 5.0
        device (str, optional) : cpu or cuda device, by default 'cuda:0'
        model_type (str, optional) : NFF model type, by default 'CHGNetNFF'
        clustering_cutoff (float, optional) : Either the distance or the maximum number of clusters,
            by default 0.2
        max_input_len (int, optional) : Maximum number of structures used in each clustering
            iteration, by default 1000
        clustering_metric (Literal['force_std', 'random', 'energy'], optional) : Metric used to
            select structure from each cluster, by default 'force_std'
        nff_paths (list[Path, str], optional) : Full path to NFF model, by default None
        cutoff_criterion (Literal['distance', 'maxclust'], optional) : Either distance or maxclust,
            by default 'distance'
        save_folder (Union[Path, str], optional) : Folder to save the plots, by default "./"
        logging_level (Literal['debug', 'info', 'warning', 'error', 'critical'], optional) : Logging
            level, by default 'info'
    """
    start_timestamp = datetime.now().isoformat(sep="-", timespec="milliseconds")

    # Initialize run folder
    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)

    # Initialize logger
    logger = setup_logger(
        "clustering",
        save_path / "clustering.log",
        level=getLevelNamesMapping()[logging_level.upper()],
    )

    logger.info("There are a total of %d input files", len(file_names))
    dset = load_dataset_from_files(file_names)
    logger.info("Loaded %d structures", len(dset))

    if torch.cuda.is_available() and "cpu" not in device:
        device = f"cuda:{cuda_devices_sorted_by_free_mem()[-1]}"
    else:
        device = "cpu"
    logger.info("Using %d device for NFF calculations", device)

    if nff_paths:
        # Load existing models
        logger.info("Loading existing models from %s", nff_paths)
        models = [
            load_model(
                path,
                model_type=model_type,
                map_location=device,
                requires_embeddings=True,
            )
            for path in nff_paths
        ]
    else:
        # raise error if no models are provided
        raise ValueError("No NFF models provided")

    # EnsembleNFF for force standard deviation prediction
    ensemble_calc = EnsembleNFF(
        models,
        device=device,
        model_units="eV/atom",
        prediction_units="eV",
    )
    # NeuralFF for latent space embedding calculation
    single_calc = NeuralFF(
        models[0],
        device=device,
        model_units="eV/atom",
        prediction_units="eV",
        properties=["energy", "forces", "embedding"],
    )

    # Perform clustering in batches
    num_batches = len(dset) // max_input_len + bool(
        len(dset) % max_input_len
    )  # additional batch for the remainder

    logger.info("Performing clustering in %d batches", num_batches)
    for i in range(num_batches):
        dset_batch = (
            dset[i * max_input_len : (i + 1) * max_input_len]
            if i < num_batches - 1
            else dset[i * max_input_len :]
        )
        batch_number = i + 1
        logger.info("Starting clustering for batch # %d", batch_number)

        file_base = f"{start_timestamp}_clustering"
        save_prepend = (
            file_base
            + f"_{len(dset_batch)}_input_structures"
            + f"_batch_{batch_number}".zfill(3)
            + f"_cutoff_{clustering_cutoff}_"
            + f"{clustering_metric}_"
        )

        # doing it singly to save memory and is faster
        embeddings = []
        metric_values = []
        for single_dset in tqdm(dset_batch):
            atoms_batch = get_atoms_batch(single_dset, nff_cutoff, device=device)
            single_calc_results = get_results_single(atoms_batch, single_calc)
            embedding = single_calc_results["embedding"].squeeze()
            if clustering_metric == "energy":
                metric_value = single_calc_results["energy"].squeeze()
            elif clustering_metric == "force_std":
                metric_value = get_std_devs_single(atoms_batch, ensemble_calc)
            else:
                metric_value = np.random.rand()
            embeddings.append(
                embedding
            )  # BUG: NFFScaleMace embeddings are given atomwise, which means we need to stack them
            metric_values.append(metric_value)
        embeddings = np.stack(embeddings)
        metric_values = np.stack(metric_values)

        # atoms_batches = get_atoms_batches(dset_batch, nff_cutoff, device=device)
        # embeddings = get_embeddings(atoms_batches, single_calc)
        # force_std_devs = get_std_devs(atoms_batches, ensemble_calc)

        # # delete the AtomsBatch to free up memory
        # del atoms_batches

        y = perform_clustering(
            embeddings, clustering_cutoff, cutoff_criterion, save_path, save_prepend
        )
        select_data_and_save(
            dset_batch, y, metric_values, clustering_metric, save_path, save_prepend
        )
    logger.info("Clustering complete!")


if __name__ == "__main__":
    args = parse_args()
    main(
        args.file_paths,
        nff_cutoff=args.nff_cutoff,
        device=args.nff_device,
        model_type=args.nff_model_type,
        clustering_cutoff=args.clustering_cutoff,
        cutoff_criterion=args.cutoff_criterion,
        clustering_metric=args.clustering_metric,
        max_input_len=args.max_input_len,
        nff_paths=args.nff_paths,
        save_folder=args.save_folder,
        logging_level=args.logging_level,
    )
