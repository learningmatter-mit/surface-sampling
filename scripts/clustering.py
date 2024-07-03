"""Clustering of structures based on their latent space embeddings."""

import argparse
import datetime
import os
import pickle as pkl
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator
from nff.data import Dataset
from nff.data.dataset import concatenate_dict
from nff.io.ase import AtomsBatch
from nff.io.ase_calcs import EnsembleNFF, NeuralFF
from nff.train.builders import load_model
from nff.utils.cuda import cuda_devices_sorted_by_free_mem
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA
from tqdm import tqdm

from mcmc.calculators import get_results_single
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

    return parser.parse_args()


def get_embeddings(atoms_batches: list[AtomsBatch], calc: Calculator) -> np.ndarray:
    """Calculate the embeddings for a list of AtomsBatch objects

    Parameters
    ----------
    atoms_batches : List[AtomsBatch]
        List of AtomsBatch objects
    calc : Calculator
        NFF Calculator

    Returns:
    -------
    np.ndarray
        Latent space embeddings with each row corresponding to a structure
    """
    print(f"Calculating embeddings for {len(atoms_batches)} structures")
    embeddings = []
    for atoms_batch in tqdm(atoms_batches):
        embedding = get_embeddings_single(atoms_batch, calc)
        embeddings.append(embedding)
    return np.stack(embeddings)


def get_embeddings_single(atoms_batch: AtomsBatch, calc: Calculator) -> np.ndarray:
    """Calculate the embeddings for a single AtomsBatch object

    Parameters
    ----------
    atoms_batch : AtomsBatch
        AtomsBatch object
    calc : Calculator
        NFF Calculator

    Returns:
    -------
    np.ndarray
        Latent space embeddings
    """
    results = get_results_single(atoms_batch, calc)

    return results["embedding"].squeeze()


def get_std_devs(atoms_batches: list[AtomsBatch], calc: Calculator) -> np.ndarray:
    """Calculate the force standard deviations for a list of AtomsBatch objects

    Parameters
    ----------
    atoms_batches : List[AtomsBatch]
        List of AtomsBatch objects
    calc : Calculator
        NFF Calculator

    Returns:
    -------
    np.ndarray
        Force standard deviations with each element corresponding to a structure
    """
    print(f"Calculating force standard deviations for {len(atoms_batches)} structures")
    force_stds = []
    for atoms_batch in tqdm(atoms_batches):
        force_std = get_std_devs_single(atoms_batch, calc)
        force_stds.append(force_std)

    return np.stack(force_stds)


def get_std_devs_single(atoms_batch: AtomsBatch, calc: Calculator) -> np.ndarray:
    """Calculate the force standard deviation for a single AtomsBatch object

    Parameters
    ----------
    atoms_batch : AtomsBatch
        AtomsBatch object
    calc : Calculator
        NFF Calculator

    Returns:
    -------
    np.ndarray
        Force standard deviation
    """
    if len(calc.models) > 1:
        atoms_batch.calc = calc
        calc.calculate(atoms_batch)
        force_std = calc.results.get("forces_std", np.array([0.0])).mean()
    else:
        force_std = 0.0

    return force_std


def perform_clustering(
    embeddings: np.ndarray,
    clustering_cutoff: float,
    cutoff_criterion: Literal["distance", "maxclust"] = "distance",
    save_folder: Path | str = "./",
    save_prepend: str = "",
    **kwargs,
) -> np.ndarray:
    """Perform clustering on the embeddings using PCA and hierarchical clustering.
    Either distance or maxclust can be used as the cutoff criterion.

    Parameters
    ----------
    embeddings : np.ndarray
        Latent space embeddings with each row corresponding to a structure
    clustering_cutoff : Union[int, float]
        Either the distance or the maximum number of clusters
    cutoff_criterion : Literal['distance', 'maxclust'], optional
        Either distance or maxclust, by default 'distance'
    save_folder : Union[Path, str], optional
        Folder to save the plots, by default "./"
    save_prepend : str, optional
        Save directory prefix, by default ""

    Returns:
    -------
    y : np.ndarray
        Each element corresponds to the cluster number of the corresponding structure
    """
    # perform PCA
    X = np.stack(embeddings)
    pca = PCA(n_components=32, whiten=True).fit(X)
    X_r = pca.transform(X)

    # plot_pca(save_folder, save_prepend, X_r)

    print(f"X_r has shape {X_r.shape}")
    print(f"X has shape {X.shape}")

    print(f"The first pca explained ratios are {pca.explained_variance_ratio_[:5]}")

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

    print(f"There are {num_clusters} clusters")

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
) -> None:
    """Select one structure from each cluster according and save the corresponding Atoms objects

    Parameters
    ----------
    atoms_batches : List[Atoms]
        List of Atoms objects
    y : np.ndarray
        Each element corresponds to the cluster number of the corresponding structure
    metric_values : np.ndarray
        Metric values for each structure
    clustering_metric : Literal['force_std', 'random', 'energy'], optional
        Metric used to select the structure, by default 'force_std'
    save_folder : Union[Path, str], optional
        Folder to save the plots, by default "./"
    save_prepend : str, optional
        Save directory prefix, by default ""

    Returns:
    -------
    None
    """
    # Find the maximum per cluster
    data = {"cluster": y, "metric_values": metric_values}

    clustering_df = pd.DataFrame(data).reset_index()

    print("Before selection")
    print(clustering_df.head())
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

    print("After selection")
    print(sorted_df.head())

    print(
        f"Cluster: {y[sorted_df['index'].iloc[0]]} "
        f"metric value: {metric_values[sorted_df['index'].iloc[0]]}"
    )

    selected_indices = sorted_df["index"].to_numpy()

    # save original atoms instead of atoms_batch
    selected_atoms = [atoms_batches[x] for x in selected_indices.tolist()]

    print(f"Saving {len(selected_atoms)} Atoms objects")
    if len(selected_atoms) >= 1 and isinstance(selected_atoms[0], Atoms):
        clustered_atom_files = os.path.join(save_folder, save_prepend + "clustered.pkl")
        with open(clustered_atom_files, "wb") as f:
            pkl.dump(selected_atoms.copy(), f)
    else:
        clustered_atom_files = os.path.join(save_folder, save_prepend + "clustered.pth.tar")
        Dataset(concatenate_dict(*selected_atoms)).save(clustered_atom_files)

    print(f"Saved to {clustered_atom_files}")


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
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    file_paths = [Path(file_name) for file_name in file_names]
    print(f"There are a total of {len(file_paths)} input files")
    # file_base = file_paths[0].resolve().stem
    file_base = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    dset = load_dataset_from_files(file_paths)

    print(f"Loaded {len(dset)} structures")

    if nff_paths:
        # Load existing models
        print(f"Loading existing models from {nff_paths}")
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

    print(f"Performing clustering in {num_batches} batches")
    for i in range(num_batches):
        dset_batch = (
            dset[i * max_input_len : (i + 1) * max_input_len]
            if i < num_batches - 1
            else dset[i * max_input_len :]
        )
        batch_number = i + 1
        print(f"Starting clustering for batch # {batch_number}")

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
            embeddings, clustering_cutoff, cutoff_criterion, save_folder, save_prepend
        )
        select_data_and_save(
            dset_batch, y, metric_values, clustering_metric, save_folder, save_prepend
        )

    print("Clustering complete!")


if __name__ == "__main__":
    args = parse_args()
    if torch.cuda.is_available() and "cpu" not in args.nff_device:
        nff_device = f"cuda:{cuda_devices_sorted_by_free_mem()[-1]}"
    else:
        nff_device = "cpu"

    print(f"Using {nff_device} device for NFF calculations")

    main(
        args.file_paths,
        nff_cutoff=args.nff_cutoff,
        device=nff_device,
        model_type=args.nff_model_type,
        clustering_cutoff=args.clustering_cutoff,
        cutoff_criterion=args.cutoff_criterion,
        clustering_metric=args.clustering_metric,
        max_input_len=args.max_input_len,
        nff_paths=args.nff_paths,
        save_folder=args.save_folder,
    )
