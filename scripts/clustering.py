import argparse
import datetime
import os
import pickle as pkl
from pathlib import Path
from typing import List, Literal, Union

import ase
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator
from nff.data import Dataset
from nff.data.dataset import concatenate_dict
from nff.io.ase import AtomsBatch
from nff.io.ase_calcs import EnsembleNFF, NeuralFF
from nff.utils.cuda import cuda_devices_sorted_by_free_mem
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.decomposition import PCA
from tqdm import tqdm

from mcmc.utils.misc import get_atoms_batch


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Cluster structures based on their latent space embeddings."
    )
    parser.add_argument(
        "--file_paths",
        nargs="+",
        help="Full paths to NFF Dataset or ASE Atoms/NFF AtomsBatch",
        type=Path,
    )
    parser.add_argument(
        "--save_folder",
        type=Path,
        default="./",
        help="Folder to save cut surfaces.",
    )
    # parser.add_argument("--painn_params_file", help="PaiNN parameter file", type=str, default="painn_params.json")
    parser.add_argument(
        "--nff_model_type",
        choices=("CHGNetNFF", "DirectNffScaleMACEWrapper"),
        help="NFF model type",
        type=str,
        default="CHGNetNFF",
    )
    parser.add_argument(
        "--nff_paths", nargs="+", help="Full path to NFF model", type=str, default=""
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
        "--use_force_std",
        action="store_true",
        help="Use force standard deviation to select structures",
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
        help="Clustering cutoff, either the cutoff distance between surfaces or the maximum number of clusters",
        type=float,
        default=200,
    )

    return parser.parse_args()


def plot_dendrogram(save_folder, save_prepend, Z):
    fig = plt.figure(figsize=(25, 10), dpi=200)
    dn = dendrogram(Z, no_labels=True)
    plt.savefig(os.path.join(save_folder, save_prepend + "dendrogram_Z.png"))
    plt.show()


def plot_pca_clusters(save_folder, save_prepend, X_r, y, clusters, max_index):
    fig = plt.figure(figsize=(10, 10), dpi=200)
    for cluster in clusters:
        plt.scatter(
            X_r[y == cluster, 0],
            X_r[y == cluster, 1],
            s=5,
            alpha=0.8,
            color=plt.cm.inferno(cluster / max_index),
        )
    plt.title("Clustering based on PaiNN embeddings")
    plt.colorbar()
    plt.savefig(
        os.path.join(save_folder, save_prepend + "clustered_painn_embeddings_pca.png")
    )
    plt.show()


def plot_pca(save_folder, save_prepend, X_r):
    plt.figure()
    plt.scatter(X_r[:, 0], X_r[:, 1], s=2, alpha=0.8, color=plt.cm.inferno(0.5))
    plt.title("PCA of PaiNN embeddings")
    plt.savefig(os.path.join(save_folder, save_prepend + "painn_embeddings_pca.png"))
    plt.show()


def get_atoms_batches(
    data: Union[Dataset, List[ase.Atoms]],
    nff_cutoff: float,
    device: str = "cpu",
    structures_per_batch: int = 32,
    **kwargs,
) -> List[AtomsBatch]:
    """Generate AtomsBatch

    Parameters
    ----------
    data : Union[Dataset, List[ase.Atoms]]
        Dictionary containing the properties of the atoms
    nff_cutoff : float
        Neighbor cutoff for the NFF model
    model : Calculator
        NFF Calculator
    device : str, optional
        cpu or cuda device, by default 'cpu'

    Returns
    -------
    List[AtomsBatch]
        List of AtomsBatch objects
    """
    print(f"Data has length {len(data)}")

    if isinstance(data, Dataset):
        atoms_batches = data.as_atoms_batches()
    # elif len(data) > 0 and isinstance(data[0], AtomsBatch):
    #     atoms_batches = data
    else:
        atoms_batches = []
        # TODO: select structures_per_batch structures at a time
        for atoms in tqdm(data):
            atoms_batch = AtomsBatch.from_atoms(
                atoms,
                cutoff=nff_cutoff,
                requires_large_offsets=False,
                directed=True,
                device=device,
                **kwargs,
            )
            atoms_batches.append(atoms_batch)

    return atoms_batches


def get_embeddings(atoms_batches: List[AtomsBatch], calc: Calculator) -> np.ndarray:
    """Calculate the embeddings for a list of AtomsBatch objects

    Parameters
    ----------
    atoms_batches : List[AtomsBatch]
        List of AtomsBatch objects
    calc : Calculator
        NFF Calculator

    Returns
    -------
    np.ndarray
        Latent space embeddings with each row corresponding to a structure
    """

    print(f"Calculating embeddings for {len(atoms_batches)} structures")
    embeddings = []
    for atoms_batch in tqdm(atoms_batches):
        embedding = get_embeddings_single(atoms_batch, calc)
        embeddings.append(embedding)

    embeddings = np.stack(embeddings)
    return embeddings


def get_embeddings_single(atoms_batch: AtomsBatch, calc: Calculator) -> np.ndarray:
    """Calculate the embeddings for a single AtomsBatch object

    Parameters
    ----------
    atoms_batch : AtomsBatch
        AtomsBatch object
    calc : Calculator
        NFF Calculator

    Returns
    -------
    np.ndarray
        Latent space embeddings
    """

    atoms_batch.calc = calc
    calc.calculate(atoms_batch)

    return calc.results["embedding"].squeeze()


def get_std_devs(atoms_batches: List[AtomsBatch], calc: Calculator) -> np.ndarray:
    """Calculate the force standard deviations for a list of AtomsBatch objects

    Parameters
    ----------
    atoms_batches : List[AtomsBatch]
        List of AtomsBatch objects
    calc : Calculator
        NFF Calculator

    Returns
    -------
    np.ndarray
        Force standard deviations with each element corresponding to a structure
    """

    print(f"Calculating force standard deviations for {len(atoms_batches)} structures")
    force_stds = []
    for atoms_batch in tqdm(atoms_batches):
        force_std = get_std_devs_single(atoms_batch, calc)
        force_stds.append(force_std)

    force_stds = np.stack(force_stds)
    return force_stds


def get_std_devs_single(atoms_batch: AtomsBatch, calc: Calculator) -> np.ndarray:
    """Calculate the force standard deviation for a single AtomsBatch object

    Parameters
    ----------
    atoms_batch : AtomsBatch
        AtomsBatch object
    calc : Calculator
        NFF Calculator

    Returns
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
    clustering_cutoff: Union[int, float],
    cutoff_criterion: Literal["distance", "maxclust"] = "distance",
    save_folder: Union[Path, str] = "./",
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

    Returns
    -------
    y : np.ndarray
        Each element corresponds to the cluster number of the corresponding structure
    """

    # perform PCA
    X = np.stack(embeddings)
    pca = PCA(n_components=32, whiten=True).fit(X)
    X_r = pca.transform(X)

    plot_pca(save_folder, save_prepend, X_r)

    print(f"X_r has shape {X_r.shape}")
    print(f"X has shape {X.shape}")

    print(f"The first pca explained ratios are {pca.explained_variance_ratio_[:5]}")

    # Perform hierarchical clustering
    Z = linkage(X_r[:, :3], method="ward", metric="euclidean", optimal_ordering=True)

    # plot dendrogram
    plot_dendrogram(save_folder, save_prepend, Z)

    # t sets the distance
    if cutoff_criterion == "distance":
        y = fcluster(Z, t=clustering_cutoff, criterion="distance", depth=2)
    else:
        y = fcluster(Z, t=clustering_cutoff, criterion="maxclust", depth=2)

    clusters = np.unique(y)
    max_index = np.max(clusters)

    print(f"There are {len(clusters)} clusters")

    plot_pca_clusters(save_folder, save_prepend, X_r, y, clusters, max_index)

    return y


def select_data_and_save(
    atoms_batches: List[Atoms],
    y: np.ndarray,
    force_std: np.ndarray,
    use_force_std: bool = True,
    save_folder: Union[Path, str] = "./",
    save_prepend: str = "",
) -> None:
    """Select the highest variance structure from each cluster and save the corresponding Atoms objects

    Parameters
    ----------
    atoms_batches : List[Atoms]
        List of Atoms objects
    y : np.ndarray
        Each element corresponds to the cluster number of the corresponding structure
    force_std : np.ndarray
        Force standard deviations with each element corresponding to a structure
    use_force_std : bool, optional
        Use force standard deviation to select structures, by default True
    save_folder : Union[Path, str], optional
        Folder to save the plots, by default "./"
    save_prepend : str, optional
        Save directory prefix, by default ""

    Returns
    -------
    None
    """

    # Find the maximum per cluster
    data = {"cluster": y, "force_std": force_std}

    df = pd.DataFrame(data).reset_index()

    print("Before selection")
    print(df.head())
    if use_force_std:
        # Select the highest variance structure from each cluster
        sorted_std_df = (
            df.sort_values(["cluster", "force_std"], ascending=[True, False])
            .groupby("cluster", as_index=False)
            .first()
        )
    else:
        # Select a random structure from each cluster
        sorted_std_df = (
            df.sort_values(["cluster", "force_std"], ascending=[True, False])
            .groupby("cluster", as_index=False)
            .apply(lambda x: x.sample(1))
        )

    print("After selection")
    print(sorted_std_df.head())

    print(
        f"Cluster: {y[sorted_std_df['index'].iloc[0]]} force std: {force_std[sorted_std_df['index'].iloc[0]]}"
    )

    selected_indices = sorted_std_df["index"].to_numpy()

    # save original atoms instead of atoms_batch
    selected_atoms = [atoms_batches[x] for x in selected_indices.tolist()]

    print(f"Saving {len(selected_atoms)} Atoms objects")
    if len(selected_atoms) >= 1 and isinstance(selected_atoms[0], Atoms):
        clustered_atom_files = os.path.join(save_folder, save_prepend + "clustered.pkl")
        with open(clustered_atom_files, "wb") as f:
            pkl.dump(selected_atoms.copy(), f)
    else:
        clustered_atom_files = os.path.join(
            save_folder, save_prepend + "clustered.pth.tar"
        )
        Dataset(concatenate_dict(*selected_atoms)).save(clustered_atom_files)

    print(f"Saved to {clustered_atom_files}")


def main(
    file_names: List[str],
    nff_cutoff: float = 5.0,
    device: str = "cuda:0",
    model_type: str = "CHGNetNFF",
    clustering_cutoff: Union[int, float] = 0.2,
    max_input_len: int = 1000,
    use_force_std: bool = True,
    nff_paths: List[Union[Path, str]] = None,
    cutoff_criterion: Literal["distance", "maxclust"] = "distance",
    save_folder: Union[Path, str] = "./",
) -> None:
    """Main function to perform clustering on a list of structures

    Parameters
    ----------
    file_names : List[str]
        List of file paths to load structures from
    nff_cutoff : float, optional
        Neighbor cutoff for the NFF model, by default 5.0
    device : str, optional
        cpu or cuda device, by default 'cuda:0'
    model_type : str, optional
        NFF model type, by default 'CHGNetNFF'
    clustering_cutoff : Union[int, float], optional
        Either the distance or the maximum number of clusters, by default 0.2
    max_input_len : int, optional
        Maximum number of structures used in each clustering iteration, by default 1000
    use_force_std : bool, optional
        Use force standard deviation to select structures, by default True
    nff_paths : List[Path, str], optional
        Full path to NFF model, by default None
    cutoff_criterion : Literal['distance', 'maxclust'], optional
        Either distance or maxclust, by default 'distance'
    save_folder : Union[Path, str], optional
        Folder to save the plots, by default "./"

    Returns
    -------
    None
    """

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    file_paths = [Path(file_name) for file_name in file_names]
    print(f"There are a total of {len(file_paths)} input files")
    # file_base = file_paths[0].resolve().stem
    file_base = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    dset = []
    for x in file_paths:
        if x.suffix == ".pkl":
            f = open(x, "rb")
            dset.extend(pkl.load(f))
        else:  # .pth.tar
            data = Dataset.from_file(x)
            dset.extend(data)

    print(f"Loaded {len(dset)} structures")

    if nff_paths:
        # Load existing models
        print(f"Loading existing models from {nff_paths}")
        models = []
        for nff_path in nff_paths:
            m = NeuralFF.from_file(
                nff_path, device=device, model_type=model_type, requires_embedding=True
            ).model
            models.append(m)
    else:
        # raise error if no models are provided
        raise ValueError("No NFF models provided")
        # Initialize new Painn models
        # print(f"Initializing new Painn models")
        # painn_params = json.load(open(painn_params_file, 'r'))
        # models = [Painn(painn_params) for _ in range(3)]

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

        cluster_selection_metric = "force_std" if use_force_std else "random"
        save_prepend = (
            file_base
            + f"_{len(dset_batch)}_input_structures"
            + f"_batch_{batch_number}".zfill(3)
            + f"_cutoff_{clustering_cutoff}_"
            + f"{cluster_selection_metric}_"
        )

        # doing it singly to save memory and is faster
        embeddings = []
        force_std_devs = []
        for single_dset in tqdm(dset_batch):
            atoms_batch = get_atoms_batch(single_dset, nff_cutoff, device=device)
            embedding = get_embeddings_single(atoms_batch, single_calc)
            force_std = get_std_devs_single(atoms_batch, ensemble_calc)
            embeddings.append(embedding)
            force_std_devs.append(force_std)
        embeddings = np.stack(embeddings)
        force_std_devs = np.stack(force_std_devs)

        # atoms_batches = get_atoms_batches(dset_batch, nff_cutoff, device=device)
        # embeddings = get_embeddings(atoms_batches, single_calc)
        # force_std_devs = get_std_devs(atoms_batches, ensemble_calc)

        # # delete the AtomsBatch to free up memory
        # del atoms_batches

        y = perform_clustering(
            embeddings, clustering_cutoff, cutoff_criterion, save_folder, save_prepend
        )
        select_data_and_save(
            dset_batch, y, force_std_devs, use_force_std, save_folder, save_prepend
        )

    print("Clustering complete!")


if __name__ == "__main__":
    args = parse_args()
    if torch.cuda.is_available() and "cpu" not in args.nff_device:
        nff_device = f"cuda:{cuda_devices_sorted_by_free_mem()[-1]}"  # take the device with the most free memory
    else:
        nff_device = "cpu"

    print(f"Using {nff_device} device for NFF calculations")

    main(
        args.file_paths,
        nff_cutoff=args.nff_cutoff,
        device=nff_device,
        model_type=args.nff_model_type,
        clustering_cutoff=args.clustering_cutoff,
        max_input_len=args.max_input_len,
        use_force_std=args.use_force_std,
        nff_paths=args.nff_paths,
        cutoff_criterion=args.cutoff_criterion,
        save_folder=args.save_folder,
    )
