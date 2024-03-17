import argparse
import os
import pickle as pkl
from pathlib import Path
from typing import List, Union

import ase
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ase.calculators.calculator import Calculator
from nff.data import Dataset
from nff.io.ase import AtomsBatch
from nff.io.ase_calcs import EnsembleNFF, NeuralFF
from nff.utils.cuda import cuda_devices_sorted_by_free_mem
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.decomposition import PCA


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
    parser.add_argument("--save_folder", help="Save folder path", type=str, default="")
    # parser.add_argument("--painn_params_file", help="PaiNN parameter file", type=str, default="painn_params.json")
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


def get_atoms_batches(
    data: Union[Dataset, List[ase.Atoms]],
    nff_cutoff: float,
    device: str = "cpu",
    **kwargs,
):
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

    else:
        atoms_batches = [
            AtomsBatch.from_atoms(
                x,
                cutoff=nff_cutoff,
                requires_large_offsets=True,
                directed=True,
                device=device,
                **kwargs,
            )
            for x in data
        ]

    return atoms_batches


def get_embeddings(atoms_batches: List[AtomsBatch], calc: Calculator):
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
    for atoms_batch in atoms_batches:
        atoms_batch.calc = calc
        calc.calculate(atoms_batch)
        breakpoint()

        embeddings.append(calc.results["embedding"].sum(axis=0).detach().cpu().numpy())

    embeddings = np.stack(embeddings)
    return embeddings


def get_std_devs(atoms_batches: List[AtomsBatch], calc: Calculator):
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
    force_std = []
    for atoms_batch in atoms_batches:
        atoms_batch.calc = calc
        calc.calculate(atoms_batch)
        breakpoint()

        if "forces_std" in calc.results:
            force_std.append(calc.results["forces_std"].mean())
        else:
            force_std.append(0.0)

    force_std = np.stack(force_std)
    return force_std


def perform_clustering(
    embeddings: np.ndarray,
    clustering_cutoff: Union[int, float],
    cutoff_criterion: str = "distance",
    save_folder: str = "./",
    save_prepend: str = "",
    **kwargs,
):
    """Perform clustering on the embeddings using PCA and hierarchical clustering. Either distance or maxclust can be used as the cutoff criterion.

    Parameters
    ----------
    embeddings : np.ndarray
        Latent space embeddings with each row corresponding to a structure
    clustering_cutoff : Union[int, float]
        Either the distance or the maximum number of clusters
    cutoff_criterion : str, optional
        'distance' or 'maxclust', by default 'distance'
    save_folder : str, optional
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

    plt.figure()
    plt.scatter(X_r[:, 0], X_r[:, 1], s=2, alpha=0.8, color=plt.cm.inferno(0.5))
    plt.title("PCA of PaiNN embeddings")
    plt.savefig(os.path.join(save_folder, save_prepend + "painn_embeddings_pca.png"))
    plt.show()

    print(f"X_r has shape {X_r.shape}")
    print(f"X has shape {X.shape}")

    print(f"The first pca explained ratios are {pca.explained_variance_ratio_[:5]}")

    # Perform hierarchical clustering
    Z = linkage(X_r[:, :3], method="ward", metric="euclidean", optimal_ordering=True)

    # plot dendrogram
    fig = plt.figure(figsize=(25, 10), dpi=200)
    dn = dendrogram(Z, no_labels=True)
    plt.savefig(os.path.join(save_folder, save_prepend + "dendrogram_Z.png"))
    plt.show()

    # t sets the distance
    if cutoff_criterion == "distance":
        y = fcluster(Z, t=clustering_cutoff, criterion="distance", depth=2)
    else:
        y = fcluster(Z, t=clustering_cutoff, criterion="maxclust", depth=2)
    # breakpoint()

    clusters = np.unique(y)
    max_index = np.max(clusters)

    print(f"There are {len(clusters)} clusters")

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

    return y


def select_data_and_save(
    atoms_batches: List[AtomsBatch],
    y: np.ndarray,
    force_std: np.ndarray,
    use_force_std: bool = True,
    save_folder: str = "./",
    save_prepend: str = "",
):
    """Select the highest variance structure from each cluster and save the corresponding AtomsBatch objects

    Parameters
    ----------
    atoms_batches : List[AtomsBatch]
        List of AtomsBatch objects
    y : np.ndarray
        Each element corresponds to the cluster number of the corresponding structure
    force_std : np.ndarray
        Force standard deviations with each element corresponding to a structure
    use_force_std : bool, optional
        Use force standard deviation to select structures, by default True
    save_folder : str, optional
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

    print("before selection")
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

    print("after selection")
    print(sorted_std_df.head())

    print(
        f"cluster: {y[sorted_std_df['index'].iloc[0]]} force std: {force_std[sorted_std_df['index'].iloc[0]]}"
    )

    selected_indices = sorted_std_df["index"].to_numpy()

    selected_atoms = [atoms_batches[x] for x in selected_indices.tolist()]

    print(f"saving {len(selected_atoms)} AtomsBatch objects")

    if len(selected_atoms) >= 1:
        clustered_atom_files = os.path.join(save_folder, save_prepend + "clustered.pkl")
        with open(clustered_atom_files, "wb") as f:
            pkl.dump(selected_atoms.copy(), f)
    # else:
    #     clustered_atom_files = os.path.join(save_folder, save_prepend +  'clustered.pth.tar')
    #     Dataset(concatenate_dict(*selected_atoms)).save(clustered_atom_files)

    print(f"saved to {clustered_atom_files}")


def main(
    file_names: List[str],
    save_folder: Union[Path, str],
    nff_cutoff: float = 5.0,
    device: str = "cuda:0",
    clustering_cutoff: Union[int, float] = 0.2,
    max_input_len: int = 1000,
    use_force_std: bool = True,
    nff_paths: List[Path, str] = None,
    cutoff_criterion: str = "distance",
):
    """Main function to perform clustering on a list of structures

    Parameters
    ----------
    file_names : List[str]
        List of file paths to load structures from
    save_folder : Union[Path, str]
        Folder to save the plots
    nff_cutoff : float, optional
        Neighbor cutoff for the NFF model, by default 5.0
    device : str, optional
        cpu or cuda device, by default 'cuda:0'
    clustering_cutoff : Union[int, float], optional
        Either the distance or the maximum number of clusters, by default 0.2
    max_input_len : int, optional
        Maximum number of structures used in each clustering iteration, by default 1000
    use_force_std : bool, optional
        Use force standard deviation to select structures, by default True
    nff_paths : List[Path, str], optional
        Full path to NFF model, by default None
    cutoff_criterion : str, optional

    Returns
    -------
    None
    """

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    file_paths = [Path(file_name) for file_name in file_names]
    print(f"There are a total of {len(file_paths)} input files")
    file_base = file_paths[0].resolve().stem

    dset = []
    for x in file_paths:
        if x.suffix == ".pkl":
            f = open(x, "rb")
            dset.extend(pkl.load(f))
        else:  # .pth.tar
            data = Dataset.from_file(x)
            dset.extend(data)

    if nff_paths:
        # Load existing models
        print(f"Loading existing models from {nff_paths}")
        models = []
        for nff_path in nff_paths:
            m = NeuralFF.from_file(nff_path, device=device).model
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
        models, device=device, model_units="eV/atom", prediction_units="eV"
    )
    # NeuralFF for latent space embedding calculation
    single_calc = NeuralFF(
        models[0], device=device, model_units="eV/atom", prediction_units="eV"
    )

    # Perform clustering in batches
    num_batches = len(dset) // max_input_len + 1  # +1 to account for the remainder
    for i in range(num_batches):

        dset_batch = (
            dset[i * max_input_len : (i + 1) * max_input_len]
            if i < num_batches - 1
            else dset[i * max_input_len :]
        )
        batch_number = i + 1
        print(f"starting clustering for batch # {batch_number}")

        cluster_selection_metric = "force_std" if use_force_std else "random"
        save_prepend = (
            file_base
            + f"_{len(data)}_input_structures"
            + f"_batch_{batch_number}".zfill(3)
            + f"_cutoff_{clustering_cutoff}_"
            + f"{cluster_selection_metric}_"
        )
        atoms_batches = get_atoms_batches(dset_batch, nff_cutoff)
        embeddings = get_embeddings(atoms_batches, ensemble_calc)
        force_std_devs = get_std_devs(atoms_batches, single_calc)
        y = perform_clustering(
            embeddings, clustering_cutoff, cutoff_criterion, save_folder, save_prepend
        )
        select_data_and_save(
            atoms_batches, y, force_std_devs, use_force_std, save_folder, save_prepend
        )


if __name__ == "__main__":
    args = parse_args()
    if torch.cuda.is_available():
        nff_device = f"cuda:{cuda_devices_sorted_by_free_mem()[-1]}"  # take the device with the most free memory
    else:
        nff_device = "cpu"

    main(
        args.file_paths,
        args.save_folder,
        nff_cutoff=args.nff_cutoff,
        device=nff_device,
        clustering_cutoff=args.clustering_cutoff,
        max_input_len=args.max_input_len,
        use_force_std=args.use_force_std,
        nff_paths=args.nff_paths,
        cutoff_criterion=args.cutoff_criterion,
    )
