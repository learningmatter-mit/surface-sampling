"""Clustering of structures based on their latent space embeddings."""

import argparse
from datetime import datetime
from logging import getLevelNamesMapping
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from nff.io.ase_calcs import EnsembleNFF, NeuralFF
from nff.train.builders import load_model
from nff.utils.cuda import cuda_devices_sorted_by_free_mem
from tqdm import tqdm

from mcmc.calculators import get_embeddings_single, get_results_single, get_std_devs_single
from mcmc.utils import setup_logger
from mcmc.utils.clustering import perform_clustering, select_data_and_save
from mcmc.utils.misc import get_atoms_batch, load_dataset_from_files

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
        choices=["PaiNN", "NffScaleMACE", "CHGNetNFF"],
        default="CHGNetNFF",
        help="NFF model type",
        type=str,
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
    logger.info("Using %s device for NFF calculations", device)

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
            embedding = get_embeddings_single(
                atoms_batch,
                single_calc,
                results_cache=single_calc_results,
                flatten=True,
                flatten_axis=0,
            )
            if clustering_metric == "energy":
                metric_value = single_calc_results["energy"].squeeze()
            elif clustering_metric == "force_std":
                metric_value = get_std_devs_single(atoms_batch, ensemble_calc)
            else:
                metric_value = np.random.rand()
            embeddings.append(embedding)
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
