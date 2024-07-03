"""Surface sampling across compositions and configurations using VSSR-MC."""

import argparse
import json
import pickle
from logging import getLevelNamesMapping
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
import pandas as pd
from monty.serialization import dumpfn, loadfn
from nff.train.builders.model import load_model
from nff.utils.cuda import cuda_devices_sorted_by_free_mem

from mcmc import MCMC
from mcmc.calculators import EnsembleNFFSurface
from mcmc.system import SurfaceSystem
from mcmc.utils import setup_logger
from mcmc.utils.misc import get_atoms_batch
from mcmc.utils.plot import plot_summary_stats
from mcmc.utils.setup import setup_folders

np.set_printoptions(precision=3, suppress=True)

DEFAULT_CUTOFFS = {
    "CHGNetNFF": 6.0,
    "NffScaleMACE": 5.0,
    "PaiNN": 5.0,
}

DEFAULT_SAMPLING_SETTINGS = {
    "canonical": False,
    "total_sweeps": 100,
    "sweep_size": 20,
    "start_temp": 1.0,  # in terms of kT
    "perform_annealing": False,
    "alpha": 1.0,
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Perform VSSR-MC sampling on surfaces.")

    parser.add_argument(
        "--run_name",
        type=str,
        default="SrTiO3_001_2x2",
        help="Name of the run",
    )
    parser.add_argument(
        "--starting_structure_path",
        type=str,
        help="Path to the starting structure",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["PaiNN", "NffScaleMACE", "CHGNetNFF"],
        default="CHGNetNFF",
        help="Type of NFF machine learning force field model to use",
    )
    parser.add_argument(
        "--model_paths",
        type=str,
        nargs="*",
        default=[""],
        help="Space separated paths to NFF models",
    )
    parser.add_argument(
        "--settings_path",
        type=str,
        default="settings.json",
        help="Path to the settings file",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device to use for calculations",
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
    run_name: str,
    starting_structure_path: Path | str,
    model_type: Literal["PaiNN", "NffScaleMACE", "CHGNetNFF"],
    model_paths: list[str],
    settings_path: str = "settings.json",
    device: Literal["cpu", "cuda"] = "cuda",
    logging_level: Literal["debug", "info", "warning", "error", "critical"] = "info",
):
    """Perform VSSR-MC sampling for surfaces.

    Args:
        run_name (str): name of the run
        starting_structure_path (Path | str): path to the starting structure
        model_type (Literal["PaiNN", "NffScaleMACE", "CHGNetNFF"]): type of NFF model to use
        model_paths (list[str]): paths to the models
        settings_path (str, optional): path to the settings file. Defaults to "settings.json"
        device (Literal["cpu", "cuda"], optional): device to use for calculations.
            Defaults to "cuda"
        logging_level (Literal["debug", "info", "warning", "error", "critical"], optional): logging
    """
    # Load settings
    all_settings = loadfn(settings_path)
    calc_settings, system_settings, sampling_settings = (
        all_settings["calc_settings"],
        all_settings["system_settings"],
        all_settings["sampling_settings"],
    )

    # Update empty settings with default params
    system_settings["surface_name"] = system_settings.get("surface_name", run_name)
    system_settings["cutoff"] = system_settings.get("cutoff", DEFAULT_CUTOFFS[model_type])
    sampling_settings = DEFAULT_SAMPLING_SETTINGS | sampling_settings

    # Initialize run folder
    if not sampling_settings.get("run_folder"):
        run_folder = setup_folders(
            system_settings["surface_name"],
            canonical=sampling_settings["canonical"],
            total_sweeps=sampling_settings["total_sweeps"],
            start_temp=sampling_settings["start_temp"],
            alpha=sampling_settings["alpha"],
        )
        sampling_settings["run_folder"] = run_folder
    else:
        run_folder = Path(sampling_settings["run_folder"])
        run_folder.mkdir(parents=True, exist_ok=True)

    # Initialize logger
    logger = setup_logger(
        "mcmc", run_folder / "mc.log", level=getLevelNamesMapping()[logging_level.upper()]
    )

    # Load offset data if offset path is provided
    if "offset_data" in calc_settings:
        offset_data = calc_settings["offset_data"]
        if isinstance(offset_data, str | Path):
            try:
                with open(offset_data, "r", encoding="utf-8") as f:
                    offset_data = json.load(f)
                    calc_settings["offset_data"] = offset_data
            except FileNotFoundError as e:
                logger.error("Offset data file not found at the provided path.")
                raise e

    # Save updated run settings
    all_settings = {
        "system_settings": system_settings,
        "sampling_settings": sampling_settings,
        "calc_settings": calc_settings,
    }
    dumpfn(all_settings, run_folder / "settings.json", indent=4)

    # Load prepared pristine slab
    try:
        with open(starting_structure_path, "rb") as f:
            starting_slab = pickle.load(f)
    except FileNotFoundError as e:
        logger.error("Pristine surface pkl file not found.")
        raise e

    # Initialize Calculator
    device = (
        f"cuda:{cuda_devices_sorted_by_free_mem()[-1]}" if device == "cuda" else "cpu"
    )  # get the gpu with most free memory
    models = []
    for model_path in model_paths:
        model = load_model(model_path, model_type=model_type, map_location=device)
        models.append(model)
    nff_surf_calc = EnsembleNFFSurface(
        models,
        device=device,
        model_units=models[0].units if model_type != "PaiNN" else "kcal/mol",
        prediction_units="eV",
        offset_units="eV" if model_type != "PaiNN" else "atomic",
    )
    nff_surf_calc.set(**calc_settings)

    # Initialize SurfaceSystem
    slab_batch = get_atoms_batch(
        starting_slab,
        nff_cutoff=system_settings["cutoff"],
        device=device,
        props={"energy": 0, "energy_grad": []},  # needed for NFF
    )
    surface = SurfaceSystem(
        slab_batch,
        calc=nff_surf_calc,
        system_settings=system_settings,
        save_folder=run_folder,
    )
    surface.all_atoms.write(run_folder / "all_virtual_ads.cif")
    logger.info("Starting surface energy: %.3f eV", float(surface.get_surface_energy()))
    logger.info("Offset units: %s", nff_surf_calc.offset_units)

    # Perform MCMC
    mcmc = MCMC(**sampling_settings)
    start = perf_counter()
    results = mcmc.run(
        surface=surface,
        **sampling_settings,
    )
    stop = perf_counter()
    logger.info("Time taken = %.3f seconds", stop - start)

    # Save SurfaceSystem objects for later use in latent space clustering or analysis
    structures = results["history"]
    with open(run_folder / f"{len(structures)}_mcmc_structures.pkl", "wb") as f:
        pickle.dump(structures, f)
    logger.info("Saving all %d surfaces", len(structures))

    # Save relaxation trajectories
    trajectories = results["trajectories"]
    traj_structures = [traj_info["atoms"] for traj_info in trajectories]
    traj_structures = [
        item for sublist in traj_structures for item in sublist
    ]  # flatten nested list
    with open(run_folder / f"{len(traj_structures)}_relaxation_structures.pkl", "wb") as f:
        pickle.dump(traj_structures, f)
    logger.info("Saving all %d slabs in relaxation trajectories", len(traj_structures))

    # Save statistics in csv
    stats_df = pd.DataFrame(
        {
            "energy": results["energy_hist"],
            "frac_accept": results["frac_accept_hist"],
            "adsorption_count": results["adsorption_count_hist"],
        }
    )
    stats_df.to_csv(run_folder / "stats.csv", index=False, float_format="%.3f")
    logger.info("Saving statistics in csv")

    # Plot statistics
    plot_summary_stats(
        results["energy_hist"],
        results["frac_accept_hist"],
        results["adsorption_count_hist"],
        mcmc.total_sweeps,
        save_folder=run_folder,
    )


if __name__ == "__main__":
    args = parse_args()
    main(
        args.run_name,
        args.starting_structure_path,
        args.model_type,
        args.model_paths,
        args.settings_path,
        args.device,
        args.logging_level,
    )
