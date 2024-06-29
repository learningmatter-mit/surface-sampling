"""Script to sample SrIrO3 bulk system using VSSR-MC."""

import argparse
import json
import logging
import pickle
from copy import deepcopy
from pathlib import Path
from time import perf_counter

from nff.io.ase_calcs import NeuralFF
from nff.nn.models.chgnet import CHGNetNFF
from nff.utils.cuda import cuda_devices_sorted_by_free_mem

from mcmc import MCMC
from mcmc.calculators import EnsembleNFFSurface
from mcmc.system import SurfaceSystem
from mcmc.utils.misc import get_atoms_batch

logger = logging.getLogger("mcmc")
logger.setLevel(logging.INFO)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="perform mcmc")
    parser.add_argument(
        "--starting_structure",
        type=str,
        default="data/SrIrO3_2x2x2_bulk.pkl",
        help="path to the starting structure",
    )
    parser.add_argument(
        "--save_folder",
        type=Path,
        default="./",
        help="Folder to save sampled surfaces.",
    )
    parser.add_argument(
        "--nnids",
        type=int,
        nargs="+",
        default=[2537, 2538, 2539],
        help="ids of the nnpotentials to use",
    )
    parser.add_argument(
        "--chem_pot",
        type=float,
        nargs="+",
        default=[0, 0, 0],
        help="chemical potential for each element",
    )
    parser.add_argument("--sweeps", type=int, default=100, help="MCMC sweeps")
    parser.add_argument("--sweep_size", type=int, default=50, help="MCMC sweep size")
    parser.add_argument("--temp", type=float, default=1.0, help="temperature in kbT")
    parser.add_argument("--relax", action="store_true", help="perform relaxation for the steps")
    parser.add_argument("--relax_steps", type=int, default=5, help="max relaxation steps")
    parser.add_argument(
        "--record_interval", type=int, default=5, help="record interval for relaxation"
    )
    parser.add_argument("--offset", action="store_true", help="whether to use energy offsets")
    parser.add_argument(
        "--offset_data",
        type=str,
        default="/mnt/data0/dux/surf_samp_working/SrIrO3/data/nff/chgnet_offset_data_eV.json",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="device to use for calculations",
    )

    return parser.parse_args()


def main(
    starting_structure: Path | str,
    save_folder: Path,
    nnids: list[int],
    chem_pot: list[float],
    sweeps: int,
    sweep_size: int,
    temp: float,
    relax: bool,
    offset_data_path: str,
    relax_steps: int = 20,
    record_interval: int = False,
    offset: bool = False,
    device: str = "cuda",
):
    """Perform VSSR-MC sampling for SrIrO3 bulk system.

    Args:
        starting_structure (Path | str): path to the starting structure
        save_folder (Path): Folder to save sampled surfaces.
        nnids (list[int]): ids of the nnpotentials to use
        chem_pot (list[float]): chemical potential for each element
        sweeps (int): MCMC sweeps
        sweep_size (int): MCMC steps (iterations) per sweep
        temp (float): temperature in kbT
        relax (bool): perform relaxation for the steps
        offset_data_path (str): path to the offset data
        relax_steps (int, optional): max relaxation steps. Defaults to 20.
        record_interval (int, optional): save interval for relaxation. Defaults to False.
        offset (bool, optional): whether to use energy offsets. Defaults to False.
        device (str, optional): device to use for calculations. Defaults to "cuda".
    """
    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)

    DEVICE = f"cuda:{cuda_devices_sorted_by_free_mem()[-1]}" if device == "cuda" else "cpu"
    try:
        with open(offset_data_path, "r") as f:
            offset_data = json.load(f)
    except FileNotFoundError as e:
        logger.error("Could not find offset data at %s", offset_data_path)
        raise e

    system_settings = {
        "surface_name": "SrIrO3_bulk",
        "cutoff": 5.0,
        "device": DEVICE,
        # "near_reduce": 0.01,
        # "planar_distance": 1.55,
        # "no_obtuse_hollow": True,
    }

    sampling_settings = {
        "alpha": 1.0,  # no annealing
        "temperature": temp,  # in terms of kbT, 1000 K
        "num_sweeps": sweeps,
        "sweep_size": sweep_size,
    }

    calc_settings = {
        "calc_name": "NFF",
        "chem_pots": {
            "Sr": chem_pot[0],
            "Ir": chem_pot[1],
            "O": chem_pot[2],
        },
        "offset_data": offset_data,
        "optimizer": "BFGS",
        "relax_atoms": relax,
        "relax_steps": relax_steps,
        "record_interval": record_interval,  # record structure every n steps
        "offset": offset,
    }

    # open 2x2x2 cubic from file
    with open(starting_structure, "rb") as f:
        cubic_cell_2x2x2 = pickle.load(f)

    chem_symbols = cubic_cell_2x2x2.get_chemical_symbols()
    print(f"Chemical symbols: {chem_symbols}")

    # Use Pretrained NFF model
    chgnet_nff = CHGNetNFF.load(device=system_settings["device"])

    nff_calc = NeuralFF(
        chgnet_nff,
        device=system_settings["device"],
        model_units="eV/atom",
        prediction_units="eV",
    )

    nff_surf_calc = EnsembleNFFSurface(
        [nff_calc.model],
        device=system_settings["device"],
        model_units="eV/atom",
        prediction_units="eV",
        offset_units="eV",
    )
    nff_surf_calc.set(**calc_settings)

    # Initialize SurfaceSystem (actually bulk system)
    ads_positions = cubic_cell_2x2x2.get_positions()
    ads_idx = list(range(len(ads_positions)))

    # remove the first ads site due to 0 index
    # 0 index will always be thought of as being empty
    # TODO might have to fix this
    ads_positions = ads_positions[1:]
    ads_idx = ads_idx[1:]

    # set attributes
    slab_batch = get_atoms_batch(
        cubic_cell_2x2x2,
        nff_cutoff=system_settings["cutoff"],
        device=system_settings["device"],
        props={"energy": 0, "energy_grad": []},  # needed for NFF
    )
    # slab_batch = AtomsBatch(
    #     cubic_cell_2x2x2,
    #     cutoff=system_settings["cutoff"],
    #     props={"energy": 0, "energy_grad": []},
    #     calculator=nff_surf_calc,
    #     requires_large_offsets=True,
    #     directed=True,
    #     device=DEVICE,
    # )

    # Fake as adsorbate slab
    slab_batch.set_tags([1] * len(slab_batch))
    print(f"Tags are {slab_batch.get_tags()}")

    bulk = SurfaceSystem(
        slab_batch,
        calc=nff_surf_calc,
        ads_coords=ads_positions,
        occ=ads_idx,
        system_settings=system_settings,
    )
    bulk.all_atoms.write(save_path / "SrTiO3_2x2_bulk_starting.cif")

    print(f"Starting chemical formula {slab_batch.get_chemical_formula()}")

    print(f"NFF calc predicts {nff_calc.get_potential_energy(slab_batch)} eV")

    print(f"Offset units: {nff_surf_calc.offset_units}")

    # Do different bulk defect sampling for the 2x2x2 cell
    # Sample across chemical potentials
    starting_bulk = deepcopy(bulk)

    # Perform MCMC and view results.
    mcmc = MCMC(
        system_settings["surface_name"],
        calc=nff_surf_calc,
        canonical=False,
        testing=False,
        element=[],
        adsorbates=list(calc_settings["chem_pots"].keys()),
        relax=calc_settings["relax_atoms"],
        relax_steps=calc_settings["relax_steps"],
        offset=calc_settings["offset"],
        offset_data=calc_settings["offset_data"],
        optimizer=calc_settings["optimizer"],
    )

    start = perf_counter()
    # Call the main function
    mcmc.mcmc_run(
        total_sweeps=sampling_settings["num_sweeps"],
        sweep_size=sampling_settings["sweep_size"],
        start_temp=sampling_settings["temperature"],
        pot=list(calc_settings["chem_pots"].values()),
        alpha=sampling_settings["alpha"],
        surface=starting_bulk,
        run_folder=save_path,
    )
    stop = perf_counter()
    print(f"Time taken = {stop - start} seconds")

    # Save structures for later use in latent space clustering or analysis
    relaxed_structures = mcmc.history
    print(f"saving all relaxed structures with length {len(relaxed_structures)}")

    with open(f"{mcmc.run_folder}/full_run_{len(relaxed_structures)}.pkl", "wb") as f:
        pickle.dump(relaxed_structures, f)

    relax_trajectories = mcmc.trajectories
    traj_structures = [traj_info["atoms"] for traj_info in relax_trajectories]

    # Flatten list of lists
    traj_structures = [item for sublist in traj_structures for item in sublist]
    print(f"saving all structures in relaxation paths with length {len(traj_structures)}")

    with open(f"{mcmc.run_folder}/relax_traj_{len(traj_structures)}.pkl", "wb") as f:
        pickle.dump(traj_structures, f)


if __name__ == "__main__":
    args = parse_args()
    main(
        args.starting_structure,
        args.save_folder,
        args.nnids,
        args.chem_pot,
        args.sweeps,
        args.sweep_size,
        args.temp,
        args.relax,
        args.offset_data,
        args.relax_steps,
        args.record_interval,
        args.offset,
        args.device,
    )
