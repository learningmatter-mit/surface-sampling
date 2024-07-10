"""Surface sampling under aqueous electrochemical conditions using VSSR-MC."""

import argparse
import datetime
import logging
import pickle
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
from monty.serialization import dumpfn, loadfn
from nff.train.builders.model import load_model
from nff.utils.cuda import get_final_device
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.ase import AseAtomsAdaptor

from mcmc import MCMC
from mcmc.calculators import NFFPourbaix
from mcmc.pourbaix.atoms import generate_pourbaix_atoms
from mcmc.system import SurfaceSystem
from mcmc.utils.misc import get_atoms_batch

logger = logging.getLogger("mcmc")
logger.setLevel(logging.INFO)

np.set_printoptions(precision=3, suppress=True)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Perform MCMC on surfaces under electrochemical conditions."
    )
    parser.add_argument(
        "--surface_name",
        type=str,
        default="SrIrO3_001_2x2",
        help="name of the surface",
    )
    parser.add_argument(
        "--starting_structure_path",
        type=str,
        help="path to the starting structure",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["NffScaleMACE", "CHGNetNFF"],
        default="NffScaleMACE",
        help="type of model to use",
    )
    parser.add_argument(
        "--model_paths",
        type=str,
        nargs="*",
        default=[""],
        help="paths to the models",
    )
    parser.add_argument(
        "--phase_diagram_path",
        type=str,
        help="path to the saved pymatgen PhaseDiagram",
    )
    parser.add_argument(
        "--pourbaix_diagram_path",
        type=str,
        help="path to the saved pymatgen PourbaixDiagram",
    )
    parser.add_argument(
        "--save_folder",
        type=Path,
        default="./",
        help="Folder to output.",
    )
    parser.add_argument(
        "--chem_pot",
        type=float,
        nargs="+",
        default=[0, 0, 0],
        help="chemical potential for each element",
    )
    parser.add_argument("--phi", type=float, default=0.0, help="electrical potential")
    parser.add_argument("--pH", type=float, default=7.0, help="pH")
    parser.add_argument("--sweeps", type=int, default=100, help="MCMC sweeps")
    parser.add_argument("--sweep_size", type=int, default=50, help="MCMC sweep size")
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="annealing parameter, fraction to multiply with temperature",
    )
    parser.add_argument("--samp_temp", type=float, default=1.0, help="sampling temperature in kbT")
    parser.add_argument("--system_temp", type=float, default=1.0, help="system temperature in kbT")
    parser.add_argument("--relax", action="store_true", help="perform relaxation for the steps")
    parser.add_argument("--relax_steps", type=int, default=5, help="max relaxation steps")
    parser.add_argument(
        "--surface_depth",
        type=int,
        default=1,
        help="layers of atoms from the surface to relax",
    )
    parser.add_argument(
        "--ads_site_planar_distance",
        type=float,
        default=2.0,
        help="distance of adsorption sites from surface",
    )
    parser.add_argument(
        "--ads_site_type",
        choices=["ontop", "bridge", "hollow", "all"],
        default="all",
        help="type of adsorption sites to include",
    )
    parser.add_argument(
        "--record_interval", type=int, default=5, help="record interval for relaxation"
    )
    parser.add_argument(
        "--neighbor_cutoff",
        type=float,
        default=5.0,
        help="cutoff for neighbor calculations",
    )
    parser.add_argument("--offset", action="store_true", help="whether to use energy offsets")
    parser.add_argument(
        "--offset_data_path",
        type=str,
        default="data/nff/offset_data.json",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="device to use for calculations",
    )

    return parser.parse_args()


def main(
    surface_name: str,
    starting_structure_path: Path | str,
    model_type: Literal["NffScaleMACE", "CHGNetNFF"],
    model_paths: list[str],
    phase_diagram_path: Path | str,
    pourbaix_diagram_path: Path | str,
    chem_pot: list[float],
    phi: float,
    pH: float,
    system_temp: float,
    sweeps: int,
    sweep_size: int,
    alpha: float,
    samp_temp: float,
    relax: bool,
    relax_steps: int = 20,
    surface_depth: int = 1,
    ads_site_planar_distance: float = 2.0,
    ads_site_type: Literal["ontop", "bridge", "hollow", "all"] = "all",
    record_interval: int = False,
    neighbor_cutoff: float = 5.0,
    offset: bool = False,
    offset_data_path: str = "",
    device: str = "cuda",
    save_folder: str = "./",
) -> None:
    """Perform VSSR-MC sampling for surfaces.

    Args:
        surface_name (str): name of the surface
        starting_structure_path (Union[Path, str]): path to the starting structure
        model_type (Literal["NffScaleMACE", "CHGNetNFF"]): type of model to use
        model_paths (List[str]): paths to the models
        phase_diagram_path (Union[Path, str]): path to the saved pymatgen PhaseDiagram
        pourbaix_diagram_path (Union[Path, str]): path to the saved pymatgen PourbaixDiagram
        chem_pot (List[float]): chemical potential for each element
        phi (float): electrical potential
        pH (float): pH
        system_temp (float): system temperature in kbT
        sweeps (int): MCMC sweeps
        sweep_size (int): MCMC sweep size
        alpha (float): alpha for MCMC
        samp_temp (float): sampling temperature in kbT
        relax (bool): perform relaxation for the steps
        relax_steps (int): max relaxation steps
        surface_depth (int): layers of atoms from the surface to relax
        ads_site_planar_distance (float): distance of adsorption sites from surface
        ads_site_type (Literal["ontop", "bridge", "hollow", "all"]): type of adsorption sites to
            include
        record_interval (int): record interval for relaxation
        neighbor_cutoff (float): cutoff for neighbor calculations
        offset (bool): whether to use energy offsets
        offset_data_path (str): path to offset data
        device (str): device to use for calculations
        save_folder (Path): Folder to output
    """
    start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_folder = save_folder / f"{start_time}_{surface_name}"

    run_path = Path(run_folder)
    run_path.mkdir(parents=True, exist_ok=True)

    device = get_final_device(device)

    logging.info("Using device: %s", device)

    with open(starting_structure_path, "rb") as f:
        starting_slab = pickle.load(f)

    chem_symbols = starting_slab.get_chemical_symbols()
    elements = list(set(chem_symbols))

    logger.info("Elements: %s", elements)

    pourbaix_atoms = generate_pourbaix_atoms(
        phase_diagram_path, pourbaix_diagram_path, phi, pH, elements
    )
    logger.info("Pourbaix atoms: %s", pourbaix_atoms)
    offset_data = loadfn(offset_data_path, "r") if offset else None

    system_settings = {
        "surface_name": surface_name,
        "cutoff": neighbor_cutoff,
        "device": device,
        "near_reduce": 0.01,
        "planar_distance": ads_site_planar_distance,  # 2.0 (default)
        "no_obtuse_hollow": True,
        "ads_site_type": ads_site_type,  # ontop, bridge, hollow, all
    }

    sampling_settings = {
        "alpha": alpha,  # no annealing
        "temperature": samp_temp,  # in terms of kbT, 1000 K
        "num_sweeps": sweeps,
        "sweep_size": sweep_size,
    }

    calc_settings = {
        "calc_name": "NFF",
        "optimizer": "FIRE",
        "chem_pots": dict(zip(elements, chem_pot, strict=False)),
        "relax_atoms": relax,
        "relax_steps": relax_steps,
        "record_interval": record_interval,  # record structure every n steps
        "offset": offset,
        "temperature": system_temp,
        "pH": pH,
        "phi": phi,
        "pourbaix_atoms": pourbaix_atoms,
        "offset_data": offset_data,
    }
    # TODO: make a load settings file as well as an alternative
    # Save settings to file
    all_settings = {
        "system_settings": system_settings,
        "sampling_settings": sampling_settings,
        "calc_settings": calc_settings,
    }
    dumpfn(all_settings, run_folder / "settings.json")

    # Obtain adsorption sites
    pristine_slab = starting_slab.copy()
    pristine_pmg_slab = AseAtomsAdaptor.get_structure(pristine_slab)
    site_finder = AdsorbateSiteFinder(pristine_pmg_slab)

    all_ads_positions = site_finder.find_adsorption_sites(
        put_inside=True,
        symm_reduce=False,
        near_reduce=system_settings["near_reduce"],
        distance=system_settings["planar_distance"],
        no_obtuse_hollow=system_settings["no_obtuse_hollow"],
    )
    ads_positions = all_ads_positions[system_settings["ads_site_type"]]
    logger.info("Generated adsorption coordinates are: %s...", ads_positions[:5])

    surf_atom_idx = pristine_slab.get_surface_atoms()
    surf_atom_positions = pristine_slab.get_positions()[surf_atom_idx]
    logger.info("Surface atom coordinates are: %s...", surf_atom_positions[:5])
    all_ads_coords = np.vstack([ads_positions, surf_atom_positions])

    occ = np.hstack([[0] * len(ads_positions), surf_atom_idx])

    # Load Ensemble NFF Model
    models = []
    for model_path in model_paths:
        model = load_model(model_path, model_type=model_type, map_location=device)
        models.append(model)

    # TODO write support for multiple models
    nff_surf_calc = NFFPourbaix(
        models[0],
        device=device,
        model_units=models[0].units,
        prediction_units="eV",
    )
    nff_surf_calc.set(**calc_settings)

    # Initialize SurfaceSystem
    slab_batch = get_atoms_batch(
        pristine_slab,
        nff_cutoff=system_settings["cutoff"],
        device=system_settings["device"],
        props={"energy": 0, "energy_grad": []},  # needed for NFF
    )
    surface = SurfaceSystem(
        slab_batch,
        all_ads_coords,
        calc=nff_surf_calc,
        occ=occ,
        surface_depth=surface_depth,
        system_settings=system_settings,
    )
    starting_atoms_path = run_folder / "all_virtual_ads.cif"
    logger.info("Saving surface with virtual atoms to %s", starting_atoms_path)
    surface.all_atoms.write(starting_atoms_path)

    logger.info("Starting chemical formula: %s", surface.real_atoms.get_chemical_formula())

    logger.info("NFF calc starting energy: %.3f eV", surface.get_potential_energy())

    logger.info("Starting surface energy: %.3f eV", surface.get_surface_energy())

    if hasattr(nff_surf_calc, "offset_units"):
        logger.info("Offset units: %s", nff_surf_calc.offset_units)

    # Do different bulk defect sampling for the 2x2x2 cell
    # Sample across chemical potentials
    starting_surface = deepcopy(surface)
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
        surface=starting_surface,
        run_folder=run_folder,
    )
    stop = perf_counter()
    logger.info("time taken = %.3f seconds", stop - start)

    # Save structures for later use in latent space clustering or analysis
    relaxed_structures = mcmc.history
    logger.info("saving all relaxed structures with length %d", len(relaxed_structures))

    with open(f"{mcmc.run_folder}/full_run_{len(relaxed_structures)}.pkl", "wb") as f:
        pickle.dump(relaxed_structures, f)

    relax_trajectories = mcmc.trajectories
    traj_structures = [traj_info["atoms"] for traj_info in relax_trajectories]

    # Flatten list of lists
    traj_structures = [item for sublist in traj_structures for item in sublist]
    logger.info("saving all structures in relaxation paths with length %d", len(traj_structures))

    with open(f"{mcmc.run_folder}/relax_traj_{len(traj_structures)}.pkl", "wb") as f:
        pickle.dump(traj_structures, f)


if __name__ == "__main__":
    args = parse_args()
    main(
        args.surface_name,
        args.starting_structure_path,
        args.model_type,
        args.model_paths,
        args.phase_diagram_path,
        args.pourbaix_diagram_path,
        args.chem_pot,
        args.phi,
        args.pH,
        args.system_temp,
        args.sweeps,
        args.sweep_size,
        args.alpha,
        args.samp_temp,
        args.relax,
        args.relax_steps,
        args.surface_depth,
        args.ads_site_planar_distance,
        args.ads_site_type,
        args.record_interval,
        args.neighbor_cutoff,
        args.offset,
        args.offset_data_path,
        args.device,
        save_folder=args.save_folder,
    )
