import argparse
import json
import logging
import pickle
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import List, Literal, Union

import numpy as np
from ase.constraints import FixAtoms
from nff.io.ase_calcs import NeuralFF
from nff.utils.cuda import cuda_devices_sorted_by_free_mem
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.ase import AseAtomsAdaptor

from mcmc import MCMC
from mcmc.calculators import EnsembleNFFSurface, NFFPourbaix
from mcmc.pourbaix.atoms import PourbaixAtom
from mcmc.system import SurfaceSystem
from mcmc.utils.misc import get_atoms_batch

logger = logging.getLogger("mcmc")
logger.setLevel(logging.INFO)

np.set_printoptions(precision=3, suppress=True)


def parse_args():
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
        nargs="+",
        help="paths to the models",
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
    parser.add_argument(
        "--samp_temp", type=float, default=1.0, help="sampling temperature in kbT"
    )
    parser.add_argument(
        "--system_temp", type=float, default=1.0, help="system temperature in kbT"
    )
    parser.add_argument(
        "--relax", action="store_true", help="perform relaxation for the steps"
    )
    parser.add_argument(
        "--relax_steps", type=int, default=5, help="max relaxation steps"
    )
    parser.add_argument(
        "--record_interval", type=int, default=5, help="record interval for relaxation"
    )
    parser.add_argument(
        "--offset", action="store_true", help="whether to use energy offsets"
    )
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

    args = parser.parse_args()
    return args


def main(
    surface_name: str,
    starting_structure_path: Union[Path, str],
    model_type: Literal["NffScaleMACE", "CHGNetNFF"],
    model_paths: List[str],
    chem_pot: List[float],
    phi: float,
    pH: float,
    system_temp: float,
    sweeps: int,
    sweep_size: int,
    alpha: float,
    samp_temp: float,
    relax: bool,
    offset_data_path: str,
    relax_steps: int = 20,
    record_interval: int = False,
    offset: bool = False,
    device: str = "cuda",
    save_folder: str = "./",
):
    """Perform VSSR-MC sampling for surfaces.

    Args:
        surface_name (str): name of the surface
        starting_structure_path (Union[Path, str]): path to the starting structure
        model_type (Literal["NffScaleMACE", "CHGNetNFF"]): type of model to use
        model_paths (List[str]): paths to the models
        save_folder (Path): Folder to output
        chem_pot (List[float]): chemical potential for each element
        phi (float): electrical potential
        pH (float): pH
        system_temp (float): system temperature in kbT
        sweeps (int): MCMC sweeps
        sweep_size (int): MCMC sweep size
        alpha (float): alpha for MCMC
        samp_temp (float): sampling temperature in kbT
        relax (bool): perform relaxation for the steps

    Returns:
        None
    """

    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)

    if device == "cuda":
        DEVICE = f"cuda:{cuda_devices_sorted_by_free_mem()[-1]}"
    else:
        DEVICE = "cpu"

    with open(starting_structure_path, "rb") as f:
        starting_slab = pickle.load(f)

    chem_symbols = starting_slab.get_chemical_symbols()
    elements = list(set(chem_symbols))

    print(f"Elements: {elements}")

    # TODO probably a way to more automatically extract from Pymatgen
    Sr_pourbaix_atom = PourbaixAtom(
        "Sr",
        dominant_species="Sr2+",  # oxidized Sr -> Sr2+ + 2e-
        species_conc=1e-6,
        num_e=2,
        num_H=0,
        atom_std_state_energy=-1.664947875,  # from DFT
        delta_G2_std=-5.798,  # -(-2.899) * -2 = -5.798 from Bratsch, S. G. (1989)
    )
    Ir_pourbaix_atom = PourbaixAtom(
        "Ir",
        dominant_species="IrO2",  # oxidized Ir + H2O -> IrO2 + 4H+ + 4e-
        species_conc=1,
        num_e=4,
        num_H=4,
        atom_std_state_energy=-8.84254924,  # from DFT
        delta_G2_std=1.76738,  # from pymatgen pourbaix_diagram, original task mp-1440326 GGA
    )
    O_pourbaix_atom = PourbaixAtom(
        "O",
        dominant_species="H2O",  # reduced 1/2 O2 + 2H+ + 2e- -> H2O
        species_conc=1,
        num_e=-2,
        num_H=-2,
        atom_std_state_energy=-5.2647,  # from task mp-1933400 GGA
        delta_G2_std=-2.46,  # 1.23 * -2 = -2.46 from Bratsch, S. G. (1989). Standard electrode potentials and temperature coefficients in water at 298.15 K. Journal of Physical and Chemical Reference Data, 18(1), 1-21.
    )

    pourbaix_atoms = dict(Sr=Sr_pourbaix_atom, Ir=Ir_pourbaix_atom, O=O_pourbaix_atom)

    system_settings = {
        "surface_name": surface_name,
        "cutoff": 5.0,
        "device": DEVICE,
        "near_reduce": 0.01,
        "planar_distance": 1.55,
        "no_obtuse_hollow": True,
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
        "chem_pots": {elem: chem_pot for elem, chem_pot in zip(elements, chem_pot)},
        "relax_atoms": relax,
        "relax_steps": relax_steps,
        "record_interval": record_interval,  # record structure every n steps
        "offset": offset,
        "temperature": system_temp,
        "pH": pH,
        "phi": phi,
        "pourbaix_atoms": pourbaix_atoms,
        "offset_data": json.load(open(offset_data_path, "r")),
    }

    # Obtain adsorption sites
    pristine_slab = starting_slab.copy()
    pristine_pmg_slab = AseAtomsAdaptor.get_structure(pristine_slab)
    site_finder = AdsorbateSiteFinder(pristine_pmg_slab)

    ads_positions = site_finder.find_adsorption_sites(
        put_inside=True,
        symm_reduce=False,
        near_reduce=system_settings["near_reduce"],
        distance=system_settings["planar_distance"],
        no_obtuse_hollow=system_settings["no_obtuse_hollow"],
    )["all"]

    print(f"adsorption coordinates are: {ads_positions[:5]}...")

    # Load Ensemble NFF Model
    # requires an ensemble of models in this path and an `offset_data.json` file
    models = []
    for modeldir in model_paths:
        m = NeuralFF.from_file(modeldir, device=DEVICE, model_type=model_type).model
        models.append(m)

    # TODO write support for multiple models
    nff_surf_calc = NFFPourbaix(
        models[0],
        device=DEVICE,
        model_units=models[0].units,
        prediction_units="eV",
    )
    nff_surf_calc.set(**calc_settings)

    # Use Pretrained NFF model
    # chgnet_nff = CHGNetNFF.load(device=system_settings["device"])

    # nff_calc = NeuralFF(
    #     chgnet_nff,
    #     device=system_settings["device"],
    #     model_units="eV/atom",
    #     prediction_units="eV",
    # )

    # nff_surf_calc = EnsembleNFFSurface(
    #     [nff_calc.model],
    #     device=system_settings["device"],
    #     model_units="eV/atom",
    #     prediction_units="eV",
    #     offset_units="eV",
    # )
    # nff_surf_calc.set(**calc_settings)

    # Initialize SurfaceSystem (actually bulk system)
    slab_batch = get_atoms_batch(
        pristine_slab,
        nff_cutoff=system_settings["cutoff"],
        device=system_settings["device"],
        props={"energy": 0, "energy_grad": []},  # needed for NFF
    )

    # Fix atoms in the bulk
    num_bulk_atoms = len(slab_batch)
    bulk_indices = list(range(num_bulk_atoms))
    print(f"bulk indices {bulk_indices}")
    surf_indices = pristine_slab.get_surface_atoms()

    fix_indices = list(set(bulk_indices) - set(surf_indices))
    print(f"fix indices {fix_indices}")

    c = FixAtoms(indices=fix_indices)
    slab_batch.set_constraint(c)

    surface = SurfaceSystem(
        slab_batch, ads_positions, nff_surf_calc, system_settings=system_settings
    )
    starting_atoms_path = save_path / "all_virtual_ads.cif"

    print(f"Saving surface with virtual atoms to {starting_atoms_path}")
    surface.all_atoms.write(starting_atoms_path)

    print(f"Starting chemical formula: {slab_batch.get_chemical_formula()}")

    print(f"NFF calc starting energy: {surface.get_potential_energy()} eV")

    print(f"Starting surface energy: {surface.get_surface_energy()} eV")

    if hasattr(nff_surf_calc, "offset_units"):
        print(f"Offset units: {nff_surf_calc.offset_units}")

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
    print(
        f"saving all structures in relaxation paths with length {len(traj_structures)}"
    )

    with open(f"{mcmc.run_folder}/relax_traj_{len(traj_structures)}.pkl", "wb") as f:
        pickle.dump(traj_structures, f)


if __name__ == "__main__":
    args = parse_args()
    main(
        args.surface_name,
        args.starting_structure_path,
        args.model_type,
        args.model_paths,
        args.chem_pot,
        args.phi,
        args.pH,
        args.system_temp,
        args.sweeps,
        args.sweep_size,
        args.alpha,
        args.samp_temp,
        args.relax,
        args.offset_data_path,
        args.relax_steps,
        args.record_interval,
        args.offset,
        args.device,
        save_folder=args.save_folder,
    )
