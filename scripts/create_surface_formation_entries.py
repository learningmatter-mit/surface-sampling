"""Create pymatgen surface formation energy entries from VSSR-MC sampled surfaces"""

import argparse
import logging
import pickle as pkl
from pathlib import Path
from typing import Literal

import ase
import numpy as np
from monty.serialization import loadfn
from nff.io.ase_calcs import NeuralFF
from nff.train.builders.model import load_model
from nff.utils.cuda import get_final_device
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Structure
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from tqdm import tqdm

from mcmc.calculators import get_results_single
from mcmc.dynamics import optimize_slab
from mcmc.utils.misc import get_atoms_batch, load_dataset_from_files

logger = logging.getLogger("mcmc")
logger.setLevel(logging.INFO)

np.set_printoptions(precision=3, suppress=True)

SYMBOLS = {
    "La": "PAW_PBE La 06Sep2000",
    "O": "PAW_PBE O 08Apr2002",
    "Mn": "PAW_PBE Mn_sv 23Jul2007",
    "H": "PAW_PBE H 15Jun2001",
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Create pymatgen surface formation energy Entry's from VSSR-MC sampled surfaces "
            "under electrochemical conditions."
        )
    )
    parser.add_argument(
        "--surface_name",
        type=str,
        default="LaMnO3_001_2x2",
        help="name of the surface",
    )
    parser.add_argument(
        "--file_paths",
        nargs="+",
        help="Full paths to NFF Dataset, ASE Atoms/NFF AtomsBatch, or a text file of file paths.",
        type=Path,
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
        "--elements",
        nargs="+",
        type=str,
        help="list of elements",
    )
    parser.add_argument(
        "--save_folder",
        type=Path,
        default="./",
        help="Folder to output.",
    )
    parser.add_argument("--relax", action="store_true", help="perform relaxation for the steps")
    parser.add_argument("--relax_steps", type=int, default=5, help="max relaxation steps")
    parser.add_argument(
        "--neighbor_cutoff",
        type=float,
        default=5.0,
        help="cutoff for neighbor calculations",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="device to use for calculations",
    )

    return parser.parse_args()


def get_params(elements: list[str]) -> dict:
    """Get the parameters for the ComputedStructureEntry.

    Args:
        elements (list[str]): list of elements

    Returns:
        dict: parameters for the ComputedStructureEntry
    """
    return {
        "run_type": "GGA",
        "is_hubbard": False,
        "hubbards": {},
        "potcar_symbols": [SYMBOLS[elem] for elem in elements],
    }


def create_computed_entry(
    slab: ase.Atoms, energy: float, slab_name: str | None = None
) -> ComputedStructureEntry:
    """Create a ComputedStructureEntry from an ASE Atoms object.

    Args:
        slab (Atoms): surface slab
        energy (float): raw predicted energy of the slab in eV (preferably with MP 2020 corrections)
        slab_name (str): name of the slab

    Returns:
        ComputedStructureEntry: ComputedStructureEntry object
    """
    pmg_struct = Structure.from_ase_atoms(slab)
    params = get_params(slab.get_chemical_symbols())
    return ComputedStructureEntry(pmg_struct, energy, parameters=params, entry_id=slab_name)


def create_surface_formation_entry(
    raw_energy_entry: ComputedEntry, phase_diagram: PhaseDiagram
) -> ComputedEntry:
    """Create a ComputedEntry with surface formation energy. Required as input to
    SurfacePourbaixDiagram.

    Args:
        raw_energy_entry (ComputedEntry): Entry with the raw energies in eV (preferably with
            MP 2020 corrections)
        phase_diagram (PhaseDiagram): pymatgen PhaseDiagram object for the relevant atomic species

    Returns:
        ComputedEntry: ComputedEntry object with surface formation energy
    """
    return ComputedEntry(
        raw_energy_entry.composition,
        phase_diagram.get_form_energy(raw_energy_entry),
        parameters=raw_energy_entry.parameters,
        entry_id=raw_energy_entry.entry_id,
    )


def main(
    surface_name: str,
    file_names: list[str],
    model_type: Literal["NffScaleMACE", "CHGNetNFF"],
    model_paths: list[str],
    phase_diagram_path: Path | str,
    pourbaix_diagram_path: Path | str,
    neighbor_cutoff: float = 5.0,
    device: str = "cuda",
    relax: bool = False,
    relax_steps: int = 20,
    save_folder: str = "./",
) -> None:
    """Create pymatgen ComputedEntries for surface formation energies. Uses NFF models to predict
    energies. Relaxes the structures if relax is True.

    Args:
        surface_name (str): name of the surface
        file_names (list[str]): list of file paths to the ASE Atoms objects
        model_type (Literal["NffScaleMACE", "CHGNetNFF"]): type of model to use
        model_paths (list[str]): list of paths to the models
        phase_diagram_path (Path | str): path to the saved pymatgen PhaseDiagram
        pourbaix_diagram_path (Path | str): path to the saved pymatgen PourbaixDiagram
        neighbor_cutoff (float, optional): cutoff for neighbor calculations. Defaults to 5.0.
        device (str, optional): device to use for calculations. Defaults to "cuda".
        relax (bool, optional): perform relaxation for the steps. Defaults to False.
        relax_steps (int, optional): max relaxation steps. Defaults to 20.
        save_folder (str, optional): folder to output. Defaults to "./".
    """
    run_path = Path(save_folder) / surface_name
    run_path.mkdir(parents=True, exist_ok=True)

    file_paths = [Path(file_name) for file_name in file_names]
    print(f"There are a total of {len(file_paths)} input files")

    dset = load_dataset_from_files(file_paths)
    print(f"Loaded {len(dset)} structures")

    device = get_final_device(device)
    logging.info("Using device: %s", device)

    phase_diagram = loadfn(phase_diagram_path)
    # pourbaix_diagram = loadfn(pourbaix_diagram_path)

    # Load Ensemble NFF Model
    models = [
        load_model(model_path, model_type=model_type, map_location=device)
        for model_path in model_paths
    ]
    # TODO write support for multiple models
    nff_calc = NeuralFF(
        models[0],
        device=device,
        model_units=models[0].units,
        prediction_units="eV",
    )
    surf_form_entries = []
    for i, slab in enumerate(tqdm(dset)):
        slab_batch = get_atoms_batch(
            slab,
            nff_cutoff=neighbor_cutoff,
            device=device,
            props={"energy": 0, "energy_grad": []},  # needed for NFF
        )
        if relax:
            if i == 0:
                print("Relaxing the first slab")
                # save before relaxation
                slab_batch.write(run_path / f"unrelaxed_{slab_batch.get_chemical_formula()}.cif")
            slab_batch = optimize_slab(
                slab_batch,
                optimizer="FIRE",
                save_traj=False,
                relax_steps=relax_steps,
            )[0]
            if i == 0:
                # save after relaxation
                slab_batch.write(run_path / f"relaxed_{slab_batch.get_chemical_formula()}.cif")
        results = get_results_single(slab_batch, nff_calc)
        raw_energy = float(results["energy"])
        raw_entry = create_computed_entry(
            slab_batch, raw_energy, slab_name=slab.get_chemical_formula()
        )
        surface_formation_entry = create_surface_formation_entry(raw_entry, phase_diagram)
        surf_form_entries.append(surface_formation_entry)

    # Save surface formation entries to pkl
    print(f"Saving {len(surf_form_entries)} surface formation entries")
    relaxed = "relaxed" if relax else "unrelaxed"
    with open(run_path / f"additional_{relaxed}_surface_formation_entries.pkl", "wb") as f:
        pkl.dump(surf_form_entries, f)


if __name__ == "__main__":
    args = parse_args()
    main(
        args.surface_name,
        args.file_paths,
        args.model_type,
        args.model_paths,
        args.phase_diagram_path,
        args.pourbaix_diagram_path,
        args.neighbor_cutoff,
        args.device,
        args.relax,
        args.relax_steps,
        args.save_folder,
    )
