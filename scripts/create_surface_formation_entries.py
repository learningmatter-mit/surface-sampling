"""Create pymatgen surface formation energy entries from VSSR-MC sampled surfaces"""

import argparse
import pickle as pkl
from datetime import datetime
from logging import getLevelNamesMapping
from pathlib import Path
from typing import Literal

import ase
import numpy as np
from monty.serialization import loadfn
from nff.io.ase_calcs import EnsembleNFF
from nff.train.builders.model import load_model
from nff.utils.constants import HARTREE_TO_EV
from nff.utils.cuda import get_final_device
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Structure
from pymatgen.entries.compatibility import (
    MaterialsProject2020Compatibility,
    MaterialsProjectAqueousCompatibility,
)
from pymatgen.entries.computed_entries import ComputedStructureEntry
from tqdm import tqdm

from mcmc.calculators import get_results_single
from mcmc.dynamics import optimize_slab
from mcmc.pourbaix.utils import SurfaceOHCompatibility
from mcmc.utils import setup_logger
from mcmc.utils.misc import get_atoms_batch, load_dataset_from_files

np.set_printoptions(precision=3, suppress=True)

SYMBOLS = {
    "La": "PAW_PBE La 06Sep2000",
    "O": "PAW_PBE O 08Apr2002",
    "Ir": "PAW_PBE Ir 06Sep2000",
    "Pt": "PAW_PBE Pt 04Feb2005",
    "Mn": "PAW_PBE Mn_pv 02Aug2007",
    "H": "PAW_PBE H 15Jun2001",
}

DFT_U_VALUES = {
    "La": 0.0,
    "Mn": 3.9,
    "Pt": 0.0,  # no need for metals
    "O": 0.0,
    "Ir": 0.0,
    "H": 0.0,
}

OH_CORRECTION = 0.23  # eV, from Rong and Kolpak, J. Phys. Chem. Lett., 2015

O2_DFT_ENERGY = -4.94795546875  # DFT energy before any entropy correction
H2O_DFT_ENERGY = -5.192751548333333  # DFT energy before any entropy correction
H2O_ADJUSTMENTS = -0.229  # already counted in the H2O energy


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
        choices=["NffScaleMACE", "CHGNetNFF", "DFT"],
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
        "--correct_hydroxide_energy",
        action="store_true",
        help="correct hydroxide energy (add ZPE-TS)",
    )
    parser.add_argument(
        "--aq_compat",
        action="store_true",
        help="use MaterialsProjectAqueousCompatibility",
    )
    parser.add_argument(
        "--neighbor_cutoff",
        type=float,
        default=5.0,
        help="cutoff for neighbor calculations",
    )
    parser.add_argument(
        "--input_slab_name",
        action="store_true",
        help="Input stoichiometry of the slab as the slab name",
    )
    parser.add_argument(
        "--input_job_id",
        action="store_true",
        help="Input job ID as the slab name",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="device to use for calculations",
    )
    parser.add_argument(
        "--logging_level",
        type=str,
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Logging level",
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
        "run_type": "GGA+U",
        "is_hubbard": True,
        "hubbards": {elem: DFT_U_VALUES[elem] for elem in elements},
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
    params = get_params(list(set(slab.get_chemical_symbols())))
    return ComputedStructureEntry(pmg_struct, energy, parameters=params, entry_id=slab_name)


def create_surface_formation_entry(
    raw_energy_entry: ComputedStructureEntry, phase_diagram: PhaseDiagram
) -> ComputedStructureEntry:
    """Create a ComputedStructureEntry with surface formation energy. Required as input to
    SurfacePourbaixDiagram.

    Args:
        raw_energy_entry (ComputedStructureEntry): Entry with the raw energies in eV (preferably
            with MP 2020 corrections)
        phase_diagram (PhaseDiagram): pymatgen PhaseDiagram object for the relevant atomic species

    Returns:
        ComputedStructureEntry: ComputedStructureEntry object with surface formation energy
    """
    return ComputedStructureEntry(
        raw_energy_entry,
        phase_diagram.get_form_energy(raw_energy_entry),
        parameters=raw_energy_entry.parameters,
        entry_id=raw_energy_entry.entry_id,
    )


def main(
    surface_name: str,
    file_names: list[str],
    model_type: Literal["NffScaleMACE", "CHGNetNFF", "DFT"],
    model_paths: list[str],
    phase_diagram_path: Path | str,
    pourbaix_diagram_path: Path | str,
    correct_hydroxide_energy: bool = False,
    aq_compat: bool = False,
    input_slab_name: bool = False,
    input_job_id: bool = False,
    neighbor_cutoff: float = 5.0,
    device: str = "cuda",
    relax: bool = False,
    relax_steps: int = 20,
    save_folder: str = "./",
    logging_level: Literal["debug", "info", "warning", "error", "critical"] = "info",
) -> None:
    """Create pymatgen ComputedEntries for surface formation energies. Uses NFF models to predict
    energies. Relaxes the structures if relax is True.

    Args:
        surface_name (str): name of the surface
        file_names (list[str]): list of file paths to the ASE Atoms objects
        model_type (Literal["NffScaleMACE", "CHGNetNFF", "DFT"]): type of model to use
        model_paths (list[str]): list of paths to the models
        phase_diagram_path (Path | str): path to the saved pymatgen PhaseDiagram
        pourbaix_diagram_path (Path | str): path to the saved pymatgen PourbaixDiagram
        correct_hydroxide_energy (bool, optional): correct hydroxide energy (add ZPE-TS). Defaults
            to False.
        aq_compat (bool, optional): use MaterialsProjectAqueousCompatibility. Defaults to False.
        input_slab_name (bool, optional): Input stoichiometry of the slab as the slab name. Defaults
            to False.
        input_job_id (bool, optional): Input job ID as the slab name. Defaults to False.
        neighbor_cutoff (float, optional): cutoff for neighbor calculations. Defaults to 5.0.
        device (str, optional): device to use for calculations. Defaults to "cuda".
        relax (bool, optional): perform relaxation for the steps. Defaults to False.
        relax_steps (int, optional): max relaxation steps. Defaults to 20.
        save_folder (str, optional): folder to output. Defaults to "./".
        logging_level (Literal["debug", "info", "warning", "error", "critical"], optional):
            logging level. Defaults to "info".
    """
    start_timestamp = datetime.now().isoformat(sep="-", timespec="milliseconds")

    # Initialize save folder
    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)
    file_base = f"{start_timestamp}_generate_surface_formation_entries_{surface_name}"

    # Initialize logger
    logger = setup_logger(
        "generate_formation_entries",
        save_path / "generate_formation_entries.log",
        level=getLevelNamesMapping()[logging_level.upper()],
    )

    logger.info("There are a total of %d input files", len(file_names))
    dset = load_dataset_from_files(file_names)
    logger.info("Loaded %d structures", len(dset))

    device = get_final_device(device)
    logger.info("Using device: %s", device)

    phase_diagram = loadfn(phase_diagram_path)
    # pourbaix_diagram = loadfn(pourbaix_diagram_path)

    if model_type not in ["DFT"]:
        # Load Ensemble NFF Model
        models = [
            load_model(model_path, model_type=model_type, map_location=device)
            for model_path in model_paths
        ]
        nff_calc = EnsembleNFF(
            models,
            device=device,
            model_units=models[0].units,
            prediction_units="eV",
        )
    else:
        # Set up compatibility adjustments
        solid_compat = MaterialsProject2020Compatibility()
    if correct_hydroxide_energy:
        oh_compat = SurfaceOHCompatibility(correction=OH_CORRECTION)

    if aq_compat:
        solid_compat = MaterialsProjectAqueousCompatibility(
            solid_compat=solid_compat,
            o2_energy=-4.94795546875,  # DFT energy before any entropy correction
            h2o_energy=-5.192751548333333,  # DFT energy before any entropy correction
            h2o_adjustments=-0.229,  # already counted in the H2O energy
        )

    raw_entries = []
    final_slab_batches = []
    surf_form_entries = []

    for i, slab in enumerate(tqdm(dset)):
        if model_type in ["DFT"]:
            slab_batch = slab
            # try to get DFT energies
            try:
                raw_energy = float(slab_batch.props["energy"]) * HARTREE_TO_EV  # convert to eV
            except KeyError:
                logger.error("No DFT energy found for %s", slab.get_chemical_formula())
                continue
        else:
            slab_batch = get_atoms_batch(
                slab,
                nff_cutoff=neighbor_cutoff,
                device=device,
                props={"energy": 0, "energy_grad": []},  # needed for NFF
            )
            slab_batch.set_calculator(nff_calc)
            if relax:
                if i == 0:
                    logger.info("Relaxing the first slab")
                    # save before relaxation
                    slab_batch.write(
                        save_path / f"unrelaxed_{slab_batch.get_chemical_formula()}.cif"
                    )
                slab_batch = optimize_slab(
                    slab_batch,
                    optimizer="FIRE",
                    save_traj=False,
                    relax_steps=relax_steps,
                )[0]
                if i == 0:
                    # save after relaxation
                    slab_batch.write(save_path / f"relaxed_{slab_batch.get_chemical_formula()}.cif")
            results = get_results_single(slab_batch, nff_calc)
            raw_energy = float(results["energy"])  # DFT-like energy

        # Use constraints to set fake surface atoms so that they relax
        if (len(slab_batch.constraints) > 0) and (
            slab_batch.constraints[0].__class__.__name__ == "FixAtoms"
        ):
            fixed_indices = slab_batch.constraints[0].get_indices()
            surface_indices = np.isin(
                np.arange(len(slab_batch)), fixed_indices, invert=True
            ).astype(int)
            slab_batch.set_tags(surface_indices)
        final_slab_batches.append(slab_batch)
        if input_slab_name:
            slab_name = slab.get_chemical_formula()
        elif input_job_id:
            slab_name = (
                slab_batch.props.get("job_id", None)
                if hasattr(slab_batch, "props")
                else slab_batch.info.get("job_id", None)
            )
        else:
            slab_name = None
        raw_entry = create_computed_entry(slab_batch, raw_energy, slab_name=slab_name)
        raw_entries.append(raw_entry)
        if model_type in ["DFT"]:
            # aqcompat.process_entries([raw_entry], inplace=True)  # process the entry
            solid_compat.process_entries([raw_entry], inplace=True)  # process the entry
        if correct_hydroxide_energy:
            oh_compat.process_entries([raw_entry], clean=False, inplace=True)

        # aqcompat.get_adjustments(raw_entry)  #
        # solid_compat.get_adjustments(raw_entry)  #

        surface_formation_entry = create_surface_formation_entry(raw_entry, phase_diagram)
        surf_form_entries.append(surface_formation_entry)

    # Save surface formation entries
    relaxed = "relaxed" if relax else "unrelaxed"
    save_entries_path = (
        save_path / f"{file_base}_{relaxed}_surface_formation_entries_{len(surf_form_entries)}.pkl"
    )
    with open(save_entries_path, "wb") as f:
        pkl.dump(surf_form_entries, f)
    logger.info("Create surface formation entries complete. Saved to %s", save_entries_path)

    # Save final slab batches if relaxed
    if relax:
        save_slab_batches_path = (
            save_path / f"{file_base}_{relaxed}_slab_batches_{len(final_slab_batches)}.pkl"
        )
        with open(save_slab_batches_path, "wb") as f:
            pkl.dump(final_slab_batches, f)
        logger.info(
            "Saved final slab batches to %s. Total number of slabs: %d",
            save_slab_batches_path,
            len(final_slab_batches),
        )


if __name__ == "__main__":
    args = parse_args()
    main(
        args.surface_name,
        args.file_paths,
        args.model_type,
        args.model_paths,
        args.phase_diagram_path,
        args.pourbaix_diagram_path,
        args.correct_hydroxide_energy,
        args.aq_compat,
        args.input_slab_name,
        args.input_job_id,
        args.neighbor_cutoff,
        args.device,
        args.relax,
        args.relax_steps,
        args.save_folder,
    )
