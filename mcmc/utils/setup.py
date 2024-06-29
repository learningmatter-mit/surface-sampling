"""Set up folders for simulation depending on whether it's semi-grand canonical or canonical."""

import os
from datetime import datetime
from pathlib import Path


def setup_folders(
    surface_name: str,
    canonical: bool = False,
    run_folder: str | None = None,
    total_sweeps: int = 0,
    start_temp: float = 1.0,
    alpha: float = 1.0,
) -> Path:
    """Set up folders for simulation depending on whether it's semi-grand canonical or canonical.

    Args:
        surface_name (str): Name of the surface.
        canonical (bool, optional): Whether the simulation is canonical. Defaults to False.
        run_folder (str, optional): Path to the run folder. Defaults to None.
        total_sweeps (int, optional): Total number of sweeps. Defaults to 0.
        start_temp (float, optional): Starting temperature. Defaults to 1.0.
        alpha (float, optional): Alpha value. Defaults to 1.0.

    Returns:
        Path: Path to the run folder.
    """
    start_timestamp = datetime.now().isoformat(sep="-", timespec="milliseconds")
    run_folder_path = Path(os.getcwd()) / surface_name
    run_folder_base = (
        f"{start_timestamp}_sweeps_{total_sweeps}_start_temp_{start_temp}_alpha_{alpha}"
    )

    # default to semi-grand canonical run folder unless canonical is specified
    if canonical:
        run_folder = run_folder_path / (run_folder_base + "_canonical")
    else:
        run_folder = run_folder_path / (run_folder_base + "_semigrand")

    Path(run_folder).mkdir(parents=True, exist_ok=False)

    return run_folder
