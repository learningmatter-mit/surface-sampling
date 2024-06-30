"""Utility functions for sampling and annealing."""

from pathlib import Path

import numpy as np

from .plot import plot_anneal_schedule


def create_anneal_schedule(
    start_temp: float = 1.0,
    total_sweeps: int = 1000,
    alpha: float = 0.99,
    multiple_anneal: bool = False,
    save_folder: Path | str = ".",
    save_fig: bool = True,
    save_csv: bool = True,
    **kwargs,
) -> list[float]:
    """Create an annealing schedule for simulated annealing reduction of temperature.

    Args:
        start_temp (float, optional): Starting temperature in units of kB T. Defaults to 1.0.
        total_sweeps (int, optional): Total number of MC sweeps. Defaults to 1000.
        alpha (float, optional): Cooling factor. Defaults to 0.99.
        multiple_anneal (bool, optional): Whether to use multiple annealing steps. Defaults to
            False.
        save_folder (Union[Path, str], optional): Folder to save the output in. Defaults to ".".
        save_fig (bool, optional): Whether to export a plot of the temperature schedule.
            Defaults to True.
        save_csv (bool, optional): Whether to export a csv of the temperature schedule.
            Defaults to True.
        **kwargs: Additional keyword arguments.

    Returns:
        list: List of temperatures for each MC sweep.
    """
    save_path = Path(save_folder)
    temp_list = [start_temp]

    curr_sweep = 1
    curr_temp = start_temp

    if not multiple_anneal:
        while curr_sweep < total_sweeps:
            curr_temp *= alpha
            temp_list.append(curr_temp)
            curr_sweep += 1
    else:
        # multiple annealing steps
        while curr_sweep < total_sweeps:
            # new low temperature annealing schedule
            # **0.2 to 0.10 relatively fast, say 100 steps**
            # **then 0.10 to 0.08 for 200 steps**
            # **0.08 for 200 steps, go up to 0.2 in 10 steps**
            temp_list.extend(np.linspace(curr_temp, 0.10, 100).tolist())
            curr_sweep += 100
            temp_list.extend(np.linspace(0.10, 0.08, 200).tolist())
            curr_sweep += 200
            temp_list.extend(np.repeat(0.08, 200).tolist())
            curr_sweep += 200
            temp_list.extend(np.linspace(0.08, curr_temp, 10).tolist())

    temp_list = temp_list[:total_sweeps]

    if save_fig:
        plot_anneal_schedule(temp_list, save_folder=save_path)
    if save_csv:
        with open(save_path / "anneal_schedule.csv", "w", encoding="utf-8") as f:
            f.write(",".join([str(temp) for temp in temp_list]))
    return temp_list
