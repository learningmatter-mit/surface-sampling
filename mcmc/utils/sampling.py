"""Utility functions for sampling and annealing."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

DPI = 200


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


def plot_anneal_schedule(
    schedule: list,
    save_folder: Path | str = ".",
) -> Figure:
    """Plot the annealing schedule.

    Args:
        schedule (list): List of temperatures for each MC sweep.
        save_folder (Union[Path, str], optional): Folder to save the output in. Defaults to ".".

    Returns:
        Figure: Matplotlib figure object.
    """
    save_path = Path(save_folder)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)
    ax.plot(schedule)
    ax.set_xlabel("Sweep number", fontsize=14)
    ax.set_ylabel("Temperature (kB T)", fontsize=14)
    ax.set_title("Annealing schedule", fontsize=16)
    plt.savefig(save_path / "anneal_schedule.png")
    return fig


# from pathlib import Path

# from mcmc.system import SurfaceSystem
# from mcmc.utils.clustering import find_closest_points_indices


# def prepare_canonical(
#     surface: SurfaceSystem,
#     num_ads_atoms: int,
#     even_adsorption_sites: bool = False,
#     save_folder: str = None,
# ):
#     """This function prepares a canonical slab by performing semi-grand canonical adsorption runs
#     until the desired number of adsorbed atoms are obtained.

#     """
#     assert num_ads_atoms > 0, "for canonical runs, need number of adsorbed atoms greater than 0"
#     if not save_folder:
#         save_folder = surface.save_folder
#     if even_adsorption_sites:
#         logger.info("evenly adsorbing sites")
#         # Do clustering
#         centers, labels = get_cluster_centers(surface.ads_coords[:, :2], num_ads_atoms)
#         sites_idx = find_closest_points_indices(surface.ads_coords[:, :2], centers, labels)
#         plot_clustering_results(
#             surface.ads_coords,
#             num_ads_atoms,
#             labels,
#             sites_idx,
#             save_folder=save_folder,
#         )

#         for site_idx in sites_idx:
#             self.curr_energy, _ = self.change_site(prev_energy=self.curr_energy,
# site_idx=site_idx)
#     else:
#         logger.info("randomly adsorbing sites")
#         # perform semi-grand canonical until num_ads_atoms are obtained
#         while len(self.surface) < self.num_pristine_atoms + self.num_ads_atoms:
#             self.curr_energy, _ = self.change_site(prev_energy=self.curr_energy)

#     surface.real_atoms.write(Path(save_folder) / f"{self.surface_name}_canonical_init.cif")
