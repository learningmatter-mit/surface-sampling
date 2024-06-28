"""Performs sampling of surface reconstructions using an MCMC-based algorithm"""

import logging
import os
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from mcmc.events.criterion import (
    DistanceCriterion,
    MetropolisCriterion,
    TestingCriterion,
)
from mcmc.events.event import Change, Exchange
from mcmc.events.proposal import ChangeProposal, SwitchProposal
from mcmc.system import SurfaceSystem
from mcmc.utils import create_anneal_schedule, setup_folders, setup_logger
from mcmc.utils.misc import (
    find_closest_points_indices,
    get_cluster_centers,
    plot_clustering_results,
)

logger = logging.getLogger(__name__)
file_dir = os.path.dirname(__file__)

ENERGY_DIFF_LIMIT = 1e3  # in eV

# TODO: idea of final code
# class MCMCSampling:
#     def __init__(self) -> None:
#         ...
#         self.surface = None

#     def run(self, surface=None, nsteps: int = 100):
#         ...
#         if surface is None:
#             surface = self.surface
#         else:
#             self.surface = surface

#         if self.surface is None:
#             raise ValueError("Surface not set")

#         # run the MCMC sampling
#         for i in range(nsteps):
#             self.step()

#     def step(self):
#         event = self.event_generator.get_event(self.surface, **kwargs)

#         accept = self.acceptance_criterion(event)

#         # can do something like this
#         log_dict = {
#             "action": action,
#             "probability": probability,
#             "yesorno": yesorno,
#             "Ef": Ef,
#             "Ei": Ei,
#             "N": N,
#         }
#         return log_dict


class MCMC:
    """MCMC-based class for sampling surface reconstructions."""

    def __init__(
        self,
        canonical=False,
        num_ads_atoms=0,
        testing=False,
        adsorbates=None,
        relax=False,
        filter_distance: float = 0.0,
        **kwargs,
    ) -> None:
        self.canonical = canonical
        self.num_ads_atoms = num_ads_atoms  # TODO can be (re)moved
        self.testing = testing
        self.adsorbates = adsorbates
        self.relax = (
            relax  # TODO remove after writing the copy methods for SurfaceSystem
        )
        self.filter_distance = filter_distance
        self.kwargs = kwargs

        # initialize here for subsequent runs
        self.surface = None
        self.total_sweeps = 100
        self.sweep_size = 100
        self.temp = 1.0
        self.alpha = 1.0
        self.run_folder = ""
        self.curr_energy = 0  # TODO: move to elsewhere or exclude

        if self.canonical:
            # perform canonical runs
            # adsorb num_ads_atoms
            assert (
                self.num_ads_atoms > 0
            ), "for canonical runs, need number of adsorbed atoms greater than 0"

    def initialize(
        self,
        even_adsorption_sites: bool = False,
        perform_annealing=False,
        anneal_schedule=None,
        multiple_anneal=False,
    ) -> np.ndarray:
        # TODO update with logger
        """Initialize the MCMC simulation by setting up the run folder, preparing the canonical slab, and creating the
        annealing schedule.

        Args:
            even_adsorption_sites (bool, optional): If True, evenly adsorb the sites. Defaults to False.
            perform_annealing (bool, optional): If True, perform annealing. Defaults to False.
            anneal_schedule (list, optional): The annealing schedule. Defaults to None.
            multiple_anneal (bool, optional): If True, perform multiple annealing. Defaults to False.

        Returns:
            np.ndarray: The annealing schedule.
        """

        if not self.run_folder:
            self.run_folder = setup_folders(
                self.surface.surface_name,
                canonical=self.canonical,
                total_sweeps=self.total_sweeps,
                start_temp=self.temp,
                alpha=self.alpha,
            )
            logger.info("Generating run folder %s", self.run_folder)
        else:
            logger.info("Using user specified run folder %s", self.run_folder)

        Path(self.run_folder).mkdir(parents=True, exist_ok=True)

        # if not self.logger:
        # self.logger = setup_logger(
        #     __name__, f"{self.run_folder}/mc.log", level=logging.INFO
        # )

        if self.canonical:
            self.prepare_canonical(even_adsorption_sites=even_adsorption_sites)

        if isinstance(anneal_schedule, Iterable):
            temp_list = anneal_schedule  # user-defined annealing schedule
        elif perform_annealing:
            temp_list = create_anneal_schedule(
                start_temp=self.temp,
                total_sweeps=self.total_sweeps,
                alpha=self.alpha,
                multiple_anneal=multiple_anneal,
                save_folder=self.run_folder,
            )
        else:
            temp_list = np.repeat(self.temp, self.total_sweeps)  # constant temperature
        return temp_list

    def run(self, surface: SurfaceSystem):
        """Alias for `mcmc_run` function.

        Args:
            surface (SurfaceSystem): The surface system on which the MCMC simulation is to be performed.
        """
        self.mcmc_run(surface)

    def get_initial_energy(self):
        """Calculate the energy of the initial surface structure. If the calculator is not set, energy will be
        set to 0.

        Returns:
            float: The energy of the initial surface structure.
        """
        try:
            energy = float(self.surface.get_surface_energy(recalculate=True))
        except RuntimeError:
            energy = 0  # Calculator does not exist
        return energy

    # TODO: refactor out, might take some effort
    def prepare_canonical(self, even_adsorption_sites: bool = False):
        """Prepare a canonical slab by performing semi-grand canonical adsorption runs until the desired number of
        adsorbed atoms are obtained.

        Args:
            even_adsorption_sites (bool, optional): If True, evenly adsorb the sites. Defaults to False.

        Raises:
            AssertionError: If the number of adsorbed atoms is less than 0.
        """
        assert (
            self.num_ads_atoms > 0
        ), "for canonical runs, need number of adsorbed atoms greater than 0"

        if even_adsorption_sites:
            logger.info("evenly adsorbing sites")
            # Do clustering
            centers, labels = get_cluster_centers(
                self.surface.ads_coords[:, :2], self.num_ads_atoms
            )
            sites_idx = find_closest_points_indices(
                self.surface.ads_coords[:, :2], centers, labels
            )
            plot_clustering_results(
                self.surface.ads_coords,
                self.num_ads_atoms,
                labels,
                sites_idx,
                save_folder=self.run_folder,
            )

            for site_idx in sites_idx:
                self.curr_energy, _ = self.change_site(
                    prev_energy=self.curr_energy, site_idx=site_idx
                )
        else:
            logger.info("randomly adsorbing sites")
            # perform semi-grand canonical until num_ads_atoms are obtained
            while self.surface.num_adsorbates < self.num_ads_atoms:
                self.curr_energy, _ = self.change_site(prev_energy=self.curr_energy)
                # site_idx = next(site_iterator)
                # self.curr_energy, _ = self.change_site(
                #     prev_energy=self.curr_energy, site_idx=site_idx
                # )

        self.surface.real_atoms.write(
            os.path.join(
                self.run_folder, f"{self.surface.surface_name}_canonical_init.cif"
            )
        )

    # TODO: merge change_site and change_site_canonical to step() with step_num or iter_num
    def change_site_canonical(self, prev_energy: float = 0, iter_num: int = 1):
        """This function performs a canonical sampling step. It switches the adsorption sites of two
        adsorbates and checks if the change is energetically favorable.

        Parameters
        ----------
        prev_energy : float, optional
            The energy of the current state before attempting to change it. If it is not provided and the
        `testing` flag is not set, it will be calculated using the `slab_energy` function.
        iter_num : int, optional
            An integer representing the current iteration number of the function.

        Returns
        -------
            the energy of the new slab and a boolean value indicating whether the proposed change was
        accepted or not.

        """
        if iter_num % self.sweep_size == 0:
            logger.info("At iter %s", iter_num)
            plot_specific_distance_weights = True
        else:
            plot_specific_distance_weights = False

        if not prev_energy and not self.testing:
            prev_energy = float(self.surface.get_surface_energy(recalculate=True))

        proposal = SwitchProposal(
            system=self.surface,
            adsorbate_list=self.adsorbates.copy(),
            require_per_atom_energies=self.kwargs.get(
                "require_per_atom_energies", False
            ),
            require_distance_decay=self.kwargs.get("require_distance_decay", False),
            temp=self.temp,
            run_folder=self.run_folder,
            plot_specific_distance_weights=plot_specific_distance_weights,
            run_iter=iter_num,
        )
        logger.debug("\n we are at iter %s", iter)

        if self.filter_distance:
            criterion = DistanceCriterion(
                filter_distance=self.kwargs["filter_distance"],
            )
            logger.debug("Using distance filter")
        elif self.testing:
            criterion = TestingCriterion()
            logger.debug("Using test criterion, always accept")
        else:
            criterion = MetropolisCriterion(self.temp)
            logger.debug("Using Metropolis criterion")

        event = Exchange(self.surface, proposal, criterion)

        accept, self.surface = event.acceptance()
        energy = self.surface.results["surface_energy"]
        return energy, accept

    def change_site(
        self, prev_energy: float = 0, iter_num: int = 1, site_idx: int = None
    ):
        """Performs a semigrand canonical sampling iteration. It randomly chooses a site to change identity in a slab, adds or removes an atom from the site,
        optionally performs relaxation, calculates the energy of the new slab, and accepts or rejects the change based
        on the Boltzmann-weighted energy difference, chemical potential change, and temperature.

        Parameters
        ----------
        prev_energy : float, optional
            the energy of the slab before the
        iter_num : int, optional
            The iteration number of the simulation.
        site_idx : int, optional
            Specify the index of the site to switch.

        Returns
        -------
            the energy of the new slab and a boolean value indicating whether the proposed change was
        accepted or not.

        """
        logger.debug("\n we are at iter %s", iter_num)
        if not prev_energy and not self.testing:
            prev_energy = float(self.surface.get_surface_energy(recalculate=True))

        proposal = ChangeProposal(
            system=self.surface,
            adsorbate_list=self.adsorbates.copy(),
            site_idx=site_idx,
        )

        if self.filter_distance:
            criterion = DistanceCriterion(
                filter_distance=self.kwargs["filter_distance"]
            )
            logger.debug("Using distance filter")

        elif self.testing:
            criterion = TestingCriterion()
            logger.debug("Using test criterion, always accept")

        else:
            criterion = MetropolisCriterion(self.temp)
            logger.debug("Using Metropolis criterion")

        event = Change(self.surface, proposal, criterion)

        accept, self.surface = event.acceptance()
        energy = self.surface.results["surface_energy"]
        return energy, accept

    def sweep(self, i: int = 0) -> dict:
        """Perform MC sweep.

        Args:
            i (int, optional): The sweep number. Defaults to 0.

        Returns:
            dict: A dictionary containing the history, trajectory, energy, adsorption count, and acceptance rate.
        """
        num_accept = 0
        logger.info("In sweep %s out of %s", i + 1, self.total_sweeps)
        for j in range(self.sweep_size):
            run_idx = self.sweep_size * i + j + 1
            # TODO change to self.step()
            if self.canonical:
                self.curr_energy, accept = self.change_site_canonical(
                    prev_energy=self.curr_energy, iter_num=run_idx
                )
            else:
                self.curr_energy, accept = self.change_site(
                    prev_energy=self.curr_energy, iter_num=run_idx
                )
            num_accept += accept

        # save structure and traj for easy viewing
        self.surface.save_structures(sweep_num=i + 1, save_folder=self.run_folder)
        # surface = self.surface.copy_without_calc()
        # BUG: not working for example.ipynb `TypeError: cannot pickle '_thread.lock' object`
        # TODO fix
        surface = (
            self.surface.relaxed_atoms.copy()
            if self.relax
            else self.surface.real_atoms.copy()
        )

        result = {
            "history": surface,
            "trajectory": self.surface.relax_traj,
            "energy": self.surface.get_surface_energy(),
            "adsorption_count": self.surface.num_adsorbates,
            "acceptance_rate": num_accept / self.sweep_size,
        }
        return result

    def mcmc_run(
        self,
        surface: SurfaceSystem,
        total_sweeps: int = 800,
        sweep_size: int = 300,
        start_temp: float = 1.0,
        perform_annealing=False,
        alpha: float = 0.9,
        multiple_anneal: bool = False,
        anneal_schedule: list = None,
        run_folder: str = None,
        starting_iteration: list = 0,
        even_adsorption_sites: bool = False,
        **kwargs,
    ):
        # TODO separate out annealing schedule
        """This function runs an MC simulation for a given number of sweeps and temperature, and
        returns the history of the simulation along with summary statistics.

        Parameters
        ----------
        num_sweeps : int, optional
            The number of MCMC sweeps to perform.
        temp : float, optional
            The temperature parameter is used in the Metropolis-Hastings algorithm for MC simulations.
            It controls the probability of accepting a proposed move during the simulation. A higher temperature
            leads to a higher probability of accepting a move, while a lower temperature leads to a lower probability
            of accepting a move.
        alpha : float, optional
            The alpha parameter is a value between 0 and 1 that determines the annealing rate. A higher
            alpha results in a slower annealing rate, while a lower alpha results in a faster annealing rate.
        surface : SurfaceSystem
            The `surface` is the starting surface structure on which the MC simulation is
            being performed.

        Returns
        -------
            a tuple containing `self.history`, `self.energy_hist`, `self.frac_accept_hist`,
        `self.adsorption_count_hist`, and `self.run_folder`.

        """
        logger.info(
            "Running with num_sweeps = %d, sweep_size = %d, start_temp = %.3f",
            total_sweeps,
            sweep_size,
            start_temp,
        )

        if run_folder:
            self.run_folder = run_folder

        # TODO: add logger, reduce the number of arguments
        self.surface = surface
        logger.info(
            "There are %d atoms in pristine slab", self.surface.num_pristine_atoms
        )
        self.curr_energy = self.get_initial_energy()
        logger.info("Initial energy is %.3f", self.curr_energy)

        self.total_sweeps = total_sweeps
        self.sweep_size = sweep_size
        self.temp = start_temp
        self.alpha = alpha

        temp_list = self.initialize(
            even_adsorption_sites, perform_annealing, anneal_schedule, multiple_anneal
        )

        logger.info("Starting with iteration %d", starting_iteration)
        logger.info(
            "Temperature schedule is: %s", [f"{temp:.3f}" for temp in temp_list]
        )

        # perform MC sweeps
        results = defaultdict(list)  # TODO: make a dataclass
        for i in range(starting_iteration, self.total_sweeps):
            self.temp = temp_list[i]
            sweep_result = self.sweep(i=i)  # TODO change to .step
            results["history"].append(sweep_result["history"])
            results["trajectories"].append(sweep_result["trajectory"])
            results["energy_hist"].append(sweep_result["energy"])
            results["frac_accept_hist"].append(sweep_result["acceptance_rate"])
            results["adsorption_count_hist"].append(sweep_result["adsorption_count"])

        return results


if __name__ == "__main__":
    pass
