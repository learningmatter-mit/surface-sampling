"""Performs sampling of surface reconstructions using an MCMC-based algorithm"""

import copy
import json
import logging
import os
import pickle as pkl
import random
from collections import Counter, defaultdict
from datetime import datetime
from typing import Union

import ase
import catkit
import numpy as np
from ase.calculators.eam import EAM
from ase.constraints import FixAtoms
from ase.io import write
from ase.io.trajectory import TrajectoryWriter
from catkit.gen.adsorption import get_adsorption_sites
from nff.io.ase import AtomsBatch
from nff.io.ase_calcs import EnsembleNFF, NeuralFF
from nff.utils.cuda import batch_to
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from scipy.special import softmax

from mcmc.events.criterion import (
    DistanceCriterion,
    MetropolisCriterion,
    TestingCriterion,
)
from mcmc.events.event import Change, Exchange
from mcmc.events.proposal import ChangeProposal, SwitchProposal

from .energy import optimize_slab
from .plot import plot_summary_stats
from .slab import change_site, count_adsorption_sites
from .system import SurfaceSystem
from .utils.misc import (
    compute_distance_weight_matrix,
    filter_distances,
    find_closest_points_indices,
    get_cluster_centers,
    plot_clustering_results,
    plot_decay_curve,
    plot_distance_weight_matrix,
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
        surface_name,
        calc=EAM(
            potential=os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "potentials", "Cu2.eam.fs"
            )
        ),
        canonical=False,
        num_ads_atoms=0,
        testing=False,
        adsorbates=None,
        relax=False,
        distance_weight_matrix=None,
        **kwargs,
    ) -> None:
        self.calc = calc
        self.surface_name = surface_name
        self.canonical = canonical
        self.num_ads_atoms = num_ads_atoms
        # self.ads_coords = np.array(ads_coords)
        self.distance_weight_matrix = distance_weight_matrix
        self.testing = testing
        self.adsorbates = adsorbates
        self.relax = relax
        self.kwargs = kwargs

        # initialize here for subsequent runs
        self.total_sweeps = 800
        self.start_temp = 1.0
        self.peak_scale = 1 / 2
        self.ramp_up_sweeps = 10
        self.ramp_down_sweeps = 200

        self.temp = 1.0
        self.pot = 1.0
        self.alpha = 1.0
        self.surface: SurfaceSystem = None

        self.num_pristine_atoms = 0
        self.run_folder = ""
        self.curr_energy = 0
        self.sweep_size = 100
        # self.state = None
        self.connectivity = None
        self.history = None
        self.energy_hist = None
        self.adsorption_count_hist = None
        self.frac_accept_hist = None

        self.reference_structure = kwargs.get("reference_structure", None)
        self.reference_structure_embeddings = None
        self.device = kwargs.get("device", "cpu")
        # self.rmsd_criterion = kwargs.get("rmsd_criterion", False)

        if self.canonical:
            # perform canonical runs
            # adsorb num_ads_atoms
            assert (
                self.num_ads_atoms > 0
            ), "for canonical runs, need number of adsorbed atoms greater than 0"

    def run(self, surface: SurfaceSystem):
        """The function "run" calls the function "mcmc_run"."""
        self.mcmc_run(surface)

    def get_adsorption_coords(self):
        """If not already set, this function sets the absolute adsorption coordinates for a given slab and element
        using the catkit `get_adsorption_sites` method.

        """
        self.connectivity = np.ones(len(self.surface.ads_coords), dtype=int)

        # if require distance decay
        distance_decay_factor = self.kwargs.get("distance_decay_factor", 1.0)
        if self.kwargs.get("require_distance_decay", False):
            if self.distance_weight_matrix is None:
                logger.info("computing distance weight matrix")
                self.distance_weight_matrix = compute_distance_weight_matrix(
                    self.surface.ads_coords, distance_decay_factor
                )
            else:
                logger.info("using provided distance weight matrix")
            plot_distance_weight_matrix(
                self.distance_weight_matrix, save_folder=self.run_folder
            )
            plot_decay_curve(distance_decay_factor, save_folder=self.run_folder)

        self.site_types = set(self.connectivity)

        logger.info(
            f"In pristine slab, there are a total of {len(self.surface.ads_coords)} sites"
        )

    def setup_folders(self):
        """Set up folders for simulation depending on whether it's semi-grand canonical or canonical."""
        if not self.run_folder:
            start_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S.%f.%f")

            # prepare both run folders
            canonical_run_folder = os.path.join(
                os.getcwd(),
                f"{self.surface_name}/runs{self.total_sweeps}_temp{self.start_temp}_adsatoms{self.num_ads_atoms:02}_alpha{self.alpha}_{start_timestamp}",
            )
            sgc_run_folder = os.path.join(
                os.getcwd(),
                f"{self.surface_name}/runs{self.total_sweeps}_temp{self.start_temp}_pot{self.pot}_alpha{self.alpha}_{start_timestamp}",
            )

            # default to semi-grand canonical run folder unless canonical is specified
            if self.canonical:
                self.run_folder = canonical_run_folder
            else:
                self.run_folder = sgc_run_folder

        if not os.path.exists(self.run_folder):
            os.makedirs(self.run_folder)

        logging.basicConfig(
            filename=os.path.join(self.run_folder, "logging.log"),
            filemode="a",
            format="%(levelname)s:%(message)s",
            level=logging.INFO,
            datefmt="%m/%d/%Y %I:%M:%S %p",
        )

        logger.info(
            f"Running with num_sweeps = {self.total_sweeps}, temp = {self.start_temp}, pot = {self.pot}, alpha = {self.alpha}"
        )

    def get_initial_energy(self):
        """This function returns the initial energy of a slab, which is calculated using the slab_energy
        function if the slab does not exists.

        Returns
        -------
            The function `get_initial_energy` returns the energy of the slab calculated using the `slab_energy`
        function if `self.slab.calc` is not `None`, otherwise it returns 0.

        """
        # sometimes slab.calc does not exist
        if self.surface.calc:
            energy = float(self.surface.get_surface_energy(recalculate=True))
        else:
            energy = 0

        return energy

    def prepare_canonical(self, even_adsorption_sites: bool = False):
        # TODO can move to System initialization
        """This function prepares a canonical slab by performing semi-grand canonical adsorption runs until the
        desired number of adsorbed atoms are obtained.

        """
        if self.canonical:
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
                while len(self.surface) < self.num_pristine_atoms + self.num_ads_atoms:
                    self.curr_energy, _ = self.change_site(prev_energy=self.curr_energy)
                    # site_idx = next(site_iterator)
                    # self.curr_energy, _ = self.change_site(
                    #     prev_energy=self.curr_energy, site_idx=site_idx
                    # )

            self.surface.real_atoms.write(
                os.path.join(self.run_folder, f"{self.surface_name}_canonical_init.cif")
            )

    # TODO change to save per iter and save per entry
    # TODO move to SurfaceSystem
    # def save_structures(
    #     self, energy: float, chemical_formula: str, i: int = 0, save_folder: str = "."
    # ):
    #     """This function saves the optimized structure of a slab."""
    #     chemical_formula = self.surface.relaxed_atoms.get_chemical_formula()
    #     logger.info("optim structure has Energy = %.3f", energy)

    #     if self.relax:
    #         write(
    #             f"{save_folder}/relaxed_slab_run_{i+1:03}_{energy:.3f}_{chemical_formula}.cif",
    #             self.surface.relaxed_atoms,
    #         )
    #     write(
    #         f"{save_folder}/unrelaxed_slab_run_{i+1:03}_{energy:.3f}_{chemical_formula}.cif",
    #         self.surface.real_atoms,
    #     )

    #     save_slab = self.surface.real_atoms.copy()
    #     save_slab.calc = None
    #     with open(
    #         f"{save_folder}/unrelaxed_slab_run_{i+1:03}_{energy:.3f}_{chemical_formula}.pkl",
    #         "wb",
    #     ) as f:
    #         pkl.dump(save_slab, f)

    #     # save trajectories
    #     if self.surface.relax_traj:
    #         # use TrajectoryWriter
    #         atoms_list = self.surface.relax_traj["atoms"]
    #         writer = TrajectoryWriter(
    #             f"{save_folder}/traj_{i+1:03}_{energy:.3f}_{chemical_formula}.traj",
    #             mode="a",
    #         )
    #         for atoms in atoms_list:
    #             writer.write(atoms)

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

        if self.kwargs.get("filter_distance", None):
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

        if self.kwargs.get("filter_distance", None):
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

    def mcmc_sweep(self, i: int = 0):
        """This function performs a Monte Carlo sweep and saves the resulting structures, energies, and adsorption site counts
        to a history.

        Parameters
        ----------
        i, optional
            The parameter "i" is an optional integer argument that represents the current sweep number in the
            MCMC simulation. It is used to keep track of the progress of the simulation and to append the
            results to the appropriate indices in the output arrays.

        """
        num_accept = 0
        trajectories = []
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
        # surface = self.surface.copy_without_calc() # BUG: not workign for example.ipynb `TypeError: cannot pickle '_thread.lock' object`
        surface = (
            self.surface.relaxed_atoms.copy()
            if self.relax
            else self.surface.real_atoms.copy()
        )
        self.history.append(surface)
        trajectories.append(self.surface.relax_traj)

        # append values
        self.energy_hist[i] = self.surface.get_surface_energy()

        ads_counts = count_adsorption_sites(self.surface, self.connectivity)
        for key in set(self.site_types):
            if ads_counts[key]:
                self.adsorption_count_hist[key].append(ads_counts[key])
            else:
                self.adsorption_count_hist[key].append(0)

        frac_accept = num_accept / self.sweep_size
        self.frac_accept_hist[i] = frac_accept

    def mcmc_run(
        self,
        surface: SurfaceSystem,
        peak_scale: float = 1 / 2,
        ramp_up_sweeps: int = 10,
        ramp_down_sweeps: int = 200,
        total_sweeps: int = 800,
        start_temp: float = 1.0,
        pot: Union[float, list] = 1.0,
        alpha: float = 0.9,
        perform_annealing=False,
        anneal_schedule: list = None,
        run_folder: str = None,
        starting_iteration: list = 0,
        sweep_size: int = 300,
        even_adsorption_sites: bool = False,
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
        pot : float or list, optional
            The chemical potential used in the simulation. The chemical potential can be a single value for one adsorbate or a list
            of values for each adsorbate type.
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
        if run_folder:
            self.run_folder = run_folder

        self.total_sweeps = total_sweeps
        self.start_temp = start_temp
        self.peak_scale = peak_scale
        self.ramp_up_sweeps = ramp_up_sweeps
        self.ramp_down_sweeps = ramp_down_sweeps
        self.pot = pot
        self.alpha = alpha
        self.surface = surface
        self.num_pristine_atoms = self.surface.num_pristine_atoms
        logger.info("there are %d atoms in pristine slab", self.num_pristine_atoms)

        # initialize history
        self.history = []
        self.energy_hist = np.random.rand(self.total_sweeps)
        self.adsorption_count_hist = defaultdict(list)
        self.frac_accept_hist = np.random.rand(self.total_sweeps)

        self.setup_folders()

        self.get_adsorption_coords()

        self.curr_energy = self.get_initial_energy()

        self.prepare_canonical(even_adsorption_sites=even_adsorption_sites)

        self.sweep_size = sweep_size

        logger.info(
            "running for %s iterations per run over a total of %s runs",
            self.sweep_size,
            self.total_sweeps,
        )
        # new parameters
        # self.start_temp
        # self.peak_scale
        # self.ramp_up_sweeps
        # self.ramp_down_sweeps
        # self.total_sweeps
        if type(anneal_schedule) == list or type(anneal_schedule) == np.ndarray:
            temp_list = anneal_schedule
        elif perform_annealing:
            temp_list = self.create_anneal_schedule()
        else:
            temp_list = np.repeat(
                self.start_temp, self.total_sweeps
            )  # constant temperature
        logger.info("starting with iteration %d", starting_iteration)
        logger.info("temp list is: %s", temp_list)
        for i in range(starting_iteration, self.total_sweeps):
            self.temp = temp_list[i]
            self.mcmc_sweep(i=i)  # TODO change to .step

        # plot and save the results
        # TODO should be moved outside to script
        plot_summary_stats(
            self.energy_hist,
            self.frac_accept_hist,
            self.adsorption_count_hist,
            self.total_sweeps,
            self.run_folder,
        )

        return (
            self.history,
            self.energy_hist,
            self.frac_accept_hist,
            self.adsorption_count_hist,
            self.run_folder,
        )

    # TODO move to utils/sampling.py
    def create_anneal_schedule(self):
        temp_list = [self.start_temp]

        curr_sweep = 1
        curr_temp = self.start_temp
        while curr_sweep < self.total_sweeps:
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

        temp_list = temp_list[: self.total_sweeps]

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(temp_list)
        plt.savefig(f"{self.run_folder}/anneal_schedule.png")
        with open(f"{self.run_folder}/anneal_schedule.csv", "w") as f:
            f.write(",".join([str(temp) for temp in temp_list]))
        return temp_list


if __name__ == "__main__":
    pass
