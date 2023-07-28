"""Performs sampling of surface reconstructions using an MCMC-based algorithm"""

import copy
import json
import logging
import os
import pickle as pkl
from collections import Counter, defaultdict
from datetime import datetime

import ase
import catkit
import numpy as np
from ase.calculators.eam import EAM
from ase.constraints import FixAtoms
from ase.io import write
from catkit.gen.adsorption import get_adsorption_sites
from nff.io.ase import AtomsBatch

from .energy import optimize_slab, slab_energy
from .plot import plot_summary_stats
from .slab import (
    change_site,
    count_adsorption_sites,
    get_adsorption_coords,
    get_complementary_idx,
    get_random_idx,
    initialize_slab,
)
from .utils import filter_distances

logger = logging.getLogger(__name__)
file_dir = os.path.dirname(__file__)

ENERGY_DIFF_LIMIT = 1e3  # in eV
LOW_ENERGY_THRESHOLD = -1500  # for Si(111) 7x7 in eV
# LOW_ENERGY_THRESHOLD = -750  # for Si(111) 5x5 in eV
# LOW_ENERGY_THRESHOLD = -282  # for Si(111) 3x3 in eV


class MCMC:
    """MCMC-based class for sampling surface reconstructions."""

    def __init__(
        self,
        calc=EAM(
            potential=os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "potentials", "Cu2.eam.fs"
            )
        ),
        surface_name=None,
        element="Cu",
        canonical=False,
        num_ads_atoms=0,
        ads_coords=[],
        testing=False,
        adsorbates=None,
        relax=False,
        **kwargs,
    ) -> None:
        self.calc = calc
        self.surface_name = surface_name
        self.element = element  # review this, might not be useful
        self.canonical = canonical
        self.num_ads_atoms = num_ads_atoms
        self.ads_coords = ads_coords
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
        self.slab = None

        self.num_pristine_atoms = 0
        self.run_folder = ""
        self.curr_energy = 0
        self.sweep_size = 100
        self.state = None
        self.connectivity = None
        self.history = None
        self.energy_hist = None
        self.adsorption_count_hist = None
        self.frac_accept_hist = None

        self.per_atom_energies = []

        if self.canonical:
            # perform canonical runs
            # adsorb num_ads_atoms
            assert (
                self.num_ads_atoms > 0
            ), "for canonical runs, need number of adsorbed atoms greater than 0"

    def run(self):
        """The function "run" calls the function "mcmc_run"."""
        self.mcmc_run()

    def get_adsorption_coords(self):
        """If not already set, this function sets the absolute adsorption coordinates for a given slab and element
        using the catkit `get_adsorption_sites` method.

        """
        # get absolute adsorption coords
        elem = catkit.gratoms.Gratoms(self.element)

        if not (
            (isinstance(self.ads_coords, list) and (len(self.ads_coords) > 0))
            or isinstance(self.ads_coords, np.ndarray)
        ):
            # get ALL the adsorption sites
            # top should have connectivity 1, bridge should be 2 and hollow more like 4
            _, self.connectivity, _ = get_adsorption_sites(
                self.slab, symmetry_reduced=False
            )
            self.ads_coords = get_adsorption_coords(
                self.slab, elem, self.connectivity, debug=True
            )
        else:
            # fake connectivity for user defined adsorption sites
            self.connectivity = np.ones(len(self.ads_coords), dtype=int)

        self.site_types = set(self.connectivity)

        logger.info(
            f"In pristine slab, there are a total of {len(self.ads_coords)} sites"
        )

    def initialize_state(self):
        # state of each adsorption site in slab. for state > 0, it's filled, and that's the index of the adsorbate atom in slab
        if not (
            (isinstance(self.state, list) and (len(self.state) > 0))
            or isinstance(self.state, np.ndarray)
        ):
            self.state = np.zeros(len(self.ads_coords), dtype=int)
        logger.info(f"initial state is {self.state}")

    def set_constraints(self):
        """This function sets constraints on the atoms in a slab object, fixing the positions of bulk atoms
        and allowing surface atoms to move.

        """
        num_bulk_atoms = len(self.slab)
        # constraint all the bulk atoms
        bulk_indices = list(range(num_bulk_atoms))
        # constraint only the surface elements

        c = FixAtoms(indices=bulk_indices)
        self.slab.set_constraint(c)

    def set_adsorbates(self):
        """This function sets the adsorbates if not defined and converts the potential and adsorbate to a list if they are
        not already in list format.

        """
        # set adsorbate
        if not self.adsorbates:
            self.adsorbates = self.element
        logger.info(f"adsorbate(s) is(are) {self.adsorbates}")

        # change int pot and adsorbate to list
        if type(self.pot) == int:
            self.pot = list([self.pot])
        if isinstance(self.adsorbates, str):
            self.adsorbates = list([self.adsorbates])

    def initialize_tags(self):
        """This function initializes tags for surface and bulk atoms in a slab object."""
        # initialize tags
        # set tags; 1 for surface atoms and 0 for bulk
        if type(self.slab) is catkit.gratoms.Gratoms:
            surface_atoms = self.slab.get_surface_atoms()
            atoms_arr = np.arange(0, len(self.slab))
            base_tags = np.int8(np.isin(atoms_arr, surface_atoms)).tolist()
            self.slab.set_tags(list(base_tags))

    def prepare_slab(self):
        """Initializes a default slab if not supplied by user. Attaches the calculator and sets the number of pristine atoms."""
        # Cu lattice at 293 K, 3.6147 Å, potential ranges from 0 - 2
        if (
            (type(self.slab) is not catkit.gratoms.Gratoms)
            and (type(self.slab) is not ase.atoms.Atoms)
            and (type(self.slab) is not AtomsBatch)
        ):
            # initialize slab
            logger.info("initializing slab")
            # Cu alat from https://www.copper.org/resources/properties/atomic_properties.html
            Cu_alat = 3.6147
            self.slab = initialize_slab(Cu_alat)

        # attach slab calculator
        self.slab.calc = self.calc
        logger.info(f"using slab calc {self.slab.calc}")

        self.slab.write(os.path.join(self.run_folder, "starting_slab.cif"))
        # with open(os.path.join(self.run_folder, "starting_slab.pkl")) as f:
        #     pkl.dump(self.slab, f)
        self.initialize_tags()

    def setup_folders(self):
        """Set up folders for simulation depending on whether it's semi-grand canonical or canonical."""
        # set surface_name
        if not self.surface_name:
            self.surface_name = self.element

        if not self.run_folder:
            start_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

            # prepare both run folders
            canonical_run_folder = os.path.join(
                os.getcwd(),
                f"{self.surface_name}/runs{self.total_sweeps}_temp{self.temp}_adsatoms{self.num_ads_atoms:02}_alpha{self.alpha}_{start_timestamp}",
            )
            sgc_run_folder = os.path.join(
                os.getcwd(),
                f"{self.surface_name}/runs{self.total_sweeps}_temp{self.temp}_pot{self.pot}_alpha{self.alpha}_{start_timestamp}",
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
            f"Running with num_sweeps = {self.total_sweeps}, temp = {self.temp}, pot = {self.pot}, alpha = {self.alpha}"
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
        if self.slab.calc:
            results = slab_energy(
                self.slab, relax=self.relax, folder_name=self.run_folder, **self.kwargs
            )
            energy = results[0]
            self.per_atom_energies = results[-1]
        else:
            energy = 0

        return energy

    def prepare_canonical(self):
        """This function prepares a canonical slab by performing semi-grand canonical adsorption runs until the
        desired number of adsorbed atoms are obtained.

        """
        if self.canonical:
            # new method to adsorb evenly
            # change_site(
            #     self.slab,
            #     state,
            #     pots,
            #     adsorbates,
            #     coords,
            #     site_idx,
            #     start_ads=None,
            #     end_ads=None,
            # )

            # self.slab,
            # self.state,
            # self.pot,
            # self.adsorbates,
            # self.ads_coords,
            # site_idx,
            # start_ads=None,
            # end_ads=None,

            # perform canonical runs
            # adsorb num_ads_atoms
            assert (
                self.num_ads_atoms > 0
            ), "for canonical runs, need number of adsorbed atoms greater than 0"
            # new method to adsorb evenly
            # select subset of sites to adsorb
            # starting_sites = (
            #     np.linspace(0, len(self.ads_coords), self.num_ads_atoms)
            #     .astype(np.int)
            #     .tolist()
            # )
            # print("canonical starting sites are", starting_sites)
            # import itertools

            # site_iterator = itertools.cycle(starting_sites)
            if even_adsorption_sites:
                logger.info("evenly adsorbing sites")
                # Method 1
                # sites_idx = np.linspace(0, len(self.ads_coords)-1, self.num_ads_atoms).astype(    
                #     np.int
                # )

                # Method 2
                # analytically determine the centroids
                cell_lengths = self.slab.get_cell_lengths_and_angles()[0:3]
                x = np.linspace(0, cell_lengths[0]*0.95, np.ceil(np.sqrt(self.num_ads_atoms)).astype(np.int))
                y = np.linspace(0, cell_lengths[1]*0.95, np.ceil(np.sqrt(self.num_ads_atoms)).astype(np.int))
                xx, yy = np.meshgrid(x, y)
                centroids = np.vstack([xx.ravel(), yy.ravel()]).T
                # sort by first coordinate then second
                sorted_coords = np.array(sorted(self.ads_coords, key=lambda x: (x[0], x[1])))
                # TBD

                # then find the closest points to the centroids
                closest_points_indices = cdist(self.ads_coords[:, :2], centroids).argmin(axis=0)
                sites_idx = np.random.choice(closest_points_indices, size=self.num_ads_atoms, replace=False)

                # Method 3
                # do clustering
                centers, labels = get_cluster_centers(self.ads_coords[:, :2], self.num_ads_atoms)
                sites_idx = find_closest_points_indices(self.ads_coords[:, :2], centers, labels)
                plot_clustering_results(self.ads_coords, self.num_ads_atoms, labels, sites_idx, save_folder=self.run_folder,)

                for site_idx in sites_idx:
                    self.curr_energy, _ = self.change_site(
                        prev_energy=self.curr_energy, site_idx=site_idx
                    )
            else:
                logger.info("randomly adsorbing sites")
                # perform semi-grand canonical until num_ads_atoms are obtained
                while len(self.slab) < self.num_pristine_atoms + self.num_ads_atoms:
                    self.curr_energy, _ = self.change_site(prev_energy=self.curr_energy)

                    # site_idx = next(site_iterator)
                    # self.curr_energy, _ = self.change_site(
                    #     prev_energy=self.curr_energy, site_idx=site_idx
                    # )

            self.slab.write(
                os.path.join(self.run_folder, f"{self.surface_name}_canonical_init.cif")
            )

    def save_structures(self, i: int = 0, **kwargs):
        """This function saves the optimized structure of a slab and calculates its energy and force error.

        Parameters
        ----------
        i : int, optional
            The parameter `i` is an integer that represents the iteration number. It has a default value of
        0.

        Returns
        -------
            the energy of the optimized structure.

        """
        energy, energy_std, _, force_std, _ = slab_energy(
            self.slab,
            relax=self.relax,
            folder_name=self.run_folder,
            iter=i + 1,
            save=True,
            **self.kwargs,
        )
        if type(self.slab) is AtomsBatch:
            logger.info(
                f"current energy is {self.curr_energy}, calculated energy is {energy}"
            )
            assert np.allclose(
                energy, self.curr_energy, atol=1.0
            ), "self.curr_energy doesn't match calculated energy of current slab"

            if kwargs.get("offset_data", None):
                ads_pot_dict = dict(zip(self.adsorbates, self.pot))
                ads_count = Counter(self.slab.get_chemical_symbols())

                with open(kwargs["offset_data"]) as f:
                    offset_data = json.load(f)
                stoics = offset_data["stoics"]
                ref_element = offset_data["ref_element"]

                pot = 0
                for ele, _ in ads_count.items():
                    if ele != ref_element:
                        pot += (
                            ads_count[ele]
                            - stoics[ele] / stoics[ref_element] * ads_count[ref_element]
                        ) * ads_pot_dict[ele]

                energy -= pot
                logger.info(
                    "optim structure has Free Energy = {:.3f}+/-{:.3f}".format(
                        energy, energy_std
                    )
                )
            else:
                logger.info(
                    f"optim structure has Energy = {energy:.3f}+/-{energy_std:.3f}"
                )

            logger.info(f"average force error = {force_std:.3f}")

            # save cif and pkl file
            write(
                f"{self.run_folder}/final_slab_run_{i+1:03}_{energy:.3f}err{force_std:.3f}_{self.slab.get_chemical_formula()}.cif",
                self.slab,
            )
            save_slab = self.slab.copy()
            save_slab.calc = None
            with open(
                f"{self.run_folder}/final_slab_run_{i+1:03}_{energy:.3f}err{force_std:.3f}_{self.slab.get_chemical_formula()}.pkl",
                "wb",
            ) as f:
                pkl.dump(save_slab, f)

        else:
            energy = self.curr_energy
            logger.info(f"optim structure has Energy = {energy}")

            # save cif file
            write(
                f"{self.run_folder}/final_slab_run_{i+1:03}_{energy:.3f}_{self.slab.get_chemical_formula()}.cif",
                self.slab,
            )
            save_slab = self.slab.copy()
            save_slab.calc = None
            with open(
                f"{self.run_folder}/final_slab_run_{i+1:03}_{energy:.3f}_{self.slab.get_chemical_formula()}.pkl",
                "wb",
            ) as f:
                pkl.dump(save_slab, f)

        return energy

    def change_site_canonical(self, prev_energy: float = 0, iter: int = 1):
        """This function performs a canonical sampling step. It switches the adsorption sites of two
        adsorbates and checks if the change is energetically favorable.

        Parameters
        ----------
        prev_energy : float, optional
            The energy of the current state before attempting to change it. If it is not provided and the
        `testing` flag is not set, it will be calculated using the `slab_energy` function.
        iter : int, optional
            An integer representing the current iteration number of the function.

        Returns
        -------
            the energy of the new slab and a boolean value indicating whether the proposed change was
        accepted or not.

        """

        if not prev_energy and not self.testing:
            # calculate energy of current state
            results = slab_energy(
                self.slab, relax=self.relax, folder_name=self.run_folder, **self.kwargs
            )
            prev_energy = results[0]
            self.per_atom_energies = results[-1]

        # choose 2 sites of different ads (empty counts too) to switch
        site1_idx, site2_idx, site1_ads, site2_ads = get_complementary_idx(
            self.state,
            slab=self.slab,
            require_per_atom_energies=self.kwargs.get(
                "require_per_atom_energies", False
            ),
            per_atom_energies=self.per_atom_energies,
            temp=self.temp,
        )

        site1_coords = self.ads_coords[site1_idx]
        site2_coords = self.ads_coords[site2_idx]

        logger.debug(f"\n we are at iter {iter}")
        logger.debug(
            f"idx is {site1_idx} with connectivity {self.connectivity[site1_idx]} at {site1_coords}"
        )
        logger.debug(
            f"idx is {site2_idx} with connectivity {self.connectivity[site2_idx]} at {site2_coords}"
        )

        logger.debug(f"before proposed state is")
        logger.debug(self.state)

        logger.debug(f"current slab has {len(self.slab)} atoms")

        # effectively switch ads at both sites
        self.slab, self.state, _, _, _ = change_site(
            self.slab,
            self.state,
            self.pot,
            self.adsorbates,
            self.ads_coords,
            site1_idx,
            start_ads=site1_ads,
            end_ads=site2_ads,
            **self.kwargs,
        )
        self.slab, self.state, _, _, _ = change_site(
            self.slab,
            self.state,
            self.pot,
            self.adsorbates,
            self.ads_coords,
            site2_idx,
            start_ads=site2_ads,
            end_ads=site1_ads,
            **self.kwargs,
        )

        # make sure num atoms is conserved
        logger.debug(f"proposed slab has {len(self.slab)} atoms")

        logger.debug(f"after proposed state is")
        logger.debug(self.state)

        if self.kwargs.get("save_cif", False):
            if not os.path.exists(self.run_folder):
                os.makedirs(self.run_folder)
            write(f"{self.run_folder}/proposed_slab_iter_{iter:03}.cif", self.slab)

        # to test, always accept
        accept = False

        if self.kwargs.get("filter_distance", None):
            filter_distance = self.kwargs["filter_distance"]
            energy = 0

            if filter_distances(
                self.slab, ads=self.adsorbates, cutoff_distance=filter_distance
            ):
                # succeeds! keep already changed slab
                logger.debug("state changed with filtering!")
                accept = True
            else:
                # failed, keep current state and revert slab back to original
                self.slab, self.state, _, _, _ = change_site(
                    self.slab,
                    self.state,
                    self.pot,
                    self.adsorbates,
                    self.ads_coords,
                    site1_idx,
                    start_ads=site2_ads,
                    end_ads=site1_ads,
                    **self.kwargs,
                )
                self.slab, self.state, _, _, _ = change_site(
                    self.slab,
                    self.state,
                    self.pot,
                    self.adsorbates,
                    self.ads_coords,
                    site2_idx,
                    start_ads=site1_ads,
                    end_ads=site2_ads,
                    **self.kwargs,
                )

                logger.debug("state kept the same with filtering")

        elif self.testing:
            energy = 0
        else:
            # use relaxation only to get lowest energy
            # but don't update adsorption positions
            results = slab_energy(
                self.slab, relax=self.relax, folder_name=self.run_folder, **self.kwargs
            )
            curr_energy = results[0]
            self.per_atom_energies = results[-1]
            logger.debug(f"prev energy is {prev_energy}")
            logger.debug(f"curr energy is {curr_energy}")

            # energy change due to site change
            energy_diff = curr_energy - prev_energy

            # check if transition succeeds
            logger.debug(f"energy diff is {energy_diff}")
            logger.debug(f"k_b T {self.temp}")
            # set limit for energy_diff, reject very large energy changes
            if np.abs(energy_diff) > ENERGY_DIFF_LIMIT:
                base_prob = 0.0
            else:
                base_prob = np.exp(-energy_diff / self.temp)
            logger.debug(f"base probability is {base_prob}")

            if np.random.rand() < base_prob:
                # succeeds! keep already changed slab
                # state = state.copy()
                logger.debug("state changed!")
                energy = curr_energy
                accept = True
            else:
                # failed, keep current state and revert slab back to original
                self.slab, self.state, _, _, _ = change_site(
                    self.slab,
                    self.state,
                    self.pot,
                    self.adsorbates,
                    self.ads_coords,
                    site1_idx,
                    start_ads=site2_ads,
                    end_ads=site1_ads,
                    **self.kwargs,
                )
                self.slab, self.state, _, _, _ = change_site(
                    self.slab,
                    self.state,
                    self.pot,
                    self.adsorbates,
                    self.ads_coords,
                    site2_idx,
                    start_ads=site1_ads,
                    end_ads=site2_ads,
                    **self.kwargs,
                )

                # state, slab = add_to_slab(slab, state, adsorbate, coords, site1_idx)
                # state, slab = remove_from_slab(slab, state, site2_idx)

                logger.debug("state kept the same")
                energy = prev_energy
                accept = False

        return energy, accept

    def change_site(self, prev_energy: float = 0, iter: int = 1, site_idx: int = None):
        """This function performs a semi-grand canonical sampling iteration. It randomly chooses a site to change identity in a slab, adds or removes an atom from the site,
        optionally performs relaxation, calculates the energy of the new slab, and accepts or rejects the change based
        on the Boltzmann-weighted energy difference, chemical potential change, and temperature.

        Parameters
        ----------
        prev_energy : float, optional
            the energy of the slab before the
        iter : int, optional
            The iteration number of the simulation.
        site_idx : int, optional
            Specify the index of the site to switch.

        Returns
        -------
            the energy of the new slab and a boolean value indicating whether the proposed change was
        accepted or not.

        """
        if not site_idx:
            site_idx = get_random_idx(self.connectivity)
        rand_site = self.ads_coords[site_idx]

        logger.debug(f"\n we are at iter {iter}")
        logger.debug(
            f"idx is {site_idx} with connectivity {self.connectivity[site_idx]} at {rand_site}"
        )

        logger.debug("before proposed state is")
        logger.debug(self.state)

        # change in number of adsorbates (atoms)
        delta_N = 0

        if not prev_energy and not self.testing:
            results = slab_energy(
                self.slab, relax=self.relax, folder_name=self.run_folder, **self.kwargs
            )
            prev_energy = results[0]
            self.per_atom_energies = results[-1]
        self.slab, self.state, delta_pot, start_ads, end_ads = change_site(
            self.slab,
            self.state,
            self.pot,
            self.adsorbates,
            self.ads_coords,
            site_idx,
            start_ads=None,
            end_ads=None,
            **self.kwargs,
        )

        logger.debug("after proposed state is")
        logger.debug(self.state)

        if self.kwargs.get("save_cif", False):
            if not os.path.exists(self.run_folder):
                os.makedirs(self.run_folder)
            write(f"{self.run_folder}/proposed_slab_iter_{iter:03}.cif", self.slab)

        # to test, always accept
        accept = False
        if self.kwargs.get("filter_distance", None):
            filter_distance = self.kwargs["filter_distance"]
            energy = 0

            if filter_distances(
                self.slab, ads=self.adsorbates, cutoff_distance=filter_distance
            ):
                # succeeds! keep already changed slab
                logger.debug("state changed with filtering!")
                accept = True
            else:
                # failed, keep current state and revert slab back to original
                self.slab, self.state, _, _, _ = change_site(
                    self.slab,
                    self.state,
                    self.pot,
                    self.adsorbates,
                    self.ads_coords,
                    site_idx,
                    start_ads=end_ads,
                    end_ads=start_ads,
                    **self.kwargs,
                )
                logger.debug("state kept the same with filtering")

        elif self.testing:
            energy = 0
            accept = True

        else:
            # use relaxation only to get lowest energy
            # but don't update adsorption positions
            results = slab_energy(
                self.slab,
                relax=self.relax,
                folder_name=self.run_folder,
                iter=iter,
                **self.kwargs,
            )
            curr_energy = results[0]
            self.per_atom_energies = results[-1]

            logger.debug(f"prev energy is {prev_energy}")
            logger.debug(f"curr energy is {curr_energy}")

            # energy change due to site change
            energy_diff = curr_energy - prev_energy

            # check if transition succeeds
            # min(1, exp(-(\delta_E-(delta_N*pot))))
            logger.debug(f"energy diff is {energy_diff}")
            logger.debug(f"chem pot(s) is(are) {self.pot}")
            logger.debug(f"delta_N {delta_N}")
            logger.debug(f"delta_pot_{delta_pot}")
            logger.debug(f"k_b T {self.temp}")

            if np.abs(energy_diff) > ENERGY_DIFF_LIMIT:
                base_prob = 0.0
            else:
                base_prob = np.exp(-(energy_diff - delta_pot) / self.temp)

            logger.debug(f"base probability is {base_prob}")
            if np.random.rand() < base_prob:
                # succeeds! keep already changed slab
                # state = state.copy()
                logger.debug("state changed!")
                energy = curr_energy
                accept = True
            else:
                # failed, keep current state and revert slab back to original
                self.slab, self.state, _, _, _ = change_site(
                    self.slab,
                    self.state,
                    self.pot,
                    self.adsorbates,
                    self.ads_coords,
                    site_idx,
                    start_ads=end_ads,
                    end_ads=start_ads,
                    **self.kwargs,
                )

                logger.debug("state kept the same")
                energy = prev_energy
                accept = False

            # logger.debug(f"energy after accept/reject {slab_energy(slab, relax=relax, folder_name=folder_name, iter=iter, **kwargs)}")
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
        logger.info(f"In sweep {i+1} out of {self.total_sweeps}")
        for j in range(self.sweep_size):
            run_idx = self.sweep_size * i + j + 1
            if self.canonical:
                self.curr_energy, accept = self.change_site_canonical(
                    prev_energy=self.curr_energy, iter=run_idx
                )
            else:
                self.curr_energy, accept = self.change_site(
                    prev_energy=self.curr_energy, iter=run_idx
                )
            num_accept += accept

            # save low energy structure

            if self.curr_energy < LOW_ENERGY_THRESHOLD:
                optimized_slab, _ = optimize_slab(
                    self.slab,
                    optimizer=self.kwargs["optimizer"],
                    kim_potential=self.kwargs.get("kim_potential", None),
                    folder_name=self.run_folder,
                )
                optimized_slab.write(
                    f"{self.run_folder}/optim_slab_run_idx_{run_idx:06}_{optimized_slab.get_chemical_formula()}_energy_{optimized_slab.get_potential_energy():.3f}.cif"
                )
                with open(
                    f"{self.run_folder}/optim_slab_run_idx_{run_idx:06}_{optimized_slab.get_chemical_formula()}_energy_{optimized_slab.get_potential_energy():.3f}.pkl",
                    "wb",
                ) as f:
                    pkl.dump(optimized_slab, f)

        # end of sweep, append to history
        if self.relax:
            history_slab, _ = optimize_slab(
                self.slab,
                kim_potential=self.kwargs.get("kim_potential", None),
                relax_steps=self.kwargs.get("relax_steps", 20),
                optimizer=self.kwargs.get("optimizer", None),
                folder_name=self.run_folder,
            )
            history_slab.calc = None
        elif type(self.slab) is AtomsBatch:
            history_slab = copy.deepcopy(self.slab)
            history_slab.calc = None
        else:
            history_slab = self.slab.copy()
        self.history.append(history_slab)
        # TODO can save some compute here

        final_energy = self.save_structures(i=i, **self.kwargs)

        # append values
        self.energy_hist[i] = final_energy

        ads_counts = count_adsorption_sites(self.slab, self.state, self.connectivity)
        for key in set(self.site_types):
            if ads_counts[key]:
                self.adsorption_count_hist[key].append(ads_counts[key])
            else:
                self.adsorption_count_hist[key].append(0)

        frac_accept = num_accept / self.sweep_size
        self.frac_accept_hist[i] = frac_accept

    def mcmc_run(
        self,
        peak_scale: float = 1 / 2,
        ramp_up_sweeps: int = 10,
        ramp_down_sweeps: int = 200,
        total_sweeps: int = 800,
        start_temp: float = 1.0,
        pot: float or list = 1.0,
        alpha: float = 0.9,
        slab: ase.atoms.Atoms or catkit.gratoms.Gratoms or AtomsBatch = None,
        state: list or np.ndarray = None,
        num_pristine_atoms: int = 0,
        anneal_schedule: list = None,
        run_folder: str = None,
        starting_iteration: list = 0
    ):
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
        slab : ase.atoms.Atoms or catkit.gratoms.Gratoms or AtomsBatch, optional
            The `slab` is the starting surface structure on which the MC simulation is
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
        self.slab = slab
        self.state = state

        if num_pristine_atoms == 0:
            self.num_pristine_atoms = len(self.slab)
        else:
            self.num_pristine_atoms = num_pristine_atoms
        logger.info("there are %d atoms in pristine slab", self.num_pristine_atoms)

        # initialize history
        self.history = []
        self.energy_hist = np.random.rand(self.total_sweeps)
        self.adsorption_count_hist = defaultdict(list)
        self.frac_accept_hist = np.random.rand(self.total_sweeps)

        self.setup_folders()

        self.prepare_slab()

        self.set_adsorbates()

        self.get_adsorption_coords()

        self.initialize_state()

        self.curr_energy = self.get_initial_energy()

        self.prepare_canonical()

        # sweep over # sites
        # self.sweep_size = len(self.ads_coords)
        self.sweep_size = 300

        logger.info(
            f"running for {self.sweep_size} iterations per run over a total of {self.total_sweeps} runs"
        )
        # new parameters
        # self.start_temp
        # self.peak_scale
        # self.ramp_up_sweeps
        # self.ramp_down_sweeps
        # self.total_sweeps
        if type(anneal_schedule) == list or type(anneal_schedule) == np.ndarray :
            temp_list = anneal_schedule
        else:
            temp_list = self.create_anneal_schedule()

        logger.info(f"starting with iteration {starting_iteration}")
        for i in range(starting_iteration, self.total_sweeps):
            self.temp = temp_list[i]
            self.mcmc_sweep(i=i)

        # plot and save the results
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

    def create_anneal_schedule(self):
        temp_list = [self.start_temp]

        curr_sweep = 1
        curr_temp = self.start_temp
        while curr_sweep < self.total_sweeps:
            for i in range(self.ramp_down_sweeps):
                # simulated annealing schedule
                curr_temp = curr_temp * self.alpha
                temp_list.append(curr_temp)
                curr_sweep += 1
            if self.start_temp >= 1.0:
                self.start_temp *= self.peak_scale
            else:
                self.start_temp -= 0.1
            # ramp up
            temp_list.extend(
                np.linspace(curr_temp, self.start_temp, self.ramp_up_sweeps).tolist()
            )
            curr_temp = self.start_temp  # reset to new start temp

            curr_sweep += self.ramp_up_sweeps
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
