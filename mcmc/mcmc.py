"""Performs Semi-Grand Monte Carlo (SGMC) reconstruction of a surface.
Produces a temperature/structure map
"""

import copy
import os
import sys

from nff.io.ase import AtomsBatch

sys.path.append("/home/dux/")
import cProfile
import logging
from collections import Counter, defaultdict
from datetime import datetime
from pstats import Stats
from time import perf_counter

import ase
import catkit
import matplotlib.pyplot as plt
import numpy as np
from ase.calculators.eam import EAM
from ase.calculators.lammpsrun import LAMMPS
from ase.constraints import FixAtoms
from ase.io import read, write
from catkit.gen.adsorption import get_adsorption_sites

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
from .utils import filter_distances, filter_distances_new

# from htvs.djangochem.pgmols.utils import surfaces

logger = logging.getLogger(__name__)
file_dir = os.path.dirname(__file__)


class MCMC:
    def __init__(
        self,
        num_sweeps=1000,
        temp=1,
        pot=1,
        alpha=0.9,
        slab=None,
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
        self.num_sweeps = num_sweeps
        self.temp = temp
        self.pot = pot
        self.alpha = alpha
        self.slab = slab
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
        self.history = None
        self.energy_hist = None
        self.adsorption_count_hist = None
        self.frac_accept_hist = None

        if self.canonical:
            # perform canonical runs
            # adsorb num_ads_atoms
            assert (
                self.num_ads_atoms > 0
            ), "for canonical runs, need number of adsorbed atoms greater than 0"

    def run(self):
        self.mcmc_run()

    def get_adsorption_coords(self):
        # get absolute adsorption coords
        metal = catkit.gratoms.Gratoms(self.element)

        if not (
            (isinstance(self.ads_coords, list) and (len(self.ads_coords) > 0))
            or isinstance(self.ads_coords, np.ndarray)
        ):
            # get ALL the adsorption sites
            # top should have connectivity 1, bridge should be 2 and hollow more like 4
            coords, self.connectivity, sym_idx = get_adsorption_sites(
                self.slab, symmetry_reduced=False
            )
            self.ads_coords = get_adsorption_coords(
                self.slab, metal, self.connectivity, debug=True
            )
        else:
            # fake connectivity
            self.connectivity = np.ones(len(self.ads_coords), dtype=int)

        self.site_types = set(self.connectivity)

        # state of each vacancy in slab. for state > 0, it's filled, and that's the index of the adsorbate atom in slab
        self.state = np.zeros(len(self.ads_coords), dtype=int)

        logger.info(
            f"In pristine slab, there are a total of {len(self.ads_coords)} sites"
        )

    def set_constraints(self):
        num_bulk_atoms = len(self.slab)
        bulk_indices = list(range(num_bulk_atoms))

        c = FixAtoms(indices=bulk_indices)
        self.slab.set_constraint(c)

    def set_adsorbates(self):
        # set adsorbate
        if not self.adsorbates:
            self.adsorbates = self.element
        logger.info(f"adsorbate(s) is(are) {self.adsorbates}")

        # change int pot and adsorbate to list
        if type(self.pot) == int:
            self.pot = list([self.pot])
        if type(self.adsorbates) == str:
            self.adsorbates = list([self.adsorbates])

    def initialize_tags(self):
        # initialize tags
        # set tags; 1 for surface atoms, 2 for adsorbates, 0 for others
        if type(self.slab) is catkit.gratoms.Gratoms:
            surface_atoms = self.slab.get_surface_atoms()
            atoms_arr = np.arange(0, len(self.slab))
            base_tags = np.int8(np.isin(atoms_arr, surface_atoms)).tolist()
            self.slab.set_tags(list(base_tags))

    def prepare_slab(self):
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

        self.set_constraints()

        self.num_pristine_atoms = len(self.slab)
        logger.info(f"there are {self.num_pristine_atoms} atoms in pristine slab")

        self.initialize_tags()

    def setup_folders(self):
        # set surface_name
        if not self.surface_name:
            self.surface_name = self.element

        start_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # prepare both run folders
        self.canonical_run_folder = os.path.join(
            os.getcwd(),
            f"{self.surface_name}/runs{self.num_sweeps}_temp{self.temp}_adsatoms{self.num_ads_atoms:02}_alpha{self.alpha}_{start_timestamp}",
        )
        self.sgc_run_folder = os.path.join(
            os.getcwd(),
            f"{self.surface_name}/runs{self.num_sweeps}_temp{self.temp}_pot{self.pot}_alpha{self.alpha}_{start_timestamp}",
        )

        # default to sgc fun folder
        if self.canonical:
            self.run_folder = self.canonical_run_folder
        else:
            self.run_folder = self.sgc_run_folder

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
            f"Running with num_sweeps = {self.num_sweeps}, temp = {self.temp}, pot = {self.pot}, alpha = {self.alpha}"
        )

    def get_initial_energy(self):
        # sometimes slab.calc is fake
        if self.slab.calc:
            energy, _, _ = slab_energy(self.slab, **self.kwargs)
        else:
            energy = 0

        return energy

    def prepare_canonical(self):
        if self.canonical:
            # perform canonical runs
            # adsorb num_ads_atoms
            assert (
                self.num_ads_atoms > 0
            ), "for canonical runs, need number of adsorbed atoms greater than 0"

            # perform grand canonical until num_ads_atoms are obtained
            while len(self.slab) < self.num_pristine_atoms + self.num_ads_atoms:
                self.curr_energy, accept = self.spin_flip(prev_energy=self.curr_energy)

            self.slab.write(f"{self.surface_name}_canonical_init.cif")

    def save_structures(self, i=0):
        if type(self.slab) is AtomsBatch:
            # add in uncertainty information
            # slab.update_nbr_list(update_atoms=True)
            # slab.calc.calculate(slab)
            # energy = float(slab.results["energy"])
            energy, energy_std, force_std = slab_energy(
                self.slab, relax=self.relax, folder_name=self.run_folder, **self.kwargs
            )

            assert np.allclose(
                energy, self.curr_energy
            ), "self.curr_energy doesn't match calculated energy of current slab"

            if not set(["O", "Sr", "Ti"]) ^ set(self.adsorbates):
                ads_count = Counter(self.slab.get_chemical_symbols())
                ads_pot_dict = dict(zip(self.adsorbates, self.pot))
                delta_pot = (ads_count["O"] - 3 * ads_count["Ti"]) * ads_pot_dict[
                    "O"
                ] + (ads_count["Sr"] - ads_count["Ti"]) * ads_pot_dict["Sr"]
                energy -= delta_pot
                logger.info(
                    f"optim structure has Free Energy = {energy:.3f}+/-{energy_std:.3f}"
                )
            else:
                # energy = float(slab.results["energy"])
                logger.info(
                    f"optim structure has Energy = {energy:.3f}+/-{energy_std:.3f}"
                )

            logger.info(f"average force error = {force_std:.3f}")

            # save cif file
            write(
                f"{self.run_folder}/final_slab_run_{i+1:03}_{energy:.3f}err{force_std:.3f}.cif",
                self.slab,
            )

        else:
            logger.info(f"optim structure has Energy = {self.curr_energy}")

            # save cif file
            write(
                f"{self.run_folder}/final_slab_run_{i+1:03}_{self.curr_energy:.3f}.cif",
                self.slab,
            )

        if self.relax:
            opt_slab = optimize_slab(
                self.slab,
                folder_name=self.run_folder,
                relax_steps=self.kwargs.get("relax_steps", 20),
            )
            opt_slab.write(f"{self.run_folder}/optim_slab_run_{i+1:03}.cif")

    def spin_flip_canonical(self, prev_energy=0, iter=1):
        """Based on the Ising model, models the adsorption/desorption of atoms from surface lattice sites.
        Uses canonical ensemble, fixed number of atoms.

        Parameters
        ----------
        state : np.array
            dimension the number of sites
        slab : catkit.gratoms.Gratoms
            model of the surface slab
        temp : float
            temperature

        Returns
        -------
        np.array, float
            new state, energy change
        """

        if not prev_energy and not self.testing:
            # calculate energy of current state
            prev_energy, _, _ = slab_energy(
                self.slab, relax=self.relax, folder_name=self.run_folder, **self.kwargs
            )

        # choose 2 sites of different ads (empty counts too) to flip
        site1_idx, site2_idx, site1_ads, site2_ads = get_complementary_idx(
            self.state, slab=self.slab
        )

        # fake pots
        # pots = list(range(len(self.adsorbates)))

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

            if filter_distances_new(
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
                )

                logger.debug("state kept the same with filtering")

        elif self.testing:
            # state = state.copy() # obviously inefficient but here for a reason
            energy = 0
        else:
            # use relaxation only to get lowest energy
            # but don't update adsorption positions
            curr_energy, _, _ = slab_energy(
                self.slab, relax=self.relax, folder_name=self.run_folder, **self.kwargs
            )

            logger.debug(f"prev energy is {prev_energy}")
            logger.debug(f"curr energy is {curr_energy}")

            # energy change due to flipping spin
            energy_diff = curr_energy - prev_energy

            # check if transition succeeds
            logger.debug(f"energy diff is {energy_diff}")
            logger.debug(f"k_b T {self.temp}")
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
                )

                # state, slab = add_to_slab(slab, state, adsorbate, coords, site1_idx)
                # state, slab = remove_from_slab(slab, state, site2_idx)

                logger.debug("state kept the same")
                energy = prev_energy
                accept = False

        return energy, accept

    def spin_flip(self, prev_energy=0, iter=1, site_idx=None):

        """It takes in a slab, a state, and a temperature, and it randomly chooses a site to flip. If the site
        is empty, it adds an atom to the slab and updates the state. If the site is filled, it removes an
        atom from the slab and updates the state. It then calculates the energy of the new slab and compares
        it to the energy of the old slab. If the new energy is lower, it accepts the change. If the new
        energy is higher, it accepts the change with a probability that depends on the temperature

        Parameters
        ----------
        state
            the current state of the adsorption sites, which is a list of the corresponding indices in the slab.
        slab
            the slab object
        temp
            the temperature of the system
        pot
            the chemical potential of the adsorbate
        coords
            the coordinates of the adsorption sites
        connectivity
            a list of lists, where each list is the indices of the sites that are connected to the site at the
        same index.
        prev_energy
            the energy of the slab before the spin flip
        save_cif, optional
            if True, will save the proposed slab to a cif file
        iter, optional
            the iteration number
        site_idx
            the index of the site to switch
        testing, optional
            if True, always accept the proposed state
        folder_name, optional
            the folder where the cif files will be saved
        adsorbate, optional
            the type of atom to adsorb
        relax, optional
            whether to relax the slab after adsorption

        Returns
        -------
            state, slab, energy, accept

        """
        if not site_idx:
            site_idx = get_random_idx(self.connectivity)
        rand_site = self.ads_coords[site_idx]

        logger.debug(f"\n we are at iter {iter}")
        logger.debug(
            f"idx is {site_idx} with connectivity {self.connectivity[site_idx]} at {rand_site}"
        )

        # determine if site vacant or filled
        # filled = (state > 0)[site_idx]
        logger.debug(f"before proposed state is")
        logger.debug(self.state)

        # change in number of adsorbates (atoms)
        delta_N = 0

        if not prev_energy and not self.testing:
            prev_energy, _, _ = slab_energy(
                self.slab, relax=self.relax, folder_name=self.run_folder, **self.kwargs
            )

        self.slab, self.state, delta_pot, start_ads, end_ads = change_site(
            self.slab,
            self.state,
            self.pot,
            self.adsorbates,
            self.ads_coords,
            site_idx,
            start_ads=None,
            end_ads=None,
        )

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

            if filter_distances_new(
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
                )
                logger.debug("state kept the same with filtering")

        elif self.testing:
            energy = 0
            accept = True

        else:
            # use relaxation only to get lowest energy
            # but don't update adsorption positions
            curr_energy, _, _ = slab_energy(
                self.slab,
                relax=self.relax,
                folder_name=self.run_folder,
                iter=iter,
                **self.kwargs,
            )

            logger.debug(f"prev energy is {prev_energy}")
            logger.debug(f"curr energy is {curr_energy}")

            # energy change due to flipping spin
            energy_diff = curr_energy - prev_energy

            # check if transition succeeds
            # min(1, exp(-(\delta_E-(delta_N*pot))))
            logger.debug(f"energy diff is {energy_diff}")
            logger.debug(f"chem pot(s) is(are) {self.pot}")
            logger.debug(f"delta_N {delta_N}")
            logger.debug(f"delta_pot_{delta_pot}")
            logger.debug(f"k_b T {self.temp}")
            base_prob = np.exp(-(energy_diff - delta_pot) / self.temp)
            logger.debug(f"base probability is {base_prob}")
            # breakpoint()
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
                )

                logger.debug("state kept the same")
                energy = prev_energy
                accept = False

            # logger.debug(f"energy after accept/reject {slab_energy(slab, relax=relax, folder_name=folder_name, iter=iter, **kwargs)}")
        return energy, accept

    def mcmc_sweep(self, i=0):
        num_accept = 0
        # simulated annealing schedule
        self.temp = self.temp * self.alpha
        logger.info(f"In sweep {i+1} out of {self.num_sweeps}")
        for j in range(self.sweep_size):
            # logger.info(f"In iter {j+1}")
            run_idx = self.sweep_size * i + j + 1
            if self.canonical:
                self.curr_energy, accept = self.spin_flip_canonical(
                    prev_energy=self.curr_energy, iter=run_idx
                )
            else:
                self.curr_energy, accept = self.spin_flip(
                    prev_energy=self.curr_energy, iter=run_idx
                )
            num_accept += accept

        # end of sweep; append to history
        if self.relax:
            history_slab = optimize_slab(
                self.slab, relax_steps=self.kwargs.get("relax_steps", 20)
            )
            history_slab.calc = None
        elif type(self.slab) is AtomsBatch:
            history_slab = copy.deepcopy(self.slab)
            history_slab.calc = None
        else:
            history_slab = self.slab.copy()
        self.history.append(history_slab)
        # TODO can save some compute here

        self.save_structures(i=i)

        # append values
        self.energy_hist[i] = self.curr_energy

        ads_counts = count_adsorption_sites(self.slab, self.state, self.connectivity)
        for key in set(self.site_types):
            if ads_counts[key]:
                self.adsorption_count_hist[key].append(ads_counts[key])
            else:
                self.adsorption_count_hist[key].append(0)

        frac_accept = num_accept / self.sweep_size
        self.frac_accept_hist[i] = frac_accept

    def mcmc_run(self, num_sweeps=1000, temp=1, pot=1, alpha=0.9, slab=None):
        """Performs MCMC sweep with given parameters, initializing with a random slab if not given an input.
        Each sweep is defined as running a number of trials equal to the number adsorption sites. Each trial
        consists of randomly picking a site and proposing (and accept/reject) a flip (adsorption or desorption).
        Only the resulting slab after one sweep is appended to the history. Corresponding observables are
        calculated also after each run.

        Returns
        -------
            history is a list of slab objects, energy_hist is a list of energies, frac_accept_hist is a list of
        fraction of accepted moves, adsorption_count_hist is a dictionary of lists of adsorption counts for
        each site type

        """

        self.num_sweeps = num_sweeps
        self.temp = temp
        self.pot = pot
        self.alpha = alpha
        self.slab = slab

        # initialize history
        self.history = []
        self.energy_hist = np.random.rand(self.num_sweeps)
        self.adsorption_count_hist = defaultdict(list)
        self.frac_accept_hist = np.random.rand(self.num_sweeps)

        self.setup_folders()

        self.prepare_slab()

        self.set_adsorbates()

        self.get_adsorption_coords()

        self.curr_energy = self.get_initial_energy()

        self.prepare_canonical()

        # perform actual mcmc sweeps
        # sweep over # sites
        self.sweep_size = len(self.ads_coords)
        # self.sweep_size = 2

        logger.info(
            f"running for {self.sweep_size} iterations per run over a total of {self.num_sweeps} runs"
        )

        for i in range(self.num_sweeps):
            self.mcmc_sweep(i=i)

        # plot and save the results
        plot_summary_stats(
            self.energy_hist,
            self.frac_accept_hist,
            self.adsorption_count_hist,
            self.num_sweeps,
            self.run_folder,
        )

        # early stopping
        # if i > 0 and abs(energy_hist[i-1] - energy_hist[i]) < 1e-2:
        #     return history, energy_hist, frac_accept_hist, adsorption_count_hist

        # TODO don't really need these, can cancel
        return (
            self.history,
            self.energy_hist,
            self.frac_accept_hist,
            self.adsorption_count_hist,
            self.run_folder,
        )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s:%(message)s",
        level=logging.DEBUG,
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )

    do_profiling = True

    # use EAM
    # eam_calc = EAM(potential='Cu2.eam.fs')

    # use LAMMPS
    alloy_parameters = {
        "pair_style": "eam/alloy",
        "pair_coeff": ["* * cu_ag_ymwu.eam.alloy Ag"],
    }
    alloy_potential_file = os.path.join(
        os.path.dirname(__file__), "cu_ag_ymwu.eam.alloy"
    )
    alloy_calc = LAMMPS(
        files=[alloy_potential_file],
        keep_tmp_files=False,
        keep_alive=False,
        tmp_dir="/home/dux/surface_sampling/tmp_files",
    )
    alloy_calc.set(**alloy_parameters)

    # Au from standard cell
    atoms = read("Ag_mp-124_conventional_standard.cif")
    # slab, surface_atoms = surfaces.surface_from_bulk(atoms, [1, 1, 1], size=[5, 5])
    # TODO: fix this
    slab.write("Ag_111_5x5_pristine_slab.cif")

    element = "Ag"
    # num_ads_atoms = 16 + 8
    adsorbate = "Cu"
    if do_profiling:
        with cProfile.Profile() as pr:
            start = perf_counter()
            # chem pot 0 to less complicate things
            # temp in terms of kbT
            # history, energy_hist, frac_accept_hist, adsorption_count_hist = mcmc_run(num_sweeps=10, temp=1, pot=0, slab=slab, calc=lammps_calc, element=element, canonical=True, num_ads_atoms=num_ads_atoms)
            history, energy_hist, frac_accept_hist, adsorption_count_hist = mcmc_run(
                num_sweeps=1,
                temp=1,
                pot=0,
                alpha=0.99,
                slab=slab,
                calc=alloy_calc,
                element=element,
                adsorbates=adsorbate,
            )
            stop = perf_counter()
            logger.info(f"Time taken = {stop - start} seconds")

        with open("profiling_stats.txt", "w") as stream:
            stats = Stats(pr, stream=stream)
            stats.strip_dirs()
            stats.sort_stats("time")
            stats.dump_stats(".prof_stats")
            stats.print_stats()
