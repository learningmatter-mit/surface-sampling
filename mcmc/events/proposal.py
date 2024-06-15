import logging
import random
from copy import deepcopy
from typing import Any, Dict, Iterable

import numpy as np

from mcmc.slab import get_complementary_idx
from mcmc.system import SurfaceSystem

logger = logging.getLogger(__name__)


class Proposal:
    """Base class for proposal moves in the Monte Carlo simulation.

    Args:
        system (SurfaceSystem): The surface system to propose changes to.
        adsorbate_list (Iterable[str]): The list of adsorbates that can be proposed.

    Attributes:
        adsorbate_list (List[str]): The list of adsorbates that can be proposed.
    """

    def __init__(self, system: SurfaceSystem, adsorbate_list: Iterable[str]) -> None:
        self.system = system
        self.adsorbate_list = list(adsorbate_list).copy()

    def get_action(self) -> Dict[str, Any]:
        """Obtain a dictionary containing the proposed change.

        Raises:
            NotImplementedError: This method should be implemented in the derived classes.
        """
        raise NotImplementedError


class ChangeProposal(Proposal):
    """Proposal to change single adsorbate at a site.

    Args:
        system (SurfaceSystem): The surface system to propose changes to.
        adsorbate_list (Iterable[str]): The list of adsorbates that can be proposed.

    Attributes:
        adsorbate_list (List[str]): The list of adsorbates that can be proposed.
    """

    def __init__(
        self,
        system: SurfaceSystem,
        adsorbate_list: Iterable[str] = ("Sr", "O"),
        site_idx: int = None,
    ) -> None:
        super().__init__(system, adsorbate_list)
        self.adsorbate_list.append("None")
        self.site_idx = site_idx

    def get_action(self) -> Dict[str, Any]:
        """Get an index and the elemental identity of the adsorbate to change into.

        Returns:
            Dict[str, Any]: A dictionary containing the index of the site and the elemental identity of the adsorbate to change into.
        """
        ads_choices = deepcopy(self.adsorbate_list)
        if isinstance(self.site_idx, int):
            site_idx = self.site_idx
        else:
            site_idx = np.random.choice(range(len(self.system.occ)))

        logger.debug("before proposed state is")
        logger.debug(self.system.occ)

        if self.system.occ[site_idx] != 0:
            # not an empty virtual site, remove the adsorbate
            start_ads = self.system.real_atoms[self.system.occ[site_idx]].symbol
            ads_choices.remove(start_ads)
        else:
            start_ads = "None"
            ads_choices.remove("None")

        end_ads = random.choice(ads_choices)

        action = {
            "name": "change",
            "site_idx": site_idx,
            "start_ads": start_ads,
            "end_ads": end_ads,
        }
        return action


class SwitchProposal(Proposal):
    """Proposal to switch two adsorbates at different sites.

    Args:
        system (SurfaceSystem): The surface system to propose changes to.
        adsorbate_list (Iterable[str]): The list of adsorbates that can be proposed.
        require_per_atom_energies (bool): Whether to require per atom energies.
        require_distance_decay (bool): Whether to require distance decay.
        per_atom_energies (np.ndarray): The per atom energies.
        distance_weight_matrix (np.ndarray): The distance weight matrix.
        temp (float): The temperature.
        run_folder (str): The folder to run the calculations in.
        plot_specific_distance_weights (bool): Whether to plot the specific distance weights.
        run_iter (int): The iteration number.

    Attributes:
        adsorbate_list (List[str]): The list of adsorbates that can be proposed.
        require_per_atom_energies (bool): Whether to require per atom energies.
        require_distance_decay (bool): Whether to require distance decay.
        per_atom_energies (np.ndarray): The per atom energies.
        distance_weight_matrix (np.ndarray): The distance weight matrix.
        temp (float): The temperature.
        run_folder (str): The folder to run the calculations in.
        plot_weights (bool): Whether to plot the specific distance weights.
        run_iter (int): The iteration number.
    """

    def __init__(
        self,
        system: SurfaceSystem,
        adsorbate_list: Iterable[str] = ("Sr", "O"),
        require_per_atom_energies: bool = False,
        require_distance_decay: bool = False,
        per_atom_energies: np.ndarray = None,
        distance_weight_matrix: np.ndarray = None,
        temp: float = None,
        run_folder: str = None,
        plot_specific_distance_weights: bool = False,
        run_iter: int = None,
    ) -> None:
        super().__init__(system, adsorbate_list)
        self.require_per_atom_energies = require_per_atom_energies
        self.require_distance_decay = require_distance_decay
        self.per_atom_energies = per_atom_energies
        self.distance_weight_matrix = distance_weight_matrix
        self.temp = temp
        self.run_folder = run_folder
        self.plot_weights = plot_specific_distance_weights
        self.run_iter = run_iter

    def get_action(self) -> Dict[str, Any]:
        """Get two indices, site1 and site2 of different elemental identities.

        Returns:
            Dict[str, Any]: A dictionary containing the indices of the two sites and the elemental identities of the adsorbates.
        """
        logger.debug("before proposed state is")
        logger.debug(self.system.occ)

        site1_idx, site2_idx, site1_ads, site2_ads = get_complementary_idx(
            self.system,
            require_per_atom_energies=self.require_per_atom_energies,
            require_distance_decay=self.require_distance_decay,
            per_atom_energies=self.per_atom_energies,  # TODO: move to Calculator
            distance_weight_matrix=self.distance_weight_matrix,  # TODO: move to SurfaceSystem
            temp=self.temp,
            ads_coords=self.system.ads_coords,
            run_folder=self.run_folder,
            plot_weights=self.plot_weights,
            run_iter=self.run_iter,
        )

        site1_coords = self.system.ads_coords[site1_idx]
        site2_coords = self.system.ads_coords[site2_idx]

        logger.debug("\n we are at iter %s", iter)
        logger.debug("idx is %s at %s", site1_idx, site1_coords)
        logger.debug("idx is %s at %s", site2_idx, site2_coords)

        action = {
            "name": "switch",
            "site1_idx": site1_idx,
            "site2_idx": site2_idx,
            "site1_ads": site1_ads,
            "site2_ads": site2_ads,
        }
        return action
