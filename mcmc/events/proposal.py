"""Module for proposing changes to the surface system in the Monte Carlo simulation."""

import logging
import random
from collections.abc import Iterable
from copy import deepcopy
from typing import Any

import numpy as np

from mcmc.slab import get_complementary_idx
from mcmc.system import SurfaceSystem


class Proposal:
    """Base class for proposal moves in the Monte Carlo simulation."""

    def __init__(
        self,
        system: SurfaceSystem,
        adsorbate_list: Iterable[str],
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the base Proposal.

        Args:
            system (SurfaceSystem): The surface system to propose changes to.
            adsorbate_list (Iterable[str]): The list of adsorbates that can be proposed.
            logger (logging.Logger): The logger object.

        Attributes:
            system (SurfaceSystem): The surface system to propose changes to.
            adsorbate_list (List[str]): The list of adsorbates that can be proposed.
            logger (logging.Logger): The logger object.
        """
        self.system = system
        self.adsorbate_list = list(adsorbate_list).copy()
        self.adsorbate_list.append("None")
        self.logger = logger or logging.getLogger(__name__)

    def get_action(self) -> dict[str, Any]:
        """Obtain a dictionary containing the proposed change.

        Raises:
            NotImplementedError: This method should be implemented in the derived classes.
        """
        raise NotImplementedError


class ChangeProposal(Proposal):
    """Proposal to change single adsorbate at a site."""

    def __init__(
        self,
        system: SurfaceSystem,
        adsorbate_list: Iterable[str] = ("Sr", "O"),
        site_idx: int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the ChangeProposal.

        Args:
            system (SurfaceSystem): The surface system to propose changes to.
            adsorbate_list (Iterable[str]): The list of adsorbates that can be proposed.
            site_idx (int, optional): The index of the site to propose the change. Defaults to None.
            logger (logging.Logger, optional): The logger object. Defaults to None.

        Attributes:
            site_idx (int): The index of the site to propose the change.
        """
        super().__init__(system, adsorbate_list, logger=logger)
        self.site_idx = site_idx

    def get_action(self) -> dict[str, Any]:
        """Get an index and the elemental identity of the adsorbate to change into.

        Returns:
            Dict[str, Any]: A dictionary containing the index of the site and the elemental
                identity of the adsorbate to change into.
        """
        ads_choices = deepcopy(self.adsorbate_list)
        if isinstance(self.site_idx, int):
            site_idx = self.site_idx
        else:
            site_idx = np.random.choice(range(len(self.system.occ)))

        self.logger.debug("before proposed state is")
        self.logger.debug(self.system.occ)

        if self.system.occ[site_idx] != 0:
            # not an empty virtual site, remove the adsorbate
            start_ads = self.system.real_atoms[self.system.occ[site_idx]].symbol
            ads_choices.remove(start_ads)
        else:
            start_ads = "None"
            ads_choices.remove("None")

        end_ads = random.choice(ads_choices)

        return {
            "name": "change",
            "site_idx": site_idx,
            "start_ads": start_ads,
            "end_ads": end_ads,
        }


class SwitchProposal(Proposal):
    """Proposal to switch two adsorbates at different sites."""

    def __init__(
        self,
        system: SurfaceSystem,
        adsorbate_list: Iterable[str] = ("Sr", "O"),
        require_per_atom_energies: bool = False,
        require_distance_decay: bool = False,
        temp: float | None = None,
        run_folder: str | None = None,
        plot_specific_distance_weights: bool = False,
        run_iter: int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the SwitchProposal.

        Args:
            system (SurfaceSystem): The surface system to propose changes to.
            adsorbate_list (Iterable[str]): The list of adsorbates that can be proposed.
            require_per_atom_energies (bool): Whether to require per atom energies.
            require_distance_decay (bool): Whether to require distance decay.
            temp (float, optional): The temperature. Defaults to None.
            run_folder (str, optional): The folder to run the calculations in. Defaults to None.
            plot_specific_distance_weights (bool, optional): Whether to plot the specific distance
                weights. Defaults to False.
            run_iter (int, optional): The iteration number. Defaults to None.
            logger (logging.Logger, optional): The logger object. Defaults to None.

        Attributes:
            require_per_atom_energies (bool): Whether to require per atom energies.
            require_distance_decay (bool): Whether to require distance decay.
            temp (float): The temperature.
            run_folder (str): The folder to run the calculations in.
            plot_weights (bool): Whether to plot the specific distance weights.
            run_iter (int): The iteration number.
        """
        super().__init__(system, adsorbate_list, logger=logger)
        self.require_per_atom_energies = require_per_atom_energies
        self.require_distance_decay = require_distance_decay
        self.temp = temp
        self.run_folder = run_folder
        self.plot_weights = plot_specific_distance_weights
        self.run_iter = run_iter

    def get_action(self) -> dict[str, Any]:
        """Get two indices, site1 and site2 of different elemental identities.

        Returns:
            dict[str, Any]: A dictionary containing the indices of the two sites and the elemental
                identities of the adsorbates.
        """
        self.logger.debug("before proposed state is")
        self.logger.debug(self.system.occ)

        site1_idx, site2_idx, site1_ads, site2_ads = get_complementary_idx(
            self.system,
            require_per_atom_energies=self.require_per_atom_energies,
            require_distance_decay=self.require_distance_decay,
            temperature=self.temp,
            run_folder=self.run_folder,
            plot_weights=self.plot_weights,
            run_iter=self.run_iter,
        )

        site1_coords = self.system.ads_coords[site1_idx]
        site2_coords = self.system.ads_coords[site2_idx]

        self.logger.debug("\n we are at iter %s", self.run_iter)
        self.logger.debug("idx is %s at %s", site1_idx, site1_coords)
        self.logger.debug("idx is %s at %s", site2_idx, site2_coords)

        return {
            "name": "switch",
            "site1_idx": site1_idx,
            "site2_idx": site2_idx,
            "site1_ads": site1_ads,
            "site2_ads": site2_ads,
        }
