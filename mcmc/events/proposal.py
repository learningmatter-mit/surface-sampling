import logging
import random
from copy import deepcopy
from typing import Any, Dict, List

import numpy as np

from mcmc.system import SurfaceSystem

logger = logging.getLogger(__name__)


class Proposal:
    """Base class for proposal moves in the Monte Carlo simulation.

    Args:
        system (SurfaceSystem): The surface system to propose changes to.
        adsorbate_list (List[str]): The list of adsorbates that can be proposed.
    """

    def __init__(self, system: SurfaceSystem, adsorbate_list: List[str]) -> None:
        self.system = system
        self.adsorbate_list = adsorbate_list.copy()

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
        adsorbate_list (List[str]): The list of adsorbates that can be proposed.
    """

    def __init__(
        self, system: SurfaceSystem, adsorbate_list: List[str] = ["Sr", "O"]
    ) -> None:
        super().__init__(system, adsorbate_list)
        self.adsorbate_list.append("None")

    def get_action(self) -> Dict[str, Any]:
        """Get two indices, site1 and site2 of different elemental identities.

        Returns:
            Dict[str, Any]: A dictionary containing the indices of the two sites and the elemental identities of the adsorbates.
        """
        ads_choices = deepcopy(self.adsorbate_list)
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
