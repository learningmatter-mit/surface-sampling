import logging
from typing import Iterable

import numpy as np

from mcmc.system import SurfaceSystem
from mcmc.utils.misc import filter_distances

logger = logging.getLogger(__name__)


class AcceptanceCriterion:
    """Base class for acceptance criteria."""

    def __init__(self):
        pass

    def criterion(self, system: SurfaceSystem, **kwargs):
        """Check if the criterion is met.

        Args:
            system (SurfaceSystem): The surface system to check.
            **kwargs: Additional keyword arguments.

        Raises:
            NotImplementedError: This method should be implemented in the derived classes.
        """
        raise NotImplementedError

    def __call__(self, system: SurfaceSystem, **kwargs):
        return self.criterion(system, **kwargs)


class TestingCriterion(AcceptanceCriterion):
    """Testing acceptance criterion for debugging purposes."""

    def __init__(self, always_accept=True) -> None:
        self.always_accept = always_accept

    def criterion(self, *args, **kwargs) -> bool:
        """Always returns one value for testing purposes.

        Returns:
            bool: The value to return.
        """
        logger.debug("Testing criterion called")
        return self.always_accept


class DistanceCriterion(AcceptanceCriterion):
    """Acceptance criterion based on distances between atoms. Ensures that atoms are not too close to each other."""

    def __init__(self, filter_distance: float = 1.5):
        self.filter_distance = filter_distance

    def criterion(
        self,
        system: SurfaceSystem,
        adsorbate_types: Iterable[str] = ("Sr", "Ti"),
        **kwargs,
    ) -> bool:
        """Check if any atoms are too close to each other.

        Args:
            system (SurfaceSystem): The surface system to check.
            adsorbate_types (List[str]): The list of adsorbate types to check.

        Returns:
            bool: Whether the criterion is met.
        """
        if filter_distances(
            system.real_atoms,
            ads=adsorbate_types,
            cutoff_distance=self.filter_distance,
        ):
            logger.debug("state changed with filtering!")
            return True
        return False


class MetropolisCriterion(AcceptanceCriterion):
    """Metropolis acceptance criterion for Monte Carlo simulations.

    Args:
        temperature (float): The temperature of the system in kB*T units.
    """

    def __init__(self, temperature: float):
        super().__init__()
        self.temp = temperature

    def criterion(self, system: SurfaceSystem, **kwargs) -> bool:
        """Metropolis acceptance criterion for Monte Carlo simulations.

        Args:
            system (SurfaceSystem): The surface system to check.
            **kwargs: Additional keyword arguments.

        Returns:
            bool: Whether the criterion is met.
        """
        system.restore_state("before")
        try:
            prev_energy = system.results["surface_energy"]
        except KeyError:
            logger.error("No surface energy found in results dict")
            prev_energy = system.get_surface_energy(recalculate=True)
        logger.debug("prev energy is %s", prev_energy)

        system.restore_state("after")
        curr_energy = system.get_surface_energy(recalculate=True)
        # system.save_state("after") # update energy in results dict
        logger.debug("curr energy is %s", curr_energy)

        energy_diff = float(curr_energy - prev_energy)
        logger.debug("energy diff is %s", energy_diff)

        logger.debug("k_b T = %s", self.temp)
        # pot should be accounted for in the energy_diff
        try:
            base_prob = np.exp(-energy_diff / self.temp)
        except OverflowError:
            base_prob = 0.0

        logger.debug("base probability is %s", base_prob)
        return np.random.rand() < base_prob
