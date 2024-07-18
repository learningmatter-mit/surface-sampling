"""Classes for acceptance criteria in Monte Carlo simulations."""

import logging
from collections.abc import Iterable

import numpy as np

from mcmc.system import SurfaceSystem
from mcmc.utils.misc import filter_distances


class AcceptanceCriterion:
    """Base class for acceptance criteria."""

    def __init__(self, logger: logging.Logger | None = None):
        """Initialize the acceptance criterion.

        Args:
            logger (logging.Logger, optional): Logger object. Defaults to None.

        Attributes:
            logger (logging.Logger): The logger object.
        """
        self.logger = logger or logging.getLogger(__name__)

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
        """Call the criterion method.

        Args:
            system (SurfaceSystem): The surface system to check.
            **kwargs: Additional keyword arguments.
        """
        return self.criterion(system, **kwargs)


class TestingCriterion(AcceptanceCriterion):
    """Testing acceptance criterion for debugging purposes."""

    def __init__(self, logger: logging.Logger | None = None, always_accept=True) -> None:
        """Initialize the testing criterion.

        Args:
            logger (logging.Logger, optional): Logger object. Defaults to None.
            always_accept (bool, optional): Whether to always accept the change. Defaults to True.

        Attributes:
            always_accept (bool): Whether to always accept the change.
        """
        super().__init__(logger)
        self.always_accept = always_accept

    def criterion(self, *args, **kwargs) -> bool:
        """Always returns one value for testing purposes.

        Returns:
            bool: The value to return.
        """
        self.logger.debug("Testing criterion called")
        return self.always_accept


class DistanceCriterion(AcceptanceCriterion):
    """Acceptance criterion based on distances between atoms. Ensures that atoms are not too close
    to each other.
    """

    def __init__(self, logger: logging.Logger | None = None, filter_distance: float = 1.5):
        """Initialize the distance criterion.

        Args:
            logger (logging.Logger, optional): Logger object. Defaults to None.
            filter_distance (float, optional): The minimum distance between atoms. Defaults to 1.5.

        Attributes:
            filter_distance (float): The minimum distance between atoms.
        """
        super().__init__(logger)
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
            **kwargs: Additional keyword arguments.

        Returns:
            bool: Whether the criterion is met.
        """
        if filter_distances(
            system.real_atoms,
            ads=adsorbate_types,
            cutoff_distance=self.filter_distance,
        ):
            self.logger.debug("state changed with filtering!")
            return True
        return False


class MetropolisCriterion(AcceptanceCriterion):
    """Metropolis acceptance criterion for Monte Carlo simulations."""

    def __init__(self, temperature: float, logger: logging.Logger | None = None):
        """Initialize the Metropolis criterion.

        Args:
            temperature (float): The temperature of the system in kB*T units.
            logger (logging.Logger, optional): Logger object. Defaults to None.

        Attributes:
            temp (float): The temperature of the system in kB*T units.
        """
        super().__init__(logger)
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
            self.logger.error("No surface energy found in results dict")
            prev_energy = system.get_surface_energy(recalculate=True)
        self.logger.debug("prev energy is %s", prev_energy)

        system.restore_state("after")
        curr_energy = system.get_surface_energy(recalculate=True)
        # system.save_state("after") # update energy in results dict
        self.logger.debug("curr energy is %s", curr_energy)

        energy_diff = float(curr_energy - prev_energy)
        self.logger.debug("energy diff is %s", energy_diff)

        self.logger.debug("k_b T = %s", self.temp)
        # pot should be accounted for in the energy_diff
        try:
            base_prob = np.exp(-energy_diff / self.temp)
        except OverflowError:
            base_prob = 0.0

        self.logger.debug("base probability is %s", base_prob)
        return np.random.rand() < base_prob
