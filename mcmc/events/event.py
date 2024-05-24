import logging

from mcmc.events.criterion import AcceptanceCriterion as Criterion
from mcmc.events.proposal import Proposal
from mcmc.slab import change_site
from mcmc.system import SurfaceSystem

logger = logging.getLogger(__name__)


class Event:
    """Base class for Monte Carlo events.

    Args:
        system (SurfaceSystem): The surface system to propose changes to.
        proposal (Proposal): The proposal object to generate the change.
        criterion (Criterion): The criterion object to determine acceptance or rejection.
    """

    def __init__(
        self, system: SurfaceSystem, proposal: Proposal, criterion: Criterion, **kwargs
    ) -> None:
        self.system = system
        self.proposal = proposal
        self.criterion = criterion
        self.kwargs = kwargs


class Change(Event):
    """Semigrand Canonical Monte Carlo event for changing the adsorbate at one site.

    Args:
        system (SurfaceSystem): The surface system to propose changes to.
        proposal (Proposal): The proposal object to generate the change.
        criterion (Criterion): The criterion object to determine acceptance or rejection.
    """

    def __init__(
        self, system: SurfaceSystem, proposal: Proposal, criterion: Criterion, **kwargs
    ) -> None:
        super().__init__(system, proposal, criterion, **kwargs)
        self.action = self.proposal.get_action()
        self.site_idx = self.action["site_idx"]
        self.start_ads = self.action["start_ads"]
        self.end_ads = self.action["end_ads"]

    def forward(self) -> None:
        """Perform the forward step of the event and saves the state before and after the change."""
        self.system.save_state("before")
        self.system = change_site(
            self.system,
            self.site_idx,
            self.end_ads,
        )
        self.system.save_state("after")
        logger.debug("after proposed state is")
        logger.debug(self.system.occ)
        # TODO it's a bit slower now, add a backward method

    def acceptance(self, **kwargs) -> tuple[bool, SurfaceSystem]:
        """Perform the acceptance step of the event and determine whether the change is accepted or rejected.
        If rejected, the system state is restored to the state before the change.

        Returns:
            bool: Whether the change is accepted or rejected.
            SurfaceSystem: The surface system after the change.
        """
        self.forward()
        accept = self.criterion(self.system, **kwargs)
        if not accept:
            self.system.restore_state("before")
            logger.debug("state not changed!")
        else:
            logger.debug("state changed!")

        return accept, self.system


class CanonicalEvent(Event):
    # distance-based decay
    # random
    def __init__(
        self, system: SurfaceSystem, proposal: Proposal, criterion: Criterion, **kwargs
    ) -> None:
        super().__init__(system, proposal, criterion, **kwargs)
