import numpy as np
import pytest

from mcmc.events.criterion import TestingCriterion
from mcmc.events.event import Change
from mcmc.events.proposal import ChangeProposal
from tests.events.test_fixtures import system


@pytest.fixture
def proposal(system):
    # Create a dummy Proposal object for testing
    adsorbate_list = ("Ga", "As")
    return ChangeProposal(system, adsorbate_list)


@pytest.fixture
def criterion():
    # Create a dummy AcceptanceCriterion object for testing
    return TestingCriterion()


def test_change_forward(system, proposal, criterion):
    # Create a Change event object
    event = Change(system, proposal, criterion)

    # Perform the forward step of the event
    event.forward()

    # Assert that the system state has been changed
    assert ~np.allclose(event.system.occ, system.occ)


def test_change_acceptance(system, proposal, criterion):
    # Create a Change event object
    event = Change(system, proposal, criterion)

    # Perform the acceptance step of the event
    accept, new_system = event.acceptance()

    # Assert that the acceptance result is a boolean
    assert isinstance(accept, bool)

    # Assert that the system state has been changed if the change is accepted
    if accept:
        assert ~np.allclose(event.system.occ, system.occ)
    else:
        assert ~np.allclose(event.system.occ, system.occ)
