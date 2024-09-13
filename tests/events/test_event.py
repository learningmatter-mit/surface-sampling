import numpy as np
import pytest

from mcmc.events.criterion import TestingCriterion
from mcmc.events.event import Change, Exchange
from mcmc.events.proposal import ChangeProposal, SwitchProposal
from tests.events.test_fixtures import system


@pytest.fixture
def change_proposal(system):
    # Create a dummy Proposal object for testing
    adsorbate_list = ("Ga", "As")
    return ChangeProposal(system, adsorbate_list)


@pytest.fixture
def switch_proposal(system):
    # Create a dummy Proposal object for testing
    adsorbate_list = ("Ga", "As")
    return SwitchProposal(system, adsorbate_list)


@pytest.fixture
def criterion():
    # Create a dummy AcceptanceCriterion object for testing
    return TestingCriterion(always_accept=True)


def test_change_forward(system, change_proposal, criterion):
    # Create a Change event object
    event = Change(system, change_proposal, criterion)
    # Perform the forward step of the event
    event.forward()

    # Assert that the system state has been changed
    assert ~np.allclose(event.system.occ, system.occ)


def test_change_backward(system, change_proposal, criterion):
    start_ads_pos = system.real_atoms.get_positions()[system.occ]
    start_chem_symbols = system.real_atoms.get_chemical_formula()
    start_occ = system.occ

    # Create a Change event object
    event = Change(system, change_proposal, criterion)

    # Perform the forward step of the event
    event.forward()

    # Perform the backward step of the event
    event.backward()
    curr_ads_pos = system.real_atoms.get_positions()[system.occ]
    curr_chem_symbols = system.real_atoms.get_chemical_formula()
    curr_occ = system.occ

    # Assert that the ads positions are the same before and after the event
    assert np.allclose(start_ads_pos, curr_ads_pos)

    # Assert that the chemical symbols are the same before and after the event
    assert start_chem_symbols == curr_chem_symbols

    # Assert that the occupation numbers are different before and after the event
    assert ~np.allclose(start_occ, curr_occ)


def test_change_acceptance(system, change_proposal, criterion):
    # Create a Change event object
    event = Change(system, change_proposal, criterion)

    # Perform the acceptance step of the event
    accept, new_system = event.acceptance()

    # Assert that the acceptance result is a boolean
    assert isinstance(accept, bool)

    # Assert that the system state has been changed if the change is accepted
    if accept:
        assert ~np.allclose(event.system.occ, system.occ)
    else:
        assert np.allclose(event.system.occ, system.occ)


def test_exchange_forward(system, switch_proposal, criterion):
    # Create a Change event object
    event = Exchange(system, switch_proposal, criterion)
    # Perform the forward step of the event
    event.forward()

    # Assert that the system state has been changed
    assert ~np.allclose(event.system.occ, system.occ)


def test_exchange_backward(system, switch_proposal, criterion):
    start_ads_pos = system.real_atoms.get_positions()[system.occ]
    start_chem_symbols = system.real_atoms.get_chemical_formula()
    start_occ = system.occ

    # Create a Change event object
    event = Exchange(system, switch_proposal, criterion)

    # Perform the forward step of the event
    event.forward()
    intermediate_ads_pos = system.real_atoms.get_positions()[system.occ]
    intermediate_chem_symbols = system.real_atoms.get_chemical_formula()
    intermediate_occ = system.occ
    # Assert that the ads positions are different before and during the event
    assert ~np.allclose(start_ads_pos, intermediate_ads_pos)

    # Assert that the chemical symbols are the same before and during the event
    assert start_chem_symbols == intermediate_chem_symbols

    # Assert that the occupation numbers are different before and during the event
    assert ~np.allclose(start_occ, intermediate_occ)

    # Perform the backward step of the event
    event.backward()
    final_ads_pos = system.real_atoms.get_positions()[system.occ]
    final_chem_symbols = system.real_atoms.get_chemical_formula()
    final_occ = system.occ

    # Assert that the ads positions are the same before and after the event
    assert np.allclose(start_ads_pos, final_ads_pos)

    # Assert that the chemical symbols are the same before and after the event
    assert start_chem_symbols == final_chem_symbols

    # Assert that the occupation numbers are different before and after the event
    assert ~np.allclose(start_occ, final_occ)


def test_exchange_acceptance(system, switch_proposal, criterion):
    # Create a Change event object
    event = Exchange(system, switch_proposal, criterion)

    # Perform the acceptance step of the event
    accept, new_system = event.acceptance()

    # Assert that the acceptance result is a boolean
    assert isinstance(accept, bool)

    # Assert that the system state has been changed if the change is accepted
    if accept:
        assert ~np.allclose(event.system.occ, system.occ)
    else:
        assert np.allclose(event.system.occ, system.occ)
