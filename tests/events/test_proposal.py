import random

import numpy as np
import pytest

from mcmc.events.proposal import ChangeProposal
from tests.events.test_fixtures import system

SEED = 11
# Set the seed for the random module
random.seed(SEED)

# Set the seed for np.random.choice
np.random.seed(SEED)


def test_change_proposal_get_action(system):
    # Create a ChangeProposal object
    adsorbate_list = ("Ga", "As")
    proposal = ChangeProposal(system, adsorbate_list)

    # Get the action dictionary
    action = proposal.get_action()

    # Assert that the action dictionary contains the required keys
    assert "name" in action
    assert "site_idx" in action
    assert "start_ads" in action
    assert "end_ads" in action

    # Assert that the start_ads and end_ads values are valid adsorbates
    assert action["start_ads"] in proposal.adsorbate_list
    assert action["end_ads"] in proposal.adsorbate_list

    # Assert that the site_idx value is a valid index
    assert action["site_idx"] >= 0
    assert action["site_idx"] < len(system.occ)


def test_adsorb_at_empty_site(system):
    # Create a ChangeProposal object
    adsorbate_list = ("Ga", "As")
    proposal = ChangeProposal(system, adsorbate_list)

    # Set the occupancy of the system to all empty sites
    system.occ = np.zeros(len(system.occ))

    # Get the action dictionary
    action = proposal.get_action()

    # Assert that the start_ads value is "None"
    assert action["start_ads"] == "None"

    # Assert that the end_ads value is a valid adsorbate
    assert action["end_ads"] in set(proposal.adsorbate_list) - set("None")


def test_change_at_adsorbed_site(system):
    # Create a ChangeProposal object
    adsorbate_list = ("Ga", "As")
    proposal = ChangeProposal(system, adsorbate_list)

    # system.occ should be [1, 3]

    # Get the action dictionary
    action = proposal.get_action()

    # Assert that the start_ads value is a valid adsorbate
    assert action["start_ads"] in set(proposal.adsorbate_list) - set("None")

    # Assert that the end_ads value is a valid adsorbate
    assert action["end_ads"] in set(proposal.adsorbate_list)