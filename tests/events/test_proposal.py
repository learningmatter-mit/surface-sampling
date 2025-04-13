"""Unit tests for the proposal module."""

import random

import numpy as np

from mcmc.events.proposal import ChangeProposal, SwitchProposal

SEED = 11
# Set the seed for the random module
random.seed(SEED)

# Set the seed for np.random.choice
np.random.seed(SEED)


def test_change_proposal_get_action(system):
    """Test the get_action method of the ChangeProposal class."""
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
    """Test the get_action method of the ChangeProposal class at empty sites."""
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
    """Test the get_action method of the ChangeProposal class at adsorbed sites."""
    # Create a ChangeProposal object
    adsorbate_list = ("Ga", "As")
    proposal = ChangeProposal(system, adsorbate_list)

    # system.occ should be [1, 3]

    # Get the action dictionary
    action = proposal.get_action()

    # Assert that the start_ads value is a valid adsorbate
    assert str(action["start_ads"]) in set(proposal.adsorbate_list) - set("None")

    # Assert that the end_ads value is a valid adsorbate
    assert action["end_ads"] in set(proposal.adsorbate_list)


def test_switch_at_adsorbed_sites_simple(system):
    """Test the get_action method of the SwitchProposal class."""
    # None of the distance-based or energy-based proposal

    # Create a SwitchProposal object
    adsorbate_list = ("Ga", "As")
    proposal = SwitchProposal(system, adsorbate_list)

    # system.occ should be [1, 3]

    # Get the action dictionary
    action = proposal.get_action()

    # Assert that the site1_ads and site2_ads values are valid adsorbates
    assert action["site1_ads"] in set(proposal.adsorbate_list)
    assert action["site2_ads"] in set(proposal.adsorbate_list)

    # Assert that the site1_ads and site2_ads values are different
    assert action["site1_ads"] != action["site2_ads"]

    # Assert that the site_idx values are valid indices
    assert action["site1_idx"] >= 0
    assert action["site1_idx"] < len(system.occ)
    assert action["site2_idx"] >= 0
    assert action["site2_idx"] < len(system.occ)

    # Assert that the site_idx values are different
    assert action["site1_idx"] != action["site2_idx"]
