"""Test slab.py module."""

import logging

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator

from mcmc.slab import (
    change_site,
    compute_boltzmann_weights,
    get_adsorbate_indices,
    get_complementary_idx,
    get_complementary_idx_distance_decay,
)
from mcmc.system import SurfaceSystem


@pytest.fixture()
def system():
    """Create a dummy SurfaceSystem object for testing."""
    atoms = Atoms(
        "GaAsGaAs", positions=[[0, 0, 0], [0, 0, 3], [1, 1, 1], [1, 1, 4]]
    )  # fake positions for now
    ads_coords = [(0, 0, 3), (1, 1, 1), (2, 2, 5)]
    occ = [1, 2, 0]
    distance_weight_matrix = np.random.rand(3, 3)
    return SurfaceSystem(
        atoms, ads_coords=ads_coords, occ=occ, distance_weight_matrix=distance_weight_matrix
    )


@pytest.fixture()
def logger():
    """Create a dummy logger object for testing."""
    return logging.getLogger("test")


def test_change_site_with_existing_adsorbate(system):
    """Test changing an existing adsorbate to a new one."""
    # Change the adsorbate to a new one
    new_surface = change_site(system, 0, "O")

    # Check that the adsorbate has been changed
    assert len(new_surface.real_atoms) == 4
    assert new_surface.occ[0] == 3
    assert new_surface.real_atoms[3].symbol == "O"


def test_change_site_with_empty_site(system):
    """Test changing an empty site to an adsorbate."""
    # Change the adsorbate to a new one
    new_surface = change_site(system, 2, "Ir")

    # Check that the adsorbate has been added
    assert len(new_surface.real_atoms) == 5
    assert new_surface.occ[2] == 4
    assert new_surface.real_atoms[4].symbol == "Ir"


def test_change_site_with_desorption(system):
    """Test desorbing an adsorbate."""
    # Change the adsorbate to None (desorption)
    new_surface = change_site(system, 0, "None")

    # Check that the adsorbate has been removed
    assert len(new_surface.real_atoms) == 3
    assert new_surface.occ[0] == 0


def test_change_site_with_invalid_site_index(system):
    """Test changing an adsorbate at an invalid site index."""
    # Try to change an adsorbate at an invalid site index
    with pytest.raises(IndexError):
        change_site(system, 5, "As")


def test_get_adsorbate_indices(system):
    """Test getting the indices of each adsorbate type."""
    adsorbates = get_adsorbate_indices(system)
    assert adsorbates == {"As": [0], "Ga": [1], "None": [2]}


def test_compute_boltzmann_weights(system):
    """Test computing the Boltzmann weights for each adsorbate type for per-atom energies."""
    # Set up the test case
    per_atom_energies = [1.0, 0.5, 1.0, 0.6]
    calc = Calculator()
    calc.results = {"per_atom_energies": per_atom_energies}
    system.calc = calc
    temperature = 1.0
    curr_ads = get_adsorbate_indices(system)

    weights = compute_boltzmann_weights(system, temperature, curr_ads)
    # Perform assertions
    assert len(weights) == 3
    assert all(val > 0 for val in weights.values())
    assert np.allclose(
        sorted(weights.values()),
        sorted(
            {
                "As": [0.1850956],
                "Ga": [0.30517106],
                "None": [1],
            }.values()
        ),
    )


def test_get_complementary_idx_distance_decay(system):
    """Test getting the indices of complementary adsorbates with distance decay weights."""
    # Set up the test case
    curr_ads = get_adsorbate_indices(system)
    type1 = "As"
    type2 = "Ga"
    weights1 = np.array([0.1850956])
    weights2 = np.array([0.30517106])
    plot_weights = False
    run_folder = "."
    run_iter = 0

    # Call the function
    site1_idx, site2_idx = get_complementary_idx_distance_decay(
        system,
        curr_ads,
        type1,
        type2,
        weights1,
        weights2,
        plot_weights,
        run_folder,
        run_iter,
    )

    # Perform assertions
    assert site1_idx != site2_idx
    if system.occ[site1_idx] != 0:
        assert type1 in system.real_atoms[system.occ[site1_idx]].symbol
    else:
        assert type1 == "None"
    if system.occ[site2_idx] != 0:
        assert type2 in system.real_atoms[system.occ[site2_idx]].symbol
    else:
        assert type2 == "None"


def test_get_complementary_idx(system):
    """Test getting the indices of complementary adsorbates without distance decay weights or
    per-atom energies.
    """
    # Set up the test case
    require_per_atom_energies = False
    require_distance_decay = False
    temperature = 1.0
    plot_weights = False
    run_folder = "."
    run_iter = 0

    # Call the function
    site1_idx, site2_idx, type1, type2 = get_complementary_idx(
        system,
        require_per_atom_energies,
        require_distance_decay,
        temperature,
        plot_weights,
        run_folder,
        run_iter,
    )

    # Perform assertions
    assert site1_idx != site2_idx
    if system.occ[site1_idx] != 0:
        assert type1 in system.real_atoms[system.occ[site1_idx]].symbol
    else:
        assert type1 == "None"
    if system.occ[site2_idx] != 0:
        assert type2 in system.real_atoms[system.occ[site2_idx]].symbol
    else:
        assert type2 == "None"
