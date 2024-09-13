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


@pytest.fixture(scope="module")
def system():
    """Create a dummy SurfaceSystem object for testing."""
    atoms = Atoms(
        "GaAsGaAs", positions=[[0, 0, 0], [0, 0, 3], [1, 1, 1], [1, 1, 4]]
    )  # fake positions for now
    atoms.set_array("ads_group", np.array([0, 1, 2, 0]))
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


def test_change_site_with_existing_adsorbate_add_group(system):
    """Test changing an existing adsorbate to a new one."""
    # Change the adsorbate to a new one
    new_surface = change_site(system, 0, "HO")

    # Check that the adsorbate has been changed
    assert len(new_surface.real_atoms) == 5
    assert np.allclose(new_surface.occ, [3, 1, 0])
    assert new_surface.real_atoms[3].symbol == "O"
    assert new_surface.real_atoms[4].symbol == "H"
    assert np.allclose(new_surface.real_atoms.get_array("ads_group"), [0, 1, 0, 3, 3])


def test_change_site_with_empty_site(system):
    """Test changing an empty site to an adsorbate."""
    # Change the adsorbate to a new one
    new_surface = change_site(system, 2, "Ir")

    # Check that the adsorbate has been added
    assert len(new_surface.real_atoms) == 6
    assert np.allclose(new_surface.occ, [3, 1, 5])
    assert new_surface.real_atoms[5].symbol == "Ir"
    assert np.allclose(new_surface.real_atoms.get_array("ads_group"), [0, 1, 0, 3, 3, 5])


def test_change_site_remove_group(system):
    """Test changing an existing adsorbate to a new one."""
    # Change the adsorbate to a new one
    new_surface = change_site(system, 0, "None")

    # Check that the adsorbate has been changed
    assert len(new_surface.real_atoms) == 4
    assert np.allclose(new_surface.occ, [0, 1, 3])
    assert new_surface.real_atoms[3].symbol == "Ir"
    assert np.allclose(new_surface.real_atoms.get_array("ads_group"), [0, 1, 0, 3])


def test_change_site_remove_single(system):
    """Test changing an existing adsorbate to a new one."""
    # Change the adsorbate to a new one
    new_surface = change_site(system, 1, "None")

    # Check that the adsorbate has been changed
    assert len(new_surface.real_atoms) == 3
    assert np.allclose(new_surface.occ, [0, 0, 2])
    assert new_surface.real_atoms[2].symbol == "Ir"
    assert np.allclose(new_surface.real_atoms.get_array("ads_group"), [0, 0, 2])
