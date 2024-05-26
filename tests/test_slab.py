import pytest
from ase import Atoms

from mcmc.slab import change_site
from mcmc.system import SurfaceSystem


@pytest.fixture
def system():
    # Create a dummy SurfaceSystem object for testing
    atoms = Atoms(
        "GaAsGaAs", positions=[[0, 0, 0], [0, 0, 3], [1, 1, 1], [1, 1, 4]]
    )  # fake positions for now
    ads_coords = [(0, 0, 3), (1, 1, 4), (2, 2, 5)]
    occ = [1, 3, 0]
    return SurfaceSystem(atoms, ads_coords, occ=occ)


def test_change_site_with_existing_adsorbate(system):
    # Change the adsorbate to a new one
    new_surface = change_site(system, 0, "O")

    # Check that the adsorbate has been changed
    assert len(new_surface.real_atoms) == 4
    assert new_surface.occ[0] == 3
    assert new_surface.real_atoms[3].symbol == "O"


def test_change_site_with_empty_site(system):
    # Change the adsorbate to a new one
    new_surface = change_site(system, 2, "Ir")

    # Check that the adsorbate has been added
    assert len(new_surface.real_atoms) == 5
    assert new_surface.occ[2] == 4
    assert new_surface.real_atoms[4].symbol == "Ir"


def test_change_site_with_desorption(system):
    # Change the adsorbate to None (desorption)
    new_surface = change_site(system, 0, "None")

    # Check that the adsorbate has been removed
    assert len(new_surface.real_atoms) == 3
    assert new_surface.occ[0] == 0


def test_change_site_with_invalid_site_index(system):
    # Try to change an adsorbate at an invalid site index
    with pytest.raises(IndexError):
        change_site(system, 5, "As")
