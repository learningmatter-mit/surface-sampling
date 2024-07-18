"""Test the filter_distances function in mcmc.utils.misc."""

import os
import pickle as pkl

import ase
import pytest
from ase.io import read

from mcmc.utils.misc import filter_distances

current_dir = os.path.dirname(__file__)

# get absolute adsorption coords
element = "O"

# adsorption coords
ase_bridge = [1.96777, 1.99250, 18.59954]
ase_top1 = [5.90331, 0.14832, 19.49200]
ase_top2 = [1.96777, 4.13332, 19.49200]


# test_fixtures
@pytest.fixture()
def pristine_slab() -> ase.Atoms:
    """The pristine slab.

    Returns:
        ase.Atoms: The pristine slab.

    Raises:
        FileNotFoundError: If the slab file is not found.
    """
    with open(os.path.join(current_dir, "data/SrTiO3_001/SrTiO3_unit_cell.pkl"), "rb") as slab_file:
        unit_slab = pkl.load(slab_file)
        return unit_slab * (2, 2, 1)
    raise FileNotFoundError("Slab file not found.")


def test_one_O_fail(pristine_slab):
    """Test the filter_distances function with one O atom adsorbed at a bridge site."""
    test_slab = pristine_slab.copy()  # starting with a pristine slab

    # adsorb at one site
    test_slab.append(element)
    test_slab.positions[-1] = ase_bridge

    assert not filter_distances(test_slab, ads=[element], cutoff_distance=1.5)


def test_one_O_pass(pristine_slab):
    """Test the filter_distances function with one O atom adsorbed at a top site."""
    test_slab = pristine_slab.copy()  # starting with a pristine slab

    # adsorb at one site
    test_slab.append(element)
    test_slab.positions[-1] = ase_top1

    assert filter_distances(test_slab, ads=[element], cutoff_distance=1.5)


def test_two_O_fail(pristine_slab):
    """Test the filter_distances function with two O atoms adsorbed at a bridge and top site."""
    test_slab = pristine_slab.copy()  # starting with a pristine slab

    # adsorb at two sites
    test_slab.append(element)
    test_slab.positions[-1] = ase_bridge
    test_slab.append(element)
    test_slab.positions[-1] = ase_top1

    assert not filter_distances(test_slab, ads=[element], cutoff_distance=1.5)


def test_two_O_pass(pristine_slab):
    """Test the filter_distances function with two O atoms adsorbed at two top sites."""
    test_slab = pristine_slab.copy()  # starting with a pristine slab

    # adsorb at two sites
    test_slab.append(element)
    test_slab.positions[-1] = ase_top1
    test_slab.append(element)
    test_slab.positions[-1] = ase_top2

    assert filter_distances(test_slab, ads=[element], cutoff_distance=1.5)


def test_cell_distance_failed():
    """Test the filter_distances function with a slab that is too close to the cell."""
    test_slab = read(
        os.path.join(
            current_dir,
            "data/SrTiO3_001/SrTiO3_001_distance_failed.cif",
        )
    )

    assert not filter_distances(test_slab, ads=[element], cutoff_distance=1.5)
