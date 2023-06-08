import os
import pickle as pkl

import pytest
from ase.io import read

from mcmc.utils import filter_distances

current_dir = os.path.dirname(__file__)

# get absolute adsorption coords
element = "O"

# adsorption coords
ase_bridge = [1.96777, 1.99250, 18.59954]
ase_top1 = [5.90331, 0.14832, 19.49200]
ase_top2 = [1.96777, 4.13332, 19.49200]


# test_fixtures
@pytest.fixture
def pristine_slab():
    slab_file = open(os.path.join(current_dir, "resources/SrTiO3_unit_cell.pkl"), "rb")
    unit_slab = pkl.load(slab_file)
    pristine_slab = unit_slab * (2, 2, 1)
    return pristine_slab


def test_one_O_fail(pristine_slab):
    test_slab = pristine_slab.copy()  # starting with a pristine slab

    # adsorb at one site
    test_slab.append(element)
    test_slab.positions[-1] = ase_bridge

    assert filter_distances(test_slab, ads=[element], cutoff_distance=1.5) == False


def test_one_O_pass(pristine_slab):
    test_slab = pristine_slab.copy()  # starting with a pristine slab

    # adsorb at one site
    test_slab.append(element)
    test_slab.positions[-1] = ase_top1

    assert filter_distances(test_slab, ads=[element], cutoff_distance=1.5) == True


def test_two_O_fail(pristine_slab):
    test_slab = pristine_slab.copy()  # starting with a pristine slab

    # adsorb at two sites
    test_slab.append(element)
    test_slab.positions[-1] = ase_bridge
    test_slab.append(element)
    test_slab.positions[-1] = ase_top1

    assert filter_distances(test_slab, ads=[element], cutoff_distance=1.5) == False


def test_two_O_pass(pristine_slab):
    test_slab = pristine_slab.copy()  # starting with a pristine slab

    # adsorb at two sites
    test_slab.append(element)
    test_slab.positions[-1] = ase_top1
    test_slab.append(element)
    test_slab.positions[-1] = ase_top2

    assert filter_distances(test_slab, ads=[element], cutoff_distance=1.5) == True


def test_cell_distance_failed():
    test_slab = read(
        os.path.join(current_dir, "resources/SrTiO3_001_distance_failed.cif")
    )

    assert filter_distances(test_slab, ads=[element], cutoff_distance=1.5) == False
