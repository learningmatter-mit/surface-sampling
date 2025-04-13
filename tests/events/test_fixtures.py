"""Fixtures for the tests in the events module."""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator

from mcmc.system import SurfaceSystem


@pytest.fixture()
def system():
    """Create a dummy SurfaceSystem object for testing."""
    atoms = Atoms(
        "GaAsGaAs", positions=[[0, 0, 0], [0, 0, 3], [1, 1, 1], [1, 1, 4]]
    )  # fake positions for now
    atoms.set_array("ads_group", np.array([0, 1, 0, 3]))
    ads_coords = [(0, 0, 3), (1, 1, 4), (2, 2, 5)]
    calc = Calculator()
    occ = [1, 3, 0]
    return SurfaceSystem(atoms, ads_coords=ads_coords, calc=calc, occ=occ)
