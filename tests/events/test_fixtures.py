import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator

from mcmc.system import SurfaceSystem


@pytest.fixture
def system():
    # Create a dummy SurfaceSystem object for testing
    atoms = Atoms(
        "GaAsGaAs", positions=[[0, 0, 0], [0, 0, 3], [1, 1, 1], [1, 1, 4]]
    )  # fake positions for now
    ads_coords = [(0, 0, 3), (1, 1, 4)]
    calc = Calculator()
    occ = [1, 3]
    return SurfaceSystem(atoms, ads_coords, calc, occ)
