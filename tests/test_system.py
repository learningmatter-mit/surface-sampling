import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator

from mcmc.system import SurfaceSystem


@pytest.fixture
def surface_system():
    atoms = Atoms(symbols=["Ga", "As"], positions=[(0, 0, 0), (0, 0, 3)])
    ads_coords = [(0, 0, 2), (0, 0, 3)]
    calc = Calculator()
    occ = [0, 1]
    system_settings = {"relax_atoms": True}
    calc_settings = {"optimizer": "BFGS"}

    system = SurfaceSystem(atoms, ads_coords, calc, occ, system_settings, calc_settings)

    # assert surface_system.all_atoms is None
    # assert surface_system.real_atoms == atoms
    # assert surface_system.system_settings == system_settings
    # assert surface_system.calc_settings == calc_settings
    # assert surface_system.relax_atoms == system_settings.get("relax_atoms", False)
    # assert surface_system.results == {}
    # assert surface_system._states == {}
    # assert surface_system.constraints == []
    # assert surface_system.surface_area == 0.0
    # assert surface_system.surface_idx == []
    # assert surface_system.bulk_idx == []
    # assert surface_system.ads_idx == []
    # assert surface_system.ads_coords == ads_coords
    # assert surface_system.occ == occ
    # assert surface_system.distance_matrix == []

    return system


def test_surface_system_save_and_restore_state(surface_system):
    starting_occ = [0, 1]
    starting_results = {"energy": 0.0, "forces": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]}
    surface_system.occ = starting_occ
    surface_system.results = starting_results
    surface_system.save_state("start_state")

    ending_occ = [1, 0]
    ending_results = {
        "energy": 1.0,
        "forces": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    }
    surface_system.occ = ending_occ
    surface_system.results = ending_results
    surface_system.save_state("end_state")

    surface_system.restore_state("start_state")
    assert surface_system.occ == starting_occ
    # test the restored results are a dict
    assert isinstance(surface_system.results, dict)
    assert surface_system.results == starting_results

    surface_system.restore_state("end_state")
    assert surface_system.occ == ending_occ
    # test the restored results are a dict
    assert isinstance(surface_system.results, dict)
    assert surface_system.results == ending_results


# def test_surface_system_get_relaxed_energy():
#     atoms = Atoms(symbols=["Ga", "As"], positions=[(0, 0, 0), (0, 0, 1)])
#     ads_coords = [(0, 0, 2), (0, 0, 3)]
#     calc = Calculator()
#     occ = [0, 1]
#     system_settings = {"relax_atoms": True}
#     calc_settings = {"optimizer": "BFGS"}

#     surface_system = SurfaceSystem(atoms, ads_coords, calc, occ, system_settings, calc_settings)

#     energy = surface_system.get_relaxed_energy()
#     assert isinstance(energy, float)

# def test_surface_system_get_unrelaxed_energy():
#     atoms = Atoms(symbols=["Ga", "As"], positions=[(0, 0, 0), (0, 0, 1)])
#     ads_coords = [(0, 0, 2), (0, 0, 3)]
#     calc = Calculator()
#     occ = [0, 1]
#     system_settings = {"relax_atoms": True}
#     calc_settings = {"optimizer": "BFGS"}

#     surface_system = SurfaceSystem(atoms, ads_coords, calc, occ, system_settings, calc_settings)

#     energy = surface_system.get_unrelaxed_energy()
#     assert isinstance(energy, float)

# def test_surface_system_get_potential_energy():
#     atoms = Atoms(symbols=["Ga", "As"], positions=[(0, 0, 0), (0, 0, 1)])
#     ads_coords = [(0, 0, 2), (0, 0, 3)]
#     calc = Calculator()
#     occ = [0, 1]
#     system_settings = {"relax_atoms": True}
#     calc_settings = {"optimizer": "BFGS"}

#     surface_system = SurfaceSystem(atoms, ads_coords, calc, occ, system_settings, calc_settings)

#     energy = surface_system.get_potential_energy()
#     assert isinstance(energy, float)

# def test_surface_system_get_surface_energy():
#     atoms = Atoms(symbols=["Ga", "As"], positions=[(0, 0, 0), (0, 0, 1)])
#     ads_coords = [(0, 0, 2), (0, 0, 3)]
#     calc = Calculator()
#     occ = [0, 1]
#     system_settings = {"relax_atoms": True}
#     calc_settings = {"optimizer": "BFGS"}

#     surface_system = SurfaceSystem(atoms, ads_coords, calc, occ, system_settings, calc_settings)

#     surface_energy = surface_system.get_surface_energy()
#     assert isinstance(surface_energy, float)

# def test_surface_system_get_forces():
#     atoms = Atoms(symbols=["Ga", "As"], positions=[(0, 0, 0), (0, 0, 1)])
#     ads_coords = [(0, 0, 2), (0, 0, 3)]
#     calc = Calculator()
#     occ = [0, 1]
#     system_settings = {"relax_atoms": True}
#     calc_settings = {"optimizer": "BFGS"}

#     surface_system = SurfaceSystem(atoms, ads_coords, calc, occ, system_settings, calc_settings)

#     forces = surface_system.get_forces()
#     assert isinstance(forces, list) or isinstance(forces, np.ndarray)

# def test_surface_system_length():
#     atoms = Atoms(symbols=["Ga", "As"], positions=[(0, 0, 0), (0, 0, 1)])
#     ads_coords = [(0, 0, 2), (0, 0, 3)]
#     calc = Calculator()
#     occ = [0, 1]
#     system_settings = {"relax_atoms": True}
#     calc_settings = {"optimizer": "BFGS"}

#     surface_system = SurfaceSystem(atoms, ads_coords, calc, occ, system_settings, calc_settings)

#     assert len(surface_system) == len(atoms)

# from ase import Atoms
# from ase.calculators.calculator import Calculator
# from surface_system import SurfaceSystem


# def test_surface_system_initialization():
#     atoms = Atoms(symbols=["Ga", "As"], positions=[(0, 0, 0), (0, 0, 1)])
#     ads_coords = [(0, 0, 2), (0, 0, 3)]
#     calc = Calculator()
#     occ = [0, 1]
#     system_settings = {"relax_atoms": True}
#     calc_settings = {"optimizer": "BFGS"}

#     surface_system = SurfaceSystem(atoms, ads_coords, calc, occ, system_settings, calc_settings)

#     assert surface_system.all_atoms is None
#     assert surface_system.real_atoms == atoms
#     assert surface_system.system_settings == system_settings
#     assert surface_system.calc_settings == calc_settings
#     assert surface_system.relax_atoms == system_settings.get("relax_atoms", False)
#     assert surface_system.results == {}
#     assert surface_system._states == {}
#     assert surface_system.constraints == []
#     assert surface_system.surface_area == 0.0
#     assert surface_system.surface_idx == []
#     assert surface_system.bulk_idx == []
#     assert surface_system.ads_idx == []
#     assert surface_system.ads_coords == ads_coords
#     assert surface_system.occ == occ
#     assert surface_system.distance_matrix == []

# def test_surface_system_save_and_restore_state():
#     atoms = Atoms(symbols=["Ga", "As"], positions=[(0, 0, 0), (0, 0, 1)])
#     ads_coords = [(0, 0, 2), (0, 0, 3)]
#     calc = Calculator()
#     occ = [0, 1]
#     system_settings = {"relax_atoms": True}
#     calc_settings = {"optimizer": "BFGS"}

#     surface_system = SurfaceSystem(atoms, ads_coords, calc, occ, system_settings, calc_settings)

#     surface_system.save_state("start_state")
#     surface_system.occ = [1, 0]
#     surface_system.save_state("end_state")

#     surface_system.restore_state("start_state")
#     assert surface_system.occ == occ

#     surface_system.restore_state("end_state")
#     assert surface_system.occ == [1, 0]

# def test_surface_system_get_relaxed_energy():
#     atoms = Atoms(symbols=["Ga", "As"], positions=[(0, 0, 0), (0, 0, 1)])
#     ads_coords = [(0, 0, 2), (0, 0, 3)]
#     calc = Calculator()
#     occ = [0, 1]
#     system_settings = {"relax_atoms": True}
#     calc_settings = {"optimizer": "BFGS"}

#     surface_system = SurfaceSystem(atoms, ads_coords, calc, occ, system_settings, calc_settings)

#     energy = surface_system.get_relaxed_energy()
#     assert isinstance(energy, float)

# def test_surface_system_get_unrelaxed_energy():
#     atoms = Atoms(symbols=["Ga", "As"], positions=[(0, 0, 0), (0, 0, 1)])
#     ads_coords = [(0, 0, 2), (0, 0, 3)]
#     calc = Calculator()
#     occ = [0, 1]
#     system_settings = {"relax_atoms": True}
#     calc_settings = {"optimizer": "BFGS"}

#     surface_system = SurfaceSystem(atoms, ads_coords, calc, occ, system_settings, calc_settings)

#     energy = surface_system.get_unrelaxed_energy()
#     assert isinstance(energy, float)

# def test_surface_system_get_potential_energy():
#     atoms = Atoms(symbols=["Ga", "As"], positions=[(0, 0, 0), (0, 0, 1)])
#     ads_coords = [(0, 0, 2), (0, 0, 3)]
#     calc = Calculator()
#     occ = [0, 1]
#     system_settings = {"relax_atoms": True}
#     calc_settings = {"optimizer": "BFGS"}

#     surface_system = SurfaceSystem(atoms, ads_coords, calc, occ, system_settings, calc_settings)

#     energy = surface_system.get_potential_energy()
#     assert isinstance(energy, float)

# def test_surface_system_get_surface_energy():
#     atoms = Atoms(symbols=["Ga", "As"], positions=[(0, 0, 0), (0, 0, 1)])
#     ads_coords = [(0, 0, 2), (0, 0, 3)]
#     calc = Calculator()
#     occ = [0, 1]
#     system_settings = {"relax_atoms": True}
#     calc_settings = {"optimizer": "BFGS"}

#     surface_system = SurfaceSystem(atoms, ads_coords, calc, occ, system_settings, calc_settings)

#     surface_energy = surface_system.get_surface_energy()
#     assert isinstance(surface_energy, float)

# def test_surface_system_get_forces():
#     atoms = Atoms(symbols=["Ga", "As"], positions=[(0, 0, 0), (0, 0, 1)])
#     ads_coords = [(0, 0, 2), (0, 0, 3)]
#     calc = Calculator()
#     occ = [0, 1]
#     system_settings = {"relax_atoms": True}
#     calc_settings = {"optimizer": "BFGS"}

#     surface_system = SurfaceSystem(atoms, ads_coords, calc, occ, system_settings, calc_settings)

#     forces = surface_system.get_forces()
#     assert isinstance(forces, list) or isinstance(forces, np.ndarray)

# def test_surface_system_length():
#     atoms = Atoms(symbols=["Ga", "As"], positions=[(0, 0, 0), (0, 0, 1)])
#     ads_coords = [(0, 0, 2), (0, 0, 3)]
#     calc = Calculator()
#     occ = [0, 1]
#     system_settings = {"relax_atoms": True}
#     calc_settings = {"optimizer": "BFGS"}

#     surface_system = SurfaceSystem(atoms, ads_coords, calc, occ, system_settings, calc_settings)

#     assert len(surface_system) == len(atoms)
