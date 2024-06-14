import pytest
from ase import Atoms
from numpy import array_equal
from ase.calculators.calculator import Calculator
from catkit.gen.surface import SlabGenerator
from mcmc.system import SurfaceSystem
from chgnet.model.dynamics import CHGNetCalculator


@pytest.fixture
def surface_system():
    atoms = Atoms(symbols=["Ga", "As"], positions=[(0, 0, 0), (0, 0, 3)])
    ads_coords = [(0, 0, 2), (0, 0, 3)]
    calc = Calculator()
    surface_depth=None
    occ = [0, 1]
    system_settings = {"relax_atoms": True}
    calc_settings = {"optimizer": "BFGS"}

    system = SurfaceSystem(atoms, ads_coords, calc, occ, surface_depth, system_settings, calc_settings)

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

@pytest.fixture
def si_slab():
    # initializing Si-149 [1,1,0] unit cell
    positions = [
        [0.        , 0.        , 0.        ],
        [1.36092559, 1.36092559, 1.36092559],
        [0.        , 2.72185118, 2.72185118],
        [4.08277678, 1.36092559, 4.08277678],
        [1.36092559, 4.08277678, 4.08277678],
        [4.08277678, 4.08277678, 1.36092559],
        [2.72185118, 0.        , 2.72185118],
        [2.72185118, 2.72185118, 0.        ],
    ]
    symbols = ['Si']*8
    pbc = [ True,  True,  True]
    cell = [5.44370237, 5.44370237, 5.44370237]
    atoms = Atoms(positions=positions,symbols=symbols,pbc=pbc,cell=cell)
    # fixed and surface_depth work in opposite directions
    gen = SlabGenerator(
         atoms,miller_index=[1,1,0],layers=3,fixed=2,layer_type='trim',
         standardize_bulk=True,tol=1e-05)
    slab = gen.get_slab(iterm=0)
    slab = gen.set_size(slab, [2,2])
    slab.center(vacuum=15, axis=2)
    return slab

def test_surface_system_constraint_retention(si_slab):
    unchanged = SurfaceSystem(si_slab,ads_coords=[],calc=CHGNetCalculator(),calc_settings={"relax_atoms":True})
    unchanged_constraints = unchanged.real_atoms.constraints[0].todict()['kwargs']['indices']
    unchanged_relaxed_constraints = unchanged.relaxed_atoms.constraints[0].todict()['kwargs']['indices']
    original_constraints = si_slab.constraints[0].todict()['kwargs']['indices']
    print(f'HEREEEEEEEEEEE {unchanged_constraints}')
    tags = unchanged.real_atoms.get_tags()
    #checking proper constraint application
    assert unchanged_constraints == unchanged_relaxed_constraints
    assert original_constraints == unchanged_constraints
    assert len(unchanged_constraints) == 16
    #checking lower two layers are constrained
    for idx in unchanged_constraints:
        assert tags[idx] != 1
    

def test_surface_system_constraint_setting(si_slab):
    unconstrained = SurfaceSystem(si_slab,ads_coords=[],calc=CHGNetCalculator(),surface_depth=3,calc_settings={"relax_atoms":True})
    #checking full removal of constraints
    unconstrained_constraints = unconstrained.real_atoms.constraints[0].todict()['kwargs']['indices']
    unconstrained_relaxed = unconstrained.relaxed_atoms.constraints[0].todict()['kwargs']['indices']
    assert unconstrained_constraints == []
    assert unconstrained_constraints == unconstrained_relaxed

    partially_constrained = SurfaceSystem(si_slab,ads_coords=[],calc=CHGNetCalculator(),surface_depth=2,calc_settings={"relax_atoms":True})
    partial_constraints = partially_constrained.real_atoms.constraints[0].todict()['kwargs']['indices']
    relaxed_constraints = partially_constrained.relaxed_atoms.constraints[0].todict()['kwargs']['indices']
    tags = partially_constrained.real_atoms.get_tags()
    relaxed_tags = partially_constrained.relaxed_atoms.get_tags()
    
    assert partial_constraints == relaxed_constraints
    print(type(tags))
    print(type(relaxed_tags))
    assert array_equal(tags,relaxed_tags)
    assert len(partial_constraints) == 8
    #checking if constraints were set on bottom layer
    for idx in partial_constraints:
         assert tags[idx] == 3
    

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
