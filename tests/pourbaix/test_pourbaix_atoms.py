from pathlib import Path

import pytest
from pytest import approx

from mcmc.pourbaix.atoms import PourbaixAtom, generate_pourbaix_atoms


@pytest.fixture
def phase_diagram_path():
    return Path(__file__).parent / "data/pd_dict.json"


@pytest.fixture
def pourbaix_diagram_path():
    return Path(__file__).parent / "data/pbx_dict.json"


@pytest.fixture
def elements():
    return ("Sr", "Ir", "O")


def test_generate_pourbaix_atoms_1(phase_diagram_path, pourbaix_diagram_path, elements):
    # Define test data
    phi = 1.0
    pH = 0.0

    # Generate pourbaix atoms
    pourbaix_atoms = generate_pourbaix_atoms(
        phase_diagram_path, pourbaix_diagram_path, phi, pH, elements
    )

    # Check the generated pourbaix atoms
    assert isinstance(pourbaix_atoms, dict)
    assert len(pourbaix_atoms) == 3

    # Check Sr pourbaix atom
    sr_pourbaix_atom = pourbaix_atoms["Sr"]
    assert isinstance(sr_pourbaix_atom, PourbaixAtom)
    assert sr_pourbaix_atom.symbol == "Sr"
    assert sr_pourbaix_atom.dominant_species == "Sr[+2]"
    assert sr_pourbaix_atom.species_conc == 1e-6
    assert sr_pourbaix_atom.num_e == 2
    assert sr_pourbaix_atom.num_H == 0
    assert sr_pourbaix_atom.atom_std_state_energy == approx(-1.68949, rel=1e-5)
    assert sr_pourbaix_atom.delta_G2_std == approx(-5.79807, rel=1e-5)

    # Check Ir pourbaix atom
    ir_pourbaix_atom = pourbaix_atoms["Ir"]
    assert isinstance(ir_pourbaix_atom, PourbaixAtom)
    assert ir_pourbaix_atom.symbol == "Ir"
    assert ir_pourbaix_atom.dominant_species == "IrO2"
    assert ir_pourbaix_atom.species_conc == 1
    assert ir_pourbaix_atom.num_e == 4
    assert ir_pourbaix_atom.num_H == 4
    assert ir_pourbaix_atom.atom_std_state_energy == approx(-8.83843, rel=1e-5)
    assert ir_pourbaix_atom.delta_G2_std == approx(1.76738, rel=1e-5)

    # Check O pourbaix atom
    o_pourbaix_atom = pourbaix_atoms["O"]
    assert isinstance(o_pourbaix_atom, PourbaixAtom)
    assert o_pourbaix_atom.symbol == "O"
    assert o_pourbaix_atom.dominant_species == "H2O"
    assert o_pourbaix_atom.species_conc == 1
    assert o_pourbaix_atom.num_e == -2
    assert o_pourbaix_atom.num_H == -2
    assert o_pourbaix_atom.atom_std_state_energy == approx(-5.26469, rel=1e-5)
    assert o_pourbaix_atom.delta_G2_std == approx(-2.45830, rel=1e-5)


def test_generate_pourbaix_atoms_2(phase_diagram_path, pourbaix_diagram_path, elements):
    # Define test data
    phi = 0.0
    pH = 0.0

    # Generate pourbaix atoms
    pourbaix_atoms = generate_pourbaix_atoms(
        phase_diagram_path, pourbaix_diagram_path, phi, pH, elements
    )

    # Check the generated pourbaix atoms
    assert isinstance(pourbaix_atoms, dict)
    assert len(pourbaix_atoms) == 3

    # Check Sr pourbaix atom
    sr_pourbaix_atom = pourbaix_atoms["Sr"]
    assert isinstance(sr_pourbaix_atom, PourbaixAtom)
    assert sr_pourbaix_atom.symbol == "Sr"
    assert sr_pourbaix_atom.dominant_species == "Sr[+2]"
    assert sr_pourbaix_atom.species_conc == 1e-6
    assert sr_pourbaix_atom.num_e == 2
    assert sr_pourbaix_atom.num_H == 0
    assert sr_pourbaix_atom.atom_std_state_energy == approx(-1.68949, rel=1e-5)
    assert sr_pourbaix_atom.delta_G2_std == approx(-5.79807, rel=1e-5)

    # Check Ir pourbaix atom
    ir_pourbaix_atom = pourbaix_atoms["Ir"]
    assert isinstance(ir_pourbaix_atom, PourbaixAtom)
    assert ir_pourbaix_atom.symbol == "Ir"
    assert ir_pourbaix_atom.dominant_species == "Ir"
    assert ir_pourbaix_atom.species_conc == 1
    assert ir_pourbaix_atom.num_e == 0
    assert ir_pourbaix_atom.num_H == 0
    assert ir_pourbaix_atom.atom_std_state_energy == approx(-8.83843, rel=1e-5)
    assert ir_pourbaix_atom.delta_G2_std == approx(0.0, rel=1e-5)

    # Check O pourbaix atom
    o_pourbaix_atom = pourbaix_atoms["O"]
    assert isinstance(o_pourbaix_atom, PourbaixAtom)
    assert o_pourbaix_atom.symbol == "O"
    assert o_pourbaix_atom.dominant_species == "H2O"
    assert o_pourbaix_atom.species_conc == 1
    assert o_pourbaix_atom.num_e == -2
    assert o_pourbaix_atom.num_H == -2
    assert o_pourbaix_atom.atom_std_state_energy == approx(-5.26469, rel=1e-5)
    assert o_pourbaix_atom.delta_G2_std == approx(-2.45830, rel=1e-5)