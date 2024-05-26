import pytest

from mcmc.events.criterion import DistanceCriterion, MetropolisCriterion
from tests.events.test_fixtures import system


# test distance criterion
def test_distance_criterion_accept(system):
    criterion = DistanceCriterion(filter_distance=1.5)

    assert criterion.criterion(system, ["Ga"]) is True


def test_metropolis_criterion_accept(system):
    criterion = MetropolisCriterion(0.0257)  # Set temperature to 300 K
    # Set the energies for the system
    system.results = {"surface_energy": 10.0}
    system.save_state("before")
    system.get_surface_energy = lambda recalculate=True: 5.0
    system.save_state("after")

    assert criterion.criterion(system) == True


def test_metropolis_criterion_reject(system):
    criterion = MetropolisCriterion(0.0257)  # Set temperature to 300 K
    # Set the energies for the system
    system.results = {"surface_energy": 10.0}
    system.save_state("before")
    system.get_surface_energy = lambda recalculate=True: 15.0
    system.save_state("after")

    assert criterion.criterion(system) == False


def test_metropolis_criterion_no_energy(system):
    criterion = MetropolisCriterion(0.0257)  # Set temperature to 300 K
    # Set the energies for the system
    system.results = {}
    system.save_state("before")
    system.get_surface_energy = lambda recalculate=True: 5.0
    system.save_state("after")
    # delta energy is 0, exp(0) = 1

    assert criterion.criterion(system) == True
