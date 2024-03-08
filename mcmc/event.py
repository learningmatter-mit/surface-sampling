# should be good with these classes


class EventInfo:
    # hoje's event info has the following
    def __init__(self) -> None:
        pass


class EventGenerator:
    def __init__(self) -> None:
        pass

    def get_event(self, surface, **kwargs):
        pass


class Event:
    def __init__(self, surface) -> None:
        self.complete = False
        self.surface = surface

    def forward_action(self):
        # do some adsorption
        self.complete = True

    def backward_action(self):
        self.complete = False

    def compute_boltzman_factor(self):
        # compute the boltzman factor
        # Ef - Ei
        # N
        # return the probability
        return probability


class CanonicalEvent(Event):
    # distance-based decay
    # random
    def __init__(self, surface) -> None:
        super().__init__(surface)


class SemigrandEvent(Event):
    # distance-based decay
    # random
    def __init__(self, surface) -> None:
        super().__init__(surface)


class MCMCSampling:
    def __init__(self) -> None:
        ...
        self.surface = None

    def run(self, surface=None, nsteps: int = 100):
        ...
        if surface is None:
            surface = self.surface
        else:
            self.surface = surface

        if self.surface is None:
            raise ValueError("Surface not set")

        # run the MCMC sampling
        for i in range(nsteps):
            self.step()

    def step(self):
        event = self.event_generator.get_event(self.surface, **kwargs)

        accept = self.acceptance_criterion(event)

        # can do something like this
        log_dict = {
            "action": action,
            "probability": probability,
            "yesorno": yesorno,
            "Ef": Ef,
            "Ei": Ei,
            "N": N,
        }
        return log_dict


# follow Hoje's Event and actions
# something like this following
class Desorption(Event):
    def __init__(self, system: EpitaxySystem, info: EventInfo, **kwargs):
        super().__init__(system, info)
        self.site_idx: int = self.info["site_idx"]
        self.adsorbate_idx: int = system.occ[self.site_idx] - 1
        self.adsorbate: str = system.gratoms.get_chemical_symbols()[self.adsorbate_idx]

    def get_atoms_pair(self) -> Tuple[Atoms, Atoms]:
        assert not self.done, "Event has taken happened"
        # mask, _ = self.system.create_mask(center_index, cutoff)
        prev_atomic_numbers = self.system.gratoms.get_atomic_numbers()
        curr_atomic_numbers = self.system.gratoms.get_atomic_numbers()
        positions = self.system.gratoms.get_positions()
        cell = self.system.gratoms.cell
        pbc = self.system.gratoms.pbc

        prev_atoms = Atoms(prev_atomic_numbers, positions=positions, cell=cell, pbc=pbc)
        curr_atomic_numbers[self.adsorbate_idx] = 0
        curr_atoms = Atoms(curr_atomic_numbers, positions=positions, cell=cell, pbc=pbc)
        # curr_atoms.set_atomic_numbers(atomic_numbers)

        return prev_atoms, curr_atoms

    def action(self):
        self.system.save_state("previous")
        self.system = remove_atom(self.system, self.site_idx)
        self.system.save_state("current")
        self.system.save_state("next")
        self.done = True

    def _calculate_deltaG(self) -> float:
        assert self.done, "Event has not happened yet"
        # final_state = self.system.restore_state("next")
        final_e = self.system.gratoms.get_potential_energy()
        self.system.restore_state("previous")
        initial_e = self.system.gratoms.get_potential_energy()
        self.system.restore_state("next")
        adsorbate_e = self.system.atomic_energies.get(self.adsorbate)
        deltaG = final_e + adsorbate_e - initial_e

        return deltaG
