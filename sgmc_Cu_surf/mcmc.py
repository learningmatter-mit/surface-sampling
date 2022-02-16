import numpy as np  
import matplotlib.pyplot as plt

"""Performs Semi-Grand Monte Carlo (SGMC) reconstruction of Cu surface.
Produces a temperature/structure map
"""

import os
import sys
sys.path.append("/home/dux/")

import matplotlib.pyplot as plt
import numpy as np

# from labutil.src.plugins.lammps import (lammps_run, get_lammps_energy)
# from labutil.src.objects import (ClassicalPotential, Struc, ase2struc, Dir)
from ase.spacegroup import crystal
from ase.build import make_supercell, bulk
from ase.io import write
import ase
import catkit
from catkit.gen.adsorption import get_adsorption_sites
import random
from collections import Counter

from ase.calculators.eam import EAM

def initialize_slab(alat, elem='Cu', vacuum=15.0):
    """Creates the slab structure using ASE.

    Parameters
    ----------
    alat : float
        Lattice parameter in angstroms
    """

    # slab = fcc100(elem, size=(4,4,4), a=alat, vacuum=vacuum)


    # TODO: adjust size of surface if necessary
    a1 = bulk(elem, 'fcc', a=alat)
    catkit_slab = catkit.build.surface(a1, size=(4,4,4), miller=(1,0,0), termination=0, fixed=0, vacuum=vacuum, orthogonal=False)

    write('catkit_slab.cif', catkit_slab)
    return catkit_slab

def get_random_idx(connectivity, type=None):
    """Get random site index
    """
    connectivities = {
        'top': 1,
        'bridge': 2,
        'hollow': 4
    } # defaults to hollow

    # top should have connectivity 1, bridge should be 2 and hollow more like 4

    if type:
        # TODO: implemente preferred connecitivity
        pass
    
    site_idx = random.randrange(len(connectivity))

    return site_idx

'''
Code block to calculate energy of the slab
'''
def slab_energy(slab):
    """Calculate slab energy.
    Probably will have to call external lib
    """
    potential = EAM(potential='Cu2.eam.fs')
    
    slab.calc = potential
    energy = slab.get_potential_energy()
    return energy

def spin_flip(state, slab, temp, pot, prev_energy=None, save_cif=False, iter=1, site_idx=None, testing=False, folder_name="."):
    """Based on the Ising model, models the adsorption/desorption of atoms from surface lattice sites

    Parameters
    ----------
    state : np.array
        dimension the number of sites
    slab : catkit.gratoms.Gratoms
        model of the surface slab
    temp : float
        temperature
    pot : float
        chemical potential of metal (Cu)

    Returns
    -------
    np.array, float
        new state, energy change
    """

    # choose a site to flip
    coords, connectivity, sym_idx = get_adsorption_sites(slab, symmetry_reduced=False)

    if not site_idx:
        site_idx = get_random_idx(connectivity)
    rand_site = coords[site_idx]

    print(f"\n we are at iter {iter}")
    print(f"idx is {site_idx} with connectivity {connectivity[site_idx]} at {rand_site}")

    # determine if site vacant or filled
    filled = (state > 0)[site_idx]
    proposed_state = state.copy()
    print(f"before proposed state is")
    print(proposed_state)

    # change in number of adsorbates (Cu atoms)
    delta_N = 0

    # case site is vacant (spin down)
    if not filled:
        delta_N = 1 # add one Cu atom
        print("site is not filled, attempting to adsorb")
        print(f"current slab has {len(slab)} atoms")
        Cu = catkit.gratoms.Gratoms('Cu')
        # tag the atom to be adsorbed with its to-be index (last position on slab)    
        adsorbate_idx = len(slab)
        # Cu.set_tags(adsorbate_idx)
        proposed_state[site_idx] = adsorbate_idx
        proposed_slab_builder = catkit.gen.adsorption.Builder(slab)
        proposed_slab = proposed_slab_builder.add_adsorbate(Cu, bonds=[0], index=site_idx, auto_construct=False, symmetric=False)
        print(f"proposed slab has {len(proposed_slab)} atoms")

    # case site is filled (spin up)
    else:
        delta_N = -1 # remove one Cu atom
        print("site is filled, attempting to desorb")
        print(f"current slab has {len(slab)} atoms")
        adsorbate_idx = state[site_idx]
        # slab_tags = slab.get_tags()

        assert len(np.argwhere(state==adsorbate_idx)) <= 1, "more than 1 site found"
        assert len(np.argwhere(state==adsorbate_idx)) == 1, "no sites found"
        # np.argwhere(state==adsorbate_idx)[0,0]

        # import pdb;pdb.set_trace()

        proposed_slab = slab.copy()
        # import pdb;pdb.set_trace()
        del proposed_slab[int(adsorbate_idx)] #networkxx needs python int 
        # import pdb;pdb.set_trace()

        # lower the index for higher index items
        proposed_state = np.where(state>=int(adsorbate_idx), state-1, state)
        # remove negatives
        proposed_state = np.where(proposed_state<0, 0, proposed_state)

        # remove the adsorbate from tracking
        proposed_state[site_idx] = 0

        print(f"proposed slab has {len(proposed_slab)} atoms")
    
    print(f"after proposed state is")
    print(proposed_state)

    if save_cif:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        write(f'{folder_name}/proposed_slab_iter_{iter:03}.cif', proposed_slab)

    # to test, always accept
    accept = False
    if testing:
        slab = proposed_slab.copy()
        state = proposed_state.copy()
        energy = 0
    else:
        if not prev_energy:
            # calculate energy of current state
            prev_energy = slab_energy(slab)

        curr_energy = slab_energy(proposed_slab)

        print("prev energy is", prev_energy)
        print("curr energy is", curr_energy)

        # energy change due to flipping spin
        energy_diff = curr_energy - prev_energy

        # check if transition succeeds
        # min(1, exp(-(\delta_E-(delta_N*pot))))
        print(f"energy diff is {energy_diff}")
        print(f"potential is", pot)
        print(f"delta_N", delta_N)
        print(f"k_b T", temp)
        base_prob = np.exp(-(energy_diff-pot*delta_N)/temp)
        print(f"base probability is {base_prob}")
        if np.random.rand() < base_prob:
            # succeeds! change state
            slab = proposed_slab.copy()
            state = proposed_state.copy()
            print("state changed!")
            energy = curr_energy
            accept = True
        else:
            # failed, keep current state
            print("state kept the same")
            energy = prev_energy
            accept = False
            
    return state, slab, energy, accept

def mcmc_run(num_runs=1000, temp=1, pot=1, slab=None):
    """Performs MCMC run with given parameters, initializing with a random lattice if not given an input.
    Each run is defined as one complete sweep through the lattice. Each sweep consists of randomly picking
    a site and proposing (and accept/reject) a flip (adsorption or desorption) for a total number of times equals to the number of cells
    in the lattice. Only the resulting lattice after one run is appended to the history. Corresponding
    obversables are calculated also after each run.
    """

    # Cu lattice at 293 K, 3.6147 Å, potential ranges from 0 - 2
    if type(slab) is not catkit.gratoms.Gratoms or ase.Atoms:
        # initialize slab
        print("initializing slab")
        # Cu alat from https://www.copper.org/resources/properties/atomic_properties.html
        Cu_alat = 3.6147
        slab = initialize_slab(Cu_alat)

    # get ALL the adsorption sites
    # top should have connectivity 1, bridge should be 2 and hollow more like 4
    coords, connectivity, sym_idx = get_adsorption_sites(slab, symmetry_reduced=False)
    print(f"In pristine slab, there are a total of {len(connectivity)} sites")

    # state of each vacancy in slab. for state > 0, it's filled, and that's the index of the adsorbate atom in slab 
    state = np.zeros(len(coords), dtype=int)
    
    history = []
    energy_hist = np.random.rand(num_runs)
    # energy_sq_hist = np.random.rand(num_runs)

    adsorption_count_hist = []

    frac_accept_hist = np.random.rand(num_runs)
    

    # sweep over # sites
    sweep_size = len(coords)
    
    energy = slab_energy(slab)

    print(f"running for {sweep_size} iterations per run over a total of {num_runs} runs")

    run_folder = f"runs{num_runs}_temp{temp}_pot{pot}"

    for i in range(num_runs):
        num_accept = 0
        for j in range(sweep_size):
            # possible actions are:
            # 1) add -- choose an element with equal prob and bias the config away from closest existing atom
            # 2) remove -- randomly remove an element
            # 3) swap -- add + remove, then relax
            # According to SI flow chart, there are only 2 actions, add & remove. After each action, relax structure

            run_idx = sweep_size*i + j+1

            state, slab, energy, accept = spin_flip(state, slab, temp, pot, prev_energy=energy, save_cif=True, iter=run_idx, testing=False, folder_name=run_folder)
            num_accept += accept

        # end of sweep; append to history
        history.append(slab.copy())
        # save cif file
        write(f'{run_folder}/final_slab_run_{i+1:03}.cif', slab)

        # append values
        energy_hist[i] = energy
        # energy_sq_hist[i] = energy**2
        # import pdb; pdb.set_trace()
        ads_counts = count_adsorption_sites(slab, state)
        adsorption_count_hist.append(ads_counts)

        frac_accept = num_accept/sweep_size
        frac_accept_hist[i] = frac_accept

        

    return history, energy_hist, frac_accept_hist, adsorption_count_hist

def count_adsorption_sites(slab, state):
    _, connectivity, _ = get_adsorption_sites(slab, symmetry_reduced=False)
    occ_idx = state > 0
    return Counter(connectivity[occ_idx])



if __name__ == "__main__":
    from time import perf_counter

    start = perf_counter()
    # chem pot 0 to less complicate things
    # temp in terms of kbT
    history, energy_hist, frac_accept_hist, adsorption_count_hist = mcmc_run(num_runs=3, temp=1, pot=0, slab=None)
    stop = perf_counter()
    print(f"Time taken = {stop - start} seconds")