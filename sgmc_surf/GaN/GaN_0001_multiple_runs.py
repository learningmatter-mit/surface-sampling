import argparse
import sys
sys.path.append("/home/dux/")
sys.path.append("/home/dux/surface_sampling/sgmc_surf")

from datetime import datetime
from mcmc import mcmc_run

from ase.calculators.lammpsrun import LAMMPS
from ase.io import read
from ase.build import make_supercell
from catkit.gen.adsorption import get_adsorption_sites

import catkit
import os
import numpy as np
import matplotlib.pyplot as plt

from htvs.djangochem.pgmols.utils import surfaces

def run_process(num_sweeps=100):
    # Get pristine surface
    # GaN 0001 surface
    atoms = read('GaN_hexagonal.cif')

    # supercell_atoms = atoms*(2,2,2)
    # supercell_atoms.write('GaN_hexagonal_2x2.cif')

    supercell_atoms = atoms*(3,3,3)
    supercell_atoms.write('GaN_hexagonal_3x3.cif')

    slab, surface_atoms = surfaces.surface_from_bulk(supercell_atoms, [0,0,0,-1], size=[3,3], vacuum=10)

    # try 2003 tersoff potential 
    parameters = {
        'pair_style': 'tersoff',
        'pair_coeff': ['* * GaN.tersoff Ga N']
    }
    potential_file = os.path.join(os.environ["LAMMPS_POTENTIALS"], 'GaN.sw')
    lammps_calc = LAMMPS(files=[potential_file], keep_tmp_files=False, keep_alive=False, tmp_dir="/home/dux/surface_sampling/tmp_files")
    lammps_calc.set(**parameters)

    element = 'Ga'
    ads = catkit.gratoms.Gratoms(element)

    # starting from more random initial positions
    num_ads_atoms = 12 # needs to have so many atoms

    slab, surface_atoms = surfaces.surface_from_bulk(supercell_atoms, [0,0,0,-1], size=[3,3], vacuum=10)
    # get initial adsorption sites
    # proper_adsorbed = read("GaN_0001_3x3_12_Ga_ads_initial_slab.cif")
    # ads_positions = proper_adsorbed.get_positions()[len(slab):]
    # assert len(ads_positions) == num_ads_atoms, "num of adsorption sites does not match num ads atoms"

    # canonical with relaxation
    # num_sweeps = 100
    surface_name = "GaN_0001_3x3"
    alpha = 0.99
    slab, surface_atoms = surfaces.surface_from_bulk(supercell_atoms, [0,0,0,-1], size=[3,3], vacuum=10)
    # set surface atoms from the other side
    all_atoms = np.arange(len(slab))
    curr_surf_atoms = slab.get_surface_atoms()
    new_surf_atoms = np.setdiff1d(all_atoms, curr_surf_atoms)
    slab.set_surface_atoms(new_surf_atoms)
    # invert the positions
    slab.set_scaled_positions(1 - slab.get_scaled_positions())

    # try positive chem pot
    chem_pot = 5
    history, energy_hist, frac_accept_hist, adsorption_count_hist = mcmc_run(num_sweeps=num_sweeps, temp=1, pot=chem_pot, alpha=alpha, slab=slab, calc=lammps_calc, surface_name=surface_name, element=element, canonical=True, num_ads_atoms=num_ads_atoms, relax=True)

    runs = range(1, num_sweeps+1)

    # do the plots
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    ax[0, 0].plot(runs, energy_hist)
    ax[0, 0].set_xlabel("Iter")
    ax[0, 0].set_ylabel("Energy (E)")
    ax[0, 0].set_title("Energy (E) vs Sweeps")

    ax[0, 1].plot(runs, frac_accept_hist)
    ax[0, 1].set_xlabel("Iter")
    ax[0, 1].set_ylabel("Fraction accepted")
    ax[0, 1].set_title("Fraction accepted vs Sweeps")

    ax[1, 1].plot(runs, np.array(list(adsorption_count_hist.values())).T)
    ax[1, 1].set_xlabel("Iter")
    ax[1, 1].set_ylabel("Adsorption count")
    ax[1, 1].legend(adsorption_count_hist.keys())
    ax[1, 1].set_title("Adsorption count vs Iterations")

    # fig.show()
    start_timestamp = datetime.now().strftime("%Y%m%d-%H%M")

    fig.savefig(f"iter{num_sweeps}_{start_timestamp}.png")
    fig.tight_layout()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run MCMC')
    parser.add_argument('--runs', type=int, help='Num runs')
    args = parser.parse_args()
    print(f"Submitting with iter={args.runs}")
    run_process(args.runs)
