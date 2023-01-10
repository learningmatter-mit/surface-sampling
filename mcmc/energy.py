import copy
import json
import logging
import os
from collections import Counter

import numpy as np
from ase import io
from ase.optimize import BFGS
from lammps import lammps
from nff.io.ase import AtomsBatch
from nff.utils.constants import HARTREE_TO_EV

logger = logging.getLogger(__name__)


def run_lammps_opt(slab, main_dir=os.getcwd()):
    curr_dir = os.getcwd()

    # config file is assumed to be stored in the folder you run lammps
    config = json.load(open(f"{curr_dir}/lammps_config.json"))
    potential_file = config["potential_file"]
    atoms = config["atoms"]

    # define necessary file locations
    lammps_data_file = f"{main_dir}/lammps.data"
    lammps_in_file = f"{main_dir}/lammps.in"
    lammps_out_file = f"{main_dir}/lammps.out"
    cif_from_lammps_path = f"{main_dir}/lammps.cif"

    # write current surface into lammps.data
    slab.write(
        lammps_data_file, format="lammps-data", units="real", atom_style="atomic"
    )

    TEMPLATE = open(f"{curr_dir}/lammps_template.txt", "r").read()
    # write lammps.in file
    with open(lammps_in_file, "w") as f:
        f.writelines(
            TEMPLATE.format(lammps_data_file, potential_file, *atoms, lammps_out_file)
        )

    lmp = lammps()

    # run the LAMMPS here
    logger.debug(lmp.file(lammps_in_file))
    lmp.close()

    # Read from LAMMPS out
    opt_slab = io.read(lammps_out_file, format="lammps-data", style="atomic")

    atomic_numbers_dict = config["atomic_numbers_dict"]
    actual_atomic_numbers = [
        atomic_numbers_dict[str(x)] for x in opt_slab.get_atomic_numbers()
    ]

    opt_slab.set_atomic_numbers(actual_atomic_numbers)
    opt_slab.calc = slab.calc

    return opt_slab


def optimize_slab(slab, optimizer="BFGS", **kwargs):
    """Run relaxation for slab

    Parameters
    ----------
    slab : ase.Atoms
        Surface slab
    optimizer : str, optional
        Either  BFGS or LAMMPS, by default 'BFGS'

    Returns
    -------
    ase.Atoms
        Relaxed slab
    """
    if "LAMMPS" in optimizer:
        if "folder_name" in kwargs:
            folder_name = kwargs["folder_name"]
            calc_slab = run_lammps_opt(slab, main_dir=folder_name)
        else:
            calc_slab = run_lammps_opt(slab)
    else:
        if type(slab) is AtomsBatch:
            slab.update_nbr_list(update_atoms=True)
        calc_slab = copy.deepcopy(slab)
        calc_slab.calc = slab.calc
        if (
            kwargs.get("folder_name", None)
            and kwargs.get("iter", None)
            and kwargs.get("save", False)
        ):
            # save every 10 steps
            iter = int(kwargs.get("iter"))
            # if iter % 10 == 0:
            # save only when told to
            dyn = BFGS(
                calc_slab,
                trajectory=os.path.join(
                    kwargs["folder_name"],
                    f"final_slab_traj_{iter:04}.traj",
                ),
            )
        else:
            dyn = BFGS(calc_slab)

        # default steps is 20 and max forces are 0.2
        # TODO set up a config file to change this
        steps = kwargs.get("relax_steps", 20)
        dyn.run(steps=steps, fmax=0.2)

        if (
            kwargs.get("folder_name", None)
            and kwargs.get("iter", None)
            and kwargs.get("save", False)
        ):
            # save the final frame as cif
            iter = int(kwargs.get("iter"))
            calc_slab.write(
                f"{kwargs['folder_name']}/optim_slab_run_{iter:03}_{calc_slab.get_chemical_formula()}.cif"
            )

    return calc_slab


def slab_energy(slab, relax=False, **kwargs):
    """Calculate slab energy."""

    if relax:
        # calculate without relax first
        energy, energy_std, max_force, force_std = slab_energy(
            slab, relax=False, **kwargs
        )
        # threshold for unrelaxed energy
        UNRELAXED_ENERGY_THRESHOLD = 1000
        UNRELAXED_MAX_FORCE_THRESHOLD = 1000

        if (
            np.abs(energy) > UNRELAXED_ENERGY_THRESHOLD
            or max_force > UNRELAXED_MAX_FORCE_THRESHOLD
        ):
            logger.info("encountered energy or forces out of bounds")
            logger.info(f"max_force {max_force:.3f}, energy {energy:.3f}")

            if kwargs.get("folder_name", None) and kwargs.get("iter", None):
                # save the final frame as cif
                logger.info("saving this slab")
                iter = int(kwargs.get("iter"))
                slab.write(
                    f"{kwargs['folder_name']}/oob_trial_slab_run_{energy:.3f}_{max_force:.3f}_{iter:03}_{slab.get_chemical_formula()}.cif"
                )

            # TODO: save the trajectory
            # optimize_slab(slab, **kwargs)

            # these energies or forces are out of bounds, thus
            # we set a high energy for mcmc to reject
            energy = UNRELAXED_ENERGY_THRESHOLD
            # energy = np.sign(energy) * UNRELAXED_ENERGY_THRESHOLD

            return energy, energy_std, max_force, force_std

        slab = optimize_slab(slab, **kwargs)

    if type(slab) is AtomsBatch:
        slab.update_nbr_list(update_atoms=True)
        slab.calc.calculate(slab)
        energy = float(slab.results["energy"])
        if kwargs.get("offset", None):
            if not kwargs.get("offset_data", None):
                raise Exception(f"No offset_data.json file specified!")
            else:
                with open(kwargs["offset_data"]) as f:
                    offset_data = json.load(f)
                bulk_energies = offset_data["bulk_energies"]
                stoidict = offset_data["stoidict"]

            ad = Counter(slab.get_chemical_symbols())

            # procedure is
            # 1: to add the linear regression coeffs back in
            ref_en = 0
            for ele, num in ad.items():
                ref_en += num * stoidict[ele]
            ref_en += stoidict["offset"]

            energy += ref_en * HARTREE_TO_EV

            # 2: subtract the bulk energies
            # TODO: generalize this
            bulk_ref_en = (
                ad["Ti"] * bulk_energies["SrTiO3"]
                + (ad["Sr"] - ad["Ti"]) * bulk_energies["Sr"]
                + (ad["O"] - 3 * ad["Ti"]) * bulk_energies["O2"] / 2
            )

            energy -= bulk_ref_en * HARTREE_TO_EV

        energy_std = float(slab.results["energy_std"])
        max_force = float(np.abs(slab.results["forces"]).max())
        force_std = float(slab.results["forces_std"].mean())

    else:
        energy = float(slab.get_potential_energy())
        energy_std = 0.0
        max_force = 0.0
        force_std = 0.0

    return energy, energy_std, max_force, force_std
