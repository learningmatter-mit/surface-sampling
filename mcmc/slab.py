import itertools
import logging
import random
from collections import Counter

import catkit
import numpy as np
from ase.build import bulk
from ase.io import write

logger = logging.getLogger(__name__)


def initialize_slab(
    alat,
    elem="Cu",
    vacuum=15.0,
    miller=(1, 0, 0),
    termination=0,
    orthogonal=False,
    size=(4, 4, 4),
    **kwargs,
):
    """Creates the slab structure using ASE.

    Parameters
    ----------
    alat : float
        Lattice parameter in angstroms
    """
    # slab = fcc100(elem, size=(4,4,4), a=alat, vacuum=vacuum)

    # TODO: adjust size of surface if necessary
    a1 = bulk(elem, "fcc", a=alat)
    write(f"{elem}_a1_bulk.cif", a1)
    catkit_slab = catkit.build.surface(
        a1,
        size=size,
        miller=miller,
        termination=termination,
        fixed=0,
        vacuum=vacuum,
        orthogonal=orthogonal,
        **kwargs,
    )

    write(f"{elem}_pristine_slab.cif", catkit_slab)
    return catkit_slab


def get_random_idx(connectivity, type=None):
    """Get random site index"""
    connectivities = {"top": 1, "bridge": 2, "hollow": 4}  # defaults to hollow

    # top should have connectivity 1, bridge should be 2 and hollow more like 4
    if type:
        site_idx = random.choice(
            np.argwhere(connectivity == connectivities[type]).flatten()
        )

    else:
        site_idx = random.randrange(len(connectivity))

    return site_idx


def get_complementary_idx(state, slab):
    """Get two indices, site1 occupied and site2 unoccupied."""
    adsorbed_idx = np.argwhere(state != 0).flatten()

    # select adsorbates present in slab
    curr_ads = {
        k: list(g)
        for k, g in itertools.groupby(adsorbed_idx, key=lambda x: slab[state[x]].symbol)
    }
    # add empty sites
    empty_idx = np.argwhere(state == 0).flatten().tolist()
    curr_ads["None"] = empty_idx
    logger.debug(f"current ads {curr_ads}")

    # choose two types
    type1, type2 = random.sample(curr_ads.keys(), 2)

    # get random idx belonging to those types
    site1_idx, site2_idx = [random.choice(curr_ads[x]) for x in [type1, type2]]

    return site1_idx, site2_idx, type1, type2


def change_site(
    slab, state, pots, adsorbates, coords, site_idx, start_ads=None, end_ads=None
):
    ads_pot_dict = dict(zip(adsorbates, pots))
    chosen_ads = None

    if state[site_idx] == 0:  # empty list, no ads
        logger.debug(f"chosen site is empty")
        start_ads = "None"

        if not end_ads:
            chosen_ads = random.choice(adsorbates)
        else:
            # choose a random site
            chosen_ads = end_ads
        chosen_pot = ads_pot_dict[chosen_ads]

        logger.debug(
            f"adsorbing from empty to adsorbate {chosen_ads} and potential {chosen_pot}"
        )
        delta_pot = chosen_pot

        # modularize
        logger.debug(f"current slab has {len(slab)} atoms")

        state, slab = add_to_slab(slab, state, chosen_ads, coords, site_idx)

        logger.debug(f"proposed slab has {len(slab)} atoms")

    else:
        logger.debug(f"chosen site already adsorbed")
        if not start_ads:
            start_ads = slab[state[site_idx]].symbol

        ads_choices = adsorbates.copy()
        ads_choices.remove(start_ads)
        ads_choices.append("None")  # 'None' for desorption
        prev_pot = ads_pot_dict[start_ads]

        # chosen_idx = random.randrange(len(adsorbates))
        if not end_ads:
            chosen_ads = random.choice(ads_choices)
        else:
            # choose a random site
            chosen_ads = end_ads

        logger.debug(f"chosen ads is {chosen_ads}")

        logger.debug(f"current slab has {len(slab)} atoms")

        # desorb first, regardless of next chosen state
        state, slab = remove_from_slab(slab, state, site_idx)

        # adsorb
        if "None" not in chosen_ads:
            chosen_pot = ads_pot_dict[chosen_ads]

            logger.debug(f"replacing {start_ads} with {chosen_ads}")

            state, slab = add_to_slab(slab, state, chosen_ads, coords, site_idx)

            delta_pot = chosen_pot - prev_pot
        else:
            logger.debug(f"desorbing {start_ads}")
            delta_pot = -prev_pot  # vacant site has pot = 0

        logger.debug(f"proposed slab has {len(slab)} atoms")

    end_ads = chosen_ads

    return slab, state, delta_pot, start_ads, end_ads


def add_to_slab(slab, state, adsorbate, coords, site_idx):
    """It adds an adsorbate to a slab, and updates the state to reflect the new adsorbate

    Parameters
    ----------
    slab : ase.Atoms
        the slab we're adding adsorbates to
    state : list
        a dict of integers, where each integer represents the slab index of the adsorbate on that site. If the
    site is empty, the integer is 0.
    adsorbate : ase.Atoms
        the adsorbate molecule
    coords : list
        the coordinates of the sites on the surface
    site_idx : int
        the index of the site on the slab where the adsorbate will be placed

    Returns
    -------
        The state and slab are being returned.
    """

    adsorbate_idx = len(slab)
    state[site_idx] = adsorbate_idx
    slab.append(adsorbate)
    slab.positions[-1] = coords[site_idx]
    return state, slab


def remove_from_slab(slab, state, site_idx):
    """Remove the adsorbate from the slab and update the state

    Parameters
    ----------
    slab : ase.Atoms
        the slab object
    state : list
        a list of integers, where each integer represents the slab index of the adsorbate on that site. If the
    site is empty, the integer is 0.
    site_idx : int
        the index of the site to remove the adsorbate from

    Returns
    -------
        The state and slab are being returned.
    """
    adsorbate_idx = state[site_idx]
    assert len(np.argwhere(state == adsorbate_idx)) <= 1, "more than 1 site found"
    assert len(np.argwhere(state == adsorbate_idx)) == 1, "no sites found"

    del slab[int(adsorbate_idx)]

    # lower the index for higher index items
    state = np.where(state >= int(adsorbate_idx), state - 1, state)
    # remove negatives
    state = np.where(state < 0, 0, state)

    # remove the adsorbate from tracking
    state[site_idx] = 0
    return state, slab


def get_adsorption_coords(slab, atom, connectivity, debug=False):
    """Takes a slab, an atom, and a list of site indices, and returns the actual coordinates of the
    adsorbed atoms

    Parameters
    ----------
    slab : ase.Atoms
        the original slab
    atom : ase.Atoms
        the atom you want to adsorb
    connectivity : list
        list of lists of integers, each list is a list of the indices of the atoms that are connected to
    the atom at the index of the list.

    Returns
    -------
        The positions of the adsorbed atoms.

    """
    logger.debug(f"getting actual adsorption site coordinates")
    new_slab = slab.copy()

    proposed_slab_builder = catkit.gen.adsorption.Builder(new_slab)

    # add multiple adsorbates
    site_indices = list(range(len(connectivity)))

    # use proposed_slab_builder._single_adsorption multiple times
    for i, index in enumerate(site_indices):
        new_slab = proposed_slab_builder._single_adsorption(
            atom,
            bond=0,
            slab=new_slab,
            site_index=site_indices[i],
            auto_construct=False,
            symmetric=False,
        )

    if debug:
        write(f"ads_{str(atom.symbols)}_all_adsorbed_slab.cif", new_slab)

    # store the actual positions of the sides
    logger.debug(
        f"new slab has {len(new_slab)} atoms and original slab has {len(slab)} atoms."
    )

    return new_slab.get_positions(wrap=True)[len(slab) :]


def count_adsorption_sites(slab, state, connectivity):
    """It takes a slab, a state, and a connectivity matrix, and returns a dictionary of the number of
    adsorption sites of each type

    Parameters
    ----------
    slab
        ase.Atoms object
    state
        a list of the same length as the number of sites in the slab.
    connectivity
        a list of the same length as the number of sites in the slab.

    Returns
    -------
        A dictionary with the number of adsorption sites as keys and the number of atoms with that number
    of adsorption sites as values.

    """
    occ_idx = state > 0
    return Counter(connectivity[occ_idx])
