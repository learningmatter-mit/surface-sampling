import itertools
import json
import logging
import random
from collections import Counter

import catkit
import numpy as np
from ase.build import bulk
from ase.io import write
from scipy.special import softmax

from mcmc.system import SurfaceSystem
from mcmc.utils import plot_specific_weights

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


def get_complementary_idx(
    surface: SurfaceSystem,
    require_per_atom_energies=False,
    require_distance_decay=False,
    **kwargs,
):
    """Get two indices, site1 and site2 of different elemental identities."""
    adsorbed_idx = np.argwhere(surface.occ != 0).flatten()

    # TODO use per_site energies
    # optimized_slab, _ = optimize_slab(
    #     self.slab,
    #     optimizer=self.kwargs["optimizer"],

    #     folder_name=self.run_folder,
    # )
    # optimized_slab.write(
    #     f"{self.run_folder}/optim_slab_run_idx_{run_idx:06}_{optimized_slab.get_chemical_formula()}_energy_{optimized_slab.get_potential_energy():.3f}.cif"
    # )
    per_atom_energies = kwargs.get("per_atom_energies", None)
    distance_weight_matrix = kwargs.get("distance_weight_matrix", None)
    # logger.info(f"distance weight matrix has shape {distance_weight_matrix.shape}")
    # select adsorbates present in slab
    curr_ads = {
        k: list(g)
        for k, g in itertools.groupby(
            adsorbed_idx, key=lambda x: surface.real_atoms[surface.occ[x]].symbol
        )
    }
    # add empty sites
    empty_idx = np.argwhere(surface.occ == 0).flatten().tolist()
    curr_ads["None"] = empty_idx
    logger.debug(f"current ads {curr_ads}")

    if require_per_atom_energies:
        if per_atom_energies is None:
            raise ValueError(
                "require_per_atom_energies is True, but no per_atom_energies were provided"
            )
        logger.debug("in `get_complementary_idx` using per atom energies")
        logger.debug("per atom energies are %s", per_atom_energies)
        # TODO might want to change the "temperature"
        # temp = kwargs.get("temp", 0.5)  # in terms of eV
        temp = 1  # fixed at 1 eV
        logger.debug("temp is %s", temp)
        boltzmann_weights = softmax(per_atom_energies / temp)
        logger.debug("boltzmann weights are %s", boltzmann_weights)
        # creat weights for each adsorbate except empty sites
        weights = {
            k: boltzmann_weights[surface.occ[v]] if k != "None" else np.ones_like(v)
            for k, v in curr_ads.items()
        }
    else:
        # all uniform weights
        weights = {k: np.ones_like(v) for k, v in curr_ads.items()}
    # choose two types
    type1, type2 = random.sample(curr_ads.keys(), 2)

    # Checking if the weights are valid, and if not, replace them with an array of ones.
    weights1 = (
        weights[type1] if weights[type1].any() > 0 else np.ones(len(curr_ads[type1]))
    )
    weights2 = (
        weights[type2] if weights[type2].any() > 0 else np.ones(len(curr_ads[type2]))
    )

    if require_distance_decay:
        # TODO merge with per atom energies
        if distance_weight_matrix is None:
            raise ValueError(
                "require_distance_decay is True, but no distance_weight_matrix was provided"
            )
        logger.debug("in `get_complementary_idx` using distance decay")
        # get random idx for type 1 first
        # weights_type1 = np.ones(len(curr_ads[type1]))
        # weights_type1_new = weights[type1] if weights[type1].any() > 0 else np.ones(len(curr_ads[type1]))
        site1_idx = random.choices(curr_ads[type1], weights=weights1, k=1)[
            0
        ]  # even weights

        ads_coords = kwargs.get("ads_coords", None)
        specific_distance_weights = distance_weight_matrix[
            site1_idx
        ]  # get the weights for the second type
        logger.debug(f"specific weights shape is {specific_distance_weights.shape}")
        if kwargs.get("plot_weights", False):
            logger.debug(f"plotting weights")
            plot_specific_weights(
                ads_coords,
                specific_distance_weights,
                site1_idx,
                save_folder=kwargs.get("run_folder", "."),
                run_iter=kwargs.get("run_iter", 0),
            )
        combined_type2_weights = (
            weights2 * specific_distance_weights[curr_ads[type2]]
        )  # energy based weights * distance decay weights
        site2_idx = random.choices(
            curr_ads[type2], weights=combined_type2_weights, k=1
        )[
            0
        ]  # weighted by distance decay
    else:
        # get random idx belonging to those types
        site1_idx, site2_idx = [
            random.choices(curr_ads[x], weights=w, k=1)[0]
            for x, w in zip([type1, type2], [weights1, weights2])
        ]
    slab_idx_1, slab_idx_2 = surface.occ[site1_idx], surface.occ[site2_idx]
    logger.debug("type1 %s, type2 %s", type1, type2)
    logger.debug("site1_idx %s, site2_idx %s", slab_idx_1, slab_idx_2)
    logger.debug(
        "coordinates are %s",
        surface.real_atoms.get_positions(wrap=True)[[slab_idx_1, slab_idx_2]],
    )

    return site1_idx, site2_idx, type1, type2


def change_site(
    surface: SurfaceSystem,
    pots,
    adsorbates,
    site_idx,
    start_ads=None,
    end_ads=None,
    **kwargs,
):
    """The `change_site` function takes in various parameters related to a surface slab and adsorbates, and
    performs operations to change the adsorption state of a specific site on the slab.

    Parameters
    ----------
    slab
        The `slab` parameter represents the slab structure on which the adsorption/desorption is being
    performed. It is typically a `pymatgen` `Structure` object.
    state
        The `state` parameter is a list that represents the current state of the adsorption sites on the
    slab. Each element in the list corresponds to a site on the slab, and the value of the element
    indicates the adsorbate occupying that site. A value of 0 indicates an empty site.
    pots
        The `pots` parameter is a list of potentials corresponding to each adsorbate in the `adsorbates`
    list. It is used to assign a potential to each adsorbate when adsorbing it onto the slab.
    adsorbates
        A list of adsorbate species that can be adsorbed onto the slab surface.
    site_idx
        The `site_idx` parameter represents the index of the site on the slab where the adsorption or
    desorption will take place.
    start_ads
        The `start_ads` parameter is used to specify the initial adsorbate on the site before the function
    is called. It is an optional parameter and its default value is `None`.
    end_ads
        The `end_ads` parameter is used to specify the adsorbate that will be adsorbed on the chosen site.
    If `end_ads` is not provided, a random adsorbate will be chosen from the available options.

    Returns
    -------
        The function `change_site` returns multiple values: `slab`, `state`, `delta_pot`, `start_ads`, and
    `end_ads`.

    """
    ads_pot_dict = dict(zip(adsorbates, pots))
    chosen_ads = None

    old_ads_count = Counter(surface.real_atoms.get_chemical_symbols())

    if surface.occ[site_idx] == 0:  # empty list, no ads
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
        logger.debug(f"current slab has {len(surface)} atoms")

        surface = add_to_slab(surface, chosen_ads, site_idx)

        logger.debug(f"proposed slab has {len(surface)} atoms")

    else:
        logger.debug(f"chosen site already adsorbed")
        if not start_ads:
            start_ads = surface.real_atoms[surface.occ[site_idx]].symbol

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

        logger.debug(f"current slab has {len(surface)} atoms")

        # desorb first, regardless of next chosen state
        surface = remove_from_slab(surface, site_idx)

        # adsorb
        if "None" not in chosen_ads:
            chosen_pot = ads_pot_dict[chosen_ads]

            logger.debug(f"replacing {start_ads} with {chosen_ads}")

            surface = add_to_slab(surface, chosen_ads, site_idx)

            delta_pot = chosen_pot - prev_pot
        else:
            logger.debug(f"desorbing {start_ads}")
            delta_pot = -prev_pot  # vacant site has pot = 0

        logger.debug(f"proposed slab has {len(surface)} atoms")

    end_ads = chosen_ads

    # new_ads_count = Counter(surface.real_atoms.get_chemical_symbols())

    # if kwargs.get("offset_data", None):
    #     with open(kwargs["offset_data"]) as f:
    #         offset_data = json.load(f)
    # stoics = offset_data["stoics"]
    # ref_element = offset_data["ref_element"]

    # old_pot = 0
    # new_pot = 0
    # for ele, _ in old_ads_count.items():
    #     if ele != ref_element:
    #         old_pot += (
    #             old_ads_count[ele]
    #             - stoics[ele] / stoics[ref_element] * old_ads_count[ref_element]
    #         ) * ads_pot_dict[ele]
    #         new_pot += (
    #             new_ads_count[ele]
    #             - stoics[ele] / stoics[ref_element] * new_ads_count[ref_element]
    #         ) * ads_pot_dict[ele]

    # delta_pot = new_pot - old_pot

    return surface, delta_pot, start_ads, end_ads


def add_to_slab(surface, adsorbate, site_idx):
    """It adds an adsorbate to a surface, and updates the state to reflect the new adsorbate

    Parameters
    ----------
    surface : SurfaceSystem
        the surface system
    site is empty, the integer is 0.
    adsorbate : ase.Atoms
        the adsorbate molecule
    site_idx : int
        the index of the site on the slab where the adsorbate will be placed

    Returns
    -------
        The state and slab are being returned.
    """

    adsorbate_idx = len(surface)
    surface.occ[site_idx] = adsorbate_idx
    surface.real_atoms.append(adsorbate)
    surface.real_atoms.positions[-1] = surface.ads_coords[site_idx]
    return surface


def remove_from_slab(surface, site_idx):
    """Remove the adsorbate from the slab and update the state

    Parameters
    ----------
    surface : SurfaceSystem
        the surface system
    site_idx : int
        the index of the site to remove the adsorbate from

    Returns
    -------
        The surface being returned.
    """
    adsorbate_idx = surface.occ[site_idx]
    assert len(np.argwhere(surface.occ == adsorbate_idx)) <= 1, "more than 1 site found"
    assert len(np.argwhere(surface.occ == adsorbate_idx)) == 1, "no sites found"

    del surface.real_atoms[int(adsorbate_idx)]

    # lower the index for higher index items
    surface.occ = np.where(
        surface.occ >= int(adsorbate_idx), surface.occ - 1, surface.occ
    )
    # remove negatives
    surface.occ = np.where(surface.occ < 0, 0, surface.occ)

    # remove the adsorbate from tracking
    surface.occ[site_idx] = 0
    return surface


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
