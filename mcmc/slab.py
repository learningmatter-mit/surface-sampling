import itertools
import logging
import random
from collections import Counter
from typing import Union

import ase
import catkit
import numpy as np
from ase.build import bulk
from ase.io import write
from scipy.special import softmax

from mcmc.system import SurfaceSystem
from mcmc.utils.misc import plot_specific_weights

logger = logging.getLogger(__name__)


def initialize_slab(
    alat: float,
    elem: str = "Cu",
    vacuum: float = 15.0,
    miller: tuple[int] = (1, 0, 0),
    termination: int = 0,
    orthogonal: bool = False,
    size: tuple[int] = (4, 4, 4),
    **kwargs,
) -> ase.Atoms:
    """Creates the slab structure using ASE.

    Args:
        alat (float): The lattice constant in angstroms.
        elem (str): The element to use.
        vacuum (float): The vacuum thickness.
        miller (tuple): The Miller indices.
        termination (int): The termination.
        orthogonal (bool): Whether to use orthogonal coordinates.
        size (tuple): The size of the slab in each dimension.

    Returns:
        ase.Atoms: The slab structure.
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


def get_complementary_idx(
    surface: SurfaceSystem,
    require_per_atom_energies: bool = False,
    require_distance_decay: bool = False,
    **kwargs,
) -> tuple[int, int, str, str]:
    """Get two indices, site1 and site2 of different elemental identities.

    Args:
        surface (SurfaceSystem): The surface system to propose changes to.
        require_per_atom_energies (bool): Whether to use per atom energies.
        require_distance_decay (bool): Whether to use distance decay.

    Returns:
        Tuple[int, int, str, str]: A tuple containing the indices of the two sites and the elemental identities of the adsorbates.
    """
    # TODO: split into two functions
    # find indices of adsorbed atoms
    adsorbed_idx = np.argwhere(surface.occ != 0).flatten()

    per_atom_energies = kwargs.get("per_atom_energies", None)
    distance_weight_matrix = kwargs.get("distance_weight_matrix", None)
    # logger.info(f"distance weight matrix has shape {distance_weight_matrix.shape}")
    # group adsorbates present in slab
    curr_ads = {
        k: list(g)
        for k, g in itertools.groupby(
            adsorbed_idx, key=lambda x: surface.real_atoms[surface.occ[x]].symbol
        )
    }
    # add empty sites
    # find indices of empty sites
    empty_idx = np.argwhere(surface.occ == 0).flatten().tolist()
    curr_ads["None"] = empty_idx
    # complelety populate the adsorbates including empty sites
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

        # create weight dict for each adsorbate except empty sites
        weights = {
            k: boltzmann_weights[surface.occ[v]] if k != "None" else np.ones_like(v)
            for k, v in curr_ads.items()
        }
    else:
        # all uniform weights
        weights = {k: np.ones_like(v) for k, v in curr_ads.items()}

    # choose two types
    types = list(curr_ads.keys())
    type1, type2 = random.sample(types, 2)

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
        logger.debug("specific weights shape is %s", specific_distance_weights.shape)
        if kwargs.get("plot_weights", False):
            logger.debug("plotting weights")
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
    surface: SurfaceSystem, site_idx: int, end_ads: Union[str, ase.Atoms]
) -> SurfaceSystem:
    """Change the adsorbate at a site to a new adsorbate.

    Args:
        surface (SurfaceSystem): The surface system to propose changes to.
        site_idx (int): The index of the site to change.
        end_ads (Union[str, ase.Atoms]): The new adsorbate to place at the site.

    Returns:
        SurfaceSystem: The updated surface system
    """
    logger.debug("current slab has %s atoms", len(surface))

    if site_idx >= len(surface.occ):
        raise IndexError("site index out of range")

    if surface.occ[site_idx] != 0:
        logger.debug("chosen site already adsorbed")
        start_ads = surface.real_atoms[surface.occ[site_idx]].symbol
        # desorb first, regardless of next chosen state
        surface = remove_atom(surface, site_idx)

    else:
        logger.debug("chosen site is empty")
        start_ads = "None"
        # modularize

    if end_ads != "None":
        logger.debug("replacing %s with %s", start_ads, end_ads)
        surface = add_atom(surface, site_idx, end_ads)
    else:
        logger.debug("desorbing %s", start_ads)

    logger.debug("proposed slab has %s atoms", len(surface))
    return surface


def add_atom(
    surface: SurfaceSystem, site_idx: int, adsorbate: Union[str, ase.Atoms]
) -> SurfaceSystem:
    """Add an adsorbate at an empty site and updates the state.

    Args:
        surface (SurfaceSystem): The surface system.
        site_idx (int): The index of the site to change.
        adsorbate (Union[str, ase.Atoms]): The adsorbate to add.

    Returns:
        SurfaceSystem: The updated surface system
    """

    adsorbate_idx = len(surface)
    surface.occ[site_idx] = adsorbate_idx
    surface.real_atoms.append(adsorbate)
    surface.real_atoms.positions[-1] = surface.ads_coords[site_idx]
    return surface


def remove_atom(surface: SurfaceSystem, site_idx: int) -> SurfaceSystem:
    """Remove an adsorbate from the slab and updates the state.

    Args:
        surface (SurfaceSystem): The surface system.
        site_idx (int): The index of the site to change.

    Returns:
        SurfaceSystem: The updated surface system
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


def count_adsorption_sites(
    surface: SurfaceSystem, connectivity: Union[list, np.ndarray]
) -> Counter:
    """Count the number of adsorption sites with a given number of adsorbates.

    Args:
        surface (SurfaceSystem): The surface system.
        connectivity (Union[list, np.ndarray]): The connectivity matrix.

    Returns:
        Counter: The number of adsorption sites with a given number of adsorbates.
    """
    occ_idx = surface.occ > 0
    return Counter(connectivity[occ_idx])
