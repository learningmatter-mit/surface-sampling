"""Functions for manipulating the surface system and adsorbates."""

import itertools
import logging
import random
from collections import Counter

import ase
import numpy as np
from scipy.special import softmax

from mcmc.system import SurfaceSystem
from mcmc.utils.misc import plot_specific_weights

logger = logging.getLogger(__name__)


# TODO move prepare_canonical to here


def get_adsorbate_indices(surface: SurfaceSystem) -> dict[str, list[int]]:
    """Get a dictionary of adsorbate indices grouped by elemental identity.

    Args:
        surface (SurfaceSystem): The surface system.

    Returns:
        dict: A dictionary of adsorbate indices grouped by elemental identity.
    """
    adsorbed_idx = surface.filled_occ_idx
    # Group adsorbates present in slab
    adsorbates = {
        k: list(g)
        for k, g in itertools.groupby(
            adsorbed_idx, key=lambda x: surface.real_atoms[surface.occ[x]].symbol
        )
    }
    # Add virtual sites only if there are empty ones
    empty_idx = surface.empty_occ_idx
    if len(empty_idx) > 0:
        adsorbates["None"] = empty_idx

    return adsorbates


def choose_adsorbate_type(adsorbates: dict) -> tuple[str, str]:
    """Choose two types of adsorbates from the given dictionary.

    Args:
        adsorbates (dict): A dictionary of adsorbate indices grouped by elemental identity.

    Returns:
        tuple: A tuple containing the two chosen adsorbate types.
    """
    types = list(adsorbates.keys())
    type1, type2 = random.sample(types, 2)
    return type1, type2


def compute_boltzmann_weights(
    surface: SurfaceSystem,
    temperature: float,
    curr_ads: dict[str, list],
) -> dict[str, np.ndarray]:
    """Compute the Boltzmann weights for each adsorbate type.

    Args:
        surface (SurfaceSystem): The surface system.
        temperature (float): The temperature in eV.
        curr_ads (dict): A dictionary of adsorbate indices grouped by elemental identity.

    Returns:
        dict: A dictionary containing the Boltzmann weights for each adsorbate type.

    Raises:
        ValueError: If require_per_atom_energies is True, but no per_atom_energies were provided.
    """
    per_atom_energies = surface.calc.results.get("per_atom_energies", [])
    # TODO write a dedicated surface.get_per_atom_energies() method

    if len(per_atom_energies) == 0:
        raise ValueError(
            "require_per_atom_energies is True, but no per_atom_energies were provided"
        )
    assert len(per_atom_energies) == len(
        surface.real_atoms
    ), "length mismatch in per atom energies and number of real atoms"
    logger.debug("in `get_complementary_idx` using per atom energies")
    logger.debug("per atom energies are %s", per_atom_energies)
    logger.debug("temperature is %s", temperature)
    boltzmann_weights = softmax(np.array(per_atom_energies) / temperature)
    logger.debug("boltzmann weights are %s", boltzmann_weights)

    # create weight dict for each adsorbate except empty sites
    return {
        k: boltzmann_weights[surface.occ[v]] if k != "None" else np.ones_like(v)
        for k, v in curr_ads.items()
    }


def get_complementary_idx_distance_decay(
    surface: SurfaceSystem,
    curr_ads: dict[str, list],
    type1: str,
    type2: str,
    weights1: np.ndarray,
    weights2: np.ndarray,
    plot_weights: bool,
    run_folder: str,
    run_iter: int,
) -> tuple[int, int]:
    """Get two indices, site1 and site2 of different elemental identities using distance decay.

    Args:
        surface (SurfaceSystem): The surface system.
        curr_ads (dict): A dictionary of adsorbate indices grouped by elemental identity.
        type1 (str): The first adsorbate type.
        type2 (str): The second adsorbate type.
        weights1 (np.ndarray): The weights for the first adsorbate type.
        weights2 (np.ndarray): The weights for the second adsorbate type.
        plot_weights (bool): Whether to plot the specific distance weights.
        run_folder (str): The folder to save the plots in.
        run_iter (int): The iteration number.

    Returns:
        Tuple[int, int]: A tuple containing the indices of the two sites.

    Raises:
        ValueError: If require_distance_decay is True, but no distance_weight_matrix was provided.
    """
    if surface.distance_weight_matrix is None:
        raise ValueError(
            "require_distance_decay is True, but no distance_weight_matrix was provided"
        )

    # TODO: run_folder and run_iter should be saved to the SurfaceSystem
    logger.debug("Length of curr_ads[type1]: %s", len(curr_ads[type1]))
    logger.debug("Length of weights1: %s", len(weights1))
    site1_idx = random.choices(curr_ads[type1], weights=weights1, k=1)[0]
    specific_distance_weights = surface.distance_weight_matrix[site1_idx]
    if plot_weights:
        plot_specific_weights(
            surface.ads_coords,
            specific_distance_weights,
            site1_idx,
            save_folder=run_folder,
            run_iter=run_iter,
        )
    combined_type2_weights = weights2 * specific_distance_weights[curr_ads[type2]]
    site2_idx = random.choices(curr_ads[type2], weights=combined_type2_weights, k=1)[0]
    return site1_idx, site2_idx


def get_complementary_idx(
    surface: SurfaceSystem,
    require_per_atom_energies: bool = False,
    require_distance_decay: bool = False,
    temperature: float = 1.0,
    plot_weights: bool = False,
    run_folder: str = ".",
    run_iter: int = 0,
) -> tuple[int, int, str, str]:
    """Get two indices, site1 and site2 of different elemental identities.

    Args:
        surface (SurfaceSystem): The surface system to propose changes to.
        require_per_atom_energies (bool): Whether to use per atom energies.
        require_distance_decay (bool): Whether to use distance decay.
        temperature (float): The temperature in eV.
        plot_weights (bool): Whether to plot the specific distance weights.
        run_folder (str): The folder to save the plots in.
        run_iter (int): The iteration number.

    Returns:
        Tuple[int, int, str, str]: A tuple containing the indices of the two sites and the
            elemental identities of the adsorbates.
    """
    curr_ads = get_adsorbate_indices(surface)
    logger.debug("current ads %s", curr_ads)

    if require_per_atom_energies:
        weights = compute_boltzmann_weights(surface, temperature, curr_ads)
    else:
        # Uniform weights
        weights = {k: np.ones_like(v) for k, v in curr_ads.items()}

    # Randomly pick two adsorbate types
    type1, type2 = choose_adsorbate_type(curr_ads)
    weights1, weights2 = weights[type1], weights[type2]
    # Checking if the weights are valid, and if not, replace them with an array of ones.
    # weights1 = (
    #     weights[type1] if weights[type1].any() > 0 else np.ones(len(curr_ads[type1]))
    # )
    # weights2 = (
    #     weights[type2] if weights[type2].any() > 0 else np.ones(len(curr_ads[type2]))
    # )

    if require_distance_decay:
        site1_idx, site2_idx = get_complementary_idx_distance_decay(
            surface,
            curr_ads,
            type1,
            type2,
            weights1,
            weights2,
            plot_weights,
            run_folder,
            run_iter,
        )
    else:
        # get random idx belonging to those types
        site1_idx, site2_idx = (
            random.choices(curr_ads[x], weights=w, k=1)[0]
            for x, w in zip([type1, type2], [weights1, weights2], strict=False)
        )

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
    site_idx: int,
    end_ads: str | ase.Atoms,
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


def add_atom(surface: SurfaceSystem, site_idx: int, adsorbate: str | ase.Atoms) -> SurfaceSystem:
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
    surface.occ = np.where(surface.occ >= int(adsorbate_idx), surface.occ - 1, surface.occ)
    # remove negatives
    surface.occ = np.where(surface.occ < 0, 0, surface.occ)

    # remove the adsorbate from tracking
    surface.occ[site_idx] = 0
    return surface


def count_adsorption_sites(surface: SurfaceSystem, connectivity: list | np.ndarray) -> Counter:
    """Count the number of adsorption sites with a given number of adsorbates.

    Args:
        surface (SurfaceSystem): The surface system.
        connectivity (Union[list, np.ndarray]): The connectivity matrix.

    Returns:
        Counter: The number of adsorption sites with a given number of adsorbates.
    """
    occ_idx = surface.occ > 0
    return Counter(connectivity[occ_idx])
