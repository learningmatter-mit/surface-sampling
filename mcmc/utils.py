import numpy as np


def filter_distances(slab, ads=["O"], cutoff_distance: float = 1.5):
    """This function filters out slabs that have atoms too close to each other based on a
    specified cutoff distance.

    Parameters
    ----------
    slab : ase.atoms.Atoms or catkit.gratoms.Gratoms or AtomsBatch
        The `slab` is the surface structure
    ads
        The ads parameter is a list of chemical symbols representing the atoms that are being checked
        for their distances from each other. By default, it is set to ["O"], which means that the function
        will check the distances of oxygen atoms from each other.
    cutoff_distance
        The minimum distance allowed between any two atoms in the slab. If any two adsorbate
        atoms are closer than this distance, the function will return False.

    Returns
    -------
        a boolean value. It returns True if all distances between the specified adsorbates in the given
    slab are greater than the specified cutoff distance, and False otherwise.

    """
    # Checks distances of all adsorbates are greater than cutoff
    ads_arr = np.isin(slab.get_chemical_symbols(), ads)
    unique_dists = np.unique(
        np.triu(slab.get_all_distances(mic=True)[ads_arr][:, ads_arr])
    )
    # get upper triangular matrix of ads dist
    if any(unique_dists[(unique_dists > 0) & (unique_dists <= cutoff_distance)]):
        return False  # fail because atoms are too close
    return True
