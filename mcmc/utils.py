import numpy as np


def filter_distances(slab, pristine_len=0, cutoff_distance=1.5):
    # OXYGEN_CUTOFF_DIST = 1.5
    # Checks distances of all adsorbates are greater than cutoff
    all_dist = slab.get_all_distances(mic=True)
    unique_dist = np.unique(
        np.triu(all_dist[pristine_len:, pristine_len:])
    )  # get upper triangular matrix of ads dist
    if any(unique_dist[(unique_dist > 0) & (unique_dist <= cutoff_distance)]):
        return False  # fail because oxygens are too close
    return True


def filter_distances_new(slab, ads=["O"], cutoff_distance=1.5):
    # OXYGEN_CUTOFF_DIST = 1.5
    # Checks distances of all adsorbates are greater than cutoff
    ads_arr = np.isin(slab.get_chemical_symbols(), ads)
    unique_dists = np.unique(
        np.triu(slab.get_all_distances(mic=True)[ads_arr][:, ads_arr])
    )
    # get upper triangular matrix of ads dist
    if any(unique_dists[(unique_dists > 0) & (unique_dists <= cutoff_distance)]):
        return False  # fail because oxygens are too close
    return True
