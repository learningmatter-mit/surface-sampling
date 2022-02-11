# Cu alat from https://www.copper.org/resources/properties/atomic_properties.html
Cu_alat = 3.6147
slab = initialize_slab(Cu_alat)

# get ALL the adsorption sites
# top should have connectivity 1, bridge should be 2 and hollow more like 4
coords, connectivity, sym_idx = get_adsorption_sites(slab, symmetry_reduced=False)
print(f"In pristine slab, there are a total of {len(connectivity)} sites")

# state of each vacancy in slab. for state > 0, it's filled, and that's the index of the adsorbate atom in slab 
state = np.zeros(len(coords), dtype=int)

temp = 300
pot = 2

# perform 5 test iterations
site1 = len(state)-1
site2 = len(state)-2
# add to two sites
state, slab, energy_diff, mag_diff = spin_flip(state, slab, temp, pot, save_cif=True, iter=1, site_idx=site1, testing=True)
state, slab, energy_diff, mag_diff = spin_flip(state, slab, temp, pot, save_cif=True, iter=2, site_idx=site2, testing=True)

# remove from 1st site
state, slab, energy_diff, mag_diff = spin_flip(state, slab, temp, pot, save_cif=True, iter=3, site_idx=site1, testing=True)
# remove from 2nd site
state, slab, energy_diff, mag_diff = spin_flip(state, slab, temp, pot, save_cif=True, iter=4, site_idx=site2, testing=True)

# add to 1st site again
state, slab, energy_diff, mag_diff = spin_flip(state, slab, temp, pot, save_cif=True, iter=5, site_idx=site2, testing=True