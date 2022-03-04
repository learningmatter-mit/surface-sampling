from catkit import Gratoms

import sys
import os

sys.path.append("/home/dux/")
from htvs.djangochem.pgmols.utils import surfaces


from ase.io import read

atoms = read('Au_mp-81_conventional_standard.cif')

slab, surface_atoms = surfaces.surface_from_bulk(atoms, [1,0,0], size=[4,4])

slab.write('Au_surface_100.cif')
