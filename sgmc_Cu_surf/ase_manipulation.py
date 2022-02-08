
def shift_slab(self, destination='cell_bottom'):
    """Dumty.q

    Parameters
    ----------
    destination : str, optional
        either 'cell_bottom' or 'cell_middle', by default 'cell_bottom'

    Returns
    -------
    Atoms
        Atoms object returned
    """
        slab = self.gratoms.copy()
        zmin = self.cell_length

        for atom in slab:
            if atom.position[2] < zmin:
                zmin = atom.position[2]

        if destination == 'cell_bottom':
            slab.translate([0, 0, - zmin + 2])
        elif destination == 'cell_middle':
            slab.translate([0, 0, self.vacuum - zmin])
        else:
            logger.warning('Destination not recognised. Returning original slab')
        return slab


def determine_vacuum(self):
    """Determines vacuum spacing for one side in ASE Atoms objects.

    Returns
    -------
    float
        vacuum spacing in angstroms
    """
    slab = self.gratoms.copy()
    zmin = zmax = slab[0].position[2]
    for atom in slab:
        if atom.position[2] > zmax:
            zmax = atom.position[2]
        elif atom.position[2] < zmin:
            zmin = atom.position[2]
    slab_thickness = zmax - zmin
    vacuum = (self.cell_length - slab_thickness) / 2
    return vacuum