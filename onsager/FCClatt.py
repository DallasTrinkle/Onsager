"""
FCClatt module

Generates a neighbor vector list for an FCC lattice, and constructs the list of
inverse indices.
"""

__author__ = 'Dallas R. Trinkle'

import numpy as np


def lattice():
    """
    Constructs the FCC lattice vectors.

    Returns
    -------
    a : array [:, :]
        lattice vectors; a[:, i] = cartesian coordinates of vector a_i
    """
    return np.array(((0, 1, 1),
                     (1, 0, 1),
                     (1, 1, 0)),
                    dtype=float)

def NNvect():
    """
    Constructs the FCC <110> transition vectors.

    Returns
    -------
    NNvect : array [12, 3]
        nearest neighbor vectors
    """
    return np.array([(n0, n1, n2)
                     for n0 in xrange(-1, 2)
                     for n1 in xrange(-1, 2)
                     for n2 in xrange(-1, 2)
                     if (n0, n1, n2) != (0, 0, 0) and (n0 + n1 + n2) % 2 == 0],
                    dtype=float)


def invlist(NNvect):
    """
    Constructs the list of inverse vectors for each vector.

    Parameters
    ----------
    nnlist : array [:,:]
        list of nearest neighbor vectors; nnlist[k,:] = is the kth nearest neighbor

    Returns
    -------
    invlist : int array [:]
        for each k1, `nnlist`[k1] = -`nnlist`[`invlist`[k1]]
    """
    z = np.shape(NNvect)[0]
    invlist = np.empty((z), dtype=int)
    for k1 in xrange(z):
        for k2 in xrange(z):
            if all((NNvect[k1] + NNvect[k2]) == (0, 0, 0)):
                invlist[k1] = k2
    return invlist

