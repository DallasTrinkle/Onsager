"""
Crystal class

Class to store definition of a crystal, along with some analysis
1. geometric analysis (nearest neighbor displacements)
2. space group operations
3. point group operations for each basis position
4. Wyckoff position generation (for interstitials)
"""

__author__ = 'Dallas R. Trinkle'

import numpy as np

def incell(vec):
    """
    Returns the vector inside the unit cell (in [0,1)**3)
    """
    return vec - np.floor(vec)

def inhalf(vec):
    """
    Returns the vector inside the centered cell (in [-0.5,0.5)**3)
    """
    return vec - np.rint(vec)

class Crystal(object):
    """
    A class that defines a crystal, as well as the symmetry analysis that goes along with it.
    """

    def __init__(self, lattice, basis):
        """
        Initialization; starts off with the lattice vector definition and the
        basis vectors. While it does not explicitly store the specific chemical
        elements involved, it does store that there are different elements.

        Parameters
        ----------
        lattice : array[3,3] or list of array[3]
            lattice vectors; if [3,3] array, then the vectors need to be in *column* format
            so that the first lattice vector is lattice[:,0]

        basis : list of array[3] or list of list of array[3]
            crystalline basis vectors, in unit cell coordinates. If a list of lists, then
            there are multiple chemical elements, with each list corresponding to a unique
            element
        """
        # Do some basic type checking and "formatting"
        self.lattice = None
        if type(lattice) == list:
            assert len(lattice) == 3, "lattice is a list, but does not contain 3 members"
            self.lattice = np.array(lattice).T
        if type(lattice) == np.ndarray:
            self.lattice = lattice
        assert self.lattice is not None, "lattice is not a recognized type"
        assert self.lattice.shape == (3,3), "lattice contains vectors that are not 3 dimensional"
        assert type(basis) is list, "basis needs to be a list or list of lists"
        if type(basis[0]) == np.ndarray:
            for u in basis:
                assert type(u) is np.ndarray, "{} in {} is not an array".format(u, basis)
            self.basis = [ [incell(u) for u in basis] ]
        else:
            for elem in basis:
                assert type(elem) is list, "{} in basis is not a list".format(elem)
                for u in elem:
                    assert type(u) is np.ndarray, "{} in {} is not an array".format(u, elem)
            self.basis = [ [ incell(u) for u in atombasis] for atombasis in basis]
        self.reduce()
        self.calcmetric()

    def reduce(self):
        """
        Reduces the lattice and basis, if needed. Works recursively.
        """
        # Work with the shortest possible list first
        maxlen = 0
        atomindex = 0
        for i, ulist in enumerate(self.basis):
            if len(ulist) > maxlen:
                maxlen = len(ulist)
                atomindex = i
        if maxlen == 1:
            return
        # We need to first check against reducibility of atomic positions: try out non-trivial displacements
        initpos = self.basis[atomindex][0]
        for newpos in self.basis[atomindex]:
            t = newpos - initpos
            if np.all(t == 0): continue
            trans = True
            for atomlist in self.basis:
                for u in atomlist:
                    if np.all([ not np.all(np.isclose(inhalf(u+t-v), 0)) for v in atomlist]):
                        trans = False
                        break
            if trans:
                break
        if not trans:
            return
        # reduce that lattice and basis
        # 1. determine what the new lattice needs to look like.
        for d in xrange(3):
            super = np.eye(3)
            super[:,d] = t[:]
            if np.linalg.det(super) != 0:
                break
        invsuper = np.linalg.inv(super)
        self.lattice = np.dot(self.lattice, super)
        # 2. update the basis
        newbasis = []
        for atomlist in self.basis:
            newatomlist = []
            for u in atomlist:
                v = incell(np.dot(invsuper, u))
                if np.all([ not np.all(np.isclose(v, v1)) for v1 in newatomlist]):
                    newatomlist.append(v)
            newbasis.append(newatomlist)
        self.basis = newbasis
        self.reduce()

    def calcmetric(self):
        """
        Computes the volume of the cell and the metric tensor
        """
        self.volume = abs(np.linalg.det(self.lattice))
        self.metric = np.dot(self.lattice.T, self.lattice)
