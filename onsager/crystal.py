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
            for v in basis:
                assert type(v) is np.ndarray, "{} in {} is not an array".format(v, basis)
            self.basis = [basis]
        else:
            for elem in basis:
                assert type(elem) is list, "{} in basis is not a list".format(elem)
                for v in elem:
                    assert type(v) is np.ndarray, "{} in {} is not an array".format(v, elem)
            self.basis = basis
        self.reduce()
        self.calcmetric()

    def reduce(self):
        """
        Reduces the lattice and basis, if needed.
        """
        return

    def calcmetric(self):
        """
        Computes the volume of the cell and the metric tensor
        """
        self.volume = abs(np.linalg.det(self.lattice))
        self.metric = np.dot(self.lattice.T, self.lattice)
