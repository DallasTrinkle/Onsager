"""
KPTmesh module

Class definition for KPTmesh class; allows automated construction of kpt meshes
"""

import numpy as np

class KPTmesh:
    """
    A class to construct (symmetrized, reduced to the irreducible wedge) k-point meshes.
    """
    def __init__(self, lattice):
        """
        Creates an instance of a k-point mesh generator.
        
        Parameters
        ----------
        lattice : array [3,3}
            lattice vectors, in *column* form, so that a1 = a[:,0], a2 = a[:,1], a3 = a[:,2]
        """
        self.lattice = lattice
        self.volume = np.linalg.det(lattice)
        self.rlattice = 2.*np.pi*(np.linalg.inv(lattice)).T
        self.Nmesh = (0,0,0)
        self.Nkpt = 0
        self.__genBZG()

    def genmesh(self, N):
        """
        Initiates, if mesh doesn't already exist, the construction of a mesh of size N.
        
        Parameters
        ----------
        N : list
            should have length 3; specifies number of divisions in 1, 2, 3 directions.
        """
        self.Nmesh = N
        self.Nkpt = np.product(N)

    def __genBZG(self):
        """
        Generates the reciprocal lattice G points that define the Brillouin zone.
        """
        BZG = [[0,0,0], [1,1,1]]
        self.BZG = np.array(BZG)
        