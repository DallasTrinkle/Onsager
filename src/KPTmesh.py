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
        self.genBZG()

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

    def incell(self, BZG, vec):
        """
        Tells us if vec is inside our set of defining points.
        
        Parameters
        ----------
        G : array [:,3]
            array of vectors that define the BZ
        vec : array [3]
            vector to be tested
            
        Returns
        -------
        False if outside the BZ, True otherwise
        """
        for G in BZG:
            if np.all(vec == G): continue
            if np.dot(vec, G) >= np.dot(G,G): return False
        return True
        
    def genBZG(self):
        """
        Generates the reciprocal lattice G points that define the Brillouin zone.
        """
        # Start with a list of possible vectors; add those that define the BZ...
        BZG = []
        nv = [0,0,0]
        for nv[0] in xrange(-3,4):
            for nv[1] in xrange(-3,4):
                for nv[2] in xrange(-3,4):
                    if nv==[0,0,0]: continue
                    vec = np.dot(self.lattice, nv)
                    if self.incell(BZG, vec): BZG.append(np.dot(self.rlattice, nv))
        # ... and use a list comprehension to only keep those that still remain
        self.BZG = np.array([0.5*vec for vec in BZG if self.incell(BZG,vec)])
        