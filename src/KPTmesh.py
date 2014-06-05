"""
KPTmesh module

Class definition for KPTmesh class; allows automated construction of kpt meshes

Test that we can git.
"""

import numpy as np

class KPTmesh:
    """
    A class to construct (symmetrized, reduced to the irreducible wedge) k-point meshes.
    """
    def __init__(self, lattice, Nmesh=(0,0,0)):
        """
        Creates an instance of a k-point mesh generator.
        
        Parameters
        ----------
        lattice : array [3,3}
            lattice vectors, in *column* form, so that a1 = a[:,0], a2 = a[:,1], a3 = a[:,2]
        Nmesh : list [3], optional
            number of divisions; can be specified later using genmesh().
        """
        self.lattice = lattice
        self.volume = np.linalg.det(lattice)
        self.rlattice = 2.*np.pi*(np.linalg.inv(lattice)).T
        self.Nmesh = (-1,-1,-1)
        self.Nkpt = -1
        self.kptfull = np.array(((0)))
        self.genBZG()
        if (Nmesh != (0,0,0)): self.genmesh(self, self.Nmesh)

    def genmesh(self, Nmesh):
        """
        Initiates, if mesh doesn't already exist, the construction of a mesh of size N.
        
        Parameters
        ----------
        Nmesh : list
            should have length 3; specifies number of divisions in 1, 2, 3 directions.
        """
        if Nmesh[0] == self.Nmesh[0] and Nmesh[1] == self.Nmesh[1] and Nmesh[2] == self.Nmesh[2] : return
        self.Nmesh = Nmesh
        self.Nkpt = np.product(Nmesh)
        if self.Nkpt == 0: return
        dN = np.array([1./x for x in Nmesh])
        meshrange = [xrange(-Nmesh[0]/2+1, Nmesh[0]/2+1),
                     xrange(-Nmesh[1]/2+1, Nmesh[1]/2+1),
                     xrange(-Nmesh[2]/2+1, Nmesh[2]/2+1)]
        # use a list comprehension to iterate and build:
        self.kptfull = np.array([ np.dot(self.rlattice, (n0*dN[0], n1*dN[1], n2*dN[2]))
                                  for n0 in meshrange[0] for n1 in meshrange[1] for n2 in meshrange[2] ])

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
        
    def fullmesh(self):
        """
        Returns (after generating, if need be) the full (unfolded) k-point mesh, with weights.
        
        Returns
        -------
        kpt : array [:,3]
            individual k-points, in Cartesian coordinates
        wts : array [:]
            weight of each k-point
        """
        if (np.shape(self.kptfull) != (self.Nkpt, 3)):
            # generate those kpoints!
            self.genmesh(self.Nmesh)
        if self.Nkpt == 0 :
            return np.array(((0))), np.array((0))
        return self.kptfull, np.array((1./self.Nkpt,)*self.Nkpt)
    