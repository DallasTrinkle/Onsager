"""
KPTmesh module

Class definition for KPTmesh class; allows automated construction of kpt meshes
"""

import numpy as np


class KPTmesh:
    """
    A class to construct (symmetrized, reduced to the irreducible wedge) k-point meshes.
    """
    def __init__(self, lattice, nmesh=(0, 0, 0), groupops=None):
        """
        Creates an instance of a k-point mesh generator.
        
        Parameters
        ----------
        lattice : array [3,3}
            lattice vectors, in *column* form, so that a1 = a[:,0], a2 = a[:,1], a3 = a[:,2]
        Nmesh : list [3], optional
            number of divisions; can be specified later using genmesh().
        groupops : list [Nop, 3, 3], optional
            point group operations; if not explicitly included, then generate from RLV
        """
        self.lattice = lattice
        self.volume = np.linalg.det(lattice)
        self.rlattice = 2.*np.pi*(np.linalg.inv(lattice)).T
        self.Nmesh = (-1, -1, -1)
        self.Nkpt = -1
        self.kptfull = np.array([[0]])
        self.genBZG()
        if nmesh != (0, 0, 0):
            self.genmesh(nmesh)
        if groupops != None :
            self.groupops = groupops
        else:
            self.gengroupops()

    def genmesh(self, Nmesh):
        """
        Initiates, if mesh doesn't already exist, the construction of a mesh of size N.
        
        Parameters
        ----------
        Nmesh : list
            should have length 3; specifies number of divisions in 1, 2, 3 directions.
        """
        if Nmesh == self.Nmesh : return
        self.Nmesh = Nmesh
        self.Nkpt = np.product(Nmesh)
        if self.Nkpt == 0: return
        dN = np.array([1./x for x in Nmesh])
        # use a list comprehension to iterate and build:
        self.kptfull = np.array([np.dot(self.rlattice, (n0*dN[0], n1*dN[1], n2*dN[2]))
                                 for n0 in xrange(-Nmesh[0]/2+1, Nmesh[0]/2+1)
                                 for n1 in xrange(-Nmesh[1]/2+1, Nmesh[1]/2+1)
                                 for n2 in xrange(-Nmesh[2]/2+1, Nmesh[2]/2+1)])
        # run through list to ensure that all k-points are inside the BZ
        Gmin = min([ np.dot(G, G) for G in self.BZG])
        for i, k in enumerate(self.kptfull):
            if np.dot(k, k)>=Gmin:
                for G in self.BZG:
                    if np.dot(k, G)>np.dot(G, G):
                        k -= 2. * G
                self.kptfull[i] = k

    def gengroupops(self, threshold=1e-8):
        """
        Generates the point group operations (stored in cartesian coord), given the reciprocal lattice vectors.

        Parameters
        ----------
        threshold : double, optional
            threshold for equality in generating point group operations

        Notes
        -----
        The principle of the algorithm rests on a simple idea: a point group operation can be expressed
        as a "supercell" of size 1 (which means an integer, idempotent matrix), and an orthogonal matrix
        (a "rotation", so its transpose is its inverse). Then, if g is a potential op,
        g.[b] = [b].n
        where n is an integer, idempotent matrix and g is an orthogonal matrix. We generate all possible
        integer, idempotent n matrices, construct g, and test for orthogonality.
        """
        groupops = []
        invrlatt = np.linalg.inv(self.rlattice)
        supercellvect = [np.array((n0, n1, n2))
                         for n0 in xrange(-1, 2)
                         for n1 in xrange(-1, 2)
                         for n2 in xrange(-1, 2)
                         if (n0, n1, n2) != (0, 0, 0)]
        for g in [np.dot(self.rlattice, np.dot(nmat, invrlatt))
                  for nmat in [np.array((n0, n1, n2))
                               for n0 in supercellvect
                               for n1 in supercellvect
                               for n2 in supercellvect]
                  if abs(np.linalg.det(nmat))==1]:
            if np.all(abs(np.dot(g.T, g)-np.eye(3))<threshold):
                groupops.append(g)
        self.groupops = np.array(groupops)

    def incell(self, vec, BZG=None, threshold=1e-5):
        """
        Tells us if vec is inside our set of defining points.
        
        Parameters
        ----------
        vec : array [3]
            vector to be tested
        BGZ : array [:,3], optional (default = self.BZG)
            array of vectors that define the BZ
        threshold : double, options
            threshold to use for "equality"

        Returns
        -------
        False if outside the BZ, True otherwise
        """
        if BZG == None :
            BZG=self.BZG
        # checks that vec.G < G^2 for all G (and throws out the option that vec == G, in case threshold == 0)
        return all([(np.dot(vec, G) < np.dot(G, G)+threshold) for G in BZG if not np.all(vec == G)])

    def genBZG(self):
        """
        Generates the reciprocal lattice G points that define the Brillouin zone.
        """
        # Start with a list of possible vectors; add those that define the BZ...
        BZG = []
        for nv in [[n0, n1, n2]
                   for n0 in xrange(-3, 4)
                   for n1 in xrange(-3, 4)
                   for n2 in xrange(-3, 4)
                   if (n0, n1, n2) != (0, 0, 0)]:
            vec = np.dot(self.lattice, nv)
            if self.incell(vec, BZG, threshold=0): BZG.append(np.dot(self.rlattice, nv))
        # ... and use a list comprehension to only keep those that still remain
        self.BZG = np.array([0.5*vec for vec in BZG if self.incell(vec, BZG, threshold=0)])
        
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
        if np.shape(self.kptfull) != (self.Nkpt, 3):
            # generate those kpoints!
            self.genmesh(self.Nmesh)
        if self.Nkpt == 0 :
            return np.array(((0))), np.array((0))
        return self.kptfull, np.array((1./self.Nkpt,)*self.Nkpt)
    