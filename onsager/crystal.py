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
    return vec - np.floor(vec+0.5)

def maptranslation(oldpos, newpos):
    """
    Given a list of transformed positions, identify if there's a translation vector
    that maps from the current positions to the new position.

    :param oldpos: list of list of array[3]
    :param newpos: list of list of array[3], same layout as oldpos
    :return: translation (array[3]), mapping (list of list of indices)

    The mapping specifies the index that the *translated* atom corresponds to in the
    original position set. If unable to construct a mapping, the mapping return is
    None; the translation vector will be meaningless.
    """
    # type-checking; may remove in production
    assert type(oldpos) == list, "oldpos is not a list"
    assert type(newpos) == list, "newpos is not a list"
    assert len(oldpos) == len(newpos), "{} and {} do not have the same length".format(oldpos, newpos)
    for a,b in zip(oldpos, newpos):
        assert type(a) == list, "element of oldpos {} is not a list".format(a)
        assert type(b) == list, "element of newpos {} is not a list".format(b)
        assert len(a) == len(b), "{} and {} do not have the same length".format(a,b)
    # Work with the shortest possible list for identifying translations
    maxlen = 0
    atomindex = 0
    for i, ulist in enumerate(oldpos):
        if len(ulist) > maxlen:
            maxlen = len(ulist)
            atomindex = i
    ru0 = newpos[atomindex][0]
    for ub in oldpos[atomindex]:
        trans = inhalf(ub - ru0)
        foundmap = True
        # now check against all the others, and construct the mapping
        indexmap = []
        for atomlist0, atomlist1 in zip(oldpos, newpos):
            # work through the "new" positions
            if not foundmap: break
            maplist = []
            for rua in atomlist1:
                for j, ub in enumerate(atomlist0):
                    if np.all(np.isclose(inhalf(ub-rua-trans), 0)):
                        maplist.append(j)
                        break
            if len(maplist) != len(atomlist0):
                foundmap = False
            else:
                indexmap.append(maplist)
        if foundmap: break
    if foundmap:
        return trans, indexmap
    else:
        return None, None


class GroupOp(object):
    """
    A class corresponding to a group operation. May add more here later beyond just storage.
    """

    def __init__(self, rot, trans, cartrot, carttrans, indexmap):
        """
        Initialization
        """
        self.rot = rot
        self.trans = trans
        self.cartrot = cartrot
        self.carttrans = carttrans
        self.indexmap = indexmap


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
        self.reduce() # clean up basis as needed
        self.minlattice()  # clean up lattice vectors as needed
        self.N = sum(len(atomlist) for atomlist in self.basis)
        self.volume, self.metric = self.calcmetric()
        self.center() # should do before gengroup so that inversion is centered at origin
        self.g = self.gengroup() # do before genpoint
        self.pointindex = self.genpoint()

    def center(self):
        """
        Center the atoms in the cell if there is an inversion operation present.
        """
        # trivial case:
        if self.N == 1:
            self.basis = [[ np.array([0., 0., 0.]) ]]
            return
        # else, invert positions!
        trans, indexmap = maptranslation(self.basis, [ [-u for u in atomlist] for atomlist in self.basis])
        if indexmap is None:
            return
        # translate by -1/2 * trans for inversion
        self.basis = [ [ incell(u-0.5*trans) for u in atomlist] for atomlist in self.basis]
        # now, check for "aesthetics" of our basis choice
        shift = np.array([0.,0.,0.])
        for d in xrange(3):
            if np.any([ np.isclose(u[d],0) for atomlist in self.basis for u in atomlist ]):
                shift[d] = 0
            elif np.any([ np.isclose(u[d],0.5) for atomlist in self.basis for u in atomlist ]):
                shift[d] = 0.5
            elif sum([ 1 for atomlist in self.basis for u in atomlist if u[d] < 0.25 or u[d] > 0.75]) > self.N/2:
                shift[d] = 0.5
        self.basis = [ [ incell(atom + shift) for atom in atomlist] for atomlist in self.basis]

    def reduce(self):
        """
        Reduces the lattice and basis, if needed. Works (tail) recursively.
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

    def remapbasis(self, super):
        """
        Takes the basis definition, and using a supercell definition, returns a new basis
        :param super: integer array[3,3]
        :return: atomic basis
        """
        invsuper = np.linalg.inv(super)
        return [ [ incell(np.dot(invsuper, u)) for u in atomlist] for atomlist in self.basis]

    def minlattice(self):
        """
        Try to find the optimal lattice vector definition for a crystal. Our definition of optimal
        is (a) length of each lattice vector is minimal; (b) the vectors are ordered from
        shortest to longest; (c) the vectors have minimal dot product; (d) the basis is right-handed.

        Works recursively.
        """
        magnlist = sorted( (np.dot(v,v), idx) for idx, v in enumerate(self.lattice.T) )
        super = np.zeros((3,3), dtype=int)
        for i, (magn, j) in enumerate(magnlist):
            super[j, i] = 1
        # check that we have a right-handed lattice
        if np.linalg.det(self.lattice) * np.linalg.det(super) < 0:
            super[:,2] = -super[:,2]
        if not np.all(super == np.eye(3, dtype=int)):
            self.lattice = np.dot(self.lattice, super)
            self.basis = self.remapbasis(super)

        super = np.eye(3, dtype=int)
        modified = False
        # check the possible vector reductions
        asq = np.dot(self.lattice.T, self.lattice)
        u = np.around([asq[0,1]/asq[0,0], asq[0,2]/asq[0,0], asq[1,2]/asq[1,1]])
        if u[0] != 0:
            super[0,1] = -int(u[0])
            modified = True
        elif u[1] != 0:
            super[0,2] = -int(u[1])
            modified = True
        elif u[2] != 0:
            super[1,2] = -int(u[2])
            modified = True

        if not modified:
            return
        self.lattice = np.dot(self.lattice, super)
        self.basis = self.remapbasis(super)
        self.minlattice()

    def calcmetric(self):
        """
        Computes the volume of the cell and the metric tensor
        :return: volume, metric tensor
        """
        return abs(np.linalg.det(self.lattice)), np.dot(self.lattice.T, self.lattice)

    def gengroup(self):
        """
        Generate all of the space group operations.
        :return: list of group operations
        """
        invlatt = np.linalg.inv(self.lattice)
        groupops = []
        supercellvect = [np.array((n0, n1, n2))
                         for n0 in xrange(-1, 2)
                         for n1 in xrange(-1, 2)
                         for n2 in xrange(-1, 2)
                         if (n0, n1, n2) != (0, 0, 0)]
        matchvect = [ [ u for u in supercellvect
                        if np.isclose(np.dot(u, np.dot(self.metric, u)),
                                      self.metric[d,d]) ] for d in xrange(3) ]
        for super in [ np.array((r0, r1, r2)).T
                       for r0 in matchvect[0]
                       for r1 in matchvect[1]
                       for r2 in matchvect[2] ]:
            if abs(np.linalg.det(super)) == 1:
                if np.all(np.isclose(np.dot(super.T, np.dot(self.metric, super)), self.metric)):
                    # possible operation--need to check the atomic positions
                    trans, indexmap = maptranslation(self.basis,
                                                     [[ np.dot(super, u)
                                                        for u in atomlist]
                                                        for atomlist in self.basis])
                    if indexmap is None: continue
                    groupops.append(GroupOp(super,
                                            trans,
                                            np.dot(self.lattice, np.dot(super,invlatt)),
                                            np.dot(self.lattice, trans),
                                            indexmap))
        return groupops

    def genpoint(self):
        """
        Generate our point group indices. Done with crazy list comprehension due to the
        structure of our basis.
        :return: list of list of indices of group operations that leave a site unchanged
        """
        return [ [ [ gind for gind, g in enumerate(self.g)
                     if g.indexmap[atomtypeindex][atomindex] == atomindex ]
                   for atomindex in range(len(atomlist))]
                 for atomtypeindex, atomlist in enumerate(self.basis) ]
