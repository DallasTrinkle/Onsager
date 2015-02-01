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
import collections


def incell(vec):
    """
    Returns the vector inside the unit cell (in [0,1)**3)
    """
    return vec - np.floor(vec)


def inhalf(vec):
    """
    Returns the vector inside the centered cell (in [-0.5,0.5)**3)
    """
    return vec - np.floor(vec + 0.5)


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
    # type-checking:
    if __debug__:
        if type(oldpos) is not list: raise TypeError('oldpos is not a list')
        if type(newpos) is not list: raise TypeError('newpos is not a list')
        if len(oldpos) != len(newpos): raise IndexError("{} and {} do not have the same length".format(oldpos, newpos))
        for a, b in zip(oldpos, newpos):
            if type(a) is not list: raise TypeError("element of oldpos {} is not a list".format(a))
            if type(b) is not list: raise TypeError("element of newpos {} is not a list".format(b))
            if len(a) != len(b): raise IndexError("{} and {} do not have the same length".format(a, b))
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
                    if np.all(np.isclose(inhalf(ub - rua - trans), 0)):
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


class GroupOp(collections.namedtuple('GroupOp', 'rot trans cartrot indexmap')):
    """
    A class corresponding to a group operation. Based on namedtuple, so it is immutable.

    Intended to be used in combination with Crystal, we have a few operations that
    can be defined out-of-the-box.

    :param rot: np.array(3,3) integer idempotent matrix
    :param trans: np.array(3) real vector
    :param cartrot: np.array(3,3) real unitary matrix
    :param indexmap: list of list, containing the atom mapping
    """

    def incell(self):
        """Return a version of groupop where the translation is in the unit cell"""
        return GroupOp(self.rot, incell(self.trans), self.cartrot, self.indexmap)

    def inhalf(self):
        """Return a version of groupop where the translation is in the centered unit cell"""
        return GroupOp(self.rot, inhalf(self.trans), self.cartrot, self.indexmap)

    def __eq__(self, other):
        """Test for equality--we use numpy.isclose for comparison, since that's what we usually care about"""
        if __debug__:
            if type(other) is not GroupOp: raise TypeError
        return isinstance(other, self.__class__) and \
               np.all(self.rot == other.rot) and \
               np.all(np.isclose(self.trans, other.trans)) and \
               np.all(np.isclose(self.cartrot, other.cartrot)) and \
               self.indexmap == other.indexmap

    def __ne__(self, other):
        """Inequality == not __eq__"""
        return not self.__eq__(other)

    def __hash__(self):
        """Hash, so that we can make sets of group operations"""
        ### we are a little conservative, and only use the rotation to define the hash. This means
        ### we will get collisions for the same rotation but different translations. The reason is
        ### that __eq__ uses "isclose" on our translations, and we don't have a good way to handle
        ### that in a hash function. We lose a little bit on efficiency if we construct a set that
        ### has a whole lot of translation operations, but that's not usually what we will do.
        return reduce(lambda x,y: x^y, [256, 128, 64, 32, 16, 8, 4, 2, 1] * self.rot.reshape((9,)))
        # ^ reduce(lambda x,y: x^y, [hash(x) for x in self.trans])

    def __add__(self, other):
        """Add a translation to our group operation"""
        if __debug__:
            if type(other) is not np.ndarray: raise TypeError('Can only add a translation to a group operation')
            if other.shape != (3,): raise IndexError('Can only add a 3 dimensional vector')
            if not np.issubdtype(other.dtype, int): raise TypeError('Can only add a lattice vector translation')
        return GroupOp(self.rot, self.trans + other, self.cartrot, self.indexmap)

    def __sub__(self, other):
        """Add a (negative) translation to our group operation"""
        return self.__add__(-other)

    def __mul__(self, other):
        """Multiply two group operations to produce a new group operation"""
        if __debug__:
            if type(other) is not GroupOp: raise TypeError
        return GroupOp(np.dot(self.rot, other.rot),
                       np.dot(self.rot, other.trans) + self.trans,
                       np.dot(self.cartrot, other.cartrot),
                       [ [atomlist0[i] for i in atomlist1]
                         for atomlist0, atomlist1 in zip(self.indexmap, other.indexmap)])

    def inv(self):
        """Construct and return the inverse of the group operation"""
        inverse = (np.round(np.linalg.inv(self.rot))).astype(int)
        return GroupOp(inverse,
                       -np.dot(inverse, self.trans),
                       self.cartrot.T,
                       [ [ x for i,x in sorted([(y,j) for j,y in enumerate(atomlist)])]
                         for atomlist in self.indexmap])

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
        if type(lattice) is list:
            if len(lattice) != 3: raise TypeError('lattice is a list, but does not contain 3 members')
            self.lattice = np.array(lattice).T
        if type(lattice) is np.ndarray:
            self.lattice = lattice
        if self.lattice is None: raise TypeError('lattice is not a recognized type')
        if self.lattice.shape != (3, 3): raise TypeError('lattice contains vectors that are not 3 dimensional')
        if type(basis) is not list: raise TypeError('basis needs to be a list or list of lists')
        if type(basis[0]) == np.ndarray:
            for u in basis:
                if type(u) is not np.ndarray: raise TypeError("{} in {} is not an array".format(u, basis))
            self.basis = [[incell(u) for u in basis]]
        else:
            for elem in basis:
                if type(elem) is not list: raise TypeError("{} in basis is not a list".format(elem))
                for u in elem:
                    if type(u) is not np.ndarray: raise TypeError("{} in {} is not an array".format(u, elem))
            self.basis = [[incell(u) for u in atombasis] for atombasis in basis]
        self.reduce()  # clean up basis as needed
        self.minlattice()  # clean up lattice vectors as needed
        self.invlatt = np.linalg.inv(self.lattice)
        # this lets us, in a flat list, enumerate over indices of atoms as needed
        self.atomindices = [(atomtype, atomindex)
                            for atomtype,atomlist in enumerate(self.basis)
                            for atomindex in range(len(atomlist))]
        self.N = len(self.atomindices)
        self.volume, self.metric = self.calcmetric()
        self.center()  # should do before gengroup so that inversion is centered at origin
        self.G = self.gengroup()  # do before genpoint
        self.pointG = self.genpoint()

    def center(self):
        """
        Center the atoms in the cell if there is an inversion operation present.
        """
        # trivial case:
        if self.N == 1:
            self.basis = [[np.array([0., 0., 0.])]]
            return
        # else, invert positions!
        trans, indexmap = maptranslation(self.basis, [[-u for u in atomlist] for atomlist in self.basis])
        if indexmap is None:
            return
        # translate by -1/2 * trans for inversion
        self.basis = [[incell(u - 0.5 * trans) for u in atomlist] for atomlist in self.basis]
        # now, check for "aesthetics" of our basis choice
        shift = np.zeros(3)
        for d in xrange(3):
            if np.any([np.isclose(u[d], 0) for atomlist in self.basis for u in atomlist]):
                shift[d] = 0
            elif np.any([np.isclose(u[d], 0.5) for atomlist in self.basis for u in atomlist]):
                shift[d] = 0.5
            elif sum([1 for atomlist in self.basis for u in atomlist if u[d] < 0.25 or u[d] > 0.75]) > self.N / 2:
                shift[d] = 0.5
        self.basis = [[incell(atom + shift) for atom in atomlist] for atomlist in self.basis]

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
                    if np.all([not np.all(np.isclose(inhalf(u + t - v), 0)) for v in atomlist]):
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
            super[:, d] = t[:]
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
                if np.all([not np.all(np.isclose(v, v1)) for v1 in newatomlist]):
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
        return [[incell(np.dot(invsuper, u)) for u in atomlist] for atomlist in self.basis]

    def minlattice(self):
        """
        Try to find the optimal lattice vector definition for a crystal. Our definition of optimal
        is (a) length of each lattice vector is minimal; (b) the vectors are ordered from
        shortest to longest; (c) the vectors have minimal dot product; (d) the basis is right-handed.

        Works recursively.
        """
        magnlist = sorted((np.dot(v, v), idx) for idx, v in enumerate(self.lattice.T))
        super = np.zeros((3, 3), dtype=int)
        for i, (magn, j) in enumerate(magnlist):
            super[j, i] = 1
        # check that we have a right-handed lattice
        if np.linalg.det(self.lattice) * np.linalg.det(super) < 0:
            super[:, 2] = -super[:, 2]
        if not np.all(super == np.eye(3, dtype=int)):
            self.lattice = np.dot(self.lattice, super)
            self.basis = self.remapbasis(super)

        super = np.eye(3, dtype=int)
        modified = False
        # check the possible vector reductions
        asq = np.dot(self.lattice.T, self.lattice)
        u = np.around([asq[0, 1] / asq[0, 0], asq[0, 2] / asq[0, 0], asq[1, 2] / asq[1, 1]])
        if u[0] != 0:
            super[0, 1] = -int(u[0])
            modified = True
        elif u[1] != 0:
            super[0, 2] = -int(u[1])
            modified = True
        elif u[2] != 0:
            super[1, 2] = -int(u[2])
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
        groupops = []
        supercellvect = [np.array((n0, n1, n2))
                         for n0 in xrange(-1, 2)
                         for n1 in xrange(-1, 2)
                         for n2 in xrange(-1, 2)
                         if (n0, n1, n2) != (0, 0, 0)]
        matchvect = [[u for u in supercellvect
                      if np.isclose(np.dot(u, np.dot(self.metric, u)),
                                    self.metric[d, d])] for d in xrange(3)]
        for super in [np.array((r0, r1, r2)).T
                      for r0 in matchvect[0]
                      for r1 in matchvect[1]
                      for r2 in matchvect[2]
                      if abs(np.inner(r0, np.cross(r1, r2))) == 1]:
            if np.all(np.isclose(np.dot(super.T, np.dot(self.metric, super)), self.metric)):
                # possible operation--need to check the atomic positions
                trans, indexmap = maptranslation(self.basis,
                                                 [[np.dot(super, u)
                                                   for u in atomlist]
                                                  for atomlist in self.basis])
                if indexmap is not None:
                    groupops.append(GroupOp(super,
                                            trans,
                                            np.dot(self.lattice, np.dot(super, self.invlatt)),
                                            indexmap))
        return frozenset(groupops)

    def pos2cart(self, lattvec, ind):
        """
        Return the cartesian coordinates of an atom specified by its lattice and index
        :param lattvec: 3-vector (integer) lattice vector in direct coordinates
        :param ind: two-tuple index specifying the atom: (atomtype, atomindex)
        :return: 3-vector (float) in Cartesian coordinates
        """
        return np.dot(self.lattice, lattvec + self.basis[ind[0]][ind[1]])

    def unit2cart(self, lattvec, uvec):
        """
        Return the cartesian coordinates of a position specified by its lattice and
        unit cell coordinates
        :param lattvec: 3-vector (integer) lattice vector in direct coordinates
        :param uvec: 3-vector (float) unit cell vector in direct coordinates
        :return: 3-vector (float) in Cartesian coordinates
        """
        return np.dot(self.lattice, lattvec + uvec)

    def cart2unit(self, v):
        """
        Return the lattvec and unit cell coord. corresponding to a position
        in cartesian coord.
        :param v: 3-vector (float) position in Cartesian coordinates
        :return: 3-vector (integer) lattice vector in direct coordinates,
         3-vector (float) inside unit cell
        """
        u = np.dot(self.invlatt, v)
        ucell = incell(u)
        return (u-ucell).astype(int), ucell

    def cart2pos(self, v):
        """
        Return the lattvec and index corresponding to an atomic position in cartesian coord.
        :param v: 3-vector (float) position in Cartesian coordinates
        :return: 3-vector (integer) lattice vector in direct coordinates, index tuple
         of corresponding atom.
         Returns None on tuple if no match
        """
        latt, u = self.cart2unit(v)
        indlist = [ind for ind in self.atomindices
                   if np.all(np.isclose(u, self.basis[ind[0]][ind[1]]))]
        if len(indlist) != 1:
            return latt, None
        else:
            return latt, indlist[0]

    def g_direc(self, g, direc):
        """
        Apply a space group operation to a direction
        :param g: group operation (GroupOp)
        :param direc: 3-vector direction
        :return: 3-vector direction
        """
        if __debug__:
            if type(g) is not GroupOp: raise TypeError
            if type(direc) is not np.ndarray: raise TypeError
        return np.dot(g.cartrot, direc)

    def g_pos(self, g, lattvec, ind):
        """
        Apply a space group operation to an atom position specified by its lattice and index
        :param g: group operation (GroupOp)
        :param lattvec: 3-vector (integer) lattice vector in direct coordinates
        :param ind: two-tuple index specifying the atom: (atomtype, atomindex)
        :return: 3-vector (integer) lattice vector in direct coordinates, index
        """
        if __debug__:
            if type(g) is not GroupOp: raise TypeError
            if type(lattvec) is not np.ndarray: raise TypeError
        rotlatt = np.dot(g.rot, lattvec)
        rotind = (ind[0], g.indexmap[ind[0]][ind[1]])
        delu = (np.round(np.dot(g.rot, self.basis[ind[0]][ind[1]]) + g.trans -
                         self.basis[rotind[0]][rotind[1]])).astype(int)
        return rotlatt + delu, rotind

    def g_vect(self, g, lattvec, uvec):
        """
        Apply a space group operation to a vector position specified by its lattice and a location
        in the unit cell in direct coordinates
        :param g:  group operation (GroupOp)
        :param lattvec: 3-vector (integer) lattice vector in direct coordinates
        :param uvec: 3-vector (float) vector in direct coordinates
        :return: 3-vector (integer) lattice vector in direct coordinates, location in unit cell in
         direct coordinates
        """
        if __debug__:
            if type(g) is not GroupOp: raise TypeError
            if type(lattvec) is not np.ndarray: raise TypeError
            if type(uvec) is not np.ndarray: raise TypeError
        rotlatt = np.dot(g.rot, lattvec)
        rotu = np.dot(g.rot, uvec)
        incellu = incell(rotu)
        return rotlatt + (np.round(rotu - incellu)).astype(int), incellu

    def genpoint(self):
        """
        Generate our point group indices. Done with crazy list comprehension due to the
        structure of our basis.
        :return: list of sets of point group operations that leave a site unchanged
        """
        if self.N == 1:
            return [[self.G]]
        origin = np.zeros(3, dtype=int)
        return [[frozenset([g - self.g_pos(g, origin, (atomtypeindex, atomindex))[0]
                            for g in self.G
                            if g.indexmap[atomtypeindex][atomindex] == atomindex])
                 for atomindex in range(len(atomlist))]
                for atomtypeindex, atomlist in enumerate(self.basis)]
