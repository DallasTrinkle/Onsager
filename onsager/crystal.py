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
import yaml ### use crystal.yaml to call--may need to change in the future

# YAML tags:
# interfaces are either at the bottom, or staticmethods in the corresponding object
NDARRAY_YAMLTAG = u'!numpy.ndarray'
GROUPOP_YAMLTAG = u'!GroupOp'
#FROZENSET_YAMLTAG = u'!set'


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
        # if not type(other) is not GroupOp: return False
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

    def eigen(self):
        """Returns the type of group operation (single integer) and eigenvectors.
        1 = identity
        2, 3, 4, 6 = n- fold rotation around an axis
        negative = rotation + mirror operation, perpendicular to axis
        "special cases": -1 = mirror, -2 = inversion

        eigenvect[0] = axis of rotation / mirror
        eigenvect[1], eigenvect[2] = orthonormal vectors to define the plane giving a right-handed
          coordinate system and where rotation around [0] is positive, and the positive imaginary
          eigenvector for the complex eigenvalue is [1] + i [2].
        """
        tr = np.int(np.round(self.cartrot.trace()))
        if np.linalg.det(self.cartrot) > 0:
            det = 1
            optype = (2, 3, 4, 6, 1)[tr + 1] # trace determines the rotation type
        else:
            det = -1
            optype = (-2, -3, -4, -6, -1)[tr + 3] # trace determines the rotation type
        # two trivial cases: identity, inversion:
        if optype == 1 or optype == -2:
            return optype, np.eye(3)
        # otherwise, there's an axis to find:
        vmat = np.eye(3)
        vsum = np.zeros((3,3))
        if det>0:
            for n in range(optype):
                vsum += vmat
                vmat = np.dot(self.cartrot, vmat)
        else:
            for n in range((0, 6, 4, 3, 2)[tr + 3]): #
                vsum += vmat
                vmat = -np.dot(self.cartrot, vmat)
        # vmat *should* equal identity if we didn't fail...
        if __debug__:
            if not np.all(np.isclose(vmat, np.eye(3))): raise ArithmeticError('eigenvalue analysis fail')
        vsum *= 1./n
        # now the columns of vsum should either be (a) our rotation / mirror axis, or (b) zero
        eig0 = vsum[:,0]
        magn0 = np.dot(eig0, eig0)
        if magn0 < 1e-2:
            eig0 = vsum[:,1]
            magn0 = np.dot(eig0, eig0)
            if magn0 < 1e-2:
                eig0 = vsum[:,2]
                magn0 = np.dot(eig0, eig0)
        eig0 /= np.sqrt(magn0)
        # now, construct the other two directions:
        if abs(eig0[2]) < 0.75:
            eig1 = np.array([eig0[1], -eig0[0], 0])
        else:
            eig1 = np.array([-eig0[2], 0, eig0[0]])
        eig1 /= np.sqrt(np.dot(eig1, eig1))
        eig2 = np.cross(eig0, eig1)
        # we have a right-handed coordinate system; test that we have a positive rotation around the axis
        if abs(optype) > 2:
            if np.dot(eig2, np.dot(self.cartrot, eig1)) > 0:
                eig0 = -eig0
                eig2 = -eig2
        return optype, [eig0, eig1, eig2]

    @staticmethod
    def GroupOp_representer(dumper, data):
        """Output a GroupOp"""
        # asdict() returns an OrderedDictionary, so pass through dict()
        return dumper.represent_mapping(GROUPOP_YAMLTAG, dict(data._asdict()))

    @staticmethod
    def GroupOp_constructor(loader, node):
        """Construct a GroupOp from YAML"""
        # ** turns the dictionary into parameters for GroupOp constructor
        return GroupOp(**loader.construct_mapping(node, deep=True))


def VectorBasis(rottype, eigenvect):
    """
    Returns a vector basis corresponding to the optype and eigenvectors for a GroupOp
    :param rottype: output from eigen()
    :param eigenvect: eigenvectors
    :return: (dim, vect) -- dimensionality (0..3), vector defining line direction (1) or plane normal (2)
    """
    # edge cases first:
    if rottype == 1: return (3, np.zeros(3)) # sphere (identity)
    if rottype == -2: return (0, np.zeros(3)) # point (inversion)
    if rottype == -1: return (2, eigenvect[0]) # plane (pure mirror)
    return (1, eigenvect[0]) # line (all others--there's a rotation axis involved

def CombineBasis(b1, b2):
    """
    Combines (intersects) two vector spaces into one.
    :param b1: (dim, vect) -- dimensionality (0..3), vector defining line direction (1) or plane normal (2)
    :param b2: (dim, vect)
    :return: (dim, vect)
    """
    # edge cases first
    if b1[0] == 3: return b2 # sphere with anything
    if b2[0] == 3: return b1
    if b1[0] == 0: return b1 # point with anything
    if b2[0] == 0: return b2
    if b1[0] == b2[0]:
        if abs(np.dot(b1[1], b2[1])) > (1.-1e-8): # parallel vectors
            return b1 # equal bases
        else: # vectors not equal...
            if b1[0] == 1: # for a line, that's death:
                return (0, np.zeros(3))
            else: # for a plane, need the mutual line:
                v = np.cross(b1[1], b2[1])
                return (1, v/np.sqrt(np.dot(v,v)))
    # finally: one is a plane, other is a line:
    if abs(np.dot(b1[1], b2[1])) > 1e-8: # if the vectors are not perpendicular, death:
        return (0, np.zeros(3))
    else: # return whichever is a line:
        if b1[0] == 1:
            return b1
        else:
            return b2


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
            self.lattice = np.array(lattice)
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
        self.reciplatt = 2.*np.pi*self.invlatt.T
        self.BZvol = abs(float(np.linalg.det(self.reciplatt)))
        self.center()  # should do before gengroup so that inversion is centered at origin
        self.G = self.gengroup()  # do before genpoint
        self.pointG = self.genpoint()
        self.Wyckoff = self.genWyckoffsets()

    def __repr__(self):
        """String representation of crystal (lattice + basis)"""
        return 'Crystal(' + repr(self.lattice).replace('\n','').replace('\t','') + ',' + repr(self.basis) + ')'

    def __str__(self):
        """Human-readable version of crystal (lattice + basis)"""
        str = "#Lattice:\n  a1 = {}\n  a2 = {}\n  a3 = {}\n#Basis:".format(
            self.lattice.T[0], self.lattice.T[1], self.lattice.T[2])
        for chemind, atoms in enumerate(self.basis):
            for atomind, pos in enumerate(atoms):
                str = str + "\n  {}.{} = {}".format(chemind+1, atomind+1, pos)
        return str

    @classmethod
    def fromdict(cls, yamldict):
        """
        Creates a Crystal object from a YAML-created dictionary
        :param yamldict: dictionary; must contain 'lattice' (using *row* vectors!) and 'basis';
        can contain optional 'lattice_constant'
        :return: Crystal(lattice.T, basis)
        """
        if 'lattice' not in yamldict: raise IndexError('{} does not contain "lattice"'.format(yamldict))
        if 'basis' not in yamldict: raise IndexError('{} does not contain "basis"'.format(yamldict))
        lattice_constant = 1.
        if 'lattice_constant' in yamldict: lattice_constant = yamldict['lattice_constant']
        return Crystal((lattice_constant*yamldict['lattice']).T, yamldict['basis'])

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
        return abs(float(np.linalg.det(self.lattice))), np.dot(self.lattice.T, self.lattice)

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
        for super in (np.array((r0, r1, r2)).T
                      for r0 in matchvect[0]
                      for r1 in matchvect[1]
                      for r2 in matchvect[2]
                      if abs(np.inner(r0, np.cross(r1, r2))) == 1):
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
        rotu = np.dot(g.rot, uvec) + g.trans
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

    def genWyckoffsets(self):
        """
        Generate our Wykcoff sets.
        :return: set of sets of tuples of positions that correspond to identical Wyckoff positions
        """
        if self.N == 1:
            return frozenset([frozenset([(0,0)])])
        # this is a little suboptimal if our basis is huge--it leans heavily
        # on the construction of sets to make the checks easy.
        return frozenset([ frozenset([ (ind[0], g.indexmap[ind[0]][ind[1]])
                                       for g in self.G])
                           for ind in self.atomindices])

    def Wyckoffpos(self, uvec):
        """
        Generates all the equivalent Wyckoff positions for a unit cell vector.
        :param uvec: 3-vector (float) vector in direct coordinates
        :return: list of equivalent Wyckoff positions
        """
        lis = []
        zero = np.zeros(3, dtype=int)
        for u in ( self.g_vect(g, zero, uvec)[1] for g in self.G ):
            if not np.any([np.all(np.isclose(u, u1)) for u1 in lis]):
                lis.append(u)
        return lis

    def VectorBasis(self, ind):
        """
        Generates the vector basis corresponding to an atomic site
        :param ind: tuple index for atom
        :return: (dim, vect) -- dimension of basis, vector = normal for plane, direction for line
        """
        # need to work with the point group operations for the site
        return reduce(CombineBasis,
                      [ VectorBasis(*g.eigen()) for g in self.pointG[ind[0]][ind[1]] ] )
        # , (3, np.zeros(3)) -- don't need initial value; if there's only one group op, it's identity

    def nnlist(self, ind, cutoff):
        """
        Generate the nearest neighbor list for a given cutoff. Only consider
        neighbor vectors for atoms of the same type. Returns a list of
        cartesian vectors.
        :param ind: tuple index for atom
        :param cutoff:  distance cutoff
        :return: list of nearest neighbor vectors
        """
        r2 = cutoff*cutoff
        nmax = [int(np.round(np.sqrt(self.metric[i,i])))+1
                for i in range(3)]
        supervect = [ np.array([n0, n1, n2])
                      for n0 in xrange(-nmax[0],nmax[0]+1)
                      for n1 in xrange(-nmax[1],nmax[1]+1)
                      for n2 in xrange(-nmax[2],nmax[2]+1) ]
        lis = []
        u0 = self.basis[ind[0]][ind[1]]
        for u1 in self.basis[ind[0]]:
            du = u1-u0
            for n in supervect:
                dx = self.unit2cart(n, du)
                if np.dot(dx, dx) > 0 and np.dot(dx,dx) < r2:
                    lis.append(dx)
        return lis


# YAML interfaces for types outside of this module
def ndarray_representer(dumper, data):
    """Output a numpy array"""
    return dumper.represent_sequence(NDARRAY_YAMLTAG, data.tolist())

### NOTE: deep=True is THE KEY here for reading
### hat-tip: https://stackoverflow.com/questions/19439765/is-there-a-way-to-construct-an-object-using-pyyaml-construct-mapping-after-all-n
def ndarray_constructor(loader, node):
    return np.array(loader.construct_sequence(node, deep=True))

# YAML registration:
yaml.add_representer(np.ndarray, ndarray_representer)
yaml.add_constructor(NDARRAY_YAMLTAG, ndarray_constructor)

yaml.add_representer(GroupOp, GroupOp.GroupOp_representer)
yaml.add_constructor(GROUPOP_YAMLTAG, GroupOp.GroupOp_constructor)

