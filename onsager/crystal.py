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
import collections, copy, itertools
from numbers import Number
from math import gcd
import yaml  # use crystal.yaml to call--may need to change in the future
from functools import reduce

# YAML tags:
# interfaces are either at the bottom, or staticmethods in the corresponding object
NDARRAY_YAMLTAG = '!numpy.ndarray'
GROUPOP_YAMLTAG = '!GroupOp'


def gcdlist(lis):
    """Returns the GCD of a list of integers"""
    return reduce(gcd, lis)


def incell(vec):
    """
    Returns the vector inside the unit cell (in [0,1)**3)

    :param vec: 3-vector (unit coord)
    :return: 3-vector
    """
    return vec - np.floor(vec + 1.0e-8)


def inhalf(vec):
    """
    Returns the vector inside the centered cell (in [-0.5,0.5)**3)

    :param vec: 3-vector (unit coord)
    :return: 3-vector
    """
    return vec - np.floor(vec + 0.5)


def maptranslation(oldpos, newpos, oldspins=None, newspins=None, threshold=1e-8):
    """
    Given a list of transformed positions, identify if there's a translation vector
    that maps from the current positions to the new position.

    The mapping specifies the index that the *translated* atom corresponds to in the
    original position set. If unable to construct a mapping, the mapping return is
    None; the translation vector will be meaningless.

    If old/newspins are given then ONLY mappings that maintain spin are considered.
    This means that a loop is needed to consider possible spin phase factors.

    :param oldpos: list of list of array[3]
    :param newpos: list of list of array[3], same layout as oldpos
    :param oldspins: (optional) list of list of numbers/arrays
    :param newspins: (optional) list of list of numbers/arrays
    :return translation: array[3]
    :return mapping: list of list of indices
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
        if (oldspins is None) != (newspins is None): raise TypeError('give both or neither spin arguments')
        if oldspins is not None:
            if type(oldspins) is not list: raise TypeError('oldspins is not a list')
            if type(newspins) is not list: raise TypeError('newspins is not a list')
            if len(oldspins) != len(newspins): raise IndexError(
                "{} and {} do not have the same length".format(oldspins, newspins))
            for a, b in zip(oldspins, newspins):
                if type(a) is not list: raise TypeError("element of oldspins {} is not a list".format(a))
                if type(b) is not list: raise TypeError("element of newspins {} is not a list".format(b))
                if len(a) != len(b): raise IndexError("{} and {} do not have the same length".format(a, b))
    if oldspins is None:
        oldspins = [[0 for u in atomlist] for atomlist in oldpos]
    if newspins is None:
        newspins = oldspins

    # Work with the shortest possible list for identifying translations
    atomindex = 0
    maxlen = len(oldpos[atomindex])
    for i, ulist in enumerate(oldpos):
        if len(ulist) < maxlen:
            maxlen = len(ulist)
            atomindex = i
    ru0 = newpos[atomindex][0]
    for ub in oldpos[atomindex]:
        trans = inhalf(ub - ru0)
        foundmap = True
        # now check against all the others, and construct the mapping
        indexmap = []
        for atomlist0, spinlist0, atomlist1, spinlist1 in zip(oldpos, oldspins, newpos, newspins):
            # work through the "new" positions
            if not foundmap: break
            maplist = []
            for rua, sp1 in zip(atomlist1, spinlist1):
                for j, uj, sp0 in zip(itertools.count(), atomlist0, spinlist0):
                    if not np.allclose(sp0, sp1, atol=threshold): continue  # only allow maps that have same spin
                    if np.allclose(inhalf(uj - rua - trans), 0, atol=threshold):
                        maplist.append(j)
                        break
            if len(maplist) != len(atomlist0):
                foundmap = False
            else:
                indexmap.append(tuple(maplist))
        if foundmap: break
    if foundmap:
        return trans, tuple(indexmap)
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
    :param indexmap: tuples of tuples, containing the atom mapping
    """

    def incell(self):
        """Return a version of groupop where the translation is in the unit cell"""
        return GroupOp(self.rot, incell(self.trans), self.cartrot, self.indexmap)

    def inhalf(self):
        """Return a version of groupop where the translation is in the centered unit cell"""
        return GroupOp(self.rot, inhalf(self.trans), self.cartrot, self.indexmap)

    @classmethod
    def ident(cls, basis):
        """Return a group operation corresponding to identity for a given basis"""
        return cls(rot=np.eye(3, dtype=int), trans=np.zeros(3), cartrot=np.eye(3),
                   indexmap=tuple(tuple(i for i in range(len(atomlist))) for atomlist in basis))

    def __str__(self):
        """Human-readable version of groupop"""
        str_rep = "#Rotation (lattice, cartesian):\n {}\t{}\n {}\t{}\n".format(
            self.rot[0], self.cartrot[0],
            self.rot[1], self.cartrot[1])
        if self.rot.shape == (3,3):
            str_rep += " {}\t{}\n".format(self.rot[2], self.cartrot[2])
        str_rep += "#Translation: {}\n#Indexmap:".format(self.trans)
        for chemind, atoms in enumerate(self.indexmap):
            for origind, finalind in enumerate(atoms):
                str_rep = str_rep + "\n  {chem}.{o} -> {chem}.{f}".format(chem=chemind,
                                                                          o=origind, f=finalind)
        return str_rep

    def _asdict(self):
        """Return a proper dict"""
        return {'rot': self.rot,
                'trans': self.trans,
                'cartrot': self.cartrot,
                'indexmap': self.indexmap}

    def __eq__(self, other):
        """Test for equality--we use numpy.isclose for comparison, since that's what we usually care about"""
        return isinstance(other, self.__class__) and \
               np.all(self.rot == other.rot) and \
               np.allclose(self.trans, other.trans) and \
               np.allclose(self.cartrot, other.cartrot) and \
               self.indexmap == other.indexmap

    def __ne__(self, other):
        """Inequality == not __eq__"""
        return not self.__eq__(other)

    def __hash__(self):
        """Hash, so that we can make sets of group operations"""
        ### we are a little conservative, and only use the rotation and indexmap to define the hash. This means
        ### we will get collisions for the same rotation but different unit cell translations. The reason is
        ### that __eq__ uses "isclose" on our translations, and we don't have a good way to handle
        ### that in a hash function. We lose a little bit on efficiency if we construct a set that
        ### has a whole lot of translation operations, but that's not usually what we will do.
        # return hash(self.rot.data.tobytes())
        return hash(self.rot.data.tobytes()) ^ hash(self.indexmap)

    def __add__(self, other):
        """Add a translation to our group operation"""
        if __debug__:
            if type(other) is not np.ndarray: raise TypeError('Can only add a translation to a group operation')
            if other.shape != (self.rot.shape[0],):
                raise IndexError('Can only add a {} dimensional vector'.format(self.rot.shape[0]))
            if not np.issubdtype(other.dtype, np.integer): raise TypeError('Can only add a lattice vector translation')
        return GroupOp(self.rot, self.trans + other, self.cartrot, self.indexmap)

    def __sub__(self, other):
        """Add a (negative) translation to our group operation"""
        return self.__add__(-other)

    def __mul__(self, other):
        """Multiply two group operations to produce a new group operation"""
        if __debug__:
            if type(other) is not GroupOp: return NotImplemented
        return GroupOp(np.dot(self.rot, other.rot),
                       np.dot(self.rot, other.trans) + self.trans,
                       np.dot(self.cartrot, other.cartrot),
                       tuple(tuple(atomlist0[i] for i in atomlist1)
                             for atomlist0, atomlist1 in zip(self.indexmap, other.indexmap)))

    def __sane__(self):
        """Return true if the cartrot and rot are consistent and 'sane'"""
        tr = self.rot.trace()
        det = np.int(np.round(np.linalg.det(self.rot)))
        # consistency:
        if np.int(np.round(self.cartrot.trace())) != tr: return False
        if np.int(np.round(np.linalg.det(self.cartrot))) != det: return False
        # sanity:
        if abs(det) != 1: return False
        dimshift = 0 if self.rot.shape[0] == 3 else -1
        if det * tr < (-1+dimshift) or det * tr > (3+dimshift): return False
        return True

    def inv(self):
        """Construct and return the inverse of the group operation"""
        inverse = (np.round(np.linalg.inv(self.rot))).astype(int)
        return GroupOp(inverse,
                       -np.dot(inverse, self.trans),
                       self.cartrot.T,
                       tuple(tuple(x for i, x in sorted([(y, j) for j, y in enumerate(atomlist)]))
                             for atomlist in self.indexmap))

    @staticmethod
    def optype(rot):
        """Returns the type of group operation (single integer):
        1 = identity
        2, 3, 4, 6 = n- fold rotation around an axis
        negative = rotation + mirror operation, perpendicular to axis
        "special cases": -1 = mirror, -2 = inversion

        :param rot: rotation matrix (can be the integer rot)
        :return type: integer
        """
        # dim = rot.shape[0]
        dimindexpos, dimindexneg = (1, 3) if rot.shape[0] == 3 else (2, 4)
        tr = np.int(rot.trace())
        if np.linalg.det(rot) > 0:
            return (2, 3, 4, 6, 1)[tr + dimindexpos]  # trace determines the rotation type [tr + 1] for 3d
        else:
            return (-2, -3, -4, -6, -1)[tr + dimindexneg]  # trace determines the rotation type [tr + 3] fpr 3d

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

        :return type: integer
        :return eigenvectors: list of [ev0, ev1, ev2]
        """
        if __debug__:
            if not self.__sane__():
                raise ValueError('Bad GroupOp:\n{}'.format(self))
        optype = self.optype(self.rot)
        det = 1 if optype > 0 else -1
        tr = np.int(self.rot.trace())
        # two trivial cases: identity, inversion:
        if optype == 1 or optype == -2:
            return optype, np.eye(self.rot.shape[0])
        if self.rot.shape[0] == 2:
            if optype != -1:
                return optype, np.eye(self.rot.shape[0])
            # only interesting case is how to deal with is the mirror plane; find the angle of the mirror
            phi = 0.5*np.arctan2(self.cartrot[0,1]+self.cartrot[1,0], self.cartrot[0,0]-self.cartrot[1,1])
            return optype, np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        # otherwise, there's an axis to find:
        vmat = np.eye(3)
        vsum = np.zeros((3, 3))
        if det > 0:
            for n in range(optype):
                vsum += vmat
                vmat = np.dot(self.cartrot, vmat)
        else:
            for n in range((0, 6, 4, 3, 2)[tr + 3]):  #
                vsum += vmat
                vmat = -np.dot(self.cartrot, vmat)
        # vmat *should* equal identity if we didn't fail...
        if __debug__:
            if not np.allclose(vmat, np.eye(3)): raise ArithmeticError('eigenvalue analysis fail')
        vsum *= 1. / n
        # now the columns of vsum should either be (a) our rotation / mirror axis, or (b) zero
        eig0 = vsum[:, 0]
        magn0 = np.dot(eig0, eig0)
        if magn0 < 1e-2:
            eig0 = vsum[:, 1]
            magn0 = np.dot(eig0, eig0)
            if magn0 < 1e-2:
                eig0 = vsum[:, 2]
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
        # had to rewrite _asdict() for some reason...?
        return dumper.represent_mapping(GROUPOP_YAMLTAG, data._asdict())

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
    :return dim: dimensionality, 0..3
    :return vect: vector defining line direction (1) or plane normal (2)
    """
    # 2d first
    if len(eigenvect) == 2:
        if rottype == 1: return (2, np.zeros(2))  # sphere (identity)
        if rottype == -1: return (1, eigenvect[0])  # plane (pure mirror)
        return (0, np.zeros(2))  # all others are rotation, which leaves nothing unchanged in 2d
    # edge cases first:
    if rottype == 1: return (3, np.zeros(3))  # sphere (identity)
    if rottype == -2: return (0, np.zeros(3))  # point (inversion)
    if rottype == -1: return (2, eigenvect[0])  # plane (pure mirror)
    return (1, eigenvect[0])  # line (all others--there's a rotation axis involved


def SymmTensorBasis(rottype, eigenvect):
    """
    Returns a symmetric second-rank tensor basis corresponding to the optype and eigenvectors
    for a GroupOp

    :param rottype: output from eigen()
    :param eigenvect: eigenvectors
    :return tensorbasis: list of 2nd-rank symmetric tensors making up the basis
    """

    def SymmTensor1(v1):
        """Make a normalized, symmetric tensor from two vectors"""
        return np.outer(v1, v1)

    def SymmTensor2(v1, v2):
        """Make a normalized, symmetric tensor from two vectors"""
        return (np.outer(v1, v1) + np.outer(v2, v2)) / np.sqrt(2)

    def SymmTensorCross(v1, v2):
        """Make a normalized, symmetric tensor from two vectors"""
        return (np.outer(v1, v2) + np.outer(v2, v1)) / np.sqrt(2)

    # 2d first:
    if len(eigenvect) == 2:
        if rottype == 1 or rottype == -2:
            return [SymmTensor1(np.array([1.,0.])), SymmTensor1(np.array([0.,1.])),
                    SymmTensorCross(np.array([1.,0]), np.array([0.,1.]))]
        if rottype == -1:
            return [SymmTensor1(eigenvect[0]), SymmTensor1(eigenvect[1])]
        # rotations kill everything except the isotropic case:
        return [SymmTensor2(np.array([1.,0.]), np.array([0.,1.]))]

    if rottype == 1 or rottype == -2:
        # identity / inversion: all symmetric tensors
        return [SymmTensor1(np.array([1., 0., 0.])),  # xx
                SymmTensor1(np.array([0., 1., 0.])),  # yy
                SymmTensor1(np.array([0., 0., 1.])),  # zz
                SymmTensorCross(np.array([0., 1., 0.]), np.array([0., 0., 1.])),  # yz
                SymmTensorCross(np.array([1., 0., 0.]), np.array([0., 0., 1.])),  # zx
                SymmTensorCross(np.array([1., 0., 0.]), np.array([0., 1., 0.]))]  # xy
    if rottype == -1 or rottype == 2:
        # mirror plane or 2-fold rotation:
        # 4 symmetric tensors: e0 x e0, e1 x e1, e2 x e2, e1 x e2
        return [SymmTensor1(eigenvect[0]),
                SymmTensor1(eigenvect[1]),
                SymmTensor1(eigenvect[2]),
                SymmTensorCross(eigenvect[1], eigenvect[2])]
    # else: 3-, 4-, or 6-fold rotation (with or without mirror):
    # 2 symmetric tensors: e0 x e0, e1 x e1 + e2 x e2
    return [SymmTensor1(eigenvect[0]),
            SymmTensor2(eigenvect[1], eigenvect[2])]


def CombineVectorBasis(b1, b2):
    """
    Combines (intersects) two vector spaces into one.

    :param b1: (dim, vect) -- dimensionality (0..3), vector defining line direction (1) or plane normal (2)
    :param b2: (dim, vect)
    :return dim: dimensionality, 0..3
    :return vect: vector defining line direction (1) or plane normal (2)
    """
    # edge cases first
    if b1[0] == 0: return b1  # point with anything
    if b2[0] == 0: return b2
    # 2d first:
    if b1[1].shape[0] == 2:
        if b1[0] == 2: return b2
        if b2[0] == 2: return b1
        # all that remains now is b1[0] == b2[0] == 1 (two lines)
        if abs(np.dot(b1[1], b2[1])) > (1. - 1e-8):  # parallel vectors
            return (0, np.zeros(2))
        else:  # two parallel lines
            return b1

    if b1[0] == 3: return b2  # sphere with anything
    if b2[0] == 3: return b1
    if b1[0] == b2[0]:
        if abs(np.dot(b1[1], b2[1])) > (1. - 1e-8):  # parallel vectors
            return b1  # equal bases
        else:  # vectors not equal...
            if b1[0] == 1:  # for a line, that's death:
                return (0, np.zeros(3))
            else:  # for a plane, need the mutual line:
                v = np.cross(b1[1], b2[1])
                return (1, v / np.sqrt(np.dot(v, v)))
    # finally: one is a plane, other is a line:
    if abs(np.dot(b1[1], b2[1])) > 1e-8:  # if the vectors are not perpendicular, death:
        return (0, np.zeros(3))
    else:  # return whichever is a line:
        if b1[0] == 1:
            return b1
        else:
            return b2


def CombineTensorBasis(b1, b2, symmetric=True):
    """
    Combines (intersects) two tensor spaces into one; uses SVD to compute null space.

    :param b1: list of tensors
    :param b2: list of tensors
    :return tensorbasis: list of 2nd-rank symmetric tensors making up the basis
    """
    # edge cases first (empty or full basis)
    if len(b1) == 0: return b1
    if len(b2) == 0: return b2
    if len(b1) == b1[0].size: return b2
    if len(b2) == b2[0].size: return b1
    # make the combined matrix with the two column spaces D = [b1 b2], then
    # find its nullspace
    u, s, vh = np.linalg.svd(np.array([v.flatten() for v in b1] + [v.flatten() for v in b2]).T)
    # this is sneaky: the first is to pull out the size of the nullspace, the second slices
    # the part of b1 that we have to deal with, but then we have to *renormalize* these vectors
    # by multiplying by sqrt(2), since the slice in each vector space would be normalized.
    nullspace = vh[sum(s >= 1e-8):, 0:len(b1)] * np.sqrt(2)
    # now to reconstruct our normalized basis from those
    # list comprehension to run over the elements of our nullspace, and the
    # generator in the sum to construct the basis
    return [sum(b1[i] * alpha for i, alpha in enumerate(v)) for v in nullspace]


def ProjectTensorBasis(tensor, basis):
    """
    Given a tensor, project it onto the basis.

    :param tensor: tensor
    :param basis: list consisting of an orthonormal basis
    :return tensor: tensor, projected
    """
    if __debug__:
        if tensor.shape != basis[0].shape: raise TypeError("Tensor and basis not compatible")
    return sum(b * np.sum(tensor * b) for b in basis)


def Voigtstrain(e1, e2, e3, e4, e5, e6):
    """
    Returns a symmetric strain tensor from the Voigt reduced strain values.

    :param e1: xx
    :param e2: yy
    :param e3: zz
    :param e4: yz + zx
    :param e5: zx + xz
    :param e6: xy + yx
    :return strain: symmetric strain tensor
    """
    return np.array([[e1, 0.5 * e6, 0.5 * e5], [0.5 * e6, e2, 0.5 * e4], [0.5 * e5, 0.5 * e4, e3]])


def isotropicFourthRank(average, shear):
    """
    Returns a symmetrized, isotropic fourth-rank tensor based on an average value and "shear" value

    :param average: averaged value = (F11+2F12)/3
    :param shear: shear value = F44 = (F11-F12)/2
    :return F[a,b,c,d]: isotropic fourth-rank tensor
    """
    F = np.zeros((3,3,3,3))
    F11, F12, F44 = average + 4*shear/3, average - 2*shear/3, shear
    for a in range(3):
        F[a,a,a,a] = F11
        for b in range(3):
            if b != a:
                F[a,a,b,b] = F12
                F[a,b,a,b] = F44
                F[a,b,b,a] = F44
    return F


def FourthRankIsotropic(F):
    """
    Returns the average and shear values from orientational averaging of a symmetric fourth-rank
    tensor.

    :param F[a,b,c,d]: symmetric fourth-rank tensor (F[abcd]=F[abcd]=F[bacd]=F[cdab])
    :return average: average value = (F11+2F12)/3, orientationally averaged
    :return shear: shear value = F44, orientationally averaged
    """
    average = sum(F[a,a,b,b] for a in range(3) for b in range(3))/9
    shear = ((F[0,0,0,0] + F[1,1,1,1] + F[2,2,2,2] - F[0,0,1,1] - F[0,0,2,2] - F[1,1,2,2])/3 +
             F[0,1,0,1] + F[0,2,0,2] + F[1,2,1,2])/5
    return average, shear


# TODO: Add the ability to explicitly specify "metastable" states
# that should be considered the same chemistry, but not subject to reduction
class Crystal(object):
    """
    A class that defines a crystal, as well as the symmetry analysis that goes along with it.
    Now includes optional spins. These can be vectors or "scalar" spins, for which we need
    to consider a phase factor. In general, they can be complex. Ideally, they should have
    magnitude either 0 or 1.

    Specified by a lattice (3 vectors), a basis (list of lists of positions in direct coordinates).
    Can also name the elements (chemistry), and specify spin degrees of freedom.
    """

    def __init__(self, lattice, basis, chemistry=None, spins=None,
                 NOSYM=False, noreduce=False, threshold=1e-8):
        """
        Initialization; starts off with the lattice vector definition and the
        basis vectors. While it does not explicitly store the specific chemical
        elements involved, it does store that there are different elements.

        :param lattice: array[3,3] or list of array[3] (or 2 if 2-dimensional)
            lattice vectors; if [3,3] array, then the vectors need to be in *column* format
            so that the first lattice vector is lattice[:,0]
        :param basis: list of array[3] or list of list of array[3] (or 2 if 2-dimensional)
            crystalline basis vectors, in unit cell coordinates. If a list of lists, then
            there are multiple chemical elements, with each list corresponding to a unique
            element
        :param chemistry: (optional) list of names of chemical elements
        :param spins: (optional) list of numbers (complex) / vectors or list of list of same
            spins for individual atoms; if not None, needs to match the basis. Can either be
            scalars or vectors, corresponding to collinear or non-collinear magnetism
        :param NOSYM: turn off all symmetry finding (except identity)
        :param noreduce: do not attempt to reduce the atomic basis
        :param threshold: threshold for symmetry equivalence (stored in the class)
        """
        # Do some basic type checking and "formatting"
        self.lattice = None
        if type(lattice) is list:
            if len(lattice) != 3 and len(lattice) != 2:
                raise TypeError('lattice is a list, but does not contain 2 or 3 members')
            self.lattice = np.array(lattice).T
        if type(lattice) is np.ndarray:
            self.lattice = np.array(lattice)
        if self.lattice is None: raise TypeError('lattice is not a recognized type')
        if self.lattice.shape != (3, 3) and self.lattice.shape != (2, 2):
            raise TypeError('lattice contains vectors that are not 2 or 3 dimensional')
        self.dim = self.lattice.shape[0] # dimensionality of our lattice
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
        if spins is not None:
            if type(spins) is not list: raise TypeError('spins needs to be a list or list of lists')
            if type(spins[0]) is list:
                self.spins = copy.deepcopy(spins)
            else:
                self.spins = [copy.deepcopy(spins)]
        else:
            self.spins = None
        self.threshold = threshold
        if not noreduce: self.reduce()  # clean up basis as needed
        self.minlattice()  # clean up lattice vectors as needed
        self.invlatt = np.linalg.inv(self.lattice)
        # this lets us, in a flat list, enumerate over indices of atoms as needed
        self.atomindices = [(atomtype, atomindex)
                            for atomtype, atomlist in enumerate(self.basis)
                            for atomindex in range(len(atomlist))]
        self.N = len(self.atomindices)
        self.Nchem = len(self.basis)
        if chemistry is None:
            self.chemistry = ['{}'.format(i) for i in range(self.Nchem)]
        else:
            self.chemistry = chemistry.copy()
        self.volume, self.metric = self.calcmetric()
        self.reciplatt = 2. * np.pi * self.invlatt.T
        self.BZvol = abs(float(np.linalg.det(self.reciplatt)))
        self.BZG = self.genBZG()
        self.center()  # should do before gengroup so that inversion is centered at origin
        if NOSYM:
            self.G = frozenset([GroupOp.ident(self.basis)])
        else:
            self.G = self.gengroup()  # do before genpoint
        self.pointG = self.genpoint()
        self.Wyckoff = self.genWyckoffsets()

    def __repr__(self):
        """String representation of crystal (lattice + basis)"""
        return 'Crystal(' + repr(self.lattice).replace('\n', '').replace('\t', '') + ',' + \
               repr(self.basis) + ', spins=' + repr(self.spins) + \
               ', chemistry=' + repr(self.chemistry) + ')'

    def __str__(self):
        """Human-readable version of crystal (lattice + basis)"""
        str_rep = "#Lattice:\n  a1 = {}\n  a2 = {}\n".format(
            self.lattice.T[0], self.lattice.T[1])
        if self.dim > 2:
            str_rep += "  a3 = {}\n".format(self.lattice.T[2])
        str_rep += "#Basis:"
        for chemind, atoms in enumerate(self.basis):
            for atomind, pos in enumerate(atoms):
                if self.spins is None:
                    s = ''
                else:
                    s = ' sp={}'.format(self.spins[chemind][atomind])
                str_rep = str_rep + "\n  ({}) {}.{} = {}{}".format(self.chemistry[chemind],
                                                                   chemind, atomind, pos, s)
        return str_rep

    @classmethod
    def fromdict(cls, yamldict):
        """
        Creates a Crystal object from a *very simple* YAML-created dictionary

        :param yamldict: dictionary; must contain 'lattice' (using *row* vectors!) and 'basis';
            can contain optional 'lattice_constant'
        :return Crystal(lattice.T, basis): new crystal object
        """
        if 'lattice' not in yamldict: raise IndexError('{} does not contain "lattice"'.format(yamldict))
        if 'basis' not in yamldict: raise IndexError('{} does not contain "basis"'.format(yamldict))
        lattice_constant = 1.
        if 'lattice_constant' in yamldict: lattice_constant = yamldict['lattice_constant']
        return cls((lattice_constant * yamldict['lattice']).T, yamldict['basis'],
                   chemistry=(yamldict['chemistry'] if 'chemistry' in yamldict else None),
                   spins=(yamldict['spins'] if 'spins' in yamldict else None),
                   threshold=(yamldict['threshold'] if 'threshold' in yamldict else 1e-8))

    def simpleYAML(self, a0=1.0):
        """
        Creates a simplified YAML dump, in case we don't want to output the full symmetry analysis

        :return YAML: string dump
        """
        return yaml.dump({'lattice_constant': a0,
                          'lattice': self.lattice.T / a0,
                          'basis': self.basis,
                          'spins': self.spins,
                          'chemistry': self.chemistry,
                          'threshold': self.threshold})

    def chemindex(self, chemistry):
        """
        Return index corresponding to chemistry; None if not present.

        :param chemistry: value to check
        :return index: corresponding to chemistry
        """
        try:
            return self.chemistry.index(chemistry)
        except:
            return None

    # a few "convenient" lattices
    @classmethod
    def FCC(cls, a0, chemistry=None):
        """
        Create a face-centered cubic crystal with lattice constant a0

        :param a0: lattice constant
        :return FCC crystal:
        """
        if chemistry is None or isinstance(chemistry, (list, tuple)):
            chem = chemistry
        else:
            chem = [chemistry]
        return cls(np.array([[0., 0.5, 0.5], [0.5, 0., 0.5], [0.5, 0.5, 0.]]) * a0, [np.zeros(3)], chem)

    @classmethod
    def BCC(cls, a0, chemistry=None):
        """
        Create a body-centered cubic crystal with lattice constant a0

        :param a0: lattice constant
        :return BCC crystal:
        """
        if chemistry is None or isinstance(chemistry, (list, tuple)):
            chem = chemistry
        else:
            chem = [chemistry]
        return cls(np.array([[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]]) * a0, [np.zeros(3)], chem)

    @classmethod
    def HCP(cls, a0, c_a=np.sqrt(8. / 3.), chemistry=None):
        """
        Create a hexagonal closed packed crystal with lattice constant a0, c/a ratio c_a

        :param a0: lattice constant
        :param c_a: (optional) c/a ratio, default=ideal :math:`\sqrt{8/3}`
        :return HCP crystal:
        """
        if chemistry is None or isinstance(chemistry, (list, tuple)):
            chem = chemistry
        else:
            chem = [chemistry]
        return cls(np.array([[0.5, 0.5, 0.],
                             [-np.sqrt(0.75), np.sqrt(0.75), 0.],
                             [0., 0., c_a]]) * a0,
                   [np.array([1. / 3., 2. / 3., 1. / 4.]),
                    np.array([2. / 3., 1. / 3., 3. / 4])], chem)

    def __iszero__(self, v):
        return np.allclose(v, 0, atol=self.threshold)

    def __isclose__(self, a, b):
        return np.allclose(a, b, atol=self.threshold)

    def center(self):
        """Center the atoms in the cell if there is an inversion operation present."""
        # trivial case:
        if self.N == 1:
            self.basis = [[np.zeros(self.dim)]]
            return
        # else, invert positions!
        trans, indexmap = maptranslation(self.basis, [[-u for u in atomlist] for atomlist in self.basis])
        if indexmap is None:
            return
        # translate by -1/2 * trans for inversion
        self.basis = [[incell(u - 0.5 * trans) for u in atomlist] for atomlist in self.basis]
        # now, check for "aesthetics" of our basis choice
        shift = np.zeros(self.dim)
        for d in range(self.dim):
            if np.any([np.isclose(u[d], 0, atol=self.threshold)
                       for atomlist in self.basis for u in atomlist]):
                shift[d] = 0
            elif np.any([np.isclose(u[d], 0.5, atol=self.threshold)
                         for atomlist in self.basis for u in atomlist]):
                shift[d] = 0.5
            elif sum([1 for atomlist in self.basis for u in atomlist if u[d] < 0.25 or u[d] > 0.75]) > self.N / 2:
                shift[d] = 0.5
        self.basis = [[incell(atom + shift) for atom in atomlist] for atomlist in self.basis]

    def reduce(self, threshold=None):
        """
        Reduces the lattice and basis, if needed. Works (tail) recursively.

        :param threshold: threshold for symmetry comparison; default = self.threshold

        Algorithm is slightly complicated: we attempt to identify if there is a internal
        translation symmetry in the crystal (called `t`) that applies to all sites. Once identified,
        we transform the lattice vectors and basis into the "reduced" form of the cell. We use
        tail recursion to continue until no further reduction is possible. Will usually require
        some "polishing" on the unit cell after the fact.

        We try to do this efficiently: we check the GCD of the site counts (called `M`); if it's 1,
        we kick out. We check translations against the smallest site set first.

        We try to do this carefully: We make sure that our translation can be expressed rationally
        with `M` as the denominator; this helps protect against roundoff error. When we reduce the
        atomic basis, we *average* the values that match. Finally, as we reduce, we also change the
        `self.threshold` value accordingly so that recursion uses the same "effective" threshold.
        """
        if threshold is not None:
            self.threshold = threshold
        sitecount = [len(ulist) for ulist in self.basis]
        M = gcdlist(sitecount)
        if M==1: return
        atomindex = min(range(len(sitecount)), key=sitecount.__getitem__) # index of shortest sitecount
        # if we don't have spins, just make a big list of lists of 0, otherwise there's too many "if spins None..."
        if self.spins is None:
            spins = [[0 for u in atomlist] for atomlist in self.basis]
        else:
            spins = self.spins
        # We need to first check against reducibility of atomic positions: try out non-trivial displacements
        initpos, initsp = self.basis[atomindex][0], spins[atomindex][0]
        trans = False
        for newpos, newsp in zip(self.basis[atomindex], spins[atomindex]):
            t = newpos - initpos
            if np.allclose(t, 0): continue
            if not self.__isclose__(initsp, newsp): continue
            # reconstruct `t` as a rational vector; if fail, kick out
            T = np.around(M*t).astype(int)
            if not self.__isclose__(t, T/M): continue
            t = T/M
            trans = True
            for atomlist, spinlist in zip(self.basis, spins):
                for u, s in zip(atomlist, spinlist):
                    # edited to only check against translations with the same spin:
                    if np.all([not self.__iszero__(inhalf(u + t - v))
                               for v, vs in zip(atomlist, spinlist)
                               if self.__isclose__(s, vs)]):
                        trans = False
                        break
            if trans: break
        # end the recursion here:
        if not trans: return
        # reduce that lattice and basis
        # 1. determine what the new lattice needs to look like.
        # m = index of smallest non-zero value in T:
        m = min([i for (i, v) in enumerate(T) if v != 0], key=lambda n: abs(T[n]))
        # i, j = other indices, ordered so that T, e_i, e_j == right-handed coordinate system
        if self.dim == 3:
            if T[m] > 0:
                i, j = (m+1)%3, (m+2)%3
            else:
                i, j = (m+2)%3, (m+1)%3
            # new lattice: A0 = [a]*t, A1 = a_i, A2 = a_j
            self.lattice = np.array([np.dot(self.lattice, t),
                                     self.lattice[:, i],
                                     self.lattice[:, j]]).T
        else:
            # 2-d
            i = (m+1)%2
            # new lattice: A0 = [a]*t, A1 = a_i
            self.lattice = np.array([np.dot(self.lattice, t),
                                     self.lattice[:, i]]).T
        reduction = abs(T[m]/M)
        mult = np.array([M/T[m], T[i]/T[m], T[j]/T[m]]) if self.dim == 3 else np.array([M/T[m], T[i]/T[m]])

        # 2. update the basis
        self.threshold *= abs(mult[0]) # need to update *before* doing matching below:
        newbasis = []
        newspins = []
        for atomlist, spinlist in zip(self.basis, spins):
            newatomlist = []
            avedisplist = []
            newspinlist = []
            for u, s in zip(atomlist, spinlist):
                v = incell(np.array([u[m]*mult[0],
                                     u[i] - u[m]*mult[1],
                                     u[j] - u[m]*mult[2]])) if self.dim == 3 else \
                    incell(np.array([u[m]*mult[0],
                                     u[i] - u[m]*mult[1]]))
                ind = 0
                for v1 in newatomlist:
                    # dv = relative displacement of site
                    dv = inhalf(v-v1)
                    if self.__iszero__(dv): break
                    ind += 1
                if ind<len(newatomlist):
                    # matched position: accumulate displacement and spin
                    avedisplist[ind] += dv
                    newspinlist[ind] += s
                else:
                    # unmatched position!
                    newatomlist.append(v)
                    avedisplist.append(np.zeros(self.dim))
                    newspinlist.append(s)
            if len(newatomlist)*(M//abs(T[m])) != len(atomlist):
                raise ArithmeticError('Reduction did not produce correct reduced basis: {}*{} != {}'.format(len(newatomlist), M//abs(T[m]), len(atomlist)))
            newbasis.append([incell(v+reduction*dv) for v, dv in zip(newatomlist, avedisplist)])
            newspins.append([reduction*s for s in newspinlist])
        self.basis = newbasis
        if self.spins is not None: self.spins = newspins
        # 3. tail recursion:
        self.reduce()

    def remapbasis(self, supercell):
        """
        Takes the basis definition, and using a supercell definition, returns a new basis

        :param supercell: integer array[3,3]
        :return atomic basis: list of list of positions
        """
        invsuper = np.linalg.inv(supercell)
        return [[incell(np.dot(invsuper, u)) for u in atomlist] for atomlist in self.basis]

    def minlattice(self):
        """
        Try to find the optimal lattice vector definition for a crystal. Our definition of optimal
        is (a) length of each lattice vector is minimal; (b) the vectors are ordered from
        shortest to longest; (c) the vectors have minimal dot product; (d) the basis is right-handed.

        Works recursively, and in-place.
        """
        magnlist = sorted((np.dot(v, v), idx) for idx, v in enumerate(self.lattice.T))
        super = np.zeros((self.dim, self.dim), dtype=int)
        for i, (magn, j) in enumerate(magnlist):
            super[j, i] = 1
        # check that we have a right-handed lattice
        if np.linalg.det(self.lattice) * np.linalg.det(super) < 0:
            super[:, -1] = -super[:, -1]
        if not np.all(super == np.eye(self.dim, dtype=int)):
            self.lattice = np.dot(self.lattice, super)
            self.basis = self.remapbasis(super)

        super = np.eye(self.dim, dtype=int)
        modified = False
        # check the possible vector reductions (edited to handle 2 and 3 dimensions)
        asq = np.dot(self.lattice.T, self.lattice)
        u = np.around(asq[0, 1] / asq[0, 0])
        if u != 0:
            super[0, 1] = -int(u)
            modified = True
        elif self.dim > 2:
            u = np.around(asq[0, 2] / asq[0, 0])
            if u != 0:
                super[0, 2] = -int(u)
                modified = True
            else:
                u = np.around(asq[1, 2] / asq[1, 1])
                if u != 0:
                    super[1, 2] = -int(u)
                    modified = True

        if not modified:
            return
        self.lattice = np.dot(self.lattice, super)
        self.basis = self.remapbasis(super)
        self.minlattice()

    def calcmetric(self):
        """
        Computes the volume of the cell and the metric tensor

        :return volume: cell volume
        :return metric tensor: 3x3
        """
        return abs(float(np.linalg.det(self.lattice))), np.dot(self.lattice.T, self.lattice)

    def inBZ(self, vec, BZG=None, threshold=1e-5):
        """
        Tells us if vec is inside our set of defining points.

        :param vec: array [3], vector to be tested
        :param BGZ: array [:,3], optional (default = self.BZG), array of vectors that define the BZ
        :param threshold: double, optional, threshold to use for "equality"
        :return inBZ: False if outside the BZ, True otherwise
        """
        if BZG is None: BZG = self.BZG
        # checks that vec.G < G^2 for all G (and throws out the option that vec == G, in case threshold == 0)
        return all(np.dot(vec, G) < (np.dot(G, G) + threshold) for G in BZG if not np.all(vec == G))

    def genBZG(self):
        """
        Generates the reciprocal lattice G points that define the Brillouin zone.

        :return Garray: array of G vectors that define the BZ, in Cartesian coordinates
        """
        # Start with a list of possible vectors; add those that define the BZ...
        BZG = []
        for nv in itertools.product(range(-3, 4), repeat = self.dim):
            if all(n == 0 for n in nv): continue
            vec = np.dot(self.lattice, nv)
            if self.inBZ(vec, BZG, threshold=0): BZG.append(np.dot(self.reciplatt, nv))
        # ... and use a list comprehension to only keep those that still remain
        return np.array([0.5 * vec for vec in BZG if self.inBZ(vec, BZG, threshold=0)])

    def gengroup(self):
        """
        Generate all of the space group operations. Now handles spins! Doesn't store
        spin phase factors for each group operation, though.

        :return Gset: frozenset of group operations
        """

        def rootsofunity(optype):
            """Return an iterable of roots of unity to try for GroupOp type optype"""
            # always include negation
            rot2, rot4, rot6 = (1, -1), (1, -1, 1j, -1j), tuple(np.exp(n * np.pi * 2j / 6) for n in range(6))
            return (rot2, rot2, rot6, rot4, None, rot6)[abs(optype) - 1]  # (+-1, +-2, +-3, +-4, .., +-6)

        def quickabsdet(M):
            if M.shape == (2,2): return abs(M[0,0]*M[1,1]-M[0,1]*M[1,0])
            if M.shape == (3,3): return abs(M[0,0]*(M[1,1]*M[2,2]-M[1,2]*M[2,1])
                                            -M[0,1]*(M[1,0]*M[2,2]-M[1,2]*M[2,0])
                                            +M[0,2]*(M[1,0]*M[2,1]-M[1,1]*M[2,0]))

        groupops = []
        supercellvect = [np.array(nv) for nv in itertools.product(range(-1,2), repeat=self.dim)
                         if any(n != 0 for n in nv)]
        matchvect = [[u for u in supercellvect
                      if self.__isclose__(np.dot(u, np.dot(self.metric, u)),
                                          self.metric[d, d])] for d in range(self.dim)]
        # if we don't have spins, just make a big list of lists of 0, otherwise there's too many "if spins None..."
        if self.spins is None:
            spins = [[0 for u in atomlist] for atomlist in self.basis]
        else:
            spins = self.spins
        for supertuple in itertools.product(*matchvect):
            supercell = np.array(supertuple).T
            if quickabsdet(supercell) != 1: continue
            if self.__isclose__(np.dot(supercell.T, np.dot(self.metric, supercell)), self.metric):
                # possible operation--need to check the atomic positions with spin phase factors
                optype = GroupOp.optype(supercell)
                cartrot = np.dot(self.lattice, np.dot(supercell, self.invlatt))
                detrot = 1 if optype > 0 else -1
                # apply cartesian rotation to spins... if they're vectors; else, do nothing
                rotspins = [[detrot * s if isinstance(s, Number) else np.dot(cartrot, s)
                             for s in spinlist]
                            for spinlist in spins]
                # if det * tr < -1 or det * tr > 3: return False
                for phase in rootsofunity(optype):
                    newspins = [[phase * s for s in spinlist] for spinlist in rotspins]
                    trans, indexmap = maptranslation(self.basis,
                                                     [[np.dot(supercell, u)
                                                       for u in atomlist]
                                                      for atomlist in self.basis],
                                                     spins, newspins, threshold=self.threshold)
                    if indexmap is not None:
                        groupops.append(GroupOp(supercell,
                                                trans,
                                                cartrot,
                                                indexmap))
        return frozenset(groupops)

    def strain(self, eps):
        """
        Returns a new Crystal object that is a strained version of the current.

        :param eps: strain tensor
        :return Crystal: new crystal object, strained
        """
        if __debug__:
            if type(eps) is not np.ndarray or eps.shape != (self.dim, self.dim):
                raise TypeError('strain is not a 3x3 tensor')
        return Crystal(np.dot(np.eye(self.dim) + eps, self.lattice), self.basis,
                       chemistry=self.chemistry, spins=self.spins, threshold=self.threshold)

    def addbasis(self, basis, chemistry=None, spins=None):
        """
        Returns a new Crystal object that contains additional sites (assumed to be new chemistry).
        This is intended to "add in" interstitial sites. Note: if the symmetry is to be
        maintained, should be the output from Wyckoffpos().

        :param basis: list (or list of lists) of new sites
        :param chemistry: (optional) list of chemistry names
        :param spins: (optional) list of spins
        :return Crystal: new crystal object, with additional sites
        """
        if type(basis) is not list: raise TypeError('basis needs to be a list or list of lists')
        if spins is not None and type(spins) is not list: raise TypeError('spins needs to be a list or list of lists')
        if type(basis[0]) == np.ndarray:
            for u in basis:
                if type(u) is not np.ndarray: raise TypeError("{} in {} is not an array".format(u, basis))
            newbasis = [[incell(u) for u in basis]]
        else:
            for elem in basis:
                if type(elem) is not list: raise TypeError("{} in basis is not a list".format(elem))
                for u in elem:
                    if type(u) is not np.ndarray: raise TypeError("{} in {} is not an array".format(u, elem))
            newbasis = [[incell(u) for u in atombasis] for atombasis in basis]
        if chemistry is None:
            newchemistry = self.chemistry + [i + self.Nchem for i in range(len(newbasis))]
        else:
            newchemistry = self.chemistry + chemistry
        # a little complicated: need to deal with (1) no spin at all; (2) having no spin and adding;
        # (3) having spin and adding something without; (4) having spin and adding it.
        if spins is not None:
            if type(spins[0]) is not list:
                sp = [spins]
            else:
                sp = spins
            if self.spins is None:
                newspins = [[0 for u in atomlist] for atomlist in self.basis] + sp
            else:
                newspins = self.spins + sp
        else:
            if self.spins is None:
                newspins = None
            else:
                newspins = self.spins + [[0 for u in atomlist] for atomlist in newbasis]
        return Crystal(self.lattice, self.basis + newbasis,
                       chemistry=newchemistry, spins=newspins,
                       threshold=self.threshold)

    def pos2cart(self, lattvec, ind):
        """
        Return the cartesian coordinates of an atom specified by its lattice and index

        :param lattvec: 3-vector (integer) lattice vector in direct coordinates
        :param ind: two-tuple index specifying the atom: (atomtype, atomindex)
        :return v: 3-vector (float) in Cartesian coordinates
        """
        return np.dot(self.lattice, lattvec + self.basis[ind[0]][ind[1]])

    def unit2cart(self, lattvec, uvec):
        """
        Return the cartesian coordinates of a position specified by its lattice and
        unit cell coordinates

        :param lattvec: 3-vector (integer) lattice vector in direct coordinates
        :param uvec: 3-vector (float) unit cell vector in direct coordinates
        :return v: 3-vector (float) in Cartesian coordinates
        """
        return np.dot(self.lattice, lattvec + uvec)

    def cart2unit(self, v):
        """
        Return the lattvec and unit cell coord. corresponding to a position
        in cartesian coord.

        :param v: 3-vector (float) position in Cartesian coordinates
        :return lattvec: 3-vector (integer) lattice vector in direct coordinates,
        :return uvec: 3-vector (float) inside unit cell, in direct coordinates
        """
        u = np.dot(self.invlatt, v)
        ucell = incell(u)
        return (u - ucell).astype(int), ucell

    def cart2pos(self, v):
        """
        Return the lattvec and index corresponding to an atomic position in cartesian coord.

        :param v: 3-vector (float) position in Cartesian coordinates
        :return lattvec: 3-vector (integer) lattice vector in direct coordinates,
        :return (c,i): tuple of matching basis atom; None if no match
        """
        latt, u = self.cart2unit(v)
        indlist = [ind for ind in self.atomindices
                   if self.__isclose__(u, self.basis[ind[0]][ind[1]])]
        if len(indlist) != 1:
            return latt, None
        else:
            return latt, indlist[0]

    @staticmethod
    def g_direc(g, direc):
        """
        Apply a space group operation to a direction

        :param g: group operation (GroupOp)
        :param direc: 3-vector direction
        :return gdirec: 3-vector direction
        """
        if __debug__:
            if type(g) is not GroupOp: raise TypeError
            if type(direc) is not np.ndarray: raise TypeError
        return np.dot(g.cartrot, direc)

    @staticmethod
    def g_tensor(g, tensor):
        """
        Apply a space group operation to a 2nd-rank tensor

        :param g: group operation (GroupOp)
        :param tensor: 2nd-rank tensor
        :return gtensor: 2nd-rank tensor
        """
        if __debug__:
            if type(g) is not GroupOp: raise TypeError
            if type(tensor) is not np.ndarray: raise TypeError
            if tensor.shape != g.rot.shape: raise TypeError
        return np.dot(g.cartrot, np.dot(tensor, g.cartrot.T))

    def g_pos(self, g, lattvec, ind):
        """
        Apply a space group operation to an atom position specified by its lattice and index

        :param g: group operation (GroupOp)
        :param lattvec: 3-vector (integer) lattice vector in direct coordinates
        :param ind: two-tuple index specifying the atom: (atomtype, atomindex)
        :return glatt: 3-vector (integer) lattice vector in direct coordinates
        :return gindex: tuple of new basis atom
        """
        if __debug__:
            if type(g) is not GroupOp: raise TypeError
            if type(lattvec) is not np.ndarray: raise TypeError
        rotlatt = np.dot(g.rot, lattvec)
        rotind = (ind[0], g.indexmap[ind[0]][ind[1]])
        delu = (np.round(np.dot(g.rot, self.basis[ind[0]][ind[1]]) + g.trans -
                         self.basis[rotind[0]][rotind[1]])).astype(int)
        return rotlatt + delu, rotind

    @staticmethod
    def g_vect(g, lattvec, uvec):
        """
        Apply a space group operation to a vector position specified by its lattice and a location
        in the unit cell in direct coordinates

        :param g:  group operation (GroupOp)
        :param lattvec: 3-vector (integer) lattice vector in direct coordinates
        :param uvec: 3-vector (float) vector in direct coordinates
        :return glatt: 3-vector (integer) lattice vector in direct coordinates
        :param guvec: 3-vector (float) vector in direct coordinates
        """
        if __debug__:
            if type(g) is not GroupOp: raise TypeError
            if type(lattvec) is not np.ndarray: raise TypeError
            if type(uvec) is not np.ndarray: raise TypeError
        rotlatt = np.dot(g.rot, lattvec)
        rotu = np.dot(g.rot, uvec) + g.trans
        incellu = incell(rotu)
        return rotlatt + (np.round(rotu - incellu)).astype(int), incellu

    def g_cart(self, g, x):
        """
        Apply a space group operation to a (Cartesian) vector position

        :param g: group operation (GroupOp)
        :param x: 3-vector position in space
        :return gx: 3-vector position in space (Cartesian coordinates)
        """
        if __debug__:
            if type(g) is not GroupOp: raise TypeError
            if type(x) is not np.ndarray: raise TypeError
        return np.dot(g.cartrot, x) + np.dot(self.lattice, g.trans)

    def g_direc_equivalent(self, d1, d2, threshold=1e-8):
        """
        Tells us if two directions are equivalent by according to the space group

        :param d1: direction one (array[3])
        :param d2: direction two (array[3])
        :param threshold: threshold for equality
        :return equivalent: True if equivalent by a point group operation
        """
        return any(np.all(abs(d1 - self.g_direc(g, d2)) < threshold) for g in self.G)

    def genpoint(self):
        """
        Generate our point group indices. Done with crazy list comprehension due to the
        structure of our basis.

        :return Gpointlists: list of lists of frozensets of point group operations that leave a site unchanged
        """
        if self.N == 1:
            return [[self.G]]
        origin = np.zeros(self.dim, dtype=int)
        return [[frozenset([g - self.g_pos(g, origin, (atomtypeindex, atomindex))[0]
                            for g in self.G
                            if g.indexmap[atomtypeindex][atomindex] == atomindex])
                 for atomindex in range(len(atomlist))]
                for atomtypeindex, atomlist in enumerate(self.basis)]

    def genWyckoffsets(self):
        """
        Generate our Wykcoff sets.

        :return Wyckoffsets: set of sets of tuples of positions that correspond to identical Wyckoff positions
        """
        if self.N == 1:
            return frozenset([frozenset([(0, 0)])])
        # this is a little suboptimal if our basis is huge--it leans heavily
        # on the construction of sets to make the checks easy.
        return frozenset([frozenset([(ind[0], g.indexmap[ind[0]][ind[1]])
                                     for g in self.G])
                          for ind in self.atomindices])

    def Wyckoffpos(self, uvec):
        """
        Generates all the equivalent Wyckoff positions for a unit cell vector.

        :param uvec: 3-vector (float) vector in direct coordinates
        :return Wyckofflist: list of equivalent Wyckoff positions
        """
        lis = []
        zero = np.zeros(self.dim, dtype=int)
        for u in (self.g_vect(g, zero, uvec)[1] for g in self.G):
            if not np.any([self.__isclose__(u, u1) for u1 in lis]):
                lis.append(u)
        return lis

    def VectorBasis(self, ind):
        """
        Generates the vector basis corresponding to an atomic site

        :param ind: tuple index for atom
        :return dim: dimensionality, 0..3
        :return vect: vector defining line direction (1) or plane normal (2)
        """
        # need to work with the point group operations for the site
        return reduce(CombineVectorBasis,
                      [VectorBasis(*g.eigen()) for g in self.pointG[ind[0]][ind[1]]])
        # , (3, np.zeros(3)) -- don't need initial value; if there's only one group op, it's identity

    # implemented as a static method as its a utility function
    @staticmethod
    def vectlist(vb):
        """Returns a list of orthonormal vectors corresponding to our vector basis.

        :param vb: (dim, v)
        :return vlist: list of vectors
        """
        if vb[0] == 0: return []
        if vb[0] == 1: return [vb[1]]
        if vb[0] == vb[1].shape[0]: return [v for v in np.eye(vb[1].shape[0])]
        if vb[0] == 2: # 3d only
            # now, construct the other two directions:
            norm = vb[1]
            if abs(norm[2]) < 0.75:
                v1 = np.array([norm[1], -norm[0], 0])
            else:
                v1 = np.array([-norm[2], 0, norm[0]])
            v1 /= np.sqrt(np.dot(v1, v1))
            v2 = np.cross(norm, v1)
            return [v1, v2]

    def FullVectorBasis(self, chem=None):
        """
        Generate our full vector basis, using the information from our crystal

        :param chem: (optional) chemical index to consider; otherwise return a list of such
        :return VBfunctions: (list) of our unique vector basis lattice functions, normalized; each is an array
            (NVbasis x Nsites x 3)
        :return VVouter: (list) of ouf VV "outer" expansion (NVbasis x NVbasis for each chemistry)
        """
        if chem is None:
            chemlist = [c for c in range(len(self.basis))]
        else:
            chemlist = [chem]
        VBlist = []
        VVlist = []
        for c in chemlist:
            lis = []
            for s in self.sitelist(c):
                N = len(self.basis[c])
                for v in self.vectlist(self.VectorBasis((c, s[0]))):
                    v /= np.sqrt(len(s))  # additional normalization
                    # we have some constructing to do... first, make the vector we want to use
                    vb = np.zeros((N, self.dim))
                    for g in self.G:
                        # what site do we land on, and what's the vector? (this is slight overkill)
                        vb[g.indexmap[c][s[0]]] = self.g_direc(g, v)
                    lis.append(vb)
            # need the *full matrix of this tensor*; could probably be done using tensordot?
            VV = np.zeros((self.dim, self.dim, len(lis), len(lis)))
            for i, vb_i in enumerate(lis):
                for j, vb_j in enumerate(lis):
                    VV[:, :, i, j] = np.dot(vb_i.T, vb_j)
            VBlist.append(np.array(lis))
            VVlist.append(VV)
        # if we didn't specify which chemical element, return the lists; else, just the individual arrays
        if chem is None:
            return VBlist, VVlist
        else:
            return VBlist[0], VVlist[0]

    def SymmTensorBasis(self, ind):
        """
        Generates the symmetric tensor basis corresponding to an atomic site

        :param ind: tuple index for atom
        :return tensorbasis: list of 2nd-rank symmetric tensors making up the basis
        """
        # need to work with the point group operations for the site
        return reduce(CombineTensorBasis,
                      [SymmTensorBasis(*g.eigen()) for g in self.pointG[ind[0]][ind[1]]])
        # , (3, np.zeros(3)) -- don't need initial value; if there's only one group op, it's identity

    def nnlist(self, ind, cutoff):
        """
        Generate the nearest neighbor list for a given cutoff. Only consider
        neighbor vectors for atoms of the same type. Returns a list of
        cartesian vectors.

        :param ind: tuple index for atom
        :param cutoff:  distance cutoff
        :return nnlist: list of nearest neighbor vectors
        """
        r2 = cutoff * cutoff
        nmax = [int(np.round(np.sqrt(self.metric[i, i]))) + 1
                for i in range(self.dim)]
        nranges = [range(-n, n+1) for n in nmax]
        supervect = [np.array(ntup) for ntup in itertools.product(*nranges)]
        lis = []
        u0 = self.basis[ind[0]][ind[1]]
        for u1 in self.basis[ind[0]]:
            du = u1 - u0
            for n in supervect:
                dx = self.unit2cart(n, du)
                if np.dot(dx, dx) > 0 and np.dot(dx, dx) < r2:
                    lis.append(dx)
        return lis

    def jumpnetwork(self, chem, cutoff, closestdistance=0):
        """
        Generate the full jump network for a specific chemical index, out to a cutoff. Organized
        by symmetry-unique transitions. Note that i->j and j->i are always related to one-another,
        but by equivalence of transition state, not symmetry. Now updated with closest-distance
        parameter.

        :param chem: index corresponding to the chemistry to consider
        :param cutoff: distance cutoff
        :param closestdistance: closest distance allowed in transition (can be a list)
        :return jumpnetwork: list of symmetry-unique transitions; each is a list of tuples:
          ``((i,j), dx)`` corresponding to jump from :math:`i \\to j` with vector :math:`\mathbf{\delta x}`
        """

        # should we consider changing the lists to tuples? Not sure if there's any efficiency gain
        def inlist(tup, dx, lis):
            """Determines if (i,j), dx is in our list"""
            # a little confusing: run through all transition tuples, see if we find our example
            return any(tup == ij and self.__isclose__(dx, v) for translist in lis for ij, v in translist)

        r2 = cutoff * cutoff
        nmax = [int(np.round(np.sqrt(r2/self.metric[i, i]))) + 1
                for i in range(self.dim)]
        nranges = [range(-n, n+1) for n in nmax]
        supervect = [np.array(ntup) for ntup in itertools.product(*nranges)]
        lis = []
        center = np.zeros(self.dim, dtype=int)
        for i, u0 in enumerate(self.basis[chem]):
            for j, u1 in enumerate(self.basis[chem]):
                du = u1 - u0
                for n in supervect:
                    dx = self.unit2cart(n, du)
                    if np.dot(dx, dx) > 0 and np.dot(dx, dx) < r2:
                        # we have a valid transition; first check that we haven't already looked at it
                        if not inlist((i, j), dx, lis):
                            trans = []
                            for g in self.G:
                                # rotate through all combinations of i->j using space group symmetry
                                R1, ind1 = self.g_pos(g, center, (chem, i))
                                R2, ind2 = self.g_pos(g, n, (chem, j))
                                tup = (ind1[1], ind2[1])
                                dx = self.pos2cart(R2, ind2) - self.pos2cart(R1, ind1)
                                if not any(tup == ij and self.__isclose__(dx, v) for ij, v in trans):
                                    trans.append((tup, dx))
                                    trans.append(((tup[1], tup[0]), -dx))
                            lis.append(trans)
        # now for collision detection:
        if type(closestdistance) is list:
            # quick sanity check to make sure we don't include collision detection on
            # our interstitial site
            closest2list = [x ** 2 if c != chem else -1. for c, x in enumerate(closestdistance)]
        else:
            closest2list = [closestdistance ** 2 if c != chem else -1. for c in range(self.Nchem)]
        for c, mindist2 in enumerate(closest2list):
            if mindist2 < 0:
                # skip the negative distances; we still check 0 because straight line paths
                # through sites should (probably) still be excluded
                continue
            for u0 in self.basis[c]:
                for n in supervect:
                    # check each transition in the list (we need to do list(lis) because
                    # we will modify lis in place with remove's, and its dangerous to pull
                    # off as we iterate through:
                    # for trans in lis.copy():
                    for ntrans in range(len(lis)-1,-1,-1):
                        trans = lis[ntrans]
                        t = trans[0]  # representative transition
                        dx = t[1]
                        # take our starting point relative to the first item in the tuple
                        xRa = self.unit2cart(n, u0 - self.basis[chem][t[0][0]])
                        xRa2 = np.dot(xRa, xRa)
                        xRa_dx = np.dot(xRa, dx)
                        dx2 = np.dot(dx, dx)
                        if 0 <= xRa_dx <= dx2:
                            d2 = (xRa2 * dx2 - xRa_dx * xRa_dx) / dx2
                            if np.isclose(d2, mindist2) or d2 < mindist2:
                                # lis.remove(trans)
                                lis.pop(ntrans)
        lis.sort(key=lambda entry: min(i + j + 1e-3 * np.dot(dx, dx) for (i, j), dx in entry))
        return lis

    def jumpnetwork2lattice(self, chem, jumpnetwork):
        """
        Convert a "standard" jumpnetwork (that specifies displacement vectors dx) into a lattice
        representation, where we replace dx with the lattice vector from i to j.

        :param chem: index corresponding to the chemistry to consider
        :param jumpnetwork: list of symmetry-unique transitions; each is a list of tuples:
          ``((i,j), dx)`` corresponding to jump from :math:`i \\to j` with vector :math:`\mathbf{\delta x}`
        :return jumplattice: list of symmetry-unique transitions; each is a list of tuples:
          ``((i,j), R)`` corresponding to jump from i in unit cell 0 -> j in unit cell R
        """
        return [[((i, j),
                  np.round(np.dot(self.invlatt, dx) + self.basis[chem][i] - self.basis[chem][j]).astype(int))
                 for (i, j), dx in jumplist]
                for jumplist in jumpnetwork]

    def sitelist(self, chem):
        """
        Return a list of lists of Wyckoff-related sites for a given chemistry.
        Done with a single list comprehension--useful as input for diffusion calculation

        :param chem: index corresponding to chemistry to consider
        :return symmequivsites: list of lists of indices that are equivalent by symmetry
        """
        return sorted([sorted(i for c, i in l)  # strips out the chemistry index; sorted for readability
                       for l in [list(s) for s in self.Wyckoff]  # converts to list of lists
                       if l[0][0] == chem])  # select only those with correct chemistry

    def fullkptmesh(self, Nmesh):
        """
        Creates a k-point mesh of density given by Nmesh; does not symmetrize but does put the
        k-points inside the BZ. Does not return any *weights* as every point is equally weighted.

        :param Nmesh: mesh divisions Nmesh[0] x Nmesh[1] x Nmesh[2]
        :return kpt: array[Nkpt][3] of kpoints
        """
        Nkpt = np.product(Nmesh)
        if Nkpt == 0: return
        # dN = np.array([1 / x for x in Nmesh])
        # # use a list comprehension to iterate and build:
        # kptfull = np.array([np.dot(self.reciplatt, (n0 * dN[0], n1 * dN[1], n2 * dN[2]))
        #                     for n0 in range(-Nmesh[0] // 2 + 1, Nmesh[0] // 2 + 1)
        #                     for n1 in range(-Nmesh[1] // 2 + 1, Nmesh[1] // 2 + 1)
        #                     for n2 in range(-Nmesh[2] // 2 + 1, Nmesh[2] // 2 + 1)])
        kdiv = [np.linspace(1/2,-1/2,Nm,endpoint=False) for Nm in Nmesh]
        kptfull = np.array([np.dot(self.reciplatt, ktup) for ktup in itertools.product(*kdiv)])
        # run through list to ensure that all k-points are inside the BZ
        Gmin = min(np.dot(G, G) for G in self.BZG)
        for k in kptfull:
            if np.dot(k, k) >= Gmin:
                for G in self.BZG:
                    if np.dot(k, G) > np.dot(G, G):
                        k -= 2. * G
        return kptfull

    def reducekptmesh(self, kptfull, threshold=None):
        """
        Takes a fully expanded mesh, and reduces it by symmetry. Assumes every point is
        equally weighted. We would need a different (more complicated) algorithm if not true...

        :param kptfull: array[Nkpt][3] of kpoints
        :param threshold: threshold for symmetry equality
        :return kptsymm: array[Nsymm][3] of kpoints
        :return weight: array[Nsymm] of weights (integrates to 1)
        """
        eps = self.threshold if threshold is None else threshold
        kptlist = list(kptfull)
        Nkpt = len(kptlist)
        kptlist.sort(key=lambda k: np.vdot(k, k))
        k2_indices = []
        k2old = np.vdot(kptlist[0], kptlist[0])
        for i, k2 in enumerate([np.vdot(k, k) for k in kptlist]):
            if k2 > (k2old + eps):
                k2_indices.append(i)
                k2old = k2
        k2_indices.append(Nkpt)
        # k2_indices now contains a list of indices with the same magnitudes
        kptsym = []
        wsym = []  # unscaled at this point
        kmin = 0
        basewt = 1 / Nkpt
        for kmax in k2_indices:
            complist = []
            symmcomplist = []
            wtlist = []
            for k in kptlist[kmin:kmax]:
                match = False
                for i, symmcomp in enumerate(symmcomplist):
                    # if any(np.allclose(k, gk, rtol=0, atol=threshold) for gk in symmcomp):
                    if any(np.all(abs(k - gk) < eps) for gk in symmcomp):
                        # update weight, kick out
                        wtlist[i] += basewt
                        match = True
                        continue
                if not match:
                    # new symmetry point!
                    complist.append(k)
                    symmcomplist.append([self.g_direc(g, k) for g in self.G])
                    wtlist.append(basewt)
            kptsym += complist
            wsym += wtlist
            kmin = kmax
        return np.array(kptsym), np.array(wsym)


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


# representers for numpy types that trip up YAML
# (from https://github.com/leifdenby/pycfd/blob/master/yaml_serialize.py)
def bool_representer(dumper, data):
    return dumper.represent_bool(data)


yaml.add_representer(np.bool_, bool_representer)


def int_representer(dumper, data):
    return dumper.represent_int(data)


yaml.add_representer(np.int32, int_representer)
yaml.add_representer(np.dtype(np.int32), int_representer)


def long_representer(dumper, data):
    return dumper.represent_long(data)


yaml.add_representer(np.int64, int_representer)


def float_representer(dumper, data):
    return dumper.represent_float(data)


yaml.add_representer(np.float32, float_representer)
yaml.add_representer(np.float64, float_representer)
