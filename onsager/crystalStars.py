"""
Stars module, modified to work with crystal class

Classes to generate star sets, double star sets, and vector star sets; a lot of indexing functionality.

NOTE: The naming follows that of stars; the functionality is extremely similar, and this code
was modified as little as possible to translate that functionality to *crystals* which possess
a basis. In the case of a single atom basis, this should reduce to the stars object functionality.

The big changes are:

* Replacing NNvect star (which represents the jumps) with the jumpnetwork type found in crystal
* Using the jumpnetwork_latt representation from crystal
* Representing a "point" as a solute + vacancy. In this case, it is a tuple (s,v) of unit cell
  indices and a vector dx or dR (dx = Cartesian vector pointing from solute to vacancy;
  dR = lattice vector pointing from unit cell of solute to unit cell of vacancy). This is equivalent
  to our old representation if the tuple (s,v) = (0,0) for all sites. Due to translational invariance,
  the solute always stays inside the unit cell
* Using indices into the point list rather than just making lists of the vectors themselves. This
  is because the "points" now have a more complex representation (see above).
"""

# TODO: need to make sure we can read / write stars for optimal functionality (YAML?)

__author__ = 'Dallas R. Trinkle'

import numpy as np
import collections
import copy
from . import crystal

# YAML tags
PAIRSTATE_YAMLTAG = '!PairState'

class PairState(collections.namedtuple('PairState', 'i j R dx')):
    """
    A class corresponding to a "pair" state; in this case, a solute-vacancy pair, but can
    also be a transition state pair. The solute (or initial state) is in unit cell 0, in position
    indexed i; the vacancy (or final state) is in unit cell R, in position indexed j.
    The cartesian vector dx connects them. We can add and subtract, negate, and "endpoint"
    subtract (useful for determining what Green function entry to use)

    :param i: index of the first member of the pair (solute)
    :param j: index of the second member of the pair (vacancy)
    :param R: lattice vector pointing from unit cell of i to unit cell of j
    :param dx: Cartesian vector pointing from first to second member of pair
    """

    @classmethod
    def zero(cls):
        """Return the "zero" state"""
        return cls(i=-1, j=-1, R=np.zeros(3, dtype=int), dx=np.zeros(3))

    @classmethod
    def fromcrys(cls, crys, chem, ij, dx):
        """Convert (i,j), dx into PairState"""
        return cls(i=ij[0],
                   j=ij[1],
                   R=np.round(np.dot(crys.invlatt,dx) - crys.basis[chem][ij[1]] + crys.basis[chem][ij[0]]).astype(int),
                   dx=dx)

    @classmethod
    def fromcrys_latt(cls, crys, chem, ij, R):
        """Convert (i,j), R into PairState"""
        return cls(i=ij[0],
                   j=ij[1],
                   R=R,
                   dx=np.dot(crys.lattice, R + crys.basis[chem][ij[1]] - crys.basis[chem][ij[0]]))

    def _asdict(self):
        """Return a proper dict"""
        return {'i': self.i, 'j': self.j, 'R': self.R, 'dx': self.dx}

    def __sane__(self, crys, chem):
        """Determine if the dx value makes sense given everything else..."""
        return np.allclose(self.dx, np.dot(crys.lattice, self.R + crys.basis[chem][self.j]- crys.basis[chem][self.i]))

    def iszero(self):
        """Quicker than self == PairState.zero()"""
        return self.i == self.j and np.all(self.R == 0)

    def __eq__(self, other):
        """Test for equality--we don't bother checking dx"""
        return isinstance(other, self.__class__) and \
               ((self.i == other.i and self.j == other.j and np.all(self.R == other.R)) or \
                (self.iszero() and other.iszero()))
            #   and np.isclose(self.dx, other.dx)

    def __ne__(self, other):
        """Inequality == not __eq__"""
        return not self.__eq__(other)

    def __hash__(self):
        """Hash, so that we can make sets of states"""
        return self.i ^ (self.j << 1) ^ (self.R[0] << 2) ^ (self.R[1] << 3) ^ (self.R[2] << 4)

    def __add__(self, other):
        """Add two states: works if and only if self.j == other.i
        (i,j) R + (j,k) R' = (i,k) R+R'  : works for thinking about transitions...
        """
        if not isinstance(other, self.__class__): return NotImplemented
        if self.iszero(): return other
        if other.iszero(): return self
        if self.j != other.i:
            raise ArithmeticError('Can only add matching endpoints: ({} {})+({} {}) not compatible'.format(self.i, self.j, other.i, other.j))
        if self.i == other.j and np.all(self.R == -other.R):
            return self.zero()
        return self.__class__(i=self.i, j=other.j, R=self.R+other.R, dx=self.dx+other.dx)

    def __neg__(self):
        """Negation of state (swap members of pair)
        - (i,j) R = (j,i) -R
        Note: a + (-a) == (-a) + a == 0 because we define what "zero" is.
        """
        return self.__class__(i=self.j, j=self.i, R=-self.R, dx=-self.dx)

    def __sub__(self, other):
        """Add a negative:
        (i,j) R - (k,j) R' = (i,k) R-R'
        Note: this means that (a-b) + b = a, but b + (a-b) is an error. (b-a) + a = b
        """
        if not isinstance(other, self.__class__): return NotImplemented
        return self.__add__(-other)

    def __xor__(self, other):
        """Subtraction on the endpoints (sort of the "opposite" of a-b)
        (i,j) R ^ (i,k) R' = (k,j) R-R'
        Note: b + (a^b) = b but (a^b) + b is an error. a + (b^a) = a
        """
        if not isinstance(other, self.__class__): return NotImplemented
        if self.iszero(): raise ArithmeticError('Cannot endpoint substract from zero')
        if other.iszero(): raise ArithmeticError('Cannot endpoint subtract zero')
        if self.i != other.i:
            raise ArithmeticError('Can only endpoint subtract matching starts: ({} {})^({} {}) not compatible'.format(self.i, self.j, other.i, other.j))
        if self == other: return self.zero()
        return self.__class__(i=other.j, j=self.j, R=self.R-other.R, dx=self.dx - other.dx)

    def g(self, crys, chem, g):
        """
        Apply group operation.

        :param crys: crystal
        :param chem: chemical index
        :param g: group operation (from crys)
        :return: PairState corresponding to group operation applied to self
        """
        if self.iszero(): return self.zero()
        gRi, (c, gi) = crys.g_pos(g, np.zeros(3, dtype=int), (chem, self.i))
        gRj, (c, gj) = crys.g_pos(g, self.R, (chem, self.j))
        gdx = crys.g_direc(g, self.dx)
        return self.__class__(i=gi, j=gj, R=gRj-gRi, dx=gdx)

    def __str__(self):
        """Human readable version"""
        if self.iszero(): return "*.[0,0,0]:*.[0,0,0] (dx=0)"
        return "{}.[0,0,0]:{}.[{},{},{}] (dx=[{},{},{}])".format(self.i, self.j,
                                                                 self.R[0], self.R[1], self.R[2],
                                                                 self.dx[0], self.dx[1], self.dx[2])

    @classmethod
    def sortkey(cls, entry):
        return np.dot(entry.dx, entry.dx)

    @staticmethod
    def PairState_representer(dumper, data):
        """Output a PairState"""
        # asdict() returns an OrderedDictionary, so pass through dict()
        # had to rewrite _asdict() for some reason...?
        return dumper.represent_mapping(PAIRSTATE_YAMLTAG, data._asdict())

    @staticmethod
    def PairState_constructor(loader, node):
        """Construct a GroupOp from YAML"""
        # ** turns the dictionary into parameters for GroupOp constructor
        return PairState(**loader.construct_mapping(node, deep=True))


class StarSet:
    """
    A class to construct stars, and be able to efficiently index.
    """
    def __init__(self, jumpnetwork, crys, chem, Nshells=0, lattice=False, empty=False):
        """
        Initiates a star set generator for a given jumpnetwork, crystal, and specified
        chemical index.

        :param jumpnetwork: list of symmetry unique jumps, as a list of list of tuples; either
          ((i,j), dx) for jump from i to j with displacement dx, or
          ((i,j), R) for jump from i in unit cell 0 -> j in unit cell R
        :param crys: crystal where jumps take place
        :param chem: chemical index of atom to consider jumps
        :param Nshells: number of shells to generate
        :param lattice: which form does the jumpnetwork take?
        """
        if empty:
            if __debug__:
                if any(x is not None for x in (jumpnetwork, crys, chem, Nshells)):
                    raise TypeError('Tried to create empty StarSet with none-None parameters')
            return
        self.jumpnetwork_index = []  # list of list of indices into...
        self.jumplist = []  # list of our jumps, as PairStates
        ind = 0
        for jlist in jumpnetwork:
            self.jumpnetwork_index.append([])
            for ij, v in jlist:
                self.jumpnetwork_index[-1].append(ind)
                ind += 1
                if lattice: PS = PairState.fromcrys_latt(crys, chem, ij, v)
                else: PS = PairState.fromcrys(crys, chem, ij, v)
                self.jumplist.append(PS)
        self.crys = crys
        self.chem = chem
        self.Nshells = self.generate(Nshells)

    def generate(self, Nshells, threshold=1e-8):
        """
        Construct the points and the stars in the set.
        :param Nshells: number of shells to generate; this is interpreted as subsequent
          "sums" of jumplist (as we need the solute to be connected to the vacancy by at least one jump)
        :param threshold: threshold for determining equality with symmetry
        """
        if Nshells == getattr(self, 'Nshells', -1): return
        self.Nshells = Nshells
        if Nshells > 0: self.states = self.jumplist.copy()
        else: self.states = []
        lastshell = list(self.states)
        for i in range(Nshells-1):
            # add all NNvect to last shell produced, always excluding 0
            # lastshell = [v1+v2 for v1 in lastshell for v2 in self.NNvect if not all(abs(v1 + v2) < threshold)]
            nextshell = []
            for s1 in lastshell:
                for s2 in self.jumplist:
                    # this try/except structure lets us attempt addition and kick out if not possible
                    try:
                        s = s1 + s2
                        if not s.iszero():
                            if not any(s == st for st in self.states):
                                nextshell.append(s)
                                self.states.append(s)
                    except: pass
            lastshell = nextshell
        # now to sort our set of vectors (easiest by magnitude, and then reduce down:
        self.states.sort(key=PairState.sortkey)
        self.Nstates = len(self.states)
        if self.Nstates > 0:
            x2_indices = []
            x2old = np.dot(self.states[0].dx, self.states[0].dx)
            for i, x2 in enumerate([np.dot(st.dx, st.dx) for st in self.states]):
                if x2 > (x2old + threshold):
                    x2_indices.append(i)
                    x2old = x2
            x2_indices.append(len(self.states))
            # x2_indices now contains a list of indices with the same magnitudes
            self.stars = []
            xmin = 0
            for xmax in x2_indices:
                complist_stars = [] # for finding unique stars
                for xi in range(xmin, xmax):
                    x = self.states[xi]
                    # is this a new rep. for a unique star?
                    match = False
                    for i, s in enumerate(complist_stars):
                        if self.symmatch(x, self.states[s[0]]):
                            # update star
                            complist_stars[i].append(xi)
                            match = True
                            continue
                    if not match:
                        # new symmetry point!
                        complist_stars.append([xi])
                self.stars += complist_stars
                xmin=xmax
        else: self.stars = [[]]
        self.Nstars = len(self.stars)
        # generate index: which star is each state a member of?
        self.index = np.zeros(self.Nstates, dtype=int)
        for si, star in enumerate(self.stars):
            for xi in star:
                self.index[xi] = si

    def copy(self):
        """Return a copy of the StarSet; done as efficiently as possible"""
        newStarSet = StarSet(None, None, None, None, empty=True)  # a little hacky... creates an empty class
        newStarSet.jumpnetwork_index = copy.deepcopy(self.jumpnetwork_index)
        newStarSet.jumplist = self.jumplist.copy()
        newStarSet.crys = self.crys
        newStarSet.chem = self.chem
        newStarSet.Nshells = self.Nshells
        newStarSet.stars = copy.deepcopy(self.stars)
        newStarSet.states = self.states.copy()
        newStarSet.Nstars = self.Nstars
        newStarSet.Nstates = self.Nstates
        newStarSet.index = self.index.copy()
        return newStarSet

    # removed combine; all it does is generate(s1.Nshells + s2.Nshells) with lots of checks...
    # replaced with (more efficient?) __add__ and __radd__.

    def __add__(self, other):
        """Add two StarSets together; done by making a copy of one, and radding"""
        if not isinstance(other, self.__class__): return NotImplemented
        if self.Nshells >= other.Nshells:
            scopy = self.copy()
            scopy.__radd__(other)
        else:
            scopy = other.copy()
            scopy.__radd__(self)
        return scopy

    def __radd__(self, other):
        """Add another StarSet to this one; very similar to generate()"""
        threshold = 1e-8
        if not isinstance(other, self.__class__): return NotImplemented
        if self.chem != other.chem: return ArithmeticError('Cannot add different chemistry index')
        self.Nshells += other.Nshells
        newshell = []
        Nold = self.Nstates
        for s1 in self.states[:Nold]:
            for s2 in other.states:
                # this try/except structure lets us attempt addition and kick out if not possible
                try:
                    s = s1 + s2
                    if not s.iszero() and not any(s == st for st in self.states): self.states.append(s)
                except: pass
        # now to sort our set of vectors (easiest by magnitude, and then reduce down:
        self.states[Nold:] = sorted(self.states[Nold:], key=PairState.sortkey)
        Nnew = len(self.states)
        x2_indices = []
        x2old = np.dot(self.states[Nold].dx, self.states[Nold].dx)
        for i in range(Nold, Nnew):
            x2 = np.dot(self.states[i].dx, self.states[i].dx)
            if x2 > (x2old + threshold):
                x2_indices.append(i)
                x2old = x2
        x2_indices.append(Nnew)
        # x2_indices now contains a list of indices with the same magnitudes
        xmin = Nold
        for xmax in x2_indices:
            complist_stars = [] # for finding unique stars
            for xi in range(xmin, xmax):
                x = self.states[xi]
                # is this a new rep. for a unique star?
                match = False
                for i, s in enumerate(complist_stars):
                    if self.symmatch(x, self.states[s[0]]):
                        # update star
                        complist_stars[i].append(xi)
                        match = True
                        continue
                if not match:
                    # new symmetry point!
                    complist_stars.append([xi])
            self.stars += complist_stars
            xmin=xmax
        self.Nstates = Nnew
        # generate new index entries: which star is each state a member of?
        self.index = np.pad(self.index, (0, Nnew-Nold), mode='constant')
        Nold = self.Nstars
        Nnew = len(self.stars)
        for si in range(Nold, Nnew):
            star = self.stars[si]
            for xi in star:
                self.index[xi] = si
        self.Nstars = Nnew

    # replaces pointindex:
    def stateindex(self, PS):
        """Return the index of pair state PS; None if not found"""
        try: return self.states.index(PS)
        except: return None

    def starindex(self, PS):
        """Return the index for the star to which pair state PS belongs; None if not found"""
        ind = self.stateindex(PS)
        if ind is None: return None
        return self.index[ind]

    def symmatch(self, PS1, PS2):
        """True if there exists a group operation that makes PS1 == PS2."""
        return any(PS1 == PS2.g(self.crys, self.chem, g) for g in self.crys.G)

### LEFT OFF HERE
class DoubleStarSet:
    """
    A class to construct double-stars (pairs of sites,
    where each pair is related by a single group op).
    """
    def __init__(self, star=None):
        """
        Initiates a double-star-generator; is designed to work with a given star.

        Parameters
        ----------
        star : Star, optional
            all of our input parameters will come from this, if non-empty
        """
        self.star = None
        self.Ndstars = 0
        self.Npairs = 0
        self.Npts = 0
        if star is not None:
            if star.Nshells > 0:
                self.generate(star)

    def generate(self, star, threshold=1e-8):
        """
        Construct the actual double-stars.

        Parameters
        ----------
        star : Star, optional
            all of our input parameters will come from this, if non-empty
        """
        if star.Nshells == 0:
            self.Npairs = 0
            return
        if star == self.star and star.Npts == self.Npts:
            return
        self.star = star
        self.Npts = star.Npts
        self.NNvect = star.NNvect
        self.groupops = star.groupops
        self.pairs = []
        # make the pairs first
        for i1, v1 in enumerate(self.star.pts):
            for dv in self.NNvect:
                v2 = v1 + dv
                i2 = self.star.pointindex(v2)
                # check that i2 is a valid point
                if i2 >= 0:
                    self.pairs.append((i1, i2))
        # sort the pairs
        self.pairs.sort(key=lambda x: max(np.dot(x[0], x[0]), np.dot(x[1], x[1])))
        self.Npairs = len(self.pairs)
        # now to make the unique sets of pairs (double-stars)
        self.dstars = []
        for pair in self.pairs:
            # Q: is this a new rep. for a unique double-star?
            match = False
            for i, ds in enumerate(self.dstars):
                if self.symmatch(pair, ds[0], threshold):
                    # update star
                    self.dstars[i].append(pair)
                    match = True
                    continue
            if not match:
                # new symmetry point!
                self.dstars.append([pair])
        self.Ndstars = len(self.dstars)
        self.index = None

    def generateindices(self):
        """
        Generates the double star indices for the set of points, if not already generated
        """
        if self.index is not None:
            return
        self.index = np.zeros(self.Npairs, dtype=int)
        for i, p in enumerate(self.pairs):
            for nds, ds in enumerate(self.dstars):
                if any([p == p1 for p1 in ds]):
                    self.index[i] = nds

    def dstarindex(self, p):
        """
        Returns the index for the double-star to which pair p belongs.

        Parameters
        ----------
        p : two-tuple
            pair to find the double-star for

        Returns
        -------
        index corresponding to double-star; -1 if not found.
        """
        self.generateindices()
        for i, p1 in enumerate(self.pairs):
            if p1 == p:
                return self.index[i]
        return -1

    def pairindex(self, p):
        """
        Returns the index corresponding to the pair p.

        Parameters
        ----------
        p : two-tuple
            pair to index

        Returns
        -------
        index corresponding to pair; -1 if not found.
        """
        for i, v in enumerate(self.pairs):
            if v == p:
                return i
        return -1

    def symmatch(self, x, xcomp, threshold=1e-8):
        """
        Tells us if x and xcomp are equivalent by a symmetry group

        Parameters
        ----------
        x : 2-tuple
            two indices corresponding to a pair
        xcomp : 2-tuple
            two indices corresponding to a pair to compare
        threshold : double, optional
            threshold to use for "equality"

        Returns
        -------
        True if equivalent by a point group operation, False otherwise
        """
        # first, try the tuple "forward"
        v00 = self.star.pts[x[0]]
        v01 = self.star.pts[x[1]]
        v10 = self.star.pts[xcomp[0]]
        v11 = self.star.pts[xcomp[1]]
        if any([np.all(abs(v00 - np.dot(g, v10)) < threshold) and np.all(abs(v01 - np.dot(g, v11)) < threshold)
                for g in self.groupops]):
            return True
        return any([np.all(abs(v00 - np.dot(g, v11)) < threshold) and np.all(abs(v01 - np.dot(g, v10)) < threshold)
                for g in self.groupops])

class VectorStarSet:
    """
    A class to construct vector stars, and be able to efficiently index.
    """
    def __init__(self, star=None):
        """
        Initiates a vector-star generator; is designed to work with a given star.

        Parameters
        ----------
        star : Star, optional
            all of our input parameters will come from this, if non-empty
        """
        self.star = None
        self.Npts = 0
        self.Nvstars = 0
        self.Nstars = 0
        if star is not None:
            self.NNvect = star.NNvect
            self.groupops = star.groupops
            if star.Nshells > 0:
                self.generate(star)

    def generate(self, star, threshold=1e-8):
        """
        Construct the actual vectors stars

        Parameters
        ----------
        star : Star, optional
            all of our input parameters will come from this, if non-empty
        """
        if star.Nshells == 0:
            return
        if star == self.star and star.Npts == self.Npts:
            return
        self.star = star
        self.Npts = star.Npts
        self.NNvect = star.NNvect
        self.groupops = star.groupops
        self.vecpos = []
        self.vecvec = []
        for s in self.star.stars:
            # start by generating the parallel star-vector; always trivially present:
            self.vecpos.append(s)
            scale = 1./np.sqrt(len(s)*np.dot(s[0],s[0])) # normalization factor
            self.vecvec.append([v*scale for v in s])
            # next, try to generate perpendicular star-vectors, if present:
            v0 = np.cross(s[0], np.array([0, 0, 1.]))
            if np.dot(v0, v0) < threshold:
                v0 = np.cross(s[0], np.array([1., 0, 0]))
            v1 = np.cross(s[0], v0)
            # normalization:
            v0 /= np.sqrt(np.dot(v0, v0))
            v1 /= np.sqrt(np.dot(v1, v1))
            Nvect = 2
            # run over the invariant group operations for vector s[0]:
            for g in [g0 for g0 in self.groupops if all(abs(np.dot(g0, s[0]) - s[0]) < threshold)]:
                if Nvect == 0:
                    continue
                gv0 = np.dot(g, v0)
                if Nvect == 1:
                    # we only need to check that we still have an invariant vector
                    if any(abs(gv0 - v0) > threshold):
                        Nvect = 0
                if Nvect == 2:
                    gv1 = np.dot(g, v1)
                    g00 = np.dot(v0, gv0)
                    g11 = np.dot(v1, gv1)
                    g01 = np.dot(v0, gv1)
                    g10 = np.dot(v1, gv0)
                    if abs((abs(g00*g11 - g01*g10) - 1)) > threshold or abs(g01-g10) > threshold:
                        # we don't have an orthogonal matrix, or we have a rotation, so kick out
                        Nvect = 0
                        continue
                    if (abs(g00 - 1) > threshold) or (abs(g11 - 1) > threshold):
                        # if we don't have the identify matrix, then we have to find the one vector that survives
                        if abs(g00 - 1) < threshold:
                            Nvect = 1
                            continue
                        if abs(g11 - 1) < threshold:
                            v0 = v1
                            Nvect = 1
                            continue
                        v0 = (g01*v0 + (1 - g00)*v1)/np.sqrt(g01*g10 + (1 - g00)**2)
                        Nvect = 1
            # so... do we have any vectors to add?
            if Nvect > 0:
                v0 /= np.sqrt(len(s)*np.dot(v0, v0))
                v1 /= np.sqrt(len(s)*np.dot(v1, v1))
                vlist = [v0]
                if Nvect > 1:
                    vlist.append(v1)
                # add the positions
                for v in vlist:
                    self.vecpos.append(s)
                    veclist = []
                    for R in s:
                        for g in self.groupops:
                            if all(abs(R - np.dot(g, s[0])) < threshold):
                                veclist.append(np.dot(g, v))
                                break
                    self.vecvec.append(veclist)
        self.Nvstars = len(self.vecpos)
        self.generateouter()

    def generateouter(self):
        """
        Generate our outer products for our star-vectors.

        Returns
        -------
        outer : array [3, 3, Nvstars, Nvstars]
            outer[:, :, i, j] is the 3x3 tensor outer product for two vector-stars vs[i] and vs[j]
        """
        self.outer = np.zeros((3, 3, self.Nvstars, self.Nvstars))
        for i, (sR0, sv0) in enumerate(zip(self.vecpos, self.vecvec)):
            for j, (sR1, sv1) in enumerate(zip(self.vecpos, self.vecvec)):
                if (sR0[0] == sR1[0]).all():
                    self.outer[:, :, i, j] = sum([np.outer(v0, v1) for v0, v1 in zip(sv0, sv1)])
        #[sum([np.outer(v, v) for v in veclist]) for veclist in self.vecvec]

    def GFexpansion(self, starGF):
        """
        Construct the GF matrix expansion in terms of the star vectors, and indexed
        to starGF.

        Parameters
        ----------
        starGF: Star
            stars that reference the GF; should be at least twice the size of the
            stars used to construct the star vector

        Returns
        -------
        GFexpansion: array[Nsv, Nsv, Nstars+1]
            the GF matrix[i, j] = GFexpansion[i, j, 0]*GF(0) + sum(GFexpansion[i, j, k+1] * GF(starGF[k]))
        """
        if self.Nvstars == 0:
            return None
        if not isinstance(starGF, StarSet):
            raise TypeError('need a star')
        GFexpansion = np.zeros((self.Nvstars, self.Nvstars, starGF.Nstars+1))
        for i in range(self.Nvstars):
            for j in range(self.Nvstars):
                if i <= j :
                    for Ri, vi in zip(self.vecpos[i], self.vecvec[i]):
                        for Rj, vj in zip(self.vecpos[j], self.vecvec[j]):
                            if (Ri == Rj).all():
                                k = 0
                            else:
                                k = starGF.starindex(Ri - Rj) + 1
                                if k == 0:
                                    raise ArithmeticError('GF star not large enough to include {}'.format(Ri - Rj))
                            GFexpansion[i, j, k] += np.dot(vi, vj)
                else:
                    GFexpansion[i, j, :] = GFexpansion[j, i, :]
        return GFexpansion

    def rate0expansion(self, NNstar):
        """
        Construct the omega0 matrix expansion in terms of the NN stars. Note: includes
        on-site terms.

        Parameters
        ----------
        NNstar: Star
            nearest-neighbor stars

        Returns
        -------
        rate0expansion: array[Nsv, Nsv, Nstars]
            the omega0 matrix[i, j] = sum(rate0expansion[i, j, k] * omega0(NNstar[k]))
        """
        if self.Nvstars == 0:
            return None
        if not isinstance(NNstar, StarSet):
            raise TypeError('need a star')
        rate0expansion = np.zeros((self.Nvstars, self.Nvstars, NNstar.Nstars))
        for i in range(self.Nvstars):
            for j in range(self.Nvstars):
                if i <= j :
                    for Ri, vi in zip(self.vecpos[i], self.vecvec[i]):
                        for Rj, vj in zip(self.vecpos[j], self.vecvec[j]):
                            # which NN shell is this? k == -1 indicates a pair that does not appear
                            k = NNstar.starindex(Ri - Rj)
                            if k >= 0:
                                rate0expansion[i, j, k] += np.dot(vi, vj)
                # note: we do *addition* here because we may have on-site contributions above
                if i == j:
                    for k, s in enumerate(NNstar.stars):
                        rate0expansion[i, i, k] += -len(s)
                if i > j:
                    rate0expansion[i, j, :] = rate0expansion[j, i, :]
        return rate0expansion

    def rate1expansion(self, dstar):
        """
        Construct the omega1 matrix expansion in terms of the double stars.

        Parameters
        ----------
        dstar: DoubleStar
            double-stars (i.e., pairs that are related by a symmetry operation; usually the sites
            are connected by a NN vector to facilitate a jump; indicates unique vacancy jumps
            around a solute)

        Returns
        -------
        rate1expansion: array[Nsv, Nsv, Ndstars]
            the omega1 matrix[i, j] = sum(rate1expansion[i, j, k] * omega1(dstar[k]))
        """
        if self.Nvstars == 0:
            return None
        if not isinstance(dstar, DoubleStarSet):
            raise TypeError('need a double star')
        rate1expansion = np.zeros((self.Nvstars, self.Nvstars, dstar.Ndstars))
        for i in range(self.Nvstars):
            for j in range(self.Nvstars):
                if i <= j :
                    for Ri, vi in zip(self.vecpos[i], self.vecvec[i]):
                        for Rj, vj in zip(self.vecpos[j], self.vecvec[j]):
                            # note: double-stars are tuples of point indices
                            k = dstar.dstarindex((dstar.star.pointindex(Ri),
                                                  dstar.star.pointindex(Rj)))
                            # note: k == -1 indicates now a pair that does not appear, not an error
                            if k >= 0:
                                rate1expansion[i, j, k] += np.dot(vi, vj)
                else:
                    rate1expansion[i, j, :] = rate1expansion[j, i, :]
        return rate1expansion

    def rate2expansion(self, NNstar):
        """
        Construct the omega2 matrix expansion in terms of the nearest-neighbor stars. Includes
        the "on-site" terms as well, hence there's a factor of 2 in the output.

        Parameters
        ----------
        NNstar: Star
            stars representing the unique nearest-neighbor jumps

        Returns
        -------
        rate2expansion: array[Nsv, Nsv, NNstars]
            the omega2 matrix[i, j] = sum(rate2expansion[i, j, k] * omega2(NNstar[k]))
        """
        if self.Nvstars == 0:
            return None
        if not isinstance(NNstar, StarSet):
            raise TypeError('need a star')
        rate2expansion = np.zeros((self.Nvstars, self.Nvstars, NNstar.Nstars))
        for i in range(self.Nvstars):
            # this is a diagonal matrix, so...
            ind = NNstar.starindex(self.vecpos[i][0])
            if ind != -1:
                rate2expansion[i, i, ind] = -2.*np.dot(self.vecvec[i][0], self.vecvec[i][0])*len(NNstar.stars[ind])
        return rate2expansion

    def bias2expansion(self, NNstar):
        """
        Construct the bias2 vector expansion in terms of the nearest-neighbor stars.

        Parameters
        ----------
        NNstar: Star
            stars representing the unique nearest-neighbor jumps

        Returns
        -------
        bias2expansion: array[Nsv, NNstars]
            the bias2 vector[i] = sum(bias2expansion[i, k] * omega2(NNstar[k]))
        """
        if self.Nvstars == 0:
            return None
        if not isinstance(NNstar, StarSet):
            raise TypeError('need a star')
        bias2expansion = np.zeros((self.Nvstars, NNstar.Nstars))
        for i in range(self.Nvstars):
            ind = NNstar.starindex(self.vecpos[i][0])
            if ind != -1:
                bias2expansion[i, ind] = np.dot(self.vecpos[i][0], self.vecvec[i][0])*len(NNstar.stars[ind])
        return bias2expansion

    def bias1expansion(self, dstar, NNstar):
        """
        Construct the bias1 or omega1 onsite vector expansion in terms of the
        nearest-neighbor stars. There are three pieces to this that we need to
        construct now, so it's more complicated. Since we use the *identical* algorithm,
        we return both bias1ds and omega1ds, and bias1NN and omega1NN.

        Parameters
        ----------
        dstar: DoubleStar
            double-stars (i.e., pairs that are related by a symmetry operation; usually the sites
            are connected by a NN vector to facilitate a jump; indicates unique vacancy jumps
            around a solute)

        NNstar: Star
            stars representing the unique nearest-neighbor jumps

        Returns
        -------
        bias1ds: array[Nsv, Ndstars]
            the gen1 vector[i] = sum(gen1ds[i, k] * sqrt(prob_star[gen1prob[i, k]) * omega1[dstar[k]])

        omega1ds: array[Nsv, Ndstars]
            the omega1 onsite vector[i] = sum(omega1ds[i, k] * sqrt(prob_star[gen1prob[i, k]) * omega1[dstar[k]])

        gen1prob: array[Nsv, Ndstars], dtype=int
            index for the corresponding *star* whose probability defines the endpoint.

        bias1NN: array[Nsv, NNNstars]
            we have an additional contribution to the bias1 vector:
            bias1 vector[i] += sum(bias1NN[i, k] * omega0[NNstar[k]])

        oemga1NN: array[Nsv, NNNstars]
            we have an additional contribution to the omega1 onsite vector:
            omega1 onsite vector[i] += sum(omega1NN[i, k] * omega0[NNstar[k]])
        """
        if self.Nvstars == 0:
            return None
        if not isinstance(dstar, DoubleStarSet):
            raise TypeError('need a double star')
        if not isinstance(NNstar, StarSet):
            raise TypeError('need a star')
        NNstar.generateindices()
        bias1ds = np.zeros((self.Nvstars, dstar.Ndstars))
        omega1ds = np.zeros((self.Nvstars, dstar.Ndstars))
        gen1bias = np.empty((self.Nvstars, dstar.Ndstars), dtype=int)
        gen1bias[:, :] = -1
        bias1NN = np.zeros((self.Nvstars, NNstar.Nstars))
        omega1NN = np.zeros((self.Nvstars, NNstar.Nstars))

        # run through the star-vectors
        for i, svR, svv in zip(list(range(self.Nvstars)),
                               self.vecpos, self.vecvec):
            # run through the NN stars
            p1 = dstar.star.pointindex(svR[0]) # first half of our pair
            # nnst = star index, vec = NN jump vector
            for nnst, vec in zip(NNstar.index, NNstar.pts):
                endpoint = svR[0] + vec
                # throw out the origin as an endpoint
                if all(abs(endpoint) < 1e-8):
                    continue
                geom_bias = np.dot(svv[0], vec) * len(svR)
                geom_omega1 = -1. #len(svR)
                p2 = dstar.star.pointindex(endpoint)
                if p2 == -1:
                    # we landed outside our range of double-stars, so...
                    bias1NN[i, nnst] += geom_bias
                    omega1NN[i, nnst] += geom_omega1
                else:
                    ind = dstar.dstarindex((p1, p2))
                    if ind == -1:
                        raise ArithmeticError('Problem with DoubleStar indexing; could not find double-star for pair')
                    bias1ds[i, ind] += geom_bias
                    omega1ds[i, ind] += geom_omega1
                    sind = dstar.star.index[p2]
                    if sind == -1:
                        raise ArithmeticError('Could not locate endpoint in a star in DoubleStar')
                    if gen1bias[i, ind] == -1:
                        gen1bias[i, ind] = sind
                    else:
                        if gen1bias[i, ind] != sind:
                            raise ArithmeticError('Inconsistent DoubleStar endpoints found')
        return bias1ds, omega1ds, gen1bias, bias1NN, omega1NN

crystal.yaml.add_representer(PairState, PairState.PairState_representer)
crystal.yaml.add_constructor(PAIRSTATE_YAMLTAG, PairState.PairState_constructor)

