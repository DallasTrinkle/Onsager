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
    def zero(cls, n=-1):
        """Return the "zero" state"""
        return cls(i=n, j=n, R=np.zeros(3, dtype=int), dx=np.zeros(3))

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
               (self.i == other.i and self.j == other.j and np.all(self.R == other.R))

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
        return self.__class__(i=self.i, j=other.j, R=self.R+other.R, dx=self.dx+other.dx)

    def __neg__(self):
        """Negation of state (swap members of pair)
        - (i,j) R = (j,i) -R
        Note: a + (-a) == (-a) + a == 0 because we define what "zero" is.
        """
        return self.__class__(i=self.j, j=self.i, R=-self.R, dx=-self.dx)

    def __sub__(self, other):
        """Add a negative:
        a-b points from initial of a to initial of b if same final state
        (i,j) R - (k,j) R' = (i,k) R-R'
        Note: this means that (a-b) + b = a, but b + (a-b) is an error. (b-a) + a = b
        """
        if not isinstance(other, self.__class__): return NotImplemented
        return self.__add__(-other)

    def __xor__(self, other):
        """Subtraction on the endpoints (sort of the "opposite" of a-b):
        a^b points from final of b to final of a if same initial state
        (i,j) R ^ (i,k) R' = (k,j) R-R'
        Note: b + (a^b) = a but (a^b) + b is an error. a + (b^a) = b
        """
        if not isinstance(other, self.__class__): return NotImplemented
        if self.iszero(): raise ArithmeticError('Cannot endpoint substract from zero')
        if other.iszero(): raise ArithmeticError('Cannot endpoint subtract zero')
        if self.i != other.i:
            raise ArithmeticError('Can only endpoint subtract matching starts: ({} {})^({} {}) not compatible'.format(self.i, self.j, other.i, other.j))
        return self.__class__(i=other.j, j=self.j, R=self.R-other.R, dx=self.dx - other.dx)

    def g(self, crys, chem, g):
        """
        Apply group operation.

        :param crys: crystal
        :param chem: chemical index
        :param g: group operation (from crys)
        :return: PairState corresponding to group operation applied to self
        """
        gRi, (c, gi) = crys.g_pos(g, np.zeros(3, dtype=int), (chem, self.i))
        gRj, (c, gj) = crys.g_pos(g, self.R, (chem, self.j))
        gdx = crys.g_direc(g, self.dx)
        return self.__class__(i=gi, j=gj, R=gRj-gRi, dx=gdx)

    def __str__(self):
        """Human readable version"""
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


class StarSet(object):
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

        crys: crystal structure
        chem: chemical index of atom that's jumping
        Nshells: number of shells (addi
        jumpnetwork_index: list of lists of indices into jumplist; matches structure of jumpnetwork
        jumplist: list of jumps, as pair states (i=initial state, j=final state)
        states: list of pair states, out to Nshells
        Nstates: size of list
        stars: list of lists of indices into states; each list are states equivalent by symmetry
        Nstars: size of list
        index[Nstates]: index of star that state belongs to
        """
        if empty:
            # this is really just used by copy() to circumvent __init__
            if __debug__:
                if any(x is not None for x in (jumpnetwork, crys, chem)):
                    raise TypeError('Tried to create empty StarSet with non-None parameters')
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

    def __str__(self):
        """Human readable version"""
        str = "Nshells: {}  Nstates: {}  Nstars: {}\n".format(self.Nshells, self.Nstates, self.Nstars)
        for si in range(self.Nstars):
            str += "Star {} ({})\n".format(si, len(self.stars[si]))
            for i in self.stars[si]:
                str += "  {}: {}\n".format(i, self.states[i])
        return str

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
                    try: s = s1 + s2
                    except: continue
                    if not s.iszero():
                        if not any(s == st for st in self.states):
                            nextshell.append(s)
                            self.states.append(s)
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

    def copy(self, empty=False):
        """Return a copy of the StarSet; done as efficiently as possible; empty means skip the shells, etc."""
        newStarSet = StarSet(None, None, None, -1, empty=True)  # a little hacky... creates an empty class
        newStarSet.jumpnetwork_index = copy.deepcopy(self.jumpnetwork_index)
        newStarSet.jumplist = self.jumplist.copy()
        newStarSet.crys = self.crys
        newStarSet.chem = self.chem
        if not empty:
            newStarSet.Nshells = self.Nshells
            newStarSet.stars = copy.deepcopy(self.stars)
            newStarSet.states = self.states.copy()
            newStarSet.Nstars = self.Nstars
            newStarSet.Nstates = self.Nstates
            newStarSet.index = self.index.copy()
        else: newStarSet.generate(0)
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
        if other.Nshells < 1: return self
        if self.Nshells < 1:
            self.Nshells = other.Nshells
            self.stars = copy.deepcopy(other.stars)
            self.states = other.states.copy()
            self.Nstars = other.Nstars
            self.Nstates = other.Nstates
            self.index = other.index.copy()
            return self
        self.Nshells += other.Nshells
        Nold = self.Nstates
        for s1 in self.states[:Nold]:
            for s2 in other.states:
                # this try/except structure lets us attempt addition and kick out if not possible
                try: s = s1 + s2
                except: continue
                if not s.iszero() and not any(s == st for st in self.states): self.states.append(s)
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
        return self

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

    # replaces DoubleStarSet
    def jumpnetwork_omega1(self):
        """
        Generate a jumpnetwork corresponding to vacancy jumping while the solute remains fixed.
        :return jumpnetwork: list of symmetry unique jumps; list of list of tuples (i,f), dx where
          i,f index into states for the initial and final states, and dx = displacement of vacancy
          in Cartesian coordinates. Note: if (i,f), dx is present, so if (f,i), -dx
        :return jumptype: list of indices corresponding to the (original) jump type for each
          symmetry unique jump; useful for constructing a LIMB approximation
        :return starpair: list of tuples of the star indices of the i and f states for each
          symmetry unique jump
        """
        if self.Nshells < 1: return []
        jumpnetwork = []
        jumptype = []
        starpair = []
        for jt, jumpindices in enumerate(self.jumpnetwork_index):
            for jump in [ self.jumplist[j] for j in jumpindices]:
                for i, PSi in enumerate(self.states):
                    # attempt to add...
                    try: PSf = PSi + jump
                    except: continue
                    if PSf.iszero(): continue
                    f = self.stateindex(PSf)
                    if f is None: continue  # outside our StarSet
                    # see if we've already generated this jump (works since all of our states are distinct)
                    if any(any( i==i0 and f==f0 for (i0,f0), dx in jlist) for jlist in jumpnetwork): continue
                    dx = PSf.dx - PSi.dx
                    jumpnetwork.append(self.symmequivjumplist(i, f, dx))
                    jumptype.append(jt)
                    starpair.append((self.index[i], self.index[f]))
        return jumpnetwork, jumptype, starpair

    def jumpnetwork_omega2(self):
        """
        Generate a jumpnetwork corresponding to vacancy exchanging with a solute.
        :return jumpnetwork: list of symmetry unique jumps; list of list of tuples (i,f), dx where
          i,f index into states for the initial and final states, and dx = displacement of vacancy
          in Cartesian coordinates. Note: if (i,f), dx is present, so if (f,i), -dx
        :return jumptype: list of indices corresponding to the (original) jump type for each
          symmetry unique jump; useful for constructing a LIMB approximation
        :return starpair: list of tuples of the star indices of the i and f states for each
          symmetry unique jump
        """
        if self.Nshells < 1: return []
        jumpnetwork = []
        jumptype = []
        starpair = []
        for jt, jumpindices in enumerate(self.jumpnetwork_index):
            for jump in [ self.jumplist[j] for j in jumpindices]:
                for i, PSi in enumerate(self.states):
                    # attempt to add...
                    try: PSf = PSi + jump
                    except: continue
                    if not PSf.iszero(): continue
                    f = self.stateindex(-PSi)  # exchange
                    # see if we've already generated this jump (works since all of our states are distinct)
                    if any(any( i==i0 and f==f0 for (i0,f0), dx in jlist) for jlist in jumpnetwork): continue
                    dx = -PSi.dx  # the vacancy jumps into the solute position (exchange)
                    jumpnetwork.append(self.symmequivjumplist(i, f, dx))
                    jumptype.append(jt)
                    starpair.append((self.index[i], self.index[f]))
        return jumpnetwork, jumptype, starpair

    def symmequivjumplist(self, i, f, dx):
        """
        Returns a list of tuples of symmetry equivalent jumps
        :param i: index of initial state
        :param f: index of final state
        :param dx: displacement vector
        :return symmjumplist: list of tuples of ((gi, gf), gdx) for every group op
        """
        PSi = self.states[i]
        PSf = self.states[f]
        symmjumplist = [((i,f), dx)]
        if i != f: symmjumplist.append(((f,i), -dx)) # i should not equal f... but in case we allow 0 as a jump
        for g in self.crys.G:
            gi, gf, gdx = self.stateindex(PSi.g(self.crys, self.chem, g)),\
                          self.stateindex(PSf.g(self.crys, self.chem, g)),\
                          self.crys.g_direc(g, dx)
            if not any( gi==i0 and gf==f0 for (i0,f0), dx in symmjumplist):
                symmjumplist.append(((gi,gf), gdx))
                if gi != gf: symmjumplist.append(((gf, gi), -gdx))
        return symmjumplist

    def diffgenerate(self, S1, S2, threshold=1e-8):
        """
        Construct a starSet using endpoint subtraction from starset S1 to starset S2. Can (will)
        include zero. Points from vacancy states of S1 to vacancy states of S2.
        :param S1: starSet for start
        :param S2: starSet for final
        :param threshold: threshold for sorting magnitudes (can influence symmetry efficiency)
        """
        if S1.Nshells < 1 or S2.Nshells < 1: raise ValueError('Need to initialize stars')
        self.Nshells = S1.Nshells + S2.Nshells  # an estimate...
        self.states = []
        for s1 in S1.states:
            for s2 in S2.states:
                # this try/except structure lets us attempt addition and kick out if not possible
                try: s = s2 ^ s1  # points from vacancy state of s1 to vacancy state of s2
                except: continue
                # now we include zero.
                if not any(s == st for st in self.states): self.states.append(s)
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


class VectorStarSet(object):
    """
    A class to construct vector star sets, and be able to efficiently index.
    """
    def __init__(self, starset=None):
        """
        Initiates a vector-star generator; work with a given star.
        :param starset: StarSet, from which we pull nearly all of the info that we need

        vecpos: list of "positions" (state indices) for each vector star (list of lists)
        vecvec: list of vectors for each vector star (list of lists of vectors)
        Nvstars: number of vector stars
        """
        self.starset = None
        self.Nvstars = 0
        if starset is not None:
            self.Nstars = starset.Nstars
            if starset.Nshells > 0:
                self.generate(starset)

    def generate(self, starset, threshold=1e-8):
        """
        Construct the actual vectors stars
        :param starset: StarSet, from which we pull nearly all of the info that we need
        """
        if starset.Nshells == 0: return
        if starset == self.starset: return
        self.starset = starset
        self.vecpos = []
        self.vecvec = []
        states = starset.states
        for s in starset.stars:
            # start by generating the parallel star-vector; always trivially present:
            self.vecpos.append(s.copy())
            PS0 = states[s[0]]
            vpara = PS0.dx
            scale = 1./np.sqrt(len(s)*np.dot(vpara, vpara)) # normalization factor
            self.vecvec.append([states[si].dx*scale for si in s])
            # next, try to generate perpendicular star-vectors, if present:
            v0 = np.cross(vpara, np.array([0, 0, 1.]))
            if np.dot(v0, v0) < threshold:
                v0 = np.cross(vpara, np.array([1., 0, 0]))
            v1 = np.cross(vpara, v0)
            # normalization:
            v0 /= np.sqrt(np.dot(v0, v0))
            v1 /= np.sqrt(np.dot(v1, v1))
            Nvect = 2
            # run over the invariant group operations for state PS0
            for g in self.starset.crys.G:
                if Nvect == 0: continue
                if PS0 != PS0.g(starset.crys, starset.chem, g): continue
                gv0 = starset.crys.g_direc(g, v0)
                if Nvect == 1:
                    # we only need to check that we still have an invariant vector
                    if any(abs(gv0 - v0) > threshold): Nvect = 0
                if Nvect == 2:
                    gv1 = starset.crys.g_direc(g, v1)
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
                    self.vecpos.append(s.copy())
                    veclist = []
                    for PSi in [states[si] for si in s]:
                        for g in starset.crys.G:
                            if PS0.g(starset.crys, starset.chem, g) == PSi:
                                veclist.append(starset.crys.g_direc(g, v))
                                break
                    self.vecvec.append(veclist)
        self.Nvstars = len(self.vecpos)
        self.outer = self.generateouter()

    def generateouter(self):
        """
        Generate our outer products for our star-vectors.
        :return outer: array [3, 3, Nvstars, Nvstars]
            outer[:, :, i, j] is the 3x3 tensor outer product for two vector-stars vs[i] and vs[j]
        """
        outer = np.zeros((3, 3, self.Nvstars, self.Nvstars))
        for i, (sR0, sv0) in enumerate(zip(self.vecpos, self.vecvec)):
            for j, (sR1, sv1) in enumerate(zip(self.vecpos, self.vecvec)):
                if sR0[0] == sR1[0]:
                    outer[:, :, i, j] = sum([np.outer(v0, v1) for v0, v1 in zip(sv0, sv1)])
        return outer

    def GFexpansion(self):
        """
        Construct the GF matrix expansion in terms of the star vectors, and indexed
        to GFstarset
        :return GFexpansion: array[Nsv, Nsv, NGFstars]
            the GF matrix[i, j] = GFexpansion[i, j, 0]*GF(0) + sum(GFexpansion[i, j, k+1] * GF(starGF[k]))
        :return GFstarset: starSet corresponding to the GF
        """
        if self.Nvstars == 0:
            return None
        GFstarset = self.starset.copy(empty=True)
        GFstarset.diffgenerate(self.starset, self.starset)
        GFexpansion = np.zeros((self.Nvstars, self.Nvstars, GFstarset.Nstars))
        for i in range(self.Nvstars):
            for j in range(self.Nvstars):
                if i <= j :
                    for si, vi in zip(self.vecpos[i], self.vecvec[i]):
                        for sj, vj in zip(self.vecpos[j], self.vecvec[j]):
                            try: ds = self.starset.states[sj] ^ self.starset.states[si]
                            except: continue
                            k = GFstarset.starindex(ds)
                            if k is None: raise ArithmeticError('GF star not large enough to include {}?'.format(ds))
                            GFexpansion[i, j, k] += np.dot(vi, vj)
                else:
                    GFexpansion[i, j, :] = GFexpansion[j, i, :]
        return GFexpansion, GFstarset

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

