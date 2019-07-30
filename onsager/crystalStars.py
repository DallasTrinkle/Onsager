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

__author__ = 'Dallas R. Trinkle'

import numpy as np
import collections
import copy
import itertools
from onsager import crystal

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
    def zero(cls, n=0, dim=3):
        """Return a "zero" state"""
        return cls(i=n, j=n, R=np.zeros(dim, dtype=int), dx=np.zeros(dim))

    @classmethod
    def fromcrys(cls, crys, chem, ij, dx):
        """Convert (i,j), dx into PairState"""
        return cls(i=ij[0],
                   j=ij[1],
                   R=np.round(np.dot(crys.invlatt, dx) - crys.basis[chem][ij[1]] + crys.basis[chem][ij[0]]).astype(int),
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
        return np.allclose(self.dx, np.dot(crys.lattice, self.R + crys.basis[chem][self.j] - crys.basis[chem][self.i]))

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
        # return self.i ^ (self.j << 1) ^ (self.R[0] << 2) ^ (self.R[1] << 3) ^ (self.R[2] << 4)
        return hash((self.i, self.j) + tuple(self.R))

    def __add__(self, other):
        """Add two states: works if and only if self.j == other.i
        (i,j) R + (j,k) R' = (i,k) R+R'  : works for thinking about transitions...
        Note: a + b != b + a, and may be that only one of those is even defined
        """
        if not isinstance(other, self.__class__): return NotImplemented
        if self.iszero() and self.j == -1: return other
        if other.iszero() and other.i == -1: return self
        if self.j != other.i:
            raise ArithmeticError(
                'Can only add matching endpoints: ({} {})+({} {}) not compatible'.format(self.i, self.j, other.i,
                                                                                         other.j))
        return self.__class__(i=self.i, j=other.j, R=self.R + other.R, dx=self.dx + other.dx)

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
        # if self.iszero(): raise ArithmeticError('Cannot endpoint substract from zero')
        # if other.iszero(): raise ArithmeticError('Cannot endpoint subtract zero')
        if self.i != other.i:
            raise ArithmeticError(
                'Can only endpoint subtract matching starts: ({} {})^({} {}) not compatible'.format(self.i, self.j,
                                                                                                    other.i, other.j))
        return self.__class__(i=other.j, j=self.j, R=self.R - other.R, dx=self.dx - other.dx)

    def g(self, crys, chem, g):
        """
        Apply group operation.

        :param crys: crystal
        :param chem: chemical index
        :param g: group operation (from crys)
        :return g*PairState: corresponding to group operation applied to self
        """
        gRi, (c, gi) = crys.g_pos(g, np.zeros(len(self.R), dtype=int), (chem, self.i))
        gRj, (c, gj) = crys.g_pos(g, self.R, (chem, self.j))
        gdx = crys.g_direc(g, self.dx)
        return self.__class__(i=gi, j=gj, R=gRj - gRi, dx=gdx)

    def __str__(self):
        """Human readable version"""
        if len(self.R) == 3:
            return "{}.[0,0,0]:{}.[{},{},{}] (dx=[{},{},{}])".format(self.i, self.j,
                                                                     self.R[0], self.R[1], self.R[2],
                                                                     self.dx[0], self.dx[1], self.dx[2])
        else:
            return "{}.[0,0]:{}.[{},{}] (dx=[{},{}])".format(self.i, self.j,
                                                             self.R[0], self.R[1],
                                                             self.dx[0], self.dx[1])

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
        # ** turns the dictionary into parameters for PairState constructor
        return PairState(**loader.construct_mapping(node, deep=True))


crystal.yaml.add_representer(PairState, PairState.PairState_representer)
crystal.yaml.add_constructor(PAIRSTATE_YAMLTAG, PairState.PairState_constructor)


# HDF5 conversion routines: PairState, and list-of-list structures
def PSlist2array(PSlist):
    """
    Take in a list of pair states; return arrays that can be stored in HDF5 format

    :param PSlist: list of pair states
    :return ij: int_array[N][2] = (i,j)
    :return R: int[N][3]
    :return dx: float[N][3]
    """
    N = len(PSlist)
    ij = np.zeros((N, 2), dtype=int)
    dim = len(PSlist[0].R)
    R = np.zeros((N, dim), dtype=int)
    dx = np.zeros((N, dim))
    for n, PS in enumerate(PSlist):
        ij[n, 0], ij[n, 1], R[n, :], dx[n, :] = PS.i, PS.j, PS.R, PS.dx
    return ij, R, dx


def array2PSlist(ij, R, dx):
    """
    Take in arrays of ij, R, dx (from HDF5), return a list of PairStates

    :param ij: int_array[N][2] = (i,j)
    :param R: int[N][3]
    :param dx: float[N][3]
    :return PSlist: list of pair states
    """
    return [PairState(i=ij0[0], j=ij0[1], R=R0, dx=dx0) for ij0, R0, dx0 in zip(ij, R, dx)]


def doublelist2flatlistindex(listlist):
    """
    Takes a list of lists, returns a flattened list and an index array

    :param listlist: list of lists of objects
    :return flatlist: flat list of objects (preserving order)
    :return indexarray: array indexing which original list it came from
    """
    flatlist = []
    indexlist = []
    for ind, entries in enumerate(listlist):
        flatlist += entries
        indexlist += [ind for j in entries]
    return flatlist, np.array(indexlist)


def flatlistindex2doublelist(flatlist, indexarray):
    """
    Takes a flattened list and an index array, returns a list of lists

    :param flatlist: flat list of objects (preserving order)
    :param indexarray: array indexing which original list it came from
    :return listlist: list of lists of objects
    """
    Nlist = max(indexarray) + 1
    listlist = [[] for n in range(Nlist)]
    for entry, ind in zip(flatlist, indexarray):
        listlist[ind].append(entry)
    return listlist


class StarSet(object):
    """
    A class to construct crystal stars, and be able to efficiently index.

    Takes in a jumpnetwork, which is used to construct the corresponding stars, a crystal
    object with which to operate, a specification of the chemical index for the atom moving
    (needs to be consistent with jumpnetwork and crys), and then the number of shells.

    In this case, ``shells`` = number of successive "jumps" from a state. As an example,
    in FCC, 1 shell = 1st neighbor, 2 shell = 1-4th neighbors.
    """

    def __init__(self, jumpnetwork, crys, chem, Nshells=0, originstates=False, lattice=False):
        """
        Initiates a star set generator for a given jumpnetwork, crystal, and specified
        chemical index. Does not include "origin states" by default; these are PairStates that
        iszero() is True; they are only needed if crystal has a nonzero VectorBasis.

        :param jumpnetwork: list of symmetry unique jumps, as a list of list of tuples; either
            ``((i,j), dx)`` for jump from i to j with displacement dx, or
            ``((i,j), R)`` for jump from i in unit cell 0 -> j in unit cell R
        :param crys: crystal where jumps take place
        :param chem: chemical index of atom to consider jumps
        :param Nshells: number of shells to generate
        :param originstates: include origin states in generate?
        :param lattice: which form does the jumpnetwork take?
        """
        # jumpnetwork_index: list of lists of indices into jumplist; matches structure of jumpnetwork
        # jumplist: list of jumps, as pair states (i=initial state, j=final state)
        # states: list of pair states, out to Nshells
        # Nstates: size of list
        # stars: list of lists of indices into states; each list are states equivalent by symmetry
        # Nstars: size of list
        # index[Nstates]: index of star that state belongs to

        # empty StarSet
        if all(x is None for x in (jumpnetwork, crys, chem)): return
        self.jumpnetwork_index = []  # list of list of indices into...
        self.jumplist = []  # list of our jumps, as PairStates
        ind = 0
        for jlist in jumpnetwork:
            self.jumpnetwork_index.append([])
            for ij, v in jlist:
                self.jumpnetwork_index[-1].append(ind)
                ind += 1
                if lattice:
                    PS = PairState.fromcrys_latt(crys, chem, ij, v)
                else:
                    PS = PairState.fromcrys(crys, chem, ij, v)
                self.jumplist.append(PS)
        self.crys = crys
        self.chem = chem
        self.generate(Nshells, threshold=crys.threshold, originstates=originstates)

    def __str__(self):
        """Human readable version"""
        str = "Nshells: {}  Nstates: {}  Nstars: {}\n".format(self.Nshells, self.Nstates, self.Nstars)
        for si in range(self.Nstars):
            str += "Star {} ({})\n".format(si, len(self.stars[si]))
            for i in self.stars[si]:
                str += "  {}: {}\n".format(i, self.states[i])
        return str

    def generate(self, Nshells, threshold=1e-8, originstates=False):
        """
        Construct the points and the stars in the set. Does not include "origin states" by default; these
        are PairStates that iszero() is True; they are only needed if crystal has a nonzero VectorBasis.

        :param Nshells: number of shells to generate; this is interpreted as subsequent
          "sums" of jumplist (as we need the solute to be connected to the vacancy by at least one jump)
        :param threshold: threshold for determining equality with symmetry
        :param originstates: include origin states in generate?
        """
        if Nshells == getattr(self, 'Nshells', -1): return
        self.Nshells = Nshells
        if Nshells > 0:
            stateset = set(self.jumplist)
        else:
            stateset = set([])
        lastshell = stateset.copy()
        if originstates:
            for i in range(len(self.crys.basis[self.chem])):
                stateset.add(PairState.zero(i, self.crys.dim))
        for i in range(Nshells - 1):
            # add all NNvect to last shell produced, always excluding 0
            # lastshell = [v1+v2 for v1 in lastshell for v2 in self.NNvect if not all(abs(v1 + v2) < threshold)]
            nextshell = set([])
            for s1 in lastshell:
                for s2 in self.jumplist:
                    # this try/except structure lets us attempt addition and kick out if not possible
                    try:
                        s = s1 + s2
                    except:
                        continue
                    if not s.iszero():
                        nextshell.add(s)
                        stateset.add(s)
            lastshell = nextshell
        # now to sort our set of vectors (easiest by magnitude, and then reduce down:
        self.states = sorted([s for s in stateset], key=PairState.sortkey)
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
                complist_stars = []  # for finding unique stars
                symmstate_list = []  # list of sets corresponding to those stars...
                for xi in range(xmin, xmax):
                    x = self.states[xi]
                    # is this a new rep. for a unique star?
                    match = False
                    for i, gs in enumerate(symmstate_list):
                        if x in gs:
                            # update star
                            complist_stars[i].append(xi)
                            match = True
                            continue
                    if not match:
                        # new symmetry point!
                        complist_stars.append([xi])
                        symmstate_list.append(set([x.g(self.crys, self.chem, g) for g in self.crys.G]))
                self.stars += complist_stars
                xmin = xmax
        else:
            self.stars = [[]]
        self.Nstars = len(self.stars)
        # generate index: which star is each state a member of?
        self.index = np.zeros(self.Nstates, dtype=int)
        self.indexdict = {}
        for si, star in enumerate(self.stars):
            for xi in star:
                self.index[xi] = si
                self.indexdict[self.states[xi]] = (xi, si)

    def addhdf5(self, HDF5group):
        """
        Adds an HDF5 representation of object into an HDF5group (needs to already exist).

        Example: if f is an open HDF5, then StarSet.addhdf5(f.create_group('StarSet')) will
        (1) create the group named 'StarSet', and then (2) put the StarSet representation in that group.

        :param HDF5group: HDF5 group
        """
        HDF5group.attrs['type'] = self.__class__.__name__
        HDF5group.attrs['crystal'] = self.crys.__repr__()
        HDF5group.attrs['chem'] = self.chem
        HDF5group['Nshells'] = self.Nshells
        # convert jumplist (list of PS) into arrays to store:
        HDF5group['jumplist_ij'], HDF5group['jumplist_R'], HDF5group['jumplist_dx'] = \
            PSlist2array(self.jumplist)
        HDF5group['jumplist_Nunique'] = len(self.jumpnetwork_index)
        jumplistinvmap = np.zeros(len(self.jumplist), dtype=int)
        for j, jlist in enumerate(self.jumpnetwork_index):
            for i in jlist: jumplistinvmap[i] = j
        HDF5group['jumplist_invmap'] = jumplistinvmap
        # convert states into arrays to store:
        HDF5group['states_ij'], HDF5group['states_R'], HDF5group['states_dx'] = \
            PSlist2array(self.states)
        HDF5group['states_index'] = self.index

    @classmethod
    def loadhdf5(cls, crys, HDF5group):
        """
        Creates a new StarSet from an HDF5 group.

        :param crys: crystal object--MUST BE PASSED IN as it is not stored with the StarSet
        :param HDFgroup: HDF5 group
        :return StarSet: new StarSet object
        """
        SSet = cls(None, None, None)  # initialize
        SSet.crys = crys
        SSet.chem = HDF5group.attrs['chem']
        SSet.Nshells = HDF5group['Nshells'].value
        SSet.jumplist = array2PSlist(HDF5group['jumplist_ij'].value,
                                     HDF5group['jumplist_R'].value,
                                     HDF5group['jumplist_dx'].value)
        SSet.jumpnetwork_index = [[] for n in range(HDF5group['jumplist_Nunique'].value)]
        for i, jump in enumerate(HDF5group['jumplist_invmap'].value):
            SSet.jumpnetwork_index[jump].append(i)
        SSet.states = array2PSlist(HDF5group['states_ij'].value,
                                   HDF5group['states_R'].value,
                                   HDF5group['states_dx'].value)
        SSet.Nstates = len(SSet.states)
        SSet.index = HDF5group['states_index'].value
        # construct the states, and the index dictionary:
        SSet.Nstars = max(SSet.index) + 1
        SSet.stars = [[] for n in range(SSet.Nstars)]
        SSet.indexdict = {}
        for xi, si in enumerate(SSet.index):
            SSet.stars[si].append(xi)
            SSet.indexdict[SSet.states[xi]] = (xi, si)
        return SSet

    def copy(self, empty=False):
        """Return a copy of the StarSet; done as efficiently as possible; empty means skip the shells, etc."""
        newStarSet = self.__class__(None, None, None)  # a little hacky... creates an empty class
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
            newStarSet.indexdict = self.indexdict.copy()
        else:
            newStarSet.generate(0)
        return newStarSet

    # removed combine; all it does is generate(s1.Nshells + s2.Nshells) with lots of checks...
    # replaced with (more efficient?) __add__ and __iadd__.

    def __add__(self, other):
        """Add two StarSets together; done by making a copy of one, and iadding"""
        if not isinstance(other, self.__class__): return NotImplemented
        if self.Nshells >= other.Nshells:
            scopy = self.copy()
            scopy.__iadd__(other)
        else:
            scopy = other.copy()
            scopy.__iadd__(self)
        return scopy

    def __iadd__(self, other):
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
            self.indexdict = other.indexdict.copy()
            return self
        self.Nshells += other.Nshells
        Nold = self.Nstates
        oldstateset = set(self.states)
        newstateset = set([])
        for s1 in self.states[:Nold]:
            for s2 in other.states:
                # this try/except structure lets us attempt addition and kick out if not possible
                try:
                    s = s1 + s2
                except:
                    continue
                if not s.iszero() and not s in oldstateset: newstateset.add(s)
        # now to sort our set of vectors (easiest by magnitude, and then reduce down:
        self.states += sorted([s for s in newstateset], key=PairState.sortkey)
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
            complist_stars = []  # for finding unique stars
            symmstate_list = []  # list of sets corresponding to those stars...
            for xi in range(xmin, xmax):
                x = self.states[xi]
                # is this a new rep. for a unique star?
                match = False
                for i, gs in enumerate(symmstate_list):
                    if x in gs:
                        # update star
                        complist_stars[i].append(xi)
                        match = True
                        continue
                if not match:
                    # new symmetry point!
                    complist_stars.append([xi])
                    symmstate_list.append(set([x.g(self.crys, self.chem, g) for g in self.crys.G]))
            self.stars += complist_stars
            xmin = xmax
        self.Nstates = Nnew
        # generate new index entries: which star is each state a member of?
        self.index = np.pad(self.index, (0, Nnew - Nold), mode='constant')
        Nold = self.Nstars
        Nnew = len(self.stars)
        for si in range(Nold, Nnew):
            star = self.stars[si]
            for xi in star:
                self.index[xi] = si
                self.indexdict[self.states[xi]] = (xi, si)
        self.Nstars = Nnew
        return self

    def __contains__(self, PS):
        """Return true if PS is in the star"""
        return PS in self.indexdict

    # replaces pointindex:
    def stateindex(self, PS):
        """Return the index of pair state PS; None if not found"""
        try:
            return self.indexdict[PS][0]
        except:
            return None

    def starindex(self, PS):
        """Return the index for the star to which pair state PS belongs; None if not found"""
        try:
            return self.indexdict[PS][1]
        except:
            return None

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
            symmetry unique jump; useful for constructing a LIMB approximation, and needed to
            construct delta_omega
        :return starpair: list of tuples of the star indices of the i and f states for each
            symmetry unique jump
        """
        if self.Nshells < 1: return []
        jumpnetwork = []
        jumptype = []
        starpair = []
        for jt, jumpindices in enumerate(self.jumpnetwork_index):
            for jump in [self.jumplist[j] for j in jumpindices]:
                for i, PSi in enumerate(self.states):
                    if PSi.iszero(): continue
                    # attempt to add...
                    try:
                        PSf = PSi + jump
                    except:
                        continue
                    if PSf.iszero(): continue
                    f = self.stateindex(PSf)
                    if f is None: continue  # outside our StarSet
                    # see if we've already generated this jump (works since all of our states are distinct)
                    if any(any(i == i0 and f == f0 for (i0, f0), dx in jlist) for jlist in jumpnetwork): continue
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
            symmetry unique jump; useful for constructing a LIMB approximation, and needed to
            construct delta_omega
        :return starpair: list of tuples of the star indices of the i and f states for each
            symmetry unique jump
        """
        if self.Nshells < 1: return []
        jumpnetwork = []
        jumptype = []
        starpair = []
        for jt, jumpindices in enumerate(self.jumpnetwork_index):
            for jump in [self.jumplist[j] for j in jumpindices]:
                for i, PSi in enumerate(self.states):
                    if PSi.iszero(): continue
                    # attempt to add...
                    try:
                        PSf = PSi + jump
                    except:
                        continue
                    if not PSf.iszero(): continue
                    f = self.stateindex(-PSi)  # exchange
                    # see if we've already generated this jump (works since all of our states are distinct)
                    if any(any(i == i0 and f == f0 for (i0, f0), dx in jlist) for jlist in jumpnetwork): continue
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
        symmjumplist = [((i, f), dx)]
        if i != f: symmjumplist.append(((f, i), -dx))  # i should not equal f... but in case we allow 0 as a jump
        for g in self.crys.G:
            gi, gf, gdx = self.stateindex(PSi.g(self.crys, self.chem, g)), \
                          self.stateindex(PSf.g(self.crys, self.chem, g)), \
                          self.crys.g_direc(g, dx)
            if not any(gi == i0 and gf == f0 for (i0, f0), dx in symmjumplist):
                symmjumplist.append(((gi, gf), gdx))
                if gi != gf: symmjumplist.append(((gf, gi), -gdx))
        return symmjumplist

    def diffgenerate(self, S1, S2, threshold=1e-8):
        """
        Construct a starSet using endpoint subtraction from starset S1 to starset S2. Will
        include zero. Points from vacancy states of S1 to vacancy states of S2.

        :param S1: starSet for start
        :param S2: starSet for final
        :param threshold: threshold for sorting magnitudes (can influence symmetry efficiency)
        """
        if S1.Nshells < 1 or S2.Nshells < 1: raise ValueError('Need to initialize stars')
        self.Nshells = S1.Nshells + S2.Nshells  # an estimate...
        stateset = set([])
        # self.states = []
        for s1 in S1.states:
            for s2 in S2.states:
                # this try/except structure lets us attempt addition and kick out if not possible
                try:
                    s = s2 ^ s1  # points from vacancy state of s1 to vacancy state of s2
                except:
                    continue
                stateset.add(s)
        # now to sort our set of vectors (easiest by magnitude, and then reduce down:
        self.states = sorted([s for s in stateset], key=PairState.sortkey)
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
                complist_stars = []  # for finding unique stars
                symmstate_list = []  # list of sets corresponding to those stars...
                for xi in range(xmin, xmax):
                    x = self.states[xi]
                    # is this a new rep. for a unique star?
                    match = False
                    for i, gs in enumerate(symmstate_list):
                        if x in gs:
                            # update star
                            complist_stars[i].append(xi)
                            match = True
                            continue
                    if not match:
                        # new symmetry point!
                        complist_stars.append([xi])
                        symmstate_list.append(set([x.g(self.crys, self.chem, g) for g in self.crys.G]))
                self.stars += complist_stars
                xmin = xmax
        else:
            self.stars = [[]]
        self.Nstars = len(self.stars)
        # generate index: which star is each state a member of?
        self.index = np.zeros(self.Nstates, dtype=int)
        self.indexdict = {}
        for si, star in enumerate(self.stars):
            for xi in star:
                self.index[xi] = si
                self.indexdict[self.states[xi]] = (xi, si)


def zeroclean(x, threshold=1e-8):
    """Modify x in place, return 0 if x is below a threshold; useful for "symmetrizing" our expansions"""
    for v in np.nditer(x, op_flags=['readwrite']):
        if abs(v) < threshold: v[...] = 0
    return x


class VectorStarSet(object):
    """
    A class to construct vector star sets, and be able to efficiently index.

    All based on a StarSet
    """

    def __init__(self, starset=None):
        """
        Initiates a vector-star generator; work with a given star.

        :param starset: StarSet, from which we pull nearly all of the info that we need
        """
        # vecpos: list of "positions" (state indices) for each vector star (list of lists)
        # vecvec: list of vectors for each vector star (list of lists of vectors)
        # Nvstars: number of vector stars

        self.starset = None
        self.Nvstars = 0
        if starset is not None:
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
        dim = starset.crys.dim
        self.vecpos = []
        self.vecvec = []
        states = starset.states
        for s in starset.stars:
            # start by generating the parallel star-vector; always trivially present:
            PS0 = states[s[0]]
            if PS0.iszero():
                # origin state; we can easily generate our vlist
                vlist = starset.crys.vectlist(starset.crys.VectorBasis((self.starset.chem, PS0.i)))
                scale = 1. / np.sqrt(len(s))  # normalization factor; vectors are already normalized
                vlist = [v * scale for v in vlist]
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
            else:
                # not an origin state
                vpara = PS0.dx
                scale = 1. / np.sqrt(len(s) * np.dot(vpara, vpara))  # normalization factor
                self.vecpos.append(s.copy())
                self.vecvec.append([states[si].dx * scale for si in s])
                # next, try to generate perpendicular star-vectors, if present:
                if dim == 3:
                    v0 = np.cross(vpara, np.array([0, 0, 1.]))
                    if np.dot(v0, v0) < threshold:
                        v0 = np.cross(vpara, np.array([1., 0, 0]))
                    v1 = np.cross(vpara, v0)
                    # normalization:
                    v0 /= np.sqrt(np.dot(v0, v0))
                    v1 /= np.sqrt(np.dot(v1, v1))
                    Nvect = 2
                else:
                    # 2d is very simple...
                    v0 = np.array([vpara[1], -vpara[0]])
                    v0 /= np.sqrt(np.dot(v0, v0))
                    Nvect = 1
                # run over the invariant group operations for state PS0
                for g in self.starset.crys.G:
                    if Nvect == 0: continue
                    if PS0 != PS0.g(starset.crys, starset.chem, g): continue
                    gv0 = starset.crys.g_direc(g, v0)
                    if Nvect == 1:
                        # we only need to check that we still have an invariant vector
                        if not np.isclose(np.dot(v0, v0), 1): raise ArithmeticError('Somehow got unnormalized vector?')
                        if not np.allclose(gv0, v0): Nvect = 0
                    if Nvect == 2:
                        if not np.isclose(np.dot(v0, v0), 1): raise ArithmeticError('Somehow got unnormalized vector?')
                        if not np.isclose(np.dot(v1, v1), 1): raise ArithmeticError('Somehow got unnormalized vector?')
                        gv1 = starset.crys.g_direc(g, v1)
                        g00 = np.dot(v0, gv0)
                        g11 = np.dot(v1, gv1)
                        g01 = np.dot(v0, gv1)
                        g10 = np.dot(v1, gv0)
                        if abs((abs(g00 * g11 - g01 * g10) - 1)) > threshold or abs(g01 - g10) > threshold:
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
                            v0 = (g01 * v0 + (1 - g00) * v1) / np.sqrt(g01 * g10 + (1 - g00) ** 2)
                            Nvect = 1
                # so... do we have any vectors to add?
                if Nvect > 0:
                    v0 /= np.sqrt(len(s) * np.dot(v0, v0))
                    vlist = [v0]
                    if Nvect > 1:
                        v1 /= np.sqrt(len(s) * np.dot(v1, v1))
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
        # dim = len(self.vecvec[0][0])
        dim = self.starset.crys.dim
        outer = np.zeros((dim, dim, self.Nvstars, self.Nvstars))
        for i, sR0, sv0 in zip(itertools.count(), self.vecpos, self.vecvec):
            for j, sR1, sv1 in zip(itertools.count(), self.vecpos, self.vecvec):
                if sR0[0] == sR1[0]:
                    outer[:, :, i, j] = sum([np.outer(v0, v1) for v0, v1 in zip(sv0, sv1)])
        return zeroclean(outer)

    def addhdf5(self, HDF5group):
        """
        Adds an HDF5 representation of object into an HDF5group (needs to already exist).

        Example: if f is an open HDF5, then StarSet.addhdf5(f.create_group('VectorStarSet')) will
          (1) create the group named 'VectorStarSet', and then (2) put the VectorStarSet
          representation in that group.

        :param HDF5group: HDF5 group
        """
        HDF5group.attrs['type'] = self.__class__.__name__
        HDF5group['Nvstars'] = self.Nvstars
        HDF5group['vecposlist'], HDF5group['vecposindex'] = doublelist2flatlistindex(self.vecpos)
        HDF5group['vecveclist'], HDF5group['vecvecindex'] = doublelist2flatlistindex(self.vecvec)
        HDF5group['outer'] = self.outer

    @classmethod
    def loadhdf5(cls, SSet, HDF5group):
        """
        Creates a new VectorStarSet from an HDF5 group.

        :param SSet: StarSet--MUST BE PASSED IN as it is not stored with the VectorStarSet
        :param HDFgroup: HDF5 group
        :return VectorStarSet: new VectorStarSet object
        """
        VSSet = cls(None)  # initialize
        VSSet.starset = SSet
        VSSet.Nvstars = HDF5group['Nvstars'].value
        VSSet.vecpos = flatlistindex2doublelist(HDF5group['vecposlist'].value,
                                                HDF5group['vecposindex'].value)
        VSSet.vecvec = flatlistindex2doublelist(HDF5group['vecveclist'].value,
                                                HDF5group['vecvecindex'].value)
        VSSet.outer = HDF5group['outer'].value
        return VSSet

    def GFexpansion(self):
        """
        Construct the GF matrix expansion in terms of the star vectors, and indexed
        to GFstarset.

        :return GFexpansion: array[Nsv, Nsv, NGFstars]
            the GF matrix[i, j] = sum(GFexpansion[i, j, k] * GF(starGF[k]))
        :return GFstarset: starSet corresponding to the GF
        """
        if self.Nvstars == 0:
            return None
        GFstarset = self.starset.copy(empty=True)
        GFstarset.diffgenerate(self.starset, self.starset)
        GFexpansion = np.zeros((self.Nvstars, self.Nvstars, GFstarset.Nstars))
        for i in range(self.Nvstars):
            for si, vi in zip(self.vecpos[i], self.vecvec[i]):
                for j in range(i, self.Nvstars):
                    for sj, vj in zip(self.vecpos[j], self.vecvec[j]):
                        try:
                            ds = self.starset.states[sj] ^ self.starset.states[si]
                        except:
                            continue
                        k = GFstarset.starindex(ds)
                        if k is None: raise ArithmeticError('GF star not large enough to include {}?'.format(ds))
                        GFexpansion[i, j, k] += np.dot(vi, vj)
        # symmetrize
        for i in range(self.Nvstars):
            for j in range(0, i):
                GFexpansion[i, j, :] = GFexpansion[j, i, :]
        # cleanup on return:
        return zeroclean(GFexpansion), GFstarset

    def rateexpansions(self, jumpnetwork, jumptype, omega2=False):
        """
        Construct the omega0 and omega1 matrix expansions in terms of the jumpnetwork;
        includes the escape terms separately. The escape terms are tricky because they have
        probability factors that differ from the transitions; the PS (pair stars) is useful for
        finding this. We just call it the 'probfactor' below.
        *Note:* this used to be separated into rate0expansion, and rate1expansion, and
        partly in bias1expansion. Note also that if jumpnetwork_omega2 is passed, it also works
        for that. However, in that case we have a different approach for the calculation of
        rate0expansion: if there are origin states, then we need to "jump" to those; if there
        is a non-empty VectorBasis we will want to account for them there.

        :param jumpnetwork: jumpnetwork of symmetry unique omega1-type jumps,
            corresponding to our starset. List of lists of (IS, FS), dx tuples, where IS and FS
            are indices corresponding to states in our starset.
        :param jumptype: specific omega0 jump type that the jump corresponds to
        :param omega2: (optional) are we dealing with the omega2 list, so we need to remove
            origin states? (default=False)
        :return rate0expansion: array[Nsv, Nsv, Njump_omega0]
            the omega0 matrix[i, j] = sum(rate0expansion[i, j, k] * omega0[k]); *IF* NVB>0
            we "hijack" this and use it for [NVB, Nsv, Njump_omega0], as we're doing an omega2
            calc and rate0expansion won't be used *anyway*.
        :return rate0escape: array[Nsv, Njump_omega0]
            the escape contributions: omega0[i,i] += sum(rate0escape[i,k]*omega0[k]*probfactor0(PS[k]))
        :return rate1expansion: array[Nsv, Nsv, Njump_omega1]
            the omega1 matrix[i, j] = sum(rate1expansion[i, j, k] * omega1[k])
        :return rate1escape: array[Nsv, Njump_omega1]
            the escape contributions: omega1[i,i] += sum(rate1escape[i,k]*omega1[k]*probfactor(PS[k]))
        """
        if self.Nvstars == 0: return None
        rate0expansion = np.zeros((self.Nvstars, self.Nvstars, len(self.starset.jumpnetwork_index)))
        rate1expansion = np.zeros((self.Nvstars, self.Nvstars, len(jumpnetwork)))
        rate0escape = np.zeros((self.Nvstars, len(self.starset.jumpnetwork_index)))
        rate1escape = np.zeros((self.Nvstars, len(jumpnetwork)))
        for k, jumplist, jt in zip(itertools.count(), jumpnetwork, jumptype):
            for (IS, FS), dx in jumplist:
                for i in range(self.Nvstars):
                    for Ri, vi in zip(self.vecpos[i], self.vecvec[i]):
                        if Ri == IS:
                            rate0escape[i, jt] -= np.dot(vi, vi)
                            rate1escape[i, k] -= np.dot(vi, vi)
                            # for j in range(i+1):
                            for j in range(self.Nvstars):
                                for Rj, vj in zip(self.vecpos[j], self.vecvec[j]):
                                    if Rj == FS:
                                        if not omega2: rate0expansion[i, j, jt] += np.dot(vi, vj)
                                        rate1expansion[i, j, k] += np.dot(vi, vj)
                            if omega2:
                                # find the "origin state" corresponding to the solute; "remove" those rates
                                OSindex = self.starset.stateindex(PairState.zero(self.starset.states[IS].i,
                                                                                 self.starset.crys.dim))
                                if OSindex is not None:
                                    for j in range(self.Nvstars):
                                        for Rj, vj in zip(self.vecpos[j], self.vecvec[j]):
                                            if Rj == OSindex:
                                                rate0expansion[i, j, jt] += np.dot(vi, vj)
                                                rate0expansion[j, i, jt] += np.dot(vi, vj)
                                                rate0escape[j, jt] -= np.dot(vj, vj)
        # cleanup on return
        return zeroclean(rate0expansion), zeroclean(rate0escape), \
               zeroclean(rate1expansion), zeroclean(rate1escape)

    def biasexpansions(self, jumpnetwork, jumptype, omega2=False):
        """
        Construct the bias1 and bias0 vector expansion in terms of the jumpnetwork.
        We return the bias0 contribution so that the db = bias1 - bias0 can be determined.
        This saves us from having to deal with issues with our outer shell where we only
        have a fraction of the escapes, but as long as the kinetic shell is one more than
        the thermodynamics (so that the interaction energy is 0, hence no change in probability),
        this will work. The PS (pair stars) is useful for including the probability factor
        for the endpoint of the jump; we just call it the 'probfactor' below.
        *Note:* this used to be separated into bias1expansion, and bias2expansion,and
        had terms that are now in rateexpansions.
        Note also that if jumpnetwork_omega2 is passed, it also works for that. However,
        in that case we have a different approach for the calculation of bias1expansion:
        if there are origin states, they get the negative summed bias of the others.

        :param jumpnetwork: jumpnetwork of symmetry unique omega1-type jumps,
            corresponding to our starset. List of lists of (IS, FS), dx tuples, where IS and FS
            are indices corresponding to states in our starset.
        :param jumptype: specific omega0 jump type that the jump corresponds to
        :param omega2: (optional) are we dealing with the omega2 list, so we need to remove
            origin states? (default=False)
        :return bias0expansion: array[Nsv, Njump_omega0]
            the gen0 vector[i] = sum(bias0expasion[i, k] * sqrt(probfactor0[PS[k]]) * omega0[k])
        :return bias1expansion: array[Nsv, Njump_omega1]
            the gen1 vector[i] = sum(bias1expansion[i, k] * sqrt(probfactor[PS[k]] * omega1[k])
        """
        if self.Nvstars == 0: return None
        bias0expansion = np.zeros((self.Nvstars, len(self.starset.jumpnetwork_index)))
        bias1expansion = np.zeros((self.Nvstars, len(jumpnetwork)))

        for k, jumplist, jt in zip(itertools.count(), jumpnetwork, jumptype):
            for (IS, FS), dx in jumplist:
                # run through the star-vectors; just use first as representative
                for i, svR, svv in zip(itertools.count(), self.vecpos, self.vecvec):
                    if svR[0] == IS:
                        geom_bias = np.dot(svv[0], dx) * len(svR)
                        bias1expansion[i, k] += geom_bias
                        bias0expansion[i, jt] += geom_bias
                if omega2:
                    # find the "origin state" corresponding to the solute; incorporate the change in bias
                    OSindex = self.starset.stateindex(PairState.zero(self.starset.states[IS].i,
                                                                     self.starset.crys.dim))
                    if OSindex is not None:
                        for j in range(self.Nvstars):
                            for Rj, vj in zip(self.vecpos[j], self.vecvec[j]):
                                if Rj == OSindex:
                                    geom_bias = -np.dot(vj, dx)
                                    bias1expansion[j, k] += geom_bias  # do we need this??
                                    bias0expansion[j, jt] += geom_bias

        # cleanup on return
        return zeroclean(bias0expansion), zeroclean(bias1expansion)

    # this is *almost* a static method--it only need to know how many omega0 type jumps there are
    # in the starset. We *could* make it static and use max(jumptype), but that may not be strictly safe
    def bareexpansions(self, jumpnetwork, jumptype):
        """
        Construct the bare diffusivity expansion in terms of the jumpnetwork.
        We return the reference (0) contribution so that the change can be determined; this
        is useful for the vacancy contributions.
        This saves us from having to deal with issues with our outer shell where we only
        have a fraction of the escapes, but as long as the kinetic shell is one more than
        the thermodynamics (so that the interaction energy is 0, hence no change in probability),
        this will work. The PS (pair stars) is useful for including the probability factor
        for the endpoint of the jump; we just call it the 'probfactor' below.

        Note also: this *currently assumes* that the displacement vector *does not change* between
        omega0 and omega(1/2).

        :param jumpnetwork: jumpnetwork of symmetry unique omega1-type jumps,
            corresponding to our starset. List of lists of (IS, FS), dx tuples, where IS and FS
            are indices corresponding to states in our starset.
        :param jumptype: specific omega0 jump type that the jump corresponds to
        :return D0expansion: array[3,3, Njump_omega0]
            the D0[a,b,jt] = sum(D0expansion[a,b, jt] * sqrt(probfactor0[PS[jt][0]]*probfactor0[PS[jt][1]) * omega0[jt])
        :return D1expansion: array[3,3, Njump_omega1]
            the D1[a,b,k] = sum(D1expansion[a,b, k] * sqrt(probfactor[PS[k][0]]*probfactor[PS[k][1]) * omega[k])
        """
        if self.Nvstars == 0: return None
        # dim = len(jumpnetwork[0][0][1])
        dim = self.starset.crys.dim
        D0expansion = np.zeros((dim, dim, len(self.starset.jumpnetwork_index)))
        D1expansion = np.zeros((dim, dim, len(jumpnetwork)))
        for k, jt, jumplist in zip(itertools.count(), jumptype, jumpnetwork):
            d0 = sum(0.5 * np.outer(dx, dx) for ISFS, dx in jumplist)  # we don't need initial/final state
            D0expansion[:, :, jt] += d0
            D1expansion[:, :, k] += d0
        # cleanup on return
        return zeroclean(D0expansion), zeroclean(D1expansion)

    def originstateVectorBasisfolddown(self, elemtype='solute'):
        """
        Construct the expansion to "fold down" from vector stars to origin states.

        :param elemtype: 'solute' of 'vacancy', depending on which site we need to reduce
        :return OSindices: list of indices corresponding to origin states
        :return folddown: [NOS, Nvstars] to map vector stars to origin states
        :return OS_VB: [NOS, Nsites, 3] mapping of origin state to a vector basis
        """
        attr = {'solute': 'i', 'vacancy': 'j'}.get(elemtype)
        if attr is None: raise ValueError('elemtype needs to be "solute" or "vacancy" not {}'.format(elemtype))
        OSindices = [n for n in range(self.Nvstars) if self.starset.states[self.vecpos[n][0]].iszero()]
        NOS, Nsites = len(OSindices), len(self.starset.crys.basis[self.starset.chem])
        folddown = np.zeros((NOS, self.Nvstars))
        # dim = len(self.vecvec[0][0])
        dim = self.starset.crys.dim
        OS_VB = np.zeros((NOS, Nsites, dim))
        if NOS==0:
            return OSindices, folddown, OS_VB
        for i, ni in enumerate(OSindices):
            for OS, OSv in zip(self.vecpos[ni], self.vecvec[ni]):
                index = getattr(self.starset.states[OS], attr)
                OS_VB[i, index, :] = OSv[:]
                for j, svR, svv in zip(itertools.count(), self.vecpos, self.vecvec):
                    for s, v in zip(svR, svv):
                        if getattr(self.starset.states[s], attr) == index:
                            folddown[i, j] += np.dot(OSv, v)
        # cleanup on return
        return OSindices, zeroclean(folddown), zeroclean(OS_VB)
