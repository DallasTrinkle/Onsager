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
        self.generate(Nshells, originstates)

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

# Dumbbells start here
from onsager.crystal import DB_disp, DB_disp4, pureDBContainer, mixedDBContainer
from onsager.DB_structs import *
from onsager.DB_collisions import *
from collections import defaultdict
import time
from functools import reduce

class DBStarSet(object):
    """
    class to form the crystal stars (or state orbits) of complex solute-dumbbell states, with shells indicated by the number of jumps.
    The minimum shell (Nshells=0) is composed of dumbbells situated atleast one jump away.
    Contains mixed dumbbell states as well.
    """

    def __init__(self, pdbcontainer, mdbcontainer, jnetwrk0, jnetwrk2, Nshells=None):
        """
        To create solute-dumbbell state orbits, we'll use the pure and mixed dumbbell containers and the jump
        network to build the shells.

        :param pdbcontainer: pureDBContainer container object (see crystal.py) containing the pure dumbbell information.
        :param mdbcontainer: mixedDBContainer container object (see crystal.py) containing the mixed dumbbell information.
        :param jnetwrk0: omega0 jumps of pure dumbbells generated from pdbcontainer. Must be a tuple containing the two
        outputs of the pdbcontainer's jumpnetwork function (see crystal.py).
        :param jnetwrk2: omega2 jumps of mixed dumbbells generated from mdbcontainer. Must be a tuple containing the two
        outputs of the mdbcontainer's jumpnetwork function (see crystal.py).
        :param Nshells: number of shells. Minimum (one jump away) corresponds to Nshells=0
        """
        # check that we have the same crystal structures for pdbcontainer and mdbcontainer
        if not np.allclose(pdbcontainer.crys.lattice, mdbcontainer.crys.lattice):
            raise TypeError("pdbcontainer and mdbcontainer have different crystals")

        if not len(pdbcontainer.crys.basis) == len(mdbcontainer.crys.basis):
            raise TypeError("pdbcontainer and mdbcontainer have different basis")

        for atom1, atom2 in zip(pdbcontainer.crys.chemistry, mdbcontainer.crys.chemistry):
            if not atom1 == atom2:
                raise TypeError("pdbcontainer and mdbcontainer basis atom types don't match")
        for l1, l2 in zip(pdbcontainer.crys.basis, mdbcontainer.crys.basis):
            if not l1 == l2:
                raise TypeError("basis atom types have different numbers in pdbcontainer and mdbcontainer")

        if not pdbcontainer.chem == mdbcontainer.chem:
            raise TypeError("pdbcontainer and mdbcontainer have states on different sublattices")

        self.crys = pdbcontainer.crys
        self.chem = pdbcontainer.chem
        self.pdbcontainer = pdbcontainer
        self.mdbcontainer = mdbcontainer

        self.jnet0 = jnetwrk0[0]
        self.jnet0_ind = jnetwrk0[1]

        self.jnet2 = jnetwrk2[0]
        self.jnet2_ind = jnetwrk2[1]

        self.jumplist = [j for l in self.jnet0 for j in l]
        # self.jumpset = set(self.jumplist)

        self.jumpindices = []
        count = 0
        for l in self.jnet0:
            self.jumpindices.append([])
            for j in l:
                if isinstance(j.state1, SdPair):
                    raise TypeError("The jumpnetwork for bare dumbbells cannot have Sdpairs")
                self.jumpindices[-1].append(count)
                count += 1
        if not Nshells == None:
            self.generate(Nshells)

    def _sortkey(self, entry):
        """
        A key function to compute solute-dumbbell distances to sort the crystal stars.
        Parameter:
            entry : SdPair object
        Returns:
            (float) the distance between the solute and dumbbell sites.
        """
        sol_pos = self.crys.unit2cart(entry.R_s, self.crys.basis[self.chem][entry.i_s])
        db_pos = self.crys.unit2cart(entry.db.R,
                                     self.crys.basis[self.chem][self.pdbcontainer.iorlist[entry.db.iorind][0]])
        return np.dot(db_pos - sol_pos, db_pos - sol_pos)

    def genIndextoContainer(self, complexStates, mixedstates):
        """
        Function to get the (i, or) index of the dumbbells in their containers
        Parameters:
            complexStates : list of solute-pure dumbbell complex states as SdPair objects.
            mixedstates : list of mixed dumbbells as SdPair objects.
        Returns:
            pureDict, mixedDict : SdPair object -> (i, or) dicts containing the indices of the pure and mixed states
            in their respective containers.
        """
        pureDict = {}
        mixedDict = {}
        for st in complexStates:
            db = st.db - st.db.R
            pureDict[st] = self.pdbcontainer.iorindex[db]

        for st in mixedstates:
            db = st.db - st.db.R
            mixedDict[st] = self.mdbcontainer.iorindex[db]
        return pureDict, mixedDict

    def generate(self, Nshells):
        """
        Generate the set of all solute-dumbbell states within a cuttoff shell.

        Also indexes the states contained in the starset
        All the indexing are done into the following lists
        ->complexStates, mixedstates - lists SdPair objects, containing the complex and mixed dumbbells that make up the starset.
        States are assgined indices based on their position in these lists.

        ->stars, starindexed -> contain symmetry grouped lists of states, and the corresponding indexed version.
        --complexIndexdict, mixedindexdict -> tell us the location of a state within the starset
        Example - if the "i^th" group of states (stars[i]) contains the state "s" which is the j^th state in "complexStates",
        then complexIndexdict[s] = (j, i)

        Corresponding indices are also built for pure dumbbells.
        ->bareStates - list of dumbbell objects containing the pure dumbbells in a unit cell.
        ->barePeriodicStars, bareStarindexed - symmetry grouped pure dumbbell objects that periodically repeat in the lattice.
                              The corresponding (i, or) version can also be found in the "symorlist" and "symIndlist"
                              of the pure dumbbell container

        ->bareindexdict - gives the symmetry position and position within the symmetry list in barePeriodicStars for a pure dumbbell.

        Parameters:
            Nshells : number of shells to look at - the shells are counted in terms of the jumps.
            Example - Nshells = 2 indicated 1nn of 1nn sites from the solute will be considered.
            All dumbbell orientations are considered for every site to generate the states.
        """

        # Return nothing if Nshells are not specified
        if Nshells is None:
            return
        self.Nshells = Nshells
        z = np.zeros(self.crys.dim, dtype=int)
        if Nshells < 1:
            Nshells = 0
        stateset = set([])
        start = time.time()
        if Nshells >= 1:
            # build the starting shell
            # Build the first shell from the jump network
            # One by one, keeping the solute at the origin unit cell, put all possible dumbbell
            # states at one jump distance away.
            # We won't put in origin states just yet.
            for j in self.jumplist:
                dx = DB_disp(self.pdbcontainer, j.state1, j.state2)
                if np.allclose(dx, np.zeros(self.crys.dim, dtype=int), atol=self.pdbcontainer.crys.threshold):
                    continue
                # Now go through the all the dumbbell states in the (i,or) list:
                for idx, (i, o) in enumerate(self.pdbcontainer.iorlist):
                    if i != self.pdbcontainer.iorlist[j.state2.iorind][0]:
                        continue
                    dbstate = dumbbell(idx, j.state2.R)
                    pair = SdPair(self.pdbcontainer.iorlist[j.state1.iorind][0], j.state1.R, dbstate)
                    stateset.add(pair)

            # Now, we add in the origin states
            for ind, tup in enumerate(self.pdbcontainer.iorlist):
                pair = SdPair(tup[0], np.zeros(self.crys.dim, dtype=int), dumbbell(ind, np.zeros(self.crys.dim, dtype=int)))
                stateset.add(pair)
        print("built shell {}: time - {}".format(1, time.time() - start))
        lastshell = stateset.copy()
        # Now build the next shells:
        for step in range(Nshells - 1):
            start = time.time()
            nextshell = set([])
            for j in self.jumplist:
                for pair in lastshell:
                    if not np.allclose(pair.R_s, 0, atol=self.crys.threshold):
                        raise ValueError("The solute is not at the origin in a complex state")
                    try:
                        pairnew = pair.addjump(j)
                    except ArithmeticError:
                        # If there is somehow a type error, we will get the message.
                        continue
                    if not (pair.i_s == pairnew.i_s and np.allclose(pairnew.R_s, pair.R_s, atol=self.crys.threshold)):
                        raise ArithmeticError("Solute shifted by a complex jump!(?)")
                    # Now, when we find a new dumbbell location, we have to consider all possible orientations in that location.
                    # Let's get the dumbbell location
                    site_db, Rdb = self.pdbcontainer.iorlist[pairnew.db.iorind][0], pairnew.db.R.copy()
                    for idx, (site_db2, o) in enumerate(self.pdbcontainer.iorlist):
                        if site_db2 == site_db:  # make sure we are making dumbbells at the correct site
                            dbstateNew = dumbbell(idx, Rdb)
                            pairnew = SdPair(pair.i_s, pair.R_s, dbstateNew)
                            nextshell.add(pairnew)
                            stateset.add(pairnew)

            lastshell = nextshell.copy()
            print("built shell {}: time - {}".format(step+2, time.time()-start))

        self.stateset = stateset
        # group the states by symmetry - form the stars
        self.complexStates = sorted(list(self.stateset), key=self._sortkey)
        self.bareStates = [dumbbell(idx, z) for idx in range(len(self.pdbcontainer.iorlist))]
        stars = []
        self.complexIndexdict = {}
        starindexed=[]
        allset = set([])
        start = time.time()
        for state in self.stateset:
            if state in allset:  # see if already considered before.
                continue
            newstar = []
            newstar_index = []
            for gdumb in self.pdbcontainer.G:
                newstate = state.gop(self.pdbcontainer, gdumb)[0]
                newstate = newstate - newstate.R_s  # Shift the solute back to the origin unit cell.
                if newstate in self.stateset:  # Check if this state is allowed to be present.
                    if not newstate in allset:  # Check if this state has already been considered.
                        try:
                            newstateind = self.complexStates.index(newstate)
                        except:
                            raise KeyError("Something wrong in finding index for newstate")
                        newstar.append(newstate)
                        newstar_index.append(newstateind)
                        allset.add(newstate)
            if len(newstar) == 0:
                raise ValueError("A star must have at least one state.")
            if not len(newstar) == len(newstar_index):
                raise ValueError("star and index star have different lengths")
            stars.append(newstar)
            starindexed.append(newstar_index)
        print("grouped states by symmetry: {}".format(time.time() - start))
        self.stars = stars
        self.starindexed = starindexed
        self.sortstars()

        for starind, star in enumerate(self.stars):
            for state in star:
                self.complexIndexdict[state] = (self.complexStates.index(state), starind)

        # Keep the indices of the origin states. May be necessary when dealing with their rates and probabilities
        # self.originstates = []
        # for starind, star in enumerate(self.stars):
        #     if star[0].is_zero(self.pdbcontainer):
        #         self.originstates.append(starind)

        start = time.time()
        self.mixedstartindex = len(self.stars)
        # Now add in the mixed states
        self.mixedstates = []
        for idx, tup in enumerate(self.mdbcontainer.iorlist):
            db = dumbbell(idx, z)
            mdb = SdPair(tup[0], z, db)
            if not mdb.is_zero(self.mdbcontainer):
                raise ValueError("mdb not origin state")
            self.mixedstates.append(mdb)

        for l in self.mdbcontainer.symIndlist:
            # The sites and orientations are already grouped - convert them into SdPairs
            newlist = []
            for idx in l:
                db = dumbbell(idx, z)
                mdb = SdPair(self.mdbcontainer.iorlist[idx][0], z, db)
                newlist.append(mdb)
            self.stars.append(newlist)

        print("built mixed dumbbell stars: {}".format(time.time() - start))

        # Next, we build up the jtags for omega2 (see Onsager_calc module).
        # Note - the jtags have been tested in test_vec_star since they were added later.
        start = time.time()
        j2initlist = []
        for jt, jlist in enumerate(self.jnet2_ind):
            initindices = defaultdict(list)
            # defaultdict(list) - dictionary creator. (key, value) pairs are such that the value corresponding to a
            # given key is a list. If a key is created for the first time, an empty list is created simultaneously.
            for (i, j), dx in jlist:
                initindices[i].append(j)
            j2initlist.append(initindices)

        self.jtags2 = []
        for initdict in j2initlist:
            jtagdict = {}
            for IS, lst in initdict.items():
                # jarr = np.zeros((len(lst), len(self.complexStates) + len(self.mixedstates)), dtype=int)
                FSList = []
                for idx, FS in enumerate(lst):
                    # if IS == FS:
                    #     # jarr[idx][IS+len(self.complexStates)]= 1
                    #     continue
                    # jarr[idx][IS + len(self.complexStates)] += 1
                    # jarr[idx][FS + len(self.complexStates)] -= 1
                    FSList.append(FS + len(self.complexStates))
                jtagdict[IS + len(self.complexStates)] = [s for s in FSList]
            self.jtags2.append(jtagdict)
        print("built jtags2: {}".format(time.time() - start))

        start = time.time()
        for star in self.stars[self.mixedstartindex:]:
            indlist = []
            for state in star:
                for j, st in enumerate(self.mixedstates):
                    if st == state:
                        indlist.append(j)
            self.starindexed.append(indlist)
        print("built mixed indexed star: {}".format(time.time() - start))
        # self.starindexed = starindexed

        self.star2symlist = np.zeros(len(self.stars), dtype=int)
        # The i_th element of this index list gives the corresponding symorlist from which the dumbbell of the
        # representative state of the i_th star comes from.
        start = time.time()
        for starind, star in enumerate(self.stars[:self.mixedstartindex]):
            # get the dumbbell of the representative state of the star
            db = star[0].db - star[0].db.R
            # now get the symorlist index in which the dumbbell belongs
            symind = self.pdbcontainer.invmap[db.iorind]
            self.star2symlist[starind] = symind

        for starind, star in enumerate(self.stars[self.mixedstartindex:]):
            # get the dumbbell from the representative state of the star
            db = star[0].db - star[0].db.R
            # now get the symorlist index in which the dumbbell belongs
            symind = self.mdbcontainer.invmap[db.iorind]
            self.star2symlist[starind + self.mixedstartindex] = symind
        print("building star2symlist : {}".format(time.time() - start))

        self.mixedindexdict = {}
        for si, star, starind in zip(itertools.count(), self.stars[self.mixedstartindex:],
                                     self.starindexed[self.mixedstartindex:]):
            for state, ind in zip(star, starind):
                self.mixedindexdict[state] = (ind, si + self.mixedstartindex)

        # create the starset for the bare dumbbell space
        self.barePeriodicStars = [[dumbbell(idx, np.zeros(self.crys.dim, dtype=int)) for idx in idxlist] for idxlist in
                                  self.pdbcontainer.symIndlist]

        self.bareStarindexed = self.pdbcontainer.symIndlist.copy()

        self.bareindexdict = {}
        for si, star, starind in zip(itertools.count(), self.barePeriodicStars, self.bareStarindexed):
            for state, ind in zip(star, starind):
                self.bareindexdict[state] = (ind, si)
        print("building bare, mixed index dicts : {}".format(time.time() - start))

    def sortstars(self):
        """sorts the solute-dumbbell complex crystal stars in order of increasing solute-dumbbell separation distance.
        Note that this is called before mixed dumbbell stars are added in. The mixed dumbbells being in a periodic state
        space, all the mixed dumbbell states are at the origin anyway.
        """
        inddict = {}
        for i, star in enumerate(self.stars):
            inddict[i] = self._sortkey(star[0])
        # Now sort the stars according to dx^2, i.e, sort the dictionary by value
        sortlist = sorted(inddict.items(), key=lambda x: x[1])
        # print(sortlist)
        starnew = []
        starIndexnew = []
        for (ind, dx2) in sortlist:
            starnew.append(self.stars[ind])
            starIndexnew.append(self.starindexed[ind])

        self.stars = starnew
        self.starindexed = starIndexnew

    def jumpnetwork_omega1(self):
        """
        Builds the omega-1 jump network from the omega_0 jump network between the complex states that have been
        considered in the starset.
        Only jumps between states that have been considered within the starset are allowed to be present.
        Also, omega-1 jumps do not move the solute.

        Returns:
            - 3-tuple containing:
                - jumpnetwork - (list of lists of "jump" objects) the symmetry grouped jumps between solute-pure dumbbell
                complex states within the starset.

                - jumpindexed - (list of lists of tuples) indexed version of the jumpnetwork, containing tuples of the form
                (i, j), dx - where i and j denote the indices of the initial and final states and dx
                is the site-to-site distance for the jump.

                - jtags - dictionary of lists containing initial(key) and all final states(values) of dumbbell jumps. This is used in
                non-local relaxation vector calculations.

            - jumptype - list mapping back the omega_1 jumps to omega_0 jumps.
        """
        jumpnetwork = []
        jumpindexed = []
        initstates = []  # list of dicitionaries that store numpy arrays of the form +1 for initial state, -1 for final state
        jumptype = []
        starpair = []
        jumpset = set([])  # set where newly produced jumps will be stored
        print("building omega1")
        start = time.time()
        for jt, jlist in enumerate(self.jnet0):
            for jnum, j0 in enumerate(jlist):
                # these contain dumbell->dumbell jumps
                for pairind, pair in enumerate(self.complexStates):
                    try:
                        pairnew = pair.addjump(j0)
                    except ArithmeticError:
                        # If anything other than ArithmeticError occurs, we'll get the message.
                        continue
                    if pairnew not in self.stateset:
                        continue
                    # convert them to pair jumps
                    jpair = jump(pair, pairnew, j0.c1, j0.c2)
                    if not jpair in jumpset:  # see if the jump has not already been considered
                        newlist = []
                        indices = []
                        initdict = defaultdict(list)
                        for gdumb in self.pdbcontainer.G:
                            # The solute must be at the origin unit cell - shift it
                            state1new, flip1 = jpair.state1.gop(self.pdbcontainer, gdumb)
                            state2new, flip2 = jpair.state2.gop(self.pdbcontainer, gdumb)
                            if not (state1new.i_s == state2new.i_s and np.allclose(state1new.R_s, state2new.R_s)):
                                raise ValueError("Same gop takes solute to different locations")
                            state1new -= state1new.R_s
                            state2new -= state2new.R_s
                            if (not state1new in self.stateset) or (not state2new in self.stateset):
                                raise ValueError("symmetrically obtained complex state not found in stateset(?)")
                            jnew = jump(state1new, state2new, jpair.c1 * flip1, jpair.c2 * flip2)

                            if not jnew in jumpset:
                                # if not (np.allclose(jnew.state1.R_s, 0., atol=self.crys.threshold) and np.allclose(
                                #         jnew.state2.R_s, 0., atol=self.crys.threshold)):
                                #     raise RuntimeError("Solute shifted from origin")
                                # if not (jnew.state1.i_s == jnew.state1.i_s):
                                #     raise RuntimeError(
                                #         "Solute must remain in exactly the same position before and after the jump")
                                newlist.append(jnew)
                                newlist.append(-jnew)
                                # we can add the negative since solute always remains at the origin
                                jumpset.add(jnew)
                                jumpset.add(-jnew)

                        # remove redundant rotations.
                        if np.allclose(DB_disp(self.pdbcontainer, newlist[0].state1, newlist[0].state2), np.zeros(self.crys.dim),
                                       atol=self.pdbcontainer.crys.threshold):
                            for jind in range(len(newlist)-1, -1, -1):
                                # start from the last, so we don't skip elements while removing.
                                j = newlist[jind]
                                j_equiv = jump(j.state1, j.state2, -j.c1, -j.c2)
                                if j_equiv in jumpset:
                                    newlist.remove(j)
                                    # keep the equivalent, discard the original.
                                    jumpset.remove(j)
                                    # Also discard the original from the jumpset, or the equivalent will be
                                    # removed later.
                        if len(newlist) == 0:
                            continue
                        for jmp in newlist:
                            if not (jmp.state1 in self.stateset):
                                raise ValueError("state not found in stateset?\n{}".format(jmp.state1))
                            if not (jmp.state2 in self.stateset):
                                raise ValueError("state not found in stateset?\n{}".format(jmp.state2))
                            initial = self.complexIndexdict[jmp.state1][0]
                            final = self.complexIndexdict[jmp.state2][0]
                            indices.append(((initial, final), DB_disp(self.pdbcontainer, jmp.state1, jmp.state2)))
                            initdict[initial].append(final)
                        jumpnetwork.append(newlist)
                        jumpindexed.append(indices)
                        initstates.append(initdict)
                        # initdict contains all the initial states as keys, and the values as the lists final states
                        # from the initial states for the given jump type.
                        jumptype.append(jt)
        print("built omega1 : time - {}".format(time.time()-start))
        jtags = []
        for initdict in initstates:
            arrdict = {}
            for IS, lst in initdict.items():
                # jtagarr = np.zeros((len(lst), len(self.complexStates) + len(self.mixedstates)), dtype=int)
                FSList = []
                for jnum, FS in enumerate(lst):
                    # jtagarr[jnum][IS] += 1
                    # jtagarr[jnum][FS] -= 1
                    FSList.append(FS)
                arrdict[IS] = [s for s in FSList] #jtagarr.copy()
            jtags.append(arrdict)

        return (jumpnetwork, jumpindexed, jtags), jumptype

    def jumpnetwork_omega34(self, cutoff, solv_solv_cut, solt_solv_cut, closestdistance):
        """
         Builds the omega_4 and omega_3 jump networks between the complex states jumping to mixed dumbbells and vice versa.
         Parameters:
             - cutoff - the cutoff distance for the jump.
             - solv_solv_cut - collision threshold distance between two solvent atoms.
             - solt_solv_cut - collision threshold distance between the solute and a solvent atom.

         Returns:
             3 tuples (T1, T2, T3).
             T1 contains both omega4 and omega3 jumps together. T2 has omega4 and T3 omega_3 jumps
             respectively.
             Each elemnet of T1, T2 and T3 correspond to:
                 - jumpnetwork - (list of lists of "jump" objects) - The even jumps are omega_4 jumps in T1, and the odd ones omega_3

                 - jumpindexed - indexed version of the jumpnetworks, containing tuples of the form (i, j), dx
                 Note - The indices for solute-pure dumbbell complexes are indexed found in complexIndexdict
                 and that of the mixed dumbbells in the mixedindexdict dictionaries.

                 - jtags - dictionary of lists containing initial (key) and all final states (values) of dumbbell jumps. This is used in
                 non-local relaxation vector calculations.
         """
        # building omega_4 -> association - c2=-1 -> since solvent movement is tracked
        # cutoff required - solute-solvent as well as solvent solvent
        alljumpset_omega4 = set([])
        symjumplist_omega4 = []
        symjumplist_omega4_indexed = []
        omega4inits = []

        # alljumpset_omega3=set([])

        symjumplist_omega3 = []
        symjumplist_omega3_indexed = []
        omega3inits = []

        symjumplist_omega43_all = []
        symjumplist_omega43_all_indexed = []
        alljumpset_omega43_all = set([])
        start = time.time()
        print("building omega43")
        for p_pure in self.complexStates:
            if p_pure.is_zero(self.pdbcontainer):  # Spectator rotating into mixed dumbbell does not make sense.
                continue
            for p_mixed in self.mixedstates:
                if not p_mixed.is_zero(self.mdbcontainer):  # Specator rotating into mixed does not make sense.
                    raise ValueError("Mixed dumbbell must be origin state")
                if not (np.allclose(p_pure.R_s, 0, atol=self.crys.threshold)
                        and np.allclose(p_mixed.R_s, 0, atol=self.crys.threshold)):
                    raise RuntimeError("Solute shifted from origin - cannot happen")
                if not (p_pure.i_s == p_mixed.i_s):
                    # The solute must remain in exactly the same position before and after the jump
                    continue
                for c1 in [-1, 1]:
                    j = jump(p_pure, p_mixed, c1, -1)
                    dx = DB_disp4(self.pdbcontainer, self.mdbcontainer, j.state1, j.state2)
                    if np.dot(dx, dx) > cutoff ** 2:
                        continue
                    if not j in alljumpset_omega4:
                        # check if jump already considered
                        # if a jump is in alljumpset_omega4, it's negative will have to be in alljumpset_omega3
                        if not collision_self(self.pdbcontainer, self.mdbcontainer, j, solv_solv_cut,
                                              solt_solv_cut) and not collision_others(self.pdbcontainer,
                                                                                      self.mdbcontainer, j,
                                                                                      closestdistance):
                            newlist = []
                            newneglist = []
                            newalllist = []
                            for g in self.crys.G:
                                for pgdumb, gval in self.pdbcontainer.G_crys.items():
                                    if gval == g:
                                        gdumb_pure = pgdumb

                                for mgdumb, gval in self.mdbcontainer.G_crys.items():
                                    if gval == g:
                                        gdumb_mixed = mgdumb

                                state1new, flip1 = j.state1.gop(self.pdbcontainer, gdumb_pure, complex=True)
                                state2new = j.state2.gop(self.mdbcontainer, gdumb_mixed, complex=False)

                                if not (np.allclose(state1new.R_s, state2new.R_s)):
                                    raise ValueError("Same group op but different resultant solute sites")
                                state1new -= state1new.R_s
                                state2new -= state2new.R_s
                                jnew = jump(state1new, state2new, j.c1*flip1, -1)
                                if not jnew in alljumpset_omega4:
                                    if jnew.state1.i_s == self.pdbcontainer.iorlist[jnew.state1.db.iorind][0]:
                                        if np.allclose(jnew.state1.R_s, jnew.state1.db.R, atol=self.crys.threshold):
                                            raise RuntimeError("Initial state mixed")
                                    if not (jnew.state2.i_s == self.mdbcontainer.iorlist[jnew.state2.db.iorind][0]
                                            and np.allclose(jnew.state2.R_s, jnew.state2.db.R, self.crys.threshold)):
                                        raise RuntimeError("Final state not mixed")
                                    newlist.append(jnew)
                                    newneglist.append(-jnew)
                                    newalllist.append(jnew)
                                    newalllist.append(-jnew)
                                    alljumpset_omega4.add(jnew)

                            new4index = []
                            new3index = []
                            newallindex = []
                            jinitdict3 = defaultdict(list)
                            jinitdict4 = defaultdict(list)

                            for jmp in newlist:
                                pure_ind = self.complexIndexdict[jmp.state1][0]
                                mixed_ind = self.mixedindexdict[jmp.state2][0]
                                # omega4 has pure as initial, omega3 has pure as final
                                jinitdict4[pure_ind].append(mixed_ind)
                                jinitdict3[mixed_ind].append(pure_ind)
                                dx = DB_disp4(self.pdbcontainer, self.mdbcontainer, jmp.state1, jmp.state2)
                                new4index.append(((pure_ind, mixed_ind), dx.copy()))
                                new3index.append(((mixed_ind, pure_ind), -dx))
                                newallindex.append(((pure_ind, mixed_ind), dx.copy()))
                                newallindex.append(((mixed_ind, pure_ind), -dx))

                            symjumplist_omega4.append(newlist)
                            omega4inits.append(jinitdict4)
                            symjumplist_omega4_indexed.append(new4index)

                            symjumplist_omega3.append(newneglist)
                            omega3inits.append(jinitdict3)
                            symjumplist_omega3_indexed.append(new3index)

                            symjumplist_omega43_all.append(newalllist)
                            symjumplist_omega43_all_indexed.append(newallindex)

        # Now build the jtags
        print("built omega43 : time {}".format(time.time()-start))
        jtags4 = []
        jtags3 = []

        for initdict in omega4inits:
            jarrdict = {}
            for IS, lst in initdict.items():
                # jarr = np.zeros((len(lst), len(self.complexStates) + len(self.mixedstates)), dtype=int)
                FSList = []
                for idx, FS in enumerate(lst):
                    # jarr[idx][IS] += 1
                    # jarr[idx][FS + len(self.complexStates)] -= 1
                    FSList.append(FS + len(self.complexStates))
                jarrdict[IS] = [s for s in FSList] #jarr.copy()
            jtags4.append(jarrdict)

        for initdict in omega3inits:
            jarrdict = {}
            for IS, lst in initdict.items():
                FSList = []
                # jarr = np.zeros((len(lst), len(self.complexStates) + len(self.mixedstates)), dtype=int)
                for idx, FS in enumerate(lst):
                    # jarr[idx][IS + len(self.complexStates)] += 1
                    # jarr[idx][FS] -= 1
                    FSList.append(FS)
                jarrdict[IS + len(self.complexStates)] = [s for s in FSList] #jarr.copy()
            jtags3.append(jarrdict)

        return (symjumplist_omega43_all, symjumplist_omega43_all_indexed), (
            symjumplist_omega4, symjumplist_omega4_indexed, jtags4), (
                   symjumplist_omega3, symjumplist_omega3_indexed, jtags3)

# Next, vector stars for dumbbells

class DBVectorStars(object):
    """
    Stores the vector stars corresponding to a given starset of dumbbell states
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

    def generate(self, starset):
        """
        Follows almost the same as that for solute-vacancy case. Only generalized to keep the state
        under consideration unchanged.
        Parameters:
            - starset - the symmetry grouped list of lists of states.
        Generates the full set of basis vectors for each state in the starset
        """
        self.starset = None
        if starset.Nshells == 0: return
        if starset == self.starset: return
        self.starset = starset
        self.crys = self.starset.crys
        self.vecpos = []
        self.vecpos_indexed = []
        self.vecvec = []
        self.Nvstars_spec = 0
        # first do it for the complexes
        for star, indstar in zip(starset.stars[:starset.mixedstartindex],
                                 starset.starindexed[:starset.mixedstartindex]):
            pair0 = star[0]
            glist = []
            # Find group operations that leave state unchanged
            for gdumb in starset.pdbcontainer.G:
                pairnew = pair0.gop(starset.pdbcontainer, gdumb)[0]
                pairnew = pairnew - pairnew.R_s
                if pairnew == pair0:
                    glist.append(starset.pdbcontainer.G_crys[gdumb])  # Although appending gdumb itself also works
            # Find the intersected vector basis for these group operations - same as function "VectorBasis" in crystal module.
            vb = reduce(crystal.CombineVectorBasis, [crystal.VectorBasis(*g.eigen()) for g in glist])
            # Get orthonormal vectors
            vlist = starset.crys.vectlist(vb)
            scale = 1. / np.sqrt(len(star))
            vlist = [v * scale for v in vlist]  # see equation 80 in the paper - (there is a typo, this is correct).
            Nvect = len(vlist)
            if Nvect > 0:
                if pair0.is_zero(self.starset.pdbcontainer):
                    self.Nvstars_spec += Nvect
                for v in vlist:
                    self.vecpos.append(star)
                    self.vecpos_indexed.append(indstar)
                    # implement a copy function like in case of vacancies
                    veclist = []
                    for pairI in star:
                        for gdumb in starset.pdbcontainer.G:
                            pairnew = pair0.gop(starset.pdbcontainer, gdumb)[0]
                            pairnew = pairnew - pairnew.R_s  # translate solute back to origin
                            # This is because the vectors associated with a state are translationally invariant.
                            # Wherever the solute is, if the relative position of the solute and the solvent is the
                            # same, the vector remains unchanged due to that group op.
                            # Remember that only the rotational part of the group op will act on the vector.
                            if pairnew == pairI:
                                veclist.append(starset.crys.g_direc(starset.pdbcontainer.G_crys[gdumb], v))
                                break
                    self.vecvec.append(veclist)

        self.Nvstars_pure = len(self.vecpos)

        # Now do it for the mixed dumbbells - all negative checks disappear
        for star, indstar in zip(starset.stars[starset.mixedstartindex:],
                                 starset.starindexed[starset.mixedstartindex:]):
            pair0 = star[0]
            glist = []
            # Find group operations that leave state unchanged
            for gdumb in starset.mdbcontainer.G:
                pairnew = pair0.gop(starset.mdbcontainer, gdumb, complex=False)
                pairnew = pairnew - pairnew.R_s  # again, only the rotation part matters.
                # what about dumbbell rotations? Does not matter - the state has to remain unchanged
                # Is this valid for origin states too? verify - because we have origin states.
                if pairnew == pair0:
                    glist.append(starset.mdbcontainer.G_crys[gdumb])
            # Find the intersected vector basis for these group operations
            vb = reduce(crystal.CombineVectorBasis, [crystal.VectorBasis(*g.eigen()) for g in glist])
            # Get orthonormal vectors
            vlist = starset.crys.vectlist(vb)  # This also normalizes with respect to length of the vectors.
            scale = 1. / np.sqrt(len(star))
            vlist = [v * scale for v in vlist]
            Nvect = len(vlist)
            if Nvect > 0:
                for v in vlist:
                    self.vecpos.append(star)
                    self.vecpos_indexed.append(indstar)
                    veclist = []
                    for pairI in star:
                        for gdumb in starset.mdbcontainer.G:
                            pairnew = pair0.gop(starset.mdbcontainer, gdumb, complex=False)
                            pairnew = pairnew - pairnew.R_s  # translate solute back to origin
                            if pairnew == pairI:
                                veclist.append(starset.crys.g_direc(starset.mdbcontainer.G_crys[gdumb], v))
                                break
                    self.vecvec.append(veclist)

        self.Nvstars = len(self.vecpos)

        # build the vector star for the bare bare dumbbell state
        self.vecpos_bare = []
        self.vecvec_bare = []
        for star in starset.barePeriodicStars:
            db0 = star[0]
            glist = []
            for gdumb in starset.pdbcontainer.G:
                dbnew = db0.gop(starset.pdbcontainer, gdumb)[0]
                dbnew = dbnew - dbnew.R  # cancel out the translation
                if dbnew == db0:
                    glist.append(starset.pdbcontainer.G_crys[gdumb])
            vb = reduce(crystal.CombineVectorBasis, [crystal.VectorBasis(*g.eigen()) for g in glist])
            # Get orthonormal vectors
            vlist = starset.crys.vectlist(vb)  # This also normalizes with respect to length of the vectors.
            scale = 1. / np.sqrt(len(star))
            vlist = [v * scale for v in vlist]
            Nvect = len(vlist)
            if Nvect > 0:
                for v in vlist:
                    veclist = []
                    self.vecpos_bare.append(star)
                    for st in star:
                        for gdumb in starset.pdbcontainer.G:
                            dbnew, flip = db0.gop(starset.pdbcontainer, gdumb)
                            dbnew = dbnew - dbnew.R
                            if dbnew == st:
                                veclist.append(starset.crys.g_direc(starset.pdbcontainer.G_crys[gdumb], v))
                                break
                    self.vecvec_bare.append(veclist)

        self.stateToVecStar_pure = defaultdict(list)
        for IndofStar, crStar in enumerate(self.vecpos[:self.Nvstars_pure]):
            for IndofState, state in enumerate(crStar):
                self.stateToVecStar_pure[state].append((IndofStar, IndofState))

        self.stateToVecStar_pure.default_factory = None

        self.stateToVecStar_mixed = defaultdict(list)
        for IndOfStar, crStar in enumerate(self.vecpos[self.Nvstars_pure:]):
            for IndOfState, state in enumerate(crStar):
                self.stateToVecStar_mixed[state].append((IndOfStar + self.Nvstars_pure, IndOfState))

        self.stateToVecStar_mixed.default_factory = None

        self.stateToVecStar_bare = defaultdict(list)
        if len(self.vecpos_bare) > 0:
            for IndOfStar, crStar in enumerate(self.vecpos_bare):
                for IndOfState, state in enumerate(crStar):
                    self.stateToVecStar_bare[state].append((IndOfStar, IndOfState))

        self.stateToVecStar_bare.default_factory = None

        # We must produce two expansions. One for pure dumbbell states pointing to pure dumbbell state
        # and the other from mixed dumbbell states to mixed states.
        self.vwycktowyck_bare = np.zeros(len(self.vecpos_bare), dtype=int)
        for vstarind, vstar in enumerate(self.vecpos_bare):
            # get the index of the wyckoff set (symorlist) in which the representative state belongs
            wyckindex = self.starset.bareindexdict[vstar[0]][1]
            self.vwycktowyck_bare[vstarind] = wyckindex

        # Need an indexing from the vector stars to the crystal stars
        self.vstar2star = np.zeros(self.Nvstars, dtype=int)
        for vstindex, vst in enumerate(self.vecpos[:self.Nvstars_pure]):
            # get the crystal star of the representative state of the vector star
            starindex = self.starset.complexIndexdict[vst[0]][1]
            self.vstar2star[vstindex] = starindex

        for vstindex, vst in enumerate(self.vecpos[self.Nvstars_pure:]):
            # get the crystal star of the representative state of the vector star
            starindex = self.starset.mixedindexdict[vst[0]][1]
            # The starindex is already with respect to the total number of (pure+mixed) crystal stars - see stars.py.
            self.vstar2star[vstindex + self.Nvstars_pure] = starindex

    def genGFstarset(self):
        """
        Makes symmetrically grouped connections between the states in the starset, to be used as GFstarset for the pure
        and mixed state spaces.
        The connections must lie within the starset and must connect only those states that are connected by omega_0 or
        omega_2 jumps.
        The GFstarset is to be returned in the form of (i,j),dx. where the indices i and j correspond to the states in
        the iorlist
        """
        complexStates = self.starset.complexStates
        mixedstates = self.starset.mixedstates
        # Connect the states - major bottleneck
        connectset= set([])
        self.connect_ComplexPair = {}
        start = time.time()
        for i, st1 in enumerate(complexStates):
            for j, st2 in enumerate(complexStates[:i+1]):
                try:
                    s = st1 ^ st2
                except:
                    continue
                connectset.add(s)
                connectset.add(-s)
                # if i==j and not s==-s:
                #     raise ValueError("Same state connection producing different connector")
                self.connect_ComplexPair[(st1, st2)] = s
                self.connect_ComplexPair[(st2, st1)] = -s
        print("\tComplex connections creation time: {}".format(time.time() - start))

        # Now group the connections
        GFstarset_pure=[]
        GFPureStarInd = {}
        start = time.time()
        for s in connectset:
            if s in GFPureStarInd:
                continue
            connectlist = []
            for gdumb in self.starset.pdbcontainer.G:
                snew = s.gop(self.starset.pdbcontainer, gdumb)
                # Bring the dumbbell of the initial state to the origin
                # snew = snew.shift() No need for shifting. Automatically done in gop function.
                if snew in GFPureStarInd:
                    continue

                if snew not in connectset:
                    raise TypeError("connector list is not closed under symmetry operations for the complex starset.{}"
                                    .format(snew))

                dx = DB_disp(self.starset.pdbcontainer, snew.state1, snew.state2)
                ind1 = self.starset.pdbcontainer.db2ind(snew.state1)
                # db2ind does not care about which unit cell the dumbbell is at
                ind2 = self.starset.pdbcontainer.db2ind(snew.state2)
                tup = ((ind1, ind2), dx.copy())
                connectlist.append(tup)
                GFPureStarInd[snew] = len(GFstarset_pure)
            GFstarset_pure.append(connectlist)
        print("\tComplex connections symmetry grouping time: {}".format(time.time() - start))
        print("No. of pure dumbbell connections: {}".format(len(connectset)))

        return GFstarset_pure, GFPureStarInd

    def GFexpansion(self):
        """
        carries out the expansion of the Green's function in the basis of the vector stars.
        """
        print("building GF starsets")
        start = time.time()
        GFstarset_pure, GFPureStarInd = self.genGFstarset()
        print("GF star sets built: {}".format(time.time() - start))

        Nvstars_pure = self.Nvstars_pure
        Nvstars_mixed = self.Nvstars - self.Nvstars_pure

        GFexpansion_pure = np.zeros((Nvstars_pure, Nvstars_pure, len(GFstarset_pure)))
        start = time.time()
        for ((st1, st2), s) in self.connect_ComplexPair.items():
            # get the vector stars in which the initial state belongs
            try:
                i = self.stateToVecStar_pure[st1]
                # get the vector stars in which the final state belongs
                j = self.stateToVecStar_pure[st2]
            except KeyError:
                continue
            k = GFPureStarInd[s]
            for (indOfStar_i, indOfState_i) in i:
                for (indOfStar_j, indOfState_j) in j:
                    GFexpansion_pure[indOfStar_i, indOfStar_j, k] += \
                        np.dot(self.vecvec[indOfStar_i][indOfState_i], self.vecvec[indOfStar_j][indOfState_j])


        print("Built Complex GF expansions: {}".format(time.time() - start))

        # symmetrize
        for i in range(Nvstars_pure):
            for j in range(0, i):
                GFexpansion_pure[i, j, :] = GFexpansion_pure[j, i, :]

        return (GFstarset_pure, GFPureStarInd, zeroclean(GFexpansion_pure))

    # See group meeting update slides of sept 10th to see how this works.
    def biasexpansion(self, jumpnetwork_omega1, jumpnetwork_omega2, jumptype, jumpnetwork_omega34):
        """
        Returns an expansion of the bias vector in terms of the displacements produced by jumps.
        Parameters:
            jumpnetwork_omega* - the jumpnetwork for the "*" kind of jumps (1,2,3 or 4)
            jumptype - the omega_0 jump type that gives rise to a omega_1 jump type (see jumpnetwork_omega1 function
            in stars.py module)
        Returns:
            bias0, bias1, bias2, bias4 and bias3 expansions, one each for solute and solvent
            Note - bias0 for solute makes no sense, so we return only for solvent.
        """
        z = np.zeros(self.crys.dim, dtype=float)

        biasBareExpansion = np.zeros((len(self.vecpos_bare), len(self.starset.jnet0)))
        # Expansion of pure dumbbell initial state bias vectors and complex state bias vectors
        bias0expansion = np.zeros((self.Nvstars_pure, len(self.starset.jumpindices)))
        bias1expansion_solvent = np.zeros((self.Nvstars_pure, len(jumpnetwork_omega1)))
        # bias1expansion_solute = np.zeros((self.Nvstars_pure, len(jumpnetwork_omega1)))

        bias4expansion_solvent = np.zeros((self.Nvstars_pure, len(jumpnetwork_omega34)))
        # bias4expansion_solute = np.zeros((self.Nvstars_pure, len(jumpnetwork_omega34)))

        # Expansion of mixed dumbbell initial state bias vectors.
        bias2expansion_solvent = np.zeros((self.Nvstars - self.Nvstars_pure, len(jumpnetwork_omega2)))
        bias2expansion_solute = np.zeros((self.Nvstars - self.Nvstars_pure, len(jumpnetwork_omega2)))

        bias3expansion_solvent = np.zeros((self.Nvstars - self.Nvstars_pure, len(jumpnetwork_omega34)))
        # bias3expansion_solute = np.zeros((self.Nvstars - self.Nvstars_pure, len(jumpnetwork_omega34)))

        # First, let's build the periodic bias expansions
        for i, star, vectors in zip(itertools.count(), self.vecpos_bare, self.vecvec_bare):
            for k, jumplist in zip(itertools.count(), self.starset.jnet0):
                for j in jumplist:
                    IS = j.state1
                    if star[0] == IS:
                        dx = DB_disp(self.starset.pdbcontainer, j.state1, j.state2)
                        geom_bias_solvent = np.dot(vectors[0], dx) * len(star)
                        biasBareExpansion[i, k] += geom_bias_solvent

        for i, purestar, vectors in zip(itertools.count(), self.vecpos[:self.Nvstars_pure],
                                        self.vecvec[:self.Nvstars_pure]):
            # iterates over the rows of the matrix
            # First construct bias1expansion and bias0expansion
            # This contains the expansion of omega_0 jumps and omega_1 type jumps
            # See slides of Sept. 10 for diagram.
            # omega_0 : pure -> pure
            # omega_1 : complex -> complex
            for k, jumplist, jt in zip(itertools.count(), jumpnetwork_omega1, jumptype):
                # iterates over the columns of the matrix
                for j in jumplist:
                    IS = j.state1
                    # for i, states, vectors in zip(itertools.count(),self.vecpos,self.vecvec):
                    if purestar[0] == IS:
                        # sees if there is a jump of the kth type with purestar[0] as the initial state.
                        dx = DB_disp(self.starset.pdbcontainer, j.state1, j.state2)
                        # dx_solute = z
                        dx_solvent = dx.copy()  # just for clarity that the solvent mass transport is dx itself.

                        geom_bias_solvent = np.dot(vectors[0], dx_solvent) * len(
                            purestar)  # should this be square root? check with tests.
                        # geom_bias_solute = np.dot(vectors[0], dx_solute) * len(purestar)

                        bias1expansion_solvent[
                            i, k] += geom_bias_solvent  # this is contribution of kth_type of omega_1 jumps, to the bias
                        # bias1expansion_solute[i, k] += geom_bias_solute
                        # vector along v_i
                        # so to find the total bias along v_i due to omega_1 jumps, we sum over k
                        bias0expansion[i, jt] += geom_bias_solvent  # These are the contributions of the omega_0 jumps
                        # to the bias vector along v_i, for bare dumbbells
                        # so to find the total bias along v_i, we sum over k.

            # Next, omega_4: complex -> mixed
            for k, jumplist in zip(itertools.count(), jumpnetwork_omega34):
                for j in jumplist[::2]:  # Start from the first element, skip every other
                    IS = j.state1
                    if not j.state2.is_zero(self.starset.mdbcontainer):
                        # check if initial state is mixed dumbbell -> then skip - it's omega_3
                        raise TypeError ("final state not origin in mixed dbcontainer for omega4")
                    # for i, states, vectors in zip(itertools.count(),self.vecpos,self.vecvec):
                    if purestar[0] == IS:
                        dx = DB_disp4(self.starset.pdbcontainer, self.starset.mdbcontainer, j.state1, j.state2)
                        # dx_solute = z  # self.starset.mdbcontainer.iorlist[j.state2.db.iorind][1] / 2.
                        dx_solvent = dx  # - self.starset.mdbcontainer.iorlist[j.state2.db.iorind][1] / 2.
                        # geom_bias_solute = np.dot(vectors[0], dx_solute) * len(purestar)
                        geom_bias_solvent = np.dot(vectors[0], dx_solvent) * len(purestar)
                        # bias4expansion_solute[i, k] += geom_bias_solute
                        # this is contribution of omega_4 jumps, to the bias
                        bias4expansion_solvent[i, k] += geom_bias_solvent
                        # vector along v_i
                        # So, to find the total bias along v_i due to omega_4 jumps, we sum over k.

        # Now, construct the bias2expansion and bias3expansion
        for i, mixedstar, vectors in zip(itertools.count(), self.vecpos[self.Nvstars_pure:],
                                         self.vecvec[self.Nvstars_pure:]):
            # First construct bias2expansion
            # omega_2 : mixed -> mixed
            for k, jumplist in zip(itertools.count(), jumpnetwork_omega2):
                for j in jumplist:
                    IS = j.state1
                    # for i, states, vectors in zip(itertools.count(),self.vecpos,self.vecvec):
                    if mixedstar[0] == IS:
                        dx = DB_disp(self.starset.mdbcontainer, j.state1, j.state2)
                        dx_solute = dx  # + self.starset.mdbcontainer.iorlist[j.state2.db.iorind][1] / 2. - \
                                    # self.starset.mdbcontainer.iorlist[j.state1.db.iorind][1] / 2.
                        dx_solvent = dx  #- self.starset.mdbcontainer.iorlist[j.state2.db.iorind][1] / 2. + \
                                     #self.starset.mdbcontainer.iorlist[j.state1.db.iorind][1] / 2.
                        geom_bias_solute = np.dot(vectors[0], dx_solute) * len(mixedstar)
                        geom_bias_solvent = np.dot(vectors[0], dx_solvent) * len(mixedstar)
                        bias2expansion_solute[i, k] += geom_bias_solute
                        bias2expansion_solvent[i, k] += geom_bias_solvent

            # Next, omega_3: mixed -> complex
            for k, jumplist in zip(itertools.count(), jumpnetwork_omega34):
                for j in jumplist[1::2]:  # start from the second element, skip every other
                    if not j.state1.is_zero(self.starset.mdbcontainer):
                        # check if initial state is not a mixed state -> skip if not mixed
                        print(self.starset.mdbcontainer.iorlist)
                        print(j.state1)
                        raise TypeError("initial state not origin in mdbcontainer")
                    # for i, states, vectors in zip(itertools.count(),self.vecpos,self.vecvec):
                    if mixedstar[0] == j.state1:
                        try:
                            dx = -DB_disp4(self.starset.pdbcontainer, self.starset.mdbcontainer, j.state2, j.state1)
                        except IndexError:
                            print(len(self.starset.pdbcontainer.iorlist), len(self.starset.mdbcontainer.iorlist))
                            print(j.state2.db.iorind, j.state1.db.iorind)
                            raise IndexError("list index out of range")
                        # dx_solute = z  # -self.starset.mdbcontainer.iorlist[j.state1.db.iorind][1] / 2.
                        dx_solvent = dx  # + self.starset.mdbcontainer.iorlist[j.state1.db.iorind][1] / 2.
                        # geom_bias_solute = np.dot(vectors[0], dx_solute) * len(mixedstar)
                        geom_bias_solvent = np.dot(vectors[0], dx_solvent) * len(mixedstar)
                        # bias3expansion_solute[i, k] += geom_bias_solute
                        bias3expansion_solvent[i, k] += geom_bias_solvent

        if len(self.vecpos_bare) != 0:
            biasBareExpansion = zeroclean(biasBareExpansion)
        # if len(self.vecpos_bare) == 0:
        #     return zeroclean(bias0expansion), (zeroclean(bias1expansion_solute), zeroclean(bias1expansion_solvent)), \
        #            (zeroclean(bias2expansion_solute), zeroclean(bias2expansion_solvent)), \
        #            (None, zeroclean(bias3expansion_solvent)), \
        #            (None, zeroclean(bias4expansion_solvent)), biasBareExpansion
        # else:
        return zeroclean(bias0expansion), (None, zeroclean(bias1expansion_solvent)), \
               (zeroclean(bias2expansion_solute), zeroclean(bias2expansion_solvent)), \
               (None, zeroclean(bias3expansion_solvent)), \
               (None, zeroclean(bias4expansion_solvent)), biasBareExpansion

    def rateexpansion(self, jumpnetwork_omega1, jumptype, jumpnetwork_omega34):
        """
        Implements expansion of the jump rates in terms of the basis function of the vector stars.
        Parameters:
            jumpnetwork_omega1 - the omega_1 jump network
            jumptype - the omega_0 jump type that gives rise to a omega_1 jump type (see jumpnetwork_omega1 function
            in stars.py module)
            jumpnetwork_omega34 - the omega_4 and omega_3 jump networks.
        Returns:
            rateexpansion and escape rate expansions for omega0, omega1, omega2, omega3 and omega4 jumps
        """
        # See my slides of Sept. 10 for diagram
        rate0expansion = np.zeros((self.Nvstars_pure, self.Nvstars_pure, len(self.starset.jnet0)))
        rate1expansion = np.zeros((self.Nvstars_pure, self.Nvstars_pure, len(jumpnetwork_omega1)))
        rate0escape = np.zeros((self.Nvstars_pure, len(self.starset.jumpindices)))
        rate1escape = np.zeros((self.Nvstars_pure, len(jumpnetwork_omega1)))

        # First, we do the rate1 and rate0 expansions
        for k, jumplist, jt in zip(itertools.count(), jumpnetwork_omega1, jumptype):
            for jmp in jumplist:
                # Get the vector star indices for the initial and final states of the jumps
                try:
                    indlist1 = self.stateToVecStar_pure[jmp.state1]
                except KeyError:
                    continue
                try:
                    indlist2 = self.stateToVecStar_pure[jmp.state2]
                except KeyError:
                    indlist2 = []

                for tup1 in indlist1:
                    # print(tup1)
                    rate0escape[tup1[0], jt] -= np.dot(self.vecvec[tup1[0]][tup1[1]], self.vecvec[tup1[0]][tup1[1]])
                    rate1escape[tup1[0], k] -= np.dot(self.vecvec[tup1[0]][tup1[1]], self.vecvec[tup1[0]][tup1[1]])
                    for tup2 in indlist2:
                        rate0expansion[tup1[0], tup2[0], jt] += np.dot(self.vecvec[tup1[0]][tup1[1]],
                                                                       self.vecvec[tup2[0]][tup2[1]])

                        rate1expansion[tup1[0], tup2[0], k] += np.dot(self.vecvec[tup1[0]][tup1[1]],
                                                                      self.vecvec[tup2[0]][tup2[1]])

        # Next, we expand the omega3 an omega4 rates
        rate4expansion = np.zeros((self.Nvstars_pure, self.Nvstars - self.Nvstars_pure, len(jumpnetwork_omega34)))
        rate3expansion = np.zeros((self.Nvstars - self.Nvstars_pure, self.Nvstars_pure, len(jumpnetwork_omega34)))
        rate3escape = np.zeros((self.Nvstars - self.Nvstars_pure, len(jumpnetwork_omega34)))
        rate4escape = np.zeros((self.Nvstars_pure, len(jumpnetwork_omega34)))

        for k, jumplist in enumerate(jumpnetwork_omega34):
            for jmp in jumplist[::2]:  # iterate only through the omega4 jumps, the negatives are omega3
                try:
                    indlist1 = self.stateToVecStar_pure[jmp.state1]  # The initial state is a complex in omega4
                    indlist2 = self.stateToVecStar_mixed[jmp.state2]  # The final state is a mixed dumbbell in omega4
                except KeyError:
                    raise ValueError("Empty vector stars for omega43 jumps?")
                for tup1 in indlist1:
                    rate4escape[tup1[0], k] -= np.dot(self.vecvec[tup1[0]][tup1[1]], self.vecvec[tup1[0]][tup1[1]])
                for tup2 in indlist2:
                    rate3escape[tup2[0] - self.Nvstars_pure, k] -= np.dot(self.vecvec[tup2[0]][tup2[1]],
                                                                          self.vecvec[tup2[0]][tup2[1]])

                for tup1 in indlist1:
                    for tup2 in indlist2:

                        rate4expansion[tup1[0], tup2[0] - self.Nvstars_pure, k] += np.dot(self.vecvec[tup1[0]][tup1[1]],
                                                                                          self.vecvec[tup2[0]][tup2[1]])

                        rate3expansion[tup2[0] - self.Nvstars_pure, tup1[0], k] += np.dot(self.vecvec[tup1[0]][tup1[1]],
                                                                                          self.vecvec[tup2[0]][tup2[1]])

        # Next, we expand omega2
        rate2expansion = np.zeros((self.Nvstars - self.Nvstars_pure, self.Nvstars - self.Nvstars_pure,
                                   len(self.starset.jnet2)))
        rate2escape = np.zeros((self.Nvstars - self.Nvstars_pure, len(self.starset.jnet2)))

        for k, jumplist in zip(itertools.count(), self.starset.jnet2):
            for jmp in jumplist:
                try:
                    indlist1 = self.stateToVecStar_mixed[jmp.state1]
                    indlist2 = self.stateToVecStar_mixed[jmp.state2 - jmp.state2.R_s]
                except KeyError:
                    raise ValueError("Empty vector stars for omega2 jumps?")
                for tup1 in indlist1:
                    rate2escape[tup1[0] - self.Nvstars_pure, k] -= np.dot(self.vecvec[tup1[0]][tup1[1]],
                                                                          self.vecvec[tup1[0]][tup1[1]])
                    for tup2 in indlist2:
                        rate2expansion[tup1[0] - self.Nvstars_pure, tup2[0] - self.Nvstars_pure, k] +=\
                            np.dot(self.vecvec[tup1[0]][tup1[1]], self.vecvec[tup2[0]][tup2[1]])

        return (zeroclean(rate0expansion), zeroclean(rate0escape)),\
               (zeroclean(rate1expansion), zeroclean(rate1escape)),\
               (zeroclean(rate2expansion), zeroclean(rate2escape)),\
               (zeroclean(rate3expansion), zeroclean(rate3escape)),\
               (zeroclean(rate4expansion), zeroclean(rate4escape))

    def outer(self):
        """
        computes the outer product tensor to perform 'bias *outer* gamma', i.e., the correlated part in the vector
        star basis.
        Returns:
            - outerprod: dimxdimxNvstarsxNvstars outer product tensor.
        """
        # print("Building outer product tensor")
        outerprod = np.zeros((self.crys.dim, self.crys.dim, self.Nvstars, self.Nvstars))

        for st in self.starset.complexStates:
            try:
                vecStarList = self.stateToVecStar_pure[st]
            except KeyError:
                continue
            for (indStar1, indState1) in vecStarList:
                for (indStar2, indState2) in vecStarList:
                    outerprod[:, :, indStar1, indStar2] += np.outer(self.vecvec[indStar1][indState1],
                                                                    self.vecvec[indStar2][indState2])

        # There should be no non-zero outer product tensors between the pure and mixed dumbbells.

        for st in self.starset.mixedstates:
            try:
                indlist = self.stateToVecStar_mixed[st]
            except KeyError:
                raise ValueError("Empty vector star for mixed state?")

            for (IndofStar1, IndofState1) in indlist:
                for (IndofStar2, IndofState2) in indlist:
                    outerprod[:, :, IndofStar1, IndofStar2] += np.outer(self.vecvec[IndofStar1][IndofState1],
                                                                      self.vecvec[IndofStar2][IndofState2])

        return zeroclean(outerprod)