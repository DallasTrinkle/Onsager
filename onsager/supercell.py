"""
Supercell class

Class to store supercells of crystals. A supercell is a lattice model of a crystal, with
periodically repeating unit cells. In that framework we can

1. add/remove/substitute atoms
2. find the transformation map between two different representations of the same supercell
3. output POSCAR format (possibly other formats?)
"""

__author__ = 'Dallas R. Trinkle'

import numpy as np
import collections, copy, itertools, warnings
from numbers import Integral
from onsager import crystal, cluster

# TODO: add "parser"--read CONTCAR file, create Supercell
# TODO: output PairState from Supercell

class Supercell(object):
    """
    A class that defines a Supercell of a crystal.

    Takes in a crystal, a supercell (3x3 integer matrix). We can identify sites
    as interstitial sites, and specify if we'll have solutes.
    """

    def __init__(self, crys, superlatt, interstitial=(), Nsolute=0, empty=False, NOSYM=False):
        """
        Initialize our supercell to an empty supercell.

        :param crys: crystal object
        :param superlatt: 3x3 integer matrix
        :param interstitial: (optional) list/tuple of indices that correspond to interstitial sites
        :param Nsolute: (optional) number of substitutional solute elements to consider; default=0
        :param empty: (optional) designed to allow "copy" to work--skips all derived info
        :param NOSYM: (optional) does not do symmetry analysis (intended ONLY for testing purposes)
        """
        self.crys = crys
        self.superlatt = superlatt.copy()
        self.interstitial = copy.deepcopy(interstitial)
        self.Nchem = crys.Nchem + Nsolute if Nsolute > 0 else crys.Nchem
        if empty: return
        # everything else that follows is "derived" from those initial parameters
        self.lattice = np.dot(self.crys.lattice, self.superlatt)
        self.N = self.crys.N
        self.atomindices, self.indexatom = self.crys.atomindices, \
                                           {ci: n for n, ci in enumerate(self.crys.atomindices)}
        self.chemistry = [crys.chemistry[n] if n < crys.Nchem else '' for n in range(self.Nchem + 1)]
        self.chemistry[-1] = 'v'
        self.Wyckofflist, self.Wyckoffchem = [], []
        for n, (c, i) in enumerate(self.atomindices):
            for wset in self.Wyckofflist:
                if n in wset: break
            if len(self.Wyckofflist) == 0 or n not in wset:
                # grab the set of (c,i) of Wyckoff sets (next returns first that matches, None if none):
                indexset = next((iset for iset in self.crys.Wyckoff if (c, i) in iset), None)
                self.Wyckofflist.append(frozenset([self.indexatom[ci] for ci in indexset]))
                self.Wyckoffchem.append(self.crys.chemistry[c])
        self.size, self.invsuper, self.translist, self.transdict = self.maketrans(self.superlatt)
        # self.transdict = {tuple(t):n for n,t in enumerate(self.translist)}
        self.pos, self.occ = self.makesites(), -1 * np.ones(self.N * self.size, dtype=int)
        self.chemorder = [[] for n in range(self.Nchem)]
        if NOSYM:
            self.G = frozenset([crystal.GroupOp.ident([self.pos])])
        else:
            self.G = self.gengroup()

    # some attributes we want to do equate, others we want deepcopy. Equate should not be modified.
    __copyattr__ = ('lattice', 'N', 'chemistry', 'size', 'invsuper',
                    'Wyckofflist', 'Wyckoffchem', 'occ', 'chemorder')
    __eqattr__ = ('atomindices', 'indexatom', 'translist', 'transdict', 'pos', 'G')

    def copy(self):
        """
        Make a copy of the supercell; initializes, then copies over ``__copyattr__`` and
        ``__eqattr__``.

        :return: new supercell object, copy of the original
        """
        supercopy = self.__class__(self.crys, self.superlatt, self.interstitial, self.Nchem - self.crys.Nchem,
                                   empty=True)
        for attr in self.__copyattr__: setattr(supercopy, attr, copy.deepcopy(getattr(self, attr)))
        for attr in self.__eqattr__: setattr(supercopy, attr, getattr(self, attr))
        return supercopy

    def __eq__(self, other):
        """
        Return True if two supercells are equal; this means they should have the same occupancy.
        *and* the same ordering

        :param other: supercell for comparison
        :return: True if same crystal, supercell, occupancy, and ordering; False otherwise
        """
        return isinstance(other, self.__class__) and np.all(self.superlatt == other.superlatt) and \
               self.interstitial == other.interstitial and np.allclose(self.pos, other.pos) and \
               np.all(self.occ == other.occ) and self.chemorder == other.chemorder

    def __ne__(self, other):
        """Inequality == not __eq__"""
        return not self.__eq__(other)

    def stoichiometry(self):
        """Return a string representing the current stoichiometry"""
        return ','.join([c + '_i({})'.format(len(l))
                         if n in self.interstitial
                         else c + '({})'.format(len(l))
                         for n, c, l in zip(itertools.count(),
                                            self.chemistry,
                                            self.chemorder)])

    def __str__(self):
        """Human readable version of supercell"""
        str = "Supercell of crystal:\n{crys}\n".format(crys=self.crys)
        str += "Supercell vectors:\n{}\nChemistry: ".format(self.superlatt.T)
        str += self.stoichiometry()
        str += '\nKroger-Vink: ' + self.KrogerVink()
        str += '\nPositions:\n'
        str += '\n'.join([u.__str__() + ' ' + self.chemistry[o] for u, o in zip(self.pos, self.occ)])
        str += '\nOrdering:\n'
        str += '\n'.join([u.__str__() + ' ' + c for c, ulist in zip(self.chemistry, self.occposlist())
                          for u in ulist])
        return str

    def __mul__(self, other):
        """
        Multiply by a GroupOp; returns a new supercell (constructed via copy).

        :param other: must be a GroupOp (and *should* be a GroupOp of the supercell!)
        :return: rotated supercell
        """
        if not isinstance(other, crystal.GroupOp): return NotImplemented
        gsuper = self.copy()
        gsuper *= other
        return gsuper

    def __rmul__(self, other):
        """
        Multiply by a GroupOp; returns a new supercell (constructed via copy).

        :param other: must be a GroupOp (and *should* be a GroupOp of the supercell!)
        :return: rotated supercell
        """
        if not isinstance(other, crystal.GroupOp): return NotImplemented
        return self.__mul__(other)

    def __imul__(self, other):
        """
        Multiply by a GroupOp, in place.

        :param other: must be a GroupOp (and *should* be a GroupOp of the supercell!)
        :return: self
        """
        if not isinstance(other, crystal.GroupOp): return NotImplemented
        # This requires some careful manipulation: we need to modify (1) occ, and (2) chemorder
        indexmap = other.indexmap[0]
        gocc = self.occ.copy()
        for ind, gind in enumerate(indexmap):
            gocc[gind] = self.occ[ind]
        self.occ = gocc
        self.chemorder = [[indexmap[ind] for ind in clist] for clist in self.chemorder]
        return self

    def index(self, pos, threshold=1.):
        """
        Return the index that corresponds to the position *closest* to pos in the supercell.
        Done in direct coordinates of the supercell, using periodic boundary conditions.

        :param pos: 3-vector
        :param threshold: (optional) minimum squared "distance" in supercell for a match; default=1.
        :return index: index of closest position
        """
        index, dist2 = None, threshold
        for ind, u in enumerate(self.pos):
            delta = crystal.inhalf(pos - u)
            d2 = np.sum(delta * delta)
            if d2 < dist2: index, dist2 = ind, d2
        return index

    def __getitem__(self, key):
        """
        Index into supercell

        :param key: index (either an int, a slice, or a position)
        :return: chemical occupation at that point
        """
        if isinstance(key, Integral) or isinstance(key, slice): return self.occ[key]
        if isinstance(key, np.ndarray) and key.shape == (3,): return self.occ[self.index(key)]
        raise TypeError('Inappropriate key {}'.format(key))

    def __setitem__(self, key, value):
        """
        Set specific composition for site; uses same indexing as __getitem__

        :param key: index (either an int, a slice, or a position)
        :param value: chemical occupation at that point
        """
        if isinstance(key, slice): return NotImplemented
        index = None
        if isinstance(key, Integral): index = key
        if isinstance(key, np.ndarray) and key.shape == (3,): index = self.index(key)
        self.setocc(index, value)

    def __sane__(self):
        """Return True if supercell occupation and chemorder are consistent"""
        occset = set()
        for c, clist in enumerate(self.chemorder):
            for ind in clist:
                # check that occupancy (from chemorder) is correct:
                if self.occ[ind] != c: return False
                # record as an occupied state
                occset.add(ind)
        # now make sure that every site *not* in occset is, in fact, vacant
        for ind, c in enumerate(self.occ):
            if ind not in occset:
                if c != -1: return False
        return True

    @staticmethod
    def maketrans(superlatt):
        """
        Takes in a supercell matrix, and returns a list of all translations of the unit
        cell that remain inside the supercell

        :param superlatt: 3x3 integer matrix
        :return size: integer, corresponding to number of unit cells
        :return invsuper: integer matrix inverse of supercell (needs to be divided by size)
        :return translist: list of integer vectors (to be divided by ``size``) corresponding
            to unit cell positions
        :return transdict: dictionary of tuples and their corresponding index (inverse of trans)
        """
        size = abs(int(np.round(np.linalg.det(superlatt))))
        if size==0: raise ZeroDivisionError('Tried to use a singular supercell.')
        invsuper = np.round(np.linalg.inv(superlatt) * size).astype(int)
        maxN = abs(superlatt).max()
        translist, transdict = [], {}
        for nvect in [np.array((n0, n1, n2))
                      for n0 in range(-maxN, maxN + 1)
                      for n1 in range(-maxN, maxN + 1)
                      for n2 in range(-maxN, maxN + 1)]:
            tv = np.dot(invsuper, nvect) % size
            ttup = tuple(tv)
            if ttup not in transdict:
                transdict[ttup] = len(translist)
                translist.append(tv)
        if len(translist) != size:
            raise ArithmeticError(
                'Somehow did not generate the correct number of translations? {}!={}'.format(size, len(translist)))
        return size, invsuper, translist, transdict

    def makesites(self):
        """
        Generate the array corresponding to the sites; the indexing is based on the translations
        and the atomindices in crys. These may not all be filled when the supercell is finished.

        :return pos: array [N*size, 3] of supercell positions in direct coordinates
        """
        invsize = 1 / self.size
        basislist = [np.dot(self.invsuper, self.crys.basis[c][i]) for (c, i) in self.atomindices]
        return np.array([crystal.incell((t + u) * invsize) for t in self.translist for u in basislist])

    def gengroup(self):
        """
        Generate the group operations internal to the supercell

        :return G: set of GroupOps
        """
        Glist = []
        unittranslist = [np.dot(self.superlatt, t) // self.size for t in self.translist]
        invsize = 1 / self.size
        for g0 in self.crys.G:
            Rsuper = np.dot(self.invsuper, np.dot(g0.rot, self.superlatt))
            if not np.all(Rsuper % self.size == 0):
                warnings.warn(
                    'Broken symmetry? GroupOp:\n{}\nnot a symmetry operation of supercell?\nRsuper=\n{}'.format(g0,
                                                                                                                Rsuper),
                    RuntimeWarning, stacklevel=2)
                continue
            else:
                # divide out the size (in inverse superlatt). Should still be an integer matrix (and hence, a symmetry)
                Rsuper //= self.size
            for u in unittranslist:
                # first, make the corresponding group operation by adding the unit cell translation:
                g = g0 + u
                # translation vector *in the supercell*; go ahead and keep it inside the supercell, too.
                tsuper = (np.dot(self.invsuper, g.trans) % self.size) * invsize
                # finally: indexmap!!
                indexmap = []
                for R in unittranslist:
                    for ci in self.atomindices:
                        Rp, ci1 = self.crys.g_pos(g, R, ci)
                        # A little confusing, but:
                        # [n]^-1*Rp -> translation, but needs to be mod self.size
                        # convert to a tuple, to the index into transdict
                        # THEN multiply by self.N, and add the index of the new Wyckoff site. Whew!
                        indexmap.append(
                            self.transdict[tuple(np.dot(self.invsuper, Rp) % self.size)] * self.N + self.indexatom[ci1])
                if len(set(indexmap)) != self.N * self.size:
                    raise ArithmeticError('Did not produce a correct index mapping for GroupOp:\n{}'.format(g))
                Glist.append(crystal.GroupOp(rot=Rsuper, cartrot=g0.cartrot, trans=tsuper,
                                             indexmap=(tuple(indexmap),)))
        return frozenset(Glist)

    def definesolute(self, c, chemistry):
        """
        Set the name of the chemistry of chemical index c. Only works for substitutional solutes.

        :param c: index
        :param chemistry: string
        """
        if c < self.crys.Nchem or c >= self.Nchem:
            raise IndexError('Trying to set the chemistry for a lattice atom / vacancy')
        self.chemistry[c] = chemistry

    def setocc(self, ind, c):
        """
        Set the occupancy of position indexed by ind, to chemistry c. Used by all the other algorithms.

        :param ind: integer index
        :param c: chemistry index
        """
        if c < -2 or c > self.crys.Nchem:
            raise IndexError('Trying to occupy with a non-defined chemistry: {} out of range'.format(c))
        corig = self.occ[ind]
        if corig != c:
            if corig >= 0:
                # remove from chemorder list (if not vacancy)
                co = self.chemorder[corig]
                co.pop(co.index(ind))
            if c >= 0:
                # add to chemorder list (if not vacancy)
                self.chemorder[c].append(ind)
            # finally: set the occupancy
            self.occ[ind] = c

    def fillperiodic(self, ci, Wyckoff=True):
        """
        Occupies all of the (Wyckoff) sites corresponding to chemical index with the appropriate chemistry.

        :param ci: tuple of (chem, index) in crystal
        :param Wyckoff: (optional) if False, *only* occupy the specific tuple, but still periodically
        :return self:
        """
        if __debug__:
            if ci not in self.indexatom: raise IndexError('Tuple {} not a corresponding atom index'.format(ci))
        ind = self.indexatom[ci]
        indlist = next((nset for nset in self.Wyckofflist if ind in nset), None) if Wyckoff else (ind,)
        for i in [n * self.N + i for n in range(self.size) for i in indlist]:
            self.setocc(i, ci[0])
        return self

    def occposlist(self):
        """
        Returns a list of lists of occupied positions, in (chem)order.

        :return occposlist: list of lists of supercell coord. positions
        """
        return [[self.pos[ind] for ind in clist] for clist in self.chemorder]

    def POSCAR(self, name=None, stoichiometry=True):
        """
        Return a VASP-style POSCAR, returned as a string.

        :param name: (optional) name to use for first list
        :param stoichiometry: (optional) if True, append stoichiometry to name
        :return POSCAR: string
        """
        POSCAR = "" if name is None else name
        if stoichiometry: POSCAR += " " + self.stoichiometry()
        POSCAR += """
1.0
{a[0][0]:21.16f} {a[1][0]:21.16f} {a[2][0]:21.16f}
{a[0][1]:21.16f} {a[1][1]:21.16f} {a[2][1]:21.16f}
{a[0][2]:21.16f} {a[1][2]:21.16f} {a[2][2]:21.16f}
""".format(a=self.lattice)
        POSCAR += ' '.join(['{}'.format(len(clist)) for clist in self.chemorder])
        POSCAR += '\nDirect\n'
        POSCAR += '\n'.join([" {u[0]:19.16f} {u[1]:19.16f} {u[2]:19.16f}".format(u=u)
                             for clist in self.occposlist() for u in clist])
        # needs a trailing newline
        return POSCAR + '\n'

    __vacancyformat__ = "v_{sitechem}"
    __interstitialformat__ = "{chem}_i"
    __antisiteformat__ = "{chem}_{sitechem}"

    def defectindices(self):
        """
        Return a dictionary that corresponds to the "defect" content of the supercell.

        :return defects: dictionary, keyed by defect type, with a set of indices of corresponding defects
        """

        def adddefect(name, index):
            if name in defects:
                defects[name].add(index)
            else:
                defects[name] = set([index])

        defects = {}
        sitechem = [self.chemistry[c] for (c, i) in self.atomindices]
        for wset, chem in zip(self.Wyckofflist, self.Wyckoffchem):
            for i in wset:
                if self.atomindices[i][0] in self.interstitial:
                    for n in range(self.size):
                        ind = n * self.N + i
                        c = self.occ[ind]
                        if c != -1: adddefect(self.__interstitialformat__.format(chem=self.chemistry[c]), ind)
                else:
                    sc = sitechem[i]
                    for n in range(self.size):
                        ind = n * self.N + i
                        c = self.occ[ind]
                        if self.chemistry[c] != sc:
                            name = self.__vacancyformat__.format(sitechem=sitechem[i]) \
                                if c == -1 else \
                                self.__antisiteformat__.format(chem=self.chemistry[c], sitechem=sc)
                            adddefect(name, ind)
        return defects

    def KrogerVink(self):
        """
        Attempt to make a "simple" string based on the defectindices, using Kroger-Vink notation.
        That is, we identify: vacancies, antisites, and interstitial sites, and return a string.
        NOTE: there is no relative charges, so this is a pseudo-KV notation.

        :return KV: string representation
        """
        defects = self.defectindices()
        return '+'.join(["{}{}".format(len(defects[name]), name)
                         if len(defects[name]) > 1 else name
                         for name in sorted(defects.keys())])

    def reorder(self, mapping):
        """
        Reorder (in place) the occupied sites. Does not change the occupancies, only the ordering
        for "presentation".

        :param mapping: list of maps; will make newchemorder[c][i] = chemorder[c][mapping[c][i]]
        :return self:

        If mapping is not a proper permutation, raises ValueError.
        """
        neworder = [[clist[cmap[i]] for i in range(len(clist))]
                    for clist, cmap in zip(self.chemorder, mapping)]
        self.chemorder, oldorder = neworder, self.chemorder
        if not self.__sane__():
            self.chemorder = oldorder
            raise ValueError('Mapping {} is not a proper permutation'.format(mapping))
        return self

    def equivalencemap(self, other):
        """
        Given the superlatt ``other`` we want to find a group operation that transforms ``self``
        into other. This is a GroupOp *along* with an index mapping of chemorder. The index
        mapping is to get the occposlist to match up:
        ``(g*self).occposlist()[c][mapping[c][i]] == other.occposlist()[c][i]``
        (We can write a similar expression using chemorder, since chemorder indexes into pos).
        We're going to return both g and mapping.

        *Remember:* ``g`` does not change the presentation ordering; ``mapping`` is
        necessary for full equivalence. If no such equivalence, return ``None,None``.

        :param other: Supercell
        :return g: GroupOp to transform sites from ``self`` to ``other``
        :return mapping: list of maps, such that (g*self).chemorder[c][mapping[c][i]] == other.chemorder[c][i]
        """
        # 1. check that our defects even match up:
        selfdefects, otherdefects = self.defectindices(), other.defectindices()
        for k, v in selfdefects.items():
            if k not in otherdefects: return None, None
            if len(v) != len(otherdefects[k]): return None, None
        for k, v in otherdefects.items():
            if k not in selfdefects: return None, None
            if len(v) != len(selfdefects[k]): return None, None

        # 2. identify the shortest common set of defects:
        defcount = {k: len(v) for k, v in selfdefects.items()}
        deftype = min(defcount, key=defcount.get)  # key to min value from dictionary
        shortset, matchset = selfdefects[deftype], otherdefects[deftype]

        mapping = None
        gocc = self.occ.copy()
        for g in self.G:
            # 3. check against the shortest list of defects:
            indexmap = g.indexmap[0]
            if any(indexmap[i] not in matchset for i in shortset): continue
            # 4. having checked that shortlist, check the full mapping:
            for ind, gind in enumerate(indexmap):
                gocc[gind] = self.occ[ind]
            if np.any(gocc != other.occ): continue
            # 5. we have a winner. Now it's all up to getting the mapping; done with index()
            gorder = [[indexmap[ind] for ind in clist] for clist in self.chemorder]
            mapping = []
            for gclist, otherlist in zip(gorder, other.chemorder):
                mapping.append([gclist.index(index) for index in otherlist])
            break

        if mapping is None: return None, mapping
        return g, mapping


class ClusterSupercell(object):
    """
    A class that defines a Supercell of a crystal for purposes of evaluating a cluster expansion.
    We intend to use this with Monte Carlo sampling.

    Takes in a crystal, a supercell (3x3 integer matrix). We can identify sites
    as spectator sites (that is, they can have different occupancies, but we do not intend
    for those to change during a Monte Carlo simulation.
    """

    def __init__(self, crys, superlatt, spectator=()):
        """
        Initialize our supercell to an empty supercell.

        :param crys: crystal object
        :param superlatt: 3x3 integer matrix
        :param spectator: list of indices of chemistries that will be considered "spectators"
        """
        self.crys = crys
        self.superlatt = superlatt.copy()
        self.spectator = [c for c in set(spectator)] # only keep unique values
        self.spectator.sort()
        self.mobile = [c for c in range(crys.Nchem) if c not in self.spectator]
        self.Nchem = crys.Nchem - len(self.spectator)
        # everything else that follows is "derived" from those initial parameters
        self.lattice = np.dot(self.crys.lattice, self.superlatt)
        self.spectatorindices = [(c, i) for c in self.spectator for i in range(len(crys.basis[c]))]
        self.mobileindices = [(c, i) for c in self.mobile for i in range(len(crys.basis[c]))]
        self.indexspectator = {ci: n for n, ci in enumerate(self.spectatorindices)}
        self.indexmobile = {ci: n for n, ci in enumerate(self.mobileindices)}
        self.Nspec, self.Nmobile = len(self.spectatorindices), len(self.mobileindices)
        self.size, self.invsuper, self.translist, self.transdict = Supercell.maketrans(self.superlatt)
        self.specpos, self.mobilepos = self.makesites()
        self.Rveclist = [np.dot(self.superlatt, t) // self.size for t in self.translist]
        # self.pos, self.occ = self.makesites(), -1 * np.ones(self.N * self.size, dtype=int)

    def makesites(self):
        """
        Generate the array corresponding to the sites; the indexing is based on the translations
        and the atomindices in crys. These may not all be filled when the supercell is finished.

        :return pos: array [N*size, 3] of supercell positions in direct coordinates
        """
        invsize = 1 / self.size
        specbasislist = [np.dot(self.invsuper, self.crys.basis[c][i]) for (c, i) in self.spectatorindices]
        mobilebasislist = [np.dot(self.invsuper, self.crys.basis[c][i]) for (c, i) in self.mobileindices]
        return np.array([crystal.incell((t + u) * invsize) for t in self.translist for u in specbasislist]), \
               np.array([crystal.incell((t + u) * invsize) for t in self.translist for u in mobilebasislist])

    def incell(self, R):
        """Map a lattice vector into a translation vector in the cell"""
        return np.dot(self.invsuper, R) % self.size

    def ciR(self, ind, mobile=True):
        """
        Return the chem/index and lattice vector for a specific indexed position
        :param ind: index of site
        :param mobile: True if mobile; false if spectator
        :return ci: (c, i) index
        :return R: lattice vector
        """
        if mobile:
            N, indices = self.Nmobile, self.mobileindices
        else:
            N, indices = self.Nspec, self.spectatorindices
        return indices[ind%N], self.Rveclist[ind//N]

    def index(self, R, ci):
        """
        Return the index that corresponds to a position specified by a lattice vector R and
        a chem/index (c,i). We also need to specify if its in the spectator basis or mobile basis.

        :param R: lattice vector
        :param ci: (c, i) index
        :return ind: index of site in our position
        :return mobile: boolean; True if mobile, False if spectator
        """
        if ci in self.mobileindices:
            return self.transdict[tuple(self.incell(R))]*self.Nmobile + self.indexmobile[ci], True
        else:
            return self.transdict[tuple(self.incell(R))]*self.Nspec + self.indexspectator[ci], False

    def indexpos(self, pos, threshold=1., CARTESIAN=False):
        """
        Return the index that corresponds to the position *closest* to pos in the supercell.
        Done in direct coordinates of the supercell, using periodic boundary conditions.

        :param pos: 3-vector
        :param threshold: (optional) minimum squared "distance" in supercell for a match; default=1.
        :return index: index of closest position
        :return mobile: boolean; True if mobile, False if spectator
        """
        index, dist2 = None, threshold
        for ind, u in enumerate(self.specpos):
            delta = crystal.inhalf(pos - u)
            d2 = np.sum(delta * delta)
            if d2 < dist2: index, dist2 = ind, d2
        dspec_min = dist2
        for ind, u in enumerate(self.mobilepos):
            delta = crystal.inhalf(pos - u)
            d2 = np.sum(delta * delta)
            if d2 < dist2: index, dist2 = ind, d2
        # if dist2 is smaller than dspec_min, it's mobile:
        return index, (dist2 < dspec_min)

    def evalcluster(self, mocc, socc, clusters):
        """
        Evaluate a cluster expansion for a given mobile occupancy and spectator occupancy.
        Indexing corresponds to `mobilepos` and `specpos`. The clusters are input as a
        list of lists of clusters (where it is assumed that all of the clusters in a given
        sublist have equal coefficients (i.e., grouped by symmetry). We return a vector
        of length Nclusters + 1; each entry is the number of times each cluster appears,
        and the *last* entry is equal to the size of the supercell (which would be an
        "empty" cluster). This can then be dotted into the vector of values to get the
        cluster expansion value.

        :param mocc: mobile occupancy vector (0 or 1 only)
        :param socc: spectator occupancy vector (0 or 1 only)
        :param clusters: list of lists (or sets) of Cluster objects
        :return clustercount: count of how many of each cluster is in this supercell.
        """
        def isocc(R, ci):
            n, mob = self.index(R, ci)
            if mob: return mocc[n] == 1
            else: return socc[n] == 1
        clustercount = np.zeros(len(clusters)+1, dtype=int)
        clustercount[-1] = self.size
        for mc, clusterlist in enumerate(clusters):
            for clust in clusterlist:
                for nR, R in enumerate(self.Rveclist):
                    if all(isocc(R+site.R, site.ci) for site in clust):
                        clustercount[mc] += 1
        return clustercount

    def clusterevaluator(self, socc, clusters, values):
        """
        Construct the information necessary for an (efficient) cluster evaluator,
        for a given spectator occupancy, set of clusters, and values for those clusters.

        :param socc: spectator occupancy vector (0 or 1 only)
        :param clusters: list of lists (or sets) of Cluster objects
        :param values: vector of values for the clusters; if it is longer than the
          list of clusters by one, the last values is assumed to be the constant value.
        :return siteinteract: list of lists of interactions for each site
        :return interact: list of interaction values
        """
        E0 = 0
        if len(values) > len(clusters):
            E0 = self.size*values[-1]
        Ninteract = 0
        interact, interdict = [], {}
        siteinteract = [[] for n in range(self.Nmobile*self.size)]
        for clusterlist, value in zip(clusters, values):
            for clust in clusterlist:
                # split into mobile and spectator
                mobilesites = [site for site in clust if site.ci in self.indexmobile]
                specsites = [site for site in clust if site.ci in self.indexspectator]
                for R in self.Rveclist:
                    if all(socc[self.index(R+site.R, site.ci)[0]] == 1 for site in specsites):
                        if len(mobilesites) == 0:
                            # spectator only == constant
                            E0 += value
                        else:
                            intertuple = tuple(sorted([self.index(R+site.R, site.ci)[0] for site in mobilesites]))
                            if intertuple in interdict:
                                # if we've already seen this particular interaction, add to the value
                                interact[interdict[intertuple]] += value
                            else:
                                # new interaction!
                                interact.append(value)
                                interdict[intertuple] = Ninteract
                                for n in intertuple:
                                    siteinteract[n].append(Ninteract)
                                Ninteract += 1
        # add on our constant term
        interact.append(E0)
        return siteinteract, interact

    def jumpnetworkevaluator(self, socc, clusters, values, chem, jumpnetwork, KRAvalues=0,
                             TSclusters=(), TSvalues=(),
                             siteinteract=(), interact=()):
        """
        Build out an efficient jump network evaluator. Similar inputs to `clusterevaluator`,
        with the addition of a jumpnetwork and energies. The interactions can be appended
        onto existing interactions, if included. The information about all of the
        transitions is: initial state, final state, delta x.

        :param socc: spectator occupancy vector (0 or 1 only)
        :param clusters: list of lists (or sets) of Cluster objects
        :param values: vector of values for the clusters; if it is longer than the
          list of clusters by one, the last values is assumed to be the constant value.
        :param chem: index of species that transitions
        :param jumpnetwork: list of lists of jumps; each is ((i, j), dx) where `i` and `j` are
          unit cell indices for species `chem`
        :param KRAvalues: list of "KRA" values for barriers (relative to average energy of endpoints);
          if `TSclusters` are used, choosing 0 is more straightforward.
        :param TSclusters: (optional) list of transition state cluster expansion terms; this is
          always added on to KRAvalues (thus using 0 is recommended if TSclusters are also used)
        :param TSvalues: (optional) values for TS cluster expansion entries
        :param siteinteract: (optional) list of lists of interactions for each site, to append
        :param interact: (optional) list of interaction values, to append

        :return siteinteract: list of lists of interactions for each site
        :return interact: list of interaction values
        :return jumps: list of ((initial, final), dx)
        :return interactrange: range of indices to count in interact for each jump; for the nth
          jump, sum over interactrange[n-1]:interactrange[n]; interactrange[-1] == range for energy
        """
        if hasattr(KRAvalues, '__len__'):
            if len(KRAvalues) != len(jumpnetwork):
                raise ValueError('Incorrect length for KRAvalues: {}'.format(KRAvalues))
        else:
            KRAvalues = KRAvalues*np.ones(len(jumpnetwork))
        if len(clusters) != len(values) != len(clusters)+1:
            raise ValueError('Incorrect length for values: {}'.format(values))
        if len(TSclusters) != len(TSvalues):
            raise ValueError('Incorrect length for TSvalues: {}'.format(TSvalues))
        siteinteract = list(siteinteract)
        interact = list(interact)
        Ninteract = len(interact)
        Ninteract0 = Ninteract # we store this now, so that we can make interactrange[-1] = Ninteract0
        # "flatten" the clusters for more efficient operations:
        # clusterinteract[(c,i)] = list of ([cs list], value) of interactions centered on (c, i)
        # NOTE: in order to maintain detailed balance, we use half the energy difference of the
        # initial and final states, so we go ahead and multiply by 0.5 here for efficiency.
        clusterinteract = {}
        for clusterlist, value in zip(clusters, values):
            for clust in clusterlist:
                for cs in clust:
                    if cs.ci in self.mobileindices:
                        # get the list of other sites, and split into mobile and spectator:
                        cllist = clust - cs
                        mobilesites = [site for site in cllist if site.ci in self.mobileindices]
                        specsites = [site for site in cllist if site.ci in self.spectatorindices]
                        if cs.ci in clusterinteract:
                            clusterinteract[cs.ci].append((mobilesites, specsites, 0.5*value))
                        else:
                            clusterinteract[cs.ci] = [(mobilesites, specsites, 0.5*value)]
        # we need to proceed one transition at a time
        Njumps, interactrange = 0, []
        jumps = []
        for jn, Etrans in zip(jumpnetwork, KRAvalues):
            for (i0, j0), deltax in jn:
                ci0, cj0 = (chem, i0), (chem, j0)
                # to get final position, it's a bit more complex... need to use dx:
                dR, cj = self.crys.cart2pos(self.crys.pos2cart(np.zeros(self.crys.dim), (chem, i0)) + deltax)
                if cj != cj0:
                    raise ArithmeticError(
                        'Transition ({},{}), {} did not land at correct site?\n{} != P{'.format(i0, j0, deltax, cj, cj0))
                # NOTE: we will need the *reverse* endpoint for the initial state...
                cs_i = cluster.ClusterSite(ci0, -dR)
                cs_j = cluster.ClusterSite(cj0, dR)
                # now, run through all lattice sites...
                for Ri in self.Rveclist:
                    # each possible *transition* is treated like its own mini-cluster expansion:
                    E0 = Etrans
                    interdict = {}
                    i = self.index(Ri, ci0)[0]
                    Rj = Ri + dR
                    j = self.index(Rj, cj0)[0]
                    jumps.append(((i, j), deltax))
                    # now, to run through our clusters, adding interactions as appropriate:
                    # -0.5*Einitial
                    for mobilesites, specsites, value in clusterinteract[ci0]:
                        # if our endpoint is also in our cluster, kick out now:
                        if cs_j in mobilesites: continue
                        # check that all of the spectator sites are occupied:
                        if all(socc[self.index(Ri + site.R, site.ci)[0]] == 1 for site in specsites):
                            if len(mobilesites) == 0:
                                # spectator only == constant
                                E0 -= value
                            else:
                                intertuple = tuple(sorted([self.index(Ri + site.R, site.ci)[0] for site in mobilesites]))
                                if intertuple in interdict:
                                    # if we've already seen this particular interaction, add to the value
                                    interact[interdict[intertuple]] -= value
                                else:
                                    # new interaction!
                                    interact.append(-value)
                                    interdict[intertuple] = Ninteract
                                    for n in intertuple:
                                        siteinteract[n].append(Ninteract)
                                    Ninteract += 1
                    # +0.5*Efinal
                    for mobilesites, specsites, value in clusterinteract[cj0]:
                        # if our initial point is also in our cluster, kick out now:
                        if cs_i in mobilesites: continue
                        # check that all of the spectator sites are occupied:
                        if all(socc[self.index(Rj + site.R, site.ci)[0]] == 1 for site in specsites):
                            if len(mobilesites) == 0:
                                # spectator only == constant
                                E0 += value
                            else:
                                intertuple = tuple(sorted([self.index(Rj + site.R, site.ci)[0] for site in mobilesites]))
                                if intertuple in interdict:
                                    # if we've already seen this particular interaction, add to the value
                                    interact[interdict[intertuple]] += value
                                else:
                                    # new interaction!
                                    interact.append(value)
                                    interdict[intertuple] = Ninteract
                                    for n in intertuple:
                                        siteinteract[n].append(Ninteract)
                                    Ninteract += 1
                    # finally, here is where we'd put the code to include the KRA expansion... TBD
                    # now add on our constant value...
                    interact.append(E0)
                    Ninteract += 1
                    interactrange.append(Ninteract)
                    Njumps += 1
        interactrange.append(Ninteract0)
        return siteinteract, interact, jumps, interactrange



