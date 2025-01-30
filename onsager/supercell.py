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
import collections, copy, itertools, warnings, yaml
from numbers import Integral
from onsager import crystal, cluster


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
        if size == 0: raise ZeroDivisionError('Tried to use a singular supercell.')
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

    def defect_chemmapping(self):
        """Returns a ``chemmapping`` dictionary corresponding to defects"""
        chemmapping = {}
        for n in range(self.crys.Nchem):
            # start with anything on a site being "occupied"
            cmap = {i: 1 for i in range(-1, self.Nchem)}
            # if it's an interstitial, vacancy == unoccupied; if its native, native chem == unoccupied
            if n in self.interstitial:
                cmap[-1] = 0  # vacancies are "unoccupied"
            else:
                cmap[n] = 0  # "correct" site
            chemmapping[n] = cmap
        return chemmapping

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

    def POSCAR_occ(self, POSCAR_str, EMPTY_SUPER=True, disp_threshold=-1, latt_threshold=-1):
        """
        Takes in a POSCAR_str, and sets the occupancy of the supercell accordingly.
        Note: if we want to read a POSCAR from a file instead, the proper usage is

        ::

            with open(POSCAR_filename, "r") as f:
                sup.from_POSCAR(f.read())

        Warning: there is only minimal validity checking; it makes a strong assumption
        that a reasonable POSCAR file is being given, and that the sites should correspond
        to the supercell object. Should that not be the case, the behavior is unspecified.

        :param POSCAR_str: string form of a POSCAR
        :param EMPTY_SUPER: initialize supercell by emptying it first (default=True)
        :param disp_threshold: threshold for difference in position to raise an error, in
            unit cell coordinates. For a negative value, we compute a value that is ~0.1A.
            If you do *not* want this check to be meaningful, choose a value >sqrt(3)=1.733
        :param latt_threshold: threshold for supercell lattice check, in units of strain.
            For a negative values **do not check**.
        :return: name from the POSCAR
        """
        POSCAR_list = POSCAR_str.split('\n')  # break into lines
        name = POSCAR_list.pop(0)
        a0 = float(POSCAR_list.pop(0))
        alist = []
        for _ in range(3):
            alist.append(a0 * np.array([float(astr) for astr in (POSCAR_list.pop(0)).split()]))
        super_latt = np.array(alist).T
        super_inv = np.linalg.inv(super_latt)
        if latt_threshold > 0:
            metric = np.dot(super_latt.T, super_latt)  # metric tensor
            super_metric = np.dot(self.lattice.T, self.lattice)  # supercell metric tensor
            max_diff = max(abs(metric[i,j] - super_metric[i,j])/np.sqrt(super_metric[i,i]*super_metric[j,j])
                           for i in range(3) for j in range(3))
            if max_diff > latt_threshold*latt_threshold:
                msg = '{}\n (supercell) and\n{}\n (POSCAR)\ndiffer by {} > {}'.format(self.lattice.T, super_latt.T,
                                                                                      np.sqrt(max_diff), latt_threshold)
                raise ValueError(msg)
        if disp_threshold < 0:
            # try to make a fairly liberal choice, based on the dimensions of the
            # supercell; essentially, this is ~0.1A
            disp_threshold = 0.1/np.sqrt(np.max(np.diag(np.dot(super_latt.T, super_latt))))
        # we should probably do a sanity check: are we trying to occupy with a sensible
        # supercell? for now, we'll skip this.
        chemlist = (POSCAR_list.pop(0)).split()
        # this optional (?) line may specify the chemical element ordering, or it may
        # just be the element numbers; if it's the former, we need to parse:
        if '0' <= chemlist[0][0] <= '9':
            chemident = [n for n in range(len(chemlist))]
        else:
            chemident = [self.chemistry.index(elem) for elem in chemlist]
            chemlist = (POSCAR_list.pop(0)).split()
        Nspecies = [int(s) for s in chemlist]
        coord_type = (POSCAR_list.pop(0)).strip()
        # check for "selective dynamics" switch
        if coord_type[0] in {'s', 'S'}:
            coord_type = (POSCAR_list.pop(0)).strip()
        cart_coord = coord_type[0] in {'c', 'C', 'k', 'K'}
        if EMPTY_SUPER:
            for n in range(self.N * self.size):
                self.setocc(n, -1)
        # finally, read all of the entries...
        for N, c in zip(Nspecies, chemident):
            for _ in range(N):
                ustr = (POSCAR_list.pop(0)).split()
                uvec = np.array([float(u) for u in ustr[:3]])
                if cart_coord:
                    uvec = np.dot(super_inv, uvec)
                i = self.index(uvec, threshold=disp_threshold)
                if i is None:
                    msg = 'Unable to map {} into supercell'.format(uvec)
                    raise ValueError(msg)
                self.setocc(i, c)
        return name

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
        self.spectator = [c for c in set(spectator)]  # only keep unique values
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
        self.vacancy = None  # assume there no vacancy
        # self.pos, self.occ = self.makesites(), -1 * np.ones(self.N * self.size, dtype=int)

    def addvacancy(self, ind):
        """Adds a vacancy into the mobile species at a specific index"""
        if ind is None:
            self.vacancy = None
        else:
            i_ind = int(ind)
            if i_ind < 0 or i_ind >= self.Nmobile*self.size:
                raise IndexError('{} is out of range; should be between 0 and {}'.format(ind, self.Nmobile*self.size))
            self.vacancy = i_ind

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
        return tuple(np.dot(self.invsuper, R) % self.size)

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
        return indices[ind % N], self.Rveclist[ind // N]

    def index(self, R, ci):
        """
        Return the index that corresponds to a position specified by a lattice vector R and
        a chem/index (c,i). We also need to specify if its in the spectator basis or mobile basis.

        :param R: lattice vector
        :param ci: (c, i) index
        :return ind: index of site in our position
        :return mobile: boolean; True if mobile, False if spectator
        """
        if ci in self.indexmobile:
            return self.transdict[self.incell(R)] * self.Nmobile + self.indexmobile[ci], True
        else:
            return self.transdict[self.incell(R)] * self.Nspec + self.indexspectator[ci], False

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

    def expiqx(self):
        """
        Construct a Fourier transform matrix for our mobile species

        :return: exp( I q[i] . x[j]) as a matrix
        :return: gamma_index of the gamma point (0)
        """
        # we need to get the rlv this way:
        size, _, qtranslist, transdict = Supercell.maketrans(self.superlatt.T)
        invsize = 1./size
        phase = 2j*np.pi
        return np.array([[np.exp(phase*np.dot(np.dot(self.superlatt, u), q*invsize))
                          for u in self.mobilepos]
                         for q in qtranslist]), \
               transdict[(0,)*self.crys.dim]

    def Supercell_occ(self, sup, chemmapping=None):
        """
        Takes in a Supercell object (that is assumed to be consistent with this supercell!)
        and produces the corresponding occupancy vectors for *this* supercell, using a
        specific chemical mapping (described below).

        In a Supercell object, each *site* has a "native" chemistry; moreover, those sites
        may be occupied by everything from a vacancy (-1) to a different chemical element (>=0).
        We need to define how that happens, since ClusterSupercells only have occupancies of 0 or 1.

        ``chemmapping`` is a dictionary of dictionaries. ``chemmapping[csite][cocc]`` = 0 or 1
        to dictate what the occupancy for a site *should* be if chemistry of type ``cocc`` occurs
        on a site with native chemistry ``csite``.

        If the ``chemmapping`` is None, we use a default "defect" occupancy mapping; namely,
        if ``csite`` != Interstitial, then we use 0 when ``csite==cocc``, 1 otherwise; and
        if ``csite`` == Interstitial, we use 0 when ``csite==-1``, 1 otherwise. See
        ``Supercell.defect_chemmapping()``

        :param sup: Supercell object, with appropriate chemical occupancies
        :param chemmapping: mapping of chemical identities to occupancy variables.
        :return mocc: mobile occupancy vector
        :return socc: spectator occupancy vector
        """
        if chemmapping is None:
            chemmapping = sup.defect_chemmapping()
        mocc = np.zeros(self.Nmobile * self.size, dtype=int)
        socc = np.zeros(self.Nspec * self.size, dtype=int)
        # now, we just run through the positions in *this* supercell, and get the occupancy
        # in the other supercell. Consistency is key!
        for ind, pos in enumerate(self.mobilepos):
            csite = self.mobileindices[ind % self.Nmobile][0]  # get the site chemistry
            cocc = sup[pos]  # get the occupancy chemistry
            mocc[ind] = chemmapping[csite][cocc]
        for ind, pos in enumerate(self.specpos):
            csite = self.spectatorindices[ind % self.Nspec][0]  # get the site chemistry
            cocc = sup[pos]  # get the occupancy chemistry
            socc[ind] = chemmapping[csite][cocc]
        return mocc, socc

    def evalcluster(self, mocc, socc, clusters):
        """
        Evaluate a cluster expansion for a given mobile occupancy and spectator occupancy.
        Indexing corresponds to ``mobilepos`` and ``specpos``. The clusters are input as a
        list of lists of clusters (where it is assumed that all of the clusters in a given
        sublist have equal coefficients (i.e., grouped by symmetry). We return a vector
        of length Nclusters + 1; each entry is the number of times each cluster appears,
        and the *last* entry is equal to the size of the supercell (which would be an
        "empty" cluster). This can then be dotted into the vector of values to get the
        cluster expansion value.

        :param mocc: mobile occupancy vector (0 or 1 only)
        :param socc: spectator occupancy vector (0 or 1 only)
        :param clusters: list of lists (or sets) of Cluster objects
        :return: clustercount: count of how many of each cluster is in this supercell.
        """

        def isocc(R, ci):
            n, mob = self.index(R, ci)
            if mob:
                return mocc[n] == 1
            else:
                return socc[n] == 1

        # treatment for vacancy clusters...
        if self.vacancy is not None:
            # sanity check:
            if mocc[self.vacancy] == 1:
                raise RuntimeWarning('Supercell contains a vacancy at {} but mobile occupancy == 1?'.format(self.vacancy))
            ci_vac, R_vac = self.ciR(self.vacancy)

        clustercount = np.zeros(len(clusters) + 1, dtype=int)
        clustercount[-1] = self.size
        for mc, clusterlist in enumerate(clusters):
            for clust in clusterlist:
                if clust.__vacancy__:
                    if self.vacancy is None: continue
                    elif clust.vacancy().ci != ci_vac: continue
                    Rveclist = [R_vac]
                else:
                    Rveclist = self.Rveclist
                for R in Rveclist:
                    if all(isocc(R + site.R, site.ci) for site in clust):
                        clustercount[mc] += 1
        return clustercount

    def expandcluster_matrices(self, socc, clusters):
        """
        Expand a cluster expansion for a given spectator occupancy into matrices of indices.
        This is designed for rapid evaluation for a fixed spectator occupancy. The clusters are
        input as a list of lists of clusters (i.e., grouped by symmetry). We return a list of
        lists of integer matrices of indices. This can then be used to efficiently evaluate cluster
        counts for a given mobile occupancy. The given row of indices must be all 1 in order to
        increment the particular cluster count.

        :param socc: spectator occupancy vector (0 or 1 only)
        :param clusters: list of lists (or sets) of Cluster objects
        :return: clustermatrices: list of lists of matrices of indices
        """
        # treatment for vacancy clusters...
        if self.vacancy is not None:
            # sanity check:
            ci_vac, R_vac = self.ciR(self.vacancy)

        clustermatrices = []
        for clusterlist in clusters:
            clmat_list = []
            for clust in clusterlist:
                cl_indices = []
                if clust.__vacancy__:
                    if self.vacancy is None: continue
                    elif clust.vacancy().ci != ci_vac: continue
                    Rveclist = [R_vac]
                else:
                    Rveclist = self.Rveclist
                for R in Rveclist:
                    # list of indices, and whether the particular set is "active"
                    ind, active = [], True
                    for site in clust:
                        n, mob = self.index(R + site.R, site.ci)
                        if mob:
                            ind.append(n)
                        else:
                            active &= (socc[n] == 1)
                    if active:
                        cl_indices.append(ind)
                clmat_list.append(np.array(cl_indices, dtype=int))
            clustermatrices.append(clmat_list)
        return clustermatrices

    def evalTScluster(self, mocc, socc, TSclusters, initial, final, dx):
        """
        Evaluate a TS cluster expansion for a given mobile occupancy and spectator occupancy.
        Indexing corresponds to ``mobilepos`` and ``specpos``. The clusters are input as a
        list of lists of clusters (where it is assumed that all of the clusters in a given
        sublist have equal coefficients (i.e., grouped by symmetry). We return a vector
        of length Nclusters; each entry is the number of times each cluster appears.
        This can then be dotted into the vector of values to get the cluster expansion value.
        This is evaluated for the transition where the mobile species at ``initial`` jumps
        to the position at ``final``. Requires mocc[initial] == 1 and mocc[final] == 0

        :param mocc: mobile occupancy vector (0 or 1 only)
        :param socc: spectator occupancy vector (0 or 1 only)
        :param TSclusters: list of lists (or sets) of (transition state) Cluster objects
        :param initial: index of initial state
        :param final: index of final state
        :param dx: displacement vector (necessary to deal with PBC)
        :return: clustercount: count of how many of each cluster is in this supercell.
        """

        def isocc(R, ci):
            n, mob = self.index(R, ci)
            if mob:
                if n == self.vacancy:
                    raise RuntimeError('Checked the occupancy for the vacancy?')
                return mocc[n] == 1
            else:
                return socc[n] == 1

        clustercount = np.zeros(len(TSclusters), dtype=int)
        vacancy = (self.vacancy is not None)
        if vacancy:
            if initial != self.vacancy:
                raise RuntimeWarning('Attempting to evaluate in cell with vacancy at '
                                     '{} but TS= {}->{}'.format(self.vacancy, initial, final))
                # return clustercount  # we can only evaluate this meaningfully for our vacancy
        elif (mocc[initial] == 0 or mocc[final] == 1):
            return clustercount  # trivial result...
        ci_i, Ri = self.ciR(initial)
        ci_j, Rj = self.ciR(final)
        # need to fix this so that it matches dx!!
        # dR = Rj - Ri
        chem, i, j = ci_i[0], ci_i[1], ci_j[1]
        dR = np.round(np.dot(self.crys.invlatt, dx) - self.crys.basis[chem][j] + self.crys.basis[chem][i]).astype(int)
        cs_i0 = cluster.ClusterSite(ci_i, np.zeros(self.crys.dim))
        cs_j0 = cluster.ClusterSite(ci_j, dR)
        cs_i1 = cluster.ClusterSite(ci_i, -dR)
        cs_j1 = cluster.ClusterSite(ci_j, np.zeros(self.crys.dim))
        for mc, clusterlist in enumerate(TSclusters):
            # if next(iter(clusterlist)).__vacancy__:
            #     raise NotImplementedError('TS cluster evaluation for vacancy jumps not currently implemented')
            for clust in clusterlist:
                if (cs_i0, cs_j0) == clust.transitionstate():
                    if all(isocc(Ri + site.R, site.ci) for site in clust):
                        clustercount[mc] += 1
                elif not vacancy and (cs_j1, cs_i1) == clust.transitionstate():
                # elif (cs_j1, cs_i1) == clust.transitionstate():
                    if all(isocc(Rj + site.R, site.ci) for site in clust):
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
            E0 = self.size * values[-1]
        Ninteract = 0
        interact, interdict = [], {}
        siteinteract = [[] for n in range(self.Nmobile * self.size)]
        if self.vacancy is not None:
            ci_vac, R_vac = self.ciR(self.vacancy)
        for clusterlist, value in zip(clusters, values):
            for clust in clusterlist:
                if clust.__vacancy__:
                    # do we have a vacancy in the right place?
                    if self.vacancy is None: continue
                    elif clust.vacancy().ci != ci_vac: continue
                    # now, set it up!
                    Rveclist = [R_vac]
                else:
                    Rveclist = self.Rveclist
                # split into mobile and spectator
                mobilesites = [site for site in clust if site.ci in self.indexmobile]
                specsites = [site for site in clust if site.ci in self.indexspectator]
                for R in Rveclist:
                    if all(socc[self.index(R + site.R, site.ci)[0]] == 1 for site in specsites):
                        if len(mobilesites) == 0:
                            # spectator only == constant
                            E0 += value
                        else:
                            intertuple = tuple(sorted(self.index(R + site.R, site.ci)[0] for site in mobilesites))
                            if self.vacancy in intertuple:
                                continue
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
        Build out an efficient jump network evaluator. Similar inputs to ``clusterevaluator``,
        with the addition of a jumpnetwork and energies. The interactions can be appended
        onto existing interactions, if included. The information about all of the
        transitions is: initial state, final state, delta x.

        :param socc: spectator occupancy vector (0 or 1 only)
        :param clusters: list of lists (or sets) of Cluster objects
        :param values: vector of values for the clusters; if it is longer than the
          list of clusters by one, the last values is assumed to be the constant value.
        :param chem: index of species that transitions
        :param jumpnetwork: list of lists of jumps; each is ((i, j), dx) where ``i`` and ``j`` are
          unit cell indices for species ``chem``
        :param KRAvalues: list of "KRA" values for barriers (relative to average energy of endpoints);
          if ``TSclusters`` are used, choosing 0 is more straightforward.
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
            KRAvalues = KRAvalues * np.ones(len(jumpnetwork))
        if len(clusters) != len(values) != len(clusters) + 1:
            raise ValueError('Incorrect length for values: {}'.format(values))
        if len(TSclusters) != len(TSvalues):
            raise ValueError('Incorrect length for TSvalues: {}'.format(TSvalues))
        siteinteract = list(siteinteract)
        interact = list(interact)
        Ninteract = len(interact)
        Ninteract0 = Ninteract  # we store this now, so that we can make interactrange[-1] = Ninteract0
        # "flatten" the clusters for more efficient operations:
        # clusterinteract[(c,i)] = list of ([cs list], value) of interactions centered on (c, i)
        # NOTE: in order to maintain detailed balance, we use half the energy difference of the
        # initial and final states, so we go ahead and multiply by 0.5 here for efficiency.
        clusterinteract = {}
        for clusterlist, value in zip(clusters, values):
            for clust in clusterlist:
                for cs in clust:
                    if cs.ci in self.indexmobile:
                        # get the list of other sites, and split into mobile and spectator:
                        cllist = clust - cs
                        mobilesites = [site for site in cllist if site.ci in self.indexmobile]
                        specsites = [site for site in cllist if site.ci in self.indexspectator]
                        if cs.ci in clusterinteract:
                            clusterinteract[cs.ci].append((mobilesites, specsites, 0.5 * value))
                        else:
                            clusterinteract[cs.ci] = [(mobilesites, specsites, 0.5 * value)]
        # "flatten" the TS clusters. To simplify, we put in both forward and backward jumps
        TSclusterinteract = {}
        for TSclusterlist, value in zip(TSclusters, TSvalues):
            for TSclust in TSclusterlist:
                TS = TSclust.transitionstate()
                if TS[0].ci in self.indexmobile and TS[1].ci in self.indexmobile:
                    R0 = TS[0].R
                    TS0 = (TS[0] - R0, TS[1] - R0)
                    mobilesites = [site - R0 for site in TSclust if site.ci in self.indexmobile]
                    specsites = [site - R0 for site in TSclust if site.ci in self.indexspectator]
                    if TS0 in TSclusterinteract:
                        TSclusterinteract[TS0].append((mobilesites, specsites, value))
                    else:
                        TSclusterinteract[TS0] = [(mobilesites, specsites, value)]
                    R1 = TS[1].R
                    mobilesites = [site - R1 for site in TSclust if site.ci in self.indexmobile]
                    specsites = [site - R1 for site in TSclust if site.ci in self.indexspectator]
                    TS1 = (TS[1] - R1, TS[0] - R1)
                    if TS1 in TSclusterinteract:
                        TSclusterinteract[TS1].append((mobilesites, specsites, value))
                    else:
                        TSclusterinteract[TS1] = [(mobilesites, specsites, value)]
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
                        'Transition ({},{}), {} did not land at correct site?\n{} != P{'.format(i0, j0, deltax, cj,
                                                                                                cj0))
                cs_i0 = cluster.ClusterSite(ci0, np.zeros(self.crys.dim, dtype=int))
                # NOTE: we will need the *reverse* endpoint for the initial state...
                cs_i = cluster.ClusterSite(ci0, -dR)
                cs_j = cluster.ClusterSite(cj0, dR)
                # construct sublists of cluster expansions that explicitly *exclude* our endpoints:
                clusterinteract_ci0 = [(ms, ss, -val) for (ms, ss, val) in clusterinteract[ci0] if cs_j not in ms]
                clusterinteract_cj0 = [(ms, ss, val) for (ms, ss, val) in clusterinteract[cj0] if cs_i not in ms]
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
                    # -0.5*Einitial +0.5*Efinal
                    for mobilesites, specsites, value, Rsite in [msssval + (Ri,) for msssval in clusterinteract_ci0] + \
                                                                [msssval + (Rj,) for msssval in clusterinteract_cj0]:
                        # if our initial point is also in our cluster, kick out now:
                        # if cs_i in mobilesites: continue
                        # check that all of the spectator sites are occupied:
                        if all(socc[self.index(Rsite + site.R, site.ci)[0]] == 1 for site in specsites):
                            if len(mobilesites) == 0:
                                # spectator only == constant
                                E0 += value
                            else:
                                intertuple = tuple(sorted(self.index(Rsite + site.R, site.ci)[0]
                                                          for site in mobilesites))
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
                    # finally, here is where we'd put the code to include the KRA expansion...
                    if (cs_i0, cs_j) in TSclusterinteract:
                        for mobilesites, specsites, value in TSclusterinteract[cs_i0, cs_j]:
                            if all(socc[self.index(Ri + site.R, site.ci)[0]] == 1 for site in specsites):
                                if len(mobilesites) == 0:
                                    # spectator only == constant
                                    E0 += value
                                else:
                                    intertuple = tuple(
                                        sorted(self.index(Ri + site.R, site.ci)[0] for site in mobilesites))
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
                    # now add on our constant value...
                    interact.append(E0)
                    Ninteract += 1
                    interactrange.append(Ninteract)
                    Njumps += 1
        interactrange.append(Ninteract0)
        return siteinteract, interact, jumps, interactrange

    def jumpnetworkevaluator_vacancy(self, socc, clusters, values, chem, jumpnetwork, KRAvalues=0,
                                     TSclusters=(), TSvalues=(),
                                     siteinteract=(), interact=()):
        """
        Build out an efficient jump network evaluator for a vacancy. Similar inputs to
        ``jumpnetworkevaluator``. This is designed for a "stationary" vacancy, where we'll
        just look at its jumps in the supercell.

        :param socc: spectator occupancy vector (0 or 1 only)
        :param clusters: list of lists (or sets) of Cluster objects
        :param values: vector of values for the clusters; if it is longer than the
          list of clusters by one, the last values is assumed to be the constant value.
        :param chem: index of species that transitions
        :param jumpnetwork: list of lists of jumps; each is ((i, j), dx) where ``i`` and ``j`` are
          unit cell indices for species ``chem``
        :param KRAvalues: list of "KRA" values for barriers (relative to average energy of endpoints);
          if ``TSclusters`` are used, choosing 0 is more straightforward.
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
        zerovec = np.zeros(self.crys.dim, dtype=int)
        if self.vacancy is None:
            raise RuntimeError('Supercell does not contain a vacancy; use `addvacancy()` first')
        ci_vac, R_vac = self.ciR(self.vacancy)
        if ci_vac[0] != chem:
            raise RuntimeError('Vacancy {} has chemistry {} not {}'.format(self.vacancy, ci_vac[0], chem))
        if hasattr(KRAvalues, '__len__'):
            if len(KRAvalues) != len(jumpnetwork):
                raise ValueError('Incorrect length for KRAvalues: {}'.format(KRAvalues))
        else:
            KRAvalues = KRAvalues * np.ones(len(jumpnetwork))
        if len(clusters) != len(values) != len(clusters) + 1:
            raise ValueError('Incorrect length for values: {}'.format(values))
        if len(TSclusters) != len(TSvalues):
            raise ValueError('Incorrect length for TSvalues: {}'.format(TSvalues))
        siteinteract = list(siteinteract)
        interact = list(interact)
        Ninteract = len(interact)
        Ninteract0 = Ninteract  # we store this now, so that we can make interactrange[-1] = Ninteract0
        # "flatten" the clusters for more efficient operations:
        # clusterinteract[(c,i)] = list of ([cs list], value) of interactions centered on (c, i)
        # NOTE: in order to maintain detailed balance, we use half the energy difference of the
        # initial and final states, so we go ahead and multiply by 0.5 here for efficiency.
        clusterinteract = {}
        vacclusterinteract = {}
        for clusterlist, value in zip(clusters, values):
            for clust in clusterlist:
                if clust.__vacancy__:
                    civ = clust.vacancy().ci
                    mobilesites = [site for site in clust if site.ci in self.indexmobile]
                    specsites = [site for site in clust if site.ci in self.indexspectator]
                    if civ in vacclusterinteract:
                        vacclusterinteract[civ].append((mobilesites, specsites, 0.5 * value))
                    else:
                        vacclusterinteract[civ] = [(mobilesites, specsites, 0.5 * value)]
                else:
                    for cs in clust:
                        if cs.ci in self.indexmobile:
                            # get the list of other sites, and split into mobile and spectator:
                            cs0 = cluster.ClusterSite(cs.ci, zerovec)
                            cllist = clust - cs
                            mobilesites = [cs0] + [site for site in cllist if site.ci in self.indexmobile]
                            specsites = [site for site in cllist if site.ci in self.indexspectator]
                            if cs.ci in clusterinteract:
                                clusterinteract[cs.ci].append((mobilesites, specsites, 0.5 * value))
                            else:
                                clusterinteract[cs.ci] = [(mobilesites, specsites, 0.5 * value)]
        # "flatten" the TS clusters. To simplify, we put in both forward and backward jumps
        TSclusterinteract = {}
        for TSclusterlist, value in zip(TSclusters, TSvalues):
            for TSclust in TSclusterlist:
                # just check that we're dealing with vacancies, with the right chemistry
                if not TSclust.__vacancy__: continue
                TS = TSclust.transitionstate()
                if TS[0].ci[0] != chem or TS[1].ci[0] != chem: continue
                R0 = TS[0].R
                TS0 = (TS[0] - R0, TS[1] - R0)
                mobilesites = [site - R0 for site in TSclust if site.ci in self.indexmobile]
                specsites = [site - R0 for site in TSclust if site.ci in self.indexspectator]
                if TS0 in TSclusterinteract:
                    TSclusterinteract[TS0].append((mobilesites, specsites, value))
                else:
                    TSclusterinteract[TS0] = [(mobilesites, specsites, value)]
                # Need to add "reverse" jumps to consider in our TS expansions (proper consideration of trans. state)
                # R1 = TS[1].R
                # TS1 = (TS[1] - R1, TS[0] - R1)
                # mobilesites = [site - R1 for site in TSclust if site.ci in self.indexmobile]
                # specsites = [site - R1 for site in TSclust if site.ci in self.indexspectator]
                # if TS1 in TSclusterinteract:
                #     TSclusterinteract[TS1].append((mobilesites, specsites, value))
                # else:
                #     TSclusterinteract[TS1] = [(mobilesites, specsites, value)]
        # we need to proceed one transition at a time
        Njumps, interactrange = 0, []
        jumps = []
        for jn, Etrans in zip(jumpnetwork, KRAvalues):
            for (i0, j0), deltax in jn:
                ci0, cj0 = (chem, i0), (chem, j0)
                if ci0 != ci_vac: continue
                # to get final position, it's a bit more complex... need to use dx:
                dR, cj = self.crys.cart2pos(self.crys.pos2cart(zerovec, (chem, i0)) + deltax)
                if cj != cj0:
                    raise ArithmeticError(
                        'Transition ({},{}), {} did not land at correct site?\n{} != P{'.format(i0, j0, deltax, cj,
                                                                                                cj0))
                cs_i0 = cluster.ClusterSite(ci0, zerovec)
                # NOTE: we will need the *reverse* endpoint for the initial state...
                cs_i = cluster.ClusterSite(ci0, -dR)
                cs_j = cluster.ClusterSite(cj0, dR)
                # Ri = R_vac
                # each possible *transition* is treated like its own mini-cluster expansion:
                E0 = Etrans
                interdict = {}
                i = self.index(R_vac, ci0)[0]
                if i != self.vacancy:
                    raise RuntimeError('Somehow did not correctly map to the vacancy? Should never happen')
                Rj = R_vac + dR
                j = self.index(Rj, cj0)[0]
                jumps.append(((i, j), deltax))

                # construct sublists of cluster expansions based on the "i" site and the "j" site
                # half of these are for the initial, and half are for the final. For simplicity, we do
                # the vacancy based ones first.
                # clusterinteract_ci0 = [(ms, ss, val) for (ms, ss, val) in vacclusterinteract[ci0]]
                # clusterinteract_cj0 = [(ms, ss, val) for (ms, ss, val) in vacclusterinteract[cj0]]
                # clusterinteract_ci0 += [(ms, ss, -val) for (ms, ss, val) in clusterinteract[ci0] if cs_j not in ms]
                # clusterinteract_cj0 += [(ms, ss, -val) for (ms, ss, val) in clusterinteract[cj0] if cs_i not in ms]

                # we need to make an index mapping for our end-point state (the two sites that are switched!)
                init_map = np.array(range(self.Nmobile*self.size))
                rev_map = init_map.copy()
                rev_map[i] = j
                rev_map[j] = i
                # -0.5*Einitial + 0.5*Efinal: vac(initial), vac(final), solute(initial), solute(final)
                clusterinteract_map = [(ms, ss, -val, R_vac, init_map) for (ms, ss, val) in vacclusterinteract[ci0]] + \
                                      [(ms, ss, +val, Rj, rev_map) for (ms, ss, val) in vacclusterinteract[cj0]] + \
                                      [(ms, ss, -val, Rj, init_map) for (ms, ss, val) in clusterinteract[cj0]
                                       if cs_i not in ms] + \
                                      [(ms, ss, +val, R_vac, rev_map) for (ms, ss, val) in clusterinteract[ci0]
                                       if cs_j not in ms]
                for mobilesites, specsites, value, Rsite, mapping in clusterinteract_map:
                    # if our endpoint is also in our cluster, kick out now:
                    # if cs_j in mobilesites: continue
                    # check that all of the spectator sites are occupied:
                    if all(socc[self.index(Rsite + site.R, site.ci)[0]] == 1 for site in specsites):
                        if len(mobilesites) == 0:
                            # spectator only == constant
                            E0 += value
                        else:
                            intertuple = tuple(sorted(mapping[self.index(Rsite + site.R, site.ci)[0]]
                                                      for site in mobilesites))
                            if intertuple in interdict:
                                # if we've already seen this particular interaction, add to the value
                                interact[interdict[intertuple]] += value
                            else:
                                # new interaction!
                                interact.append(+value)
                                interdict[intertuple] = Ninteract
                                for n in intertuple:
                                    siteinteract[n].append(Ninteract)
                                Ninteract += 1
                # finally, here is where we'd put the code to include the KRA expansion...
                if (cs_i0, cs_j) in TSclusterinteract:
                    for mobilesites, specsites, value in TSclusterinteract[cs_i0, cs_j]:
                        if all(socc[self.index(R_vac + site.R, site.ci)[0]] == 1 for site in specsites):
                            if len(mobilesites) == 0:
                                # spectator only == constant
                                E0 += value
                            else:
                                intertuple = tuple(
                                    sorted(self.index(R_vac + site.R, site.ci)[0] for site in mobilesites))
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
                # now add on our constant value...
                interact.append(E0)
                Ninteract += 1
                interactrange.append(Ninteract)
                Njumps += 1
        interactrange.append(Ninteract0)
        return siteinteract, interact, jumps, interactrange
