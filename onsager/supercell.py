"""
Supercell class

Class to store supercells of crystals: along with some analysis
1. add/remove/substitute atoms
2. output POSCAR format (possibly other formats?)
3. find the transformation map between two different representations of the same supercell
4. construct an NEB pathway between two supercells
5. possibly input from CONTCAR? extract displacements?
"""

__author__ = 'Dallas R. Trinkle'

import numpy as np
import collections, copy, itertools, warnings
from . import crystal
from functools import reduce


# YAML tags:
# interfaces are either at the bottom, or staticmethods in the corresponding object
# NDARRAY_YAMLTAG = '!numpy.ndarray'
# GROUPOP_YAMLTAG = '!GroupOp'

class Supercell(object):
    """
    A class that defines a Supercell of a crystal
    """

    def __init__(self, crys, super, interstitial=(), Nchem=-1, empty=False, NOSYM=False):
        """
        Initialize our supercell

        :param crys: crystal object
        :param super: 3x3 integer matrix
        :param interstitial: (optional) list/tuple of indices that correspond to interstitial sites
        :param Nchem: (optional) number of distinct chemical elements to consider; default = crys.Nchem+1
        :param empty: optional; designed to allow "copy" to work
        :param NOSYM: optional; does not do symmetry analysis (intended ONLY for testing purposes)
        """
        self.crys = crys
        self.super = super.copy()
        self.interstitial = copy.deepcopy(interstitial)
        self.Nchem = crys.Nchem + 1 if Nchem < crys.Nchem else Nchem
        if empty: return
        # everything else that follows is "derived" from those initial parameters
        self.lattice = np.dot(self.crys.lattice, self.super)
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
                # grab the set of (c,i) of Wyckoff sets (next returns first that matches, None if none:
                indexset = next((iset for iset in self.crys.Wyckoff if (c, i) in iset), None)
                self.Wyckofflist.append(frozenset([self.indexatom[ci] for ci in indexset]))
                self.Wyckoffchem.append(self.crys.chemistry[c])
        self.size, self.invsuper, self.translist, self.transdict = self.maketrans(self.super)
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
    __eqattr__ = ('translist', 'transdict', 'pos', 'G')

    def copy(self):
        """
        Make a copy of the supercell; initializes, then copies over copyattr's.
        :return: new supercell object, copy of the original
        """
        supercopy = Supercell(self.crys, self.super, self.interstitial, self.Nchem)
        for attr in self.__copyattr__: setattr(supercopy, attr, copy.deepcopy(getattr(self, attr)))
        for attr in self.__eqattr__: setattr(supercopy, attr, getattr(self, attr))
        return supercopy

    def __eq__(self, other):
        """
        Return True if two supercells are equal; this means they should have the same occupancy
        *and* the same ordering
        :param other: supercell for comparison
        :return: True if same crystal, supercell, occupancy, and ordering; False otherwise
        """
        ### Will need more....
        return isinstance(other, self.__class__) and np.all(self.super == other.super) and \
               self.interstitial == other.interstitial and np.allclose(self.pos, other.pos)

    def __ne__(self, other):
        """Inequality == not __eq__"""
        return not self.__eq__(other)

    def __str__(self):
        """Human readable version of supercell"""
        str = "Supercell of crystal:\n{crys}\n".format(crys=self.crys)
        # if self.interstitial != (): str = str + "Interstitial sites: {}\n".format(self.interstitial)
        str = str + "Supercell vectors:\n{}\nChemistry: ".format(self.super.T)
        str = str + ','.join([c + '_i({})'.format(len(l)) if n in self.interstitial else c + '({})'.format(len(l))
                              for n, c, l in
                              zip(itertools.count(), self.chemistry[:-1], self.chemorder)])
        str = str + '\nPositions:\n'
        str = str + '\n'.join([u.__str__() + ': ' + self.chemistry[o] for u, o in zip(self.pos, self.occ)])
        return str

    @staticmethod
    def maketrans(super):
        """
        Takes in a supercell matrix, and returns a list of all translations of the unit cell that
        remain inside the supercell
        :param super: 3x3 integer matrix
        :return size: integer, corresponding to number of unit cells
        :return invsuper: integer matrix inverse of supercell (needs to be divided by size)
        :return translist: list of integer vectors (to be divided by `size`) corresponding to unit cell positions
        :return transdict: dictionary of tuples and their corresponding index (inverse of trans)
        """
        size = abs(int(np.round(np.linalg.det(super))))
        invsuper = np.round(np.linalg.inv(super) * size).astype(int)
        maxN = abs(super).max()
        translist, transdict = [], {}
        for nvect in [np.array((n0, n1, n2))
                      for n0 in range(-maxN, maxN + 1)
                      for n1 in range(-maxN, maxN + 1)
                      for n2 in range(-maxN, maxN + 1)]:
            tv = np.dot(invsuper, nvect) % size
            ttup = tuple(tv)
            # if np.all(tv>=0) and np.all(tv<N): trans.append(tv)
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
        :return pos: array [N*size, 3] of (supercell) unit cell positions.
        """
        invsize = 1 / self.size
        basislist = [np.dot(self.invsuper, self.crys.basis[c][i]) for (c, i) in self.atomindices]
        return np.array([crystal.incell((t + u) * invsize) for t in self.translist for u in basislist])

    def gengroup(self):
        """
        Generate the group operations internal to the supercell
        :return Gset: set of GroupOps
        """
        Glist = []
        unittranslist = [np.dot(self.super, t)//self.size for t in self.translist]
        invsize = 1 / self.size
        for g0 in self.crys.G:
            Rsuper = np.dot(self.invsuper, np.dot(g0.rot, self.super))
            if not np.all(Rsuper % self.size == 0):
                warnings.warn('Broken symmetry? GroupOp:\n{}\nnot a symmetry operation of supercell?\nRsuper=\n{}'.format(g0, Rsuper),
                              RuntimeWarning, stacklevel=2)
                continue
            else:
                # divide out the size (in inverse super). Should still be an integer matrix (and hence, a symmetry)
                Rsuper //= self.size
            # for t, u in zip(self.translist, unittranslist):
            for u in unittranslist:
                # first, make the corresponding group operation by adding the unit cell translation:
                g = g0 + u
                # translation vector *in the supercell*; go ahead and keep it inside the supercell, too.
                # tsuper = ((np.dot(self.invsuper, g0.trans) + t) % self.size) * invsize
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
                if len(set(indexmap)) != self.N*self.size:
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
        cind = c%(self.Nchem+1)
        if c<self.crys.Nchem or c==self.Nchem:
            raise IndexError('Trying to set the chemistry for a lattice atom / vacancy')
        self.chemistry[c] = chemistry

    def setocc(self, ind, c):
        """
        Set the occupancy of position indexed by ind, to chemistry c. Used by all the other algorithms.
        :param ind: integer index
        :param c: chemistry index
        """
        if c<-2 or c>self.crys.Nchem:
            raise IndexError('Trying to occupy with a non-defined chemistry: {} out of range'.format(c))
        corig = self.occ[ind]
        if corig != c:
            if corig>=0:
                # remove from chemorder list (if not vacancy)
                co = self.chemorder[corig]
                co.pop(co.index(ind))
            if c>=0:
                # add to chemorder list (if not vacancy)
                self.chemorder[c].append(ind)
            # finally: set the occupancy
            self.occ[ind] = c

    def fillperiodic(self, ci, Wyckoff=True):
        """
        Occupies all of the (Wyckoff) sites corresponding to chemical index with the appropriate chemistry
        :param ci: tuple of (chem, index) in crystal
        :param Wyckoff: (optional) if False, *only* occupy the specific tuple, but still periodically
        """
        if __debug__:
            if ci not in self.indexatom: raise IndexError('Tuple {} not a corresponding atom index'.format(ci))
        ind = self.indexatom[ci]
        indlist = next((nset for nset in self.Wyckofflist if ind in nset), None) if Wyckoff else (ind,)
        for i in [n*self.N+i for n in range(self.size) for i in indlist]:
            self.setocc(i, ci[0])
