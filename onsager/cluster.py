"""
Cluster expansion module: types necessary to implement cluster expansions of
quantities based on crystals.
"""

__author__ = 'Dallas R. Trinkle'

import numpy as np
import copy, collections, itertools, warnings
from onsager import crystal

# YAML tags
CLUSTERSITE_YAMLTAG = '!ClusterSite'


class ClusterSite(collections.namedtuple('ClusterSite', 'ci R')):
    """
    A class corresponding to a site in a cluster.

    :param ci: (chem, index) of the site
    :param R: lattice vector of the site
    """

    def _asdict(self):
        """Return a proper dict"""
        return {'ci': self.ci, 'R': self.R}

    @classmethod
    def fromcryscart(cls, crys, cart_pos):
        """Return a ClusterSite corresponding to Cartesian position `cart_pos` in crystal `crys`"""
        R, ci = crys.cart2pos(cart_pos)
        return cls(ci=ci, R=R)

    @classmethod
    def fromcrysunit(cls, crys, unit_pos):
        """Return a ClusterSite corresponding to unit cell position `unit_pos` in crystal `crys`"""
        cart_pos = crys.unit2cart(np.zeros(crys.dim, dtype=int), unit_pos)
        return cls.fromcryscart(cart_pos)

    def __eq__(self, other):
        """Test for equality--we don't bother checking dx"""
        return isinstance(other, self.__class__) and \
               (self.ci == other.ci and np.all(self.R == other.R))

    def __ne__(self, other):
        """Inequality == not __eq__"""
        return not self.__eq__(other)

    def __hash__(self):
        """Hash, so that we can make sets of states"""
        # return self.i ^ (self.j << 1) ^ (self.R[0] << 2) ^ (self.R[1] << 3) ^ (self.R[2] << 4)
        return hash(self.ci + tuple(self.R))

    def __neg__(self):
        """Negation of site"""
        return self.__class__(ci=self.ci, R=-self.R)

    def __add__(self, other):
        """Add a vector to a site; other *must* be a vector
        """
        if len(other) != len(self.R):
            raise ArithmeticError('Dimensionality problem? Adding {} to {}'.format(other, self))
        return self.__class__(ci=self.ci, R=self.R + np.array(other))

    def __sub__(self, other):
        return self.__add__(-np.array(other))

    def g(self, crys, g):
        """
        Apply group operation.

        :param crys: crystal
        :param g: group operation (from crys)
        :return g*site: corresponding to group operation applied to self
        """
        gR, gci = crys.g_pos(g, self.R, self.ci)
        return self.__class__(ci=gci, R=gR)

    def __str__(self):
        """Human readable version"""
        if len(self.R) == 3:
            return "{}.[{},{},{}]".format(self.ci, self.R[0], self.R[1], self.R[2])
        else:
            return "{}.[{},{}]".format(self.ci, self.R[0], self.R[1])

    @staticmethod
    def ClusterSite_representer(dumper, data):
        """Output a ClusterSite"""
        # asdict() returns an OrderedDictionary, so pass through dict()
        # had to rewrite _asdict() for some reason...?
        return dumper.represent_mapping(CLUSTERSITE_YAMLTAG, data._asdict())

    @staticmethod
    def ClusterSite_constructor(loader, node):
        """Construct a ClusterSite from YAML"""
        # ** turns the dictionary into parameters for ClusterSite constructor
        return ClusterSite(**loader.construct_mapping(node, deep=True))


crystal.yaml.add_representer(ClusterSite, ClusterSite.ClusterSite_representer)
crystal.yaml.add_constructor(CLUSTERSITE_YAMLTAG, ClusterSite.ClusterSite_constructor)


class Cluster(object):
    """
    Class to define (arbitrary) cluster interactions. A cluster is defined as
    a set of ClusterSites. We don't implement this using sets, however, because
    we make a choice of a "reference" site in each cluster, which has a lattice
    vector of 0, to account for translational invariance.
    """

    def __init__(self, clustersitelist, NOSORT=False):
        """
        Cluster interaction, from an iterable of ClusterSites
        :param clustersitelist: iterable of ClusterSites
        """
        # this sorting is a *little* hacked together, but as long as we don't have
        # more than 2^32 sites in our crystal, we're good to go:
        def sortkey(cs): return cs.ci[0]*(2**32)+cs.ci[1]
        # first, dump contents of iterable into a list to manipulate:
        lis = [cs for cs in clustersitelist]
        if not NOSORT: lis.sort(key=sortkey)
        R0 = lis[0].R
        self.sites = tuple([cs-R0 for cs in lis])
        self.Norder = len(self.sites)
        # a little mapping of the positions into sets to make equality checking faster,
        # and explicit evaluation of hash function one time using XOR of individual values
        # so that it respects permutations
        self.__center__ = sum(cs.R for cs in self.sites)
        self.__equalitymap__ = {}
        hashcache = 0
        for cs in self.sites:
            shiftpos = self.__shift_pos__(cs)  # our tuple representation of site
            hashcache ^= hash(cs.ci + shiftpos)
            if cs.ci not in self.__equalitymap__:
                self.__equalitymap__[cs.ci] = set([shiftpos])
            else:
                self.__equalitymap__[cs.ci].add(shiftpos)
        self.__hashcache__ = hashcache

    def __shift_pos__(self, cs):
        return tuple(cs.R*self.Norder - self.__center__)

    def __eq__(self, other):
        """
        Test for equality of two clusters. This is a bit trickier than one would expect, since
        clusters are essentially sets where all of the lattice vectors are shifted by the
        center of mass of the sites.
        """
        if not isinstance(other, self.__class__): return False
        if self.Norder != other.Norder: return False
        if self.__equalitymap__.keys() != other.__equalitymap__.keys(): return False
        for k, v in self.__equalitymap__.items():
            if other.__equalitymap__[k] != v: return False
        return True

    def __hash__(self):
        """Return our hash value, precomputed"""
        return self.__hashcache__

    def __contains__(self, elem):
        if elem.ci not in self.__equalitymap__: return False
        return self.__shift_pos__(elem) in self.__equalitymap__[elem.ci]

    def __getitem__(self, item):
        return self.sites[item]

    def __add__(self, other):
        """Add a clustersite to a cluster expansion"""
        if other not in self:
            return self.__class__(self.sites + (other,))
        else:
            return self

    def __radd__(self, other):
        return self.__add__(self, other)

    def __len__(self):
        return self.Norder

    def g(self, crys, g):
        """
        Apply group operation.

        :param crys: crystal
        :param g: group operation (from crys)
        :return g*cluster: corresponding to group operation applied to self
        """
        return self.__class__([cs.g(crys, g) for cs in self.sites])

    def __str__(self):
        """Human readable version"""
        s = "{} order: ".format(self.Norder)
        return s + " ".join([str(cs) for cs in self.sites])
