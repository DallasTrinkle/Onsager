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
        return self.__class__(**loader.construct_mapping(node, deep=True))


crystal.yaml.add_representer(ClusterSite, ClusterSite.ClusterSite_representer)
crystal.yaml.add_constructor(CLUSTERSITE_YAMLTAG, ClusterSite.ClusterSite_constructor)


class Cluster(object):
    """
    Class to define (arbitrary) cluster interactions. A cluster is defined as
    a set of ClusterSites. We don't implement this using sets, however, because
    we make a choice of a "reference" site in each cluster, which has a lattice
    vector of 0, to account for translational invariance.
    """

    def __init__(self, iterable):
        """
        Cluster interaction, from an iterable of ClusterSites
        :param iterable: set of ClusterSites
        """
        # this sorting is a *little* hacked together, but as long as we don't have
        # more than 2^32 sites in our crystal, we're good to go:
        def sortkey(cs): return cs[0]*(2**32)+cs[1]
        # first, dump contents of iterable into a list to manipulate:
        lis = [cs for cs in iterable]
        lis.sort(key=sortkey)
        R0 = lis[0].R
        self.sites = tuple([cs-R0 for cs in lis])

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
        s = "{} order: ".format(len(self.sites))
        return s + " ".join([str(cs) for cs in self.sites])
