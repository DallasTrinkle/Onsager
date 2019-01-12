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
