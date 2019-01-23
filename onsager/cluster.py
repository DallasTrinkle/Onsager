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

    The flag transition dictates whether the cluster is a transition state cluster
    or not. The difference with a transition state cluster is that the first two
    states in the cluster are the initial and final states of the transition, while
    all of the remaining parts are the cluster.
    """

    def __init__(self, clustersitelist, transition=False, NOSORT=False):
        """
        Cluster interaction, from an iterable of ClusterSites
        :param clustersitelist: iterable of ClusterSites
        :param transition: True if a transition state cluster; cl[0] = initial, cl[1] = final
        """
        # this sorting is a *little* hacked together, but as long as we don't have
        # more than 2^32 sites in our crystal, we're good to go:
        def sortkey(cs): return cs.ci[0]*(2**32)+cs.ci[1]
        # first, dump contents of iterable into a list to manipulate:
        lis = [cs for cs in clustersitelist]
        if not NOSORT:
            if not transition:
                lis.sort(key=sortkey)
            else:
                lis = lis[0:2] + sorted(lis[2:], key=sortkey)
        R0 = lis[0].R
        self.sites = tuple([cs-R0 for cs in lis])
        self.Norder = len(self.sites)
        if transition:
            self.Norder -= 2
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
        self.__transition__ = transition

    def __shift_pos__(self, cs):
        return tuple(cs.R*len(self.sites) - self.__center__)

    def __eq__(self, other):
        """
        Test for equality of two clusters. This is a bit trickier than one would expect, since
        clusters are essentially sets where all of the lattice vectors are shifted by the
        center of mass of the sites.
        """
        if not isinstance(other, self.__class__): return False
        if self.__transition__ != other.__transition__: return False
        if self.Norder != other.Norder: return False
        if self.__equalitymap__.keys() != other.__equalitymap__.keys(): return False
        for k, v in self.__equalitymap__.items():
            if other.__equalitymap__[k] != v: return False
        if self.__transition__:
            TSself, TSother = self.transitionstate(), other.transitionstate()
            if TSself != TSother:
                R0 = TSother[1].R
                if TSself != (TSother[1] - R0, TSother[0] - R0):
                    return False
        return True

    def __hash__(self):
        """Return our hash value, precomputed"""
        return self.__hashcache__

    def __contains__(self, elem):
        if elem.ci not in self.__equalitymap__: return False
        return self.__shift_pos__(elem) in self.__equalitymap__[elem.ci]

    def __getitem__(self, item):
        if self.__transition__:
            return self.sites[item-2]
        else:
            return self.sites[item]

    def transitionstate(self):
        if not self.__transition__: return None
        return self.sites[0], self.sites[1]

    def __add__(self, other):
        """Add a clustersite to a cluster expansion"""
        if other not in self:
            return self.__class__(self.sites + (other,), transition=self.__transition__)
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
        return self.__class__([cs.g(crys, g) for cs in self.sites], transition=self.__transition__)

    def __str__(self):
        """Human readable version"""
        s = "{} order: ".format(self.Norder)
        if self.__transition__:
            s += str(self.sites[0]) + " -> " + str(self.sites[1]) + " : "
            s += " ".join([str(cs) for cs in self.sites[2:]])
        else:
            s += " ".join([str(cs) for cs in self.sites])
        return s


def makeclusters(crys, cutoff, maxorder, exclude=()):
    """
    Function to make clusters up to a maximum order involving all sites within a cutoff
    distance. We can exclude certain chemistries; default is to use all.

    :param crys: crystal to construct our clusters for
    :param cutoff: distance between sites; all sites in a cluster must
      have this mutual distance
    :param maxorder: maximum order of our clusters
    :param exclude: list of chemistries to exclude
    :return clusterexp: list of sets of clusters
    """
    sitelist = [ci for ci in crys.atomindices if ci not in exclude]
    # We construct our clusters in increasing order for maximum efficiency
    # 1st order (sites) is slightly different than the rest.
    clusterexp = []
    clusters = set()
    for ci in sitelist:
        # single sites:
        cl = Cluster([ClusterSite(ci, np.zeros(crys.dim, dtype=int))])
        if cl not in clusters:
            clset = set([cl.g(crys, g) for g in crys.G])
            clusterexp.append(clset)
            clusters.update(clset)
    if maxorder < 2:
        return clusterexp
    # now, we can proceed to higher and higher orders...
    # first, make lists of all our pairs within a given nn distance
    # we could modify this to use different cutoff between different chemistries...
    r2 = cutoff * cutoff
    nmax = [int(np.round(np.sqrt(crys.metric[i, i]))) + 1
            for i in range(crys.dim)]
    nranges = [range(-n, n+1) for n in nmax]
    supervect = [np.array(ntup) for ntup in itertools.product(*nranges)]
    nndict = {}
    for ci0 in sitelist:
        u0 = crys.basis[ci0[0]][ci0[1]]
        nnset = set()
        for ci1 in sitelist:
            u1 = crys.basis[ci1[0]][ci1[1]]
            du = u1 - u0
            for R in supervect:
                dx = crys.unit2cart(R, du)
                if 0 < np.dot(dx, dx) < r2:
                    nnset.add(ClusterSite(ci1, R))
        nndict[ci0] = nnset
    for K in range(maxorder-1):
        # we build based on our lower order clusters:
        prevclusters, clusters = clusters, set()
        for clprev in prevclusters:
            for neigh in nndict[clprev[0].ci]:
                # if this neighbor is already in our cluster, move on
                if neigh in clprev: continue
                # now check that all of the sites in the cluster are also neighbors:
                neighlist = nndict[neigh.ci]
                R0 = neigh.R
                if all(ClusterSite(cl.ci, cl.R-R0) in neighlist for cl in clprev):
                    # new cluster!
                    clnew = clprev + neigh
                    if clnew not in clusters:
                        clset = set([clnew.g(crys, g) for g in crys.G])
                        clusterexp.append(clset)
                        clusters.update(clset)
    return clusterexp
