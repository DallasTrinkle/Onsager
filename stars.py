"""
Stars module

Classes to generate stars and double stars; a lot of indexing functionality.
"""

# TODO: need to make sure we can read / write stars for optimal functionality (or pickle?)

__author__ = 'Dallas R. Trinkle'

import numpy as np


class Star:
    """
    A class to construct stars, and be able to efficiently index.
    """
    def __init__(self, NNvect, groupops, Nshells=0):
        """
        Initiates a star-generator for a given set of nearest-neighbor vectors
        and group operations. Explicitly *excludes* the trivial star R=0.

        Parameters
        ----------
        NNvect : array [:, 3]
            set of nearest-neighbor vectors; will also be used to generate indexing

        groupops : array [:, 3, 3]
            point group operations, in Cartesian coordinates

        Nshells : integer, optional
            number of shells to generate
        """
        self.NNvect = NNvect
        self.groupops = groupops
        self.generate(Nshells)

    def generate(self, Nshells, threshold=1e-8):
        """
        Construct the actual shells, indexing.

        Parameters
        ----------
        Nshells : integer
            number of shells to generate

        threshold : float, optional
            threshold for determining equality with symmetry

        Notes
        -----
        This code is rather similar to what's in KPTmesh.
        """
        if Nshells == 0:
            self.Nshells = 0
            self.Nstars = 0
            self.Npts = 0
            return
        if Nshells == self.Nshells:
            return
        self.Nshells = Nshells
        # list of all vectors to consider
        vectlist = [v for v in self.NNvect]
        lastshell = list(vectlist)
        for i in range(Nshells-1):
            # add all NNvect to last shell produced, always excluding 0
            lastshell = [v1+v2 for v1 in lastshell for v2 in self.NNvect if not all(abs(v1 + v2) < threshold)]
            vectlist += lastshell
        # now to sort our set of vectors (easiest by magnitude, and then reduce down:
        vectlist.sort(key=lambda x: np.vdot(x, x))
        x2_indices = []
        x2old = np.vdot(vectlist[0], vectlist[0])
        for i, x2 in enumerate([np.vdot(x, x) for x in vectlist]):
            if x2 > (x2old + threshold):
                x2_indices.append(i)
                x2old = x2
        x2_indices.append(len(vectlist))
        # x2_indices now contains a list of indices with the same magnitudes
        self.stars = []
        self.pts = []
        xmin = 0
        for xmax in x2_indices:
            complist_stars = [] # for finding unique stars
            complist_pts = []   # for finding unique points
            for x in vectlist[xmin:xmax]:
                # Q1: is this a unique point?
                if any([np.all(abs(x - xcomp) < threshold) for xcomp in complist_pts]):
                    continue
                complist_pts.append(x)
                # Q2: is this a new rep. for a unique star?
                match = False
                for i, s in enumerate(complist_stars):
                    if self.symmatch(x, s[0], threshold):
                        # update star
                        complist_stars[i].append(x)
                        match = True
                        continue
                if not match:
                    # new symmetry point!
                    complist_stars.append([x])
            self.stars += complist_stars
            self.pts += complist_pts
            xmin=xmax
        self.Nstars = len(self.stars)
        self.Npts = len(self.pts)
        self.index = None

    def generateindices(self):
        """
        Generates the star indices for the set of points, if not already generated
        """
        if self.index != None:
            return
        self.index = np.zeros(self.Npts, dtype=int)
        for i, v in enumerate(self.pts):
            for ns, s in enumerate(self.stars):
                if np.dot(v, v) != np.dot(s[0], s[0]):
                    continue
                if any([all(v == v1) for v1 in s]):
                    self.index[i] = ns

    def starindex(self, x, threshold=1e-8):
        """
        Returns the index for the star to which vector x belongs.

        Parameters
        ----------
        x : array [3]
            vector to check against star

        threshold : float, optional
            threshold for equality

        Returns
        -------
        index corresponding to star; -1 if not found.
        """
        self.generateindices()
        for i, v in enumerate(self.pts):
            if all(abs(x - v) < threshold):
                return self.index[i]
        return -1

    def pointindex(self, x, threshold=1e-8):
        """
        Returns the index corresponding to the point x.

        Parameters
        ----------
        x : array [3]
            vector to check against star

        threshold : float, optional
            threshold for equality

        Returns
        -------
        index corresponding to point; -1 if not found.
        """
        for i, v in enumerate(self.pts):
            if all(abs(x - v) < threshold):
                return i
        return -1

    def symmatch(self, x, xcomp, threshold=1e-8):
        """
        Tells us if x and xcomp are equivalent by a symmetry group

        Parameters
        ----------
        x : array [3]
            vector to be tested
        xcomp : array [3]
            vector to compare
        threshold : double, optional
            threshold to use for "equality"

        Returns
        -------
        True if equivalent by a point group operation, False otherwise
        """
        return any([np.all(abs(x - np.dot(g, xcomp)) < threshold) for g in self.groupops])


class DoubleStar:
    """
    A class to construct double-stars (pairs of sites,
    where each pair is related by a single group op).
    """
    def __init__(self, star=None):
        """
        Initiates a star-generator for a given set of nearest-neighbor vectors
        and group operations. Explicitly *excludes* the trivial star R=0.

        Parameters
        ----------
        star : Star, optional
            all of our input parameters will come from this, if non-empty
        """
        if star != None:
            self.NNvect = star.NNvect
            self.groupops = star.groupops
            if self.star.Nshells > 0:
                self.generate(star)
        self.star = None
        self.Ndstars = 0
        self.Npairs = 0
        self.Npts = 0

    def generate(self, star, threshold=1e-8):
        """
        Construct the actual double-shells, indexing.

        Parameters
        ----------
        star : Star, optional
            all of our input parameters will come from this, if non-empty
        """
        if star.Nshells == 0:
            self.Npairs = 0
            return
        if star == self.star and star.Npts == self.Npts:
            return
        self.star = star
        self.Npts = star.Npts
        self.NNvect = star.NNvect
        self.groupops = star.groupops
        self.pairs = []
        # make the pairs first
        for i1, v1 in enumerate(self.star.pts):
            for dv in self.NNvect:
                v2 = v1 + dv
                if not all(abs(v2) < threshold):
                    self.pairs += (i1, self.star.pointindex(v2))
        self.Npairs = len(self.pairs)
        # now to make the unique sets of pairs (double-stars)

    def symmatch(self, x, xcomp, threshold=1e-8):
        """
        Tells us if x and xcomp are equivalent by a symmetry group

        Parameters
        ----------
        x : array [3]
            vector to be tested
        xcomp : array [3]
            vector to compare
        threshold : double, optional
            threshold to use for "equality"

        Returns
        -------
        True if equivalent by a point group operation, False otherwise
        """
        return any([np.all(abs(x - np.dot(g, xcomp)) < threshold) for g in self.groupops])

