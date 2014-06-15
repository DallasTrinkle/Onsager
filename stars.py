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
            return
        if Nshells == self.Nshells:
            return
        self.Nshells = Nshells
        # list of all vectors to consider
        vectlist = [v for v in self.NNvect]
        lastshell = list(vectlist)
        for i in range(Nshells-1):
            # add all NNvect to last shell produced, always excluding 0
            lastshell = [v1+v2 for v1 in lastshell for v2 in self.NNvect if not all(abs(v1+v2)<threshold)]
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
                if any([np.all(abs(x-xcomp) < threshold) for xcomp in complist_pts]):
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

