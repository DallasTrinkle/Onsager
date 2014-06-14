"""
Unit tests for star (and double-star) generation and indexing
"""

__author__ = 'dallas'

#

import unittest
import FCClatt
import KPTmesh
import numpy as np
import stars


class StarTests(unittest.TestCase):
    """Set of tests that our star code is behaving correctly"""

    def setUp(self):
        self.lattice = FCClatt.lattice()
        self.NNvect = FCClatt.NNvect()
        self.invlist = FCClatt.invlist(self.NNvect)
        self.kpt = KPTmesh.KPTmesh(self.lattice, )
        self.groupops = self.kpt.groupops
        self.star = stars.Star(self.NNvect, self.groupops)

    def testStarcount(self):
        """Check that the counts (Npts, Nstars) make sense for FCC, with Nshells = 1"""
        self.star.generate(1)
        self.assertEqual(self.star.Nstars, 1)
        self.assertEqual(self.star.Npts, np.shape(self.NNvect)[0])

    def testStarmembers(self):
        """Are the members correct?"""
        self.star.generate(1)
        pass
