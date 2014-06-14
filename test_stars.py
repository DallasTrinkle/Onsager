"""
Unit tests for star (and double-star) generation and indexing
"""

__author__ = 'Dallas R. Trinkle'

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
        """Check that the counts (Npts, Nstars) make sense for FCC, with Nshells = 1, 2"""
        self.star.generate(1)
        self.assertEqual(self.star.Nstars, 1)
        self.assertEqual(self.star.Npts, np.shape(self.NNvect)[0])
        self.star.generate(2)
        for s in self.star.stars:
            print s
        self.assertEqual(self.star.Nstars, 4)

    def testStarmembers(self):
        """Are the members correct?"""
        self.star.generate(1)
        s = self.star.stars[0]
        for v in self.NNvect:
            self.assertTrue(any(all(abs(v-v1)<1e-8) for v1 in s))

