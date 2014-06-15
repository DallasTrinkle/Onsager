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
        # 110
        self.assertEqual(self.star.Nstars, 1)
        self.assertEqual(self.star.Npts, np.shape(self.NNvect)[0])
        self.assertEqual(self.star.Npts, sum([len(s) for s in self.star.stars]))
        self.star.generate(2)
        # 110, 200, 211, 220
        self.assertEqual(self.star.Nstars, 4)
        self.assertEqual(self.star.Npts, sum([len(s) for s in self.star.stars]))
        for s in self.star.stars:
            x = s[0]
            num = (2 ** (3 - list(x).count(0)))
            if x[0] != x[1] and x[1] != x[2]:
                num *= 6
            elif x[0] != x[1] or x[1] != x[2]:
                num *= 3
            self.assertEqual(num, len(s))
        self.star.generate(3)
        # 110, 200, 211, 220, 310, 321, 330, 222
        self.assertEqual(self.star.Nstars, 8)
        self.assertEqual(self.star.Npts, sum([len(s) for s in self.star.stars]))
        for s in self.star.stars:
            x = s[0]
            num = (2 ** (3 - list(x).count(0)))
            if x[0] != x[1] and x[1] != x[2]:
                num *= 6
            elif x[0] != x[1] or x[1] != x[2]:
                num *= 3
            self.assertEqual(num, len(s))


    def testStarmembers(self):
        """Are the members correct?"""
        self.star.generate(1)
        s = self.star.stars[0]
        for v in self.NNvect:
            self.assertTrue(any(all(abs(v-v1)<1e-8) for v1 in s))

