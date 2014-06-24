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
    """Set of tests that our star code is behaving correctly for a general materials"""

    def setUp(self):
        self.lattice = np.array([[3, 0, 0],
                                 [0, 2, 0],
                                 [0, 0, 1]])
        self.NNvect = np.array([[3, 0, 0], [-3, 0, 0],
                                [0, 2, 0], [0, -2, 0],
                                [0, 0, 1], [0, 0, -1]])
        self.kpt = KPTmesh.KPTmesh(self.lattice)
        self.groupops = self.kpt.groupops
        self.star = stars.Star(self.NNvect, self.groupops)

    def isclosed(self, s, groupops, threshold=1e-8):
        """
        Evaluate if star s is closed against group operations.

        Parameters
        ----------
        s : list of vectors
            star

        groupops : list (or array) of 3x3 matrices
            all group operations

        threshold : float, optional
            threshold for equality in comparison

        Returns
        -------
        True if every pair of vectors in star are related by a group operation;
        False otherwise
        """
        for v1 in s:
            for v2 in s:
                if not any([all(abs(v1 - np.dot(g, v2)) < threshold) for g in groupops]):
                    return False
        return True


    def testStarConsistent(self):
        """Check that the counts (Npts, Nstars) make sense, with Nshells = 1..4"""
        for n in xrange(1,5):
            self.star.generate(n)
            self.assertEqual(self.star.Npts, sum([len(s) for s in self.star.stars]))
            for s in self.star.stars:
                self.assertTrue(self.isclosed(s, self.groupops))

    def testStarindices(self):
        """Check that our indexing is correct."""
        self.star.generate(4)
        for ns, s in enumerate(self.star.stars):
            for v in s:
                self.assertEqual(ns, self.star.starindex(v))
        self.assertEqual(-1, self.star.starindex(np.zeros(3)))
        for i, v in enumerate(self.star.pts):
            self.assertEqual(i, self.star.pointindex(v))
        self.assertEqual(-1, self.star.pointindex(np.zeros(3)))


class CubicStarTests(StarTests):
    """Set of tests that our star code is behaving correctly for cubic materials"""

    def setUp(self):
        self.lattice = np.array([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
        self.NNvect = np.array([[1, 0, 0], [-1, 0, 0],
                                [0, 1, 0], [0, -1, 0],
                                [0, 0, 1], [0, 0, -1]])
        self.kpt = KPTmesh.KPTmesh(self.lattice)
        self.groupops = self.kpt.groupops
        self.star = stars.Star(self.NNvect, self.groupops)

    def testStarConsistent(self):
        """Check that the counts (Npts, Nstars) make sense for cubic, with Nshells = 1..4"""
        for n in xrange(1,5):
            self.star.generate(n)
            self.assertEqual(self.star.Npts, sum([len(s) for s in self.star.stars]))
            for s in self.star.stars:
                x = s[0]
                num = (2 ** (3 - list(x).count(0)))
                if x[0] != x[1] and x[1] != x[2]:
                    num *= 6
                elif x[0] != x[1] or x[1] != x[2]:
                    num *= 3
                self.assertEqual(num, len(s))
                self.assertTrue(self.isclosed(s, self.groupops))

    def testStarmembers(self):
        """Are the members correct?"""
        self.star.generate(1)
        s = self.star.stars[0]
        for v in self.NNvect:
            self.assertTrue(any(all(abs(v-v1)<1e-8) for v1 in s))


class FCCStarTests(CubicStarTests):
    """Set of tests that our star code is behaving correctly for FCC"""

    def setUp(self):
        self.lattice = FCClatt.lattice()
        self.NNvect = FCClatt.NNvect()
        self.kpt = KPTmesh.KPTmesh(self.lattice)
        self.groupops = self.kpt.groupops
        self.star = stars.Star(self.NNvect, self.groupops)

    def testStarCount(self):
        """Check that the counts (Npts, Nstars) make sense for FCC, with Nshells = 1, 2, 3"""
        # 110
        self.star.generate(1)
        self.assertEqual(self.star.Nstars, 1)
        self.assertEqual(self.star.Npts, np.shape(self.NNvect)[0])

        # 110, 200, 211, 220
        self.star.generate(2)
        self.assertEqual(self.star.Nstars, 4)

        # 110, 200, 211, 220, 310, 321, 330, 222
        self.star.generate(3)
        self.assertEqual(self.star.Nstars, 8)


class DoubleStarTests(unittest.TestCase):
    """Set of tests that our DoubleStar class is behaving correctly."""

    def setUp(self):
        self.lattice = FCClatt.lattice()
        self.NNvect = FCClatt.NNvect()
        self.kpt = KPTmesh.KPTmesh(self.lattice)
        self.groupops = self.kpt.groupops
        self.star = stars.Star(self.NNvect, self.groupops)
        self.dstar = stars.DoubleStar()

    def testDoubleStarGeneration(self):
        """Can we generate a double-star?"""
        self.star.generate(1)
        self.dstar.generate(self.star)
        self.assertTrue(self.dstar.Ndstars > 0)
        self.assertTrue(self.dstar.Npairs > 0)

    def testDoubleStarCount(self):
        """Check that the counts (Npts, Nstars) make sense for FCC, with Nshells = 1, 2"""
        # each of the 12 <110> pairs to 101, 10-1, 011, 01-1 = 4, so should be 48 pairs
        # (which includes "double counting": i->j and j->i)
        # but *all* of those 48 are all equivalent to each other by symmetry: one double-star.
        self.star.generate(1)
        self.dstar.generate(self.star)
        self.assertEqual(self.dstar.Npairs, 48)
        self.assertEqual(self.dstar.Ndstars, 1)
        # Now have four stars (110, 200, 211, 220), so this means
        # 12 <110> pairs to 11 (no 000!); 12*11
        # 6 <200> pairs to 110, 101, 1-10, 10-1; 211, 21-1, 2-11, 2-1-1 = 8; 6*8
        # 24 <211> pairs to 110, 101; 200; 112, 121; 202, 220 = 7; 24*7
        # 12 <220> pairs to 110; 12-1, 121, 21-1, 211 = 5; 12*5
        # unique pairs: (110, 101); (110, 200); (110, 211); (110, 220); (200, 211); (211, 112); (211, 220)
        self.star.generate(2)
        self.dstar.generate(self.star)
        self.assertEqual(self.dstar.Npairs, 12*11 + 6*8 + 24*7 + 12*5)
        # for ds in self.dstar.dstars:
        #     print self.star.pts[ds[0][0]], self.star.pts[ds[0][1]]
        self.assertEqual(self.dstar.Ndstars, 4 + 1 + 2)

    def testPairIndices(self):
        """Check that our pair indexing works correctly for Nshell=1..3"""
        for nshells in xrange(1, 4):
            self.star.generate(nshells)
            self.dstar.generate(self.star)
            for pair in self.dstar.pairs:
                self.assertTrue(pair == self.dstar.pairs[self.dstar.pairindex(pair)])

    def testDoubleStarindices(self):
        """Check that our double-star indexing works correctly for Nshell=1..3"""
        for nshells in xrange(1, 4):
            self.star.generate(nshells)
            self.dstar.generate(self.star)
            for pair in self.dstar.pairs:
                self.assertTrue(any(pair == p for p in self.dstar.dstars[self.dstar.dstarindex(pair)]))

class StarVectorTests(unittest.TestCase):
    """Set of tests that our StarVector class is behaving correctly"""
    def setUp(self):
        self.lattice = np.array([[3, 0, 0],
                                 [0, 2, 0],
                                 [0, 0, 1]])
        self.NNvect = np.array([[3, 0, 0], [-3, 0, 0],
                                [0, 2, 0], [0, -2, 0],
                                [0, 0, 1], [0, 0, -1]])
        self.kpt = KPTmesh.KPTmesh(self.lattice)
        self.groupops = self.kpt.groupops
        self.star = stars.Star(self.NNvect, self.groupops)
        self.starvec = stars.StarVector(self.star)

    def testStarVectorGenerate(self):
        """Can we generate star-vectors that make sense?"""
        self.star.generate(1)
        self.starvec.generate(self.star)
        self.assertTrue(self.starvec.Nstarvects>0)

    def StarVectorConsistent(self, nshells):
        """Do the star vectors obey the definition?"""
        self.star.generate(nshells)
        self.starvec.generate(self.star)
        for s, vec in zip(self.starvec.starvecpos, self.starvec.starvecvec):
            for R, v in zip(s, vec):
                for g in self.groupops:
                    R1 = np.dot(g, R)
                    for m, R2 in enumerate(s):
                        if all(abs(R1 - R2) < 1e-8):
                            self.assertTrue(all(abs(vec[m] - np.dot(g, v)) < 1e-8))

    def TestStarVectorConsistent(self):
        """Do the star vectors obey the definition?"""
        self.StarVectorConsistent(self, 1)

    def testStarVectorCount(self):
        """Does our star vector count make any sense?"""
        self.star.generate(1)
        self.starvec.generate(self.star)
        self.assertEqual(self.starvec.Nstarvects, 3)

class StarVectorFCCTests(StarVectorTests):
    """Set of tests that our StarVector class is behaving correctly, for FCC"""
    def setUp(self):
        self.lattice = FCClatt.lattice()
        self.NNvect = FCClatt.NNvect()
        self.kpt = KPTmesh.KPTmesh(self.lattice)
        self.groupops = self.kpt.groupops
        self.star = stars.Star(self.NNvect, self.groupops)
        self.starvec = stars.StarVector(self.star)

    def testStarVectorCount(self):
        """Does our star vector count make any sense?"""
        self.star.generate(2)
        self.starvec.generate(self.star)
        # nn + nn = 4 stars, and that should make 5 star-vectors!
        self.assertEqual(self.starvec.Nstarvects, 5)

    def TestStarVectorConsistent(self):
        """Do the star vectors obey the definition?"""
        self.StarVectorConsistent(self, 2)

