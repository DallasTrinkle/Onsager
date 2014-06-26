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
        self.groupops = KPTmesh.KPTmesh(self.lattice).groupops
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
        self.groupops = KPTmesh.KPTmesh(self.lattice).groupops
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
        self.groupops = KPTmesh.KPTmesh(self.lattice).groupops
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
        self.groupops = KPTmesh.KPTmesh(self.lattice).groupops
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
        self.groupops = KPTmesh.KPTmesh(self.lattice).groupops
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
                    Rrot = np.dot(g, R)
                    vrot = np.dot(g, v)
                    for R1, v1 in zip(s, vec):
                        if (abs(R1 - Rrot) < 1e-8).all():
                            self.assertTrue((abs(v1 - vrot) < 1e-8).all())

    def testStarVectorConsistent(self):
        """Do the star vectors obey the definition?"""
        self.StarVectorConsistent(1)

    def testStarVectorCount(self):
        """Does our star vector count make any sense?"""
        self.star.generate(1)
        self.starvec.generate(self.star)
        self.assertEqual(self.starvec.Nstarvects, 3)

    def testStarVectorOuterProduct(self):
        """Do we generate the correct outer products for our star-vectors?"""
        self.star.generate(1)
        self.starvec.generate(self.star)
        for outer in self.starvec.outer:
            self.assertAlmostEqual(np.trace(outer), 1)
            # should also be symmetric:
            for g in self.groupops:
                g_out_gT = np.dot(g, np.dot(outer, g.T))
                self.assertTrue((abs(outer - g_out_gT) < 1e-8).all())


class StarVectorFCCTests(StarVectorTests):
    """Set of tests that our StarVector class is behaving correctly, for FCC"""
    def setUp(self):
        self.lattice = FCClatt.lattice()
        self.NNvect = FCClatt.NNvect()
        self.groupops = KPTmesh.KPTmesh(self.lattice).groupops
        self.star = stars.Star(self.NNvect, self.groupops)
        self.starvec = stars.StarVector(self.star)

    def testStarVectorCount(self):
        """Does our star vector count make any sense?"""
        self.star.generate(2)
        self.starvec.generate(self.star)
        # nn + nn = 4 stars, and that should make 5 star-vectors!
        self.assertEqual(self.starvec.Nstarvects, 5)

    def testStarVectorConsistent(self):
        """Do the star vectors obey the definition?"""
        self.StarVectorConsistent(2)

    def testStarVectorOuterProductMore(self):
        """Do we generate the correct outer products for our star-vectors?"""
        self.star.generate(2)
        self.starvec.generate(self.star)
        # with cubic symmetry, these all have to equal 1/3 * identity
        testouter = 1./3.*np.eye(3)
        for outer in self.starvec.outer:
            self.assertTrue((abs(outer - testouter) < 1e-8).all())

import GFcalc


class StarVectorGFlinearTests(unittest.TestCase):
    """Set of tests that make sure we can construct the GF matrix as a linear combination"""
    def setUp(self):
        self.lattice = np.array([[3, 0, 0],
                                 [0, 2, 0],
                                 [0, 0, 1]])
        self.NNvect = np.array([[3, 0, 0], [-3, 0, 0],
                                [0, 2, 0], [0, -2, 0],
                                [0, 0, 1], [0, 0, -1]])
        self.groupops = KPTmesh.KPTmesh(self.lattice).groupops
        self.star = stars.Star(self.NNvect, self.groupops)
        self.star2 = stars.Star(self.NNvect, self.groupops)
        self.starvec = stars.StarVector(self.star)
        self.rates = np.array((3., 3., 2., 2., 1., 1.))
        self.GF = GFcalc.GFcalc(self.lattice, self.NNvect, self.rates)

    def ConstructGF(self, nshells):
        self.star.generate(nshells)
        self.star2.generate(2*nshells)
        self.starvec.generate(self.star)
        GFexpand = self.starvec.GFexpansion(self.star2)
        self.assertEqual(np.shape(GFexpand),
                         (self.starvec.Nstarvects, self.starvec.Nstarvects, self.star2.Nstars + 1))
        gexpand = np.zeros(self.star2.Nstars + 1)
        gexpand[0] = self.GF.GF(np.zeros(3))
        for i in xrange(self.star2.Nstars):
            gexpand[i + 1] = self.GF.GF(self.star2.stars[i][0])
        for i in xrange(self.starvec.Nstarvects):
            for j in xrange(self.starvec.Nstarvects):
                # test the construction
                self.assertAlmostEqual(sum(GFexpand[i, j, :]), 0)
                g = 0
                for Ri, vi in zip(self.starvec.starvecpos[i], self.starvec.starvecvec[i]):
                    for Rj, vj in zip(self.starvec.starvecpos[j], self.starvec.starvecvec[j]):
                        g += np.dot(vi, vj)*self.GF.GF(Ri - Rj)
                self.assertAlmostEqual(g, np.dot(GFexpand[i, j, :], gexpand))
        # print(np.dot(GFexpand, gexpand))

    def testConstructGF(self):
        """Test the construction of the GF using double-nn shell"""
        self.ConstructGF(2)

class StarVectorGFFCClinearTests(StarVectorGFlinearTests):
    """Set of tests that make sure we can construct the GF matrix as a linear combination for FCC"""
    def setUp(self):
        self.lattice = FCClatt.lattice()
        self.NNvect = FCClatt.NNvect()
        self.groupops = KPTmesh.KPTmesh(self.lattice).groupops
        self.star = stars.Star(self.NNvect, self.groupops)
        self.star2 = stars.Star(self.NNvect, self.groupops)
        self.starvec = stars.StarVector(self.star)
        self.rates = np.array((1./12., ) * 12)
        self.GF = GFcalc.GFcalc(self.lattice, self.NNvect, self.rates)

    def testConstructGF(self):
        """Test the construction of the GF using double-nn shell"""
        self.ConstructGF(2)


class StarVectorOmegalinearTests(unittest.TestCase):
    """Set of tests for our expansion of omega_1 in double-stars"""
    def setUp(self):
        self.lattice = np.array([[3, 0, 0],
                                 [0, 2, 0],
                                 [0, 0, 1]])
        self.NNvect = np.array([[3, 0, 0], [-3, 0, 0],
                                [0, 2, 0], [0, -2, 0],
                                [0, 0, 1], [0, 0, -1]])
        self.groupops = KPTmesh.KPTmesh(self.lattice).groupops
        self.star = stars.Star(self.NNvect, self.groupops)
        self.dstar = stars.DoubleStar()
        self.starvec = stars.StarVector()
        self.rates = np.array((3., 3., 2., 2., 1., 1.))

    def testConstructOmega1(self):
        self.star.generate(2) # we need at least 2nd nn to even have double-stars to worry about...
        self.dstar.generate(self.star)
        self.starvec.generate(self.star)
        rate1expand = self.starvec.rate1expansion(self.dstar)
        self.assertEqual(np.shape(rate1expand),
                         (self.starvec.Nstarvects, self.starvec.Nstarvects, self.dstar.Ndstars))
        om1expand = np.zeros(self.dstar.Ndstars)
        for nd, ds in enumerate(self.dstar.dstars):
            pair = ds[0]
            dv = self.star.pts[pair[0]]-self.star.pts[pair[1]]
            for vec, rate in zip(self.NNvect, self.rates):
                if all(abs(dv - vec) < 1e-8):
                    om1expand[nd] = rate
                    break
        # print om1expand
        for i in xrange(self.starvec.Nstarvects):
            for j in xrange(self.starvec.Nstarvects):
                # test the construction
                om1 = 0
                for Ri, vi in zip(self.starvec.starvecpos[i], self.starvec.starvecvec[i]):
                    for Rj, vj in zip(self.starvec.starvecpos[j], self.starvec.starvecvec[j]):
                        dv = Ri - Rj
                        for vec, rate in zip(self.NNvect, self.rates):
                            if all(abs(dv - vec) < 1e-8):
                                om1 += np.dot(vi, vj) * rate
                                break
                self.assertAlmostEqual(om1, np.dot(rate1expand[i, j, :], om1expand))
        # print(np.dot(rateexpand, om1expand))


class StarVectorFCCOmegalinearTests(StarVectorOmegalinearTests):
    """Set of tests for our expansion of omega_1 in double-stars for FCC"""
    def setUp(self):
        self.lattice = FCClatt.lattice()
        self.NNvect = FCClatt.NNvect()
        self.groupops = KPTmesh.KPTmesh(self.lattice).groupops
        self.star = stars.Star(self.NNvect, self.groupops)
        self.dstar = stars.DoubleStar()
        self.starvec = stars.StarVector()
        self.rates = np.array((1./12.,) * 12)


class StarVectorOmega2linearTests(unittest.TestCase):
    """Set of tests for our expansion of omega_2 in NN stars"""
    def setUp(self):
        self.lattice = np.array([[3, 0, 0],
                                 [0, 2, 0],
                                 [0, 0, 1]])
        self.NNvect = np.array([[3, 0, 0], [-3, 0, 0],
                                [0, 2, 0], [0, -2, 0],
                                [0, 0, 1], [0, 0, -1]])
        self.groupops = KPTmesh.KPTmesh(self.lattice).groupops
        self.NNstar = stars.Star(self.NNvect, self.groupops)
        self.star = stars.Star(self.NNvect, self.groupops)
        self.starvec = stars.StarVector()
        self.rates = np.array((3., 3., 2., 2., 1., 1.))

    def testConstructOmega2(self):
        self.NNstar.generate(1) # we need the NN set of stars for NN jumps
        # construct the set of rates corresponding to the unique stars:
        om2expand = np.zeros(self.NNstar.Nstars)
        for vec, rate in zip(self.NNvect, self.rates):
            om2expand[self.NNstar.starindex(vec)] = rate
        self.star.generate(2) # go ahead and make a "large" set of stars
        self.starvec.generate(self.star)
        rate2expand = self.starvec.rate2expansion(self.NNstar)
        self.assertEqual(np.shape(rate2expand),
                         (self.starvec.Nstarvects, self.starvec.Nstarvects, self.NNstar.Nstars))
        for i in xrange(self.starvec.Nstarvects):
            # test the construction
            om2 = 0
            for Ri, vi in zip(self.starvec.starvecpos[i], self.starvec.starvecvec[i]):
                for vec, rate in zip(self.NNvect, self.rates):
                    if (vec == Ri).all():
                        om2 += -np.dot(vi, vi) * rate
                        break
            self.assertAlmostEqual(om2, np.dot(rate2expand[i, i, :], om2expand))
            for j in xrange(self.starvec.Nstarvects):
                if j != i:
                    for d in xrange(self.NNstar.Nstars):
                        self.assertAlmostEquals(0, rate2expand[i, j, d])
        # print(np.dot(rate2expand, om2expand))


class StarVectorFCCOmega2linearTests(StarVectorOmega2linearTests):
    """Set of tests for our expansion of omega_2 in NN stars for FCC"""
    def setUp(self):
        self.lattice = FCClatt.lattice()
        self.NNvect = FCClatt.NNvect()
        self.groupops = KPTmesh.KPTmesh(self.lattice).groupops
        self.NNstar = stars.Star(self.NNvect, self.groupops)
        self.star = stars.Star(self.NNvect, self.groupops)
        self.starvec = stars.StarVector()
        self.rates = np.array((1./12.,) * 12)


class StarVectorBias2linearTests(unittest.TestCase):
    """Set of tests for our expansion of bias vector (2) in NN stars"""
    def setUp(self):
        self.lattice = np.array([[3, 0, 0],
                                 [0, 2, 0],
                                 [0, 0, 1]])
        self.NNvect = np.array([[3, 0, 0], [-3, 0, 0],
                                [0, 2, 0], [0, -2, 0],
                                [0, 0, 1], [0, 0, -1]])
        self.groupops = KPTmesh.KPTmesh(self.lattice).groupops
        self.NNstar = stars.Star(self.NNvect, self.groupops)
        self.star = stars.Star(self.NNvect, self.groupops)
        self.starvec = stars.StarVector()
        self.rates = np.array((3., 3., 2., 2., 1., 1.))

    def testConstructBias2(self):
        self.NNstar.generate(1) # we need the NN set of stars for NN jumps
        # construct the set of rates corresponding to the unique stars:
        om2expand = np.zeros(self.NNstar.Nstars)
        for vec, rate in zip(self.NNvect, self.rates):
            om2expand[self.NNstar.starindex(vec)] = rate
        self.star.generate(2) # go ahead and make a "large" set of stars
        self.starvec.generate(self.star)
        bias2expand = self.starvec.bias2expansion(self.NNstar)
        self.assertEqual(np.shape(bias2expand),
                         (self.starvec.Nstarvects, self.NNstar.Nstars))
        biasvec = np.zeros((self.star.Npts, 3)) # bias vector
        for i, pt in enumerate(self.star.pts):
            for vec, rate in zip(self.NNvect, self.rates):
                if (vec == pt).all():
                    biasvec[i, :] += vec*rate
        # construct the same bias vector using our expansion
        biasveccomp = np.zeros((self.star.Npts, 3))
        for om2, svpos, svvec in zip(np.dot(bias2expand, om2expand),
                                     self.starvec.starvecpos,
                                     self.starvec.starvecvec):
            # test the construction
            for Ri, vi in zip(svpos, svvec):
                biasveccomp[self.star.pointindex(Ri), :] = om2*vi
        for i in xrange(self.star.Npts):
            for d in xrange(3):
                self.assertAlmostEqual(biasvec[i, d], biasveccomp[i, d])
        print(biasvec)
        print(np.dot(bias2expand, om2expand))

