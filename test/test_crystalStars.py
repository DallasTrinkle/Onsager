"""
Unit tests for star, double-star and vector-star generation and indexing,
rebuilt to use crystal
"""

__author__ = 'Dallas R. Trinkle'

#

import unittest
import onsager.crystal as crystal
import numpy as np
import onsager.crystalStars as stars


# Setup for orthorhombic, simple cubic, and FCC cells:
def setuportho():
    crys = crystal.Crystal(np.array([[3, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=float),
                           [[np.zeros(3)]])
    jumpnetwork = [ [((0,0), np.array([3.,0.,0.])),((0,0), np.array([-3.,0.,0.]))],
                    [((0,0), np.array([0.,2.,0.])),((0,0), np.array([0.,-2.,0.]))],
                    [((0,0), np.array([0.,0.,1.])),((0,0), np.array([0.,0.,-1.]))] ]
    return crys, jumpnetwork

def orthorates():
    return np.array([3., 2., 1.])

def setupcubic():
    crys = crystal.Crystal(np.eye(3), [[np.zeros(3)]])
    jumpnetwork = [ [((0,0), np.array([1.,0.,0.])), ((0,0), np.array([-1.,0.,0.])),
                     ((0,0), np.array([0.,1.,0.])), ((0,0), np.array([0.,-1.,0.])),
                     ((0,0), np.array([0.,0.,1.])), ((0,0), np.array([0.,0.,-1.]))]]
    return crys, jumpnetwork

def cubicrates():
    return np.array([1./6.])

def setupFCC():
    lattice = crystal.Crystal.FCC(1.)
    jumpnetwork = lattice.jumpnetwork(0, np.sqrt(0.5)+0.01)
    return lattice, jumpnetwork

def FCCrates():
    return np.array([1./12.])


class PairStateTests(unittest.TestCase):
    """Tests of the PairState class"""
    def setUp(self):
        self.hcp = crystal.Crystal.HCP(1.0)

    def testZero(self):
        """Is zero consistent?"""
        zero = stars.PairState.zero()
        self.assertTrue(zero.iszero())

    def testAddition(self):
        """Does addition work as expected?"""
        pos1 = self.hcp.pos2cart(np.array([0,0,0]), (0,0))  # first site
        pos2 = self.hcp.pos2cart(np.array([0,0,0]), (0,1))  # second site
        pos1a1 = self.hcp.pos2cart(np.array([1,0,0]), (0,0))  # first + a1
        pos2a1 = self.hcp.pos2cart(np.array([1,0,0]), (0,1))  # second + a1
        pos1a2 = self.hcp.pos2cart(np.array([0,1,0]), (0,0))  # first + a2
        pos2a2 = self.hcp.pos2cart(np.array([0,1,0]), (0,1))  # second + a2
        ps1 = stars.PairState.fromcrys(self.hcp, 0, (0,0), pos1a1 - pos1)
        ps2 = stars.PairState.fromcrys(self.hcp, 0, (0,0), 2*(pos1a1 - pos1))
        self.assertEqual(ps1 + ps1, ps2)
        ps2 = stars.PairState.fromcrys(self.hcp, 0, (1,0), pos1 - pos2)
        with self.assertRaises(ArithmeticError):
            ps1 + ps2
        ps3 = stars.PairState.fromcrys(self.hcp, 0, (1,0), pos1a1 - pos2)
        self.assertEqual(ps2 + ps1, ps3)


class StarTests(unittest.TestCase):
    """Set of tests that our star code is behaving correctly for a general materials"""

    def setUp(self):
        self.crys, self.jumpnetwork = setuportho()

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
        for n in range(1,5):
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

    def assertEqualStars(self, s1, s2):
        """Asserts that two stars are equal."""
        self.assertEqual(s1.Npts, s2.Npts,
                         msg='Number of points in two star sets are not equal: {} != {}'.format(
                             s1.Npts, s2.Npts
                         ))
        self.assertEqual(s1.Nshells, s2.Nshells,
                         msg='Number of shells in two star sets are not equal: {} != {}'.format(
                             s1.Nshells, s2.Nshells
                         ))
        self.assertEqual(s1.Nstars, s2.Nstars,
                         msg='Number of stars in two star sets are not equal: {} != {}'.format(
                             s1.Nstars, s2.Nstars
                         ))
        for s in s1.stars:
            ind = s2.starindex(s[0])
            self.assertNotEqual(ind, -1,
                                msg='Could not find {} from s1 in s2'.format(
                                    s[0]
                                ))
            for R in s:
                self.assertEqual(ind, s2.starindex(R),
                                 msg='Point {} and {} from star in s1 belong to different stars in s2'.format(
                                     s[0], R
                                 ))
        for s in s2.stars:
            ind = s1.starindex(s[0])
            self.assertNotEqual(ind, -1,
                                msg='Could not find {} from s2 in s1'.format(
                                    s[0]
                                ))
            for R in s:
                self.assertEqual(ind, s1.starindex(R),
                                 msg='Point {} and {} from star in s2 belong to different stars in s1'.format(
                                     s[0], R
                                 ))

    def testStarCombine(self):
        """Check that we can combine two stars and get what we expect."""
        s1 = stars.StarSet(self.NNvect, self.groupops)
        s2 = stars.StarSet(self.NNvect, self.groupops)
        s3 = stars.StarSet(self.NNvect, self.groupops)
        s4 = stars.StarSet(self.NNvect, self.groupops)
        s1.generate(1)
        s2.generate(1)
        s3.combine(s1, s2)
        s4.generate(2)
        # s3 = s1 + s2, should equal s4
        self.assertEqualStars(s1, s2)
        self.assertEqualStars(s3, s4)


class CubicStarTests(StarTests):
    """Set of tests that our star code is behaving correctly for cubic materials"""

    def setUp(self):
        self.lattice, self.NNvect, self.groupops, self.star = setupcubic()

    def testStarConsistent(self):
        """Check that the counts (Npts, Nstars) make sense for cubic, with Nshells = 1..4"""
        for n in range(1,5):
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
        self.lattice, self.NNvect, self.groupops, self.star = setupFCC()

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
        self.lattice, self.NNvect, self.groupops, self.star = setupFCC()
        self.dstar = stars.DoubleStarSet()

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
        for nshells in range(1, 4):
            self.star.generate(nshells)
            self.dstar.generate(self.star)
            for pair in self.dstar.pairs:
                self.assertTrue(pair == self.dstar.pairs[self.dstar.pairindex(pair)])

    def testDoubleStarindices(self):
        """Check that our double-star indexing works correctly for Nshell=1..3"""
        for nshells in range(1, 4):
            self.star.generate(nshells)
            self.dstar.generate(self.star)
            for pair in self.dstar.pairs:
                self.assertTrue(any(pair == p for p in self.dstar.dstars[self.dstar.dstarindex(pair)]))

class VectorStarTests(unittest.TestCase):
    """Set of tests that our VectorStar class is behaving correctly"""
    def setUp(self):
        self.lattice, self.NNvect, self.groupops, self.star = setuportho()
        self.vecstar = stars.VectorStarSet(self.star)

    def testVectorStarGenerate(self):
        """Can we generate star-vectors that make sense?"""
        self.star.generate(1)
        self.vecstar.generate(self.star)
        self.assertTrue(self.vecstar.Nvstars>0)

    def VectorStarConsistent(self, nshells):
        """Do the star vectors obey the definition?"""
        self.star.generate(nshells)
        self.vecstar.generate(self.star)
        for s, vec in zip(self.vecstar.vecpos, self.vecstar.vecvec):
            for R, v in zip(s, vec):
                for g in self.groupops:
                    Rrot = np.dot(g, R)
                    vrot = np.dot(g, v)
                    for R1, v1 in zip(s, vec):
                        if (abs(R1 - Rrot) < 1e-8).all():
                            self.assertTrue((abs(v1 - vrot) < 1e-8).all())

    def VectorStarOrthonormal(self, nshells):
        """Are the star vectors orthonormal?"""
        self.star.generate(nshells)
        self.vecstar.generate(self.star)
        for s1, v1 in zip(self.vecstar.vecpos, self.vecstar.vecvec):
            for s2, v2 in zip(self.vecstar.vecpos, self.vecstar.vecvec):
                if (s1[0] == s2[0]).all():
                    dp = 0
                    for vv1, vv2 in zip(v1, v2):
                        dp += np.dot(vv1, vv2)
                    if (v1[0] == v2[0]).all():
                        self.assertAlmostEqual(1., dp,
                                               msg='Failed normality for {}/{} and {}/{}'.format(
                                                   s1[0], v1[0], s2[0], v2[0]
                                               ))
                    else:
                        self.assertAlmostEqual(0., dp,
                                               msg='Failed orthogonality for {}/{} and {}/{}'.format(
                                                   s1[0], v1[0], s2[0], v2[0]
                                               ))

    def testVectorStarConsistent(self):
        """Do the star vectors obey the definition?"""
        self.VectorStarConsistent(2)

    def testVectorStarOrthonormal(self):
        self.VectorStarOrthonormal(2)

    def testVectorStarCount(self):
        """Does our star vector count make any sense?"""
        self.star.generate(1)
        self.vecstar.generate(self.star)
        self.assertEqual(self.vecstar.Nvstars, 3)

    def testVectorStarOuterProduct(self):
        """Do we generate the correct outer products for our star-vectors (symmetry checks)?"""
        self.star.generate(1)
        self.vecstar.generate(self.star)
        self.assertEqual(np.shape(self.vecstar.outer), (3, 3, self.vecstar.Nvstars, self.vecstar.Nvstars))
        # check our diagonal blocks first:
        for outer in [self.vecstar.outer[:, :, i, i]
                      for i in range(self.vecstar.Nvstars)]:
            self.assertAlmostEqual(np.trace(outer), 1)
            # should also be symmetric:
            for g in self.groupops:
                g_out_gT = np.dot(g, np.dot(outer, g.T))
                self.assertTrue((abs(outer - g_out_gT) < 1e-8).all())
        # off-diagonal terms now
        for outer in [self.vecstar.outer[:, :, i, j]
                      for i in range(self.vecstar.Nvstars)
                      for j in range(self.vecstar.Nvstars)
                      if i != j]:
            self.assertAlmostEqual(np.trace(outer), 0)
            # should also be symmetric:
            for g in self.groupops:
                g_out_gT = np.dot(g, np.dot(outer, g.T))
                self.assertTrue((abs(outer - g_out_gT) < 1e-8).all())
        for i, (sRv0, svv0) in enumerate(zip(self.vecstar.vecpos, self.vecstar.vecvec)):
            for j, (sRv1, svv1) in enumerate(zip(self.vecstar.vecpos, self.vecstar.vecvec)):
                testouter = np.zeros((3, 3))
                if (sRv0[0] == sRv1[0]).all():
                    # we have the same underlying star to work with, so...
                    for v0, v1 in zip(svv0, svv1):
                        testouter += np.outer(v0, v1)
                self.assertTrue((abs(self.vecstar.outer[:, :, i, j] - testouter) < 1e-8).all(),
                                msg='Failed for vector stars {} and {}:\n{} !=\n{}'.format(
                                    i, j, outer, testouter
                                ))


class VectorStarFCCTests(VectorStarTests):
    """Set of tests that our VectorStar class is behaving correctly, for FCC"""
    def setUp(self):
        self.lattice, self.NNvect, self.groupops, self.star = setupFCC()
        self.vecstar = stars.VectorStarSet(self.star)

    def testVectorStarCount(self):
        """Does our star vector count make any sense?"""
        self.star.generate(2)
        self.vecstar.generate(self.star)
        # nn + nn = 4 stars, and that should make 5 star-vectors!
        self.assertEqual(self.vecstar.Nvstars, 5)

    def testVectorStarConsistent(self):
        """Do the star vectors obey the definition?"""
        self.VectorStarConsistent(2)

    def testVectorStarOuterProductMore(self):
        """Do we generate the correct outer products for our star-vectors?"""
        self.star.generate(2)
        self.vecstar.generate(self.star)
        # with cubic symmetry, these all have to equal 1/3 * identity, and
        # with a diagonal matrix
        testouter = 1./3.*np.eye(3)
        for outer in [self.vecstar.outer[:, :, i, i]
                      for i in range(self.vecstar.Nvstars)]:
            self.assertTrue((abs(outer - testouter) < 1e-8).all())
        for outer in [self.vecstar.outer[:, :, i, j]
                      for i in range(self.vecstar.Nvstars)
                      for j in range(self.vecstar.Nvstars)
                      if i != j]:
            self.assertTrue((abs(outer) < 1e-8).all())

import onsager.GFcalc as GFcalc


class VectorStarGFlinearTests(unittest.TestCase):
    """Set of tests that make sure we can construct the GF matrix as a linear combination"""
    def setUp(self):
        self.lattice, self.NNvect, self.groupops, self.star = setuportho()
        self.star2 = stars.StarSet(self.NNvect, self.groupops)
        self.vecstar = stars.VectorStarSet(self.star)
        self.rates = orthorates()
        self.GF = GFcalc.GFcalc(self.lattice, self.NNvect, self.rates)

    def ConstructGF(self, nshells):
        self.star.generate(nshells)
        self.star2.generate(2*nshells)
        self.vecstar.generate(self.star)
        GFexpand = self.vecstar.GFexpansion(self.star2)
        self.assertEqual(np.shape(GFexpand),
                         (self.vecstar.Nvstars, self.vecstar.Nvstars, self.star2.Nstars + 1))
        gexpand = np.zeros(self.star2.Nstars + 1)
        gexpand[0] = self.GF.GF(np.zeros(3))
        for i in range(self.star2.Nstars):
            gexpand[i + 1] = self.GF.GF(self.star2.stars[i][0])
        for i in range(self.vecstar.Nvstars):
            for j in range(self.vecstar.Nvstars):
                # test the construction
                self.assertAlmostEqual(sum(GFexpand[i, j, :]), 0)
                g = 0
                for Ri, vi in zip(self.vecstar.vecpos[i], self.vecstar.vecvec[i]):
                    for Rj, vj in zip(self.vecstar.vecpos[j], self.vecstar.vecvec[j]):
                        g += np.dot(vi, vj)*self.GF.GF(Ri - Rj)
                self.assertAlmostEqual(g, np.dot(GFexpand[i, j, :], gexpand))
        # print(np.dot(GFexpand, gexpand))

    def testConstructGF(self):
        """Test the construction of the GF using double-nn shell"""
        self.ConstructGF(2)

class VectorStarGFFCClinearTests(VectorStarGFlinearTests):
    """Set of tests that make sure we can construct the GF matrix as a linear combination for FCC"""
    def setUp(self):
        self.lattice, self.NNvect, self.groupops, self.star = setupFCC()
        self.star2 = stars.StarSet(self.NNvect, self.groupops)
        self.vecstar = stars.VectorStarSet(self.star)
        self.rates = FCCrates()
        self.GF = GFcalc.GFcalc(self.lattice, self.NNvect, self.rates)

    def testConstructGF(self):
        """Test the construction of the GF using double-nn shell"""
        self.ConstructGF(2)


class VectorStarOmega0Tests(unittest.TestCase):
    """Set of tests for our expansion of omega_0 in NN vectors"""
    def setUp(self):
        self.lattice, self.NNvect, self.groupops, self.star = setuportho()
        self.NNstar = stars.StarSet(self.NNvect, self.groupops)
        self.vecstar = stars.VectorStarSet()
        self.rates = orthorates()

    def testConstructOmega0(self):
        self.star.generate(2) # we need at least 2nd nn to even have double-stars to worry about...
        self.NNstar.generate(1) # just nearest-neighbor stars
        self.vecstar.generate(self.star)
        rate0expand = self.vecstar.rate0expansion(self.NNstar)
        self.assertEqual(np.shape(rate0expand),
                         (self.vecstar.Nvstars, self.vecstar.Nvstars, self.NNstar.Nstars))
        om0expand = np.zeros(self.NNstar.Nstars)
        for vec, rate in zip(self.NNvect, self.rates):
            om0expand[self.NNstar.starindex(vec)] = rate

        om0matrix = -sum(self.rates)*np.eye(self.star.Npts)
        for i, pt in enumerate(self.star.pts):
            for vec, rate in zip(self.NNvect, self.rates):
                j = self.star.pointindex(pt + vec)
                if j >= 0:
                    om0matrix[i, j] = rate
        # now, we need to convert that omega0 matrix into the "folded down"
        for i, (sRv0, svv0) in enumerate(zip(self.vecstar.vecpos, self.vecstar.vecvec)):
            for j, (sRv1, svv1) in enumerate(zip(self.vecstar.vecpos, self.vecstar.vecvec)):
                om0_sv = 0
                for R0, v0 in zip(sRv0, svv0):
                    for R1, v1 in zip(sRv1, svv1):
                        om0_sv += np.dot(v0, v1)*\
                                  om0matrix[self.star.pointindex(R0),
                                            self.star.pointindex(R1)]
                om0_sv_comp = np.dot(rate0expand[i, j], om0expand)
                self.assertAlmostEqual(om0_sv, om0_sv_comp,
                                       msg='Failed to match {}, {}: {} != {}'.format(
                                           i, j, om0_sv, om0_sv_comp)
                                       )


class VectorStarFCCOmega0Tests(VectorStarOmega0Tests):
    """Set of tests for our expansion of omega_0 in NN vect for FCC"""
    def setUp(self):
        self.lattice, self.NNvect, self.groupops, self.star = setupFCC()
        self.NNstar = stars.StarSet(self.NNvect, self.groupops)
        self.vecstar = stars.VectorStarSet()
        self.rates = FCCrates()


class VectorStarOmegalinearTests(unittest.TestCase):
    """Set of tests for our expansion of omega_1 in double-stars"""
    def setUp(self):
        self.lattice, self.NNvect, self.groupops, self.star = setuportho()
        self.dstar = stars.DoubleStarSet()
        self.vecstar = stars.VectorStarSet()
        self.rates = orthorates()

    def testConstructOmega1(self):
        self.star.generate(2) # we need at least 2nd nn to even have double-stars to worry about...
        self.dstar.generate(self.star)
        self.vecstar.generate(self.star)
        rate1expand = self.vecstar.rate1expansion(self.dstar)
        self.assertEqual(np.shape(rate1expand),
                         (self.vecstar.Nvstars, self.vecstar.Nvstars, self.dstar.Ndstars))
        om1expand = np.zeros(self.dstar.Ndstars)
        for nd, ds in enumerate(self.dstar.dstars):
            pair = ds[0]
            dv = self.star.pts[pair[0]]-self.star.pts[pair[1]]
            for vec, rate in zip(self.NNvect, self.rates):
                if all(abs(dv - vec) < 1e-8):
                    om1expand[nd] = rate
                    break
        # print om1expand
        for i in range(self.vecstar.Nvstars):
            for j in range(self.vecstar.Nvstars):
                # test the construction
                om1 = 0
                for Ri, vi in zip(self.vecstar.vecpos[i], self.vecstar.vecvec[i]):
                    for Rj, vj in zip(self.vecstar.vecpos[j], self.vecstar.vecvec[j]):
                        dv = Ri - Rj
                        for vec, rate in zip(self.NNvect, self.rates):
                            if all(abs(dv - vec) < 1e-8):
                                om1 += np.dot(vi, vj) * rate
                                break
                self.assertAlmostEqual(om1, np.dot(rate1expand[i, j, :], om1expand))
        # print(np.dot(rateexpand, om1expand))


class VectorStarFCCOmegalinearTests(VectorStarOmegalinearTests):
    """Set of tests for our expansion of omega_1 in double-stars for FCC"""
    def setUp(self):
        self.lattice, self.NNvect, self.groupops, self.star = setupFCC()
        self.dstar = stars.DoubleStarSet()
        self.vecstar = stars.VectorStarSet()
        self.rates = FCCrates()


class VectorStarOmega2linearTests(unittest.TestCase):
    """Set of tests for our expansion of omega_2 in NN stars"""
    def setUp(self):
        self.lattice, self.NNvect, self.groupops, self.star = setuportho()
        self.NNstar = stars.StarSet(self.NNvect, self.groupops)
        self.vecstar = stars.VectorStarSet()
        self.rates = orthorates()

    def testConstructOmega2(self):
        self.NNstar.generate(1) # we need the NN set of stars for NN jumps
        # construct the set of rates corresponding to the unique stars:
        om2expand = np.zeros(self.NNstar.Nstars)
        for vec, rate in zip(self.NNvect, self.rates):
            om2expand[self.NNstar.starindex(vec)] = rate
        self.star.generate(2) # go ahead and make a "large" set of stars
        self.vecstar.generate(self.star)
        rate2expand = self.vecstar.rate2expansion(self.NNstar)
        self.assertEqual(np.shape(rate2expand),
                         (self.vecstar.Nvstars, self.vecstar.Nvstars, self.NNstar.Nstars))
        for i in range(self.vecstar.Nvstars):
            # test the construction
            om2 = 0
            for Ri, vi in zip(self.vecstar.vecpos[i], self.vecstar.vecvec[i]):
                for vec, rate in zip(self.NNvect, self.rates):
                    if (vec == Ri).all():
                        # includes the factor of 2 to account for on-site terms in matrix.
                        om2 += -2. * np.dot(vi, vi) * rate
                        break
            self.assertAlmostEqual(om2, np.dot(rate2expand[i, i, :], om2expand))
            for j in range(self.vecstar.Nvstars):
                if j != i:
                    for d in range(self.NNstar.Nstars):
                        self.assertAlmostEqual(0, rate2expand[i, j, d])
        # print(np.dot(rate2expand, om2expand))


class VectorStarFCCOmega2linearTests(VectorStarOmega2linearTests):
    """Set of tests for our expansion of omega_2 in NN stars for FCC"""
    def setUp(self):
        self.lattice, self.NNvect, self.groupops, self.star = setupFCC()
        self.NNstar = stars.StarSet(self.NNvect, self.groupops)
        self.vecstar = stars.VectorStarSet()
        self.rates = FCCrates()


class VectorStarBias2linearTests(unittest.TestCase):
    """Set of tests for our expansion of bias vector (2) in NN stars"""
    def setUp(self):
        self.lattice, self.NNvect, self.groupops, self.star = setuportho()
        self.NNstar = stars.StarSet(self.NNvect, self.groupops)
        self.vecstar = stars.VectorStarSet()
        self.rates = orthorates()

    def testConstructBias2(self):
        self.NNstar.generate(1) # we need the NN set of stars for NN jumps
        # construct the set of rates corresponding to the unique stars:
        om2expand = np.zeros(self.NNstar.Nstars)
        for vec, rate in zip(self.NNvect, self.rates):
            om2expand[self.NNstar.starindex(vec)] = rate
        self.star.generate(2) # go ahead and make a "large" set of stars
        self.vecstar.generate(self.star)
        bias2expand = self.vecstar.bias2expansion(self.NNstar)
        self.assertEqual(np.shape(bias2expand),
                         (self.vecstar.Nvstars, self.NNstar.Nstars))
        biasvec = np.zeros((self.star.Npts, 3)) # bias vector: only the exchange hops
        for i, pt in enumerate(self.star.pts):
            for vec, rate in zip(self.NNvect, self.rates):
                if (vec == pt).all():
                    biasvec[i, :] += vec*rate
        # construct the same bias vector using our expansion
        biasveccomp = np.zeros((self.star.Npts, 3))
        for om2, svpos, svvec in zip(np.dot(bias2expand, om2expand),
                                     self.vecstar.vecpos,
                                     self.vecstar.vecvec):
            # test the construction
            for Ri, vi in zip(svpos, svvec):
                biasveccomp[self.star.pointindex(Ri), :] = om2*vi
        for i in range(self.star.Npts):
            for d in range(3):
                self.assertAlmostEqual(biasvec[i, d], biasveccomp[i, d])
        # print(biasvec)
        # print(np.dot(bias2expand, om2expand))


class VectorStarFCCBias2linearTests(VectorStarBias2linearTests):
    """Set of tests for our expansion of bias vector (2) in NN stars for FCC"""
    def setUp(self):
        self.lattice, self.NNvect, self.groupops, self.star = setupFCC()
        self.NNstar = stars.StarSet(self.NNvect, self.groupops)
        self.vecstar = stars.VectorStarSet()
        self.rates = FCCrates()


class VectorStarBias1linearTests(unittest.TestCase):
    """Set of tests for our expansion of bias vector (1) in double + NN stars"""
    def setUp(self):
        self.lattice, self.NNvect, self.groupops, self.star = setuportho()
        self.NNstar = stars.StarSet(self.NNvect, self.groupops)
        self.dstar = stars.DoubleStarSet()
        self.vecstar = stars.VectorStarSet()
        self.rates = orthorates()

    def testConstructBias1(self):
        self.NNstar.generate(1) # we need the NN set of stars for NN jumps
        # construct the set of rates corresponding to the unique stars:
        om0expand = np.zeros(self.NNstar.Nstars)
        for vec, rate in zip(self.NNvect, self.rates):
            om0expand[self.NNstar.starindex(vec)] = rate
        self.star.generate(2) # go ahead and make a "large" set of stars
        self.dstar.generate(self.star)
        om1expand = np.zeros(self.dstar.Ndstars)
        # in this case, we pick up omega_1 from omega_0... maybe not the most interesting case?
        # I think we make up for the "boring" rates here by having unusual probabilities below
        for i, ds in enumerate(self.dstar.dstars):
            p1, p2 = ds[0]
            dv = self.star.pts[p1] - self.star.pts[p2]
            sind = self.NNstar.starindex(dv)
            self.assertNotEqual(sind, -1)
            om1expand[i] = om0expand[sind]
        # print 'om0:', om0expand
        # print 'om1:', om1expand
        self.vecstar.generate(self.star)
        bias1ds, omega1ds, gen1prob, bias1NN, omega1NN = self.vecstar.bias1expansion(self.dstar, self.NNstar)
        self.assertEqual(np.shape(bias1ds),
                         (self.vecstar.Nvstars, self.dstar.Ndstars))
        self.assertEqual(np.shape(omega1ds),
                         (self.vecstar.Nvstars, self.dstar.Ndstars))
        self.assertEqual(np.shape(gen1prob),
                         (self.vecstar.Nvstars, self.dstar.Ndstars))
        self.assertEqual(np.shape(bias1NN),
                         (self.vecstar.Nvstars, self.NNstar.Nstars))
        self.assertEqual(np.shape(omega1NN),
                         (self.vecstar.Nvstars, self.NNstar.Nstars))
        self.assertTrue(np.issubdtype(gen1prob.dtype, int)) # needs to be for indexing
        # make sure that we don't have -1 as our endpoint probability for any ds that are used.
        for b1ds, o1ds, b1p in zip(bias1ds, omega1ds, gen1prob):
            for bds, ods, p in zip(b1ds, o1ds, b1p):
                if p == -1:
                    self.assertEqual(bds, 0)
                    self.assertEqual(ods, 0)

        # construct some fake probabilities for testing, with an "extra" star, set it's probability to 1
        # this is ONLY needed for testing purposes--the expansion should never access it.
        # note: this probability is to be the SQRT of the true probability
        # probsqrt = np.array([1,]*(self.star.Nstars+1)) # very little bias...
        probsqrt = np.sqrt(np.array([1.10**(self.star.Nstars-n) for n in range(self.star.Nstars + 1)]))
        probsqrt[-1] = 1 # this is important, as it represents our baseline "far-field"
        biasvec = np.zeros((self.star.Npts, 3)) # bias vector: all the hops *excluding* exchange
        omega1onsite = np.zeros(self.star.Npts) # omega1 onsite vector: all the hops *excluding* exchange
        for i, pt in enumerate(self.star.pts):
            for vec, rate in zip(self.NNvect, self.rates):
                if not all(abs(pt + vec) < 1e-8):
                    # note: starindex returns -1 if not found, which defaults to the final probability of 1.
                    biasvec[i, :] += vec*rate*probsqrt[self.star.starindex(pt + vec)]
                    omega1onsite[i] -=  rate*probsqrt[self.star.starindex(pt + vec)]
            omega1onsite[i] /= probsqrt[self.star.index[i]]
        # construct the same bias vector using our expansion
        biasveccomp = np.zeros((self.star.Npts, 3))
        omega1onsitecomp = np.zeros(self.star.Npts)
        # om = om1 + om0 contributions
        for om_b, om_on, svpos, svvec in zip(np.dot(bias1ds * probsqrt[gen1prob], om1expand) +
                                                     np.dot(bias1NN, om0expand),
                                             np.dot(omega1ds * probsqrt[gen1prob], om1expand) +
                                                     np.dot(omega1NN, om0expand),
                                             self.vecstar.vecpos,
                                             self.vecstar.vecvec):
            # test the construction
            for Ri, vi in zip(svpos, svvec):
                biasveccomp[self.star.pointindex(Ri), :] += om_b*vi
                omega1onsitecomp[self.star.pointindex(Ri)] += om_on # * np.dot(vi, vi)
        for i in range(self.star.Npts):
            omega1onsitecomp[i] /= probsqrt[self.star.index[i]]
        # this is a little confusing, but we need to take into account the multiplicity;
        # this has to do with how we're building out the onsite matrix, and that there
        # can be multiple vector-stars that contain the same "base" star.
        multiplicity = np.zeros(self.star.Npts, dtype=int)
        for R in [Rv for svpos in self.vecstar.vecpos for Rv in svpos]:
            multiplicity[self.star.pointindex(R)] += 1
        omega1onsitecomp /= multiplicity

        for svpos, svvec in zip(self.vecstar.vecpos, self.vecstar.vecvec):
            for Ri, vi in zip(svpos, svvec):
                self.assertAlmostEqual(np.dot(vi, biasvec[self.star.pointindex(Ri)]),
                                       np.dot(vi, biasveccomp[self.star.pointindex(Ri)]),
                                       msg='Did not match dot product for {} along {} where {} != {}'.format(
                                           Ri, vi, biasvec[self.star.pointindex(Ri)],
                                           biasveccomp[self.star.pointindex(Ri)]
                                       ))

        for i, pt in enumerate(self.star.pts):
            self.assertAlmostEqual(omega1onsite[i], omega1onsitecomp[i],
                                   msg='Did not match onsite point[{}] {} where {} != {}'.format(
                                       i, pt, omega1onsite[i], omega1onsitecomp[i]))
            for d in range(3):
                self.assertAlmostEqual(biasvec[i, d], biasveccomp[i, d],
                                       msg='Did not match point[{}] {} direction {} where {} != {}'.format(
                                           i, pt, d, biasvec[i, d], biasveccomp[i, d]))

        # print(np.dot(bias1ds * probsqrt[bias1prob], om1expand))
        # print(np.dot(bias1NN, om0expand))
        # print 'bias1ds', bias1ds
        # print 'bias1prob', bias1prob
        # print 'bias1NN', bias1NN
        # print 'biasvec, biasveccomp:'
        # for bv, bvc in zip(biasvec, biasveccomp):
        #     print bv, bvc


class VectorStarFCCBias1linearTests(VectorStarBias1linearTests):
    """Set of tests for our expansion of bias vector (1) in double + NN stars for FCC"""
    def setUp(self):
        self.lattice, self.NNvect, self.groupops, self.star = setupFCC()
        self.NNstar = stars.StarSet(self.NNvect, self.groupops)
        self.dstar = stars.DoubleStarSet()
        self.vecstar = stars.VectorStarSet()
        self.rates = FCCrates()
