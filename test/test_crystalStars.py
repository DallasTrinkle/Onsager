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
    jumpnetwork = [[((0, 0), np.array([3., 0., 0.])), ((0, 0), np.array([-3., 0., 0.]))],
                   [((0, 0), np.array([0., 2., 0.])), ((0, 0), np.array([0., -2., 0.]))],
                   [((0, 0), np.array([0., 0., 1.])), ((0, 0), np.array([0., 0., -1.]))]]
    return crys, jumpnetwork


def orthorates():
    return np.array([3., 2., 1.])


def setupcubic():
    crys = crystal.Crystal(np.eye(3), [[np.zeros(3)]])
    jumpnetwork = [[((0, 0), np.array([1., 0., 0.])), ((0, 0), np.array([-1., 0., 0.])),
                    ((0, 0), np.array([0., 1., 0.])), ((0, 0), np.array([0., -1., 0.])),
                    ((0, 0), np.array([0., 0., 1.])), ((0, 0), np.array([0., 0., -1.]))]]
    return crys, jumpnetwork

def setupsquare():
    crys = crystal.Crystal(np.eye(2), [[np.zeros(2)]])
    jumpnetwork = [[((0, 0), np.array([1., 0.])), ((0, 0), np.array([-1., 0.])),
                    ((0, 0), np.array([0., 1.])), ((0, 0), np.array([0., -1.]))]]
    return crys, jumpnetwork



def cubicrates():
    return np.array([1. / 6.])

def squarerates():
    return np.array([1. / 4.])


def setupFCC():
    lattice = crystal.Crystal.FCC(2.)
    jumpnetwork = lattice.jumpnetwork(0, 2. * np.sqrt(0.5) + 0.01)
    return lattice, jumpnetwork


def FCCrates():
    return np.array([1. / 12.])


def setupHCP():
    lattice = crystal.Crystal.HCP(1.)
    jumpnetwork = lattice.jumpnetwork(0, 1.01)
    return lattice, jumpnetwork


def HCPrates():
    return np.array([1. / 12., 1. / 12.])


def setupBCC():
    lattice = crystal.Crystal.BCC(1.)
    jumpnetwork = lattice.jumpnetwork(0, np.sqrt(0.75) + 0.01)
    return lattice, jumpnetwork


def BCCrates():
    return np.array([1. / 8.])


def setupB2():
    lattice = crystal.Crystal(np.eye(3), [np.array([0., 0., 0.]), np.array([0.45, 0.45, 0.45])])
    jumpnetwork = lattice.jumpnetwork(0, 0.9)
    return lattice, jumpnetwork


def B2rates():
    return np.array([1. / 8., 1. / 8., 1. / 8.])


class PairStateTests(unittest.TestCase):
    """Tests of the PairState class"""
    longMessage = False

    def setUp(self):
        self.hcp = crystal.Crystal.HCP(1.0)

    def testZeroClean(self):
        """Does the zero clean function work?"""
        threshold = 1e-8
        for s in (1, 3, (3,3), (3,3,3), (3,3,3,3)):
            a = np.random.uniform(-0.99*threshold,0.99*threshold, s)
            b = np.random.uniform(-10*threshold,10*threshold, s)
            azero = stars.zeroclean(a, threshold=threshold)
            it = np.nditer(a, flags=['multi_index'])
            while not it.finished:
                self.assertEqual(azero[it.multi_index], 0,
                                 msg='Failed on {} for small {} matrix?'.format(it.multi_index, s))
                it.iternext()
            # self.assertTrue(np.all(azero == 0), msg='Failed for {} matrix?'.format(s))
            bnonzero = stars.zeroclean(b, threshold=threshold)
            it = np.nditer(b, flags=['multi_index'])
            while not it.finished:
                if abs(b[it.multi_index]) > threshold:
                    self.assertEqual(b[it.multi_index], bnonzero[it.multi_index],
                                     msg='Failed on nonzero {} for large {} matrix?'.format(it.multi_index, s))
                else:
                    self.assertEqual(bnonzero[it.multi_index], 0,
                                     msg='Failed on zero {} for large {} matrix?'.format(it.multi_index, s))
                it.iternext()

    def testZero(self):
        """Is zero consistent?"""
        zero = stars.PairState.zero()
        self.assertTrue(zero.iszero())
        self.assertTrue(stars.PairState.fromcrys(self.hcp, 0, (0, 0), np.zeros(3)).iszero())

    def testArithmetic(self):
        """Does addition,subtraction, and endpoint subtraction work as expected?"""
        pos1 = self.hcp.pos2cart(np.array([0, 0, 0]), (0, 0))  # first site
        pos2 = self.hcp.pos2cart(np.array([0, 0, 0]), (0, 1))  # second site
        pos1a1 = self.hcp.pos2cart(np.array([1, 0, 0]), (0, 0))  # first + a1
        pos2a1 = self.hcp.pos2cart(np.array([1, 0, 0]), (0, 1))  # second + a1
        pos1a2 = self.hcp.pos2cart(np.array([0, 1, 0]), (0, 0))  # first + a2
        pos2a2 = self.hcp.pos2cart(np.array([0, 1, 0]), (0, 1))  # second + a2
        ps1 = stars.PairState.fromcrys(self.hcp, 0, (0, 0), pos1a1 - pos1)
        ps2 = stars.PairState.fromcrys(self.hcp, 0, (0, 0), 2 * (pos1a1 - pos1))
        self.assertEqual(ps1 + ps1, ps2)
        ps2 = stars.PairState.fromcrys(self.hcp, 0, (1, 0), pos1 - pos2)
        with self.assertRaises(ArithmeticError):
            ps1 + ps2
        ps3 = stars.PairState.fromcrys(self.hcp, 0, (1, 0), pos1a1 - pos2)
        self.assertEqual(ps2 + ps1, ps3)
        ps4 = stars.PairState.fromcrys(self.hcp, 0, (0, 1), pos2a1 - pos1)
        self.assertEqual(ps1 - ps2, ps4)
        for i in [ps1, ps2, ps3, ps4]:
            self.assertTrue((i - i).iszero())
            self.assertTrue((i ^ i).iszero())  # endpoint subtraction!
        # endpoint subtraction: 0.(000):1.(010) ^ 0.(000):1.(000)  == 1.(000):1.(010)
        ps1 = stars.PairState.fromcrys(self.hcp, 0, (0, 1), pos2a2 - pos1)
        ps2 = stars.PairState.fromcrys(self.hcp, 0, (0, 1), pos2 - pos1)
        ps3 = stars.PairState.fromcrys(self.hcp, 0, (1, 1), pos2a2 - pos2)
        self.assertEqual(ps1 ^ ps2, ps3)
        with self.assertRaises(ArithmeticError):
            ps1 ^ ps3
        self.assertEqual(stars.PairState.fromcrys(self.hcp, 0, (1, 0), pos1 - pos2),
                         - stars.PairState.fromcrys(self.hcp, 0, (0, 1), pos2 - pos1))

    def testGroupOps(self):
        """Applying group operations?"""
        pos1 = self.hcp.pos2cart(np.array([0, 0, 0]), (0, 0))  # first site
        pos2 = self.hcp.pos2cart(np.array([0, 0, 0]), (0, 1))  # second site
        pos1a1 = self.hcp.pos2cart(np.array([1, 0, 0]), (0, 0))  # first + a1
        pos2a1 = self.hcp.pos2cart(np.array([1, 0, 0]), (0, 1))  # second + a1
        pos1a2 = self.hcp.pos2cart(np.array([0, 1, 0]), (0, 0))  # first + a2
        pos2a2 = self.hcp.pos2cart(np.array([0, 1, 0]), (0, 1))  # second + a2
        pos1a3 = self.hcp.pos2cart(np.array([0, 0, 1]), (0, 0))  # first + a3
        pos2a3 = self.hcp.pos2cart(np.array([0, 0, 1]), (0, 1))  # second + a3
        iterlist = [(0, pos1), (1, pos2),
                    (0, pos1a1), (1, pos2a1),
                    (0, pos1a2), (1, pos2a2),
                    (0, pos1a3), (1, pos2a3)]
        zero = stars.PairState.zero()
        for g in self.hcp.G:
            self.assertTrue((zero.g(self.hcp, 0, g)).iszero())
        for i, x1 in iterlist:
            for j, x2 in iterlist:
                ps = stars.PairState.fromcrys(self.hcp, 0, (i, j), x2 - x1)
                self.assertTrue(ps.__sane__(self.hcp, 0), msg="{} is not sane?".format(ps))
                for g in self.hcp.G:
                    # apply directly to the vectors, and get the site index from the group op
                    gi, gx1 = g.indexmap[0][i], self.hcp.g_cart(g, x1)
                    gj, gx2 = g.indexmap[0][j], self.hcp.g_cart(g, x2)
                    gpsdirect = stars.PairState.fromcrys(self.hcp, 0, (gi, gj), gx2 - gx1)
                    gps = ps.g(self.hcp, 0, g)
                    self.assertEqual(gps, gpsdirect,
                                     msg="{}\n*{} =\n{} !=\n{}".format(g, ps, gps, gpsdirect))


class StarTests(unittest.TestCase):
    """Set of tests that our star code is behaving correctly for a general materials"""
    longMessage = False

    def setUp(self):
        self.crys, self.jumpnetwork = setuportho()
        self.chem = 0
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)

    def isclosed(self, starset, starindex):
        """Evaluate if star s is closed against group operations."""
        for i1 in starset.stars[starindex]:
            ps1 = starset.states[i1]
            for i2 in starset.stars[starindex]:
                ps2 = starset.states[i2]
                if not any(ps1 == ps2.g(self.crys, self.chem, g) for g in self.crys.G):
                    return False
        return True

    def testStarConsistent(self):
        """Check that the counts (Npts, Nstars) make sense, with Nshells = 1..4"""
        for n in range(1, 5):
            self.starset.generate(n)
            for starindex in range(self.starset.Nstars):
                self.assertTrue(self.isclosed(self.starset, starindex))

    def testStarindices(self):
        """Check that our indexing is correct."""
        dim = self.crys.dim
        self.starset.generate(4)
        for si, star in enumerate(self.starset.stars):
            for i in star:
                self.assertEqual(si, self.starset.starindex(self.starset.states[i]))
        self.assertEqual(None, self.starset.starindex(stars.PairState.zero(dim=dim)))
        self.assertEqual(None, self.starset.stateindex(stars.PairState.zero(dim=dim)))
        self.assertNotIn(stars.PairState.zero(dim=dim), self.starset)  # test __contains__ (PS in starset)

    def assertEqualStars(self, s1, s2):
        """Asserts that two star sets are equal."""
        self.assertEqual(s1.Nstates, s2.Nstates,
                         msg='Number of states in two star sets are not equal: {} != {}'.format(
                             s1.Nstates, s2.Nstates))
        self.assertEqual(s1.Nshells, s2.Nshells,
                         msg='Number of shells in two star sets are not equal: {} != {}'.format(
                             s1.Nshells, s2.Nshells))
        self.assertEqual(s1.Nstars, s2.Nstars,
                         msg='Number of stars in two star sets are not equal: {} != {}'.format(
                             s1.Nstars, s2.Nstars))
        for s in s1.stars:
            # grab the first entry, and index it into a star; then check that all the others are there.
            ps0 = s1.states[s[0]]
            s2ind = s2.starindex(s1.states[s[0]])
            self.assertNotEqual(s2ind, -1,
                                msg='Could not find state {} from s1 in s2'.format(s1.states[s[0]]))
            self.assertEqual(len(s), len(s2.stars[s2ind]),
                             msg='Star in s1 has different length than star in s2? {} != {}'.format(
                                 len(s), len(s2.stars[s2ind])))
            for i1 in s:
                ps1 = s1.states[i1]
                self.assertEqual(s2ind, s2.starindex(ps1),
                                 msg='States {} and {} from star in s1 belong to different stars in s2'.format(
                                     ps0, ps1))

    def testStarCombine(self):
        """Check that we can combine two stars and get what we expect."""
        s1 = self.starset.copy()
        s2 = self.starset.copy()
        # s3 = self.starset.copy()
        s4 = self.starset.copy()
        s1.generate(1)
        s2.generate(1)
        s3 = s1 + s2
        s4.generate(2)
        # s3 = s1 + s2, should equal s4
        self.assertEqualStars(s1, s2)
        self.assertEqualStars(s3, s4)


class CubicStarTests(StarTests):
    """Set of tests that our star code is behaving correctly for cubic materials"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupcubic()
        self.chem = 0
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)

    def testStarConsistent(self):
        """Check that the counts (Npts, Nstars) make sense for cubic, with Nshells = 1..4"""
        dim = self.crys.dim
        for n in range(1, 5):
            self.starset.generate(n)
            for starindex in range(self.starset.Nstars):
                self.assertTrue(self.isclosed(self.starset, starindex))
            self.assertEqual(None, self.starset.starindex(stars.PairState.zero(dim=dim)))
            self.assertEqual(None, self.starset.stateindex(stars.PairState.zero(dim=dim)))

            for s in self.starset.stars:
                x = np.sort(abs(self.starset.states[s[0]].dx))
                num = (2 ** (3 - list(x).count(0)))
                if not np.isclose(x[0], x[1]) and not np.isclose(x[1], x[2]):
                    num *= 6
                elif not np.isclose(x[0], x[1]) or not np.isclose(x[1], x[2]):
                    num *= 3
                self.assertEqual(num, len(s),
                                 msg='Count for {} should be {}, got {}'.format(
                                     self.starset.states[s[0]].dx, num, len(s)))


class HCPStarTests(StarTests):
    """Set of tests that our star code is behaving correctly for HCP"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupHCP()
        self.chem = 0
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)

    def testDiffGenerate(self):
        self.starset.generate(1)
        dS = self.starset.copy(empty=True)
        dS.diffgenerate(self.starset, self.starset)
        self.assertEqual(len(dS.stars[0]), 2)
        for si in dS.stars[0]:
            self.assertTrue(dS.states[si].iszero())
        # 0, a1, a1+a2, 2a1; p, p+a1, p+a1+a2, p+2a1; c, c+a1 (p=pyramidal vector)
        self.assertEqual(dS.Nstars, 4 + 4 + 2)


class FCCStarTests(CubicStarTests):
    """Set of tests that our star code is behaving correctly for FCC"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupFCC()
        self.chem = 0
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)

    def testStarCount(self):
        """Check that the counts (Npts, Nstars) make sense for FCC, with Nshells = 1, 2, 3"""
        # 110
        self.starset.generate(1)
        self.assertEqual(self.starset.Nstars, 1)
        self.assertEqual(self.starset.Nstates, 12)

        # 110, 200, 211, 220
        self.starset.generate(2)
        self.assertEqual(self.starset.Nstars, 4)

        # 110, 200, 211, 220, 310, 321, 330, 222
        self.starset.generate(3)
        self.assertEqual(self.starset.Nstars, 8)


class SquareStarTests(StarTests):
    """Set of tests that our star code is behaving correctly for square"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupsquare()
        self.chem = 0
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)


# replaced DoubleStarTests
class JumpNetworkTests(unittest.TestCase):
    """Set of tests that our JumpNetwork is behaving correctly."""
    longMessage = False

    def setUp(self):
        self.crys, self.jumpnetwork = setupFCC()
        self.chem = 0
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)

    def testJumpNetworkGeneration(self):
        """Can we generate jumpnetworks?"""
        self.starset.generate(1)
        jumpnetwork1, jt, sp = self.starset.jumpnetwork_omega1()
        self.assertEqual(len(jumpnetwork1), 1)
        self.assertEqual(jt[0], 0)
        self.assertEqual(sp[0], (0, 0))
        jumpnetwork2, jt, sp = self.starset.jumpnetwork_omega2()
        self.assertEqual(len(jumpnetwork2), 1)
        self.assertEqual(len(jumpnetwork2[0]), len(self.jumpnetwork[0]))
        self.assertEqual(jt[0], 0)
        self.assertEqual(sp[0], (0, 0))

    def testJumpNetworkCount(self):
        """Check that the counts in the jumpnetwork make sense for FCC, with Nshells = 1, 2"""
        # each of the 12 <110> pairs to 101, 10-1, 011, 01-1 = 4, so should be 48 pairs
        # (which includes "double counting": i->j and j->i)
        # but *all* of those 48 are all equivalent to each other by symmetry: one jump network.
        self.starset.generate(1)
        jumpnetwork, jt, sp = self.starset.jumpnetwork_omega1()
        self.assertEqual(len(jumpnetwork), 1)
        self.assertEqual(len(jumpnetwork[0]), 48)
        # Now have four stars (110, 200, 211, 220), so this means
        # 12 <110> pairs to 11 (no 000!); 12*11
        # 6 <200> pairs to 110, 101, 1-10, 10-1; 211, 21-1, 2-11, 2-1-1 = 8; 6*8
        # 24 <211> pairs to 110, 101; 200; 112, 121; 202, 220 = 7; 24*7
        # 12 <220> pairs to 110; 12-1, 121, 21-1, 211 = 5; 12*5
        # unique pairs: (110, 101); (110, 200); (110, 211); (110, 220); (200, 211); (211, 112); (211, 220)
        self.starset.generate(2)
        self.assertEqual(self.starset.Nstars, 4)
        jumpnetwork, jt, sp = self.starset.jumpnetwork_omega1()
        self.assertEqual(len(jumpnetwork), 4 + 1 + 2)
        self.assertEqual(sum(len(jlist) for jlist in jumpnetwork), 12 * 11 + 6 * 8 + 24 * 7 + 12 * 5)
        # check that nothing changed with the larger StarSet
        jumpnetwork2, jt, sp = self.starset.jumpnetwork_omega2()
        self.assertEqual(len(jumpnetwork2), 1)
        self.assertEqual(len(jumpnetwork2[0]), len(self.jumpnetwork[0]))
        self.assertEqual(jt[0], 0)
        self.assertEqual(sp[0], (0, 0))

    def testJumpNetworkindices(self):
        """Check that our indexing works correctly for Nshell=1..3"""
        for nshells in range(1, 4):
            self.starset.generate(nshells)
            jumpnetwork, jt, sp = self.starset.jumpnetwork_omega1()
            for jumplist, (s1, s2) in zip(jumpnetwork, sp):
                for (i, f), dx in jumplist:
                    si = self.starset.index[i]
                    sf = self.starset.index[f]
                    self.assertTrue((s1, s2) == (si, sf) or (s1, s2) == (sf, si))


class VectorStarTests(unittest.TestCase):
    """Set of tests that our VectorStar class is behaving correctly"""
    longMessage = False

    def setUp(self):
        self.crys, self.jumpnetwork = setuportho()
        self.chem = 0
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)

    def testVectorStarGenerate(self):
        """Can we generate star-vectors that make sense?"""
        self.starset.generate(1)
        self.vecstarset = stars.VectorStarSet(self.starset)
        self.assertTrue(self.vecstarset.Nvstars > 0)

    def VectorStarConsistent(self, nshells):
        """Do the star vectors obey the definition?"""
        self.starset.generate(nshells)
        self.vecstarset = stars.VectorStarSet(self.starset)
        for s, vec in zip(self.vecstarset.vecpos, self.vecstarset.vecvec):
            for si, v in zip(s, vec):
                PS = self.starset.states[si]
                for g in self.crys.G:
                    gsi = self.starset.stateindex(PS.g(self.crys, self.chem, g))
                    vrot = self.crys.g_direc(g, v)
                    for si1, v1 in zip(s, vec):
                        if gsi == si1: self.assertTrue(np.allclose(v1, vrot))

    def VectorStarOrthonormal(self, nshells):
        """Are the star vectors orthonormal?"""
        self.starset.generate(1)
        self.vecstarset = stars.VectorStarSet(self.starset)
        for s1, v1 in zip(self.vecstarset.vecpos, self.vecstarset.vecvec):
            for s2, v2 in zip(self.vecstarset.vecpos, self.vecstarset.vecvec):
                if s1[0] == s2[0]:
                    dp = sum(np.dot(vv1, vv2) for vv1, vv2 in zip(v1, v2))
                    if np.allclose(v1[0], v2[0]):
                        self.assertAlmostEqual(1., dp,
                                               msg='Failed normality for {}/{} and {}/{}'.format(
                                                   self.starset.states[s1[0]], v1[0],
                                                   self.starset.states[s2[0]], v2[0]))
                    else:
                        self.assertAlmostEqual(0., dp,
                                               msg='Failed orthogonality for {}/{} and {}/{}'.format(
                                                   self.starset.states[s1[0]], v1[0],
                                                   self.starset.states[s2[0]], v2[0]))

    def testVectorStarConsistent(self):
        """Do the star vectors obey the definition?"""
        self.VectorStarConsistent(2)

    def testVectorStarOrthonormal(self):
        self.VectorStarOrthonormal(2)

    def testVectorStarCount(self):
        """Does our star vector count make any sense?"""
        self.starset.generate(1)
        self.vecstarset = stars.VectorStarSet(self.starset)
        self.assertEqual(self.vecstarset.Nvstars, 3)

    def testVectorStarOuterProduct(self):
        """Do we generate the correct outer products for our star-vectors (symmetry checks)?"""
        dim = self.crys.dim
        self.starset.generate(2)
        self.vecstarset = stars.VectorStarSet(self.starset)
        self.assertEqual(np.shape(self.vecstarset.outer),
                         (dim, dim, self.vecstarset.Nvstars, self.vecstarset.Nvstars))
        # check our diagonal blocks first:
        for outer in [self.vecstarset.outer[:, :, i, i]
                      for i in range(self.vecstarset.Nvstars)]:
            self.assertAlmostEqual(np.trace(outer), 1)
            # should also be symmetric:
            for g in self.crys.G:
                g_out_gT = self.crys.g_tensor(g, outer)
                self.assertTrue(np.allclose(g_out_gT, outer))
        # off-diagonal terms now
        for outer in [self.vecstarset.outer[:, :, i, j]
                      for i in range(self.vecstarset.Nvstars)
                      for j in range(self.vecstarset.Nvstars)
                      if i != j]:
            self.assertAlmostEqual(np.trace(outer), 0)
            # should also be symmetric:
            for g in self.crys.G:
                g_out_gT = self.crys.g_tensor(g, outer)
                self.assertTrue(np.allclose(g_out_gT, outer))
        for i, (s0, svv0) in enumerate(zip(self.vecstarset.vecpos, self.vecstarset.vecvec)):
            for j, (s1, svv1) in enumerate(zip(self.vecstarset.vecpos, self.vecstarset.vecvec)):
                testouter = np.zeros((dim, dim))
                if s0[0] == s1[0]:
                    # we have the same underlying star to work with, so...
                    for v0, v1 in zip(svv0, svv1):
                        testouter += np.outer(v0, v1)
                self.assertTrue(np.allclose(self.vecstarset.outer[:, :, i, j], testouter),
                                msg='Failed for vector stars {} and {}:\n{} !=\n{}'.format(
                                    i, j, self.vecstarset.outer[:, :, i, j], testouter))


class VectorStarFCCTests(VectorStarTests):
    """Set of tests that our VectorStar class is behaving correctly, for FCC"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupFCC()
        self.chem = 0
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)

    def testVectorStarCount(self):
        """Does our star vector count make any sense?"""
        self.starset.generate(2)
        self.vecstarset = stars.VectorStarSet(self.starset)
        # nn + nn = 4 stars, and that should make 5 star-vectors!
        self.assertEqual(self.starset.Nstars, 4)
        self.assertEqual(self.vecstarset.Nvstars, 5)

    def testVectorStarConsistent(self):
        """Do the star vectors obey the definition?"""
        self.VectorStarConsistent(2)

    def testVectorStarOuterProductMore(self):
        """Do we generate the correct outer products for our star-vectors?"""
        self.starset.generate(2)
        self.vecstarset = stars.VectorStarSet(self.starset)
        # with cubic symmetry, these all have to equal 1/3 * identity, and
        # with a diagonal matrix
        testouter = 1. / 3. * np.eye(3)
        for outer in [self.vecstarset.outer[:, :, i, i]
                      for i in range(self.vecstarset.Nvstars)]:
            self.assertTrue(np.allclose(outer, testouter))
        for outer in [self.vecstarset.outer[:, :, i, j]
                      for i in range(self.vecstarset.Nvstars)
                      for j in range(self.vecstarset.Nvstars)
                      if i != j]:
            self.assertTrue(np.allclose(outer, 0))


class VectorStarHCPTests(VectorStarTests):
    """Set of tests that our VectorStar class is behaving correctly, for HCP"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupHCP()
        self.chem = 0
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)

    def testVectorStarCount(self):
        """Does our star vector count make any sense?"""
        self.starset.generate(1)
        self.vecstarset = stars.VectorStarSet(self.starset)
        # two stars, with two vectors: one basal, one along c (more or less)
        self.assertEqual(self.starset.Nstars, 2)
        self.assertEqual(self.vecstarset.Nvstars, 2 + 2)


class VectorStarBCCTests(VectorStarTests):
    """Set of tests that our VectorStar class is behaving correctly, for BCC"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupBCC()
        self.chem = 0
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)

    def testVectorStarCount(self):
        """Does our star vector count make any sense?"""
        self.starset.generate(2)
        self.vecstarset = stars.VectorStarSet(self.starset)
        # nn + nn = 4 stars, and that should make 4 star-vectors!
        self.assertEqual(self.starset.Nstars, 4)
        self.assertEqual(self.vecstarset.Nvstars, 4)


class VectorStarB2Tests(VectorStarTests):
    """Set of tests that our VectorStar class is behaving correctly, for B2"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupB2()
        self.chem = 0
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)

    def testVectorStarCount(self):
        """Does our star vector count make any sense?"""
        self.starset.generate(2)
        self.vecstarset = stars.VectorStarSet(self.starset)
        # nn + nn = 10 stars, and that should make 20 star-vectors!
        self.assertEqual(self.starset.Nstars, 10)
        self.assertEqual(self.vecstarset.Nvstars, 20)


class VectorStarSquareTests(VectorStarTests):
    """Set of tests that our VectorStar class is behaving correctly, for square"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupsquare()
        self.chem = 0
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)

    def testVectorStarCount(self):
        """Does our star vector count make any sense?"""
        self.starset.generate(3)
        self.vecstarset = stars.VectorStarSet(self.starset)
        # nn + nn + nn = 5 stars (10,11,20,21,30), and that should make 6 star-vectors!
        self.assertEqual(self.starset.Nstars, 5)
        self.assertEqual(self.vecstarset.Nvstars, 6)


import onsager.GFcalc as GFcalc


class VectorStarGFlinearTests(unittest.TestCase):
    """Set of tests that make sure we can construct the GF matrix as a linear combination"""

    def setUp(self):
        self.crys, self.jumpnetwork = setuportho()
        self.rates = orthorates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)

        # not an accurate value for Nmax; but since we just need some values, it's fine
        self.GF = GFcalc.GFCrystalcalc(self.crys, self.chem, self.sitelist, self.jumpnetwork, Nmax=2)
        self.GF.SetRates([1 for s in self.sitelist],
                         [0 for s in self.sitelist],
                         self.rates,
                         [0 for j in self.rates])

    def ConstructGF(self, nshells):
        self.starset.generate(nshells)
        self.vecstarset = stars.VectorStarSet(self.starset)
        GFexpand, GFstarset = self.vecstarset.GFexpansion()
        gexpand = np.zeros(GFstarset.Nstars)
        for i, star in enumerate(GFstarset.stars):
            st = GFstarset.states[star[0]]
            gexpand[i] = self.GF(st.i, st.j, st.dx)
        for i in range(self.vecstarset.Nvstars):
            for j in range(self.vecstarset.Nvstars):
                # test the construction
                # GFsum = np.sum(GFexpand[i,j,:])
                # if abs(GFsum) > 1e-5:
                #     print('GF vector star set between:')
                #     for R, v in zip(self.vecstarset.vecpos[i], self.vecstarset.vecvec[i]):
                #         print('  {} / {}'.format(self.starset.states[R], v))
                #     print('and')
                #     for R, v in zip(self.vecstarset.vecpos[j], self.vecstarset.vecvec[j]):
                #         print('  {} / {}'.format(self.starset.states[R], v))
                #     print('expansion:')
                #     for k, g in enumerate(GFexpand[i,j,:]):
                #         if abs(g) > 1e-5:
                #             print('  {:+0.15f}*{}'.format(g, GFstarset.states[k]))
                # self.assertAlmostEqual(GFsum, 0, msg='Failure for {},{}: GF= {}'.format(i,j,GFsum))
                g = 0
                for si, vi in zip(self.vecstarset.vecpos[i], self.vecstarset.vecvec[i]):
                    for sj, vj in zip(self.vecstarset.vecpos[j], self.vecstarset.vecvec[j]):
                        try:
                            ds = self.starset.states[sj] ^ self.starset.states[si]
                        except:
                            continue
                        g += np.dot(vi, vj) * self.GF(ds.i, ds.j, ds.dx)
                self.assertAlmostEqual(g, np.dot(GFexpand[i, j, :], gexpand))
                # Removed this test. It's not generally true.
                # self.assertAlmostEqual(np.sum(GFexpand), 0)
                # print(np.dot(GFexpand, gexpand))

    def testConstructGF(self):
        """Test the construction of the GF using double-nn shell"""
        self.ConstructGF(2)


class VectorStarGFFCClinearTests(VectorStarGFlinearTests):
    """Set of tests that make sure we can construct the GF matrix as a linear combination for FCC"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupFCC()
        self.rates = FCCrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)

        # not an accurate value for Nmax; but since we just need some values, it's fine
        self.GF = GFcalc.GFCrystalcalc(self.crys, self.chem, self.sitelist, self.jumpnetwork, Nmax=2)
        self.GF.SetRates([1 for s in self.sitelist],
                         [0 for s in self.sitelist],
                         self.rates,
                         [0 for j in self.rates])

    def testConstructGF(self):
        """Test the construction of the GF using double-nn shell"""
        self.ConstructGF(2)


class VectorStarGFHCPlinearTests(VectorStarGFlinearTests):
    """Set of tests that make sure we can construct the GF matrix as a linear combination for HCP"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupHCP()
        self.rates = HCPrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)

        # not an accurate value for Nmax; but since we just need some values, it's fine
        self.GF = GFcalc.GFCrystalcalc(self.crys, self.chem, self.sitelist, self.jumpnetwork, Nmax=2)
        self.GF.SetRates([1 for s in self.sitelist],
                         [0 for s in self.sitelist],
                         self.rates,
                         [0 for j in self.rates])

    def testConstructGF(self):
        """Test the construction of the GF using double-nn shell"""
        self.ConstructGF(2)


class VectorStarGFBCClinearTests(VectorStarGFlinearTests):
    """Set of tests that make sure we can construct the GF matrix as a linear combination for BCC"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupBCC()
        self.rates = BCCrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)

        # not an accurate value for Nmax; but since we just need some values, it's fine
        self.GF = GFcalc.GFCrystalcalc(self.crys, self.chem, self.sitelist, self.jumpnetwork, Nmax=2)
        self.GF.SetRates([1 for s in self.sitelist],
                         [0 for s in self.sitelist],
                         self.rates,
                         [0 for j in self.rates])

    def testConstructGF(self):
        """Test the construction of the GF using double-nn shell"""
        self.ConstructGF(2)


class VectorStarGFSquarelinearTests(VectorStarGFlinearTests):
    """Set of tests that make sure we can construct the GF matrix as a linear combination for square"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupsquare()
        self.rates = squarerates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)

        # not an accurate value for Nmax; but since we just need some values, it's fine
        self.GF = GFcalc.GFCrystalcalc(self.crys, self.chem, self.sitelist, self.jumpnetwork, Nmax=2)
        self.GF.SetRates([1 for s in self.sitelist],
                         [0 for s in self.sitelist],
                         self.rates,
                         [0 for j in self.rates])

    def testConstructGF(self):
        """Test the construction of the GF using double-nn shell"""
        self.ConstructGF(2)


class VectorStarGFB2linearTests(VectorStarGFlinearTests):
    """Set of tests that make sure we can construct the GF matrix as a linear combination for B2"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupB2()
        self.rates = B2rates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)

        # not an accurate value for Nmax; but since we just need some values, it's fine
        self.GF = GFcalc.GFCrystalcalc(self.crys, self.chem, self.sitelist, self.jumpnetwork, Nmax=2)
        self.GF.SetRates([1 for s in self.sitelist],
                         [0 for s in self.sitelist],
                         self.rates,
                         [0 for j in self.rates])

    def testConstructGF(self):
        """Test the construction of the GF using double-nn shell"""
        self.ConstructGF(2)


class VectorStarOmega0Tests(unittest.TestCase):
    """Set of tests for our expansion of omega_0"""

    longMessage = False

    def setUp(self):
        self.crys, self.jumpnetwork = setuportho()
        self.rates = orthorates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)

    def testConstructOmega0(self):
        # NOTE: now we only take omega0 *here* to be those equivalent to omega1 jumps; the exchange
        # terms are handled in omega2; the jumps outside the kinetic shell simply contributed onsite escape
        # terms that get subtracted away, since the outer kinetic shell *has* to have zero energy
        self.starset.generate(2)  # we need at least 2nd nn to even have jump networks to worry about...
        jumpnetwork_omega1, jt, sp = self.starset.jumpnetwork_omega1()
        self.vecstarset = stars.VectorStarSet(self.starset)
        rate0expand, rate0escape, rate1expand, rate1escape = self.vecstarset.rateexpansions(jumpnetwork_omega1, jt)
        # rate0expand = self.vecstarset.rate0expansion()
        self.assertEqual(np.shape(rate0expand),
                         (self.vecstarset.Nvstars, self.vecstarset.Nvstars, len(self.jumpnetwork)))
        om0expand = self.rates.copy()
        # put together the onsite and off-diagonal terms for our matrix:
        # go through each state, and add the escapes for the vacancy; see if vacancy (PS.j)
        # is the initial state (i) for a transition out (i,f), dx
        om0matrix = np.zeros((self.starset.Nstates, self.starset.Nstates))
        for ns, PS in enumerate(self.starset.states):
            for rate, jumplist in zip(self.rates, self.starset.jumpnetwork_index):
                for TS in [self.starset.jumplist[jumpindex] for jumpindex in jumplist]:
                    if PS.j == TS.i:
                        nsend = self.starset.stateindex(PS + TS)
                        if nsend is not None:
                            om0matrix[ns, nsend] += rate
                            om0matrix[ns, ns] -= rate
        # now, we need to convert that omega0 matrix into the "folded down"
        for i, (sRv0, svv0) in enumerate(zip(self.vecstarset.vecpos, self.vecstarset.vecvec)):
            for j, (sRv1, svv1) in enumerate(zip(self.vecstarset.vecpos, self.vecstarset.vecvec)):
                om0_sv = 0
                for i0, v0 in zip(sRv0, svv0):
                    for i1, v1 in zip(sRv1, svv1):
                        om0_sv += np.dot(v0, v1) * om0matrix[i0, i1]
                om0_sv_comp = np.dot(rate0expand[i, j], om0expand)
                if i == j: om0_sv_comp += np.dot(rate0escape[i], om0expand)
                self.assertAlmostEqual(om0_sv, om0_sv_comp,
                                       msg='Failed to match {}, {}: {} != {}'.format(
                                           i, j, om0_sv, om0_sv_comp))


class VectorStarFCCOmega0Tests(VectorStarOmega0Tests):
    """Set of tests for our expansion of omega_0 for FCC"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupFCC()
        self.rates = FCCrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)


class VectorStarHCPOmega0Tests(VectorStarOmega0Tests):
    """Set of tests for our expansion of omega_0 for HCP"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupHCP()
        self.rates = HCPrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)


class VectorStarBCCOmega0Tests(VectorStarOmega0Tests):
    """Set of tests for our expansion of omega_0 for BCC"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupBCC()
        self.rates = BCCrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)


class VectorStarSquareOmega0Tests(VectorStarOmega0Tests):
    """Set of tests for our expansion of omega_0 for square"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupsquare()
        self.rates = squarerates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)


class VectorStarB2Omega0Tests(VectorStarOmega0Tests):
    """Set of tests for our expansion of omega_0 for B2"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupB2()
        self.rates = B2rates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)


class VectorStarOmegalinearTests(unittest.TestCase):
    """Set of tests for our expansion of omega_1"""

    longMessage = False

    def setUp(self):
        self.crys, self.jumpnetwork = setuportho()
        # self.rates = orthorates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)

    def testConstructOmega1(self):
        self.starset.generate(2)  # we need at least 2nd nn to even have jump networks to worry about...
        self.vecstarset = stars.VectorStarSet(self.starset)
        jumpnetwork_omega1, jt, sp = self.starset.jumpnetwork_omega1()
        rate0expand, rate0escape, rate1expand, rate1escape = self.vecstarset.rateexpansions(jumpnetwork_omega1, jt)
        # rate1expand = self.vecstarset.rate1expansion(jumpnetwork)
        self.assertEqual(np.shape(rate1expand),
                         (self.vecstarset.Nvstars, self.vecstarset.Nvstars, len(jumpnetwork_omega1)))
        # make some random rates
        om1expand = np.random.uniform(0, 1, len(jumpnetwork_omega1))
        for i in range(self.vecstarset.Nvstars):
            for j in range(self.vecstarset.Nvstars):
                # test the construction
                om1 = 0
                for Ri, vi in zip(self.vecstarset.vecpos[i], self.vecstarset.vecvec[i]):
                    for Rj, vj in zip(self.vecstarset.vecpos[j], self.vecstarset.vecvec[j]):
                        for jumplist, rate in zip(jumpnetwork_omega1, om1expand):
                            for (IS, FS), dx in jumplist:
                                if IS == Ri:
                                    if IS == Rj: om1 -= np.dot(vi, vj) * rate  # onsite terms...
                                    if FS == Rj: om1 += np.dot(vi, vj) * rate
                om1_sv_comp = np.dot(rate1expand[i, j], om1expand)
                if i == j: om1_sv_comp += np.dot(rate1escape[i], om1expand)
                self.assertAlmostEqual(om1, om1_sv_comp,
                                       msg='Failed to match {}, {}: {} != {}'.format(
                                           i, j, om1, om1_sv_comp))
                # print(np.dot(rateexpand, om1expand))


class VectorStarFCCOmegalinearTests(VectorStarOmegalinearTests):
    """Set of tests for our expansion of omega_1 for FCC"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupFCC()
        # self.rates = FCCrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)


class VectorStarHCPOmegalinearTests(VectorStarOmegalinearTests):
    """Set of tests for our expansion of omega_1 for HCP"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupHCP()
        # self.rates = HCPrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)


class VectorStarBCCOmegalinearTests(VectorStarOmegalinearTests):
    """Set of tests for our expansion of omega_1 for BCC"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupBCC()
        # self.rates = HCPrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)


class VectorStarSquareOmegalinearTests(VectorStarOmegalinearTests):
    """Set of tests for our expansion of omega_1 for square"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupsquare()
        # self.rates = HCPrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)


class VectorStarB2OmegalinearTests(VectorStarOmegalinearTests):
    """Set of tests for our expansion of omega_1 for B2"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupB2()
        # self.rates = HCPrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)


class VectorStarOmega2linearTests(unittest.TestCase):
    """Set of tests for our expansion of omega_2"""

    longMessage = False

    def setUp(self):
        self.crys, self.jumpnetwork = setuportho()
        self.rates = orthorates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)

    def testConstructOmega2(self):
        self.starset.generate(2)  # we need at least 2nd nn to even have jump networks to worry about...
        self.vecstarset = stars.VectorStarSet(self.starset)
        jumpnetwork_omega2, jt, sp = self.starset.jumpnetwork_omega2()
        rate0expand, rate0escape, rate2expand, rate2escape = self.vecstarset.rateexpansions(jumpnetwork_omega2, jt)

        # construct the set of rates corresponding to the unique stars:
        om2expand = self.rates.copy()
        self.assertEqual(np.shape(rate2expand),
                         (self.vecstarset.Nvstars, self.vecstarset.Nvstars, len(jumpnetwork_omega2)))
        for i in range(self.vecstarset.Nvstars):
            for j in range(self.vecstarset.Nvstars):
                # test the construction
                om2 = 0
                for Ri, vi in zip(self.vecstarset.vecpos[i], self.vecstarset.vecvec[i]):
                    for Rj, vj in zip(self.vecstarset.vecpos[j], self.vecstarset.vecvec[j]):
                        for jumplist, rate in zip(jumpnetwork_omega2, om2expand):
                            for (IS, FS), dx in jumplist:
                                if IS == Ri:
                                    if IS == Rj: om2 -= np.dot(vi, vj) * rate  # onsite terms...
                                    if FS == Rj: om2 += np.dot(vi, vj) * rate
                om2_sv_comp = np.dot(rate2expand[i, j], om2expand)
                if i == j: om2_sv_comp += np.dot(rate2escape[i], om2expand)
                self.assertAlmostEqual(om2, om2_sv_comp,
                                       msg='Failed to match {}, {}: {} != {}'.format(
                                           i, j, om2, om2_sv_comp))


class VectorStarFCCOmega2linearTests(VectorStarOmega2linearTests):
    """Set of tests for our expansion of omega_2 for FCC"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupFCC()
        self.rates = FCCrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)


class VectorStarHCPOmega2linearTests(VectorStarOmega2linearTests):
    """Set of tests for our expansion of omega_2 for HCP"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupHCP()
        self.rates = HCPrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)


class VectorStarBCCOmega2linearTests(VectorStarOmega2linearTests):
    """Set of tests for our expansion of omega_2 for BCC"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupBCC()
        self.rates = BCCrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)


class VectorStarSquareOmega2linearTests(VectorStarOmega2linearTests):
    """Set of tests for our expansion of omega_2 for square"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupsquare()
        self.rates = squarerates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)


class VectorStarB2Omega2linearTests(VectorStarOmega2linearTests):
    """Set of tests for our expansion of omega_2 for B2"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupB2()
        self.rates = B2rates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)


class VectorStarBias2linearTests(unittest.TestCase):
    """Set of tests for our expansion of bias vector (2)"""

    longMessage = False

    def setUp(self):
        self.crys, self.jumpnetwork = setuportho()
        self.rates = orthorates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)

    def testConstructBias2(self):
        dim = self.crys.dim
        self.starset.generate(2)  # we need at least 2nd nn to even have jump networks to worry about...
        self.vecstarset = stars.VectorStarSet(self.starset)
        jumpnetwork_omega2, jt, sp = self.starset.jumpnetwork_omega2()
        bias0expand, bias2expand = self.vecstarset.biasexpansions(jumpnetwork_omega2, jt)
        # make omega2 twice omega0:
        alpha = 2.
        om2expand = alpha * self.rates
        om0expand = self.rates.copy()
        self.assertEqual(np.shape(bias2expand),
                         (self.vecstarset.Nvstars, len(self.jumpnetwork)))
        biasvec = np.zeros((self.starset.Nstates, dim))  # bias vector: only the exchange hops
        for jumplist, rate in zip(jumpnetwork_omega2, self.rates):
            for (IS, FS), dx in jumplist:
                for i in range(self.starset.Nstates):
                    if IS == i:
                        biasvec[i, :] += dx * (alpha - 1) * rate
        # construct the same bias vector using our expansion
        biasveccomp = np.zeros((self.starset.Nstates, dim))
        for b2, b0, svpos, svvec in zip(np.dot(bias2expand, om2expand),
                                        np.dot(bias0expand, om0expand),
                                        self.vecstarset.vecpos,
                                        self.vecstarset.vecvec):
            # test the construction
            for Ri, vi in zip(svpos, svvec):
                biasveccomp[Ri, :] += (b2 - b0) * vi
        for i in range(self.starset.Nstates):
            self.assertTrue(np.allclose(biasvec[i], biasveccomp[i]),
                            msg='Failure for state {}: {}\n{} != {}'.format(
                                i, self.starset.states[i], biasvec[i], biasveccomp[i]))


class VectorStarFCCBias2linearTests(VectorStarBias2linearTests):
    """Set of tests for our expansion of bias vector (2) for FCC"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupFCC()
        self.rates = FCCrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)


class VectorStarHCPBias2linearTests(VectorStarBias2linearTests):
    """Set of tests for our expansion of bias vector (2) for HCP"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupHCP()
        self.rates = HCPrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)


class VectorStarBCCBias2linearTests(VectorStarBias2linearTests):
    """Set of tests for our expansion of bias vector (2) for BCC"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupBCC()
        self.rates = BCCrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)


class VectorStarSquareBias2linearTests(VectorStarBias2linearTests):
    """Set of tests for our expansion of bias vector (2) for square"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupsquare()
        self.rates = squarerates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)


class VectorStarB2Bias2linearTests(VectorStarBias2linearTests):
    """Set of tests for our expansion of bias vector (2) for B2"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupB2()
        self.rates = B2rates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)


class VectorStarBias1linearTests(unittest.TestCase):
    """Set of tests for our expansion of bias vector (1)"""

    longMessage = False

    def setUp(self):
        self.crys, self.jumpnetwork = setuportho()
        self.rates = orthorates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)

    def testConstructBias1(self):
        """Do we construct our omega1 bias correctly?"""
        dim = self.crys.dim
        self.starset.generate(2)  # we need at least 2nd nn to even have jump networks to worry about...
        self.vecstarset = stars.VectorStarSet(self.starset)
        jumpnetwork_omega1, jt, sp = self.starset.jumpnetwork_omega1()
        bias0expand, bias1expand = self.vecstarset.biasexpansions(jumpnetwork_omega1, jt)
        om1expand = np.random.uniform(0, 1, len(jumpnetwork_omega1))
        om0expand = self.rates.copy()
        self.assertEqual(np.shape(bias1expand),
                         (self.vecstarset.Nvstars, len(jumpnetwork_omega1)))
        biasvec = np.zeros((self.starset.Nstates, dim))  # bias vector: only the exchange hops
        for jumplist, rate, om0type in zip(jumpnetwork_omega1, om1expand, jt):
            om0 = om0expand[om0type]
            for (IS, FS), dx in jumplist:
                for i in range(self.starset.Nstates):
                    if IS == i:
                        biasvec[i, :] += dx * (rate - om0)
        # construct the same bias vector using our expansion
        biasveccomp = np.zeros((self.starset.Nstates, dim))
        for b1, b0, svpos, svvec in zip(np.dot(bias1expand, om1expand),
                                        np.dot(bias0expand, om0expand),
                                        self.vecstarset.vecpos,
                                        self.vecstarset.vecvec):
            # test the construction
            for Ri, vi in zip(svpos, svvec):
                biasveccomp[Ri, :] += (b1 - b0) * vi
        for i in range(self.starset.Nstates):
            self.assertTrue(np.allclose(biasvec[i], biasveccomp[i]),
                            msg='Failure for state {}: {}\n{} != {}'.format(
                                i, self.starset.states[i], biasvec[i], biasveccomp[i]))

    def testPeriodicBias(self):
        """Do we have no periodic bias?"""
        dim = self.crys.dim
        vectorbasislist = self.crys.FullVectorBasis(self.chem)[0]  # just check the VB list
        # we *should* have some projection if there's a vectorbasis, so only continue if this is empty
        if len(vectorbasislist) != 0: return
        self.starset.generate(1, originstates=True)  # turn on origin state generation
        self.vecstarset = stars.VectorStarSet(self.starset)
        for elemtype in ('solute', 'vacancy'):
            OSindices, folddown, OS_VB = self.vecstarset.originstateVectorBasisfolddown(elemtype)
            NOS = len(OSindices)
            self.assertEqual(NOS, 0)
            self.assertEqual(folddown.shape, (NOS, self.vecstarset.Nvstars))
            self.assertEqual(OS_VB.shape, (NOS, len(self.crys.basis[self.chem]), dim))


class VectorStarFCCBias1linearTests(VectorStarBias1linearTests):
    """Set of tests for our expansion of bias vector (1) for FCC"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupFCC()
        self.rates = FCCrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)


class VectorStarHCPBias1linearTests(VectorStarBias1linearTests):
    """Set of tests for our expansion of bias vector (1) for HCP"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupHCP()
        self.rates = HCPrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)


class VectorStarBCCBias1linearTests(VectorStarBias1linearTests):
    """Set of tests for our expansion of bias vector (1) for BCC"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupBCC()
        self.rates = BCCrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)


class VectorStarSquareBias1linearTests(VectorStarBias1linearTests):
    """Set of tests for our expansion of bias vector (1) for square"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupsquare()
        self.rates = squarerates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)


class VectorStarB2Bias1linearTests(VectorStarBias1linearTests):
    """Set of tests for our expansion of bias vector (1) for B2"""

    def setUp(self):
        self.crys, self.jumpnetwork = setupB2()
        self.rates = B2rates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)


class VectorStarPeriodicBias(unittest.TestCase):
    """Set of tests for our expansion of periodic bias vector (1)"""

    longMessage = False

    def setUp(self):
        self.crys, self.jumpnetwork = setupB2()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)

    def testOriginStates(self):
        """Does our origin state treatment work correctly to produce the vector bases?"""
        self.starset.generate(2, originstates=True)  # we need at least 2nd nn to even have jump networks to worry about...
        self.vecstarset = stars.VectorStarSet(self.starset)
        vectorbasislist = self.crys.FullVectorBasis(self.chem)[0]
        NVB = len(vectorbasislist)
        for elemtype, attr in zip(['vacancy', 'solute'], ['j', 'i']):
            OSindices, folddown, OS_VB = self.vecstarset.originstateVectorBasisfolddown(elemtype)
            NOS = len(OSindices)
            self.assertEqual(NOS, NVB)
            self.assertEqual(folddown.shape, (NOS, self.vecstarset.Nvstars))
            self.assertEqual(OS_VB.shape, (NOS,) + vectorbasislist[0].shape)
            for n, svR in enumerate(self.vecstarset.vecpos):
                if n in OSindices:
                    for i in svR:
                        self.assertTrue(self.starset.states[i].iszero())
                else:
                    for i in svR:
                        self.assertFalse(self.starset.states[i].iszero())
            for n in range(10):
                # test that our OS in our VectorStar make a proper basis according to our VB:
                vb = sum((2. * u - 1) * vect for u, vect in zip(np.random.random(len(vectorbasislist)), vectorbasislist))
                vb_proj = np.tensordot(OS_VB, np.tensordot(OS_VB, vb, axes=((1, 2), (0, 1))), axes=(0, 0))
                self.assertTrue(np.allclose(vb, vb_proj))
                # expand out to all sites:
                svexp = np.dot(folddown.T, np.tensordot(OS_VB, vb, axes=((1, 2), (0, 1))))
                vbdirect = np.array([vb[getattr(PS, attr)] for PS in self.starset.states])
                vbexp = np.zeros((self.starset.Nstates, 3))
                for svcoeff, svR, svv in zip(svexp, self.vecstarset.vecpos, self.vecstarset.vecvec):
                    for s, v in zip(svR, svv):
                        vbexp[s, :] += v * svcoeff
                self.assertTrue(np.allclose(vbdirect, vbexp))

