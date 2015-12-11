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
    lattice = crystal.Crystal.FCC(2.)
    jumpnetwork = lattice.jumpnetwork(0, 2.*np.sqrt(0.5)+0.01)
    return lattice, jumpnetwork

def FCCrates():
    return np.array([1./12.])

def setupHCP():
    lattice = crystal.Crystal.HCP(1.)
    jumpnetwork = lattice.jumpnetwork(0, 1.01)
    return lattice, jumpnetwork

def HCPrates():
    return np.array([1./12., 1./12.])


class PairStateTests(unittest.TestCase):
    """Tests of the PairState class"""
    longMessage=False

    def setUp(self):
        self.hcp = crystal.Crystal.HCP(1.0)

    def testZero(self):
        """Is zero consistent?"""
        zero = stars.PairState.zero()
        self.assertTrue(zero.iszero())
        self.assertTrue(stars.PairState.fromcrys(self.hcp, 0, (0,0), np.zeros(3)).iszero())

    def testArithmetic(self):
        """Does addition,subtraction, and endpoint subtraction work as expected?"""
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
        ps4 = stars.PairState.fromcrys(self.hcp, 0, (0,1), pos2a1 - pos1)
        self.assertEqual(ps1 - ps2, ps4)
        for i in [ps1, ps2, ps3, ps4]:
            self.assertTrue((i - i).iszero())
            self.assertTrue((i ^ i).iszero())  # endpoint subtraction!
        # endpoint subtraction: 0.(000):1.(010) ^ 0.(000):1.(000)  == 1.(000):1.(010)
        ps1 = stars.PairState.fromcrys(self.hcp, 0, (0,1), pos2a2 - pos1)
        ps2 = stars.PairState.fromcrys(self.hcp, 0, (0,1), pos2 - pos1)
        ps3 = stars.PairState.fromcrys(self.hcp, 0, (1,1), pos2a2 - pos2)
        self.assertEqual(ps1 ^ ps2, ps3)
        with self.assertRaises(ArithmeticError):
            ps1 ^ ps3
        self.assertEqual(stars.PairState.fromcrys(self.hcp, 0, (1,0), pos1 - pos2),
                         - stars.PairState.fromcrys(self.hcp, 0, (0,1), pos2 - pos1))

    def testGroupOps(self):
        """Applying group operations?"""
        pos1 = self.hcp.pos2cart(np.array([0,0,0]), (0,0))  # first site
        pos2 = self.hcp.pos2cart(np.array([0,0,0]), (0,1))  # second site
        pos1a1 = self.hcp.pos2cart(np.array([1,0,0]), (0,0))  # first + a1
        pos2a1 = self.hcp.pos2cart(np.array([1,0,0]), (0,1))  # second + a1
        pos1a2 = self.hcp.pos2cart(np.array([0,1,0]), (0,0))  # first + a2
        pos2a2 = self.hcp.pos2cart(np.array([0,1,0]), (0,1))  # second + a2
        pos1a3 = self.hcp.pos2cart(np.array([0,0,1]), (0,0))  # first + a3
        pos2a3 = self.hcp.pos2cart(np.array([0,0,1]), (0,1))  # second + a3
        iterlist = [(0,pos1), (1,pos2),
                    (0,pos1a1), (1,pos2a1),
                    (0,pos1a2), (1,pos2a2),
                    (0,pos1a3), (1,pos2a3)]
        zero = stars.PairState.zero()
        for g in self.hcp.G:
            self.assertTrue((zero.g(self.hcp, 0, g)).iszero())
        for i,x1 in iterlist:
            for j,x2 in iterlist:
                ps = stars.PairState.fromcrys(self.hcp, 0, (i,j), x2-x1)
                self.assertTrue(ps.__sane__(self.hcp, 0), msg="{} is not sane?".format(ps))
                for g in self.hcp.G:
                    # apply directly to the vectors, and get the site index from the group op
                    gi, gx1 = g.indexmap[0][i], self.hcp.g_cart(g, x1)
                    gj, gx2 = g.indexmap[0][j], self.hcp.g_cart(g, x2)
                    gpsdirect = stars.PairState.fromcrys(self.hcp, 0, (gi,gj), gx2-gx1)
                    gps = ps.g(self.hcp, 0, g)
                    self.assertEqual(gps, gpsdirect,
                                     msg="{}\n*{} =\n{} !=\n{}".format(g, ps, gps, gpsdirect))


class StarTests(unittest.TestCase):
    """Set of tests that our star code is behaving correctly for a general materials"""
    longMessage=False

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
                if not any( ps1 == ps2.g(self.crys, self.chem, g) for g in self.crys.G):
                    return False
        return True

    def testStarConsistent(self):
        """Check that the counts (Npts, Nstars) make sense, with Nshells = 1..4"""
        for n in range(1,5):
            self.starset.generate(n)
            for starindex in range(self.starset.Nstars):
                self.assertTrue(self.isclosed(self.starset, starindex))

    def testStarindices(self):
        """Check that our indexing is correct."""
        self.starset.generate(4)
        for si, star in enumerate(self.starset.stars):
            for i in star:
                self.assertEqual(si, self.starset.starindex(self.starset.states[i]))
        self.assertEqual(None, self.starset.starindex(stars.PairState.zero()))
        self.assertEqual(None, self.starset.stateindex(stars.PairState.zero()))

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
        for n in range(1,5):
            self.starset.generate(n)
            for starindex in range(self.starset.Nstars):
                self.assertTrue(self.isclosed(self.starset, starindex))
            self.assertEqual(None, self.starset.starindex(stars.PairState.zero()))
            self.assertEqual(None, self.starset.stateindex(stars.PairState.zero()))

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
        self.assertEqual(dS.Nstars, 4+4+2)

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
        self.assertEqual(len(jumpnetwork1),1)
        self.assertEqual(jt[0], 0)
        self.assertEqual(sp[0], (0,0))
        jumpnetwork2, jt, sp = self.starset.jumpnetwork_omega2()
        self.assertEqual(len(jumpnetwork2),1)
        self.assertEqual(len(jumpnetwork2[0]), len(self.jumpnetwork[0]))
        self.assertEqual(jt[0], 0)
        self.assertEqual(sp[0], (0,0))

    def testJumpNetworkCount(self):
        """Check that the counts in the jumpnetwork make sense for FCC, with Nshells = 1, 2"""
        # each of the 12 <110> pairs to 101, 10-1, 011, 01-1 = 4, so should be 48 pairs
        # (which includes "double counting": i->j and j->i)
        # but *all* of those 48 are all equivalent to each other by symmetry: one double-star.
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
        self.assertEqual(len(jumpnetwork), 4+1+2)
        self.assertEqual(sum(len(jlist) for jlist in jumpnetwork), 12*11 + 6*8 + 24*7 + 12*5)
        # check that nothing changed with the larger StarSet
        jumpnetwork2, jt, sp = self.starset.jumpnetwork_omega2()
        self.assertEqual(len(jumpnetwork2),1)
        self.assertEqual(len(jumpnetwork2[0]), len(self.jumpnetwork[0]))
        self.assertEqual(jt[0], 0)
        self.assertEqual(sp[0], (0,0))

    def testJumpNetworkindices(self):
        """Check that our indexing works correctly for Nshell=1..3"""
        for nshells in range(1, 4):
            self.starset.generate(nshells)
            jumpnetwork, jt, sp = self.starset.jumpnetwork_omega1()
            for jumplist, (s1,s2) in zip(jumpnetwork, sp):
                for (i,f), dx in jumplist:
                    si = self.starset.index[i]
                    sf = self.starset.index[f]
                    self.assertTrue((s1,s2) == (si,sf) or (s1,s2) == (sf,si))


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
        self.assertTrue(self.vecstarset.Nvstars>0)

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
        self.starset.generate(1)
        self.vecstarset = stars.VectorStarSet(self.starset)
        self.assertEqual(np.shape(self.vecstarset.outer),
                         (3, 3, self.vecstarset.Nvstars, self.vecstarset.Nvstars))
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
                testouter = np.zeros((3, 3))
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
        testouter = 1./3.*np.eye(3)
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
        self.assertEqual(self.vecstarset.Nvstars, 2+2)


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
                self.assertAlmostEqual(sum(GFexpand[i, j, :]), 0)
                g = 0
                for si, vi in zip(self.vecstarset.vecpos[i], self.vecstarset.vecvec[i]):
                    for sj, vj in zip(self.vecstarset.vecpos[j], self.vecstarset.vecvec[j]):
                        try: ds = self.starset.states[sj] ^ self.starset.states[si]
                        except: continue
                        g += np.dot(vi, vj)*self.GF(ds.i, ds.j, ds.dx)
                self.assertAlmostEqual(g, np.dot(GFexpand[i, j, :], gexpand))
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
    """Set of tests that make sure we can construct the GF matrix as a linear combination for FCC"""
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


class VectorStarOmega0Tests(unittest.TestCase):
    """Set of tests for our expansion of omega_0 in NN vectors"""

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
        self.starset.generate(2) # we need at least 2nd nn to even have double-stars to worry about...
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
                for TS in [ self.starset.jumplist[jumpindex] for jumpindex in jumplist]:
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
                        om0_sv += np.dot(v0, v1)*om0matrix[i0,i1]
                om0_sv_comp = np.dot(rate0expand[i, j], om0expand)
                if i == j: om0_sv_comp += np.dot(rate0escape[i], om0expand)
                self.assertAlmostEqual(om0_sv, om0_sv_comp,
                                       msg='Failed to match {}, {}: {} != {}'.format(
                                           i, j, om0_sv, om0_sv_comp))

class VectorStarFCCOmega0Tests(VectorStarOmega0Tests):
    """Set of tests for our expansion of omega_0 in NN vect for FCC"""
    def setUp(self):
        self.crys, self.jumpnetwork = setupFCC()
        self.rates = FCCrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)

class VectorStarHPCOmega0Tests(VectorStarOmega0Tests):
    """Set of tests for our expansion of omega_0 in NN vect for FCC"""
    def setUp(self):
        self.crys, self.jumpnetwork = setupHCP()
        self.rates = HCPrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)


class VectorStarOmegalinearTests(unittest.TestCase):
    """Set of tests for our expansion of omega_1 in double-stars"""

    longMessage = False
    def setUp(self):
        self.crys, self.jumpnetwork = setuportho()
        # self.rates = orthorates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)

    def testConstructOmega1(self):
        self.starset.generate(2) # we need at least 2nd nn to even have double-stars to worry about...
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
    """Set of tests for our expansion of omega_1 in double-stars for FCC"""
    def setUp(self):
        self.crys, self.jumpnetwork = setupFCC()
        # self.rates = FCCrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)

class VectorStarHPCOmegalinearTests(VectorStarOmegalinearTests):
    """Set of tests for our expansion of omega_1 in double-stars for FCC"""
    def setUp(self):
        self.crys, self.jumpnetwork = setupHCP()
        # self.rates = HCPrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)


class VectorStarOmega2linearTests(unittest.TestCase):
    """Set of tests for our expansion of omega_2 in NN stars"""

    longMessage = False
    def setUp(self):
        self.crys, self.jumpnetwork = setuportho()
        self.rates = orthorates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)

    def testConstructOmega2(self):
        self.starset.generate(2) # we need at least 2nd nn to even have double-stars to worry about...
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
    """Set of tests for our expansion of omega_2 in NN stars for FCC"""
    def setUp(self):
        self.crys, self.jumpnetwork = setupFCC()
        self.rates = FCCrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)

class VectorStarHCPOmega2linearTests(VectorStarOmega2linearTests):
    """Set of tests for our expansion of omega_2 in NN stars for FCC"""
    def setUp(self):
        self.crys, self.jumpnetwork = setupHCP()
        self.rates = HCPrates()
        self.chem = 0
        self.sitelist = self.crys.sitelist(self.chem)
        self.starset = stars.StarSet(self.jumpnetwork, self.crys, self.chem)


class VectorStarBias2linearTests(unittest.TestCase):
    """Set of tests for our expansion of bias vector (2) in NN stars"""
    def setUp(self):
        self.lattice, self.NNvect, self.groupops, self.star = setuportho()
        self.NNstar = stars.StarSet(self.NNvect, self.groupops)
        self.vecstar = stars.VectorStarSet()
        self.rates = orthorates()

    def TESTConstructBias2(self):
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

    def TESTConstructBias1(self):
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
