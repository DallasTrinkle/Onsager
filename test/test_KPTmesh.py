"""
Unit tests for k-point mesh generation including symmetry
"""

__author__ = 'Dallas R. Trinkle'

import unittest
import numpy as np
import KPTmesh


class KPTMeshTests(unittest.TestCase):
    """Set of tests for our kpt-mesh generation class"""

    def setUp(self):
        self.lattice = np.eye(3)
        self.N = (4, 4, 4)
        self.kptmesh = KPTmesh.KPTmesh(self.lattice)

    def testKPTclassexists(self):
        """Does it exist?"""
        self.assertIsInstance(self.kptmesh, KPTmesh.KPTmesh)

    def testKPTreciprocallattice(self):
        """Have we correctly constructed the reciprocal lattice vectors?"""
        dotprod = np.dot(self.kptmesh.rlattice.T, self.kptmesh.lattice)
        dotprod0 = 2. * np.pi * np.eye(3)
        for a in xrange(3):
            for b in xrange(3):
                self.assertAlmostEqual(dotprod[a, b], dotprod0[a, b])

    def testKPTvolume(self):
        """Correct volume for cell?"""
        self.assertAlmostEqual(1., self.kptmesh.volume)

    def testKPTconstruct(self):
        """Can we construct a mesh with the correct number of points?"""
        # reset
        self.kptmesh.genmesh((0, 0, 0))
        self.assertEqual(self.kptmesh.Nkpt, 0)
        self.kptmesh.genmesh(self.N)
        self.assertEqual(self.kptmesh.Nkpt, np.product(self.N))

    def testKPT_BZ_Gpoints(self):
        """Do we have the correct G points that define the BZ?"""
        self.assertEqual(np.shape(self.kptmesh.BZG), (6, 3))
        self.assertTrue(any(all((np.pi, 0, 0) == x) for x in self.kptmesh.BZG))
        self.assertTrue(any(all((-np.pi, 0, 0) == x) for x in self.kptmesh.BZG))
        self.assertTrue(any(all((0, np.pi, 0) == x) for x in self.kptmesh.BZG))
        self.assertTrue(any(all((0, -np.pi, 0) == x) for x in self.kptmesh.BZG))
        self.assertTrue(any(all((0, 0, np.pi) == x) for x in self.kptmesh.BZG))
        self.assertTrue(any(all((0, 0, -np.pi) == x) for x in self.kptmesh.BZG))
        self.assertFalse(any(all((0, 0, 0) == x) for x in self.kptmesh.BZG))
        vec = np.array((1, 1, 1))
        self.assertTrue(self.kptmesh.incell(vec))
        vec = np.array((4, 0, -4))
        self.assertFalse(self.kptmesh.incell(vec))

    def testKPT_fullmesh_points(self):
        """Are the points in the k-point mesh that we expect to see?"""
        self.kptmesh.genmesh(self.N)
        kpts, wts = self.kptmesh.fullmesh()
        self.assertAlmostEqual(sum(wts), 1)
        self.assertAlmostEqual(wts[0], 1. / self.kptmesh.Nkpt)
        self.assertTrue(any(all((2. * np.pi / self.N[0], 0, 0) == x) for x in kpts))

    def testKPT_insideBZ(self):
        """Do we only have points that are inside the BZ?"""
        self.kptmesh.genmesh(self.N)
        kpts, wts = self.kptmesh.fullmesh()
        for q in kpts:
            self.assertTrue(self.kptmesh.incell(q),
                            msg="Failed with vector {} not in BZ".format(q))

    def testKPT_symmetry(self):
        """Do we have the correct number of point group operations? Are they unique? Are they symmetries?"""
        self.assertEqual(np.shape(self.kptmesh.groupops), (48, 3, 3))
        binv = np.linalg.inv(self.kptmesh.rlattice)
        for g in self.kptmesh.groupops:
            self.assertAlmostEqual(abs(np.linalg.det(g)), 1)
        for i, g1 in enumerate(self.kptmesh.groupops):
            for g2 in self.kptmesh.groupops[:i]:
                self.assertFalse(np.all(g1 == g2),
                                 msg="Group operations {} and {} are duplicated in kptmesh".format(g1, g2))
            bgb = np.dot(binv, np.dot(g1, self.kptmesh.rlattice))
            self.assertTrue(np.all(bgb == np.round(bgb)))

    def testKPT_IRZ(self):
        """Do we produce a correct irreducible wedge?"""
        self.kptmesh.genmesh(self.N)
        kpts, wts = self.kptmesh.symmesh()
        self.assertAlmostEqual(sum(wts), 1)
        for i, k in enumerate(kpts):
            # We want to determine what the weight for each point should be, and compare
            # dealing with the BZ edges is complicated; so we skip that in our tests
            if all([np.dot(k, G) < (np.dot(G, G) - 1e-8) for G in self.kptmesh.BZG]):
                basewt = 1. / self.kptmesh.Nkpt
                sortk = sorted(k)
                basewt *= (2 ** (3 - list(k).count(0)))
                if sortk[0] != sortk[1] and sortk[1] != sortk[2]:
                    basewt *= 6
                elif sortk[0] != sortk[1] or sortk[1] != sortk[2]:
                    basewt *= 3
                self.assertAlmostEqual(basewt, wts[i])
        # integration test
        kptfull, wtfull = self.kptmesh.fullmesh()
        self.assertAlmostEqual(sum(wtfull * [np.cos(sum(k)) for k in kptfull]),
                               sum(wts * [np.cos(sum(k)) for k in kpts]))
        self.assertNotAlmostEquals(sum(wtfull * [np.cos(k[0]) for k in kptfull]),
                                   sum(wts * [np.cos(k[0]) for k in kpts]))

    def testKPT_integration(self):
        """Do we get integral values that we expect? 1/(2pi)^3 int cos(kx+ky+kz)^3 = 1/2"""
        self.kptmesh.genmesh(self.N)
        kptfull, wtfull = self.kptmesh.fullmesh()
        kpts, wts = self.kptmesh.symmesh()
        self.assertAlmostEqual(sum(wtfull * [np.cos(sum(k)) ** 2 for k in kptfull]), 0.5)
        self.assertAlmostEqual(sum(wts * [np.cos(sum(k)) ** 2 for k in kpts]), 0.5)
        self.assertAlmostEqual(sum(wtfull * [np.cos(sum(k)) for k in kptfull]), 0)
        self.assertAlmostEqual(sum(wts * [np.cos(sum(k)) for k in kpts]), 0)
        # Note: below we have the true values of the integral, but these should disagree
        # due to numerical error.
        self.assertNotAlmostEqual(sum(wtfull * [sum(k) ** 2 for k in kptfull]), 9.8696044010893586188)
        self.assertNotAlmostEqual(sum(wts * [sum(k) ** 2 for k in kpts]), 9.8696044010893586188)
