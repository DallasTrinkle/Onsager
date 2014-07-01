"""
Unit tests for Onsager calculator for vacancy-mediated diffusion.
"""

__author__ = 'Dallas R. Trinkle'

# TODO: add the five-frequency model as a test for FCC; add in BCC
# TODO: additional tests using the 14 frequency model for FCC?

import unittest
import FCClatt
import KPTmesh
import numpy as np
import stars
import OnsagerCalc


# Setup for orthorhombic, simple cubic, and FCC cells; different than test_stars
def setuportho():
    lattice = np.array([[3, 0, 0],
                        [0, 2, 0],
                        [0, 0, 1]], dtype=float)
    NNvect = np.array([[3, 0, 0], [-3, 0, 0],
                       [0, 2, 0], [0, -2, 0],
                       [0, 0, 1], [0, 0, -1]], dtype=float)
    groupops = KPTmesh.KPTmesh(lattice).groupops
    rates = np.array([3, 3, 2, 2, 1, 1], dtype=float)
    return lattice, NNvect, groupops, rates

def setupcubic():
    lattice = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]], dtype=float)
    NNvect = np.array([[1, 0, 0], [-1, 0, 0],
                       [0, 1, 0], [0, -1, 0],
                       [0, 0, 1], [0, 0, -1]], dtype=float)
    groupops = KPTmesh.KPTmesh(lattice).groupops
    rates = np.array([1./6.,]*6, dtype=float)
    return lattice, NNvect, groupops, rates

def setupFCC():
    lattice = FCClatt.lattice()
    NNvect = FCClatt.NNvect()
    groupops = KPTmesh.KPTmesh(lattice).groupops
    rates = np.array([1./12.,]*12, dtype=float)
    return lattice, NNvect, groupops, rates

def setupBCC():
    lattice = np.array([[-1, 1, 1],
                        [1, -1, 1],
                        [1, 1, -1]], dtype=float)
    NNvect = np.array([[-1, 1, 1], [1, -1, -1],
                       [1, -1, 1], [-1, 1, -1],
                       [1, 1, -1], [-1, -1, 1],
                       [1, 1, 1], [-1, -1, -1]], dtype=float)
    groupops = KPTmesh.KPTmesh(lattice).groupops
    rates = np.array([1./8.,]*8, dtype=float)
    return lattice, NNvect, groupops, rates


class BaseTests(unittest.TestCase):
    """Set of tests that our Onsager calculator is well-behaved"""

    def setUp(self):
        self.lattice, self.NNvect, self.groupops, self.rates = setuportho()
        self.Lcalc = OnsagerCalc.VacancyMediated(self.NNvect, self.groupops)
        self.Njumps = 3
        self.Ninteract = [3, 9]
        self.Nomega1 = [9, 27]
        self.NGF = [35, 84]

    def testGenerate(self):
        # try to generate with a single interaction shell
        self.Lcalc.generate(1)
        jumplist = self.Lcalc.omega0list()
        # we expect to have three unique jumps to calculate:
        self.assertEqual(len(jumplist), self.Njumps)
        for j in jumplist:
            self.assertTrue(any([ (v == j).all() for v in self.NNvect]))
        for thermo, nint, nom1, nGF in zip(range(len(self.Ninteract)),
                                           self.Ninteract,
                                           self.Nomega1,
                                           self.NGF):
            self.Lcalc.generate(thermo+1)
            interactlist = self.Lcalc.interactlist()
            self.assertEqual(len(interactlist), nint)
            for v in self.NNvect:
                self.assertTrue(any([all(abs(v - np.dot(g, R)) < 1e-8)
                                     for R in interactlist
                                     for g in self.groupops]))
            omega1list, omega1index = self.Lcalc.omega1list()
            # print 'omega1list: [{}]'.format(len(omega1list))
            # for pair in omega1list:
            #     print pair[0], pair[1]
            self.assertEqual(len(omega1list), nom1)
            self.assertEqual(len(omega1index), nom1)
            for vecpair, omindex in zip(omega1list, omega1index):
                jump = jumplist[omindex]
                self.assertTrue(any([all(abs(np.dot(g, jump) - (vecpair[0]-vecpair[1])) < 1e-8)
                                     for g in self.groupops]))
            GFlist = self.Lcalc.GFlist()
            self.assertEqual(len(GFlist), nGF)
            # we test this by making the list of endpoints from omega1, and doing all
            # possible additions with point group ops, and making sure it shows up.
            vlist = [pair[0] for pair in omega1list] + [pair[1] for pair in omega1list]
            for v1 in vlist:
                for gv in [np.dot(g, v1) for g in self.groupops]:
                    for v2 in vlist:
                        vsum = gv + v2
                        match = False
                        for vGF in GFlist:
                            if np.dot(vGF, vGF) != np.dot(vsum, vsum):
                                continue
                            if any([all(abs(vsum - np.dot(g, vGF)) < 1e-8) for g in self.groupops]):
                                match = True
                                break
                        self.assertTrue(match)

    def testTracerIndexing(self):
        """Test out the generation of the tracer example indexing."""
        for n in [1, 2]:
            self.Lcalc.generate(n)
            prob, om2, om1 = self.Lcalc.maketracer()
            self.assertEqual(len(prob), len(self.Lcalc.interactlist()))
            self.assertEqual(len(om2), len(self.Lcalc.omega0list()))
            om1list, om1index = self.Lcalc.omega1list()
            self.assertEqual(len(om1), len(om1list))
            for p in prob:
                self.assertEqual(p, 1)
            for om in om2:
                self.assertEqual(om, -1)
            for om in om1:
                self.assertEqual(om, -1)

    def testTracerFake(self):
        """Test the (fake) evaluation of the tracer diffusion value."""
        for n in [1, 2]:
            self.Lcalc.generate(n)
            prob, om2, om1 = self.Lcalc.maketracer()
            om0 = np.array((1.,)*len(om2), dtype=float)
            gf = np.array((1.,)*len(self.Lcalc.GFlist()))
            Lvv, Lss, Lsv = self.Lcalc.Lij(gf, om0, prob, om2, om1)
            for lvv, lsv, lss in zip(Lvv.flat, Lsv.flat, Lss.flat):
                self.assertAlmostEqual(lvv, lsv)
                self.assertAlmostEqual(lvv, -lss)


import GFcalc


class SCBaseTests(BaseTests):
    """Set of tests that our Onsager calculator is well-behaved"""

    def setUp(self):
        self.lattice, self.NNvect, self.groupops, self.rates = setupcubic()
        self.Lcalc = OnsagerCalc.VacancyMediated(self.NNvect, self.groupops)
        self.Njumps = 1
        self.Ninteract = [1, 3]
        self.Nomega1 = [2, 6]
        self.NGF = [11, 23]
        self.GF = GFcalc.GFcalc(self.lattice, self.NNvect, self.rates)
        self.D0 = self.GF.D2[0, 0]
        self.correl = 0.653

    def testTracerValue(self):
        """Make sure we get the correct tracer correlation coefficients"""
        self.Lcalc.generate(1)
        prob, om2, om1 = self.Lcalc.maketracer()
        om0 = np.array((self.rates[0],)*len(om2), dtype=float)
        om2 = om0.copy()
        gf = np.array([self.GF.GF(R) for R in self.Lcalc.GFlist()])
        Lvv, Lss, Lsv = self.Lcalc.Lij(gf, om0, prob, om2, om1)
        # print 'Lvv:', Lvv
        # print 'Lss:', Lss
        # print 'Lsv:', Lsv
        for p in [(i, j) for i in range(3) for j in range(3) if i != j]:
            self.assertAlmostEqual(0, Lvv[p])
            self.assertAlmostEqual(0, Lss[p])
            self.assertAlmostEqual(0, Lsv[p])
        for L in [Lvv, Lss, Lsv]:
            self.assertAlmostEqual(L[0, 0], L[1, 1])
            self.assertAlmostEqual(L[1, 1], L[2, 2])
            self.assertAlmostEqual(L[2, 2], L[0, 0])
        # No solute drag, so Lsv = -Lvv; Lvv = normal vacancy diffusion
        # all correlation is in that geometric prefactor of Lss.
        self.assertAlmostEqual(Lvv[0, 0], self.D0)
        self.assertAlmostEqual(-Lsv[0, 0], self.D0)
        self.assertAlmostEqual(-Lss[0, 0]/Lsv[0, 0], self.correl, delta=1e-3)


class FCCBaseTests(SCBaseTests):
    """Set of tests that our Onsager calculator is well-behaved"""

    def setUp(self):
        self.lattice, self.NNvect, self.groupops, self.rates = setupFCC()
        self.Lcalc = OnsagerCalc.VacancyMediated(self.NNvect, self.groupops)
        self.Njumps = 1
        self.Ninteract = [1, 4]
        self.Nomega1 = [7, 20]
        self.NGF = [16, 37]
        self.GF = GFcalc.GFcalc(self.lattice, self.NNvect, self.rates)
        self.D0 = self.GF.D2[0, 0]
        self.correl = 0.781


class BCCBaseTests(SCBaseTests):
    """Set of tests that our Onsager calculator is well-behaved"""

    def setUp(self):
        self.lattice, self.NNvect, self.groupops, self.rates = setupBCC()
        self.Lcalc = OnsagerCalc.VacancyMediated(self.NNvect, self.groupops)
        self.Njumps = 1
        self.Ninteract = [1, 4]
        self.Nomega1 = [3, 20]
        self.NGF = [14, 37]
        self.GF = GFcalc.GFcalc(self.lattice, self.NNvect, self.rates)
        self.D0 = self.GF.D2[0, 0]
        self.correl = 0.727
