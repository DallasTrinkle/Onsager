"""
Unit tests for Onsager calculator for vacancy-mediated diffusion.
"""

__author__ = 'Dallas R. Trinkle'

#

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


class BaseTests(unittest.TestCase):
    """Set of tests that our Onsager calculator is well-behaved"""

    def setUp(self):
        self.lattice, self.NNvect, self.groupops, self.rates = setuportho()
        self.Lcalc = OnsagerCalc.VacancyMediated(self.NNvect, self.groupops)
        self.Njumps = 3
        self.Ninteract = [3, 9]

    def testGenerate(self):
        # try to generate with a single interaction shell
        self.Lcalc.generate(1)
        jumplist = self.Lcalc.omega0list()
        # we expect to have three unique jumps to calculate:
        self.assertEqual(len(jumplist), self.Njumps)
        for j in jumplist:
            self.assertTrue(any([ (v == j).all() for v in self.NNvect]))
        for i, n in enumerate(self.Ninteract):
            self.Lcalc.generate(i+1)
            interactlist = self.Lcalc.interactlist()
            self.assertEqual(len(interactlist), n)
            for v in self.NNvect:
                self.assertTrue(any([all(abs(v - np.dot(g, R)) < 1e-8)
                                     for R in interactlist
                                     for g in self.groupops]))


class FCCBaseTests(BaseTests):
    """Set of tests that our Onsager calculator is well-behaved"""

    def setUp(self):
        self.lattice, self.NNvect, self.groupops, self.rates = setupFCC()
        self.Lcalc = OnsagerCalc.VacancyMediated(self.NNvect, self.groupops)
        self.Njumps = 1
        self.Ninteract = [1, 4]


class SCBaseTests(BaseTests):
    """Set of tests that our Onsager calculator is well-behaved"""

    def setUp(self):
        self.lattice, self.NNvect, self.groupops, self.rates = setupcubic()
        self.Lcalc = OnsagerCalc.VacancyMediated(self.NNvect, self.groupops)
        self.Njumps = 1
        self.Ninteract = [1, 3]
