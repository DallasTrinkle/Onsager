import onsager.crystal as crystal
import numpy as np
from scipy import special
from onsager.GFcalc import *
from onsager.crystal import Crystal, pureDBContainer
from crysts import *
import unittest
"""
Tests to check if inhereting the original Greens Function calculator works for dumbbells
The test is imported directly from Prof. Dallas R. Trinkle's test suite, since this is an extension of the same.
We just need to ensure we did not break anything.
"""
def poleFT(di, u, pm, erfupm=-1):
    """
    Calculates the pole FT (excluding the volume prefactor) given the `di` eigenvalues,
    the value of u magnitude (available from unorm), and the pmax scaling factor.

    :param di: array [:]  eigenvalues of `D2`
    :param u: double  magnitude of u, from unorm() = x.D^-1.x
    :param pm: double  scaling factor pmax for exponential cutoff function
    :param erfupm: double, optional  value of erf(0.5*u*pm) (negative = not set, then its calculated)
    :return poleFT: double
        integral of Gaussian cutoff function corresponding to a l=0 pole;
        :math:`\\erf(0.5 u pm)/(4\\pi u \\sqrt{d1 d2 d3})` if u>0
        :math:`pm/(4\pi^3/2 \\sqrt{d1 d2 d3})` if u==0
    """

    if (u == 0):
        return 0.25 * pm / np.sqrt(np.product(di * np.pi))
    if (erfupm < 0):
        erfupm = special.erf(0.5 * u * pm)
    return erfupm * 0.25 / (np.pi * u * np.sqrt(np.product(di)))

class GreenFuncCrystalTests(unittest.TestCase):

    def test_cube(self):
        """Test on simple cubic lattice"""
        """Same tests as in FCC lattice in the original GFcalc, just using cubic structure"""

        famp0 = [np.array([1.,0.,0.])/np.linalg.norm(np.array([1.,0.,0.]))*0.126]
        family = [famp0]
        pdbcontainer = pureDBContainer(cube,0,family)
        jset,jind = pdbcontainer.jumpnetwork(0.3,0.01,0.01)
        cube_GF = GF_dumbbells(pdbcontainer, jind)
        preT = list(np.ones(len(jind)))
        betaeneT = list(np.ones(len(jind)))
        cube_GF.SetRates([1], [0], preT, betaeneT)
        # test the pole function:
        for u in np.linspace(0, 5, 21):
            pole_orig = cube_GF.crys.volume * poleFT(cube_GF.d, u, cube_GF.pmax)
            pole_new = cube_GF.g_Taylor_fnlu[(-2, 0)](u).real
            self.assertAlmostEqual(pole_orig, pole_new, places=15, msg="Pole (-2,0) failed for u={}".format(u))
        # test the discontinuity function:
        for u in np.linspace(0, 5, 21):
            disc_orig = cube_GF.crys.volume * (cube_GF.pmax / (2 * np.sqrt(np.pi))) ** 3 * \
                        np.exp(-(0.5 * u * cube_GF.pmax) ** 2) / np.sqrt(np.product(cube_GF.d))
            disc_new = cube_GF.g_Taylor_fnlu[(0, 0)](u).real
            self.assertAlmostEqual(disc_orig, disc_new, places=15, msg="Disc (0,0) failed for u={}".format(u))

    def test_BCC(self):
        """Tests on BCC lattice"""

        famp0 = [np.array([1., 1., 0.])/np.linalg.norm(np.array([1., 1., 0.]))*0.126]
        family = [famp0]
        pdbcontainer = pureDBContainer(Fe_bcc, 0, family)
        jset,jind = pdbcontainer.jumpnetwork(0.25, 0.01, 0.01)
        cube_GF = GF_dumbbells(pdbcontainer, jind)
        preT = list(np.ones(len(jind)))
        betaeneT = list(np.ones(len(jind)))
        cube_GF.SetRates([1], [0], preT, betaeneT)
        # test the pole function:
        for u in np.linspace(0, 5, 21):
            pole_orig = cube_GF.crys.volume * poleFT(cube_GF.d, u, cube_GF.pmax)
            pole_new = cube_GF.g_Taylor_fnlu[(-2, 0)](u).real
            self.assertAlmostEqual(pole_orig, pole_new, places=15, msg="Pole (-2,0) failed for u={}".format(u))
        # test the discontinuity function:
        for u in np.linspace(0, 5, 21):
            disc_orig = cube_GF.crys.volume * (cube_GF.pmax / (2 * np.sqrt(np.pi))) ** 3 * \
                        np.exp(-(0.5 * u * cube_GF.pmax) ** 2) / np.sqrt(np.product(cube_GF.d))
            disc_new = cube_GF.g_Taylor_fnlu[(0, 0)](u).real
            self.assertAlmostEqual(disc_orig, disc_new, places=15, msg="Disc (0,0) failed for u={}".format(u))

    def test_FCC(self):
        """Tests on FCC lattice"""

        famp0 = [np.array([1., 0., 0.])/np.linalg.norm(np.array([1., 0., 0.]))*0.126]
        family = [famp0]
        FCC = Crystal.FCC(0.3, "Ni")
        pdbcontainer = pureDBContainer(FCC, 0, family)
        jset,jind = pdbcontainer.jumpnetwork(0.22, 0.01, 0.01)
        cube_GF = GF_dumbbells(pdbcontainer, jind)
        preT = list(np.ones(len(jind)))
        betaeneT = list(np.ones(len(jind)))
        cube_GF.SetRates([1], [0], preT, betaeneT)
        # test the pole function:
        for u in np.linspace(0, 5, 21):
            pole_orig = cube_GF.crys.volume * poleFT(cube_GF.d, u, cube_GF.pmax)
            pole_new = cube_GF.g_Taylor_fnlu[(-2, 0)](u).real
            self.assertAlmostEqual(pole_orig, pole_new, places=15, msg="Pole (-2,0) failed for u={}".format(u))
        # test the discontinuity function:
        for u in np.linspace(0, 5, 21):
            disc_orig = cube_GF.crys.volume * (cube_GF.pmax / (2 * np.sqrt(np.pi))) ** 3 * \
                        np.exp(-(0.5 * u * cube_GF.pmax) ** 2) / np.sqrt(np.product(cube_GF.d))
            disc_new = cube_GF.g_Taylor_fnlu[(0, 0)](u).real
            self.assertAlmostEqual(disc_orig, disc_new, places=15, msg="Disc (0,0) failed for u={}".format(u))
