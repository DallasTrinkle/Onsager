"""
Unit tests for calculation of lattice Green function for diffusion
"""

__author__ = 'Dallas R. Trinkle'

import unittest
import numpy as np
from scipy import special
import onsager.GFcalc as GFcalc

def poleFT(di, u, pm, erfupm=-1):
    """
    Calculates the pole FT (excluding the volume prefactor) given the `di` eigenvalues,
    the value of u magnitude (available from unorm), and the pmax scaling factor.

    :param di : array [:]  eigenvalues of `D2`
    :param u : double  magnitude of u, from unorm() = x.D^-1.x
    :param pm : double  scaling factor pmax for exponential cutoff function
    :param erfupm : double, optional  value of erf(0.5*u*pm) (negative = not set, then its calculated)
    :return poleFT : double
        integral of Gaussian cutoff function corresponding to a l=0 pole;
        erf(0.5*u*pm)/(4*pi*u*sqrt(d1*d2*d3)) if u>0
        pm/(4*pi^3/2 * sqrt(d1*d2*d3)) if u==0
    """

    if (u == 0):
        return 0.25 * pm / np.sqrt(np.product(di * np.pi))
    if (erfupm < 0):
        erfupm = special.erf(0.5 * u * pm)
    return erfupm * 0.25 / (np.pi * u * np.sqrt(np.product(di)))


class GreenFuncCrystalTests(unittest.TestCase):
    """Test new implementation of GF calculator, based on Crystal class"""
    def setUp(self):
        self.FCC = GFcalc.crystal.Crystal.FCC(1.)
        self.HCP = GFcalc.crystal.Crystal.HCP(1., np.sqrt(8/3))
        self.FCC_sitelist = self.FCC.sitelist(0)
        self.FCC_jumpnetwork = self.FCC.jumpnetwork(0, 0.75)
        self.HCP_sitelist = self.HCP.sitelist(0)
        self.HCP_jumpnetwork = self.HCP.jumpnetwork(0, 1.01)

    def testFCC(self):
        """Test on FCC"""
        FCC_GF = GFcalc.GFCrystalcalc(self.FCC, 0, self.FCC_sitelist, self.FCC_jumpnetwork, Nmax=4)
        FCC_GF.SetRates([1],[0],[1],[0])
        # test the pole function:
        for u in np.linspace(0,5,21):
            pole_orig = FCC_GF.crys.volume*poleFT(FCC_GF.d, u, FCC_GF.pmax)
            pole_new = FCC_GF.g_Taylor_fnlu[(-2,0)](u).real
            self.assertAlmostEqual(pole_orig, pole_new, places=15, msg="Pole (-2,0) failed for u={}".format(u))
        # test the discontinuity function:
        for u in np.linspace(0,5,21):
            disc_orig = FCC_GF.crys.volume*(FCC_GF.pmax/(2*np.sqrt(np.pi)))**3*\
                        np.exp(-(0.5*u*FCC_GF.pmax)**2)/np.sqrt(np.product(FCC_GF.d))
            disc_new = FCC_GF.g_Taylor_fnlu[(0,0)](u).real
            self.assertAlmostEqual(disc_orig, disc_new, places=15, msg="Disc (0,0) failed for u={}".format(u))
        # test the GF evaluation against the original
        NNvect = np.array([dx for (i,j), dx in self.FCC_jumpnetwork[0]])
        rates = np.array([1 for jump in NNvect])
        # old_FCC_GF = GFcalc.GFcalc(self.FCC.lattice, NNvect, rates)
        # for R in [np.array([0.,0.,0.]), np.array([0.5, 0.5, 0.]), np.array([0.5, 0., 0.5]), \
        #          np.array([1.,0.,0.]), np.array([1.,0.5,0.5]), np.array([1.,1.,0.])]:
        #     GF_orig = old_FCC_GF.GF(R)
        #     GF_new = FCC_GF(0,0,R)
        #     # print("R={}: dG= {}  G_orig= {}  G_new= {}".format(R, GF_new-GF_orig, GF_orig, GF_new))
        #     self.assertAlmostEqual(GF_orig, GF_new, places=5,
        #                            msg="Failed for R={}".format(R))

    def testHCP(self):
        """Test on HCP"""
        HCP_GF = GFcalc.GFCrystalcalc(self.HCP, 0, self.HCP_sitelist, self.HCP_jumpnetwork, Nmax=4)
        HCP_GF.SetRates([1],[0],[1,1],[0,0])  # one unique site, two types of jumps
        # print(HCP_GF.Diffusivity())
        # make some basic vectors:
        hcp_basal = self.HCP.pos2cart(np.array([1.,0.,0.]), (0,0)) - \
                    self.HCP.pos2cart(np.array([0.,0.,0.]), (0,0))
        hcp_pyram = self.HCP.pos2cart(np.array([0.,0.,0.]), (0,1)) - \
                    self.HCP.pos2cart(np.array([0.,0.,0.]), (0,0))
        hcp_zero = np.zeros(3)
        for R in [hcp_zero, hcp_basal, hcp_pyram]:
            self.assertAlmostEqual(HCP_GF(0,0,R), HCP_GF(1,1,R), places=15)
        self.assertAlmostEqual(HCP_GF(0,0,hcp_basal), HCP_GF(0,0,-hcp_basal), places=15)
        self.assertAlmostEqual(HCP_GF(0,1,hcp_pyram), HCP_GF(1,0,-hcp_pyram), places=15)
        g0 = HCP_GF(0,0,hcp_zero)
        gbasal = HCP_GF(0,0,hcp_basal)
        gpyram = HCP_GF(0,1,hcp_pyram)
        self.assertAlmostEqual(-12*g0 + 6*gbasal + 6*gpyram, 1, places=6)
        # Try again, but with different rates:
        HCP_GF.SetRates([1],[0],[1,3],[0,0])  # one unique site, two types of jumps
        g0 = HCP_GF(0,0,hcp_zero)
        gw = 0
        for jumplist, omega in zip(self.HCP_jumpnetwork, HCP_GF.symmrate*HCP_GF.maxrate):
            for (i,j), dx in jumplist:
                if (i==0):
                    gw += omega*(HCP_GF(i,j,dx) - g0)
        self.assertAlmostEqual(gw, 1, places=6)

