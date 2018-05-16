"""
Unit tests for calculation of lattice Green function for diffusion
"""

__author__ = 'Dallas R. Trinkle'

import unittest
import numpy as np
from scipy import special
import onsager.GFcalc as GFcalc
import onsager.crystal as crystal


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
    """Test new implementation of GF calculator, based on Crystal class"""

    longMessage = False

    def setUp(self):
        pass

    def testFCC(self):
        """Test on FCC"""
        FCC = crystal.Crystal.FCC(1.)
        FCC_sitelist = FCC.sitelist(0)
        FCC_jumpnetwork = FCC.jumpnetwork(0, 0.75)
        FCC_GF = GFcalc.GFCrystalcalc(FCC, 0, FCC_sitelist, FCC_jumpnetwork, Nmax=4)
        FCC_GF.SetRates([1], [0], [1], [0])
        # test the pole function:
        for u in np.linspace(0, 5, 21):
            pole_orig = FCC_GF.crys.volume * poleFT(FCC_GF.d, u, FCC_GF.pmax)
            pole_new = FCC_GF.g_Taylor_fnlu[(-2, 0)](u).real
            self.assertAlmostEqual(pole_orig, pole_new, places=15, msg="Pole (-2,0) failed for u={}".format(u))
        # test the discontinuity function:
        for u in np.linspace(0, 5, 21):
            disc_orig = FCC_GF.crys.volume * (FCC_GF.pmax / (2 * np.sqrt(np.pi))) ** 3 * \
                        np.exp(-(0.5 * u * FCC_GF.pmax) ** 2) / np.sqrt(np.product(FCC_GF.d))
            disc_new = FCC_GF.g_Taylor_fnlu[(0, 0)](u).real
            self.assertAlmostEqual(disc_orig, disc_new, places=15, msg="Disc (0,0) failed for u={}".format(u))
            # test the GF evaluation against the original
            # NNvect = np.array([dx for (i,j), dx in FCC_jumpnetwork[0]])
            # rates = np.array([1 for jump in NNvect])
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
        HCP = crystal.Crystal.HCP(1., np.sqrt(8 / 3))
        HCP_sitelist = HCP.sitelist(0)
        HCP_jumpnetwork = HCP.jumpnetwork(0, 1.01)
        HCP_GF = GFcalc.GFCrystalcalc(HCP, 0, HCP_sitelist, HCP_jumpnetwork, Nmax=4)
        HCP_GF.SetRates([1], [0], [1, 1], [0, 0])  # one unique site, two types of jumps
        # print(HCP_GF.Diffusivity())
        # make some basic vectors:
        hcp_basal = HCP.pos2cart(np.array([1., 0., 0.]), (0, 0)) - \
                    HCP.pos2cart(np.array([0., 0., 0.]), (0, 0))
        hcp_pyram = HCP.pos2cart(np.array([0., 0., 0.]), (0, 1)) - \
                    HCP.pos2cart(np.array([0., 0., 0.]), (0, 0))
        hcp_zero = np.zeros(3)
        for R in [hcp_zero, hcp_basal, hcp_pyram]:
            self.assertAlmostEqual(HCP_GF(0, 0, R), HCP_GF(1, 1, R), places=15)
        self.assertAlmostEqual(HCP_GF(0, 0, hcp_basal), HCP_GF(0, 0, -hcp_basal), places=15)
        self.assertAlmostEqual(HCP_GF(0, 1, hcp_pyram), HCP_GF(1, 0, -hcp_pyram), places=15)
        g0 = HCP_GF(0, 0, hcp_zero)
        gbasal = HCP_GF(0, 0, hcp_basal)
        gpyram = HCP_GF(0, 1, hcp_pyram)
        self.assertAlmostEqual(-12 * g0 + 6 * gbasal + 6 * gpyram, 1, places=6)
        # Try again, but with different rates:
        HCP_GF.SetRates([1], [0], [1, 3], [0, 0])  # one unique site, two types of jumps
        g0 = HCP_GF(0, 0, hcp_zero)
        gw = 0
        for jumplist, omega in zip(HCP_jumpnetwork, HCP_GF.symmrate * HCP_GF.maxrate):
            for (i, j), dx in jumplist:
                if (i == 0):
                    gw += omega * (HCP_GF(i, j, dx) - g0)
        self.assertAlmostEqual(gw, 1, places=6)

    def testsquare(self):
        """Test on square"""
        square = crystal.Crystal(np.eye(2), [np.zeros(2)])
        square_sitelist = square.sitelist(0)
        square_jumpnetwork = square.jumpnetwork(0, 1.01)
        square_GF = GFcalc.GFCrystalcalc(square, 0, square_sitelist, square_jumpnetwork, Nmax=4)
        square_GF.SetRates([1], [0], [1], [0])
        square_zero = np.zeros(2)
        square_1nn = np.array([1.,0.])
        square_2nn = np.array([1.,1.])
        square_3nn = np.array([2.,0.])
        g0 = square_GF(0, 0, square_zero)
        g1 = square_GF(0, 0, square_1nn)
        g2 = square_GF(0, 0, square_2nn)
        g3 = square_GF(0, 0, square_3nn)
        self.assertAlmostEqual(-4 * g0 + 4 * g1, 1, places=6)
        self.assertAlmostEqual(-4 * g1 + g0 + 2*g2 + g3, 0, places=6)

    def testtria(self):
        """Test on triagonal"""
        tria = crystal.Crystal(np.array([[1/2, 1/2], [-np.sqrt(3/4), np.sqrt(3/4)]]), [np.zeros(2)])
        tria_sitelist = tria.sitelist(0)
        tria_jumpnetwork = tria.jumpnetwork(0, 1.01)
        tria_GF = GFcalc.GFCrystalcalc(tria, 0, tria_sitelist, tria_jumpnetwork, Nmax=4)
        tria_GF.SetRates([1], [0], [1], [0])
        tria_zero = np.zeros(2)
        tria_1nn = np.array([1.,0.])
        g0 = tria_GF(0, 0, tria_zero)
        g1 = tria_GF(0, 0, tria_1nn)
        self.assertAlmostEqual(-6 * g0 + 6 * g1, 1, places=6)

    def testhoneycomb(self):
        """Test on honeycomb"""
        honey = crystal.Crystal(np.array([[1/2, 1/2], [-np.sqrt(3/4), np.sqrt(3/4)]]),
                                [np.array([2/3, 1/3]), np.array([1/3, 2/3])])
        honey_sitelist = honey.sitelist(0)
        honey_jumpnetwork = honey.jumpnetwork(0, 0.6)
        honey_GF = GFcalc.GFCrystalcalc(honey, 0, honey_sitelist, honey_jumpnetwork, Nmax=4)
        honey_GF.SetRates([1], [0], [1], [0])
        honey_zero = np.zeros(2)
        honey_1nn = honey.pos2cart(np.zeros(2), (0, 1)) - honey.pos2cart(np.zeros(2), (0, 0))
        g0 = honey_GF(0, 0, honey_zero)
        g1 = honey_GF(0, 1, honey_1nn)
        self.assertAlmostEqual(-3 * g0 + 3 * g1, 1, places=6)


    def testBCC_B2(self):
        """Test that BCC and B2 produce the same GF"""
        a0 = 1.
        chem = 0
        BCC = crystal.Crystal.BCC(a0)
        BCC_sitelist = BCC.sitelist(chem)
        BCC_jumpnetwork = BCC.jumpnetwork(chem, 0.87 * a0)
        BCC_GF = GFcalc.GFCrystalcalc(BCC, chem, BCC_sitelist, BCC_jumpnetwork, Nmax=6)
        BCC_GF.SetRates(np.ones(len(BCC_sitelist)), np.zeros(len(BCC_sitelist)),
                        2. * np.ones(len(BCC_jumpnetwork)), np.zeros(len(BCC_jumpnetwork)))

        B2 = crystal.Crystal(a0 * np.eye(3), [np.zeros(3), np.array([0.45, 0.45, 0.45])])
        B2_sitelist = B2.sitelist(chem)
        B2_jumpnetwork = B2.jumpnetwork(chem, 0.99 * a0)
        B2_GF = GFcalc.GFCrystalcalc(B2, chem, B2_sitelist, B2_jumpnetwork, Nmax=6)
        B2_GF.SetRates(np.ones(len(B2_sitelist)), np.zeros(len(B2_sitelist)),
                       2. * np.ones(len(B2_jumpnetwork)), np.zeros(len(B2_jumpnetwork)))
        veclist = [np.array([a0, 0, 0]), np.array([0, a0, 0]), np.array([0, 0, a0]),
                   np.array([-a0, 0, 0]), np.array([0, -a0, 0]), np.array([0, 0, -a0])]
        for v1 in veclist:
            for v2 in veclist:
                # print('{}: '.format(v1+v2) + '{} vs {} vs {}'.format(B2_GF(0,0,v1+v2),B2_GF(1,1,v1+v2),BCC_GF(0,0,v1+v2)))
                self.assertAlmostEqual(BCC_GF(0, 0, v1 + v2), B2_GF(0, 0, v1 + v2), places=5)
                self.assertAlmostEqual(BCC_GF(0, 0, v1 + v2), B2_GF(1, 1, v1 + v2), places=5)
        for jlist in B2_jumpnetwork:
            for (i, j), dx in jlist:
                # convert our B2 dx into a corresponding BCC dx:
                BCCdx = (0.5 * a0) * np.round(dx / (0.5 * a0))
                # print('({},{}), {} / {}: '.format(i,j,dx,BCCdx) + '{} vs {}'.format(B2_GF(i,j,dx), BCC_GF(0,0,BCCdx)))
                self.assertAlmostEqual(BCC_GF(0, 0, BCCdx), B2_GF(i, j, dx), places=5)

    def testPyrope(self):
        """Test using the pyrope structure: two disconnected symmetry-related networks"""
        a0 = 1.
        chem = 0
        cutoff = 0.31*a0
        alatt = a0 * np.array([[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]])
        invlatt = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        uMg = ((1 / 8, 0, 1 / 4), (3 / 8, 0, 3 / 4), (1 / 4, 1 / 8, 0), (3 / 4, 3 / 8, 0),
               (0, 1 / 4, 1 / 8), (0, 3 / 4, 3 / 8), (7 / 8, 0, 3 / 4), (5 / 8, 0, 1 / 4),
               (3 / 4, 7 / 8, 0), (1 / 4, 5 / 8, 0), (0, 3 / 4, 7 / 8), (0, 1 / 4, 5 / 8))
        tovec = lambda x: np.dot(invlatt, x)
        # this is a reduced version of pyrope: just the Mg (24c sites in 230)
        # pyrope2 = half of the sites; makes for a single, connected network
        pyropeMg = crystal.Crystal(alatt, [[vec(w) for w in uMg for vec in (tovec,)]], ['Mg'])
        pyropeMg2 = crystal.Crystal(alatt, [[vec(w) for w in uMg[:6] for vec in (tovec,)]], ['Mg'])
        sitelist = pyropeMg.sitelist(chem)
        sitelist2 = pyropeMg2.sitelist(chem)
        jumpnetwork = pyropeMg.jumpnetwork(chem, cutoff)
        jumpnetwork2 = pyropeMg2.jumpnetwork(chem, cutoff)
        self.assertEqual(len(jumpnetwork), 1)
        self.assertEqual(len(jumpnetwork2), 1)
        GF = GFcalc.GFCrystalcalc(pyropeMg, chem, sitelist, jumpnetwork)
        GF2 = GFcalc.GFCrystalcalc(pyropeMg2, chem, sitelist2, jumpnetwork2)
        GF.SetRates(np.ones(1), np.zeros(1), 0.25*np.ones(1), np.zeros(1))  # simple tracer
        GF2.SetRates(np.ones(1), np.zeros(1), 0.25*np.ones(1), np.zeros(1))  # simple tracer
        D0 = np.eye(3)*(1/64)
        for D in (GF.D,GF2.D):
            self.assertTrue(np.allclose(D0, D),
                            msg='Diffusivity does not match?\n{}\n!=\n{}'.format(D0,D))
        basis = pyropeMg.basis[chem]
        # order of testing: 000, 211
        ijlist = ((0,0), (0,2))
        dxlist = [np.dot(alatt, basis[j]-basis[i]) for (i,j) in ijlist]
        glist = np.array([GF(i,j,dx) for (i,j), dx in zip(ijlist, dxlist)])
        g2list = np.array([GF2(i,j,dx) for (i,j), dx in zip(ijlist, dxlist)])
        Gref = np.array([2.30796022, 1.30807261])
        self.assertTrue(np.allclose(glist, -Gref, rtol=1e-4),
                        msg='Does not match Carlson and Wilson values?\n{} !=\n{}'.format(glist, Gref))
        # with the nearly disconnected, the rate anisotropy makes comparison of differences
        # much more stable
        self.assertTrue(np.allclose(glist, g2list, rtol=1e-12),
                        msg='Does not match single network GF values?\n{} !=\n{}'.format(glist, g2list))
        for i in range(12):
            for j in range(12):
                dx = np.dot(alatt, basis[j]-basis[i])
                if i//6 != j//6:
                    self.assertAlmostEqual(GF(i,j,dx), 0,
                                           msg='Does not give disconnected networks? {},{}'.format(i,j))
                else:
                    if i>=6: dxmap = -dx  # inversion
                    else: dxmap = dx
                    self.assertAlmostEqual(GF(i,j,dx), GF2(i%6,j%6,dxmap),
                                           msg='Does not match single network? {},{}'.format(i,j))