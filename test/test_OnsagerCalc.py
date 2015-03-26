"""
Unit tests for Onsager calculator for vacancy-mediated diffusion.
"""

__author__ = 'Dallas R. Trinkle'

# TODO: add the five-frequency model as a test for FCC; add in BCC
# TODO: additional tests using the 14 frequency model for FCC?

import unittest
import onsager.FCClatt as FCClatt
import onsager.KPTmesh as KPTmesh
import numpy as np
import onsager.OnsagerCalc as OnsagerCalc
import onsager.crystal as crystal


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
            Lvv, Lss, Lsv, L1vv = self.Lcalc.Lij(gf, om0, prob, om2, om1)
            for lvv, lsv, lss in zip(Lvv.flat, Lsv.flat, Lss.flat):
                self.assertAlmostEqual(lvv, -lsv)
                self.assertAlmostEqual(lvv, lss)


import onsager.GFcalc as GFcalc


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
        Lvv, Lss, Lsv, L1vv = self.Lcalc.Lij(gf, om0, prob, om2, om1)
        # print 'Lvv:', Lvv
        # print 'Lss:', Lss
        # print 'Lsv:', Lsv
        for p in [(i, j) for i in range(3) for j in range(3) if i != j]:
            self.assertAlmostEqual(0, Lvv[p])
            self.assertAlmostEqual(0, Lss[p])
            self.assertAlmostEqual(0, Lsv[p])
            self.assertAlmostEqual(0, L1vv[p])
        for L in [Lvv, Lss, Lsv, L1vv]:
            self.assertAlmostEqual(L[0, 0], L[1, 1])
            self.assertAlmostEqual(L[1, 1], L[2, 2])
            self.assertAlmostEqual(L[2, 2], L[0, 0])
        # No solute drag, so Lsv = -Lvv; Lvv = normal vacancy diffusion
        # all correlation is in that geometric prefactor of Lss.
        self.assertAlmostEqual(Lvv[0, 0], self.D0)
        self.assertAlmostEqual(-Lsv[0, 0], self.D0)
        self.assertAlmostEqual(L1vv[0, 0], 0.)
        self.assertAlmostEqual(-Lss[0, 0]/Lsv[0, 0], self.correl, delta=1e-3)


def fivefreq(w0, w1, w2, w3, w4):
    """The solute/solute diffusion coefficient in the 5-freq. model"""
    b = w4/w0
    # 7(1-F) = (10 b^4 + 180.3 b^3 + 924.3 b^2 + 1338.1 b)/ (2 b^4 + 40.1 b^3 + 253/3 b^2 + 596 b + 435.3)
    F7 = 7. - b*(1338.1 + b*(924.3 + b*(180.3 + b*10.)))/\
              (435.3 + b*(596. + b*(253.3 + b*(40.1 + b*2.))))
    p = w4/w3
    return p*w2*(2.*w1 + w3*F7)/(2.*w2 + 2.*w1 + w3*F7)


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
        self.correl = 0.78145

    def testFiveFreq(self):
        """Test whether we can reproduce the five frequency model"""
        self.Lcalc.generate(1)
        w0 = self.rates[0]
        w1 = 0.8 * w0
        w2 = 1.25 * w0
        w3 = 1.5 * w0
        w4 = 0.5 * w0
        w3w4 = np.sqrt(w3*w4)
        prob, om2, om1 = self.Lcalc.maketracer()
        om0 = np.array([w0])
        om2 = np.array([w2])
        prob[0] = w4/w3
        om1list, om1index = self.Lcalc.omega1list()
        # making the om1 list... a little tricky
        for i, pair in enumerate(om1list):
            p0nn = any([all(abs(pair[0] - x) < 1e-8) for x in self.NNvect])
            p1nn = any([all(abs(pair[1] - x) < 1e-8) for x in self.NNvect])
            if p0nn and p1nn:
                om1[i] = w1
                continue
            if p0nn or p1nn:
                om1[i] = w3w4
                continue
            # rely on LIMB for rest...
        gf = np.array([self.GF.GF(R) for R in self.Lcalc.GFlist()])
        Lvv, Lss, Lsv, L1vv = self.Lcalc.Lij(gf, om0, prob, om2, om1)
        for p in [(i, j) for i in range(3) for j in range(3) if i != j]:
            self.assertAlmostEqual(0, Lvv[p])
            self.assertAlmostEqual(0, Lss[p])
            self.assertAlmostEqual(0, Lsv[p])
            self.assertAlmostEqual(0, L1vv[p])
        for L in [Lvv, Lss, Lsv, L1vv]:
            self.assertAlmostEqual(L[0, 0], L[1, 1])
            self.assertAlmostEqual(L[1, 1], L[2, 2])
            self.assertAlmostEqual(L[2, 2], L[0, 0])
        self.assertAlmostEqual(Lvv[0, 0], self.D0)
        self.assertAlmostEqual(Lss[0, 0], 4.*fivefreq(w0, w1, w2, w3, w4), delta=1e-3)
        # print 'om0:', om0
        # print 'om1:', om1
        # print 'om2:', om2
        # print 'prob:', prob
        # print 'gf:', gf
        # print 'Lvv:', Lvv
        # print 'Lss:', Lss
        # print 'Lsv:', Lsv



class BCCBaseTests(SCBaseTests):
    """Set of tests that our Onsager calculator is well-behaved"""

    def setUp(self):
        self.lattice, self.NNvect, self.groupops, self.rates = setupBCC()
        self.Lcalc = OnsagerCalc.VacancyMediated(self.NNvect, self.groupops)
        self.Njumps = 1
        self.Ninteract = [1, 4]
        self.Nomega1 = [3, 9]
        self.NGF = [14, 30]
        self.GF = GFcalc.GFcalc(self.lattice, self.NNvect, self.rates)
        self.D0 = self.GF.D2[0, 0]
        self.correl = 0.727


class InterstitialTests(unittest.TestCase):
    """Tests for our interstitial diffusion calculator"""

    def setUp(self):
        # Both HCP and FCC diffusion networks with octahedral and tetrahedral sites
        self.a0 = 3
        self.c_a = np.sqrt(8./3.)
        self.fcclatt = self.a0*np.array([[0, 0.5, 0.5],
                                         [0.5, 0, 0.5],
                                         [0.5, 0.5, 0]])
        self.fccbasis = [[np.zeros(3)], [np.array([0.5,0.5,-0.5]),
                                         np.array([0.25,0.25,0.25]),
                                         np.array([0.75,0.75,0.75])]]
        self.hexlatt = self.a0*np.array([[0.5, 0.5, 0],
                                         [-np.sqrt(0.75), np.sqrt(0.75), 0],
                                         [0, 0, self.c_a]])
        self.hcpbasis = [[np.array([1./3.,2./3.,0.25]),np.array([2./3.,1./3.,0.75])],
                         [np.array([0.,0.,0.]), np.array([0.,0.,0.5]),
                          np.array([1./3.,2./3.,0.625]), np.array([1./3.,2./3.,0.875]),
                          np.array([2./3.,1./3.,0.125]), np.array([2./3.,1./3.,0.375])]]
        self.HCP_intercrys = crystal.Crystal(self.hexlatt, self.hcpbasis)
        self.HCP_jumpnetwork = self.HCP_intercrys.jumpnetwork(1, self.a0*0.7) # tuned to avoid t->t in basal plane
        self.HCP_sitelist = self.HCP_intercrys.sitelist(1)
        self.Dhcp = OnsagerCalc.Interstitial(self.HCP_intercrys, 1, self.HCP_sitelist, self.HCP_jumpnetwork)
        self.FCC_intercrys = crystal.Crystal(self.fcclatt, self.fccbasis)
        self.FCC_jumpnetwork = self.FCC_intercrys.jumpnetwork(1, self.a0*0.5)
        self.FCC_sitelist = self.FCC_intercrys.sitelist(1)
        self.Dfcc = OnsagerCalc.Interstitial(self.FCC_intercrys, 1, self.FCC_sitelist, self.FCC_jumpnetwork)

    def testVectorBasis(self):
        """Do we correctly analyze our crystals regarding their symmetry?"""
        self.assertEqual(self.Dhcp.NV, 1)
        self.assertTrue(self.Dhcp.omega_invertible)
        self.assertTrue(np.allclose(self.Dhcp.VV[:,:,0,0], np.array([[0,0,0],[0,0,0],[0,0,1]])))
        self.assertEqual(self.Dfcc.NV, 0)
        self.assertTrue(self.Dfcc.omega_invertible)

    def testInverseMap(self):
        """Do we correctly construct the inverse map?"""
        for D in [self.Dhcp, self.Dfcc]:
            for i,w in enumerate(D.invmap):
                self.assertTrue(any(i==j for j in D.sitelist[w]))
        self.assertEqual(len(self.HCP_sitelist), 2)
        self.assertEqual(len(self.FCC_sitelist), 2)
        self.assertEqual(len(self.HCP_jumpnetwork), 2)
        self.assertEqual(len(self.FCC_jumpnetwork), 1)

    def testGroupOps(self):
        """Do we have reasonable group op. lists?"""
        center = np.zeros(3, dtype=int)
        for D in [self.Dfcc, self.Dhcp]:
            for sites, groups in zip(D.sitelist, D.sitegroupops):
                i0 = sites[0]
                for site, g in zip(sites, groups):
                    # group operation g transforms the site (c, i0) into (c, i)
                    R, (c,i) = D.crys.g_pos(g, center, (D.chem, i0))
                    self.assertEqual(site, i)
            for jumps, groups in zip(D.jumpnetwork, D.jumpgroupops):
                (i0, j0), dx0 = jumps[0]
                for ((i, j), dx), g in zip(jumps, groups):
                    R, (c,inew) = D.crys.g_pos(g, center, (D.chem, i0))
                    R, (c,jnew) = D.crys.g_pos(g, center, (D.chem, j0))
                    dxnew = D.crys.g_direc(g, dx0)
                    if inew==i:
                        failmsg = "({},{}), {} != ({},{}), {}".format(inew,jnew,dxnew, i, j, dx)
                        self.assertEqual(inew, i, msg=failmsg)
                        self.assertEqual(jnew, j, msg=failmsg)
                        self.assertTrue(np.allclose(dxnew, dx), msg=failmsg)
                    else:
                        # reverse transition
                        failmsg = "({},{}), {} != ({},{}), {}".format(inew,jnew,dxnew, j, i, -dx)
                        self.assertEqual(inew, j, msg=failmsg)
                        self.assertEqual(jnew, i, msg=failmsg)
                        self.assertTrue(np.allclose(dxnew, -dx), msg=failmsg)

    def testSymmBasis(self):
        """Do we have a reasonable symmetric tensor basis?"""
        for basis in self.Dfcc.siteSymmTensorBasis:
            self.assertEqual(len(basis), 1)
        for basis in self.Dfcc.jumpSymmTensorBasis:
            self.assertEqual(len(basis), 2)
        for basis in self.Dhcp.siteSymmTensorBasis:
            self.assertEqual(len(basis), 2)
        for basis, jumps in zip(self.Dhcp.jumpSymmTensorBasis, self.Dhcp.jumpnetwork):
            if len(jumps) == 4:
                self.assertEqual(len(basis), 2)
            else:
                self.assertEqual(len(basis), 4)

    def testSiteProb(self):
        """Do we correctly construct our site probabilities?"""
        # HCP first
        preoct = 1
        pretet = 2
        BEoct = 0
        BEtet = np.log(2) # so exp(-beta*E) = 1/2
        pre = np.zeros(len(self.HCP_sitelist))
        BE = np.zeros(len(self.HCP_sitelist))
        pre[self.Dhcp.invmap[0]] = preoct
        pre[self.Dhcp.invmap[2]] = pretet
        BE[self.Dhcp.invmap[0]] = BEoct
        BE[self.Dhcp.invmap[2]] = BEtet
        # With this, we have 6 sites total, and they should all have equal probability: so 1/6 is the answer.
        self.assertTrue(np.allclose(np.ones(self.Dhcp.N)/self.Dhcp.N, self.Dhcp.siteprob(pre, BE)))
        # FCC
        pre = np.zeros(len(self.FCC_sitelist))
        BE = np.zeros(len(self.FCC_sitelist))
        pre[self.Dfcc.invmap[0]] = preoct
        pre[self.Dfcc.invmap[1]] = pretet
        BE[self.Dfcc.invmap[0]] = BEoct
        BE[self.Dfcc.invmap[1]] = BEtet
        # With this, we have 3 sites total, and they should all have equal probability: so 1/3 is the answer.
        self.assertTrue(np.allclose(np.ones(self.Dfcc.N)/self.Dfcc.N, self.Dfcc.siteprob(pre, BE)))

    def testRatelist(self):
        """Do we correctly construct our rates?"""
        # FCC first
        preoct = 1
        pretet = 2
        BEoct = 0
        BEtet = np.log(2) # so exp(-beta*E) = 1/2
        preTrans = 10
        BETrans = np.log(10) # so that our rate should be 10*exp(-BET) / (1*exp(0)) = 1
        pre = np.zeros(len(self.FCC_sitelist))
        BE = np.zeros(len(self.FCC_sitelist))
        pre[self.Dfcc.invmap[0]] = preoct
        pre[self.Dfcc.invmap[1]] = pretet
        BE[self.Dfcc.invmap[0]] = BEoct
        BE[self.Dfcc.invmap[1]] = BEtet
        preT = np.array([preTrans])
        BET = np.array([BETrans])
        self.assertTrue(all( np.isclose(rate, 1)
                             for ratelist in self.Dfcc.ratelist(pre, BE, preT, BET)
                             for rate in ratelist))
        # try changing the prefactor for tetrahedral...
        pre[self.Dfcc.invmap[1]] = 1
        ratelist = self.Dfcc.ratelist(pre, BE, preT, BET)
        for ((i, j), dx), rate in zip(self.Dfcc.jumpnetwork[0], ratelist[0]):
            if i==0: self.assertAlmostEqual(rate, 1) # oct->tet
            else:    self.assertAlmostEqual(rate, 2) # tet->oct

        # HCP
        pre = np.zeros(len(self.HCP_sitelist))
        BE = np.zeros(len(self.HCP_sitelist))
        pre[self.Dhcp.invmap[0]] = preoct
        pre[self.Dhcp.invmap[2]] = pretet
        BE[self.Dhcp.invmap[0]] = BEoct
        BE[self.Dhcp.invmap[2]] = BEtet
        preTransOT = 10.
        preTransTT = 100.
        BETransOT = np.log(10.)
        BETransTT = np.log(10.)
        preT = np.zeros(len(self.Dhcp.jumpnetwork))
        BET = np.zeros(len(self.Dhcp.jumpnetwork))
        for i, jump in enumerate(self.Dhcp.jumpnetwork):
            if len(jump) == 4:
                preT[i] = preTransTT
                BET[i] = BETransTT
            else:
                preT[i] = preTransOT
                BET[i] = BETransOT
        # oct->tet jumps have rate 1, tet->tet jumps have rate 10.
        ratelist = self.Dhcp.ratelist(pre, BE, preT, BET)
        for jumps, rates in zip(self.Dhcp.jumpnetwork, ratelist):
            for ((i, j), dx), rate in zip(jumps, rates):
                if i<2 or j<2: self.assertAlmostEqual(rate, 1) # oct->tet
                else:          self.assertAlmostEqual(rate, 10) # tet->oct

    def testDiffusivity(self):
        """Diffusivity"""
        # What we all came for...
        preoct = 1.
        pretet = 2.
        BEoct = 0.
        BEtet = np.log(2) # so exp(-beta*E) = 1/2
        preTrans = 10.
        BETrans = np.log(10) # so that our rate should be 10*exp(-BET) / (1*exp(0)) = 1
        pre = np.zeros(len(self.FCC_sitelist))
        BE = np.zeros(len(self.FCC_sitelist))
        pre[self.Dfcc.invmap[0]] = preoct
        pre[self.Dfcc.invmap[1]] = pretet
        BE[self.Dfcc.invmap[0]] = BEoct
        BE[self.Dfcc.invmap[1]] = BEtet
        preT = np.array([preTrans])
        BET = np.array([BETrans])

        Dfcc_anal = 0.5*self.a0**2 *preTrans*np.exp(-BETrans)/(preoct*np.exp(-BEoct) + 2*pretet*np.exp(-BEtet))
        self.assertTrue(np.allclose(Dfcc_anal*np.eye(3), self.Dfcc.diffusivity(pre, BE, preT, BET)))

        # HCP
        pre = np.zeros(len(self.HCP_sitelist))
        BE = np.zeros(len(self.HCP_sitelist))
        pre[self.Dhcp.invmap[0]] = preoct
        pre[self.Dhcp.invmap[2]] = pretet
        BE[self.Dhcp.invmap[0]] = BEoct
        BE[self.Dhcp.invmap[2]] = BEtet
        preTransOT = 10.
        preTransTT = 100.
        BETransOT = np.log(10.)
        BETransTT = np.log(10.)
        preT = np.zeros(len(self.Dhcp.jumpnetwork))
        BET = np.zeros(len(self.Dhcp.jumpnetwork))
        for i, jump in enumerate(self.Dhcp.jumpnetwork):
            if len(jump) == 4:
                preT[i] = preTransTT
                BET[i] = BETransTT
            else:
                preT[i] = preTransOT
                BET[i] = BETransOT
        Dhcp_basal = self.a0**2 * preTransOT*np.exp(-BETransOT)/(preoct*np.exp(-BEoct) + 2*pretet*np.exp(-BEtet))
        Dhcp_c = 0.75*self.c_a**2 * Dhcp_basal/ (3*preTransOT/preTransTT * np.exp(-BETransOT+BETransTT) + 2)
        D = self.Dhcp.diffusivity(pre, BE, preT, BET)
        self.assertTrue(np.allclose(np.array([[Dhcp_basal,0,0],[0,Dhcp_basal,0],[0,0,Dhcp_c]]), D),
                        msg="Diffusivity doesn't match:\n{}\nnot {} and {}".format(D, Dhcp_basal,Dhcp_c))

    def testSymmTensorMapping(self):
        """Do we correctly map our elastic dipoles onto sites and transitions?"""
        # put a little "error" in from our calculation... shouldn't really be present
        dipole = [np.array([[1.,1e-4, -2e-4],[1e-4,1.,3e-4],[-2e-4, 3e-4, 1.]]),
                  np.array([[1.,1e-4, -2e-4],[1e-4,1.,3e-4],[-2e-4, 3e-4, 1.]])]
        (i,j), dx = self.Dfcc.jumpnetwork[0][0] # our representative jump
        dipoleT = [ -0.5*np.eye(3) + 2.*np.outer(dx, dx)] # this should remain unchanged
        sitedipoles = self.Dfcc.siteDipoles(dipole)
        jumpdipoles = self.Dfcc.jumpDipoles(dipoleT)
        for dip in sitedipoles:
            self.assertTrue(np.allclose(dip, np.eye(3)))
        self.assertTrue(np.allclose(dipoleT[0], jumpdipoles[0][0]))
        self.assertTrue(np.allclose(np.trace(dipoleT[0])*np.eye(3)/3., sum(jumpdipoles[0])/len(jumpdipoles[0])))
        for ((i,j), dx), dipole in zip(self.Dfcc.jumpnetwork[0], jumpdipoles[0]):
            self.assertTrue(np.allclose(-0.5*np.eye(3) + 2.*np.outer(dx, dx), dipole))

    def testFCCElastodiffusion(self):
        """Elastodiffusion tensor without correlation; compare with finite difference"""
        # FCC first:
        preoct = 1.
        pretet = 0.5
        BEoct = 0.
        BEtet = np.log(2) # so exp(-beta*E) = 1/2
        preTrans = 10.
        BETrans = np.log(10) # so that our rate should be 10*exp(-BET) / (1*exp(0)) = 1
        pre = np.zeros(len(self.FCC_sitelist))
        BE = np.zeros(len(self.FCC_sitelist))
        pre[self.Dfcc.invmap[0]] = preoct
        pre[self.Dfcc.invmap[1]] = pretet
        BE[self.Dfcc.invmap[0]] = BEoct
        BE[self.Dfcc.invmap[1]] = BEtet
        preT = np.array([preTrans])
        BET = np.array([BETrans])

        octP = 1e-4
        tetP = -1e-4
        transPpara = 2e-4
        transPperp = -1.5e-4
        dipole = [0, 0]
        dipole[self.Dfcc.invmap[0]] = octP*np.eye(3)
        dipole[self.Dfcc.invmap[1]] = tetP*np.eye(3)
        (i,j), dx = self.Dfcc.jumpnetwork[0][0] # our representative jump
        dipoleT = [ transPperp*np.eye(3) + (transPpara-transPperp)*np.outer(dx, dx)/np.dot(dx, dx)]
        sitedipoles = self.Dfcc.siteDipoles(dipole)
        jumpdipoles = self.Dfcc.jumpDipoles(dipoleT)

        # strain
        D0, Dp = self.Dfcc.elastodiffusion(pre, BE, dipole, preT, BET, dipoleT)
        eps = 1e-4
        for straintype in [ np.array([[1.,0.,0],[0.,0.,0,],[0.,0.,0.]]),
                            np.array([[0.,0.,0],[0.,1.,0,],[0.,0.,0.]]),
                            np.array([[0.,0.,0],[0.,0.,0,],[0.,0.,1.]]),
                            np.array([[0.,0.,0],[0.,0.,0.5],[0.,0.5,0.]]),
                            np.array([[0.,0.,0.5],[0.,0.,0,],[0.5,0.,0.]]),
                            np.array([[0.,0.5,0],[0.5,0.,0,],[0.,0.,0.]]) ]:
            strainmat = eps*straintype
            strainedFCC = crystal.Crystal(np.dot(np.eye(3) + strainmat, self.fcclatt), self.fccbasis)
            strainedFCC_jumpnetwork = strainedFCC.jumpnetwork(1, self.a0*0.5)
            strainedFCC_sitelist = strainedFCC.sitelist(1)
            strainedDfcc =OnsagerCalc.Interstitial(strainedFCC, 1, strainedFCC_sitelist, strainedFCC_jumpnetwork)
            strainedBE = np.zeros(len(strainedFCC_sitelist))
            # apply dipoles to site energies:
            strainedBE[strainedDfcc.invmap[0]] = BEoct + np.sum(sitedipoles[0] * strainmat)
            strainedBE[strainedDfcc.invmap[1]] = BEtet + np.sum(sitedipoles[1] * strainmat)
            strainedBET = np.zeros(len(strainedFCC_jumpnetwork))
            strainedpreT = np.zeros(len(strainedFCC_jumpnetwork))
            # this gets more complicated... just work it "by hand"
            for ind, jumps in enumerate(strainedFCC_jumpnetwork):
                (i,j), dx = jumps[0]
                dx0 = np.dot(np.linalg.inv(np.eye(3) + strainmat), dx)
                strainedpreT[ind] = preTrans
                dip = transPperp*np.eye(3) +(transPpara-transPperp)*np.outer(dx0, dx0)/np.dot(dx0, dx0)
                strainedBET[ind] = BETrans + np.sum(dip*strainmat)
            Deps = strainedDfcc.diffusivity(pre, strainedBE, strainedpreT, strainedBET)
            Deps0 = np.tensordot(Dp, strainmat, axes=((2,3), (0,1)))/eps
            failmsg = "strainmatrix:\n{}\nD0:\n{}\nfinite difference:\n{}".format(strainmat,
                                                                                  D0, (Deps-D0)/eps, Deps0)
            self.assertTrue(np.allclose((Deps-D0)/eps, Deps0, rtol=eps, atol=eps), msg=failmsg)

    def testHCPElastodiffusion(self):
        """Elastodiffusion tensor with correlation; compare with finite difference"""
        # HCP; note: *uncorrelated* requires lambda(t->t) = 1.5*lambda(t->o)
        preoct = 1.
        pretet = 0.5
        BEoct = 0.
        BEtet = np.log(2) # so exp(-beta*E) = 1/2
        preTransOT = 10.
        preTransTT = 100.
        BETransOT = np.log(10.)
        BETransTT = np.log(10.)

        pre = np.zeros(len(self.HCP_sitelist))
        BE = np.zeros(len(self.HCP_sitelist))
        pre[self.Dhcp.invmap[0]] = preoct
        pre[self.Dhcp.invmap[2]] = pretet
        BE[self.Dhcp.invmap[0]] = BEoct
        BE[self.Dhcp.invmap[2]] = BEtet
        preT = np.zeros(len(self.Dhcp.jumpnetwork))
        BET = np.zeros(len(self.Dhcp.jumpnetwork))
        for i, jump in enumerate(self.Dhcp.jumpnetwork):
            if len(jump) == 4:
                preT[i] = preTransTT
                BET[i] = BETransTT
            else:
                preT[i] = preTransOT
                BET[i] = BETransOT

        octPb = 1e-4
        octPc = 2e-4
        tetPb = -1e-4
        tetPc = -2e-4
        transPpara = 0 # 3e-4
        transPperp = 0 # -1.5e-4
        dipole = [0, 0]
        dipole[self.Dhcp.invmap[0]] = np.array([[octPb,0,0],[0,octPb,0],[0,0,octPc]])
        dipole[self.Dhcp.invmap[2]] = np.array([[tetPb,0,0],[0,tetPb,0],[0,0,tetPc]])
        dipoleT = []
        for jumps in self.Dhcp.jumpnetwork:
            (i,j), dx = jumps[0] # our representative jump
            dipoleT.append(transPperp*np.eye(3) + (transPpara-transPperp)*np.outer(dx, dx)/np.dot(dx, dx))
        sitedipoles = self.Dhcp.siteDipoles(dipole)
        jumpdipoles = self.Dhcp.jumpDipoles(dipoleT)

        # strain
        eps = 1e-5
        D0, Dp = self.Dhcp.elastodiffusion(pre, BE, dipole, preT, BET, dipoleT)
        for straintype in [ np.array([[1.,0.,0],[0.,0.,0,],[0.,0.,0.]]),
                            np.array([[0.,0.,0],[0.,1.,0,],[0.,0.,0.]]),
                            np.array([[0.,0.,0],[0.,0.,0,],[0.,0.,1.]]),
                            np.array([[0.,0.,0],[0.,0.,0.5],[0.,0.5,0.]]),
                            np.array([[0.,0.,0.5],[0.,0.,0,],[0.5,0.,0.]]),
                            np.array([[0.,0.5,0],[0.5,0.,0,],[0.,0.,0.]]) ]:
            strainmat = eps*straintype
            strainedHCP = crystal.Crystal(np.dot(np.eye(3) + strainmat, self.hexlatt), self.hcpbasis)
            strainedHCP_jumpnetwork = strainedHCP.jumpnetwork(1, self.a0*0.7)
            strainedHCP_sitelist = strainedHCP.sitelist(1)
            strainedDhcp =OnsagerCalc.Interstitial(strainedHCP, 1, strainedHCP_sitelist, strainedHCP_jumpnetwork)
            strainedpre = np.zeros(len(strainedHCP_sitelist))
            strainedBE = np.zeros(len(strainedHCP_sitelist))
            # apply dipoles to site energies:
            for octind in xrange(2):
                strainedpre[strainedDhcp.invmap[octind]] = preoct
                strainedBE[strainedDhcp.invmap[octind]] = BEoct + np.sum(sitedipoles[octind] * strainmat)
            for tetind in xrange(2,6):
                strainedpre[strainedDhcp.invmap[tetind]] = pretet
                strainedBE[strainedDhcp.invmap[tetind]] = BEtet + np.sum(sitedipoles[tetind] * strainmat)
            strainedBET = np.zeros(len(strainedHCP_jumpnetwork))
            strainedpreT = np.zeros(len(strainedHCP_jumpnetwork))
            # this gets more complicated...
            for ind, jumps in enumerate(strainedHCP_jumpnetwork):
                (i,j), dx = jumps[0]
                dx0 = np.dot(np.linalg.inv(np.eye(3) + strainmat), dx)
                dip = transPperp*np.eye(3) +(transPpara-transPperp)*np.outer(dx0, dx0)/np.dot(dx0, dx0)
                if i>=2 and j>=2:
                    strainedpreT[ind] = preTransTT
                    strainedBET[ind] = BETransTT + np.sum(dip*strainmat)
                else:
                    strainedpreT[ind] = preTransOT
                    strainedBET[ind] = BETransOT + np.sum(dip*strainmat)
            # print strainedHCP_sitelist
            # print strainedBE
            # print strainedpre
            # print strainedBET
            # print strainedpreT
            Deps = strainedDhcp.diffusivity(strainedpre, strainedBE, strainedpreT, strainedBET)
            Deps0 = np.tensordot(Dp, strainmat, axes=((2,3), (0,1)))/eps
            failmsg = """
strainmatrix:
{}
D0:
{}
finite difference:
{}
elastodiffusion:
{}""".format(strainmat, D0, (Deps-D0)/eps, Deps0)
            self.assertTrue(np.allclose((Deps-D0)/eps, Deps0, rtol=2*eps, atol=2*eps), msg=failmsg)
