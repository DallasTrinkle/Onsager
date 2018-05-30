"""
Unit tests for PowerExpansion class
"""

__author__ = 'Dallas R. Trinkle'

import unittest
import numpy as np
from scipy.special import sph_harm
import onsager.PowerExpansion as PE

T3D = PE.Taylor3D
T2D = PE.Taylor2D

class PowerExpansionTests(unittest.TestCase):
    """Tests to make sure our power expansions are constructed correctly and behaving as advertised"""

    def setUp(self):
        """initial setup for testing"""
        self.phi = np.pi * 0.2234
        self.theta = np.pi * 0.7261
        self.c = T3D()
        self.basis = [(np.eye(2), np.array([0.5, -np.sqrt(0.75), 0.])),
                      (np.eye(2), np.array([0.5, np.sqrt(0.75), 0.])),
                      (np.eye(2), np.array([-1., 0., 0.])),
                      (np.eye(2), np.array([-0.5, -np.sqrt(0.75), 0.])),
                      (np.eye(2), np.array([-0.5, np.sqrt(0.75), 0.])),
                      (np.eye(2), np.array([1., 0., 0.])),
                      (np.eye(2) * 2, np.array([0., 0., 1.])),
                      (np.eye(2) * 2, np.array([0., 0., -1.])),
                      ]

    def testExpansionYlmpow(self):
        """Test the expansion of Ylm into powers"""
        for (phi, theta) in [(self.phi + dp, self.theta + dt)
                             for dp in (0., 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi)
                             for dt in (0., 0.5 * np.pi, np.pi, 1.5 * np.pi)]:
            utest, umagn = T3D.powexp(np.array([np.sin(phi) * np.cos(theta),
                                                np.sin(phi) * np.sin(theta),
                                                np.cos(phi)]))
            self.assertAlmostEqual(umagn, 1)
            Ylm0 = np.zeros(T3D.NYlm, dtype=complex)
            # Ylm as power expansions
            for lm in range(T3D.NYlm):
                l, m = T3D.ind2Ylm[lm, 0], T3D.ind2Ylm[lm, 1]
                Ylm0[lm] = sph_harm(m, l, theta, phi)
                Ylmexp = np.dot(T3D.Ylmpow[lm], utest)
                self.assertAlmostEqual(Ylm0[lm], Ylmexp,
                                       msg="Failure for Ylmpow " 
                    "l={} m={}; theta={}, phi={}\n{} != {}".format(l, m, theta, phi, Ylm0[lm], Ylmexp))
            # power expansions in Ylm's
            for p in range(T3D.NYlm):
                pYlm = np.dot(T3D.powYlm[p], Ylm0)
                self.assertAlmostEqual(utest[p], pYlm,
                                       msg="Failure for powYlm " 
                    "{}; theta={}, phi={}\n{} != {}".format(T3D.ind2pow[p], theta, phi, utest[p], pYlm))
            # projection (note that Lproj is not symmetric): so this test ensures that v.u and (proj.v).u
            # give the same value
            uproj = np.tensordot(T3D.Lproj[-1], utest, axes=(0, 0))
            for p in range(T3D.NYlm):
                self.assertAlmostEqual(utest[p], uproj[p],
                                       msg="Projection failure for " 
                    "{}\n{} != {}".format(T3D.ind2pow[p], uproj[p], utest[p]))

    def testProjection(self):
        """Test that the L-projections are correct"""
        # Try to do this sequentially
        for tup in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            v = np.zeros(T3D.Npower)
            v[T3D.pow2ind[tup]] = 1.
            Pv = np.tensordot(T3D.Lproj[-1], v, axes=1)
            # now, try with multiplying by x^2+y^2+z^2:
            vxyz = np.zeros(T3D.Npower)
            vxyz[T3D.pow2ind[tup[0] + 2, tup[1], tup[2]]] = 1.
            vxyz[T3D.pow2ind[tup[0], tup[1] + 2, tup[2]]] = 1.
            vxyz[T3D.pow2ind[tup[0], tup[1], tup[2] + 2]] = 1.
            Pvxyz = np.tensordot(T3D.Lproj[-1], vxyz, axes=1)
            self.assertTrue(np.allclose(v, Pv))
            self.assertTrue(np.allclose(v, Pvxyz))
            for l in range(T3D.Lmax + 1):
                Pv = np.tensordot(T3D.Lproj[l], v, axes=1)
                Pvxyz = np.tensordot(T3D.Lproj[l], vxyz, axes=1)
                if l == sum(tup):
                    self.assertTrue(np.allclose(v, Pv))
                    self.assertTrue(np.allclose(v, Pvxyz))
                else:
                    self.assertTrue(np.allclose(Pv, 0))
                    self.assertTrue(np.allclose(Pvxyz, 0))

    def testEvaluation(self):
        """Test out the evaluation functions in an expansion, including with scalar multiply and addition"""

        def approxexp(u):
            """4th order expansion of exp(u)"""
            return 1 + u * (1 + u * (0.5 + u * (1 / 6 + u / 24)))

        def createExpansion(n):
            return lambda u: u ** n / PE.factorial(n, True)

        c = T3D()
        for coeff in c.constructexpansion(self.basis):
            c.addterms(coeff)
        for (n, l) in c.nl():
            self.assertEqual(n, l)
        fnu = {(n, l): createExpansion(n) for (n, l) in c.nl()}  # or could do this in previous loop

        # c2 = 2*c
        c2 = c.copy()
        c2 *= 2
        c3 = c + c
        c4 = c2 - c
        ### NOTE! We have to do it *this way*; otherwise, it will try to use the sum in np.array,
        ### and that WILL NOT WORK with our expansion.
        c5 = c + np.eye(2)
        prod = np.array([[-4.2, 2.67], [1.3, 3.21]])
        c6 = c.ldot(prod)
        c7 = c.copy()
        c7.irdot(prod)
        sum([c, c2, c3])  # tests whether we can use sum

        for u in [np.zeros(3), np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.]),
                  np.array([0.234, -0.85, 1.25]),
                  np.array([1.24, 0.71, -0.98])]:
            umagn = np.sqrt(np.dot(u, u))
            fval = {nl: f(umagn) for nl, f in fnu.items()}
            # comparison value:
            value = sum(pre * approxexp(np.dot(u, vec)) for pre, vec in self.basis)
            valsum = c(u, fval)
            funcsum = c(u, fnu)
            dictsum = sum(fval[k] * v for k, v in c(u).items())

            self.assertTrue(np.allclose(value, valsum),
                            msg="Failure for call with values for {}\n{} != {}".format(u, value, valsum))
            self.assertTrue(np.allclose(value, funcsum),
                            msg="Failure for call with function for {}\n{} != {}".format(u, value, funcsum))
            self.assertTrue(np.allclose(value, dictsum),
                            msg="Failure for call with dictionary for {}\n{} != {}".format(u, value, dictsum))
            self.assertTrue(np.allclose(2 * value, c2(u, fval)),
                            msg="Failure with scalar multiply?")
            self.assertTrue(np.allclose(2 * value, c3(u, fval)),
                            msg="Failure with addition?")
            self.assertTrue(np.allclose(value, c4(u, fval)),
                            msg="Failure with subtraction?")
            self.assertTrue(np.allclose(value + np.eye(2), c5(u, fval)),
                            msg="Failure with scalar addition?")
            self.assertTrue(np.allclose(np.dot(prod, value), c6(u, fval)),
                            msg="Failure with tensor dot product?")
            self.assertTrue(np.allclose(np.dot(value, prod), c7(u, fval)),
                            msg="Failure with tensor dot product inplace?")

    def testProduct(self):
        """Test out the evaluation functions in an expansion, using coefficient products"""

        def approxexp(u):
            """2nd order expansion of exp(u)"""
            # return 1 + u*(1 + u*(0.5 + u*(1/6 + u/24)))
            return 1 + u * (1 + u * 0.5)

        def createExpansion(n):
            return lambda u: u ** n

        c = T3D()
        for coeff in c.constructexpansion(self.basis, N=2):
            c.addterms(coeff)
        c *= {(n, l): 1. / PE.factorial(n, True) for (n, l) in
              c.nl()}  # scalar multiply to create a Taylor expansion for exp
        c2 = c * c
        for (n, l) in c2.nl():
            self.assertEqual(n, l)
        fnu = {(n, l): createExpansion(n) for (n, l) in c2.nl()}  # or could do this in previous loop

        for u in [np.zeros(3), np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.]),
                  np.array([0.234, -0.85, 1.25]),
                  np.array([1.24, 0.71, -0.98])]:
            umagn = np.sqrt(np.dot(u, u))
            fval = {nl: f(umagn) for nl, f in fnu.items()}
            # comparison value:
            value = sum(pre * approxexp(np.dot(u, vec)) for pre, vec in self.basis)
            valsum = c(u, fval)
            funcsum = c(u, fnu)
            dictsum = sum(fval[k] * v for k, v in c(u).items())

            value2 = np.dot(value, value)
            valsum2 = c2(u, fval)
            funcsum2 = c2(u, fnu)
            dictsum2 = sum(fval[k] * v for k, v in c2(u).items())

            self.assertTrue(np.allclose(value, valsum),
                            msg="Failure for call with values for {}\n{} != {}".format(u, value, valsum))
            self.assertTrue(np.allclose(value, funcsum),
                            msg="Failure for call with function for {}\n{} != {}".format(u, value, funcsum))
            self.assertTrue(np.allclose(value, dictsum),
                            msg="Failure for call with dictionary for {}\n{} != {}".format(u, value, dictsum))

            self.assertTrue(np.allclose(value2, valsum2),
                            msg="Failure for call with values for {}\n{} != {}".format(u, value2, valsum2))
            self.assertTrue(np.allclose(value2, funcsum2),
                            msg="Failure for call with function for {}\n{} != {}".format(u, value2, funcsum2))
            self.assertTrue(np.allclose(value2, dictsum2),
                            msg="Failure for call with dictionary for {}\n{} != {}".format(u, value2, dictsum2))

    def testReduceExpand(self):
        """Test our reduction and expansion operations"""

        def approxexp(u):
            """4th order expansion of exp(u)"""
            return 1 + u * (1 + u * (0.5 + u * (1 / 6 + u / 24)))

        def createExpansion(n):
            return lambda u: u ** n

        c = T3D([c[0] for c in T3D.constructexpansion(self.basis, N=4, pre=(0, 1, 1 / 2, 1 / 6, 1 / 24))])
        self.assertEqual(len(c.coefflist), 5)  # should have all n from 0 to 4
        c2 = c.copy()
        c2.reduce()
        # check the reduction: should be just two terms remaining: n=2, n=4
        self.assertEqual(len(c2.coefflist), 2)
        for n, l, coeff in c2.coefflist:
            self.assertTrue(n == 2 or n == 4)
            if n == 2:
                self.assertEqual(l, 2)
            else:
                self.assertEqual(l, 4)
        c3 = c2.copy()
        c3.separate()
        # print("c2:\n{}".format(c2))
        # print("c3:\n{}".format(c3))
        # now should have 2 + 3 = 5 terms
        self.assertEqual(len(c3.coefflist), 5)
        for n, l, coeff in c3.coefflist:
            self.assertTrue(n == 2 or n == 4)
            if n == 2:
                self.assertTrue(l == 0 or l == 2)
            else:
                self.assertTrue(l == 0 or l == 2 or l == 4)
            # also check that the only non-zero terms for a given l are value are those values
            if l == 0:
                lmin = 0
            else:
                lmin = T3D.powlrange[l - 1]
            lmax = T3D.powlrange[l]
            # self.assertTrue(np.allclose(coeff[0:lmin], 0))
            self.assertTrue(np.allclose(coeff[lmax:T3D.Npower], 0))
            self.assertFalse(np.allclose(coeff[lmin:lmax], 0))
            Ylmcoeff = np.tensordot(T3D.powYlm[:T3D.powlrange[l], :], coeff, axes=(0, 0))  # now in Ylm
            lmin = l ** 2
            lmax = (l + 1) ** 2
            self.assertTrue(np.allclose(Ylmcoeff[0:lmin], 0))
            self.assertFalse(np.allclose(Ylmcoeff[lmin:lmax], 0))
            self.assertTrue(np.allclose(Ylmcoeff[lmax:T3D.NYlm], 0))

        # a little tricky to make sure we get ALL the functions (instead of making multiple dictionaries)
        fnu = {(n, l): createExpansion(n) for (n, l) in c.nl()}  # or could do this in previous loop
        for (n, l) in c3.nl():
            if (n, l) not in fnu:
                fnu[(n, l)] = createExpansion(n)

        for u in [np.zeros(3), np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.]),
                  np.array([0.234, -0.85, 1.25]),
                  np.array([1.24, 0.71, -0.98])]:
            umagn = np.sqrt(np.dot(u, u))
            # compare values:
            self.assertTrue(np.allclose(c(u, fnu), c2(u, fnu)),
                            msg="Failure on reduce:\n{} != {}".format(c(u, fnu), c2(u, fnu)))
            self.assertTrue(np.allclose(c(u, fnu), c3(u, fnu)),
                            msg="Failure on expand:\n{} != {}".format(c(u, fnu), c3(u, fnu)))
        # do a test of projection using some random coefficients
        coeffrand = np.random.uniform(-1, 1, T3D.Npower)
        coeffrand.shape = (T3D.Npower, 1, 1)
        crand = T3D([(0, T3D.Lmax, coeffrand)])
        crand.separate()
        for (n, l, c) in crand.coefflist:
            Ylmcoeff = np.tensordot(T3D.powYlm[:T3D.powlrange[l], :], c, axes=(0, 0))  # now in Ylm
            lmin = l ** 2
            lmax = (l + 1) ** 2
            self.assertTrue(np.allclose(Ylmcoeff[0:lmin], 0))
            self.assertFalse(np.allclose(Ylmcoeff[lmin:lmax], 0))
            self.assertTrue(np.allclose(Ylmcoeff[lmax:T3D.NYlm], 0))

    def testInverse(self):
        """Test our inverse expansion"""

        # This is *very tricky* because the inverse expansion is *strictly* a Taylor series;
        # it won't be exact. Should be up to order u^2
        def approxexp(u):
            """4th order expansion of exp(u)"""
            return 1 + u * (1 + u * (0.5 + u * (1 / 6 + u / 24)))

        def createExpansion(n):
            return lambda u: u ** n

        cubicbasis = [(np.eye(2), np.array([1., 0., 0.])),
                      (np.eye(2), np.array([-1., 0., 0.])),
                      (np.eye(2), np.array([0., 1., 0.])),
                      (np.eye(2), np.array([0., -1., 0.])),
                      (np.eye(2), np.array([0., 0., 1.])),
                      (np.eye(2), np.array([0., 0., -1.]))
                      ]

        c = T3D([c[0] for c in T3D.constructexpansion(cubicbasis, N=4, pre=(0, 1, 1 / 2, 1 / 6, 1 / 24))])
        c.reduce()
        cinv = c.inv(Nmax=0)  # since c ~ x^2, cinv ~ 1/x^2, and L=4 should take us to x^0

        fnu = {(n, l): createExpansion(n) for (n, l) in c.nl()}  # or could do this in previous loop
        for (n, l) in cinv.nl():
            if (n, l) not in fnu:
                fnu[(n, l)] = createExpansion(n)

        for u in [np.array([0.25, 0., 0.]), np.array([0., 0.1, 0.]), np.array([0., 0., 0.1]),
                  np.array([0.0234, -0.085, 0.125]),
                  np.array([0.124, 0.071, -0.098])]:
            umagn = np.sqrt(np.dot(u, u))
            cval = c(u, fnu)
            cinvval = cinv(u, fnu)
            cval_inv = np.dot(cval, cinvval) - np.eye(2)
            # cval_directinv = np.linalg.inv(cval)
            self.assertTrue(np.all(abs(cval_inv) < (1 / 120) * umagn ** 4),
                            msg="cinv * c != 1?\nc={}\ncinv={}\nc*cinv-1={}".format(cval, cinvval, cval_inv))

    def testTruncation(self):
        """Make sure truncation works how we expect"""
        c = T3D([nlc[0] for nlc in T3D.constructexpansion(self.basis, N=4)])
        c.reduce()
        self.assertEqual(max(n for n, l, c in c.coefflist), 4)
        c2 = c.truncate(2)
        self.assertEqual(max(n for n, l, c in c2.coefflist), 2)
        self.assertEqual(max(n for n, l, c in c.coefflist), 4)
        c.truncate(2, inplace=True)
        self.assertEqual(max(n for n, l, c in c.coefflist), 2)

    def testIndexingSlicing(self):
        """Can we index into our expansions to get a new expansion? Can we slice? Can we assign?"""

        def createExpansion(n):
            return lambda u: u ** n

        newbasis = [(np.array([[1., 6., 5.], [5., 2., 4.], [5., 4., 3.]]), np.array([2 / 3., 1 / 3, -1 / 2]))]
        c = T3D([nlc[0] for nlc in T3D.constructexpansion(newbasis, N=4)])
        fnu = {(n, l): createExpansion(n) for n in range(5) for l in range(5)}
        # now we have something to work with. We should have a basis from n=0 up to n=4 of 3x3 matrices.
        c00 = c[0, 0]
        for u in [np.array([0.25, 0., 0.]), np.array([0., 0.1, 0.]), np.array([0., 0., 0.1]),
                  np.array([0.0234, -0.085, 0.125]),
                  np.array([0.124, 0.071, -0.098])]:
            cval = c(u, fnu)
            c00val = c00(u, fnu)
            self.assertEqual(cval[0, 0], c00val)
        # now, an assignment test. This will be funky; first, we do a "copy" operation so that
        # c00 is clean--it is no longer a "view" or slice of c, but it's own thing.
        c00 = c00.copy()
        # now, set the 0,0 value to be twice what it was before:
        c[0, 0] = 2. * c00
        for u in [np.array([0.25, 0., 0.]), np.array([0., 0.1, 0.]), np.array([0., 0., 0.1]),
                  np.array([0.0234, -0.085, 0.125]),
                  np.array([0.124, 0.071, -0.098])]:
            cval = c(u, fnu)
            c00val = c00(u, fnu)
            self.assertEqual(cval[0, 0], 2. * c00val)
        c00inv = c00.inv(Nmax=4)
        c00inv.reduce()
        for u in [np.array([0.025, 0., 0.]), np.array([0., 0.025, 0.]), np.array([0., 0., 0.025]),
                  np.array([0.0234, -0.05, 0.05]),
                  np.array([-0.024, 0.041, -0.033])]:
            c00val = c00(u, fnu)
            c00invval = c00inv(u, fnu)
            self.assertAlmostEqual(c00val * c00invval, 1)

    def testRotation(self):
        """Set of tests for rotating directions"""

        def createExpansion(n):
            return lambda u: u ** n

        newbasis = [(0.89 * np.eye(1), np.array([2 / 3., 1 / 3, -1 / 2]))]
        c = T3D([nlc[0] for nlc in T3D.constructexpansion(newbasis, N=4)])
        # does this still work if we do this?
        # c.reduce()
        fnu = {(n, l): createExpansion(n) for n in range(5) for l in range(5)}
        for rot in [np.eye(3), 2. * np.eye(3), 0.5 * np.eye(3),
                    np.array([[1.25, 0.5, 0.25], [-0.25, 0.9, 0.5], [-0.75, -0.4, 0.6]]),
                    np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])]:
            rotbasis = [(newbasis[0][0], np.dot(newbasis[0][1], rot))]
            crotdirect = T3D([nlc[0] for nlc in T3D.constructexpansion(rotbasis, N=4)])
            crot = c.rotate(c.rotatedirections(rot))
            for u in [np.array([1.2, 0., 0.]),
                      np.array([0., 1.2, 0.]),
                      np.array([0., 0., 1.2]),
                      np.array([0.234, -0.5, 0.5]),
                      np.array([-0.24, 0.41, -1.3])]:
                self.assertAlmostEqual(crot(u, fnu)[0, 0], crotdirect(u, fnu)[0, 0],
                                       msg="Failed before reduce()")
        # now, a more detailed test: do a reduce.
        c2 = c.copy()
        c.reduce()
        for rot in [np.eye(3), 2. * np.eye(3), 0.5 * np.eye(3),
                    np.array([[1.25, 0.5, 0.25], [-0.25, 0.9, 0.5], [-0.75, -0.4, 0.6]]),
                    np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])]:
            rotbasis = [(newbasis[0][0], np.dot(newbasis[0][1], rot))]
            crotdirect = T3D([nlc[0] for nlc in T3D.constructexpansion(rotbasis, N=4)])
            crot = c.rotate(c.rotatedirections(rot))
            for u in [np.array([1.2, 0., 0.]),
                      np.array([0., 1.2, 0.]),
                      np.array([0., 0., 1.2]),
                      np.array([0.234, -0.5, 0.5]),
                      np.array([-0.24, 0.41, -1.3])]:
                self.assertAlmostEqual(c(u, fnu)[0, 0], c2(u, fnu)[0, 0],
                                       msg="Failure in reduce() to produce equal function values?")
                self.assertAlmostEqual(crot(u, fnu)[0, 0], crotdirect(u, fnu)[0, 0],
                                       msg="Failed after reduce() for\n{}".format(rot))


def FourierCoeff(l, theta):
    """This is the equivalent of sph_harm for the two-dimensional case"""
    return np.exp(1j*l*theta)


class PowerExpansion2DTests(unittest.TestCase):
    """Tests to make sure our power expansions are constructed correctly and behaving as advertised"""

    def setUp(self):
        """initial setup for testing"""
        self.theta = np.pi * 0.2234
        self.c = T2D()
        self.basis = [(np.eye(2), np.array([0.5, -np.sqrt(0.75)])),
                      (np.eye(2), np.array([0.5, np.sqrt(0.75)])),
                      (2.*np.eye(2), np.array([-1., 0.])),
                      (np.eye(2), np.array([-0.5, -np.sqrt(0.75)])),
                      (np.eye(2), np.array([-0.5, np.sqrt(0.75)])),
                      (2.*np.eye(2), np.array([1., 0.]))
                      ]

    def testIndexing(self):
        for ind, l in enumerate(self.c.ind2FC):
            self.assertEqual(ind, self.c.FC2ind[l])
        for l in range(-self.c.Lmax, self.c.Lmax+1):
            self.assertEqual(l, self.c.ind2FC[self.c.FC2ind[l]])

    def testExpansionFCpow(self):
        """Test the expansion of FC into powers"""
        for theta in [self.theta + dt*np.pi for dt in np.linspace(0, 2, num=16, endpoint=False)]:
            utest, umagn = T2D.powexp(np.array([np.cos(theta), np.sin(theta)]))
            self.assertAlmostEqual(umagn, 1)
            FC0 = np.zeros(T2D.NFC, dtype=complex)
            # FC as power expansions
            for lind in range(T2D.NFC):
                l = T2D.ind2FC[lind]
                FC0[lind] = FourierCoeff(l, theta)
                FCexp = np.dot(T2D.FCpow[lind], utest)
                self.assertAlmostEqual(FC0[lind], FCexp,
                                       msg="Failure for FCpow " 
                    "l={}; theta={}\n{} != {}".format(l, theta, FC0[lind], FCexp))
            # power expansions in FC's
            for p in range(T2D.NFC):
                pFC = np.dot(T2D.powFC[p], FC0)
                self.assertAlmostEqual(utest[p], pFC,
                                       msg="Failure for powFC " 
                    "{}; theta={}\n{} != {}".format(T2D.ind2pow[p], theta, utest[p], pFC))
            # projection (note that Lproj is not symmetric): so this test ensures that v.u and (proj.v).u
            # give the same value
            uproj = np.tensordot(T2D.Lproj[-1], utest, axes=(0, 0))
            for p in range(T2D.NFC):
                self.assertAlmostEqual(utest[p], uproj[p],
                                       msg="Projection failure for " 
                    "{}\n{} != {}".format(T2D.ind2pow[p], uproj[p], utest[p]))

    def testProjection(self):
        """Test that the L-projections are correct"""
        # Try to do this sequentially
        for tup in [(0, 0), (1, 0), (0, 1)]:
            v = np.zeros(T2D.Npower)
            v[T2D.pow2ind[tup]] = 1.
            Pv = np.tensordot(T2D.Lproj[-1], v, axes=1)
            # now, try with multiplying by x^2+y^2+z^2:
            vxy = np.zeros(T2D.Npower)
            vxy[T2D.pow2ind[tup[0] + 2, tup[1]]] = 1.
            vxy[T2D.pow2ind[tup[0], tup[1] + 2]] = 1.
            Pvxy = np.tensordot(T2D.Lproj[-1], vxy, axes=1)
            self.assertTrue(np.allclose(v, Pv))
            self.assertTrue(np.allclose(v, Pvxy))
            for l in range(T2D.Lmax + 1):
                Pv = np.tensordot(T2D.Lproj[l], v, axes=1)
                Pvxy = np.tensordot(T2D.Lproj[l], vxy, axes=1)
                if l == sum(tup):
                    self.assertTrue(np.allclose(v, Pv))
                    self.assertTrue(np.allclose(v, Pvxy))
                else:
                    self.assertTrue(np.allclose(Pv, 0))
                    self.assertTrue(np.allclose(Pvxy, 0))

    def testEvaluation(self):
        """Test out the evaluation functions in an expansion, including with scalar multiply and addition"""

        def approxexp(u):
            """4th order expansion of exp(u)"""
            return 1 + u * (1 + u * (0.5 + u * (1 / 6 + u / 24)))

        def createExpansion(n):
            return lambda u: u ** n / PE.factorial(n, True)

        c = T2D()
        for coeff in c.constructexpansion(self.basis):
            c.addterms(coeff)
        for (n, l) in c.nl():
            self.assertEqual(n, l)
        fnu = {(n, l): createExpansion(n) for (n, l) in c.nl()}  # or could do this in previous loop

        # c2 = 2*c
        c2 = c.copy()
        c2 *= 2
        c3 = c + c
        c4 = c2 - c
        ### NOTE! We have to do it *this way*; otherwise, it will try to use the sum in np.array,
        ### and that WILL NOT WORK with our expansion.
        c5 = c + np.eye(2)
        prod = np.array([[-4.2, 2.67], [1.3, 3.21]])
        c6 = c.ldot(prod)
        c7 = c.copy()
        c7.irdot(prod)
        sum([c, c2, c3])  # tests whether we can use sum

        for u in [np.zeros(2), np.array([1., 0.]), np.array([0., 1.]), np.array([0., 0.]),
                  np.array([0.234, -0.85]),
                  np.array([1.24, 0.71])]:
            umagn = np.sqrt(np.dot(u, u))
            fval = {nl: f(umagn) for nl, f in fnu.items()}
            # comparison value:
            value = sum(pre * approxexp(np.dot(u, vec)) for pre, vec in self.basis)
            valsum = c(u, fval)
            funcsum = c(u, fnu)
            dictsum = sum(fval[k] * v for k, v in c(u).items())

            self.assertTrue(np.allclose(value, valsum),
                            msg="Failure for call with values for {}\n{} != {}".format(u, value, valsum))
            self.assertTrue(np.allclose(value, funcsum),
                            msg="Failure for call with function for {}\n{} != {}".format(u, value, funcsum))
            self.assertTrue(np.allclose(value, dictsum),
                            msg="Failure for call with dictionary for {}\n{} != {}".format(u, value, dictsum))
            self.assertTrue(np.allclose(2 * value, c2(u, fval)),
                            msg="Failure with scalar multiply?")
            self.assertTrue(np.allclose(2 * value, c3(u, fval)),
                            msg="Failure with addition?")
            self.assertTrue(np.allclose(value, c4(u, fval)),
                            msg="Failure with subtraction?")
            self.assertTrue(np.allclose(value + np.eye(2), c5(u, fval)),
                            msg="Failure with scalar addition?")
            self.assertTrue(np.allclose(np.dot(prod, value), c6(u, fval)),
                            msg="Failure with tensor dot product?")
            self.assertTrue(np.allclose(np.dot(value, prod), c7(u, fval)),
                            msg="Failure with tensor dot product inplace?")

    def testProduct(self):
        """Test out the evaluation functions in an expansion, using coefficient products"""

        def approxexp(u):
            """2nd order expansion of exp(u)"""
            # return 1 + u*(1 + u*(0.5 + u*(1/6 + u/24)))
            return 1 + u * (1 + u * 0.5)

        def createExpansion(n):
            return lambda u: u ** n

        c = T2D()
        for coeff in c.constructexpansion(self.basis, N=2):
            c.addterms(coeff)
        c *= {(n, l): 1. / PE.factorial(n, True) for (n, l) in
              c.nl()}  # scalar multiply to create a Taylor expansion for exp
        c2 = c * c
        for (n, l) in c2.nl():
            self.assertEqual(n, l)
        fnu = {(n, l): createExpansion(n) for (n, l) in c2.nl()}  # or could do this in previous loop

        for u in [np.zeros(2), np.array([1., 0.]), np.array([0., 1.]),
                  np.array([0.234, -0.85]), np.array([1.24, 0.71])]:
            umagn = np.sqrt(np.dot(u, u))
            fval = {nl: f(umagn) for nl, f in fnu.items()}
            # comparison value:
            value = sum(pre * approxexp(np.dot(u, vec)) for pre, vec in self.basis)
            valsum = c(u, fval)
            funcsum = c(u, fnu)
            dictsum = sum(fval[k] * v for k, v in c(u).items())

            value2 = np.dot(value, value)
            valsum2 = c2(u, fval)
            funcsum2 = c2(u, fnu)
            dictsum2 = sum(fval[k] * v for k, v in c2(u).items())

            self.assertTrue(np.allclose(value, valsum),
                            msg="Failure for call with values for {}\n{} != {}".format(u, value, valsum))
            self.assertTrue(np.allclose(value, funcsum),
                            msg="Failure for call with function for {}\n{} != {}".format(u, value, funcsum))
            self.assertTrue(np.allclose(value, dictsum),
                            msg="Failure for call with dictionary for {}\n{} != {}".format(u, value, dictsum))

            self.assertTrue(np.allclose(value2, valsum2),
                            msg="Failure for call with values for {}\n{} != {}".format(u, value2, valsum2))
            self.assertTrue(np.allclose(value2, funcsum2),
                            msg="Failure for call with function for {}\n{} != {}".format(u, value2, funcsum2))
            self.assertTrue(np.allclose(value2, dictsum2),
                            msg="Failure for call with dictionary for {}\n{} != {}".format(u, value2, dictsum2))

    def testReduceExpand(self):
        """Test our reduction and expansion operations"""

        def approxexp(u):
            """4th order expansion of exp(u)"""
            return 1 + u * (1 + u * (0.5 + u * (1 / 6 + u / 24)))

        def createExpansion(n):
            return lambda u: u ** n

        c = T2D([c[0] for c in T2D.constructexpansion(self.basis, N=4, pre=(0, 1, 1 / 2, 1 / 6, 1 / 24))])
        self.assertEqual(len(c.coefflist), 5)  # should have all n from 0 to 4
        c2 = c.copy()
        c2.reduce()
        # check the reduction: should be just two terms remaining: n=2, n=4
        self.assertEqual(len(c2.coefflist), 2)
        for n, l, coeff in c2.coefflist:
            self.assertTrue(n == 2 or n == 4)
            if n == 2:
                self.assertEqual(l, 2)
            else:
                self.assertEqual(l, 4)
        c3 = c2.copy()
        c3.separate()
        # print("c2:\n{}".format(c2))
        # print("c3:\n{}".format(c3))
        # now should have 2 + 3 = 5 terms
        self.assertEqual(len(c3.coefflist), 5)
        for n, l, coeff in c3.coefflist:
            self.assertTrue(n == 2 or n == 4)
            if n == 2:
                self.assertTrue(l == 0 or l == 2)
            else:
                self.assertTrue(l == 0 or l == 2 or l == 4)
            # also check that the only non-zero terms for a given l are value are those values
            lmin, lmax = T2D.powlrange[l-1], T2D.powlrange[l]
            self.assertTrue(np.allclose(coeff[0:lmin], 0))
            self.assertTrue(np.allclose(coeff[lmax:T2D.Npower], 0))
            self.assertFalse(np.allclose(coeff[lmin:lmax], 0))
            # check directly the Fourier transform:
            FCcoeff = np.tensordot(T2D.powFC[:T2D.powlrange[l], :], coeff, axes=(0, 0))  # now in FC
            # only the lplus and lminus should be non-zero:
            lp, lm = T2D.FC2ind[l], T2D.FC2ind[-l]
            for lind in range(T2D.NFC):
                if lind != lp and lind != lm:
                    self.assertAlmostEqual(np.sum(np.abs(FCcoeff[lind])), 0)
            self.assertNotAlmostEqual(np.sum(np.abs(FCcoeff[lp])+np.abs(FCcoeff[lm])), 0)

        # a little tricky to make sure we get ALL the functions (instead of making multiple dictionaries)
        fnu = {(n, l): createExpansion(n) for (n, l) in c.nl()}  # or could do this in previous loop
        for (n, l) in c3.nl():
            if (n, l) not in fnu:
                fnu[(n, l)] = createExpansion(n)

        for u in [np.zeros(2), np.array([1., 0.]), np.array([0., 1.]),
                  np.array([0.234, -0.85]), np.array([1.24, 0.71])]:
            umagn = np.sqrt(np.dot(u, u))
            # compare values:
            self.assertTrue(np.allclose(c(u, fnu), c2(u, fnu)),
                            msg="Failure on reduce:\n{} != {}".format(c(u, fnu), c2(u, fnu)))
            self.assertTrue(np.allclose(c(u, fnu), c3(u, fnu)),
                            msg="Failure on expand:\n{} != {}".format(c(u, fnu), c3(u, fnu)))
        # do a test of projection using some random coefficients
        coeffrand = np.random.uniform(-1, 1, T2D.Npower)
        coeffrand.shape = (T2D.Npower, 1, 1)
        crand = T2D([(0, T2D.Lmax, coeffrand)])
        crand.separate()
        for (n, l, c) in crand.coefflist:
            FCcoeff = np.tensordot(T2D.powFC[:T2D.powlrange[l], :], c, axes=(0, 0))  # now in FC
            # only the lplus and lminus should be non-zero:
            lp, lm = T2D.FC2ind[l], T2D.FC2ind[-l]
            for lind in range(T2D.NFC):
                if lind != lp and lind != lm:
                    self.assertAlmostEqual(np.sum(np.abs(FCcoeff[lind])), 0)
            self.assertNotAlmostEqual(np.sum(np.abs(FCcoeff[lp])+np.abs(FCcoeff[lm])), 0)

    def testInverse(self):
        """Test our inverse expansion"""

        # This is *very tricky* because the inverse expansion is *strictly* a Taylor series;
        # it won't be exact. Should be up to order u^2
        def approxexp(u):
            """4th order expansion of exp(u)"""
            return 1 + u * (1 + u * (0.5 + u * (1 / 6 + u / 24)))

        def createExpansion(n):
            return lambda u: u ** n

        cubicbasis = [(np.eye(2), np.array([1., 0.])),
                      (np.eye(2), np.array([-1., 0.])),
                      (np.eye(2), np.array([0., 1.])),
                      (np.eye(2), np.array([0., -1.]))
                      ]

        c = T2D([c[0] for c in T2D.constructexpansion(cubicbasis, N=4, pre=(0, 1, 1 / 2, 1 / 6, 1 / 24))])
        c.reduce()
        cinv = c.inv(Nmax=0)  # since c ~ x^2, cinv ~ 1/x^2, and L=4 should take us to x^0

        fnu = {(n, l): createExpansion(n) for (n, l) in c.nl()}  # or could do this in previous loop
        for (n, l) in cinv.nl():
            if (n, l) not in fnu:
                fnu[(n, l)] = createExpansion(n)

        for u in [np.array([0.25, 0.]), np.array([0., 0.1]),
                  np.array([0.0234, -0.085]), np.array([0.124, 0.071])]:
            umagn = np.sqrt(np.dot(u, u))
            cval = c(u, fnu)
            cinvval = cinv(u, fnu)
            cval_inv = np.dot(cval, cinvval) - np.eye(2)
            # cval_directinv = np.linalg.inv(cval)
            self.assertTrue(np.all(abs(cval_inv) < (1 / 120) * umagn ** 4),
                            msg="cinv * c != 1?\nc={}\ncinv={}\nc*cinv-1={}".format(cval, cinvval, cval_inv))

    def testTruncation(self):
        """Make sure truncation works how we expect"""
        c = T2D([nlc[0] for nlc in T2D.constructexpansion(self.basis, N=4)])
        c.reduce()
        self.assertEqual(max(n for n, l, c in c.coefflist), 4)
        c2 = c.truncate(2)
        self.assertEqual(max(n for n, l, c in c2.coefflist), 2)
        self.assertEqual(max(n for n, l, c in c.coefflist), 4)
        c.truncate(2, inplace=True)
        self.assertEqual(max(n for n, l, c in c.coefflist), 2)

    def testIndexingSlicing(self):
        """Can we index into our expansions to get a new expansion? Can we slice? Can we assign?"""

        def createExpansion(n):
            return lambda u: u ** n

        newbasis = [(np.array([[1., -2.], [-2., 2.]]), np.array([2 / 3., 1 / 3]))]
        c = T2D([nlc[0] for nlc in T2D.constructexpansion(newbasis, N=4)])
        fnu = {(n, l): createExpansion(n) for n in range(5) for l in range(5)}
        # now we have something to work with. We should have a basis from n=0 up to n=4 of 2x2 matrices.
        c00 = c[0, 0]
        for u in [np.array([0.25, 0.]), np.array([0., 0.1]),
                  np.array([0.0234, -0.085]), np.array([0.124, 0.071])]:
            cval = c(u, fnu)
            c00val = c00(u, fnu)
            self.assertEqual(cval[0, 0], c00val)
        # now, an assignment test. This will be funky; first, we do a "copy" operation so that
        # c00 is clean--it is no longer a "view" or slice of c, but it's own thing.
        c00 = c00.copy()
        # now, set the 0,0 value to be twice what it was before:
        c[0, 0] = 2. * c00
        for u in [np.array([0.25, 0.]), np.array([0., 0.1]),
                  np.array([0.0234, -0.085]), np.array([0.124, 0.071])]:
            cval = c(u, fnu)
            c00val = c00(u, fnu)
            self.assertEqual(cval[0, 0], 2. * c00val)
        c00inv = c00.inv(Nmax=4)
        c00inv.reduce()
        for u in [np.array([0.025, 0.]), np.array([0., 0.01]),
                  np.array([0.0234, -0.085]), np.array([0.0124, 0.071])]:
            c00val = c00(u, fnu)
            c00invval = c00inv(u, fnu)
            self.assertAlmostEqual(c00val * c00invval, 1)

    def testRotation(self):
        """Set of tests for rotating directions"""

        def createExpansion(n):
            return lambda u: u ** n

        newbasis = [(0.89 * np.eye(1), np.array([2 / 3., 1 / 3]))]
        c = T2D([nlc[0] for nlc in T2D.constructexpansion(newbasis, N=4)])
        # does this still work if we do this?
        # c.reduce()
        fnu = {(n, l): createExpansion(n) for n in range(5) for l in range(5)}
        for rot in [np.eye(2), 2. * np.eye(2), 0.5 * np.eye(2),
                    np.array([[1.25, 0.5], [-0.25, 0.9]]),
                    np.array([[0., 1.], [-1., 0.]])]:
            rotbasis = [(newbasis[0][0], np.dot(newbasis[0][1], rot))]
            crotdirect = T2D([nlc[0] for nlc in T2D.constructexpansion(rotbasis, N=4)])
            crot = c.rotate(c.rotatedirections(rot))
            for u in [np.array([1.2, 0.]),
                      np.array([0., 1.2]),
                      np.array([0., 0.]),
                      np.array([0.234, -0.5]),
                      np.array([-0.24, 0.41])]:
                self.assertAlmostEqual(crot(u, fnu)[0, 0], crotdirect(u, fnu)[0, 0],
                                       msg="Failed before reduce()")
        # now, a more detailed test: do a reduce.
        c2 = c.copy()
        c.reduce()
        for rot in [np.eye(2), 2. * np.eye(2), 0.5 * np.eye(2),
                    np.array([[1.25, 0.5], [-0.25, 0.9]]),
                    np.array([[0., 1.], [-1., 0.]])]:
            rotbasis = [(newbasis[0][0], np.dot(newbasis[0][1], rot))]
            crotdirect = T2D([nlc[0] for nlc in T2D.constructexpansion(rotbasis, N=4)])
            crot = c.rotate(c.rotatedirections(rot))
            for u in [np.array([1.2, 0.]),
                      np.array([0., 1.2]),
                      np.array([0., 0.]),
                      np.array([0.234, -0.5]),
                      np.array([-0.24, 0.41])]:
                self.assertAlmostEqual(c(u, fnu)[0, 0], c2(u, fnu)[0, 0],
                                       msg="Failure in reduce() to produce equal function values?")
                self.assertAlmostEqual(crot(u, fnu)[0, 0], crotdirect(u, fnu)[0, 0],
                                       msg="Failed after reduce() for\n{}".format(rot))
