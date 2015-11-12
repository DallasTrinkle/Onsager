"""
Unit tests for PowerExpansion class
"""

__author__ = 'Dallas R. Trinkle'

import unittest
import numpy as np
from scipy.special import sph_harm
import onsager.PowerExpansion as PE

T3D = PE.Taylor3D

class PowerExpansionTests(unittest.TestCase):
    """Tests to make sure our power expansions are constructed correclty and behaving as advertised"""
    def setUp(self):
        """initial setup for testing"""
        self.phi = np.pi*0.2234
        self.theta = np.pi*0.7261
        self.c = T3D()
        self.basis = [(np.eye(2), np.array([0.5,-np.sqrt(0.75),0.])),
         (np.eye(2), np.array([0.5,np.sqrt(0.75),0.])),
         (np.eye(2), np.array([-1.,0.,0.])),
         (np.eye(2), np.array([-0.5,-np.sqrt(0.75),0.])),
         (np.eye(2), np.array([-0.5,np.sqrt(0.75),0.])),
         (np.eye(2), np.array([1.,0.,0.])),
         (np.eye(2)*2, np.array([0.,0.,1.])),
         (np.eye(2)*2, np.array([0.,0.,-1.])),
        ]

    def testExpansionYlmpow(self):
        """Test the expansion of Ylm into powers"""
        for (phi, theta) in [ (self.phi + dp, self.theta + dt)
                              for dp in (0., 0.25*np.pi, 0.5*np.pi, 0.75*np.pi)
                              for dt in (0., 0.5*np.pi, np.pi, 1.5*np.pi)]:
            utest,umagn = T3D.powexp(np.array([np.sin(phi)*np.cos(theta),
                                               np.sin(phi)*np.sin(theta),
                                               np.cos(phi)]))
            self.assertAlmostEquals(umagn, 1)
            Ylm0 = np.zeros(T3D.NYlm, dtype=complex)
            # Ylm as power expansions
            for lm in range(T3D.NYlm):
                l, m = T3D.ind2Ylm[lm][0], T3D.ind2Ylm[lm][1]
                Ylm0[lm] = sph_harm(m, l, theta, phi)
                Ylmexp = np.dot(T3D.Ylmpow[lm], utest)
                self.assertAlmostEquals(Ylm0[lm], Ylmexp,
                                        msg="Failure for Ylmpow l={} m={}; theta={}, phi={}\n{} != {}".format(l, m, theta, phi, Ylm0[lm], Ylmexp))
            # power expansions in Ylm's
            for p in range(T3D.NYlm):
                pYlm = np.dot(T3D.powYlm[p], Ylm0)
                self.assertAlmostEquals(utest[p], pYlm,
                                        msg="Failure for powYlm {}; theta={}, phi={}\n{} != {}".format(T3D.ind2pow[p], theta, phi, utest[p], pYlm))
            # projection (note that Lproj is symmetric)
            for p in range(T3D.NYlm):
                uproj = np.dot(T3D.Lproj[5][p], utest)
                self.assertAlmostEquals(utest[p], uproj,
                                        msg="Projection failure for {}\n{} != {}".format(T3D.ind2pow[p], uproj, utest[p]))

    def testEvaluation(self):
        """Test out the evaluation functions in an expansion, including with scalar multiply and addition"""
        def approxexp(u):
            """4th order expansion of exp(u)"""
            return 1 + u*(1 + u*(0.5 + u*(1/6 + u/24)))
        def createExpansion(n):
            return lambda u: u**n / PE.factorial(n, True)

        c = T3D()
        for coeff in c.constructexpansion(self.basis):
            c.addterms(coeff)
        for (n,l) in c.nl():
            self.assertEqual(n, l)
        fnu = { (n,l): createExpansion(n) for (n,l) in c.nl() } # or could do this in previous loop

        c2 = 2.*c
        c3 = c + c
        c4 = c2 - c
        ### NOTE! We have to do it *this way*; otherwise, it will try to use the sum in np.array,
        ### and that WILL NOT WORK with our expansion.
        c5 = c + np.eye(2)

        for u in [ np.zeros(3), np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.]),
                   np.array([0.234, -0.85, 1.25]),
                   np.array([1.24, 0.71, -0.98])]:
            umagn = np.sqrt(np.dot(u, u))
            fval = { nl: f(umagn) for nl,f in fnu.items()}
            # comparison value:
            value = sum(pre*approxexp(np.dot(u, vec)) for pre, vec in self.basis)
            valsum = c(u, fval)
            funcsum = c(u, fnu)
            dictsum = sum( fval[k]*v for k,v in c(u).items())

            self.assertTrue(np.all(np.isclose(value, valsum)),
                            msg="Failure for call with values for {}\n{} != {}".format(u, value, valsum))
            self.assertTrue(np.all(np.isclose(value, funcsum)),
                            msg="Failure for call with function for {}\n{} != {}".format(u, value, funcsum))
            self.assertTrue(np.all(np.isclose(value, dictsum)),
                            msg="Failure for call with dictionary for {}\n{} != {}".format(u, value, dictsum))
            self.assertTrue(np.all(np.isclose(2*value, c2(u, fval))),
                            msg="Failure with scalar multiply?")
            self.assertTrue(np.all(np.isclose(2*value, c3(u, fval))),
                            msg="Failure with addition?")
            self.assertTrue(np.all(np.isclose(value, c4(u, fval))),
                            msg="Failure with subtraction?")
            self.assertTrue(np.all(np.isclose(value + np.eye(2), c5(u, fval))),
                            msg="Failure with scalar addition?")

    def testProduct(self):
        """Test out the evaluation functions in an expansion, including with scalar multiply and addition"""
        def approxexp(u):
            """2nd order expansion of exp(u)"""
            # return 1 + u*(1 + u*(0.5 + u*(1/6 + u/24)))
            return 1 + u*(1 + u*0.5)
        def createExpansion(n):
            return lambda u: u**n

        c = T3D()
        # print("c: ", c.coefflist)
        # print(c.constructexpansion(self.basis, N=2))
        for coeff in c.constructexpansion(self.basis, N=2):
            c.addterms(coeff)
        c *= { (n,l): 1./PE.factorial(n, True) for (n,l) in c.nl() } # scalar multiply to match the expansion
        print("c:\n", c)
        c2 = c * c
        print("c^2:\n", c2)
        for (n,l) in c2.nl():
            self.assertEqual(n, l)
        fnu = { (n,l): createExpansion(n) for (n,l) in c2.nl() } # or could do this in previous loop

        for u in [ np.zeros(3), np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.]),
                   np.array([0.234, -0.85, 1.25]),
                   np.array([1.24, 0.71, -0.98])]:
            umagn = np.sqrt(np.dot(u, u))
            fval = { nl: f(umagn) for nl,f in fnu.items()}
            # comparison value:
            value = sum(pre*approxexp(np.dot(u, vec)) for pre, vec in self.basis)
            valsum = c(u, fval)
            funcsum = c(u, fnu)
            dictsum = sum( fval[k]*v for k,v in c(u).items())

            value2 = np.dot(value, value)
            valsum2 = c2(u, fval)
            funcsum2 = c2(u, fnu)
            dictsum2 = sum( fval[k]*v for k,v in c2(u).items())

            self.assertTrue(np.all(np.isclose(value, valsum)),
                            msg="Failure for call with values for {}\n{} != {}".format(u, value, valsum))
            self.assertTrue(np.all(np.isclose(value, funcsum)),
                            msg="Failure for call with function for {}\n{} != {}".format(u, value, funcsum))
            self.assertTrue(np.all(np.isclose(value, dictsum)),
                            msg="Failure for call with dictionary for {}\n{} != {}".format(u, value, dictsum))

            self.assertTrue(np.all(np.isclose(value2, valsum2)),
                            msg="Failure for call with values for {}\n{} != {}".format(u, value2, valsum2))
            self.assertTrue(np.all(np.isclose(value2, funcsum2)),
                            msg="Failure for call with function for {}\n{} != {}".format(u, value2, funcsum2))
            self.assertTrue(np.all(np.isclose(value2, dictsum2)),
                            msg="Failure for call with dictionary for {}\n{} != {}".format(u, value2, dictsum2))
