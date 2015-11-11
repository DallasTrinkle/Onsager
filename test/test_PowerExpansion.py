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
