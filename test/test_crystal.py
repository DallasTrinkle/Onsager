"""
Unit tests for crystal class
"""

__author__ = 'Dallas R. Trinkle'

import unittest
import numpy as np
import onsager.crystal as crystal

class CrystalClassTests(unittest.TestCase):
    """Tests for the crystal class and symmetry analysis."""

    def setUp(self):
        self.a0 = 2.5
        self.sclatt = self.a0*np.eye(3)
        self.fcclatt = self.a0*np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
        self.bcclatt = self.a0*np.array([[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]])
        self.basis = [np.array([0.,0.,0.])]

    def testscMetric(self):
        """Does the simple cubic lattice have the right volume and metric?"""
        crys = crystal.Crystal(self.sclatt, self.basis)
        self.assertAlmostEqual(crys.volume, self.a0**3)
        for i, a2 in enumerate(crys.metric.flatten()):
            if i%4 == 0:
                # diagonal element
                self.assertAlmostEqual(a2, self.a0**2)
            else:
                # off-diagonal element
                self.assertAlmostEqual(a2, 0)

    def testfccMetric(self):
        """Does the face-centered cubic lattice have the right volume and metric?"""
        crys = crystal.Crystal(self.fcclatt, self.basis)
        self.assertAlmostEqual(crys.volume, 0.25*self.a0**3)
        for i, a2 in enumerate(crys.metric.flatten()):
            if i%4 == 0:
                # diagonal element
                self.assertAlmostEqual(a2, 0.5*self.a0**2)
            else:
                # off-diagonal element
                self.assertAlmostEqual(a2, 0.25*self.a0**2)

    def testbccMetric(self):
        """Does the body-centered cubic lattice have the right volume and metric?"""
        crys = crystal.Crystal(self.bcclatt, self.basis)
        self.assertAlmostEqual(crys.volume, 0.5*self.a0**3)
        for i, a2 in enumerate(crys.metric.flatten()):
            if i%4 == 0:
                # diagonal element
                self.assertAlmostEqual(a2, 0.75*self.a0**2)
            else:
                # off-diagonal element
                self.assertAlmostEqual(a2, -0.25*self.a0**2)
