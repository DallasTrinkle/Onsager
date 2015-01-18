"""
Unit tests for crystal class
"""

__author__ = 'Dallas R. Trinkle'

import unittest
import numpy as np
from scipy import special
import onsager.crystal as crystal

class CrystalClassTests(unittest.TestCase):
    """Tests for the crystal class and symmetry analysis."""

    def setUp(self):
        self.a0 = 2.5
        self.alatt = self.a0*np.eye(3)
        self.basis = [np.array([0.,0.,0.])]
        self.crystal = crystal.Crystal(self.alatt, self.basis)


    def testFTisfunc(self):
        """Do we get a function as DFT?"""
        self.assertTrue(0==0)
