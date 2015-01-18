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
        self.NNvect = FCClatt.NNvect()


    def testFTisfunc(self):
        """Do we get a function as DFT?"""
        self.assertTrue(callable(self.DFT))
