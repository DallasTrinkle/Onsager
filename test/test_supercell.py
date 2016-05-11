"""
Unit tests for supercell class
"""

__author__ = 'Dallas R. Trinkle'

import unittest
import numpy as np
import onsager.crystal as crystal
import onsager.supercell as supercell

class TypeTests(unittest.TestCase):
    """Tests to make sure we can make a supercell object."""
    def setUp(self):
        self.crys = crystal.Crystal.FCC(1.,'Al')

    def testSuper(self):
        """Can we make a supercell object?"""
        super = supercell.Supercell(self.crys, np.eye(3, dtype=int))
        self.assertNotEqual(super, None)
