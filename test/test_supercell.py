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
    longMessage = False

    def setUp(self):
        self.crys = crystal.Crystal.FCC(1.,'Al')
        self.one = np.eye(3, dtype=int)

    def testSuper(self):
        """Can we make a supercell object?"""
        super = supercell.Supercell(self.crys, self.one)
        self.assertNotEqual(super, None)
        self.assertEqual(super.Nchem, 2)
        super = supercell.Supercell(self.crys, self.one, interstitial = (1,))
        self.assertNotEqual(super, None)
        super = supercell.Supercell(self.crys, self.one, Nchem=5)
        self.assertNotEqual(super, None)
        self.assertEqual(super.Nchem, 5)

    def testEqualityCopy(self):
        """Can we copy a supercell, and is it equal to itself?"""
        super = supercell.Supercell(self.crys, self.one)
        super2 = super.copy()
        self.assertEqual(super, super2, msg="{}\n!=\n{}".format(super, super2))
