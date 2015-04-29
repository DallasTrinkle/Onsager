"""
Unit tests for lattice structure
"""

__author__ = 'Dallas R. Trinkle'

# TODO: additional lattice structures?
# TODO: use spglib and/or interface with pymatgen to automatically construct accordingly

import unittest
import onsager.FCClatt as FCClatt
import numpy as np

class LatticeTests(unittest.TestCase):
    """Set of tests that our lattice code is behaving correctly"""

    def setUp(self):
        self.lattice = FCClatt.lattice()
        self.NNvect = FCClatt.NNvect()
        self.invlist = FCClatt.invlist(self.NNvect)

    def testFCCcount(self):
        """Check that we have z=12 neighbors, and we're in 3D"""
        self.assertEqual(np.shape(self.NNvect), (12, 3))

    def testFCCinversioncount(self):
        """Right dimensions for matrices?"""
        self.assertEqual(np.shape(self.NNvect)[0], np.shape(self.invlist)[0])
        self.assertEqual(np.shape(self.lattice), (3, 3))

    def testFCCbasic(self):
        """Do we have the right <110> nearest neighbor vectors for FCC?"""
        self.assertTrue(any(all((1, 1, 0) == x) for x in self.NNvect))
        self.assertTrue(any(all((1, -1, 0) == x) for x in self.NNvect))
        self.assertTrue(any(all((0, 1, 1) == x) for x in self.NNvect))
        self.assertTrue(any(all((0, 1, -1) == x) for x in self.NNvect))
        self.assertTrue(any(all((1, 0, 1) == x) for x in self.NNvect))
        self.assertTrue(any(all((-1, 0, 1) == x) for x in self.NNvect))
        self.assertFalse(any(all((0, 0, 0) == x) for x in self.NNvect))
        self.assertFalse(any(all((1, 1, 1) == x) for x in self.NNvect))

    def testFCCinversion(self):
        """Check that for each NN vector, we have its inverse too, from invlist"""
        for k1, k2 in enumerate(self.invlist):
            self.assertTrue(all(self.NNvect[k1] == -self.NNvect[k2]))

    def testNNvectlatticevect(self):
        """Check that our NN vectors are all lattice vectors"""
        invlatt = np.linalg.inv(self.lattice)
        for vec in self.NNvect:
            uvec = np.dot(invlatt, vec)
            for d in range(3):
                self.assertAlmostEqual(uvec[d], round(uvec[d]))
