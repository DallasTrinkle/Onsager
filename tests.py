#!/usr/bin/env python
"""
Unit tests for vacancy mediated diffusion code.
"""

import unittest
import numpy as np
import FCClatt

class LatticeTests(unittest.TestCase):
    def setUp(self):
        self.nnvect = FCClatt.makeNNvect()
        self.invlist = FCClatt.invlist(self.nnvect)

    def testFCCcount(self):
        # check that we have z=12 neighbors, and we're in 3D        
        self.assertEqual(np.shape(self.nnvect), (12,3))

    def testFCCinversion(self):
        # check that for each NN vector, we have its inverse too
        for k1, k2 in enumerate(self.invlist):
            self.assertEqual(self.nnvect[k1], -self.nnvect[k2])

def main():
    unittest.main()

if __name__ == '__main__':
    main()
