#!/usr/bin/env python
"""
Unit tests for vacancy mediated diffusion code.
"""

import unittest
import numpy as np
import FCClatt

class LatticeTests(unittest.TestCase):
        
    def testbase(self):
        self.failUnless(1+1 == 2)
        
    def testFCCcount(self):
        nnvect = FCClatt.makeNNvect()
        print np.shape(nnvect)
        self.failUnless(np.shape(nnvect) == [12,3])

def main():
    unittest.main()

if __name__ == '__main__':
    main()
