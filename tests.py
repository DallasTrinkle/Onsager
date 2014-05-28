#!/usr/bin/env python
"""
Unit tests for vacancy mediated diffusion code.
"""

import unittest

class LatticeTests(unittest.TestCase):
    def testbase(self):
        self.failUnless(1+1 == 2)
        
    def testFCCcount(self):
        nnvect = FCClatt.makeNNvect()
        self.failUnless(nnvect.dim() = [12,3])

def main():
    unittest.main()

if __name__ == '__main__':
    main()
