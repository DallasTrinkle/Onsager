#!/usr/bin/env python
"""
Unit tests for vacancy mediated diffusion code.
"""

import unittest
import numpy as np
import FCClatt

class LatticeTests(unittest.TestCase):
    def setUp(self):
        self.NNvect = FCClatt.NNvect()
        self.invlist = FCClatt.invlist(self.NNvect)

    def testFCCcount(self):
        # check that we have z=12 neighbors, and we're in 3D        
        self.assertEqual(np.shape(self.NNvect), (12,3))

    def testFCCinversioncount(self):
        # right dimensions?
        self.assertEqual(np.shape(self.NNvect)[0], np.shape(self.invlist)[0])

    def testFCCbasic(self):
        # do we have the right <110> nearest neighbor vectors?
        self.assertTrue(any( all((1,1,0)==x) for x in self.NNvect ))
        self.assertTrue(any( all((1,-1,0)==x) for x in self.NNvect ))
        self.assertTrue(any( all((0,1,1)==x) for x in self.NNvect ))
        self.assertTrue(any( all((0,1,-1)==x) for x in self.NNvect ))
        self.assertTrue(any( all((1,0,1)==x) for x in self.NNvect ))
        self.assertTrue(any( all((-1,0,1)==x) for x in self.NNvect ))
        self.assertFalse(any( all((0,0,0)==x) for x in self.NNvect ))
        self.assertFalse(any( all((1,1,1)==x) for x in self.NNvect ))
        
    def testFCCinversion(self):
        # check that for each NN vector, we have its inverse too
        for k1, k2 in enumerate(self.invlist):
            self.assertTrue(all(self.NNvect[k1] == -self.NNvect[k2]))

# At some point, we'll probably want to included point group operations, and appropriate tests...

import GFcalc

class GreenFuncFourierTransformTests(unittest.TestCase):
    def setUp(self):
        self.NNvect = FCClatt.NNvect()
        self.rates = np.array((1,)*np.shape(self.NNvect)[0])

    def testFTfunc(self):
        GFFT = GFcalc.GFFTfunc(self.NNvect, self.rates)
        self.assertTrue(callable(GFFT))
        
    def testFTdim(self):
        GFdiff = GFcalc.GFdiff(self.NNvect, self.rates)
        self.assertTrue(np.shape(GFdiff)==(3,3))
            
def main():
    unittest.main()

if __name__ == '__main__':
    main()
