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
        self.GFFT = GFcalc.GFFTfunc(self.NNvect, self.rates)
        self.GFdiff = GFcalc.GFdiff(self.NNvect, self.rates)

    def testFTisfunc(self):
        self.assertTrue(callable(self.GFFT))

    def testFTfuncZero(self):
        q=np.array((0,0,0))
        self.assertEqual(self.GFFT(q),0)
        
    def testFTfuncZeroBZ(self):
        q=np.array((2*np.pi,0,0))
        self.assertEqual(self.GFFT(q),0)
        q=np.array((2*np.pi,2*np.pi,0))
        self.assertEqual(self.GFFT(q),0)
        q=np.array((2*np.pi,2*np.pi,2*np.pi))
        self.assertEqual(self.GFFT(q),0)
        
    def testFTfuncValues(self):
        # note: equality here doesn't quite work due to roundoff error at the 15th digit
        q=np.array((1,0,0))
        self.assertTrue(self.GFFT(q)<0)
        self.assertAlmostEqual(self.GFFT(q), 8*(np.cos(1)-1))
        q=np.array((1,1,0))
        self.assertTrue(self.GFFT(q)<0)
        self.assertAlmostEqual(self.GFFT(q), 2*(np.cos(2)-1)+8*(np.cos(1)-1))
        q=np.array((1,1,1))
        self.assertTrue(self.GFFT(q)<0)
        self.assertAlmostEqual(self.GFFT(q), 6*(np.cos(2)-1))

    def testFTfuncSymmetry(self):
        q=np.array((1,0,0))
        q2=np.array((-1,0,0))
        self.assertEqual(self.GFFT(q),self.GFFT(q2))
        q2=np.array((0,1,0))
        self.assertEqual(self.GFFT(q),self.GFFT(q2))
        q2=np.array((0,0,1))
        self.assertEqual(self.GFFT(q),self.GFFT(q2))

    def testFTdim(self):
        self.assertTrue(np.shape(self.GFdiff)==(3,3))

    def testFTDiffValue(self):
        # note: equality here doesn't work here, as we're using finite difference
        # to evaluate a second derivative, so we use a threshold value.
        delta=2.e-4
        eps=1e-5
        qsmall=np.array((delta,0,0))
        D0 = self.GFFT(qsmall)/(delta*delta)
        self.assertTrue(
            abs(np.dot(qsmall,np.dot(self.GFdiff,qsmall))/(delta*delta)-D0) < eps )

        qsmall=np.array((delta,delta,0))
        D0 = self.GFFT(qsmall)/(delta*delta)
        self.assertTrue(
            abs(np.dot(qsmall,np.dot(self.GFdiff,qsmall))/(delta*delta)-D0) < eps )

        qsmall=np.array((delta,delta,delta))
        D0 = self.GFFT(qsmall)/(delta*delta)
        self.assertTrue(
            abs(np.dot(qsmall,np.dot(self.GFdiff,qsmall))/(delta*delta)-D0) < eps )

def main():
    unittest.main()

if __name__ == '__main__':
    main()
