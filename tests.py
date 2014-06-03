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

class GreenFuncDerivativeTests(unittest.TestCase):
    def setUp(self):
        self.NNvect = FCClatt.NNvect()
        self.rates = np.array((1,)*np.shape(self.NNvect)[0])
        self.DFT = GFcalc.DFTfunc(self.NNvect, self.rates) # Fourier transform
        self.D2 = GFcalc.D2(self.NNvect, self.rates) # - 2nd deriv. of FT (>0)
        self.D4 = GFcalc.D4(self.NNvect, self.rates) # + 4th deriv. of FT (>0)

    def testFTisfunc(self):
        self.assertTrue(callable(self.DFT))

    def testFTfuncZero(self):
        q=np.array((0,0,0))
        self.assertEqual(self.DFT(q),0)
        
    def testFTfuncZeroBZ(self):
        q=np.array((2*np.pi,0,0))
        self.assertEqual(self.DFT(q),0)
        q=np.array((2*np.pi,2*np.pi,0))
        self.assertEqual(self.DFT(q),0)
        q=np.array((2*np.pi,2*np.pi,2*np.pi))
        self.assertEqual(self.DFT(q),0)
        
    def testFTfuncValues(self):
        # note: equality here doesn't quite work due to roundoff error at the 15th digit
        q=np.array((1,0,0))
        self.assertTrue(self.DFT(q)<0) # negative everywhere...
        self.assertAlmostEqual(self.DFT(q), 8*(np.cos(1)-1))
        q=np.array((1,1,0))
        self.assertTrue(self.DFT(q)<0)
        self.assertAlmostEqual(self.DFT(q), 2*(np.cos(2)-1)+8*(np.cos(1)-1))
        q=np.array((1,1,1))
        self.assertTrue(self.DFT(q)<0)
        self.assertAlmostEqual(self.DFT(q), 6*(np.cos(2)-1))

    def testFTfuncSymmetry(self):
        q=np.array((1,0,0))
        q2=np.array((-1,0,0))
        self.assertEqual(self.DFT(q),self.DFT(q2))
        q2=np.array((0,1,0))
        self.assertEqual(self.DFT(q),self.DFT(q2))
        q2=np.array((0,0,1))
        self.assertEqual(self.DFT(q),self.DFT(q2))

    def testFTdim(self):
        self.assertTrue(np.shape(self.D2)==(3,3))
        self.assertTrue(np.shape(self.D4)==(3,3,3,3))

    def testFTDiffSymmetry(self):
        self.assertTrue(np.all(self.D2 == self.D2.T))
        self.assertEqual(self.D2[0,0], self.D2[1,1])
        self.assertEqual(self.D2[0,0], self.D2[2,2])
        for a in xrange(3):
            for b in xrange(3):
                for c in xrange(3):
                    for d in xrange(3):
                        ind=(a,b,c,d)
                        inds = tuple(sorted(ind))
                        self.assertEqual(self.D4[ind], self.D4[inds])

    def testFTDiffValue(self):
        # note: equality here doesn't work here, as we're using finite difference
        # to evaluate a second derivative, so we use a threshold value.
        # Remember: D2 is negative of the second derivative (to make it positive def.)
        delta=2.e-4
        eps=1e-5
        qsmall=np.array((delta,0,0))
        D0 = self.DFT(qsmall)
        D2 = np.dot(qsmall,np.dot(qsmall,self.D2))
        self.assertTrue(abs(D0+D2) < eps*(delta**2) )
        self.assertFalse(abs(D0) < eps*(delta**2) )

        qsmall=np.array((delta,delta,0))
        D0 = self.DFT(qsmall)
        D2 = np.dot(qsmall,np.dot(qsmall,self.D2))
        self.assertTrue(abs(D0+D2) < eps*(delta**2) )
        self.assertFalse(abs(D0) < eps*(delta**2) )

        qsmall=np.array((delta,delta,delta))
        D0 = self.DFT(qsmall)
        D2 = np.dot(qsmall,np.dot(qsmall,self.D2))
        self.assertTrue(abs(D0+D2) < eps*(delta**2) )
        self.assertFalse(abs(D0) < eps*(delta**2) )

    def testFTDiff4Value(self):
        # note: equality here doesn't work here, as we're using finite difference
        # to evaluate a second derivative, so we use a threshold value.
        # Remember: D2 is negative of the second derivative (to make it positive def.)
        delta=1e-1
        eps=1e-1
        qsmall=np.array((delta,0,0))
        D = self.DFT(qsmall)
        D2 = np.dot(qsmall,np.dot(qsmall,self.D2))
        D4 = np.dot(qsmall,np.dot(qsmall,np.dot(qsmall,np.dot(qsmall,self.D4))))
        self.assertTrue(abs(D+D2-D4) < eps*(delta**4) )
        self.assertFalse(abs(D+D2) < eps*(delta**4) )

        qsmall=np.array((delta,delta,0))
        D = self.DFT(qsmall)
        D2 = np.dot(qsmall,np.dot(qsmall,self.D2))
        D4 = np.dot(qsmall,np.dot(qsmall,np.dot(qsmall,np.dot(qsmall,self.D4))))
        self.assertTrue(abs(D+D2-D4) < eps*(delta**4) )
        self.assertFalse(abs(D+D2) < eps*(delta**4) )

        qsmall=np.array((delta,delta,delta))
        D = self.DFT(qsmall)
        D2 = np.dot(qsmall,np.dot(qsmall,self.D2))
        D4 = np.dot(qsmall,np.dot(qsmall,np.dot(qsmall,np.dot(qsmall,self.D4))))
        self.assertTrue(abs(D+D2-D4) < eps*(delta**4) )
        self.assertFalse(abs(D+D2) < eps*(delta**4) )

# code that does Fourier transforms
class GreenFuncFourierTransformTests(unittest.TestCase):
    def setUp(self):
        self.di0 = np.array([0.5, 1., 2.])
        self.ei0 = np.array([[np.sqrt(0.5), np.sqrt(0.5),0],
                             [np.sqrt(1./6.),-np.sqrt(1./6.),np.sqrt(2./3.)],
                             [np.sqrt(1./3.),-np.sqrt(1./3.),-np.sqrt(1./3.)]])
        self.D2 = np.dot(self.ei0.T, np.dot(np.diag(self.di0), self.ei0))
        self.GF2_0 = np.dot(self.ei0.T, np.dot(np.diag(1./self.di0), self.ei0))
        self.di, self.ei_vect = GFcalc.calcDE(self.D2)
        self.GF2 = GFcalc.invertD2(self.D2)

    def testEigendim(self):
        self.assertTrue(np.shape(self.di)==(3,))
        self.assertTrue(np.shape(self.ei_vect)==(3,3))

    def testEigenvalueVect(self):
        # a little painful, due to thresholds (and possible negative eigenvectors)
        eps=1e-8
        for eig in self.di0: self.assertTrue(any(abs(self.di-eig)<eps) )
        for vec in self.ei0: self.assertTrue(any(abs(np.dot(x,vec))>(1-eps) for x in self.ei_vect))

    def testInverse(self):
        for a in xrange(3):
            for b in xrange(3):
                self.assertAlmostEqual(self.GF2_0[a,b], self.GF2[a,b])

# test spherical harmonics code
import scipy
import SphereHarm
        
class SphereHarmTests(unittest.TestCase):
    def setUp(self):
        self.D2iso = np.eye(3)
        self.Npolar=4
        self.Ntrunc=(self.Npolar*(self.Npolar+1))/2

    def testCarttoSphere(self):
        qv = np.array([1,0,0])
        qsphere=SphereHarm.CarttoSphere(qv)
        self.assertEqual(qsphere[0], 0) # theta (azimuthal)
        self.assertEqual(qsphere[1], np.pi*0.5) # phi (polar)
        self.assertEqual(qsphere[2], 1) # magnitude
        
        qv = np.array([0,0,0])
        qv2 = SphereHarm.SpheretoCart(SphereHarm.CarttoSphere(qv))
        for d in xrange(3): self.assertAlmostEqual(qv[d], qv2[d])

        qv = np.array([1,0,0])
        qv2 = SphereHarm.SpheretoCart(SphereHarm.CarttoSphere(qv))
        for d in xrange(3): self.assertAlmostEqual(qv[d], qv2[d])

        qv = np.array([1,1,0])
        qv2 = SphereHarm.SpheretoCart(SphereHarm.CarttoSphere(qv))
        for d in xrange(3): self.assertAlmostEqual(qv[d], qv2[d])

        qv = np.array([-1,1,-1])
        qv2 = SphereHarm.SpheretoCart(SphereHarm.CarttoSphere(qv))
        for d in xrange(3): self.assertAlmostEqual(qv[d], qv2[d])

    def testYlmTransformDim(self):
        Dcoeff,indices=SphereHarm.YlmTransform(self.D2iso, Npolar=self.Npolar)
        self.assertEqual(np.shape(Dcoeff)[0], self.Ntrunc)
        self.assertEqual(np.shape(indices[0])[0], self.Ntrunc)
        self.assertEqual(np.shape(indices[1])[0], self.Ntrunc)

    def testYlmTransformValuesIsotropic(self):
        threshold=1e-7
        Dcoeff,indices=SphereHarm.YlmTransform(self.D2iso, Npolar=self.Npolar)
        # isotropic; only the l=0,m=0 value should be nonzero
        for i, m in enumerate(indices[0]):
            l = indices[1][i]
            if (l==0): self.assertAlmostEqual(Dcoeff[i], np.sqrt(2))
            else: self.assertTrue(abs(Dcoeff[i])<threshold)

def main():
    unittest.main()

if __name__ == '__main__':
    main()
