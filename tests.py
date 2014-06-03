#!/usr/bin/env python
"""
Unit tests for vacancy mediated diffusion code.
"""

import unittest
import numpy as np
import FCClatt

class LatticeTests(unittest.TestCase):
    """
    Set of tests that our lattice code is behaving correctly
    """
    def setUp(self):
        self.NNvect = FCClatt.NNvect()
        self.invlist = FCClatt.invlist(self.NNvect)

    def testFCCcount(self):
        """
        check that we have z=12 neighbors, and we're in 3D
        """
        self.assertEqual(np.shape(self.NNvect), (12,3))

    def testFCCinversioncount(self):
        """
        Right dimensions for matrices?
        """
        self.assertEqual(np.shape(self.NNvect)[0], np.shape(self.invlist)[0])

    def testFCCbasic(self):
        """
        Do we have the right <110> nearest neighbor vectors for FCC?
        """
        self.assertTrue(any( all((1,1,0)==x) for x in self.NNvect ))
        self.assertTrue(any( all((1,-1,0)==x) for x in self.NNvect ))
        self.assertTrue(any( all((0,1,1)==x) for x in self.NNvect ))
        self.assertTrue(any( all((0,1,-1)==x) for x in self.NNvect ))
        self.assertTrue(any( all((1,0,1)==x) for x in self.NNvect ))
        self.assertTrue(any( all((-1,0,1)==x) for x in self.NNvect ))
        self.assertFalse(any( all((0,0,0)==x) for x in self.NNvect ))
        self.assertFalse(any( all((1,1,1)==x) for x in self.NNvect ))
        
    def testFCCinversion(self):
        """
        Check that for each NN vector, we have its inverse too, from invlist
        """
        for k1, k2 in enumerate(self.invlist):
            self.assertTrue(all(self.NNvect[k1] == -self.NNvect[k2]))

# At some point, we'll probably want to included point group operations, and appropriate tests...

import GFcalc

class GreenFuncDerivativeTests(unittest.TestCase):
    """
    Tests for the construction of D as a fourier transform, and the 2nd and 4th
    derivatives.
    """
    def setUp(self):
        self.NNvect = FCClatt.NNvect()
        self.rates = np.array((1,)*np.shape(self.NNvect)[0])
        self.DFT = GFcalc.DFTfunc(self.NNvect, self.rates) # Fourier transform
        self.D2 = GFcalc.D2(self.NNvect, self.rates) # - 2nd deriv. of FT (>0)
        self.D4 = GFcalc.D4(self.NNvect, self.rates) # + 4th deriv. of FT (>0)

    def testFTisfunc(self):
        """
        Do we get a function as DFT?
        """
        self.assertTrue(callable(self.DFT))

    def testFTfuncZero(self):
        """
        Is the FT zero at gamma?
        """
        q=np.array((0,0,0))
        self.assertEqual(self.DFT(q),0)
        
    def testFTfuncZeroRLV(self):
        """
        Is the FT zero for reciprocal lattice vectors?
        """
        q=np.array((2*np.pi,0,0))
        self.assertEqual(self.DFT(q),0)
        q=np.array((2*np.pi,2*np.pi,0))
        self.assertEqual(self.DFT(q),0)
        q=np.array((2*np.pi,2*np.pi,2*np.pi))
        self.assertEqual(self.DFT(q),0)
        
    def testFTfuncValues(self):
        """
        Do we match some specific values?
        Testing that we're negative, and "by hand" evaluation of a few cases.
        Note: equality here doesn't quite work due to roundoff error at the 15th digit
        """
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
        """
        Does our FT obey basic cubic symmetry operations?
        """
        q=np.array((1,0,0))
        q2=np.array((-1,0,0))
        self.assertEqual(self.DFT(q),self.DFT(q2))
        q2=np.array((0,1,0))
        self.assertEqual(self.DFT(q),self.DFT(q2))
        q2=np.array((0,0,1))
        self.assertEqual(self.DFT(q),self.DFT(q2))

    def testFTdim(self):
        """
        Do we have the correct dimensionality for our second and fourth derivatives?
        """
        self.assertTrue(np.shape(self.D2)==(3,3))
        self.assertTrue(np.shape(self.D4)==(3,3,3,3))

    def testFTDiffSymmetry(self):
        """
        Do we obey basic symmetry for these values? That means that D2 should be
        symmetric, and that any permutation of [abcd] should give the same value
        in D4.
        """
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

    def testEval2(self):
        """
        Tests eval2(q,D) gives qDq
        """
        qvec = np.array((0.5, 0.75, -0.25))
        self.assertAlmostEqual(np.dot(qvec, np.dot(qvec, self.D2)),
                               GFcalc.eval2(qvec, self.D2))

    def testEval4(self):
        """
        Tests eval4(q,D) gives qqDqq
        """
        qvec = np.array((0.5, 0.75, -0.25))
        self.assertAlmostEqual(np.dot(qvec, np.dot(qvec, np.dot(qvec, np.dot(qvec, self.D4)))),
                               GFcalc.eval4(qvec, self.D4))

    def testFTDiffValue(self):
        """
        Test out that the 2nd derivatives behave as expected, by doing a finite
        difference evaluation. Requires using a threshold value.
        """
        # Remember: D2 is negative of the second derivative (to make it positive def.)
        delta=2.e-4
        eps=1e-5
        qsmall=np.array((delta,0,0))
        D0 = self.DFT(qsmall)
        D2 = GFcalc.eval2(qsmall,self.D2)
        self.assertTrue(abs(D0+D2) < eps*(delta**2) )
        self.assertFalse(abs(D0) < eps*(delta**2) )

        qsmall=np.array((delta,delta,0))
        D0 = self.DFT(qsmall)
        D2 = GFcalc.eval2(qsmall,self.D2)
        self.assertTrue(abs(D0+D2) < eps*(delta**2) )
        self.assertFalse(abs(D0) < eps*(delta**2) )

        qsmall=np.array((delta,delta,delta))
        D0 = self.DFT(qsmall)
        D2 = GFcalc.eval2(qsmall,self.D2)
        self.assertTrue(abs(D0+D2) < eps*(delta**2) )
        self.assertFalse(abs(D0) < eps*(delta**2) )

    def testFTDiff4Value(self):
        """
        Test out that the 4th derivatives behave as expected, by doing a finite
        difference evaluation. Requires using a threshold value.
        """
        # Remember: D2 is negative of the second derivative (to make it positive def.)
        delta=1e-1
        eps=1e-1
        qsmall=np.array((delta,0,0))
        D = self.DFT(qsmall)
        D2 = GFcalc.eval2(qsmall,self.D2)
        D4 = GFcalc.eval4(qsmall,self.D4)
        self.assertTrue(abs(D+D2-D4) < eps*(delta**4) )
        self.assertFalse(abs(D+D2) < eps*(delta**4) )

        qsmall=np.array((delta,delta,0))
        D = self.DFT(qsmall)
        D2 = GFcalc.eval2(qsmall,self.D2)
        D4 = GFcalc.eval4(qsmall,self.D4)
        self.assertTrue(abs(D+D2-D4) < eps*(delta**4) )
        self.assertFalse(abs(D+D2) < eps*(delta**4) )

        qsmall=np.array((delta,delta,delta))
        D = self.DFT(qsmall)
        D2 = GFcalc.eval2(qsmall,self.D2)
        D4 = GFcalc.eval4(qsmall,self.D4)
        self.assertTrue(abs(D+D2-D4) < eps*(delta**4) )
        self.assertFalse(abs(D+D2) < eps*(delta**4) )

# code that does Fourier transforms
class GreenFuncFourierTransformPoleTests(unittest.TestCase):
    """
    Tests for code involved in the Fourier transform of the second-order pole.
    """
    def setUp(self):
        # di0/ei0 are the "original" eigenvalues / eigenvectors, and di/ei are the
        # calculated versions
        self.di0 = np.array([0.5, 1., 2.])
        self.ei0 = np.array([[np.sqrt(0.5), np.sqrt(0.5),0],
                             [np.sqrt(1./6.),-np.sqrt(1./6.),np.sqrt(2./3.)],
                             [np.sqrt(1./3.),-np.sqrt(1./3.),-np.sqrt(1./3.)]])
        self.D2 = np.dot(self.ei0.T, np.dot(np.diag(self.di0), self.ei0))
        self.GF2_0 = np.dot(self.ei0.T, np.dot(np.diag(1./self.di0), self.ei0))
        self.di, self.ei_vect = GFcalc.calcDE(self.D2)
        self.GF2 = GFcalc.invertD2(self.D2)

    def testEigendim(self):
        """
        Correct dimensionality of eigenvalues and vectors?
        """
        self.assertTrue(np.shape(self.di)==(3,))
        self.assertTrue(np.shape(self.ei_vect)==(3,3))

    def testEigenvalueVect(self):
        """
        Test that the eigenvalues and vectors by direct comparison with thresholds.
        """
        # a little painful, due to thresholds (and possible negative eigenvectors)
        eps=1e-8
        for eig in self.di0: self.assertTrue(any(abs(self.di-eig)<eps) )
        for vec in self.ei0: self.assertTrue(any(abs(np.dot(x,vec))>(1-eps) for x in self.ei_vect))

    def testInverse(self):
        """
        Check the evaluation of the inverse.
        """
        for a in xrange(3):
            for b in xrange(3):
                self.assertAlmostEqual(self.GF2_0[a,b], self.GF2[a,b])

    def testCalcUnorm(self):
        """
        Test the normalized u vector and magnitude; ui = (x.ei)/sqrt(di), including
        the handling of x=0.
        """
        # Graceful handling of 0?
        x = np.zeros(3)
        ui, umagn = GFcalc.unorm(self.di, self.ei_vect, x)
        self.assertEqual(umagn, 0)
        self.assertTrue(all(ui == 0))

        # "arbitrary" vector
        x = np.array([0.5, 0.25, -1])
        ui, umagn = GFcalc.unorm(self.di, self.ei_vect, x)
        self.assertAlmostEqual(np.dot(ui,ui), 1)
        self.assertAlmostEqual(umagn, np.sqrt(GFcalc.eval2(x, self.GF2)))
        for a in xrange(3):
            self.assertAlmostEqual(ui[a]*umagn,
                                   np.dot(x, self.ei_vect[a,:])/np.sqrt(self.di[a]))

    def testCalcPnorm(self):
        """
        Test the normalized p vector and magnitude; pi = (q.ei)*sqrt(di), including
        the handling of q=0.
        """
        # Graceful handling of 0?
        q = np.zeros(3)
        pi, pmagn = GFcalc.pnorm(self.di, self.ei_vect, q)
        self.assertEqual(pmagn, 0)
        self.assertTrue(all(pi == 0))

        # "arbitrary" vector
        q = np.array([0.5, 0.25, -1])
        pi, pmagn = GFcalc.pnorm(self.di, self.ei_vect, q)
        self.assertAlmostEqual(np.dot(pi,pi), 1)
        self.assertAlmostEqual(pmagn, np.sqrt(GFcalc.eval2(q, self.D2)))
        for a in xrange(3):
            self.assertAlmostEqual(pi[a]*pmagn,
                                   np.dot(q, self.ei_vect[a,:])*np.sqrt(self.di[a]))

    def testPoleFT(self):
        """
        Test the evaluation of the fourier transform of the second-order pole,
        including at 0.
        """
        # Graceful handling of 0?
        pm = 0.5 # arbitrary at this point...
        x = np.zeros(3)
        ui, umagn = GFcalc.unorm(self.di, self.ei_vect, x)
        g = GFcalc.poleFT(self.di,umagn, pm)
        # pm*sqrt(d1*d2*d3)/4 pi^(3/2)
        self.assertAlmostEqual(pm*0.25/np.sqrt(np.product(self.di*np.pi)), g)

        x=np.array((0.25, 0.5, 1))
        ui, umagn = GFcalc.unorm(self.di, self.ei_vect, x)

        erfupm = 0.125 # just to use the "cached" version
        self.assertNotEqual(GFcalc.poleFT(self.di,umagn, pm),
                            GFcalc.poleFT(self.di,umagn, pm, erfupm))
        
        g = GFcalc.poleFT(self.di, umagn, pm, erfupm)
        self.assertAlmostEqual(erfupm*0.25/(umagn*np.pi*np.sqrt(np.product(self.di))), g)

class GreenFuncFourierTransformDiscTests(unittest.TestCase):
    """
    Tests for the fourier transform of the discontinuity correction (4th derivative).
    """
    def setUp(self):
        # GFcalc.ConstructExpToIndex()
        pass

    def testPowerExpansion(self):
        """
        Check that there are (a) 15 entries, (b) all non-negative, (c) summing to 4,
        (d) uniquely in our power expansion.
        """
        self.assertEqual(np.shape(GFcalc.PowerExpansion),(15,3))
        self.assertTrue(np.all(GFcalc.PowerExpansion>=0))
        for i in xrange(15):
            self.assertEqual(GFcalc.PowerExpansion[i].sum(), 4)
            for j in xrange(i):
                self.assertFalse(all(GFcalc.PowerExpansion[i]==GFcalc.PowerExpansion[j]))

    def testExpToIndex(self):
        """
        Checks that ExpToIndex is correctly constructed.
        """
        for n1 in xrange(5):
            for n2 in xrange(5):
                for n3 in xrange(5):
                    if (n1+n2+n3 != 4):
                        self.assertEqual(GFcalc.ExpToIndex[n1,n2,n3], 15)
                    else:
                        ind = GFcalc.ExpToIndex[n1,n2,n3]
                        self.assertNotEqual(ind, 15)
                        self.assertTrue(all(GFcalc.PowerExpansion[ind]==(n1,n2,n3)))

    def testConvD4toNNN(self):
        """
        Tests conversion of the 4th-rank 4th derivative into power expansion.
        """
        D4=np.zeros((3,3,3,3))
        D4[0,0,0,0]=1
        D15=GFcalc.D4toNNN(D4)
        self.assertEqual(np.shape(D15), (15,))
        self.assertEqual(D15[GFcalc.ExpToIndex[4,0,0]], 1)
        for ind in xrange(15):
            if ind != GFcalc.ExpToIndex[4,0,0]:
                self.assertEqual(D15[ind], 0)
                
        D4=np.zeros((3,3,3,3))
        D4[1,1,1,1]=1
        D15=GFcalc.D4toNNN(D4)
        self.assertEqual(np.shape(D15), (15,))
        self.assertEqual(D15[GFcalc.ExpToIndex[0,4,0]], 1)
        for ind in xrange(15):
            if ind != GFcalc.ExpToIndex[0,4,0]:
                self.assertEqual(D15[ind], 0)

        D4=np.zeros((3,3,3,3))
        D4[0,0,0,1]=1
        D4[0,0,1,0]=1
        D4[0,1,0,0]=1
        D4[1,0,0,0]=1
        D15=GFcalc.D4toNNN(D4)
        self.assertEqual(np.shape(D15), (15,))
        self.assertEqual(D15[GFcalc.ExpToIndex[3,1,0]], 4)
        for ind in xrange(15):
            if ind != GFcalc.ExpToIndex[3,1,0]:
                self.assertEqual(D15[ind], 0)

    def testRotateD4(self):
        """
        Tests the rotation of D4 with the eigenvalues/vectors of D
        """
        di = np.array([0.5, 1., 2.])
        ei = np.array([[np.sqrt(0.5), np.sqrt(0.5),0],
                       [np.sqrt(1./6.),-np.sqrt(1./6.),np.sqrt(2./3.)],
                       [np.sqrt(1./3.),-np.sqrt(1./3.),-np.sqrt(1./3.)]])
        D4=np.zeros((3,3,3,3))
        D4[0,0,0,0]=1
        Drot4 = GFcalc.RotateD4(D4, di, ei)
        self.assertEqual(np.shape(Drot4),(3,3,3,3))
        for a in xrange(3):
            self.assertAlmostEqual(Drot4[a,a,a,a],
                                   GFcalc.eval4(ei[a]/np.sqrt(di[a]), D4))
        q = np.array([1,-0.5, -0.25])
        pi, pnorm = GFcalc.pnorm(di, ei, q)
        self.assertAlmostEqual(GFcalc.eval4(pi, Drot4)*(pnorm**4),
                               GFcalc.eval4(q, D4))
        for a in xrange(3):
            for b in xrange(3):
                for c in xrange(3):
                    for d in xrange(3):
                        self.assertAlmostEqual(Drot4[a,b,c,d],
                                               np.sqrt(di[a]*di[b]*di[c]*di[d])*
                                               np.dot(ei[a],
                                                      np.dot(ei[b],
                                                             np.dot(ei[c],
                                                                    np.dot(ei[d],
                                                                           D4)))))

def main():
    unittest.main()

if __name__ == '__main__':
    main()
