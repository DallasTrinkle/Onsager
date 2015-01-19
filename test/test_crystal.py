"""
Unit tests for crystal class
"""

__author__ = 'Dallas R. Trinkle'

import unittest
import numpy as np
import onsager.crystal as crystal

class UnitCellTests(unittest.TestCase):
    """Tests to make sure incell and halfcell work as expected."""
    def testincell(self):
        """In cell testing"""
        a = np.array([4./3., -2./3.,19./9.])
        b = np.array([1./3., 1./3., 1./9.])
        self.assertTrue(np.all(np.isclose(crystal.incell(a), b)))

    def testhalfcell(self):
        """Half cell testing"""
        a = np.array([4./3., -2./3.,17./9.])
        b = np.array([1./3., 1./3., -1./9.])
        self.assertTrue(np.all(np.isclose(crystal.inhalf(a), b)))


class CrystalClassTests(unittest.TestCase):
    """Tests for the crystal class and symmetry analysis."""

    def setUp(self):
        self.a0 = 2.5
        self.c_a = np.sqrt(8./3.)
        self.sclatt = self.a0*np.eye(3)
        self.fcclatt = self.a0*np.array([[0, 0.5, 0.5],
                                         [0.5, 0, 0.5],
                                         [0.5, 0.5, 0]])
        self.bcclatt = self.a0*np.array([[-0.5, 0.5, 0.5],
                                         [0.5, -0.5, 0.5],
                                         [0.5, 0.5, -0.5]])
        self.hexlatt = self.a0*np.array([[0.5, 0.5, 0],
                                         [-np.sqrt(0.75), np.sqrt(0.75), 0],
                                         [0, 0, self.c_a]])
        self.basis = [np.array([0.,0.,0.])]

    def isscMetric(self, crys, a0=0):
        if a0==0: a0=self.a0
        self.assertAlmostEqual(crys.volume, a0**3)
        for i, a2 in enumerate(crys.metric.flatten()):
            if i%4 == 0:
                # diagonal element
                self.assertAlmostEqual(a2, a0**2)
            else:
                # off-diagonal element
                self.assertAlmostEqual(a2, 0)

    def isfccMetric(self, crys, a0=0):
        if a0==0: a0=self.a0
        self.assertAlmostEqual(crys.volume, 0.25*a0**3)
        for i, a2 in enumerate(crys.metric.flatten()):
            if i%4 == 0:
                # diagonal element
                self.assertAlmostEqual(a2, 0.5*a0**2)
            else:
                # off-diagonal element
                self.assertAlmostEqual(a2, 0.25*a0**2)

    def isbccMetric(self, crys, a0=0):
        if a0==0: a0=self.a0
        self.assertAlmostEqual(crys.volume, 0.5*a0**3)
        for i, a2 in enumerate(crys.metric.flatten()):
            if i%4 == 0:
                # diagonal element
                self.assertAlmostEqual(a2, 0.75*a0**2)
            else:
                # off-diagonal element
                self.assertAlmostEqual(a2, -0.25*a0**2)

    def ishexMetric(self, crys, a0=0, c_a=0):
        if a0==0: a0=self.a0
        if c_a==0: c_a=self.c_a
        self.assertAlmostEqual(crys.volume, np.sqrt(0.75)*c_a*a0**3)
        self.assertAlmostEqual(crys.metric[0,0], a0**2)
        self.assertAlmostEqual(crys.metric[1,1], a0**2)
        self.assertAlmostEqual(crys.metric[0,1], -0.5*a0**2)
        self.assertAlmostEqual(crys.metric[2,2], (c_a*a0)**2)
        self.assertAlmostEqual(crys.metric[0,2], 0)
        self.assertAlmostEqual(crys.metric[1,2], 0)

    def testscMetric(self):
        """Does the simple cubic lattice have the right volume and metric?"""
        crys = crystal.Crystal(self.sclatt, self.basis)
        self.isscMetric(crys)
        self.assertEqual(len(crys.basis), 1)    # one chemistry
        self.assertEqual(len(crys.basis[0]), 1) # one atom in the unit cell

    def testfccMetric(self):
        """Does the face-centered cubic lattice have the right volume and metric?"""
        crys = crystal.Crystal(self.fcclatt, self.basis)
        self.isfccMetric(crys)
        self.assertEqual(len(crys.basis), 1)    # one chemistry
        self.assertEqual(len(crys.basis[0]), 1) # one atom in the unit cell

    def testbccMetric(self):
        """Does the body-centered cubic lattice have the right volume and metric?"""
        crys = crystal.Crystal(self.bcclatt, self.basis)
        self.isbccMetric(crys)
        self.assertEqual(len(crys.basis), 1)    # one chemistry
        self.assertEqual(len(crys.basis[0]), 1) # one atom in the unit cell

    def testscReduce(self):
        """If we start with a supercell, does it get reduced back to our start?"""
        nsuper = np.array([[2,0,0],[0,2,0],[0,0,1]], dtype=int)
        doublebasis = [self.basis[0], np.array([0.5, 0, 0]) + self.basis[0],
                       np.array([0, 0.5, 0]) + self.basis[0], np.array([0.5, 0.5, 0]) + self.basis[0]]
        crys = crystal.Crystal(np.dot(self.sclatt, nsuper), doublebasis)
        self.isscMetric(crys)
        self.assertEqual(len(crys.basis), 1)    # one chemistry
        self.assertEqual(len(crys.basis[0]), 1) # one atom in the unit cell

    def testscReduce2(self):
        """If we start with a supercell, does it get reduced back to our start?"""
        nsuper = np.array([[5,-3,0],[1,-1,3],[-2,1,1]], dtype=int)
        crys = crystal.Crystal(np.dot(self.sclatt, nsuper), self.basis)
        self.isscMetric(crys)
        self.assertEqual(len(crys.basis), 1)    # one chemistry
        self.assertEqual(len(crys.basis[0]), 1) # one atom in the unit cell

    def testbccReduce2(self):
        """If we start with a supercell, does it get reduced back to our start?"""
        basis = [[np.array([0.,0.,0.]), np.array([0.5,0.5,0.5])]]
        crys = crystal.Crystal(self.sclatt, basis)
        self.isbccMetric(crys)
        self.assertEqual(len(crys.basis), 1)    # one chemistry
        self.assertEqual(len(crys.basis[0]), 1) # one atom in the unit cell

    def testscShift(self):
        """If we start with a supercell, does it get reduced back to our start?"""
        nsuper = np.array([[5,-3,0],[1,-1,3],[-2,1,1]], dtype=int)
        basis = [np.array([0.33, -0.25, 0.45])]
        crys = crystal.Crystal(np.dot(self.sclatt, nsuper), basis)
        self.isscMetric(crys)
        self.assertEqual(len(crys.basis), 1)    # one chemistry
        self.assertEqual(len(crys.basis[0]), 1) # one atom in the unit cell
        self.assertTrue(np.all(np.isclose(crys.basis[0][0], np.array([0,0,0]))))

    def testhcp(self):
        """If we start with a supercell, does it get reduced back to our start?"""
        basis = [np.array([0, 0, 0]), np.array([1./3., 2./3., 1./2.])]
        crys = crystal.Crystal(self.hexlatt, basis)
        print crys.lattice
        print crys.basis
        self.ishexMetric(crys)
        self.assertEqual(len(crys.basis), 1)    # one chemistry
        self.assertEqual(len(crys.basis[0]), 2) # two atoms in the unit cell
        # there needs to be [1/3,2/3,1/4] or [1/3,2/3,3/4], and then the opposite

    def testscgroupops(self):
        """Do we have 48 space group operations?"""
        crys = crystal.Crystal(self.sclatt, self.basis)
        self.assertEqual(len(crys.g), 48)

    def testmaptrans(self):
        """Does our map translation operate correctly?"""
        basis = [[np.array([0,0,0])]]
        trans, indexmap = crystal.maptranslation(basis, basis)
        self.assertTrue(np.all(np.isclose(trans, np.array([0,0,0]))))
        self.assertEqual(indexmap, [[0]])

        oldbasis = [[np.array([0.2,0,0])]]
        newbasis = [[np.array([-0.2,0,0])]]
        trans, indexmap = crystal.maptranslation(oldbasis, newbasis)
        self.assertTrue(np.all(np.isclose(trans, np.array([0.4,0,0]))))
        self.assertEqual(indexmap, [[0]])

        oldbasis = [[np.array([0.,0.,0.]), np.array([1./3.,2./3.,1./2.])]]
        newbasis = [[np.array([0.,0.,0.]), np.array([-1./3.,-2./3.,-1./2.])]]
        trans, indexmap = crystal.maptranslation(oldbasis, newbasis)
        self.assertTrue(np.all(np.isclose(trans, np.array([1./3.,-1./3.,1./2.]))))
        self.assertEqual(indexmap, [[1,0]])

        oldbasis = [[np.array([0.,0.,0.])], [np.array([1./3.,2./3.,1./2.]), np.array([2./3.,1./3.,1./2.])]]
        newbasis = [[np.array([0.,0.,0.])], [np.array([2./3.,1./3.,1./2.]), np.array([1./3.,2./3.,1./2.])]]
        trans, indexmap = crystal.maptranslation(oldbasis, newbasis)
        self.assertTrue(np.all(np.isclose(trans, np.array([0.,0.,0.]))))
        self.assertEqual(indexmap, [[0],[1,0]])

        oldbasis = [[np.array([0.,0.,0.]), np.array([1./3.,2./3.,1./2.])]]
        newbasis = [[np.array([0.,0.,0.]), np.array([-1./4.,-1./2.,-1./2.])]]
        trans, indexmap = crystal.maptranslation(oldbasis, newbasis)
        self.assertEqual(indexmap, None)
