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
        a = np.array([4. / 3., -2. / 3., 19. / 9.])
        b = np.array([1. / 3., 1. / 3., 1. / 9.])
        self.assertTrue(np.allclose(crystal.incell(a), b))

    def testhalfcell(self):
        """Half cell testing"""
        a = np.array([4. / 3., -2. / 3., 17. / 9.])
        b = np.array([1. / 3., 1. / 3., -1. / 9.])
        self.assertTrue(np.allclose(crystal.inhalf(a), b))


class GroupOperationTests(unittest.TestCase):
    """Tests for our group operations."""

    def setUp(self):
        self.rot = np.array([[0, 1, 0],
                             [1, 0, 0],
                             [0, 0, 1]])
        self.trans = np.zeros(3)
        self.cartrot = np.array([[0., 1., 0.],
                                 [1., 0., 0.],
                                 [0., 0., 1.]])
        self.indexmap = ((0,),)
        self.mirrorop = crystal.GroupOp(self.rot, self.trans, self.cartrot, self.indexmap)
        self.ident = crystal.GroupOp(np.eye(3, dtype=int), np.zeros(3), np.eye(3), ((0,),))

    def testEquality(self):
        """Can we check if two group operations are equal?"""
        self.assertNotEqual(self.mirrorop, self.rot)
        self.assertEqual(self.mirrorop.incell(), self.mirrorop)
        # self.assertEqual(self.mirrorop.__hash__(), (self.mirrorop + np.array([1,0,0])).__hash__())

    def testAddition(self):
        """Can we add a vector to our group operation and get a new one?"""
        with self.assertRaises(TypeError):
            self.mirrorop + 0
        v1 = np.array([1, 0, 0])
        newop = self.mirrorop + v1
        mirroroptrans = crystal.GroupOp(self.rot, self.trans + v1, self.cartrot, self.indexmap)
        self.assertEqual(newop, mirroroptrans)
        self.assertTrue(np.allclose((self.ident - v1).trans, -v1))

    def testMultiplication(self):
        """Does group operation multiplication work correctly?"""
        self.assertEqual(self.mirrorop * self.mirrorop, self.ident)
        v1 = np.array([1, 0, 0])
        trans = self.ident + v1
        self.assertEqual(trans * trans, self.ident + 2 * v1)
        rot3 = crystal.GroupOp(np.eye(3, dtype=int), np.zeros(3), np.eye(3), ((1, 2, 0),))
        ident3 = crystal.GroupOp(np.eye(3, dtype=int), np.zeros(3), np.eye(3), ((0, 1, 2),))
        self.assertEqual(rot3 * rot3 * rot3, ident3)

    def testInversion(self):
        """Is the product with the inverse equal to identity?"""
        self.assertEqual(self.ident.inv, self.ident.inv)
        self.assertEqual(self.mirrorop * (self.mirrorop.inv()), self.ident)
        v1 = np.array([1, 0, 0])
        trans = self.ident + v1
        self.assertEqual(trans.inv(), self.ident - v1)
        inversion = crystal.GroupOp(-np.eye(3, dtype=int), np.zeros(3), -np.eye(3), ((0,),))
        self.assertEqual(inversion.inv(), inversion)
        invtrans = inversion + v1
        self.assertEqual(invtrans.inv(), invtrans)

    def testHash(self):
        """Can we construct a frozenset? --requires __hash__"""
        fr = frozenset([self.ident, self.mirrorop])
        self.assertTrue(len(fr), 2)

    def testGroupAnalysis(self):
        """If we determine the eigenvalues / vectors of a group operation, are they what we expect?"""
        # This is entirely dictated by the cartrot part of a GroupOp, so we will only look at that
        # identity
        # rotation type: 1 = identity; 2..6 : 2- .. 6- fold rotation; negation includes a
        # perpendicular mirror
        # therefore: a single mirror is -1, and inversion is -2 (since 2-fold rotation + mirror = i)
        rot = np.eye(3, dtype=int)
        cartrot = np.eye(3)
        rottype, eigenvect = (crystal.GroupOp(rot, self.trans, cartrot, self.indexmap)).eigen()
        self.assertTrue(np.isclose(np.linalg.det(eigenvect), 1))
        self.assertEqual(rottype, 1)  # should be the identity
        self.assertTrue(np.allclose(eigenvect, np.eye(3)))
        basis = crystal.VectorBasis(rottype, eigenvect)
        self.assertEqual(basis[0], 3)  # should be a sphere
        tensorbasis = crystal.SymmTensorBasis(rottype, eigenvect)  # at some point in the future, generalize
        self.assertEqual(len(tensorbasis), 6)  # should be 6 unique symmetric tensors
        for t in tensorbasis:
            self.assertTrue(np.all(t == t.T), msg="{} is not symmetric".format(t))
            for t2 in tensorbasis:
                if np.any(t2 != t):
                    self.assertAlmostEqual(np.dot(t.flatten(), t2.flatten()), 0)

        # inversion
        rot = -np.eye(3, dtype=int)
        cartrot = -np.eye(3)
        rottype, eigenvect = (crystal.GroupOp(rot, self.trans, cartrot, self.indexmap)).eigen()
        self.assertTrue(np.isclose(np.linalg.det(eigenvect), 1))
        self.assertEqual(rottype, -2)  # should be the identity
        self.assertTrue(np.allclose(eigenvect, np.eye(3)))
        basis = crystal.VectorBasis(rottype, eigenvect)
        self.assertEqual(basis[0], 0)  # should be a point
        tensorbasis = crystal.SymmTensorBasis(rottype, eigenvect)  # at some point in the future, generalize
        self.assertEqual(len(tensorbasis), 6)  # should be 6 unique symmetric tensors
        for t in tensorbasis:
            self.assertTrue(np.all(t == t.T), msg="{} is not symmetric".format(t))
            self.assertAlmostEqual(np.dot(t.flatten(), t.flatten()), 1)
            for t2 in tensorbasis:
                if np.any(t2 != t):
                    self.assertAlmostEqual(np.dot(t.flatten(), t2.flatten()), 0)

        # mirror through the y=x line: (x,y) -> (y,x)
        rot = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        cartrot = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
        rottype, eigenvect = (crystal.GroupOp(rot, self.trans, cartrot, self.indexmap)).eigen()
        self.assertTrue(np.isclose(np.linalg.det(eigenvect), 1))
        self.assertEqual(rottype, -1)
        self.assertTrue(np.isclose(abs(np.dot(eigenvect[0],
                                              np.array([1 / np.sqrt(2), -1 / np.sqrt(2), 0]))), 1))
        self.assertTrue(np.allclose(-eigenvect[0], np.dot(rot, eigenvect[0])))  # inverts
        self.assertTrue(np.allclose(eigenvect[1], np.dot(rot, eigenvect[1])))  # leaves unchanged
        self.assertTrue(np.allclose(eigenvect[2], np.dot(rot, eigenvect[2])))  # leaves unchanged
        basis = crystal.VectorBasis(rottype, eigenvect)
        self.assertEqual(basis[0], 2)  # should be a plane
        self.assertTrue(np.allclose(basis[1], eigenvect[0]))
        tensorbasis = crystal.SymmTensorBasis(rottype, eigenvect)  # at some point in the future, generalize
        self.assertEqual(len(tensorbasis), 4)  # should be 4 unique symmetric tensors
        for t in tensorbasis:
            # check symmetry, and remaining unchanged with operations
            self.assertTrue(np.all(t == t.T), msg="{} is not symmetric".format(t))
            rott = np.dot(rot, np.dot(t, rot.T))
            self.assertTrue(np.allclose(t, rott),
                            msg="\n{}\nis not unchanged with\n{}\n{}".format(t, rot, rott))
            self.assertAlmostEqual(np.dot(t.flatten(), t.flatten()), 1)
            for t2 in tensorbasis:
                if np.any(t2 != t):
                    self.assertAlmostEqual(np.dot(t.flatten(), t2.flatten()), 0)

        # three-fold rotation around the body-center
        rot = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        cartrot = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
        rottype, eigenvect = (crystal.GroupOp(rot, self.trans, cartrot, self.indexmap)).eigen()
        self.assertEqual(rottype, 3)
        self.assertTrue(np.isclose(np.linalg.det(eigenvect), 1))
        self.assertTrue(np.isclose(abs(np.dot(eigenvect[0],
                                              np.array([1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)]))), 1))
        self.assertTrue(np.allclose(eigenvect[0], np.dot(rot, eigenvect[0])))  # our rotation axis
        basis = crystal.VectorBasis(rottype, eigenvect)
        self.assertEqual(basis[0], 1)  # should be a line
        self.assertTrue(np.allclose(basis[1], eigenvect[0]))
        tensorbasis = crystal.SymmTensorBasis(rottype, eigenvect)  # at some point in the future, generalize
        self.assertEqual(len(tensorbasis), 2)  # should be 2 unique symmetric tensors
        for t in tensorbasis:
            # check symmetry, and remaining unchanged with operations
            self.assertTrue(np.all(t == t.T), msg="{} is not symmetric".format(t))
            rott = np.dot(rot, np.dot(t, rot.T))
            self.assertTrue(np.allclose(t, rott),
                            msg="\n{}\nis not unchanged with\n{}\n{}".format(t, rot, rott))
            self.assertAlmostEqual(np.dot(t.flatten(), t.flatten()), 1)
            for t2 in tensorbasis:
                if np.any(t2 != t):
                    self.assertAlmostEqual(np.dot(t.flatten(), t2.flatten()), 0)

    def testCombineVectorBasis(self):
        """Test our ability to combine a few vector basis choices"""
        # these are all (d, vect) tuples that we work with
        sphere = (3, np.zeros(3))
        point = (0, np.zeros(3))
        plane1 = (2, np.array([0., 0., 1.]))
        plane2 = (2, np.array([1., 1., 1.]) / np.sqrt(3))
        line1 = (1, np.array([1., 0., 0.]))
        line2 = (1, np.array([0., 1., 0.]))
        line3 = (1, np.array([1., -1., 0.]) / np.sqrt(2))

        for t in [sphere, point, plane1, plane2, line1, line2, line3]:
            self.assertEqual(crystal.CombineVectorBasis(t, t)[0], t[0])
        res = crystal.CombineVectorBasis(line1, plane1)
        self.assertEqual(res[0], 1)  # should be a line
        self.assertTrue(np.isclose(abs(np.dot(res[1], line1[1])), 1))
        res = crystal.CombineVectorBasis(plane1, plane2)
        self.assertEqual(res[0], 1)  # should be a line
        self.assertTrue(np.isclose(abs(np.dot(res[1], line3[1])), 1))
        res = crystal.CombineVectorBasis(plane1, line1)
        self.assertEqual(res[0], 1)  # should be a line
        self.assertTrue(np.isclose(abs(np.dot(res[1], line1[1])), 1))
        res = crystal.CombineVectorBasis(plane2, line1)
        self.assertEqual(res[0], 0)  # should be a point
        res = crystal.CombineVectorBasis(line1, line2)
        self.assertEqual(res[0], 0)  # should be a point

    def testCombineTensorBasis(self):
        """Test the intersection of tensor bases"""
        fullbasis = crystal.SymmTensorBasis(1, np.eye(3))  # full basis (identity)
        yzbasis = crystal.SymmTensorBasis(-1, np.eye(3))  # mirror through the x axis
        xzbasis = crystal.SymmTensorBasis(-1, [np.array([0., 1., 0.]), np.array([0., 0., 1.]), np.array([1., 0., 0.])])
        rotbasis = crystal.SymmTensorBasis(3, np.eye(3))  # 120 deg rot through the x axis
        rotbasis2 = crystal.SymmTensorBasis(3, [np.array([0., 0., 1.]), np.array([1., 0., 0.]), np.array([0., 1., 0.])])
        for b in [fullbasis, yzbasis, xzbasis, rotbasis, rotbasis2]:
            combbasis = crystal.CombineTensorBasis(fullbasis, b)
            self.assertEqual(len(b), len(combbasis))
            combbasis = crystal.CombineTensorBasis(b, fullbasis)
            self.assertEqual(len(b), len(combbasis))
        combbasis = crystal.CombineTensorBasis(yzbasis, rotbasis)
        self.assertEqual(len(combbasis), len(crystal.CombineTensorBasis(rotbasis, yzbasis)))
        self.assertEqual(len(combbasis), len(rotbasis))  # should be two left here
        combbasis = crystal.CombineTensorBasis(rotbasis, rotbasis2)
        self.assertEqual(len(combbasis), 1)  # if there's only one, it has to be 1/sqrt(3).
        self.assertAlmostEqual(1, abs(np.dot(combbasis[0].flatten(), np.eye(3).flatten() / np.sqrt(3))))
        combbasis = crystal.CombineTensorBasis(yzbasis, xzbasis)
        self.assertEqual(len(combbasis), 3)


class CrystalClassTests(unittest.TestCase):
    """Tests for the crystal class and symmetry analysis."""

    def setUp(self):
        self.a0 = 2.5
        self.c_a = np.sqrt(8. / 3.)
        self.sclatt = self.a0 * np.eye(3)
        self.fcclatt = self.a0 * np.array([[0, 0.5, 0.5],
                                           [0.5, 0, 0.5],
                                           [0.5, 0.5, 0]])
        self.bcclatt = self.a0 * np.array([[-0.5, 0.5, 0.5],
                                           [0.5, -0.5, 0.5],
                                           [0.5, 0.5, -0.5]])
        self.hexlatt = self.a0 * np.array([[0.5, 0.5, 0],
                                           [-np.sqrt(0.75), np.sqrt(0.75), 0],
                                           [0, 0, self.c_a]])
        self.basis = [np.array([0., 0., 0.])]
        self.squarelatt = self.a0 * np.eye(2)  # two-dimensional crystal
        self.basis2d = [np.zeros(2)]

    def isscMetric(self, crys, a0=0):
        if a0 == 0: a0 = self.a0
        self.assertAlmostEqual(crys.volume, a0 ** 3)
        for i, a2 in enumerate(crys.metric.flatten()):
            if i % 4 == 0:
                # diagonal element
                self.assertAlmostEqual(a2, a0 ** 2)
            else:
                # off-diagonal element
                self.assertAlmostEqual(a2, 0)

    def isfccMetric(self, crys, a0=0):
        if a0 == 0: a0 = self.a0
        self.assertAlmostEqual(crys.volume, 0.25 * a0 ** 3)
        for i, a2 in enumerate(crys.metric.flatten()):
            if i % 4 == 0:
                # diagonal element
                self.assertAlmostEqual(a2, 0.5 * a0 ** 2)
            else:
                # off-diagonal element
                self.assertAlmostEqual(a2, 0.25 * a0 ** 2)

    def isbccMetric(self, crys, a0=0):
        if a0 == 0: a0 = self.a0
        self.assertAlmostEqual(crys.volume, 0.5 * a0 ** 3)
        for i, a2 in enumerate(crys.metric.flatten()):
            if i % 4 == 0:
                # diagonal element
                self.assertAlmostEqual(a2, 0.75 * a0 ** 2)
            else:
                # off-diagonal element
                self.assertAlmostEqual(a2, -0.25 * a0 ** 2)

    def ishexMetric(self, crys, a0=0, c_a=0):
        if a0 == 0: a0 = self.a0
        if c_a == 0: c_a = self.c_a
        self.assertAlmostEqual(crys.volume, np.sqrt(0.75) * c_a * a0 ** 3)
        self.assertAlmostEqual(crys.metric[0, 0], a0 ** 2)
        self.assertAlmostEqual(crys.metric[1, 1], a0 ** 2)
        self.assertAlmostEqual(crys.metric[0, 1], -0.5 * a0 ** 2)
        self.assertAlmostEqual(crys.metric[2, 2], (c_a * a0) ** 2)
        self.assertAlmostEqual(crys.metric[0, 2], 0)
        self.assertAlmostEqual(crys.metric[1, 2], 0)

    def issquareMetric(self, crys, a0=0):
        if a0 == 0: a0 = self.a0
        self.assertAlmostEqual(crys.volume, a0 ** 2)
        for i, a2 in enumerate(crys.metric.flatten()):
            if i % 3 == 0:
                # diagonal element
                self.assertAlmostEqual(a2, a0 ** 2)
            else:
                # off-diagonal element
                self.assertAlmostEqual(a2, 0)

    def isspacegroup(self, crys):
        """Check that the space group obeys all group definitions: not fast."""
        # 1. Contains the identity: O(group size)
        identity = None
        dim = crys.dim
        for g in crys.G:
            if np.all(g.rot == np.eye(dim, dtype=int)):
                identity = g
                self.assertTrue(np.allclose(g.trans, 0),
                                msg="Identity has bad translation: {}".format(g.trans))
                for atommap in g.indexmap:
                    for i, j in enumerate(atommap):
                        self.assertTrue(i == j,
                                        msg="Identity has bad indexmap: {}".format(g.indexmap))
        self.assertTrue(identity is not None, msg="Missing identity")
        # 2. Check for inverses: O(group size^2)
        for g in crys.G:
            inverse = g.inv().inhalf()
            self.assertIn(inverse, crys.G,
                          msg="Missing inverse op:\n{}\n{}|{}^-1 =\n{}\n{}|{}".format(
                              g.rot, g.cartrot, g.trans,
                              inverse.rot, inverse.cartrot, inverse.trans))
        # 3. Closed under multiplication: g.g': O(group size^3)
        for g in crys.G:
            for gp in crys.G:
                product = (g * gp).inhalf()
                self.assertIn(product, crys.G,
                              msg="Missing product op:\n{}\n{}|{} *\n{}\n{}|{} = \n{}\n{}|{}".format(
                                  g.rot, g.cartrot, g.trans,
                                  gp.rot, gp.cartrot, gp.trans,
                                  product.rot, product.cartrot, product.trans))

    def testscMetric(self):
        """Does the simple cubic lattice have the right volume and metric?"""
        crys = crystal.Crystal(self.sclatt, self.basis)
        self.isscMetric(crys)
        self.assertEqual(len(crys.basis), 1)  # one chemistry
        self.assertEqual(len(crys.basis[0]), 1)  # one atom in the unit cell

    def testfccMetric(self):
        """Does the face-centered cubic lattice have the right volume and metric?"""
        crys = crystal.Crystal(self.fcclatt, self.basis)
        self.isfccMetric(crys)
        self.assertEqual(len(crys.basis), 1)  # one chemistry
        self.assertEqual(len(crys.basis[0]), 1)  # one atom in the unit cell

    def testbccMetric(self):
        """Does the body-centered cubic lattice have the right volume and metric?"""
        crys = crystal.Crystal(self.bcclatt, self.basis)
        self.isbccMetric(crys)
        self.assertEqual(len(crys.basis), 1)  # one chemistry
        self.assertEqual(len(crys.basis[0]), 1)  # one atom in the unit cell

    def testsquareMetric(self):
        """Does the square lattice have the right volume and metric?"""
        crys = crystal.Crystal(self.squarelatt, self.basis2d)
        self.issquareMetric(crys)
        self.assertEqual(len(crys.basis), 1)  # one chemistry
        self.assertEqual(len(crys.basis[0]), 1)  # one atom in the unit cell

    def testscReduce(self):
        """If we start with a supercell, does it get reduced back to our start?"""
        nsuper = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=int)
        doublebasis = [self.basis[0], np.array([0.5, 0, 0]) + self.basis[0],
                       np.array([0, 0.5, 0]) + self.basis[0], np.array([0.5, 0.5, 0]) + self.basis[0]]
        crys = crystal.Crystal(np.dot(self.sclatt, nsuper), doublebasis)
        self.isscMetric(crys)
        self.assertEqual(len(crys.basis), 1)  # one chemistry
        self.assertEqual(len(crys.basis[0]), 1)  # one atom in the unit cell

    def testscReduce2(self):
        """If we start with a supercell, does it get reduced back to our start?"""
        nsuper = np.array([[5, -3, 0], [1, -1, 3], [-2, 1, 1]], dtype=int)
        crys = crystal.Crystal(np.dot(self.sclatt, nsuper), self.basis)
        self.isscMetric(crys)
        self.assertEqual(len(crys.basis), 1)  # one chemistry
        self.assertEqual(len(crys.basis[0]), 1)  # one atom in the unit cell

    def testbccReduce2(self):
        """If we start with a supercell, does it get reduced back to our start?"""
        basis = [[np.array([0., 0., 0.]), np.array([0.5, 0.5, 0.5])]]
        crys = crystal.Crystal(self.sclatt, basis)
        self.isbccMetric(crys)
        self.assertEqual(len(crys.basis), 1)  # one chemistry
        self.assertEqual(len(crys.basis[0]), 1)  # one atom in the unit cell

    def testscShift(self):
        """If we start with a supercell, does it get reduced back to our start?"""
        nsuper = np.array([[5, -3, 0], [1, -1, 3], [-2, 1, 1]], dtype=int)
        basis = [np.array([0.33, -0.25, 0.45])]
        crys = crystal.Crystal(np.dot(self.sclatt, nsuper), basis)
        self.isscMetric(crys)
        self.assertEqual(len(crys.basis), 1)  # one chemistry
        self.assertEqual(len(crys.basis[0]), 1)  # one atom in the unit cell
        self.assertTrue(np.allclose(crys.basis[0][0], np.array([0, 0, 0])))

    def testsquareReduce(self):
        """If we start with a supercell, does it get reduced back to our start?"""
        nsuper = np.array([[2, 0], [0, 2]], dtype=int)
        doublebasis = [self.basis2d[0], np.array([0.5, 0]) + self.basis2d[0],
                       np.array([0, 0.5]) + self.basis2d[0], np.array([0.5, 0.5]) + self.basis2d[0]]
        crys = crystal.Crystal(np.dot(self.squarelatt, nsuper), doublebasis)
        self.issquareMetric(crys)
        self.assertEqual(len(crys.basis), 1)  # one chemistry
        self.assertEqual(len(crys.basis[0]), 1)  # one atom in the unit cell

    def testhcp(self):
        """If we start with a supercell, does it get reduced back to our start?"""
        basis = [np.array([0, 0, 0]), np.array([1. / 3., 2. / 3., 1. / 2.])]
        crys = crystal.Crystal(self.hexlatt, basis)
        self.ishexMetric(crys)
        self.assertEqual(len(crys.basis), 1)  # one chemistry
        self.assertEqual(len(crys.basis[0]), 2)  # two atoms in the unit cell
        # there needs to be [1/3,2/3,1/4] or [1/3,2/3,3/4], and then the opposite
        # it's a little clunky; there's probably a better way to test this:
        if np.any([np.allclose(u, np.array([1. / 3., 2. / 3., 0.25]))
                   for atomlist in crys.basis for u in atomlist]):
            self.assertTrue(np.any([np.allclose(u, np.array([2. / 3., 1. / 3., 0.75]))
                                    for atomlist in crys.basis for u in atomlist]))
        elif np.any([np.allclose(u, np.array([1. / 3., 2. / 3., 0.75]))
                     for atomlist in crys.basis for u in atomlist]):
            self.assertTrue(np.any([np.allclose(u, np.array([2. / 3., 1. / 3., 0.25]))
                                    for atomlist in crys.basis for u in atomlist]))
        else:
            self.assertTrue(False, msg="HCP basis not correct")
        self.assertEqual(len(crys.G), 24)
        # for g in crys.G:
        #     print g.rot
        #     print g.cartrot, g.trans, g.indexmap
        self.isspacegroup(crys)
        self.assertEqual(len(crys.pointG[0][0]), 12)
        self.assertEqual(len(crys.pointG[0][1]), 12)

    def testLaGaO3(self):
        """Can we properly reduce down an LaGaO3 structure with errors in positions?"""
        # this uses "real" DFT relaxation data to test the reduction capabilities
        LaGa03latt = [np.array([  7.88040734e+00,   5.87657472e-05,  -1.95441808e-02]),
                      np.array([ -7.59206882e-05,   7.87786508e+00,   8.28811636e-05]),
                      np.array([ -1.95315626e-02,  -5.74109318e-05,   7.88041614e+00])]
        LaGaO3basis = [[np.array([  2.02290790e-02,   2.32539034e-04,   9.91922251e-01]),
                        np.array([  1.26313454e-02,   2.30601523e-04,   4.84327798e-01]),
                        np.array([ 0.97941805,  0.50023385,  0.01754055]),
                        np.array([ 0.98701667,  0.50023207,  0.52514002]),
                        np.array([  5.12632654e-01,   2.30909936e-04,   9.84337122e-01]),
                        np.array([  5.20224990e-01,   2.32577464e-04,   4.91932968e-01]),
                        np.array([ 0.48701525,  0.50023187,  0.02514135]),
                        np.array([ 0.47942077,  0.5002339 ,  0.51754884])],
                       [np.array([ 0.24982273,  0.25023308,  0.25473045]),
                        np.array([ 0.24982282,  0.25023333,  0.75473148]),
                        np.array([ 0.249823  ,  0.75023368,  0.25472946]),
                        np.array([ 0.24982247,  0.75023396,  0.75473027]),
                        np.array([ 0.74982257,  0.2502339 ,  0.25473326]),
                        np.array([ 0.74982307,  0.25023197,  0.75473186]),
                        np.array([ 0.74982204,  0.75023295,  0.25473187]),
                        np.array([ 0.74982322,  0.75023469,  0.75473098])],
                       [np.array([ 0.28414742,  0.20916336,  0.00430709]),
                        np.array([ 0.0002463 ,  0.20916015,  0.22041692]),
                        np.array([  2.80317156e-01,   2.28151610e-04,   3.00655890e-01]),
                        np.array([ 0.21550181,  0.29129973,  0.50516544]),
                        np.array([ 0.99940227,  0.29128777,  0.78906602]),
                        np.array([  2.03918412e-01,   2.36510236e-04,   7.24241274e-01]),
                        np.array([ 0.2841317 ,  0.791303  ,  0.00431445]),
                        np.array([  2.54313708e-04,   7.91306290e-01,   2.20429168e-01]),
                        np.array([ 0.21933007,  0.50023581,  0.2088184 ]),
                        np.array([ 0.21551645,  0.70916116,  0.50515561]),
                        np.array([ 0.99939381,  0.7091728 ,  0.78904879]),
                        np.array([ 0.29572872,  0.50022831,  0.78523308]),
                        np.array([ 0.71550064,  0.29129386,  0.00516782]),
                        np.array([ 0.4994013 ,  0.29130198,  0.28906235]),
                        np.array([  7.03903980e-01,   2.36323588e-04,   2.24257240e-01]),
                        np.array([ 0.78414767,  0.20916926,  0.50430849]),
                        np.array([ 0.50024549,  0.20917445,  0.72041481]),
                        np.array([  7.80305988e-01,   2.27988377e-04,   8.00654063e-01]),
                        np.array([ 0.71551543,  0.7091663 ,  0.0051578 ]),
                        np.array([ 0.49939281,  0.70915813,  0.28904503]),
                        np.array([ 0.79574297,  0.50022792,  0.28522595]),
                        np.array([ 0.78413198,  0.79129631,  0.50431609]),
                        np.array([ 0.50025359,  0.79129237,  0.72042732]),
                        np.array([ 0.71934128,  0.50023592,  0.70882833])]]
        LaGaO3strict = crystal.Crystal(LaGa03latt, LaGaO3basis, ['La', 'Ga', 'O'],
                                       threshold=1e-8)
        LaGaO3toler = crystal.Crystal(LaGa03latt, LaGaO3basis, ['La', 'Ga', 'O'],
                                      threshold=2e-5)
        self.assertEqual(len(LaGaO3strict.G), 1)
        self.assertEqual(len(LaGaO3toler.G), 2)
        self.assertEqual([len(ulist) for ulist in LaGaO3strict.basis],
                         [len(ulist) for ulist in LaGaO3basis])
        self.assertEqual([2*len(ulist) for ulist in LaGaO3toler.basis],
                         [len(ulist) for ulist in LaGaO3basis])
        self.assertAlmostEqual(LaGaO3strict.volume, 2*LaGaO3toler.volume)


    def testscgroupops(self):
        """Do we have 48 space group operations?"""
        crys = crystal.Crystal(self.sclatt, self.basis)
        self.assertEqual(len(crys.G), 48)
        self.isspacegroup(crys)
        # for g in crys.G:
        #     print g.rot, g.trans, g.indexmap
        #     print g.cartrot, g.carttrans

    def testfccpointgroup(self):
        """Test out that we generate point groups correctly"""
        crys = crystal.Crystal(self.fcclatt, self.basis)
        for g in crys.G:
            self.assertIn(g, crys.pointG[0][0])

    def testsquaregroupops(self):
        """Do we have 8 space group operations?"""
        crys = crystal.Crystal(self.squarelatt, self.basis2d)
        self.assertEqual(len(crys.G), 8)
        self.isspacegroup(crys)
        # for g in crys.G:
        #     print g.rot, g.trans, g.indexmap
        #     print g.cartrot, g.carttrans

    def testomegagroupops(self):
        """Build the omega lattice; make sure the space group is correct"""
        basis = [[np.array([0., 0., 0.]),
                  np.array([1. / 3., 2. / 3., 0.5]),
                  np.array([2. / 3., 1. / 3., 0.5])]]
        crys = crystal.Crystal(self.hexlatt, basis)
        self.assertEqual(crys.N, 3)
        self.assertEqual(crys.atomindices, [(0, 0), (0, 1), (0, 2)])
        self.assertEqual(len(crys.G), 24)
        self.isspacegroup(crys)

    def testcartesianposition(self):
        """Do we correctly map out our atom position (lattice vector + indices) in cartesian coord.?"""
        crys = crystal.Crystal(self.fcclatt, self.basis)
        lattvect = np.array([2, -1, 3])
        for ind in crys.atomindices:
            b = crys.basis[ind[0]][ind[1]]
            pos = crys.lattice[:, 0] * (lattvect[0] + b[0]) + \
                  crys.lattice[:, 1] * (lattvect[1] + b[1]) + \
                  crys.lattice[:, 2] * (lattvect[2] + b[2])
            self.assertTrue(np.allclose(pos, crys.pos2cart(lattvect, ind)))
        basis = [[np.array([0., 0., 0.]),
                  np.array([1. / 3., 2. / 3., 0.5]),
                  np.array([2. / 3., 1. / 3., 0.5])]]
        crys = crystal.Crystal(self.hexlatt, basis)
        for ind in crys.atomindices:
            b = crys.basis[ind[0]][ind[1]]
            pos = crys.lattice[:, 0] * (lattvect[0] + b[0]) + \
                  crys.lattice[:, 1] * (lattvect[1] + b[1]) + \
                  crys.lattice[:, 2] * (lattvect[2] + b[2])
            self.assertTrue(np.allclose(pos, crys.pos2cart(lattvect, ind)))

    def testmaptrans(self):
        """Does our map translation operate correctly?"""
        basis = [[np.array([0, 0, 0])]]
        trans, indexmap = crystal.maptranslation(basis, basis)
        self.assertTrue(np.allclose(trans, np.array([0, 0, 0])))
        self.assertEqual(indexmap, ((0,),))

        oldbasis = [[np.array([0.2, 0, 0])]]
        newbasis = [[np.array([-0.2, 0, 0])]]
        trans, indexmap = crystal.maptranslation(oldbasis, newbasis)
        self.assertTrue(np.allclose(trans, np.array([0.4, 0, 0])))
        self.assertEqual(indexmap, ((0,),))

        oldbasis = [[np.array([0., 0., 0.]), np.array([1. / 3., 2. / 3., 1. / 2.])]]
        newbasis = [[np.array([0., 0., 0.]), np.array([-1. / 3., -2. / 3., -1. / 2.])]]
        trans, indexmap = crystal.maptranslation(oldbasis, newbasis)
        self.assertTrue(np.allclose(trans, np.array([1. / 3., -1. / 3., -1. / 2.])))
        self.assertEqual(indexmap, ((1, 0),))

        oldbasis = [[np.array([0., 0., 0.])],
                    [np.array([1. / 3., 2. / 3., 1. / 2.]), np.array([2. / 3., 1. / 3., 1. / 2.])]]
        newbasis = [[np.array([0., 0., 0.])],
                    [np.array([2. / 3., 1. / 3., 1. / 2.]), np.array([1. / 3., 2. / 3., 1. / 2.])]]
        trans, indexmap = crystal.maptranslation(oldbasis, newbasis)
        self.assertTrue(np.allclose(trans, np.array([0., 0., 0.])))
        self.assertEqual(indexmap, ((0,), (1, 0)))

        oldbasis = [[np.array([0., 0., 0.]), np.array([1. / 3., 2. / 3., 1. / 2.])]]
        newbasis = [[np.array([0., 0., 0.]), np.array([-1. / 4., -1. / 2., -1. / 2.])]]
        trans, indexmap = crystal.maptranslation(oldbasis, newbasis)
        self.assertEqual(indexmap, None)

    def testfccgroupops_directions(self):
        """Test out that we can apply group operations to directions"""
        crys = crystal.Crystal(self.fcclatt, self.basis)
        # 1. direction
        direc = np.array([2., 0., 0.])
        direc2 = np.dot(direc, direc)
        count = np.zeros(3, dtype=int)
        for g in crys.G:
            rotdirec = crys.g_direc(g, direc)
            self.assertAlmostEqual(np.dot(rotdirec, rotdirec), direc2)
            costheta = np.dot(rotdirec, direc) / direc2
            self.assertTrue(np.isclose(costheta, 1) or np.isclose(costheta, 0) or np.isclose(costheta, -1))
            count[int(round(costheta + 1))] += 1
        self.assertEqual(count[0], 8)  ## antiparallel
        self.assertEqual(count[1], 32)  ## perpendicular
        self.assertEqual(count[2], 8)  ## parallel

    def testomegagroupops_positions(self):
        """Test out that we can apply group operations to positions"""
        # 2. position = lattice vector + 2-tuple atom-index
        basis = [[np.array([0., 0., 0.]),
                  np.array([1. / 3., 2. / 3., 0.5]),
                  np.array([2. / 3., 1. / 3., 0.5])]]
        crys = crystal.Crystal(self.hexlatt, basis)
        lattvec = np.array([-2, 3, 1])
        for ind in crys.atomindices:
            pos = crys.pos2cart(lattvec, ind)
            for g in crys.G:
                # testing g_pos: (transform an atomic position)
                rotpos = crys.g_direc(g, pos)
                self.assertTrue(np.allclose(rotpos,
                                            crys.pos2cart(*crys.g_pos(g, lattvec, ind))))
                # testing g_vect: (transform a vector position in the crystal)
                rotlatt, rotind = crys.g_pos(g, lattvec, ind)
                rotlatt2, u = crys.g_vect(g, lattvec, crys.basis[ind[0]][ind[1]])
                self.assertTrue(np.allclose(rotpos, crys.unit2cart(rotlatt2, u)))
                self.assertTrue(np.all(rotlatt == rotlatt2))
                self.assertTrue(np.allclose(u, crys.basis[rotind[0]][rotind[1]]))

            # test point group operations:
            for g in crys.pointG[ind[0]][ind[1]]:
                origin = np.zeros(3, dtype=int)
                rotlatt, rotind = crys.g_pos(g, origin, ind)
                self.assertTrue(np.all(rotlatt == origin))
                self.assertEqual(rotind, ind)

    def testinverspos(self):
        """Test the inverses of pos2cart and unit2cart"""
        basis = [[np.array([0., 0., 0.]),
                  np.array([1. / 3., 2. / 3., 0.5]),
                  np.array([2. / 3., 1. / 3., 0.5])]]
        crys = crystal.Crystal(self.hexlatt, basis)
        lattvec = np.array([-2, 3, 1])
        for ind in crys.atomindices:
            lattback, uback = crys.cart2unit(crys.pos2cart(lattvec, ind))
            self.assertTrue(np.all(lattback == lattvec))
            self.assertTrue(np.allclose(uback, crys.basis[ind[0]][ind[1]]))
            lattback, indback = crys.cart2pos(crys.pos2cart(lattvec, ind))
            self.assertTrue(np.all(lattback == lattvec))
            self.assertEqual(indback, ind)
        lattback, indback = crys.cart2pos(np.array([0.25 * self.a0, 0.25 * self.a0, 0.]))
        self.assertIsNone(indback)

    def testWyckoff(self):
        """Test grouping for Wyckoff positions"""
        basis = [[np.array([0., 0., 0.]),
                  np.array([1. / 3., 2. / 3., 0.5]),
                  np.array([2. / 3., 1. / 3., 0.5])]]
        crys = crystal.Crystal(self.hexlatt, basis)
        # crys.Wyckoff : frozen set of frozen sets of tuples that are all equivalent
        Wyckoffind = {frozenset([(0, 0)]),
                      frozenset([(0, 1), (0, 2)])}
        self.assertEqual(crys.Wyckoff, Wyckoffind)
        # now ask it to generate the set of all equivalent points
        for wyckset in crys.Wyckoff:
            for ind in wyckset:
                # construct our own Wyckoff set using cart2pos...
                wyckset2 = crys.Wyckoffpos(crys.basis[ind[0]][ind[1]])
                # test equality:
                for i in wyckset:
                    self.assertTrue(np.any([np.allclose(crys.basis[i[0]][i[1]], u) for u in wyckset2]))
                for u in wyckset2:
                    self.assertTrue(np.any([np.allclose(crys.basis[i[0]][i[1]], u) for i in wyckset]))

    def testVectorBasis(self):
        """Test for the generation of a vector (and tensor) basis for sites in a crystal: oct. + tet."""
        # start with HCP, then "build out" a lattice that includes interstitial sites
        basis = [[np.array([1. / 3., 2. / 3., 0.25]),
                  np.array([2. / 3., 1. / 3., 0.75])]]
        HCPcrys = crystal.Crystal(self.hexlatt, basis)
        octset = HCPcrys.Wyckoffpos(np.array([0., 0., 0.5]))
        tetset = HCPcrys.Wyckoffpos(np.array([1. / 3., 2. / 3., 0.5]))
        self.assertEqual(len(octset), 2)
        self.assertEqual(len(tetset), 4)
        # now, build up HCP + interstitials (which are of a *different chemistry*)
        HCP_intercrys = crystal.Crystal(self.hexlatt, basis + [octset + tetset])
        for i in range(2):
            vbas = HCP_intercrys.VectorBasis((1, i))  # for our octahedral site
            self.assertEqual(vbas[0], 0)  # should be a point
            tbas = HCP_intercrys.SymmTensorBasis((1, i))
            self.assertEqual(len(tbas), 2)
            for t in tbas:
                for tij in (t[i, j] for i in range(3) for j in range(3) if i != j):
                    self.assertAlmostEqual(0, tij)
                self.assertAlmostEqual(t[0, 0], t[1, 1])
        for i in range(2, 6):
            vbas = HCP_intercrys.VectorBasis((1, i))  # for our tetrahedal sites
            self.assertEqual(vbas[0], 1)  # should be a line
            self.assertEqual(vbas[1][0], 0)  # pointing vertically up
            self.assertEqual(vbas[1][1], 0)  # pointing vertically up
            tbas = HCP_intercrys.SymmTensorBasis((1, i))
            self.assertEqual(len(tbas), 2)
            for t in tbas:
                for tij in (t[i, j] for i in range(3) for j in range(3) if i != j):
                    self.assertAlmostEqual(0, tij)
                self.assertAlmostEqual(t[0, 0], t[1, 1])

    def testJumpNetwork(self):
        """Test for the generation of our jump network between octahedral and tetrahedral sites."""
        # start with HCP, then "build out" a lattice that includes interstitial sites
        basis = [[np.array([1. / 3., 2. / 3., 0.25]),
                  np.array([2. / 3., 1. / 3., 0.75])]]
        HCPcrys = crystal.Crystal(self.hexlatt, basis)
        octset = HCPcrys.Wyckoffpos(np.array([0., 0., 0.5]))
        tetset = HCPcrys.Wyckoffpos(np.array([1. / 3., 2. / 3., 0.625]))
        self.assertEqual(len(octset), 2)
        self.assertEqual(len(tetset), 4)
        # now, build up HCP + interstitials (which are of a *different chemistry*)
        HCP_intercrys = crystal.Crystal(self.hexlatt, basis + [octset + tetset])
        jumpnetwork = HCP_intercrys.jumpnetwork(1, self.a0 * 0.8, 0.5 * self.a0)  # tuned to avoid t->t in basal plane
        self.assertEqual(len(jumpnetwork), 2)  # should contain o->t/t->o and t->t networks
        self.assertEqual(sorted(len(t) for t in jumpnetwork), [4, 24])
        # print crystal.yaml.dump(jumpnetwork)
        # for i, t in enumerate(jumpnetwork):
        #     print i, len(t)
        #     for ij, dx in t:
        #         print "{} -> {}: {}".format(ij[0], ij[1], dx)

    def testNNfcc(self):
        """Test of the nearest neighbor construction"""
        crys = crystal.Crystal(self.fcclatt, self.basis)
        nnlist = crys.nnlist((0, 0), 0.9 * self.a0)
        self.assertEqual(len(nnlist), 12)
        for x in nnlist:
            self.assertTrue(np.isclose(np.dot(x, x), 0.5 * self.a0 * self.a0))


class CrystalSpinTests(unittest.TestCase):
    """Tests for crystal class when spins are involved"""
    longMessage = False

    def setUp(self):
        self.a0 = 1.0
        self.latt = self.a0 * np.eye(3)
        # RockSalt:
        self.basis = [[np.array([0., 0., 0.]), np.array([0., 0.5, 0.5]),
                       np.array([0.5, 0., 0.5]), np.array([0.5, 0.5, 0.])],
                      [np.array([0., 0., 0.5]), np.array([0., 0.5, 0.]),
                       np.array([0.5, 0., 0.]), np.array([0.5, 0.5, 0.5])]]
        self.spins = [[1, -1, -1, 1], [0, 0, 0, 0]]

    def testUN(self):
        """Uranium-Nitride structure"""
        crys = crystal.Crystal(self.latt, self.basis, ['U', 'N'], self.spins)
        # print(crys)
        self.assertTrue(crys is not None)
        self.assertEqual(len(crys.basis), 2)
        self.assertEqual(len(crys.basis[0]), 2)
        self.assertEqual(len(crys.basis[1]), 2)
        self.assertEqual(len(crys.Wyckoff), 2,
                         msg='Not matching Wyckoff?\n{}\n{}'.format(crys.Wyckoff, crys))
        tetlatt = self.a0 * np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.01]])
        tetcrys = crystal.Crystal(tetlatt, self.basis, ['U', 'N'])
        # the tetragonal distortion has 1 U and 1 N (when there's no spin), so the group op list
        # is twice as big, and includes translations for each
        self.assertEqual(len(crys.G), 2 * len(tetcrys.G))

    def testmaptrans(self):
        """Does our map translation operate correctly with spins?"""
        basis = [[np.array([0, 0, 0])]]
        spins = [[1]]
        trans, indexmap = crystal.maptranslation(basis, basis, spins, spins)
        self.assertTrue(np.allclose(trans, np.array([0, 0, 0])))
        self.assertEqual(indexmap, ((0,),))

        oldbasis = [[np.array([0.2, 0, 0])]]
        newbasis = [[np.array([-0.2, 0, 0])]]
        oldspins = [[1]]
        newspins = [[-1]]
        trans, indexmap = crystal.maptranslation(oldbasis, newbasis, oldspins, newspins)
        self.assertEqual(indexmap, None)  # should NOT be able to do this mapping with spins!

        oldbasis = [[np.array([0., 0., 0.]), np.array([0., 0.5, 0.5]),
                     np.array([0.5, 0., 0.5]), np.array([0.5, 0.5, 0.])]]
        newbasis = oldbasis
        oldspins, newspins = [[2, -2, -1, 1]], [[2, -2, -1, 1]]
        trans, indexmap = crystal.maptranslation(oldbasis, newbasis, oldspins, newspins)
        self.assertTrue(np.allclose(trans, np.array([0., 0., 0.])))
        self.assertEqual(indexmap, ((0, 1, 2, 3),))

        oldspins, newspins = [[2, -2, -1, 1]], [[1, -1, -2, 2]]
        trans, indexmap = crystal.maptranslation(oldbasis, newbasis, oldspins, newspins)
        self.assertTrue(np.allclose(np.abs(trans[0]), 0.5))
        self.assertTrue(np.allclose(np.abs(trans[1]), 0.5))
        self.assertTrue(np.allclose(np.abs(trans[2]), 0.))
        self.assertEqual(indexmap, ((3, 2, 1, 0),))

    def testAddBasis(self):
        """Uranium-Nitride, with addbasis"""
        UNcrys = crystal.Crystal(self.latt, self.basis, ['U', 'N'], self.spins)
        Ucrys = crystal.Crystal(self.latt, self.basis[0], ['U'], self.spins[0])
        UNnewcrys = Ucrys.addbasis(Ucrys.Wyckoffpos(Ucrys.cart2unit(np.array([0., 0., 0.5 * self.a0]))[1]),
                                   ['N'], [0, 0])
        self.assertEqual(len(UNnewcrys.basis), 2,
                         msg="Failed? {}\n+ basis:\n{}\ndoesn't match:\n{}".format(Ucrys, UNnewcrys, UNcrys))
        self.assertEqual(len(UNnewcrys.basis[0]), 2,
                         msg="Failed? {}\n+ basis:\n{}\ndoesn't match:\n{}".format(Ucrys, UNnewcrys, UNcrys))
        self.assertEqual(len(UNnewcrys.basis[1]), 2,
                         msg="Failed? {}\n+ basis:\n{}\ndoesn't match:\n{}".format(Ucrys, UNnewcrys, UNcrys))


class YAMLTests(unittest.TestCase):
    """Tests to make sure we can use YAML to write and read our classes."""

    def testarrayYAML(self):
        """Test that we can write and read an array"""
        for a in [np.array([1]), np.array([1., 0., 0.]), np.eye(3, dtype=float), np.eye(3, dtype=int)]:
            # we could do this with one call; if we want to add tests for the format later
            # they should go in between here.
            awrite = crystal.yaml.dump(a)
            aread = crystal.yaml.load(awrite)
            self.assertTrue(np.allclose(a, aread))
            self.assertIsInstance(aread, np.ndarray)

    def testSetYAML(self):
        """Test that we can use YAML to write and read a frozenset"""
        for a in [frozenset([]), frozenset([1]), frozenset([0, 1, 1]), frozenset(list(range(10)))]:
            awrite = crystal.yaml.dump(a)
            aread = crystal.yaml.load(awrite)
            self.assertEqual(a, aread)
            self.assertIsInstance(aread, frozenset)

    def testGroupOpYAML(self):
        """Test that we can write and read a GroupOp"""
        g = crystal.GroupOp(np.eye(3, dtype=int),
                            np.array([0., 0., 0.]),
                            np.eye(3),
                            ((0,),))
        # crystal.yaml.add_representer(crystal.GroupOp, crystal.GroupOp_representer)
        gwrite = crystal.yaml.dump(g)
        gread = crystal.yaml.load(gwrite)
        self.assertEqual(g, gread)
        self.assertIsInstance(gread, crystal.GroupOp)

    def testCrystalYAML(self):
        """Test that we can write and read a crystal"""
        a0 = 2.5
        c_a = np.sqrt(8. / 3.)
        hexlatt = a0 * np.array([[0.5, 0.5, 0],
                                 [-np.sqrt(0.75), np.sqrt(0.75), 0],
                                 [0, 0, c_a]])
        basis = [[np.array([0., 0., 0.]),
                  np.array([1. / 3., 2. / 3., 0.5]),
                  np.array([2. / 3., 1. / 3., 0.5])]]
        crys = crystal.Crystal(hexlatt, basis)
        cryswrite = crystal.yaml.dump(crys)
        crysread = crystal.yaml.load(cryswrite)
        self.assertIsInstance(crysread, crystal.Crystal)
        self.assertTrue(np.allclose(crys.lattice, crysread.lattice))
        self.assertTrue(np.allclose(crys.invlatt, crysread.invlatt))
        self.assertTrue(np.allclose(crys.reciplatt, crysread.reciplatt))
        self.assertTrue(np.allclose(crys.metric, crysread.metric))
        self.assertEqual(crys.N, crysread.N)
        self.assertTrue(np.allclose(crys.basis, crysread.basis))
        self.assertEqual(crys.G, crysread.G)
        self.assertAlmostEqual(crys.volume, crysread.volume)
        self.assertAlmostEqual(crys.BZvol, crysread.BZvol)
        self.assertEqual(crys.atomindices, crysread.atomindices)
        self.assertEqual(crys.pointG, crysread.pointG)
        self.assertEqual(crys.Wyckoff, crysread.Wyckoff)

    def testCrystalYAMLsimplified(self):
        """Test that we can read a simplified crystal input"""
        a0 = 2.5
        c_a = np.sqrt(8. / 3.)
        hexlatt = a0 * np.array([[0.5, 0.5, 0],
                                 [-np.sqrt(0.75), np.sqrt(0.75), 0],
                                 [0, 0, c_a]])
        basis = [[np.array([0., 0., 0.]),
                  np.array([1. / 3., 2. / 3., 0.5]),
                  np.array([2. / 3., 1. / 3., 0.5])]]
        crys = crystal.Crystal(hexlatt, basis)
        yamlstr = """lattice_constant: 2.5
lattice: {YAMLtag}
- [0.5, -0.8660254037844386, 0]
- [0.5,  0.8660254037844386, 0]
- [0.0, 0.0, 1.6329931618554521]
basis:
- - {YAMLtag} [0.0, 0.0, 0.0]
  - {YAMLtag} [0.3333333333333333, 0.6666666666666666, 0.5]
  - {YAMLtag} [0.6666666666666666, 0.3333333333333333, 0.5]
chemistry:
- Ti""".format(YAMLtag=crystal.NDARRAY_YAMLTAG)  # rather than hard-coding the tag...
        crysread = crystal.Crystal.fromdict(crystal.yaml.load(yamlstr))
        self.assertIsInstance(crysread, crystal.Crystal)
        self.assertTrue(np.allclose(crys.lattice, crysread.lattice))
        self.assertTrue(np.allclose(crys.invlatt, crysread.invlatt))
        self.assertTrue(np.allclose(crys.reciplatt, crysread.reciplatt))
        self.assertTrue(np.allclose(crys.metric, crysread.metric))
        self.assertEqual(crys.N, crysread.N)
        self.assertTrue(np.allclose(crys.basis, crysread.basis))
        self.assertEqual(crys.G, crysread.G)
        self.assertAlmostEqual(crys.volume, crysread.volume)
        self.assertAlmostEqual(crys.BZvol, crysread.BZvol)
        self.assertEqual(crys.atomindices, crysread.atomindices)
        self.assertEqual(crys.pointG, crysread.pointG)
        self.assertEqual(crys.Wyckoff, crysread.Wyckoff)


class KPTgentest(unittest.TestCase):
    """Tests for KPT mesh pieces of Crystal class"""

    def setUp(self):
        self.a0 = 1.0
        self.sclatt = self.a0 * np.eye(3)
        self.basis = [np.array([0., 0., 0.])]
        self.crys = crystal.Crystal(self.sclatt, self.basis)
        self.N = (4, 4, 4)
        self.squarelatt = self.a0 * np.eye(2)
        self.basis2d = [np.array([0., 0.])]

    def testKPTreciprocallattice(self):
        """Have we correctly constructed the reciprocal lattice vectors?"""
        dotprod = np.dot(self.crys.reciplatt.T, self.crys.lattice)
        dotprod0 = 2. * np.pi * np.eye(3)
        for a in range(3):
            for b in range(3):
                self.assertAlmostEqual(dotprod[a, b], dotprod0[a, b])

    def testKPTconstruct(self):
        """Can we construct a mesh with the correct number of points?"""
        # reset
        kpts = self.crys.fullkptmesh(self.N)
        self.assertEqual(kpts.shape[0], np.product(self.N))
        self.assertEqual(kpts.shape[1], 3)

    def testKPT_BZ_Gpoints(self):
        """Do we have the correct G points that define the BZ?"""
        self.assertEqual(np.shape(self.crys.BZG), (6, 3))
        self.assertTrue(any(all((np.pi, 0, 0) == x) for x in self.crys.BZG))
        self.assertTrue(any(all((-np.pi, 0, 0) == x) for x in self.crys.BZG))
        self.assertTrue(any(all((0, np.pi, 0) == x) for x in self.crys.BZG))
        self.assertTrue(any(all((0, -np.pi, 0) == x) for x in self.crys.BZG))
        self.assertTrue(any(all((0, 0, np.pi) == x) for x in self.crys.BZG))
        self.assertTrue(any(all((0, 0, -np.pi) == x) for x in self.crys.BZG))
        self.assertFalse(any(all((0, 0, 0) == x) for x in self.crys.BZG))
        vec = np.array((1, 1, 1))
        self.assertTrue(self.crys.inBZ(vec))
        vec = np.array((4, 0, -4))
        self.assertFalse(self.crys.inBZ(vec))

    def testKPT_fullmesh_points(self):
        """Are the points in the k-point mesh that we expect to see?"""
        kpts = self.crys.fullkptmesh(self.N)
        self.assertTrue(any(all((2. * np.pi / self.N[0], 0, 0) == x) for x in kpts))

    def testKPT_insideBZ(self):
        """Do we only have points that are inside the BZ?"""
        for q in self.crys.fullkptmesh(self.N):
            self.assertTrue(self.crys.inBZ(q),
                            msg="Failed with vector {} not in BZ".format(q))

    def testKPT_IRZ(self):
        """Do we produce a correct irreducible wedge?"""
        Nkpt = np.prod(self.N)
        kptfull = self.crys.fullkptmesh(self.N)
        kpts, wts = self.crys.reducekptmesh(kptfull)  # self.crys.fullkptmesh(self.N)
        for i, k in enumerate(kpts):
            # We want to determine what the weight for each point should be, and compare
            # dealing with the BZ edges is complicated; so we skip that in our tests
            if all([np.dot(k, G) < (np.dot(G, G) - 1e-8) for G in self.crys.BZG]):
                basewt = 1. / Nkpt
                sortk = sorted(k)
                basewt *= (2 ** (3 - list(k).count(0)))
                if sortk[0] != sortk[1] and sortk[1] != sortk[2]:
                    basewt *= 6
                elif sortk[0] != sortk[1] or sortk[1] != sortk[2]:
                    basewt *= 3
                self.assertAlmostEqual(basewt, wts[i])
        # integration test
        wtfull = np.array((1 / Nkpt,) * Nkpt)
        self.assertAlmostEqual(sum(wtfull * [np.cos(sum(k)) for k in kptfull]),
                               sum(wts * [np.cos(sum(k)) for k in kpts]))
        self.assertNotAlmostEqual(sum(wtfull * [np.cos(k[0]) for k in kptfull]),
                                  sum(wts * [np.cos(k[0]) for k in kpts]))

    def testKPT_integration(self):
        """Do we get integral values that we expect? 1/(2pi)^3 int cos(kx+ky+kz)^3 = 1/2"""
        Nkpt = np.prod(self.N)
        kptfull = self.crys.fullkptmesh(self.N)
        wtfull = np.array((1 / Nkpt,) * Nkpt)
        kpts, wts = self.crys.reducekptmesh(kptfull)  # self.crys.fullkptmesh(self.N)
        self.assertAlmostEqual(sum(wtfull * [np.cos(sum(k)) ** 2 for k in kptfull]), 0.5)
        self.assertAlmostEqual(sum(wts * [np.cos(sum(k)) ** 2 for k in kpts]), 0.5)
        self.assertAlmostEqual(sum(wtfull * [np.cos(sum(k)) for k in kptfull]), 0)
        self.assertAlmostEqual(sum(wts * [np.cos(sum(k)) for k in kpts]), 0)
        # Note: below we have the true values of the integral, but these should disagree
        # due to numerical error.
        self.assertNotAlmostEqual(sum(wtfull * [sum(k) ** 2 for k in kptfull]), 9.8696044010893586188)
        self.assertNotAlmostEqual(sum(wts * [sum(k) ** 2 for k in kpts]), 9.8696044010893586188)

    def test2DKPT_integration(self):
        """Do we get integral values that we expect? 1/(2pi)^2 int cos(kx+ky)^2 = 1/2"""
        crys = crystal.Crystal(self.squarelatt, self.basis2d)
        N = self.N[:2]
        Nkpt = np.prod(N)
        kptfull = crys.fullkptmesh(N)
        wtfull = np.array((1 / Nkpt,) * Nkpt)
        kpts, wts = crys.reducekptmesh(kptfull)  # self.crys.fullkptmesh(self.N)
        self.assertAlmostEqual(sum(wtfull * [np.cos(sum(k)) ** 2 for k in kptfull]), 0.5)
        self.assertAlmostEqual(sum(wts * [np.cos(sum(k)) ** 2 for k in kpts]), 0.5)
        self.assertAlmostEqual(sum(wtfull * [np.cos(k[0])*np.cos(k[1]) for k in kptfull]), 0)
        self.assertAlmostEqual(sum(wts * [np.cos(k[0])*np.cos(k[1]) for k in kpts]), 0)
        self.assertAlmostEqual(sum(wtfull * [np.cos(k[0])**2*np.cos(k[1])**2 for k in kptfull]), 0.25)
        self.assertAlmostEqual(sum(wts * [np.cos(k[0])**2*np.cos(k[1])**2 for k in kpts]), 0.25)
        # Note: below we have the true values of the integral, but these should disagree
        # due to numerical error.
        self.assertNotAlmostEqual(sum(wtfull * [sum(k) ** 2 for k in kptfull]), 6.579736267392905)
        self.assertNotAlmostEqual(sum(wts * [sum(k) ** 2 for k in kpts]), 6.579736267392905)
