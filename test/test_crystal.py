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


class GroupOperationTests(unittest.TestCase):
    """Tests for our group operations."""
    def setUp(self):
        self.rot = np.array([[0,1,0],
                             [1,0,0],
                             [0,0,1]])
        self.trans = np.zeros(3)
        self.cartrot = np.array([[0.,1.,0.],
                                 [1.,0.,0.],
                                 [0.,0.,1.]])
        self.indexmap = [[0]]
        self.mirrorop = crystal.GroupOp(self.rot,self.trans,self.cartrot,self.indexmap)
        self.ident = crystal.GroupOp(np.eye(3, dtype=int), np.zeros(3), np.eye(3), [[0]])

    def testEquality(self):
        """Can we check if two group operations are equal?"""
        with self.assertRaises(TypeError):
            self.mirrorop == self.rot
        self.assertEqual(self.mirrorop.incell(), self.mirrorop)
        # self.assertEqual(self.mirrorop.__hash__(), (self.mirrorop + np.array([1,0,0])).__hash__())

    def testAddition(self):
        """Can we add a vector to our group operation and get a new one?"""
        with self.assertRaises(TypeError):
            self.mirrorop + 0
        v1 = np.array([1,0,0])
        newop = self.mirrorop + v1
        mirroroptrans = crystal.GroupOp(self.rot,self.trans + v1,self.cartrot,self.indexmap)
        self.assertEqual(newop, mirroroptrans)
        self.assertTrue(np.all(np.isclose((self.ident - v1).trans, -v1)))

    def testMultiplication(self):
        """Does group operation multiplication work correctly?"""
        self.assertEqual(self.mirrorop*self.mirrorop, self.ident)
        v1 = np.array([1,0,0])
        trans = self.ident + v1
        self.assertEqual(trans*trans, self.ident + 2*v1)
        rot3 = crystal.GroupOp(np.eye(3, dtype=int), np.zeros(3), np.eye(3), [[1,2,0]])
        ident3 = crystal.GroupOp(np.eye(3, dtype=int), np.zeros(3), np.eye(3), [[0,1,2]])
        self.assertEqual(rot3*rot3*rot3, ident3)

    def testInversion(self):
        """Is the product with the inverse equal to identity?"""
        self.assertEqual(self.ident.inv, self.ident.inv)
        self.assertEqual(self.mirrorop*(self.mirrorop.inv()), self.ident)
        v1 = np.array([1,0,0])
        trans = self.ident + v1
        self.assertEqual(trans.inv(), self.ident - v1)
        inversion = crystal.GroupOp(-np.eye(3,dtype=int), np.zeros(3), -np.eye(3), [[0]])
        self.assertEqual(inversion.inv(), inversion)
        invtrans = inversion + v1
        self.assertEqual(invtrans.inv(), invtrans)


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

    def isspacegroup(self, crys):
        """Check that the space group obeys all group definitions: not fast."""
        # 1. Contains the identity: O(group size)
        identity = None
        for g in crys.G:
            if np.all(g.rot == np.eye(3, dtype=int) ):
                identity = g
                self.assertTrue(np.all(np.isclose(g.trans, 0)),
                                msg="Identity has bad translation: {}".format(g.trans))
                for atommap in g.indexmap:
                    for i, j in enumerate(atommap):
                        self.assertTrue(i==j,
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
                product = (g*gp).inhalf()
                self.assertIn(product, crys.G,
                              msg="Missing product op:\n{}\n{}|{} *\n{}\n{}|{} = \n{}\n{}|{}".format(
                                  g.rot, g.cartrot, g.trans,
                                  gp.rot, gp.cartrot, gp.trans,
                                  product.rot, product.cartrot, product.trans))

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
        self.ishexMetric(crys)
        self.assertEqual(len(crys.basis), 1)    # one chemistry
        self.assertEqual(len(crys.basis[0]), 2) # two atoms in the unit cell
        # there needs to be [1/3,2/3,1/4] or [1/3,2/3,3/4], and then the opposite
        # it's a little clunky; there's probably a better way to test this:
        if np.any([ np.all(np.isclose(u, np.array([1./3.,2./3.,0.25])))
                    for atomlist in crys.basis for u in atomlist]):
            self.assertTrue(np.any([ np.all(np.isclose(u, np.array([2./3.,1./3.,0.75])))
                                     for atomlist in crys.basis for u in atomlist]))
        elif np.any([ np.all(np.isclose(u, np.array([1./3.,2./3.,0.75])))
                      for atomlist in crys.basis for u in atomlist]):
            self.assertTrue(np.any([ np.all(np.isclose(u, np.array([2./3.,1./3.,0.25])))
                                     for atomlist in crys.basis for u in atomlist]))
        else: self.assertTrue(False, msg="HCP basis not correct")
        self.assertEqual(len(crys.G), 24)
        # for g in crys.G:
        #     print g.rot
        #     print g.cartrot, g.trans, g.indexmap
        self.isspacegroup(crys)
        self.assertEqual(len(crys.pointG[0][0]), 12)
        self.assertEqual(len(crys.pointG[0][1]), 12)

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

    def testomegagroupops(self):
        """Build the omega lattice; make sure the space group is correct"""
        basis = [[np.array([0.,0.,0.]),
                  np.array([1./3.,2./3.,0.5]),
                  np.array([2./3.,1./3.,0.5])]]
        crys = crystal.Crystal(self.hexlatt, basis)
        self.assertEqual(crys.N, 3)
        self.assertEqual(crys.atomindices, [(0,0), (0,1), (0,2)])
        self.assertEqual(len(crys.G), 24)
        self.isspacegroup(crys)

    def testcartesianposition(self):
        """Do we correctly map out our atom position (lattice vector + indices) in cartesian coord.?"""
        crys = crystal.Crystal(self.fcclatt, self.basis)
        lattvect = np.array([2, -1, 3])
        for ind in crys.atomindices:
            b = crys.basis[ind[0]][ind[1]]
            pos = crys.lattice[:,0]*(lattvect[0] + b[0]) + \
                  crys.lattice[:,1]*(lattvect[1] + b[1]) + \
                  crys.lattice[:,2]*(lattvect[2] + b[2])
            self.assertTrue(np.all(np.isclose(pos, crys.pos2cart(lattvect, ind))))
        basis = [[np.array([0.,0.,0.]),
                  np.array([1./3.,2./3.,0.5]),
                  np.array([2./3.,1./3.,0.5])]]
        crys = crystal.Crystal(self.hexlatt, basis)
        for ind in crys.atomindices:
            b = crys.basis[ind[0]][ind[1]]
            pos = crys.lattice[:,0]*(lattvect[0] + b[0]) + \
                  crys.lattice[:,1]*(lattvect[1] + b[1]) + \
                  crys.lattice[:,2]*(lattvect[2] + b[2])
            self.assertTrue(np.all(np.isclose(pos, crys.pos2cart(lattvect, ind))))

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
        self.assertTrue(np.all(np.isclose(trans, np.array([1./3.,-1./3.,-1./2.]))))
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

    def testfccgroupops_directions(self):
        """Test out that we can apply group operations to directions"""
        crys = crystal.Crystal(self.fcclatt, self.basis)
        # 1. direction
        direc = np.array([2.,0.,0.])
        direc2 = np.dot(direc, direc)
        count = np.zeros(3, dtype=int)
        for g in crys.G:
            rotdirec = crys.g_direc(g, direc)
            self.assertAlmostEqual(np.dot(rotdirec, rotdirec), direc2)
            costheta = np.dot(rotdirec, direc)/direc2
            self.assertTrue(np.isclose(costheta, 1) or np.isclose(costheta, 0) or np.isclose(costheta, -1))
            count[int(round(costheta+1))] += 1
        self.assertEqual(count[0], 8) ## antiparallel
        self.assertEqual(count[1], 32) ## perpendicular
        self.assertEqual(count[2], 8) ## parallel

    def testomegagroupops_positions(self):
        """Test out that we can apply group operations to positions"""
        # 2. position = lattice vector + 2-tuple atom-index
        basis = [[np.array([0.,0.,0.]),
                  np.array([1./3.,2./3.,0.5]),
                  np.array([2./3.,1./3.,0.5])]]
        crys = crystal.Crystal(self.hexlatt, basis)
        lattvec = np.array([-2, 3, 1])
        for ind in crys.atomindices:
            pos = crys.pos2cart(lattvec, ind)
            for g in crys.G:
                # testing g_pos: (transform an atomic position)
                rotpos = crys.g_direc(g, pos)
                self.assertTrue(np.all(np.isclose(rotpos,
                                                  crys.pos2cart(*crys.g_pos(g, lattvec, ind)))))
                # testing g_vect: (transform a vector position in the crystal)
                rotlatt, rotind = crys.g_pos(g, lattvec, ind)
                rotlatt2, u = crys.g_vect(g, lattvec, crys.basis[ind[0]][ind[1]])
                self.assertTrue(np.all(np.isclose(rotpos, crys.unit2cart(rotlatt2, u))))
                self.assertTrue(np.all(rotlatt == rotlatt2))
                self.assertTrue(np.all(np.isclose(u, crys.basis[rotind[0]][rotind[1]])))

            # test point group operations:
            for g in crys.pointG[ind[0]][ind[1]]:
                origin = np.zeros(3, dtype=int)
                rotlatt, rotind = crys.g_pos(g, origin, ind)
                self.assertTrue(np.all(rotlatt == origin))
                self.assertEqual(rotind, ind)

    def testinverspos(self):
        """Test the inverses of pos2cart and unit2cart"""
        basis = [[np.array([0.,0.,0.]),
                  np.array([1./3.,2./3.,0.5]),
                  np.array([2./3.,1./3.,0.5])]]
        crys = crystal.Crystal(self.hexlatt, basis)
        lattvec = np.array([-2, 3, 1])
        for ind in crys.atomindices:
            lattback, uback = crys.cart2unit(crys.pos2cart(lattvec, ind))
            self.assertTrue(np.all(lattback == lattvec))
            self.assertTrue(np.all(np.isclose(uback, crys.basis[ind[0]][ind[1]])))
            lattback, indback = crys.cart2pos(crys.pos2cart(lattvec, ind))
            self.assertTrue(np.all(lattback == lattvec))
            self.assertEqual(indback,ind)
        lattback, indback = crys.cart2pos(np.array([0.25*self.a0, 0.25*self.a0, 0.]))
        self.assertIsNone(indback)

    def testWyckoff(self):
        """Test grouping for Wyckoff positions"""
        basis = [[np.array([0.,0.,0.]),
                  np.array([1./3.,2./3.,0.5]),
                  np.array([2./3.,1./3.,0.5])]]
        crys = crystal.Crystal(self.hexlatt, basis)
        # crys.Wyckoff : frozen set of frozen sets of tuples that are all equivalent
        Wyckoffind = {frozenset([(0,0)]),
                      frozenset([(0,1), (0,2)])}
        self.assertEqual(crys.Wyckoff, Wyckoffind)

    def testNNfcc(self):
        """Test of the nearest neighbor construction"""
        self.assertTrue(False)
