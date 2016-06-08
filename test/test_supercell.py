"""
Unit tests for supercell class
"""

__author__ = 'Dallas R. Trinkle'

import unittest
import itertools
import numpy as np
import onsager.crystal as crystal
import onsager.supercell as supercell


class FCCSuperTests(unittest.TestCase):
    """Tests to make sure we can make a supercell object."""
    longMessage = False

    def setUp(self):
        self.crys = crystal.Crystal.FCC(1., 'Al')
        self.one = np.eye(3, dtype=int)
        self.groupsupers = (self.one, 2 * self.one, np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]))

    def testSuper(self):
        """Can we make a supercell object?"""
        super = supercell.Supercell(self.crys, self.one)
        self.assertNotEqual(super, None)
        self.assertEqual(super.Nchem, 2)
        super = supercell.Supercell(self.crys, self.one, interstitial=(1,))
        self.assertNotEqual(super, None)
        super = supercell.Supercell(self.crys, self.one, Nchem=5)
        self.assertNotEqual(super, None)
        self.assertEqual(super.Nchem, 5)

    def testEqualityCopy(self):
        """Can we copy a supercell, and is it equal to itself?"""
        super = supercell.Supercell(self.crys, self.one)
        super2 = super.copy()
        self.assertEqual(super, super2, msg="{}\n!=\n{}".format(super, super2))

    def testTrans(self):
        """Can we correctly generates the translations?"""
        size, invsup, tlist, tdict = supercell.Supercell.maketrans(self.one)
        self.assertEqual(size, 1)
        self.assertTrue(np.all(tlist[0] == 0))
        size, invsup, tlist, tdict = supercell.Supercell.maketrans(2 * self.one)
        self.assertEqual(size, 8)
        for tv in tlist:
            self.assertTrue(all(tvi == 0 or tvi == 4 for tvi in tv))
        super = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        size, invsup, tlist, tdict = supercell.Supercell.maketrans(super)
        self.assertEqual(size, 2)
        for tv in tlist:
            self.assertTrue(np.all(tv == 0) or np.all(tv == 1))
        # Try making a whole series of supercells; if they fail, will raise an Arithmetic exception:
        for n in range(100):
            randsuper = np.random.randint(-5, 6, size=(3, 3))
            if np.allclose(np.linalg.det(randsuper), 0): continue
            size, invsup, tlist, tdict = supercell.Supercell.maketrans(randsuper)
            self.assertTrue(len(tlist) == size)

    def testSites(self):
        """Do we have the correct sites in our supercell?"""
        for n in range(100):
            randsuper = np.random.randint(-5, 6, size=(3, 3))
            if np.allclose(np.linalg.det(randsuper), 0): continue
            # for efficiency we don't bother generating group ops,
            # and also to avoid warnings about broken symmetry
            super = supercell.Supercell(self.crys, randsuper, NOSYM=True)
            Rdictset = {ci: set() for ci in self.crys.atomindices}
            for u in super.pos:
                x = np.dot(self.crys.lattice, np.dot(randsuper, u))
                R, ci = self.crys.cart2pos(x)
                self.assertNotEqual(ci, None)
                Rtup = tuple(R)
                self.assertNotIn(Rtup, Rdictset[ci])
                Rdictset[ci].add(Rtup)
            for v in Rdictset.values():
                self.assertEqual(len(v), super.size)

    def testGroupOps(self):
        """Do we correctly generate group operations inside the supercell?"""
        for nmat in self.groupsupers:
            super = supercell.Supercell(self.crys, nmat)
            # print(super)
            # for g in super.G: if np.all(g.rot==self.one): print(g)
            self.assertEqual(len(super.G), len(self.crys.G) * super.size)
            invlatt = np.linalg.inv(super.lattice)
            superposcart = [np.dot(super.lattice, u) for u in super.pos]
            for g in super.G:
                for i,x,u in zip(itertools.count(), superposcart, super.pos):
                    gx = np.dot(g.cartrot, x) + np.dot(super.lattice, g.trans)
                    gu = crystal.incell(np.dot(invlatt,gx))
                    gu0 = crystal.incell(np.dot(g.rot, u) + g.trans)
                    gi = g.indexmap[0][i]
                    if not np.allclose(gu,gu0):
                        self.assertTrue(np.allclose(gu, gu0),
                                        msg="{}\nProblem with GroupOp:\n{}\n{} != {}".format(super, g, gu, gu0))
                    if not np.allclose(gu,super.pos[gi]):
                        self.assertTrue(np.allclose(gu, super.pos[gi]),
                                        msg="{}\nProblem with GroupOp:\n{}\nIndexing: {} != {}".format(super, g, gu, super.pos[gi]))


class HCPSuperTests(FCCSuperTests):
    """Tests to make sure we can make a supercell object: based on HCP"""
    longMessage = False

    def setUp(self):
        self.crys = crystal.Crystal.HCP(1., chemistry='Ti')
        self.one = np.eye(3, dtype=int)
        self.groupsupers = (self.one, 2 * self.one, 3 * self.one, np.array([[2, 0, 0], [0, 2, 0], [0, 0, 3]]))
