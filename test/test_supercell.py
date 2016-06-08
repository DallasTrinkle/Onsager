"""
Unit tests for supercell class
"""

__author__ = 'Dallas R. Trinkle'

import unittest
import numpy as np
import onsager.crystal as crystal
import onsager.supercell as supercell

class TypeTests(unittest.TestCase):
    """Tests to make sure we can make a supercell object."""
    longMessage = False

    def setUp(self):
        self.crys = crystal.Crystal.FCC(1.,'Al')
        self.one = np.eye(3, dtype=int)

    def testSuper(self):
        """Can we make a supercell object?"""
        super = supercell.Supercell(self.crys, self.one)
        self.assertNotEqual(super, None)
        self.assertEqual(super.Nchem, 2)
        super = supercell.Supercell(self.crys, self.one, interstitial = (1,))
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
        size,invsup,tlist = supercell.Supercell.maketrans(self.one)
        self.assertEqual(size,1)
        self.assertTrue(np.all(tlist[0]==0))
        size,invsup,tlist = supercell.Supercell.maketrans(2*self.one)
        self.assertEqual(size, 8)
        for tv in tlist:
            self.assertTrue(all(tvi ==0 or tvi == 4 for tvi in tv))
        super=np.array([[0,1,1],[1,0,1],[1,1,0]])
        size,invsup,tlist = supercell.Supercell.maketrans(super)
        self.assertEqual(size, 2)
        for tv in tlist:
            self.assertTrue(np.all(tv == 0) or np.all(tv == 1))
        # Try making a whole series of supercells; if they fail, will raise an Arithmetic exception:
        for n in range(100):
            randsuper = np.random.randint(-5,6,size=(3,3))
            if np.allclose(np.linalg.det(randsuper), 0): continue
            size,invsup,tlist = supercell.Supercell.maketrans(randsuper)
            self.assertTrue(len(tlist)==size)

    def testSites(self):
        """Do we have the correct sites in our supercell?"""
        super = supercell.Supercell(self.crys, self.one)
        self.assertTrue(np.allclose(super.pos[0], np.zeros(3)))

    def testGroupOps(self):
        """Do we correctly generate group operations inside the supercell?"""
        for nmat in (self.one, 2*self.one, np.array([[0,1,1],[1,0,1],[1,1,0]])):
            super = supercell.Supercell(self.crys, nmat)
            print(super)
            # self.assertEqual(len(super.G), len(self.crys.G)*super.size)





