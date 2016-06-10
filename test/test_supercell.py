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

    def assertOrderingSuperEqual(self, s0, s1, msg=""):
        if s0 != s1:
            failmsg = msg + '\n'
            for line0, line1 in itertools.zip_longest(s0.__str__().splitlines(),
                                                      s1.__str__().splitlines(),
                                                      fillvalue=' - '):
                failmsg += line0 + '\t' + line1 + '\n'
            self.fail(msg=failmsg)

    def testSuper(self):
        """Can we make a supercell object?"""
        super = supercell.Supercell(self.crys, self.one)
        self.assertNotEqual(super, None)
        self.assertEqual(super.Nchem, self.crys.Nchem)
        super = supercell.Supercell(self.crys, self.one, interstitial=(1,))
        self.assertNotEqual(super, None)
        super = supercell.Supercell(self.crys, self.one, Nsolute=5)
        self.assertNotEqual(super, None)
        self.assertEqual(super.Nchem, self.crys.Nchem+5)

    def testEqualityCopy(self):
        """Can we copy a supercell, and is it equal to itself?"""
        super = supercell.Supercell(self.crys, self.one)
        super2 = super.copy()
        self.assertOrderingSuperEqual(super, super2, msg="copy not equal")

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

    def testSanity(self):
        """Does __sane__ operate as it should?"""
        # we use NOSYM for speed only
        super = supercell.Supercell(self.crys, 3*self.one, Nsolute=1, NOSYM=True)
        self.assertTrue(super.__sane__(), msg='Empty supercell not sane?')
        # do a bunch of random operations, make sure we remain sane:
        Ntests = 100
        for c,ind in zip(np.random.randint(-1, super.Nchem, size=Ntests),
                         np.random.randint(super.size*super.N, size=Ntests)):
            super.setocc(ind, c)
            if not super.__sane__():
                self.assertTrue(False, msg='Supercell:\n{}\nnot sane?'.format(super))
        # Now! Break sanity (and then repair it)
        for c, ind in zip(np.random.randint(-1, super.Nchem, size=Ntests),
                          np.random.randint(super.size * super.N, size=Ntests)):
            c0 = super.occ[ind]
            if c==c0: continue
            super.occ[ind] = c
            self.assertFalse(super.__sane__())
            super.occ[ind] = c0
            if not super.__sane__():
                self.assertTrue(False, msg='Supercell:\n{}\nnot sane?'.format(super))

    def testIndex(self):
        """Test that we can use index into our supercell appropriately"""
        for n in range(10):
            randsuper = np.random.randint(-5, 6, size=(3, 3))
            if np.allclose(np.linalg.det(randsuper), 0): continue
            # for efficiency we don't bother generating group ops,
            # and also to avoid warnings about broken symmetry
            super = supercell.Supercell(self.crys, randsuper, NOSYM=True)
            for ind, u in enumerate(super.pos):
                self.assertEqual(ind, super.index(u))
                delta = np.random.uniform(-0.01,0.01,size=3)
                self.assertEqual(ind, super.index(crystal.incell(u+delta)))
            # test out setting by making a copy "by hand"
            randcopy = super.copy()  # starts out empty, too.
            Ntests = 30
            for c, ind in zip(np.random.randint(-1, super.Nchem, size=Ntests),
                              np.random.randint(super.size * super.N, size=Ntests)):
                super.setocc(ind, c)
            for c,poslist in enumerate(super.occposlist()):
                for pos in poslist:
                    randcopy[pos] = c
            self.assertOrderingSuperEqual(super, randcopy, msg='Indexing fail?')

    def testMultiply(self):
        """Can we multiply a supercell by our group operations successfully?"""
        super = supercell.Supercell(self.crys, 3 * self.one, Nsolute=1)
        # set up some random occupancy
        Ntests = 100
        for c, ind in zip(np.random.randint(-1, super.Nchem, size=Ntests),
                          np.random.randint(super.size * super.N, size=Ntests)):
            super.setocc(ind, c)
        g_occ = super.occ.copy()
        for g in super.G:
            gsuper = g*super
            if not gsuper.__sane__():
                self.assertTrue(False, msg='GroupOp:\n{}\nbreaks sanity?'.format(g))
            # because it's sane, we *only* need to test that occupation is correct
            # indexmap[0]: each entry is the index where it "lands"
            for n in range(super.size*super.N):
                g_occ[g.indexmap[0][n]] = super.occ[n]
            self.assertTrue(np.all(g_occ == gsuper.occ))
            # rotate a few sites, see if they match up:
            for ind in np.random.randint(super.size*super.N, size=Ntests//10):
                gu = crystal.incell(np.dot(g.rot, super.pos[ind]) + g.trans)
                self.assertIsInstance(gu, np.ndarray)
                self.assertOrderingSuperEqual(gsuper[gu], super[ind], msg='Group operation fail?')
        # quick test of multiplying the other direction, and in-place (which should all call the same code)
        self.assertOrderingSuperEqual(gsuper, super*g, msg='Other rotation fail?')
        super *= g
        self.assertOrderingSuperEqual(gsuper, super, msg='In place rotation fail?')

    def testReorder(self):
        """Can we reorder a supercell?"""
        super = supercell.Supercell(self.crys, 3 * self.one, Nsolute=1)
        super.definesolute(super.Nchem-1, 's')
        # set up some random occupancy
        Ntests = 100
        for c, ind in zip(np.random.randint(-1, super.Nchem, size=Ntests),
                          np.random.randint(super.size * super.N, size=Ntests)):
            super.setocc(ind, c)
        # Try some simple reorderings: 1. unity permutation; 2. pop+push; 3. reversal
        supercopy = super.copy()
        unitymap = [[i for i in range(len(clist))] for clist in super.chemorder]
        supercopy.reorder(unitymap)
        self.assertOrderingSuperEqual(super, supercopy, msg='Reordering fail with unity?')
        popmap = []
        for c, clist in enumerate(super.chemorder):
            n = len(clist)
            popmap.append([(i+1)%n for i in range(n)])
            indpoppush = clist[0]
            super.setocc(indpoppush, -1)
            super.setocc(indpoppush, c)  # *now* should be at the *end* of the chemorder list
        supercopy.reorder(popmap)
        self.assertOrderingSuperEqual(super, supercopy, msg='Reordering fail with "pop/push"?')
        revmap = []
        for c, clist in enumerate(super.chemorder):
            n = len(clist)
            revmap.append([(n-1-i) for i in range(n)])
            cl = clist.copy()  # need to be careful, since clist gets modified by our popping...
            for indpoppush in cl:
                super.setocc(indpoppush, -1)
            cl.reverse()
            for indpoppush in cl:
                super.setocc(indpoppush, c)
        supercopy.reorder(revmap)
        self.assertOrderingSuperEqual(super, supercopy, msg='Reordering fail with reverse?')
        # test out a bad mapping:
        badmap = [[i%2 for i in range(len(clist))] for clist in super.chemorder]
        with self.assertRaises(ValueError):
            supercopy.reorder(badmap)
        self.assertOrderingSuperEqual(super, supercopy, msg='Reordering is not safe after fail?')


    def testEquivalenceMap(self):
        """Can we construct an equivalence map between two supercells?"""
        super = supercell.Supercell(self.crys, 3 * self.one, Nsolute=1)
        super.definesolute(super.Nchem-1, 's')
        # set up some random occupancy
        Ntests = 100
        for c, ind in zip(np.random.randint(-1, super.Nchem, size=Ntests),
                          np.random.randint(super.size * super.N, size=Ntests)):
            super.setocc(ind, c)
        supercopy = super.copy()
        # first equivalence test: introduce some random permutations of ordering of supercell
        for ind in np.random.randint(super.size*super.N, size=Ntests):
            c, super[ind] = super[ind], -1
            super[ind] = c
        g, mapping = supercopy.equivalencemap(super)
        self.assertNotEqual(g, None, msg='Cannot map between permutation?')
        supercopy.reorder(mapping)
        self.assertOrderingSuperEqual(super, supercopy, msg='Improper map from random permutation')
        # apply all of the group operations, and see how they perform:
        for g in super.G:
            gsuper = g*super
            for ind in np.random.randint(super.size * super.N, size=Ntests):
                c, gsuper[ind] = gsuper[ind], -1
                gsuper[ind] = c
            g0, mapping = supercopy.equivalencemap(gsuper)
            if g!=g0:
                msg='Group operations not equal?\n'
                for line0, line1 in itertools.zip_longest(g.__str__().splitlines(),
                                                          g0.__str__().splitlines(),
                                                          fillvalue=' - '):
                      msg += line0 + '\t' + line1 + '\n'
                self.fail(msg=msg)
            self.assertOrderingSuperEqual((g0*supercopy).reorder(mapping), gsuper,
                                          msg='Group operation + mapping failure?')

class HCPSuperTests(FCCSuperTests):
    """Tests to make sure we can make a supercell object: based on HCP"""
    longMessage = False

    def setUp(self):
        self.crys = crystal.Crystal.HCP(1., chemistry='Ti')
        self.one = np.eye(3, dtype=int)
        self.groupsupers = (self.one, 2 * self.one, 3 * self.one,
                            np.array([[2, 0, 0], [0, 2, 0], [0, 0, 3]]),
                            np.array([[1, 1, 0], [0, 1, 0], [0, 0, 2]]))


class InterstitialSuperTests(HCPSuperTests):
    """Tests to make sure we can make a supercell object: HCP + interstitials"""
    longMessage = False

    def setUp(self):
        crys = crystal.Crystal.HCP(1., chemistry='Ti')
        self.crys = crys.addbasis(crys.Wyckoffpos(np.array([0., 0., 0.5])), chemistry=['O'])
        self.one = np.eye(3, dtype=int)
        self.groupsupers = (self.one, 2 * self.one,
                            np.array([[2, 0, 0], [0, 2, 0], [0, 0, 3]]),
                            np.array([[1, 1, 0], [0, 1, 0], [0, 0, 2]]))

    def testFillPeriodic(self):
        """Can we fill up our periodic cell?"""
        super = supercell.Supercell(self.crys, 3*self.one, interstitial=[1])
        super.fillperiodic((0,0))
        self.assertEqual(len(super.chemorder[0]), super.size*len(self.crys.basis[0]))
        for n in range(1,super.Nchem):
            self.assertEqual(len(super.chemorder[n]), 0)
        for ind in range(super.size*super.N):
            if ind in super.chemorder[0]:
                self.assertEqual(super.occ[ind], 0)  # occupied with correct chemistry
            else:
                self.assertEqual(super.occ[ind], -1)  # vacancy
        for ci in next((wset for wset in self.crys.Wyckoff if (0,0) in wset), None):
            i = self.crys.atomindices.index(ci)
            for n in range(super.size):
                self.assertIn(n*super.N+i, super.chemorder[0])

    def testIndexExceptions(self):
        """Do we get raise indexing errors as appropriate?"""
        super = supercell.Supercell(self.crys, 3*self.one, interstitial=[1])
        with self.assertRaises(IndexError):
            super.definesolute(0, 'Mg')
        with self.assertRaises(IndexError):
            super.definesolute(-1, 'Mg')
        with self.assertRaises(IndexError):
            super.definesolute(2, 'Mg')
        with self.assertRaises(IndexError):
            super.setocc(0, 2)
        with self.assertRaises(IndexError):
            super.setocc(super.size*super.N, 0)
        # and... all of the following should be *safe* operations:
        super.setocc(0,0)
        super.setocc(-1,0)
        super = supercell.Supercell(self.crys, self.one, interstitial=[1], Nsolute=1)
        super.definesolute(2, 'Mg')

    def testYAML(self):
        """Can we read/write YAML representation of a supercell?"""
        super = supercell.Supercell(self.crys, 3*self.one, interstitial=[1])
        YAMLstring = crystal.yaml.dump(super)
        superYAML = crystal.yaml.load(YAMLstring)
        self.assertOrderingSuperEqual(super, superYAML, msg='YAML write/read fail?')
        # print(YAMLstring)


