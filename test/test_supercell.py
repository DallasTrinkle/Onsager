"""
Unit tests for supercell class
"""

__author__ = 'Dallas R. Trinkle'

import unittest
import itertools, copy
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
        sup = supercell.Supercell(self.crys, self.one)
        self.assertNotEqual(sup, None)
        self.assertEqual(sup.Nchem, self.crys.Nchem)
        sup = supercell.Supercell(self.crys, self.one, interstitial=(1,))
        self.assertNotEqual(sup, None)
        sup = supercell.Supercell(self.crys, self.one, Nsolute=5)
        self.assertNotEqual(sup, None)
        self.assertEqual(sup.Nchem, self.crys.Nchem + 5)
        with self.assertRaises(ZeroDivisionError):
            supercell.Supercell(self.crys, np.zeros((3, 3), dtype=int))

    def testEqualityCopy(self):
        """Can we copy a supercell, and is it equal to itself?"""
        super0 = supercell.Supercell(self.crys, self.one)
        super2 = super0.copy()
        self.assertOrderingSuperEqual(super0, super2, msg="copy not equal")

    def testTrans(self):
        """Can we correctly generates the translations?"""
        size, invsup, tlist, tdict = supercell.Supercell.maketrans(self.one)
        self.assertEqual(size, 1)
        self.assertTrue(np.all(tlist[0] == 0))
        size, invsup, tlist, tdict = supercell.Supercell.maketrans(2 * self.one)
        self.assertEqual(size, 8)
        for tv in tlist:
            self.assertTrue(all(tvi == 0 or tvi == 4 for tvi in tv))
        sup = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        size, invsup, tlist, tdict = supercell.Supercell.maketrans(sup)
        self.assertEqual(size, 2)
        for tv in tlist:
            self.assertTrue(np.all(tv == 0) or np.all(tv == 1))
        # Try making a whole series of supercells; if they fail, will raise an Arithmetic exception:
        for n in range(100):
            randsuper = np.random.randint(-5, 6, size=(3, 3))
            if np.allclose(np.linalg.det(randsuper), 0):
                with self.assertRaises(ZeroDivisionError):
                    supercell.Supercell.maketrans(randsuper)
                continue
            size, invsup, tlist, tdict = supercell.Supercell.maketrans(randsuper)
            self.assertTrue(len(tlist) == size)

    def testSites(self):
        """Do we have the correct sites in our supercell?"""
        for n in range(100):
            randsuper = np.random.randint(-5, 6, size=(3, 3))
            if np.allclose(np.linalg.det(randsuper), 0): continue
            # for efficiency we don't bother generating group ops,
            # and also to avoid warnings about broken symmetry
            sup = supercell.Supercell(self.crys, randsuper, NOSYM=True)
            Rdictset = {ci: set() for ci in self.crys.atomindices}
            for u in sup.pos:
                x = np.dot(self.crys.lattice, np.dot(randsuper, u))
                R, ci = self.crys.cart2pos(x)
                self.assertNotEqual(ci, None)
                Rtup = tuple(R)
                self.assertNotIn(Rtup, Rdictset[ci])
                Rdictset[ci].add(Rtup)
            for v in Rdictset.values():
                self.assertEqual(len(v), sup.size)

    def testGroupOps(self):
        """Do we correctly generate group operations inside the supercell?"""
        for nmat in self.groupsupers:
            sup = supercell.Supercell(self.crys, nmat)
            # print(super)
            # for g in super.G: if np.all(g.rot==self.one): print(g)
            self.assertEqual(len(sup.G), len(self.crys.G) * sup.size)
            invlatt = np.linalg.inv(sup.lattice)
            superposcart = [np.dot(sup.lattice, u) for u in sup.pos]
            for g in sup.G:
                for i, x, u in zip(itertools.count(), superposcart, sup.pos):
                    gx = np.dot(g.cartrot, x) + np.dot(sup.lattice, g.trans)
                    gu = crystal.incell(np.dot(invlatt, gx))
                    gu0 = crystal.incell(np.dot(g.rot, u) + g.trans)
                    gi = g.indexmap[0][i]
                    if not np.allclose(gu, gu0):
                        self.assertTrue(np.allclose(gu, gu0),
                                        msg="{}\nProblem with GroupOp:\n{}\n{} != {}".format(sup, g, gu, gu0))
                    if not np.allclose(gu, sup.pos[gi]):
                        self.assertTrue(np.allclose(gu, sup.pos[gi]),
                                        msg="{}\nProblem with GroupOp:\n{}\nIndexing: {} != {}".format(sup, g, gu,
                                                                                                       sup.pos[gi]))
        # do we successfully raise a Warning about broken symmetry?
        with self.assertWarns(RuntimeWarning):
            brokensymmsuper = np.array([[3, -5, 2], [-1, 2, 3], [4, -2, 1]])
            supercell.Supercell(self.crys, brokensymmsuper)

    def testSanity(self):
        """Does __sane__ operate as it should?"""
        # we use NOSYM for speed only
        sup = supercell.Supercell(self.crys, 3 * self.one, Nsolute=1, NOSYM=True)
        self.assertTrue(sup.__sane__(), msg='Empty supercell not sane?')
        # do a bunch of random operations, make sure we remain sane:
        Ntests = 100
        for c, ind in zip(np.random.randint(-1, sup.Nchem, size=Ntests),
                          np.random.randint(sup.size * sup.N, size=Ntests)):
            sup.setocc(ind, c)
            if not sup.__sane__():
                self.assertTrue(False, msg='Supercell:\n{}\nnot sane?'.format(sup))
        # Now! Break sanity (and then repair it)
        for c, ind in zip(np.random.randint(-1, sup.Nchem, size=Ntests),
                          np.random.randint(sup.size * sup.N, size=Ntests)):
            c0 = sup.occ[ind]
            if c == c0: continue
            sup.occ[ind] = c
            self.assertFalse(sup.__sane__())
            sup.occ[ind] = c0
            if not sup.__sane__():
                self.assertTrue(False, msg='Supercell:\n{}\nnot sane?'.format(sup))

    def testIndex(self):
        """Test that we can use index into our supercell appropriately"""
        for n in range(10):
            randsuper = np.random.randint(-5, 6, size=(3, 3))
            if np.allclose(np.linalg.det(randsuper), 0): continue
            # for efficiency we don't bother generating group ops,
            # and also to avoid warnings about broken symmetry
            sup = supercell.Supercell(self.crys, randsuper, NOSYM=True)
            for ind, u in enumerate(sup.pos):
                self.assertEqual(ind, sup.index(u))
                delta = np.random.uniform(-0.01, 0.01, size=3)
                self.assertEqual(ind, sup.index(crystal.incell(u + delta)))
            # test out setting by making a copy "by hand"
            randcopy = sup.copy()  # starts out empty, too.
            Ntests = 30
            for c, ind in zip(np.random.randint(-1, sup.Nchem, size=Ntests),
                              np.random.randint(sup.size * sup.N, size=Ntests)):
                sup.setocc(ind, c)
            for c, poslist in enumerate(sup.occposlist()):
                for pos in poslist:
                    randcopy[pos] = c
            self.assertOrderingSuperEqual(sup, randcopy, msg='Indexing fail?')

    def testMultiply(self):
        """Can we multiply a supercell by our group operations successfully?"""
        sup = supercell.Supercell(self.crys, 3 * self.one, Nsolute=1)
        # set up some random occupancy
        Ntests = 100
        for c, ind in zip(np.random.randint(-1, sup.Nchem, size=Ntests),
                          np.random.randint(sup.size * sup.N, size=Ntests)):
            sup.setocc(ind, c)
        g_occ = sup.occ.copy()
        for g in sup.G:
            gsuper = g * sup
            if not gsuper.__sane__():
                self.assertTrue(False, msg='GroupOp:\n{}\nbreaks sanity?'.format(g))
            # because it's sane, we *only* need to test that occupation is correct
            # indexmap[0]: each entry is the index where it "lands"
            for n in range(sup.size * sup.N):
                g_occ[g.indexmap[0][n]] = sup.occ[n]
            self.assertTrue(np.all(g_occ == gsuper.occ))
            # rotate a few sites, see if they match up:
            for ind in np.random.randint(sup.size * sup.N, size=Ntests // 10):
                gu = crystal.incell(np.dot(g.rot, sup.pos[ind]) + g.trans)
                self.assertIsInstance(gu, np.ndarray)
                self.assertOrderingSuperEqual(gsuper[gu], sup[ind], msg='Group operation fail?')
        # quick test of multiplying the other direction, and in-place (which should all call the same code)
        self.assertOrderingSuperEqual(gsuper, sup * g, msg='Other rotation fail?')
        sup *= g
        self.assertOrderingSuperEqual(gsuper, sup, msg='In place rotation fail?')

    def testReorder(self):
        """Can we reorder a supercell?"""
        sup = supercell.Supercell(self.crys, 3 * self.one, Nsolute=1)
        sup.definesolute(sup.Nchem - 1, 's')
        # set up some random occupancy
        Ntests = 100
        for c, ind in zip(np.random.randint(-1, sup.Nchem, size=Ntests),
                          np.random.randint(sup.size * sup.N, size=Ntests)):
            sup.setocc(ind, c)
        # Try some simple reorderings: 1. unity permutation; 2. pop+push; 3. reversal
        supercopy = sup.copy()
        unitymap = [[i for i in range(len(clist))] for clist in sup.chemorder]
        supercopy.reorder(unitymap)
        self.assertOrderingSuperEqual(sup, supercopy, msg='Reordering fail with unity?')
        popmap = []
        for c, clist in enumerate(sup.chemorder):
            n = len(clist)
            popmap.append([(i + 1) % n for i in range(n)])
            indpoppush = clist[0]
            sup.setocc(indpoppush, -1)
            sup.setocc(indpoppush, c)  # *now* should be at the *end* of the chemorder list
        supercopy.reorder(popmap)
        self.assertOrderingSuperEqual(sup, supercopy, msg='Reordering fail with "pop/push"?')
        revmap = []
        for c, clist in enumerate(sup.chemorder):
            n = len(clist)
            revmap.append([(n - 1 - i) for i in range(n)])
            cl = clist.copy()  # need to be careful, since clist gets modified by our popping...
            for indpoppush in cl:
                sup.setocc(indpoppush, -1)
            cl.reverse()
            for indpoppush in cl:
                sup.setocc(indpoppush, c)
        supercopy.reorder(revmap)
        self.assertOrderingSuperEqual(sup, supercopy, msg='Reordering fail with reverse?')
        # test out a bad mapping:
        badmap = [[i % 2 for i in range(len(clist))] for clist in sup.chemorder]
        with self.assertRaises(ValueError):
            supercopy.reorder(badmap)
        self.assertOrderingSuperEqual(sup, supercopy, msg='Reordering is not safe after fail?')

    def testEquivalenceMap(self):
        """Can we construct an equivalence map between two supercells?"""
        sup = supercell.Supercell(self.crys, 3 * self.one, Nsolute=1)
        sup.definesolute(sup.Nchem - 1, 's')
        # set up some random occupancy
        Ntests = 100
        for c, ind in zip(np.random.randint(-1, sup.Nchem, size=Ntests),
                          np.random.randint(sup.size * sup.N, size=Ntests)):
            sup.setocc(ind, c)
        supercopy = sup.copy()
        # first equivalence test: introduce some random permutations of ordering of supercell
        for ind in np.random.randint(sup.size * sup.N, size=Ntests):
            c, sup[ind] = sup[ind], -1
            sup[ind] = c
        g, mapping = supercopy.equivalencemap(sup)
        self.assertNotEqual(g, None, msg='Cannot map between permutation?')
        supercopy.reorder(mapping)
        self.assertOrderingSuperEqual(sup, supercopy, msg='Improper map from random permutation')
        # apply all of the group operations, and see how they perform:
        for g in sup.G:
            gsuper = g * sup
            for ind in np.random.randint(sup.size * sup.N, size=Ntests):
                c, gsuper[ind] = gsuper[ind], -1
                gsuper[ind] = c
            g0, mapping = supercopy.equivalencemap(gsuper)
            if g != g0:
                msg = 'Group operations not equal?\n'
                for line0, line1 in itertools.zip_longest(g.__str__().splitlines(),
                                                          g0.__str__().splitlines(),
                                                          fillvalue=' - '):
                    msg += line0 + '\t' + line1 + '\n'
                self.fail(msg=msg)
            self.assertOrderingSuperEqual((g0 * supercopy).reorder(mapping), gsuper,
                                          msg='Group operation + mapping failure?')
            # do the testing with occposlist, since that's what this is really for...
            rotoccposlist = [[crystal.incell(np.dot(g0.rot, pos) + g0.trans) for pos in poslist]
                             for poslist in supercopy.occposlist()]
            # now, reorder:
            reorderoccposlist = copy.deepcopy(rotoccposlist)
            for reposlist, poslist, remap in zip(reorderoccposlist, rotoccposlist, mapping):
                for i, m in enumerate(remap):
                    reposlist[i] = poslist[m]
            for reposlist, gposlist in zip(reorderoccposlist, gsuper.occposlist()):
                for repos, gpos in zip(reposlist, gposlist):
                    self.assertTrue(np.allclose(repos, gpos),
                                    msg='Reordering the unit cell position failed?')

        # now try something that *shouldn't* be equivalent:
        for ind in np.random.randint(sup.size * sup.N, size=Ntests):
            # a chemical "permutation":
            sup[ind] = (sup[ind] + 2) % (sup.Nchem + 1) - 1
        g, mapping = supercopy.equivalencemap(sup)
        self.assertEqual(g, None, msg='Found a mapping where one should not exist?')


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
        sup = supercell.Supercell(self.crys, 3 * self.one, interstitial=[1])
        sup.fillperiodic((0, 0))
        self.assertEqual(len(sup.chemorder[0]), sup.size * len(self.crys.basis[0]))
        for n in range(1, sup.Nchem):
            self.assertEqual(len(sup.chemorder[n]), 0)
        for ind in range(sup.size * sup.N):
            if ind in sup.chemorder[0]:
                self.assertEqual(sup.occ[ind], 0)  # occupied with correct chemistry
            else:
                self.assertEqual(sup.occ[ind], -1)  # vacancy
        for ci in next((wset for wset in self.crys.Wyckoff if (0, 0) in wset), None):
            i = self.crys.atomindices.index(ci)
            for n in range(sup.size):
                self.assertIn(n * sup.N + i, sup.chemorder[0])

    def testIndexExceptions(self):
        """Do we get raise indexing errors as appropriate?"""
        sup = supercell.Supercell(self.crys, 3 * self.one, interstitial=[1])
        with self.assertRaises(IndexError):
            sup.definesolute(0, 'Mg')
        with self.assertRaises(IndexError):
            sup.definesolute(-1, 'Mg')
        with self.assertRaises(IndexError):
            sup.definesolute(2, 'Mg')
        with self.assertRaises(IndexError):
            sup.setocc(0, 2)
        with self.assertRaises(IndexError):
            sup.setocc(sup.size * sup.N, 0)
        # and... all of the following should be *safe* operations:
        sup.setocc(0, 0)
        sup.setocc(-1, 0)
        sup = supercell.Supercell(self.crys, self.one, interstitial=[1], Nsolute=1)
        sup.definesolute(2, 'Mg')

    def testYAML(self):
        """Can we read/write YAML representation of a supercell?"""
        sup = supercell.Supercell(self.crys, 3 * self.one, interstitial=[1])
        YAMLstring = crystal.yaml.dump(sup)
        superYAML = crystal.yaml.load(YAMLstring)
        self.assertOrderingSuperEqual(sup, superYAML, msg='YAML write/read fail?')
        # print(YAMLstring)
