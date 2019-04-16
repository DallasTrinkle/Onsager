"""
Unit tests for supercell class
"""

__author__ = 'Dallas R. Trinkle'

import unittest
import itertools, copy, textwrap
import numpy as np
from onsager import crystal, supercell, cluster


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
            # print(superlatt)
            # for g in superlatt.G: if np.all(g.rot==self.one): print(g)
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


class SupercellParsing(unittest.TestCase):
    """Tests to make sure we can read a POSCAR into a Supercell object, and manipulate."""
    longMessage = False

    def setUp(self):
        self.crys = crystal.Crystal.FCC(1., 'Al')
        self.one = np.eye(3, dtype=int)

    def testReadPOSCAR(self):
        """Can we read what we write? (simple FCC)"""
        sup = supercell.Supercell(self.crys, 4 * self.one)
        sup2 = sup.copy()
        sup.fillperiodic((0,0))
        testname = 'test'
        POSCAR_str = sup.POSCAR(testname)
        name = sup2.POSCAR_occ(POSCAR_str)
        self.assertEqual(testname + ' {}({})'.format(self.crys.chemistry[0], sup.N*sup.size), name)
        for n, occ, occ2 in zip(itertools.count(), sup.occ, sup2.occ):
            self.assertEqual(occ, occ2, msg='Failure at {}'.format(n))
        POSCAR_str2 = sup2.POSCAR(testname)
        self.assertEqual(POSCAR_str, POSCAR_str2)

    def testReadB2POSCAR(self):
        """Can we read what we write? (B2)"""
        B2 = crystal.Crystal(np.eye(3), [[np.array([0.,0.,0.])], [np.array([0.5,0.5,0.5])]], ['A', 'B'])
        nsuper = 4*self.one
        sup = supercell.Supercell(B2, nsuper, Nsolute=1)
        sup.definesolute(2, 'C')
        # make up some random occupancies, and see how we do...
        sup2 = sup.copy()
        Ntests = sup.size*sup.N
        for _ in range(16):
            for c, ind in zip(np.random.randint(-1, sup.Nchem, size=Ntests),
                              np.random.randint(sup.size * sup.N, size=Ntests)):
                sup.setocc(ind, c)
            sup2.POSCAR_occ(sup.POSCAR())
            for n, occ, occ2 in zip(itertools.count(), sup.occ, sup2.occ):
                self.assertEqual(occ, occ2, msg='Failure at {}'.format(n))

    def testReadPOSCAR_hand(self):
        """Can we read a simple example POSCAR?"""
        nsuper = np.array([[-1,1,1], [1,-1,1], [1,1,-1]]) # 4 atom cubic unit cell, FCC
        poslist = [np.array([0., 0., 0.]), np.array([0.5, 0.5, 0.]), np.array([0.0, 0.5, 0.5])]
        vaclist = [np.array([0.5, 0., 0.5])]
        POSCAR_str = textwrap.dedent("""\
        Test supercell (written by hand!)
        1.0
        1.0 0.0 0.0
        0.0 1.0 0.0
        0.0 0.0 1.0
        3
        Direct
        """)
        for pos in poslist:
            POSCAR_str += "{} {} {}\n".format(pos[0], pos[1], pos[2])
        sup = supercell.Supercell(self.crys, nsuper)
        sup.POSCAR_occ(POSCAR_str, disp_threshold=1e-2, latt_threshold=1e-2)
        for pos in poslist:
            self.assertEqual(0, sup[pos])
        for vac in vaclist:
            self.assertEqual(-1, sup[vac])

    def test_chemmapping(self):
        """Does the chemmapping function behave logically?"""
        B2 = crystal.Crystal(np.eye(3), [[np.array([0.,0.,0.])], [np.array([0.5,0.5,0.5])]], ['A', 'B'])
        nsuper = 1*self.one
        sup = supercell.Supercell(B2, nsuper, Nsolute=1)
        sup.definesolute(2, 'C')
        chemmapping = sup.defect_chemmapping()
        for csite in range(B2.Nchem):
            for cocc in range(-1, sup.Nchem):
                if csite == cocc:
                    self.assertEqual(0, chemmapping[csite][cocc])
                else:
                    self.assertEqual(1, chemmapping[csite][cocc])

    def testSuperIntoOccupancies(self):
        """Can we construct our mobile and spectator occupancy vectors? (B2)"""
        B2 = crystal.Crystal(np.eye(3), [[np.array([0.,0.,0.])], [np.array([0.5,0.5,0.5])]], ['A', 'B'])
        nsuper = 4*self.one
        sup = supercell.Supercell(B2, nsuper, Nsolute=1)
        sup.definesolute(2, 'C')
        sup_c = supercell.ClusterSupercell(B2, nsuper, spectator=(1,))
        # make up some random occupancies, and see how we do...
        Ntests = sup.size*sup.N
        for _ in range(16):
            for c, ind in zip(np.random.randint(-1, sup.Nchem, size=Ntests),
                              np.random.randint(sup.size * sup.N, size=Ntests)):
                sup.setocc(ind, c)
            mocc, socc = sup_c.Supercell_occ(sup)
            # Now... to check that it all makes sense.
            for ind, n in enumerate(mocc):
                csite = sup_c.ciR(ind, mobile=True)[0][0]
                pos = sup_c.mobilepos[ind]
                cocc = sup[pos]
                self.assertEqual(0 if csite==cocc else 1, n,
                                 msg="Mobile failure: {}, {} has occupancy {}".format(pos, csite, cocc))
            for ind, n in enumerate(socc):
                csite = sup_c.ciR(ind, mobile=False)[0][0]
                pos = sup_c.specpos[ind]
                cocc = sup[pos]
                self.assertEqual(0 if csite==cocc else 1, n,
                                 msg="Spectator failure: {}, {} has occupancy {}".format(pos, csite, cocc))

    def testThresholds(self):
        """Do our thresholds function as expected?"""
        crys_strained = self.crys.strain(crystal.Voigtstrain(0.1, 0., -0.1, 0., 0., 0.))
        sup = supercell.Supercell(self.crys, 2*self.one)
        sup2 = supercell.Supercell(crys_strained, 2*self.one)
        sup2.fillperiodic((0,0), 0)
        POSCAR_str = sup2.POSCAR()
        # first, the safe operation:
        sup.POSCAR_occ(POSCAR_str)
        # now, with our threshold, which should raise a ValueError
        with self.assertRaises(ValueError):
            sup.POSCAR_occ(POSCAR_str, latt_threshold=1e-2)
        # now, let's make some displacements:
        sup2.pos += np.random.uniform(-0.1, 0.1, size=(sup2.size, 3))
        POSCAR_str = sup2.POSCAR()
        # first, the safe operation:
        sup.POSCAR_occ(POSCAR_str)
        # now, with our threshold, which should raise a ValueError
        with self.assertRaises(ValueError):
            sup.POSCAR_occ(POSCAR_str, disp_threshold=1e-5)


class ClusterSupercellTests(unittest.TestCase):
    """Tests of the Supercell Cluster class"""
    longMessage = False

    def setUp(self):
        self.crys = crystal.Crystal.FCC(1., chemistry='FCC')
        self.one = np.eye(3, dtype=int)

    def testClusterSupercellCreation(self):
        """Can we make a cluster supercell?"""
        sup = supercell.ClusterSupercell(self.crys, 3*self.one)
        self.assertIsInstance(sup, supercell.ClusterSupercell)

    def testIndexingSimple(self):
        """Check that the indexing behaves as we expect: FCC"""
        superlatt = np.array([[3,2,0], [-2, 3, 1], [2, -1, 4]])
        # R1, R2, R3 = superlatt[:,0], superlatt[:,1], superlatt[:,2]
        sup = supercell.ClusterSupercell(self.crys, superlatt)
        for n, Rv in enumerate(sup.Rveclist):
            m, mob = sup.index(Rv, (0,0))
            self.assertTrue(mob)
            self.assertEqual(n, m,
                             msg='Failure to match {} in supercell'.format(Rv))
            for Rext in superlatt.T:
                m, mob = sup.index(Rv+Rext, (0, 0))
                self.assertTrue(mob)
                self.assertEqual(n, m,
                                 msg='Failure to match {} in supercell'.format(Rv+Rext))

    def testIndexingComplex(self):
        """Check that the indexing behaves as we expect: HCP"""
        Ti = crystal.Crystal.HCP(1., chemistry='Ti')
        TiO= Ti.addbasis(Ti.Wyckoffpos(np.array([0., 0., 0.5])), chemistry=['O'])
        superlatt = np.array([[3,2,0], [-2, 3, 1], [2, -1, 4]])
        superinv = np.linalg.inv(superlatt)
        # R1, R2, R3 = superlatt[:,0], superlatt[:,1], superlatt[:,2]
        sup = supercell.ClusterSupercell(TiO, superlatt, spectator=[0])
        for c, mob in zip([0, 1], [False, True]):
            for i in range(len(TiO.basis[c])):
                ci = (c,i)
                for n, Rv in enumerate(sup.Rveclist):
                    m, mobile = sup.index(Rv, ci)
                    self.assertEqual(mob, mobile)
                    # positions in unit cell coordinates of our supercell:
                    pos = np.dot(superinv, TiO.basis[c][i] + Rv)
                    poscompare = sup.mobilepos[m] if mobile else sup.specpos[m]
                    # are we within a lattice vector *for the supercell* for our position?
                    posdiff = pos - poscompare
                    posdiff -= np.round(posdiff)
                    self.assertTrue(np.allclose(posdiff, 0),
                                    msg='Failure to match {} {}:\nindex: {} and {}, {} != {}'.format(ci, Rv, n, m, pos, poscompare))

    def testIndexingReverse(self):
        """Check that the ciR evaluates to index properly: HCP"""
        Ti = crystal.Crystal.HCP(1., chemistry='Ti')
        TiO= Ti.addbasis(Ti.Wyckoffpos(np.array([0., 0., 0.5])), chemistry=['O'])
        superlatt = np.array([[3,2,0], [-2, 3, 1], [2, -1, 4]])
        sup = supercell.ClusterSupercell(TiO, superlatt, spectator=[0])
        for N, mob in zip((sup.Nmobile, sup.Nspec), (True, False)):
            for n in range(sup.size*N):
                ci, R = sup.ciR(n, mob)
                self.assertEqual((n, mob), sup.index(R, ci))

    def testClusterEvalSimple(self):
        """Check that we can evaluate a cluster expansion: FCC"""
        FCC = self.crys
        Nsuper = 4
        A1 = FCC.cart2unit(np.array([Nsuper,0.,0.]))[0]
        A2 = FCC.cart2unit(np.array([0.,Nsuper,0.]))[0]
        A3 = FCC.cart2unit(np.array([0.,0.,Nsuper]))[0]
        sup = supercell.ClusterSupercell(FCC, np.array([A1, A2, A3]))
        # build some sites...
        s1 = cluster.ClusterSite.fromcryscart(FCC, np.array([0, 0, 0]))
        s2 = cluster.ClusterSite.fromcryscart(FCC, np.array([0., 0.5, 0.5]))
        s3 = cluster.ClusterSite.fromcryscart(FCC, np.array([0.5, 0., 0.5]))
        s4 = cluster.ClusterSite.fromcryscart(FCC, np.array([0.5, 0.5, 0.]))
        s5 = cluster.ClusterSite.fromcryscart(FCC, np.array([0., 0.5, -0.5]))
        # build some base clusters...
        c1 = cluster.Cluster([s1])
        c2 = cluster.Cluster([s1, s2])
        c3 = cluster.Cluster([s1, s2, s3])
        c3w = cluster.Cluster([s1, s2, s5])
        c4 = cluster.Cluster([s1, s2, s3, s4])
        # expand out into symmetric sets...
        clusterexp = [set([cl.g(FCC, g) for g in FCC.G]) for cl in [c1, c2, c3, c3w, c4]]
        mocc, socc = np.zeros(sup.size), np.zeros(0)
        clustercount = sup.evalcluster(mocc, socc, clusterexp)
        self.assertTrue(np.all(np.array([0,]*5 + [sup.size]) == clustercount))

        mocc, socc = np.ones(sup.size), np.zeros(0)
        clustercount = sup.evalcluster(mocc, socc, clusterexp)
        self.assertTrue(np.all(np.array([1, 6, 8, 12, 2, 1])*sup.size == clustercount))

        socc = np.zeros(0)
        for u1, u2, u3 in itertools.product([-0.5/Nsuper, 0.5/Nsuper], repeat=3):
            mocc = np.zeros(sup.size)
            # make a tetrahedron...
            for x in [(0., 0., 0.), (0., u2, u3), (u1, 0, u3), (u1, u2, 0)]:
                mocc[sup.indexpos(np.array(x))] = 1
            clustercount = sup.evalcluster(mocc, socc, clusterexp)
            self.assertTrue(np.all(np.array([4, 6, 4, 0, 1, sup.size]) == clustercount))

        for u in [-0.5/Nsuper, 0.5/Nsuper]:
            mocc = np.zeros(sup.size)
            # make an octahedron...
            for x in [(0., 0., 0.), (u, 0, u), (u, 0, -u), (u, u, 0), (u, -u, 0), (2*u, 0, 0)]:
                mocc[sup.indexpos(np.array(x))] = 1
            clustercount = sup.evalcluster(mocc, socc, clusterexp)
            self.assertTrue(np.all(np.array([6, 12, 8, 12, 0, sup.size]) == clustercount))

    def testClusterEvalSimple2(self):
        """Check that we can evaluate a cluster expansion: B2"""
        B2 = crystal.Crystal(np.eye(3), [[np.array([0., 0., 0.])],
                                         [np.array([0.5, 0.5, 0.5])]], ['A', 'B'])
        Nsuper = 4
        # build a supercell, but call the "A" atoms spectators to the "B" atoms
        sup = supercell.ClusterSupercell(B2, Nsuper*self.one, spectator=[0])
        # build some sites...
        sA1 = cluster.ClusterSite.fromcryscart(B2, np.array([0, 0, 0]))
        sB1 = cluster.ClusterSite.fromcryscart(B2, np.array([0.5, 0.5, 0.5]))
        sA2 = cluster.ClusterSite.fromcryscart(B2, np.array([1., 0., 0.]))
        sB2 = cluster.ClusterSite.fromcryscart(B2, np.array([-0.5, 0.5, 0.5]))
        # build some base clusters...
        cA = cluster.Cluster([sA1])
        cB = cluster.Cluster([sB1])
        cAB = cluster.Cluster([sA1, sB1])
        cAA = cluster.Cluster([sA1, sA2])
        cBB = cluster.Cluster([sB1, sB2])
        cABB = cluster.Cluster([sA1, sB1, sB2])
        # expand out into symmetric sets...
        clusterexp = [set([cl.g(B2, g) for g in B2.G]) for cl in [cA, cB, cAB, cAA, cBB]]
        mocc, socc = np.zeros(sup.size), np.zeros(sup.size)
        clustercount = sup.evalcluster(mocc, socc, clusterexp)
        self.assertTrue(np.all(np.array([0,]*5 + [sup.size]) == clustercount))

        mocc, socc = np.ones(sup.size), np.ones(sup.size)
        clustercount = sup.evalcluster(mocc, socc, clusterexp)
        self.assertTrue(np.all(np.array([1, 1, 8, 3, 3, 1])*sup.size == clustercount))

    def testClusterEvaluator(self):
        """Check that our cluster evaluator works"""
        B2 = crystal.Crystal(np.eye(3), [[np.array([0., 0., 0.])],
                                         [np.array([0.5, 0.5, 0.5])]], ['A', 'B'])
        Nsuper = 4
        # build a supercell, but call the "A" atoms spectators to the "B" atoms
        sup = supercell.ClusterSupercell(B2, Nsuper*self.one, spectator=[0])
        # build some sites...
        sA1 = cluster.ClusterSite.fromcryscart(B2, np.array([0, 0, 0]))
        sB1 = cluster.ClusterSite.fromcryscart(B2, np.array([0.5, 0.5, 0.5]))
        sA2 = cluster.ClusterSite.fromcryscart(B2, np.array([1., 0., 0.]))
        sB2 = cluster.ClusterSite.fromcryscart(B2, np.array([-0.5, 0.5, 0.5]))
        # build some base clusters...
        cA = cluster.Cluster([sA1])
        cB = cluster.Cluster([sB1])
        cAB = cluster.Cluster([sA1, sB1])
        cAA = cluster.Cluster([sA1, sA2])
        cBB = cluster.Cluster([sB1, sB2])
        cABB = cluster.Cluster([sA1, sB1, sB2])
        # expand out into symmetric sets...
        clusterexp = [set([cl.g(B2, g) for g in B2.G]) for cl in [cA, cB, cAB, cAA, cBB, cABB]]
        # ene = np.zeros(len(clusterexp)+1)
        ene = np.random.normal(size=len(clusterexp) + 1) # random interactions
        # work with a random spectator occupancy, then try out some mobile occupancies.
        # mocc, socc = np.zeros(sup.size), np.zeros(sup.size)
        for spec_try in range(10):
            socc = np.random.choice((0,1), size=sup.size)
            siteinteract, interact = sup.clusterevaluator(socc, clusterexp, ene)
            for mobile_try in range(10):
                mocc = np.random.choice((0,1), size=sup.size)
                clustercount = sup.evalcluster(mocc, socc, clusterexp)
                ene_direct = np.dot(ene, clustercount)
                interact_count = np.zeros(len(interact), dtype=int)
                for s, interlist in zip(mocc, siteinteract):
                    if s==0:
                        for m in interlist:
                            interact_count[m] += 1
                ene_count = sum(E for E, c in zip(interact, interact_count) if c==0)
                self.assertAlmostEqual(ene_direct, ene_count)

    def testJumpNetworkEvaluator(self):
        """Can we construct an efficient jump network evaluator?"""
        # *displaced* B2, to break symmetry
        # B2 = crystal.Crystal(np.eye(3), [[np.array([0., 0., 0.])],
        #                                  [np.array([0.5, 0.5, 0.5])]], ['A', 'B'])
        B2 = crystal.Crystal(np.eye(3), [[np.array([0., 0., 0.])],
                                         [np.array([0.55, 0.55, 0.55])]], ['A', 'B'])
        Nsuper = 4
        # build a supercell, but call the "A" atoms spectators to the "B" atoms
        sup = supercell.ClusterSupercell(B2, Nsuper*self.one, spectator=[0])
        chem = 1 # other atom is the spectator, so...
        clusterexp = cluster.makeclusters(B2, 1.2, 4)
        ene = np.random.normal(size=len(clusterexp) + 1)  # random interactions
        # ene = np.ones(len(clusterexp)+1)
        jumpnetwork = B2.jumpnetwork(chem, 1.01)
        eneT = np.random.normal(size=len(jumpnetwork))  # random barriers
        TSclusterexp = cluster.makeTSclusters(B2, chem, jumpnetwork, clusterexp)
        TSvalues = np.random.normal(size=len(TSclusterexp))
        # eneT = np.zeros(len(jumpnetwork))
        for spec_try in range(5):
            socc = np.random.choice((0,1), size=sup.size)
            MCsamp = cluster.MonteCarloSampler(sup, socc, clusterexp, ene)
            siteinteract, interact = sup.clusterevaluator(socc, clusterexp, ene)
            # make copies for testing comparisons...
            siteinteract0 = siteinteract.copy()
            interact0 = interact.copy()
            siteinteract, interact, jumps, interactrange = \
                sup.jumpnetworkevaluator(socc, clusterexp, ene, chem, jumpnetwork, eneT,
                                         TSclusterexp, TSvalues,
                                         siteinteract=siteinteract, interact=interact)
            # first, test that we have a reasonable setup...
            self.assertEqual(len(interact0), interactrange[-1])
            self.assertEqual(sum(len(jn) for jn in jumpnetwork)*sup.size, len(jumps))
            self.assertEqual(len(jumps)+1, len(interactrange))
            self.assertEqual(interact0, interact[:len(interact0)])
            for sint0, sint in zip(siteinteract0, siteinteract):
                self.assertEqual(sint0, sint[:len(sint0)])
            # now, let's make a mobile species distribution, and evaluate all possible transition
            # energies, and make sure that they agree with what our evaluator provides.
            mocc = np.random.choice((0,1), size=sup.size)
            interact_count = np.zeros(len(interact), dtype=int)
            for s, interlist in zip(mocc, siteinteract):
                if s == 0:
                    for m in interlist:
                        interact_count[m] += 1
            Nene, Njumps = interactrange[-1], len(jumps)
            ene_count = sum(E for E, c in zip(interact[:Nene], interact_count[:Nene]) if c == 0)
            ET = np.zeros(Njumps)
            for n in range(Njumps):
                ran = slice(interactrange[n-1], interactrange[n])
                ET[n] = sum(E for E, c in zip(interact[ran], interact_count[ran]) if c == 0)
            # now, to compare all of the jumps!
            for ((i, j), dx), Etrans in zip(jumps, ET):
                if mocc[i] == 0 or mocc[j] == 1: continue
                # we have a valid jump; now we need to back out which particular jump this would be:
                ci0, cj0 = sup.ciR(i)[0], sup.ciR(j)[0]
                E0 = 0
                for jn, ET_trial in zip(jumpnetwork, eneT):
                    for (i0, j0), dx0 in jn:
                        if ci0[1] == i0 and cj0[1] == j0 and np.allclose(dx, dx0):
                            E0 = ET_trial
                # now, we need to get the "LIMB" part of the barrier:
                # compute the interaction count after deoccupying i and occupying j:
                new_interact_count = interact_count.copy()
                for m in siteinteract[i]:
                    new_interact_count[m] += 1
                for m in siteinteract[j]:
                    new_interact_count[m] -= 1
                new_ene_count = sum(E for E, c in zip(interact[:Nene], new_interact_count[:Nene]) if c == 0)
                E0 += 0.5*(new_ene_count - ene_count)
                E0 += np.dot(TSvalues, sup.evalTScluster(mocc, socc, TSclusterexp, i, j, dx))
                self.assertAlmostEqual(E0, Etrans)
