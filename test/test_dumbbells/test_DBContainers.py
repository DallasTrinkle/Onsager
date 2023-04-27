import numpy as np

import crystal
from onsager.DB_structs import dumbbell, SdPair, jump, connector
from crysts import *
import itertools
import unittest

class test_DB_structs(unittest.TestCase):
    def setUp(self):
        # DC_Si - same symmetry as FCC, except twice the number of jumps, since we have two basis
        # atoms belonging to the same Wyckoff site, in a crystal with the same lattice vectors.
        latt = np.array([[0., 0.5, 0.5], [0.5, 0., 0.5], [0.5, 0.5, 0.]]) * 0.55
        DC_Si = crystal.Crystal(latt, [[np.array([0., 0., 0.]), np.array([0.25, 0.25, 0.25])]], ["Si"])
        famp0 = [np.array([1., 1., 0.]) / np.sqrt(2) * 0.2]
        family = [famp0]
        self.pdbcontainer = crystal.pureDBContainer(DC_Si, 0, family)
        self.mdbcontainer = crystal.mixedDBContainer(DC_Si, 0, family)

        iorInd_test = 0
        for iorInd in range(len(self.pdbcontainer.iorlist)):
            i, o = self.pdbcontainer.iorlist[iorInd]
            if np.allclose(o, famp0[0]) or np.allclose(o, -famp0[0]):
                iorInd_test = iorInd

        print(self.pdbcontainer.iorlist[iorInd_test])
        self.iorInd_test = iorInd_test
        self.Rdb_test = np.random.randint(0, 5, 3)

    def test_dumbbells(self):
        db_test = dumbbell(self.iorInd_test, self.Rdb_test)
        db_test_2 = dumbbell(self.iorInd_test, self.Rdb_test + 2) # change the lattice position
        db_test_3 = dumbbell(self.iorInd_test + 1, self.Rdb_test) # change the site, orientation index
        db_test_4 = db_test_2 - np.array([2,2,2], dtype=int)  # translate db_test2 back to check addition

        self.assertNotEqual(db_test_2, db_test)
        self.assertNotEqual(db_test_3, db_test)
        self.assertEqual(db_test_4, db_test)

    def test_SdPairs(self):
        db_test = dumbbell(self.iorInd_test, self.Rdb_test)

        i_s_test, R_s_test = 0, np.array([0, 0, 0])
        i_s_test_2 = 1
        R_s_test_2 = np.array([0, 0, 2])

        pair1 = SdPair(i_s_test, R_s_test, db_test)
        db_test_2 = dumbbell(self.iorInd_test, self.Rdb_test + R_s_test_2)
        pair1_trans = SdPair(i_s_test, R_s_test_2, db_test_2)

        pair2 = SdPair(i_s_test_2, R_s_test, db_test)
        pair3 = SdPair(i_s_test_2, R_s_test_2, db_test)
        pair4 = SdPair(i_s_test_2, R_s_test_2, db_test_2)

        self.assertEqual(pair1, pair1)
        self.assertNotEqual(pair1, pair2)
        self.assertNotEqual(pair1, pair3)
        self.assertNotEqual(pair1, pair4)

        # check for additions
        self.assertEqual(pair1, pair1_trans - np.array([0, 0, 2]), msg="found \n{} \n {}".format(pair1, pair1_trans))

    def test_Jumps(self):
        # first, let's check dumbbell jumps
        db_test = dumbbell(self.iorInd_test, self.Rdb_test)
        db_test_2 = dumbbell(self.iorInd_test, self.Rdb_test + 2)

        jmp = jump(db_test, db_test_2, -1, 1)
        jmpNeg = -jmp

        self.assertEqual(jmpNeg.state1, db_test_2)
        self.assertEqual(jmpNeg.state2, db_test)
        self.assertEqual(jmpNeg.c1, 1)
        self.assertEqual(jmpNeg.c2, -1)
        self.assertNotEqual(jmp, jmpNeg)

        # test adding jumps to pairs
        i_s_test, R_s_test = 0, np.array([0, 0, 0])

        pair1 = SdPair(i_s_test, R_s_test, db_test)
        # In jmp, the dumbbell is not at R=0
        with self.assertRaises(ValueError):
            pair2 = pair1.addjump(jmp)

        jmp = jump(db_test - db_test.R, db_test_2, -1, 1)
        pair2 = pair1.addjump(jmp)
        # check that solute does not move
        self.assertEqual(pair1.i_s, pair2.i_s)
        self.assertTrue(np.all(pair1.R_s == pair2.R_s))

        # check the final db location
        self.assertEqual(pair2.db.iorind, db_test_2.iorind)
        self.assertTrue(np.all(pair2.db.R == db_test_2.R + db_test.R))

        db_test_wrong = dumbbell(self.iorInd_test + 3, self.Rdb_test)
        j2 = jump(db_test_wrong - db_test_wrong.R, db_test, -1, 1)
        with self.assertRaises(ArithmeticError):
            pair2 = pair1.addjump(j2)

    def test_Connectors(self):
        # first, let's check dumbbell jumps
        db_test = dumbbell(self.iorInd_test, np.array([0, 0, 0]))
        db_test_2 = dumbbell(self.iorInd_test + 3, self.Rdb_test)

        with self.assertRaises(ValueError):
            conn = connector(db_test_2, db_test)

        conn = connector(db_test, db_test_2)
        conn_neg = -conn
        self.assertEqual(conn_neg.state1, db_test_2 - db_test_2.R)
        self.assertEqual(conn_neg.state2, db_test - db_test_2.R)

        # test connector creation by xor-ing
        i_s_test, R_s_test = 0, np.array([0, 0, 0])
        i_s_test_2 = 1
        R_s_test_2 = np.array([0, 0, 2])

        pair1 = SdPair(i_s_test, R_s_test, db_test)
        pair2 = SdPair(i_s_test_2, R_s_test_2, db_test_2)

        with self.assertRaises(ArithmeticError):
            c1 = pair1 ^ pair2

        pair2 = SdPair(i_s_test, R_s_test_2, db_test_2)
        with self.assertRaises(ArithmeticError):
            c1 = pair1 ^ pair2

        pair2 = SdPair(i_s_test_2, R_s_test, db_test_2)
        with self.assertRaises(ArithmeticError):
            c1 = pair1 ^ pair2

        pair2 = SdPair(i_s_test, R_s_test, db_test_2)

        c1 = pair1 ^ pair2
        c2 = pair2 ^ pair1
        self.assertEqual(c1, -c2)

    def test_gops(self):

        # test the group operations on the structures one by one

        # First, for dumbbells
        # Make a dumbbell object
        db_test = dumbbell(self.iorInd_test, self.Rdb_test)
        for g in self.pdbcontainer.G:
            g_crys = self.pdbcontainer.G_crys[g]
            dbnew, flip = db_test.gop(self.pdbcontainer, g, pure=True)
            # first check the sites
            self.assertEqual(dbnew.iorind, g.indexmap[0][self.iorInd_test])
            if flip == -1:
                o = self.pdbcontainer.iorlist[self.iorInd_test][1]
                o2 = self.pdbcontainer.iorlist[dbnew.iorind][1]
                self.assertTrue(np.allclose(np.dot(g.cartrot, o), -o2))

        # Then for pairs - complex states
        db_test = dumbbell(self.iorInd_test, self.Rdb_test)
        i_s_test, R_s_test = 0, np.array([0, 0, 2])
        pair1 = SdPair(i_s_test, R_s_test, db_test)

        for g in self.pdbcontainer.G:
            g_crys = self.pdbcontainer.G_crys[g]
            pair2, flip = pair1.gop(self.pdbcontainer, g, complex=True)
            # first check the sites
            self.assertEqual(pair2.db.iorind, g.indexmap[0][self.iorInd_test])
            if flip == -1:
                o = self.pdbcontainer.iorlist[self.iorInd_test][1]
                o2 = self.pdbcontainer.iorlist[pair2.db.iorind][1]
                self.assertTrue(np.allclose(np.dot(g.cartrot, o), -o2))

            RNew, (c, i_new) = self.pdbcontainer.crys.g_pos(g_crys, pair1.R_s, (self.pdbcontainer.chem, pair1.i_s))
            self.assertEqual(c, self.pdbcontainer.chem)
            self.assertEqual(i_new, pair2.i_s)
            self.assertTrue(np.all(RNew == pair2.R_s))

        # Then for pairs - mixed states
        db_test = dumbbell(self.iorInd_test, self.Rdb_test)
        i = self.mdbcontainer.iorlist[self.iorInd_test][0]
        pair1 = SdPair(i, self.Rdb_test, db_test)

        for g in self.mdbcontainer.G:
            g_crys = self.mdbcontainer.G_crys[g]
            pair2 = pair1.gop(self.mdbcontainer, g, complex=False)
            # first check the sites
            self.assertEqual(pair2.db.iorind, g.indexmap[0][self.iorInd_test])
            # since mixed state, the orientation change should be a simples rotation
            o1 = self.mdbcontainer.iorlist[self.iorInd_test][1]
            o2 = self.mdbcontainer.iorlist[pair2.db.iorind][1]
            self.assertTrue(np.allclose(o2, np.dot(g.cartrot, o1)),
                            msg="{} {} \n {}".format(o2, self.mdbcontainer.crys.g_direc(g_crys, o1), g.cartrot))

            RNew, (c, i_new) = self.mdbcontainer.crys.g_pos(g_crys, pair1.R_s, (self.mdbcontainer.chem, pair1.i_s))
            self.assertEqual(c, self.mdbcontainer.chem)
            self.assertEqual(i_new, pair2.i_s)
            self.assertTrue(np.all(RNew == pair2.R_s))

            # check that the mixed state remains mixed
            self.assertTrue(np.all(RNew == pair2.db.R))
            inew_db = self.mdbcontainer.iorlist[pair2.db.iorind][0]
            self.assertEqual(i_new, inew_db)

        # test gops for connectors
        db_test = dumbbell(self.iorInd_test, np.array([0, 0, 0]))
        # select another dumbbell to connect this to - could be the same or different - doesn't matter
        newInd = np.random.randint(0, len(self.pdbcontainer.iorlist))
        print("Connector between : {} and {}".format(self.iorInd_test, newInd))
        db_test_2 = dumbbell(newInd, self.Rdb_test)

        conn = connector(db_test, db_test_2)
        for g in self.pdbcontainer.G:
            g_crys = self.pdbcontainer.G_crys[g]
            conn_g = conn.gop(self.pdbcontainer, g)

            db_test_g = db_test.gop(self.pdbcontainer, g, pure=True)[0]
            Rt = db_test_g.R  #.copy()
            db_test_g -= Rt

            db_test_2_g = db_test_2.gop(self.pdbcontainer, g, pure=True)[0]
            db_test_2_g -= Rt

            self.assertEqual(db_test_g, conn_g.state1)
            self.assertEqual(db_test_2_g, conn_g.state2)


class test_statemaking(unittest.TestCase):
    def setUp(self):
        # famp0 = [np.array([1., 1., 0.]), np.array([1., 0., 0.])]
        # famp12 = [np.array([1., 1., 1.]), np.array([1., 1., 0.])]
        # self.family = [famp0, famp12]
        # self.crys = tet2

        latt = np.array([[0., 0.5, 0.5], [0.5, 0., 0.5], [0.5, 0.5, 0.]]) * 0.55
        DC_Si = crystal.Crystal(latt, [[np.array([0., 0., 0.]), np.array([0.25, 0.25, 0.25])]], ["Si"])
        famp0 = [np.array([1., 1., 0.]) / np.sqrt(2) * 0.2]
        self.family = [famp0]
        self.crys = DC_Si

    def test_dbStates(self):
        # check that symmetry analysis is correct
        dbstates = crystal.pureDBContainer(self.crys, 0, self.family)
        print(len(dbstates.iorlist))
        self.assertEqual(len(dbstates.symorlist), 1)
        # check that every (i,or) set is accounted for
        sm = 0
        for i in dbstates.symorlist:
            sm += len(i)
        self.assertEqual(sm, len(dbstates.iorlist))

        for g in self.crys.G:
            gdumb_found = None
            count = 0
            for gdumb, gval in dbstates.G_crys.items():
                if gval == g:
                    gdumb_found = gdumb
                    count += 1

            self.assertEqual(count, 1)
            self.assertTrue(np.allclose(gdumb_found.cartrot, gdumb_found.cartrot))
            self.assertTrue(np.allclose(gdumb_found.rot, gdumb_found.rot))
            self.assertTrue(np.allclose(gdumb_found.trans, gdumb_found.trans))


        # test indexmapping
        for gdumb in dbstates.G:
            # First check that all states are accounted for.
            self.assertEqual(len(gdumb.indexmap[0]), len(dbstates.iorlist))
            for idx1, tup1 in enumerate(dbstates.iorlist):
                i, o = tup1[0], tup1[1]
                R, (ch, inew) = dbstates.crys.g_pos(dbstates.G_crys[gdumb], np.array([0, 0, 0]), (dbstates.chem, i))
                onew = np.dot(gdumb.cartrot, o)
                count = 0
                for idx2, tup2 in enumerate(dbstates.iorlist):
                    if inew == tup2[0] and (np.allclose(tup2[1], onew, atol=dbstates.crys.threshold) or
                                      np.allclose(tup2[1], -onew, atol=dbstates.crys.threshold)):
                        count +=1
                        self.assertEqual(gdumb.indexmap[0][idx1], idx2, msg="{}, {}".format(gdumb.indexmap[0][idx1], idx2))
                self.assertEqual(count, 1)

        # test_indexedsymlist
        for i1, symindlist, symstatelist in zip(itertools.count(),dbstates.symIndlist, dbstates.symorlist):
            for stind, state in zip(symindlist, symstatelist):
                st_iorlist = dbstates.iorlist[stind]
                self.assertEqual(st_iorlist[0], state[0])
                self.assertTrue(np.all(st_iorlist[1] == state[1]))
                self.assertEqual(dbstates.invmap[stind], i1)

        # test indexing
        for idx, (i, o) in enumerate(dbstates.iorlist):
            idxNew = dbstates.getIndex((i, o))
            self.assertEqual(idxNew, idx)
            idxNew = dbstates.getIndex((i, -o))
            self.assertEqual(idxNew, idx) # check that negative is accounted for

            db = dumbbell(idx, np.array([3,3,3]))
            self.assertEqual(dbstates.db2ind(db), idx)

    # Test jumpnetwork
    def test_jnet0(self):
        # cube
        famp0 = [np.array([1., 0., 0.]) / np.linalg.norm(np.array([1., 0., 0.])) * 0.126]
        family = [famp0]
        pdbcontainer_cube = crystal.pureDBContainer(cube, 0, family)
        jset_cube, jind_cube = pdbcontainer_cube.jumpnetwork(0.3, 0.01, 0.01)
        test_dbi = dumbbell(pdbcontainer_cube.getIndex((0, np.array([0.126, 0., 0.]))), np.array([0, 0, 0]))
        test_dbf = dumbbell(pdbcontainer_cube.getIndex((0, np.array([0.126, 0., 0.]))), np.array([0, 1, 0]))
        count = 0
        indices = []
        for i, jlist in enumerate(jset_cube):
            for q, j in enumerate(jlist):
                if j.state1 == test_dbi:
                    if j.state2 == test_dbf:
                        if j.c1 == j.c2 == -1:
                            count += 1
                            indices.append((i, q))
                            jtest = jlist
        # print (indices)
        self.assertEqual(count, 1)  # see that this jump has been taken only once into account
        self.assertEqual(len(jtest), 24)

        o1 = np.array([0.126, 0., 0.])
        o2 = np.array([0., 0.126, 0.])
        idx1 = pdbcontainer_cube.getIndex((0, o1))
        idx2 = pdbcontainer_cube.getIndex((0, o2))

        if np.allclose(o1, pdbcontainer_cube.iorlist[idx1][1]):
            c1 = 1
        else:
            self.assertTrue(np.allclose(o1, -pdbcontainer_cube.iorlist[idx1][1]))
            c1 = -1

        if np.allclose(o2, pdbcontainer_cube.iorlist[idx2][1]):
            c2 = 1
        else:
            self.assertTrue(np.allclose(o2, -pdbcontainer_cube.iorlist[idx2][1]))
            c2 = -1

        test_dbi = dumbbell(idx1, np.array([0, 0, 0]))
        test_dbf = dumbbell(idx2, np.array([0, 0, 0]))
        count = 0
        for i, jlist in enumerate(jset_cube):
            for q, j in enumerate(jlist):
                if j.state1 == test_dbi:
                    if j.state2 == test_dbf:
                        if (j.c1 == c1 and j.c2 == c2) or (j.c1 == -c1 and j.c2 == -c2):
                            count += 1
                            jtest = jlist
        self.assertEqual(count, 1)  # see that this jump has been taken only once into account
        self.assertEqual(len(jtest), 12)




        # Next FCC
        # test this out with FCC
        fcc = crystal.Crystal.FCC(0.55)
        famp0 = [np.array([1., 1., 0.]) / np.sqrt(2) * 0.2]
        family = [famp0]
        pdbcontainer_fcc = crystal.pureDBContainer(fcc, 0, family)
        jset_fcc, jind_fcc = pdbcontainer_fcc.jumpnetwork(0.4, 0.01, 0.01)
        o1 = np.array([1., -1., 0.]) * 0.2 / np.sqrt(2)
        if any(np.allclose(-o1, o) for i, o in pdbcontainer_fcc.iorlist):
            o1 = -o1.copy()
        db1 = dumbbell(pdbcontainer_fcc.getIndex((0, o1)), np.array([0, 0, 0]))
        db2 = dumbbell(pdbcontainer_fcc.getIndex((0, o1)), np.array([0, 0, 1]))
        jmp = jump(db1, db2, 1, 1)
        jtest = []
        for jl in jset_fcc:
            for j in jl:
                if j == jmp:
                    jtest.append(jl)
        # see that the jump has been accounted just once.
        self.assertEqual(len(jtest), 1)
        # See that there 24 jumps. 24 0->0.
        self.assertEqual(len(jtest[0]), 24)

        o1 = np.array([1., 1., 0.]) * 0.2 / np.sqrt(2)
        o2 = np.array([-1., 1., 0.]) * 0.2 / np.sqrt(2)
        idx1 = pdbcontainer_fcc.getIndex((0, o1))
        idx2 = pdbcontainer_fcc.getIndex((0, o2))

        if np.allclose(o1, pdbcontainer_fcc.iorlist[idx1][1]):
            c1 = 1
        else:
            self.assertTrue(np.allclose(o1, -pdbcontainer_fcc.iorlist[idx1][1]))
            c1 = -1

        if np.allclose(o2, pdbcontainer_fcc.iorlist[idx2][1]):
            c2 = 1
        else:
            self.assertTrue(np.allclose(o2, -pdbcontainer_fcc.iorlist[idx2][1]))
            c2 = -1

        test_dbi = dumbbell(idx1, np.array([0, 0, 0]))
        test_dbf = dumbbell(idx2, np.array([0, 0, 0]))
        count = 0
        for i, jlist in enumerate(jset_fcc):
            for q, j in enumerate(jlist):
                if j.state1 == test_dbi:
                    if j.state2 == test_dbf:
                        if (j.c1 == c1 and j.c2 == c2) or (j.c1 == -c1 and j.c2 == -c2):
                            count += 1
                            jtest = jlist
        self.assertEqual(count, 1)  # see that this jump has been taken only once into account
        self.assertEqual(len(jtest), 12)

        # check that across all jumps, a rotation jump only occurs once and its equivalent does not occur
        rotSet = []
        allJumps = []
        for jlist in jset_fcc:
            for j in jlist:
                allJumps.append(j)
                (i1, o1) = pdbcontainer_fcc.iorlist[j.state1.iorind]
                (i2, o2) = pdbcontainer_fcc.iorlist[j.state2.iorind]
                R1 = j.state1.R
                R2 = j.state2.R
                dx_explicit = pdbcontainer_fcc.crys.pos2cart(R2, (pdbcontainer_fcc.chem, i2)) - \
                              pdbcontainer_fcc.crys.pos2cart(R1, (pdbcontainer_fcc.chem, i1))
                if np.allclose(dx_explicit, 0):
                    rotSet.append(j)

        for jrot in rotSet:
            jequiv = jump(jrot.state1, jrot.state2, -jrot.c1, -jrot.c2)
            self.assertTrue(jequiv not in rotSet)
            self.assertTrue(jequiv not in allJumps)




        # DC_Si - same symmetry as FCC, except twice the number of jumps, since we have two basis
        # atoms belonging to the same Wyckoff site, in a crystal with the same lattice vectors.
        latt = np.array([[0., 0.5, 0.5], [0.5, 0., 0.5], [0.5, 0.5, 0.]]) * 0.55
        DC_Si = crystal.Crystal(latt, [[np.array([0., 0., 0.]), np.array([0.25, 0.25, 0.25])]], ["Si"])
        famp0 = [np.array([1., 1., 0.]) / np.sqrt(2) * 0.2]
        family = [famp0]
        pdbcontainer_si = crystal.pureDBContainer(DC_Si, 0, family)
        jset_si, jind_si = pdbcontainer_si.jumpnetwork(0.4, 0.01, 0.01)
        o1 = np.array([1., -1., 0.]) * 0.2 / np.sqrt(2)
        if any(np.allclose(-o1, o) for i, o in pdbcontainer_si.iorlist):
            o1 = -o1.copy()
        db1 = dumbbell(pdbcontainer_si.getIndex((0, o1)), np.array([0, 0, 0]))
        db2 = dumbbell(pdbcontainer_si.getIndex((0, o1)), np.array([0, 0, 1]))
        jmp = jump(db1, db2, 1, 1)
        jtest = []
        for jl in jset_si:
            for j in jl:
                if j == jmp:
                    jtest.append(jl)
        # see that the jump has been accounted just once.
        self.assertEqual(len(jtest), 1)
        # See that there 48 jumps. 24 0->0 and 24 1->1.
        self.assertEqual(len(jtest[0]), 48)

        o1 = np.array([1., 1., 0.]) * 0.2 / np.sqrt(2)
        o2 = np.array([-1., 1., 0.]) * 0.2 / np.sqrt(2)
        idx1 = pdbcontainer_si.getIndex((0, o1))
        idx2 = pdbcontainer_si.getIndex((0, o2))

        if np.allclose(o1, pdbcontainer_si.iorlist[idx1][1]):
            c1 = 1
        else:
            self.assertTrue(np.allclose(o1, -pdbcontainer_si.iorlist[idx1][1]))
            c1 = -1

        if np.allclose(o2, pdbcontainer_si.iorlist[idx2][1]):
            c2 = 1
        else:
            self.assertTrue(np.allclose(o2, -pdbcontainer_si.iorlist[idx2][1]))
            c2 = -1

        test_dbi = dumbbell(idx1, np.array([0, 0, 0]))
        test_dbf = dumbbell(idx2, np.array([0, 0, 0]))
        count = 0
        for i, jlist in enumerate(jset_si):
            for q, j in enumerate(jlist):
                if j.state1 == test_dbi:
                    if j.state2 == test_dbf:
                        if (j.c1 == c1 and j.c2 == c2) or (j.c1 == -c1 and j.c2 == -c2):
                            count += 1
                            jtest = jlist
        self.assertEqual(count, 1)  # see that this jump has been taken only once into account
        self.assertEqual(len(jtest), 24)

        # check that across all jumps, a rotation jump only occurs once and its equivalent does not occur
        rotSet = []
        allJumps = []
        for jlist in jset_si:
            for j in jlist:
                allJumps.append(j)
                (i1, o1) = pdbcontainer_si.iorlist[j.state1.iorind]
                (i2, o2) = pdbcontainer_si.iorlist[j.state2.iorind]
                R1 = j.state1.R
                R2 = j.state2.R
                dx_explicit = pdbcontainer_si.crys.pos2cart(R2, (pdbcontainer_si.chem, i2)) - \
                              pdbcontainer_si.crys.pos2cart(R1, (pdbcontainer_si.chem, i1))
                if np.allclose(dx_explicit, 0):
                    rotSet.append(j)

        for jrot in rotSet:
            jequiv = jump(jrot.state1, jrot.state2, -jrot.c1, -jrot.c2)
            self.assertTrue(jequiv not in rotSet)
            self.assertTrue(jequiv not in allJumps)



        # HCP
        Mg = crystal.Crystal.HCP(0.3294, chemistry=["Mg"])
        famp0 = [np.array([1., 0., 0.]) * 0.145]
        family = [famp0]
        pdbcontainer_hcp = crystal.pureDBContainer(Mg, 0, family)
        jset_hcp, jind_hcp = pdbcontainer_hcp.jumpnetwork(0.45, 0.01, 0.01)
        o = np.array([0.145, 0., 0.])
        if any(np.allclose(-o, o1) for i, o1 in pdbcontainer_hcp.iorlist):
            o = -o + 0.
        db1 = dumbbell(pdbcontainer_hcp.getIndex((0, o)), np.array([0, 0, 0], dtype=int))
        db2 = dumbbell(pdbcontainer_hcp.getIndex((1, o)), np.array([0, 0, 0], dtype=int))
        testjump = jump(db1, db2, 1, 1)
        count = 0
        testlist = []
        for jl in jset_hcp:
            for j in jl:
                if j == testjump:
                    count += 1
                    testlist = jl
        self.assertEqual(len(testlist), 24)
        self.assertEqual(count, 1)

        # check that across all jumps, a rotation jump only occurs once and its equivalent does not occur
        rotSet = []
        allJumps = []
        for jlist in jset_hcp:
            for j in jlist:
                allJumps.append(j)
                (i1, o1) = pdbcontainer_hcp.iorlist[j.state1.iorind]
                (i2, o2) = pdbcontainer_hcp.iorlist[j.state2.iorind]
                R1 = j.state1.R
                R2 = j.state2.R
                dx_explicit = pdbcontainer_hcp.crys.pos2cart(R2, (pdbcontainer_hcp.chem, i2)) - \
                              pdbcontainer_hcp.crys.pos2cart(R1, (pdbcontainer_hcp.chem, i1))
                if np.allclose(dx_explicit, 0):
                    rotSet.append(j)

        for jrot in rotSet:
            jequiv = jump(jrot.state1, jrot.state2, -jrot.c1, -jrot.c2)
            self.assertTrue(jequiv not in rotSet)
            self.assertTrue(jequiv not in allJumps)

        # test_indices
        # First check if they have the same number of lists and elements
        jindlist = [jind_cube, jind_fcc, jind_si, jind_hcp]
        jsetlist = [jset_cube, jset_fcc, jset_si, jset_hcp]
        pdbcontainerlist = [pdbcontainer_cube, pdbcontainer_fcc, pdbcontainer_si, pdbcontainer_hcp]
        for pdbcontainer, jset, jind in zip(pdbcontainerlist, jsetlist, jindlist):
            self.assertEqual(len(jind), len(jset))
            # now check if all the elements are correctly correspondent
            for lindex in range(len(jind)):
                self.assertEqual(len(jind[lindex]), len(jset[lindex]))
                for jindex in range(len(jind[lindex])):
                    (i1, o1) = pdbcontainer.iorlist[jind[lindex][jindex][0][0]]
                    (i2, o2) = pdbcontainer.iorlist[jind[lindex][jindex][0][1]]
                    R1 = jset[lindex][jindex].state1.R
                    R2 = jset[lindex][jindex].state2.R
                    self.assertEqual(pdbcontainer.iorlist[jset[lindex][jindex].state1.iorind][0], i1)
                    self.assertEqual(pdbcontainer.iorlist[jset[lindex][jindex].state2.iorind][0], i2)
                    self.assertTrue(np.allclose(pdbcontainer.iorlist[jset[lindex][jindex].state1.iorind][1], o1))
                    self.assertTrue(np.allclose(pdbcontainer.iorlist[jset[lindex][jindex].state2.iorind][1], o2))
                    dx = crystal.DB_disp(pdbcontainer, jset[lindex][jindex].state1, jset[lindex][jindex].state2)
                    dx_explicit = pdbcontainer.crys.pos2cart(R2, (pdbcontainer.chem, i2)) - \
                                  pdbcontainer.crys.pos2cart(R1, (pdbcontainer.chem, i1))
                    self.assertTrue(np.allclose(dx, jind[lindex][jindex][1]))
                    self.assertTrue(np.allclose(dx, dx_explicit))
                    self.assertTrue(np.allclose(dx, jind[lindex][jindex][1]))

    def test_mStates(self):
        dbstates = crystal.pureDBContainer(self.crys, 0, self.family)
        mstates = crystal.mixedDBContainer(self.crys, 0, self.family)

        # # check that symmetry analysis is correct
        self.assertEqual(len(mstates.symorlist), 1)
        #
        # # check that negative orientations are accounted for
        self.assertEqual(len(mstates.symorlist[0]) / len(dbstates.symorlist[0]), 2)

        # check that every (i,or) set is accounted for
        sm = 0
        for i in mstates.symorlist:
            sm += len(i)
        self.assertEqual(sm, len(mstates.iorlist))
        
        for g in self.crys.G:
            gdumb_found = None
            count = 0
            for gdumb, gval in mstates.G_crys.items():
                if gval == g:
                    gdumb_found = gdumb
                    count += 1
            self.assertEqual(count, 1)
            self.assertTrue(np.allclose(gdumb_found.cartrot, g.cartrot))
            self.assertTrue(np.allclose(gdumb_found.rot, g.rot))
            self.assertTrue(np.allclose(gdumb_found.trans, g.trans))

            gdumb_found_pure = None
            count = 0
            for gdumb, gval in dbstates.G_crys.items():
                if gval == g:
                    gdumb_found_pure = gdumb
                    count += 1
            self.assertEqual(count, 1)
            self.assertTrue(np.allclose(gdumb_found_pure.cartrot, gdumb_found.cartrot))
            self.assertTrue(np.allclose(gdumb_found_pure.rot, gdumb_found.rot))
            self.assertTrue(np.allclose(gdumb_found_pure.trans, gdumb_found.trans))
        
        # check indexmapping
        for gdumb in mstates.G:
            self.assertEqual(len(gdumb.indexmap[0]), len(mstates.iorlist))
            for stateind, tup in enumerate(mstates.iorlist):
                i, o = tup[0], tup[1]
                R, (ch, inew) = mstates.crys.g_pos(mstates.G_crys[gdumb], np.array([0, 0, 0]), (mstates.chem, i))
                onew = np.dot(gdumb.cartrot, o)
                count = 0
                for j, t in enumerate(mstates.iorlist):
                    if t[0] == inew and np.allclose(t[1], onew):
                        foundindex = j
                        count += 1
                self.assertEqual(count, 1)
                self.assertEqual(foundindex, gdumb.indexmap[0][stateind])

        # Check indexing of symlist
        for symind, symIndlist, symstlist in zip(itertools.count(), mstates.symIndlist, mstates.symorlist):
            for idx, state in zip(symIndlist, symstlist):
                self.assertEqual(mstates.iorlist[idx][0],state[0])
                self.assertTrue(np.all(mstates.iorlist[idx][1] == state[1]))
                self.assertEqual(mstates.invmap[idx], symind)

        for idx, (i, o) in enumerate(mstates.iorlist):
            idxNew = mstates.getIndex((i, o))
            self.assertEqual(idxNew, idx)
            idxNew = mstates.getIndex((i, -o))
            self.assertNotEqual(idxNew, idx) # check that negative is treated differently

            db = dumbbell(idx, np.array([3,3,3]))
            self.assertEqual(mstates.db2ind(db), idx)

    def test_mixedjumps(self):
        latt = np.array([[0., 0.5, 0.5], [0.5, 0., 0.5], [0.5, 0.5, 0.]]) * 0.55
        DC_Si = crystal.Crystal(latt, [[np.array([0., 0., 0.]), np.array([0.25, 0.25, 0.25])]], ["Si"])
        famp0 = [np.array([1., 1., 0.]) / np.sqrt(2) * 0.2]
        family = [famp0]
        mdbcontainer = crystal.mixedDBContainer(DC_Si, 0, family)
        jset, jind = mdbcontainer.jumpnetwork(0.4, 0.01, 0.01)
        o1 = np.array([1., 1., 0.]) * 0.2 / np.sqrt(2)
        # if any(np.allclose(-o1, o) for i, o in pdbcontainer.iorlist):
        #     o1 = -o1.copy()
        # db1 = dumbbell(pdbcontainer_si.getIndex((0, o1)), np.array([0, 0, 0]))
        # db2 = dumbbell(pdbcontainer_si.getIndex((0, o1)), np.array([0, 0, 1]))
        test_dbi = dumbbell(mdbcontainer.getIndex((0, o1)), np.array([0, 0, 0]))
        test_dbf = dumbbell(mdbcontainer.getIndex((1, o1)), np.array([0, 0, 0]))
        count = 0
        jtest = None
        for i, jlist in enumerate(jset):
            for q, j in enumerate(jlist):
                self.assertTrue(j.c1 == j.c2 == 1) # mixed dumbbell jumps always involve the solute
                if j.state1.db == test_dbi:
                    if j.state2.db == test_dbf:
                        count += 1
                        jtest = jlist
        self.assertEqual(count, 1)  # see that this jump has been taken only once into account
        self.assertEqual(len(jtest), 48)

        # check if conditions for mixed dumbbell transitions are satisfied
        count = 0
        for jl in jset:
            for j in jl:
                if j.c1 == -1 or j.c2 == -1:
                    count += 1
                    break
                if not (j.state1.i_s == mdbcontainer.iorlist[j.state1.db.iorind][0] and
                        j.state2.i_s == mdbcontainer.iorlist[j.state2.db.iorind][0] and
                        np.allclose(j.state1.R_s, j.state1.db.R) and
                        np.allclose(j.state2.R_s, j.state2.db.R)):
                    count += 1
                    break
            if count == 1:
                break
        self.assertEqual(count, 0)

        # test_indices
        # First check if they have the same number of lists and elements
        self.assertEqual(len(jind), len(jset))
        # now check if all the elements are correctly correspondent
        for lindex in range(len(jind)):
            self.assertEqual(len(jind[lindex]), len(jset[lindex]))
            for jindex in range(len(jind[lindex])):
                (i1, o1) = mdbcontainer.iorlist[jind[lindex][jindex][0][0]]
                (i2, o2) = mdbcontainer.iorlist[jind[lindex][jindex][0][1]]
                R2 = jset[lindex][jindex].state2.db.R
                R1 = jset[lindex][jindex].state1.db.R
                self.assertEqual(mdbcontainer.iorlist[jset[lindex][jindex].state1.db.iorind][0], i1)
                self.assertEqual(mdbcontainer.iorlist[jset[lindex][jindex].state2.db.iorind][0], i2)
                self.assertTrue(np.allclose(mdbcontainer.iorlist[jset[lindex][jindex].state1.db.iorind][1], o1))
                self.assertTrue(np.allclose(mdbcontainer.iorlist[jset[lindex][jindex].state2.db.iorind][1], o2))
                dx = crystal.DB_disp(mdbcontainer, jset[lindex][jindex].state1, jset[lindex][jindex].state2)
                dx_explicit =  mdbcontainer.crys.pos2cart(R2, (mdbcontainer.chem, i2)) - \
                      mdbcontainer.crys.pos2cart(R1, (mdbcontainer.chem, i1))
                self.assertTrue(np.allclose(dx, jind[lindex][jindex][1]))
                self.assertTrue(np.allclose(dx, dx_explicit))

class test_2d(unittest.TestCase):
    def setUp(self):
        o = np.array([0., .2])
        famp0 = [o.copy()]
        self.family = [famp0]

        latt = np.array([[1., 0.], [0., np.sqrt(2)]])
        self.crys = crystal.Crystal(latt, [np.array([0., 0.]), np.array([0.5, 0.5])], ["A"])
        print(self.crys)

    def test_dbStates(self):
        # check that symmetry analysis is correct
        dbstates = crystal.pureDBContainer(self.crys, 0, self.family)
        print(len(dbstates.iorlist))
        self.assertEqual(len(dbstates.symorlist), 1)
        # check that every (i,or) set is accounted for
        sm = 0
        for i in dbstates.symorlist:
            sm += len(i)
        self.assertEqual(sm, len(dbstates.iorlist))

        # test indexmapping
        for gdumb in dbstates.G:
            # First check that all states are accounted for.
            self.assertEqual(len(gdumb.indexmap[0]), len(dbstates.iorlist))
            for idx1, tup1 in enumerate(dbstates.iorlist):
                i, o = tup1[0], tup1[1]
                R, (ch, inew) = dbstates.crys.g_pos(dbstates.G_crys[gdumb], np.array([0, 0]), (dbstates.chem, i))
                onew = np.dot(gdumb.cartrot, o)
                count = 0
                for idx2, tup2 in enumerate(dbstates.iorlist):
                    if inew == tup2[0] and (np.allclose(tup2[1], onew, atol=dbstates.crys.threshold) or
                                      np.allclose(tup2[1], -onew, atol=dbstates.crys.threshold)):
                        count +=1
                        self.assertEqual(gdumb.indexmap[0][idx1], idx2, msg="{}, {}".format(gdumb.indexmap[0][idx1], idx2))
                self.assertEqual(count, 1)

                # test_indexedsymlist
            for i1, symindlist, symstatelist in zip(itertools.count(), dbstates.symIndlist, dbstates.symorlist):
                for stind, state in zip(symindlist, symstatelist):
                    st_iorlist = dbstates.iorlist[stind]
                    self.assertEqual(st_iorlist[0], state[0])
                    self.assertTrue(np.all(st_iorlist[1] == state[1]))
                    self.assertEqual(dbstates.invmap[stind], i1)

            # test indexing
            for idx, (i, o) in enumerate(dbstates.iorlist):
                idxNew = dbstates.getIndex((i, o))
                self.assertEqual(idxNew, idx)
                idxNew = dbstates.getIndex((i, -o))
                self.assertEqual(idxNew, idx)  # check that negative is accounted for

                db = dumbbell(idx, np.array([3, 3]))
                self.assertEqual(dbstates.db2ind(db), idx)

    def test_jnet0(self):
        # set up the container
        pdbcontainer = crystal.pureDBContainer(self.crys, 0, self.family)
        cut = 1.01 * np.linalg.norm(self.crys.lattice[:, 0])
        jset, jind = pdbcontainer.jumpnetwork(cut, 0.01, 0.01)
        self.assertEqual(len(pdbcontainer.iorlist), 1)
        idx = pdbcontainer.getIndex((0, np.array([0., .2])))
        self.assertEqual(idx, 0)
        test_dbi = dumbbell(idx, np.array([0, 0]))
        test_dbf = dumbbell(idx, np.array([0, 1]))
        count = 0
        indices = []
        for i, jlist in enumerate(jset):
            for q, j in enumerate(jlist):
                if j.state1 == test_dbi:
                    if j.state2 == test_dbf:
                        if j.c1 == -1 and j.c2 == 1:
                            count += 1
                            indices.append((i, q))
                            jtest = jlist
        # print (indices)
        self.assertEqual(count, 1)  # see that this jump has been taken only once into account
        try:
            self.assertEqual(len(jtest), 4)
        except AssertionError:
            for jmp in jtest:
                print(pdbcontainer.iorlist[0])
                print(jmp)
        self.assertEqual(len(jtest), 4)

        # check that no rotations occur here - since 180 flips are not taken into account
        count = 0
        checked = 0
        print(len(jset))
        for jlist in jset:
            for j in jlist:
                (i1, o1) = pdbcontainer.iorlist[j.state1.iorind]
                (i2, o2) = pdbcontainer.iorlist[j.state2.iorind]
                R1 = j.state1.R
                R2 = j.state2.R
                dx_explicit = pdbcontainer.crys.pos2cart(R2, (pdbcontainer.chem, i2)) - \
                              pdbcontainer.crys.pos2cart(R1, (pdbcontainer.chem, i1))
                checked += 1
                if np.allclose(dx_explicit, 0):
                    count += 1

        print(checked)
        self.assertEqual(count, 0)
        self.assertEqual(checked, 16)

        # test_indices
        # First check if they have the same number of lists and elements
        self.assertEqual(len(jind), len(jset))
        # now check if all the elements are correctly correspondent
        for lindex in range(len(jind)):
            self.assertEqual(len(jind[lindex]), len(jset[lindex]))
            for jindex in range(len(jind[lindex])):
                self.assertEqual(jind[lindex][jindex][0][0], 0)
                self.assertEqual(jind[lindex][jindex][0][1], 0)
                (i1, o1) = pdbcontainer.iorlist[jind[lindex][jindex][0][0]]
                (i2, o2) = pdbcontainer.iorlist[jind[lindex][jindex][0][1]]
                self.assertEqual(i1, 0)
                self.assertEqual(i2, 0)
                self.assertTrue(np.allclose(o1, pdbcontainer.iorlist[0][1]))
                self.assertTrue(np.allclose(o2, pdbcontainer.iorlist[0][1]))
                self.assertEqual(pdbcontainer.iorlist[jset[lindex][jindex].state1.iorind][0], i1)
                self.assertEqual(pdbcontainer.iorlist[jset[lindex][jindex].state2.iorind][0], i2)
                self.assertTrue(np.allclose(pdbcontainer.iorlist[jset[lindex][jindex].state1.iorind][1], o1))
                self.assertTrue(np.allclose(pdbcontainer.iorlist[jset[lindex][jindex].state2.iorind][1], o2))
                dx = crystal.DB_disp(pdbcontainer, jset[lindex][jindex].state1, jset[lindex][jindex].state2)
                self.assertAlmostEqual(np.linalg.norm(dx), np.linalg.norm(pdbcontainer.crys.lattice[:, 0]), places=8)
                self.assertTrue(np.allclose(dx, jind[lindex][jindex][1]))

    def test_mStates(self):
        dbstates = crystal.pureDBContainer(self.crys, 0, self.family)
        mstates = crystal.mixedDBContainer(self.crys, 0, self.family)

        # check that symmetry analysis is correct
        self.assertEqual(len(mstates.symorlist), 1)

        # check that negative orientations are accounted for
        for i in range(len(mstates.symorlist)):
            self.assertEqual(len(mstates.symorlist[i]) / len(dbstates.symorlist[i]), 2)

        # check that every (i,or) set is accounted for
        sm = 0
        for i in mstates.symorlist:
            sm += len(i)
        self.assertEqual(sm, len(mstates.iorlist))

        # check indexmapping
        for gdumb in mstates.G:
            self.assertEqual(len(gdumb.indexmap[0]), len(mstates.iorlist))
            for stateind, tup in enumerate(mstates.iorlist):
                i, o = tup[0], tup[1]
                R, (ch, inew) = mstates.crys.g_pos(mstates.G_crys[gdumb], np.array([0, 0]), (mstates.chem, i))
                onew = np.dot(gdumb.cartrot, o)
                count = 0
                for j, t in enumerate(mstates.iorlist):
                    if t[0] == inew and np.allclose(t[1], onew):
                        foundindex = j
                        count += 1
                self.assertEqual(count, 1)
                self.assertEqual(foundindex, gdumb.indexmap[0][stateind])

        # Check indexing of symlist
        for symind, symIndlist, symstlist in zip(itertools.count(), mstates.symIndlist, mstates.symorlist):
            for idx, state in zip(symIndlist, symstlist):
                self.assertEqual(mstates.iorlist[idx][0],state[0])
                self.assertTrue(np.all(mstates.iorlist[idx][1] == state[1]))
                self.assertEqual(mstates.invmap[idx], symind)

        for idx, (i, o) in enumerate(mstates.iorlist):
            idxNew = mstates.getIndex((i, o))
            self.assertEqual(idxNew, idx)
            idxNew = mstates.getIndex((i, -o))
            self.assertNotEqual(idxNew, idx) # check that negative is treated differently

            db = dumbbell(idx, np.array([3,3]))
            self.assertEqual(mstates.db2ind(db), idx)

    def test_jnet2(self):
        # set up the container
        mdbcontainer = crystal.mixedDBContainer(self.crys, 0, self.family)
        cut = 1.01 * np.linalg.norm(self.crys.lattice[:, 0])
        jset, jind = mdbcontainer.jumpnetwork(cut, 0.01, 0.01)
        self.assertEqual(len(mdbcontainer.iorlist), 2)
        idx = mdbcontainer.getIndex((0, np.array([0., .2])))
        test_dbi = dumbbell(idx, np.array([0, 0]))
        test_dbf = dumbbell(idx, np.array([0, 1]))
        count = 0
        indices = []
        for i, jlist in enumerate(jset):
            for q, j in enumerate(jlist):
                self.assertTrue(j.c1 == j.c2 == 1) # solute always moves
                if j.state1.db == test_dbi:
                    if j.state2.db == test_dbf:
                        count += 1
                        indices.append((i, q))
                        jtest = jlist
        # print (indices)
        self.assertEqual(count, 1)  # see that this jump has been taken only once into account
        try:
            self.assertEqual(len(jtest), 8)
        except AssertionError:
            for jmp in jtest:
                print(mdbcontainer.iorlist[0])
                print(jmp)
        self.assertEqual(len(jtest), 8)