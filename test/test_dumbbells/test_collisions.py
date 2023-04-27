import numpy as np
from onsager.DB_collisions import *
import unittest
from crysts import *
import onsager.crystal as crystal
from onsager.DB_structs import dumbbell, SdPair, jump

class collision_tests(unittest.TestCase):

    def setUp(self):
        famp0 = [np.array([0.10, 0., 0.])]
        family = [famp0]
        self.pdbcontainer_cube = crystal.pureDBContainer(cube, 0, family)
        self.pdbcontainer_BCC = crystal.pureDBContainer(Fe_bcc, 0, family)

        or_tet2 = np.array([1.0, -1.0, 0.]) / np.sqrt(2) * 0.1
        famp0 = [np.zeros(3)]
        famp12 = [or_tet2.copy()]
        family = [famp0, famp12]
        self.pdbcontainer_tet2 = crystal.pureDBContainer(tet2, 0, family)

    def test_self(self):

        # Get cube orientations
        for iorInd, (i, o) in enumerate(self.pdbcontainer_cube.iorlist):
            if np.allclose(np.array([0.1, 0., 0.]), o):
                c1_cube = 1
                or1_cube = iorInd

            elif np.allclose(np.array([-0.1,0.,0.]), o):
                c1_cube = -1
                or1_cube = iorInd

        db1Cube = crystal.dumbbell(or1_cube, np.array([0, 0, 0]))
        db2Cube = crystal.dumbbell(or1_cube, np.array([1, 0, 0]))

        # Get BCC orientations
        for iorInd, (i, o) in enumerate(self.pdbcontainer_BCC.iorlist):
            if np.allclose(np.array([0.1, 0., 0.]), o):
                c1_BCC = 1
                or1_BCC = iorInd

            elif np.allclose(np.array([-0.1,0.,0.]), o):
                c1_BCC = -1
                or1_BCC = iorInd

        x = np.array([0.286, 0., 0.])
        R, u = Fe_bcc.cart2pos(x)

        db1BCC = dumbbell(or1_BCC, np.array([0, 0, 0]))
        db2BCC = dumbbell(or1_BCC, R)

        jmp = jump(db1Cube, db2Cube, c1_cube*1, c1_cube*1)
        check = collision_self(self.pdbcontainer_cube, None, jmp, 0.01, 0.01)
        self.assertTrue(check)

        jmp = jump(db1BCC, db2BCC, c1_BCC * 1, c1_BCC * 1)
        check = collision_self(self.pdbcontainer_BCC, None, jmp, 0.01, 0.01)
        self.assertTrue(check, msg="{} \n{}".format(jmp, self.pdbcontainer_BCC.iorlist))

        jmp = jump(db1Cube, db2Cube, c1_cube * -1, c1_cube * -1)
        check = collision_self(self.pdbcontainer_cube, None, jmp, 0.01, 0.01)
        self.assertTrue(check)

        jmp = jump(db1BCC, db2BCC, c1_BCC * -1, c1_BCC * -1)
        check = collision_self(self.pdbcontainer_BCC, None, jmp, 0.01, 0.01)
        self.assertTrue(check)

        jmp = jump(db1Cube, db2Cube, c1_cube * 1, c1_cube * -1)
        check = collision_self(self.pdbcontainer_cube, None, jmp, 0.01, 0.01)
        self.assertFalse(check)

        jmp = jump(db1BCC, db2BCC, c1_BCC * 1, c1_BCC * -1)
        check = collision_self(self.pdbcontainer_BCC, None, jmp, 0.01, 0.01)
        self.assertFalse(check)

    def test_others(self):
        # Test where collision is sure to be detected
        or1 = np.array([1.0, -1.0, 0.])/np.sqrt(2)*0.1

        for iorInd, (i, o) in enumerate(self.pdbcontainer_tet2.iorlist):
            if np.allclose(or1, o) and i == 2:
                c1 = 1
                or1Ind = iorInd

            elif np.allclose(-or1, o) and i == 2:
                c1 = -1
                or1Ind = iorInd

        db1 = dumbbell(or1Ind, np.array([0, 0, 0]))
        db2 = dumbbell(or1Ind, np.array([1, -1, 0]))
        jmp = jump(db1, db2, c1*1, c1*-1)
        check1 = collision_self(self.pdbcontainer_tet2, None, jmp, 0.01, 0.01)
        check2 = collision_others(self.pdbcontainer_tet2, None, jmp, 0.01)
        self.assertFalse(check1)
        self.assertTrue(check2)

        # Test where collision is sure to be not detected

        for iorInd, (i, o) in enumerate(self.pdbcontainer_tet2.iorlist):
            if np.allclose(or1, o) and i == 1:
                c2 = 1
                or2Ind = iorInd
                print(or1, i, o, iorInd)

            elif np.allclose(-or1, o) and i == 1:
                c2 = -1
                or2Ind = iorInd
                print(or1, i, o, iorInd)


        db1 = dumbbell(or1Ind, np.array([0, 0, 0]))
        db2 = dumbbell(or2Ind, np.array([0, 0, 0]))

        jmp = jump(db1, db2, c1*1, c2*-1)
        check1 = collision_self(self.pdbcontainer_tet2, None, jmp, 0.01, 0.01)
        check2 = collision_others(self.pdbcontainer_tet2, None, jmp, 0.01)
        self.assertFalse(check1, msg="\n{} \n{} \n{}".format(jmp, self.pdbcontainer_tet2.iorlist, self.pdbcontainer_tet2.crys))
        self.assertFalse(check2)
