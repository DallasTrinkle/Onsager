"""
Unit tests for HDF5 parsing; runs tests in multiple libraries
"""

__author__ = 'Dallas R. Trinkle'

import unittest
import numpy as np
import h5py
import onsager.crystal as crystal
import onsager.PowerExpansion as PE
import onsager.GFcalc as GFcalc
import onsager.crystalStars as stars
import onsager.OnsagerCalc as OnsagerCalc
T3D = PE.Taylor3D

class HDF5ParsingTests(unittest.TestCase):
    def setUp(self):
        self.f = h5py.File('/dev/null', 'w', driver='core', backing_store=False)

    def tearDown(self):
        self.f.close()

    def testPowerExpansion(self):
        """Test whether we can write and read an HDF5 group containing a PowerExpansion"""
        basis = [(np.eye(2), np.array([0.5,-np.sqrt(0.75),0.])),
         (np.eye(2), np.array([0.5,np.sqrt(0.75),0.])),
         (np.eye(2), np.array([-1.,0.,0.])),
         (np.eye(2), np.array([-0.5,-np.sqrt(0.75),0.])),
         (np.eye(2), np.array([-0.5,np.sqrt(0.75),0.])),
         (np.eye(2), np.array([1.,0.,0.])),
         (np.eye(2)*2, np.array([0.,0.,1.])),
         (np.eye(2)*2, np.array([0.,0.,-1.])),
        ]
        T3D()
        c1 = T3D([c[0] for c in T3D.constructexpansion(basis, N=4, pre=(0,1,1/2,1/6,1/24))])
        c2 = T3D([c[0] for c in T3D.constructexpansion(basis, N=4, pre=(0,-1j,-1/2,+1j/6,1/24))])
        c1.addhdf5(self.f.create_group('T3D-c1'))
        c2.addhdf5(self.f.create_group('T3D-c2'))
        c1copy = T3D.loadhdf5(self.f['T3D-c1'])
        c2copy = T3D.loadhdf5(self.f['T3D-c2'])
        for (a, b) in [(c1, c1copy), (c2, c2copy)]:
            self.assertEqual(len(a.coefflist), len(b.coefflist))
            for (n0, l0, coeff0), (n1, l1, coeff1) in zip(a.coefflist, b.coefflist):
                self.assertEqual(n0, n1)
                self.assertEqual(l0, l1)
                self.assertTrue(np.all(coeff0 == coeff1))
        c1.dumpinternalsHDF5(self.f.create_group('Taylor3Dinternals'))
        self.assertTrue(T3D.checkinternalsHDF5(self.f['Taylor3Dinternals']))

    def testGreenFunction(self):
        """Test whether we can write and read an HDF5 group containing a GFcalc"""
        HCP = crystal.Crystal.HCP(1., np.sqrt(8/3))
        HCP_sitelist = HCP.sitelist(0)
        HCP_jumpnetwork = HCP.jumpnetwork(0, 1.01)
        HCP_GF = GFcalc.GFCrystalcalc(HCP, 0, HCP_sitelist, HCP_jumpnetwork, Nmax=4)
        HCP_GF.addhdf5(self.f.create_group('GFcalc'))
        GFcopy = GFcalc.GFCrystalcalc.loadhdf5(HCP, self.f['GFcalc'])  # note: we need to pass crystal!
        HCP_GF.SetRates([2.],[0],[1.5,0.5],[0.5,1.])  # one unique site, two types of jumps
        GFcopy.SetRates([2.],[0],[1.5,0.5],[0.5,1.])  # one unique site, two types of jumps
        self.assertEqual(HCP_GF(0,0,np.zeros(3)), GFcopy(0,0,np.zeros(3)))

    def testPairState(self):
        """Test whether conversion of different PairState groups back and forth to arrays works"""
        PSlist = [ stars.PairState(i=0, j=1, R=np.array([1,0,-1]), dx=np.array([1.,0.,-1.])),
                   stars.PairState(i=1, j=0, R=np.array([-1,0,1]), dx=np.array([-1.,0.,1.]))]
        ij, R, dx = stars.PSlist2array(PSlist)
        self.assertEqual(ij.shape, (2,2))
        self.assertEqual(R.shape, (2,3))
        self.assertEqual(dx.shape, (2,3))
        PSlistcopy = stars.array2PSlist(ij, R, dx)
        for PS0, PS1 in zip(PSlist, PSlistcopy):
            self.assertEqual(PS0, PS1)

    def testFlattening(self):
        """Test whether conversion between lists of lists and flat lists works"""
        l1 = [ [n for n in range(10)] ]
        fl1, ind1 = stars.doublelist2flatlistindex(l1)
        self.assertTrue(np.all(ind1 == 0))
        self.assertEqual(fl1,l1[0])
        l1copy = stars.flatlistindex2doublelist(fl1, ind1)
        self.assertEqual(len(l1), len(l1copy))
        for lis1, lis1copy in zip(l1, l1copy):
            self.assertEqual(lis1, lis1copy)
        l2 = [ [n for n in range(5)], [n for n in range(1)], [n for n in range(10)]]
        l2copy = stars.flatlistindex2doublelist(*stars.doublelist2flatlistindex(l2))
        self.assertEqual(len(l2), len(l2copy))
        for lis1, lis1copy in zip(l2, l2copy):
            self.assertEqual(lis1, lis1copy)

    def testStarSet(self):
        """Test whether we can write and read an HDF5 group containing a StarSet"""
        HCP = crystal.Crystal.HCP(1., np.sqrt(8/3))
        HCP_jumpnetwork = HCP.jumpnetwork(0, 1.01)
        HCP_StarSet = stars.StarSet(HCP_jumpnetwork, HCP, 0, Nshells=2)
        HCP_StarSet.addhdf5(self.f.create_group('thermo'))
        HCP_StarSetcopy = stars.StarSet.loadhdf5(HCP, self.f['thermo'])  # note: we need to pass crystal!
        self.assertEqual(HCP_StarSet.Nstates, HCP_StarSetcopy.Nstates)
        self.assertEqual(HCP_StarSet.Nshells, HCP_StarSetcopy.Nshells)
        for s1, s2 in zip(HCP_StarSet.states, HCP_StarSetcopy.states):
            self.assertEqual(s1, s2)
            self.assertEqual(HCP_StarSet.stateindex(s1), HCP_StarSet.stateindex(s2))
            self.assertEqual(HCP_StarSet.starindex(s1), HCP_StarSet.starindex(s2))

    def testVectorStarSet(self):
        """Test whether we can write and read an HDF5 group containing a VectorStarSet"""
        HCP = crystal.Crystal.HCP(1., np.sqrt(8/3))
        HCP_jumpnetwork = HCP.jumpnetwork(0, 1.01)
        HCP_StarSet = stars.StarSet(HCP_jumpnetwork, HCP, 0, Nshells=2)
        HCP_VectorStarSet = stars.VectorStarSet(HCP_StarSet)
        HCP_VectorStarSet.addhdf5(self.f.create_group('vkinetic'))
        HCP_VectorStarSetcopy = stars.VectorStarSet.loadhdf5(HCP_StarSet,
                                                             self.f['vkinetic'])  # note: we need to pass StarSet!
        self.assertEqual(HCP_VectorStarSet.Nvstars, HCP_VectorStarSetcopy.Nvstars)
        self.assertTrue(np.all(HCP_VectorStarSet.outer == HCP_VectorStarSetcopy.outer))
        for p1list, v1list, p2list, v2list in zip(HCP_VectorStarSet.vecpos, HCP_VectorStarSet.vecvec,
                                                  HCP_VectorStarSetcopy.vecpos, HCP_VectorStarSetcopy.vecvec):
            self.assertEqual(p1list, p2list)
            self.assertTrue(all(np.all(v1 == v2) for v1, v2 in zip(v1list, v2list)))

    def testvTKdict(self):
        """Test whether we can write and read an HDF5 group containing a dictionary indexed by vTK"""
        self.assertEqual(OnsagerCalc.arrays2vTKdict(*OnsagerCalc.vTKdict2arrays({})), {})
        dict1 = {}
        vTK = OnsagerCalc.vacancyThermoKinetics(pre=np.ones(2), betaene=np.zeros(2),
                                                preT=np.ones(4), betaeneT=np.zeros(4))
        dict1[vTK] = np.eye(3)
        vTK = OnsagerCalc.vacancyThermoKinetics(pre=2.*np.ones(2), betaene=np.zeros(2),
                                                preT=np.ones(4), betaeneT=np.ones(4))
        dict1[vTK] = 2.*np.eye(3)
        dict1copy = OnsagerCalc.arrays2vTKdict(*OnsagerCalc.vTKdict2arrays(dict1))
        for k,v in zip(dict1.keys(), dict1.values()):
            self.assertTrue(np.all(dict1copy[k] == v))
        for k,v in zip(dict1copy.keys(), dict1copy.values()):
            self.assertTrue(np.all(dict1[k] == v))

    def testOnsagerVacancyMediated(self):
        """Test whether we can write and read an HDF5 group containing a VacancyMediated Onsager Calculator"""
        HCP = crystal.Crystal.HCP(1., np.sqrt(8/3))
        HCP_sitelist = HCP.sitelist(0)
        HCP_jumpnetwork = HCP.jumpnetwork(0, 1.01)
        HCP_diffuser = OnsagerCalc.VacancyMediated(HCP, 0, HCP_sitelist, HCP_jumpnetwork, 1)
        HCP_diffuser.addhdf5(self.f)  # we'll usually dump it in main
        HCP_diffuser_copy = OnsagerCalc.VacancyMediated.loadhdf5(self.f)  # should be fully self-contained
        thermaldef = {'preV': np.array([1.]), 'eneV': np.array([0.]),
                      'preT0': np.array([1.,1.5]), 'eneT0': np.array([0.25,0.35])}
        thermaldef.update(HCP_diffuser.maketracerpreene(**thermaldef))
        for L0, Lcopy in zip(HCP_diffuser.Lij(*HCP_diffuser.preene2betafree(1.0, **thermaldef)),
                             HCP_diffuser_copy.Lij(*HCP_diffuser_copy.preene2betafree(1.0, **thermaldef))):
            self.assertTrue(np.allclose(L0, Lcopy), msg='{}\n!=\n{}'.format(L0, Lcopy))
        # compare tags
        for k in HCP_diffuser.tags.keys():
            self.assertEqual(HCP_diffuser.tags[k], HCP_diffuser_copy.tags[k])
        # do a dictionary check (dictionaries are only added *after* a minimum of one call
        HCP_diffuser.addhdf5(self.f.create_group('new'))
        HCP_diffuser_copy = OnsagerCalc.VacancyMediated.loadhdf5(self.f['new'])  # should be fully self-contained
        for L0, Lcopy in zip(HCP_diffuser.Lij(*HCP_diffuser.preene2betafree(1.0, **thermaldef)),
                             HCP_diffuser_copy.Lij(*HCP_diffuser_copy.preene2betafree(1.0, **thermaldef))):
            self.assertTrue(np.allclose(L0, Lcopy), msg='{}\n!=\n{}'.format(L0, Lcopy))
        # Test with B2 (there are additional terms that get used when we have origin states)
        B2 = crystal.Crystal(np.eye(3), [np.zeros(3), np.array([0.45, 0.45, 0.45])])
        B2diffuser = OnsagerCalc.VacancyMediated(B2, 0, B2.sitelist(0), B2.jumpnetwork(0, 0.99), 1)
        B2diffuser.addhdf5(self.f.create_group('B2'))
        B2diffuser_copy = OnsagerCalc.VacancyMediated.loadhdf5(self.f['B2'])
        Nsites, Njumps = len(B2diffuser.sitelist), len(B2diffuser.om0_jn)
        tdef = {'preV': np.ones(Nsites), 'eneV': np.zeros(Njumps),
                'preT0': np.ones(Njumps), 'eneT0': np.zeros(Njumps)}
        tdef.update(B2diffuser.maketracerpreene(**tdef))
        for L0, Lcopy in zip(B2diffuser.Lij(*B2diffuser.preene2betafree(1.0, **tdef)),
                             B2diffuser_copy.Lij(*B2diffuser_copy.preene2betafree(1.0, **tdef))):
            self.assertTrue(np.allclose(L0, Lcopy), msg='{}\n!=\n{}'.format(L0, Lcopy))
        # Test with displaced triangle (2D "B2" example):
        tria2 = crystal.Crystal(np.array([[1.,0.], [0.,np.sqrt(3.)]]),
                                [np.zeros(2),np.array([0.5, 0.4])])
        tria2diffuser = OnsagerCalc.VacancyMediated(tria2, 0, tria2.sitelist(0),
                                                    tria2.jumpnetwork(0, 1.2), 1)
        tria2diffuser.addhdf5(self.f.create_group('tria'))
        tria2diffuser_copy = OnsagerCalc.VacancyMediated.loadhdf5(self.f['tria'])
        Nsites, Njumps = len(tria2diffuser.sitelist), len(tria2diffuser.om0_jn)
        tdef2 = {'preV': np.ones(Nsites), 'eneV': np.zeros(Njumps),
                'preT0': np.ones(Njumps), 'eneT0': np.zeros(Njumps)}
        tdef2.update(tria2diffuser.maketracerpreene(**tdef2))
        for L0, Lcopy in zip(tria2diffuser.Lij(*tria2diffuser.preene2betafree(1.0, **tdef2)),
                             tria2diffuser_copy.Lij(*tria2diffuser_copy.preene2betafree(1.0, **tdef2))):
            self.assertTrue(np.allclose(L0, Lcopy), msg='{}\n!=\n{}'.format(L0, Lcopy))
        # compare tags
        for k in tria2diffuser.tags.keys():
            self.assertEqual(tria2diffuser.tags[k], tria2diffuser_copy.tags[k])
