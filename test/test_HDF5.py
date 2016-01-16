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
        self.assertEqual(HCP_VectorStarSet.Nstars, HCP_VectorStarSetcopy.Nstars)
        self.assertEqual(HCP_VectorStarSet.Nvstars, HCP_VectorStarSetcopy.Nvstars)
        self.assertTrue(np.all(HCP_VectorStarSet.outer == HCP_VectorStarSetcopy.outer))
        for p1list, v1list, p2list, v2list in zip(HCP_VectorStarSet.vecpos, HCP_VectorStarSet.vecvec,
                                                  HCP_VectorStarSetcopy.vecpos, HCP_VectorStarSetcopy.vecvec):
            self.assertEqual(p1list, p2list)
            self.assertTrue(all(np.all(v1 == v2) for v1, v2 in zip(v1list, v2list)))

