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
        GFcopy = GFcalc.GFCrystalcalc.loadhdf5(self.f['GFcalc'])
        HCP_GF.SetRates([2.],[0],[1.5,-1.0],[0.5,1.])  # one unique site, two types of jumps
        GFcopy.SetRates([2.],[0],[1.5,-1.0],[0.5,1.])  # one unique site, two types of jumps
        self.assertEqual(HCP_GF(0,0,np.zeros(3)), GFcopy(0,0,np.zeros(3)))
