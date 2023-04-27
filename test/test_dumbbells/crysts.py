import numpy as np
import onsager.crystal as crystal
#This will be used to store crystal structures needed to create tests
#omega_Ti crystal structure
a0=4.576855
c_a=2.828683/a0
alatt=np.array([[0.5, 0.5, 0.],[-np.sqrt(0.75), np.sqrt(0.75), 0.],[0., 0., c_a]]) * a0
u1=np.array([0.,0.,0.])
u2=np.array([1./3., 2./3., 1./2.])
u3=np.array([2./3., 1./3., 1./2.])
omega_Ti = crystal.Crystal(alatt,[[u1,u2,u3]],["Ti"])
#simple tetragonal crystal structure
# tet = crystal.Crystal(np.array([[1.2,0.,0.],[0.,1.2,0.],[0.,0.,1.5]]),[[np.zeros(3)]])
# tet2 = crystal.Crystal(np.array([[0.32,0.,0.],[0.,0.32,0.],[0.,0.,0.45]]),[[np.zeros(3),np.array([0.5,0.,0.]),np.array([0.,0.5,0.])]])
# tet3 = crystal.Crystal(np.array([[0.32,0.,0.],[0.,0.32,0.],[0.,0.,0.45]]),[[np.zeros(3),np.array([0.0,0.5,0.])],[np.array([0.5,0.,0.])]],["A","B"])
# print (tet3)
tet = crystal.Crystal(np.array([[0.28,0.,0.],[0.,0.28,0.],[0.,0.,0.32]]),[[np.zeros(3)]])
tet2 = crystal.Crystal(np.array([[0.28,0.,0.],[0.,0.28,0.],[0.,0.,0.32]]),[[np.zeros(3),np.array([0.5,0.,0.]),np.array([0.,0.5,0.])]])
tet3 = crystal.Crystal(np.array([[0.32,0.,0.],[0.,0.32,0.],[0.,0.,0.45]]),[[np.zeros(3),np.array([0.0,0.5,0.])],[np.array([0.5,0.,0.])]],["A","B"])
tet4 = crystal.Crystal(np.array([[0.28,0.,0.],[0.,0.28,0.],[0.,0.,0.32]]),[[np.zeros(3),np.array([0.5,0.,0.]),np.array([0.5,0.5,0.])]])
#simple cubic
cube = crystal.Crystal(np.array([[0.28,0.,0.],[0.,0.28,0.],[0.,0.,0.28]]),[[np.zeros(3)]])
#BCC Fe
Fe_bcc = crystal.Crystal.BCC(0.286,"Fe")
