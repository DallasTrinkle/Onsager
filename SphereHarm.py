import scipy as sp
import numpy as np

def CarttoSphere(xv):
   """
   Converts a cartesian vector into spherical coordinates, for use with Ylm's

   [0]: theta (azimuthal angle in xy plane, between 0 and 2*pi)
   [1]: phi (polar angle, relative to z axis, between 0 and pi)
   [2]: magnitude (>=0)
   
   Parameters
   ----------
   xv[3]: cartesian vector to convert
   """
   spv = np.zeros(3)
   spv[2] = np.sqrt(np.dot(xv,xv))
   if spv[2] > 0:
      uz = xv[2]/spv[2]
      spv[1] = np.arccos(uz)
      if abs(uz)<1:
         spv[0] = np.arctan2(xv[1], xv[0])
   return spv

def SpheretoCart(spv):
   """
   Converts a spherical coordinates into cartesian, for use with Ylm's
   Spherical coordinates:

   [0]: theta (azimuthal angle in xy plane, between 0 and 2*pi)
   [1]: phi (polar angle, relative to z axis, between 0 and pi)
   [2]: magnitude (>=0)
   
   Parameters
   ----------
   spv[3]: cartesian vector to convert
   """
   return np.array(spv[2]*(np.cos(spv[0])*np.sin(spv[1]),
                           np.sin(spv[0])*np.sin(spv[1]),
                           np.cos(spv[1])))
