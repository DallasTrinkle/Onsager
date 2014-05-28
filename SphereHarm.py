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
   return np.array([0,0,0])

