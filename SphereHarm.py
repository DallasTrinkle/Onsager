# scipy, numpy, and spharm python interface
import scipy as sp
import numpy as np
import spharm

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
   return np.array((spv[2]*np.cos(spv[0])*np.sin(spv[1]),
                    spv[2]*np.sin(spv[0])*np.sin(spv[1]),
                    spv[2]*np.cos(spv[1])))

# Interface needed:
#   transform from grid function to Ylm
#   given Ylm expansion, evaluate at point
#   these need to be done using the spharm interface
def YlmTransform(D, Npolar=64):
   """
   Takes in a 3x3 matrix D, returns the spherical harmonic expansion of qDq.

   Parameters
   ----------
   D: 3x3 matrix
   """
   Nazim=Npolar*2
   # construct our spherical harmonic transforming object, and datagrid
   sphtrans = spharm.Spharmt(Nazim, Npolar, rsphere=1,
                             gridtype='gaussian', legfunc='stored')
   datagrid = np.empty((Npolar, Nazim))
   # lattitudes for a Gaussian quadrature, converted to radians
   lats = spharm.gaussian_lats_wts(Npolar)[0]*(np.pi/180)
   lons = 2*np.pi/Nazim * np.array(range(Nazim))
   # generate our data: construct qvector from spherical coordinates
   spv = np.empty(3)
   spv[2] = 1
   dazim = 2*np.pi/Nazim
   for p, spv[1] in enumerate(lats):
      for az, spv[0] in enumerate(lons):
         qv = SpheretoCart(spv)
         datagrid[p, az] = np.dot(qv, np.dot(qv, D))
   return sphtrans.grdtospec(datagrid), spharm.getspecindx(Npolar-1)
