import numpy as np

def GFFTfunc(NNvect, rates):
   """
   Returns a Fourier-transform function given the NNvect and rates
   
   Parameters
   ----------
   NNvect[z,3]: list of nearest-neighbor vectors
   rates[z]:    jump rate for each neighbor
   """
   return lambda q: np.sum(np.cos(np.dot(NNvect,q))*rates)-np.sum(rates)

def GF2(NNvect, rates):
   """
   Construct the diffusivity matrix (small q limit of Fourier transform
   as a second derivative).
   Returns a 3x3 matrix that can be dotted into q to get FT.

   Parameters
   ----------
   NNvect[z,3]: list of nearest-neighbor vectors
   rates[z]:    jump rate for each neighbor
   """
   # return np.zeros((3,3))
   return -0.5*np.dot(NNvect.T*rates, NNvect)
   
def GF4(NNvect, rates):
   """
   Construct the discontinuity matrix (fourth derivative wit respect to q of
   Fourier transform).
   Returns a 3x3x3x3 matrix that can be dotted into q to get the FT.

   Parameters
   ----------
   NNvect[z,3]: list of nearest-neighbor vectors
   rates[z]:    jump rate for each neighbor
   """
   return np.zeros((3,3,3,3))
   
