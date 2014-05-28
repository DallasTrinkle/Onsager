import numpy as np

def GFFTfunc(NNvect, rates):
   """
   Returns a Fourier-transform function given the NNvect and rates
   
   Parameters
   ----------
   NNvect[z,3]: list of nearest-neighbor vectors
   rates[z]:    jump rate for each neighbor
   """
   pass

def GFdiff(NNvect, rates):
   """
   Construct the diffusivity matrix (small q limit of Fourier transform).
   Returns a 3x3 matrix that can be dotted into q to get FT.

   Parameters
   ----------
   NNvect[z,3]: list of nearest-neighbor vectors
   rates[z]:    jump rate for each neighbor
   """
   pass
   
