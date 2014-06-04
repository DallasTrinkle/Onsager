import numpy as np
import scipy as sp
from scipy import special

def DFTfunc(NNvect, rates):
   """
   Returns a Fourier-transform function given the NNvect and rates
   
   Parameters
   ----------
   NNvect[z,3]: list of nearest-neighbor vectors
   rates[z]:    jump rate for each neighbor
   """
   return lambda q: np.sum(np.cos(np.dot(NNvect,q))*rates)-np.sum(rates)

def D2(NNvect, rates):
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
   return 0.5*np.dot(NNvect.T*rates, NNvect)
   
def D4(NNvect, rates):
   """
   Construct the discontinuity matrix (fourth derivative wit respect to q of
   Fourier transform).
   Returns a 3x3x3x3 matrix that can be dotted into q to get the FT.

   Parameters
   ----------
   NNvect[z,3]: list of nearest-neighbor vectors
   rates[z]:    jump rate for each neighbor
   """
   D4 = np.zeros((3,3,3,3))
   for a in xrange(3):
      for b in xrange(3):
         for c in xrange(3):
            for d in xrange(3):
               D4[a,b,c,d] = 1./24. * sum(NNvect[:,a]*NNvect[:,b]*NNvect[:,c]*NNvect[:,d]*rates[:])
   return D4

def eval2(q, D):
   """
   Returns q.D.q.

   Parameters
   ----------
   q[3]:   3-vector
   D[3,3]: second-rank tensor
   """
   return np.dot(q, np.dot(q, D))

def eval4(q, D):
   """
   Returns q.q.D.q.q

   Parameters
   ----------
   q[3]:   3-vector
   D[3,3,3,3]: fourth-rank tensor
   """
   return np.dot(q, np.dot(q, np.dot(q, np.dot(q, D))))

def calcDE(D2):
   """
   Takes in the D2 matrix (assumed to be real, symmetric) and diagonalizes it
   returning the eigenvalues (d_i) and corresponding normalized eigenvectors (e_i).
   Returns di[3], ei[3,3], where ei[i,:] is the eigenvector for di[i]. NOTE: this is
   the transposed version of what eigh returns.
   
   Parameters
   ----------
   D2[3,3]: symmetric, real matrix from D2()
   """

   di, ei=np.linalg.eigh(D2)
   return di, ei.T

def invertD2(D2):
   """
   Takes in the matrix D2, returns its inverse (which gets used repeatedly).

   Parameters
   ----------
   D2[3,3]: symmetric, real matrix from D2()
   """

   return np.linalg.inv(D2)

def unorm(di, ei, x):
   """
   Takes the eigenvalues di, eigenvectors ei, and the vector x, and returns the
   normalized u vector, along with its magnitude. These are the key elements needed
   in *all* of the Fourier transform expressions to follow.

   Returns: ui[3], umagn

   Parameters
   ----------
   di[3]:   eigenvalues of D2
   ei[3,3]: eigenvectors of D2 (ei[i,:] == ith eigenvector)
   x[3]:    cartesian position vector
   """

   ui = np.zeros(3)
   umagn = 0
   if (np.dot(x,x)>0):
      ui = np.dot(ei, x)/np.sqrt(di)
      umagn = np.sqrt(np.dot(ui,ui))
      ui /= umagn
   return ui, umagn

def pnorm(di, ei, q):
   """
   Takes the eigenvalues di, eigenvectors ei, and the vector q, and returns the
   normalized p vector, along with its magnitude. These are the key elements needed
   in *all* of the Fourier transform expressions to follow.

   Returns: pi[3], pmagn

   Parameters
   ----------
   di[3]:   eigenvalues of D2
   ei[3,3]: eigenvectors of D2 (ei[i,:] == ith eigenvector)
   q[3]:    cartesian reciprocal vector
   """

   pi = np.zeros(3)
   pmagn = 0
   if (np.dot(q,q)>0):
      pi = np.dot(ei, q)*np.sqrt(di)
      pmagn = np.sqrt(np.dot(pi,pi))
      pi /= pmagn
   return pi, pmagn

def poleFT(di, u, pm, erfupm=-1):
   """
   Calculates the pole FT (excluding the volume prefactor) given the di eigenvalues,
   the value of u magnitude (available from unorm), and the pmax scaling factor.
   Note: if we've already calculated the erf, don't bother recalculating it here.

   Returns erf(0.5*u*pm)/(4*pi*u*sqrt(d1*d2*d3)) if u>0
   else pm/(4*pi^3/2 * sqrt(d1*d2*d3)) if u==0

   Parameters
   ----------
   di[3]:  eigenvalues of D2
   u:      magnitude of u (from unorm())
   pm:     scaling factor for exponential cutoff function
   erfupm: value of erf(0.5*u*pm) (optional; if not set, then its calculated)
   """

   if (erfupm < 0):
      erfupm = special.erf(0.5*u*pm)
   if (erfupm==0):
      return 0.25*pm/np.sqrt(np.product(di*np.pi))
   return erfupm*0.25/(np.pi*u*np.sqrt(np.product(di)))

# Hard-coded?
PowerExpansion = np.array( (
      (0,0,4), (0,4,0), (4,0,0),
      (2,2,0), (2,0,2), (0,2,2),
      (0,1,3), (0,3,1), (2,1,1),
      (1,0,3), (3,0,1), (1,2,1),
      (1,3,0), (3,1,0), (1,1,2)), dtype=int)

# Conversion from hard-coded PowerExpansion back to index number; if not in range,
# its equal to 15. Needs to be constructed

def ConstructExpToIndex():
   """
   Setup to construct ExpToIndex to match PowerExpansion.
   """
   ExpToIndex = 15*np.ones((5,5,5), dtype=int)
   for i in xrange(15):
      ExpToIndex[tuple(PowerExpansion[i])] = i
   return ExpToIndex

ExpToIndex = ConstructExpToIndex()

def D4toNNN(D4):
   """
   Converts from a fourth-derivative expansion D4 into power expansion.

   Returns D15, the expansion coefficients for the power series.

   Parameters
   ----------
   D4[3,3,3,3]: 4th rank tensor coefficient, as in D4[a,b,c,d]*x[a]*x[b]*x[c]*x[d]
   """
   D15 = np.zeros(15)
   for a in xrange(3):
      for b in xrange(3):
         for c in xrange(3):
            for d in xrange(3):
               tup = (a,b,c,d)
               D15[ExpToIndex[tup.count(0),tup.count(1),tup.count(2)]] += D4[tup]
   return D15

def RotateD4(D4, di, ei):
   """
   Returns the rotated (and scaled) version of the fourth-ranked tensor D4,
   using the eigenvalues di and eigenvectors ei. Should be such that if you
   take q, evaluate it as pi, then eval4(pi, Drot4) = eval4(q, D4).

   Returns Drot4[3,3,3,3]

   Parameters
   ----------
   D4[3,3,3,3]: fourth-rank tensor
   di[3]:       eigenvalues
   ei[3,3]:     eigenvectors (ei[i,:] corresponds to di[i])
   """
   Drot4 = np.zeros((3,3,3,3))
   diinvsqrt = 1./np.sqrt(di)
   for a in xrange(3):
      for b in xrange(3):
         for c in xrange(3):
            for d in xrange(3):
               Drot4[a,b,c,d] = (diinvsqrt[a]*diinvsqrt[b]*diinvsqrt[c]*diinvsqrt[d]*
                                 np.dot(ei[a],
                                        np.dot(ei[b],
                                               np.dot(ei[c],
                                                      np.dot(ei[d],D4)))))
   return Drot4

# We construct the 3x15x15 matrix that gives the Fourier transform expansion
# coefficients. This is a bit messy, but necessary (pulled from Mathematica
# evaluation of the same).

def rotatetuple(tup, i):
   """
   Returns rotated version of list--shifting by i.

   >>> rotatetuple((1,2,3), 0)
   (1, 2, 3)
   >>> rotatetuple((1,2,3), 1)
   (2, 3, 1)
   >>> rotatetuple((1,2,3), 2)
   (3, 1, 2)
   """
   i = i % len(tup)
   listrot=list(tup)
   head = listrot[:i]
   del listrot[:i]
   listrot.extend(head)
   return tuple(listrot)

# For these 3x3x3 matrices, the first entry is l corresponding to l=0, 2, 4
# the next two indices correspond to our 3x3 blocks. For <004>, <220>, the
# indices are "shifts". For the <013>/<031>/<112> blocks, these correspond to that
# ordering. This is hardcoded, and comes from Mathematica. These all come from
# transforming the matrices that convert powers qx^nx qy^ny qz^nz into spherical
# harmonics, and then grouping these by l values that show up.

# F44[l, s1, s2] for the <004> type power expansions:
F44 = np.array((
      ((1./5.,1./5.,1./5.), (1./5.,1./5.,1./5.), (1./5.,1./5.,1./5.)),
      ((4./7.,-2./7.,-2./7.), (-2./7.,4./7.,-2./7.), (-2./7.,-2./7.,4./7.)),
      ((8./35.,3./35.,3./35.), (3./35.,8./35.,3./35.), (3./35.,3./35.,8./35.)))
               )
# F22[l, s1, s2] for the <220> type power expansions:
F22 = np.array((
      ((2./15.,2./15.,2./15.), (2./15.,2./15.,2./15.), (2./15.,2./15.,2./15.)),
      ((2./21.,-1./21.,-1./21.), (-1./21.,2./21.,-1./21.), (-1./21.,-1./21.,2./21.)),
      ((27./35.,-3./35.,-3./35.), (-3./35.,27./35.,-3./35.), (-3./35.,-3./35.,27./35.)))
               )
# F42[l, s1, s2] mixes the <004>/<220> types
F42 = np.array((
      ((1./15.,1./15.,1./15.), (1./15.,1./15.,1./15.), (1./15.,1./15.,1./15.)),
      ((-2./21.,1./21.,1./21.), (1./21.,-2./21.,1./21.), (1./21.,1./21.,-2./21.)),
      ((1./35.,-4./35.,-4./35.), (-4./35.,1./35.,-4./35.), (-4./35.,-4./35.,1./35.)))
               )
# F24[l, s1, s2] mixes the <220>/<004> types
F24 = np.array((
      ((2./5.,2./5.,2./5.), (2./5.,2./5.,2./5.), (2./5.,2./5.,2./5.)),
      ((-4./7.,2./7.,2./7.), (2./7.,-4./7.,2./7.), (2./7.,2./7.,-4./7.)),
      ((6./35.,-24./35.,-24./35.), (-24./35.,6./35.,-24./35.), (-24./35.,-24./35.,6./35.)))
               )
# F13[l, i1, i2] mixes the <013>/<031>/<211> types.
# Now, i=0 is [013], 1 is [031], 2 is [211]. We use the shifts to permute among these.
F13 = np.array((
      ((0,0,0), (0,0,0), (0,0,0)),
      ((3./7.,3./7.,1./7.), (3./7.,3./7.,1./7.), (3./7.,3./7.,1./7.)),
      ((4./7.,-3./7.,-1./7.), (-3./7,4./7.,-1./7.), (-3./7.,-3./7.,6./7.)))
               )

def ConstructPowerFT():
   """
   Setup to construct the 3x15x15 PowerFT matrix, which gives the linear
   transform version of our Fourier transform.
   """
   PowerFT = np.zeros((3,15,15))
   # First up, our onsite terms, for the symmetric cases:
   # <004>
   vec = (0,0,4)
   for l in xrange(3):
      for s1 in xrange(3):
         for s2 in xrange(3):
            PowerFT[l,
                    ExpToIndex[rotatetuple(vec,s1)],
                    ExpToIndex[rotatetuple(vec,s2)]] = F44[l, s1, s2]
   # <220>
   vec = (2,2,0)
   for l in xrange(3):
      for s1 in xrange(3):
         for s2 in xrange(3):
            PowerFT[l,
                    ExpToIndex[rotatetuple(vec,s1)],
                    ExpToIndex[rotatetuple(vec,s2)]] = F22[l, s1, s2]

   # <400>/<220> mixed terms:
   vec1 = (0,0,4)
   vec2 = (2,2,0)
   for l in xrange(3):
      for s1 in xrange(3):
         for s2 in xrange(3):
            PowerFT[l,
                    ExpToIndex[rotatetuple(vec1,s1)],
                    ExpToIndex[rotatetuple(vec2,s2)]] = F42[l, s1, s2]
            PowerFT[l,
                    ExpToIndex[rotatetuple(vec2,s2)],
                    ExpToIndex[rotatetuple(vec1,s1)]] = F24[l, s2, s1]
   
   # <013>/<031>/<211>; now, F13 indexes which of those three vectors we need
   veclist = ( (0,1,3), (0,3,1), (2,1,1) )
   for l in xrange(3):
      for v1 in xrange(3):
         for v2 in xrange(3):
            for s1 in xrange(3):
               PowerFT[l,
                       ExpToIndex[rotatetuple(veclist[v1],s1)],
                       ExpToIndex[rotatetuple(veclist[v2],s1)]] = F13[l, v1, v2]

   return PowerFT

PowerFT = ConstructPowerFT()

