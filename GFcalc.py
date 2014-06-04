"""
GFcalc module

Code to compute the lattice Green function for diffusion; this entails inverting
the "diffusion" matrix, which is infinite, singular, and has translational
invariance. The solution involves fourier transforming to reciprocal space,
inverting, and inverse fourier transforming back to real (lattice) space. The
complication is that the inversion produces a second order pole which must be
treated analytically. Subtracting off the pole then produces a discontinuity at
the gamma-point (q=0), which also should be treated analytically. Then, the
remaining function can be numerically inverse fourier transformed.
"""

import numpy as np
import scipy as sp
from scipy import special

def DFTfunc(NNvect, rates):
   """
   Returns a Fourier-transform function given the NNvect and rates
   
   Parameters
   ----------
   NNvect : int array [:,:]
       list of nearest-neighbor vectors
   rates : array [:]
       jump rate for each neighbor, from 1..z where z is the length of `NNvect`[,:]

   Returns
   -------
   DFTfunc : callable function (q)
       a callable function (constructed with lambda) that takes q and
       returns the fourier transform of the D(R) matrix
   """
   return lambda q: np.sum(np.cos(np.dot(NNvect,q))*rates)-np.sum(rates)

def D2(NNvect, rates):
   """
   Construct the diffusivity matrix (small q limit of Fourier transform
   as a second derivative).

   Parameters
   ----------
   NNvect : int array [:,:]
       list of nearest-neighbor vectors
   rates : array [:]
       jump rate for each neighbor, from 1..z where z is the length of `NNvect`[,:]

   Returns
   -------
   D2 : array [3,3]
       3x3 matrix (2nd rank tensor) that can be dotted into q to get FT.
   """
   # return np.zeros((3,3))
   return 0.5*np.dot(NNvect.T*rates, NNvect)
   
def D4(NNvect, rates):
   """
   Construct the discontinuity matrix (fourth derivative wit respect to q of
   Fourier transform).

   Parameters
   ----------
   NNvect : int array [:,:]
       list of nearest-neighbor vectors
   rates : array [:]
       jump rate for each neighbor, from 1..z where z is the length of `NNvect`[,:]

   Returns
   -------
   D4 : array [3,3,3,3]
       3x3x3x3 matrix (4th rank tensor) that can be dotted into q to get FT.
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
   q : array [3]
       3-vector
   D : array[3,3]
       second-rank tensor
   """
   return np.dot(q, np.dot(q, D))

def eval4(q, D):
   """
   Returns q.q.D.q.q

   Parameters
   ----------
   Parameters
   ----------
   q : array [3]
       3-vector
   D : array[3,3,3,3]
       fourth-rank tensor
   """
   return np.dot(q, np.dot(q, np.dot(q, np.dot(q, D))))

def calcDE(D2):
   """
   Takes in the `D2` matrix (assumed to be real, symmetric) and diagonalizes it
   returning the eigenvalues (`di`) and corresponding normalized eigenvectors (`ei`).
   
   Parameters
   ----------
   D2 : array[:,:]
       symmetric, real matrix from `D2`()

   Returns
   -------
   di : array [:]
       eigenvalues of `D2`
   ei : array [:,:]
       eigenvectors of `D2`, where `ei`[i,:] is the eigenvector for `di`[i]

   Notes
   -----
   This uses eigh, but returns the transposed version of output from eigh.
   """

   di, ei=np.linalg.eigh(D2)
   return di, ei.T

def invertD2(D2):
   """
   Takes in the matrix `D2`, returns its inverse (which gets used repeatedly).

   Parameters
   ----------
   D2 : array[:,:]
       symmetric, real matrix from `D2`()

   Returns
   -------
   invD2 : array[:,:]
       inverse of `D2`
   """

   return np.linalg.inv(D2)

def unorm(di, ei, x):
   """
   Takes the eigenvalues `di`, eigenvectors `ei`, and the vector x, and returns the
   normalized u vector, along with its magnitude. These are the key elements needed
   in *all* of the Fourier transform expressions to follow.

   Parameters
   ----------
   di : array [:]
       eigenvalues of `D2`
   ei : array [:,:]
       eigenvectors of `D2`, where `ei`[i,:] is the eigenvector for `di`[i]
   x : array [:]
       cartesian position vector

   Returns
   -------
   ui : array [:]
       normalized components ui = (`di`^-1/2 x.`ei`)/umagn
   umagn : double
       magnitude = sum_i `di`^-1 (x.`ei`)^2 = x.D^-1.x
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
   Takes the eigenvalues `di`, eigenvectors `ei`, and the vector q, and returns the
   normalized p vector, along with its magnitude. These are the key elements needed
   in *all* of the Fourier transform expressions to follow.

   Parameters
   ----------
   di : array [:]
       eigenvalues of `D2`
   ei : array [:,:]
       eigenvectors of `D2`, where `ei`[i,:] is the eigenvector for `di`[i]
   q : array [:]
       cartesian reciprocal vector

   Returns
   -------
   pi : array [:]
       normalized components pi = (`di`^1/2 q.`ei`)/pmagn
   pmagn : double
       magnitude = sum_i `di` (q.`ei`)^2 = q.D.q
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
   Calculates the pole FT (excluding the volume prefactor) given the `di` eigenvalues,
   the value of u magnitude (available from unorm), and the pmax scaling factor.

   Parameters
   ----------
   di : array [:]
       eigenvalues of `D2`
   u : double
       magnitude of u, from unorm() = x.D^-1.x
   pm : double
       scaling factor pmax for exponential cutoff function
   erfupm : double, optional
       value of erf(0.5*u*pm) (negative = not set, then its calculated)

   Returns
   -------
   poleFT : double
       integral of Gaussian cutoff function corresponding to a l=0 pole;
       erf(0.5*u*pm)/(4*pi*u*sqrt(d1*d2*d3)) if u>0
       pm/(4*pi^3/2 * sqrt(d1*d2*d3)) if u==0
   """

   if (u==0):
      return 0.25*pm/np.sqrt(np.product(di*np.pi))
   if (erfupm < 0):
      erfupm = special.erf(0.5*u*pm)
   return erfupm*0.25/(np.pi*u*np.sqrt(np.product(di)))

def discFT(di, u, pm, erfupm=-1, gaussupm=-1):
   """
   Calculates the discontinuity FT (excluding the volume prefactor) given the
   `di` eigenvalues, the value of u magnitude (available from unorm), and the pmax
   scaling factor. Returns a 3-vector for l=0, 2, and 4.

   Parameters
   ----------
   di : array [:]
       eigenvalues of `D2`
   u : double
       magnitude of u, from unorm() = `x`.`D2`^-1.`x`
   pm : double
       scaling factor pmax for exponential cutoff function
   erfupm : double, optional
       value of erf(`u` `pm` / 2) (negative = not set, then its calculated)
   gaussupm : double, optional
       value of exp(-(`u` `pm` / 2)**2) (negative = not set, then its calculated)

   Returns
   -------
   poleFT : array [:]
       integral of Gaussian cutoff function corresponding to a l=0,2,4 discontinuities;
       z = `u` `pm`
       l=0: 1/(4pi u^3 (d1 d2 d3)^1/2 * z^3 * exp(-z^2/4)/2 sqrt(pi)
       l=2: 1/(4pi u^3 (d1 d2 d3)^1/2 * (-15/2*erf(z/2)
             + (15/2 + 5/4 z^2)exp(-z^2/4)/sqrt(pi)
       l=4: 1/(4pi u^3 (d1 d2 d3)^1/2 * (63*15/8*(1-14/z^2)*erf(z/2)
             + (63*15*14/8z + 63*5/2 z + 63/8 z^3)exp(-z^2/4)/sqrt(pi)
   """

   if (u==0):
      return np.array((0,0,0))
   pi1 = 1./np.sqrt(np.pi)
   pre = 0.25/(np.pi*u*u*u*np.sqrt(np.product(di)))
   z = u*pm
   z2 = z*z
   z3 = z*z2
   zm1 = 1./z
   zm2 = zm1*zm1
   zm3 = zm1*zm2
   if (erfupm < 0):
      erfupm = special.erf(0.5*z)
   if (gaussupm < 0):
      gaussupm = np.exp(-0.25*z2)
   return pre*np.array((0.5*pi1*z3*gaussupm,
                        -7.5*erfupm + pi1*gaussupm*(7.5*z+1.25*z3),
                        118.125*(1-14.*zm2)*erfupm + pi1*gaussupm*(
            1653.75*zm1 + 157.5*z + 7.875*z3)
                        ))

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

   Returns
   -------
   ExpToIndex : array [5,5,5]
       array that gives the corresponding index in PowerExpansion list for
       (n1, n2, n3)
   """
   ExpToIndex = 15*np.ones((5,5,5), dtype=int)
   for i in xrange(15):
      ExpToIndex[tuple(PowerExpansion[i])] = i
   return ExpToIndex

ExpToIndex = ConstructExpToIndex()

def D4toNNN(D4):
   """
   Converts from a fourth-derivative expansion `D4` into power expansion.

   Parameters
   ----------
   D4 : array [3,3,3,3]
       4th rank tensor coefficient, as in `D4`[a,b,c,d]*x[a]*x[b]*x[c]*x[d]

   Returns
   -------
   D15 : array [15]
       expansion coefficients in terms of powers
   """
   D15 = np.zeros(15)
   for a in xrange(3):
      for b in xrange(3):
         for c in xrange(3):
            for d in xrange(3):
               tup = (a,b,c,d)
               D15[ExpToIndex[tup.count(0),tup.count(1),tup.count(2)]] += D4[tup]
   return D15

def powereval(u):
   """
   Takes the 3-vector u, and returns the 15-vector of the powers of u,
   corresponding to PowerExpansion terms.

   Parameters
   ----------
   u : array [3]
       3-vector to power-expand.

   Returns
   -------
   powers : array [15]
       `u` components raised to the powers in PowerExpansion
   """
   powers = np.zeros(15)
   for ind, power in enumerate(PowerExpansion):
      powers[ind] = (u[0]**power[0])*(u[1]**power[1])*(u[2]**power[2])
   return powers

def RotateD4(D4, di, ei):
   """
   Returns the rotated (and scaled) version of the fourth-ranked tensor `D4`,
   using the eigenvalues `di` and eigenvectors `ei` into `Drot4`.

   Parameters
   ----------
   D4 : array [3,3,3,3]
       4th rank tensor coefficient, as in `D4`[a,b,c,d]*x[a]*x[b]*x[c]*x[d]
   di : array [:]
       eigenvalues of `D2`
   ei : array [:,:]
       eigenvectors of `D2`, where `ei`[i,:] is the eigenvector for `di`[i]

   Returns
   -------
   Drot4 : array [3,3,3,3]
       4th rank tensor coefficients, rotated so that for `q`, converted to normalized
       `pi` with magnitude `pmagn`, `pmagn`**4 eval4(`pi`, `Drot4`) = eval4(`q`, `D4`).
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

"""
For these 3x3x3 matrices, the first entry is l corresponding to l=0, 2, 4
the next two indices correspond to our 3x3 blocks. For <004>, <220>, the
indices are "shifts". For the <013>/<031>/<112> blocks, these correspond to that
ordering. This is hardcoded, and comes from Mathematica. These all come from
transforming the matrices that convert powers qx^nx qy^ny qz^nz into spherical
harmonics, and then grouping these by l values that show up. This is transposed
so that we can make the 3x15 matrix as PowerFT[:,:,:] * D15[:], and then we need
to make two vectors: a 3-vector for (f0(z), f2(z), f4(z)) and a 15 vector of our powers
"""

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

   Returns
   -------
   PowerFT : array [3,15,15]
       The [l, n, m] matrix corresponding to the l=0, 2, and 4 FT, where if
       we right multiply a 15 vector D15 by `PowerFT`, we get the 3x15 FT matrix.
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
