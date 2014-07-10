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

__author__ = 'Dallas R. Trinkle'

import numpy as np
import scipy as sp
from scipy import special
import KPTmesh


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
    return lambda q: np.sum(np.cos(np.dot(NNvect, q)) * rates) - np.sum(rates)


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
    return 0.5 * np.dot(NNvect.T * rates, NNvect)


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
    D4 = np.zeros((3, 3, 3, 3))
    for a in xrange(3):
        for b in xrange(3):
            for c in xrange(3):
                for d in xrange(3):
                    D4[a, b, c, d] = 1. / 24. * sum(
                        NNvect[:, a] * NNvect[:, b] * NNvect[:, c] * NNvect[:, d] * rates[:])
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

    di, ei = np.linalg.eigh(D2)
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
    if (np.dot(x, x) > 0):
        ui = np.dot(ei, x) / np.sqrt(di)
        umagn = np.sqrt(np.dot(ui, ui))
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
    if (np.dot(q, q) > 0):
        pi = np.dot(ei, q) * np.sqrt(di)
        pmagn = np.sqrt(np.dot(pi, pi))
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

    if (u == 0):
        return 0.25 * pm / np.sqrt(np.product(di * np.pi))
    if (erfupm < 0):
        erfupm = special.erf(0.5 * u * pm)
    return erfupm * 0.25 / (np.pi * u * np.sqrt(np.product(di)))


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
    discFT : array [:]
        integral of Gaussian cutoff function corresponding to a l=0,2,4 discontinuities;
        z = `u` `pm`
        l=0: 1/(4pi u^3 (d1 d2 d3)^1/2) * z^3 * exp(-z^2/4)/2 sqrt(pi)
        l=2: 1/(4pi u^3 (d1 d2 d3)^1/2) * (-15/2*erf(z/2)
                + (15/2 + 5/4 z^2)exp(-z^2/4)/sqrt(pi)
        l=4: 1/(4pi u^3 (d1 d2 d3)^1/2) * (63*15/8*(1-14/z^2)*erf(z/2)
                + (63*15*14/8z + 63*5/2 z + 63/8 z^3)exp(-z^2/4)/sqrt(pi)
    """

    if (u == 0):
        return np.array((0, 0, 0))
    pi1 = 1. / np.sqrt(np.pi)
    pre = 0.25 / (np.pi * u * u * u * np.sqrt(np.product(di)))
    z = u * pm
    z2 = z * z
    z3 = z * z2
    zm1 = 1. / z
    zm2 = zm1 * zm1
    if (erfupm < 0):
        erfupm = special.erf(0.5 * z)
    if (gaussupm < 0):
        gaussupm = np.exp(-0.25 * z2)
    return pre * np.array((0.5 * pi1 * z3 * gaussupm,
                           -7.5 * erfupm + pi1 * gaussupm * (7.5 * z + 1.25 * z3),
                           118.125 * (1 - 14. * zm2) * erfupm + pi1 * gaussupm * (
                               1653.75 * zm1 + 157.5 * z + 7.875 * z3)
    ))

# Hard-coded?
PowerExpansion = np.array((
                              (0, 0, 4), (0, 4, 0), (4, 0, 0),
                              (2, 2, 0), (2, 0, 2), (0, 2, 2),
                              (0, 1, 3), (0, 3, 1), (2, 1, 1),
                              (1, 0, 3), (3, 0, 1), (1, 2, 1),
                              (1, 3, 0), (3, 1, 0), (1, 1, 2)),
                          dtype=int)

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
    ExpToIndex = 15 * np.ones((5, 5, 5), dtype=int)
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
                    tup = (a, b, c, d)
                    D15[ExpToIndex[tup.count(0), tup.count(1), tup.count(2)]] += D4[tup]
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
        powers[ind] = (u[0] ** power[0]) * (u[1] ** power[1]) * (u[2] ** power[2])
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
    Drot4 = np.zeros((3, 3, 3, 3))
    diinvsqrt = 1. / np.sqrt(di)
    for a in xrange(3):
        for b in xrange(3):
            for c in xrange(3):
                for d in xrange(3):
                    Drot4[a, b, c, d] = (diinvsqrt[a] * diinvsqrt[b] * diinvsqrt[c] * diinvsqrt[d] *
                                         np.dot(ei[a],
                                                np.dot(ei[b],
                                                       np.dot(ei[c],
                                                              np.dot(ei[d], D4)))))
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
    i %= len(tup)
    listrot = list(tup)
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
    ((1. / 5., 1. / 5., 1. / 5.), (1. / 5., 1. / 5., 1. / 5.), (1. / 5., 1. / 5., 1. / 5.)),
    ((4. / 7., -2. / 7., -2. / 7.), (-2. / 7., 4. / 7., -2. / 7.), (-2. / 7., -2. / 7., 4. / 7.)),
    ((8. / 35., 3. / 35., 3. / 35.), (3. / 35., 8. / 35., 3. / 35.), (3. / 35., 3. / 35., 8. / 35.)))
)
# F22[l, s1, s2] for the <220> type power expansions:
F22 = np.array((
    ((2. / 15., 2. / 15., 2. / 15.), (2. / 15., 2. / 15., 2. / 15.), (2. / 15., 2. / 15., 2. / 15.)),
    ((2. / 21., -1. / 21., -1. / 21.), (-1. / 21., 2. / 21., -1. / 21.), (-1. / 21., -1. / 21., 2. / 21.)),
    ((27. / 35., -3. / 35., -3. / 35.), (-3. / 35., 27. / 35., -3. / 35.), (-3. / 35., -3. / 35., 27. / 35.)))
)
# F42[l, s1, s2] mixes the <004>/<220> types
F42 = np.array((
    ((1. / 15., 1. / 15., 1. / 15.), (1. / 15., 1. / 15., 1. / 15.), (1. / 15., 1. / 15., 1. / 15.)),
    ((-2. / 21., 1. / 21., 1. / 21.), (1. / 21., -2. / 21., 1. / 21.), (1. / 21., 1. / 21., -2. / 21.)),
    ((1. / 35., -4. / 35., -4. / 35.), (-4. / 35., 1. / 35., -4. / 35.), (-4. / 35., -4. / 35., 1. / 35.)))
)
# F24[l, s1, s2] mixes the <220>/<004> types
F24 = np.array((
    ((2. / 5., 2. / 5., 2. / 5.), (2. / 5., 2. / 5., 2. / 5.), (2. / 5., 2. / 5., 2. / 5.)),
    ((-4. / 7., 2. / 7., 2. / 7.), (2. / 7., -4. / 7., 2. / 7.), (2. / 7., 2. / 7., -4. / 7.)),
    ((6. / 35., -24. / 35., -24. / 35.), (-24. / 35., 6. / 35., -24. / 35.), (-24. / 35., -24. / 35., 6. / 35.)))
)
# F13[l, i1, i2] mixes the <013>/<031>/<211> types.
# Now, i=0 is [013], 1 is [031], 2 is [211]. We use the shifts to permute among these.
F13 = np.array((
    ((0, 0, 0), (0, 0, 0), (0, 0, 0)),
    ((3. / 7., 3. / 7., 1. / 7.), (3. / 7., 3. / 7., 1. / 7.), (3. / 7., 3. / 7., 1. / 7.)),
    ((4. / 7., -3. / 7., -1. / 7.), (-3. / 7, 4. / 7., -1. / 7.), (-3. / 7., -3. / 7., 6. / 7.)))
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
    PowerFT = np.zeros((3, 15, 15))
    # First up, our onsite terms, for the symmetric cases:
    # <004>
    vec = (0, 0, 4)
    for l in xrange(3):
        for s1 in xrange(3):
            for s2 in xrange(3):
                PowerFT[l,
                        ExpToIndex[rotatetuple(vec, s1)],
                        ExpToIndex[rotatetuple(vec, s2)]] = F44[l, s1, s2]
    # <220>
    vec = (2, 2, 0)
    for l in xrange(3):
        for s1 in xrange(3):
            for s2 in xrange(3):
                PowerFT[l,
                        ExpToIndex[rotatetuple(vec, s1)],
                        ExpToIndex[rotatetuple(vec, s2)]] = F22[l, s1, s2]

    # <400>/<220> mixed terms:
    vec1 = (0, 0, 4)
    vec2 = (2, 2, 0)
    for l in xrange(3):
        for s1 in xrange(3):
            for s2 in xrange(3):
                PowerFT[l,
                        ExpToIndex[rotatetuple(vec1, s1)],
                        ExpToIndex[rotatetuple(vec2, s2)]] = F42[l, s1, s2]
                PowerFT[l,
                        ExpToIndex[rotatetuple(vec2, s2)],
                        ExpToIndex[rotatetuple(vec1, s1)]] = F24[l, s2, s1]

    # <013>/<031>/<211>; now, F13 indexes which of those three vectors we need
    veclist = ( (0, 1, 3), (0, 3, 1), (2, 1, 1) )
    for l in xrange(3):
        for v1 in xrange(3):
            for v2 in xrange(3):
                for s1 in xrange(3):
                    PowerFT[l,
                            ExpToIndex[rotatetuple(veclist[v1], s1)],
                            ExpToIndex[rotatetuple(veclist[v2], s1)]] = F13[l, v1, v2]

    return PowerFT


PowerFT = ConstructPowerFT()


class GFcalc:
    def __init__(self, lattice, NNvect, rates, Nmax = 4):
        """
        Initialize GF calculator with NN vector lists and rates

        Parameters
        ----------
        lattice : array [:, :]
            lattice vectors defining periodicity; a[:,i] = coordinates of lattice vector a_i.
        NNvect : array [:, 3]
            list of nearest-neighbor vectors
        rates : array [:]
            list of escape rates
        Nmax : integer, optional
            how far out will we evaluate our GF? This determines the KPTmesh that will be used.

        Notes
        -----
        The Green Function calculator has a few support pieces for the calculation:
        * kpoint mesh (symmetrized to inside the irreducible wedge)
        * point group operations
        * D-FT function calculation, with 2nd and 4th derivatives
        * cached version of G calculated for stars
        """
        self.lattice = lattice
        self.NNvect = NNvect
        self.rates = rates
        self.kptmesh = KPTmesh.KPTmesh(lattice)
        # NOTE: as [b] = [a]^-T, the Cartesian group operations in reciprocal space
        # match real space, so we can piggy-back on the KPTmesh calculation of same
        self.groupops = self.kptmesh.groupops
        self.volume = self.kptmesh.volume
        self.DFT = DFTfunc(NNvect, rates)
        self.D2 = D2(NNvect, rates)
        self.di, self.ei = calcDE(self.D2)
        # get the rotated D15 (from D4) with p_i version, and FT version
        self.D15 = D4toNNN(RotateD4(D4(NNvect, rates), self.di, self.ei))
        self.D15FT = np.array([np.dot(PowerFT[i], self.D15) for i in xrange(3)])
        # determine pmax: find smallest p value at BZ faces, then
        self.pmax = np.sqrt(min([eval2(G, self.D2) for G in self.kptmesh.BZG])/
                            -np.log(1e-11))
        # discontinuity correction at R=0
        self.Gdisc0 = self.calcGsc_zero()
        self.Nmax = Nmax
        self.Nmesh = [0, 0, 0] # we haven't constructed anything just yet; wait till first call.
        self.Gcache = [] # our cached values
        self.Gsc_calced = False

    def calcGsc_zero(self):
        """
        Calculates the R=0 value of the discontinuity correction.
        """
        return self.volume*self.pmax**3/(15*8*np.pi**1.5*np.sqrt(np.prod(self.di)))*(
            3.*(self.D15[ExpToIndex[4, 0, 0]] +
                self.D15[ExpToIndex[0, 4, 0]] +
                self.D15[ExpToIndex[0, 0, 4]]) +
            (self.D15[ExpToIndex[0, 2, 2]] +
             self.D15[ExpToIndex[2, 0, 2]] +
             self.D15[ExpToIndex[2, 2, 0]])
        )

    def genmesh(self, Nmesh=None):
        """
        Generate the kpt mesh, if not already existing.

        Parameters
        ----------
        Nmesh : array [3], optional
            the mesh that we want to generate; otherwise, use self.Nmesh
        """
        regen = False
        if Nmesh != None:
            if not self.Nmesh == Nmesh:
                self.Nmesh = Nmesh
                regen = True
        if self.Nmesh == [0, 0, 0]:
            self.Nmesh = [4*self.Nmax, 4*self.Nmax, 4*self.Nmax]
            regen = True
        if regen:
            self.kptmesh.genmesh(self.Nmesh)
            self.kpt, self.wts = self.kptmesh.symmesh()
            self.Gsc = np.zeros(self.kptmesh.Nkptsym)
            self.Gsc_calced = False
            self.Gcache = [] # our cached values

    def calccoskR(self, R):
        """
        Calculates cos(k.R) for vector R, but with all of the pt group ops on k.
        It also does a few "behind-the-scenes" checks:
        1. If the mesh doesn't even exist, we need to construct it.
        2. If the mesh already exists, it may not be sufficient, which requires reconstruction
        Mesh reconstruction uses Nmax and this value of R to make sure we're good. For best
        results, Nmax should be a reasonable "upper limit," so that we only construct the mesh
        once for a series of R's.
        After we construct the mesh, we create space for Gsc, but leave it empty.

        Parameters
        ----------
        R : array [3]
            vector (cartesian) where we'll calculate k.R

        Returns
        -------
        coskR : array [:]
            symmetrized value of cos(k.R) for each k-point
        """
        # TODO determine if we need a more accurate mesh for a given R value
        # determine if we need to recalculate everything
        self.genmesh() # make sure we have a mesh
        gRlist = [np.dot(g, R) for g in self.groupops]
        # for g in self.groupops:
        #     gRvec = np.dot(g, R)
        #     if not any([np.all(np.abs(gRvec, vec)<1e-8) for vec in gRlist]):
        #         gRlist.append(gRvec)
        coskR = np.array([np.average([np.cos(np.dot(k, gR)) for gR in gRlist]) for k in self.kpt])
        # check that kR will not produce aliasing errors: that requires that the smallest
        # non-zero value of k.R be smaller than pi/2
        # ... check goes here.
        return coskR

    def calcGsc(self):
        """
        Calculates the semi-continuum correction (G - second-order pole - discontinuity)

        Checks first that we haven't already calculated this; if so, it does nothing.
        Else, it calculated for every k-point in the irreducible wedge.
        """
        self.genmesh() # make sure we have a mesh
        if self.Gsc_calced : return
        for i, k in enumerate(self.kpt):
            if np.dot(k, k) == 0:
                self.Gsc[i] = -1./self.pmax**2
            else:
                pi, pmagn = pnorm(self.di, self.ei, k)
                self.Gsc[i] = 1./self.DFT(k) + \
                              np.exp(-(pmagn/self.pmax)**2)*(1./(pmagn**2)
                                                             + np.dot(self.D15, powereval(pi)))
        self.Gsc_calced = True

    def GF(self, R):
        """
        Evaluate the GF at point R. Takes advantage of caching with stars.

        Parameters
        ----------
        R : array [3]
            lattice position to evalute GF

        Returns
        -------
        G : double
            GF at point R
        """
        # check against the cache first:
        # Gcache = [ [R.R, [R1, R2, ...], gvalue], ... ]
        for gcache in [ gc for gc in self.Gcache if gc[0] == np.dot(R,R)]:
            if any(np.all(abs(vec - R) < 1e-8) for vec in gcache[1]):
                return gcache[2]
        coskR = self.calccoskR(R)
        self.calcGsc() # in case we don't have a cached version of this (G-G2-G4)
        # our steps are
        # 0. calculate the IFT of Gsc
        Gsc = sum(coskR * self.Gsc * self.wts)
        # 1. calculate the IFT of the 2nd order pole
        ui, umagn = unorm(self.di, self.ei, R)
        G2 = self.volume * poleFT(self.di, umagn, self.pmax)
        # 2. calculate the IFT of the 4th order pole
        if umagn > 0:
            G4 = self.volume * np.dot(discFT(self.di, umagn, self.pmax),
                                      np.dot(self.D15FT, powereval(ui)))
        else:
            G4 = self.Gdisc0
        # 3. create a cached value, and return G.
        G = Gsc - G2 - G4
        gcache = [np.dot(R, R), [R], G]
        for Rp in [np.dot(g, R) for g in self.groupops]:
            if not any(all(Rp == x) for x in gcache[1]):
                gcache[1].append(Rp)
        self.Gcache.append(gcache)
        return G
