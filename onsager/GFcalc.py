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
from onsager import PowerExpansion as PE
import itertools
from copy import deepcopy
from numpy import linalg as LA
from scipy.special import hyp1f1, gamma, expi #, gammainc

# two quick shortcuts
T3D, T2D = PE.Taylor3D, PE.Taylor2D
factorial = PE.factorial


# Some "helper objects"; mostly collected here so that YAML has full access to them

class Fnl_p(object):
    def __init__(self, n, pm):
        """
        Exponential cutoff function in Fourier space (p)

        :param n: power
        :param pm: pmax value
        """
        self.n = n
        self.inv_pmax = 1 / pm

    def __call__(self, p):
        return (p ** self.n) * np.exp(-(p * self.inv_pmax) ** 2)


class Fnl_u(object):
    def __init__(self, n, l, pm, prefactor, d=3):
        """
        Inverse Fourier transform of exponential cutoff function into real space (u)
        for 3d and 2d

        :param n: power > -2
        :param l: angular momentum >= 0
        :param pm: pmax value
        :param prefactor: V/sqrt(prod_i d_i)
        :param d: dimensionality == 2, 3
        """
        self.a = (d + l + n) / 2
        self.b = d / 2 + l
        self.l = l
        self.half_pm = 0.5 * pm
        self.log = (self.a == 0)  # (n == -2 and l == 0 and d == 2)
        self.pre = (-1j) ** l * prefactor * (pm ** (d + n + l)) * gamma(self.a) / \
                   ((np.pi ** (d/2)) * (2 ** (d + l)) * gamma(self.b)) if not self.log else \
                   prefactor/(2*np.pi)

    def __call__(self, u):
        # return self.pre * u ** self.l * hyp1f1(self.a, self.b, -(u * self.half_pm) ** 2)
        if not self.log:
            return self.pre * u ** self.l * hyp1f1(self.a, self.b, -(u * self.half_pm) ** 2)
        else:
            if u == 0:
                return self.pre * (-0.5*np.euler_gamma + np.log(self.half_pm))
            else:
                # incomplete Gamma(0,x) = -Ei(-x) (exponential integral), turns out...
                # return self.pre * (-np.euler_gamma - np.log(u) -0.5*gammainc(0, (u*self.half_pm)**2))
                return self.pre * (-np.euler_gamma - np.log(u) + 0.5*expi(-(u*self.half_pm)**2))


class GFCrystalcalc(object):
    """
    Class calculator for the Green function, designed to work with the Crystal class.

    This computes the bare vacancy GF. It requires a crystal, chemical identity for the
    vacancy, list of symmetry unique sites (to define energies / entropies uniquely), and
    a corresponding jumpnetwork for that vacancy.
    """

    def __init__(self, crys, chem, sitelist, jumpnetwork, Nmax=4, kptwt = None):
        """
        Initializes our calculator with the appropriate topology / connectivity. Doesn't
        require, at this point, the site probabilities or transition rates to be known.

        :param crys: Crystal object
        :param chem: index identifying the diffusing species
        :param sitelist: list, grouped into Wyckoff common positions, of unique sites
        :param jumpnetwork: list of unique transitions as lists of ((i,j), dx)
        :param Nmax: maximum range as estimator for kpt mesh generation
        :param kptwt: (optional) tuple of (kpts, wts) to short-circuit kpt mesh generation
        """
        # this is really just used by loadHDF5() to circumvent __init__
        if all(x is None for x in (crys, chem, sitelist, jumpnetwork)): return
        self.crys = crys
        self.chem = chem
        self.sitelist = sitelist.copy()
        # self.N = sum(1 for w in sitelist for i in w)
        # self.invmap = [0 for w in sitelist for i in w]
        self.N = sum(len(w) for w in sitelist)
        self.Ndiff = self.networkcount(jumpnetwork, self.N)
        # if self.Ndiff>1:
        #     raise NotImplementedError('Cannot currently have {} disconnected networks'.format(self.Ndiff))
        self.invmap = np.zeros(self.N, dtype=int)
        for ind, w in enumerate(sitelist):
            for i in w:
                self.invmap[i] = ind
        self.NG = len(self.crys.G)  # number of group operations
        self.grouparray, self.indexpair = self.BreakdownGroups()
        # note: currently, we don't store jumpnetwork. If we want to rewrite the class
        # to allow a new kpoint mesh to be generated "on the fly", we'd need to store
        # a copy for regeneration
        # self.jumpnetwork = jumpnetwork
        # generate a kptmesh: now we try to make the mesh more "uniform" ??
        bmagn = np.array([np.sqrt(np.dot(crys.reciplatt[:, i], crys.reciplatt[:, i]))
                          for i in range(self.crys.dim)])
        bmagn /= np.power(np.product(bmagn), 1 / self.crys.dim)
        # make sure we have even meshes
        self.kptgrid = np.array([2 * np.int(np.ceil(2 * Nmax * b)) for b in bmagn], dtype=int) \
            if kptwt is None else np.zeros(self.crys.dim, dtype=int)
        self.kpts, self.wts = crys.reducekptmesh(crys.fullkptmesh(self.kptgrid)) \
            if kptwt is None else deepcopy(kptwt)
        self.Nkpt = self.kpts.shape[0]
        # generate the Fourier transformation for each jump
        # also includes the multiplicity for the onsite terms (site expansion)
        self.FTjumps, self.SEjumps = self.FourierTransformJumps(jumpnetwork, self.N, self.kpts)
        # generate the Taylor expansion coefficients for each jump
        self.Taylorjumps = self.TaylorExpandJumps(jumpnetwork, self.N)
        # tuple of the Wyckoff site indices for each jump (needed to make symmrate)
        self.jumppairs = tuple((self.invmap[jumplist[0][0][0]], self.invmap[jumplist[0][0][1]])
                               for jumplist in jumpnetwork)
        self.D, self.eta = 0, 0  # we don't yet know the diffusivity

    @staticmethod
    def networkcount(jumpnetwork, N):
        """Return a count of how many separate connected networks there are"""
        jngraph = np.zeros((N, N), dtype=bool)
        for jlist in jumpnetwork:
            for (i, j), dx in jlist:
                jngraph[i,j] = True
        connectivity = 0  # had been a list... if we want to return the list of sets
        disconnected = {i for i in range(N)}
        while len(disconnected)>0:
            # take the "first" element out, and find everything it's connected to:
            i = min(disconnected)
            cset = {i}
            disconnected.remove(i)
            while True:
                clen = len(cset)
                for n in cset.copy():
                    for m in disconnected.copy():
                        if jngraph[n,m]:
                            cset.add(m)
                            disconnected.remove(m)
                # check if we've stopped adding new members:
                if clen == len(cset): break
            connectivity += 1
            # connectivity.append(cset)  # if we want to keep lists of connectivity sets
        return connectivity

    # this is part of our *class* definition:
    __HDF5list__ = ('N', 'Ndiff', 'invmap', 'NG', 'grouparray', 'indexpair', 'kptgrid',
                    'kpts', 'wts', 'Nkpt', 'FTjumps', 'SEjumps')

    def __str__(self):
        return 'GFcalc for crystal (chemistry={}):\n{}\nkpt grid: {} ({})'.format(self.chem,
                                                                                  self.crys,
                                                                                  self.kptgrid,
                                                                                  self.Nkpt)

    def addhdf5(self, HDF5group):
        """
        Adds an HDF5 representation of object into an HDF5group (needs to already exist).

        Example: if f is an open HDF5, then GFcalc.addhdf5(f.create_group('GFcalc')) will
        (1) create the group named 'GFcalc', and then (2) put the GFcalc representation in that group.

        :param HDF5group: HDF5 group
        """
        HDF5group.attrs['type'] = self.__class__.__name__
        HDF5group.attrs['crystal'] = self.crys.__repr__()
        HDF5group.attrs['chem'] = self.chem
        # arrays that we can deal with:
        for internal in self.__HDF5list__:
            HDF5group[internal] = getattr(self, internal)
        # note: we don't store sitelist; we reconstruct it from invmap
        # we need to deal with Taylorjumps and jumppairs separately
        NTaylorjumps = len(self.Taylorjumps)
        TaylorTag = 'T3D' if self.crys.dim == 3 else 'T2D'
        HDF5group['N' + TaylorTag + 'jumps'] = NTaylorjumps
        for i, t3d in enumerate(self.Taylorjumps):
            coeffstr = TaylorTag + 'jump-{}'.format(i)
            t3d.addhdf5(HDF5group.create_group(coeffstr))
        HDF5group['jumppairs'] = np.array(self.jumppairs)

    @classmethod
    def loadhdf5(cls, crys, HDF5group):
        """
        Creates a new GFcalc from an HDF5 group.

        :param crys: crystal object--MUST BE PASSED IN as it is not stored with the GFcalc
        :param HDFgroup: HDF5 group
        :return GFcalc: new GFcalc object
        """
        GFcalc = cls(None, None, None, None)  # initialize
        GFcalc.crys = crys
        GFcalc.chem = HDF5group.attrs['chem']
        for internal in cls.__HDF5list__:
            setattr(GFcalc, internal, HDF5group[internal].value)
        GFcalc.Taylorjumps = []
        Taylor = T3D if crys.dim == 3 else T2D
        TaylorTag = 'T3D' if crys.dim == 3 else 'T2D'
        for i in range(HDF5group['N' + TaylorTag + 'jumps'].value):
            coeffstr = TaylorTag + 'jump-{}'.format(i)
            GFcalc.Taylorjumps.append(Taylor.loadhdf5(HDF5group[coeffstr]))
        # construct sitelist and jumppairs
        GFcalc.sitelist = [[] for i in range(max(GFcalc.invmap) + 1)]
        for i, site in enumerate(GFcalc.invmap):
            GFcalc.sitelist[site].append(i)
        GFcalc.jumppairs = tuple((pair[0], pair[1]) for pair in HDF5group['jumppairs'])
        GFcalc.D, GFcalc.eta = 0, 0  # we don't yet know the diffusivity
        return GFcalc

    def FourierTransformJumps(self, jumpnetwork, N, kpts):
        """
        Generate the Fourier transform coefficients for each jump

        :param jumpnetwork: list of unique transitions, as lists of ((i,j), dx)
        :param N: number of sites
        :param kpts: array[Nkpt][3], in Cartesian (same coord. as dx)
        :return FTjumps: array[Njump][Nkpt][Nsite][Nsite] of FT of the jump network
        :return SEjumps: array[Nsite][Njump] multiplicity of jump on each site
        """
        FTjumps = np.zeros((len(jumpnetwork), self.Nkpt, N, N), dtype=complex)
        SEjumps = np.zeros((N, len(jumpnetwork)), dtype=int)
        for J, jumplist in enumerate(jumpnetwork):
            for (i, j), dx in jumplist:
                FTjumps[J, :, i, j] += np.exp(1.j * np.dot(kpts, dx))
                SEjumps[i, J] += 1
        return FTjumps, SEjumps

    def TaylorExpandJumps(self, jumpnetwork, N):
        """
        Generate the Taylor expansion coefficients for each jump

        :param jumpnetwork: list of unique transitions, as lists of ((i,j), dx)
        :param N: number of sites
        :return T3Djumps: list of Taylor3D expansions of the jump network
        """
        Taylor = T3D if self.crys.dim == 3 else T2D
        Taylor()  # need to do just to initialize the class; if already initialized, won't do anything
        # Taylor expansion coefficients for exp(1j*x) = (1j)^n/n!
        pre = np.array([(1j) ** n / factorial(n, True) for n in range(Taylor.Lmax + 1)])
        Taylorjumps = []
        for jumplist in jumpnetwork:
            # coefficients; we use tuples because we'll be successively adding to the coefficients in place
            c = [(n, n, np.zeros((Taylor.powlrange[n], N, N), dtype=complex)) for n in range(Taylor.Lmax + 1)]
            for (i, j), dx in jumplist:
                pexp = Taylor.powexp(dx, normalize=False)
                for n in range(Taylor.Lmax + 1):
                    (c[n][2])[:, i, j] += pre[n] * (Taylor.powercoeff[n] * pexp)[:Taylor.powlrange[n]]
            Taylorjumps.append(Taylor(c))
        return Taylorjumps

    def BreakdownGroups(self):
        """
        Takes in a crystal, and a chemistry, and constructs the indexing breakdown for each
        (i,j) pair.
        :return grouparray: array[NG][3][3] of the NG group operations
        :return indexpair: array[N][N][NG][2] of the index pair for each group operation
        """
        grouparray = np.zeros((self.NG, self.crys.dim, self.crys.dim))
        indexpair = np.zeros((self.N, self.N, self.NG, 2), dtype=int)
        for ng, g in enumerate(self.crys.G):
            grouparray[ng, :, :] = g.cartrot[:, :]
            indexmap = g.indexmap[self.chem]
            for i in range(self.N):
                for j in range(self.N):
                    indexpair[i, j, ng, 0], indexpair[i, j, ng, 1] = indexmap[i], indexmap[j]
        return grouparray, indexpair

    def SymmRates(self, pre, betaene, preT, betaeneT):
        """Returns a list of lists of symmetrized rates, matched to jumpnetwork"""
        return np.array([pT * np.exp(0.5 * betaene[w0] + 0.5 * betaene[w1] - beT) / np.sqrt(pre[w0] * pre[w1])
                         for (w0, w1), pT, beT in zip(self.jumppairs, preT, betaeneT)])

    def SetRates(self, pre, betaene, preT, betaeneT, pmaxerror=1.e-8):
        """
        (Re)sets the rates, given the prefactors and Arrhenius factors for the sites and
        transitions, using the ordering according to sitelist and jumpnetwork. Initiates all of
        the calculations so that GF calculation is (fairly) efficient for each input.

        :param pre: list of prefactors for site probabilities
        :param betaene: list of beta*E (energy/kB T) for each site
        :param preT: list of prefactors for transition states
        :param betaeneT: list of beta*ET (energy/kB T) for each transition state
        :param pmaxerror: parameter controlling error from pmax value. Should be same order as integration error.
        """
        self.symmrate = self.SymmRates(pre, betaene, preT, betaeneT)
        self.maxrate = self.symmrate.max()
        self.symmrate /= self.maxrate
        self.escape = -np.diag([sum(self.SEjumps[i, J] * pretrans / pre[wi] * np.exp(betaene[wi] - BET)
                                    for J, pretrans, BET in zip(itertools.count(), preT, betaeneT))
                                for i, wi in enumerate(self.invmap)]) / self.maxrate
        self.omega_qij = np.tensordot(self.symmrate, self.FTjumps, axes=(0, 0))
        self.omega_qij[:] += self.escape  # adds it to every point
        self.omega_Taylor = sum(symmrate * expansion
                                for symmrate, expansion in zip(self.symmrate, self.Taylorjumps))
        self.omega_Taylor += self.escape
        Taylor = T3D if self.crys.dim == 3 else T2D

        # 1. Diagonalize gamma point value; use to rotate to diffusive / relaxive, and reduce
        self.r, self.vr = self.DiagGamma()
        if not np.allclose(self.r[:self.Ndiff], 0):
            raise ArithmeticError("Did not find {} equilibrium solution to rates?".format(self.Ndiff))
        self.omega_Taylor_rotate = (self.omega_Taylor.ldot(self.vr.T)).rdot(self.vr)
        oT_dd, oT_dr, oT_rd, oT_rr, oT_D, etav = self.BlockRotateOmegaTaylor(self.omega_Taylor_rotate)
        # 2. Calculate D and eta
        self.D = self.Diffusivity(oT_D)
        self.eta = self.biascorrection(etav)
        # 3. Spatially rotate the Taylor expansion
        self.d, self.e = LA.eigh(self.D / self.maxrate)
        # had been 1e-11; changed to 1e-7 to reflect likely integration accuracy of k-point grids
        self.pmax = np.sqrt(min([np.dot(G, np.dot(G, self.D / self.maxrate)) for G in self.crys.BZG]) / -np.log(pmaxerror))
        self.qptrans = self.e.copy()
        self.pqtrans = self.e.T.copy()
        self.uxtrans = self.e.T.copy()
        for i in range(self.crys.dim):
            self.qptrans[:, i] /= np.sqrt(self.d[i])
            self.pqtrans[i, :] *= np.sqrt(self.d[i])
            self.uxtrans[i, :] /= np.sqrt(self.d[i])
        powtrans = Taylor.rotatedirections(self.qptrans)
        for t in [oT_dd, oT_dr, oT_rd, oT_rr, oT_D]:
            t.irotate(powtrans)  # rotate in place
            t.reduce()
        if oT_D.coefflist[0][1] != 0: raise ArithmeticError("Problem isotropizing D?")
        # 4. Invert Taylor expansion using block inversion formula, and truncate at n=0
        gT_rotate = self.BlockInvertOmegaTaylor(oT_dd, oT_dr, oT_rd, oT_rr, oT_D)
        self.g_Taylor = (gT_rotate.ldot(self.vr)).rdot(self.vr.T)
        self.g_Taylor.separate()
        g_Taylor_fnlp = {(n, l): Fnl_p(n, self.pmax) for (n, l) in self.g_Taylor.nl()}
        prefactor = self.crys.volume / np.sqrt(np.product(self.d))
        self.g_Taylor_fnlu = {(n, l): Fnl_u(n, l, self.pmax, prefactor, d=self.crys.dim)
                              for (n, l) in self.g_Taylor.nl()}
        # 5. Invert Fourier expansion
        gsc_qij = np.zeros_like(self.omega_qij)
        for qind, q in enumerate(self.kpts):
            if np.allclose(q, 0):
                # gamma point... need to treat separately
                gsc_qij[qind] = (-1 / self.pmax ** 2) * \
                                sum(np.outer(self.vr[:, n], self.vr[:, n])
                                    for n in range(self.Ndiff))
            else:
                # invert, subtract off Taylor expansion to leave semicontinuum piece
                gsc_qij[qind] = np.linalg.inv(self.omega_qij[qind, :, :]) \
                                - self.g_Taylor(np.dot(self.pqtrans, q), g_Taylor_fnlp)
        # 6. Slice the pieces we want for fast(er) evaluation (since we specify i and j in evaluation)
        self.gsc_ijq = np.zeros((self.N, self.N, self.Nkpt), dtype=complex)
        for i in range(self.N):
            for j in range(self.N):
                self.gsc_ijq[i, j, :] = gsc_qij[:, i, j]
        # since we can't make an array, use tuples of tuples to do gT_ij[i][j]
        self.gT_ij = tuple(tuple(self.g_Taylor[i, j].copy().reduce().separate()
                                 for j in range(self.N))
                           for i in range(self.N))

    def exp_dxq(self, dx):
        """
        Return the array of exp(-i q.dx) evaluated over the q-points, and accounting for symmetry

        :param dx: vector
        :return exp(-i q.dx): array of :math:`\\exp(-i \\cdot dx)`
        """
        # kpts[k,3] .. g_dx_array[NR, 3]
        return np.exp(-1j * np.tensordot(self.kpts, dx, axes=(1, 0)))

    def __call__(self, i, j, dx):
        """
        Evaluate the Green function from site i to site j, separated by vector dx

        :param i: site index
        :param j: site index
        :param dx: vector pointing from i to j (can include lattice contributions)
        :return G: Green function value
        """
        if self.D is 0: raise ValueError("Need to SetRates first")
        # evaluate Fourier transform component (now with better space group treatment!)
        gIFT = 0
        for gop, pair in zip(self.grouparray, self.indexpair[i][j]):
            gIFT += np.dot(self.wts, self.gsc_ijq[pair[0], pair[1]] * self.exp_dxq(np.dot(gop, dx)))
        gIFT /= self.NG
        if not np.isclose(gIFT.imag, 0): raise ArithmeticError("Got complex IFT? {}".format(gIFT))
        # evaluate Taylor expansion component:
        gTaylor = self.gT_ij[i][j](np.dot(self.uxtrans, dx), self.g_Taylor_fnlu)
        if not np.isclose(gTaylor.imag, 0): raise ArithmeticError("Got complex IFT from Taylor? {}".format(gTaylor))
        # combine:
        return (gIFT + gTaylor).real / self.maxrate

    def DiagGamma(self, omega=None):
        """
        Diagonalize the gamma point (q=0) term

        :param omega: optional; the Taylor expansion to use. If None, use self.omega_Taylor
        :return r: array of eigenvalues, sorted from 0 to decreasing values.
        :return vr: array of eigenvectors where vr[:,i] is the vector for eigenvalue r[i]
        """
        if omega is None:
            omega = self.omega_Taylor
        gammacoeff = None
        for (n, l, coeff) in omega.coefflist:
            if n < 0: raise ValueError("Taylor expansion has terms below n=0?")
            if n == 0:
                if l != 0: raise ValueError("n=0 term has angular dependence? l != 0")
                gammacoeff = -coeff[0].real
                break
        if gammacoeff is None:
            # missing onsite term--indicates that it's been reduced to 0
            # should ONLY happen if we have a Bravais lattice, e.g.
            gammacoeff = np.zeros((self.N, self.N)) #, dtype=complex)
        r, vr = LA.eigh(gammacoeff)
        return -r, vr

    def Diffusivity(self, omega_Taylor_D=None):
        """
        Return the diffusivity, or compute it if it's not already known. Uses omega_Taylor_D
        to compute with maximum efficiency.

        :param omega_Taylor_D: Taylor expansion of the diffusivity component
        :return D: diffusivity [3,3] array
        """
        if self.D is not 0 and omega_Taylor_D is None: return self.D
        if self.D is 0 and omega_Taylor_D is None: raise ValueError("Need omega_Taylor_D value")
        Taylor = T3D if self.crys.dim == 3 else T2D
        D = np.zeros((self.crys.dim, self.crys.dim))
        for (n, l, c) in omega_Taylor_D.coefflist:
            if n < 2: raise ValueError("Reduced Taylor expansion for D doesn't begin with n==2")
            DTr = np.trace(c.real, axis1=1, axis2=2)/self.Ndiff
            if n == 2:
                # first up: constant term (if present)
                D += np.eye(self.crys.dim) * DTr[0]
                # next: l == 2 contributions
                if l >= 2:
                    # done in this way so that we get the 1/2 for the off-diagonal, and the 1 for diagonal
                    for t in ((i, j) for i in range(self.crys.dim) for j in range(i, self.crys.dim)):
                        tupind = tuple(t.count(d) for d in range(self.crys.dim))
                        ind = Taylor.pow2ind[tupind]  # count the powers
                        D[t] += 0.5 * DTr[ind]
                        D[t[1], t[0]] += 0.5 * DTr[ind]
        # note: the "D" constructed this way will be negative! (as it is -q.D.q)
        return -D * self.maxrate

    def biascorrection(self, etav=None):
        """
        Return the bias correction, or compute it if it's not already known. Uses etav to compute.

        :param etav: Taylor expansion of the bias correction
        :return eta: [N,3] array
        """
        if etav is None: return self.eta
        # a little bit of a hack: we keep the implicit vr[:,0] part, that's the square root of
        # probability that comes from the diagonalization at q=0, but it might be negative!
        rhosign = [1. if sum(self.vr[:, n0])>0 else -1. for n0 in range(self.Ndiff)]
        Taylor = T3D if self.crys.dim == 3 else T2D
        d_ind_list = [(d, Taylor.pow2ind[(0,)*d + (1,) + (0,)*(self.crys.dim-1-d)])
                       for d in range(self.crys.dim)]
        eta = np.zeros((self.N, self.crys.dim))
        if etav == 0: return eta
        for (n, l, c) in etav.coefflist:
            if n < 1: raise ValueError("Reduced Taylor expansion for etav doesn't begin with n==1")
            if n == 1:
                if l >= 1:
                    for d, ind in d_ind_list:
                        eta[:, d] -= sum(rhosign[n0]*np.dot(self.vr[:, self.Ndiff:], c[ind, :])[:, n0].imag
                                         for n0 in range(self.Ndiff)) / self.Ndiff
        return eta

    def BlockRotateOmegaTaylor(self, omega_Taylor_rotate):
        """
        Returns block partitioned Taylor expansion of a rotated omega Taylor expansion.

        :param omega_Taylor_rotate: rotated into diffusive [0] / relaxive [1:] basis
        :return dd: diffusive/diffusive block (upper left)
        :return dr: diffusive/relaxive block (lower left)
        :return rd: relaxive/diffusive block (upper right)
        :return rr: relaxive/relaxive block (lower right)
        :return D: :math:`dd - dr (rr)^{-1} rd` (diffusion)
        :return etav: :math:`(rr)^{-1} rd` (relaxation vector)
        """
        Taylor = T3D if self.crys.dim == 3 else T2D
        ND = self.Ndiff  # previously had been 1.
        dd = omega_Taylor_rotate[0:ND, 0:ND].copy()
        dr = omega_Taylor_rotate[0:ND, ND:].copy()
        rd = omega_Taylor_rotate[ND:, 0:ND].copy()
        rr = omega_Taylor_rotate[ND:, ND:].copy()
        for t in [dd, dr, rd, rr]: t.reduce()
        if self.N > ND:
            D = dd - dr * rr.inv() * rd
            etav = rr.inv() * rd
            etav.truncate(1, inplace=True)
        else:
            D = dd.copy()
            etav = 0
        D.truncate(Taylor.Lmax, inplace=True)
        D.reduce()
        return dd, dr, rd, rr, D, etav

    def BlockInvertOmegaTaylor(self, dd, dr, rd, rr, D):
        """
        Returns block inverted omega as a Taylor expansion, up to Nmax = 0 (discontinuity
        correction). Needs to be rotated such that leading order of D is isotropic.

        :param dd: diffusive/diffusive block (upper left)
        :param dr: diffusive/relaxive block (lower left)
        :param rd: relaxive/diffusive block (upper right)
        :param rr: relaxive/relaxive block (lower right)
        :param D: :math:`dd - dr (rr)^{-1} rd` (diffusion)
        :return gT: Taylor expansion of g in block form, and reduced (collected terms)
        """
        Taylor = T3D if self.crys.dim == 3 else T2D
        ND = self.Ndiff  # previously had been 1.
        gT = Taylor.zeros(-2, 0, (self.N, self.N))  # where we'll place our Taylor expansion
        D_inv = D.inv()
        gT[0:ND, 0:ND] = D_inv.truncate(0)
        if self.N > ND:
            rr_inv = rr.inv()
            gT[0:ND, ND:] = -(D_inv * dr * rr_inv).truncate(0)
            gT[ND:, 0:ND] = -(rr_inv * rd * D_inv).truncate(0)
            gT[ND:, ND:] = (rr_inv + rr_inv * rd * D_inv * dr * rr_inv).truncate(0)
        return gT.reduce()
