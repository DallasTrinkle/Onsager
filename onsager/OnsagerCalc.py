"""
Onsager calculator module

Class to create an Onsager "calculator", which brings two functionalities:
1. determines *what* input is needed to compute the Onsager (mobility, or L) tensors
2. constructs the function that calculates those tensors, given the input values.

This class is designed to be combined with code that can, e.g., automatically
run some sort of atomistic-scale (DFT, classical potential) calculation of site
energies, and energy barriers, and then in concert with scripts to convert such data
into rates and probabilities, this will allow for efficient evaluation of transport
coefficients.

This implementation will be for vacancy-mediated solute diffusion on a Bravais lattice,
and assumes the dilute limit. The mathematics is based on a Green function solution
for the vacancy diffusion. The computation of the GF is outside the scope of this
particular module; however, clever uses of the GFcalc module can be attached to this
quite easily.
"""

# TODO: need to make sure we can read / write Onsager comp. information for optimal functionality (JSON?)

__author__ = 'Dallas R. Trinkle'

import numpy as np
from scipy.linalg import pinv2, solve
import stars
import GFcalc
import crystal


class Interstitial(object):
    """
    A class to compute interstitial diffusivity; uses structure of crystal to do most
    of the heavy lifting in terms of symmetry, etc.
    """
    def __init__(self, crys, chem, sitelist, jumpnetwork):
        """
        Initialization; takes an underlying crystal, a choice of atomic chemistry,
        a corresponding Wyckoff site list and jump network.

        Parameters
        ----------
        crys : Crystal object

        chem : integer
            index into the basis of crys, corresponding to the chemical element that hops

        sitelist : list of lists of indices
            site indices where the atom may hop; grouped by symmetry equivalency

        jumpnetwork : list of lists of tuples: ( (i, j), dx )
            symmetry unique transitions; each list is all of the possible transitions
            from site i to site j with jump vector dx; includes i->j and j->i
        """
        self.crys = crys
        self.chem = chem
        self.sitelist = sitelist
        self.N = sum(1 for w in sitelist for i in w)
        self.invmap = [0 for w in sitelist for i in w]
        for ind,w in enumerate(sitelist):
            for i in w:
                self.invmap[i] = ind
        self.jumpnetwork = jumpnetwork
        self.VectorBasis, self.VV = self.generateVectorBasis()
        self.NV = len(self.VectorBasis)
        # quick check to see if our projected omega matrix will be invertible
        # only really needed if we have a non-empty vector basis
        self.omega_invertible = True
        if self.NV > 0:
            # invertible if inversion is present
            self.omega_invertible = any( np.allclose(g.cartrot, -np.eye(3)) for g in crys.G )
            # self.omega_invertible = all( np.all(np.isclose(np.sum(v, axis=0), np.zeros(3)))
            #                              for v in self.VectorBasis)

    @staticmethod
    def sitelistYAML(sitelist):
        """Dumps a "sample" YAML formatted version of the sitelist with data to be entered"""
        return crystal.yaml.dump({'Dipole': [np.zeros(3,3) for w in sitelist],
                                  'Energy': [0 for w in sitelist],
                                  'Prefactor': [1 for w in sitelist],
                                  'sitelist': sitelist})

    @staticmethod
    def jumpnetworkYAML(jumpnetwork):
        """Dumps a "sample" YAML formatted version of the jumpnetwork with data to be entered"""
        return crystal.yaml.dump({'DipoleT': [np.zeros(3,3) for t in jumpnetwork],
                                  'EnergyT': [0 for t in jumpnetwork],
                                  'PrefactorT': [1 for t in jumpnetwork],
                                  'jumpnetwork': jumpnetwork})

    def generateVectorBasis(self):
        """
        Generate our full vector basis, using the information from our crystal
        :return: list of our unique vector basis lattice functions, normalized
        """
        def vectlist(vb):
            """Returns a list of orthonormal vectors corresponding to our vector basis
            :param vb: (dim, v)
            :return: list of vectors
            """
            if vb[0] == 0: return []
            if vb[0] == 1: return [vb[1]]
            if vb[0] == 2:
                # now, construct the other two directions:
                norm = vb[1]
                if abs(norm[2]) < 0.75:
                    v1 = np.array([norm[1], -norm[0], 0])
                else:
                    v1 = np.array([-norm[2], 0, norm[0]])
                v1 /= np.sqrt(np.dot(v1, v1))
                v2 = np.cross(norm, v1)
                return [v1, v2]
            if vb[0] == 3: return [np.array([1.,0.,0.]),
                                   np.array([0.,1.,0.]),
                                   np.array([0.,0.,1.])]

        lis = []
        lisVV = []
        for s in self.sitelist:
            for v in vectlist(self.crys.VectorBasis((self.chem, s[0]))):
                v /= np.sqrt(len(s)) # additional normalization
                # we have some constructing to do... first, make the vector we want to use
                vb = np.zeros((self.N, 3))
                for g in self.crys.G:
                    # what site do we land on, and what's the vector? (this is slight overkill)
                    vb[g.indexmap[self.chem][s[0]]] = self.crys.g_direc(g, v)
                lis.append(vb)
                lisVV.append(np.dot(vb.T, vb))
        return lis, lisVV

    def siteprob(self, pre, betaene):
        """Returns our site probabilities, normalized, as a vector"""
        rho = np.array([ pre[w]*np.exp(-betaene[w]) for i,w in enumerate(self.invmap)])
        return rho/sum(rho)

    def ratelist(self, pre, betaene, preT, betaeneT):
        """Returns a list of lists of rates, matched to jumpnetwork"""
        # the ij tuple in each transition list is the i->j pair
        # invmap[i] tells you which Wyckoff position i maps to (in the sitelist)
        invrho = np.array([ np.exp(betaene[w])/pre[w] for i,w in enumerate(self.invmap)])
        return [ [ arr*invrho[i] for (i,j), dx in t ]
            for t, arr in zip(self.jumpnetwork,
                              [ pT*np.exp(-beT)
                                for pT, beT in zip(preT, betaeneT) ])]

    def diffusivity(self, pre, betaene, preT, betaeneT):
        """
        Computes the diffusivity for our element given prefactors and energies/kB T.
        The input list order corresponds to the sitelist and jumpnetwork

        Parameters
        ----------
        pre : list of prefactors for unique sites
        betaene : list of site energies divided by kB T
        preT : list of prefactors for transition states
        betaeneT: list of transition state energies divided by kB T

        Returns
        -------
        D[3,3] : diffusivity as 3x3 tensor
        """
        if __debug__:
            if len(pre) != len(self.sitelist): raise IndexError("length of prefactor {} doesn't match sitelist".format(pre))
            if len(betaene) != len(self.sitelist): raise IndexError("length of energies {} doesn't match sitelist".format(betaene))
            if len(preT) != len(self.jumpnetwork): raise IndexError("length of prefactor {} doesn't match jump network".format(preT))
            if len(betaeneT) != len(self.jumpnetwork): raise IndexError("length of energies {} doesn't match jump network".format(betaeneT))
        rho = self.siteprob(pre, betaene)
        sqrtrho = np.sqrt(rho)
        invsqrtrho = 1./sqrtrho
        ratelist = self.ratelist(pre, betaene, preT, betaeneT)
        omega_ij = np.zeros((self.N, self.N))
        bias_i = np.zeros((self.N, 3))
        D0 = np.zeros((3,3))
        for transitionset, rates in zip(self.jumpnetwork, ratelist):
            for ((i,j), dx), rate in zip(transitionset, rates):
                omega_ij[i, j] += sqrtrho[i]*invsqrtrho[j]*rate
                omega_ij[i, i] -= rate
                bias_i[i] += sqrtrho[i]*rate*dx
                D0 += 0.5*np.outer(dx, dx)*rho[i]*rate
        if self.NV > 0:
            # NOTE: there's probably a SUPER clever way to do this with higher dimensional arrays and dot...
            omega_v = np.zeros((self.NV, self.NV))
            bias_v = np.zeros(self.NV)
            for a, va in enumerate(self.VectorBasis):
                bias_v[a] = np.trace(np.dot(bias_i.T, va))
                for b, vb in enumerate(self.VectorBasis):
                    omega_v[a,b] = np.trace(np.dot(va.T, np.dot(omega_ij, vb)))
            if self.omega_invertible:
                # invertible, so just use solve for speed:
                gamma_v = -solve(-omega_v, bias_v, sym_pos=True) # technically *negative* definite
            else:
                # pseudoinverse required:
                gamma_v = np.dot(pinv2(omega_v), bias_v)
            for b, g, VV in zip(bias_v, gamma_v, self.VV):
                D0 += b*g*VV
        return D0

    def elastodiffusion(self, pre, betaene, dipole, preT, betaeneT, dipoleT):
        """
        Computes the elastodiffusion tensor for our element given prefactors, energies/kB T,
        and elastic dipoles/kB T
        The input list order corresponds to the sitelist and jumpnetwork

        Parameters
        ----------
        pre : list of prefactors for unique sites
        betaene : list of site energies divided by kB T
        dipole: list of elastic dipoles divided by kB T
        preT : list of prefactors for transition states
        betaeneT: list of transition state energies divided by kB T
        dipoleT: list of elastic dipoles divided by kB T

        Returns
        -------
        D[3,3], dD[3,3,3,3] : diffusivity as 3x3 tensor and elastodiffusion tensor as 3x3x3x3 tensor
        """
        if __debug__:
            if len(pre) != len(self.sitelist): raise IndexError("length of prefactor {} doesn't match sitelist".format(pre))
            if len(betaene) != len(self.sitelist): raise IndexError("length of energies {} doesn't match sitelist".format(betaene))
            if len(dipole) != len(self.sitelist): raise IndexError("length of dipoles {} doesn't match sitelist".format(dipole))
            if len(preT) != len(self.jumpnetwork): raise IndexError("length of prefactor {} doesn't match jump network".format(preT))
            if len(betaeneT) != len(self.jumpnetwork): raise IndexError("length of energies {} doesn't match jump network".format(betaeneT))
            if len(dipoleT) != len(self.jumpnetwork): raise IndexError("length of dipoles {} doesn't match jump network".format(dipoleT))
        return np.eye(3), np.zeros((3,3,3,3))


class VacancyMediated(object):
    """
    A class to compute vacancy-mediated solute transport coefficients, specifically
    L_vv (vacancy diffusion), L_ss (solute), and L_sv (off-diagonal). As part of that,
    it determines *what* quantities are needed as inputs in order to perform this calculation.

    Currently for Bravais lattice only.
    """
    def __init__(self, jumpvect, groupops, Nthermo = 0):
        """
        Initialization; starts off with a set of jump vectors, group operations, and an optional
        range for thermodynamic interactions.

        Parameters
        ----------
        jumpvect : either list of array[3] or array[:, 3]
            list of jump vectors that are possible

        groupops : list of array[3, 3]
            point group operations

        Nthermo : integer, optional
            range of thermodynamic interactions, in terms of "shells", which is multiple
            summations of jumpvect
        """
        # make copies, as lists (there is likely a more efficient way to do this
        self.jumpvect = [v for v in jumpvect]
        self.groupops = [g for g in groupops]
        self.NNstar = stars.StarSet(self.jumpvect, self.groupops, 1)
        self.thermo = stars.StarSet(self.jumpvect, self.groupops)
        self.kinetic = stars.StarSet(self.jumpvect, self.groupops)
        self.GF = stars.StarSet(self.jumpvect, self.groupops)
        self.omega1 = stars.DoubleStarSet(self.kinetic)
        self.biasvec = stars.VectorStarSet(self.GF)
        self.Nthermo = 0
        self.generate(Nthermo)

    def generate(self, Nthermo, genmatrices=False):
        """
        Generate the necessary stars, double-stars, vector-stars, etc. based on the
        thermodynamic range.

        Parameters
        ----------
        Nthermo : integer
            range of thermodynamic interactions, in terms of "shells", which is multiple
            summations of jumpvect

        genmatrices : logical, optional
            if set, call generate matrices (default is to wait until needed)
        """
        if Nthermo == 0:
            self.Nthermo = 0
        if Nthermo == self.Nthermo:
            return
        self.thermo.generate(Nthermo)
        self.kinetic.combine(self.thermo, self.NNstar)
        self.omega1.generate(self.kinetic)
        # the following is a list of indices corresponding to the jump-type; so that if one
        # chooses *not* to calculate omega1, the corresponding omega0 value can be substituted
        # This is returned both for reference, and used for internal consumption
        self.omega1LIMB = [self.NNstar.starindex(self.kinetic.pts[p[0][0]] - self.kinetic.pts[p[0][1]])
                           for p in self.omega1.dstars]
        self.GF.combine(self.kinetic, self.kinetic)
        self.biasvec.generate(self.kinetic)
        # this is the list of points for the GF calculation; we need to add in the origin now:
        self.GFR = [np.zeros(3)] + [sR[0] for sR in self.GF.stars]
        self.matricesgenerated = False
        if genmatrices:
            self.generatematrices()

    def generatematrices(self, Nthermo=None):
        """
        Makes all of the pieces we need to calculate the diffusion. Called by Lij, but we store
        "cached" versions for efficiency.

        Parameters
        ----------
        Nthermo : integer, optional
            range of thermodynamic interactions, in terms of "shells", which is multiple
            summations of jumpvect; if set, call generate() first.
        """
        if Nthermo is not None:
            self.generate(Nthermo)
        if not self.matricesgenerated:
            self.matricesgenerated = True
            self.thermo2kin = [self.kinetic.starindex(Rs[0]) for Rs in self.thermo.stars]
            self.NN2thermo = [self.thermo.starindex(Rs[0]) for Rs in self.NNstar.stars]
            self.vstar2kin = [self.kinetic.starindex(Rs[0]) for Rs in self.biasvec.vecpos]
            self.GFexpansion = self.biasvec.GFexpansion(self.GF)
            self.rate0expansion = self.biasvec.rate0expansion(self.NNstar)
            self.rate1expansion = self.biasvec.rate1expansion(self.omega1)
            self.rate2expansion = self.biasvec.rate2expansion(self.NNstar)
            self.bias2expansion = self.biasvec.bias2expansion(self.NNstar)
            self.bias1ds, self.omega1ds, self.gen1prob, self.bias1NN, self.omega1NN = \
                self.biasvec.bias1expansion(self.omega1, self.NNstar)

    def omega0list(self, Nthermo = None):
        """
        Return a list of endpoints for a vacancy jump, corresponding to omega0: no solute.
        Note: omega0list and omega2list are, by definition, the same. Defined by Stars.

        Parameters
        ----------
        Nthermo : integer, optional
            if set to some value, then we call generate(Nthermo) first.

        Returns
        -------
        omega0list : list of array[3]
            list of endpoints for a vacancy jump: we will expect rates for jumps from
            the origin to each of these endpoints as inputs for our calculation
        """
        if Nthermo is not None:
            self.generate(Nthermo)
        return [s[0] for s in self.NNstar.stars]

    def interactlist(self, Nthermo = None):
        """
        Return a list of solute-vacancy configurations for interactions. The points correspond
        to a vector between a solute atom and a vacancy. Defined by Stars.

        Parameters
        ----------
        Nthermo : integer, optional
            if set to some value, then we call generate(Nthermo) first.

        Returns
        -------
        interactlist : list of array[3]
            list of vectors to connect a solute and a vacancy jump.
        """
        if Nthermo is not None:
            self.generate(Nthermo)
        return [s[0] for s in self.thermo.stars]

    def GFlist(self, Nthermo = None):
        """
        Return a list of points for the vacancy GF calculation, corresponding to omega0: no solute.
        Defined by Stars.

        Parameters
        ----------
        Nthermo : integer, optional
            if set to some value, then we call generate(Nthermo) first.

        Returns
        -------
        GFlist : list of array[3]
            list of points to calculate the vacancy diffusion GF; this is based ultimately on omega0,
            and is the (pseudo)inverse of the vacancy hop rate matrix.
        """
        if Nthermo is not None:
            self.generate(Nthermo)
        return self.GFR

    def omega1list(self, Nthermo = None):
        """
        Return a list of pairs of endpoints for a vacancy jump, corresponding to omega1:
        Solute at the origin, vacancy hopping between two sites. Defined by Double Stars.

        Parameters
        ----------
        Nthermo : integer, optional
            if set to some value, then we call generate(Nthermo) first.

        Returns
        -------
        omega1list : list of tuples of array[3]
            list of paired endpoints for a vacancy jump: the solute is at the origin,
            and these define start- and end-points for the vacancy jump.
        omega1LIMB : int array [Ndstar]
            index specifying which type of jump in omega0 would correspond to the LIMB
            approximation
        """
        if Nthermo is not None:
            self.generate(Nthermo)
        return [(self.kinetic.pts[p[0][0]], self.kinetic.pts[p[0][1]]) for p in self.omega1.dstars], \
               self.omega1LIMB

    def maketracer(self):
        """
        Generates input to Lij that corresponds to a tracer atom; useful for building
        input to Lij()

        Returns
        -------
        prob : array[thermo.Nstars]
            probability for each site in thermodynamic interaction range
        om2 : array[NNstar.Nstars]
            rates for exchange
        om1 : array[omega1.Ndstars]
            rates for vacancy motion around a solute
        """
        prob = np.zeros(self.thermo.Nstars, dtype=float)
        prob[:] = 1.
        om2 = np.zeros(self.NNstar.Nstars, dtype=float)
        om2[:] = -1.
        om1 = np.zeros(self.omega1.Ndstars, dtype=float)
        om1[:] = -1.
        return prob, om2, om1

    def _lij(self, gf, om0, prob, om2, om1):
        """
        Calculates the pieces for the transport coefficients: Lvv, L0ss, L2ss, L1sv, L1vv
        from the GF, omega0, omega1, and omega2 rates along with site probabilities.
        Used by Lij.

        Parameters
        ----------
        gf : array[NGF_sites]
            Green function for vacancy evaluated at sites
        om0 : array[NNjumps]
            rates for vacancy jumps
        prob : array[thermosites]
            probability of solute-vacancy complex at each sites
        om2 : array[NNjumps]
            rates for vacancy-solute exchange; if -1, use om0 entry
        om1 : array[Ndstars]
            rates for vacancy jumps around solute; if -1, use corresponding om0 entry

        Returns
        -------
        Lvv : array[3, 3]
            vacancy-vacancy; needs to be multiplied by cv/kBT
        L0ss : array[3, 3]
            "bare" solute-solute; needs to be multiplied by cv*cs/kBT
        L2ss : array[3, 3]
            correlation for solute-solute; needs to be multiplied by cv*cs/kBT
        L1sv : array[3, 3]
            correlation for solute-vacancy; needs to be multiplied by cv*cs/kBT
        L1vv : array[3, 3]
            correlation for vacancy-vacancy; needs to be multiplied by cv*cs/kBT
        """
        self.generatematrices()

        G0 = np.dot(self.GFexpansion, gf)

        probsqrt = np.zeros(self.kinetic.Nstars)
        probsqrt[:] = 1.
        for p, ind in zip(prob, self.thermo2kin):
            probsqrt[ind] = np.sqrt(p)

        om1expand = np.array([r1 if r1 >= 0 else r0
                              for r1, r0 in zip(om1, om0[self.omega1LIMB])]) # omega_1
        om2expand = np.array([r2 if r2 >= 0 else r0
                              for r2, r0 in zip(om2, om0)]) # omega_2
        # onsite term: need to make sure we divide out by the starting point probability, too
        om1onsite = (np.dot(self.omega1ds * probsqrt[self.gen1prob], om1expand) +
                     np.dot(self.omega1NN, om0))/probsqrt[self.vstar2kin]
        delta_om = np.dot(self.rate1expansion, om1expand) + \
                   np.diag(om1onsite) + \
                   np.dot(self.rate2expansion, om2expand) - \
                   np.dot(self.rate0expansion, om0)

        bias2vec = np.dot(self.bias2expansion, om2expand*np.sqrt(prob[self.NN2thermo]))
        bias1vec = np.dot(self.bias1ds * probsqrt[self.gen1prob], om1expand) + \
                   np.dot(self.bias1NN, om0)

        # G = np.linalg.inv(np.linalg.inv(G0) + delta_om)
        G = np.dot(np.linalg.inv(np.eye(len(bias1vec)) + np.dot(G0, delta_om)), G0)
        outer_eta1vec = np.dot(self.biasvec.outer, np.dot(G, bias1vec))
        outer_eta2vec = np.dot(self.biasvec.outer, np.dot(G, bias2vec))
        L2ss = np.dot(outer_eta2vec, bias2vec)
        L1sv = np.dot(outer_eta2vec, bias1vec)
        L1vv = np.dot(outer_eta1vec, bias1vec)
        # convert jump vectors to an array, and construct the rates as an array
        L0vv = GFcalc.D2(np.array(self.jumpvect),
                         np.array(sum([[om,]*len(Rs)
                                       for om, Rs in zip(om0, self.NNstar.stars)], [])))
        L0ss = GFcalc.D2(np.array(self.jumpvect),
                         np.array(sum([[om,]*len(Rs)
                                       for om, Rs in zip(om2expand*prob[self.NN2thermo],
                                                         self.NNstar.stars)], [])))
        return L0vv, L0ss, L2ss, L1sv, L1vv
        # print 'om1:\n', np.dot(self.rate1expansion, om1expand)
        # print 'om1_onsite:\n', np.diag(om1onsite)
        # print 'om2:\n', np.dot(self.rate2expansion, om2expand)
        # print 'om0:\n', np.dot(self.rate0expansion, om0)
        # print 'delta_om:\n', delta_om

    def Lij(self, gf, om0, prob, om2, om1):
        """
        Calculates the transport coefficients Lvv, Lss, and Lsv from the GF,
        omega0, omega1, and omega2 rates along with site probabilities.

        Parameters
        ----------
        gf : array[NGF_sites]
            Green function for vacancy evaluated at sites
        om0 : array[NNjumps]
            rates for vacancy jumps
        prob : array[thermosites]
            probability of solute-vacancy complex at each sites
        om2 : array[NNjumps]
            rates for vacancy-solute exchange; if -1, use om0 entry
        om1 : array[Ndstars]
            rates for vacancy jumps around solute; if -1, use corresponding om0 entry

        Returns
        -------
        Lvv : array[3, 3]
            vacancy-vacancy; needs to be multiplied by cv/kBT
        Lss : array[3, 3]
            solute-solute; needs to be multiplied by cv*cs/kBT
        Lsv : array[3, 3]
            solute-vacancy; needs to be multiplied by cv*cs/kBT
        Lvv1 : array[3, 3]
            vacancy-vacancy correction due to solute; needs to be multiplied by cv*cs/kBT
        """
        Lvv, L0ss, L2ss, L1sv, L1vv = self._lij(gf, om0, prob, om2, om1)
        return Lvv, L0ss + L2ss, -L0ss - L2ss + L1sv, L2ss - 2*L1sv + L1vv
