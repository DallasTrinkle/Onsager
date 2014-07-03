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
import stars
import GFcalc


class VacancyMediated:
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
        if Nthermo != None:
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
        if Nthermo != None:
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
        if Nthermo != None:
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
        if Nthermo != None:
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
        if Nthermo != None:
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
        Calculates the pieces for the transport coefficients: Lvv, L0ss, L2ss, L1sv
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
        """
        self.generatematrices()
        # convert jump vectors to an array, and construct the rates as an array
        Lvv = GFcalc.D2(np.array(self.jumpvect),
                        np.array(sum([[om,]*len(Rs)
                                      for om, Rs in zip(om0, self.NNstar.stars)], [])))
        L0ss = GFcalc.D2(np.array(self.jumpvect),
                         np.array(sum([[om,]*len(Rs)
                                       for om, Rs in zip(om2*prob[self.NN2thermo],
                                                         self.NNstar.stars)], [])))

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

        print 'om1:\n', np.dot(self.rate1expansion, om1expand)
        print 'om1_onsite:\n', np.diag(om1onsite)
        print 'om2:\n', np.dot(self.rate2expansion, om2expand)
        print 'om0:\n', np.dot(self.rate0expansion, om0)
        print 'delta_om:\n', delta_om

        bias2vec = np.dot(self.bias2expansion, om2expand)
        bias1vec = np.dot(self.bias1ds * probsqrt[self.gen1prob], om1expand) + \
                   np.dot(self.bias1NN, om0)

        # G = np.linalg.inv(np.linalg.inv(G0) + delta_om)
        G = np.dot(np.linalg.inv(np.eye(len(bias1vec)) + np.dot(G0, delta_om)), G0)
        L2ss = np.zeros((3, 3))
        L1sv = np.zeros((3, 3))
        for outer, b1, b2, Gb2 in zip(self.biasvec.outer,
                                      bias1vec,
                                      bias2vec,
                                      np.dot(G, bias2vec)):
            L1sv += outer*b1*Gb2
            L2ss += outer*b2*Gb2
        return Lvv, L0ss, L2ss, L1sv

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
        """
        Lvv, L0ss, L2ss, L1sv = self._lij(gf, om0, prob, om2, om1)
        return Lvv, L0ss + L2ss, -L0ss - L2ss + L1sv
