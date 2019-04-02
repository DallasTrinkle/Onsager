"""
Onsager calculator module: Interstitialcy mechanism and Vacancy-mediated mechanism

Class to create an Onsager "calculator", which brings two functionalities:
1. determines *what* input is needed to compute the Onsager (mobility, or L) tensors
2. constructs the function that calculates those tensors, given the input values.

This class is designed to be combined with code that can, e.g., automatically
run some sort of atomistic-scale (DFT, classical potential) calculation of site
energies, and energy barriers, and then in concert with scripts to convert such data
into rates and probabilities, this will allow for efficient evaluation of transport
coefficients.

This implementation will be for vacancy-mediated solute diffusion assumes the dilute limit.
The mathematics is based on a Green function solution for the vacancy diffusion. The
computation of the GF is included in the GFcalc module.

Now with HDF5 write / read capability for VacancyMediated module
"""

__author__ = 'Dallas R. Trinkle'

import numpy as np
from scipy.linalg import pinv2, solve
import copy, collections, itertools, warnings
from functools import reduce
from onsager import GFcalc
from onsager import crystal
from onsager import crystalStars as stars
from onsager import supercell

# database tags
INTERSTITIAL_TAG = 'i'
TRANSITION_TAG = '{state1}^{state2}'
SOLUTE_TAG = 's'
VACANCY_TAG = 'v'
SINGLE_DEFECT_TAG_3D = '{type}:{u[0]:+06.3f},{u[1]:+06.3f},{u[2]:+06.3f}'
SINGLE_DEFECT_TAG_2D = '{type}:{u[0]:+06.3f},{u[1]:+06.3f}'
DOUBLE_DEFECT_TAG = '{state1}-{state2}'
OM0_TAG = 'omega0:{vac1}^{vac2}'
OM1_TAG = 'omega1:{solute}-{vac1}^{vac2}'
OM2_TAG = 'omega2:{complex1}^{complex2}'


class Interstitial(object):
    """
    A class to compute interstitial diffusivity; uses structure of crystal to do most
    of the heavy lifting in terms of symmetry.

    Takes in a crystal that contains the interstitial as one of the chemical elements,
    to be specified by ``chem``, the sitelist (list of symmetry equivalent sites), and
    jumpnetwork. Both of the latter can be computed automatically from ``crys`` methods,
    but as they are lists, can also be editted or constructed by hand.
    """

    def __init__(self, crys, chem, sitelist, jumpnetwork):
        """
        Initialization; takes an underlying crystal, a choice of atomic chemistry,
        a corresponding Wyckoff site list and jump network.

        :param crys: Crystal object
        :param chem: integer, index into the basis of crys, corresponding to the chemical element that hops
        :param sitelist: list of lists of indices, site indices where the atom may hop;
          grouped by symmetry equivalency
        :param jumpnetwork: list of lists of tuples: ( (i, j), dx )
            symmetry unique transitions; each list is all of the possible transitions
            from site i to site j with jump vector dx; includes i->j and j->i
        """
        self.crys = crys
        self.threshold = self.crys.threshold
        self.dim = crys.dim
        self.chem = chem
        self.sitelist = sitelist
        self.N = sum(1 for w in sitelist for i in w)
        self.invmap = [0 for w in sitelist for i in w]
        for ind, w in enumerate(sitelist):
            for i in w:
                self.invmap[i] = ind
        self.jumpnetwork = jumpnetwork
        self.VectorBasis, self.VV = self.crys.FullVectorBasis(self.chem)
        self.NV = len(self.VectorBasis)
        # quick check to see if our projected omega matrix will be invertible
        # only really needed if we have a non-empty vector basis
        self.omega_invertible = True
        if self.NV > 0:
            # invertible if inversion is present
            self.omega_invertible = any(np.allclose(g.cartrot, -np.eye(self.dim)) for g in crys.G)
        if self.omega_invertible:
            # invertible, so just use solve for speed (omega is technically *negative* definite)
            self.bias_solver = lambda omega, b: -solve(-omega, b, sym_pos=True)
        else:
            # pseudoinverse required:
            self.bias_solver = lambda omega, b: np.dot(pinv2(omega), b)
        # these pieces are needed in order to compute the elastodiffusion tensor
        self.sitegroupops = self.generateSiteGroupOps()  # list of group ops to take first rep. into whole list
        self.jumpgroupops = self.generateJumpGroupOps()  # list of group ops to take first rep. into whole list
        self.siteSymmTensorBasis = self.generateSiteSymmTensorBasis()  # projections for *first rep. only*
        self.jumpSymmTensorBasis = self.generateJumpSymmTensorBasis()  # projections for *first rep. only*
        self.tags, self.tagdict, self.tagdicttype = self.generatetags()  # now with tags!

    @staticmethod
    def sitelistYAML(sitelist, dim=3):
        """Dumps a "sample" YAML formatted version of the sitelist with data to be entered"""
        return crystal.yaml.dump({'Dipole': [np.zeros((dim, dim)) for w in sitelist],
                                  'Energy': [0 for w in sitelist],
                                  'Prefactor': [1 for w in sitelist],
                                  'sitelist': sitelist})

    @staticmethod
    def jumpnetworkYAML(jumpnetwork, dim=3):
        """Dumps a "sample" YAML formatted version of the jumpnetwork with data to be entered"""
        return crystal.yaml.dump({'DipoleT': [np.zeros((dim, dim)) for t in jumpnetwork],
                                  'EnergyT': [0 for t in jumpnetwork],
                                  'PrefactorT': [1 for t in jumpnetwork],
                                  'jumpnetwork': jumpnetwork})

    def generatetags(self):
        """
        Create tags for unique interstitial states, and transition states.

        :return tags: dictionary of tags; each is a list-of-lists
        :return tagdict: dictionary that maps tag into the index of the corresponding list.
        :return tagdicttype: dictionary that maps tag into the key for the corresponding list.
        """
        tags, tagdict, tagdicttype = {}, {}, {}
        basis = self.crys.basis[self.chem]  # shortcut

        def single_state(u):
            return SINGLE_DEFECT_TAG_3D.format(type=INTERSTITIAL_TAG, u=u) if self.dim == 3 else \
                    SINGLE_DEFECT_TAG_2D.format(type=INTERSTITIAL_TAG, u=u)

        def transition(ui, dx):
            return TRANSITION_TAG.format(state1=single_state(ui),
                                         state2=single_state(ui + np.dot(self.crys.invlatt, dx)))

        tags['states'] = [[single_state(basis[s]) for s in sites]
                          for sites in self.sitelist]
        tags['transitions'] = [[transition(basis[i], dx) for ((i, j), dx) in jumplist]
                               for jumplist in self.jumpnetwork]
        # make the "tagdict" for quick indexing!
        for tagtype, taglist in tags.items():
            for i, tagset in enumerate(taglist):
                for tag in tagset:
                    if tag in tagdict:
                        raise ValueError('Generated repeated tags? {} found twice.'.format(tag))
                    else:
                        tagdict[tag], tagdicttype[tag] = i, tagtype
        return tags, tagdict, tagdicttype

    def __str__(self):
        """Human readable version of diffuser"""
        s = "Diffuser for atom {} ({})\n".format(self.chem, self.crys.chemistry[self.chem])
        s += self.crys.__str__() + '\n'
        for t in ('states', 'transitions'):
            s += t + ':\n'
            s += '\n'.join([taglist[0] for taglist in self.tags[t]]) + '\n'
        return s

    def makesupercells(self, super_n):
        """
        Take in a supercell matrix, then generate all of the supercells needed to compute
        site energies and transitions (corresponding to the representatives).

        :param super_n: 3x3 integer matrix to define our supercell
        :return superdict: dictionary of ``states``, ``transitions``, ``transmapping``,
            and ``indices`` that correspond to dictionaries with tags.

            * superdict['states'][i] = supercell of site;
            * superdict['transitions'][n] = (supercell initial, supercell final);
            * superdict['transmapping'][n] = ((site tag, groupop, mapping), (site tag, groupop, mapping))
            * superdict['indices'][tag] = index of tag, where tag is either a state or transition tag.
        """
        superdict = {'states': {}, 'transitions': {}, 'transmapping': {}, 'indices': {}}
        basesupercell = supercell.Supercell(self.crys, super_n, interstitial=(self.chem,), Nsolute=0)
        basis = self.crys.basis[self.chem]
        # fill up the supercell with all the *other* atoms
        for (c, i) in self.crys.atomindices:
            if c == self.chem: continue
            basesupercell.fillperiodic((c, i), Wyckoff=False)  # for efficiency
        for sites, tags in zip(self.sitelist, self.tags['states']):
            i, tag = sites[0], tags[0]
            u = basis[i]
            super0 = basesupercell.copy()
            ind = np.dot(super0.invsuper, u) / super0.size
            # put an interstitial in that single state; the "first" one is fine:
            super0[ind] = self.chem
            superdict['states'][tag] = super0
        for jumps, tags in zip(self.jumpnetwork, self.tags['transitions']):
            (i0, j0), dx0 = jumps[0]
            tag = tags[0]
            u0 = self.crys.basis[self.chem][i0]
            u1 = u0 + np.dot(self.crys.invlatt, dx0)  # should correspond to the j0
            super0, super1 = basesupercell.copy(), basesupercell.copy()
            ind0, ind1 = np.dot(super0.invsuper, u0) / super0.size, np.dot(super1.invsuper, u1) / super0.size
            # put interstitials at our corresponding sites
            super0[ind0], super1[ind1] = self.chem, self.chem
            superdict['transitions'][tag] = (super0, super1)
            # determine the mappings:
            superdict['transmapping'][tag] = tuple()
            for s in (super0, super1):
                for k, v in superdict['states'].items():
                    # attempt the mapping
                    g, mapping = v.equivalencemap(s)
                    if g is not None:
                        superdict['transmapping'][tag] += ((k, g, mapping),)
                        break
        for d in (superdict['states'], superdict['transitions']):
            for k in d.keys():
                superdict['indices'][k] = self.tagdict[k]  # keep a local copy of the indices, for transformation later
        return superdict

    def generateSiteGroupOps(self):
        """
        Generates a list of group operations that transform the first site in each site list
        into all of the other members; one group operation for each.

        :return siteGroupOps: list of list of group ops that mirrors the structure of site list
        """
        groupops = []
        for sites in self.sitelist:
            i0 = sites[0]
            oplist = []
            for i in sites:
                for g in self.crys.G:
                    if g.indexmap[self.chem][i0] == i:
                        oplist.append(g)
                        break
            groupops.append(oplist)
        return groupops

    def generateJumpGroupOps(self):
        """
        Generates a list of group operations that transform the first jump in the jump
        network into all of the other members; one group operation for each.

        :return siteGroupOps: list of list of group ops that mirrors the structure of jumpnetwork.
        """
        groupops = []
        for jumps in self.jumpnetwork:
            (i0, j0), dx0 = jumps[0]
            oplist = []
            for (i, j), dx in jumps:
                for g in self.crys.G:
                    # more complex: have to check the tuple (i,j) *and* the rotation of dx
                    # AND against the possibility that we are looking at the reverse jump too
                    if (g.indexmap[self.chem][i0] == i
                        and g.indexmap[self.chem][j0] == j
                        and np.allclose(dx, np.dot(g.cartrot, dx0), atol=self.threshold)) or \
                            (g.indexmap[self.chem][i0] == j
                             and g.indexmap[self.chem][j0] == i
                             and np.allclose(dx, -np.dot(g.cartrot, dx0), atol=self.threshold)):
                        oplist.append(g)
                        break
            groupops.append(oplist)
        return groupops

    def generateSiteSymmTensorBasis(self):
        """
        Generates a list of symmetric tensor bases for the first representative site
        in our site list.

        :return TensorSet: list of symmetric tensors
        """
        return [self.crys.SymmTensorBasis((self.chem, sites[0])) for sites in self.sitelist]

    def generateJumpSymmTensorBasis(self):
        """
        Generates a list of symmetric tensor bases for the first representative transition
        in our jump network

        :return TensorSet: list of list of symmetric tensors
        """
        # there is probably another way to do a list comprehension here, but that
        # will likely be nigh unreadable.
        lis = []
        for jumps in self.jumpnetwork:
            (i, j), dx = jumps[0]
            # more complex: have to check the tuple (i,j) *and* the rotation of dx
            # AND against the possibility that we are looking at the reverse jump too
            lis.append(reduce(crystal.CombineTensorBasis,
                              [crystal.SymmTensorBasis(*g.eigen())
                               for g in self.crys.G
                               if (g.indexmap[self.chem][i] == i and
                                   g.indexmap[self.chem][j] == j and
                                   np.allclose(dx, np.dot(g.cartrot, dx), atol=self.threshold)) or
                               (g.indexmap[self.chem][i] == j and
                                g.indexmap[self.chem][j] == i and
                                np.allclose(dx, -np.dot(g.cartrot, dx), atol=self.threshold))]))
        return lis

    def siteprob(self, pre, betaene):
        """Returns our site probabilities, normalized, as a vector"""
        # be careful to make sure that we don't under-/over-flow on beta*ene
        minbetaene = min(betaene)
        rho = np.array([pre[w] * np.exp(minbetaene - betaene[w]) for w in self.invmap])
        return rho / sum(rho)

    def ratelist(self, pre, betaene, preT, betaeneT):
        """Returns a list of lists of rates, matched to jumpnetwork"""
        # the ij tuple in each transition list is the i->j pair
        # invmap[i] tells you which Wyckoff position i maps to (in the sitelist)
        # trying to avoid under-/over-flow
        siteene = np.array([betaene[w] for w in self.invmap])
        sitepre = np.array([pre[w] for w in self.invmap])
        return [[pT * np.exp(siteene[i] - beT) / sitepre[i]
                 for (i, j), dx in t]
                for t, pT, beT in zip(self.jumpnetwork, preT, betaeneT)]

    def symmratelist(self, pre, betaene, preT, betaeneT):
        """Returns a list of lists of symmetrized rates, matched to jumpnetwork"""
        # the ij tuple in each transition list is the i->j pair
        # invmap[i] tells you which Wyckoff position i maps to (in the sitelist)
        # trying to avoid under-/over-flow
        siteene = np.array([betaene[w] for w in self.invmap])
        sitepre = np.array([pre[w] for w in self.invmap])
        return [[pT * np.exp(0.5 * siteene[i] + 0.5 * siteene[j] - beT) / np.sqrt(sitepre[i] * sitepre[j])
                 for (i, j), dx in t]
                for t, pT, beT in zip(self.jumpnetwork, preT, betaeneT)]

    def siteDipoles(self, dipoles):
        """
        Returns a list of the elastic dipole on each site, given the dipoles
        for the representatives. ("populating" the full set of dipoles)

        :param dipoles: list of dipoles for the first representative site
        :return dipolelist: array of dipole for each site [site][3][3]
        """
        # difficult to do with list comprehension since we're mapping from Wyckoff positions
        # to site indices; need to create the "blank" list first, then map into it.
        lis = np.zeros((self.N, self.dim, self.dim))  # blank list to index into
        for dipole, basis, sites, groupops in zip(dipoles, self.siteSymmTensorBasis,
                                                  self.sitelist, self.sitegroupops):
            symmdipole = crystal.ProjectTensorBasis(dipole, basis)
            for i, g in zip(sites, groupops):
                lis[i] = self.crys.g_tensor(g, symmdipole)
        return lis
        # return [ dipoles[w] for i,w in enumerate(self.invmap) ]

    def jumpDipoles(self, dipoles):
        """
        Returns a list of the elastic dipole for each transition, given the dipoles
        for the representatives. ("populating" the full set of dipoles)

        :param dipoles: list of dipoles for the first representative transition
        :return dipolelist: list of lists of dipole for each jump[site][3][3]
        """
        # symmetrize them first via projection
        symmdipoles = [crystal.ProjectTensorBasis(dipole, basis)
                       for dipole, basis in zip(dipoles, self.jumpSymmTensorBasis)]
        return [[self.crys.g_tensor(g, dipole) for g in groupops]
                for groupops, dipole in zip(self.jumpgroupops, symmdipoles)]

    def diffusivity(self, pre, betaene, preT, betaeneT, CalcDeriv=False):
        """
        Computes the diffusivity for our element given prefactors and energies/kB T.
        Also returns the negative derivative of diffusivity with respect to beta (used to compute
        the activation barrier tensor) if CalcDeriv = True
        The input list order corresponds to the sitelist and jumpnetwork

        :param pre: list of prefactors for unique sites
        :param betaene: list of site energies divided by kB T
        :param preT: list of prefactors for transition states
        :param betaeneT: list of transition state energies divided by kB T
        :return D[3,3]: diffusivity as a 3x3 tensor
        :return DE[3,3]: diffusivity times activation barrier (if CalcDeriv == True)
        """
        if __debug__:
            if len(pre) != len(self.sitelist): raise IndexError(
                "length of prefactor {} doesn't match sitelist".format(pre))
            if len(betaene) != len(self.sitelist): raise IndexError(
                "length of energies {} doesn't match sitelist".format(betaene))
            if len(preT) != len(self.jumpnetwork): raise IndexError(
                "length of prefactor {} doesn't match jump network".format(preT))
            if len(betaeneT) != len(self.jumpnetwork): raise IndexError(
                "length of energies {} doesn't match jump network".format(betaeneT))
        rho = self.siteprob(pre, betaene)
        sqrtrho = np.sqrt(rho)
        ratelist = self.ratelist(pre, betaene, preT, betaeneT)
        symmratelist = self.symmratelist(pre, betaene, preT, betaeneT)
        omega_ij = np.zeros((self.N, self.N))
        domega_ij = np.zeros((self.N, self.N))
        bias_i = np.zeros((self.N, self.dim))
        dbias_i = np.zeros((self.N, self.dim))
        D0 = np.zeros((self.dim, self.dim))
        Dcorrection = np.zeros((self.dim, self.dim))
        Db = np.zeros((self.dim, self.dim))
        # bookkeeping for energies:
        siteene = np.array([betaene[w] for w in self.invmap])
        # transene = [ [ bET for (i,j), dx in t ] for t, bET in zip(self.jumpnetwork, betaeneT)]
        Eave = np.dot(rho, siteene)

        for transitionset, rates, symmrates, bET in zip(self.jumpnetwork, ratelist, symmratelist, betaeneT):
            for ((i, j), dx), rate, symmrate in zip(transitionset, rates, symmrates):
                # symmrate = sqrtrho[i]*invsqrtrho[j]*rate
                omega_ij[i, j] += symmrate
                omega_ij[i, i] -= rate
                domega_ij[i, j] += symmrate * (bET - 0.5 * (siteene[i] + siteene[j]))
                domega_ij[i, i] -= rate * (bET - siteene[i])
                bias_i[i] += sqrtrho[i] * rate * dx
                dbias_i[i] += sqrtrho[i] * rate * dx * (bET - 0.5 * (siteene[i] + Eave))
                D0 += 0.5 * np.outer(dx, dx) * rho[i] * rate
                Db += 0.5 * np.outer(dx, dx) * rho[i] * rate * (bET - Eave)
        if self.NV > 0:
            # NOTE: there's probably a SUPER clever way to do this with higher dimensional arrays and dot...
            omega_v = np.zeros((self.NV, self.NV))
            domega_v = np.zeros((self.NV, self.NV))
            bias_v = np.zeros(self.NV)
            dbias_v = np.zeros(self.NV)
            for a, va in enumerate(self.VectorBasis):
                bias_v[a] = np.trace(np.dot(bias_i.T, va))
                dbias_v[a] = np.trace(np.dot(dbias_i.T, va))
                for b, vb in enumerate(self.VectorBasis):
                    omega_v[a, b] = np.trace(np.dot(va.T, np.dot(omega_ij, vb)))
                    domega_v[a, b] = np.trace(np.dot(va.T, np.dot(domega_ij, vb)))
            gamma_v = self.bias_solver(omega_v, bias_v)
            dgamma_v = np.dot(domega_v, gamma_v)
            Dcorrection = np.dot(np.dot(self.VV, bias_v), gamma_v)
            Db += np.dot(np.dot(self.VV, dbias_v), gamma_v) \
                  + np.dot(np.dot(self.VV, gamma_v), dbias_v) \
                  - np.dot(np.dot(self.VV, gamma_v), dgamma_v)

        if not CalcDeriv:
            return D0 + Dcorrection
        else:
            return D0 + Dcorrection, Db

    def elastodiffusion(self, pre, betaene, dipole, preT, betaeneT, dipoleT):
        """
        Computes the elastodiffusion tensor for our element given prefactors, energies/kB T,
        and elastic dipoles/kB T
        The input list order corresponds to the sitelist and jumpnetwork

        :param pre: list of prefactors for unique sites
        :param betaene: list of site energies divided by kB T
        :param dipole: list of elastic dipoles divided by kB T
        :param preT: list of prefactors for transition states
        :param betaeneT: list of transition state energies divided by kB T
        :param dipoleT: list of elastic dipoles divided by kB T
        :return D[3,3]: diffusivity as 3x3 tensor
        :return dD[3,3,3,3]: elastodiffusion tensor as 3x3x3x3 tensor
        """

        def vector_tensor_outer(v, a):
            """Construct the outer product of v and a"""
            va = np.zeros((self.dim, self.dim, self.dim))
            for i, j, k in itertools.product(range(self.dim), repeat=3):
                va[i, j, k] = v[i] * a[j, k]
            return va

        def tensor_tensor_outer(a, b):
            """Construct the outer product of a and b"""
            ab = np.zeros((self.dim, self.dim, self.dim, self.dim))
            for i, j, k, l in itertools.product(range(self.dim), repeat=4):
                ab[i, j, k, l] = a[i, j] * b[k, l]
            return ab

        if __debug__:
            if len(pre) != len(self.sitelist): raise IndexError(
                "length of prefactor {} doesn't match sitelist".format(pre))
            if len(betaene) != len(self.sitelist): raise IndexError(
                "length of energies {} doesn't match sitelist".format(betaene))
            if len(dipole) != len(self.sitelist): raise IndexError(
                "length of dipoles {} doesn't match sitelist".format(dipole))
            if len(preT) != len(self.jumpnetwork): raise IndexError(
                "length of prefactor {} doesn't match jump network".format(preT))
            if len(betaeneT) != len(self.jumpnetwork): raise IndexError(
                "length of energies {} doesn't match jump network".format(betaeneT))
            if len(dipoleT) != len(self.jumpnetwork): raise IndexError(
                "length of dipoles {} doesn't match jump network".format(dipoleT))
        rho = self.siteprob(pre, betaene)
        sqrtrho = np.sqrt(rho)
        ratelist = self.ratelist(pre, betaene, preT, betaeneT)
        symmratelist = self.symmratelist(pre, betaene, preT, betaeneT)
        omega_ij = np.zeros((self.N, self.N))
        bias_i = np.zeros((self.N, self.dim))
        biasP_i = np.zeros((self.N, self.dim, self.dim, self.dim))
        domega_ij = np.zeros((self.N, self.N, self.dim, self.dim))
        sitedipoles = self.siteDipoles(dipole)
        jumpdipoles = self.jumpDipoles(dipoleT)
        dipoleave = np.tensordot(rho, sitedipoles, [(0), (0)])  # average dipole

        D0 = np.zeros((self.dim, self.dim))
        Dp = np.zeros((self.dim, self.dim, self.dim, self.dim))
        for transitionset, rates, symmrates, dipoles in zip(self.jumpnetwork, ratelist, symmratelist, jumpdipoles):
            for ((i, j), dx), rate, symmrate, dipole in zip(transitionset, rates, symmrates, dipoles):
                # symmrate = sqrtrho[i]*invsqrtrho[j]*rate
                omega_ij[i, j] += symmrate
                omega_ij[i, i] -= rate
                domega_ij[i, j] -= symmrate * (dipole - 0.5 * (sitedipoles[i] + sitedipoles[j]))
                domega_ij[i, i] += rate * (dipole - sitedipoles[i])
                bias_i[i] += sqrtrho[i] * rate * dx
                biasP_i[i] += vector_tensor_outer(sqrtrho[i] * rate * dx, dipole - 0.5 * (sitedipoles[i] + dipoleave))
                D0 += 0.5 * np.outer(dx, dx) * rho[i] * rate
                Dp += 0.5 * tensor_tensor_outer(np.outer(dx, dx) * rho[i] * rate, dipole - dipoleave)
        if self.NV > 0:
            omega_v = np.zeros((self.NV, self.NV))
            bias_v = np.zeros(self.NV)
            domega_v = np.zeros((self.NV, self.NV, self.dim, self.dim))
            # NOTE: there's probably a SUPER clever way to do this with higher dimensional arrays and dot...
            for a, va in enumerate(self.VectorBasis):
                bias_v[a] = np.tensordot(bias_i, va, ((0, 1), (0, 1)))  # can also use trace(dot(bias_i.T, va))
                for b, vb in enumerate(self.VectorBasis):
                    omega_v[a, b] = np.tensordot(va, np.tensordot(omega_ij, vb, ((1), (0))), ((0, 1), (0, 1)))
                    domega_v[a, b] = np.tensordot(va, np.tensordot(domega_ij, vb, ((1), (0))), ((0, 1), (0, 3)))
            gamma_v = self.bias_solver(omega_v, bias_v)
            dg = np.tensordot(domega_v, gamma_v, ((1), (0)))
            # need to project gamma_v *back onto* our sites; not sure if we can just do with a dot since
            # self.VectorBasis is a list of Nx3 matrices
            gamma_i = sum(g * va for g, va in zip(gamma_v, self.VectorBasis))
            D0 += np.dot(np.dot(self.VV, bias_v), gamma_v)
            for c, d in itertools.product(range(self.dim), repeat=2):
                Dp[:, :, c, d] += np.tensordot(gamma_i, biasP_i[:, :, c, d], ((0), (0))) + \
                                  np.tensordot(biasP_i[:, :, c, d], gamma_i, ((0), (0)))
            Dp += np.tensordot(np.tensordot(self.VV, gamma_v, ((3), (0))), dg, ((2), (0)))

        for a, b, c, d in itertools.product(range(self.dim), repeat=4):
            if a == c:
                Dp[a, b, c, d] += 0.5 * D0[b, d]
            if a == d:
                Dp[a, b, c, d] += 0.5 * D0[b, c]
            if b == c:
                Dp[a, b, c, d] += 0.5 * D0[a, d]
            if b == d:
                Dp[a, b, c, d] += 0.5 * D0[a, c]
        return D0, Dp

    def losstensors(self, pre, betaene, dipole, preT, betaeneT):
        """
        Computes the internal friction loss tensors for our element given prefactors, energies/kB T,
        and elastic dipoles/kB T
        The input list order corresponds to the sitelist and jumpnetwork

        :param pre: list of prefactors for unique sites
        :param betaene: list of site energies divided by kB T
        :param dipole: list of elastic dipoles divided by kB T
        :param preT: list of prefactors for transition states
        :param betaeneT: list of transition state energies divided by kB T
        :return lambdaL: list of tuples of (eigenmode, L-tensor) where L-tensor is a 3x3x3x3 loss tensor
            L-tensor needs to be multiplied by kB T to have proper units of energy.
        """

        def tensor_square(a):
            """Construct the outer product of a with itself"""
            aa = np.zeros((self.dim, self.dim, self.dim, self.dim))
            for i, j, k, l in itertools.product(range(self.dim), repeat=4):
                aa[i, j, k, l] = a[i, j] * a[k, l]
            return aa

        if __debug__:
            if len(pre) != len(self.sitelist): raise IndexError(
                "length of prefactor {} doesn't match sitelist".format(pre))
            if len(betaene) != len(self.sitelist): raise IndexError(
                "length of energies {} doesn't match sitelist".format(betaene))
            if len(dipole) != len(self.sitelist): raise IndexError(
                "length of dipoles {} doesn't match sitelist".format(dipole))
            if len(preT) != len(self.jumpnetwork): raise IndexError(
                "length of prefactor {} doesn't match jump network".format(preT))
            if len(betaeneT) != len(self.jumpnetwork): raise IndexError(
                "length of energies {} doesn't match jump network".format(betaeneT))
        rho = self.siteprob(pre, betaene)
        sqrtrho = np.sqrt(rho)
        ratelist = self.ratelist(pre, betaene, preT, betaeneT)
        symmratelist = self.symmratelist(pre, betaene, preT, betaeneT)
        omega_ij = np.zeros((self.N, self.N))
        sitedipoles = self.siteDipoles(dipole)

        # populate our symmetrized transition matrix:
        for transitionset, rates, symmrates in zip(self.jumpnetwork, ratelist, symmratelist):
            for ((i, j), dx), rate, symmrate in zip(transitionset, rates, symmrates):
                # symmrate = sqrtrho[i]*invsqrtrho[j]*rate
                omega_ij[i, j] += symmrate
                omega_ij[i, i] -= rate
        # next, diagonalize:
        # lamb: eigenvalues, in ascending order, with eigenvalues phi
        # then, the *largest* should be lamb = 0
        lamb, phi = np.linalg.eigh(omega_ij)
        averate = abs(omega_ij.trace()/self.N)
        lambdaL = []
        # work through the eigenvalues / vectors individually:
        # NOTE: we should have a negative definite matrix, so negate those eigenvalues...
        for l, p in zip(-lamb, phi.T):
            # need to check if lamb is (approximately) 0. Can also check if p is close to sqrtrho
            if abs(l) < 1e-8*averate: continue
            if np.isclose(np.dot(p, sqrtrho), 1): continue
            F = np.tensordot(p*sqrtrho, sitedipoles, axes=1)
            L = tensor_square(F)
            # determine if we have a new mode or not
            found = False
            for (lamb0, L0) in lambdaL:
                if np.isclose(lamb0, l):
                    L0 += L
                    found = True
            if not found:
                lambdaL.append((l, L))
        # pass back list
        return lambdaL


# YAML tags
VACANCYTHERMOKINETICS_YAMLTAG = '!VacancyThermoKinetics'


class vacancyThermoKinetics(collections.namedtuple('vacancyThermoKinetics',
                                                   'pre betaene preT betaeneT')):
    """
    Class to store (in a hashable manner) the thermodynamics and kinetics for the vacancy

    :param pre: prefactors for sites
    :param betaene: energy for sites / kBT
    :param preT: prefactors for transition states
    :param betaeneT: transition state energy for sites / kBT
    """

    def __repr__(self):
        return "{}(pre={}, betaene={}, preT={}, betaeneT={})".format(self.__class__.__name__,
                                                                     self.pre, self.betaene,
                                                                     self.preT, self.betaeneT)

    def _asdict(self):
        """Return a proper dict"""
        return {'pre': self.pre, 'betaene': self.betaene, 'preT': self.preT, 'betaeneT': self.betaeneT}

    def __eq__(self, other):
        # Note: could scale all prefactors by min(pre) and subtract all energies by min(ene)...?
        return isinstance(other, self.__class__) and \
               np.allclose(self.pre, other.pre) and np.allclose(self.betaene, other.betaene) and \
               np.allclose(self.preT, other.preT) and np.allclose(self.betaeneT, other.betaeneT)

    def __ne__(self, other):
        return not __eq__(other)

    def __hash__(self):
        return hash(self.pre.data.tobytes() + self.betaene.data.tobytes() +
                    self.preT.data.tobytes() + self.betaeneT.data.tobytes())

    @staticmethod
    def vacancyThermoKinetics_representer(dumper, data):
        """Output a PairState"""
        # asdict() returns an OrderedDictionary, so pass through dict()
        # had to rewrite _asdict() for some reason...?
        return dumper.represent_mapping(VACANCYTHERMOKINETICS_YAMLTAG, data._asdict())

    @staticmethod
    def vacancyThermoKinetics_constructor(loader, node):
        """Construct a GroupOp from YAML"""
        # ** turns the dictionary into parameters for GroupOp constructor
        return vacancyThermoKinetics(**loader.construct_mapping(node, deep=True))


# HDF5 conversion routines: vTK indexed dictionaries
def vTKdict2arrays(vTKdict):
    """
    Takes a dictionary indexed by vTK objects, returns two arrays of vTK keys and values,
    and the splits to separate vTKarray back into vTK

    :param vTKdict: dictionary, indexed by vTK objects, whose entries are arrays
    :return vTKarray: array of vTK entries
    :return valarray: array of values
    :return vTKsplits: split placement for vTK entries
    """
    if len(vTKdict.keys()) == 0: return None, None, None
    vTKexample = [k for k in vTKdict.keys()][0]
    vTKsplits = np.cumsum(np.array([len(v) for v in vTKexample]))[:-1]
    vTKlist = []
    vallist = []
    for k, v in zip(vTKdict.keys(), vTKdict.values()):
        vTKlist.append(np.hstack(k))  # k.pre, k.betaene, k.preT, k.betaeneT
        vallist.append(v)
    return np.array(vTKlist), np.array(vallist), vTKsplits


def arrays2vTKdict(vTKarray, valarray, vTKsplits):
    """
    Takes two arrays of vTK keys and values, and the splits to separate vTKarray back into vTK
    and returns a dictionary indexed by the vTK.

    :param vTKarray: array of vTK entries
    :param valarray: array of values
    :param vTKsplits: split placement for vTK entries
    :return vTKdict: dictionary, indexed by vTK objects, whose entries are arrays
    """
    if all(x is None for x in (vTKarray, valarray, vTKsplits)): return {}
    vTKdict = {}
    for vTKa, val in zip(vTKarray, valarray):
        vTKdict[vacancyThermoKinetics(*np.hsplit(vTKa, vTKsplits))] = val
    return vTKdict


class VacancyMediated(object):
    """
    A class to compute vacancy-mediated solute transport coefficients, specifically
    L_vv (vacancy diffusion), L_ss (solute), and L_sv (off-diagonal). As part of that,
    it determines *what* quantities are needed as inputs in order to perform this calculation.

    Based on crystal class. Also now includes its own GF calculator and cacheing, and
    storage in HDF5 format.

    Requires a crystal, chemical identity of vacancy, list of symmetry-equivalent
    sites for that chemistry, and a jumpnetwork for the vacancy. The thermodynamic
    range (number of "shells" -- see ``crystalStars.StarSet`` for precise definition).
    """

    def __init__(self, crys, chem, sitelist, jumpnetwork, Nthermo=0, NGFmax=4):
        """
        Create our diffusion calculator for a given crystal structure, chemical identity,
        jumpnetwork (for the vacancy) and thermodynamic shell.

        :param crys: Crystal object
        :param chem: index identifying the diffusing species
        :param sitelist: list, grouped into Wyckoff common positions, of unique sites
        :param jumpnetwork: list of unique transitions as lists of ((i,j), dx)
        :param Nthermo: range of thermodynamic interaction (in successive jumpnetworks)
        :param NGFmax: parameter controlling k-point density of GF calculator; 4 seems reasonably accurate
        """
        if all(x is None for x in (crys, chem, sitelist, jumpnetwork)): return  # blank object
        self.crys = crys
        self.threshold = self.crys.threshold
        self.dim = crys.dim
        self.chem = chem
        self.sitelist = copy.deepcopy(sitelist)
        self.jumpnetwork = copy.deepcopy(jumpnetwork)
        self.N = sum(len(w) for w in sitelist)
        self.invmap = np.zeros(self.N, dtype=int)
        for ind, w in enumerate(sitelist):
            for i in w:
                self.invmap[i] = ind
        self.om0_jn = copy.deepcopy(jumpnetwork)
        self.GFcalc = self.GFcalculator(NGFmax)
        # do some initial setup:
        # self.thermo = stars.StarSet(self.jumpnetwork, self.crys, self.chem, Nthermo)
        self.thermo = stars.StarSet(self.jumpnetwork, self.crys, self.chem)  # just create; call generate later
        self.kinetic = stars.StarSet(self.jumpnetwork, self.crys, self.chem)
        self.NNstar = stars.StarSet(self.jumpnetwork, self.crys, self.chem, 1)
        # self.kinetic = self.thermo + self.NNstar
        self.vkinetic = stars.VectorStarSet()
        self.generate(Nthermo)
        self.generatematrices()
        # dict: vacancy, solute, solute-vacancy; omega0, omega1, omega2 (see __taglist__)
        self.tags, self.tagdict, self.tagdicttype = self.generatetags()

    def GFcalculator(self, NGFmax=0):
        """Return the GF calculator; create a new one if NGFmax is being changed"""
        # if not being set (no parameter passed) or same as what we already use, return calculator
        if NGFmax == getattr(self, 'NGFmax', 0): return self.GFcalc
        if NGFmax < 0: raise ValueError('NGFmax ({}) must be >0'.format(NGFmax))
        self.NGFmax= NGFmax
        # empty dictionaries to store GF values: necessary if we're changing NGFmax
        self.clearcache()
        return GFcalc.GFCrystalcalc(self.crys, self.chem, self.sitelist, self.om0_jn, NGFmax)

    def clearcache(self):
        """Clear out the GF cache values"""
        self.GFvalues, self.Lvvvalues, self.etavvalues = {}, {}, {}

    def generate(self, Nthermo):
        """
        Generate the necessary stars, vector-stars, and jump networks based on the thermodynamic range.

        :param Nthermo: range of thermodynamic interactions, in terms of "shells",
            which is multiple summations of jumpvect
        """
        if Nthermo == getattr(self, 'Nthermo', 0): return
        self.Nthermo = Nthermo

        self.thermo.generate(Nthermo, originstates=False)
        self.kinetic.generate(Nthermo + 1, originstates=True)  # now include origin states (for removal)
        self.vkinetic.generate(self.kinetic)
        # TODO: check the GF calculator against the range in GFstarset to make sure its adequate
        self.GFexpansion, self.GFstarset = self.vkinetic.GFexpansion()

        # some indexing helpers:
        # thermo2kin maps star index in thermo to kinetic (should just be range(n), but we use this for safety)
        # kin2vacancy maps star index in kinetic to non-solute configuration from sitelist
        # outerkin is the list of stars that are in kinetic, but not in thermo
        # vstar2kin maps each vector star back to the corresponding star index
        # kin2vstar provides a list of vector stars indices corresponding to the same star index
        self.thermo2kin = [self.kinetic.starindex(self.thermo.states[s[0]]) for s in self.thermo.stars]
        self.kin2vacancy = [self.invmap[self.kinetic.states[s[0]].j] for s in self.kinetic.stars]
        self.outerkin = [s for s in range(self.kinetic.Nstars)
                         if self.thermo.stateindex(self.kinetic.states[self.kinetic.stars[s][0]]) is None]
        self.vstar2kin = [self.kinetic.index[Rs[0]] for Rs in self.vkinetic.vecpos]
        self.kin2vstar = [[j for j in range(self.vkinetic.Nvstars) if self.vstar2kin[j] == i]
                          for i in range(self.kinetic.Nstars)]
        # jumpnetwork, jumptype (omega0), star-pair for jump
        self.om1_jn, self.om1_jt, self.om1_SP = self.kinetic.jumpnetwork_omega1()
        self.om2_jn, self.om2_jt, self.om2_SP = self.kinetic.jumpnetwork_omega2()
        # Prune the om1 list: remove entries that have jumps between stars in outerkin:
        # work in reverse order so that popping is safe (and most of the offending entries are at the end
        for i, SP in zip(reversed(range(len(self.om1_SP))), reversed(self.om1_SP)):
            if SP[0] in self.outerkin and SP[1] in self.outerkin:
                self.om1_jn.pop(i), self.om1_jt.pop(i), self.om1_SP.pop(i)
        # empty dictionaries to store GF values
        self.clearcache()

    def generatematrices(self):
        """
        Generates all the matrices and "helper" pieces, based on our jump networks.
        This has been separated out in case the user wants to, e.g., prune / modify the networks
        after they've been created with generate(), then generatematrices() can be rerun.
        """

        self.Dom1_om0, self.Dom1 = self.vkinetic.bareexpansions(self.om1_jn, self.om1_jt)
        self.Dom2_om0, self.Dom2 = self.vkinetic.bareexpansions(self.om2_jn, self.om2_jt)
        self.om1_om0, self.om1_om0escape, self.om1expansion, self.om1escape = \
            self.vkinetic.rateexpansions(self.om1_jn, self.om1_jt)
        self.om2_om0, self.om2_om0escape, self.om2expansion, self.om2escape = \
            self.vkinetic.rateexpansions(self.om2_jn, self.om2_jt, omega2=True)
        self.om1_b0, self.om1bias = self.vkinetic.biasexpansions(self.om1_jn, self.om1_jt)
        self.om2_b0, self.om2bias = self.vkinetic.biasexpansions(self.om2_jn, self.om2_jt, omega2=True)
        self.OSindices, self.OSfolddown, self.OS_VB = self.vkinetic.originstateVectorBasisfolddown('solute')
        self.OSVfolddown = self.vkinetic.originstateVectorBasisfolddown('vacancy')[1]  # only need the folddown

        # more indexing helpers:
        # kineticsvWyckoff: Wyckoff position of solute and vacancy for kinetic stars
        # omega0vacancyWyckoff: Wyckoff positions of initial and final position in omega0 jumps
        self.kineticsvWyckoff = [(self.invmap[PS.i], self.invmap[PS.j]) for PS in
                                 [self.kinetic.states[si[0]] for si in self.kinetic.stars]]
        self.omega0vacancyWyckoff = [(self.invmap[jumplist[0][0][0]], self.invmap[jumplist[0][0][1]])
                                     for jumplist in self.om0_jn]

    def generatetags(self):
        """
        Create tags for vacancy states, solute states, solute-vacancy complexes;
        omega0, omega1, and omega2 transition states.

        :return tags: dictionary of tags; each is a list-of-lists
        :return tagdict: dictionary that maps tag into the index of the corresponding list.
        :return tagdicttype: dictionary that maps tag into the key for the corresponding list.
        """
        tags, tagdict, tagdicttype = {}, {}, {}
        basis = self.crys.basis[self.chem]  # shortcut

        def single_defect(DEFECT_TAG, u):
            return SINGLE_DEFECT_TAG_3D.format(type=DEFECT_TAG, u=u) if self.dim == 3 else \
                    SINGLE_DEFECT_TAG_2D.format(type=DEFECT_TAG, u=u)

        def double_defect(PS):
            return DOUBLE_DEFECT_TAG.format( \
                state1=single_defect(SOLUTE_TAG, basis[PS.i]), \
                state2=single_defect(VACANCY_TAG, basis[PS.j] + PS.R))

        def omega1(PS1, PS2):
            return OM1_TAG.format( \
                solute=single_defect(SOLUTE_TAG, basis[PS1.i]),
                vac1=single_defect(VACANCY_TAG, basis[PS1.j] + PS1.R), \
                vac2=single_defect(VACANCY_TAG, basis[PS2.j] + PS2.R))

        tags['vacancy'] = [[single_defect(VACANCY_TAG, basis[s]) for s in sites]
                           for sites in self.sitelist]
        tags['solute'] = [[single_defect(SOLUTE_TAG, basis[s]) for s in sites]
                          for sites in self.sitelist]
        tags['solute-vacancy'] = [[double_defect(self.thermo.states[s]) for s in starlist]
                                  for starlist in self.thermo.stars]
        tags['omega0'] = [[OM0_TAG.format(vac1=single_defect(VACANCY_TAG, basis[i]),
                                          vac2=single_defect(VACANCY_TAG, basis[j] + dx))
                           for ((i, j), dx) in jumplist]
                          for jumplist in self.crys.jumpnetwork2lattice(self.chem, self.om0_jn)]
        tags['omega1'] = [[omega1(self.kinetic.states[i], self.kinetic.states[j])
                           for ((i, j), dx) in jumplist] for jumplist in self.om1_jn]
        tags['omega2'] = [[OM2_TAG.format(complex1=double_defect(self.kinetic.states[i]),
                                          complex2=double_defect(self.kinetic.states[j]))
                           for ((i, j), dx) in jumplist] for jumplist in self.om2_jn]
        # make the "tagdict" for quick indexing!
        for tagtype, taglist in tags.items():
            for i, tagset in enumerate(taglist):
                for tag in tagset:
                    if tag in tagdict:
                        raise ValueError('Generated repeated tags? {} found twice.'.format(tag))
                    else:
                        tagdict[tag], tagdicttype[tag] = i, tagtype
        return tags, tagdict, tagdicttype

    def __str__(self):
        """Human readable version of diffuser"""
        s = "Diffuser for atom {} ({}), Nthermo={}\n".format(self.chem,
                                                             self.crys.chemistry[self.chem],
                                                             self.Nthermo)
        s += self.crys.__str__() + '\n'
        for t in ('vacancy', 'solute', 'solute-vacancy'):
            s += t + ' configurations:\n'
            s += '\n'.join([taglist[0] for taglist in self.tags[t]]) + '\n'
        for t in ('omega0', 'omega1', 'omega2'):
            s += t + ' jumps:\n'
            s += '\n'.join([taglist[0] for taglist in self.tags[t]]) + '\n'
        return s

    def makesupercells(self, super_n):
        """
        Take in a supercell matrix, then generate all of the supercells needed to compute
        site energies and transitions (corresponding to the representatives).

        Note: the states are lone vacancy, lone solute, solute-vacancy complexes in
        our thermodynamic range. Note that there will be escape states are endpoints of
        some omega1 jumps. They are not relaxed, and have no pre-existing tag. They will
        only be output as a single endpoint of an NEB run; there may be symmetry equivalent
        duplicates, as we construct these supercells on an as needed basis.

        We've got a few classes of warnings (from most egregious to least) that can issued
        if the supercell is too small; the analysis will continue despite any warnings:

        1. Thermodynamic shell states map to different states in supercell
        2. Thermodynamic shell states are not unique in supercell (multiplicity)
        3. Kinetic shell states map to different states in supercell
        4. Kinetic shell states are not unique in supercell (multiplicity)

        The lowest level can still be run reliably but runs the risk of errors in escape transition
        barriers. Extreme caution should be used if any of the other warnings are raised.

        :param super_n: 3x3 integer matrix to define our supercell
        :return superdict: dictionary of ``states``, ``transitions``, ``transmapping``,
            ``indices`` that correspond to dictionaries with tags; the final tag
            ``reference`` is the basesupercell for calculations without defects.

            * superdict['states'][i] = supercell of state;
            * superdict['transitions'][n] = (supercell initial, supercell final);
            * superdict['transmapping'][n] = ((site tag, groupop, mapping), (site tag, groupop, mapping))
            * superdict['indices'][tag] = (type, index) of tag, where tag is either a state or transition tag.
            * superdict['reference'] = supercell reference, no defects
        """
        ### NOTE: much of this will *need* to be reimplemented for metastable states.
        vchem, schem = -1, self.crys.Nchem
        basis = self.crys.basis[self.chem]
        basesupercell = supercell.Supercell(self.crys, super_n, Nsolute=1)
        basesupercell.definesolute(schem, 'solute')
        # check whether our cell is large enough to contain the full thermodynamic range;
        # also check that our escape endpoint doesn't accidentally coincide with a "real" state.
        # The check is simple: we map the dx vector for a PairState into the half cell of the supercell;
        # it should match exactly. If it doesn't, there are two options: it has a different magnitude
        # which indicates a *new* state (mapping error) or it has the same magnitude (multiplicity).
        # We raise the warning accordingly. We do this with all the kinetic states, and check if it's in thermo.
        invlatt = np.linalg.inv(basesupercell.lattice)
        for PS in self.kinetic.states:
            dxmap = np.dot(basesupercell.lattice, crystal.inhalf(np.dot(invlatt, PS.dx)))
            if not np.allclose(PS.dx, dxmap, atol=self.threshold):
                if PS in self.thermo:
                    failstate = 'thermodynamic range'
                else:
                    failstate = 'escape endpoint'
                if np.allclose(np.dot(PS.dx, PS.dx), np.dot(dxmap, dxmap), atol=self.threshold):
                    failtype = 'multiplicity issue'
                else:
                    failtype = 'mapping error'
                warnings.warn('Supercell:\n{}\ntoo small: {} has {}'.format(super_n, failstate, failtype),
                              RuntimeWarning, stacklevel=2)
        # fill up the supercell with all the *other* atoms
        for (c, i) in self.crys.atomindices:
            basesupercell.fillperiodic((c, i), Wyckoff=False)  # for efficiency
        superdict = {'states': {}, 'transitions': {}, 'transmapping': {}, 'indices': {},
                     'reference': basesupercell}
        for statetype, chem in (('vacancy', vchem), ('solute', schem)):
            for sites, tags in zip(self.sitelist, self.tags[statetype]):
                i, tag = sites[0], tags[0]
                u = basis[i]
                super0 = basesupercell.copy()
                ind = np.dot(super0.invsuper, u) / super0.size
                # put a vacancy / solute in that single state; the "first" one is fine:
                super0[ind] = chem
                superdict['states'][tag] = super0
        for starlist, tags in zip(self.thermo.stars, self.tags['solute-vacancy']):
            PS, tag = self.thermo.states[starlist[0]], tags[0]
            us, uv = basis[PS.i], basis[PS.j] + PS.R
            super0 = basesupercell.copy()
            inds, indv = np.dot(super0.invsuper, us) / super0.size, np.dot(super0.invsuper, uv) / super0.size
            # put a solute + vacancy in that single state; the "first" one is fine:
            super0[inds], super0[indv] = schem, vchem
            superdict['states'][tag] = super0
        for jumptype, jumpnetwork in (('omega0', self.om0_jn),
                                      ('omega1', self.om1_jn),
                                      ('omega2', self.om2_jn)):
            for jumps, tags in zip(jumpnetwork, self.tags[jumptype]):
                (i0, j0), dx0 = jumps[0]
                tag = tags[0]
                super0, super1 = basesupercell.copy(), basesupercell.copy()
                # the supercell building is a bit specific to each jump type
                if jumptype == 'omega0':
                    u0 = basis[i0]
                    u1 = u0 + np.dot(self.crys.invlatt, dx0)  # should correspond to the j0
                    ind0, ind1 = np.dot(super0.invsuper, u0) / super0.size, \
                                 np.dot(super1.invsuper, u1) / super1.size
                    # put vacancies at our corresponding sites:
                    # we do this by *removing* two atoms in each, and then *placing* the atom back in.
                    # this ensures that we have correct NEB ordering
                    super0[ind0], super0[ind1] = vchem, vchem
                    super1[ind0], super1[ind1] = vchem, vchem
                    super0[ind1], super1[ind0] = self.chem, self.chem
                else:
                    PSi, PSf = self.kinetic.states[i0], self.kinetic.states[j0]
                    if jumptype == 'omega1':
                        # solute in first; same for each
                        inds = np.dot(super0.invsuper, basis[PSi.i]) / super0.size
                        super0[inds], super1[inds] = schem, schem
                        # now get the initial and final vacancy locations
                        ind0, ind1 = np.dot(super0.invsuper, basis[PSi.j] + PSi.R) / super0.size, \
                                     np.dot(super1.invsuper, basis[PSf.j] + PSf.R) / super1.size
                        # put vacancies at our corresponding sites:
                        # we do this by *removing* two atoms in each, and then *placing* the atom back in.
                        # this ensures that we have correct NEB ordering
                        super0[ind0], super0[ind1] = vchem, vchem
                        super1[ind0], super1[ind1] = vchem, vchem
                        super0[ind1], super1[ind0] = self.chem, self.chem
                    else:
                        # omega2, we do it all using PSi: *assume* PSf is the reverse (exchange s + v)
                        inds, indv = np.dot(super0.invsuper, basis[PSi.i]) / super0.size, \
                                     np.dot(super0.invsuper, basis[PSi.j] + PSi.R) / super0.size
                        # add the solutes:
                        super0[inds], super1[indv] = schem, schem
                        # and the vacancies:
                        super0[indv], super1[inds] = vchem, vchem
                superdict['transitions'][tag] = (super0, super1)
                # determine the mappings:
                superdict['transmapping'][tag] = tuple()
                for s in (super0, super1):
                    nomap = True
                    for k, v in superdict['states'].items():
                        # attempt the mapping
                        g, mapping = v.equivalencemap(s)
                        if g is not None:
                            superdict['transmapping'][tag] += ((k, g, mapping),)
                            nomap = False
                            break
                    if nomap:
                        superdict['transmapping'][tag] += (None,)
        for d in (superdict['states'], superdict['transitions']):
            for k in d.keys():
                superdict['indices'][k] = (
                    self.tagdicttype[k], self.tagdict[k])  # keep a local copy of the indices, for transformation later
        return superdict

    # this is part of our *class* definition: list of data that can be directly assigned / read
    __HDF5list__ = ('chem', 'N', 'Nthermo', 'NGFmax', 'invmap',
                    'thermo2kin', 'kin2vacancy', 'outerkin', 'vstar2kin',
                    'om1_jt', 'om1_SP', 'om2_jt', 'om2_SP',
                    'GFexpansion',
                    'Dom1_om0', 'Dom1', 'Dom2_om0', 'Dom2',
                    'om1_om0', 'om1_om0escape', 'om1expansion', 'om1escape',
                    'om2_om0', 'om2_om0escape', 'om2expansion', 'om2escape',
                    'om1_b0', 'om1bias', 'om2_b0', 'om2bias',
                    'OSindices', 'OSfolddown', 'OS_VB', 'OSVfolddown',
                    'kineticsvWyckoff', 'omega0vacancyWyckoff')
    __taglist__ = ('vacancy', 'solute', 'solute-vacancy', 'omega0', 'omega1', 'omega2')

    def addhdf5(self, HDF5group):
        """
        Adds an HDF5 representation of object into an HDF5group (needs to already exist).

        Example: if f is an open HDF5, then VacancyMediated.addhdf5(f.create_group('Diffuser')) will
        (1) create the group named 'Diffuser', and then (2) put the VacancyMediated representation in that group.

        :param HDF5group: HDF5 group
        """
        HDF5group.attrs['type'] = self.__class__.__name__
        HDF5group['crystal_yaml'] = crystal.yaml.dump(self.crys)
        HDF5group['crystal_yaml'].attrs['pythonrep'] = self.crys.__repr__()
        HDF5group['crystal_lattice'] = self.crys.lattice.T
        basislist, basisindex = stars.doublelist2flatlistindex(self.crys.basis)
        HDF5group['crystal_basisarray'], HDF5group['crystal_basisindex'] = \
            np.array(basislist), basisindex
        # a long way around, but if you want to store an array of variable length strings, this is how to do it:
        # import h5py
        # HDF5group.create_dataset('crystal_chemistry', data=np.array(self.crys.chemistry, dtype=object),
        #                          dtype=h5py.special_dtype(vlen=str))
        HDF5group['crystal_chemistry'] = np.array(self.crys.chemistry, dtype='S')
        # arrays that we can deal with:
        for internal in self.__HDF5list__:
            HDF5group[internal] = getattr(self, internal)
        # convert jumplist:
        jumplist, jumpindex = stars.doublelist2flatlistindex(self.jumpnetwork)
        HDF5group['jump_ij'], HDF5group['jump_dx'], HDF5group['jump_index'] = \
            np.array([np.array((i, j)) for ((i, j), dx) in jumplist]), \
            np.array([dx for ((i, j), dx) in jumplist]), \
            jumpindex
        # objects with their own addhdf5 functionality:
        self.GFcalc.addhdf5(HDF5group.create_group('GFcalc'))
        self.thermo.addhdf5(HDF5group.create_group('thermo'))
        self.NNstar.addhdf5(HDF5group.create_group('NNstar'))
        self.kinetic.addhdf5(HDF5group.create_group('kinetic'))
        self.vkinetic.addhdf5(HDF5group.create_group('vkinetic'))
        self.GFstarset.addhdf5(HDF5group.create_group('GFstarset'))

        # jump networks:
        jumplist, jumpindex = stars.doublelist2flatlistindex(self.om1_jn)
        HDF5group['omega1_ij'], HDF5group['omega1_dx'], HDF5group['omega1_index'] = \
            np.array([np.array((i, j)) for ((i, j), dx) in jumplist]), \
            np.array([dx for ((i, j), dx) in jumplist]), \
            jumpindex

        jumplist, jumpindex = stars.doublelist2flatlistindex(self.om2_jn)
        HDF5group['omega2_ij'], HDF5group['omega2_dx'], HDF5group['omega2_index'] = \
            np.array([np.array((i, j)) for ((i, j), dx) in jumplist]), \
            np.array([dx for ((i, j), dx) in jumplist]), \
            jumpindex

        HDF5group['kin2vstar_array'], HDF5group['kin2vstar_index'] = \
            stars.doublelist2flatlistindex(self.kin2vstar)

        if self.GFvalues != {}:
            HDF5group['GFvalues_vTK'], HDF5group['GFvalues_values'], HDF5group['GFvalues_splits'] = \
                vTKdict2arrays(self.GFvalues)
            HDF5group['Lvvvalues_vTK'], HDF5group['Lvvvalues_values'], HDF5group['Lvvvalues_splits'] = \
                vTKdict2arrays(self.Lvvvalues)
            HDF5group['etavvalues_vTK'], HDF5group['etavvalues_values'], HDF5group['etavvalues_splits'] = \
                vTKdict2arrays(self.etavvalues)

        # tags
        for tag in self.__taglist__:
            taglist, tagindex = stars.doublelist2flatlistindex(self.tags[tag])
            HDF5group[tag + '_taglist'], HDF5group[tag + '_tagindex'] = np.array(taglist, dtype='S'), tagindex

    @classmethod
    def loadhdf5(cls, HDF5group):
        """
        Creates a new VacancyMediated diffuser from an HDF5 group.

        :param HDFgroup: HDF5 group
        :return VacancyMediated: new VacancyMediated diffuser object from HDF5
        """
        diffuser = cls(None, None, None, None)  # initialize
        diffuser.crys = crystal.yaml.load(HDF5group['crystal_yaml'].value)
        diffuser.dim = diffuser.crys.dim
        for internal in cls.__HDF5list__:
            setattr(diffuser, internal, HDF5group[internal].value)
        diffuser.sitelist = [[] for i in range(max(diffuser.invmap) + 1)]
        for i, site in enumerate(diffuser.invmap):
            diffuser.sitelist[site].append(i)

        # convert jumplist:
        diffuser.jumpnetwork = stars.flatlistindex2doublelist([((ij[0], ij[1]), dx) for ij, dx in \
                                                               zip(HDF5group['jump_ij'].value,
                                                                   HDF5group['jump_dx'].value)],
                                                              HDF5group['jump_index'])
        diffuser.om0_jn = copy.deepcopy(diffuser.jumpnetwork)

        # objects with their own addhdf5 functionality:
        diffuser.GFcalc = GFcalc.GFCrystalcalc.loadhdf5(diffuser.crys, HDF5group['GFcalc'])
        diffuser.thermo = stars.StarSet.loadhdf5(diffuser.crys, HDF5group['thermo'])
        diffuser.NNstar = stars.StarSet.loadhdf5(diffuser.crys, HDF5group['NNstar'])
        diffuser.kinetic = stars.StarSet.loadhdf5(diffuser.crys, HDF5group['kinetic'])
        diffuser.vkinetic = stars.VectorStarSet.loadhdf5(diffuser.kinetic, HDF5group['vkinetic'])
        diffuser.GFstarset = stars.StarSet.loadhdf5(diffuser.crys, HDF5group['GFstarset'])

        # jump networks:
        diffuser.om1_jn = stars.flatlistindex2doublelist([((ij[0], ij[1]), dx) for ij, dx in \
                                                          zip(HDF5group['omega1_ij'].value,
                                                              HDF5group['omega1_dx'].value)], HDF5group['omega1_index'])
        diffuser.om2_jn = stars.flatlistindex2doublelist([((ij[0], ij[1]), dx) for ij, dx in \
                                                          zip(HDF5group['omega2_ij'].value,
                                                              HDF5group['omega2_dx'].value)], HDF5group['omega2_index'])

        diffuser.kin2vstar = stars.flatlistindex2doublelist(HDF5group['kin2vstar_array'],
                                                            HDF5group['kin2vstar_index'])
        if 'GFvalues_vTK' in HDF5group:
            diffuser.GFvalues = arrays2vTKdict(HDF5group['GFvalues_vTK'],
                                               HDF5group['GFvalues_values'],
                                               HDF5group['GFvalues_splits'])
            diffuser.Lvvvalues = arrays2vTKdict(HDF5group['Lvvvalues_vTK'],
                                                HDF5group['Lvvvalues_values'],
                                                HDF5group['Lvvvalues_splits'])
            diffuser.etavvalues = arrays2vTKdict(HDF5group['etavvalues_vTK'],
                                                 HDF5group['etavvalues_values'],
                                                 HDF5group['etavvalues_splits'])
        else:
            diffuser.GFvalues, diffuser.Lvvvalues, diffuser.etavvalues = {}, {}, {}
        # tags
        diffuser.tags, diffuser.tagdict, diffuser.tagdicttype = {}, {}, {}
        for tag in cls.__taglist__:
            # needed because of how HDF5 stores strings...
            utf8list = [str(data, encoding='utf-8') for data in HDF5group[tag + '_taglist'].value]
            diffuser.tags[tag] = stars.flatlistindex2doublelist(utf8list, HDF5group[tag + '_tagindex'])
        for tagtype, taglist in diffuser.tags.items():
            for i, tags in enumerate(taglist):
                for tag in tags: diffuser.tagdict[tag], diffuser.tagdicttype[tag] = i, tagtype
        return diffuser

    def interactlist(self):
        """
        Return a list of solute-vacancy configurations for interactions. The points correspond
        to a vector between a solute atom and a vacancy. Defined by Stars.

        :return statelist: list of PairStates for the solute-vacancy interactions
        """
        if 0 == getattr(self, 'Nthermo', 0): raise ValueError('Need to set thermodynamic range first')
        return [self.thermo.states[s[0]] for s in self.thermo.stars]

    def omegalist(self, fivefreqindex=1):
        """
        Return a list of pairs of endpoints for a vacancy jump, corresponding to omega1 or omega2
        Solute at the origin, vacancy hopping between two sites. Defined by om1_jumpnetwork

        :param fivefreqindex: 1 or 2, corresponding to omega1 or omega2
        :return omegalist: list of tuples of PairStates
        :return omegajumptype: index of corresponding omega0 jumptype
        """
        if 0 == getattr(self, 'Nthermo', 0): raise ValueError('Need to set thermodynamic range first')
        om, jt = {1: (self.om1_jn, self.om1_jt),
                  2: (self.om2_jn, self.om2_jt)}.get(fivefreqindex, (None, None))
        if om is None: raise ValueError('Five frequency index should be 1 or 2')
        return [(self.kinetic.states[jlist[0][0][0]], self.kinetic.states[jlist[0][0][1]]) for jlist in om], \
               jt.copy()

    def maketracerpreene(self, preT0, eneT0, **ignoredextraarguments):
        """
        Generates corresponding energies / prefactors for an isotopic tracer. Returns a dictionary.
        (we ignore extra arguments so that a dictionary including additional entries can be passed)

        :param preT0[Nomeg0]: prefactor for vacancy jump transitions (follows jumpnetwork)
        :param eneT0[Nomega0]: transition energy state for vacancy jumps
        :return preS[NWyckoff]: prefactor for solute formation
        :return eneS[NWyckoff]: solute formation energy
        :return preSV[Nthermo]: prefactor for solute-vacancy interaction
        :return eneSV[Nthermo]: solute-vacancy binding energy
        :return preT1[Nomega1]: prefactor for omega1-style transitions (follows om1_jn)
        :return eneT1[Nomega1]: transition energy for omega1-style jumps
        :return preT2[Nomega2]: prefactor for omega2-style transitions (follows om2_jn)
        :return eneT2[Nomega2]: transition energy for omega2-style jumps
        """
        preS = np.ones(len(self.sitelist))
        eneS = np.zeros(len(self.sitelist))
        preSV = np.ones(self.thermo.Nstars)
        eneSV = np.zeros(self.thermo.Nstars)
        preT1 = np.ones(len(self.om1_jn))
        eneT1 = np.zeros(len(self.om1_jn))
        for j, jt in zip(itertools.count(), self.om1_jt): preT1[j], eneT1[j] = preT0[jt], eneT0[jt]
        preT2 = np.ones(len(self.om2_jn))
        eneT2 = np.zeros(len(self.om2_jn))
        for j, jt in zip(itertools.count(), self.om2_jt): preT2[j], eneT2[j] = preT0[jt], eneT0[jt]
        return {'preS': preS, 'eneS': eneS, 'preSV': preSV, 'eneSV': eneSV,
                'preT1': preT1, 'eneT1': eneT1, 'preT2': preT2, 'eneT2': eneT2}

    def makeLIMBpreene(self, preS, eneS, preSV, eneSV, preT0, eneT0, **ignoredextraarguments):
        """
        Generates corresponding energies / prefactors for corresponding to LIMB
        (Linearized interpolation of migration barrier approximation). Returns a dictionary.
        (we ignore extra arguments so that a dictionary including additional entries can be passed)

        :param preS[NWyckoff]: prefactor for solute formation
        :param eneS[NWyckoff]: solute formation energy
        :param preSV[Nthermo]: prefactor for solute-vacancy interaction
        :param eneSV[Nthermo]: solute-vacancy binding energy
        :param preT0[Nomeg0]: prefactor for vacancy jump transitions (follows jumpnetwork)
        :param eneT0[Nomega0]: transition energy for vacancy jumps
        :return preT1[Nomega1]: prefactor for omega1-style transitions (follows om1_jn)
        :return eneT1[Nomega1]: transition energy/kBT for omega1-style jumps
        :return preT2[Nomega2]: prefactor for omega2-style transitions (follows om2_jn)
        :return eneT2[Nomega2]: transition energy/kBT for omega2-style jumps
        """
        # we need the prefactors and energies for all of our kinetic stars... without the
        # vacancy part (since that reference is already in preT0 and eneT0); we're going
        # to add these to preT0 and eneT0 to get the TS prefactor/energy for w1 and w2 jumps
        eneSVkin = np.array([eneS[s] for (s, v) in self.kineticsvWyckoff], dtype=float)  # avoid ints
        preSVkin = np.array([preS[s] for (s, v) in self.kineticsvWyckoff], dtype=float)  # avoid ints
        for tindex, kindex in enumerate(self.thermo2kin):
            eneSVkin[kindex] += eneSV[tindex]
            preSVkin[kindex] *= preSV[tindex]
        preT1 = np.ones(len(self.om1_jn))
        eneT1 = np.zeros(len(self.om1_jn))
        for j, jt, SP in zip(itertools.count(), self.om1_jt, self.om1_SP):
            # need to include solute energy / prefactors
            preT1[j] = preT0[jt] * np.sqrt(preSVkin[SP[0]] * preSVkin[SP[1]])
            eneT1[j] = eneT0[jt] + 0.5 * (eneSVkin[SP[0]] + eneSVkin[SP[1]])
        preT2 = np.ones(len(self.om2_jn))
        eneT2 = np.zeros(len(self.om2_jn))
        for j, jt, SP in zip(itertools.count(), self.om2_jt, self.om2_SP):
            # need to include solute energy / prefactors
            preT2[j] = preT0[jt] * np.sqrt(preSVkin[SP[0]] * preSVkin[SP[1]])
            eneT2[j] = eneT0[jt] + 0.5 * (eneSVkin[SP[0]] + eneSVkin[SP[1]])
        return {'preT1': preT1, 'eneT1': eneT1, 'preT2': preT2, 'eneT2': eneT2}

    def tags2preene(self, usertagdict, VERBOSE=False):
        """
        Generates energies and prefactors based on a dictionary of tags.

        :param usertagdict: dictionary where the keys are tags, and the values are tuples: (pre, ene)
        :param VERBOSE: (optional) if True, also return a dictionary of missing tags, duplicate tags, and bad tags
        :return thermodict: dictionary of ene's and pre's corresponding to usertagdict
        :return missingdict: dictionary with keys corresponding to tag types, and the values are
          lists of lists of symmetry equivalent tags that are missing
        :return duplicatelist: list of lists of tags in usertagdict that are (symmetry) duplicates
        :return badtaglist: list of all tags in usertagdict that aren't found in our dictionary
        """
        N, Nst, Nom0 = len(self.sitelist), self.thermo.Nstars, len(self.om0_jn)
        # basic thermodict; note: we *don't* prefill omega1 and omega2, because LIMB does that later
        thermodict = {'preV': np.ones(N), 'eneV': np.zeros(N),
                      'preS': np.ones(N), 'eneS': np.zeros(N),
                      'preSV': np.ones(Nst), 'eneSV': np.zeros(Nst),
                      'preT0': np.ones(Nom0), 'eneT0': np.zeros(Nom0)}
        for tagstring, prename, enename in (('vacancy', 'preV', 'eneV'),
                                            ('solute', 'preS', 'eneS'),
                                            ('solute-vacancy', 'preSV', 'eneSV'),
                                            ('omega0', 'preT0', 'eneT0')):
            for i, tags in enumerate(self.tags[tagstring]):
                for t in tags:
                    if t in usertagdict:
                        thermodict[prename][i], thermodict[enename][i] = usertagdict[t]
                        break
        # "backfill" with LIMB so that the rest is meaningful:
        thermodict.update(self.makeLIMBpreene(**thermodict))
        for tagstring, prename, enename in (('omega1', 'preT1', 'eneT1'),
                                            ('omega2', 'preT2', 'eneT2')):
            for i, tags in enumerate(self.tags[tagstring]):
                for t in tags:
                    if t in usertagdict:
                        thermodict[prename][i], thermodict[enename][i] = usertagdict[t]
                        break
        if not VERBOSE: return thermodict
        missingdict, duplicatelist, badtaglist = {}, [], []
        tupledict = {(tagtype, n): [] for tagtype, taglist in self.tags.items() for n in range(len(taglist))}
        # go through all the types of tags and interactions, and construct a list of usertags for each
        for usertag in usertagdict:
            if usertag not in self.tagdict:
                badtaglist.append(usertag)
            else:
                tupledict[(self.tagdicttype[usertag], self.tagdict[usertag])].append(usertag)
        # each entry should appear once, and only once
        for k, v in tupledict.items():
            if len(v) == 0:
                if k[0] in missingdict:
                    missingdict[k[0]].append(self.tags[k[0]][k[1]])
                else:
                    missingdict[k[0]] = [self.tags[k[0]][k[1]]]
            elif len(v) > 1:
                duplicatelist.append(v)
        return thermodict, missingdict, duplicatelist, badtaglist

    @staticmethod
    def preene2betafree(kT, preV, eneV, preS, eneS, preSV, eneSV,
                        preT0, eneT0, preT1, eneT1, preT2, eneT2, **ignoredextraarguments):
        """
        Read in a series of prefactors (:math:`\\exp(S/k_\\text{B})`) and energies, and return
        :math:`\\beta F` for energies and transition state energies. Used to provide scaled values
        to Lij().
        Can specify all of the entries using a dictionary; e.g., ``preene2betafree(kT, **data_dict)``
        and then send that output as input to Lij: ``Lij(*preene2betafree(kT, **data_dict))``
        (we ignore extra arguments so that a dictionary including additional entries can be passed)

        :param kT: temperature times Boltzmann's constant kB
        :param preV: prefactor for vacancy formation (prod of inverse vibrational frequencies)
        :param eneV: vacancy formation energy
        :param preS: prefactor for solute formation (prod of inverse vibrational frequencies)
        :param eneS: solute formation energy
        :param preSV: excess prefactor for solute-vacancy binding
        :param eneSV: solute-vacancy binding energy
        :param preT0: prefactor for vacancy transition state
        :param eneT0: energy for vacancy transition state (relative to eneV)
        :param preT1: prefactor for vacancy swing transition state
        :param eneT1: energy for vacancy swing transition state (relative to eneV + eneS + eneSV)
        :param preT2: prefactor for vacancy exchange transition state
        :param eneT2: energy for vacancy exchange transition state (relative to eneV + eneS + eneSV)
        :return bFV: beta*eneV - ln(preV) (relative to minimum value)
        :return bFS: beta*eneS - ln(preS) (relative to minimum value)
        :return bFSV: beta*eneSV - ln(preSV) (excess)
        :return bFT0: beta*eneT0 - ln(preT0) (relative to minimum value of bFV)
        :return bFT1: beta*eneT1 - ln(preT1) (relative to minimum value of bFV + bFS)
        :return bFT2: beta*eneT2 - ln(preT2) (relative to minimum value of bFV + bFS)
        """
        # do anything to treat kT -> 0?
        beta = 1 / kT
        bFV = beta * eneV - np.log(preV)
        bFS = beta * eneS - np.log(preS)
        bFSV = beta * eneSV - np.log(preSV)
        bFT0 = beta * eneT0 - np.log(preT0)
        bFT1 = beta * eneT1 - np.log(preT1)
        bFT2 = beta * eneT2 - np.log(preT2)

        bFVmin = np.min(bFV)
        bFSmin = np.min(bFS)
        bFV -= bFVmin
        bFS -= bFSmin
        bFT0 -= bFVmin
        bFT1 -= bFVmin + bFSmin
        bFT2 -= bFVmin + bFSmin
        return bFV, bFS, bFSV, bFT0, bFT1, bFT2

    def _symmetricandescaperates(self, bFV, bFSVkinetic, bFT0, bFT1, bFT2):
        """
        Compute the symmetric, escape, and escape reference rates. Used by _lij().

        :param bFV[NWyckoff]: beta*eneV - ln(preV) (relative to minimum value)
        :param bFSVkinetic[Nkinetic]: beta*eneSV - ln(preSV) (TOTAL for solute-vacancy complex)
        :param bFT0[Nomega0]: beta*eneT0 - ln(preT0) (relative to minimum value of bFV)
        :param bFT1[Nomega1]: beta*eneT1 - ln(preT1) (relative to minimum value of bFV + bFS)
        :param bFT2[Nomega2]: beta*eneT2 - ln(preT2) (relative to minimum value of bFV + bFS)
        :return omega0[Nomega0]: symmetric rate for omega0 jumps
        :return omega1[Nomega1]: symmetric rate for omega1 jumps
        :return omega2[Nomega2]: symmetric rate for omega2 jumps
        :return omega0escape[NWyckoff, Nomega0]: escape rate elements for omega0 jumps
        :return omega1escape[NVstars, Nomega1]: escape rate elements for omega1 jumps
        :return omega2escape[NVstars, Nomega2]: escape rate elements for omega2 jumps
        """
        omega0 = np.zeros(len(self.om0_jn))
        omega0escape = np.zeros((len(self.sitelist), len(self.om0_jn)))
        for j, bF, (v1, v2) in zip(itertools.count(), bFT0, self.omega0vacancyWyckoff):
            omega0escape[v1, j] = np.exp(-bF + bFV[v1])
            omega0escape[v2, j] = np.exp(-bF + bFV[v2])
            omega0[j] = np.sqrt(omega0escape[v1, j] * omega0escape[v2, j])
        omega1 = np.zeros(len(self.om1_jn))
        omega1escape = np.zeros((self.vkinetic.Nvstars, len(self.om1_jn)))
        for j, (st1, st2), bFT in zip(itertools.count(), self.om1_SP, bFT1):
            omF, omB = np.exp(-bFT + bFSVkinetic[st1]), np.exp(-bFT + bFSVkinetic[st2])
            omega1[j] = np.sqrt(omF * omB)
            for vst1 in self.kin2vstar[st1]: omega1escape[vst1, j] = omF
            for vst2 in self.kin2vstar[st2]: omega1escape[vst2, j] = omB
        omega2 = np.zeros(len(self.om2_jn))
        omega2escape = np.zeros((self.vkinetic.Nvstars, len(self.om2_jn)))
        for j, (st1, st2), bFT in zip(itertools.count(), self.om2_SP, bFT2):
            omF, omB = np.exp(-bFT + bFSVkinetic[st1]), np.exp(-bFT + bFSVkinetic[st2])
            omega2[j] = np.sqrt(omF * omB)
            for vst1 in self.kin2vstar[st1]: omega2escape[vst1, j] = omF
            for vst2 in self.kin2vstar[st2]: omega2escape[vst2, j] = omB
        return omega0, omega1, omega2, \
               omega0escape, omega1escape, omega2escape

    def Lij(self, bFV, bFS, bFSV, bFT0, bFT1, bFT2, large_om2=1e8):
        """
        Calculates the transport coefficients: L0vv, Lss, Lsv, L1vv from the scaled free energies.
        The Green function entries are calculated from the omega0 info. As this is the most
        time-consuming part of the calculation, we cache these values with a dictionary
        and hash function.

        :param bFV[NWyckoff]: beta*eneV - ln(preV) (relative to minimum value)
        :param bFS[NWyckoff]: beta*eneS - ln(preS) (relative to minimum value)
        :param bFSV[Nthermo]: beta*eneSV - ln(preSV) (excess)
        :param bFT0[Nomega0]: beta*eneT0 - ln(preT0) (relative to minimum value of bFV)
        :param bFT1[Nomega1]: beta*eneT1 - ln(preT1) (relative to minimum value of bFV + bFS)
        :param bFT2[Nomega2]: beta*eneT2 - ln(preT2) (relative to minimum value of bFV + bFS)
        :param large_om2: threshold for changing treatment of omega2 contributions (default: 10^8)
        :return Lvv[3, 3]: vacancy-vacancy; needs to be multiplied by cv/kBT
        :return Lss[3, 3]: solute-solute; needs to be multiplied by cv*cs/kBT
        :return Lsv[3, 3]: solute-vacancy; needs to be multiplied by cv*cs/kBT
        :return Lvv1[3, 3]: vacancy-vacancy correction due to solute; needs to be multiplied by cv*cs/kBT
        """
        # 1. bare vacancy diffusivity and Green's function
        vTK = vacancyThermoKinetics(pre=np.ones_like(bFV), betaene=bFV,
                                    preT=np.ones_like(bFT0), betaeneT=bFT0)
        GF = self.GFvalues.get(vTK)
        L0vv = self.Lvvvalues.get(vTK)
        etav = self.etavvalues.get(vTK)
        if GF is None:
            # calculate, and store in dictionary for cache:
            self.GFcalc.SetRates(**(vTK._asdict()))
            L0vv = self.GFcalc.Diffusivity()
            etav = self.GFcalc.biascorrection()
            GF = np.array([self.GFcalc(PS.i, PS.j, PS.dx)
                           for PS in
                           [self.GFstarset.states[s[0]] for s in self.GFstarset.stars]])
            self.GFvalues[vTK] = GF.copy()
            self.Lvvvalues[vTK] = L0vv
            self.etavvalues[vTK] = etav

        # 2. set up probabilities for solute-vacancy configurations
        probVsites = np.array([np.exp(min(bFV) - bFV[wi]) for wi in self.invmap])
        probVsites *= self.N / np.sum(probVsites)  # normalize
        probV = np.array([probVsites[sites[0]] for sites in self.sitelist])  # Wyckoff positions
        probVsqrt = np.array([np.sqrt(probV[self.kin2vacancy[starindex]])
                              for starindex in self.vstar2kin])
        probSsites = np.array([np.exp(min(bFS) - bFS[wi]) for wi in self.invmap])
        probSsites *= self.N / np.sum(probSsites)  # normalize
        probS = np.array([probSsites[sites[0]] for sites in self.sitelist])  # Wyckoff positions
        bFSVkin = np.array([bFS[s] + bFV[v] for (s, v) in self.kineticsvWyckoff])  # NOT EXCESS: total
        prob = np.array([probS[s] * probV[v] for (s, v) in self.kineticsvWyckoff])
        for tindex, kindex in enumerate(self.thermo2kin):
            bFSVkin[kindex] += bFSV[tindex]
            prob[kindex] *= np.exp(-bFSV[tindex])
        # zero out probability of any origin states... not clear this is really needed
        for kindex, s in enumerate(self.kinetic.stars):
            if self.kinetic.states[s[0]].iszero():
                prob[kindex] = 0

        # 3. set up symmetric rates: omega0, omega1, omega2
        #    and escape rates omega0escape, omega1escape, omega2escape
        omega0, omega1, omega2, omega0escape, omega1escape, omega2escape = \
            self._symmetricandescaperates(bFV, bFSVkin, bFT0, bFT1, bFT2)

        # 4. expand out: D0ss, D0vv, domega1, domega2, bias1, bias2
        # Note: we handle the equivalent of om1_om0 for omega2 (om2_om0) differently. Those
        # jumps correspond to the vacancy *landing* on the solute site; the "origin states"
        # are treated below--they only need to be considered *if* there is broken symmetry, such
        # that we have a non-empty VectorBasis in our *unit cell* (NVB > 0)
        # 4a. Bare diffusivities
        symmprobV0 = np.array([np.sqrt(probV[i] * probV[f]) for i,f in self.omega0vacancyWyckoff])
        symmprobSV1 = np.array([np.sqrt(prob[i] * prob[f]) for i,f in self.om1_SP])
        symmprobSV2 = np.array([np.sqrt(prob[i] * prob[f]) for i,f in self.om2_SP])
        D0ss = np.dot(self.Dom2, omega2 * symmprobSV2) / self.N
        D0sv = -D0ss
        D0vv = (np.dot(self.Dom1, omega1 * symmprobSV1) -
                np.dot(self.Dom1_om0 + self.Dom2_om0, omega0 * symmprobV0)) / self.N
        D2vv = D0ss.copy()

        # 4b. Bias vectors (before correction) and rate matrices
        biasSvec = np.zeros(self.vkinetic.Nvstars)
        biasVvec = np.zeros(self.vkinetic.Nvstars)  # now, does *not* include -biasSvec
        om2 = np.dot(self.om2expansion, omega2)
        delta_om = np.dot(self.om1expansion, omega1) - np.dot(self.om1_om0, omega0) \
                   - np.dot(self.om2_om0, omega0)
        for sv, starindex in enumerate(self.vstar2kin):
            svvacindex = self.kin2vacancy[starindex]  # vacancy
            delta_om[sv, sv] += np.dot(self.om1escape[sv, :], omega1escape[sv, :]) - \
                                np.dot(self.om1_om0escape[sv, :], omega0escape[svvacindex, :]) - \
                                np.dot(self.om2_om0escape[sv, :], omega0escape[svvacindex, :])
            om2[sv, sv] += np.dot(self.om2escape[sv, :], omega2escape[sv, :])
            # note: our solute bias is negative of the contribution to the vacancy, and also the
            # reference value is 0
            biasSvec[sv] = -np.dot(self.om2bias[sv, :], omega2escape[sv, :]) * np.sqrt(prob[starindex])
            # removed the om2 contribution--will be added back in later. Separation necessary for large_om2 case
            biasVvec[sv] = np.dot(self.om1bias[sv, :], omega1escape[sv, :]) * np.sqrt(prob[starindex]) - \
                           np.dot(self.om1_b0[sv, :], omega0escape[svvacindex, :]) * probVsqrt[sv] - \
                           np.dot(self.om2_b0[sv, :], omega0escape[svvacindex, :]) * probVsqrt[sv]
            # - biasSvec[sv]
        biasVvec_om2 = -biasSvec

        # 4c. origin state corrections for solute: (corrections for vacancy appear below)
        # these corrections are due to the null space for the vacancy without solute
        if len(self.OSindices) > 0:
            # need to multiply by sqrt(probV) first
            OSprobV = self.OSfolddown*probVsqrt  # proper null space projection
            biasSbar = np.dot(OSprobV, biasSvec)
            om2bar = np.dot(OSprobV, np.dot(om2, OSprobV.T))  # OS x OS
            etaSbar = np.dot(pinv2(om2bar), biasSbar)
            dDss = np.dot(np.dot(self.vkinetic.outer[:, :, self.OSindices, :, ][:, :, :, self.OSindices],
                                 etaSbar), biasSbar) / self.N
            D0ss += dDss
            D0sv -= dDss
            biasSvec -= np.dot(om2, np.dot(OSprobV.T, etaSbar))

        # 5. compute Green function:
        G0 = np.dot(self.GFexpansion, GF)
        # Note: we first do this *just* with omega1, then ... with omega2, depending on how it behaves
        G = np.dot(np.linalg.inv(np.eye(self.vkinetic.Nvstars) + np.dot(G0, delta_om)), G0)
        # Now: to identify the omega2 contributions, we need to find all of the sv indices with a
        # non-zero contribution to om2bias. Hand been, where np.any(self.om2bias[sv,:] != 0)
        # Now, where np.any(self.om2expansion[sv,:,:] != 0)  --should we put into generatematrices?
        om2_sv_indices = [n for n in range(len(self.om2expansion)) if not np.allclose(self.om2expansion[n], 0)]
        # looks weird, but this is how we pull out a block in G corresponding to the indices in our list:
        G1 = G[om2_sv_indices, :][:, om2_sv_indices]
        om2_slice = om2[om2_sv_indices, :][:, om2_sv_indices]
        gdom2 = np.dot(G1, om2_slice)
        if np.any(np.abs(gdom2) > large_om2):
            nom2 = len(om2_sv_indices)
            om2eig, om2vec = np.linalg.eigh(om2_slice)
            G1rot = np.dot(om2vec.T, np.dot(G1, om2vec))  # rotated matrix
            # eigenvalues are sorted in ascending order, and omega2 is negative definite
            # om2min = -np.min(omega2escape)  # this is the smallest that any nonzero eigenvalue can be
            om2min = -0.5*min(om for omlist in omega2escape for om in omlist if om>0)
            nnull = next((n for n in range(nom2) if om2eig[n] > om2min), nom2)  # 0:nnull == not in nullspace
            # general update (g^-1 + w)^-1:
            G2rot = np.dot(np.linalg.inv(np.eye(nom2) + np.dot(G1rot, np.diag(om2eig))), G1rot)
            om2rot = np.diag(om2eig[0:nnull])
            # in the non-null subspace, replace with (g^-1+w)^-1-w^-1 = -(w+wgw)^-1:
            G2rot[0:nnull, 0:nnull] = -np.linalg.inv(om2rot + np.dot(om2rot,
                                                                     np.dot(G1rot[0:nnull,0:nnull],
                                                                            om2rot)))
            Greplace = np.dot(om2vec, np.dot(G2rot, om2vec.T))  # transform back
            om2_inv = np.linalg.pinv(om2_slice)  # only used here for testing purposes...
            # update with omega2, and then put in change due to omega2
            G = np.dot(np.linalg.inv(np.eye(self.vkinetic.Nvstars) + np.dot(G, om2)), G)
            Gfull = G.copy()
            for ni, i in enumerate(om2_sv_indices):
                for nj, j in enumerate(om2_sv_indices):
                    G[i, j] = Greplace[ni, nj]

            bV, bV2, bS, = biasVvec[om2_sv_indices], biasVvec_om2[om2_sv_indices], biasSvec[om2_sv_indices]
            om2_outer = self.vkinetic.outer[:, :, om2_sv_indices, :][:, :, :, om2_sv_indices]
            D0ss_correct = np.dot(np.dot(om2_outer, bS), np.dot(om2_inv, bS)) / self.N
            D0ss = np.zeros_like(D0ss)  # exact cancellation of bare term
            D0sv = np.dot(np.dot(om2_outer, bV), np.dot(om2_inv, bS)) / self.N
            D2vv = (np.dot(np.dot(om2_outer, bV), np.dot(om2_inv, bV)) +
                    2 * np.dot(np.dot(om2_outer, bV2), np.dot(om2_inv, bV))) / self.N
        else:
            # update with omega2 ("small" omega2):
            G = np.dot(np.linalg.inv(np.eye(self.vkinetic.Nvstars) + np.dot(G, om2)), G)
            Gfull = G

        # 6. Compute bias contributions to Onsager coefficients
        # 6a. add in the om2 contribution to biasVvec:
        biasVvec += biasVvec_om2

        # 6b. GF pieces:
        etaVvec, etaSvec = np.dot(G, biasVvec), np.dot(G, biasSvec)
        outer_etaVvec, outer_etaSvec = np.dot(self.vkinetic.outer, etaVvec), np.dot(self.vkinetic.outer, etaSvec)

        L1ss = np.dot(outer_etaSvec, biasSvec) / self.N
        L1sv = np.dot(outer_etaSvec, biasVvec) / self.N
        L1vv = np.dot(outer_etaVvec, biasVvec) / self.N

        # 6c. origin state corrections for vacancy:
        if len(self.OSindices) > 0:
            etaV0 = -np.tensordot(self.OS_VB, etav, axes=((1, 2), (0, 1))) * np.sqrt(self.N)
            outer_etaV0 = np.dot(self.vkinetic.outer[:, :, self.OSindices, :][:, :, :, self.OSindices], etaV0)
            dom = delta_om + om2  # sum of the terms
            # dgd = -dom + np.dot(dom, np.dot(G, dom))  # delta_g = g0*dgd*g0
            dgd = -dom + np.dot(dom, np.dot(Gfull, dom))  # delta_g = g0*dgd*g0
            G0db = np.dot(G0, biasVvec)  # G0*db
            # 2 eta0*db + 2 eta0*dgd*G0*db + eta0*dgd*eta0  (domega = delta_om + om2)
            # - etaV0*biasV0 (correction due to removing states)
            L1vv += np.dot(outer_etaV0,
                           2 * np.dot(self.OSVfolddown, biasVvec)
                           + 2 * np.dot(self.OSVfolddown, np.dot(dgd, G0db))
                           + np.dot(np.dot(self.OSVfolddown, np.dot(dgd, self.OSVfolddown.T)), etaV0)
                           - biasVvec[self.OSindices]
                           ) / self.N

        return L0vv, D0ss + L1ss, D0sv + L1sv, D0vv + D2vv + L1vv


crystal.yaml.add_representer(vacancyThermoKinetics, vacancyThermoKinetics.vacancyThermoKinetics_representer)
crystal.yaml.add_constructor(VACANCYTHERMOKINETICS_YAMLTAG, vacancyThermoKinetics.vacancyThermoKinetics_constructor)
