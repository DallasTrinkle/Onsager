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

# Onsager calculator for dumbbell mediated diffusion
import time
from scipy.linalg import pinv, pinvh

class dumbbellMediated():
    """
    Calculator class to compute dumbbell mediated solute transport coefficients.
    Here, unlike vacancies, we must compute the Green's Function for both bare pure (g0)
    and mixed(g2) dumbbells, since our Dyson equation requires so.
    Also, instead of working with crystal and chem, we work with the container objects.
    """

    def __init__(self, pdbcontainer, mdbcontainer, jnet0data, jnet2data, cutoff, solt_solv_cut, solv_solv_cut,
                 closestdistance, NGFmax=4, Nthermo=0, omega43_indices=None):
        """
        To initiate a transport coefficient calculatore, we start with the pure and mixed dumbbell containers and
        their jump networks. From these, we'll build our state and state-vector orbits all the omega 1, 2 3 and 4
        jump networks, compute the Green's functions between the various states with the Dyson equation approach,
        and then compute the transport coefficients.

        :param pdbcontainer: The container object for pure dumbbells - instance of dbStates
        :param mdbcontainer: The container object for mixed dumbbell - instance of mStates
        :param jnet0data: tuple (jnet0, jnet0_indexed) - the jumpnetworks for pure dumbbells
            jnet0 - jumps are of the form (state1, state2, c1 ,c2) - must be produced from states in pdbcontainer.
            jnet0_indexed - jumps are of the form ((i, j),d x) - indices must be matched to states in pdbcontainer.
        :param jnet2data: tuple (jnet2, jnet2_indexed) - the jumpnetworks for mixed dumbbells
            jnet2 - jumps are of the form (state1, state2, c1 ,c2) - must be produced from states in mdbcontainer.
            jnet2_indexed - jumps are of the form ((i, j), dx) - indices must be matched to states in mdbcontainer.
        :param cutoff: The maximum jump distance to be considered while building the jump networks
        :param solt_solv_cut: The collision cutoff between solute and solvent atoms
        :param solv_solv_cut: The collision cutoff between solvent and solvent atoms
        :param closestdistance: The closest distance allowable to all other atoms in the crystal.
        :param NGFmax: Parameter controlling k-point density (cf - GFcalc.py from the vacancy version)
        :param Nthermo: The number of jump-nearest neighbor sites that are to be considered within the thermodynamic
        :param self.omega43_indices - list of indices of omega43 jumps to keep.
        """
        # All the required quantities will be extracted from the containers as we move along
        self.pdbcontainer = pdbcontainer
        self.mdbcontainer = mdbcontainer
        (self.jnet0, self.jnet0_indexed), (self.jnet2, self.jnet2_indexed) = jnet0data, jnet2data
        self.omega43_indices = omega43_indices
        self.crys = pdbcontainer.crys  # we assume this is the same in both containers
        self.chem = pdbcontainer.chem

        # Create the solute invmap
        sitelist_solute = self.crys.sitelist(self.chem)
        self.invmap_solute = np.zeros(len(self.crys.basis[self.chem]), dtype=int)
        for wyckind, ls in enumerate(sitelist_solute):
            for site in ls:
                self.invmap_solute[site] = wyckind

        # self.jnet2_indexed = self.kinetic.starset.jnet2_indexed
        print("initializing thermo")
        self.thermo = stars.DBStarSet(pdbcontainer, mdbcontainer, (self.jnet0, self.jnet0_indexed),
                                    (self.jnet2, self.jnet2_indexed))

        print("initializing kin")
        self.kinetic = stars.DBStarSet(pdbcontainer, mdbcontainer, (self.jnet0, self.jnet0_indexed),
                                     (self.jnet2, self.jnet2_indexed))

        # print("initializing NN")
        # start = time.time()
        # # Note - even if empty, our starsets go out to atleast the NNstar - later we'll have to keep this in mind
        # self.NNstar = stars.StarSet(pdbcontainer, mdbcontainer, (self.jnet0, self.jnet0_indexed),
        #                             (self.jnet2, self.jnet2_indexed), 2)
        # print("2NN Shell initialization time: {}\n".format(time.time() - start))
        self.vkinetic = stars.DBVectorStars()

        # Make GF calculators.
        self.GFcalc_pure = GFcalc.GF_dumbbells(self.pdbcontainer, self.jnet0_indexed, Nmax=NGFmax, kptwt=None)
        # self.GFcalc_mixed = GF_dumbbells(self.mdbcontainer, self.jnet2_indexed, Nmax=4, kptwt=None)

        # Generate the initialized crystal and vector stars and the jumpnetworks with the kinetic shell
        self.generate(Nthermo, cutoff, solt_solv_cut, solv_solv_cut, closestdistance)

    def generate_jnets(self, cutoff, solt_solv_cut, solv_solv_cut, closestdistance):
        """
        Generate the omega 1, 3 and 4 jump networks.
        Note - for mixed dumbbells, indexing to the iorlist is the same as indexing to mixedstates, as the latter is
        just the former in the form of SdPair objects, all of which are origin states.
        """
        # first omega0 and omega2 - indexed to complexStates and mixed states
        # self.jnet2_indexed = self.vkinetic.starset.jnet2_indexed
        # self.omeg2types = self.vkinetic.starset.jnet2_types
        self.jtags2 = self.vkinetic.starset.jtags2
        # Next - omega1 - indexed to complexStates
        (self.jnet1, self.jnet1_indexed, self.jtags1), self.om1types = self.vkinetic.starset.jumpnetwork_omega1()

        # next, omega3 and omega_4, indexed to pure and mixed states
        # If data already provided, use those
        (self.jnet43, self.jnet43_indexed), (self.jnet4, self.jnet4_indexed, self.jtags4), \
        (self.jnet3, self.jnet3_indexed, self.jtags3) = self.vkinetic.starset.jumpnetwork_omega34(cutoff, solv_solv_cut,
                                                                                                  solt_solv_cut, closestdistance)

    def regenerate43(self, indices):
        """
        This will be used to extract a subset of omega 3, 4 jumps of interest.

        :param indices: list of integers corresponding to the indices of jump lists
        to keep from an existing omega 3, 4 jump network.
        """
        self.jnet43 = [self.jnet43[i] for i in indices]
        self.jnet43_indexed = [self.jnet43_indexed[i] for i in indices]

        self.jnet4 = [self.jnet4[i] for i in indices]
        self.jnet4_indexed = [self.jnet4_indexed[i] for i in indices]
        self.jtags4 = [self.jtags4[i] for i in indices]

        self.jnet3 = [self.jnet3[i] for i in indices]
        self.jnet3_indexed = [self.jnet3_indexed[i] for i in indices]
        self.jtags3 = [self.jtags3[i] for i in indices]

        self.rateExps = self.vkinetic.rateexpansion(self.jnet1, self.om1types, self.jnet43)

        # # Generate the bias expansions
        self.biases = self.vkinetic.biasexpansion(self.jnet1, self.jnet2, self.om1types, self.jnet43)

    def generate(self, Nthermo, cutoff, solt_solv_cut, solv_solv_cut, closestdistance):
        """
        Generate the thermodynamic and kinetic shells.

        :param Nthero: No. of shells (in terms of jumps starting from the solute site) to construct the thermodynamic
        shell. The kinetic shell is then constructed by taking one more jump.
        :param cutoff: the cutoff distance for the various kinds of jumps.
        :param solt_solv_cut: threshold approach distance (for collisions) between solute and solvent species.
        :param solv_solv_cut: threshold approach distance (for collisions) between two solvent atoms.
        :param closestdistance: minimum allowable distance to check for collisions with other sublattice atoms.
        """
        if Nthermo == getattr(self, "Nthermo", 0): return
        self.Nthermo = Nthermo
        print("generating thermodynamic shell")
        start = time.time()
        self.thermo.generate(Nthermo)
        print("thermodynamic shell generated: {}".format(time.time() - start))
        print("Total number of states in Thermodynamic Shell - {}, {}".format(len(self.thermo.complexStates),
                                                                              len(self.thermo.mixedstates)))
        print("generating kinetic shell")
        start = time.time()
        self.kinetic.generate(Nthermo + 1)
        print("Kinetic shell generated: {}".format(time.time() - start))
        print("Total number of states in Kinetic Shell - {}, {}".format(len(self.kinetic.complexStates),
                                                                        len(self.kinetic.mixedstates)))
        # self.Nmixedstates = len(self.kinetic.mixedstates)
        # self.NcomplexStates = len(self.kinetic.complexStates)
        print("generating kinetic shell vector starset")
        start = time.time()
        self.vkinetic.generate(self.kinetic)  # we generate the vector star out of the kinetic shell
        print("Kinetic shell vector starset generated: {}".format(time.time()-start))
        # Now generate the pure and mixed dumbbell Green functions expnsions - internalized within vkinetic.

        # Generate and indexing that takes from a star in the thermodynamic shell
        # to the corresponding star in the kinetic shell.
        self.thermo2kin = np.zeros(self.thermo.mixedstartindex, dtype=int)
        for th_ind, thstar in enumerate(self.thermo.stars[:self.thermo.mixedstartindex]):
            count = 0
            for k_ind, kstar in enumerate(self.vkinetic.starset.stars[:self.vkinetic.starset.mixedstartindex]):
                # check if the representative state of the thermo star is present in the kin star.
                if thstar[0] in set(kstar):
                    count += 1
                    self.thermo2kin[th_ind] = k_ind
            if count != 1:
                raise TypeError("thermodynamic and kinetic shells not consistent.")
        print("Generating Jump networks")
        start = time.time()
        self.generate_jnets(cutoff, solt_solv_cut, solv_solv_cut, closestdistance)
        print("Jump networks generated: {}".format(time.time() - start))

        # Generate the GF expansions
        start = time.time()
        (self.GFstarset_pure, self.GFPureStarInd, self.GFexpansion_pure) = self.vkinetic.GFexpansion()
        print("built GFstarsets: {}".format(time.time() - start))

        # generate the rate expansions
        start = time.time()
        self.rateExps = self.vkinetic.rateexpansion(self.jnet1, self.om1types, self.jnet43)
        print("built rate expansions: {}".format(time.time() - start))

        # # Generate the bias expansions
        start = time.time()
        self.biases = self.vkinetic.biasexpansion(self.jnet1, self.jnet2, self.om1types, self.jnet43)
        print("built bias expansions: {}".format(time.time() - start))
        #
        # # generate the outer products of the vector stars
        start = time.time()
        self.kinouter = self.vkinetic.outer()
        print("built outer product tensor:{}".format(time.time() - start))
        # self.clearcache()

    # staticmethod functions to compute rates and energies of isolated dumbbell states
    # These are taken from the interstitial class by Prof. Trinkle
    @staticmethod
    def stateprob(pre, betaene, invmap):
        """Returns our probabilities, normalized, as a vector.
           Straightforward extension from vacancy case.
        """
        # be careful to make sure that we don't under-/over-flow on beta*ene
        minbetaene = min(betaene)
        rho = np.array([pre[w] * np.exp(minbetaene - betaene[w]) for w in invmap])
        return rho / sum(rho)

    @staticmethod
    def ratelist(jumpnetwork, pre, betaene, preT, betaeneT, invmap):
        """Returns a list of lists of rates, matched to jumpnetwork"""
        stateene = np.array([betaene[w] for w in invmap])
        statepre = np.array([pre[w] for w in invmap])
        return [[pT * np.exp(stateene[i] - beT) / statepre[i]
                 for (i, j), dx in t]
                for t, pT, beT in zip(jumpnetwork, preT, betaeneT)]

    @staticmethod
    def symmratelist(jumpnetwork, pre, betaene, preT, betaeneT, invmap):
        """Returns a list of lists of symmetrized rates, matched to jumpnetwork"""
        stateene = np.array([betaene[w] for w in invmap])
        statepre = np.array([pre[w] for w in invmap])
        return [[pT * np.exp(0.5 * stateene[i] + 0.5 * stateene[j] - beT) / np.sqrt(statepre[i] * statepre[j])
                 for (i, j), dx in t]
                for t, pT, beT in zip(jumpnetwork, preT, betaeneT)]

    def calc_eta(self, rate0list, omega0escape):
        """
        Function to calculate the relaxation vectors due to omega_0 rates in complex states.

        :param rate0list: the non-symmetrized rate lists for the bare and mixed dumbbell spaces.
        :param omega0escape: rate expansion of omega_0 escape rates.
        """

        # The non-local bias for the complex space has to be carried out based on the omega0 jumpnetwork,
        # not the omega1 jumpnetwork.This is because all the jumps that are allowed by omega0 out of a given dumbbell
        # state are not there in omega1. That is because omega1 considers only those states that are in the kinetic
        # shell. Not outside it.

        # First, we build up G0
        W0 = np.zeros((len(self.vkinetic.starset.bareStates), len(self.vkinetic.starset.bareStates)))
        # use the indexed omega2 to fill this up - need omega2 indexed to mixed subspace of starset
        for jt, jlist in enumerate(self.jnet0_indexed):
            for jnum, ((i, j), dx) in enumerate(jlist):
                W0[i, j] += rate0list[jt][jnum]  # The unsymmetrized rate for that jump.
                W0[i, i] -= rate0list[jt][jnum]  # Add the same to the diagonal
        # Here, G0 = sum(x_s')G0(x_s') - and we have [sum(x_s')G0(x_s')][sum(x_s')W0(x_s')] = identity
        # The equation can be derived from the Fourier space inverse relations at q=0 for their symmetrized versions.
        self.G0 = pinv(W0)

        # W2 = np.zeros((len(self.kinetic.mixedstates),
        #                len(self.kinetic.mixedstates)))
        # # use the indexed omega2 to fill this up - need omega2 indexed to mixed subspace of starset
        # for jt, jlist in enumerate(self.jnet2_indexed):
        #     for jnum, ((i, j), dx) in enumerate(jlist):
        #         W2[i, j] += rate2list[jt][jnum]  # The unsymmetrized rate for that jump.
        #         W2[i, i] -= rate2list[jt][jnum]  # Add the same to the diagonal
        #
        # self.G2 = pinv(W2)
        # self.W2 = W2

        self.biasBareExpansion = self.biases[-1]

        # First check if non-local biases should be zero anyway (as is the case
        # with highly symmetric lattices - in that case vecpos_bare should be zero sized)
        if len(self.vkinetic.vecpos_bare) == 0:
            self.eta00_solvent = np.zeros((len(self.vkinetic.starset.complexStates), self.crys.dim))
            self.eta00_solute = np.zeros((len(self.vkinetic.starset.complexStates), self.crys.dim))

        # otherwise, we need to build the bare bias expansion
        else:
            # First we build up for just the bare starset

            # We first get the bias vector in the basis of the vector stars.
            # Since we are using symmetrized rates, we only need to consider them
            self.NlsolventVel_bare = np.zeros((len(self.vkinetic.starset.bareStates), self.crys.dim))

            # We evaluate the velocity vectors in the basis of vector wyckoff sets.
            # Need omega0_escape.
            velocity0SolventTotNonLoc = np.array([np.dot(self.biasBareExpansion[i, :],
                                                         omega0escape[self.vkinetic.vwycktowyck_bare[i], :])
                                                  for i in range(len(self.vkinetic.vecpos_bare))])

            # Then, we convert them to cartesian form for each state.
            for st in self.vkinetic.starset.bareStates:
                try:
                    indlist = self.vkinetic.stateToVecStar_bare[st]
                except:
                    indlist = []
                if len(indlist) != 0:
                    self.NlsolventVel_bare[self.vkinetic.starset.bareindexdict[st][0]][:] = \
                        sum([velocity0SolventTotNonLoc[tup[0]] * self.vkinetic.vecvec_bare[tup[0]][tup[1]] for tup in
                             indlist])

            # Then, we use G0 to get the eta0 vectors. The second 0 in eta00 indicates omega0 space.
            self.eta00_solvent_bare = np.tensordot(self.G0, self.NlsolventVel_bare, axes=(1, 0))
            # self.eta00_solute_bare = np.zeros_like(self.eta00_solvent_bare)

            # Now match the non-local biases for complex states to the pure states
            self.eta00_solvent = np.zeros((len(self.vkinetic.starset.complexStates), self.crys.dim))
            # self.eta00_solute = np.zeros((len(self.vkinetic.starset.complexStates), self.crys.dim))
            self.NlsolventBias0 = np.zeros((len(self.vkinetic.starset.complexStates), self.crys.dim))

            for i, state in enumerate(self.vkinetic.starset.complexStates):
                dbstate_ind = state.db.iorind
                self.eta00_solvent[i, :] = self.eta00_solvent_bare[dbstate_ind, :]
                self.NlsolventBias0[i, :] = self.NlsolventVel_bare[dbstate_ind, :]

        self.eta0total_solvent = np.zeros((len(self.vkinetic.starset.complexStates) +
                                           len(self.vkinetic.starset.mixedstates), self.crys.dim))

        # Just copy the portion for the complex states, leave mixed dumbbell state space as zeros.
        self.eta0total_solvent[:len(self.vkinetic.starset.complexStates), :] = self.eta00_solvent.copy()
        # self.eta0total_solvent[len(self.vkinetic.starset.complexStates):, :] = self.eta02_solvent.copy()
        # self.eta0total_solute[len(self.vkinetic.starset.complexStates):, :] = self.eta02_solute.copy()

    def bias_changes(self):
        """
        Function to compute changes in the solvent bias expansions based on the non-local solvent relaxation vectors
        already calculated.
        Note - function calc_eta needs to be called prior to this function to compute the omega_0 relaxations.
        """
        # create updates to the bias expansions
        # Construct the projection of eta vectors
        # self.delbias1expansion_solute = np.zeros_like(self.biases[1][0])
        self.delbias1expansion_solvent = np.zeros_like(self.biases[1][1])

        # self.delbias4expansion_solute = np.zeros_like(self.biases[4][0])
        self.delbias4expansion_solvent = np.zeros_like(self.biases[4][1])

        # self.delbias3expansion_solute = np.zeros_like(self.biases[3][0])
        self.delbias3expansion_solvent = np.zeros_like(self.biases[3][1])

        # self.delbias2expansion_solute = np.zeros_like(self.biases[2][0])
        # self.delbias2expansion_solvent = np.zeros_like(self.biases[2][0])

        if len(self.vkinetic.vecpos_bare) == 0: # and not eta2shift:
            return

        for i in range(self.vkinetic.Nvstars_pure):
            # get the representative state(its index in complexStates) and vector
            v0 = self.vkinetic.vecvec[i][0]
            st0 = self.vkinetic.starset.complexIndexdict[self.vkinetic.vecpos[i][0]][0]
            # Index of the state in the flat list
            # eta_proj_solute = np.dot(self.eta0total_solute, v0)
            # eta_proj_solvent = np.dot(self.eta0total_solvent, v0)
            # Now go through the omega1 jump network tags
            for jt, initindexdict in enumerate(self.jtags1):
                # see if there's an array corresponding to the initial state
                if not st0 in initindexdict:
                    # if the representative state does not occur as an initial state in any of the jumps, continue.
                    continue
                # self.delbias1expansion_solute[i, jt] += len(self.vkinetic.vecpos[i]) * np.sum(
                #     np.dot(initindexdict[st0], eta_proj_solute))
                else:
                    FSList = initindexdict[st0]
                    for FS in FSList:
                        self.delbias1expansion_solvent[i, jt] += len(self.vkinetic.vecpos[i]) *\
                            np.dot(v0, self.eta0total_solvent[st0] - self.eta0total_solvent[FS])

            # Now let's build it for omega4
            for jt, initindexdict in enumerate(self.jtags4):
                # see if there's an array corresponding to the initial state
                if not st0 in initindexdict:
                    continue
                # self.delbias4expansion_solute[i, jt] += len(self.vkinetic.vecpos[i]) * np.sum(
                #     np.dot(initindexdict[st0], eta_proj_solute))
                else:
                    FSList = initindexdict[st0]
                    for FS in FSList:
                        self.delbias4expansion_solvent[i, jt] += len(self.vkinetic.vecpos[i]) *\
                            np.dot(v0, self.eta0total_solvent[st0] - self.eta0total_solvent[FS])

        for i in range(self.vkinetic.Nvstars - self.vkinetic.Nvstars_pure):
            # get the representative state(its index in mixedstates) and vector
            v0 = self.vkinetic.vecvec[i + self.vkinetic.Nvstars_pure][0]
            st0 = self.vkinetic.starset.mixedindexdict[self.vkinetic.vecpos[i + self.vkinetic.Nvstars_pure][0]][0]
            # Form the projection of the eta vectors on v0
            # eta_proj_solute = np.dot(self.eta0total_solute, v0)
            # eta_proj_solvent = np.dot(self.eta0total_solvent, v0)

            # Need to update for omega3 because the solvent shift vector in the complex space is not zero.
            # Now let's build the change expansion for omega3

            for jt, initindexdict in enumerate(self.jtags3):
                # see if there's an array corresponding to the initial state
                if not (st0 + len(self.vkinetic.starset.complexStates)) in initindexdict:
                    continue
                # self.delbias3expansion_solute[i, jt] += len(self.vkinetic.vecpos[i + self.vkinetic.Nvstars_pure]) * \
                #                                         np.sum(np.dot(initindexdict[st0], eta_proj_solute))
                else:
                    FSList = initindexdict[st0 + len(self.vkinetic.starset.complexStates)]
                    for FS in FSList:
                        self.delbias3expansion_solvent[i, jt] += len(self.vkinetic.vecpos[i + self.vkinetic.Nvstars_pure]) * \
                                                                 np.dot(v0, self.eta0total_solvent[st0 + len(self.vkinetic.starset.complexStates)] -
                                                                        self.eta0total_solvent[FS])


    def update_bias_expansions(self, rate0list, omega0escape):
        """
        Updates the solvent bias expansions with the changes in the bias due to subtraction of the non-local
        (omega_0) relaxation vectors computed in bias_changes.
        Note - function bias_changes needs to be called prior to this function to compute the bias changes in the bias
        expansion due to omega_0 relaxations.
        """
        self.calc_eta(rate0list, omega0escape)
        self.bias_changes()
        # self.bias1_solute_new = self.biases[1][0] # stars.zeroclean( + self.delbias1expansion_solute)
        self.bias1_solvent_new = stars.zeroclean(self.biases[1][1] + self.delbias1expansion_solvent)

        # self.bias3_solute_new = self.biases[3][0] # stars.zeroclean( + self.delbias3expansion_solute)
        self.bias3_solvent_new = stars.zeroclean(self.biases[3][1] + self.delbias3expansion_solvent)

        # self.bias4_solute_new = self.biases[4][0] # stars.zeroclean( + self.delbias4expansion_solute)
        self.bias4_solvent_new = stars.zeroclean(self.biases[4][1] + self.delbias4expansion_solvent)

        self.bias2_solute_new = self.biases[2][0]  # stars.zeroclean( + self.delbias2expansion_solute)
        self.bias2_solvent_new = self.biases[2][1]  # + self.delbias2expansion_solvent)

    def bareExpansion(self, eta0_solvent):
        """
        Returns the expansion matrix of the uncorrelated diffusivity term in the basis of the state-vector orbits,
        grouped separately for each type of jump.

        The uncorrelated contribution expansion matrices are returned as tuples for the omega_1, omega_2, omega_3 and omega_4 jumps.
        These tuples have the form (Dexpansion_aa, Dexpansion_bb, Dexpansion_ab), where "a" corresponds to the solute and "b" to the solvent,
        and Dexpansion_aa gives the uncorrelated contribution to the solute-solute transport coefficient (L_aa) and
        so on.

        :param: The solvent relaxation vectors eta vectors in each state as obtained from the calc_eta function.
        :return D0expansion_bb: (numpy 2d array) expansion of the uncorrelated term for the solvent-solvent transport coefficient due to
        omega_0 jumps
        :return D1expansions: tuple of uncorrelated contributions due to omega_1 jumps.
        :return D2expansions: tuple of uncorrelated contributions due to omega_2 jumps.
        :return D3expansions: tuple of uncorrelated contributions due to omega_3 jumps.
        :return D4expansions: tuple of uncorrelated contributions due to omega_4 jumps.

        """
        # a = solute, b = solvent
        # eta0_solute, eta0_solvent = self.eta0total_solute, self.eta0total_solvent
        # Stores biases out of complex states, followed by mixed dumbbell states.
        jumpnetwork_omega1, jumptype, jumpnetwork_omega2, jumpnetwork_omega3, jumpnetwork_omega4 = \
            self.jnet1_indexed, self.om1types, self.jnet2_indexed, self.jnet3_indexed, \
            self.jnet4_indexed

        Ncomp = len(self.vkinetic.starset.complexStates)

        # We need the D0expansion to evaluate the modified non-local contribution
        # outside the kinetic shell.

        dim = self.crys.dim

        D0expansion_bb = np.zeros((dim, dim, len(self.jnet0)))

        # Omega1 contains the total rate and not just the change.
        D1expansion_aa = np.zeros((dim, dim, len(jumpnetwork_omega1)))
        D1expansion_bb = np.zeros((dim, dim, len(jumpnetwork_omega1)))
        D1expansion_ab = np.zeros((dim, dim, len(jumpnetwork_omega1)))

        D2expansion_aa = np.zeros((dim, dim, len(jumpnetwork_omega2)))
        D2expansion_bb = np.zeros((dim, dim, len(jumpnetwork_omega2)))
        D2expansion_ab = np.zeros((dim, dim, len(jumpnetwork_omega2)))

        D3expansion_aa = np.zeros((dim, dim, len(jumpnetwork_omega3)))
        D3expansion_bb = np.zeros((dim, dim, len(jumpnetwork_omega3)))
        D3expansion_ab = np.zeros((dim, dim, len(jumpnetwork_omega3)))

        D4expansion_aa = np.zeros((dim, dim, len(jumpnetwork_omega4)))
        D4expansion_bb = np.zeros((dim, dim, len(jumpnetwork_omega4)))
        D4expansion_ab = np.zeros((dim, dim, len(jumpnetwork_omega4)))

        # iorlist_pure = self.pdbcontainer.iorlist
        # iorlist_mixed = self.mdbcontainer.iorlist
        # Need versions for solute and solvent - solute dusplacements are zero anyway
        for k, jt, jumplist in zip(itertools.count(), jumptype, jumpnetwork_omega1):
            d0 = sum(
                0.5 * np.outer(dx + eta0_solvent[i] - eta0_solvent[j], dx + eta0_solvent[i] - eta0_solvent[j]) for
                (i, j), dx in jumplist)
            D0expansion_bb[:, :, jt] += d0
            D1expansion_bb[:, :, k] += d0
            # For solutes, don't need to do anything for omega1 and omega0 - solute does not move anyway
            # and therefore, their non-local eta corrections are also zero.

        for jt, jumplist in enumerate(jumpnetwork_omega2):
            # Build the expansions directly
            for (IS, FS), dx in jumplist:
                # o1 = iorlist_mixed[self.vkinetic.starset.mixedstates[IS].db.iorind][1]
                # o2 = iorlist_mixed[self.vkinetic.starset.mixedstates[FS].db.iorind][1]
                dx_solute = dx #+ eta0_solute[Ncomp + IS] - eta0_solute[Ncomp + FS]  # + o2 / 2. - o1 / 2.
                dx_solvent = dx + eta0_solvent[Ncomp + IS] - eta0_solvent[Ncomp + FS]  # - o2 / 2. + o1 / 2.
                D2expansion_aa[:, :, jt] += 0.5 * np.outer(dx_solute, dx_solute)
                D2expansion_bb[:, :, jt] += 0.5 * np.outer(dx_solvent, dx_solvent)
                D2expansion_ab[:, :, jt] += 0.5 * np.outer(dx_solute, dx_solvent)

        for jt, jumplist in enumerate(jumpnetwork_omega3):
            for (IS, FS), dx in jumplist:
                # o1 = iorlist_mixed[self.vkinetic.starset.mixedstates[IS].db.iorind][1]
                # dx_solute = np.zeros(self.crys.dim) # eta0_solute[Ncomp + IS] - eta0_solute[FS]  # -o1 / 2.
                dx_solvent = dx + eta0_solvent[Ncomp + IS] - eta0_solvent[FS]  # + o1 / 2.
                # D3expansion_aa[:, :, jt] += 0.5 * np.outer(dx_solute, dx_solute)
                D3expansion_bb[:, :, jt] += 0.5 * np.outer(dx_solvent, dx_solvent)
                # D3expansion_ab[:, :, jt] += 0.5 * np.outer(dx_solute, dx_solvent)

        for jt, jumplist in enumerate(jumpnetwork_omega4):
            for (IS, FS), dx in jumplist:
                # o2 = iorlist_mixed[self.vkinetic.starset.mixedstates[FS].db.iorind][1]
                # dx_solute = eta0_solute[IS] - eta0_solute[Ncomp + FS]  # o2 / 2. +
                dx_solvent = dx + eta0_solvent[IS] - eta0_solvent[Ncomp + FS]  # - o2 / 2.
                # D4expansion_aa[:, :, jt] += 0.5 * np.outer(dx_solute, dx_solute)
                D4expansion_bb[:, :, jt] += 0.5 * np.outer(dx_solvent, dx_solvent)
                # D4expansion_ab[:, :, jt] += 0.5 * np.outer(dx_solute, dx_solvent)

        zeroclean = stars.zeroclean

        return zeroclean(D0expansion_bb), \
               (zeroclean(D1expansion_aa), zeroclean(D1expansion_bb), zeroclean(D1expansion_ab)), \
               (zeroclean(D2expansion_aa), zeroclean(D2expansion_bb), zeroclean(D2expansion_ab)), \
               (zeroclean(D3expansion_aa), zeroclean(D3expansion_bb), zeroclean(D3expansion_ab)), \
               (zeroclean(D4expansion_aa), zeroclean(D4expansion_bb), zeroclean(D4expansion_ab))

    # noinspection SpellCheckingInspection
    @staticmethod
    def preene2betafree(kT, predb0, enedb0, preS, eneS, preSdb, eneSdb, predb2, enedb2, preT0, eneT0, preT2, eneT2,
                        preT1, eneT1, preT43, eneT43):
        """
        Similar to the function for vacancy mediated OnsagerCalc. Takes in the energies and entropic pre-factors for
        the states and transition states and returns the corresponding free energies. The difference from the vacancy case
        is the consideration of more types of states ans transition states.

        Parameters:
            pre* - entropic pre-factors
            ene* - state/transition state energies.
        The pre-factors for pure dumbbells are matched to the symmorlist. For mixed dumbbells the mixedstarset and
        symmorlist are equivalent and the pre-factors are energies are matched to these.
        For solute-dumbbell complexes, the pre-factors and the energies are matched to the star set.

        Note - for the solute-dumbbell complexes, eneSdb and preSdb are the binding (excess) energies and pre
        factors respectively. We need to evaluate the total configuration energy separately.

        For all the transitions, the pre-factors and energies for transition states are matched to symmetry-unique jump types.

        Returns :
        bFdb0, bFdb2, bFS, bFSdb, bFT0, bFT1, bFT2, bFT3, bFT4
        the free energies for the states and transition states. Used in L_ij() and getsymmrates() to get the
        symmetrized transition rates.


        """
        beta = 1. / kT
        bFdb0 = beta * enedb0 - np.log(predb0)
        bFdb2 = beta * enedb2 - np.log(predb2)
        bFS = beta * eneS - np.log(preS)
        bFSdb = beta * eneSdb - np.log(preSdb)

        bFT0 = beta * eneT0 - np.log(preT0)
        bFT1 = beta * eneT1 - np.log(preT1)
        bFT2 = beta * eneT2 - np.log(preT2)
        bFT3 = beta * eneT43 - np.log(preT43)
        bFT4 = beta * eneT43 - np.log(preT43)

        # Now, shift
        bFdb0_min = np.min(bFdb0)
        bFdb2_min = np.min(bFdb2)
        bFS_min = np.min(bFS)

        # bFdb0 -= bFdb0_min
        # bFdb2 -= bFdb2_min
        # bFS -= bFS_min
        # The unshifted values are required to be able to normalize the state probabilities.
        # See the L_ij function for details
        bFT0 -= bFdb0_min
        bFT2 -= bFdb2_min
        bFT3 -= bFdb2_min
        bFT1 -= (bFS_min + bFdb0_min)
        bFT4 -= (bFS_min + bFdb0_min)

        return bFdb0, bFdb2, bFS, bFSdb, bFT0, bFT1, bFT2, bFT3, bFT4

    def getsymmrates(self, bFdb0, bFdb2, bFSdb, bFT0, bFT1, bFT2, bFT3, bFT4):
        """
        :param bFdb0: beta * ene_db0 - ln(pre_db0) - relative to bFdb0min
        :param bFdb2: beta * ene_db2 - ln(pre_db2) - relative to bFdb2min
        :param bFSdb: beta * ene_Sdb - ln(pre_Sdb) - Total (not excess) - Relative to bFdb0min + bFSmin
        :param bFT0: beta * ene_T0 - ln(pre_T0) - relative to bFdb0min
        :param bFT1: beta * ene_T1 - ln(pre_T1) - relative to bFdb0min + bFSmin
        :param bFT2: beta * ene_T2 - ln(pre_T2) - relative to bFdb2min
        :param bFT3: beta * ene_T3 - ln(pre_T3) - relative to bFdb2min
        :param bFT4: beta * ene_T4 - ln(pre_T4) - relative to bFdb0min + bFSmin
        :return:
        """
        Nvstars_mixed = self.vkinetic.Nvstars - self.vkinetic.Nvstars_pure

        omega0 = np.zeros(len(self.jnet0))
        omega0escape = np.zeros((len(self.pdbcontainer.symorlist), len(self.jnet0)))

        omega2 = np.zeros(len(self.jnet2))
        omega2escape = np.zeros((Nvstars_mixed, len(self.jnet2)))

        omega1 = np.zeros(len(self.jnet1))
        omega1escape = np.zeros((self.vkinetic.Nvstars_pure, len(self.jnet1)))

        omega3 = np.zeros(len(self.jnet3))
        omega3escape = np.zeros((Nvstars_mixed, len(self.jnet3)))

        omega4 = np.zeros(len(self.jnet4))
        omega4escape = np.zeros((self.vkinetic.Nvstars_pure, len(self.jnet4)))

        # build the omega0 lists
        for jt, jlist in enumerate(self.jnet0):
            # Get the bare dumbbells between which jumps are occurring
            st1 = jlist[0].state1 - jlist[0].state1.R
            st2 = jlist[0].state2 - jlist[0].state2.R

            # get the symorindex of the states - these serve analogous to Wyckoff sets
            w1 = self.vkinetic.starset.pdbcontainer.invmap[self.vkinetic.starset.pdbcontainer.db2ind(st1)]
            w2 = self.vkinetic.starset.pdbcontainer.invmap[self.vkinetic.starset.pdbcontainer.db2ind(st2)]

            omega0escape[w1, jt] = np.exp(-bFT0[jt] + bFdb0[w1])
            omega0escape[w2, jt] = np.exp(-bFT0[jt] + bFdb0[w2])
            omega0[jt] = np.sqrt(omega0escape[w1, jt] * omega0escape[w2, jt])

        # we need omega2 only for the uncorrelated contributions.
        for jt, jlist in enumerate(self.jnet2):
            st1 = jlist[0].state1 - jlist[0].state1.R_s
            st2 = jlist[0].state2 - jlist[0].state2.R_s

            crStar1 = self.vkinetic.starset.mdbcontainer.invmap[self.vkinetic.starset.mdbcontainer.db2ind(st1.db)]
            crStar2 = self.vkinetic.starset.mdbcontainer.invmap[self.vkinetic.starset.mdbcontainer.db2ind(st2.db)]

            init2TS = np.exp(-bFT2[jt] + bFdb2[crStar1])
            fin2TS = np.exp(-bFT2[jt] + bFdb2[crStar2])

            omega2[jt] = np.sqrt(init2TS * fin2TS)

            # get the vector stars
            try:
                v1list = self.vkinetic.stateToVecStar_mixed[st1]
                v2list = self.vkinetic.stateToVecStar_mixed[st2]
            except KeyError:
                raise ValueError("Empty vector star for mixed state?")

            for (v1, in_v1) in v1list:
                omega2escape[v1 - self.vkinetic.Nvstars_pure, jt] = init2TS

            for (v2, in_v2) in v2list:
                omega2escape[v2 - self.vkinetic.Nvstars_pure, jt] = fin2TS

        # build the omega1 lists
        for jt, jlist in enumerate(self.jnet1):

            st1 = jlist[0].state1
            st2 = jlist[0].state2

            if st1.is_zero(self.vkinetic.starset.pdbcontainer) or st2.is_zero(self.vkinetic.starset.pdbcontainer):
                continue

            # get the crystal stars of the representative jumps
            crStar1 = self.vkinetic.starset.complexIndexdict[st1][1]
            crStar2 = self.vkinetic.starset.complexIndexdict[st2][1]

            init2TS = np.exp(-bFT1[jt] + bFSdb[crStar1])
            fin2TS = np.exp(-bFT1[jt] + bFSdb[crStar2])

            omega1[jt] = np.sqrt(init2TS * fin2TS)

            # Get the vector stars where they are located
            try:
                v1list = self.vkinetic.stateToVecStar_pure[st1]
                v2list = self.vkinetic.stateToVecStar_pure[st2]
            except:
                continue

            for (v1, in_v1) in v1list:
                omega1escape[v1, jt] = init2TS

            for (v2, in_v2) in v2list:
                omega1escape[v2, jt] = fin2TS

        # Next, we need to build the lists for omega3 and omega4 lists
        for jt, jlist in enumerate(self.jnet43):

            # The first state is a complex state, the second state is a mixed state.
            # This has been checked in test_crystal stars - look it up
            st1 = jlist[0].state1
            st2 = jlist[0].state2 - jlist[0].state2.R_s
            # If the solutes are not already at the origin, there is some error and it will show up
            # while getting the crystal stars.

            # get the crystal stars
            crStar1 = self.vkinetic.starset.complexIndexdict[st1][1]
            crStar2 = self.vkinetic.starset.mixedindexdict[st2][1] - self.vkinetic.starset.mixedstartindex
            # crStar2 is the same as the "Wyckoff" index for the mixed dumbbell state.

            init2TS = np.exp(-bFT4[jt] + bFSdb[crStar1])  # complex (bFSdb) to transition state
            fin2TS = np.exp(-bFT3[jt] + bFdb2[crStar2])  # mixed (bFdb2) to transition state.

            # symmetrized rates for omega3 and omega4 are equal
            omega4[jt] = np.sqrt(init2TS * fin2TS)
            omega3[jt] = omega4[jt]  # symmetry condition : = np.sqrt(fin2ts * init2Ts)

            # get the vector stars
            try:
                v1list = self.vkinetic.stateToVecStar_pure[st1]
                v2list = self.vkinetic.stateToVecStar_mixed[st2]
            except:
                continue

            for (v1, in_v1) in v1list:
                omega4escape[v1, jt] = init2TS

            for (v2, in_v2) in v2list:
                omega3escape[v2 - self.vkinetic.Nvstars_pure, jt] = fin2TS

        return (omega0, omega0escape), (omega1, omega1escape), (omega2, omega2escape), (omega3, omega3escape), \
               (omega4, omega4escape)

    def makeGF(self, bFdb0, bFT0, omegas, mixed_prob):
        """
        Constructs the N_vs x N_vs GF matrix.
        """
        # if not hasattr(self, 'G2'):
        #     raise AttributeError("G2 not found yet. Please run calc_eta first.")

        Nvstars_pure = self.vkinetic.Nvstars_pure

        (rate0expansion, rate0escape), (rate1expansion, rate1escape), (rate2expansion, rate2escape), \
        (rate3expansion, rate3escape), (rate4expansion, rate4escape) = self.rateExps

        # omega2 and omega2escape will not be needed here, but we still need them to calculate the uncorrelated part.
        (omega0, omega0escape), (omega1, omega1escape), (omega2, omega2escape), (omega3, omega3escape), \
        (omega4, omega4escape) = omegas

        GF02 = np.zeros((self.vkinetic.Nvstars, self.vkinetic.Nvstars))

        # left-upper part of GF02 = Nvstars_pure x Nvstars_pure g0 matrix
        # right-lower part of GF02 = Nvstars_mixed x Nvstars_mixed g2 matrix

        pre0, pre0T = np.ones_like(bFdb0), np.ones_like(bFT0)

        # Make g2 from omega2 and omega3 (escapes)
        om23 = np.zeros((self.vkinetic.Nvstars - Nvstars_pure, self.vkinetic.Nvstars - Nvstars_pure))

        # off diagonal elements of om23
        om23[:, :] += np.dot(rate2expansion, omega2)

        # Next, omega2 escape terms
        for i in range(self.vkinetic.Nvstars - Nvstars_pure):
            om23[i, i] += np.dot(rate2escape[i, :], omega2escape[i, :])

        # omega3 escapes
        for i in range(self.vkinetic.Nvstars - Nvstars_pure):
            om23[i, i] += np.dot(rate3escape[i, :], omega3escape[i, :])

        # Then invert it
        GF2 = pinvh(om23)

        self.GFcalc_pure.SetRates(pre0, bFdb0, pre0T, bFT0)

        GF0 = np.array([self.GFcalc_pure(tup[0][0], tup[0][1], tup[1]) for tup in
                        [star[0] for star in self.GFstarset_pure]])

        GF02[Nvstars_pure:, Nvstars_pure:] = GF2
        GF02[:Nvstars_pure, :Nvstars_pure] = np.dot(self.GFexpansion_pure, GF0)

        # make delta omega
        delta_om = np.zeros((self.vkinetic.Nvstars, self.vkinetic.Nvstars))

        # off-diagonals
        delta_om[:Nvstars_pure, :Nvstars_pure] += np.dot(rate1expansion, omega1) - np.dot(rate0expansion, omega0)
        delta_om[Nvstars_pure:, :Nvstars_pure] += np.dot(rate3expansion, omega3)
        delta_om[:Nvstars_pure, Nvstars_pure:] += np.dot(rate4expansion, omega4)

        # escapes
        # omega1 and omega4 terms
        for i, starind in enumerate(self.vkinetic.vstar2star[:Nvstars_pure]):
            #######
            symindex = self.vkinetic.starset.star2symlist[starind]
            delta_om[i, i] += \
                np.dot(rate1escape[i, :], omega1escape[i, :]) - \
                np.dot(rate0escape[i, :], omega0escape[symindex, :]) + \
                np.dot(rate4escape[i, :], omega4escape[i, :])

        GF_total = np.dot(np.linalg.inv(np.eye(self.vkinetic.Nvstars) + np.dot(GF02, delta_om)), GF02)

        return stars.zeroclean(GF_total), GF02, delta_om

    def L_ij(self, bFdb0, bFT0, bFdb2, bFT2, bFS, bFSdb, bFT1, bFT3, bFT4):

        """
        bFdb0[i] = beta*ene_pdb[i] - ln(pre_pdb[i]), i=1,2...,N_pdbcontainer.symorlist - pure dumbbell free energy
        bFdb2[i] = beta*ene_mdb[i] - ln(pre_mdb[i]), i=1,2...,N_mdbcontainer.symorlist - mixed dumbbell free energy
        bFS[i] = beta*ene_S[i] - _ln(pre_S[i]), i=1,2,..N_Wyckoff - site free energy for solute.
        THE ABOVE THREE VALUES ARE NOT SHIFTED RELATIVE TO THEIR RESPECTIVE MINIMUM VALUES.
        We need them to be unshifted to be able to normalize the state probabilities, which requires complex and
        mixed dumbbell energies to be with respect to the same reference. Shifting with their respective minimum values
        disturbs this.
        Wherever shifting is required, it is done in-place.

        bFSdb - beta*ene_Sdb[i] - ln(pre_Sdb[i]) [i=1,2...,mixedstartindex](binding)] excess free energy of interaction
        between a solute and a pure dumbbell in it's vicinity. This must be non-zero only for states within the
        thermodynamic shell. So the size is restricted to the number of thermodynamic crystal stars.

        Jump barrier free energies (See preene2betaene for details):
        bFT0[i] = beta*ene_TS[i] - ln(pre_TS[i]), i=1,2,...,N_omega0 - Shifted
        bFT2[i] = beta*ene_TS[i] - ln(pre_TS[i]), i=1,2,...,N_omega2 - Shited
        bFT1[i] = beta*eneT1[i] - len(preT1[i]) -> i = 1,2..,N_omega1 - Shifted
        bFT3[i] = beta*eneT3[i] - len(preT3[i]) -> i = 1,2..,N_omega3 - Shifted
        bFT4[i] = beta*eneT4[i] - len(preT4[i]) -> i = 1,2..,N_omega4 - Shifted
        # See the preene2betaene function to see what the shifts are.
        """
        if not len(bFSdb) == self.thermo.mixedstartindex:
            raise TypeError("Interaction energies must be present for all and only all thermodynamic shell states.")
        for en in bFSdb[self.thermo.mixedstartindex + 2:]:
            if not en == bFSdb[self.thermo.mixedstartindex + 1]:
                raise ValueError("States in kinetic shell have difference reference interaction energy")

        # 1. Get the minimum free energies of solutes, pure dumbbells and mixed dumbbells
        bFdb0_min = np.min(bFdb0)
        bFdb2_min = np.min(bFdb2)
        bFS_min = np.min(bFS)

        # 2. Make the unsymmetrized rates for calculating eta0
        # The energies of bare dumbbells, solutes and mixed dumbbells are not shifted with their minimum values
        # pass them in after shifting them.

        pre0, pre0T = np.ones_like(bFdb0), np.ones_like(bFT0)
        pre2, pre2T = np.ones_like(bFdb2), np.ones_like(bFT2)

        rate0list = self.ratelist(self.jnet0_indexed, pre0, bFdb0 - bFdb0_min, pre0T, bFT0,
                             self.vkinetic.starset.pdbcontainer.invmap)

        # rate2list = self.ratelist(self.jnet2_indexed, pre2, bFdb2 - bFdb2_min, pre2T, bFT2,
        #                      self.vkinetic.starset.mdbcontainer.invmap)

        # 3. Make the symmetrized rates and escape rates for calculating eta0, GF, bias and gamma.
        # 3a. First, make bFSdb_total from individual solute and pure dumbbell and the binding free energies,
        # i.e, bFdb0, bFS, bFSdb (binding), respectively.
        # For origin states, this should be in such a way so that omega_0 + del_omega = 0 -> this is taken care of in
        # getsymmrates function.
        # Also, we need to keep a shifted version, to calculate rates.

        bFSdb_total = np.zeros(self.vkinetic.starset.mixedstartindex)
        bFSdb_total_shift = np.zeros(self.vkinetic.starset.mixedstartindex)

        # first, just add up the solute and dumbbell energies.
        # Now adding changes to states to both within and outside the thermodynamics shell. This is because on
        # changing the energy reference, the "interaction energy" might not be zero in the kinetic shell.
        # The kinetic shell is defined as that outside which the omega1 rates are the same as the omega0 rates.
        # THAT is the definition that needs to be satisfied.
        for starind, star in enumerate(self.vkinetic.starset.stars[:self.vkinetic.starset.mixedstartindex]):
            # For origin complex states, do nothing - leave them as zero.
            if star[0].is_zero(self.vkinetic.starset.pdbcontainer):
                continue
            symindex = self.vkinetic.starset.star2symlist[starind]
            # First, get the unshifted value
            bFSdb_total[starind] = bFdb0[symindex] + bFS[self.invmap_solute[star[0].i_s]]
            bFSdb_total_shift[starind] = bFSdb_total[starind] - (bFdb0_min + bFS_min)

        # Now add in the changes for the complexes inside the thermodynamic shell.
        # Note that we are still not making any changes to the origin states.
        # We always keep them as zero.
        for starind, star in enumerate(self.thermo.stars[:self.thermo.mixedstartindex]):
            # Get the symorlist index for the representative state of the star
            if star[0].is_zero(self.thermo.pdbcontainer):
                continue
            # keep the total energies zero for origin states.
            kinStarind = self.thermo2kin[starind]  # Get the index of the thermo star in the kinetic starset
            bFSdb_total[kinStarind] += bFSdb[starind]  # add in the interaction energy to the appropriate index
            bFSdb_total_shift[kinStarind] += bFSdb[starind]

        # 3b. Get the rates and escapes
        # We incorporate a separate "shift" array so that even after shifting, the origin state energies remain
        # zero.
        betaFs = [bFdb0, bFdb2, bFS, bFSdb, bFSdb_total, bFSdb_total_shift, bFT0, bFT1, bFT2, bFT3, bFT4]
        (omega0, omega0escape), (omega1, omega1escape), (omega2, omega2escape), (omega3, omega3escape), \
        (omega4, omega4escape) = self.getsymmrates(bFdb0 - bFdb0_min, bFdb2 - bFdb2_min, bFSdb_total_shift, bFT0, bFT1,
                                                   bFT2, bFT3, bFT4)

        # 3b.1 - Put them in a tuple to use in makeGF later on - maybe simplify this process later on.
        omegas = ((omega0, omega0escape), (omega1, omega1escape), (omega2, omega2escape), (omega3, omega3escape),
                  (omega4, omega4escape))

        # 4. Update the bias expansions
        self.update_bias_expansions(rate0list, omega0escape) #, rate2list, omega2escape)

        # 5. Work out the probabilities and the normalization - will be needed to produce g2 from G2 (created in bias
        # updates)
        mixed_prob = np.zeros(len(self.vkinetic.starset.mixedstates))
        complex_prob = np.zeros(len(self.vkinetic.starset.complexStates))

        # 5a. get the complex boltzmann factors - unshifted
        # TODO Should we at least shift with respect to the minimum of the two (complex, mixed)
        # Otherwise, how do we think of preventing overflow in case it occurs?
        for starind, star in enumerate(self.vkinetic.starset.stars[:self.vkinetic.starset.mixedstartindex]):
            for state in star:
                if not (self.vkinetic.starset.complexIndexdict[state][1] == starind):
                    raise ValueError("check complexIndexdict")
                # For states outside the thermodynamics shell, there is no interaction and the probabilities are
                # just the product solute and dumbbell probabilities.
                complex_prob[self.vkinetic.starset.complexIndexdict[state][0]] = np.exp(-bFSdb_total[starind])

        # 5b. get the mixed dumbbell boltzmann factors.
        for siteind, wyckind in enumerate(self.vkinetic.starset.mdbcontainer.invmap):
            # don't need the site index but the wyckoff index corresponding to the site index.
            # The energies are not shifted with respect to the minimum
            mixed_prob[siteind] = np.exp(-bFdb2[wyckind])

        # 5c. Form the partition function
        # get the "reference energy" for non-interacting complexes. This is just the value of bFSdb (interaction)
        # for any state in the kinetic shell
        # del_en = bFSdb[self.thermo.mixedstartindex + 1]
        part_func = 0.
        # Now add in the non-interactive complex contribution to the partition function
        for dbsiteind, dbwyckind in enumerate(self.vkinetic.starset.pdbcontainer.invmap):
            for solsiteind, solwyckind in enumerate(self.invmap_solute):
                part_func += np.exp(-(bFdb0[dbwyckind] + bFS[solwyckind]))

        # 5d. Normalize - division by the partition function ensures effects of shifting go away.
        complex_prob *= 1. / part_func
        mixed_prob *= 1. / part_func

        # 6. Get the symmetrized Green's function in the basis of the vector stars and the non-local contribution
        # to solvent (Fe dumbbell) diffusivity.
        # arguments for makeGF - bFdb0 (shifted), bFT0(shifted), omegas, mixed_prob
        # Note about mixed prob: g2_ij = p_mixed(i)^0.5 * G2_ij * p_mixed(j)^-0.5
        # So, at the end of the end the day, it only depends on boltzmann factors of the mixed states.
        # All other factors cancel out (including partition function).
        GF_total, GF02, del_om = self.makeGF(bFdb0 - bFdb0_min, bFT0, omegas, mixed_prob)
        L0bb = self.GFcalc_pure.Diffusivity()
        # 7. Once the GF is built, make the correlated part of the transport coefficient
        # 7a. First we make the projection of the bias vector
        self.biases_solute_vs = np.zeros(self.vkinetic.Nvstars)
        self.biases_solvent_vs = np.zeros(self.vkinetic.Nvstars)

        Nvstars_pure = self.vkinetic.Nvstars_pure
        Nvstars = self.vkinetic.Nvstars

        # 7b. We need the square roots of the probabilities of the representative state of each vector star.
        prob_sqrt_complex_vs = np.array([np.sqrt(complex_prob[self.kinetic.complexIndexdict[vp[0]][0]])
                                         for vp in self.vkinetic.vecpos[:Nvstars_pure]])
        prob_sqrt_mixed_vs = np.array([np.sqrt(mixed_prob[self.kinetic.mixedindexdict[vp[0]][0]])
                                       for vp in self.vkinetic.vecpos[Nvstars_pure:]])


        # omega1 has total rates. So, to get the change in the rates, we must subtract out the corresponding
        # omega0 rates.
        # This gives us only the change in the rates within the kinetic shell due to solute interactions.
        # The effect of the non-local rates has been cancelled out by subtracting off the eta vectors.
        # For solvents out of complex states, both omega1 and omega4 jumps contribute to the local bias.

        self.del_W1 = np.zeros_like(omega1escape)
        for i in range(Nvstars_pure):
            for jt in range(len(self.jnet1)):
                self.del_W1[i, jt] = omega1escape[i, jt] - \
                                     omega0escape[
                                         self.kinetic.star2symlist[self.vkinetic.vstar2star[i]], self.om1types[jt]]

        self.biases_solvent_vs[:Nvstars_pure] = np.array([(np.dot(self.bias1_solvent_new[i, :], self.del_W1[i, :]) +
                                                           np.dot(self.bias4_solvent_new[i, :], omega4escape[i, :])) *
                                                          prob_sqrt_complex_vs[i] for i in range(Nvstars_pure)])

        self.biases_solvent_vs[Nvstars_pure:] = np.array([np.dot(self.bias3_solvent_new[i - Nvstars_pure, :],
                                                                 omega3escape[i - Nvstars_pure, :]) *
                                                          prob_sqrt_mixed_vs[i - Nvstars_pure]
                                                          for i in range(Nvstars_pure, self.vkinetic.Nvstars)])
        # In the mixed state space, the solvent bias comes due only to the omega3(dissociation) jumps.

        # if not eta2shift:
        # the bias2_new tensors won't be all zeros
        for i in range(Nvstars_pure, Nvstars):
            st0 = self.vkinetic.vecpos[i][0]
            dbwyck2 = self.mdbcontainer.invmap[st0.db.iorind]

            self.biases_solute_vs[i] += np.dot(self.bias2_solute_new[i - Nvstars_pure, :], omega2escape[dbwyck2, :]) * \
                                 prob_sqrt_mixed_vs[i - Nvstars_pure]

            self.biases_solvent_vs[i] += np.dot(self.bias2_solvent_new[i - Nvstars_pure, :], omega2escape[dbwyck2, :]) * \
                                  prob_sqrt_mixed_vs[i - Nvstars_pure]

        # Next, we create the gamma vector, projected onto the vector stars
        self.gamma_solute_vs = np.dot(GF_total, self.biases_solute_vs)
        self.gamma_solvent_vs = np.dot(GF_total, self.biases_solvent_vs)

        # Next we produce the outer product in the basis of the vector star vector state functions
        # a=solute, b=solvent
        L_c_aa = np.dot(np.dot(self.kinouter, self.gamma_solute_vs), self.biases_solute_vs)
        L_c_bb = np.dot(np.dot(self.kinouter, self.gamma_solvent_vs), self.biases_solvent_vs)
        L_c_ab = np.dot(np.dot(self.kinouter, self.gamma_solvent_vs), self.biases_solute_vs)

        # Next, we get to the bare or uncorrelated terms
        # First, we have to generate the probability arrays and multiply them with the ratelists. This will
        # Give the probability-square-root multiplied rates in the uncorrelated terms.
        # For the complex states, weed out the origin state probabilities
        for stateind, prob in enumerate(complex_prob):
            if self.vkinetic.starset.complexStates[stateind].is_zero(self.vkinetic.starset.pdbcontainer):
                complex_prob[stateind] = 0.

        pr_states = (complex_prob, mixed_prob)  # For testing
        # Next, we need the bare dumbbell probabilities for the non-local part of the solvent-solvent transport
        # coefficients
        bareprobs = self.stateprob(pre0, bFdb0 - bFdb0_min, self.pdbcontainer.invmap)
        # This ensured that summing over all complex + mixed states gives a probability of 1.
        # Note that this is why the bFdb0, bFS and bFdb2 values have to be entered unshifted.
        # The complex and mixed dumbbell energies need to be with respect to the same reference.

        # First, make the square root prob * rate lists to multiply with the rates
        # TODO Is there a way to combine all of the next four loops?

        prob_om0 = np.zeros(len(self.jnet0))
        for jt, ((IS, FS), dx) in enumerate([jlist[0] for jlist in self.jnet0_indexed]):
            prob_om0[jt] = np.sqrt(bareprobs[IS] * bareprobs[FS]) * omega0[jt]

        prob_om1 = np.zeros(len(self.jnet1))
        for jt, ((IS, FS), dx) in enumerate([jlist[0] for jlist in self.jnet1_indexed]):
            prob_om1[jt] = np.sqrt(complex_prob[IS] * complex_prob[FS]) * omega1[jt]

        prob_om2 = np.zeros(len(self.jnet2))
        for jt, ((IS, FS), dx) in enumerate([jlist[0] for jlist in self.jnet2_indexed]):
            prob_om2[jt] = np.sqrt(mixed_prob[IS] * mixed_prob[FS]) * omega2[jt]

        prob_om4 = np.zeros(len(self.jnet4))
        for jt, ((IS, FS), dx) in enumerate([jlist[0] for jlist in self.jnet4_indexed]):
            prob_om4[jt] = np.sqrt(complex_prob[IS] * mixed_prob[FS]) * omega4[jt]

        prob_om3 = np.zeros(len(self.jnet3))
        for jt, ((IS, FS), dx) in enumerate([jlist[0] for jlist in self.jnet3_indexed]):
            prob_om3[jt] = np.sqrt(mixed_prob[IS] * complex_prob[FS]) * omega3[jt]

        probs = (prob_om1, prob_om2, prob_om4, prob_om3)

        start = time.time()
        # Generate the bare expansions with modified displacements
        D0expansion_bb, (D1expansion_aa, D1expansion_bb, D1expansion_ab), \
        (D2expansion_aa, D2expansion_bb, D2expansion_ab), \
        (D3expansion_aa, D3expansion_bb, D3expansion_ab), \
        (D4expansion_aa, D4expansion_bb, D4expansion_ab) = self.bareExpansion(self.eta0total_solvent)

        L_uc_aa = np.dot(D1expansion_aa, prob_om1) + np.dot(D2expansion_aa, prob_om2) + \
                  np.dot(D3expansion_aa, prob_om3) + np.dot(D4expansion_aa, prob_om4)

        L_uc_bb = np.dot(D1expansion_bb, prob_om1) - np.dot(D0expansion_bb, prob_om0) + \
                  np.dot(D2expansion_bb, prob_om2) + np.dot(D3expansion_bb, prob_om3) + np.dot(D4expansion_bb, prob_om4)

        L_uc_ab = np.dot(D1expansion_ab, prob_om1) + np.dot(D2expansion_ab, prob_om2) + \
                  np.dot(D3expansion_ab, prob_om3) + np.dot(D4expansion_ab, prob_om4)

        # Make things that need to be tested as attributes
        self.GF_total = GF_total
        self.GF02 = GF02
        self.betaFs = betaFs
        self.del_om = del_om
        self.part_func = part_func
        self.probs = probs
        self.omegas = omegas
        self.pr_states = pr_states

        return L0bb, (L_uc_aa, L_c_aa), (L_uc_bb, L_c_bb), (L_uc_ab, L_c_ab)