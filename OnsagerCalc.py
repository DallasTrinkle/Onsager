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

        Nthermo : integer
            range of thermodynamic interactions, in terms of "shells", which is multiple
            summations of jumpvect
        """
        # make copies, as lists (there is likely a more efficient way to do this
        self.jumpvect = [v for v in jumpvect]
        self.groupops = [g for g in groupops]
        self.NNstar = stars.StarSet(self.jumpvect, self.groupops, 1)
        self.generate(Nthermo)

    def generate(self, Nthermo):
        """
        Generate the necessary stars, double-stars, vector-stars, etc. based on the
        thermodynamic range.

        Parameters
        ----------
        Nthermo : integer
            range of thermodynamic interactions, in terms of "shells", which is multiple
            summations of jumpvect
        """
        if Nthermo == 0:
            self.Nthermo = 0
            return


