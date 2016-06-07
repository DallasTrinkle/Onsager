"""
Supercell class

Class to store supercells of crystals: along with some analysis
1. add/remove/substitute atoms
2. output POSCAR format (possibly other formats?)
3. find the transformation map between two different representations of the same supercell
4. construct an NEB pathway between two supercells
5. possibly input from CONTCAR? extract displacements?
"""

__author__ = 'Dallas R. Trinkle'

import numpy as np
import collections, copy
from . import crystal
from functools import reduce

# YAML tags:
# interfaces are either at the bottom, or staticmethods in the corresponding object
# NDARRAY_YAMLTAG = '!numpy.ndarray'
# GROUPOP_YAMLTAG = '!GroupOp'

class Supercell(object):
    """
    A class that defines a Supercell of a crystal
    """
    def __init__(self, crys, super, interstitial=(), Nchem=-1, empty=False):
        """
        Initialize our supercell

        :param crys: crystal object
        :param super: 3x3 integer matrix
        :param interstitial: (optional) list/tuple of indices that correspond to interstitial sites
        :param Nchem: (optional) number of distinct chemical elements to consider; default = crys.Nchem+1
        :param empty: optional; designed to allow "copy" to work
        """
        if empty: return
        self.crys = crys
        self.super = super.copy()
        self.interstitial = copy.deepcopy(interstitial)
        self.Nchem = crys.Nchem+1 if Nchem<crys.Nchem else Nchem

    def copy(self):
        """
        Make a copy of the supercell
        :return: new supercell object, copy of the original
        """
        supercopy = Supercell(self.crys, self.super, self.interstitial, self.Nchem)
        return supercopy

    def __eq__(self, other):
        """
        Return True if two supercells are equal; this means they should have the same occupancy
        *and* the same ordering
        :param other: supercell for comparison
        :return: True if same crystal, supercell, occupancy, and ordering; False otherwise
        """
        return isinstance(other, self.__class__) and \
               False

    def __ne__(self, other):
        """Inequality == not __eq__"""
        return not self.__eq__(other)

    def __str__(self):
        """Human readable version of supercell"""
        str = "Supercell of crystal:\n{crys}\n".format(crys=self.crys)
        if self.interstitial != (): str = str + "Interstitial sites: {}\n".format(self.interstitial)
        str = str + "Supercell vectors:\n{}".format(self.super.T)
        return str

