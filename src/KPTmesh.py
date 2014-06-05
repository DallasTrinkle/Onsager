"""
KPTmesh module

Class definition for KPTmesh class; allows automated construction of kpt meshes
"""

import numpy as np

class KPTmesh:
    """
    A class to construct (symmetrized, reduced to the irreducible wedge) k-point meshes.
    """
    def __init__(self, lattice):
        """
        Creates an instance of a k-point mesh generator
        """
        self.lattice = lattice
 
