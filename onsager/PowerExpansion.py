"""
Power expansion class

Class to store and manipulate 3-dimensional Taylor (power) expansions of functions
Particularly useful for inverting the FT of the evolution matrix, and subtracting off
analytically calculated IFT for the Green function.

Really designed to get used by other code.
"""

__author__ = 'Dallas R. Trinkle'

import numpy as np
from scipy.special import factorial

