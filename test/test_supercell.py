"""
Unit tests for supercell class
"""

__author__ = 'Dallas R. Trinkle'

import unittest
import numpy as np
import onsager.crystal as crystal
import onsager.supercell as super

class TypeTests(unittest.TestCase):
    """Tests to make sure we can make a supercell object."""
