"""
Automator code

Functions to convert from a supercell dictionary (output from a Diffuser) into a tarball
that contains all of the input files in an organized directory structure to run the
atomic-scale transition state calculations. This includes:

1. All positions in POSCAR format (POSCAR files for states to relax, POS as reference
  for transition endpoints that need to be relaxed)
2. Transformation information from relaxed states to initial states.
3. INCAR files for relaxation and NEB runs; KPOINTS for each.
4. perl script to transform CONCAR output from a state relaxation to NEB endpoints.
5. perl script to linearly interpolate between NEB endpoints.
6. Makefile to run everything (representing the "directed graph" of calculations)
"""

__author__ = 'Dallas R. Trinkle'

import numpy as np
import collections, copy, itertools, warnings
from onsager import crystal, supercell

