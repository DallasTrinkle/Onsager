#!/usr/bin/env python
"""
Unit tests for vacancy mediated diffusion code.
"""

import unittest
import test_FCClatt
import test_KPTmesh
import test_GFcalc


# DocTests... we use this for the small "utility" functions, rather than writing
# explicit tests; doctests are compatible with unittests, so we're good here.
import doctest
import GFcalc

def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(GFcalc))
    return tests

def main():
    unittest.defaultTestLoader.discover('.')
    unittest.main()


if __name__ == '__main__':
    main()
