#!/usr/bin/env python
"""
Unit tests for vacancy mediated diffusion code; designed to run discovery in directory.
Automatically runs at verbosity level 2; doesn't read from the command line.
"""

import unittest
import os


def main():
    suite = unittest.defaultTestLoader.discover(start_dir=os.path.dirname(__file__))
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    main()
