"""
Unit tests for star, double-star and vector-star generation and indexing,
rebuilt to use crystal
"""

__author__ = 'Dallas R. Trinkle'

#

import unittest
import onsager.crystal as crystal
import numpy as np
import onsager.cluster as cluster


class ClusterSiteTests(unittest.TestCase):
    """Tests of the ClusterSite class"""
    longMessage = False

    def testClusterSiteType(self):
        """Can we make cluster sites?"""
        site = cluster.ClusterSite((0,0), np.array([0,0,0]))
        self.assertIsInstance(site, cluster.ClusterSite)



if __name__ == '__main__':
    unittest.main()
