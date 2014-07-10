#!/usr/bin/env python
"""
Example script using OnsagerCalc to compute the Onsager coefficients for
the five-frequency model. This script is actually *very general*; it only
requires a lattice with jump vectors that are all identical by symmetry.
"""

__author__ = 'Dallas R. Trinkle'

import numpy as np
from onsager import FCClatt
from onsager import GFcalc
from onsager import OnsagerCalc

class FiveFreqFreqCalc:
    """Class that does the five-frequency calculation for us."""
    def __init__(self, lattice, NNvect):
        # GF calculator (using scaled rates)
        self.GF = GFcalc.GFcalc(lattice, NNvect, np.ones(len(NNvect)))
        # Onsager calculator
        self.Lcalc = OnsagerCalc.VacancyMediated(NNvect, self.GF.groupops)
        self.Lcalc.generate(1) # everything we need, assuming that the thermodynamic range = 1
        # get our (scaled) Green function entries:
        self.gf = np.array([self.GF.GF(R) for R in self.Lcalc.GFlist()])
        # prepare our omega1 matrix
        om1list, om1index = self.Lcalc.omega1list()
        # this array is such that np.dot(om1_w0w1w34, np.array([w0, w1, w3w4])) = om1
        self.om1_w0w1w34 = np.zeros((len(om1list), 3))
        for i, pair in enumerate(om1list):
            p0nn = any([all(abs(pair[0] - x) < 1e-8) for x in NNvect])
            p1nn = any([all(abs(pair[1] - x) < 1e-8) for x in NNvect])
            if p0nn and p1nn:
                self.om1_w0w1w34[i, 1] = 1
                continue
            if p0nn or p1nn:
                self.om1_w0w1w34[i, 2] = 1
                continue
            # default
            self.om1_w0w1w34[i, 0] = 1
        # use maketracer to construct our probability list (throw away om2 and om1)
        self.prob, om2, om1 = self.Lcalc.maketracer()

    def Lij(self, w0, w1, w2, w3, w4):
        """
        Calculates the Onsager coefficients for vacancy-mediated solute diffusion in the
        five-frequency model.

        Parameters
        ----------
        w0 : float > 0
            rate for a vacancy jump absent a solute
        w1 : float > 0
            rate for a vacancy jump from 1st nn site to 1st nn site
        w2 : float > 0
            rate for vacancy / solute exchange
        w3 : float > 0
            rate for vacancy escape from 1st nn shell
        w4 : float > 0
            return rate for w3 (formation of a complex)

        Returns
        -------
        4 second-rank tensors
        Lvv : array [3, 3]
            vacancy/vacancy transport, to be multiplied cv/kB T, in cs=0 limit
        Lss : array [3, 3]
            solute/solute transport, to be multiplied cv cs/kB T
        Lsv : array [3, 3]
            solute/vacancy transport, to be multiplied cv cs/kB T
        L1vv : array [3, 3]
            vacancy/vacancy transport correction, to be multiplied cv cs/kB T
        """
        for om in (w0, w1, w2, w3, w4):
            if om <= 0:
                raise ArithmeticError('All frequencies need to be >= 0; received {}'.format(om))
        om0 = np.array([w0])
        om2 = np.array([w2])
        om1 = np.dot(self.om1_w0w1w34, np.array([w0, w1, np.sqrt(w3*w4)]))
        self.prob[0] = w4/w3
        # note: we need to divide self.gf by om0, as it was constructed with w0 = 1, so we scale here:
        return self.Lcalc.Lij(self.gf/om0, om0, self.prob, om2, om1)

if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(
        description='Compute transport coefficients for the five frequency model in FCC',
        epilog='output as L0vv Lss Lsv L1vv ; to be scaled by conc./kBT as needed')
    parser.add_argument('freq', type=np.loadtxt,
                        help='file of 5 frequencies: w0 w1 w2 w3 w4')
    args = parser.parse_args()

    w_array = np.array(args.freq)
    if np.shape(w_array)[1] < 5 :
        raise ArithmeticError('Need at least five frequencies')
    Lijcalc = FiveFreqFreqCalc(FCClatt.lattice(), FCClatt.NNvect())
    print '#w0 #w1 #w2 #w3 #w4 #Lvv #Lss #Lsv #L1vv'
    for w in w_array:
        L0vv, Lss, Lsv, L1vv = Lijcalc.Lij(w[0], w[1], w[2], w[3], w[4])
        print w[0], w[1], w[2], w[3], w[4], L0vv[0, 0], Lss[0, 0], Lsv[0, 0], L1vv[0, 0]
