#!/usr/bin/env python
"""
Example script using OnsagerCalc to compute the Onsager coefficients for
the five-frequency model. This script is actually *very general*; it only
requires a lattice with jump vectors that are all identical by symmetry.
"""

__author__ = 'Dallas R. Trinkle'

import numpy as np
# from onsager import FCClatt
# from onsager import GFcalc
from onsager import OnsagerCalc
from onsager import crystal

class FiveFreqFreqCalc:
    """Class that does the five-frequency calculation for us."""
    def __init__(self):
        # GF calculator (using scaled rates)
        fcc = crystal.Crystal.FCC(1.)
        self.diffuser = OnsagerCalc.VacancyMediated(fcc, 0, fcc.sitelist(0), fcc.jumpnetwork(0, 0.8), 1)

    def Lij(self, w0, w1, w2, w3, w4):
        """
        Calculates the Onsager coefficients for vacancy-mediated solute diffusion in the
        five-frequency model.

        :param w0 : float > 0, rate for a vacancy jump absent a solute
        :param w1 : float > 0  rate for a vacancy jump from 1st nn site to 1st nn site
        :param w2 : float > 0  rate for vacancy / solute exchange
        :param w3 : float > 0  rate for vacancy escape from 1st nn shell
        :param w4 : float > 0  return rate for w3 (formation of a complex)

        :return Lvv : array [3, 3]  vacancy/vacancy transport, to be multiplied cv/kB T, in cs=0 limit
        :return Lss : array [3, 3]  solute/solute transport, to be multiplied cv cs/kB T
        :return Lsv : array [3, 3]  solute/vacancy transport, to be multiplied cv cs/kB T
        :return L1vv : array [3, 3]  vacancy/vacancy transport correction, to be multiplied cv cs/kB T
        """
        for om in (w0, w1, w2, w3, w4):
            if om <= 0:
                raise ArithmeticError('All frequencies need to be >= 0; received {}'.format(om))

        SVprob = w4/w3  # enhanced probability of solute-vacancy complex
        thermaldef = {'preV': np.array([1.]), 'eneV': np.array([0.]),
                      'preS': np.array([1.]), 'eneS': np.array([0.]),
                      'preT0': np.array([w0]), 'eneT0': np.array([0.]),
                      'preSV': np.array([SVprob]), 'eneSV': np.array([0.])}
        thermaldef.update(self.diffuser.makeLIMBpreene(**thermaldef))
        # now, we need to get w1, w3, and w4 in there. w3 = dissociation, w4 = association, so:
        # the transition state for the association/dissociation jump is w4 as the outer prob = 1,
        # and the bound probability = w4/w3. The transition state for the "swing" jumps is
        # w1*(w4/w3), where the w4/w3 takes care of the probability factor. Finally, the
        # exchange jump is also w2*(w4/w3).
        thermaldef['preT2'][0] = w2*SVprob
        for j, (PS1, PS2) in enumerate(self.diffuser.omegalist(1)[0]):
            # check to see if the two endpoints of the transition have the solute-vacancy at same distance:
            if np.isclose(np.dot(PS1.dx, PS1.dx), np.dot(PS2.dx, PS2.dx)):
                thermaldef['preT1'][j] = w1*SVprob
            else:
                thermaldef['preT1'][j] = w4
        return self.diffuser.Lij(*self.diffuser.preene2betafree(1., **thermaldef))

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
    Lijcalc = FiveFreqFreqCalc()
    print('#w0 #w1 #w2 #w3 #w4 #Lvv #Lss #Lsv #L1vv')
    for w in w_array:
        L0vv, Lss, Lsv, L1vv = Lijcalc.Lij(*w) # w[0], w[1], w[2], w[3], w[4]
        print(w[0], w[1], w[2], w[3], w[4], L0vv[0, 0], Lss[0, 0], Lsv[0, 0], L1vv[0, 0])
