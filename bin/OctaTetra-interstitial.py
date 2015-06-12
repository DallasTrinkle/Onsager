#!/usr/bin/env python
"""
Example script using OnsagerCalc to compute the elasto-diffusion tensor for
an interstitial. It also shows how to both construct an input YAML file for
an FCC, BCC, or HCP crystal with octahedral / tetrahedral network, and
read from that same YAML file as input. It's actually *extremely general*--if
you pass it the appropriate YAML input, it will run it *regardless of whether
it corresponds to a YAML file it generated*.
"""

__author__ = 'Dallas R. Trinkle'

import numpy as np
from onsager import crystal
from onsager import OnsagerCalc

HeaderString = """# Input format for a crystal, followed by sitelist and jumpnetwork.
# Notes:
# 1. !numpy.ndarray tag is used to specifically identify numpy arrays;
#    should be used for both the lattice and basis entries
# 2. lattice is in a more "readable" format using *row* vectors; the
#    actual Crystal object stores the lattice with *column* vectors,
#    so after import, this matrix will be transposed.
# 3. lattice_constant is optional; it is used to scale lattice on input.
# 4. the basis is a list of lists; the lists are broken up in terms
#    of chemistry (see the chemistry list)
# 5. chemistry is a list of names of the unique species in the crystal;
#    it is entirely optional, and not used to construct the crystal object
# 6. the sitelist and jumpnetwork have entries for energies, elastic dipoles
#    and prefactors; each are for the *first element in the lists* as a
#    representative.
# 7. the tag interstitial defines which site is the interstitial element.
interstitial: 1
"""

def FCCoutputYAML(a0):
    """
    Generates YAML file corresponding to our FCC lattice with octahedral and tetrahedrals.
    :param a0: lattice constant
    :return: YAML string containing the *short* version, along with jump networks
    """
    FCC = crystal.Crystal.FCC(a0)
    FCCinter = FCC.addbasis(FCC.Wyckoffpos(np.array([0.5,0.5,0.5])) +
                            FCC.Wyckoffpos(np.array([0.25,0.25,0.25])))
    # this cutoffs is for: o->t
    cutoff = 0.5*a0
    return HeaderString + \
           FCCinter.simpleYAML(a0) + \
           OnsagerCalc.Interstitial.sitelistYAML(FCCinter.sitelist(1)) + \
           OnsagerCalc.Interstitial.jumpnetworkYAML(FCCinter.jumpnetwork(1, cutoff))

def BCCoutputYAML(a0):
    """
    Generates YAML file corresponding to our BCC lattice with octahedral and tetrahedrals.
    :param a0: lattice constant
    :return: YAML string containing the *short* version, along with jump networks
    """
    BCC = crystal.Crystal.BCC(a0)
    BCCinter = BCC.addbasis(BCC.Wyckoffpos(np.array([0.5,0.5,0.])) +
                            BCC.Wyckoffpos(np.array([0.25,0.5,0.75])))
    # this cutoffs is for: o->t and t-> networks
    cutoff = 0.4*a0
    return HeaderString + \
           BCCinter.simpleYAML(a0) + \
           OnsagerCalc.Interstitial.sitelistYAML(BCCinter.sitelist(1)) + \
           OnsagerCalc.Interstitial.jumpnetworkYAML(BCCinter.jumpnetwork(1, cutoff))

def HCPoutputYAML(a0, c_a, z=1./8.):
    """
    Generates YAML file corresponding to our HCP lattice with octahedral and tetrahedrals.
    :param a0: lattice constant
    :param c_a: c/a ratio
    :param z: distance of tetrahedral from basal plane (unit cell coordinates)
    :return: YAML string containing the *short* version, along with jump networks
    """
    # Note: alternatively, we could construct our HCP crystal *then* generate Wyckoff positions
    hexlatt = a0*np.array([[0.5, 0.5, 0],
                           [-np.sqrt(0.75), np.sqrt(0.75), 0],
                           [0, 0, c_a]])
    hcpbasis = [[np.array([1./3.,2./3.,1./4.]),np.array([2./3.,1./3.,3./4.])],
                [np.array([0.,0.,0.]), np.array([0.,0.,0.5]),
                 np.array([1./3.,2./3.,3./4.-z]), np.array([1./3.,2./3.,3./4.+z]),
                 np.array([2./3.,1./3.,1./4.-z]), np.array([2./3.,1./3.,1./4.+z])]]
    HCP = crystal.Crystal(hexlatt, hcpbasis)
    # these cutoffs are for: o->t, t->t, o->o along c
    # (preferably all below a). The t->t should be shortest, and o->o the longest
    cutoff = 1.01*a0*max(np.sqrt(1./3.+c_a**2/64), 2*z*c_a, 0.5*c_a)
    if __debug__:
        if cutoff > a0: raise AssertionError('Geometry such that we will include basal jumps')
        if np.abs(z) > 0.25: raise AssertionError('Tetrahedral parameter out of range (>1/4)')
        if np.abs(z) < 1e-2: raise AssertionError('Tetrahedral parameter out of range (approx. 0)')
    return HeaderString + \
           HCP.simpleYAML(a0) + \
           OnsagerCalc.Interstitial.sitelistYAML(HCP.sitelist(1)) + \
           OnsagerCalc.Interstitial.jumpnetworkYAML(HCP.jumpnetwork(1, cutoff))


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(
        description='Compute elastodiffusion tensor for interstitials; temperatures (K) read from stdin',
        epilog='output: T Dxx Dzz Exx Ezz d11 d33 d12 d13 d31 d44 d66')
    parser.add_argument('--yaml', '-y', choices=['FCC', 'BCC', 'HCP'],
                        help='Output YAML file corresponding to an o/t network')
    parser.add_argument('-a', type=float, default=1.0,
                        help='basal lattice constant')
    parser.add_argument('-c', type=float, default=np.sqrt(8./3.),
                        help='c/a ratio')
    parser.add_argument('-z', type=float, default=1./8.,
                        help='z parameter specifying distance from basal plane for tetrahedrals')
    parser.add_argument('yaml_input',
                        help='YAML formatted file containing all information about interstitial crystal/lattice')
    parser.add_argument('--eV', action='store_true',
                        help='Assume that T is input as kB T, in (same units as energies/dipoles)')
    # we use parse_known_args so that "extra" is our additional arguments, which can be files of T to read in
    args, extra = parser.parse_known_args()

    if args.yaml:
        # generate YAML input
        if args.yaml == 'FCC':
            print(FCCoutputYAML(args.a))
        elif args.yaml == 'BCC':
            print(BCCoutputYAML(args.a))
        elif args.yaml == 'HCP':
            print(HCPoutputYAML(args.a, args.c, args.z))
        else:
            parser.print_help()
    else:
        # otherwise... we need to parse our YAML file, and get to work
        with open(args.yaml_input, "r") as in_f:
            dict_def = crystal.yaml.load(in_f)
            crys = crystal.Crystal.fromdict(dict_def) # pull out the crystal part of the YAML
            # sites:
            sitelist = dict_def['sitelist']
            pre = dict_def['Prefactor']
            ene = dict_def['Energy']
            dipole = dict_def['Dipole']
            # jumps
            jumpnetwork = dict_def['jumpnetwork']
            preT = dict_def['PrefactorT']
            eneT = dict_def['EnergyT']
            dipoleT = dict_def['DipoleT']
            # we don't do any checking here... just dive on in
            chem = dict_def['interstitial']
            # create our calculator
            interstitial = OnsagerCalc.Interstitial(crys, chem, sitelist, jumpnetwork)
            if args.eV:
                kB = 1.
            else:
                from scipy.constants import physical_constants
                kB = physical_constants['Boltzmann constant in eV/K'][0]
            # now read through stdin, taking each entry as a temperature
            import fileinput
            print("#T #Dxx #Dzz #Exx #Ezz #d11 #d33 #d12 #d13 #d31 #d44 #d66")
            for line in fileinput.input(extra):
                T = float(line)
                beta = 1./(kB*T)
                BE = [ beta*E for E in ene ]
                beta_dip = [ beta*dip for dip in dipole ]
                BET = [ beta*ET for ET in eneT ]
                beta_dipT = [ beta*dipT for dipT in dipoleT ]
                D0, DB = interstitial.diffusivity(pre, BE, preT, BET, True) # calculate deriv. wrt beta
                D0, Dp = interstitial.elastodiffusion(pre, BE, beta_dip, preT, BET, beta_dipT)
                Eact = np.linalg.solve(beta*D0, DB)
                print("{} {} {} {} {} {} {} {} {} {} {} {}".format(T, D0[0,0], D0[2,2],
                                                                   Eact[0,0], Eact[2,2],
                                                                   Dp[0,0,0,0],
                                                                   Dp[2,2,2,2],
                                                                   Dp[0,0,1,1],
                                                                   Dp[0,0,2,2],
                                                                   Dp[2,2,0,0],
                                                                   Dp[1,2,1,2],
                                                                   Dp[0,1,0,1]))
