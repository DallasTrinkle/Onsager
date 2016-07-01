#!/usr/bin/env python

import sys
sys.path.append('./')  # if we want to run from the bin directory
sys.path.append('../')  # if we want to run from the bin directory
import onsager.OnsagerCalc as onsager
import h5py, json

# Tags we can use to identify components; first part specifies which Onsager matrix element while
# last part specifies Cartesian components
__diffusion_types__ = {'vv': 'L0vv', 'vv1': 'L1vv', 'ss': 'Lss', 'sv': 'Lsv', 'vs': 'Lsv'}
__cartesian_components__ = {'x': 0, 'y': 1, 'z': 2}
# string maps to (localname, (i,j)) for Cartesian components i,j
__fullcomponents__ = {t + c1 + c2: (loc, (i1, i2))
                      for t, loc in __diffusion_types__.items()
                      for c1, i1 in __cartesian_components__.items()
                      for c2, i2 in __cartesian_components__.items()}


def cleanup(strlist):
    # translate: first list into second, and delete the third
    transtable = str.maketrans('SVXYZ', 'svxyz', 'DdLl-_ \t')
    return [s.translate(transtable) for s in strlist]


def OnsagerComponents(diff, preene, kT, components):
    """
    Takes in a diffuser (diff) and thermodynamic dictionary (tdict) and temperature (kT) and returns
    particular components.

    :param diff: vacancy-mediated diffuser
    :param preene: thermodynamic prefactors + energies
    :param kT: temperature
    :param components: list of particular components to return (should be a key in __fullcomponents__)
    :return Onsagerterms: tuple of all the corresponding Onsager coefficients
    """
    L0vv, Lss, Lsv, L1vv = diff.Lij(*diffuser.preene2betafree(kT, **preene))
    loc = locals()  # get a dictionary of the local variables
    # fun bit of python: the interior list is the corresponding tuples of name of component and indices
    # then we loop over those tuples, pulling out the corresponding pieces.
    return tuple(loc[locname][ij] for locname, ij in [__fullcomponents__[c] for c in components])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Compute diffusivity using HDF5 diffuser and a JSON file of thermodynamic data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""output: T Labij ...
Each line of stdin / additional file must be in the form:
T abij abij ...

Where abij = Onsagertype + Cartesian components, such as ssxx, or vv1xy.
Onsagertype =
  vv (bare vacancy, must be multiplied by cv/kBT)
  ss (solute-solute, must be multiplied by cs*cv/kBT)
  sv (solute-vacancy, must be multiplied by cs*cv/kBT)
  vv1 (vacancy-vacancy correction, must be multiplied by cs*cv/kBT)
Cartesian components = xx, yy, zz, xy, yx, xz, zx, yz, zy
""")
    parser.add_argument('HDF5_input', help='HDF5 diffuser')
    parser.add_argument('JSON_input', help='JSON dictionary of thermodynamic data')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Do a verbose dump on diffuser and exit')
    parser.add_argument('--limb', '-l', action='store_true',
                        help='Use omega0, solute-vacancy energies with LIMB approx.')
    parser.add_argument('--eV', action='store_true',
                        help='Assume that T is input as kB T, in same units as energies')
    # we use parse_known_args so that "extra" is our additional arguments, which can be files of T to read in
    args, extra = parser.parse_known_args()

    with h5py.File(args.HDF5_input, 'r') as f:
        diffuser = onsager.VacancyMediated.loadhdf5(f)

    if args.verbose:
        print(diffuser)
        exit()

    with open(args.JSON_input, 'r') as f:
        thermodict = json.load(f)

    preene = diffuser.tags2preene(thermodict)
    if args.limb:
        preene.update(diffuser.makeLIMBpreene(**preene))

    if args.eV:
        kB = 1.
    else:
        from scipy.constants import physical_constants

        kB = physical_constants['Boltzmann constant in eV/K'][0]

    import fileinput

    # print("#T #Lss_xx #Lss_zz #Lsv_xx #Lsv_zz")
    for line in fileinput.input(extra):
        components = line.split()
        T = float(components.pop(0))  # get the first entry, and also remove it...
        Lcomponents = OnsagerComponents(diffuser, preene, kB * T, cleanup(components))
        print("{} ".format(T) + ' '.join(['{:.12g}'.format(c) for c in Lcomponents]))
