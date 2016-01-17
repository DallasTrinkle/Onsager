=======
Onsager
=======

The Onsager package provides routines for the general calculation of transport coefficients in vacancy-mediated diffusion and interstitial diffusion. It does this using a Green function approach, combined with point group symmetry reduction for maximum efficiency.

Typical usage looks like::

    #!/usr/bin/env python

	from onsager import crystal
	from onsager import OnsagerCalc

    ...

Many of the subpackages within Onsager are support for the main attraction, which is in OnsagerCalc. An example of how to use the OnsagerCalc pieces is shown in fivefreq.py, which computes the well-known five-frequency model for vacancy- mediated solute transport in an FCC lattice. It can very easily be modified to use a different lattice / NNvect set.

The tests for the package are include in test; tests.py will run all of the tests in the directory with verbosity level 2. This can be time-consuming (on the order of tens of minutes) to run all tests.

The newest update (0.2) includes an improved "crystal" class to handle a general crystal type, with full space group symmetry, and associated analysis; and an Interstitial class within OnsagerCalc that can compute diffusion and the elastodiffusion (derivative of diffusion with respect to strain) tensors. These also include YAML support for input as well as output of classes.

Update 0.2.1: corrected a sign error in the definition of elastic dipole.

Update 0.3: an entirely new implementation of the Onsager calculation is now included. This is based on the crystal class; two examples are shown in the examples directory as iPython notebooks. This allows for the use of crystals that are not simple Bravais lattices for vacancy-mediated diffusion. YAML support is also included to output a diffuser (and read back in).

Update 0.4: cleanup of code, removing old implementations and tests; updated for speed, and inclusion of HDF5 format for reading and writing of VacancyMediated objects.

Contributors
============
Dallas R. Trinkle, initial design and implementation.

Thanks to discussions with Maylise Nastar (CEA, Saclay), Thomas Garnier (CEA, Saclay and UIUC), Thomas Schuler (CEA, Saclay), and Pascal Bellon (UIUC).

Support
=======
This work has been supported in part by DOE/BES program XYZ and NSF/CDSE program XYZ.
