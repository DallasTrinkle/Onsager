=======
Onsager
=======

The Onsager package provides routines for the general calculation of transport
coefficients in vacancy-mediated diffusion. It does this using a Green function
approach, combined with point group symmetry reduction for maximum efficiency.

Typical usage looks like::

    #!/usr/bin/env python

    from onsager import GFcalc
    from onsager import OnsagerCalc

    ...

Many of the subpackages within Onsager are support for the main attraction, which
is in OnsagerCalc. An example of how to use the OnsagerCalc pieces is shown in
fivefreq.py, which computes the well-known five-frequency model for vacancy-
mediated solute transport in an FCC lattice. It can very easily be modified
to use a different lattice / NNvect set.

The tests for the package are include in test; tests.py will run all of the tests
in the directory with verbosity level 2. This can be time-consuming (on the order
of tens of minutes) to run all tests.

Contributors
============
Dallas R. Trinkle, initial design.

Thanks to discussions with Maylise Nastar (CEA, Saclay), Thomas Garnier
(CEA, Saclay and UIUC), Thomas Schuler (CEA, Saclay), and Pascal Bellon (UIUC).

Support
=======
This work has been supported in part by DOE/BES program XYZ and NSF/CDSE program XYZ.
