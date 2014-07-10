=======
Onsager
=======

The Onsager package provides routines for the general calculation of transport
coefficients in vacancy-mediated diffusion. It does this using a Green function
approach, combined with point group symmetry reduction for maximum efficiency.

Typical usage looks like::

    #!/usr/bin/env python

    from onsager import all

    ...

Many of the subpackages within Onsager are support for the main attraction, which
is in OnsagerCalc.