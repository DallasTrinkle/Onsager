Onsager
=======

Documentation now available at the [Onsager github page](http://dallastrinkle.github.io/Onsager/). Please cite as [![DOI](https://zenodo.org/badge/14172/DallasTrinkle/Onsager.svg)](https://zenodo.org/badge/latestdoi/14172/DallasTrinkle/Onsager)

The Onsager package provides routines for the general calculation of transport coefficients in vacancy-mediated diffusion and interstitial diffusion. It does this using a Green function approach, combined with point group symmetry reduction for maximum efficiency.

Typical usage looks like::

    #!/usr/bin/env python

	from onsager import crystal
	from onsager import OnsagerCalc

    ...

Many of the subpackages within Onsager are support for the main attraction, which is in OnsagerCalc. Interstitial calculation examples are avaliable in `bin`, including three YAML input files, as well as a interstitial diffuser. An example of vacancy-mediated diffusion is shown in `bin/fivefreq.py`, which computes the well-known five-frequency model for substitutional solute transport in an FCC lattice.

The tests for the package are include in `test`; `tests.py` will run all of the tests in the directory with verbosity level 2. This can be time-consuming (on the order of several of minutes) to run all tests; coverage is currently >90%.

The code uses YAML files for input/output of diffusion data for the interstitial calculator. The vacancy-mediated calculator requires much more data, and uses HDF5 format to save/reload as needed. The vacancy-mediated calculator uses tags (unique human-readable-ish strings) to identify all (symmetry-unique) vacancy, solute, and complex states, and transitions between them.

Releases:

* 0.9. Full release of Interstitial calculator, along with theory paper (see References below).
* 0.9.1. Added spin degrees of freedom to `crystal` for symmetry purposes; added `supercell` class to aid in automated setup of calculation.
* 1.0 Now including automator for supercell calculations.

References
==========
* Dallas R. Trinkle, "Diffusivity and derivatives for interstitial solutes: Activation energy, volume, and elastodiffusion tensors." [arXiv:1605.03623](http://arxiv.org/abs/1605.03623)

Contributors
============
* Dallas R. Trinkle, initial design, derivation, and implementation.
* Ravi Agarwal, testing of HCP interstitial calculations; testing of HCP vacancy-mediated diffusion calculations
* Abhinav Jain, testing of HCP vacancy-mediated diffusion calculations.

Thanks to discussions with Maylise Nastar (CEA, Saclay), Thomas Garnier (CEA, Saclay and UIUC), Thomas Schuler (CEA, Saclay), and Pascal Bellon (UIUC).

Support
=======
This work has been supported in part by

* DOE/BES grant DE-FG02-05ER46217,
* ONR grant N000141210752,
* NSF/CDSE grant 1411106.
* Dallas R. Trinkle began the theoretical work for this code during the long program on Material Defects at the [Institute for Pure and Applied Mathematics](https://www.ipam.ucla.edu/) at UCLA, Fall 2012, which is supported by the National Science Foundation.
