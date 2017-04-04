Onsager
=======

Documentation now available at the [Onsager github page](http://dallastrinkle.github.io/Onsager/). Please cite as [![DOI](https://zenodo.org/badge/14172/DallasTrinkle/Onsager.svg)](https://zenodo.org/badge/latestdoi/14172/DallasTrinkle/Onsager) or see `Onsager github <https://github.com/DallasTrinkle/Onsager>`_ for current version doi information.

The Onsager package provides routines for the general calculation of transport coefficients in vacancy-mediated diffusion and interstitial diffusion. It does this using a Green function approach, combined with point group symmetry reduction for maximum efficiency.

Typical usage can be seen in the ipython notebooks in `examples`; the usual import will be::

    #!/usr/bin/env python

	from onsager import crystal
	from onsager import OnsagerCalc

    ...

Many of the subpackages within Onsager are support for the main attraction, which is in OnsagerCalc. Interstitial calculation examples are available in `bin`, including three YAML input files, as well as a interstitial diffuser. An example of vacancy-mediated diffusion is shown in `bin/fivefreq.py`, which computes the well-known five-frequency model for substitutional solute transport in an FCC lattice. The script `CLdiffuser` is a command-line diffuser calculator that is designed to read in an HDF5 file of a diffuser, along with a JSON file that includes the thermal/kinetic data, and computes diffusivity components for different temperatures.

The tests for the package are include in `test`; `tests.py` will run all of the tests in the directory with verbosity level 2. This can be time-consuming (on the order of several of minutes) to run all tests; coverage is currently >90%.

The code uses YAML format for input/output of crystal structures, and diffusion data for the interstitial calculator. The vacancy-mediated calculator requires much more data, and uses HDF5 format to save/reload as needed. The vacancy-mediated calculator uses tags (unique human-readable-ish strings) to identify all (symmetry-unique) vacancy, solute, and complex states, and transitions between them.
The vacancy-mediated diffuser can be stored as an HDF5 file (which internally stores the crystal structure in YAML format). The thermal/kinetic data is most easily serialized as JSON, but any dictionary-compatible format will do, by making use of tags.

Releases:

* 0.9. Full release of Interstitial calculator, along with theory paper (see References below).
* 0.9.1. Added spin degrees of freedom to `crystal` for symmetry purposes; added `supercell` class to aid in automated setup of calculation.
* 1.0 Now including automator for supercell calculations.
* 1.1 Automator update with Makefile; corrections for possible overflow error when omega2 gets very large.
* 1.2 Combined "large omega2" and "non-zero bias" algorithms; notebook for Fe-C added to documentation; cleanup of code and improved testing.
* 1.2.1 Additional notebooks added for vacancy-mediated diffuser.
* 1.2.2 New internal friction calculator for interstitial diffuser; improvement in Crystal class symmetry to handle larger error in unit cell.

References
==========
* Dallas R. Trinkle, "Diffusivity and derivatives for interstitial solutes: Activation energy, volume, and elastodiffusion tensors." Philos. Mag. (2016) [doi:10.1080/14786435.2016.1212175](http://dx.doi.org/10.1080/14786435.2016.1212175); [arXiv:1605.03623](http://arxiv.org/abs/1605.03623)
* Dallas R. Trinkle, "Automatic numerical evaluation of vacancy-mediated transport for arbitrary crystals: Onsager coefficients in the dilute limit using a Green function approach." [arXiv:1608.01252](http://arxiv.org/abs/1608.01252)

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
