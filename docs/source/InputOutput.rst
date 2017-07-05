Input and output for Onsager transport calculation
====================================

The Onsager calculators currently include two computational approaches to determining transport coefficients: an "interstitial" calculation, and a "vacancy-mediated" calculator. Below we describe the

1. `Crystal class setup`_ needed to initiate a calculation,
2. `Interstitial calculator setup`_ needed for an single mobile species calculation, or
3. `Vacancy-mediated calculator setup`_ needed for a vacancy-mediated substitutional solute calculation,
4. the creation of `VASP-style input files`_ to be run to generate input data,
5. proper `Formatting of input data`_ to be compatible with the calculators, and
6. `Interpretation of output`_ which includes how to convert output into transport coefficients.

This follows the overall structure of a transport coefficient calculation. Broadly speaking, these are the steps necessary to compute transport coefficients:

1. Identify the crystal to be considered; this requires mapping whatever defects are to be considered mobile onto appropriate Wyckoff sites in the crystal, even if those exact sites are not occupied by true atoms.
2. Generate lists of symmetry unrelated "defect states" and "defect state transitions," along with the appropriate "calculator object."
3. Construct input files for total energy calculations to be run outside of the Onsager codebase; extract appropriate energy and frequency information from those runs.
4. Input the data in a format that the calculator can understand, and transform those energies and frequencies into rates at a given temperature assuming Arrhenius behavior.
5. Transform the output into physically relevant quantities (Onsager coefficients, solute diffusivities, mobilities, or drag ratios) with appropriate units.

Crystal class setup
-------------

Interstitial calculator setup
--------------------

Vacancy-mediated calculator setup
------------------------

VASP-style input files
----------------

Formatting of input data
-----------------

Interpretation of output
-----------------
