Input and output for Onsager transport calculation
====================================

The Onsager calculators currently include two computational approaches to determining transport coefficients: an "interstitial" calculation, and a "vacancy-mediated" calculator. Below we describe the

0. `Assumptions used in Onsager`_ that are necessary to understand the data to be input, and the limitations of the results;
1. `Crystal class setup`_ needed to initiate a calculation;
2. `Interstitial calculator setup`_ needed for an single mobile species calculation, or,
3. `Vacancy-mediated calculator setup`_ needed for a vacancy-mediated substitutional solute calculation;
4. the creation of `VASP-style input files`_ to be run to generate input data;
5. proper `Formatting of input data`_ to be compatible with the calculators; and
6. `Interpretation of output`_ which includes how to convert output into transport coefficients.

This follows the overall structure of a transport coefficient calculation. Broadly speaking, these are the steps necessary to compute transport coefficients:

1. Identify the crystal to be considered; this requires mapping whatever defects are to be considered mobile onto appropriate Wyckoff sites in the crystal, even if those exact sites are not occupied by true atoms.
2. Generate lists of symmetry unrelated "defect states" and "defect state transitions," along with the appropriate "calculator object."
3. Construct input files for total energy calculations to be run outside of the Onsager codebase; extract appropriate energy and frequency information from those runs.
4. Input the data in a format that the calculator can understand, and transform those energies and frequencies into rates at a given temperature assuming Arrhenius behavior.
5. Transform the output into physically relevant quantities (Onsager coefficients, solute diffusivities, mobilities, or drag ratios) with appropriate units.

Assumptions used in Onsager
-------------------

The ``Onsager`` code computes transport of defects on an infinite crystalline lattice. Currently, the code requires that the particular defects can be mapped onto Wyckoff positions in a crystal. This does not *require* that the defect be an atom occupying various Wyckoff positions (though that obviously is captured), but merely that the defect have the symmetry and transitions that can be equivalently described by an "object" that occupies Wyckoff positions. Simple examples include vacancies, substitutional solutes, simple interstitial atoms, as well as more complex cases such as split vacancy defects (e.g.: a V-O\ :sub:`i`\ -V split double vacancy with oxygen interstitial in a closed-packed crystal; the entire defect complex can be mapped on to the Wyckoff position of the oxygen interstitial). In order to calculate diffusion, a few assumptions are made:

* **defects are dilute:** we never consider more than one defect at a time in an "infinite" periodic crystal; the vacancy-mediated diffuser uses one vacancy and one solute.
* **defects diffuse via a Markovian process**: defect states are well-defined, and the transition time from state-to-state is much longer than the equilibration time in a state, so that the evolution of the system is described by the Master equation with time-independent rates.
* **defects do not alter the underlying symmetry of the crystal**: while the defect itself can have a lower symmetry (according to its Wyckoff position), the presence of a defect does not lead to a global phase transformation to a different crystal; moreover, the crystal maintains translational invariance so that the energy of the system with defect(s) is unchanged under translations.

All of these assumptions are usually good: the dilute limit is valid without strong interactions (such as site blocking), Markovian processes are valid as long as barriers are a few times *k*\ :sub:`B`\ *T*, and we are not currently aware of any (simple) defects that induce phase transformations.

Furthermore, relaxation around a defect (or defect cluster) is allowed, but the assumption is that all of the atomic positions can be easily mapped back to "perfect" crystal lattice sites. This is an "off-lattice" model. In some cases, it can be possible to incorporate "new" states, especially metastable states, that are only accessible by a defect.

Finally, the code requires that all diffusion happens on a single sublattice. This sublattice is defined by a single chemical species; it can include multiple Wyckoff positions. But the current algorithms assume that transitions do not result in the creation of *antisite defects* (where a chemical species is on an "incorrect" sublattice).

Crystal class setup
--------------

The assumption of translational invariance of our defects is captured by the use of a ``Crystal` object. Following the standard definition of a crystal, we need to specify (a) three lattice vectors, and (b) at least one basis position, corresponding to at least one site. The crystal needs to contain *at least* the Wyckoff positions on a single sublattice corresponding to the diffusing defects. It can be useful for it to contain *more* atoms that act as "spectator" atoms: they do not participate in diffusion, but define both the underlying symmetry of the crystal, and if atomic-scale calculations will be used to compute configuration and transition-state energies, are necessary to define the energy landscape of diffusion.

* The *lattice vectors* are given by three vectors, :math:`\mathbf{a}_1`, :math:`\mathbf{a}_2`, :math:`\mathbf{a}_3` in Cartesian coordinates. In python, these are input when creating a Crystal either as a list of three ``numpy`` vectors, *or* as a square ``numpy`` matrix. **Note:** if you enter the three vectors as a matrix, remember that **it assumes the vectors are column vectors**. That is, if ``amat`` is the matrix, then ``amat[:,0]`` is :math:`\mathbf{a}_1`, ``amat[:,1]`` is :math:`\mathbf{a}_2`, and ``amat[:,2]`` is :math:`\mathbf{a}_2`. **This may not be what you're expecting.** The main recommendation is to enter the lattice vectors as a list (or tuple) of three ``numpy`` vectors.
* The *atomic basis* is given by a *list* of *lists* of ``numpy`` vectors of positions in *unit cell coordinates*. For a given ``basis``, then ``basis[0]`` is a list of all positions for the first chemical element in the crystal, ``basis[1]`` is the second chemical element, and so on. **If you only have a single chemical element, you may enter a list of ``numpy`` vectors.**
* An optional *spin* degree of freedom can be included. This is a list of objects, with one for each chemical element. These can be either scalar or vectors, with the assumption that they transform as those objects under group operations. If not included, the spins are all assumed to be equal to 0. Inclusion of these additional degrees of freedom (currently) only impacts the reduction of the unit cell, and the construction of the space group operations.
* We also take in, strictly for bookkeeping purposes, a list of names for the chemical elements. *This is an optional input*, but recommended for readability.

Once initialized, two main internal operations take place:

1. The unit cell is *reduced* and *optimized*. Reduction is a process where we try to find the smallest unit cell representation for the ``Crystal``. This means that the four-atom "simple cubic" unit cell of face-centered cubic can be input, and the code will reduce it to the standard single-atom primitive cell. The reduction algorithm can end up with "unusual" choices of lattice constants, so we also optimize the lattice vectors so that they are as close to orthogonal as possible, and ordered from smallest to largest. Neither choice changes the representation of the crystal; however, the *reduction* operation can be skipped by including the option ``noreduce=True``.
2. Full symmetry analysis is performed, including: automated construction of space group generator operators, partitioning of basis sites into symmetry related Wyckoff positions, and determination of point group operations for every basis site. All of these operations are automated, and make no reference to crystallographic tables. The algorithm cannot identify which space group it has generated, nor which Wyckoff positions are present. The algorithm respects both *chemistry* and *spin*; this also makes spin a useful manipulation tool to artificially lower symmetry for testing purposes as needed.

**Note**: ``Crystal``\ s can also be constructed by manipulating existing ``Crystal`` objects. A useful case is for the interstitial diffuser: when working "interactively," it is often easier to first make the underlying "spectator" crystal, and then have that ``Crystal`` construct the set of Wyckoff positions for a single site in the crystal, and then add that to the basis. ``Crystal`` objects are intended to be read-only, so these manipulations result in the creation of a new ``Crystal`` object.

A few quick examples:

Face-centered cubic crystal, vacancy-diffusion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Face-centered cubic crystals could be created either by entering the primitive basis::

  from onsager import crystal
  a0 = 1.
  FCCcrys = crystal.Crystal([a0*np.array([0,0.5,0.5]),
                                 a0*np.array([0.5,0,0.5]),
				 a0*np.array([0.5,0.5,0])],
				 [np.array([0, 0, 0])])

or 

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
