"""
Automator code

Functions to convert from a supercell dictionary (output from a Diffuser) into a tarball
that contains all of the input files in an organized directory structure to run the
atomic-scale transition state calculations. This includes:

1. All positions in POSCAR format (POSCAR files for states to relax, POS as reference for transition endpoints that need to be relaxed)
2. Transformation information from relaxed states to initial states.
3. INCAR files for relaxation and NEB runs; KPOINTS for each.
4. perl script to transform CONTCAR output from a state relaxation to NEB endpoints.
5. perl script to linearly interpolate between NEB endpoints.*
6. Makefile to run NEB construction.

*Note:* the NEB interpolator script (nebmake.pl) is part of the `VTST scripts <http://theory.cm.utexas.edu/vtsttools/scripts.html>`_.
"""

__author__ = 'Dallas R. Trinkle'

import numpy as np
import collections, copy, itertools, warnings
from onsager import crystal, supercell
import tarfile, time, io, json
import pkg_resources


def map2string(tag, groupop, mapping):
    """
    Takes in a map: tag, groupop, mapping and constructs a string representation
    to be dumped to a file. If we want to call using the tuple, ``map2string(*(map))`` will suffice.

    :param tag: string of initial state to rotate
    :param groupop: see crystal.GroupOp; we use the rot and trans. This is in the supercell coord.
    :param mapping: in "chemorder" format; list by chemistry of lists of indices of position
        in initial cell to use.
    :return string_rep: string representation (to be used by an external script)
    """
    string_rep = tag + """
{rot[0][0]:3d} {rot[0][1]:3d} {rot[0][2]:3d}
{rot[1][0]:3d} {rot[1][1]:3d} {rot[1][2]:3d}
{rot[2][0]:3d} {rot[2][1]:3d} {rot[2][2]:3d}
{trans[0]:.16f} {trans[1]:.16f} {trans[2]:.16f}
""".format(rot=groupop.rot, trans=groupop.trans)
    # the index shift needs to be added for each subsequent chemistry
    indexshift = [0] + list(itertools.accumulate(len(remap) for remap in mapping))
    string_rep += ' '.join(['{}'.format(m + shift)
                            for remap, shift in zip(mapping, indexshift)
                            for m in remap])
    # needs a trailing newline
    return string_rep + '\n'


### Some default input files to use for our runs, and a sed formatted script to recreate INCARs

SEDstring = "s/{{system}}/{system}/\n"

INCARrelax = """SYSTEM = {system}
PREC = High
ISIF = 2
EDIFF = 1E-8
EDIFFG = -10E-3
IBRION = 2
NSW = 50
ISMEAR = 1
SIGMA = 0.1
# ENCUT =
# NGX =
# NGY =
# NGZ =
# NGXF =
# NGYF =
# NGZF =
# NPAR =
LWAVE  = .FALSE.
LCHARG = .FALSE.
LREAL  = .FALSE.
VOSKOWN = 1
"""

INCARNEB = INCARrelax + \
"""IMAGES = 1
SPRING = -5
LCLIMB = .TRUE.
NELMIN = 4
NFREE = 10
"""

KPOINTSgammaonly = """Gamma
1
Reciprocal
0. 0. 0. 1.
"""

KPOINTS_MP = """Monkhorst-Pack mesh {N1}x{N2}x{N3}
0
Monkhorst
{N1} {N2} {N3}
0. 0. 0.
"""

KPOINTS_Gamma = """Gamma-centered mesh {N1}x{N2}x{N3}
0
Gamma
{N1} {N2} {N3}
0. 0. 0.
"""

MAKEFILE = r"""# Makefile to construct NEB input from relaxation output
# we set this so that the makefile doesn't use builtin implicit rules
MAKEFLAGS = -rk

makeneb := "./nebmake.pl"
transform := "./trans.pl"

Nimages ?= 1

.PHONY: help

target := $(foreach neb, $(wildcard neb.*), $(neb)/01/POSCAR)
target: $(target)

help:
	@echo "# Creates input POSCAR for NEB runs, once relaxation runs are complete"
	@echo "# Uses CONTCAR in relaxation directories to create initial run geometry"
	@echo "# environment variable: Nimages (default: $(Nimages))"
	@echo "# target files:"
	@echo $(target) | sed 's/ /\n/g'
	@echo "# default target: all"

neb.%: neb.%/01/POSCAR neb.%/POSCAR.init neb.%/POSCAR.final

neb.%/01/POSCAR: neb.%/POSCAR.init neb.%/POSCAR.final
	@$(makeneb) $^ $(Nimages)

neb.%/POSCAR.init:
	@$(transform) $^ > $@

neb.%/POSCAR.final:
	@$(transform) $^ > $@

###############################################################
# structure of NEB runs:
"""


def supercelltar(tar, superdict, filemode=0o664, directmode=0o775, timestamp=None,
                 INCARrelax=INCARrelax, INCARNEB=INCARNEB, KPOINTS=KPOINTSgammaonly, basedir="",
                 statename='relax.', transitionname='neb.', IDformat='{:02d}',
                 JSONdict='tags.json', YAMLdef='supercell.yaml'):
    """
    Takes in a tarfile (needs to be open for writing) and a supercelldict (from a
    diffuser) and creates the full directory structure inside the tarfile. Best used in
    a form like

    ::

        with tarfile.open('supercells.tar.gz', mode='w:gz') as tar:
            automator.supercelltar(tar, supercelldict)

    :param tar: tarfile open for writing; may contain other files in advance.
    :param superdict: dictionary of ``states``, ``transitions``, ``transmapping``, ``indices`` that
        correspond to dictionaries with tags; the final tag ``reference`` is the basesupercell
        for calculations without defects.

        * superdict['states'][i] = supercell of state;
        * superdict['transitions'][n] = (supercell initial, supercell final);
        * superdict['transmapping'][n] = ((site tag, groupop, mapping), (site tag, groupop, mapping))
        * superdict['indices'][tag] = (type, index) of tag, where tag is either a state or transition tag; or...
        * superdict['indices'][tag] = index of tag, where tag is either a state or transition tag.
        * superdict['reference'] = (optional) supercell reference, no defects

    :param filemode: mode to use for files (default: 664)
    :param directmode: mode to use for directories (default: 775)
    :param timestamp: UNIX time for files; if None, use current time (default)
    :param INCARrelax: contents of INCAR file to use for relaxation; must contain {system} to be replaced
        by tag value (default: automator.INCARrelax)
    :param INCARNEB: contents of INCAR file to use for NEB; must contain {system} to be replaced
        by tag value (default: automator.INCARNEB)
    :param KPOINTS: contents of KPOINTS file (default: gamma-point only calculation);
        if None or empty, no KPOINTS file at all
    :param basedir: prepended to all files/directories (default: '')
    :param statename: prepended to all state names, before 2 digit number (default: relax.)
    :param transitionname: prepended to all transition names, before 2 digit number  (default: neb.)
    :param IDformat: format for integer tags (default: {:02d})
    :param JSONdict: name of JSON file storing the tags corresponding to each directory (default: tags.json)
    :param YAMLdef: YAML file containing full definition of supercells, relationship, etc. (default: supercell.yaml);
        set to None to not output. **may want to change this to None for the future**
    """
    if timestamp is None: timestamp = time.time()
    if len(basedir) > 0 and basedir[-1] != '/': basedir += '/'
    kpoints = not ((KPOINTS is None) or (KPOINTS == ""))

    def addfile(filename, strdata, executable=False):
        info = tarfile.TarInfo(basedir + filename)
        info.mode, info.mtime = filemode, timestamp
        if executable: info.mode = directmode
        info.size = len(strdata.encode('ascii'))
        tar.addfile(info, io.BytesIO(strdata.encode('ascii')))

    def adddirectory(dirname):
        info = tarfile.TarInfo(basedir + dirname)
        info.type = tarfile.DIRTYPE
        info.mode, info.mtime = directmode, timestamp
        tar.addfile(info)

    def addsymlink(linkname, target):
        info = tarfile.TarInfo(basedir + linkname)
        info.type = tarfile.SYMTYPE
        info.mode, info.mtime = filemode, timestamp
        info.linkname = target
        tar.addfile(info)

    # our tags make for troublesome directory names; construct a mapping:
    states, transitions, transmapping = superdict['states'], superdict['transitions'], superdict['transmapping']
    # we do a reverse sorting on state keys, so that vacancies and complexes are first; we use
    # normal order for the transitions.
    dirmapping = {k: statename + IDformat.format(n)
                  for n, k in enumerate(sorted(states.keys(), reverse=True))}
    for n, k in enumerate(sorted(transitions.keys())):
        dirmapping[k] = transitionname + IDformat.format(n)
    tagmapping = {v: k for k, v in dirmapping.items()}

    # add the common VASP input files: (weird construction to check if kpoints is True)
    for filename, strdata in (('INCAR.relax', INCARrelax), ('INCAR.NEB', INCARNEB)) + \
            ((('KPOINTS', KPOINTS),) if kpoints else tuple()):
        addfile(filename, strdata)
    addfile('trans.pl', str(pkg_resources.resource_string(__name__, 'trans.pl'), 'ascii'), executable=True)
    addfile('nebmake.pl', str(pkg_resources.resource_string(__name__, 'nebmake.pl'), 'ascii'), executable=True)
    addfile('Vasp.pm', str(pkg_resources.resource_string(__name__, 'Vasp.pm'), 'ascii'))
    # now, go through the states:
    if 'reference' in superdict:
        addfile('POSCAR', superdict['reference'].POSCAR('Defect-free reference'))
    for tag, super in states.items():
        # directory first
        dirname = dirmapping[tag]
        adddirectory(dirname)
        # POSCAR file next
        addfile(dirname + '/POSCAR', super.POSCAR(tag))
        addfile(dirname + '/INCAR', INCARrelax.format(system=tag))
        addfile(dirname + '/incar.sed', SEDstring.format(system=tag))
        if kpoints: addsymlink(dirname + '/KPOINTS', '../KPOINTS')
        addsymlink(dirname + '/POTCAR', '../POTCAR')
    # and the transitions:
    for tag, (super0, super1) in transitions.items():
        # directory first
        dirname = dirmapping[tag]
        adddirectory(dirname)
        # POS/POSCAR files next
        filename = dirname + '/POSCAR.init' \
            if superdict['transmapping'][tag][0] is None \
            else dirname + '/POS.init'
        addfile(filename, super0.POSCAR('initial ' + tag))
        filename = dirname + '/POSCAR.final' \
            if superdict['transmapping'][tag][1] is None \
            else dirname + '/POS.final'
        addfile(filename, super1.POSCAR('final ' + tag))
        addfile(dirname + '/INCAR', INCARNEB.format(system=tag))
        addfile(dirname + '/incar.sed', SEDstring.format(system=tag))
        if kpoints: addsymlink(dirname + '/KPOINTS', '../KPOINTS')
        addsymlink(dirname + '/POTCAR', '../POTCAR')

    # and the transition mappings:
    Makefile = MAKEFILE
    relaxNEB = {}
    for tag in sorted(transmapping.keys()):
        dirname = dirmapping[tag]
        for m, t in ((transmapping[tag][0], 'init'), (transmapping[tag][1], 'final')):
            if m is not None:
                relax = dirmapping[m[0]]
                addfile(dirname + '/trans.' + t, map2string(relax, m[1], m[2]))
                Makefile += \
                    "{neb}/POSCAR.{type}: {neb}/trans.{type} {relax}/CONTCAR\n".format(neb=dirname,
                                                                                       type=t, relax=relax)
                if relax not in relaxNEB: relaxNEB[relax] = {dirname}
                else: relaxNEB[relax].add(dirname)
    addfile('Makefile', Makefile)
    for relax, NEBset in relaxNEB.items():
        addfile(relax + '/NEBlist', '\n'.join(k for k in sorted(NEBset)) + '\n')

    # JSON dictionary connecting directories and tags: (needs a trailing newline?)
    addfile(JSONdict, json.dumps(tagmapping, indent=4, sort_keys=True) + '\n')
    # YAML representation of supercell:
    if YAMLdef is not None: addfile(YAMLdef, crystal.yaml.dump(superdict))
