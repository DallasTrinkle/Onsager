"""
Automator code

Functions to convert from a supercell dictionary (output from a Diffuser) into a tarball
that contains all of the input files in an organized directory structure to run the
atomic-scale transition state calculations. This includes:

1. All positions in POSCAR format (POSCAR files for states to relax, POS as reference
  for transition endpoints that need to be relaxed)
2. Transformation information from relaxed states to initial states.
3. INCAR files for relaxation and NEB runs; KPOINTS for each.
4. perl script to transform CONTCAR output from a state relaxation to NEB endpoints.
5. perl script to linearly interpolate between NEB endpoints.
6. Makefile to run everything (representing the "directed graph" of calculations)
"""

__author__ = 'Dallas R. Trinkle'

import numpy as np
import collections, copy, itertools, warnings
from onsager import crystal, supercell
import tarfile, time, io, json


def map2string(tag, groupop, mapping):
    """
    Takes in a map: tag, groupop, mapping and constructs a string representation
    to be dumped to a file. If we want to call using the tuple, `map2string(*(map))` will suffice.

    :param tag: string of initial state to rotate
    :param groupop: see crystal.GroupOp; we use the rot and trans. This is in the supercell coord.
    :param mapping: in "chemorder" format; list by chemistry of lists of indices of position
        in initial cell to use.
    :return string_rep: string representation (to be used by an external script)
    """
    string_rep = tag + """
{rot[0][0]:.15f} {rot[0][1]:.15f} {rot[0][2]:.15f}
{rot[1][0]:.15f} {rot[1][1]:.15f} {rot[1][2]:.15f}
{rot[2][0]:.15f} {rot[2][1]:.15f} {rot[2][2]:.15f}
{trans[0]:.15f} {trans[1]:.15f} {trans[2]:.15f}
""".format(rot=groupop.rot, trans=groupop.trans)
    # the index shift needs to be added for each subsequent chemistry
    indexshift = [0] + list(itertools.accumulate(len(remap) for remap in mapping))
    string_rep += ' '.join(['{}'.format(m + shift)
                            for remap, shift in zip(mapping, indexshift)
                            for m in remap])
    return string_rep


def supercelltar(tar, superdict, filemode=0o664, directmode=0o775, timestamp=None,
                 INCARrelax="", INCARNEB="", KPOINTS="", basedir="",
                 statename='relax.', transitionname='neb.', IDformat='{:02d}',
                 JSONdict='tags.json', YAMLdef='supercell.yaml'):
    """
    Takes in a tarfile (needs to be open for reading) and a supercelldict (from a
    diffuser) and creates the full directory structure inside the tarfile. Best used in a form like

        with tarfile.open('supercells.tar.gz', mode='w:gz') as tar:
            automator.supercelltar(tar, supercelldict)

    :param tar: tarfile open for writing; may contain other files in advance.
    :param superdict: dictionary of `states`, `transitions`, `transmapping`, `indices` that
        correspond to dictionaries with tags; the final tag `reference` is the basesupercell for
        calculations without defects.
        superdict['states'][i] = supercell of state;
        superdict['transitions'][n] = (supercell initial, supercell final);
        superdict['transmapping'][n] = ((site tag, groupop, mapping), (site tag, groupop, mapping))
        superdict['indices'][tag] = (type, index) of tag, where tag is either a state or transition tag; or...
        superdict['indices'][tag] = index of tag, where tag is either a state or transition tag.
        superdict['reference'] = (optional) supercell reference, no defects
    :param filemode: (optional) mode to use for files; default = 664
    :param directmode: (optional) mode to use for directories; default = 775
    :param timestamp: (optional) if None, use current time.
    :param INCARrelax: (optional) contents of INCAR file to use for relaxation
    :param INCARNEB: (optional) contents of INCAR file to use for relaxation
    :param KPOINTS: (optional) contents of KPOINTS file
    :param basedir: prepended to all files/directories
    :param statename: prepended to all state names, before 2 digit number (default: relax.)
    :param transitionname: prepended to all transition names, before 2 digit number  (default: neb.)
    :param IDformat: format for integer tags (default: {:02d})
    :param JSONdict: name of JSON file storing the tags corresponding to each directory (default: tags.json)
    :param YAMLdef: YAML file containing full definition of supercells, relationship, etc. (default: supercell.yaml);
        set to None to not output. **may want to change this to None for the future**
    """
    if timestamp is None: timestamp = time.time()
    if len(basedir) > 0 and basedir[-1] != '/': basedir += '/'

    def addfile(filename, strdata):
        info = tarfile.TarInfo(basedir + filename)
        info.mode, info.mtime = filemode, timestamp
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
    dirmapping = {k: statename + IDformat.format(n) for n, k in enumerate(sorted(states.keys()))}
    for n, k in enumerate(sorted(transitions.keys())):
        dirmapping[k] = transitionname + IDformat.format(n)
    tagmapping = {v: k for k, v in dirmapping.items()}

    # add the common VASP input files:
    for filename, strdata in (('INCAR.relax', INCARrelax),
                              ('INCAR.NEB', INCARNEB),
                              ('KPOINTS', KPOINTS)):
        addfile(filename, strdata)
    # now, go through the states:
    if 'reference' in superdict:
        addfile('POSCAR', superdict['reference'].POSCAR('Defect-free reference'))
    for tag, super in states.items():
        # directory first
        dirname = dirmapping[tag]
        adddirectory(dirname)
        # POSCAR file next
        addfile(dirname + '/POSCAR', super.POSCAR(tag))
        addsymlink(dirname + '/INCAR', '../INCAR.relax')
        addsymlink(dirname + '/KPOINTS', '../KPOINTS')
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
            if superdict['transmapping'][tag][0] is None \
            else dirname + '/POS.final'
        addfile(filename, super1.POSCAR('final ' + tag))
        addsymlink(dirname + '/INCAR', '../INCAR.NEB')
        addsymlink(dirname + '/KPOINTS', '../KPOINTS')
        addsymlink(dirname + '/POTCAR', '../POTCAR')

    # and the transition mappings:
    for tag, (map0, map1) in transmapping.items():
        dirname = dirmapping[tag]
        if map0 is not None:
            addfile(dirname + '/trans.init', map2string(dirmapping[map0[0]], map0[1], map0[2]))
        if map1 is not None:
            addfile(dirname + '/trans.final', map2string(dirmapping[map1[0]], map1[1], map1[2]))

    # JSON dictionary connecting directories and tags:
    addfile(JSONdict, json.dumps(tagmapping, indent=4, sort_keys=True))
    # YAML representation of supercell:
    if YAMLdef is not None: addfile(YAMLdef, crystal.yaml.dump(superdict))
