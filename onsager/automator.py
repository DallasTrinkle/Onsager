"""
Automator code

Functions to convert from a supercell dictionary (output from a Diffuser) into a tarball
that contains all of the input files in an organized directory structure to run the
atomic-scale transition state calculations. This includes:

1. All positions in POSCAR format (POSCAR files for states to relax, POS as reference
  for transition endpoints that need to be relaxed)
2. Transformation information from relaxed states to initial states.
3. INCAR files for relaxation and NEB runs; KPOINTS for each.
4. perl script to transform CONCAR output from a state relaxation to NEB endpoints.
5. perl script to linearly interpolate between NEB endpoints.
6. Makefile to run everything (representing the "directed graph" of calculations)
"""

__author__ = 'Dallas R. Trinkle'

import numpy as np
import collections, copy, itertools, warnings
from onsager import crystal, supercell
import tarfile, time, io


def map2string(map):
    """
    Takes in a map (a tuple of tag, groupop, mapping) and constructs a string representation
    to be dumped to a file.

    :param map: tuple of tag, groupop, mapping
        tag = string of initial state to rotate
        groupop = see crystal.GroupOp; we use the rot and trans. This is in the supercell
        mapping = in "chemorder" format; list by chemistry of lists of indices of position
            in initial cell to use.
    :return string_rep: string representation (to be used by an external script)
    """
    tag, groupop, mapping = map
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
                 INCARrelax="", INCARNEB="", KPOINTS=""):
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
    """
    if timestamp is None: timestamp = time.time()

    def addfile(filename, strdata):
        info = tarfile.TarInfo(filename)
        info.mode, info.mtime = filemode, timestamp
        info.size = len(strdata.encode('ascii'))
        tar.addfile(info, io.BytesIO(strdata.encode('ascii')))

    def adddirectory(dirname):
        info = tarfile.TarInfo(dirname)
        info.type = tarfile.DIRTYPE
        info.mode, info.mtime = directmode, timestamp
        tar.addfile(info)

    def addsymlink(linkname, target):
        info = tarfile.TarInfo(linkname)
        info.type = tarfile.SYMTYPE
        info.mode, info.mtime = filemode, timestamp
        info.linkname = target
        tar.addfile(info)

    # add the common VASP input files:
    for filename, strdata in (('INCAR.relax', INCARrelax),
                              ('INCAR.NEB', INCARNEB),
                              ('KPOINTS', KPOINTS)):
        addfile(filename, strdata)
    # now, go through the states:
    if 'reference' in superdict:
        addfile('POSCAR', superdict['reference'].POSCAR('Undefected reference'))
    for tag, super in superdict['states'].items():
        # directory first
        adddirectory(tag)
        # POSCAR file next
        addfile(tag + '/POSCAR', super.POSCAR(tag))
        addsymlink(tag + '/INCAR', '../INCAR.relax')
        addsymlink(tag + '/KPOINTS', '../KPOINTS')
        addsymlink(tag + '/POTCAR', '../POTCAR')
    # and the transitions:
    for tag, (super0, super1) in superdict['transitions'].items():
        # directory first
        adddirectory(tag)
        # POS/POSCAR files next
        filename = tag + '/POSCAR.init' \
            if superdict['transmapping'][tag][0] is None \
            else tag + '/POS.init'
        addfile(filename, super0.POSCAR('initial ' + tag))
        filename = tag + '/POSCAR.final' \
            if superdict['transmapping'][tag][0] is None \
            else tag + '/POS.final'
        addfile(filename, super1.POSCAR('final ' + tag))
        addsymlink(tag + '/INCAR', '../INCAR.NEB')
        addsymlink(tag + '/KPOINTS', '../KPOINTS')
        addsymlink(tag + '/POTCAR', '../POTCAR')

    # and the transition mappings:
    for tag, (map0, map1) in superdict['transmapping'].items():
        if map0 is not None:
            addfile(tag + '/trans.init', map2string(map0))
        if map1 is not None:
            addfile(tag + '/trans.final', map2string(map1))
