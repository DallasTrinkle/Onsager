#!/usr/bin/env python

### generates an FCC lattice

import numpy as np

def NNvect():
    r"""
    Constructs the FCC <110> transition vectors.
    """
    z = 12
    NNvect = np.empty((z,3), dtype=int)
    n=[0,0,0]
    z=0
    for n[0] in (-1,0,1):
        for n[1] in (-1,0,1):
            for n[2] in (-1,0,1):
                if sum(n)%2 == 0 and n.count(0) != 3:
                    NNvect[z] = n[:]
                    z+=1
    return NNvect

def invlist(NNvect):
    r"""
    Constructs the list of inverse vectors for each vector.

    Parameters
    ----------
    nnlist : array [:,:]
        list of nearest neighbor vectors; nnlist[k,:] = is the kth nearest neighbor

    Returns
    -------
    invlist : int array [:]
        for each k1, `nnlist`[k1] = -`nnlist`[`invlist`[k1]]
    """
    z = np.shape(NNvect)[0]
    invlist = np.empty((z), dtype=int)
    for k1 in xrange(z):
        for k2 in xrange(z):
            if all((NNvect[k1] + NNvect[k2]) == (0,0,0)) :
                invlist[k1] = k2
    return invlist

