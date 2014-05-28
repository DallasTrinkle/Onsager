#!/usr/bin/env python

### generates an FCC lattice

import numpy as np

def NNvect():
    """
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
    """
    Constructs the list of inverse vectors for each vector.
    Returns invlist[] where, for each k1, nnlist[k1] = -nnlist[invlist[k1]]

    Parameters
    ----------
    nnlist: list of nearest neighbor vectors
    """    
    z = np.shape(NNvect)[0]
    invlist = np.empty((z), dtype=int)
    for k1 in xrange(z):
        for k2 in xrange(z):
            if all((NNvect[k1] + NNvect[k2]) == (0,0,0)) :
                invlist[k1] = k2
    return invlist

def neighlist (N, NNvect):
    """
    Constructs the full neighbor list for an NxNxN cell, using NNvect as the
    transition vectors; with full periodic boundary conditions.

    Returns an array list of sites, and their neighbors; uses the hashindex
    to identify.
    neighlist[i,z]: list of neighbor indices (will be constant once constructed)

    Parameters
    ----------
    N: size of box
    NNvect: output from makeNNvect()
    """
    z = np.shape(NNvect)[0]
    Nsite = N**3/2
    neighlist = np.empty((Nsite, z), dtype=int)
    n=[0,0,0]
    for n[0] in xrange(N):
        for n[1] in xrange(N):
            for n[2] in xrange(N):
                if sum(n)%2 == 0 :
                    nind = hashindex(n, N)
                    for nn in xrange(z):
                        neighlist[nind,nn] = hashindex((n+NNvect[nn])%N, N)
    return neighlist

def hashindex(n, N):
    """
    Hashing function, designed to make generation of 3D lattice map easily to vector.
    On a face-centerd cubic lattice: atom position (n0/N,n1/N,n2/N) where n0+n1+n2 = even
    This hash converts lattice position into unique integer from 0 .. N^3/2-1

    Parameters
    ----------
    n: three vector of coordinates, in units of 2*a0
    N: size of box on a side

    Notes
    -----
    Assumes that N is even, and that sum(n) is also even, and that all are integers;
    doesn't check.
    """
    return N*N*(n[0]/2) + N*n[1] + n[2]

