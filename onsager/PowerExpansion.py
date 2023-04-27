"""
Power expansion class

Class to store and manipulate 3-dimensional Taylor (power) expansions of functions
Particularly useful for inverting the FT of the evolution matrix, and subtracting off
analytically calculated IFT for the Green function.

Really designed to get used by other code.
"""

__author__ = 'Dallas R. Trinkle'

import numpy as np
from numbers import Number
from scipy.special import factorial, comb
# from scipy.misc import comb


class Taylor3D(object):
    """
    Class that stores a Taylor expansion of a function in 3D, and defines some arithmetic
    """

    # a whole series of sorts of automated computation for setup

    @staticmethod
    def makeindexPowerYlm(Lmax):
        """
        Analyzes the spherical harmonics and powers for a given Lmax; returns a
        series of index functions.

        :param Lmax: maximum l value to consider; equal to the sum of powers
        :return NYlm: number of Ylm coefficients
        :return Npower: number of power coefficients
        :return pow2ind[n1][n2][n3]: powers to index
        :return ind2pow[n]: powers for a given index
        :return Ylm2ind[l][m]: (l,m) to index
        :return ind2Ylm[lm]: (l,m) for a given index
        :return powlrange[l]: upper limit of power indices for a given l value; note: [-1] = 0
        """
        # first, the counts
        NYlm = (Lmax + 1) ** 2
        Npower = NYlm + ((Lmax + 1) * Lmax * (Lmax - 1)) // 6
        # indexing arrays
        powlrange = np.zeros(Lmax + 2, dtype=int)
        powlrange[-1] = 0
        pow2ind = -np.ones((Lmax + 1, Lmax + 1, Lmax + 1), dtype=int)
        ind2pow = np.zeros((Npower, 3), dtype=int)
        Ylm2ind = -np.ones((Lmax + 1, 2 * Lmax + 1), dtype=int)
        ind2Ylm = np.zeros((NYlm, 2), dtype=int)
        # powers first; these are ordered by increasing l = n1+n2+n3
        ind = 0
        for l in range(Lmax + 1):
            for n1 in range(l + 1):
                for n2 in range(l + 1 - n1):
                    n3 = l - n1 - n2
                    pow2ind[n1, n2, n3] = ind
                    ind2pow[ind, 0], ind2pow[ind, 1], ind2pow[ind, 2] = n1, n2, n3
                    ind += 1
            powlrange[l] = ind
        # next, Ylm values
        ind = 0
        for l in range(Lmax + 1):
            for m in range(-l, l + 1):
                Ylm2ind[l, m] = ind
                ind2Ylm[ind, 0], ind2Ylm[ind, 1] = l, m
                ind += 1
        return NYlm, Npower, pow2ind, ind2pow, Ylm2ind, ind2Ylm, powlrange

    @classmethod
    def makeYlmpow(cls):
        """
        Construct the expansion of the Ylm's in powers of x,y,z. Done via brute force.

        :return Ylmpow[lm, p]: expansion of each Ylm in powers
        """
        Ylmpow = np.zeros((cls.NYlm, cls.Npower), dtype=complex)
        for l in range(cls.Lmax + 1):
            # do the positive m first; then easily swap to get the negative m
            for m in range(l + 1):
                ind = cls.Ylm2ind[l, m]
                pre = (-1) ** m * np.sqrt((2 * l + 1) * factorial(l - m, True) /
                                          (4 * np.pi * factorial(l + m, True)))
                for k in range((l + m + 1) // 2, l + 1):
                    zz = (-1) ** (l - k) * factorial(2 * k, True) / \
                         (2 ** l * factorial(2 * k - l - m, True) * factorial(k, True) * factorial(l - k, True))
                    for j in range(m + 1):
                        # xy = factorial(m, True) / (factorial(j, True) * factorial(m - j, True))
                        xy = comb(m, j)
                        Ylmpow[ind, cls.pow2ind[j, m - j, 2 * k - l - m]] = pre * zz * xy * (1.j) ** (m - j)
            for m in range(-l, 0):
                ind = cls.Ylm2ind[l, m]
                indpos = cls.Ylm2ind[l, -m]
                for p in range(cls.Npower):
                    Ylmpow[ind, p] = (-1) ** (-m) * Ylmpow[indpos, p].conjugate()
        return Ylmpow

    @classmethod
    def makepowYlm(cls):
        """
        Construct the expansion of the powers in Ylm's. Done using recursion relations
        instead of direct calculation. Note: an alternative approach would be Gaussian
        quadrature.

        :return powYlm[p][lm]: expansion of powers in Ylm; uses indexing scheme above
        """
        powYlm = np.zeros((cls.Npower, cls.NYlm), dtype=complex)
        Cp = np.zeros((cls.Lmax, 2 * cls.Lmax - 1))
        Cm = np.zeros((cls.Lmax, 2 * cls.Lmax - 1))
        Sp = np.zeros((cls.Lmax, 2 * cls.Lmax - 1))
        Sm = np.zeros((cls.Lmax, 2 * cls.Lmax - 1))
        # because this is for our recursion relations, we only need to work to Lmax-1 !
        for l, m in ((l, m) for l in range(cls.Lmax) for m in range(-l, l + 1)):
            Cp[l, m] = np.sqrt((l - m + 1) * (l + m + 1) / ((2 * l + 1) * (2 * l + 3)))
            Sp[l, m] = 0.5 * np.sqrt((l + m + 1) * (l + m + 2) / ((2 * l + 1) * (2 * l + 3)))
            if l > 0:  # and -l < m < l:
                Cm[l, m] = np.sqrt((l - m) * (l + m) / ((2 * l - 1) * (2 * l + 1)))
                Sm[l, m] = 0.5 * np.sqrt((l - m) * (l - m - 1) / ((2 * l - 1) * (2 * l + 1)))

        # first, prime the pump with 1
        powYlm[cls.pow2ind[0, 0, 0], cls.Ylm2ind[0, 0]] = np.sqrt(4 * np.pi)
        for n0, n1, n2 in ((n0, n1, n2) for n0 in range(cls.Lmax + 1)
                           for n1 in range(cls.Lmax + 1)
                           for n2 in range(cls.Lmax + 1)
                           if 0 < n0 + n1 + n2 <= cls.Lmax):
            ind = cls.pow2ind[n0, n1, n2]
            lmax = n0 + n1 + n2
            if n2 > 0:
                # we can recurse up from n0, n1, n2-1
                indlow = cls.pow2ind[n0, n1, n2 - 1]
                for l, m in ((l, m) for l in range(lmax) for m in range(-l, l + 1)):
                    plm = powYlm[indlow, cls.Ylm2ind[l, m]]
                    powYlm[ind, cls.Ylm2ind[l + 1, m]] += Cp[l, m] * plm
                    if l > 0 and -l < m < l:
                        powYlm[ind, cls.Ylm2ind[l - 1, m]] += Cm[l, m] * plm
            elif n1 > 0:
                # we can recurse up from n0, n1-1, n2
                indlow = cls.pow2ind[n0, n1 - 1, n2]
                for l, m in ((l, m) for l in range(lmax) for m in range(-l, l + 1)):
                    plm = powYlm[indlow, cls.Ylm2ind[l, m]]
                    powYlm[ind, cls.Ylm2ind[l + 1, m + 1]] += 1.j * Sp[l, m] * plm
                    powYlm[ind, cls.Ylm2ind[l + 1, m - 1]] += 1.j * Sp[l, -m] * plm
                    # if l>0:
                    if m < l - 1:
                        powYlm[ind, cls.Ylm2ind[l - 1, m + 1]] += -1.j * Sm[l, m] * plm
                    if m > -l + 1:
                        powYlm[ind, cls.Ylm2ind[l - 1, m - 1]] += -1.j * Sm[l, -m] * plm
            elif n0 > 0:
                # we can recurse up from n0-1, n1, n2
                indlow = cls.pow2ind[n0 - 1, n1, n2]
                for l, m in ((l, m) for l in range(lmax) for m in range(-l, l + 1)):
                    plm = powYlm[indlow, cls.Ylm2ind[l, m]]
                    powYlm[ind, cls.Ylm2ind[l + 1, m + 1]] += -Sp[l, m] * plm
                    powYlm[ind, cls.Ylm2ind[l + 1, m - 1]] += Sp[l, -m] * plm
                    # if l>0:
                    if m < l - 1:
                        powYlm[ind, cls.Ylm2ind[l - 1, m + 1]] += Sm[l, m] * plm
                    if m > -l + 1:
                        powYlm[ind, cls.Ylm2ind[l - 1, m - 1]] += -Sm[l, -m] * plm
        return powYlm

    @classmethod
    def makeLprojections(cls):
        """
        Constructs a series of projection matrices for each l component in our power series

        :return: projL[l][p][p']
            projection of powers containing *only* l component.
            -1 component = sum(l=0..Lmax, projL[l]) = simplification projection
        """
        projL = np.zeros((cls.Lmax + 2, cls.Npower, cls.Npower))
        projLYlm = np.zeros((cls.Lmax + 2, cls.NYlm, cls.NYlm), dtype=complex)
        for l, m in ((l, m) for l in range(0, cls.Lmax + 1) for m in range(-l, l + 1)):
            lm = cls.Ylm2ind[l, m]
            projLYlm[l, lm, lm] = 1.  # l,m is part of l
            projLYlm[-1, lm, lm] = 1.  # all part of the sum
        for l in range(cls.Lmax + 2):
            # projL[l] = np.dot(cls.powYlm, np.dot(projLYlm[l], cls.Ylmpow)).real
            projL[l] = np.tensordot(cls.Ylmpow,
                                    np.tensordot(projLYlm[l], cls.powYlm, axes=(1, 1)),
                                    axes=(0, 0)).real
        return projL

    @classmethod
    def makedirectmult(cls):
        """
        :return direcmult[p][p']: index that corresponds to the multiplication of power indices p and p'
        """
        directmult = -np.ones((cls.Npower, cls.Npower), dtype=int)
        for (p0, p1) in ((p0, p1) for p0 in range(cls.Npower) for p1 in range(cls.Npower)):
            nsum = cls.ind2pow[p0] + cls.ind2pow[p1]
            if sum(nsum) <= cls.Lmax:
                directmult[p0, p1] = cls.pow2ind[nsum[0], nsum[1], nsum[2]]
        return directmult

    @classmethod
    def powexp(cls, u, normalize=True):
        """
        Given a vector u, normalize it and return the power expansion of uvec

        :param u[3]: vector to apply
        :param normalize: do we normalize u first?
        :return upow[Npower]: ux uy uz products of powers
        :return umagn: magnitude of u (if normalized)
        """
        umagn = np.sqrt(np.dot(u, u))
        upow = np.zeros(cls.Npower)
        if umagn < 1e-8:
            upow[cls.pow2ind[0, 0, 0]] = 1.
            umagn = 0.
        else:
            u0 = u.copy()
            if normalize: u0 /= umagn
            xyz = np.ones((cls.Lmax + 1, 3))
            for n in range(1, cls.Lmax + 1):
                xyz[n, :] = xyz[n - 1, :] * u0[:]
            for n0, n1, n2 in ((n0, n1, n2) for n0 in range(cls.Lmax + 1)
                               for n1 in range(cls.Lmax + 1)
                               for n2 in range(cls.Lmax + 1)
                               if n0 + n1 + n2 <= cls.Lmax):
                upow[cls.pow2ind[n0, n1, n2]] = xyz[n0, 0] * xyz[n1, 1] * xyz[n2, 2]
        if normalize:
            return upow, umagn
        else:
            return upow

    @classmethod
    def makepowercoeff(cls):
        """
        Make our power coefficients for our construct expansion method

        :return powercoeff[n][p]: vector we multiply by our power expansion to get the n'th coefficients
        """
        powercoeff = np.zeros((cls.Lmax + 1, cls.Npower))
        for n0 in range(cls.Lmax + 1):
            for n1 in range(cls.Lmax + 1):
                for n2 in range(cls.Lmax + 1):
                    n = n0 + n1 + n2
                    if n <= cls.Lmax:
                        powercoeff[n, cls.pow2ind[n0, n1, n2]] = \
                            factorial(n, True) / (factorial(n0, True) * factorial(n1, True) * factorial(n2, True))
        return powercoeff

    @classmethod
    def constructexpansion(cls, basis, N=-1, pre=None):
        """
        Takes a "basis" for constructing an expansion -- list of vectors and matrices --
        and constructs the expansions up to power N (default = Lmax)

        :param basis = list((coeffmatrix, vect)): expansions to create;
          sum(coeffmatrix * (vect*q)^n), for powers n = 0..N
        :param N: maximum power to consider; for N=-1, use Lmax
        :param pre: list of prefactors, defining the Taylor expansion. Default = 1
        :return list((n, lmax, powexpansion)),...: our expansion, as input to create
          Taylor3D objects
        """
        if N < 0: N = cls.Lmax
        if pre is None:
            pre = [1 for n in range(N + 1)]
        c = []
        for n in range(N + 1):
            c.append([(n, n, np.zeros((cls.powlrange[n],) + basis[0][0].shape, dtype=complex))])
        for coeff, vect in basis:
            pexp = cls.powexp(vect, normalize=False)
            for n in range(N + 1):
                vnpow = (cls.powercoeff[n] * pexp)[:cls.powlrange[n]]
                cn = c[n][0][2]
                for p in range(cls.powlrange[n]):
                    cn[p] += pre[n] * vnpow[p] * coeff
        return tuple(c)

    @classmethod
    def rotatedirections(cls, qptrans):
        """
        Takes a transformation matrix qptrans, where q[i] = sum_j qptrans[i][j] p[j], and
        returns the Npow x Npow transformation matrix for the new components in terms of
        the old.
        NOTE: This is more complex than one might first realize. If we only work with cases
        where all of the entries for a given power n have those same n (that is, not reduced),
        then this is straightforward. However, we run into problems with *reductions*: e.g.,
        for n=2, the power :math:`x^0 y^0 z^0` is, in reality, :math:`x^2+y^2+z^2`, and hence
        *it must be transformed* because we allow non-orthogonal transformation matrices.

        :param qptrans: 3x3 matrix
        :return npowtrans: [Lmax +1][Npow][Npow] transformation matrix [n][original pow][new pow]
            for each n from 0 up to Lmax
        """
        powtrans = np.zeros((cls.Npower, cls.Npower))
        # l = 0 case
        powtrans[0, 0] = 1
        # single q value cases
        for i in range(3):
            qi_pow = cls.powexp(qptrans[i, :], normalize=False)
            for n in range(1, cls.Lmax + 1):
                powtrans[cls.pow2ind[(0,) * i + (n,) + (0,) * (2 - i)], :] = cls.powercoeff[n] * qi_pow
        # pairs of q cases: we get qi^ni qj^nj by direct multiplication
        # triplet is done inside the loop: q1^n1 q2^n2 q3^n3 = (q1^n1 q2^n2) (q3^n3)
        for i in range(3):
            for j in range(i + 1, 3):
                for ni in range(1, cls.Lmax + 1):
                    powi = cls.pow2ind[(0,) * i + (ni,) + (0,) * (2 - i)]
                    for nj in range(1, cls.Lmax + 1 - ni):
                        powj = cls.pow2ind[(0,) * j + (nj,) + (0,) * (2 - j)]
                        powij = cls.pow2ind[(0,) * i + (ni,) + (0,) * (j - i - 1) + (nj,) + (0,) * (2 - j)]
                        # multiply the pair!
                        for pi in range(cls.powlrange[ni - 1], cls.powlrange[ni]):
                            for pj in range(cls.powlrange[nj - 1], cls.powlrange[nj]):
                                powtrans[powij, cls.directmult[pi, pj]] += powtrans[powi, pi] * powtrans[powj, pj]
                        if j == 1:
                            # do the triplet
                            # k = 2 (instead of explicitly writing another loop)
                            for nk in range(1, cls.Lmax + 1 - ni - nj):
                                powk = cls.pow2ind[0, 0, nk]
                                powijk = cls.pow2ind[ni, nj, nk]
                                for pij in range(cls.powlrange[ni + nj - 1], cls.powlrange[ni + nj]):
                                    for pk in range(cls.powlrange[nk - 1], cls.powlrange[nk]):
                                        powtrans[powijk, cls.directmult[pij, pk]] += powtrans[powij, pij] * powtrans[
                                            powk, pk]
        npowtrans = np.zeros((cls.Lmax + 1, cls.Npower, cls.Npower))
        for n in range(cls.Lmax + 1):
            prange = slice(cls.powlrange[n - 1], cls.powlrange[n])
            npowtrans[n, prange, prange] = powtrans[prange, prange]
            # now, work on lower values (n-2, n-4, ...)
            for m in range(n - 2, -1, -2):
                # powers that sum up to m:
                for tup in [(n0, n1, m - n0 - n1) for n0 in range(m + 1) for n1 in range(m - n0 + 1)]:
                    npowtrans[n, cls.pow2ind[tup], :] = npowtrans[n, cls.pow2ind[tup[0] + 2, tup[1], tup[2]], :] + \
                                                        npowtrans[n, cls.pow2ind[tup[0], tup[1] + 2, tup[2]], :] + \
                                                        npowtrans[n, cls.pow2ind[tup[0], tup[1], tup[2] + 2], :]
        return npowtrans

    # for sorting our coefficient lists:
    @classmethod
    def __sortkey(cls, entry):
        return (entry[0] + entry[1] / (cls.Lmax + 1))

    __INITIALIZED__ = False

    # these are all *class* parameters, not object parameters: they are computed
    # and defined once for the entire class. It means that once, in your code, you *choose*
    # a value for Lmax, you are stuck with it. This is a choice: it makes compatibility between
    # the expansions easy, for a minor loss in flexibility.
    # Note: I believe, given the way we've set this up, that it *could* be modified to
    # allow for Lmax to be *increased* as necessary, and all of the structures should be
    # "backwards compatible". That said, this has not been tested.
    @classmethod
    def __initTaylor3Dindexing__(cls, Lmax):
        """
        This calls *all* the class methods defined above, and stores them *for the class*.
        This is intended to be done *once*

        :param Lmax: maximum power / orbital angular momentum
        """
        if cls.__INITIALIZED__:
            # we only need initialize our class once!
            return
        cls.Lmax = Lmax
        cls.NYlm, cls.Npower, \
        cls.pow2ind, cls.ind2pow, \
        cls.Ylm2ind, cls.ind2Ylm, \
        cls.powlrange = cls.makeindexPowerYlm(Lmax)
        cls.Ylmpow = cls.makeYlmpow()
        cls.powYlm = cls.makepowYlm()
        cls.Lproj = cls.makeLprojections()
        cls.directmult = cls.makedirectmult()
        cls.powercoeff = cls.makepowercoeff()
        cls.HDF5str = 'coeff.{}.{}'  # needed for addhdf5()
        cls.__internallist__ = ('pow2ind', 'ind2pow', 'Ylm2ind', 'ind2Ylm',
                                'powlrange', 'Ylmpow', 'powYlm',
                                'Lproj', 'directmult', 'powercoeff')
        cls.__INITIALIZED__ = True

    def __init__(self, coefflist=[], Lmax=4, nodeepcopy=False):
        """
        Initializes a Taylor3D object, with coefflist (default = empty)

        :param coefflist: list((n, lmax, powexpansion)). No type checking; default empty
        :param Lmax: maximum power / orbital angular momentum; can be set only once the
            first time a Taylor expansion is constructed, and is set for all objects
        :param nodeepcopy: true if we don't want to copy the matrices on creation of object
            (i.e., deep copy, which is the default) **Note:** deep copy is strongly preferred.
            The *only* real reason to use nodeepcopy is when returning slices / indexing in
            arrays, but even then we have to be careful about doing things like reductions,
            etc., that modify matrices *in place*. We always copy the list, but that
            doesn't make copies of the underlying matrices.
        """
        self.__initTaylor3Dindexing__(Lmax)
        if nodeepcopy:
            self.coefflist = coefflist.copy()
        else:
            self.coefflist = [(n, l, c.copy()) for n, l, c in coefflist]

    def copy(self):
        """Returns a copy of the current expansion"""
        return type(self)(self.coefflist)

    def addhdf5(self, HDF5group):
        """
        Adds an HDF5 representation of object into an HDF5group (needs to already exist).
        Example: if f is an open HDF5, then T3D.addhdf5(f.create_group('T3D')) will
        (1) create the group named 'T3D', and then (2) put the T3D representation in
        that group.

        :param HDF5group: HDF5 group
        """
        HDF5group.attrs['type'] = self.__class__.__name__
        HDF5group.attrs['Lmax'] = self.Lmax
        for (n, l, c) in self.coefflist:
            coeffstr = self.HDF5str.format(n, l)
            HDF5group[coeffstr] = c
            HDF5group[coeffstr].attrs['n'] = n
            HDF5group[coeffstr].attrs['l'] = l

    @classmethod
    def loadhdf5(cls, HDF5group):
        """
        Creates a new T3D from an HDF5 group.

        :param HDFgroup: HDF5 group
        :return T3D: new T3D object
        """
        t3d = cls()  # initialize
        for k, c in HDF5group.items():
            n = HDF5group[k].attrs['n']
            l = HDF5group[k].attrs['l']
            if l > t3d.Lmax or l < 0:
                raise ValueError('HDF5 group data contains illegal l = {} for {}'.format(l, k))
            t3d.coefflist.append((n, l, c.value))
        return t3d

    def dumpinternalsHDF5(self, HDF5group):
        """
        Adds the initialized power expansion internals into an HDF5group--should be stored for a
        sanity check

        :param HDF5group: HDF5 group
        """
        HDF5group.attrs['description'] = 'Internals of PowerExpansion class'
        HDF5group.attrs['Lmax'] = self.Lmax
        HDF5group.attrs['NYlm'] = self.NYlm
        HDF5group.attrs['Npower'] = self.Npower
        for internal in self.__internallist__:
            HDF5group[internal] = getattr(self, internal)

    @classmethod
    def checkinternalsHDF5(cls, HDF5group):
        """
        Reads the power expansion internals into an HDF5group, and performs sanity check

        :param HDF5group: HDF5 group
        """
        if not cls.__INITIALIZED__: raise ValueError('Must initialize first to perform sanity check')
        if HDF5group.attrs['description'] != u'Internals of PowerExpansion class':
            raise ValueError(
                'HDF5 group lacks the attribute "description" which matches "Internals of PowerExpansion class"')
        if HDF5group.attrs['Lmax'] != cls.Lmax: return False
        if HDF5group.attrs['NYlm'] != cls.NYlm: return False
        if HDF5group.attrs['Npower'] != cls.Npower: return False
        for internal in cls.__internallist__:
            if not np.all(HDF5group[internal][:] == getattr(cls, internal)): return False
        return True

    @classmethod
    def zeros(cls, nmin, nmax, shape, dtype=complex):
        """
        Constructs (and returns) a "zero" Taylor expansion with the prescribed shape.
        This will be useful for doing slicing assignments. Because of the manner in
        which slicing works for assignment, we create what looks like a *lot* of
        zeros, by explicitly making the full range of l values.

        :param nmin: minimum value of n
        :param nmax: maximum value of n (inclusive)
        :param shape: shape of matrix, as zeros would expect.
        :return Taylor3D: Taylor3D, with a zero coefficient list
        """
        return cls([(n, l, np.zeros((cls.powlrange[l],) + shape, dtype=dtype))
                    for n in range(nmin, nmax + 1)
                    for l in range(0, cls.Lmax + 1)])

    def __getitem__(self, key):
        """
        Indexes (or even slices) into our Taylor expansion.

        :param key: indices for our Taylor expansion
        :return Taylor3D: Taylor expansion after indexing
        """
        if type(key) is not tuple:
            keyt = (key,)
        else:
            keyt = key
        return type(self)([(n, l, c[(slice(0, None, None),) + keyt]) for n, l, c in self.coefflist], nodeepcopy=True)

    def __setitem__(self, key, value):
        """
        Indexes (or even slices) into our Taylor expansion and "sets"; really only intended to work
        with another Taylor expansion

        :param key: indices for our Taylor expansion
        :param value: assignment value; really, should be
        :return: Taylor expansion after indexing
        """
        if not hasattr(value, "coefflist"):
            raise ValueError("Can only do setitem ([...] = ) with another {} on the rhs".format(type(self)))
        if type(key) is not tuple:
            keyt = (key,)
        else:
            keyt = key
        for nv, lv, cv in value.coefflist:
            matched = False
            for n, l, c in self.coefflist:
                if n == nv and l == lv:
                    matched = True
                    c[(slice(0, None, None),) + keyt] = cv
            if not matched:
                raise ValueError("Attempted to do setitem where the rhs contains terms not present in lhs")

    def __str__(self):
        """Human readable string representation"""
        strrep = ""
        for n, l, coeff in self.coefflist:
            strrep = strrep + "f^({}, {})(u)*(".format(n, l)
            for p in range(self.powlrange[l]):
                if not np.all(np.isclose(coeff[p], 0)):
                    strrep = strrep + "\n{} x^{} y^{} z^{}".format(coeff[p],
                                                                   self.ind2pow[p, 0], self.ind2pow[p, 1],
                                                                   self.ind2pow[p, 2])
            strrep = strrep + " )\n"
        return strrep

    def addterms(self, coefflist):
        """
        Add additional coefficients into our object. No type checking. Only works if
        terms are completely non-overlapping (otherwise, need to use sum).

        :param coefflist: list((n, lmax, powexpansion))
        """
        # getattr is here *in case* someone passes us a Taylor3D type object...
        for coeff in getattr(coefflist, 'coefflist', coefflist):
            if any(coeff[0] == c[0] for c in self.coefflist):
                raise ValueError("Can only use addterms to include new powers; use + instead")
            else:
                self.coefflist.append((coeff[0], coeff[1], coeff[2].copy()))
        self.coefflist.sort(key=self.__sortkey)

    def __call__(self, u, fnu=None):
        """
        Method for evaluating our 3D Taylor expansion. We have two approaches: if we are
        passed a dictionary in fnu that will map (n,l) tuple pairs to either (a) values or
        (b) functions of a single parameter umagn, then we will compute and return the
        function value. Otherwise, we return a dictionary mapping (n,l) tuple pairs into
        values, and leave it at that.

        :param u: three vector to evaluate; may (or may not) be normalized
        :param fnu: dictionary of (n,l): value or function pairs.
        :return value or dictionary: depending on fnu; default is dictionary
        """
        u0, umagn = self.powexp(u)
        if fnu is not None:
            fval = [fnu[(n, l)](umagn) if callable(fnu[(n, l)]) else fnu[(n, l)]
                    for (n, l, coeff) in self.coefflist]
            return sum(fv * np.tensordot(u0[:self.powlrange[l]], coeff, axes=1)
                          for fv, (n, l, coeff) in zip(fval, self.coefflist))
        # otherwise, create a dictionary!
        return {(n, l): np.tensordot(u0[:self.powlrange[l]], coeff, axes=1) for n, l, coeff in self.coefflist}

    def nl(self):
        """
        Returns a list of (n,l) pairs in the coefflist

        :return nl_list: all of the (n,l) pairs that are present in our coefflist
        """
        return sorted([(n, l) for (n, l, coeff) in self.coefflist], key=self.__sortkey)

    @classmethod
    def negcoeff(cls, a):
        """
        Negates a coefficient expansion a

        :param a = list((n, lmax, powexpansion): expansion of function in powers
        :return coefflist: -a
        """
        acoeff = getattr(a, 'coefflist', a)
        nega = []
        for an, almax, apow in acoeff:
            nega.append((an, almax, -apow))
        return nega

    @classmethod
    def scalarproductcoeff(cls, c, a, inplace=False):
        """
        Multiplies an coefficient expansion a by a scalar c

        :param c: scalar *or* dictionary mapping (n,l) to scalars
        :param a = list((n, lmax, powexpansion): expansion of function in powers
        :param inplace: modify a in place?
        :return coefflist: c*a
        """
        acoeff = getattr(a, 'coefflist', a)
        if inplace:
            if isinstance(c, dict):
                for an, almax, apow in acoeff:
                    apow *= c[(an, almax)]
            else:
                for an, almax, apow in acoeff:
                    apow *= c
            ca = a
        else:
            # create new expansion
            ca = []
            if isinstance(c, dict):
                for an, almax, apow in acoeff:
                    ca.append((an, almax, c[(an, almax)] * apow))
            else:
                for an, almax, apow in acoeff:
                    ca.append((an, almax, c * apow))
        return ca

    @classmethod
    def rotatecoeff(cls, a, npowtrans, inplace=False):
        """
        Return a rotated version of the expansion. Needs to use pad to work with reduced representations.

        :param a: coefficiant list
        :param npowtrans: Lmax+1 x Npow x Npow matrix, of [n,oldpow,newpow] corresponding to the rotation
        :return rcoeff: coefficient list, rotated
        """
        acoeff = getattr(a, 'coefflist', a)
        if len(acoeff) == 0: return acoeff
        # needed to make padding easier: we only pad the first axis corresponding to our powers
        padtuple = ((0, 0),) * (len(acoeff[0][2].shape) - 1)
        if not inplace:
            return [(n, n, np.tensordot(npowtrans[n, :cls.powlrange[n], :cls.powlrange[n]],
                                        np.pad(c, ((0, cls.powlrange[n] - cls.powlrange[l]),) + padtuple,
                                               mode='constant'),
                                        axes=(0, 0)))
                    for n, l, c in acoeff]
        else:
            for i, (n, l, c) in enumerate(acoeff):
                acoeff[i] = (n, n, np.tensordot(npowtrans[n, :cls.powlrange[n], :cls.powlrange[n]],
                                                np.pad(c, ((0, cls.powlrange[n] - cls.powlrange[l]),) + padtuple,
                                                       mode='constant'),
                                                axes=(0, 0)))
            return acoeff

    def rotate(self, powtrans):
        """
        Return a rotated version of the expansion.

        :param powtrans: Npow x Npow matrix, of [oldpow,newpow] corresponding to the rotation
        :return rTaylor3D: Taylor expansion, rotated
        """
        return type(self)(self.rotatecoeff(self.coefflist, powtrans))

    def irotate(self, powtrans):
        """
        Rotate in place.

        :param powtrans: Npow x Npow matrix, of [oldpow,newpow] corresponding to the rotation
        :return: self
        """
        self.rotatecoeff(self.coefflist, powtrans, inplace=True)
        return self

    @classmethod
    def sumcoeff(cls, a, b, alpha=1, beta=1, inplace=False):
        """
        Takes Taylor3D expansion a and b, and returns the sum of the expansions.

        :param: a, b = list((n, lmax, powexpansion)
            written as a series of coefficients; n defines the magnitude function, which
            is additive; lmax is the largest cumulative power of coefficients, and
            powexpansion is a numpy array that can multiplied. We assume that a and b
            have consistent shapes throughout--we *do not test this*; runtime will likely
            fail if not true. The entries in the list are *tuples* of n, lmax, pow
        :param alpha, beta:
            optional scalars: c = alpha*a + beta*b; allows for more efficient expansions
        :param inplace: True if the summation should modify a in place
        :return c: coeff of sum of a and b (! NOTE ! does not return the class!)
            sum of a and b
        """
        # a little pythonic magic to work with *either* a list, or an object with a coefflist
        acoeff = getattr(a, 'coefflist', a)  # fallback to a if not there... which assumes it's a list
        bcoeff = getattr(b, 'coefflist', b)  # fallback to b if not there... which assumes it's a list
        if len(bcoeff) == 0: return cls.scalarproductcoeff(alpha, acoeff, inplace)
        if len(acoeff) == 0:
            if not inplace:
                return cls.scalarproductcoeff(beta, bcoeff, inplace)
            else:
                for entry in cls.scalarproductcoeff(beta, bcoeff, inplace=False):
                    acoeff.append(entry)
                return acoeff
        ashape = acoeff[0][2].shape
        bshape = bcoeff[0][2].shape
        if ashape[1:] != bshape[1:]:
            raise TypeError('Unable to add--not compatible')
        # make c = copy of a
        if not inplace:
            c = [(an, almax, alpha * apow) for (an, almax, apow) in acoeff]
        else:
            c = acoeff
        for bn, blmax, bpow in bcoeff:
            # now add it into the list
            cpow = beta * bpow
            matched = False
            for coeffindex, cmatch in enumerate(c):
                if cmatch[0] == bn:
                    matched = True
                    break
            if not matched:
                c.append((bn, blmax, cpow))
            else:
                # a little tricky: we need to *append* to an existing term
                clmax0 = cmatch[1]
                if blmax > clmax0:
                    # need to replace cmatch with a new tuple
                    cpow[:cls.powlrange[clmax0]] += cmatch[2]
                    c[coeffindex] = (bn, blmax, cpow)
                else:
                    # can just append in place: need to be careful, since we have a tuple
                    coeff = cmatch[2]
                    coeff[:cls.powlrange[blmax]] += cpow
        c.sort(key=cls.__sortkey)
        return c

    def __pos__(self):
        """Return +T3D"""
        return self.copy()

    def __neg__(self):
        """Return -T3D"""
        return type(self)(self.negcoeff(self))

    def __add__(self, other):
        """Add a set of Taylor expansions"""
        # if we're passed an array, just take it in stride
        if hasattr(other, 'shape'): other = [(0, 0, other.reshape((1,) + other.shape))]
        return type(self)(self.sumcoeff(self, other))

    def __radd__(self, other):
        """Add a set of Taylor expansions"""
        # note: sum(), without a start value, uses 0, which then will call __radd__:
        if other == 0: return self.copy()
        # if we're passed an array, just take it in stride
        if hasattr(other, 'shape'): other = [(0, 0, other.reshape((1,) + other.shape))]
        return type(self)(self.sumcoeff(self, other))

    def __iadd__(self, other):
        """Add a set of Taylor expansions"""
        # if we're passed an array, just take it in stride
        if hasattr(other, 'shape'): other = [(0, 0, other.reshape((1,) + other.shape))]
        self.sumcoeff(self, other, inplace=True)
        return self

    def __sub__(self, other):
        """Subtract a set of Taylor expansions"""
        # if we're passed an array, just take it in stride
        if hasattr(other, 'shape'): other = [(0, 0, other.reshape((1,) + other.shape))]
        return type(self)(self.sumcoeff(self, other, 1, -1))

    def __rsub__(self, other):
        """Subtract a set of Taylor expansions"""
        # if we're passed an array, just take it in stride
        if hasattr(other, 'shape'): other = [(0, 0, other.reshape((1,) + other.shape))]
        return type(self)(self.sumcoeff(self, other, -1, 1))

    def __isub__(self, other):
        """Subtract a set of Taylor expansions"""
        # if we're passed an array, just take it in stride
        if hasattr(other, 'shape'): other = [(0, 0, other.reshape((1,) + other.shape))]
        self.sumcoeff(self, other, 1, -1, inplace=True)
        return self

    @classmethod
    def tensorproductcoeff(cls, c, a, leftmultiply=True):
        """
        Multiplies an coefficient expansion a by a scalar c

        :param c: array *or* dictionary mapping (n,l) to arrays
        :param a = list((n, lmax, powexpansion): expansion of function in powers
        :param leftmultiply: tensordot(c,a) vs. tensordot(a,c)
        :return coefflist: c.a (or a.c)
        """
        acoeff = getattr(a, 'coefflist', a)
        if isinstance(c, Number) or \
                (isinstance(c, dict) and isinstance(c[(acoeff[0][0], acoeff[0][1])], Number)):
            return cls.scalarproductcoeff(c, a, inplace=False)
        ca = []
        for an, almax, apow in acoeff:
            if isinstance(c, dict):
                cmult = c[(an, almax)]
            else:
                cmult = c
            if leftmultiply:
                # tricky because of layout of apow
                shape = (apow.shape[0],) + cmult.shape[:-1] + apow.shape[2:]
                mat = np.zeros(shape, dtype=complex)
                for p in range(cls.powlrange[almax]):
                    mat[p] = np.tensordot(cmult, apow[p], axes=1)
                ca.append((an, almax, mat))
            else:
                ca.append((an, almax, np.tensordot(apow, cmult, axes=(-1, 0))))
        return ca

    def ldot(self, c):
        """Returns :math:`c\\cdot self`"""
        return type(self)(self.tensorproductcoeff(c, self))

    def rdot(self, c):
        """Returns :math:`self\\cdot c`"""
        return type(self)(self.tensorproductcoeff(c, self, leftmultiply=False))

    def ildot(self, c):
        """Computes :math:`c\\cdot self` in place"""
        self.coefflist = self.tensorproductcoeff(c, self)
        return self

    def irdot(self, c):
        """Computes :math:`self\\cdot c` in place"""
        self.coefflist = self.tensorproductcoeff(c, self, leftmultiply=False)
        return self

    @classmethod
    def coeffproductcoeff(cls, a, b):
        """
        Takes a direction expansion a and b, and returns the product expansion.

        :param a: list((n, lmax, powexpansion)
        :param b: list((n, lmax, powexpansion)
            written as a series of coefficients; n defines the magnitude function, which
            is additive; lmax is the largest cumulative power of coefficients, and
            powexpansion is a numpy array that can multiplied. We assume that a and b
            have consistent shapes throughout--we *do not test this*; runtime will likely
            fail if not true. The entries in the list are *tuples* of n, lmax, pow
        :return c: list((n, lmax, powexpansion)), product of ``a`` and ``b``
        """
        # a little pythonic magic to work with *either* a list, or an object with a coefflist
        acoeff = getattr(a, 'coefflist', a)  # fallback to a if not there... which assumes it's a list
        bcoeff = getattr(b, 'coefflist', b)  # fallback to b if not there... which assumes it's a list
        if len(acoeff) == 0: return acoeff  # 0*anything == 0
        if len(bcoeff) == 0: return bcoeff  # anything*0 == 0
        c = []
        ashape = acoeff[0][2].shape
        bshape = bcoeff[0][2].shape
        scalarmult = False
        if len(ashape) == 1:
            cshape = bshape[1:]
            scalarmult = True
        elif len(bshape) == 1:
            cshape = ashape[1:]
            scalarmult = True
        else:
            if ashape[-1] != bshape[1]:
                raise TypeError('Unable to multiply--not compatible')
            cshape = ashape[1:-1] + bshape[2:]  # weird piece of python to find the shape of a*b
        for an, almax, apow in acoeff:
            for bn, blmax, bpow in bcoeff:
                cn = an + bn
                clmax = almax + blmax
                if clmax > cls.Lmax:
                    # in theory... we should warn the user here
                    clmax = cls.Lmax
                # construct the expansion
                cpow = np.zeros((cls.powlrange[clmax],) + cshape, dtype=complex)
                for pa in range(cls.powlrange[almax]):
                    for pb in range(cls.powlrange[blmax]):
                        if scalarmult:
                            cpow[cls.directmult[pa, pb]] += apow[pa] * bpow[pb]
                        else:
                            cpow[cls.directmult[pa, pb]] += np.tensordot(apow[pa], bpow[pb], axes=1)
                # now add it into the list
                matched = False
                for coeffindex, cmatch in enumerate(c):
                    if cmatch[0] == cn:
                        matched = True
                        break
                if not matched:
                    c.append((cn, clmax, cpow))
                else:
                    # a little tricky: we need to *append* to an existing term
                    clmax0 = cmatch[1]
                    if clmax > clmax0:
                        # need to replace cmatch with a new tuple
                        cpow[:cls.powlrange[clmax0]] += cmatch[2]
                        c[coeffindex] = (cn, clmax, cpow)
                    else:
                        # can just append in place: need to be careful, since we have a tuple
                        coeff = cmatch[2]
                        coeff[:cls.powlrange[clmax]] += cpow
        c.sort(key=cls.__sortkey)
        return c

    def __mul__(self, other):
        """
        Multiply our expansion

        :param other:
        :return Taylor3D: expansion of product
        """
        if isinstance(other, Number) or hasattr(other, 'shape') or isinstance(other, dict):
            coeff = self.scalarproductcoeff(other, self)
        else:
            coeff = self.coeffproductcoeff(self, other)
        return type(self)(coeff)

    def __rmul__(self, other):
        """
        Multiply our expansion

        :param other:
        :return Taylor3D: expansion of product
        """
        if isinstance(other, Number) or hasattr(other, 'shape') or isinstance(other, dict):
            coeff = self.scalarproductcoeff(other, self)
        else:
            coeff = self.coeffproductcoeff(self, other)
        return type(self)(coeff)

    @classmethod
    def inversecoeff(cls, a, Nmax=0):
        """
        Takes a direction expansion , and returns the inversion expansion (approximated
        based on the Taylor expansion of :math:`1/(1-x) = \\sum_{i=0}^{\\infty} x^i`, or
        :math:`(A + B)^{-1} = ((1+BA^{-1})A)^{-1} = A^{-1}(1-(-BA{^1}))^{-1} = A^{-1} \\sum_{i=0} (-BA^{-1})^i`

        NOTE: assumes SMALLEST n coefficient is the leading order; only works if that
        coefficient is also isotropic (l=0). Otherwise, raises an error.
        NOTE: there is no sanity check on whether Nmax is reasonable given the expansion
        and Lmax values; *caveat emptor*.

        :param a: = list((n, lmax, powexpansion)
            written as a series of coefficients; n defines the magnitude function, which
            is additive; lmax is the largest cumulative power of coefficients, and
            powexpansion is a numpy array that can multiplied. We assume that a and b
            have consistent shapes throughout--we *do not test this*; runtime will likely
            fail if not true. The entries in the list are *tuples* of n, lmax, pow
        :param Nmax: maximum remaining n value in expansion. Default value of 0 means
            up to a discontinuity correction in an inversion, but higher (or lower) values are
            possible.
        :return c: list((n, lmax, powexpansion)), inverse of a
        """
        # a little pythonic magic to work with *either* a list, or an object with a coefflist
        acoeff = sorted(getattr(a, 'coefflist', a),
                        key=cls.__sortkey)  # fallback to a if not there... which assumes it's a list
        lead = acoeff[0]
        if lead[1] != 0:
            raise ValueError('Cannot invert expansion: leading-order term {} has l={}>0'.format(lead[0], lead[1]))
        if len(lead[2].shape) == 1:
            leadinvmat = 1 / lead[2][0]
        else:
            leadinvmat = np.linalg.inv(lead[2][0])
        leadinvpow = -lead[0]
        c = [(leadinvpow, 0, leadinvmat.copy().reshape(lead[2].shape))]
        if len(acoeff) == 1:  # trivial case
            return c
        else:
            if leadinvpow + acoeff[1][0] <= 0:
                raise ValueError('Cannot invert expansion: second term has same power as leading-order term?')
        tail = [(n + leadinvpow, l, -coeff)
                for n, l, coeff in cls.tensorproductcoeff(leadinvmat, acoeff[1:], leftmultiply=True)]
        # now we can calculate the number of terms necessary:
        # leadinvpow + n*(tail[0][0]) >= Nmax
        Nseries = (Nmax - leadinvpow) // tail[0][0]
        # prime the pump: leadinvmat = A^-1, tail = -A^-1 B
        cls.sumcoeff(c, [(n + leadinvpow, l, coeff)
                         for n, l, coeff in
                         cls.tensorproductcoeff(leadinvmat, tail, leftmultiply=False)
                         if n + leadinvpow <= Nmax],
                     inplace=True)
        tailn = [(n, l, coeff.copy()) for n, l, coeff in tail]
        for npower in range(2, Nseries + 1):
            # trim out the powers that are too large once they become too large:
            tailn = [(n, l, coeff) for n, l, coeff in cls.coeffproductcoeff(tailn, tail)
                     if n + leadinvpow <= Nmax]
            cls.sumcoeff(c, [(n + leadinvpow, l, coeff)
                             for n, l, coeff in
                             cls.tensorproductcoeff(leadinvmat, tailn, leftmultiply=False)],
                         inplace=True)
        return c

    def inv(self, Nmax=0):
        """
        Return the inverse of the expansion, up to order Nmax

        :param Nmax: maximum order in the inverse expansion
        :return Taylor3D^-1: Taylor series of inverse
        """
        return type(self)(self.inversecoeff(self, Nmax))

    @classmethod
    def reducecoeff(cls, a, inplace=False, atol=1e-10):
        """
        Projects coefficients through Ylm space, then eliminates any zero contributions
        (including possible reduction in l values, too).

        :param a: list((n, lmax, powexpansion), expansion of function in powers
        :param inplace: modify a in place?
        :return coefflist: a
        """
        acoeff = getattr(a, 'coefflist', a)
        if inplace:
            ra = acoeff
        else:
            ra = [(n, l, c.copy()) for (n, l, c) in acoeff]
        projector = cls.Lproj[-1]
        dellist = []
        for coeffindex, (n, l, c) in enumerate(ra):
            # first, project
            c = np.tensordot(projector[:cls.powlrange[l], :cls.powlrange[l]], c, axes=1)
            # print(c)
            # now, systematically attempt to reduce the l value
            if np.allclose(c, 0, atol=atol):
                # occasionally, it gets reduced to zero:
                dellist.append(coeffindex)
            else:
                # then we have something to look at... systematically attempt to drop l:
                # check in blocks
                for lmin in range(l, -1, -1):
                    if not np.allclose(c[cls.powlrange[lmin - 1]:cls.powlrange[lmin]], 0, atol=atol):
                        break
                # reduce! Note: we do this *every time* because c is the projected version of our coeff.
                ra[coeffindex] = (n, lmin, c[:cls.powlrange[lmin]].copy())
        # finally, let's deal with our delete list; do this by popping, and in reverse index order
        dellist.reverse()
        for ind in dellist:
            ra.pop(ind)
        return ra

    @classmethod
    def collectcoeff(cls, a, inplace=False, atol=1e-10):
        """
        Collects coefficients: sums up all the common n values. Best to be done *after*
        reduce is called.

        :param a: list((n, lmax, powexpansion), expansion of function in powers
        :param inplace: modify a in place?
        :return coefflist: a
        """
        acoeff = getattr(a, 'coefflist', a)
        if inplace:
            ca = acoeff
        else:
            ca = [(n, l, c.copy()) for (n, l, c) in acoeff]
        # so: the sort is such that all of the common n values are in ascending order
        # *and* ascending l order. Makes collecting very easy:
        ca.sort(key=cls.__sortkey)
        projector = cls.Lproj[-1]
        dellist = []  # indices to be eliminated
        for coeffindex, (n, l, c) in enumerate(ca):
            # first, project
            c = np.tensordot(projector[:cls.powlrange[l], :cls.powlrange[l]], c, axes=1)
            if np.allclose(c, 0, atol=atol):
                # if we have zero coefficients, remove from the list
                dellist.append(coeffindex)
            else:
                if coeffindex < len(ca) - 1:
                    if ca[coeffindex + 1][0] == n:
                        # same n, so collect:
                        ca[coeffindex + 1][2][:cls.powlrange[l]] += c
                        dellist.append(coeffindex)
                    else:
                        ca[coeffindex] = (n, l, c.copy())
        # finally, let's deal with our delete list; do this by popping, and in reverse index order
        dellist.reverse()
        for ind in dellist:
            ca.pop(ind)
        return ca

    def reduce(self):
        """
        Reduce the coefficients: eliminate any n that has zero coefficients, collect all of
        the same values of n together. Done in place.
        """
        self.reducecoeff(self.coefflist, inplace=True)
        self.collectcoeff(self.coefflist, inplace=True)
        return self

    @classmethod
    def separatecoeff(cls, a, inplace=False, atol=1e-10):
        """
        Projects coefficients through Ylm space, one by one. Assumes they've already been
        reduced and collected first; if not, could lead to duplicated (n,l) entries in list, which
        is inefficient (should still *evaluate* the same, just with extra steps). After this,
        each (n,l) term *only* contains terms equal to l, rather than terms <= l.

        :param a: list((n, lmax, powexpansion), expansion of function in powers
        :param inplace: modify a in place?
        :return coefflist: a
        """
        acoeff = getattr(a, 'coefflist', a)
        if inplace:
            sa = acoeff
        else:
            sa = [(n, l, c.copy()) for (n, l, c) in acoeff]
        dellist = []
        # we're going to append to sa, so... if you DON'T do this, you get an infinite loop:
        for coeffindex, (n, l, c) in enumerate(sa[:len(sa)]):
            # first, project
            for l0 in range(l):
                cl0 = np.tensordot(cls.Lproj[l0][:cls.powlrange[l], :cls.powlrange[l]],
                                   c, axes=1)[:cls.powlrange[l0]]
                if not np.allclose(cl0, 0, atol=atol):
                    sa.append((n, l0, cl0))
            c = np.tensordot(cls.Lproj[l][:cls.powlrange[l], :cls.powlrange[l]], c, axes=1)
            if not np.allclose(c, 0, atol=atol):
                sa[coeffindex] = (n, l, c)  # this *should not be zero* but just in case...
            else:
                dellist.append(coeffindex)
        # finally, let's deal with our delete list; do this by popping, and in reverse index order
        dellist.reverse()
        for ind in dellist:
            sa.pop(ind)
        sa.sort(key=cls.__sortkey)
        return sa

    def separate(self):
        """Separate out the coefficients into (n,l) terms where *only* l contributions appear in each."""
        self.separatecoeff(self.coefflist, inplace=True)
        return self

    @classmethod
    def truncatecoeff(cls, a, Nmax, inplace=False):
        """
        Remove the coefficients above a given Nmax; normally returns a new object

        :param Nmax: maximum coefficient to include
        :param a: list((n, lmax, powexpansion), expansion of function in powers
        :param inplace: do it in place?
        """
        acoeff = getattr(a, 'coefflist', a)
        if inplace:
            for ind in range(len(acoeff) - 1, -1, -1):
                if acoeff[ind][0] > Nmax:
                    acoeff.pop(ind)
            return acoeff
        else:
            return [(n, l, c.copy()) for (n, l, c) in acoeff if n <= Nmax]

    def truncate(self, Nmax, inplace=False):
        """
        Remove the coefficients above a given Nmax; normally returns a new object

        :param Nmax: maximum coefficient to include
        :param inplace: do it in place?
        """
        if inplace:
            self.truncatecoeff(self.coefflist, Nmax, inplace)
            return self
        else:
            return type(self)(self.truncatecoeff(self.coefflist, Nmax))


class Taylor2D(Taylor3D):
    """
    Class that stores a Taylor expansion of a function in 2D, and defines some arithmetic
    """

    # As much as possible, we inherit from the 2D code; below are the changes we make

    @staticmethod
    def makeindexPowerFC(Lmax):
        """
        Analyzes the Fourier coefficients and powers for a given Lmax; returns a
        series of index functions.

        :param Lmax: maximum l value to consider; equal to the sum of powers
        :return NFC: number of Fourier coefficients
        :return Npower: number of power coefficients
        :return pow2ind[n1][n2]: powers to index
        :return ind2pow[n]: powers for a given index
        :return FC2ind[l]: (l) to index
        :return ind2FC[lind]: (l) for a given index
        :return powlrange[l]: upper limit of power indices for a given l value; note: [-1] = 0
        """
        # first, the counts
        NFC = 2*Lmax + 1
        Npower = (Lmax+1)*(Lmax+2)//2
        # indexing arrays
        powlrange = np.zeros(Lmax + 2, dtype=int)
        powlrange[-1] = 0
        pow2ind = -np.ones((Lmax + 1, Lmax + 1), dtype=int)
        ind2pow = np.zeros((Npower, 2), dtype=int)
        FC2ind = np.zeros(NFC, dtype=int)
        ind2FC = np.zeros(NFC, dtype=int)
        # powers first; these are ordered by increasing l = n1+n2
        ind = 0
        for l in range(Lmax + 1):
            for n1 in range(l + 1):
                n2 = l-n1
                pow2ind[n1, n2] = ind
                ind2pow[ind,0], ind2pow[ind, 1] = n1, n2
                ind += 1
            powlrange[l] = ind
        # next, FC values
        ind = 0
        for l in range(-Lmax,Lmax+1):
            FC2ind[l] = ind
            ind2FC[ind] = l
            ind += 1
        return NFC, Npower, pow2ind, ind2pow, FC2ind, ind2FC, powlrange

    @classmethod
    def makeFCpow(cls):
        """
        Construct the expansion of the FC's in powers of x,y. Done via brute force.

        :return FCpow[l, p]: expansion of each FC in powers
        """
        FCpow = np.zeros((cls.NFC, cls.Npower), dtype=complex)
        for lind, l in enumerate(cls.ind2FC):
            labs = abs(l)
            lsign = 1j if l >= 0 else -1j
            for k in range(labs + 1):
                nmind = cls.pow2ind[k, labs - k]
                FCpow[lind, nmind] = comb(labs, k) * (lsign) ** (labs - k)
        return FCpow

    @classmethod
    def makepowFC(cls):
        """
        Construct the expansion of the powers in FC's. Done using brute force

        :return powFC[p, l]: expansion of powers in FC; uses indexing scheme above
        """
        powFC = np.zeros((cls.Npower, cls.NFC), dtype=complex)
        for nind, (n,m) in enumerate(cls.ind2pow):
            pre = (-1j) ** m / (2 ** (n + m))
            for j in range(n + 1):
                for k in range(m + 1):
                    l = 2 * j - n + 2 * k - m
                    powFC[nind, cls.FC2ind[l]] += pre * comb(n, j) * comb(m, k) * \
                                                  (-1) ** (m - k)
        return powFC

    @classmethod
    def makeLprojections(cls):
        """
        Constructs a series of projection matrices for each l component in our power series

        :return: projL[l][p][p']
            projection of powers containing *only* l component.
            -1 component = sum(l=0..Lmax, projL[l]) = simplification projection
        """
        projL = np.zeros((cls.Lmax + 2, cls.Npower, cls.Npower))
        for l in range(cls.Lmax + 1):
            lp, lm = cls.FC2ind[l], cls.FC2ind[-l]
            if l == 0:
                projL[l] = np.outer(cls.FCpow[lp, :], cls.powFC[:, lp]).real
            else:
                projL[l] = np.outer(cls.FCpow[lp, :], cls.powFC[:, lp]).real
                projL[l] += np.outer(cls.FCpow[lm, :], cls.powFC[:, lm]).real
        projL[-1] = np.sum(projL, axis=0)
        return projL

    @classmethod
    def makedirectmult(cls):
        """
        :return direcmult[p][p']: index that corresponds to the multiplication of power indices p and p'
        """
        directmult = -np.ones((cls.Npower, cls.Npower), dtype=int)
        for (p0, p1) in ((p0, p1) for p0 in range(cls.Npower) for p1 in range(cls.Npower)):
            nsum = cls.ind2pow[p0] + cls.ind2pow[p1]
            if sum(nsum) <= cls.Lmax:
                directmult[p0, p1] = cls.pow2ind[nsum[0], nsum[1]]
        return directmult

    @classmethod
    def powexp(cls, u, normalize=True):
        """
        Given a vector u, normalize it and return the power expansion of uvec

        :param u[2]: vector to apply
        :param normalize: do we normalize u first?
        :return upow[Npower]: ux uy uz products of powers
        :return umagn: magnitude of u (if normalized)
        """
        umagn = np.sqrt(np.dot(u, u))
        upow = np.zeros(cls.Npower)
        if umagn < 1e-8:
            upow[cls.pow2ind[0, 0]] = 1.
            umagn = 0.
        else:
            u0 = u.copy()
            if normalize: u0 /= umagn
            xy = np.ones((cls.Lmax + 1, 2))
            for n in range(1, cls.Lmax + 1):
                xy[n, :] = xy[n - 1, :] * u0[:]
            for n0, n1 in ((n0, n1) for n0 in range(cls.Lmax + 1)
                           for n1 in range(cls.Lmax + 1)
                           if n0 + n1 <= cls.Lmax):
                upow[cls.pow2ind[n0, n1]] = xy[n0, 0] * xy[n1, 1]
        if normalize:
            return upow, umagn
        else:
            return upow

    @classmethod
    def makepowercoeff(cls):
        """
        Make our power coefficients for our construct expansion method

        :return powercoeff[n][p]: vector we multiply by our power expansion to get the n'th coefficients
        """
        powercoeff = np.zeros((cls.Lmax + 1, cls.Npower))
        for n0 in range(cls.Lmax + 1):
            for n1 in range(cls.Lmax + 1):
                n = n0 + n1
                if n <= cls.Lmax:
                    powercoeff[n, cls.pow2ind[n0, n1]] = \
                        factorial(n, True) / (factorial(n0, True) * factorial(n1, True))
        return powercoeff

    @classmethod
    def rotatedirections(cls, qptrans):
        """
        Takes a transformation matrix qptrans, where q[i] = sum_j qptrans[i][j] p[j], and
        returns the Npow x Npow transformation matrix for the new components in terms of
        the old.
        NOTE: This is more complex than one might first realize. If we only work with cases
        where all of the entries for a given power n have those same n (that is, not reduced),
        then this is straightforward. However, we run into problems with *reductions*: e.g.,
        for n=2, the power :math:`x^0 y^0 z^0` is, in reality, :math:`x^2+y^2+z^2`, and hence
        *it must be transformed* because we allow non-orthogonal transformation matrices.

        :param qptrans: 3x3 matrix
        :return npowtrans: [Lmax +1][Npow][Npow] transformation matrix [n][original pow][new pow]
            for each n from 0 up to Lmax
        """
        powtrans = np.zeros((cls.Npower, cls.Npower))
        # l = 0 case
        powtrans[0, 0] = 1
        # single q value cases
        for i in range(2):
            qi_pow = cls.powexp(qptrans[i, :], normalize=False)
            for n in range(1, cls.Lmax + 1):
                powtrans[cls.pow2ind[(0,) * i + (n,) + (0,) * (1 - i)], :] = cls.powercoeff[n] * qi_pow
        # pairs of q cases: we get qi^ni qj^nj by direct multiplication
        # triplet is done inside the loop: q1^n1 q2^n2 q3^n3 = (q1^n1 q2^n2) (q3^n3)
        for ni in range(1, cls.Lmax + 1):
            powi = cls.pow2ind[ni,0]
            for nj in range(1, cls.Lmax + 1 - ni):
                powj = cls.pow2ind[0,nj]
                powij = cls.pow2ind[ni,nj]
                # multiply the pair!
                for pi in range(cls.powlrange[ni - 1], cls.powlrange[ni]):
                    for pj in range(cls.powlrange[nj - 1], cls.powlrange[nj]):
                        powtrans[powij, cls.directmult[pi, pj]] += powtrans[powi, pi] * powtrans[powj, pj]
        npowtrans = np.zeros((cls.Lmax + 1, cls.Npower, cls.Npower))
        for n in range(cls.Lmax + 1):
            prange = slice(cls.powlrange[n - 1], cls.powlrange[n])
            npowtrans[n, prange, prange] = powtrans[prange, prange]
            # now, work on lower values (n-2, n-4, ...)
            for m in range(n - 2, -1, -2):
                # powers that sum up to m:
                for tup in [(n0, m- n0) for n0 in range(m + 1)]:
                    npowtrans[n, cls.pow2ind[tup], :] = npowtrans[n, cls.pow2ind[tup[0] + 2, tup[1]], :] + \
                                                        npowtrans[n, cls.pow2ind[tup[0], tup[1] + 2], :]
        return npowtrans

    # for sorting our coefficient lists:
    @classmethod
    def __sortkey(cls, entry):
        return (entry[0] + entry[1] / (cls.Lmax + 1))

    __INITIALIZED__ = False

    # these are all *class* parameters, not object parameters: they are computed
    # and defined once for the entire class. It means that once, in your code, you *choose*
    # a value for Lmax, you are stuck with it. This is a choice: it makes compatibility between
    # the expansions easy, for a minor loss in flexibility.
    # Note: I believe, given the way we've set this up, that it *could* be modified to
    # allow for Lmax to be *increased* as necessary, and all of the structures should be
    # "backwards compatible". That said, this has not been tested.
    @classmethod
    def __initTaylor2Dindexing__(cls, Lmax):
        """
        This calls *all* the class methods defined above, and stores them *for the class*.
        This is intended to be done *once*

        :param Lmax: maximum power / orbital angular momentum
        """
        if cls.__INITIALIZED__:
            # we only need initialize our class once!
            return
        cls.Lmax = Lmax
        cls.NFC, cls.Npower, \
        cls.pow2ind, cls.ind2pow, \
        cls.FC2ind, cls.ind2FC, \
        cls.powlrange = cls.makeindexPowerFC(Lmax)
        cls.FCpow = cls.makeFCpow()
        cls.powFC = cls.makepowFC()
        cls.Lproj = cls.makeLprojections()
        cls.directmult = cls.makedirectmult()
        cls.powercoeff = cls.makepowercoeff()
        cls.HDF5str = 'coeff.{}.{}'  # needed for addhdf5()
        cls.__internallist__ = ('pow2ind', 'ind2pow', 'FC2ind', 'ind2FC',
                                'powlrange', 'FCpow', 'powFC',
                                'Lproj', 'directmult', 'powercoeff')
        cls.__INITIALIZED__ = True

    def __init__(self, coefflist=[], Lmax=4, nodeepcopy=False):
        """
        Initializes a Taylor3D object, with coefflist (default = empty)

        :param coefflist: list((n, lmax, powexpansion)). No type checking; default empty
        :param Lmax: maximum power / orbital angular momentum; can be set only once the
            first time a Taylor expansion is constructed, and is set for all objects
        :param nodeepcopy: true if we don't want to copy the matrices on creation of object
            (i.e., deep copy, which is the default) **Note:** deep copy is strongly preferred.
            The *only* real reason to use nodeepcopy is when returning slices / indexing in
            arrays, but even then we have to be careful about doing things like reductions,
            etc., that modify matrices *in place*. We always copy the list, but that
            doesn't make copies of the underlying matrices.
        """
        self.__initTaylor2Dindexing__(Lmax)
        if nodeepcopy:
            self.coefflist = coefflist.copy()
        else:
            self.coefflist = [(n, l, c.copy()) for n, l, c in coefflist]

    def dumpinternalsHDF5(self, HDF5group):
        """
        Adds the initialized power expansion internals into an HDF5group--should be stored for a
        sanity check

        :param HDF5group: HDF5 group
        """
        HDF5group.attrs['description'] = 'Internals of PowerExpansion class'
        HDF5group.attrs['Lmax'] = self.Lmax
        HDF5group.attrs['NFC'] = self.NFC
        HDF5group.attrs['Npower'] = self.Npower
        for internal in self.__internallist__:
            HDF5group[internal] = getattr(self, internal)

    @classmethod
    def checkinternalsHDF5(cls, HDF5group):
        """
        Reads the power expansion internals into an HDF5group, and performs sanity check

        :param HDF5group: HDF5 group
        """
        if not cls.__INITIALIZED__: raise ValueError('Must initialize first to perform sanity check')
        if HDF5group.attrs['description'] != u'Internals of PowerExpansion class':
            raise ValueError(
                'HDF5 group lacks the attribute "description" which matches "Internals of PowerExpansion class"')
        if HDF5group.attrs['Lmax'] != cls.Lmax: return False
        if HDF5group.attrs['NFC'] != cls.NFC: return False
        if HDF5group.attrs['Npower'] != cls.Npower: return False
        for internal in cls.__internallist__:
            if not np.all(HDF5group[internal][:] == getattr(cls, internal)): return False
        return True

    def __str__(self):
        """Human readable string representation"""
        strrep = ""
        for n, l, coeff in self.coefflist:
            strrep = strrep + "f^({}, {})(u)*(".format(n, l)
            for p in range(self.powlrange[l]):
                if not np.all(np.isclose(coeff[p], 0)):
                    strrep = strrep + "\n{} x^{} y^{}".format(coeff[p],
                                                              self.ind2pow[p, 0],
                                                              self.ind2pow[p, 1])
            strrep = strrep + " )\n"
        return strrep

