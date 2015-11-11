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
from scipy.special import factorial

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
        :return powlrange[l]: upper limit of power indices for a given l value
        """
        # first, the counts
        NYlm = (Lmax+1)**2
        Npower = NYlm + ((Lmax+1)*Lmax*(Lmax-1))//6
        # indexing arrays
        powlrange = np.zeros(Lmax+1, dtype=int)
        pow2ind = -np.ones((Lmax+1,Lmax+1,Lmax+1), dtype=int)
        ind2pow = np.zeros((Npower, 3), dtype=int)
        Ylm2ind = -np.ones((Lmax+1, 2*Lmax+1), dtype=int)
        ind2Ylm = np.zeros((NYlm, 2), dtype=int)
        # powers first; these are ordered by increasing l = n1+n2+n3
        ind = 0
        for l in range(Lmax+1):
            for n1 in range(l+1):
                for n2 in range(l+1-n1):
                    n3 = l-n1-n2
                    pow2ind[n1][n2][n3] = ind
                    ind2pow[ind][0], ind2pow[ind][1], ind2pow[ind][2] = n1,n2,n3
                    ind += 1
            powlrange[l] = ind
        # next, Ylm values
        ind = 0
        for l in range(Lmax+1):
            for m in range(-l, l+1):
                Ylm2ind[l][m] = ind
                ind2Ylm[ind][0], ind2Ylm[ind][1] = l,m
                ind += 1
        return NYlm, Npower, pow2ind, ind2pow, Ylm2ind, ind2Ylm, powlrange

    @classmethod
    def makeYlmpow(cls):
        """
        Construct the expansion of the Ylm's in powers of x,y,z. Done via brute force.
        :return Ylmpow[lm, p]: expansion of each Ylm in powers
        """
        Ylmpow = np.zeros((cls.NYlm, cls.Npower), dtype=complex)
        for l in range(cls.Lmax+1):
            # do the positive m first; then easily swap to get the negative m
            for m in range(l+1):
                ind = cls.Ylm2ind[l][m]
                pre = (-1)**m * np.sqrt((2*l+1)*factorial(l-m,True)/
                                        (4*np.pi*factorial(l+m,True)))
                for k in range((l+m+1)//2, l+1):
                    zz = (-1)**(l-k) * factorial(2*k,True)/\
                         (2**l*factorial(2*k-l-m,True)*factorial(k,True)*factorial(l-k,True))
                    for j in range(m+1):
                        xy = factorial(m,True)/(factorial(j,True)*factorial(m-j,True))
                        Ylmpow[ind][cls.pow2ind[j][m-j][2*k-l-m]] = pre*zz*xy*(1.j)**(m-j)
            for m in range(-l,0):
                ind = cls.Ylm2ind[l][m]
                indpos = cls.Ylm2ind[l][-m]
                for p in range(cls.Npower):
                    Ylmpow[ind][p] = (-1)**(-m) * Ylmpow[indpos][p].conjugate()
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
        Cp = np.zeros((cls.Lmax, 2*cls.Lmax-1))
        Cm = np.zeros((cls.Lmax, 2*cls.Lmax-1))
        Sp = np.zeros((cls.Lmax, 2*cls.Lmax-1))
        Sm = np.zeros((cls.Lmax, 2*cls.Lmax-1))
        # because this is for our recursion relations, we only need to work to Lmax-1 !
        for l,m in ((l,m) for l in range(cls.Lmax) for m in range(-l,l+1)):
            Cp[l][m] = np.sqrt((l-m+1)*(l+m+1)/((2*l+1)*(2*l+3)))
            Sp[l][m] = 0.5*np.sqrt((l+m+1)*(l+m+2)/((2*l+1)*(2*l+3)))
            if l>0: # and -l < m < l:
                Cm[l][m] = np.sqrt((l-m)*(l+m)/((2*l-1)*(2*l+1)))
                Sm[l][m] = 0.5*np.sqrt((l-m)*(l-m-1)/((2*l-1)*(2*l+1)))

        # first, prime the pump with 1
        powYlm[cls.pow2ind[0][0][0]][cls.Ylm2ind[0][0]] = np.sqrt(4*np.pi)
        for n0,n1,n2 in ((n0,n1,n2) for n0 in range(cls.Lmax+1)
                         for n1 in range(cls.Lmax+1)
                         for n2 in range(cls.Lmax+1)
                         if 0 < n0+n1+n2 <= cls.Lmax):
            ind = cls.pow2ind[n0][n1][n2]
            lmax = n0+n1+n2
            if n2>0:
                # we can recurse up from n0, n1, n2-1
                indlow = cls.pow2ind[n0][n1][n2-1]
                for l,m in ((l,m) for l in range(lmax) for m in range(-l,l+1)):
                    plm = powYlm[indlow][cls.Ylm2ind[l][m]]
                    powYlm[ind][cls.Ylm2ind[l+1][m]] += Cp[l][m]*plm
                    if l>0 and -l < m < l:
                        powYlm[ind][cls.Ylm2ind[l-1][m]] += Cm[l][m]*plm
            elif n1>0:
                # we can recurse up from n0, n1-1, n2
                indlow = cls.pow2ind[n0][n1-1][n2]
                for l,m in ((l,m) for l in range(lmax) for m in range(-l,l+1)):
                    plm = powYlm[indlow][cls.Ylm2ind[l][m]]
                    powYlm[ind][cls.Ylm2ind[l+1][m+1]] += 1.j*Sp[l][m]*plm
                    powYlm[ind][cls.Ylm2ind[l+1][m-1]] += 1.j*Sp[l][-m]*plm
                    # if l>0:
                    if m < l-1:
                        powYlm[ind][cls.Ylm2ind[l-1][m+1]] += -1.j*Sm[l][m]*plm
                    if m > -l+1:
                        powYlm[ind][cls.Ylm2ind[l-1][m-1]] += -1.j*Sm[l][-m]*plm
            elif n0>0:
                # we can recurse up from n0-1, n1, n2
                indlow = cls.pow2ind[n0-1][n1][n2]
                for l,m in ((l,m) for l in range(lmax) for m in range(-l,l+1)):
                    plm = powYlm[indlow][cls.Ylm2ind[l][m]]
                    powYlm[ind][cls.Ylm2ind[l+1][m+1]] += -Sp[l][m]*plm
                    powYlm[ind][cls.Ylm2ind[l+1][m-1]] += Sp[l][-m]*plm
                    # if l>0:
                    if m < l-1:
                        powYlm[ind][cls.Ylm2ind[l-1][m+1]] += Sm[l][m]*plm
                    if m > -l+1:
                        powYlm[ind][cls.Ylm2ind[l-1][m-1]] += -Sm[l][-m]*plm
        return powYlm

    @classmethod
    def makeLprojections(cls):
        """
        Constructs a series of projection matrices for each l component in our power series
        :return: projL[l][p][p']
            projection of powers containing *only* l component.
            -1 component = sum(l=0..Lmax, projL[l]) = simplification projection
        """
        projL = np.zeros((cls.Lmax+2, cls.Npower, cls.Npower))
        projLYlm = np.zeros((cls.Lmax+2, cls.NYlm, cls.NYlm), dtype=complex)
        for l,m in ((l,m) for l in range(0,cls.Lmax+1) for m in range(-l, l+1)):
            lm = cls.Ylm2ind[l][m]
            projLYlm[l][lm][lm] = 1. # l,m is part of l
            projLYlm[-1][lm][lm] = 1. # all part of the sum
        for l in range(cls.Lmax+2):
            projL[l] = np.dot(cls.powYlm, np.dot(projLYlm[l], cls.Ylmpow)).real
        return projL

    @classmethod
    def makedirectmult(cls):
        """
        :return direcmult[p][p']: index that corresponds to the multiplication of power indices p and p'
        """
        directmult = np.zeros((cls.Npower, cls.Npower), dtype=int)
        for (p0,p1) in ((p0,p1) for p0 in range(cls.Npower) for p1 in range(cls.Npower)):
            nsum = cls.ind2pow[p0] + cls.ind2pow[p1]
            if sum(nsum) <= cls.Lmax:
                directmult[p0][p1] = cls.pow2ind[nsum[0]][nsum[1]][nsum[2]]
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
            upow[cls.pow2ind[0][0][0]] = 1.
            umagn = 0.
        else:
            u0 = u.copy()
            if normalize: u0 /= umagn
            xyz = np.ones((cls.Lmax+1,3))
            for n in range(1,cls.Lmax+1):
                xyz[n][:] = xyz[n-1][:]*u0[:]
            for n0,n1,n2 in ((n0,n1,n2) for n0 in range(cls.Lmax+1)
                             for n1 in range(cls.Lmax+1)
                             for n2 in range(cls.Lmax+1)
                             if n0+n1+n2 <= cls.Lmax):
                upow[cls.pow2ind[n0][n1][n2]] = xyz[n0][0]*xyz[n1][1]*xyz[n2][2]
        if normalize: return upow, umagn
        else: return upow

    @classmethod
    def makepowercoeff(cls):
        """
        Make our power coefficients for our construct expansion method
        :return powercoeff[n][p]: vector we multiply by our power expansion to get the n'th coefficients
        """
        powercoeff = np.zeros((cls.Lmax+1, cls.Npower))
        for n0 in range(cls.Lmax+1):
            for n1 in range(cls.Lmax+1):
                for n2 in range(cls.Lmax+1):
                    n = n0+n1+n2
                    if n<=cls.Lmax:
                        powercoeff[n][cls.pow2ind[n0][n1][n2]] = \
                            factorial(n,True)/(factorial(n0,True)*factorial(n1,True)*factorial(n2,True))
        return powercoeff

    @classmethod
    def constructexpansion(cls, basis, N=-1):
        """
        Takes a "basis" for constructing an expansion -- list of vectors and matrices --
        and constructs the expansions up to power N (default = Lmax)
        Takes a direction expansion a and b, and returns the sum of the expansions.
        :param basis = list((coeffmatrix, vect)): expansions to create;
          sum(coeffmatrix * (vect*q)^n), for powers n = 0..N
        :param N: maximum power to consider; for N=-1, use Lmax

        :returns list((n, lmax, powexpansion)), ... : our expansion, as input to create
          Taylor3D objects
        """
        if N<0: N=cls.Lmax
        # in principle, we should precompute this once...
        c = []
        for n in range(N+1):
            c.append([(n, n, np.zeros((cls.powlrange[n],) + basis[0][0].shape, dtype=complex))])
        for coeff, vect in basis:
            pexp = cls.powexp(vect, normalize=False)
            for n in range(N+1):
                vnpow = (cls.powercoeff[n]*pexp)[:cls.powlrange[n]]
                cn = c[n][0][2]
                for p in range(cls.powlrange[n]):
                    cn[p] += vnpow[p]*coeff
        return tuple(c)

    # for sorting our coefficient lists:
    @classmethod
    def __sortkey(cls, entry):
        return (entry[0]+entry[1]/cls.Lmax)

    __INITIALIZED__ = False
    # these are all *class* parameters, not object parameters: they are computed
    # and defined once for the entire class. It means that once, in your code, you *choose*
    # a value for Lmax, you are stuck with it. This is a choice: it makes compatibility between
    # the expansions easy, for a minor loss in flexibility.
    @classmethod
    def __initTaylor3Dindexing__(cls, Lmax):
        """
        This calls *all* the class methods defined above, and stores them *for the class*
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
        cls.__INITIALIZED__ = True

    def __init__(self, coefflist = [], Lmax = 4):
        """
        Initializes a Taylor3D object, with coefflist (default = empty)

        :param coefflist: list((n, lmax, powexpansion)). No type checking; default empty
        :param Lmax: maximum power / orbital angular momentum
        """
        self.__initTaylor3Dindexing__(Lmax)
        self.coefflist = coefflist

    def addterms(self, coefflist):
        """
        Add additional coefficients into our object. No type checking. Only works if
        terms are completely non-overlapping (otherwise, need to use sum).
        :param coefflist: list((n, lmax, powexpansion))
        """
        # getattr is here *in case* someone passes us a Taylor3D type object...
        for coeff in getattr(coefflist, 'coefflist', coefflist):
            if any( coeff[0] == c[0] for c in self.coefflist ):
                raise ArithmeticError("Can only use addterms to include non-occuring powers")
            else:
                self.coefflist.append(coeff)
        self.coefflist.sort(key=self.__sortkey)

    # def __call__(self, *args, **kwargs):
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
            fval = [ fnu[(n,l)](umagn) if callable(fnu[(n,l)]) else fnu[(n,l)]
                     for (n, l, coeff) in self.coefflist]
            return np.sum( fv*np.tensordot(u0[:self.powlrange[l]], coeff, axes=1) for fv, (n,l , coeff) in zip(fval, self.coefflist) )
        # otherwise, create a dictionary!
        return { (n,l): np.tensordot(u0[:self.powlrange[l]], coeff, axes=1) for n, l, coeff in self.coefflist}

    def nl(self):
        """
        Returns a list of (n,l) pairs in the coefflist
        :return nl_list: all of the (n,l) pairs that are present in our coefflist
        """
        return sorted([ (n,l) for (n,l,coeff) in self.coefflist], key=self.__sortkey)

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
        acoeff = getattr(a, 'coefflist', a) # fallback to a if not there... which assumes it's a list
        bcoeff = getattr(b, 'coefflist', b) # fallback to b if not there... which assumes it's a list
        ashape = acoeff[0][2].shape
        bshape = bcoeff[0][2].shape
        if ashape[1:] != bshape[1:]:
            raise TypeError('Unable to add--not compatible')
        # make c = copy of a
        if not inplace:
            c = [(an, almax, alpha*apow) for (an, almax, apow) in acoeff]
        else:
            c = a
        for bn, blmax, bpow in bcoeff:
            # now add it into the list
            cpow = beta*bpow
            matched = False
            for cmatch in c:
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
                    cpow[:cls.powlrange[clmax0]][:cls.powlrange[clmax0]] += cmatch[2]
                    cmatch = (bn, blmax, cpow)
                else:
                    # can just append in place: need to be careful, since we have a tuple
                    coeff = cmatch[2]
                    coeff[:cls.powlrange[blmax]][:cls.powlrange[blmax]] += cpow
        c.sort(key=cls.__sortkey)
        return c

    def __add__(self, other):
        """Add a set of Taylor expansions"""
        # if we're passed an array, just take it in stride
        if hasattr(other, 'shape'): other = (0, 0, other.reshape((1,) + other.shape))
        return Taylor3D(self.sumcoeff(self, other))

    def __radd__(self, other):
        """Add a set of Taylor expansions"""
        # if we're passed an array, just take it in stride
        if hasattr(other, 'shape'): other = (0, 0, other.reshape((1,) + other.shape))
        return Taylor3D(self.sumcoeff(self, other))

    def __iadd__(self, other):
        """Add a set of Taylor expansions"""
        # if we're passed an array, just take it in stride
        if hasattr(other, 'shape'): other = (0, 0, other.reshape((1,) + other.shape))
        self.sumcoeff(self, other, inplace=True)
        return self

    def __sub__(self, other):
        """Subtract a set of Taylor expansions"""
        # if we're passed an array, just take it in stride
        if hasattr(other, 'shape'): other = (0, 0, other.reshape((1,) + other.shape))
        return Taylor3D(self.sumcoeff(self, other, 1, -1))

    def __rsub__(self, other):
        """Subtract a set of Taylor expansions"""
        # if we're passed an array, just take it in stride
        if hasattr(other, 'shape'): other = (0, 0, other.reshape((1,) + other.shape))
        return Taylor3D(self.sumcoeff(self, other, -1, 1))

    def __isub__(self, other):
        """Subtract a set of Taylor expansions"""
        # if we're passed an array, just take it in stride
        if hasattr(other, 'shape'): other = (0, 0, other.reshape((1,) + other.shape))
        self.sumcoeff(self, other, 1, -1, inplace=True)
        return self

    @classmethod
    def scalarproductcoeff(cls, c, a, inplace=False):
        """
        Multiplies an coefficient expansion a by a scalar c
        :param: c
            scalar
        :param: a = list((n, lmax, powexpansion)
            expansion of function in powers
        :param inplace: modify a in place?
        :return coefflist: c*a
        """
        acoeff = getattr(a, 'coefflist', a)
        if inplace:
            for an, almax, apow in acoeff:
                apow *= c
            ca = a
        else:
            # create new expansion
            ca = []
            for an, almax, apow in acoeff:
                ca.append((an, almax, c*apow))
        return ca

    @classmethod
    def coeffproductcoeff(cls, a, b):
        """
        Takes a direction expansion a and b, and returns the product expansion.
        :param: a, b = list((n, lmax, powexpansion)
            written as a series of coefficients; n defines the magnitude function, which
            is additive; lmax is the largest cumulative power of coefficients, and
            powexpansion is a numpy array that can multiplied. We assume that a and b
            have consistent shapes throughout--we *do not test this*; runtime will likely
            fail if not true. The entries in the list are *tuples* of n, lmax, pow

        :return: c = list((n, lmax, powexpansion))
            product of a and b
        """
        # a little pythonic magic to work with *either* a list, or an object with a coefflist
        acoeff = getattr(a, 'coefflist', a) # fallback to a if not there... which assumes it's a list
        bcoeff = getattr(b, 'coefflist', b) # fallback to b if not there... which assumes it's a list
        c = []
        ashape = acoeff[0][2].shape
        bshape = bcoeff[0][2].shape
        if ashape[-1] != bshape[1]:
            raise TypeError('Unable to multiply--not compatible')
        cshape = ashape[1:-1] + bshape[2:] # weird piece of python to find the shape of a*b
        for an, almax, apow in acoeff:
            for bn, blmax, bpow in bcoeff:
                cn = an+bn
                clmax = almax + blmax
                if clmax > cls.Lmax:
                    # in theory... we should warn the user here
                    clmax = cls.Lmax
                # construct the expansion
                cpow = np.zeros((cls.powlrange[clmax],) + cshape, dtype=complex)
                for pa in range(cls.powlrange[almax]):
                    for pb in range(cls.powlrange[blmax]):
                        cpow[cls.direcmult[pa][pb]] += np.dot(apow[pa], bpow[pb])
                # now add it into the list
                matched = False
                for cmatch in c:
                    if cmatch[0] == bn:
                        matched = True
                        break
                if not matched:
                    c.append((cn, clmax, cpow))
                else:
                    # a little tricky: we need to *append* to an existing term
                    clmax0 = cmatch[1]
                    if clmax > clmax0:
                        # need to replace cmatch with a new tuple
                        cpow[:cls.powlrange[clmax0]][:cls.powlrange[clmax0]] += cmatch[2]
                        cmatch = (cn, clmax, cpow)
                    else:
                        # can just append in place: need to be careful, since we have a tuple
                        coeff = cmatch[2]
                        coeff[:cls.powlrange[clmax]][:cls.powlrange[clmax]] += cpow
        c.sort(key=cls.__sortkey)
        return c

    def __mul__(self, other):
        """
        Multiply our expansion
        :param other:
        :return: our expansion
        """
        if isinstance(other, Number) or hasattr(other, 'shape'):
            coeff = self.scalarproductcoeff(other, self)
        else:
            coeff = self.coeffproductcoeff(self, other)
        return Taylor3D(coeff)

    def __rmul__(self, other):
        """
        Multiply our expansion
        :param other:
        :return: our expansion
        """
        if isinstance(other, Number) or hasattr(other, 'shape'):
            coeff = self.scalarproductcoeff(other, self)
        else:
            coeff = self.coeffproductcoeff(self, other)
        return Taylor3D(coeff)
