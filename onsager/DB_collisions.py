import numpy as np
from onsager.DB_structs import dumbbell, SdPair, jump

def collision_self(dbcontainer, dbcontainer2, jump, cutoff12, cutoff13=None):
    """
    Check if the three atoms involved in a dumbbell jumping from one site to the next
    are colliding or not.

    :params dbcontainer: the dumbell states container (in crystal module)
    :param dbcontainer2: the second container if the jumps are occuring between pure and mixed dumbbell spaces. "None" otherwise
    :param jump: the jump object representing the transitions
    :param cutoff12: minimum allowed distance between the two atoms in the initial dumbbell.
    :param cutoff13: minimum allowed distance between the two atoms in the final dumbbell.
    :return bool: True if atoms collide. False otherwise
    """
    crys, chem = dbcontainer.crys, dbcontainer.chem
    if cutoff13 == None:
        cutoff13 = cutoff12

    if dbcontainer2 is None:
        dbcontainer2 = dbcontainer

    def iscolliding(a0i, a1i, a0j, a1j, cutoff):
        """
        checks if two atoms are considered to be colliding within the cutoff specified.
        The position of an atom 'i' as a function of fractional time (going from 0 to 1) is given by: R(t) = a0i + a1i * t
        Then the minimum squared distance between atoms 'i' and 'j' is then minimized as a function of time.

        :param a0i: the initial position of the first atom.
        :param a1i: the total displacement of the first atom during the jump (a0i + a1i is the final position).
        :param a0j: the initial position of the second atom.
        :param a1j: the total displacement of the second atom during the jump.

        :return bool: True if the given atom pair comes closer than cutoff within t=0 or t=1, False otherwise.
        """

        num = np.dot((a1i - a1j), (a0i - a0j))
        den = np.dot((a1i - a1j), (a1i - a1j))
        tmin = np.round(-num / den, decimals=6)
        # print(tmin)
        mindist2 = np.round(
            np.dot(((a0i + a1i * tmin) - (a0j + a1j * tmin)), ((a0i + a1i * tmin) - (a0j + a1j * tmin))), decimals=6)
        # print ("mindist^2 = ",mindist2)
        # print ("cutoff^2 = ",np.round(cutoff**2,decimals=6))
        # print()
        if tmin <= 0 or tmin >= 1:
            return False  # no atoms collide within transition time.
        elif (mindist2 >= np.round(cutoff ** 2, decimals=6)):
            return False  # no atoms collide
        else:
            return True  # atoms collide

    # create the initial and final locations of the atoms
    if isinstance(jump.state1, dumbbell):
        R1i = crys.unit2cart(jump.state1.R, crys.basis[chem][dbcontainer.iorlist[jump.state1.iorind][0]]) +\
              (jump.c1 / 2.) * dbcontainer.iorlist[jump.state1.iorind][1]

        R2i = crys.unit2cart(jump.state1.R, crys.basis[chem][dbcontainer.iorlist[jump.state1.iorind][0]]) -\
              (jump.c1 / 2.) * dbcontainer.iorlist[jump.state1.iorind][1]

        R3i = crys.unit2cart(jump.state2.R, crys.basis[chem][dbcontainer2.iorlist[jump.state2.iorind][0]])

        R1f = crys.unit2cart(jump.state2.R, crys.basis[chem][dbcontainer2.iorlist[jump.state2.iorind][0]]) +\
              (jump.c2 / 2.) * dbcontainer2.iorlist[jump.state2.iorind][1]

        R2f = crys.unit2cart(jump.state1.R, crys.basis[chem][dbcontainer.iorlist[jump.state1.iorind][0]])

        R3f = crys.unit2cart(jump.state2.R, crys.basis[chem][dbcontainer2.iorlist[jump.state2.iorind][0]]) -\
              (jump.c2 / 2.) * dbcontainer2.iorlist[jump.state2.iorind][1]
        # print(R1i,R2i,R3i,R1f,R2f,R3f)
    else:
        R1i = crys.unit2cart(jump.state1.db.R, crys.basis[chem][dbcontainer.iorlist[jump.state1.db.iorind][0]]) +\
              (jump.c1 / 2.) * dbcontainer.iorlist[jump.state1.db.iorind][1]

        R2i = crys.unit2cart(jump.state1.db.R, crys.basis[chem][dbcontainer.iorlist[jump.state1.db.iorind][0]]) -\
              (jump.c1 / 2.) * dbcontainer.iorlist[jump.state1.db.iorind][1]

        R3i = crys.unit2cart(jump.state2.db.R, crys.basis[chem][dbcontainer2.iorlist[jump.state2.db.iorind][0]])

        R1f = crys.unit2cart(jump.state2.db.R, crys.basis[chem][dbcontainer2.iorlist[jump.state2.db.iorind][0]]) +\
              (jump.c2 / 2.) * dbcontainer2.iorlist[jump.state2.db.iorind][1]

        R2f = crys.unit2cart(jump.state1.db.R, crys.basis[chem][dbcontainer.iorlist[jump.state1.db.iorind][0]])
        R3f = crys.unit2cart(jump.state2.db.R, crys.basis[chem][dbcontainer2.iorlist[jump.state2.db.iorind][0]]) -\
              (jump.c2 / 2.) * dbcontainer2.iorlist[jump.state2.db.iorind][1]

    if np.allclose(R1i, R1f):
        return False  # not considering rotations(yet).
    a01 = R1i.copy()
    a11 = (R1f - R1i)
    a02 = R2i.copy()
    a12 = (R2f - R2i)
    a03 = R3i.copy()
    a13 = (R3f - R3i)
    # check the booleans for each pair
    c12 = iscolliding(a01, a11, a02, a12, cutoff12)
    # print(c12)
    c13 = iscolliding(a01, a11, a03, a13, cutoff13)
    # print(c13)
    # c23 = isnotcolliding(a02,a12,a03,a13,cutoff)
    return (c12 or c13)


def collision_others(container, container2, jmp, closestdistance):
    """
    Takes a jump and sees if the moving atom of the dumbbell collides with any other atom, within a cuttoff distance.

    :params container: the dumbbell states container.
    :param container2: the second dumbbell states container in case the jumps are occurring between pure and mixed dumbbells. "None" otherwise.
    :param jmp: the jump object to test.
    :param closestdistance: (A list or a number) minimum allowable distance to other atoms in other sublattices.
    :return bool: True if atoms collide. False otherwise.
    """
    if container2 is None:
        container2=container
    crys, chem = container.crys, container.chem
    # Format closestdistance appropriately
    if isinstance(closestdistance, list):
        closest2list = [x ** 2 for c, x in enumerate(closestdistance)]
    else:
        closest2list = [closestdistance ** 2 for c in range(crys.Nchem)]

    # First extract the necessary parameters for calculating the transport vector
    if isinstance(jmp.state1, dumbbell):
        (i1, i2) = (container.iorlist[jmp.state1.iorind][0], container2.iorlist[jmp.state2.iorind][0])
    else:
        (i1, i2) = (container.iorlist[jmp.state1.db.iorind][0], container2.iorlist[jmp.state2.db.iorind][0])

    if isinstance(jmp.state1, dumbbell):
        (R1, R2) = (jmp.state1.R, jmp.state2.R)
    else:
        (R1, R2) = (jmp.state1.db.R, jmp.state2.db.R)

    if isinstance(jmp.state1, dumbbell):
        (o1, o2) = (container.iorlist[jmp.state1.iorind][1], container2.iorlist[jmp.state2.iorind][1])
    else:
        (o1, o2) = (container.iorlist[jmp.state1.db.iorind][1], container2.iorlist[jmp.state2.db.iorind][1])

    c1, c2 = jmp.c1, jmp.c2
    dvec = (c2 / 2.) * o2 - (c1 / 2.) * o1
    dR = crys.unit2cart(R2, crys.basis[chem][i2]) - crys.unit2cart(R1, crys.basis[chem][i1])
    dx = dR + dvec
    dx2 = np.dot(dx, dx)
    # Do not consider on-site rotations
    dR2 = np.dot(dR, dR)
    if np.allclose(dR, 0, atol=crys.threshold):
        return False
    nmax = [int(np.round(np.sqrt(dx2 / crys.metric[i, i]))) + 1 for i in range(crys.dim)]
    # print(nmax)
    if crys.dim == 2:
        supervect = [np.array([n0, n1]) for n0 in range(-nmax[0], nmax[0] + 1) for n1 in range(-nmax[1], nmax[1] + 1)]
    else:
        supervect = [np.array([n0, n1, n2]) for n0 in range(-nmax[0], nmax[0] + 1) for n1 in range(-nmax[1], nmax[1] + 1)
                     for n2 in range(-nmax[2], nmax[2] + 1)]
    # print(supervect)
    # print()
    # now test against other atoms, treating the initial atom as the origin
    for c, mindist2 in enumerate(closest2list):
        for j, u0 in enumerate(crys.basis[c]):
            for n in supervect:
                # skip checking against the atom in the initial and destination site
                if np.allclose(R1, n, atol=crys.threshold) and j == i1:
                    continue
                if np.allclose(R2, n, atol=crys.threshold) and j == i2:
                    continue
                x = crys.unit2cart(n, u0) - (crys.unit2cart(R1, crys.basis[chem][i1]) + (c1 / 2.) * o1)
                # Get the location of the atom with respect to the original position of the jumping atom
                x2 = np.dot(x, x)
                x_dx = np.dot(x, dx)
                d2 = (x2 * dx2 - x_dx ** 2) / dx2
                # if j==1 and np.allclose(n,0,atol=1e-8):
                #     print(d2)
                if 0 <= x_dx <= dx2:
                    if np.isclose(d2, mindist2) or d2 < mindist2:
                        # print(c,n,u0)
                        # This print statement is only for seeing outputs in the testing phase.
                        return True
    return False  # if no collision occurs.
