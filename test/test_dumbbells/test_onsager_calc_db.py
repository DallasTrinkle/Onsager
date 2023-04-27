from onsager.crystal import Crystal
from onsager.crystalStars import zeroclean
from onsager.OnsagerCalc import dumbbellMediated

from crysts import *
from onsager.crystal import DB_disp, pureDBContainer, mixedDBContainer
from onsager.DB_structs import dumbbell, SdPair, jump, connector
import unittest
import itertools

ratelist = dumbbellMediated.ratelist
symmratelist = dumbbellMediated.symmratelist


class test_dumbbell_mediated(unittest.TestCase):
    def setUp(self):
        # We test a new weird lattice because it is more interesting

        latt = np.array([[0.5, 0.5, 0.], [0., 0.5, 0.5], [0.5, 0., 0.5]]) * 0.55
        self.DC_Si = Crystal(latt, [[np.array([0., 0., 0.]), np.array([0.25, 0.25, 0.25])]], ["Si"])
        # keep it simple with [1.,0.,0.] type orientations for now
        o = np.array([1., 1., 0.])/(np.linalg.norm(np.array([1., 1., 0.]))) * 0.1
        famp0 = [o.copy()]
        family = [famp0]

        self.pdbcontainer_si = pureDBContainer(self.DC_Si, 0, family)
        self.mdbcontainer_si = mixedDBContainer(self.DC_Si, 0, family)
        self.jset0, self.jset2 = \
            self.pdbcontainer_si.jumpnetwork(0.3, 0.01, 0.01), self.mdbcontainer_si.jumpnetwork(0.3, 0.01, 0.01)

        self.onsagercalculator = dumbbellMediated(self.pdbcontainer_si, self.mdbcontainer_si, self.jset0, self.jset2,
                                                  0.3, 0.01, 0.01, 0.01, NGFmax=4, Nthermo=1)
        # generate all the bias expansions - will separate out later
        self.biases = \
            self.onsagercalculator.vkinetic.biasexpansion(self.onsagercalculator.jnet1, self.onsagercalculator.jnet2,
                                                          self.onsagercalculator.om1types,
                                                          self.onsagercalculator.jnet43)

        self.W1list = np.random.rand(len(self.onsagercalculator.jnet1))
        self.W2list = np.random.rand(len(self.onsagercalculator.jnet0))
        self.W3list = np.random.rand(len(self.onsagercalculator.jnet3))
        self.W4list = np.random.rand(len(self.onsagercalculator.jnet4))
        print(len(self.onsagercalculator.vkinetic.vecpos_bare))
        print("Initiated Si")

    def test_thermo2kin(self):
        for th_ind, k_ind in enumerate(self.onsagercalculator.thermo2kin):
            count = 0
            for state1 in self.onsagercalculator.thermo.stars[th_ind]:
                for state2 in self.onsagercalculator.vkinetic.starset.stars[k_ind]:
                    if state1 == state2:
                        count += 1
            self.assertEqual(count, len(self.onsagercalculator.thermo.stars[th_ind]))

    def test_calc_eta(self):
        # set up random pre-factors and energies for rate calculations
        pre0 = np.random.rand(len(self.onsagercalculator.pdbcontainer.symorlist))
        betaene0 = np.random.rand(len(self.onsagercalculator.pdbcontainer.symorlist))
        pre0T = np.random.rand(len(self.onsagercalculator.jnet0))
        betaene0T = np.random.rand(len(self.onsagercalculator.jnet0))
        # pre2 = np.random.rand(len(self.onsagercalculator.mdbcontainer.symorlist))
        # betaene2 = np.random.rand(len(self.onsagercalculator.mdbcontainer.symorlist))
        # pre2T = np.random.rand(len(self.onsagercalculator.jnet2))
        # betaene2T = np.random.rand(len(self.onsagercalculator.jnet2))

        rate0list = ratelist(self.onsagercalculator.jnet0_indexed, pre0, betaene0, pre0T, betaene0T,
                                 self.onsagercalculator.vkinetic.starset.pdbcontainer.invmap)


        # rate2list = ratelist(self.onsagercalculator.jnet2_indexed, pre2, betaene2, pre2T, betaene2T,
        #                      self.onsagercalculator.vkinetic.starset.mdbcontainer.invmap)

        rate0_forward = np.array([rate0list[jt][0] for jt in range(len(rate0list))])
        rate0_backward = np.array([rate0list[jt][1] for jt in range(len(rate0list))])

        # rate2_forward = np.array([rate2list[jt][0] for jt in range(len(rate2list))])
        # rate2_backward = np.array([rate2list[jt][1] for jt in range(len(rate2list))])

        rate0_wycks = np.zeros((len(self.onsagercalculator.pdbcontainer.symorlist), len(rate0list)))
        # send in rate0_wycks as the argument to calc_eta - this is the same as omega0escape, except with a
        # randomized form.

        for jt, jlist in enumerate(self.onsagercalculator.jnet0):
            db1_ind = jlist[0].state1.iorind
            db2_ind = jlist[0].state2.iorind

            w1 = self.onsagercalculator.pdbcontainer.invmap[db1_ind]
            w2 = self.onsagercalculator.pdbcontainer.invmap[db2_ind]

            rate0_wycks[w1, jt] = rate0_forward[jt]
            rate0_wycks[w2, jt] = rate0_backward[jt]

        # rate2_wycks = np.zeros((len(self.onsagercalculator.mdbcontainer.symorlist), len(rate2list)))
        #
        # for jt, jlist in enumerate(self.onsagercalculator.jnet2):
        #     db1_ind = jlist[0].state1.db.iorind
        #     db2_ind = jlist[0].state2.db.iorind
        #
        #     w1 = self.onsagercalculator.mdbcontainer.invmap[db1_ind]
        #     w2 = self.onsagercalculator.mdbcontainer.invmap[db2_ind]
        #
        #     rate2_wycks[w1, jt] = rate2_forward[jt]
        #     rate2_wycks[w2, jt] = rate2_backward[jt]

        self.onsagercalculator.calc_eta(rate0list, rate0_wycks)#, rate2list, rate2_wycks)

        if len(self.onsagercalculator.vkinetic.vecpos_bare) == 0:
            self.assertTrue(np.allclose(self.onsagercalculator.eta00_solvent,
                                        np.zeros((len(self.onsagercalculator.vkinetic.starset.complexStates),
                                                  self.onsagercalculator.crys.dim))))
            print("Null basis of pure dumbbell - checked zero bare relaxation vectors.")

        else:
            print("Non zero basis detected for pure dumbbell.")
            # Here, we check if for periodic dumbbells, we have the same non- local solvent velocity vector.
            for i, state1 in enumerate(self.onsagercalculator.vkinetic.starset.complexStates):
                for j, state2 in enumerate(self.onsagercalculator.vkinetic.starset.complexStates):
                    if state1.db.iorind == state2.db.iorind:
                        self.assertTrue(np.allclose(self.onsagercalculator.eta00_solvent[i, :],
                                                    self.onsagercalculator.eta00_solvent[j, :]))
                        self.assertTrue(np.allclose(self.onsagercalculator.eta00_solvent[i, :],
                                                    self.onsagercalculator.eta00_solvent_bare[state1.db.iorind]))

            # Check that we get the correct non-local velocity vector
            for i, db in enumerate(self.onsagercalculator.vkinetic.starset.bareStates):
                vel_calc = self.onsagercalculator.NlsolventVel_bare[i, :]
                vel_true = np.zeros(3)
                for jt, jlist in enumerate(self.onsagercalculator.jnet0_indexed):
                    for jnum, ((IS, FS), dx) in enumerate(jlist):
                        if i == IS:
                            vel_true += rate0list[jt][jnum] * dx
                self.assertTrue(np.allclose(vel_true, vel_calc), msg="\n{}\n{}\n{}".format(i, vel_true, vel_calc))

            # The above test confirms that our NlsolventVel_bare is correct
            # Now we check if the eta vectors are true
            vMags_solvent = []
            for i, dbstate in enumerate(self.onsagercalculator.vkinetic.starset.bareStates):

                vel_test = np.zeros(3)

                # First, quickly check the barestate indexing
                self.assertEqual(i, self.onsagercalculator.vkinetic.starset.bareindexdict[dbstate][0])

                vel_calc = self.onsagercalculator.NlsolventVel_bare[i, :]

                for jt, jindlist, jlist in zip(itertools.count(), self.onsagercalculator.jnet0_indexed,
                                               self.onsagercalculator.jnet0):

                    for jnum, ((IS, FS), dx), jmp in zip(itertools.count(), jindlist, jlist):
                        if i == IS:
                            # quick check to see jump indexing in consistent (although done while testing stars)
                            self.assertTrue(dbstate == jmp.state1)
                            vel_test += rate0list[jt][jnum] * (self.onsagercalculator.eta00_solvent_bare[FS, :] -
                                                                self.onsagercalculator.eta00_solvent_bare[IS, :])

                self.assertTrue(np.allclose(vel_test, vel_calc), msg="{}{}".format(vel_test, vel_calc))
                vMags_solvent.append(np.max(np.abs(vel_test)))

            print("max pure state solvent component: {}".format(max(vMags_solvent)))

            # A small test to reaffirm that vector bases are calculated properly for the bare states.
            for i in range(len(self.onsagercalculator.vkinetic.starset.bareStates)):
                # get the indices of the state
                st = self.onsagercalculator.vkinetic.starset.bareStates[i]
                # next get the vectorstar,state indices
                indlist = self.onsagercalculator.vkinetic.stateToVecStar_bare[st]
                # The IndOfStar in indlist for the mixed case is already shifted by the number of pure vector stars.
                if len(indlist)!=0:
                    vlist = []
                    for tup in indlist:
                        # get the vectors and the length of the vector stars.
                        vlist.append((self.onsagercalculator.vkinetic.vecvec_bare[tup[0]][tup[1]],
                                      len(self.onsagercalculator.vkinetic.vecvec_bare[tup[0]])))

                    # See that all the vector stars are consistent
                    nsv0 = vlist[0][1]
                    for v, N_sv in vlist:
                        self.assertEqual(N_sv, nsv0)

                    eta_test_solvent = sum([np.dot(self.onsagercalculator.eta00_solvent_bare[i, :], v) * v * N_sv
                                            for v, N_sv in vlist])

                    self.assertTrue(np.allclose(eta_test_solvent, self.onsagercalculator.eta00_solvent_bare[i]),
                                    msg="{} {}".format(eta_test_solvent, self.onsagercalculator.eta00_solvent_bare[i]))

        # # Now we test the solute and solvent non-local eta vectors in mixed dumbbell space
        # # Check that we get the correct non-local velocity vector
        # for i, state in enumerate(self.onsagercalculator.vkinetic.starset.mixedstates):
        #     vel_calc_solvent = self.onsagercalculator.NlsolventVel_mixed[i, :]
        #     vel_true_solvent = np.zeros(self.onsagercalculator.crys.dim)
        #
        #     vel_calc_solute = self.onsagercalculator.NlsoluteVel_mixed[i, :]
        #     vel_true_solute = np.zeros(self.onsagercalculator.crys.dim)
        #
        #     for jt, jlist in enumerate(self.onsagercalculator.jnet2_indexed):
        #         for jnum, ((IS, FS), dx) in enumerate(jlist):
        #             if i == IS:
        #                 # or1 = self.onsagercalculator.mdbcontainer.iorlist[IS][1]
        #                 # or2 = self.onsagercalculator.mdbcontainer.iorlist[FS][1]
        #
        #                 dx_solute = dx  #+ or2/2. - or1/2.
        #                 dx_solvent = dx  #- or2 / 2. + or1 / 2.
        #                 vel_true_solvent += rate2list[jt][jnum] * dx_solvent
        #                 vel_true_solute += rate2list[jt][jnum] * dx_solute
        #
        #     self.assertTrue(np.allclose(vel_true_solvent, vel_calc_solvent),
        #                     msg="\n{}\n{}\n{}".format(i, vel_true_solvent, vel_calc_solvent))
        #     self.assertTrue(np.allclose(vel_true_solute, vel_calc_solute),
        #                     msg="\n{}\n{}\n{}".format(i, vel_true_solute, vel_calc_solute))
        #
        # # Now we check if the eta vectors are true
        # vNorms_solute = []
        # vNorms_solvent = []
        # for i, state in enumerate(self.onsagercalculator.vkinetic.starset.mixedstates):
        #
        #     vel_test_solute = np.zeros(self.onsagercalculator.crys.dim)
        #     vel_test_solvent = np.zeros(self.onsagercalculator.crys.dim)
        #
        #     vel_calc_solvent = self.onsagercalculator.NlsolventVel_mixed[i, :]
        #     vel_calc_solute = self.onsagercalculator.NlsoluteVel_mixed[i, :]
        #
        #     for jt, jindlist, jlist in zip(itertools.count(), self.onsagercalculator.jnet2_indexed,
        #                                    self.onsagercalculator.jnet2):
        #
        #         for jnum, ((IS, FS), dx), jmp in zip(itertools.count(), jindlist, jlist):
        #             if i == IS:
        #                 # quick check to see if jump indexing in consistent
        #                 self.assertTrue(state == jmp.state1)
        #                 vel_test_solvent += rate2list[jt][jnum] * (self.onsagercalculator.eta02_solvent[FS, :] -
        #                                                            self.onsagercalculator.eta02_solvent[IS, :])
        #
        #                 vel_test_solute += rate2list[jt][jnum] * (self.onsagercalculator.eta02_solute[FS, :] -
        #                                                           self.onsagercalculator.eta02_solute[IS, :])
        #
        #     self.assertTrue(np.allclose(vel_test_solute, vel_calc_solute), msg="{}{}".format(vel_test_solute,
        #                                                                                      vel_calc_solute))
        #     self.assertTrue(np.allclose(vel_test_solvent, vel_calc_solvent), msg="{}{}".format(vel_test_solvent,
        #                                                                                        vel_calc_solvent))
        #
        #     vNorms_solute.append(np.max(np.abs(vel_test_solute)))
        #     vNorms_solvent.append(np.max(np.abs(vel_test_solvent)))
        # print("max mixed state solute component: {}".format(max(vNorms_solute)))
        # print("max mixed state solvent component: {}".format(max(vNorms_solvent)))

    def test_bias_updates(self):
        """
        This is to check if the del_bias expansions are working fine, prod.
        """
        pre0 = np.random.rand(len(self.onsagercalculator.pdbcontainer.symorlist))
        betaene0 = np.random.rand(len(self.onsagercalculator.pdbcontainer.symorlist))
        pre0T = np.random.rand(len(self.onsagercalculator.jnet0))
        betaene0T = np.random.rand(len(self.onsagercalculator.jnet0))
        pre2 = np.random.rand(len(self.onsagercalculator.mdbcontainer.symorlist))
        betaene2 = np.random.rand(len(self.onsagercalculator.mdbcontainer.symorlist))
        pre2T = np.random.rand(len(self.onsagercalculator.jnet2))
        betaene2T = np.random.rand(len(self.onsagercalculator.jnet2))

        rate0list = ratelist(self.onsagercalculator.jnet0_indexed, pre0, betaene0, pre0T, betaene0T,
                             self.onsagercalculator.vkinetic.starset.pdbcontainer.invmap)

        rate2list = ratelist(self.onsagercalculator.jnet2_indexed, pre2, betaene2, pre2T, betaene2T,
                             self.onsagercalculator.vkinetic.starset.mdbcontainer.invmap)

        # Let's keep rates symmetric for simplicity
        # Otherwise we'll have to make sure symmetrically rotated jumps have same rates.
        rate0_forward = np.array([rate0list[jt][0] for jt in range(len(rate0list))])
        rate0_backward = rate0_forward
        # rate2_forward = np.array([rate2list[jt][0] for jt in range(len(rate2list))])
        # rate2_backward = rate2_forward


        rate0_wycks = np.zeros((len(self.onsagercalculator.pdbcontainer.symorlist), len(rate0list)))
        # send in rate0_wycks as the argument to calc_eta - this is the same as omega0escape, except with a
        # randomized form.

        for jt, jlist in enumerate(self.onsagercalculator.jnet0):
            db1_ind = jlist[0].state1.iorind
            db2_ind = jlist[0].state2.iorind

            w1 = self.onsagercalculator.pdbcontainer.invmap[db1_ind]
            w2 = self.onsagercalculator.pdbcontainer.invmap[db2_ind]

            rate0_wycks[w1, jt] = rate0_forward[jt]
            rate0_wycks[w2, jt] = rate0_backward[jt]

        # rate2_wycks = np.zeros((len(self.onsagercalculator.mdbcontainer.symorlist), len(rate2list)))
        #
        # for jt, jlist in enumerate(self.onsagercalculator.jnet2):
        #     db1_ind = jlist[0].state1.db.iorind
        #     db2_ind = jlist[0].state2.db.iorind
        #
        #     w1 = self.onsagercalculator.mdbcontainer.invmap[db1_ind]
        #     w2 = self.onsagercalculator.mdbcontainer.invmap[db2_ind]
        #
        #     rate2_wycks[w1, jt] = rate2_forward[jt]
        #     rate2_wycks[w2, jt] = rate2_backward[jt]

        self.onsagercalculator.update_bias_expansions(rate0list, rate0_wycks)  # , rate2list, rate2_wycks)

        # Now, local rates (randomized)
        # randomize the rates for every jump type - but keep them symmetric for now
        rate1_forward = np.random.rand(len(self.onsagercalculator.jnet1))
        rate1_backward = rate1_forward  # np.random.rand(len(self.onsagercalculator.jnet1))
        rate10_forward = np.array([rate0_forward[jt] for jt in self.onsagercalculator.om1types])
        rate10_backward = rate10_forward  # np.array([rate0_backward[jt] for jt in self.onsagercalculator.om1types])

        rate10_stars = np.zeros((self.onsagercalculator.vkinetic.Nvstars_pure, len(self.onsagercalculator.jnet1)))
        rate1_stars = np.zeros((self.onsagercalculator.vkinetic.Nvstars_pure, len(self.onsagercalculator.jnet1)))

        rate1list = []
        for jt, jlist in enumerate(self.onsagercalculator.jnet1):

            st1 = jlist[0].state1
            st2 = jlist[0].state2

            try:
                v1list = self.onsagercalculator.vkinetic.stateToVecStar_pure[st1]
            except KeyError:
                v1list = []
                # for solute-dumbbell complexes, only the origin state can have an empty basis
                self.assertTrue(st1.is_zero(self.onsagercalculator.pdbcontainer))

            try:
                v2list = self.onsagercalculator.vkinetic.stateToVecStar_pure[st2]
            except KeyError:
                v2list = []
                self.assertTrue(st2.is_zero(self.onsagercalculator.pdbcontainer))

            for v1, inv1 in v1list:
                rate1_stars[v1, jt] = rate1_forward[jt]
                rate10_stars[v1, jt] = rate10_forward[jt]
            for v2, inv2 in v2list:
                rate1_stars[v2, jt] = rate1_backward[jt]
                rate10_stars[v2, jt] = rate10_backward[jt]

            newlist = []
            count = 0
            while count < len(jlist) // 2:
                newlist.append(rate1_forward[jt])
                newlist.append(rate1_backward[jt])
                count += 1

            rate1list.append(newlist)

        rate43_forward = np.random.rand((len(self.onsagercalculator.jnet43)))
        rate43_backward = np.random.rand((len(self.onsagercalculator.jnet43)))

        Nvstars_mixed = self.onsagercalculator.vkinetic.Nvstars - self.onsagercalculator.vkinetic.Nvstars_pure

        rate3_stars = np.zeros((Nvstars_mixed, len(self.onsagercalculator.jnet43)))

        rate4_stars = np.zeros((self.onsagercalculator.vkinetic.Nvstars_pure,
                                len(self.onsagercalculator.jnet43)))

        rate3list = []
        rate4list = []

        for jt, jlist in enumerate(self.onsagercalculator.jnet43):
            st1 = jlist[0].state1
            st2 = jlist[0].state2
            try:
                v1list = self.onsagercalculator.vkinetic.stateToVecStar_pure[st1]
            except KeyError: # for empty basis states - skip
                v1list = []
                self.assertTrue(st1.is_zero(self.onsagercalculator.pdbcontainer))

            # Mixed state must have non-empty basis
            v2list = self.onsagercalculator.vkinetic.stateToVecStar_mixed[st2]

            for v1, inv1 in v1list:
                rate4_stars[v1, jt] = rate43_forward[jt]
            for v2, inv2 in v2list:
                rate3_stars[v2 - self.onsagercalculator.vkinetic.Nvstars_pure, jt] = rate43_backward[jt]

            newlist3 = []
            newlist4 = []
            count = 0
            while count < len(jlist) // 2:
                newlist3.append(rate43_backward[jt])
                newlist4.append(rate43_forward[jt])
                count += 1
            rate3list.append(newlist3)
            rate4list.append(newlist4)

        # VELOCITY MATCHING TESTS START HERE

        biasBareExp = self.onsagercalculator.biases[-1]
        # Next, we calculate the velocity updates explicitly.
        # First, we verify that the non-local velocity out of all the bare dumbbell states disappear
        if not len(self.onsagercalculator.vkinetic.vecpos_bare) == 0:
            print("non empty basis - checking non-local solvent velocities")
            vel0_solvent_vs = np.array([np.dot(biasBareExp[i, :],
                                                    rate0_wycks[self.onsagercalculator.vkinetic.vwycktowyck_bare[i], :])
                                             for i in range(len(self.onsagercalculator.vkinetic.vecvec_bare))])

            solvent_vel_Nl = np.zeros((len(self.onsagercalculator.vkinetic.starset.bareStates), 3))
            for i, db in enumerate(self.onsagercalculator.vkinetic.starset.bareStates):
                indlist = self.onsagercalculator.vkinetic.stateToVecStar_bare[db]
                # We have indlist as (IndOfStar, IndOfState)
                solvent_vel_Nl[i, :] = sum([vel0_solvent_vs[tup[0]] *
                                             self.onsagercalculator.vkinetic.vecvec_bare[tup[0]][tup[1]]
                                             for tup in indlist])

            self.assertTrue(np.allclose(solvent_vel_Nl, self.onsagercalculator.NlsolventVel_bare))

            for i, state in enumerate(self.onsagercalculator.vkinetic.starset.complexStates):
                dbstate_ind = state.db.iorind
                self.assertTrue(np.allclose(self.onsagercalculator.NlsolventBias0[i, :], solvent_vel_Nl[dbstate_ind, :]))

            print("max solvent non-local vel component : {}".format(np.max(solvent_vel_Nl)))

            # Next, update with eta vectors manually
            for jt, jindlist in enumerate(self.onsagercalculator.jnet0_indexed):
                for jnum, ((i, j), dx) in enumerate(jindlist):
                    solvent_vel_Nl[i, :] += rate0list[jt][jnum] * (self.onsagercalculator.eta00_solvent_bare[i] -
                                                                   self.onsagercalculator.eta00_solvent_bare[j])

            print("max solvent shifted non-local vel component : {}".format(np.max(solvent_vel_Nl)))
            self.assertTrue(np.allclose(solvent_vel_Nl, np.zeros_like(solvent_vel_Nl)))

        # Now, we check eta vectors for omega1
        bias1solute, bias1solvent = self.biases[1]
        self.assertTrue(bias1solute is None)
        # bias1_solute_vs = np.array([np.dot(bias1solute[i, :], rate1_stars[i, :])
        #                             for i in range(self.onsagercalculator.vkinetic.Nvstars_pure)])

        bias1_solvent_vs = np.array([np.dot(bias1solvent[i, :], rate1_stars[i, :])
                                     for i in range(self.onsagercalculator.vkinetic.Nvstars_pure)])

        # Now, convert this into the Nstates x 3 form
        # solute_vel_1 = np.zeros((len(self.onsagercalculator.vkinetic.starset.complexStates), self.onsagercalculator.crys.dim))
        solvent_vel_1 = np.zeros((len(self.onsagercalculator.vkinetic.starset.complexStates), self.onsagercalculator.crys.dim))
        for i, state in enumerate(self.onsagercalculator.vkinetic.starset.complexStates):
            try:
                indlist = self.onsagercalculator.vkinetic.stateToVecStar_pure[state]
            except KeyError:
                indlist = []
                self.assertTrue(state.is_zero(self.onsagercalculator.pdbcontainer))
            # We have indlist as (IndOfStar, IndOfState)
            # solute_vel_1[i, :] = sum([bias1_solute_vs[vstarind] *
            #                           self.onsagercalculator.vkinetic.vecvec[vstarind][invstarind]
            #                            for vstarind, invstarind in indlist])

            solvent_vel_1[i, :] = sum([bias1_solvent_vs[vstarind] *
                                       self.onsagercalculator.vkinetic.vecvec[vstarind][invstarind]
                                       for vstarind, invstarind in indlist])

        # Next, manually update with the eta0 vectors
        # for i in range(len(self.onsagercalculator.vkinetic.starset.complexStates)):
        for jt, jlist in enumerate(self.onsagercalculator.jnet1_indexed):
            for jnum, ((IS, FS), dx) in enumerate(jlist):
                # if i==IS:
                # solute_vel_1[IS, :] += rate1list[jt][jnum]*(self.onsagercalculator.eta00_solute[IS, :] -
                #                                             self.onsagercalculator.eta00_solute[FS, :])
                solvent_vel_1[IS, :] += rate1list[jt][jnum]*(self.onsagercalculator.eta00_solvent[IS, :] -
                                                             self.onsagercalculator.eta00_solvent[FS, :])

        # Now, get the version from the updated expansion
        # vel1_solute_new_vs = np.array([np.dot(self.onsagercalculator.bias1_solute_new[i, :], rate1_stars[i, :])
        #                                 for i in range(self.onsagercalculator.vkinetic.Nvstars_pure)])

        vel1_solvent_new_vs = np.array([np.dot(self.onsagercalculator.bias1_solvent_new[i, :], rate1_stars[i, :])
                                         for i in range(self.onsagercalculator.vkinetic.Nvstars_pure)])

        # solute_vel_1_new = np.zeros((len(self.onsagercalculator.vkinetic.starset.complexStates), self.onsagercalculator.crys.dim))
        solvent_vel_1_new = np.zeros((len(self.onsagercalculator.vkinetic.starset.complexStates), self.onsagercalculator.crys.dim))
        for i, state in enumerate(self.onsagercalculator.vkinetic.starset.complexStates):
            try:
                indlist = self.onsagercalculator.vkinetic.stateToVecStar_pure[state]
            except KeyError:
                indlist = []
                self.assertTrue(state.is_zero(self.onsagercalculator.pdbcontainer))

            # solute_vel_1_new[i, :] = sum([vel1_solute_new_vs[tup[0]] *
            #                                self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])
            solvent_vel_1_new[i, :] = sum([vel1_solvent_new_vs[tup[0]] *
                                            self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])

        # Check that they are the same
        # self.assertTrue(np.allclose(solute_vel_1, solute_vel_1_new))
        # self.assertTrue(np.allclose(solute_vel_1, np.zeros_like(solute_vel_1)))
        self.assertTrue(np.allclose(solvent_vel_1, solvent_vel_1_new), msg="{} \n{}".format(solvent_vel_1, solvent_vel_1_new))

        # For the kinetic shell, we check that for states in the thermodynamic shell,
        # out of which every (omega0-allowed) jump leads to another state in the kinetic shell,
        # the non-local velocity becomes zero, using omega1 jumps.
        # For those states in the kinetic but outside the thermodynamic shell, the corresponding non-local bias using
        # just the jumps in omega1 should be non-zero, since all possible jumps out of them are not considered.

        elim_list = np.zeros(len(self.onsagercalculator.vkinetic.starset.complexStates))

        # This array stores how many omega0 jumps are not considered in omega1, for each state
        for stindex, st in enumerate(self.onsagercalculator.vkinetic.starset.complexStates):
            for jlist in self.onsagercalculator.jnet0:
                for jmp in jlist:
                    try:
                        stnew = st.addjump(jmp)
                    except:
                        continue
                    if not stnew in self.onsagercalculator.vkinetic.starset.stateset:
                        # see that if the jump out of a shell leads to a state outside the kinetic shell,
                        # then the state is inside the kinetic shell
                        self.assertTrue(stnew not in self.onsagercalculator.thermo.complexStates)
                        elim_list[stindex] += 1

        # Get the non-local rates corresponding to a omega1 jump
        rate10list = []
        for jt, jlist in enumerate(self.onsagercalculator.jnet1):
            newlist = []
            for jnum, jmp in enumerate(jlist):
                ISdb, FSdb = jmp.state1.db.iorind, jmp.state2.db.iorind
                (IS, FS), dx = self.onsagercalculator.jnet1_indexed[jt][jnum]
                rate01 = None
                count = 0
                # We need to iterate over only that jlist0 that corresponds to the current jlist.
                for jnum0, ((ISdb0, FSdb0), dxdb0) in enumerate(self.onsagercalculator.jnet0_indexed[
                                                       self.onsagercalculator.om1types[jt]]):
                    j0 = self.onsagercalculator.jnet0[self.onsagercalculator.om1types[jt]][jnum0]
                    if np.allclose(dxdb0, np.zeros(self.onsagercalculator.crys.dim)):
                        condc = (j0.c1 == jmp.c1 and j0.c2 == jmp.c2) or (-j0.c1 == jmp.c1 and -j0.c2 == jmp.c2)
                    else:
                        condc = j0.c1 == jmp.c1 and j0.c2 == jmp.c2
                    if ISdb == ISdb0 and FSdb == FSdb0 and np.allclose(dx, dxdb0) and condc:
                        # check that the omega0 and omega1 jumps are consistent
                        self.assertEqual(j0.state1.iorind, jmp.state1.db.iorind)
                        self.assertEqual(j0.state2.iorind, jmp.state2.db.iorind)
                        # self.assertEqual(j0.c1, jmp.c1)
                        # self.assertEqual(j0.c2, jmp.c2)
                        dx0 = DB_disp(self.onsagercalculator.pdbcontainer, j0.state1, j0.state2)
                        self.assertTrue(np.allclose(dx0, dxdb0))
                        self.assertTrue(np.allclose(dx0, dx))
                        rate01 = rate0list[self.onsagercalculator.om1types[jt]][jnum0]
                        newlist.append(rate01)
                        count += 1
                self.assertEqual(count, 1)
                self.assertFalse(rate01 is None)  # there must be exactly one omega0 jump for an omega1 jump.
            rate10list.append(newlist)

        # Get the omega1 contribution to the non-local bias vectors
        # First check that there is no movement in the solutes
        # self.assertTrue(np.allclose(self.onsagercalculator.bias1_solute_new,
        #                             np.zeros_like(self.onsagercalculator.bias1_solute_new)))

        vel10_solvent_new_vs = \
            np.array([np.dot(self.onsagercalculator.bias1_solvent_new[i, :], rate10_stars[i, :])
                      for i in range(self.onsagercalculator.vkinetic.Nvstars_pure)])

        # Get the new biases in the cartesian basis.
        solvent_vel_10_new = np.zeros((len(self.onsagercalculator.vkinetic.starset.complexStates), self.onsagercalculator.crys.dim))
        for i, state in enumerate(self.onsagercalculator.vkinetic.starset.complexStates):
            try:
                indlist = self.onsagercalculator.vkinetic.stateToVecStar_pure[state]
            except KeyError:
                indlist = []
                self.assertTrue(state.is_zero(self.onsagercalculator.pdbcontainer))
            # We have indlist as (IndOfStar, IndOfState)
            solvent_vel_10_new[i, :] = sum([vel10_solvent_new_vs[tup[0]] *
                                           self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])

        # Calculate the updated bias explicitly
        vel10solvent = np.zeros((len(self.onsagercalculator.vkinetic.starset.complexStates), self.onsagercalculator.crys.dim))
        for jt, jlist, jindlist in zip(itertools.count(), self.onsagercalculator.jnet1,
                                       self.onsagercalculator.jnet1_indexed):
            for jnum, ((IS, FS), dx), jmp in zip(itertools.count(), jindlist, jlist):
                vel10solvent[IS, :] += rate10list[jt][jnum]*dx

        # Now, update with eta vectors
        for jt, jlist, jindlist in zip(itertools.count(), self.onsagercalculator.jnet1,
                                     self.onsagercalculator.jnet1_indexed):
            for jnum, ((IS, FS), dx), jmp in zip(itertools.count(), jindlist, jlist):
                vel10solvent[IS, :] += rate10list[jt][jnum]*(self.onsagercalculator.eta00_solvent[IS]-
                                                             self.onsagercalculator.eta00_solvent[FS])

        # Check that we have the same vectors using the W01list as well
        self.assertTrue(np.allclose(vel10solvent, solvent_vel_10_new))

        # Now, check that the proper states leave zero non-local bias vectors.
        for i in range(len(self.onsagercalculator.vkinetic.starset.complexStates)):
            if elim_list[i] == 0:  # if no omega0 jumps have been eliminated
                self.assertTrue(np.allclose(vel10solvent[i], np.zeros(self.onsagercalculator.crys.dim)))
            if elim_list[i]!=0:
                self.assertFalse(np.allclose(vel10solvent[i], np.zeros(self.onsagercalculator.crys.dim)))

        # Now, do it for omega2

        Nvstars_pure = self.onsagercalculator.vkinetic.Nvstars_pure
        mstartind = self.onsagercalculator.kinetic.mixedstartindex
        #
        # bias2soluteExp, bias2solventExp = self.biases[2]
        #
        # vel2_solute_vs = np.array([np.dot(bias2soluteExp[i - Nvstars_pure, :],
        #                                   rate2_wycks[self.onsagercalculator.vkinetic.vstar2star[i] - mstartind, :])
        #                            for i in range(Nvstars_pure, self.onsagercalculator.vkinetic.Nvstars)])
        #
        # vel2_solvent_vs = np.array([np.dot(bias2solventExp[i - Nvstars_pure, :],
        #                                    rate2_wycks[self.onsagercalculator.vkinetic.vstar2star[i] - mstartind, :])
        #                            for i in range(Nvstars_pure, self.onsagercalculator.vkinetic.Nvstars)])
        #
        # # Now, convert this into the Nstates x 3 form in the mixed state space - write a function to generalize this
        # # later on
        # solute_vel_2 = np.zeros((len(self.onsagercalculator.vkinetic.starset.mixedstates), self.onsagercalculator.crys.dim))
        # solvent_vel_2 = np.zeros((len(self.onsagercalculator.vkinetic.starset.mixedstates), self.onsagercalculator.crys.dim))
        # for i, state in enumerate(self.onsagercalculator.vkinetic.starset.mixedstates):
        #     indlist = self.onsagercalculator.vkinetic.stateToVecStar_mixed[state]
        #     # We have indlist as (IndOfStar, IndOfState)
        #     solute_vel_2[i,:] = sum([vel2_solute_vs[tup[0] - Nvstars_pure] *
        #                              self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])
        #     solvent_vel_2[i,:] = sum([vel2_solvent_vs[tup[0] - Nvstars_pure] *
        #                               self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])
        # # Next, manually update with the eta0 vectors
        # # for i in range(len(self.onsagercalculator.vkinetic.starset.mixedstates)):
        # for jt,jlist in enumerate(self.onsagercalculator.jnet2_indexed):
        #     for jnum, ((IS,FS),dx) in enumerate(jlist):
        #         # if i==IS:
        #         solute_vel_2[IS, :] += rate2list[jt][jnum] * (self.onsagercalculator.eta02_solute[IS] -
        #                                                       self.onsagercalculator.eta02_solute[FS])
        #         solvent_vel_2[IS, :] += rate2list[jt][jnum] * (self.onsagercalculator.eta02_solvent[IS] -
        #                                                        self.onsagercalculator.eta02_solvent[FS])
        #
        # # Now, get the version from the updated expansion
        # vel2_solute_new_vs = np.array([np.dot(self.onsagercalculator.bias2_solute_new[i - Nvstars_pure, :],
        #                                           rate2_wycks[self.onsagercalculator.vkinetic.vstar2star[i] - mstartind]
        #                                           ) for i in range(Nvstars_pure,
        #                                                            self.onsagercalculator.vkinetic.Nvstars)])
        #
        # vel2_solvent_new_vs = np.array([np.dot(self.onsagercalculator.bias2_solvent_new[i - Nvstars_pure, :],
        #                                           rate2_wycks[self.onsagercalculator.vkinetic.vstar2star[i] - mstartind]
        #                                           ) for i in range(Nvstars_pure,
        #                                                            self.onsagercalculator.vkinetic.Nvstars)])
        #
        # solute_vel_2_new = np.zeros((len(self.onsagercalculator.vkinetic.starset.mixedstates), self.onsagercalculator.crys.dim))
        # solvent_vel_2_new = np.zeros((len(self.onsagercalculator.vkinetic.starset.mixedstates), self.onsagercalculator.crys.dim))
        # for i, state in enumerate(self.onsagercalculator.vkinetic.starset.mixedstates):
        #     indlist = self.onsagercalculator.vkinetic.stateToVecStar_mixed[state]
        #     # We have indlist as (IndOfStar, IndOfState)
        #     solute_vel_2_new[i, :] = sum([vel2_solute_new_vs[tup[0] - Nvstars_pure] *
        #                                   self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])
        #
        #     solvent_vel_2_new[i, :] = sum([vel2_solvent_new_vs[tup[0] - Nvstars_pure] *
        #                                    self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])
        #
        # self.assertTrue(np.allclose(solute_vel_2, solute_vel_2_new))
        # self.assertTrue(np.allclose(solvent_vel_2, solvent_vel_2_new))
        # # The following tests must hold - the non-local biases in omega2_space must become zero after eta updates
        # self.assertTrue(np.allclose(solute_vel_2_new, np.zeros_like(solute_vel_2)), msg="\n{}\n".format(solute_vel_2))
        # self.assertTrue(np.allclose(solvent_vel_2_new, np.zeros_like(solvent_vel_2)))

        # Now, do it for omega3
        bias3solute, bias3solvent = self.biases[3]
        self.assertTrue(bias3solute is None)
        # self.assertTrue(np.allclose(bias3solute, 0))

        # vel3_solute_vs = np.array([np.dot(bias3solute[i - Nvstars_pure, :], rate3_stars[i - Nvstars_pure, :])
        #                           for i in range(Nvstars_pure, self.onsagercalculator.vkinetic.Nvstars)])
        # self.assertTrue(np.allclose(vel3_solute_vs, 0))

        vel3_solvent_vs = np.array([np.dot(bias3solvent[i - Nvstars_pure, :], rate3_stars[i - Nvstars_pure, :])
                                   for i in range(Nvstars_pure, self.onsagercalculator.vkinetic.Nvstars)])

        # Now, convert this into the Nstates x 3 form in the mixed state space
        # solute_vel_3 = np.zeros((len(self.onsagercalculator.vkinetic.starset.mixedstates), self.onsagercalculator.crys.dim))
        solvent_vel_3 = np.zeros((len(self.onsagercalculator.vkinetic.starset.mixedstates), self.onsagercalculator.crys.dim))
        for i, state in enumerate(self.onsagercalculator.vkinetic.starset.mixedstates):
            indlist = self.onsagercalculator.vkinetic.stateToVecStar_mixed[state]
            # We have indlist as (IndOfStar, IndOfState)
            # solute_vel_3[i, :] = sum([vel3_solute_vs[tup[0] - Nvstars_pure] *
            #                           self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])
            solvent_vel_3[i, :] = sum([vel3_solvent_vs[tup[0] - Nvstars_pure] *
                                       self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])
        # Next, manually update with the eta0 vectors
        for jt, jlist in enumerate(self.onsagercalculator.jnet3_indexed):
            for jnum, ((IS, FS), dx) in enumerate(jlist):
                # if i = =IS:
                # solute_vel_3[IS, :] += rate3list[jt][jnum] * (-self.onsagercalculator.eta00_solute[FS]) # solute is zero
                solvent_vel_3[IS, :] += rate3list[jt][jnum] * (- self.onsagercalculator.eta00_solvent[FS]) # self.onsagercalculator.eta02_solvent[IS]

        # Now, get the version from the updated expansion
        # vel3_solute_new_vs = np.array([np.dot(self.onsagercalculator.bias3_solute_new[i - Nvstars_pure, :],
        #                                       rate3_stars[i - Nvstars_pure, :])
        #                                for i in range(Nvstars_pure, self.onsagercalculator.vkinetic.Nvstars)])
        vel3_solvent_new_vs = np.array([np.dot(self.onsagercalculator.bias3_solvent_new[i - Nvstars_pure, :],
                                               rate3_stars[i - Nvstars_pure, :])
                                        for i in range(Nvstars_pure, self.onsagercalculator.vkinetic.Nvstars)])

        # solute_vel_3_new = np.zeros((len(self.onsagercalculator.vkinetic.starset.mixedstates), self.onsagercalculator.crys.dim))
        solvent_vel_3_new = np.zeros((len(self.onsagercalculator.vkinetic.starset.mixedstates), self.onsagercalculator.crys.dim))

        for i, state in enumerate(self.onsagercalculator.vkinetic.starset.mixedstates):
            indlist = self.onsagercalculator.vkinetic.stateToVecStar_mixed[state]
            # We have indlist as (IndOfStar, IndOfState)
            # solute_vel_3_new[i, :] = sum([vel3_solute_new_vs[tup[0] - Nvstars_pure] *
            #                               self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])
            solvent_vel_3_new[i, :] = sum([vel3_solvent_new_vs[tup[0] - Nvstars_pure] *
                                           self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])

        # self.assertTrue(np.allclose(solute_vel_3, solute_vel_3_new))
        self.assertTrue(np.allclose(solvent_vel_3, solvent_vel_3_new))

        #Now, do it for omega4
        bias4solute, bias4solvent = self.biases[4]
        bias3solute, bias3solvent = self.biases[3]

        self.assertTrue(bias4solute is None)

        # vel4_solute_vs = np.array([np.dot(bias4solute[i, :], rate4_stars[i, :]) for i in range(Nvstars_pure)])

        vel4_solvent_vs = np.array([np.dot(bias4solvent[i, :], rate4_stars[i, :]) for i in range(Nvstars_pure)])

        # Now, convert this into the Nstates x 3 form in the mixed state space
        # solute_vel_4 = np.zeros((len(self.onsagercalculator.vkinetic.starset.complexStates), self.onsagercalculator.crys.dim))
        solvent_vel_4 = np.zeros((len(self.onsagercalculator.vkinetic.starset.complexStates), self.onsagercalculator.crys.dim))

        for i, state in enumerate(self.onsagercalculator.vkinetic.starset.complexStates):
            try:
                indlist = self.onsagercalculator.vkinetic.stateToVecStar_pure[state]
            except KeyError:
                indlist = []
                self.assertTrue(state.is_zero(self.onsagercalculator.pdbcontainer))

            # solute_vel_4[i, :] = sum([vel4_solute_vs[tup[0]] * self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]]
            #                          for tup in indlist])
            solvent_vel_4[i, :] = sum([vel4_solvent_vs[tup[0]] * self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]]
                                       for tup in indlist])

        # check against explicit evaluation
        # solute_vel_4_direct = np.zeros((len(self.onsagercalculator.vkinetic.starset.complexStates),
        #                                 self.onsagercalculator.crys.dim))
        solvent_vel_4_direct = np.zeros((len(self.onsagercalculator.vkinetic.starset.complexStates),
                                         self.onsagercalculator.crys.dim))
        for jt, jlist in enumerate(self.onsagercalculator.jnet4_indexed):
            for jnum, ((IS, FS), dx) in enumerate(jlist):
                # or2 = self.onsagercalculator.mdbcontainer.iorlist[FS][1]
                # dx_solute = np.zeros(self.onsagercalculator.crys.dim)  # or2/2.
                dx_solvent = dx  # - or2/2.
                # solute_vel_4_direct[IS, :] += rate4list[jt][jnum] * dx_solute
                solvent_vel_4_direct[IS, :] += rate4list[jt][jnum] * dx_solvent
        # self.assertTrue(np.allclose(solute_vel_4_direct, solute_vel_4))
        self.assertTrue(np.allclose(solvent_vel_4_direct, solvent_vel_4))
        # Next, manually update with the eta0 vectors
        for jt,jlist in enumerate(self.onsagercalculator.jnet4_indexed):
            for jnum, ((IS, FS), dx) in enumerate(jlist):
                # solute_vel_4[IS, :] += rate4list[jt][jnum] * self.onsagercalculator.eta00_solute[IS] #-
                                                              #self.onsagercalculator.eta02_solute[FS])
                solvent_vel_4[IS, :] += rate4list[jt][jnum] * self.onsagercalculator.eta00_solvent[IS] #-
                                                               #self.onsagercalculator.eta02_solvent[FS])

        # Now, get the version from the updated expansion
        # vel4_solute_new_vs = np.array([np.dot(self.onsagercalculator.bias4_solute_new[i, :], rate4_stars[i, :])
        #                            for i in range(Nvstars_pure)])
        vel4_solvent_new_vs = np.array([np.dot(self.onsagercalculator.bias4_solvent_new[i, :], rate4_stars[i, :])
                                    for i in range(Nvstars_pure)])

        # solute_vel_4_new = np.zeros((len(self.onsagercalculator.vkinetic.starset.complexStates), self.onsagercalculator.crys.dim))
        solvent_vel_4_new = np.zeros((len(self.onsagercalculator.vkinetic.starset.complexStates), self.onsagercalculator.crys.dim))
        for i, state in enumerate(self.onsagercalculator.vkinetic.starset.complexStates):
            try:
                indlist = self.onsagercalculator.vkinetic.stateToVecStar_pure[state]
            except KeyError:
                indlist = []
                self.assertTrue(state.is_zero(self.onsagercalculator.pdbcontainer))

            # solute_vel_4_new[i, :] = sum([vel4_solute_new_vs[tup[0]] *
            #                               self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])
            solvent_vel_4_new[i, :] = sum([vel4_solvent_new_vs[tup[0]] *
                                           self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])

        # self.assertTrue(np.allclose(solute_vel_4, solute_vel_4_new))
        self.assertTrue(np.allclose(solvent_vel_4, solvent_vel_4_new))

    def test_uncorrelated_del_om(self):
        """
        Test the uncorrelated contribution to diffusivity part by part.
        Also in the process check the omega rate list creation and everything.
        """

        dim = self.onsagercalculator.crys.dim

        # First, we need some thermodynamic data
        # We randomize site and transition energies for now.

        # Set up energies and pre-factors
        kT = 1.

        predb0, enedb0 = np.ones(len(self.onsagercalculator.vkinetic.starset.pdbcontainer.symorlist)), \
                         np.random.rand(len(self.onsagercalculator.vkinetic.starset.pdbcontainer.symorlist))

        preS, eneS = np.ones(
            len(self.onsagercalculator.vkinetic.starset.crys.sitelist(self.onsagercalculator.vkinetic.starset.chem))), \
                     np.random.rand(len(self.onsagercalculator.vkinetic.starset.crys.sitelist(
                         self.onsagercalculator.vkinetic.starset.chem)))

        # These are the interaction or the excess energies and pre-factors for solutes and dumbbells.
        preSdb, eneSdb = np.ones(self.onsagercalculator.thermo.mixedstartindex), \
                         np.random.rand(self.onsagercalculator.thermo.mixedstartindex)

        predb2, enedb2 = np.ones(len(self.onsagercalculator.vkinetic.starset.mdbcontainer.symorlist)), \
                         np.random.rand(len(self.onsagercalculator.vkinetic.starset.mdbcontainer.symorlist))

        preT0, eneT0 = np.ones(len(self.onsagercalculator.vkinetic.starset.jnet0)), np.random.rand(
            len(self.onsagercalculator.vkinetic.starset.jnet0))
        preT2, eneT2 = np.ones(len(self.onsagercalculator.vkinetic.starset.jnet2)), np.random.rand(
            len(self.onsagercalculator.vkinetic.starset.jnet2))
        preT1, eneT1 = np.ones(len(self.onsagercalculator.jnet1)), np.random.rand(len(self.onsagercalculator.jnet1))

        preT43, eneT43 = np.ones(len(self.onsagercalculator.jnet43)), \
                         np.random.rand(len(self.onsagercalculator.jnet43))

        # Now get the beta*free energy values.
        bFdb0, bFdb2, bFS, bFSdb, bFT0, bFT1, bFT2, bFT3, bFT4 =\
            self.onsagercalculator.preene2betafree(kT, predb0, enedb0, preS, eneS, preSdb, eneSdb, predb2, enedb2,
                                                   preT0, eneT0, preT2, eneT2, preT1, eneT1, preT43, eneT43)

        self.onsagercalculator.L_ij(bFdb0, bFT0, bFdb2, bFT2, bFS, bFSdb, bFT1, bFT3, bFT4)

        (omega0, omega0escape), (omega1, omega1escape), (omega2, omega2escape), (omega3, omega3escape),\
        (omega4, omega4escape) = self.onsagercalculator.omegas

        # First, check the omega1 rates coming out of origin states are zero
        for jt, rate in enumerate(omega1):
            if self.onsagercalculator.jnet1[jt][0].state1.is_zero(self.onsagercalculator.pdbcontainer) \
                    or self.onsagercalculator.jnet1[jt][0].state2.is_zero(self.onsagercalculator.pdbcontainer):
                self.assertEqual(rate, 0.)

        # eta0total_solute = self.onsagercalculator.eta0total_solute
        eta0total_solvent = self.onsagercalculator.eta0total_solvent
        # Now, let's get the bias expansions
        D0expansion_bb, (D1expansion_aa, D1expansion_bb, D1expansion_ab), \
        (D2expansion_aa, D2expansion_bb, D2expansion_ab), \
        (D3expansion_aa, D3expansion_bb, D3expansion_ab), \
        (D4expansion_aa, D4expansion_bb, D4expansion_ab) =\
            self.onsagercalculator.bareExpansion(eta0total_solvent)

        complex_prob, mixed_prob = self.onsagercalculator.pr_states

        # Now set up the multiplicative quantity for each jump type.
        prob_om1 = np.zeros(len(omega1))
        for jt, ((IS, FS), dx) in enumerate([jlist[0] for jlist in self.onsagercalculator.jnet1_indexed]):
            prob_om1[jt] = np.sqrt(complex_prob[IS] * complex_prob[FS]) * omega1[jt]

        prob_om2 = np.zeros(len(self.onsagercalculator.jnet2))
        for jt, ((IS, FS), dx) in enumerate([jlist[0] for jlist in self.onsagercalculator.jnet2_indexed]):
            prob_om2[jt] = np.sqrt(mixed_prob[IS] * mixed_prob[FS]) * omega2[jt]

        prob_om4 = np.zeros(len(self.onsagercalculator.jnet4))
        for jt, ((IS, FS), dx) in enumerate([jlist[0] for jlist in self.onsagercalculator.jnet4_indexed]):
            prob_om4[jt] = np.sqrt(complex_prob[IS] * mixed_prob[FS]) * omega4[jt]

        prob_om3 = np.zeros(len(self.onsagercalculator.jnet3))
        for jt, ((IS, FS), dx) in enumerate([jlist[0] for jlist in self.onsagercalculator.jnet3_indexed]):
            prob_om3[jt] = np.sqrt(mixed_prob[IS] * complex_prob[FS]) * omega3[jt]

        # Now, let's compute the contribution by omega1 jumps
        # For solutes, it's zero anyway - let's check for solvents
        L_uc_om1_test_solvent = np.zeros((dim, dim))
        for jt, jlist in enumerate(self.onsagercalculator.jnet1_indexed):
            for (IS, FS), dx in jlist:
                L_uc_om1_test_solvent += np.outer(dx + eta0total_solvent[IS] - eta0total_solvent[FS],
                                                  dx + eta0total_solvent[IS] - eta0total_solvent[FS])*prob_om1[jt]*0.5
        L_uc_om1 = np.dot(D1expansion_bb, prob_om1)
        self.assertTrue(np.allclose(L_uc_om1_test_solvent, L_uc_om1))

        # Now, let's check the omega2 contributions
        L_uc_om2_bb = np.dot(D2expansion_bb, prob_om2)
        L_uc_om2_aa = np.dot(D2expansion_aa, prob_om2)
        L_uc_om2_ab = np.dot(D2expansion_ab, prob_om2)
        L_uc_om2_test_aa = np.zeros((dim, dim))
        L_uc_om2_test_bb = np.zeros((dim, dim))
        L_uc_om2_test_ab = np.zeros((dim, dim))
        Ncomp = len(self.onsagercalculator.vkinetic.starset.complexStates)
        for jt, jlist in enumerate(self.onsagercalculator.jnet2_indexed):
            for (IS, FS), dx in jlist:
                # o1 = self.onsagercalculator.mdbcontainer.iorlist[
                #     self.onsagercalculator.vkinetic.starset.mixedstates[IS].db.iorind][1]
                # o2 = self.onsagercalculator.mdbcontainer.iorlist[
                #     self.onsagercalculator.vkinetic.starset.mixedstates[FS].db.iorind][1]

                dx_solute = dx #+ eta0total_solute[IS + Ncomp] - eta0total_solute[FS + Ncomp]  #- o1/2. + o2/2.
                dx_solvent = dx + eta0total_solvent[IS + Ncomp] - eta0total_solvent[FS + Ncomp]  #+ o1/2. - o2/2.

                L_uc_om2_test_aa += np.outer(dx_solute, dx_solute)* prob_om2[jt] * 0.5
                L_uc_om2_test_bb += np.outer(dx_solvent, dx_solvent) * prob_om2[jt] * 0.5
                L_uc_om2_test_ab += np.outer(dx_solute, dx_solvent) * prob_om2[jt] * 0.5

        self.assertTrue(np.allclose(L_uc_om2_test_aa, L_uc_om2_aa))
        self.assertTrue(np.allclose(L_uc_om2_test_bb, L_uc_om2_bb))
        self.assertTrue(np.allclose(L_uc_om2_test_ab, L_uc_om2_ab))

        # Now, let's check the omega3 contributions
        L_uc_om3_bb = np.dot(D3expansion_bb, prob_om3)
        L_uc_om3_aa = np.dot(D3expansion_aa, prob_om3)
        L_uc_om3_ab = np.dot(D3expansion_ab, prob_om3)
        L_uc_om3_test_aa = np.zeros((dim, dim))
        L_uc_om3_test_bb = np.zeros((dim, dim))
        L_uc_om3_test_ab = np.zeros((dim, dim))

        for jt, jlist in enumerate(self.onsagercalculator.jnet3_indexed):
            # The initial state is a  mixed dumbbell and the final is a pure dumbbell
            sm_aa = np.zeros((dim, dim))
            sm_bb = np.zeros((dim, dim))
            sm_ab = np.zeros((dim, dim))
            for (IS, FS), dx in jlist:
                # o1 = self.onsagercalculator.mdbcontainer.iorlist[
                #     self.onsagercalculator.vkinetic.starset.mixedstates[IS].db.iorind][1]

                # dx_solute = eta0total_solute[IS + Ncomp] - eta0total_solute[FS]  # -o1/2.
                dx_solvent = dx + eta0total_solvent[IS + Ncomp] - eta0total_solvent[FS]  # o1/2.
                # sm_aa += zeroclean(np.outer(dx_solute, dx_solute)) * 0.5
                sm_bb += zeroclean(np.outer(dx_solvent, dx_solvent)) * 0.5
                # sm_ab += zeroclean(np.outer(dx_solute, dx_solvent)) * 0.5
                # L_uc_om3_test_aa += zeroclean(np.outer(dx_solute, dx_solute)) * prob_om3[jt] * 0.5
                L_uc_om3_test_bb += zeroclean(np.outer(dx_solvent, dx_solvent)) * prob_om3[jt] * 0.5
                # L_uc_om3_test_ab += zeroclean(np.outer(dx_solute, dx_solvent)) * prob_om3[jt] * 0.5
            self.assertTrue(np.allclose(D3expansion_aa[:, :, jt], sm_aa), msg="{}".format(jt))

        self.assertTrue(np.allclose(L_uc_om3_test_aa, L_uc_om3_aa), msg="\n {} \n {}".format(L_uc_om3_test_aa, L_uc_om3_aa))
        self.assertTrue(np.allclose(L_uc_om3_test_bb, L_uc_om3_bb))
        self.assertTrue(np.allclose(L_uc_om3_test_ab, L_uc_om3_ab))

        # Now, let's check the omega4 contributions
        L_uc_om4_bb = np.dot(D4expansion_bb, prob_om4)
        L_uc_om4_aa = np.dot(D4expansion_aa, prob_om4)
        L_uc_om4_ab = np.dot(D4expansion_ab, prob_om4)
        L_uc_om4_test_aa = np.zeros((dim, dim))
        L_uc_om4_test_bb = np.zeros((dim, dim))
        L_uc_om4_test_ab = np.zeros((dim, dim))

        for jt, jlist in enumerate(self.onsagercalculator.jnet4_indexed):
            # The initial state is a  pure dumbbell and the final is a mixed dumbbell
            for (IS, FS), dx in jlist:
                # o1 = self.onsagercalculator.pdbcontainer.iorlist[
                #     self.onsagercalculator.vkinetic.starset.complexStates[IS].db.iorind][1]
                # o2 = self.onsagercalculator.mdbcontainer.iorlist[
                #     self.onsagercalculator.vkinetic.starset.mixedstates[FS].db.iorind][1]

                # dx_solute = eta0total_solute[IS] - eta0total_solute[FS + Ncomp]  # o2 / 2.
                dx_solvent = dx + eta0total_solvent[IS] - eta0total_solvent[FS + Ncomp]  # -o2 / 2.
                # L_uc_om4_test_aa += np.outer(dx_solute, dx_solute) * prob_om4[jt] * 0.5
                L_uc_om4_test_bb += np.outer(dx_solvent, dx_solvent) * prob_om4[jt] * 0.5
                # L_uc_om4_test_ab += np.outer(dx_solute, dx_solvent) * prob_om4[jt] * 0.5

        self.assertTrue(np.allclose(L_uc_om4_test_aa, L_uc_om4_aa))
        self.assertTrue(np.allclose(L_uc_om4_test_bb, L_uc_om4_bb))
        self.assertTrue(np.allclose(L_uc_om4_test_ab, L_uc_om4_ab))

    # construct a test for the way the rates are constructed.

    def test_Lij(self):
        """
        This is to check the delta omega matrix and the GF expansions
        """
        # 1.  First get the rates and thermodynamic data.

        # 1a. Energies and pre-factors
        kT = 1.

        predb0, enedb0 = np.ones(len(self.onsagercalculator.vkinetic.starset.pdbcontainer.symorlist)), \
                         np.random.rand(len(self.onsagercalculator.vkinetic.starset.pdbcontainer.symorlist))

        preS, eneS = np.ones(
            len(self.onsagercalculator.vkinetic.starset.crys.sitelist(self.onsagercalculator.vkinetic.starset.chem))), \
                     np.random.rand(len(self.onsagercalculator.vkinetic.starset.crys.sitelist(
                         self.onsagercalculator.vkinetic.starset.chem)))

        # These are the interaction or the excess energies and pre-factors for solutes and dumbbells.
        preSdb, eneSdb = np.ones(self.onsagercalculator.thermo.mixedstartindex), \
                         np.random.rand(self.onsagercalculator.thermo.mixedstartindex)

        predb2, enedb2 = np.ones(len(self.onsagercalculator.vkinetic.starset.mdbcontainer.symorlist)), \
                         np.random.rand(len(self.onsagercalculator.vkinetic.starset.mdbcontainer.symorlist))

        preT0, eneT0 = np.ones(len(self.onsagercalculator.vkinetic.starset.jnet0)), np.random.rand(
            len(self.onsagercalculator.vkinetic.starset.jnet0))
        preT2, eneT2 = np.ones(len(self.onsagercalculator.vkinetic.starset.jnet2)), np.random.rand(
            len(self.onsagercalculator.vkinetic.starset.jnet2))
        preT1, eneT1 = np.ones(len(self.onsagercalculator.jnet1)), np.random.rand(len(self.onsagercalculator.jnet1))

        preT43, eneT43 = np.ones(len(self.onsagercalculator.jnet43)), \
                         np.random.rand(len(self.onsagercalculator.jnet43))

        # 1c. Now get the beta*free energy values.
        bFdb0, bFdb2, bFS, bFSdb, bFT0, bFT1, bFT2, bFT3, bFT4 = \
            self.onsagercalculator.preene2betafree(kT, predb0, enedb0, preS, eneS, preSdb, eneSdb, predb2, enedb2,
                                                   preT0, eneT0, preT2, eneT2, preT1, eneT1, preT43, eneT43)

        bFdb0_min = np.min(bFdb0)
        bFdb2_min = np.min(bFdb2)
        bFS_min = np.min(bFS)

        # 1d. Modify the solute-dumbell interaction energies
        bFSdb_total = np.zeros(self.onsagercalculator.vkinetic.starset.mixedstartindex)
        bFSdb_total_shift = np.zeros(self.onsagercalculator.vkinetic.starset.mixedstartindex)

        # first, just add up the solute and dumbbell energies. We will add in the corrections to the thermo shell states
        # later.
        # Also, we need to keep an unshifted version to be able to normalize probabilities
        for starind, star in enumerate(self.onsagercalculator.vkinetic.starset.stars[
                                       :self.onsagercalculator.vkinetic.starset.mixedstartindex]):
            # For origin complex states, do nothing - leave them as zero.
            if star[0].is_zero(self.onsagercalculator.vkinetic.starset.pdbcontainer):
                continue
            symindex = self.onsagercalculator.vkinetic.starset.star2symlist[starind]
            # First, get the unshifted value
            # check that invmaps are okay
            solwyck = self.onsagercalculator.invmap_solute[star[0].i_s]
            for state in star:
                self.assertEqual(self.onsagercalculator.invmap_solute[state.i_s], solwyck)

            bFSdb_total[starind] = bFdb0[symindex] + bFS[solwyck]
            bFSdb_total_shift[starind] = bFSdb_total[starind] - (bFdb0_min + bFS_min)

        # Now add in the changes for the complexes inside the thermodynamic shell.
        for starind, star in enumerate(self.onsagercalculator.thermo.stars[
                                       :self.onsagercalculator.thermo.mixedstartindex]):
            # Get the symorlist index for the representative state of the star
            if star[0].is_zero(self.onsagercalculator.thermo.pdbcontainer):
                continue
            # keep the total energies zero for origin states.

            # check thermo2kin
            kinind = self.onsagercalculator.thermo2kin[starind]
            self.assertEqual(len(star), len(self.onsagercalculator.vkinetic.starset.stars[kinind]))
            count = 0
            for state in star:
                for statekin in self.onsagercalculator.vkinetic.starset.stars[kinind]:
                    if state == statekin:
                        count += 1
            self.assertEqual(count, len(star))

            bFSdb_total[kinind] += bFSdb[starind]
            bFSdb_total_shift[kinind] += bFSdb[starind]

        print("Passed tests 1 - making complex energies")

        # 2. Next, we get all the relevant data from the L_ij function.
        self.onsagercalculator.L_ij(bFdb0, bFT0, bFdb2, bFT2, bFS, bFSdb, bFT1, bFT3, bFT4)

        GF20 = self.onsagercalculator.GF02
        del_om = self.onsagercalculator.del_om
        part_func = self.onsagercalculator.part_func
        omegas = self.onsagercalculator.omegas
        pr_states = self.onsagercalculator.pr_states

        # 2a - get the symmetrized and escape rates.
        (omega0, omega0escape), (omega1, omega1escape), (omega2, omega2escape), (omega3, omega3escape),\
        (omega4, omega4escape) = omegas
        # 2a.1 - check equivalence of symmetrized omega3 and 4 rates
        for jt in range(len(self.onsagercalculator.jnet43)):
            self.assertEqual(omega4[jt], omega3[jt])

        # 2a.2 - check consistency of non-local rates
        pre0, pre0T = np.ones_like(bFdb0), np.ones_like(bFT0)
        pre2, pre2T = np.ones_like(bFdb2), np.ones_like(bFT2)

        symrate0list = symmratelist(self.onsagercalculator.jnet0_indexed, pre0, bFdb0 - bFdb0_min, pre0T, bFT0,
                                    self.onsagercalculator.vkinetic.starset.pdbcontainer.invmap)

        symrate2list = symmratelist(self.onsagercalculator.jnet2_indexed, pre2, bFdb2 - bFdb2_min, pre2T, bFT2,
                                    self.onsagercalculator.vkinetic.starset.mdbcontainer.invmap)
        for jt in range(len(self.onsagercalculator.jnet0)):
            self.assertEqual(symrate0list[jt][0], omega0[jt])

        for jt in range(len(self.onsagercalculator.jnet2)):
            self.assertEqual(symrate2list[jt][0], omega2[jt])

        print("passed tests in 2 - checking non-local rate consistencies")

        # 2b - get the state probabilities and check that all states in a vector star have the same probability
        Nvstars = self.onsagercalculator.vkinetic.Nvstars
        Nvstars_pure = self.onsagercalculator.vkinetic.Nvstars_pure
        Nvstars_mixed = Nvstars - Nvstars_pure
        complex_prob, mixed_prob = pr_states
        for vp in self.onsagercalculator.vkinetic.vecpos[:Nvstars_pure]:
            prob0 = complex_prob[self.onsagercalculator.kinetic.complexIndexdict[vp[0]][0]]
            # check that all the other states have the same energy
            for state in vp:
                stateind = self.onsagercalculator.kinetic.complexIndexdict[state][0]
                self.assertEqual(complex_prob[stateind], prob0)
        for vp in self.onsagercalculator.vkinetic.vecpos[Nvstars_pure:]:
            prob0 = mixed_prob[self.onsagercalculator.kinetic.mixedindexdict[vp[0]][0]]
            for state in vp:
                stateind = self.onsagercalculator.kinetic.mixedindexdict[state][0]
                self.assertEqual(mixed_prob[stateind], prob0)

        # 2c - check that origin states have zero probability
        for i, state in enumerate(self.onsagercalculator.kinetic.complexStates):
            if state.is_zero(self.onsagercalculator.pdbcontainer):
                self.assertTrue(np.allclose(complex_prob[i], 0.))

        # 3. Now, that we have the symmetrized rates, we need to construct the delta_om matrix using it's mathematical
        # form
        # 3a. First, we do the non-diagonal parts
        delta_om_test = np.zeros((Nvstars, Nvstars))
        # 3a.1 - First, we concentrate on the complex-complex block and the omega1 jumps
        for jt, jlist in enumerate(self.onsagercalculator.jnet1):
            delom1 = omega1[jt] - omega0[self.onsagercalculator.om1types[jt]]
            for jmp in jlist:
                try:
                    indlist1 = self.onsagercalculator.vkinetic.stateToVecStar_pure[jmp.state1]
                except KeyError:
                    indlist1 = []
                    self.assertTrue(jmp.state1.is_zero(self.onsagercalculator.pdbcontainer))
                try:
                    indlist2 = self.onsagercalculator.vkinetic.stateToVecStar_pure[jmp.state2]
                except KeyError:
                    indlist2 = []
                    self.assertTrue(jmp.state2.is_zero(self.onsagercalculator.pdbcontainer))

                for vi, invi in indlist1:
                    for vj, invj in indlist2:
                        delta_om_test[vi, vj] += \
                            delom1 * np.dot(self.onsagercalculator.vkinetic.vecvec[vi][invi],
                                            self.onsagercalculator.vkinetic.vecvec[vj][invj])

        # 3a.2 - Next, we consider the contribution by omega3 jumps - mixed to complex
        for jt, jlist in enumerate(self.onsagercalculator.jnet3):
            for jmp in jlist:
                indlist1 = self.onsagercalculator.vkinetic.stateToVecStar_mixed[jmp.state1]
                try:
                    indlist2 = self.onsagercalculator.vkinetic.stateToVecStar_pure[jmp.state2]
                except:
                    indlist2 = []
                    self.assertTrue(jmp.state2.is_zero(self.onsagercalculator.pdbcontainer))

                for vi, invi in indlist1:
                    for vj, invj in indlist2:
                        delta_om_test[vi, vj] += \
                            omega3[jt] * np.dot(self.onsagercalculator.vkinetic.vecvec[vi][invi],
                                                self.onsagercalculator.vkinetic.vecvec[vj][invj])

        # 3a.3 - Next, we consider the contribution by only the omega4 jumps - complex to mixed
        for jt, jlist in enumerate(self.onsagercalculator.jnet4):
            for jmp in jlist:
                try:
                    indlist1 = self.onsagercalculator.vkinetic.stateToVecStar_pure[jmp.state1]
                except:
                    indlist1 = []
                    self.assertTrue(jmp.state1.is_zero(self.onsagercalculator.pdbcontainer))

                indlist2 = self.onsagercalculator.vkinetic.stateToVecStar_mixed[jmp.state2]
                for vi, invi in indlist1:
                    for vj, invj in indlist2:
                        delta_om_test[vi, vj] += \
                            omega4[jt] * np.dot(self.onsagercalculator.vkinetic.vecvec[vi][invi],
                                                self.onsagercalculator.vkinetic.vecvec[vj][invj])

        # 3b - Now, we do the off diagonal parts
        # 3b.1 - contribution by omega1
        diags = np.zeros(Nvstars)
        for jt, jlist in enumerate(self.onsagercalculator.jnet1):
            for jmp in jlist:
                si = jmp.state1

                if si.is_zero(self.onsagercalculator.pdbcontainer):
                    if not omega1[jt] == 0.:
                        raise ValueError
                if jmp.state2.is_zero(self.onsagercalculator.pdbcontainer):
                    if not omega1[jt] == 0.:
                        raise ValueError

                star_i = self.onsagercalculator.vkinetic.starset.complexIndexdict[si][1]
                dbwyck_i = self.onsagercalculator.pdbcontainer.invmap[si.db.iorind]

                try:
                    indlist1 = self.onsagercalculator.vkinetic.stateToVecStar_pure[si]
                except:
                    indlist1 = []
                    self.assertTrue(si.is_zero(self.onsagercalculator.pdbcontainer))

                for vi, invi in indlist1:
                    vec = self.onsagercalculator.vkinetic.vecvec[vi][invi]
                    vdot = np.dot(vec, vec)
                    if jmp.state2.is_zero(self.onsagercalculator.pdbcontainer) or\
                            jmp.state1.is_zero(self.onsagercalculator.pdbcontainer):
                        diags[vi] -= 0 - np.exp(-bFT0[self.onsagercalculator.om1types[jt]]
                                                                + bFdb0[dbwyck_i] - bFdb0_min) * vdot
                    else:
                        diags[vi] -= np.exp(-bFT1[jt] + bFSdb_total_shift[star_i]) * vdot - \
                                                     np.exp(-bFT0[self.onsagercalculator.om1types[jt]] + bFdb0[
                                                         dbwyck_i] - bFdb0_min) * vdot

        # 3b.2 - contribution by omega4
        for jt, jlist in enumerate(self.onsagercalculator.jnet4):
            for jmp in jlist:
                si = jmp.state1

                star_i = self.onsagercalculator.vkinetic.starset.complexIndexdict[si][1]
                #         dbwyck_i = self.onsagercalculator.pdbcontainer.invmap[si.db.iorind]
                try:
                    indlist1 = self.onsagercalculator.vkinetic.stateToVecStar_pure[si]
                except:
                    indlist1 = []
                    self.assertTrue(si.is_zero(self.onsagercalculator.pdbcontainer))

                for vi, invi in indlist1:
                    vec = self.onsagercalculator.vkinetic.vecvec[vi][invi]
                    vdot = np.dot(vec, vec)
                    diags[vi] -= np.exp(-bFT4[jt] + bFSdb_total_shift[star_i]) * vdot

        # add the diagonal contributions
        for i in range(Nvstars):
            delta_om_test[i, i] += diags[i]

        self.assertTrue(np.allclose(delta_om_test, del_om))
        print("passed tests 3 - checking delta omega")

        # 4. NEXT WE VERIFY G0
        # 4a. to verify GF20, we need the GF starsets and the GF2 and GF0 lists for each starset
        # setrates has already been run while running makeGF
        GF0 = np.array([self.onsagercalculator.GFcalc_pure(tup[0][0], tup[0][1], tup[1]) for tup in
                        [star[0] for star in self.onsagercalculator.GFstarset_pure]])

        GFstarset_pure, GFPureStarInd = self.onsagercalculator.GFstarset_pure, self.onsagercalculator.GFPureStarInd

        # 4b. Now, we must evaluate the GF20 tensor through explicit summation
        GF0_test = np.zeros((Nvstars_pure, Nvstars_pure))
        # First, we do it for the GF0 part
        for i in range(Nvstars_pure):
            for j in range(Nvstars_pure):
                for si, vi in zip(self.onsagercalculator.vkinetic.vecpos[i], self.onsagercalculator.vkinetic.vecvec[i]):
                    for sj, vj in zip(self.onsagercalculator.vkinetic.vecpos[j],
                                      self.onsagercalculator.vkinetic.vecvec[j]):
                        # try to form a connection between the states
                        try:
                            ds = si ^ sj
                        except:
                            continue
                        # If the connection has formed, locate it's star index
                        Gstarind = GFPureStarInd[ds]
                        GF0_test[i, j] += GF0[Gstarind] * np.dot(vi, vj)

        self.assertTrue(np.allclose(GF20[:Nvstars_pure, :Nvstars_pure], GF0_test))
        print("passed tests 4 - checking non-local Green's function")

        # 5. NEXT, WE TEST THE STATE PROBABILITIES
        for prob in complex_prob:
            self.assertTrue(prob >= 0.)
        for prob in mixed_prob:
            self.assertTrue(prob >= 0.)

        for i, state in enumerate(self.onsagercalculator.vkinetic.starset.complexStates):
            if state.is_zero(self.onsagercalculator.pdbcontainer):
                continue
            starind = self.onsagercalculator.vkinetic.starset.complexIndexdict[state][1]
            self.assertTrue(np.allclose(np.exp(-bFSdb_total[starind]), complex_prob[i] * part_func))

        for i, state in enumerate(self.onsagercalculator.vkinetic.starset.mixedstates):
            wyckind = self.onsagercalculator.mdbcontainer.invmap[state.db.iorind]
            self.assertTrue(np.allclose(np.exp(-bFdb2[wyckind]), mixed_prob[i] * part_func))

        # 5a. Test consistency with omega1 rates
        for jt, jlist in enumerate(self.onsagercalculator.jnet1):
            for jmp in jlist:
                st1 = jmp.state1
                st2 = jmp.state2
                stateind1, starind1 = self.onsagercalculator.vkinetic.starset.complexIndexdict[st1]
                stateind2, starind2 = self.onsagercalculator.vkinetic.starset.complexIndexdict[st2]

                if st1.is_zero(self.onsagercalculator.pdbcontainer):
                    self.assertTrue(np.allclose(complex_prob[stateind1], 0.))
                    continue
                if st2.is_zero(self.onsagercalculator.pdbcontainer):
                    self.assertTrue(np.allclose(complex_prob[stateind2], 0.))
                    continue

                rate = np.exp(-bFT1[jt] + bFSdb_total_shift[starind1])
                symrate = np.sqrt(complex_prob[stateind1]) * rate * 1. / np.sqrt(complex_prob[stateind2])
                self.assertTrue(np.allclose(symrate, omega1[jt]))

        # 5b. Test consistency with omega2 rates
        for jt, jlist in enumerate(self.onsagercalculator.jnet2):
            for jmp in jlist:
                st1 = jmp.state1
                st2 = jmp.state2
                stateind1, wyck1 = self.onsagercalculator.mdbcontainer.db2ind(st1.db),\
                                   self.onsagercalculator.mdbcontainer.invmap[st1.db.iorind]

                stateind2, wyck2 = self.onsagercalculator.mdbcontainer.db2ind(st2.db),\
                                   self.onsagercalculator.mdbcontainer.invmap[st2.db.iorind]

                rate = np.exp(-bFT2[jt] + bFdb2 - bFdb2_min)
                symrate = np.sqrt(mixed_prob[stateind1]) * rate * 1 / np.sqrt(mixed_prob[stateind2])
                self.assertTrue(np.allclose(symrate, omega2[jt]))

        # 5c. Test consistency of omega3 and omega4 rates
        for jt, jlist in enumerate(self.onsagercalculator.jnet3):
            for jmp in jlist:
                st1 = jmp.state1
                st2 = jmp.state2
                stateind1, wyck1 = self.onsagercalculator.mdbcontainer.db2ind(st1.db), \
                                   self.onsagercalculator.mdbcontainer.invmap[st1.db.iorind]

                stateind2, starind2 = self.onsagercalculator.vkinetic.starset.complexIndexdict[st2]

                rate3 = np.exp(-(bFT3[jt] + bFdb2_min) + bFdb2[wyck1])
                symrate3 = np.sqrt(mixed_prob[stateind1]) * rate3 * 1. / np.sqrt(complex_prob[stateind2])
                # if not np.allclose(omega3[jt], symrate3):
                #     print(jt, symrate3, omega3[jt])
                self.assertTrue(np.allclose(omega3[jt], symrate3))

                rate4 = np.exp(-(bFT4[jt] + bFdb0_min + bFS_min) + bFSdb_total[starind2])
                symrate4 = 1. / np.sqrt(mixed_prob[stateind1]) * rate4 * np.sqrt(complex_prob[stateind2])
                self.assertTrue(np.allclose(omega4[jt], symrate4))

                self.assertTrue( np.allclose(bFT4[jt] + bFdb0_min + bFS_min, bFT3[jt] + bFdb2_min))

                self.assertTrue(np.allclose(symrate3, symrate4))

        print("passed tests 5 - checking state probabilities")

        # 6. Next, we test the final bias vector
        Ncomp = len(self.onsagercalculator.kinetic.complexStates)
        Nmix = len(self.onsagercalculator.kinetic.mixedstates)
        bias_true_updated_solute = np.zeros((Ncomp + Nmix, self.onsagercalculator.crys.dim))
        bias_true_updated_solvent = np.zeros((Ncomp + Nmix, self.onsagercalculator.crys.dim))

        # To get the unsymmetrized rate out of a state, we locate it's vector star, and get the unsymmetrized
        # rate from the escape arrays.

        # 6a. First, we do it for the complex states - we'll use the eta vector updates, so that we don't have to worry
        # about excluded omega0 jumps from the kinetic shell states.
        for i in range(Ncomp):
            comp_state = self.onsagercalculator.kinetic.complexStates[i]
            try:
                vstar_indlist = self.onsagercalculator.vkinetic.stateToVecStar_pure[comp_state]
            except KeyError:
                vstar_indlist = []
                self.assertTrue(comp_state.is_zero(self.onsagercalculator.pdbcontainer))

            starind = self.onsagercalculator.kinetic.complexIndexdict[comp_state][1]
            dbwyckind = self.onsagercalculator.kinetic.star2symlist[starind]
            # Also, check the correctness of the omega_escape arrays.
            # if some vector stars contain the same crystal stars (and hence states), then for a given jt, they should
            # also have the same unsymmetrized rates.
            # omega1 jumps lead to no solute updates

            # Do nothing for origin states - bias out of them needs to remain zero.
            if comp_state.is_zero(self.onsagercalculator.pdbcontainer):
                # self.assertTrue(np.allclose(self.onsagercalculator.eta0total_solvent[i], np.zeros(3)))
                # self.assertTrue(np.allclose(self.onsagercalculator.eta0total_solute[i], np.zeros(3)))
                self.assertTrue(np.allclose(complex_prob[i], 0.))
                for jt in range(len(self.onsagercalculator.jnet1)):
                    for tup in vstar_indlist:
                        self.assertTrue(np.allclose(omega1escape[tup[0], jt], 0.))
                continue

            for jt, jlist in enumerate(self.onsagercalculator.jnet1):
                for jnum, jmp in enumerate(jlist):
                    jt0 = self.onsagercalculator.om1types[jt]
                    rate0 = np.exp(- bFT0[jt0] + bFdb0[dbwyckind] - bFdb0_min)
                    rate1 = np.exp(- bFT1[jt] + bFSdb_total_shift[starind])
                    if jmp.state1 == comp_state:
                        if jmp.state2.is_zero(self.onsagercalculator.pdbcontainer):
                            # If the final state is an origin state, then the omega1 rate of that jump is zero,
                            # but delta omega is still non-zero.
                            # And if the initial state is not an origin state, then that means the non-local change
                            # in the bias for that jump is non-zero and is exactly the opposite of the total bias for
                            # that jump.
                            # which means delW1 = -rate0
                            rate1 = 0. #set the omega1 rate to zero
                            for tup in vstar_indlist:
                                self.assertTrue(np.allclose(omega1escape[tup[0], jt], 0.))
                                self.assertTrue(np.allclose(self.onsagercalculator.del_W1[tup[0], jt], -rate0))
                            # continue
                        # next, we need the bare dumbbell transition rate for this jump
                        # the non-local rate is the difference between the two.
                        rate = rate1 - rate0
                        (IS, FS), dx = self.onsagercalculator.jnet1_indexed[jt][jnum]
                        self.assertEqual(dx.shape[0], self.onsagercalculator.crys.dim)
                        self.assertEqual(IS, i)
                        bias_true_updated_solvent[i, :] += rate * np.sqrt(complex_prob[i]) *\
                                                           (dx + self.onsagercalculator.eta0total_solvent[IS] -
                                                            self.onsagercalculator.eta0total_solvent[FS])
                        for tup in vstar_indlist:
                            if not jmp.state2.is_zero(self.onsagercalculator.pdbcontainer):
                                self.assertTrue(np.allclose(omega1escape[tup[0], jt], rate1),
                                                msg="\n{}\n{}\n{}\n{}\n{}".format(rate1, omega1escape[tup[0], jt],
                                                                              jmp.state1, jmp.state2,
                                                                                  self.onsagercalculator.pdbcontainer.iorlist))
                                self.assertTrue(np.allclose(self.onsagercalculator.del_W1[tup[0], jt], rate))

            # Now, update with omega4 contributions
            for jt, jlist in enumerate(self.onsagercalculator.jnet4):
                for jnum, jmp in enumerate(jlist):
                    if jmp.state1 == comp_state:
                        # get the jump rate
                        rate = np.exp(- bFT4[jt] + bFSdb_total_shift[starind])
                        (IS, FS), dx = self.onsagercalculator.jnet4_indexed[jt][jnum]
                        self.assertEqual(FS, jmp.state2.db.iorind)
                        dx_solute = np.zeros(self.onsagercalculator.crys.dim)  # + or2/2.
                        dx_solvent = dx  # - or2/2.
                        self.assertEqual(IS, i)
                        bias_true_updated_solvent[i, :] += rate * np.sqrt(complex_prob[i]) *\
                                                           (dx_solvent + self.onsagercalculator.eta0total_solvent[IS] -
                                                            self.onsagercalculator.eta0total_solvent[FS + Ncomp])
                        # bias_true_updated_solute[i, :] += rate * np.sqrt(complex_prob[i]) * \
                        #                                   (dx_solute + self.onsagercalculator.eta0total_solute[IS] -
                        #                                    self.onsagercalculator.eta0total_solute[FS + Ncomp])
                        for tup in vstar_indlist:
                            self.assertTrue(np.allclose(omega4escape[tup[0], jt], rate))

        # 6b. - Now, we do it for the mixed dumbbell states
        # In the mixed dumbbell state space, the local rates come only from the contributions by the omega3 jumps
        for i in range(Ncomp, Ncomp + Nmix):
            mixstate = self.onsagercalculator.kinetic.mixedstates[i - Ncomp]
            vstar_indlist = self.onsagercalculator.vkinetic.stateToVecStar_mixed[mixstate]
            starind = self.onsagercalculator.kinetic.mixedindexdict[mixstate][1]
            mdbwyckind = self.onsagercalculator.kinetic.star2symlist[starind]
            dbwyck2 = self.onsagercalculator.mdbcontainer.invmap[mixstate.db.iorind]

            self.assertEqual(mdbwyckind, dbwyck2)

            # Now, update with omega3 contributions
            for jt, jlist in enumerate(self.onsagercalculator.jnet3):
                for jnum, jmp in enumerate(jlist):
                    if jmp.state1 == mixstate:
                        # get the jump rate
                        rate = np.exp(- bFT3[jt] + bFdb2[mdbwyckind] - bFdb2_min)
                        (IS, FS), dx = self.onsagercalculator.jnet3_indexed[jt][jnum]
                        dx_solute = np.zeros(self.onsagercalculator.crys.dim)
                        dx_solvent = dx  # + or1 / 2.
                        self.assertEqual(IS, i - Ncomp)
                        bias_true_updated_solvent[i, :] += rate * np.sqrt(mixed_prob[i - Ncomp]) * \
                                                           (dx_solvent +
                                                            self.onsagercalculator.eta0total_solvent[IS + Ncomp] -
                                                            self.onsagercalculator.eta0total_solvent[FS])
                        # bias_true_updated_solute[i, :] += rate * np.sqrt(mixed_prob[i - Ncomp]) * \
                        #                                   (dx_solute +
                        #                                    self.onsagercalculator.eta0total_solute[IS + Ncomp] -
                        #                                    self.onsagercalculator.eta0total_solute[FS])
                        for tup in vstar_indlist:
                            self.assertTrue(np.allclose(omega3escape[tup[0] - Nvstars_pure, jt], rate))

            # Now, update with omega2 contributions
            for jt, jlist in enumerate(self.onsagercalculator.jnet2):
                for jnum, jmp in enumerate(jlist):
                    if jmp.state1 == mixstate:
                        # get the jump rate
                        rate = np.exp(- bFT2[jt] + bFdb2[mdbwyckind] - bFdb2_min)
                        (IS, FS), dx = self.onsagercalculator.jnet2_indexed[jt][jnum]
                        dx_solute = dx
                        dx_solvent = dx
                        self.assertEqual(IS, i - Ncomp)
                        bias_true_updated_solvent[i, :] += rate * np.sqrt(mixed_prob[i - Ncomp]) * \
                                                           (dx_solvent +
                                                            self.onsagercalculator.eta0total_solvent[IS + Ncomp] -
                                                            self.onsagercalculator.eta0total_solvent[FS + Ncomp])
                        bias_true_updated_solute[i, :] += rate * np.sqrt(mixed_prob[i - Ncomp]) * dx_solute
                                                           # self.onsagercalculator.eta0total_solute[IS + Ncomp] -
                                                           # self.onsagercalculator.eta0total_solute[FS + Ncomp])

                        self.assertTrue(np.allclose(omega2escape[mdbwyckind, jt], rate))

        # 6c. Now, we get the bias vector as calculated in the code from the corresponding vector stars.
        bias_solvent_calc = np.zeros((Ncomp + Nmix, self.onsagercalculator.crys.dim))
        bias_solute_calc = np.zeros((Ncomp + Nmix, self.onsagercalculator.crys.dim))

        # first, we convert the complex states into cartesian form
        for i, state in enumerate(self.onsagercalculator.kinetic.complexStates):
            try:
                indlist = self.onsagercalculator.vkinetic.stateToVecStar_pure[state]
            except KeyError:
                indlist = []
                self.assertTrue(state.is_zero(self.onsagercalculator.pdbcontainer))

            bias_solute_calc[i, :] = sum([self.onsagercalculator.biases_solute_vs[tup[0]] *
                                          self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])
            bias_solvent_calc[i, :] = sum([self.onsagercalculator.biases_solvent_vs[tup[0]] *
                                          self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]] for tup in indlist])

        for i, state in enumerate(self.onsagercalculator.kinetic.mixedstates):
            indlist = self.onsagercalculator.vkinetic.stateToVecStar_mixed[state]
            bias_solute_calc[i + Ncomp, :] = sum([self.onsagercalculator.biases_solute_vs[tup[0]] *
                                                  self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]]
                                                  for tup in indlist])
            bias_solvent_calc[i + Ncomp, :] = sum([self.onsagercalculator.biases_solvent_vs[tup[0]] *
                                                   self.onsagercalculator.vkinetic.vecvec[tup[0]][tup[1]]
                                                   for tup in indlist])

        self.assertTrue(np.allclose(bias_solvent_calc, bias_true_updated_solvent))
        self.assertTrue(np.allclose(bias_solute_calc, bias_true_updated_solute))

        print("passed tests 6 - checking the final bias vector")

class test_BCC(test_dumbbell_mediated):
    def setUp(self):
        # We test a new weird lattice because it is more interesting
        self.BCC = Crystal.BCC(0.2836, "A")
        o = np.array([0.1, 0., 0.])
        famp0 = [o.copy()]
        family = [famp0]

        self.pdbcontainer = pureDBContainer(self.BCC, 0, family)
        self.mdbcontainer = mixedDBContainer(self.BCC, 0, family)
        self.jset0, self.jset2 = \
            self.pdbcontainer.jumpnetwork(0.25, 0.01, 0.01), self.mdbcontainer.jumpnetwork(0.25, 0.01, 0.01)

        self.onsagercalculator = dumbbellMediated(self.pdbcontainer, self.mdbcontainer, self.jset0, self.jset2,
                                                  0.25, 0.01, 0.01, 0.01, NGFmax=4, Nthermo=1)
        # generate all the bias expansions - will separate out later
        self.biases = \
            self.onsagercalculator.vkinetic.biasexpansion(self.onsagercalculator.jnet1, self.onsagercalculator.jnet2,
                                                          self.onsagercalculator.om1types,
                                                          self.onsagercalculator.jnet43)

        self.W1list = np.random.rand(len(self.onsagercalculator.jnet1))
        self.W2list = np.random.rand(len(self.onsagercalculator.jnet0))
        self.W3list = np.random.rand(len(self.onsagercalculator.jnet3))
        self.W4list = np.random.rand(len(self.onsagercalculator.jnet4))

        print(self.onsagercalculator.mdbcontainer.symorlist)
        print("Initiated distorted lattice")

class test_distorted(test_dumbbell_mediated):
    def setUp(self):
        # We test a new weird lattice because it is more interesting

        latt = np.array([[0., 0.1, 0.5], [0.3, 0., 0.5], [0.5, 0.5, 0.]]) * 0.55
        self.DC_Si = Crystal(latt, [[np.array([0., 0., 0.]), np.array([0.25, 0.25, 0.25])]], ["Si"])
        # keep it simple with [1.,0.,0.] type orientations for now
        o = np.array([1., 0., 0.]) / np.linalg.norm(np.array([1., 0., 0.])) * 0.126
        famp0 = [o.copy()]
        family = [famp0]

        self.pdbcontainer_si = pureDBContainer(self.DC_Si, 0, family)
        self.mdbcontainer_si = mixedDBContainer(self.DC_Si, 0, family)
        self.jset0, self.jset2 = \
            self.pdbcontainer_si.jumpnetwork(0.3, 0.01, 0.01), self.mdbcontainer_si.jumpnetwork(0.3, 0.01, 0.01)

        self.onsagercalculator = dumbbellMediated(self.pdbcontainer_si, self.mdbcontainer_si, self.jset0, self.jset2,
                                                  0.3, 0.01, 0.01, 0.01, NGFmax=4, Nthermo=1)
        # generate all the bias expansions - will separate out later
        self.biases = \
            self.onsagercalculator.vkinetic.biasexpansion(self.onsagercalculator.jnet1, self.onsagercalculator.jnet2,
                                                          self.onsagercalculator.om1types,
                                                          self.onsagercalculator.jnet43)

        self.W1list = np.random.rand(len(self.onsagercalculator.jnet1))
        self.W2list = np.random.rand(len(self.onsagercalculator.jnet0))
        self.W3list = np.random.rand(len(self.onsagercalculator.jnet3))
        self.W4list = np.random.rand(len(self.onsagercalculator.jnet4))

        print(self.onsagercalculator.mdbcontainer.symorlist)
        print("Jump types of 1, 2, 3, 4: ", len(self.W1list), len(self.W2list), len(self.W3list), len(self.W4list))
        print("Initiated distorted lattice")

class test_2d(test_dumbbell_mediated):
    def setUp(self):

        self.DC_Si = Crystal(np.array([[1., 0.], [0., 1.5]]), [[np.array([0, 0]), np.array([0.5, 0.5])]], ["A"])
        # keep it simple with [1.,0.,0.] type orientations for now
        o = np.array([0., 0.1])
        famp02d = [o.copy()]
        family = [famp02d]

        self.pdbcontainer_si = pureDBContainer(self.DC_Si, 0, family)
        self.mdbcontainer_si = mixedDBContainer(self.DC_Si, 0, family)
        self.jset0, self.jset2 = \
            self.pdbcontainer_si.jumpnetwork(1.51, 0.01, 0.01), self.mdbcontainer_si.jumpnetwork(1.51, 0.01, 0.01)

        self.onsagercalculator = dumbbellMediated(self.pdbcontainer_si, self.mdbcontainer_si, self.jset0, self.jset2,
                                                  1.51, 0.01, 0.01, 0.01, NGFmax=4, Nthermo=1)

        # generate all the bias expansions - will separate out later
        self.biases = \
            self.onsagercalculator.vkinetic.biasexpansion(self.onsagercalculator.jnet1, self.onsagercalculator.jnet2,
                                                          self.onsagercalculator.om1types,
                                                          self.onsagercalculator.jnet43)

        self.W1list = np.random.rand(len(self.onsagercalculator.jnet1))
        self.W2list = np.random.rand(len(self.onsagercalculator.jnet0))
        self.W3list = np.random.rand(len(self.onsagercalculator.jnet3))
        self.W4list = np.random.rand(len(self.onsagercalculator.jnet4))
        print("Initiated 2d lattice")