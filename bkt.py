# BKT (Bayesian Knowledge Tracing) Model

import hmm
import numpy as np
import random

class bkt:

    def __init__(self, pi, pt, pf, ps, pg):
        self.init = pi
        self.transit = pt
        self.forget = pf
        self.slip = ps
        self.guess = pg

        '''
            bkt parameters in matrix form

            prior: 
                 unknown,   known
                [1-init,    init]

            hh:  
                                  to unknown    to known 
                from unknown    [[1-transit,    transit], 
                from known       [forget,       1-forget]]
            ho: 
                          wrong     right 
                unknown [[1-guess,  guess],
                known    [slip,     1-slip]]
        '''
        self.pi = np.array(([1.0-self.init, self.init]), np.float)

        self.hh = np.array(([[1.0-self.transit, self.transit],
                             [self.forget,      1.0-self.forget]]), np.float)

        self.ho = np.array(([[1.0-self.guess,   self.guess],
                             [self.slip,        1.0-self.slip]]), np.float)

        self.hmm = hmm.hmm(self.pi, self.hh, self.ho)


    # Todo
    def random_model(self):
        pi = random.random()
        pt = random.random()
        pf = random.random()
        pg = random.random()
        ps = random.random()
        self.__init__(pi, pt, pf, pg, ps)

    # ss: state sequence
    # os: observation sequence
    def synthetic_data(self, length):
        ss = np.zeros(length, np.int)
        os = np.zeros(length, np.int)

        next_state = self.pi

        for l in range(length):
            ss[l] = next_state[0] < random.random()
            os[l] = (ss[l] and self.slip or 1-self.guess) < random.random()
            next_state = self.hh[ss[l]]

        return ss, os


    # os: observation_sequence
    def forward(self, os):
        return self.hmm.forward(os) 
        

    def viterbi(self, os):
        return self.hmm.viterbi(os)

    
    # estimate bkt parameters
    def baum_welch(self, os, delta, max_iteration=float('inf')):
        alpha, number = self.hmm.baum_welch(os, delta, max_iteration)
        self.update_bkt_parameters(self.hmm)
        return alpha
    

    # predict the probability of answered correct next step 
    # alpha: return from baum_welch function
    def predict(self, os, alpha):
        state_predicts, os_predicts = self.hmm.predict_next_steps(alpha, len(os))
        return state_predicts, os_predicts


    # after run baum_welch function, update bkt parameters
    def update_bkt_parameters(self, hmm):
        self.pi = self.hmm.pi
        self.hh = self.hmm.hh 
        self.ho = self.hmm.ho
        self.init = self.pi[1]
        self.transit = self.hh[0][1]
        self.forget = self.hh[1][0]
        self.slip = self.ho[1][0]
        self.guess = self.ho[0][1]


    def print_parameters(self):
        print('\nbkt patameters:')
        print('\tinit:\t', self.init)
        print('\tlearn:\t', self.transit)
        print('\tforget:\t', self.forget)
        print('\tslip:\t', self.slip)
        print('\tguess:\t', self.guess)
        print('\npi\n', self.pi)
        print('\nhh\n', self.hh)
        print('\nho\n', self.ho)
        
