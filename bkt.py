# BKT (Bayesian Knowledge Tracing) Model

import hmm
import numpy as np
import random

class bkt:

    def __init__(self, pi, pt, pf, pg, ps):
        self.init = pi
        self.transit = pt
        self.forget = pf
        self.guess = pg
        self.slip = ps

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
        self.init = random.random()
        self.transit = random.random()
        self.slip = random.random()
        self.guess = random.random()

        self.pi = np.array(([1.0-self.init, self.init]), np.float)

        self.hh = np.array(([[1.0-self.transit, self.transit],
                             [self.forget,      1.0-self.forget]]), np.float)

        self.ho = np.array(([[1.0-self.guess,   self.guess],
                             [self.slip,        1.0-self.slip]]), np.float)

        self.hmm = hmm.hmm(self.pi, self.hh, self.ho)


    # ss: state sequence
    # os: observation sequence
    def synthetic_data(self, length):
        return self.hmm.synthetic_data(length)

        #ss = np.zeros(length, np.int)
        #os = np.zeros(length, np.int)

        #next_state = self.pi

        #for l in range(length):
        #    ss[l] = next_state[0] < random.random()
        #    os[l] = (ss[l] and self.slip or 1-self.guess) < random.random()
        #    next_state = self.hh[ss[l]]

        #return ss, os


    # os: observation_sequence
    def forward(self, os):
        return self.hmm.forward(os) 
        

    def viterbi(self, os):
        return self.hmm.viterbi(os)

    
    # estimate bkt parameters
    def baum_welch(self, os, delta=0.001, max_iteration=float('inf')):
        log_likelihood, number, alpha = self.hmm.baum_welch(os, delta, max_iteration)
        self.update_bkt_parameters()
        return log_likelihood, number, alpha
    

    # predict the probability of answered correct next step 
    # alpha: return from baum_welch function
    def predict(self, os, alpha):
        state_predicts, os_predicts = self.hmm.predict_next_steps(len(os), alpha)
        return state_predicts, os_predicts


    # after run baum_welch function, update bkt parameters
    def update_bkt_parameters(self):
        self.pi = self.hmm.pi
        self.hh = self.hmm.hh 
        self.ho = self.hmm.ho
        self.init = self.pi[1]
        self.transit = self.hh[0][1]
        self.forget = self.hh[1][0]
        self.guess = self.ho[0][1]
        self.slip = self.ho[1][0]


    def print_parameters(self):
        print('\nbkt patameters:')
        print('\tinit:\t', "{:.3f}".format(self.init))
        print('\ttran:\t', "{:.3f}".format(self.transit))
        print('\tforget:\t', "{:.3f}".format(self.forget))
        print('\tguess:\t', "{:.3f}".format(self.guess))
        print('\tslip:\t', "{:.3f}".format(self.slip))
        np.set_printoptions(precision=3, suppress=True)
        print('\npi\n', self.pi)
        print('\nhh\n', self.hh)
        print('\nho\n', self.ho)
        
