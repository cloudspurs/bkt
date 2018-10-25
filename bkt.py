# bkt: bayes knowledge tracing 

import hmm


class bkt:

    def __init__(self, PL0, PL, PF, PS, PG):
        self.p_l0 = PL0
        self.p_learn = PL
        self.p_forget = PF
        self.p_slip = PS
        self.p_guess = PG
        
        hmm_pi = [1.0-self.p_l0, self.p_l0]
        hmm_hh = [[1.0-self.p_learn, self.p_learn], [self.p_forget, 1.0-self.p_forget]]
        hmm_ho = [[1.0-self.p_guess, self.p_guess], [self.p_slip, 1.0-self.p_slip]]
        self.hmm_model = hmm.hmm(hmm_pi, hmm_hh, hmm_ho)


    def forward(self, observation_sequence):
        return self.hmm_model.forward(observation_sequence, len(observation_sequence))
        

    def viterbi(self, observation_sequence):
        return self.hmm_model.viterbi(observation_sequence, len(observation_sequence))

    
    def baum_welch(self, observation_sequence, delta):
        self.hmm_model.baum_welch(observation_sequence, len(observation_sequence), delta)
        self.bkt_parameters(self.hmm_model)


    def print_parameters(self):
        print('p_l0', self.p_l0)
        print('p_learn', self.p_learn)
        print('p_slip', self.p_slip)
        print('p_guess', self.p_guess)


    def bkt_parameters(self, hmm):
        self.p_l0 = hmm.PI[1]
        self.p_learn = hmm.HH[0][1]
        self.p_slip = hmm.HO[1][0]
        self.p_guess = hmm.HO[0][1]
        
    
a_model = bkt(0.5, 0.5, 0.0, 0.1, 0.05)

os = [0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1]

print(a_model.forward(os))
print(a_model.viterbi(os))
a_model.baum_welch(os, 0.001)
a_model.print_parameters()
print(a_model.forward([1]))

