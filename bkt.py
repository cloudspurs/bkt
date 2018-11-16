# BKT(Bayesian Knowledge Tracing) Model

import hmm


# BKT model for one knowledge
class bkt:

    def __init__(self, pi, pl, pf, ps, pg):
        self.init = pi
        self.learn = pl
        self.forget = pf
        self.slip = ps
        self.guess = pg
        
        '''
            pi [0, 1]
            hh [0,0 0,1
                1,0 1,1]
            ho [0,0 0,1
                1,0 1,1]
        '''
        hmm_pi = [1.0-self.init, self.init]
        hmm_hh = [[1.0-self.learn, self.learn], [self.forget, 1.0-self.forget]]
        hmm_ho = [[1.0-self.guess, self.guess], [self.slip, 1.0-self.slip]]
        self.hmm_model = hmm.hmm(hmm_pi, hmm_hh, hmm_ho)


    def forward(self, observation_sequence):
        return self.hmm_model.forward(observation_sequence, len(observation_sequence))
        

    def viterbi(self, observation_sequence):
        return self.hmm_model.viterbi(observation_sequence, len(observation_sequence))

    
    def baum_welch(self, observation_sequence, delta):
        alpha = self.hmm_model.baum_welch(observation_sequence, len(observation_sequence), delta)
        self.update_bkt_parameters(self.hmm_model)
        return alpha
    

    def predict_next_step(self, os, delta):
        alpha = self.baum_welch(os, delta)
        predicts, obs = self.hmm_model.predict_next_steps(alpha, len(os))
        return predicts, obs


    # Todo
    # def baum_welch(self, observation_sequence, iterations):

    def print_parameters(self):
        print('bkt patameters:')
        print('\tinit:\t', self.init)
        print('\tlearn:\t', self.learn)
        print('\tforget:\t', self.forget)
        print('\tslip:\t', self.slip)
        print('\tguess:\t', self.guess, '\n')


    # after baum_welch function, update bkt parameters
    def update_bkt_parameters(self, hmm):
        self.init = self.hmm_model.pi[1]
        self.learn = self.hmm_model.hh[0][1]
        self.forget = self.hmm_model.hh[1][0]
        self.slip = self.hmm_model.ho[1][0]
        self.guess = self.hmm_model.ho[0][1]
        
    

