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
        print('init: ', self.init)
        print('learn: ', self.learn)
        print('forget: ', self.forget)
        print('slip: ', self.slip)
        print('guess: ', self.guess)


    # after baum_welch function, update bkt parameters
    def update_bkt_parameters(self, hmm):
        self.init = hmm.pi[1]
        self.learn = hmm.hh[0][1]
        self.forget = hmm.hh[1][0]
        self.slip = hmm.ho[1][0]
        self.guess = hmm.ho[0][1]
        
    
a_model = bkt(0.1, 0.5, 0.0, 0.1, 0.05)

os = [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]

print("init parameters:")
a_model.print_parameters()

print(a_model.forward(os))
print(a_model.viterbi(os))

a_model.baum_welch(os, 0.001)

a_model.print_parameters()

predicts, obs = a_model.predict_next_step(os, 0.01)
print("predicts: ", predicts)
print("obs: ", obs)

