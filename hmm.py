# hmm: hidden markov model, aicfe mqg

import math
import numpy as np


class hmm:
    
    '''
        pi: hidden state init prob vector
        hh: a hidden state-transition prob matrix ( p(hidden state b | hidden state a) )
        ho: observation prob distribution matrix ( p(bversation state a | hidden state a) )
        h: the number of hidden states
        o: the number of obeservation states
    '''
    def __init__(self, pi, hh, ho):
        self.pi = np.array(pi, np.float)
        self.hh = np.array(hh, np.float)
        self.ho = np.array(ho, np.float)
        self.h = self.hh.shape[0]
        self.o = self.ho.shape[1]


    # computing the prob of observation sequence given a HMM model parameters
    # os: observation sequence
    # os_length: the length of os
    # Todo: check data, exception handle
    def forward(self, os, os_length):
        alpha = np.zeros((os_length, self.h))

        # 1. Initialization
        for h in range(self.h):
            alpha[0, h] = self.pi[h] * self.ho[h, os[0]]

        # 2. Induction
        # t: time index
        # i: time t+1 hidden state index
        # j: time t hidden state index
        for t in range(1, os_length):
            for i in range(self.h):
                sum_prob = 0.0
                for j in range(self.h):
                    sum_prob += alpha[t-1][j] * self.hh[j][i]
                alpha[t][i] = sum_prob * self.ho[i][os[t]]

        # 3. Termination
        prob = 0.0
        for i in range(self.h):
            prob += alpha[os_length-1][i]

        return prob 


    def backward(self, os, os_length):
        beta = np.zeros((os_length, self.h), np.float)

        for h in range(self.h):
            beta[os_length-1][h] = 1.0

        for t in range(os_length-2, -1, -1):
            for i in range(self.h):
                sum_prob = 0.0
                for j in range(self.h):
                    sum_prob+= self.hh[i][j] * self.ho[j][os[t+1]] * beta[t+1][j] 
                beta[t][i] = sum_prob

        prob = 0.0
        for h in range(self.h):
            prob += beta[0][h]

        return beta, prob


    # seeking the most probable hidden state sequence
    # hss: hidden state sequence
    def viterbi(self, os, os_length):
        hss = np.zeros(os_length, np.int)

        delta = np.zeros((os_length, self.h), np.float)
        psi = np.zeros((os_length, self.h), np.float)

        # 1. Initialization
        for h in range(self.h):
            delta[0][h] = self.pi[h] * self.ho[h, os[0]]
            psi[0][h] = 0

        # 2. Recursion
        for t in range(1, os_length):
            for i in range(self.h):
                max_val = 0.0
                max_val_ind = 0
                for j in range(self.h):
                    val = delta[t-1][j] * self.hh[j][i]
                    if val > max_val:
                        max_val = val
                        max_val_ind = j
                delta[t][i] = max_val * self.ho[i][os[t]]
                psi[t][i] = max_val_ind

        # 3. os_lengthermination
        big_prob = delta[os_length-1, :].max()

        # 4. Path(hidden state sequence) backtracking
        hss[os_length-1] = delta[os_length-1, :].argmax()
        for t in range(os_length-2, -1, -1):
            hss[t] = psi[t+1, hss[t+1]]

        return hss


    def forward_with_scale(self, os, os_length):
        scale = np.zeros(os_length, np.float)
        alpha = np.zeros((os_length, self.h), np.float)
        
        # 1. Initialization
        for h in range(self.h):
            alpha[0][h] = self.pi[h] * self.ho[h, os[0]]
            scale[0] += alpha[0][h]

        for i in range(self.h):
            alpha[0][i] /= scale[0]

        # 2. Induction
        for t in range(1, os_length):
            for i in range(self.h):
                sum_prob = 0.0
                for j in range(self.h):
                    sum_prob += alpha[t-1][j] * self.hh[j][i]
                alpha[t][i] = sum_prob * self.ho[i][os[t]]
                scale[t] += alpha[t][i]

            for h in range(self.h):
                alpha[t][h] /= scale[t]

        # 3. os_lengthermination
        log_likelihood = 0.0
        for t in range(os_length):
            log_likelihood += math.log(scale[t])

        return alpha, log_likelihood, scale

    
    # scale: from the forward_with_scale function
    def backward_with_scale(self, os, os_length, scale):
        beta = np.zeros((os_length, self.h), np.float)

        for h in range(self.h):
            beta[os_length-1][h] = 1.0/scale[os_length-1]

        # i: time t
        # j: time t+1
        for t in range(os_length-2, -1, -1):
            for i in range(self.h):
                sum_prob = 0.0
                for j in range(self.h):
                    sum_prob += self.hh[i][j] * self.ho[j][os[t+1]] * beta[t+1][j] 
                beta[t][i] = sum_prob / scale[t]

        return beta 


    def computer_gamma(self, os_length, alpha, beta):
        gamma = np.zeros((os_length, self.h), np.float)

        for t in range(os_length):
            denominator = 0.0
            for i in range(self.h):
                gamma[t][i] = alpha[t][i] * beta[t][i]
                denominator += gamma[t][i]

            for j in range(self.h):
                gamma[t][j] = gamma[t][j] / denominator

        return gamma


    def computer_xi(self, os, os_length, alpha, beta):
        xi = np.zeros((os_length-1, self.h, self.h), np.float)
        
        for t in range(os_length-1):
            denominator = 0.0
            for i in range(self.h):
                for j in range(self.h):
                    xi[t][i][j] = alpha[t][i] * self.hh[i][j] * self.ho[j][os[t+1]] * beta[t+1][j]
                    denominator += xi[t][i][j]
            
            for i in range(self.h):
                for j in range(self.h):
                    xi[t][i][j] /= denominator

        return xi


    # optimizing model parameters
    def baum_welch(self, os, os_length, delta):
        loop_number = 0
        
        alpha, log_prob, scale = self.forward_with_scale(os, os_length)
        beta = self.backward_with_scale(os, os_length, scale)
        gamma = self.computer_gamma(os_length, alpha, beta)
        xi = self.computer_xi(os, os_length, alpha, beta)
        log_prob_prev = log_prob
        
        while True:
            # reestimate frequency of hidden state i in time t=0
            for i in range(self.h):
                self.pi[i] = 0.001 + 0.999 * gamma[0][i]

            # reestimate transition matrix and symbol prob in each state
            for i in range(self.h):
                denominator_A = 0.0
                for t in range(os_length-1):
                    denominator_A += gamma[t][i]

                for j in range(self.h):
                    numerator_A = 0.0
                    for t in range(os_length-1):
                        numerator_A += xi[t][i][j]

                    self.hh[i][j] = 0.001 + 0.999 * numerator_A / denominator_A

                denominator_B = denominator_A + gamma[os_length-1][i]
                for k in range(self.o):
                    numerator_B = 0.0
                    for t in range(os_length):
                        if os[t] == k:
                            numerator_B += gamma[t][i]
                        
                    self.ho[i][k] = 0.001 + 0.999 * numerator_B / denominator_B

            alpha, log_prob, scale = self.forward_with_scale(os, os_length)
            beta = self.backward_with_scale(os, os_length, scale)
            gamma = self.computer_gamma(os_length, alpha, beta)
            xi = self.computer_xi(os, os_length, alpha, beta)

            now_delta = log_prob - log_prob_prev
            log_prob_prev = log_prob
            loop_number += 1

            if now_delta <= delta:
                break

        #return self.pi, self.hh, self.ho, loop_number, alpha
        return alpha


    def predict_next_steps(self, alpha, os_length):
        # next time states probability
        states_predictions = np.zeros((os_length, self.h), np.float)
        # next time observations probability
        observations_predictions = np.zeros((os_length, self.o), np.float)
        
        # 1. Initialization states_predictions
        for h in range(self.h):
            states_predictions[0][h] = self.pi[h]

        # 2. Induction states_predictions
        for t in range(1, os_length):
            for i in range(self.h):
                sum_prob = 0.0
                for j in range(self.h):
                    sum_prob += alpha[t-1][j] * self.hh[j][i]
                
                states_predictions[t][i] = sum_prob

        # 3. compute observations_predictions
        for t in range(os_length):
            for i in range(self.o):
                sum_prob = 0.0
                for j in range(self.h):
                    sum_prob += states_predictions[t][j] * self.ho[j][i]
                    
                observations_predictions[t][i] = sum_prob

        return states_predictions, observations_predictions

