# HMM: first order Hidden Markov Model

import math
import numpy as np
import random

class hmm:
    "HMM: first order Hidden Markov Model"

    '''
        pi: states initial probability vector
        hh: states transition probability matrix [ p(t+1(state_b) | t(state_a)) ]
        ho: states to observations emission probability matrix [ p(t(observation_a) | t(state_a)) ]
        h: the number of states
        o: the number of obeservations
    '''
    def __init__(self, pi, hh, ho):
        self.pi = np.array(pi, np.float)
        self.hh = np.array(hh, np.float)
        self.ho = np.array(ho, np.float)
        self.h = self.hh.shape[0]
        self.o = self.ho.shape[1]

    
    def print_parameters(self):
        np.set_printoptions(precision=3, suppress=True)
        print('\npi:', self.pi)
        print('\nhh:')
        print(self.hh)
        print('\nho:')
        print(self.ho)


    def random_model(self):
        sum_pi = 0.0
        for h in range(self.h):
            self.pi[h] = random.random()
            sum_pi += self.pi[h]
        for h in range(self.h):
            self.pi[h] /= sum_pi

        for i in range(self.h):
            sum_hh = 0.0
            for j in range(self.h):
                self.hh[i][j] = random.random()
                sum_hh += self.hh[i][j]
            
            for j in range(self.h):
                self.hh[i][j] /= sum_hh

            sum_ho = 0.0
            for j in range(self.o):
                self.ho[i][j] = random.random()
                sum_ho += self.ho[i][j]

            for j in range(self.o):
                self.ho[i][j] /= sum_ho

    
    # ss: state sequence
    # os: observation sequence
    def synthetic_data(self, length):
        ss = np.zeros(length, np.int)
        os = np.zeros(length, np.int)

        next_state = np.cumsum(self.pi)

        for l in range(length):
            ss[l] = np.argmax(next_state > random.random())
            
            ob = np.cumsum(self.ho[ss[l]])
            os[l] = np.argmax(ob > random.random())

            next_state = np.cumsum(self.hh[ss[l]])

        return ss, os
        

    # given observation sequence os=(o1, o2,..., ot) and model m=(pi, hh, ho), compute p(os|m)
    # os_length: the length of os
    def forward(self, os):
        os_length = len(os)
        alpha = np.zeros((os_length, self.h), np.float)

        # 1. Initialization
        for h in range(self.h):
            alpha[0, h] = self.pi[h] * self.ho[h, os[0]]

        # 2. Induction
        # t: time index
        # i: time t states index
        # j: time t-1 states index
        for t in range(1, os_length):
            for i in range(self.h):
                sum_prob = 0.0
                for j in range(self.h):
                    sum_prob += alpha[t-1][j] * self.hh[j][i]
                alpha[t][i] = sum_prob * self.ho[i][os[t]]

        # 3. Termination
        prob = 0.0
        for h in range(self.h):
            prob += alpha[os_length-1][h]

        return prob

    # backward probability should equal to forward probability
    def backward(self, os):
        os_length = len(os)
        beta = np.zeros((os_length, self.h), np.float)

        for h in range(self.h):
            beta[os_length-1][h] = 1.0

        for t in range(os_length-2, -1, -1):
            for i in range(self.h):
                sum_prob = 0.0
                for j in range(self.h):
                    sum_prob += self.hh[i][j] * self.ho[j][os[t+1]] * beta[t+1][j] 
                beta[t][i] = sum_prob

        prob = 0.0
        for h in range(self.h):
            prob += self.pi[h] * self.ho[h][os[0]] * beta[0][h]

        return prob


    # seeking the most probable hidden state sequence
    # hss: hidden state sequence
    def viterbi(self, os):
        os_length = len(os)
        
        # hss: hidden state sequence
        hss = np.zeros(os_length, np.int)
        delta = np.zeros((os_length, self.h), np.float)
        # psi: position index
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


    # scale: avoding float number overflow
    def forward_with_scale(self, os):
        os_length = len(os)

        scale = np.zeros(os_length, np.float)
        alpha = np.zeros((os_length, self.h), np.float)
        
        # 1. Initialization, compute alpha(0, h)
        for h in range(self.h):
            alpha[0][h] = self.pi[h] * self.ho[h, os[0]]
            scale[0] += alpha[0][h]

        for h in range(self.h):
            alpha[0][h] /= scale[0]

        # 2. Iteration, compute alpha(t, h)
        for t in range(1, os_length):
            for i in range(self.h):
                sum_prob = 0.0
                for j in range(self.h):
                    sum_prob += alpha[t-1][j] * self.hh[j][i]
                alpha[t][i] = sum_prob * self.ho[i][os[t]]
                scale[t] += alpha[t][i]

            for h in range(self.h):
                alpha[t][h] /= scale[t]

        # 3. Termination
        log_likelihood = math.log(scale[os_length-1])

#        log_likelihood = 0.0
#        for t in range(os_length):
#            log_likelihood += math.log(scale[t])

        return alpha, log_likelihood, scale

    
    # scale: from the forward_with_scale function
    def backward_with_scale(self, os, scale):
        os_length = len(os)

        beta = np.zeros((os_length, self.h), np.float)

        for h in range(self.h):
            # (1.0 / scale[os_length-1]) maybe > 1.0
            beta[os_length-1][h] = 1.0 / scale[os_length-1]

        for t in range(os_length-2, -1, -1):
            for i in range(self.h):
                sum_prob = 0.0
                for j in range(self.h):
                    sum_prob += self.hh[i][j] * self.ho[j][os[t+1]] * beta[t+1][j] 
                beta[t][i] = sum_prob / scale[t]

        return beta 


    def computer_xi(self, os, alpha, beta):
        os_length = len(os)
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


    def computer_gamma(self, os_length, alpha, beta):
        gamma = np.zeros((os_length, self.h), np.float)

        for t in range(os_length):
            denominator = 0.0
            for h in range(self.h):
                gamma[t][h] = alpha[t][h] * beta[t][h]
                denominator += gamma[t][h]

            for h in range(self.h):
                gamma[t][h] = gamma[t][h] / denominator

        return gamma


    # function: estimate model parameters
    def baum_welch(self, os, delta=0.001, max_iteration=float('inf')):
        os_length = len(os)

        iteration_number = 0
        
        log_likelihood_prev = float('-inf')

        while True:
            # compute all parameters
            alpha, log_likelihood, scale = self.forward_with_scale(os)
            beta = self.backward_with_scale(os, scale)
            xi = self.computer_xi(os, alpha, beta)
            gamma = self.computer_gamma(os_length, alpha, beta)

            # reestimate frequency of hidden state i in time t=0
            for i in range(self.h):
                #self.pi[i] = 0.001 + 0.999 * gamma[0][i]
                self.pi[i] = gamma[0][i]

            # reestimate transition matrix and symbol prob in each state
            for i in range(self.h):
                denominator_A = 0.0
                for t in range(os_length-1):
                    denominator_A += gamma[t][i]

                for j in range(self.h):
                    numerator_A = 0.0
                    for t in range(os_length-1):
                        numerator_A += xi[t][i][j]

                    #self.hh[i][j] = 0.001 + 0.999 * numerator_A / denominator_A
                    self.hh[i][j] = numerator_A / denominator_A

                denominator_B = denominator_A + gamma[os_length-1][i]

                for o in range(self.o):
                    numerator_B = 0.0
                    for t in range(os_length):
                        if os[t] == o:
                            numerator_B += gamma[t][i]
                        
                    #self.ho[i][k] = 0.001 + 0.999 * numerator_B / denominator_B
                    self.ho[i][o] = numerator_B / denominator_B

            now_delta = log_likelihood - log_likelihood_prev
            log_likelihood_prev = log_likelihood
            iteration_number += 1

            if (now_delta <= delta) or (iteration_number >= max_iteration):
                break

        #return self.pi, self.hh, self.ho, loop_number, alpha
        return log_likelihood, iteration_number, alpha 

    # fuction: predict next observation probability
    # 1. compute next time states distribution
    # 2. compute next time observations distribution
    def predict_next_steps(self, os_length, alpha):
        # next time states probability
        states_predictions = np.zeros((os_length, self.h), np.float)
        # next time observations probability
        observations_predictions = np.zeros((os_length, self.o), np.float)
        
        # 1. Initialization states_predictions
        for h in range(self.h):
            states_predictions[0][h] = self.pi[h]

        # 2. Iteration states_predictions
        for t in range(1, os_length):
            for i in range(self.h):
                sum_prob = 0.0
                for j in range(self.h):
                    sum_prob += alpha[t-1][j] * self.hh[j][i]
                
                states_predictions[t][i] = sum_prob

        # 3. Compute observations_predictions
        for t in range(os_length):
            for o in range(self.o):
                sum_prob = 0.0
                for h in range(self.h):
                    sum_prob += states_predictions[t][h] * self.ho[h][o]
                    
                observations_predictions[t][o] = sum_prob

        return states_predictions, observations_predictions

