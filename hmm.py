# hmm: hidden markov model, aicfe mqg

import math
import numpy as np


class hmm:
    
    '''
    PI: hidden state init prob vector
    HH: a hidden state-transition prob matrix ( p(hidden state b | hidden state a) )
    HO: obversation prob distribution matrix ( p(bversation state a | hidden state a) )
    H: the number of hidden states
    O: the number of obeservation states
    T: the length of obeservation sequence
    '''
    def __init__(self, PI, HH, HO):
        self.PI = np.array(PI, np.float)
        self.HH = np.array(HH, np.float)
        self.HO = np.array(HO, np.float)
        self.H = self.HH.shape[0]
        self.O = self.HO.shape[1]


    # computing the prob of observation sequence given a HMM model parameters
    # os: observation sequence
    # T: the length of os
    def forward(self, os, T):
        alpha = np.zeros((T, self.H))

        # 1. Initialization
        for h in range(self.H):
            alpha[0, h] = self.PI[h] * self.HO[h, os[0]]

        # 2. Induction
        # t: time index
        # i, j: hidden state index
        for t in range(1, T):
            for i in range(self.H):
                sum_prob = 0.0
                for j in range(self.H):
                    sum_prob += alpha[t-1][j] * self.HH[j][i]
                alpha[t][i] = sum_prob * self.HO[i][os[t]]

        # 3. Termination
        pprob = 0.0
        for i in range(self.H):
            pprob += alpha[T-1][i]

        return pprob 


    # seeking the most probable hidden state sequence
    # hss: hidden state sequence
    def viterbi(self, os, T):
        hss = np.zeros(T, np.int)

        delta = np.zeros((T, self.H), np.float)
        psi = np.zeros((T, self.H), np.float)

        # 1. Initialization
        for h in range(self.H):
            delta[0][h] = self.PI[h] * self.HO[h, os[0]]
            psi[0][h] = 0

        # 2. Recursion
        for t in range(1, T):
            for i in range(self.H):
                max_val = 0.0
                max_val_ind = 0
                for j in range(self.H):
                    val = delta[t-1][j] * self.HH[j][i]
                    if val > max_val:
                        max_val = val
                        max_val_ind = j
                delta[t][i] = max_val * self.HO[i][os[t]]
                psi[t][i] = max_val_ind

        # 3. Termination
        big_prob = delta[T-1, :].max()

        # 4. Path(hidden state sequence) backtracking
        hss[T-1] = delta[T-1, :].argmax()
        for t in range(T-2, -1, -1):
            hss[t] = psi[t+1, hss[t+1]]

        return hss


    def backward(self, os, T):
        beta = np.zeros((T, self.H), np.float)

        for h in range(self.H):
            beta[T-1][h] = 1.0

        for t in range(T-2, -1, -1):
            for i in range(self.H):
                sum_prob = 0.0
                for j in range(self.H):
                    sum_prob+= self.HH[i][j] * self.HO[j][os[t+1]] * beta[t+1][j] 
                beta[t][i] = sum_prob

        pprob = 0.0
        for h in range(self.H):
            pprob += beta[0][h]

        return beta, pprob


    def forward_with_scale(self, os, T):
        scale = np.zeros(T, np.float)
        alpha = np.zeros((T, self.H), np.float)
        
        # 1. Initialization
        for h in range(self.H):
            alpha[0][h] = self.PI[h] * self.HO[h, os[0]]
            scale[0] += alpha[0][h]

        for i in range(self.H):
            alpha[0][i] /= scale[0]

        # 2. Induction
        for t in range(1, T):
            for i in range(self.H):
                sum_prob = 0.0
                for j in range(self.H):
                    sum_prob += alpha[t-1][j] * self.HH[j][i]
                alpha[t][i] = sum_prob * self.HO[i][os[t]]
                scale[t] += alpha[t][i]

            for h in range(self.H):
                alpha[t][h] /= scale[t]

        # 3. Termination
        pprob = 0.0
        for t in range(T):
            pprob += math.log(scale[t])

        return alpha, pprob, scale

    
    # scale: from the forward_with_scale function
    def backward_with_scale(self, os, T, scale):
        beta = np.zeros((T, self.H))

        for h in range(self.H):
            beta[T-1][h] = 1.0/scale[T-1]

        # i: time t
        # j: time t+1
        for t in range(T-2, -1, -1):
            for i in range(self.H):
                sum_prob = 0.0
                for j in range(self.H):
                    sum_prob += self.HH[i][j] * self.HO[j][os[t+1]] * beta[t+1][j] 
                beta[t][i] = sum_prob / scale[t]

        return beta 


    def computer_gamma(self, T, alpha, beta):
        gamma = np.zeros((T, self.H), np.float)

        for t in range(T):
            denominator = 0.0
            for i in range(self.H):
                gamma[t][i] = alpha[t][i] * beta[t][i]
                denominator += gamma[t][i]

            for j in range(self.H):
                gamma[t][j] = gamma[t][j] / denominator

        return gamma


    def computer_xi(self, os, T, alpha, beta):
        xi = np.zeros((T-1, self.H, self.H), np.float)
        
        for t in range(T-1):
            denominator = 0.0
            for i in range(self.H):
                for j in range(self.H):
                    xi[t][i][j] = alpha[t][i] * self.HH[i][j] * self.HO[j][os[t+1]] * beta[t+1][j]
                    denominator += xi[t][i][j]
            
            for i in range(self.H):
                for j in range(self.H):
                    xi[t][i][j] /= denominator

        return xi


    # optimizing model parameters
    def baum_welch(self, os, T, delta):
        loop_number = 0
        
        alpha, log_prob, scale = self.forward_with_scale(os, T)
        beta = self.backward_with_scale(os, T, scale)
        gamma = self.computer_gamma(T, alpha, beta)
        xi = self.computer_xi(os, T, alpha, beta)
        log_prob_prev = log_prob
        
        while True:
            # reestimate frequency of hidden state i in time t=0
            for i in range(self.H):
                self.PI[i] = 0.001 + 0.999 * gamma[0][i]

            # reestimate transition matrix and symbol prob in each state
            for i in range(self.H):
                denominator_A = 0.0
                for t in range(T-1):
                    denominator_A += gamma[t][i]

                for j in range(self.H):
                    numerator_A = 0.0
                    for t in range(T-1):
                        numerator_A += xi[t][i][j]

                    self.HH[i][j] = 0.001 + 0.999 * numerator_A / denominator_A

                denominator_B = denominator_A + gamma[T-1][i]
                for k in range(self.O):
                    numerator_B = 0.0
                    for t in range(T):
                        if os[t] == k:
                            numerator_B += gamma[t][i]
                        
                    self.HO[i][k] = 0.001 + 0.999 * numerator_B / denominator_B

            alpha, log_prob, scale = self.forward_with_scale(os, T)
            beta = self.backward_with_scale(os, T, scale)
            gamma = self.computer_gamma(T, alpha, beta)
            xi = self.computer_xi(os, T, alpha, beta)

            d = log_prob - log_prob_prev
            log_prob_prev = log_prob
            loop_number += 1

            if d <= delta:
                break

        return self.PI, self.HH, self.HO, loop_number
    

#A = [[0.500, 0.375, 0.125],
#        [0.250, 0.125, 0.625],
#        [0.250, 0.375, 0.375]]
#
#B = [[0.60, 0.20, 0.15, 0.05],
#        [0.25, 0.25, 0.25, 0.25],
#        [0.05, 0.10, 0.35, 0.50]]
#
#PI = [0.63, 0.17, 0.20]
#
#ob = [0, 2, 3]
#T = 3
#
#model = hmm(PI, A, B)
#print(model.forward(ob, T))
#
#A =  [[0.500, 0.375, 0.125],
#          [0.250, 0.125, 0.625],
#          [0.250, 0.375, 0.375]]
#          
#B = [[0.50, 0.50],
# [0.75, 0.25],
# [0.25, 0.75]]
#
#PI = [0.333, 0.333, 0.333]
#
#ob = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1]
#T = len(ob)
#
#model = hmm(PI, A, B)
#print(model.viterbi(ob, T))
#print(model.baum_welch(ob, T, 0.00001))

