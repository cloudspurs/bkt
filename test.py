import numpy as np
import hmm
import bkt

#os = np.random.randint(0, 2, size=[5000])
#os = np.full(100, 1)
#os[0] = 0

#pi = [0.3, 0.7]
#A = [[0.2, 0.8], [0.4, 0.6]]
#B = [[0.7, 0.3], [0.2, 0.8]]
#
#hmm_model = hmm.hmm(pi, A, B)
#hmm_model.random_model()
#hmm_model.print_parameters()
#
#ss, os = hmm_model.synthetic_data(2000)
#
#print('\nss:', ss)
#print('\nos:', os)

pi = [0.333, 0.333, 0.333]
A = [[0.333, 0.333, 0.333], [0.333, 0.333, 0.333], [0.333, 0.333, 0.333]]
B = [[0.5, 0.5], [0.75, 0.25], [0.25, 0.75]]

os = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1]
test_model = hmm.hmm(pi, A, B)
test_model.baum_welch(os, 0.001)
test_model.print_parameters()


#bkt_model = bkt.bkt(0.1, 0.3, 0.0, 0.1, 0.03)
#bkt_model.print_parameters()
#bkt_model.random_model()
#bkt_model.print_parameters()
#
#ss, os = bkt_model.synthetic_data(20)
#print('\nss:', ss)
#print('\nos:', os)

#alpha = bkt_model.baum_welch(os, 0.001)
#print('\nestimate parameters: ') 
#bkt_model.print_parameters()

#sp, osp = bkt_model.predict(os, alpha)

#print("\nstate prediction: \n", sp)
#print("\nos prediction: \n", osp)

#pi = 0.0
#pt = 0.0
#pf = 0.0
#pg = 0.0
#ps = 0.0
#
#num = 500
#
#for i in range(num):
#    bkt_model = bkt.bkt(0.1, 0.3, 0.0, 0.1, 0.03)
#    #bkt_model.print_parameters()
#
#    ss, os = bkt_model.synthetic_data(100)
#    #print('\nss:', ss)
#
#    alpha = bkt_model.baum_welch(os, 0.001)
#
#    pi += bkt_model.init
#    pt += bkt_model.transit
#    pf += bkt_model.forget
#    pg += bkt_model.guess
#    ps += bkt_model.slip
#
#print('\npi', pi/num)
#print('\npt', pt/num)
#print('\npf', pf/num)
#print('\npg', pg/num)
#print('\nps', ps/num)
