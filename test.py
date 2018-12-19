import numpy as np
import hmm
import bkt
import hmm_no_scale as hns

#os = np.random.randint(0, 2, size=[5000])
#os = np.full(100, 1)
#os[0] = 0

pi = [0.3, 0.7]
A = [[0.2, 0.8], [0.4, 0.6]]
B = [[0.7, 0.3], [0.2, 0.8]]

hmm_model = hmm.hmm(pi, A, B)
hmm_model.print_parameters()

ss, os = hmm_model.synthetic_data(2000)

a = b = c = d = 0

for i in range(len(ss)-1):
    if ss[i] == 0:
        # 0, 0
        if ss[i+1] == ss[i]:
            a += 1
        # 0, 1
        if ss[i+1] > ss[i]:
            b += 1
    if ss[i] == 1:
        # 1, 0
        if ss[i+1] < ss[i]:
            c += 1
        # 1, 1
        if ss[i+1] == ss[i]:
            d+=1
print('\n', a, b, c, d)
print("0 -> 0: ", a/(a+b))
print("0 -> 1: ", b/(a+b))
print("1 -> 0: ", c/(c+d))
print("1 -> 1: ", d/(c+d))

e = f = g = h = 0

for i in range(len(ss)):
    if ss[i] == 0:
        if os[i] == 0:
            e += 1
        if os[i] == 1:
            f += 1
    if ss[i] == 1:
        if os[i] == 0:
            g += 1
        if os[i] == 1:
            h += 1

print('\n', e, f, g, h)
print("0 -> 0: ", e/(e+f))
print("0 -> 1: ", f/(e+f))
print("1 -> 0: ", g/(g+h))
print("1 -> 1: ", h/(g+h))


#print('\nss:', ss)
#print('\nos:', os)

#print('\nos forward prob:\t', hmm_model.forward(os))
#print('\nos backward prob:\t', hmm_model.backward(os))

#pi = [0.2, 0.8]
#A = [[0.8, 0.2], [0.1, 0.9]]
#B = [[0.7, 0.3], [0.2, 0.8]]
#
#test_model = hmm.hmm(pi, A, B)
#test_model.baum_welch(os, 0.0001)
#test_model.print_parameters()

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

#print('\nestimate parameters: ') 
#bkt_model.print_parameters()

#sp, osp = bkt_model.predict(os, alpha)

#print("\nstate prediction: \n", sp)
#print("\nos prediction: \n", osp)
