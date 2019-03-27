import sys
sys.path.append("/home/mengqinggang/workspaces/aicfe/bkt/src")
import numpy as np
import hmm
import bkt

bkt_model = bkt.bkt(0.1, 0.3, 0.0, 0.1, 0.03)
#bkt_model.random_model()

ss, os = bkt_model.synthetic_data(20)
print('\nss:', ss)
print('\nos:', os)

likelihood, ite_num, alpha = bkt_model.baum_welch(os, 0.001)
print('\nestimate parameters: ') 
bkt_model.print_parameters()

ss, os = bkt_model.predict(os, alpha)

print("\nstate prediction: \n", ss)
print("\nos prediction: \n", os)

