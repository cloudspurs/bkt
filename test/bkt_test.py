import sys
sys.path.append("/home/mengqinggang/workspaces/aicfe/bkt/src")
import numpy as np
import bkt

model = bkt.bkt(0.1, 0.4, 0.0, 0.1, 0.05)
model.print_parameters()
ss, os = model.synthetic_data(100)

best_model = bkt.bkt(0.1, 0.4, 0.0, 0.1, 0.05)
best_model.random_model()
best_log_like, number, alpha = best_model.baum_welch(os, 0.001)
print('log likelihood', best_log_like)
best_model.print_parameters()

#for i in range(10):
#    test_model = bkt.bkt(0.1, 0.3, 0.0, 0.1, 0.03)
#    test_model.random_model()
#    log_like, number, alpha = test_model.baum_welch(os, 0.001)
#    if(log_like > best_log_like):
#        best_model = test_model
#
#print(best_log_like)
#best_model.print_parameters()

