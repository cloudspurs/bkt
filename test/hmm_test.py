import sys
sys.path.append("/home/mengqinggang/workspaces/aicfe/bkt/src")
import numpy as np
from hmm import hmm


pi = [0.333, 0.333, 0.333]
A = [[0.333, 0.333, 0.333], [0.333, 0.333, 0.333], [0.333, 0.333, 0.333]]
B = [[0.5, 0.5], [0.75, 0.25], [0.25, 0.75]]

os = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1]
model = hmm(pi, A, B)
model.random_model()
print(model.__doc__)
model.baum_welch(os, 0.001)
model.print_parameters()
