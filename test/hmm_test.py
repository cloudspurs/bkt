import sys
sys.path.append("/home/mengqinggang/workspaces/aicfe/bkt/src")
import numpy as np
from hmm import hmm


pi = [0.333, 0.333, 0.333]
A = [[0.95, 0.05, 0.05], [0.45, 0.1, 0.45], [0.45, 0.45, 0.1]]
B = [[0.5, 0.5], [0.75, 0.25], [0.25, 0.75]]

#pi = [0.333, 0.333, 0.333]
#A = [[0.5, 0.3, 0.2], [0.2, 0.4, 0.4], [0.7, 0.2, 0.1]]
#B = [[0.5, 0.5], [0.75, 0.25], [0.25, 0.75]]

os_1 = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1]
os_2 = [0, 1, 0, 1, 1, 0, 1, 0, 1, 1]

os = []
os.append(os_1)
os.append(os_2)

model = hmm(pi, A, B)

#a, b, c = model.forward_with_scale(os)
#c = model.forward(os)

#model.baum_welch(os_1, 0.01)

model.multi_os_baum_welch(os)
