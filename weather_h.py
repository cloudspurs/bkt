#coding:utf-8
import numpy as np
'''
这个函数主要是来解决第二个问题:
给定一个模型和已知的观测输出序列 搜索最可能的隐藏状态序列, 解码过程
现在我们定义一个HMM 模型 weather,  现在我们需要求解序列为[0, 0, 0, 0, 1, 0, 1, 1, 1, 1], 最可能的隐藏状态序列
'''
class hmm:
    '''
    M:  是观察状态的数量, 海藻的干湿度(Dry ，Damp)
    N:  是隐含状态的数量, 天气的情况(Sunny，Cloudy，Rainy)
    A:  天气之间的隐含状态转移矩阵
    B:  天气到海藻湿度的生成概率矩阵
    PI: 状态的初始概率
    '''
    M = 2
    N = 3
    A =  [[0.500, 0.375, 0.125],
          [0.250, 0.125, 0.625],
          [0.250, 0.375, 0.375]]
    B = [[0.50, 0.50],
         [0.75, 0.25],
         [0.25, 0.75]]
    PI = [0.333, 0.333, 0.333]
 
T = 10
observation_sequence = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1]
partial_prob = np.zeros((T, hmm.N))
#隐藏状态序列
psi  = np.zeros((T, hmm.N))
 
def viterbi(observation_sequence):
    #计算t=0时刻的观测到observation_sequence[0]的概率
    for i in range(hmm.N):
        partial_prob[0][i] = hmm.PI[i] *  hmm.B[i][observation_sequence[0]]
# ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        psi[0][i] = 0 #起始点到t=0时刻各个隐含状态的点的前一个时刻的状态的id ,  由于前一个时刻只有一个起始点,所以都是0
 
    print("t=0时刻的局部概率矩阵\n", partial_prob)
    print("t=0时刻最佳的路径id集合 \n", psi)
 
    #从t=1时刻开始
    for t  in range(1, T):
        for i in range(hmm.N):
            maxval = 0.
            maxvalind = 0 #在t时刻取到maxval时对应的隐含状态的id
            for j in range(hmm.N):
                val = partial_prob[t-1][j] * hmm.A[j][i]
                if  val > maxval:
                    maxval = val
                    maxvalind = j
            partial_prob[t][i] =  maxval * hmm.B[i][observation_sequence[t]]
            psi[t][i] = maxvalind  #记录的是到目前位置到si的最优路径的上一个时刻的节点的id
    print("所有时刻的局部概率矩阵\n",partial_prob)
    print("所有时刻最佳的路径id集合 \n",psi)
 
    #计算所有T 时刻最大概率的那条路径
    result_prob = 0.
    q = [0]*T

    for i in range(hmm.N):
        if partial_prob[T-1][i] > result_prob:
            result_prob = partial_prob[T-1][i]
            q[T-1] = i
    print("最优路径的概率是: ", result_prob)
    print("最后一个时刻的概率最大值对应的隐含节点是: ", q[T-1])
    
    #路径回溯,找到最优路径, 通过最终的概率最大的那个节点开始从最优路径id表中追溯出最优的隐含状态路径
    t = T-2
    while t >= 0:
        q[t] = int(psi[t+1][q[t+1]])
        t = t-1
    print(q)
 
viterbi(observation_sequence)

