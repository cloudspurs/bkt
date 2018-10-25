#coding:utf-8
import numpy as np
import math
'''
这个函数主要是来解决第三个问题:
给定足够的观测数据， 如何估计出HMM的参数（即转移概率矩阵,生成概率矩阵, 初始隐含状态矩阵）, 训练声学模型
在前两个问题一个被用来测量一个模型的相对适用性, 另一个用来推测隐藏的部分到底在做什么, 但是它们都依赖于HMM模型参数,
然而这些参数往往都不是直接统计出来的,而是通过训练得到的
前向-后向算法首先对于隐马尔科夫模型的参数进行一个初始的估计（这很可能是完全错误的），然后通过对于给定的数据评估这些参数的的价值并减少它们所引起的错误,
并以此来重新修订这些HMM参数。
用到的前向-后向算法，是对于网格中的每一个状态，它既计算到达此状态的“前向”概率（给定当前模型的近似估计），又计算生成此模型最终状态的“后向”概率（给定当前模型的近似估计）。
这些都可以通过利用递归进行快速地计算，就像我们已经看到的前向算法和viterbi算法。可以通过利用近似的HMM模型参数来提高这些中间概率进行调整，而这些调整又形成了前向-后向算法迭代的基础。
暂时只有一次迭代
log_prob_forward 根据log_prob_pre = log_pro_forward的值进行判断,
更新之后的log_prob_forward - log_prob_pre < DELTA(default=0.001)时停止迭代
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
 
#观察序列,T个观察值
observation_sequence = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1]
T = len(observation_sequence)
 
#存储前向的每一个隐含节点的局部概率
partial_prob_forward = np.zeros((T, hmm.N))
log_prob_forward = 0.0
 
def forwardWithScale(observation_sequence, partial_prob, log_prob_forward):
    '''
    前向算法
    Args:
        observation_sequence: 观察序列
        partial_prob: 局部概率
        pprop: 局部概率和取对数
    Returns: scale, 每一个时刻的所有状态的局部概率之和
    '''
    #存储某一个时刻的局部概率之和, backward用到了forward的结果,需要返回
    scale = np.zeros(T)
 
    #t=0时刻的初始状态
    for i in range(hmm.N):
        #该状态初始的概率乘以该状态下出现观察状态的概率
        partial_prob[0][i] = hmm.PI[i] * hmm.B[i][observation_sequence[0]]
        scale[0] += partial_prob[0][i]
    print('at t=0, scale is',scale)
 
    #将t=0时刻的每一个状态的局部概率更新为 局部概率/当前时刻所有状态的局部概率之和 避免局部概率值太小,在后续的计算中出现下溢出
    for i in range(hmm.N):
        partial_prob[0][i] /= scale[0]
        print(partial_prob[0][i])
 
    for t in range(1, T):
        #求t=1...T-1时刻的局部概率
        for j in range(hmm.N):
            temp = 0.
            #前一时刻每个状态到此隐含节点
            for i in range(hmm.N):
                temp += partial_prob[t-1][i] * hmm.A[i][j]
            #当前时刻该隐含节点的局部概率值还需要乘以观察到observation_sequence[t]的生成概率
            partial_prob[t][j] = temp*hmm.B[j][observation_sequence[t]]
            scale[t] += partial_prob[t][j]
 
        for j in range(hmm.N):
            partial_prob[t][j] /= scale[t]
    #对每一时刻的局部概率之和取对数
    for t in range(T):
        log_prob_forward += math.log(scale[t])

    return scale
 
#求出前向的结果
forward_scale = forwardWithScale(observation_sequence, partial_prob_forward, log_prob_forward)
print( "前向一次的局部概率矩阵是: ", partial_prob_forward)
plogprobinit = log_prob_forward  #log P(O |intial model)
 
partial_prob_backward = np.zeros((T, hmm.N))
log_prob_backward = 0.
 
def backwardWithScale(scale, observation_sequence, partial_prob, log_prob_backward):
    '''
    Args:
        scale: 前向计算的每一时刻的局部概率和
        observation_sequence:
        partial_prob: 后向的局部概率矩阵
        log_prob_backward: 局部概率和取对数
    Returns:
    '''
    #t=T时刻各隐含节点的初始值为前向计算的 局部概率和倒数 感觉没什么意义啊
    for i in range(hmm.N):
        partial_prob[T-1][i] = 1.0/scale[T-1]
 
    #求的t=T-2, ..., 0 时刻的局部概率
    t = T-2
    while t >= 0:
        for j in range(hmm.N):
            temp = 0.
            #后一时刻到当前时刻转移,画图比较直观
            for i in range(hmm.N):
                temp += partial_prob[t+1][i] * hmm.A[i][j] * hmm.B[j][observation_sequence[t+1]]
            partial_prob[t][j] = temp / scale[t]
        t = t-1
# ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????        
    for t in range(T):
        log_prob_backward += partial_prob[1][i]
# ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????        
 
#求出后向的结果
print("前向求出的scale is :",  forward_scale)
backwardWithScale(forward_scale, observation_sequence, partial_prob_backward , log_prob_backward)
print("后向的局部概率矩阵是: ", partial_prob_backward)
 
#在已知模型和观察序列 求 P(si | O, \lambda)
gamma = np.zeros((T, hmm.N))
def computeGamma(gamma):
    '''
    计算在给定观察序列 O 和 模型 \lambda = (PI, A, B), 利用前向和后向的结果求出t时刻的的每个隐含状态si的概率
    将gamma的结果作为新的初始概率PI
    '''
    for t in range(T):
        #分母是 \sum_{i=1}^{N} P(si | O, \lambda), 其中N是隐含状态的数目
        denominator = 0.0
        for j in range(hmm.N):
            gamma[t][j] = partial_prob_forward[t][j] * partial_prob_backward[t][j]
            denominator += gamma[t][j]
 
        #最后除以分母,确保 \sum_{i=0}^{i=N-1} \gamma[i] = 1
        for i in range(hmm.N):
            gamma[t][i] = gamma[t][i] / denominator
 
computeGamma(gamma)
print("最后得到的gamma矩阵是: ", gamma)
 
xi = np.zeros((T, hmm.N, hmm.N))
def computeXi(xi, observation_sequence):
    '''
    \epsilon(t, i, j) = xi(t, i, j) 也可以表示为前向后向变量
    xi(t, i, j) = P(si, sj | O, 入) si和sj的联合概率
    计算在t时刻是si 并且 在t+1时刻是sj的概率 P(si, sj),  有gamma = P(si) 和 xi = P(si, sj)
    就可以求出状态转移概率 P(sj|si), 更新的状态转移矩阵A
    '''
    for t in range(T-1):
        temp = 0.
        for i in range(hmm.N):
            for j in range(hmm.N):
                #
                xi[t][i][j] = partial_prob_forward[t][i]*hmm.A[i][j]\
                              *hmm.B[j][observation_sequence[t+1]]\
                              *partial_prob_backward[t+1][j]
                temp += xi[t][i][j]
        for i in range(hmm.N):
            for j in range(hmm.N):
                xi[t][i][j] = xi[t][i][j] / temp
computeXi(xi, observation_sequence)
print("最后得到的 \epsilon 矩阵是: ", xi)
 
'''
其中 \gamma_{t}(i) = \sum_{j=0}^{N-1} \epsilon_{t}(i,j)
E(si) = \sum_{t=0}^{T-1} \gamma_{t}(i)  #把时间从t=0 到 t = T 某个节点的局部概率都加起来, 可以看作从其他状态到si状态的期望
E(si,sj) = \sum_{t=0}^{T-1} \epsilon_{t}(i,j) #把时间从t=0到t=T某个si 到 sj 节点的概率都加起来,可以看作从si 到sj状态转移期望值
'''
 
print("开始更新PI,  A ,  B,  动量是0.999 ")
#将gamma的t=0 时刻的状态就是更新后的状态矩阵
for i in range(hmm.N):
    hmm.PI[i] = 0.001 + 0.999 * gamma[0][i]
print("更新后的初始状态矩阵是: ", hmm.PI)
 
for i in range(hmm.N):
 
    '''
    更新状态转移矩阵A, 有gamma 和 xi 就可以求出 P(s(t+1,j)|s(t, i)),
    P(sj | si) = P(si, sj) | P(si)
               = P(si, sj) | P( si | O, \lamba)
               = \sum_{t=0}^{T} Xi[t][i][j] | \sum_{t=0}^{T} Gamma[t][i]
    '''
    denominatorA =  0.  #先用gamma求分母
    for t in range(T-1):  #由于xi的 T-1 时刻全是0
        denominatorA += gamma[t][i]

    for j in range(hmm.N):
        numeratorA = 0.  #再用 xi 求分子
        for t in range(T-1):
            numeratorA += xi[t][i][j]
        #更新隐含状态转移矩阵
        hmm.A[i][j] = 0.001 + 0.999 * numeratorA / denominatorA
 
    '''
    更新生成概率矩阵 B: P(ot | st) = P(ot, st) / P(st)
                   Bayes公式展开: = P(st | ot) * P(ot) / P(st):
                                 = P(st | ot) * P(ot) / \sum{t=0}^{T} P(st | ot, new \lambda)
    '''
    #分母是
    denominatorB = denominatorA + gamma[T-1][i]
    for k in range(hmm.M):
        numeratorB = 0.
        for t in range(T):
            if observation_sequence[t] == k:
                numeratorB += gamma[t][i]
        hmm.B[i][k] = 0.001 + 0.999 * numeratorB / denominatorB
print("更新后的状态转移矩阵A是: ", hmm.A)
print("更新后的生成概率矩阵B是: ", hmm.B)

