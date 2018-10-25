import  numpy as np
'''
这个函数主要是来解决第一个问题:
给定一个模型，如何计算出某个特定的输出序列的概率 P(O | 入),  评估计算
我们使用前向算法（forward algorithm）来计算给定隐马尔科夫模型（HMM）后的一个观察序列的概率，并因此选择最合适的隐马尔科夫模型(HMM)。
在语音识别中这种类型的问题发生在当一大堆数目的马尔科夫模型被使用，并且每一个模型都对一个特殊的单词进行建模时。
一个观察序列从一个发音单词中形成，并且通过寻找对于此观察序列最有可能的隐马尔科夫模型（HMM）识别这个单词。
前向变量定义(局部概率):
a(t , i ) =  P(o1, o2, ..., ot, si| 入) ,  在给定模型 入 时, 部分观测序列o1, o2, .., ot 和t 时刻状态处于 si 的联合概率
注意: 后边加范围的表示求和:
step1:
    a(1,i) = PI(i) *  B(i, o1)   1<= i <= N
step2:
    a(t+1, i) =  ( a(t, i) * Aij , 1<= i <= N ) * bj(ot+1)  , 1<= t <= T,  1<= j <= N
step3:
    P(o1, o2, ..., ot| 入) = a(T, i), 1<= i <= N
现在我们定义一个HMM 模型 weather,  现在我们需要求解的是输出序列T = 3 ,  Dry, Damp, Soggy 的概率
'''

class weather:
    '''
    M:  是观察状态的数量, 海藻的干湿度(Dry，Dryish，Damp，Soggy)
    N:  是隐含状态的数量, 天气的情况(Sunny，Cloudy，Rainy)
    A:  天气之间的状态转移矩阵
    B:  天气到海藻湿度的生成概率矩阵
    PI: 状态的初始概率
    '''
    M = 4
    N = 3
    A =  [[0.500, 0.375, 0.125],
          [0.250, 0.125, 0.625],
          [0.250, 0.375, 0.375]]
    B = [[0.60, 0.20, 0.15, 0.05],
         [0.25, 0.25, 0.25, 0.25],
         [0.05, 0.10, 0.35, 0.50]]
    PI = [0.63, 0.17, 0.20]
 
T = 3
observation_sequence = [0, 2, 3]
partail_prob = np.zeros((T, weather.N))
 
print("初始的局部概率矩阵\n", partail_prob)
 
def forward(observation_sequence):
    #初始化, 计算t=0时刻的所有状态的局部概率,也就是计算所有的初始隐含状态 到 t=1时刻观察到的状态的 概率
    for i in range(weather.N):
        partail_prob[0][i] = weather.PI[i] * weather.B[i][observation_sequence[0]]
 
    print("t=1时刻的局部概率矩阵\n", partail_prob)
 
    #计算t>0时刻的局部概率, t1 时刻是 [(在t=1的局部概率的si 到 sj(j从1到N) 的概率) * 生成概率矩阵中的是(si->ot)] , i从1到N
    for t in range(1, T):
        for j in range(weather.N):
            temp = 0.
            for i in range(weather.N):
                temp += partail_prob[t-1][i] * weather.A[i][j]
            partail_prob[t][j] = temp * weather.B[j][observation_sequence[t]]

    print("所有时刻的局部概率矩阵\n", partail_prob)
 
    #计算T-1时刻的所有局部概率之和 就等于 最终的观测序列的概率
    result_prob = 0.
    for i  in  range(weather.N):
        result_prob += partail_prob[T-1][i]
 
    print("最中的P(O | 入) = ", result_prob)
 
forward(observation_sequence)

