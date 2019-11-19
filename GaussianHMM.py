from hmm import HMM
from MultivariateGaussian import MultivariateGaussian
import numpy as np
import sklearn.cluster as cluster
class GaussianHMM(HMM):

    # (super)保存隐含状态数目，初始化初始状态概率，状态转移概率
    # (GaussianHMM) 初始化观测值维度，每个状态发射概率为高斯分布——初始化均值和方差
    def __init__(self, initial_prob, transition_prob, means, covs):
        super().__init__(initial_prob, transition_prob)
        self.n_dim = means.shape[1]
        self.means = means
        self.covs = covs
        #self.precisions = np.linalg.inv(self.covs) # 协方差矩阵的逆矩阵,通常称为精度矩阵(precision matrix)
    def kmeans_init(self, Qs):
        init = 1. / self.n_hidden
        self.initial_prob = np.full(self.n_hidden, init)
        self.transition_prob = np.full((self.n_hidden, self.n_hidden),init)
        # Qs 
        X = np.vstack(tuple(Qs))

        kmeans = cluster.KMeans(n_clusters=self.n_hidden,
                                    random_state=None)
        kmeans.fit(X)
        self.means = kmeans.cluster_centers_
        cv = np.cov(X.T) + 1e-3 * np.eye(X.shape[1])
        self.covs = np.tile(np.diag(cv), (self.n_hidden, 1))

    # 参数(pi, A, B)中已经有了pi, A, 由于是连续的发射概率, B未显式给定，下面根据连续分布函数求离散的B
    # 命名原因: '似然'的含义：计算生成数据的概率; 此处的'似然'的含义: 假设发射概率为多维高斯分布，计算每时刻每种状态生成给定观测值的概率
    # input  X, X[t] = ot
    #           X.shape = (T, n_dim).
    # return B, B[t, i] = P(ot|zt = i)
    #           B.shape = (T, n_hidden).
    def log_likelihood(self, X):
        
        # diff = X[:,None,:] - self.means  给定时刻t， 计算ot与每个状态发射均值的差值，所以要先扩展X，使之在时刻t有n_hidden个相同的值，再分别减去各状态的发射均值
        T = X.shape[0]
        diff = np.zeros((T, self.n_hidden, self.n_dim), dtype = 'float64')   
        for t in range(T):
            for i in range(self.n_hidden):
                diff[t][i] = X[t] - self.means[i]
        
        log_likelihood = np.zeros((T, self.n_hidden), dtype = 'float64')
        for t in range(T):
            for i in range(self.n_hidden):

                log_pb = MultivariateGaussian(self.means[i], self.covs[i]).log_prob(X[t])  # t时刻, 第i个状态的高斯分布生成该时刻观测值X[t]的概率
                log_likelihood[t][i] = log_pb
                #if np.isnan(pb):
                #    likelihood[t][i] = 1e50 # TODO:
                #else:
                #    likelihood[t][i] = min(pb, 1e50)

        return log_likelihood
    
    '''
    params:
        Qs       (num, T_k)
        epsilons (num, T_k, n_hidden, n_hidden)
        gammas   (num, T_k, n_hidden)
    update:
        initial_prob[i] = gamma(1,i)
                 shape  = (n_hidden, )
        transition_prob[i,j] = sum_t epsilon_t(i, j) / sum_t gamma_t(i)
                 shape  = (n_hidden, n_hidden)
        means[i] = ui = sum_t gamma(t, i) * ot / sum_t gamma(t, i)
                      = gamma(,i) X Q / sum_t gamma(t, i)
    '''
    def maximize(self, Qs, epsilons, gammas):
        # TODO: 应该传入取平均之后的epsilon和gamma, 传入列表没有意义。
        num = len(Qs)
        self.initial_prob = np.zeros(self.n_hidden, dtype = 'float64')
        for k in range(num):
            self.initial_prob += gammas[k][0]
        self.initial_prob /= num  # 经过debug, initial_prob正确, 总和为1        
        '''
        transition_prob = sum([epsilon.sum(axis = 0) for epsilon in epsilons]) # 先对每个epsilon分别求和, 在把这些加起来
        transition_prob = np.nan_to_num(transition_prob)
        self.transition_prob = transition_prob / np.nan_to_num(sum([gamma.sum(axis = 0) for gamma in gammas])) # 矩阵求和时axis = 0, 表示将每行加和得到一行
        '''
        transition_prob = np.zeros((self.n_hidden, self.n_hidden))
        for i in range(self.n_hidden):
            for j in range(self.n_hidden):
                for epsilon in epsilons:
                    transition_prob[i][j] += epsilon[:,i,j].sum()
                    #factor[i] += 
        
        #transition_prob = np.nan_to_num(transition_prob) # TODO:
        #transition_prob += 1e-300 # TODO: transitionProb什么时候会出现特殊情况?
        #transition_prob = (transition_prob.T / np.nan_to_num(transition_prob.sum(axis = 1))).T
        for i in range(self.n_hidden):
            tmp = 0
            for k in range(num):
                tmp += gammas[k][:,i].sum()
            #tmp /= num

            transition_prob[i] /= tmp
        # 注意，这里让矩阵每一行除以每一行的和，numpy中是矩阵除以数组，数组大小为矩阵行数，故需要将矩阵转置，使得矩阵每行元素个数等于原来的行数，方便numpy的广播
        # 联合概率除以先验概率 = 条件概率, 且每行概率和为1. 其中先验概率为每行各元素的和, 相当于对列向量求和, 故axis = 1
        self.transition_prob = transition_prob

        for i in range(self.n_hidden):
            ui = np.zeros((self.n_dim,))
            denom = 0 # gamma[:,i]之和, 分母项
            for k in range(num):
                gamma = gammas[k]
                for t in range(Qs[k].shape[0]):
                    ui += gamma[t][i] * Qs[k][t]
                    denom += gamma[t][i]
            ui /= denom
            #ui = sum([np.matmul(gamma[:,i], Q) for Q, gamma in zip(Qs, gammas)]) / num # 均值 = 所有时刻 生成状态i的概率*该时刻观测值 的平均值
            self.means[i] = ui
            #print(gammas[0][:,i].shape) (100,) 每个时刻出现状态i的概率 T维数组
            #print((Qs[0] - ui).shape)   (100,2) 每时刻与状态i方差的差值向量， T*n_dim维

        covs = np.zeros((self.n_hidden, self.n_dim, self.n_dim), dtype = 'float64')
        denom = np.zeros(self.n_hidden)
        for k in range(num):
            Q = Qs[k]
            T = Q.shape[0]
            gamma = gammas[k]
            for t in range(T):
                for i in range(self.n_hidden):
                    diff = Q[t] - self.means[i]
                    diff.shape = (1, diff.shape[0])
                    covs[i] += gamma[t][i] * np.matmul(diff.T, diff)
                    denom[i] += gamma[t][i]
        
        
        #print(sum([gamma.sum(axis = 0) for gamma in gammas]).shape): (3,) 没问题
        #print(covs.shape): (3,2,2) 没问题
        #sum_gamma = sum([gamma.sum(axis = 0) for gamma in gammas])
        for i in range(self.n_hidden):
            covs[i] /= denom[i]
            covs[i] += 1e-3
            #covs[i] /= (sum_gamma[i] + 1e-3/self.n_hidden)
        # covi = sum([np.matmul(gamma[:,i], np.matmul((Q - ui).T, Q - ui)) for Q, gamma in zip(Qs, gammas)]) / num

        # 将协方差矩阵转成对角阵
        '''
        for k in range(self.n_hidden):
            for i in range(self.n_dim):
                for j in range(self.n_dim):
                    if i == j and abs(covs[k][i][j]) < 1e-5:
                        covs[k][i][j] = np.sign(covs[k][i][j]) * 1e-5
                    elif i != j:
                        covs[k][i][j] = 0 
        '''   
        new_covs = np.zeros((self.n_hidden, self.n_dim))
        for i in range(self.n_hidden):
            new_covs[i] = np.diag(covs[i])
        self.covs = new_covs
        
        return


    def supervision_train(self, Q, states):
        #states = self.viterbi(Q)
        cnt = np.zeros(self.n_hidden, dtype = 'int64')

        # 初始化 means
        means = np.zeros((self.n_hidden, self.n_dim), dtype = 'float64')
        for t, state in enumerate(states):
            cnt[state] += 1
            means[state] += Q[t]
        for state in states:
            means[state] /= cnt[state]

        # 初始化 covs
        covs = np.zeros((self.n_hidden, self.n_dim, self.n_dim), dtype = 'float64')
        for i in range(self.n_hidden):
            covs[i] = np.eye(self.n_dim, self.n_dim)
        for t, state in enumerate(states):
            diff = Q[t] - means[state]
            diff.shape = (1, self.n_dim)
            covs[state] += np.matmul(diff.T, diff)
        for state in states:
            covs[state] /= cnt[state]

        # 初始化 transition_prob
        trans_cnt = np.zeros((self.n_hidden, self.n_hidden), dtype = 'int64')
        for t, state in enumerate(states[:-1]):
            trans_cnt[state][states[t+1]] += 1
        transition_prob = np.zeros((self.n_hidden, self.n_hidden))
        for i in range(self.n_hidden):
            for j in range(self.n_hidden):
                if np.abs(trans_cnt[i].sum()) > 1e-5:
                    transition_prob[i][j] = trans_cnt[i][j] / trans_cnt[i].sum()
        # 初始化 initial_prob
        initial_prob = np.zeros(self.n_hidden, dtype = 'float64')
        initial_prob[states[0]] = 1
        return initial_prob, transition_prob, means, covs
    def viterbi_init(self, Qs, iter_max = 5):
        params = np.hstack((self.initial_prob.ravel(), self.transition_prob.ravel(), self.means.ravel(), self.covs.ravel()))
        for _ in range(iter_max):   
            sum_initial_prob = np.zeros((self.n_hidden))
            sum_transition_prob = np.zeros((self.n_hidden, self.n_hidden))
            sum_means = np.zeros((self.n_hidden, self.n_dim))
            sum_covs = np.zeros((self.n_hidden, self.n_dim, self.n_dim))
            for Q in Qs:
                states = self.viterbi(Q)
                #print('states:',states)
                initial_prob, transition_prob, means, covs = self.supervision_train(Q, states)  # 给定观测值序列Q 和 状态序列states, 利用监督学习/频率=概率 估计hmm参数
                sum_initial_prob += initial_prob
                sum_transition_prob += transition_prob
                sum_means += means
                sum_covs += covs
            self.initial_prob = sum_initial_prob / len(Qs)
            self.transition_prob = sum_transition_prob / len(Qs)
            self.means = sum_means / len(Qs)
            
            sum_covs /= len(Qs)
            # 将协方差矩阵转成对角阵
            for k in range(self.n_hidden):
                for i in range(self.n_dim):
                    for j in range(self.n_dim):
                        if i == j and abs(sum_covs[k][i][j]) < 1e-5:
                            sum_covs[k][i][j] = np.sign(sum_covs[k][i][j]) * 1e-5
                        elif i != j:
                            sum_covs[k][i][j] = 0
            flag = 0
            for k in range(self.n_hidden):
                for i in range(self.n_dim):
                    for j in range(self.n_dim):    
                        if (i == j and sum_covs[k][i][j] == 0) or (i != j and sum_covs[k][i][j] != 0):
                            flag = 1
                            break
            #if flag:
            #    print('!!!!!!!!!!!!!')
            #    print(covs)
            #    return
            self.covs = sum_covs

            params_new = np.hstack((self.initial_prob.ravel(), self.transition_prob.ravel(), self.means.ravel(), self.covs.ravel()))
            if np.allclose(params, params_new): # 逐元素 判断参数是否收敛, array中所有参数值收敛时返回True
                break
            else:
                params = params_new
        return
