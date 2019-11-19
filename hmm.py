import numpy as np

class HMM(object):

    # 保存隐含状态数目，初始化初始状态概率，状态转移概率
    def __init__(self, initial_prob, transition_prob):
        self.n_hidden = initial_prob.shape[0]
        self.initial_prob = initial_prob
        self.transition_prob = transition_prob
    # 似然和M部都与发射概率B有关, 而B可以连续可以离散, 故这里暂不定义
    def log_likelihood(self, Q):
        raise NotImplementedError

    def maximize(self, Qs, epsilons, gammas):
        raise NotImplementedError

    # 传入多组序列进行训练
    def fit(self, Qs, iter_max = 2):
        
        params = np.hstack((self.initial_prob.ravel(), self.transition_prob.ravel()))  # 将参数平铺开, 便于下面计算参数是否以一定精度不变
        
        # 其实这里应该也计算发射概率b的变化? d高斯时计算均值和协方差的变化

        for _ in range(iter_max):             # 约定_未不打算使用的变量: 防止出现未使用变量的警告 
            epsilons, gammas = self.expect(Qs)  # E部:计算似然函数Q的系数: 根据旧参数计算后验概率
            self.maximize(Qs, epsilons, gammas) # M部:计算似然函数Q的极值点, 得到新的参数, 并更新到类里面

            params_new = np.hstack((self.initial_prob.ravel(), self.transition_prob.ravel()))
            if np.allclose(params, params_new): # 逐元素 判断参数是否收敛, array中所有参数值收敛时返回True
                break
            else:
                params = params_new

        return 

    '''
    E部:计算似然函数Q的系数: 根据旧参数计算后验概率
        epsilon[k,t,i,j] = P(zt = i, zt+1 = j | Qk), 第k组观测值的条件下, t时刻 zt = i, zt+1 = j (状态对i,j)出现的概率
        gamma[k,t,i] = P(zt = i| Qk), 第k组观测值的条件下, t时刻 zt = i (状态i)出现的概率 
        使用前向变量alpha, 后向变量belta, 用DP简化计算
        alpha[t,i] = P(o1,...,ot, zt = i)
        belta[t,i] = P(ot+1,...,oT | zt=i)
        性质:
        1.  P(o1,...,oT) 
            = sum_i alpha(T, i)
        2.  P(o1,...,oT, zt=i)  =   P(o1,...,ot, zt=i) * P(ot+1,...,oT | zt=i, o1,...,ot) = P(o1,...,ot, zt=i) * P(ot+1,...,oT | zt=i)
            = alpha(t,i) * belta(t,i)
        3.  P(o1,...,oT) 
            = sum_i alpha(t,i) * belta(t,i)
        epsilon, gamma可用alpha, belta表示
        epsilon[k,t,i,j] = P(zt = i, zt+1 = j | Qk) = P(zt = i, zt+1 = j, Qk) / P(Qk) 
                         = alpha(t,i) * a(i,j) * b(t+1,j) * belta(t+1,j) / P(Qk)
                         = alpha(t,i) * a(i,j) * b(t+1,j) * belta(t+1,j) / sum_i alpha(Tk, i)
        gamma[k,t,i] = P(zt = i| Qk) 
                     = sum_j epsilon[k,t,i,j]
                     = alpha(t,i) belta(t,i) / P(Qk)
    M部在子类中根据不同的发射概率模型实现 
    '''
    def expect(self, Qs):
        #alphas = self.forward(Qs)
        #beltas = self.backward(Qs)
        #alphas, beltas, _ = self.forward_and_backward(Qs)
        with np.errstate(divide = 'ignore'):
            log_transition_prob = np.log(self.transition_prob)
        alphas, beltas = self.forward_and_backward(Qs)
        epsilons = list()
        gammas = list()
        for k, Q in enumerate(Qs): # 对Qs中每个Q计算一次epsilon, gamma, 加入列表epsilons, gammas
            T = Q.shape[0]
            alpha = alphas[k]
            belta = beltas[k]
            
            log_likelihood = self.log_likelihood(Q)
            #likelihood = np.exp(log_likelihood) # 用log求发射概率,取指数后仍有可能得到0或者inf,故给发射概率取log, 而计算epsilon是还用乘积的话, 没有意义
            epsilon = np.zeros((T, self.n_hidden, self.n_hidden), dtype = 'float64')      
            for t in range(T-1):
                #scale = 0
                for i in range(self.n_hidden):
                    for j in range(self.n_hidden):        
                        epsilon[t][i][j] = alpha[t][i] + log_transition_prob[i][j] + log_likelihood[t+1,j] + belta[t+1,j] 
                        #scale += epsilon[t][i][j]#  删去归一化这一步？epsilon[t]的概率密度之和不一定为1
                                        #if scale != 0:
                #epsilon[t] /= scale # ------------------归一化-----------------
                epsilon[t] = self.log_normalize(epsilon[t].reshape(self.n_hidden ** 2,)).reshape(self.n_hidden, self.n_hidden)
            #Q_prob = alpha[T-1].sum()
            #if Q_prob != 0:
            #    epsilon /= Q_prob
            with np.errstate(under = 'ignore'):
                epsilon = np.exp(epsilon)
            epsilons.append(epsilon)

            gamma = np.zeros((T, self.n_hidden), dtype = 'float64')
            for t in range(T):
                for i in range(self.n_hidden):
                    gamma[t][i] = alpha[t,i] + belta[t,i]
                    # -----------前后向变量做防下溢处理后, gamma不用归一化,----------------- 
                gamma[t] = self.log_normalize(gamma[t])
            #gamma = np.exp(gamma) 取log时会出现log(0)警告
            # TODO: 对loggamma用log法做归一化
            
            with np.errstate(under = 'ignore'):
                gamma = np.exp(gamma)
            gammas.append(gamma)

        return epsilons, gammas


    # 有归一化的前后向算法
    def forward_and_backward(self, Qs):
        alphas = list()
        beltas = list()
        #scales = list()
        with np.errstate(divide = 'ignore'):
            log_initial_prob = np.log(self.initial_prob)
            log_transition_prob = np.log(self.transition_prob)
        for Q in Qs:
            T = Q.shape[0]   # 注意: shape, 不需要加括号
            log_likelihood = self.log_likelihood(Q)  # 求出各时刻各状态发射观察值的概率
            #likelihood = np.exp(log_likelihood)
            #scale = list()
            #print('means:', self.means)
            #print('covs:', self.covs)

            # --------alpha(t+1,i) = ( sum_j alpha(t, j) a[j,i] ) b[t+1,i]-----
            #scale = list()
            alpha = np.zeros((T, self.n_hidden))
            for i in range(self.n_hidden):
                alpha[0,i] = log_initial_prob[i] + log_likelihood[0][i]
            #scale.append(alpha[0].sum())  
            #alpha[0] = np.nan_to_num(alpha[0])
            #alpha[0] /= scale[0]
            for t in range(1, T):
                for i in range(self.n_hidden):
                    tmp = np.zeros(self.n_hidden,)
                    for j in range(self.n_hidden):
                        tmp[j] = alpha[t-1][j] + log_transition_prob[j][i]
                        #print(alpha[t-1][j],':1\n2:',log_transition_prob[j][i])
                    alpha[t][i] = self.logsumexp(tmp) + log_likelihood[t][i]
                #alpha[t] += 1e-33 # TODO:
                #alpha[t] = np.nan_to_num(alpha[t])
                #scale.append(alpha[t].sum())
                #scale[t] = np.nan_to_num(scale[t])
                #alpha[t] /= scale[t]
            alphas.append(alpha)

            #scales.append(scale)
            
            belta = list()
            belta_T = np.ones(self.n_hidden) # 初始时刻的belta
            #belta_T /= scale[T-1]
            belta.insert(0, belta_T)
            for t in range(T-2, -1, -1): # 这里及上文 t从T开始直到1, 故t时刻的似然存储在likelihood[t-1]内
                #belta_t = np.matmul(self.transition_prob, likelihood[t+1] * belta[0])
                #belta(t, i) = sum_j a(i,j) (b(t+1,j) belta(t+1,j))
                belta_t = np.zeros((self.n_hidden))
                for i in range(self.n_hidden):
                    tmp = np.zeros(self.n_hidden,)
                    for j in range(self.n_hidden):
                        tmp[j] = log_transition_prob[i][j] + log_likelihood[t+1,j] + belta[0][j]
                    belta_t[i] = self.logsumexp(tmp)
                #belta_t += 1e-30 
                #belta_t /= scale[t+1] # ----------------归一化------------------
                belta.insert(0, belta_t) # python list method: list.insert(index, obj) 在指定位置插入元素
            belta = np.asarray(belta)
            beltas.append(belta)

        #return alphas, beltas, scales
        return alphas, beltas


    def forward(self, Qs):
        """
        前向算法
        input : Qs 同一HMM模型的多个观测序列
        output: alpha[k, t, i] 第k个序列alpha(t,i)的值
                alpha[k].shape == (Tk, n_hidden)
        DP algorithm:
        define: alpha(t,i) = P(o1,...,ot,zt=i)
        init  : alpha(1,i) = P(o1,z1=i) 
                            = P(z1=i)P(o1|z1=i)
                alpha(1,) = init_prob * b_1
        more  : alpha(t+1,i) = P(o1,...,ot+1,zt+1=i) 
                            = sum_j P(o1,...,ot+1,zt=j, zt+1=i) 
                            = sum_j P(o1,...,ot,zt=j) P(ot+1,zt+1=i|zt=j,o1,...,ot)
                            = sum_j alpha(t, j) P(ot+1,zt+1=i|zt=j)
                            = sum_j alpha(t, j) P(zt+1=i|zt=j)P(ot+1|zt+1=i,zt=j)
                alpha(t+1,i) = ( sum_j alpha(t, j) a[j,i] ) b[t+1,i]
                alpha(t+1,i) = alpha(t,) X a[,i] * b[t+1,i]
                alpha(t+1,)  = alpha(t,) X a[,]  * b[t+1,]
        """        
        alphas = list()
        for Q in Qs:
            T = Q.shape[0]   # 注意: shape, 不需要加括号
            log_likelihood = self.log_likelihood(Q)  # 求出各时刻各状态发射观察值的概率
            likelihood = np.exp(log_likelihood)
            alpha = list()
            alpha_1 = self.initial_prob * likelihood[0] # 初始时刻的alpha
            alpha.append(alpha_1)
            for t in range(1, T):
                alpha_t = np.matmul(alpha[-1], self.transition_prob) * likelihood[t]
                alpha.append(alpha_t)
            alpha = np.asarray(alpha)
            alphas.append(alpha)
            
            '''
            for t in range(1,T):
                for i in range(self.n_hidden):
                    #alpha(t+1,i) = ( sum_j alpha(t, j) a[j,i] ) b[t+1,i]
                    x = 0
                    for j in range(self.n_hidden):
                        x += alpha[t-1, j] * self.transition_prob[j][i] * likelihood[t][i]
                    if abs(x-alpha[t, i]) > 1e-3:
                        return False
            '''
        return alphas

    def backward(self, Qs):
        '''
        后向算法
        input : Qs 同一HMM模型的多个观测序列
        output: belta[k, t, i] belta(t,i)的值, t时刻状态i生成该时刻观测值的概率
                belta[k].shape == (Tk, n_hidden)
        DP algorithm:
        define: belta(t, i) = P(ot+1,...,oT|zt=i)
        init  : belta(T, i) = 1  :make sure P(o1,...,oT,zT=i) = alpha(T,i)*belta(T,i)
        more  : belta(t, i) = P(ot+1,...,oT|zt=i)
                            = sum_j P(zt+1=j,ot+1,ot+2,...,oT|zt=i)
                            = sum_j P(zt+1=j,ot+1|zt=i) P(ot+2,...,oT|zt+1=j,ot+1,zt=i)
                            = sum_j P(zt+1=j|zt=i) P(ot+1|zt+1=j,zt=i) P(ot+2,...,oT|zt+1=j)
                belta(t, i) = sum_j a(i,j) (b(t+1,j) belta(t+1,j))
                belta(t, i) = a(i,) X ( b(t+1,) * belta(t+1,) ) = a number
                belta(t, )  = [a(,)  X ( b(t+1,) * belta(t+1,) )] = an array, so don't need traverse
        '''
        beltas = list()
        for Q in Qs:
            T = Q.shape[0]
            log_likelihood = self.log_likelihood(Q)  # 求出各时刻各状态发射观察值的概率
            likelihood = np.exp(log_likelihood)
            belta = list()
            belta_T = np.ones(self.n_hidden) # 初始时刻的belta
            belta.insert(0, belta_T)
            for t in range(T-2, -1, -1): # 这里及上文 t从T开始直到1, 故t时刻的似然存储在likelihood[t-1]内
                belta_t = np.matmul(self.transition_prob, likelihood[t+1] * belta[0])
                belta.insert(0, belta_t) # python list method: list.insert(index, obj) 在指定位置插入元素
            belta = np.asarray(belta)
            
            for t in range(1,T-1):
                for i in range(self.n_hidden):
                    # belta(t, i) = sum_j a(i,j) (b(t+1,j) belta(t+1,j))
                    x = 0
                    for j in range(self.n_hidden):
                        x += self.transition_prob[i][j] * likelihood[t+1][j] * belta[t+1][j]
                    if np.abs(belta[t,i] - x) > 1e-3:
                        return False   
            beltas.append(belta)

        return beltas

    # input: Q
    # output: 最佳状态链. 给定观测值Q时,该hmm模型中出现概率最大的状态链
    def viterbi(self, Q):
        T = Q.shape[0]
        delta = np.full((T, self.n_hidden), -1*np.inf, dtype = 'float64') # Q.shape = (T, n_dim)
        pre =np.zeros((T, self.n_hidden), dtype = 'int64')
        log_likelihood = self.log_likelihood(Q)
        for i in range(self.initial_prob.shape[0]):
            self.initial_prob[i] += 1e-50
        for i in range(self.transition_prob.shape[0]):
            for j in range(self.transition_prob.shape[1]):
                self.transition_prob[i][j] += 1e-50   
        #self.initial_prob += 1e-300
        for i in range(self.n_hidden):
            delta[0][i] = np.log(self.initial_prob[i]) + log_likelihood[0][i]
            pre[0][i] = 0
        #self.transition_prob += 1e-300
        for t in range(1, T):
            for j in range(self.n_hidden):
                for i in range(self.n_hidden):
                    if delta[t][j] < delta[t-1][i] + np.log(self.transition_prob[i][j]):
                        delta[t][j] = delta[t-1][i] + np.log(self.transition_prob[i][j])
                        pre[t][j] = i
                delta[t][j] += log_likelihood[t][j]
        maxDelta = -1 * np.inf
        q = -1
        for i in range(self.n_hidden):
            if delta[T-1][i] > maxDelta:
                maxDelta = delta[T-1][i]
                q = i
        states = [q]
        for t in range(T-1, 0, -1):
            q = pre[t][q]
            states.insert(0, q)
        return states


    def generate_prob(self, Q):
        alphas, beltas = self.forward_and_backward([Q])
        #print(scale)
        alpha = alphas[0]
        return self.logsumexp(alpha[len(Q)-1])

    def logsumexp(self, X):
        X_max = np.max(X)
        if np.isinf(X_max):
            return -1 * np.inf
        acc = 0
        for i in range(X.shape[0]):
            acc += np.exp(X[i] - X_max)
        return np.log(acc) + X_max

    def log_normalize(self, a, axis = None):
        """Normalizes the input array so that the exponent of the sum is 1.

        Parameters
        ----------
        a : array
            Non-normalized input data.

        axis : int
            Dimension along which normalization is performed.

        Notes
        -----
        Modifies the input **inplace**.
        """
        with np.errstate(under="ignore"):
            a_lse = self.logsumexp(a)
        a -= a_lse
        return a
    

