import numpy as np
from scipy.stats import multivariate_normal

class MultivariateGaussian(object):
    # initialize n_dim、 mean、 covariance matrix
    def __init__(self, mean, cov):
        
        self.n_dim = mean.shape[0]
        self.mean = mean
        self.cov = cov
        
        for i in range(self.n_dim):
            if np.abs(self.cov[i]) < 1e-3:
                self.cov[i] = 1e-3
        
        #self.cov += np.random.randn(cov.shape[0], cov.shape[1]) * 40
        #self.precision = np.linalg.inv(self.cov)
        

        #self.normal = multivariate_normal(mean = mean, cov = cov)
    
    # 计算x在该多维高斯分下出现的概率, 注意：概率密度可以大于1，是分布函数一定<=1
    def log_prob(self, X):
             
        diff = X - self.mean
        log_det = 0
        for i in range(self.n_dim):
            log_det += np.log(np.abs(self.cov[i]))
        exponent = 1
        for i in range(self.n_dim):
            exponent += (diff[i] ** 2) / self.cov[i]
        log_pro = -0.5 * (self.n_dim * np.log(2 * np.pi) + log_det + exponent)
        return log_pro
        
    '''
    exponent = -1/2 * np.matmul(np.matmul(diff.T, self.precision), diff)
    
    det = np.linalg.det(self.cov + np.random.randn(self.cov.shape[0], self.cov.shape[1]))
    #print(det)
    factor = 1 / np.sqrt(abs(det) * ((2 * np.pi) ** self.n_dim)) # np.pi与math.pi相同, 都是float类型

    prob = factor * np.exp(exponent) + 1e-300
    #return self.normal.pdf(x)
    
    return min(prob, 1e5)
    '''
    
        
        