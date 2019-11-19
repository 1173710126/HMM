import numpy as np
from get_mfc_data import get_mfc_data
from GaussianHMM import GaussianHMM

if __name__ == "__main__":
    datas = get_mfc_data('C:/Users/18341/Desktop/book/听觉/实验3-语音识别/语料/features/')
    '''
    for key in datas:
        print(len(datas[key]))
        print(len(datas[key][0]))
    '''
    hmms = dict()
        
    '''
    原始版本的训练和测试, 训练过程中只能看到最后的测试结果
    下面未注释版本每训练一次就测试一次, 可以看到多次测试结果
    for category in datas:
        Qs = datas[category]
        n_hidden = 6
        #initial_prob = np.random.randn(n_hidden)
        #transition_prob = np.random.randn(n_hidden, n_hidden)
        initial_prob = np.ones((n_hidden))
        initial_prob /= n_hidden
        transition_prob = np.ones((n_hidden, n_hidden))
        transition_prob /= n_hidden
        
        n_dim = len(Qs[0][0])
        means = np.random.randn(n_hidden, n_dim)  
        covs = np.random.randn(n_hidden, n_dim, n_dim)
        for i in range(n_hidden):
            covs[i] = np.eye(n_dim, n_dim)

        hmm = GaussianHMM(initial_prob, transition_prob, means, covs)
        hmm.viterbi_init(Qs, iter_max=5)
        print('success viterbi_init')
        hmm.fit(Qs[:-3], iter_max = 10)
        print('success fit')
        hmms[category] = hmm
    '''  
    for category in datas:
        Qs = datas[category]
        n_hidden = 5
        #initial_prob = np.random.randn(n_hidden)
        #transition_prob = np.random.randn(n_hidden, n_hidden)
        initial_prob = np.ones((n_hidden))
        initial_prob /= n_hidden
        transition_prob = np.ones((n_hidden, n_hidden))
        transition_prob /= n_hidden
        
        n_dim = len(Qs[0][0])
        means = np.random.randn(n_hidden, n_dim) * 10
        covs = list()
        for _ in range(n_hidden):
            cov = np.eye(n_dim) 
            for i in range(n_dim):
                cov[i,i] *= (np.random.randn() * 10)
            covs.append(cov)
        covs = np.asarray(covs)
        hmm = GaussianHMM(initial_prob, transition_prob, means, covs)
        #hmm.viterbi_init(Qs, iter_max=5) # TODO:
        hmm.kmeans_init(Qs)
        hmms[category] = hmm

    evaluate_num = 5
    for evaluate_cnt in range(evaluate_num):
        print(evaluate_cnt, 'start fit')
        for category in hmms:
            hmm = hmms[category]
            #print(hmm.covs)
            Qs = datas[category]
            hmm.fit(Qs[:-3], iter_max = 10)
            hmms[category] = hmm
            print('fit success')
        
        # test
        correct_num = 0
        for category in datas:
            for test_sample in datas[category][-3:]:
                print('real_category:', category)
                max_like = -1 * np.inf
                predict = -1
                for predict_category in hmms:
                    hmm = hmms[predict_category]
                    like = hmm.generate_prob(test_sample)
                    print('category', predict_category, '. like:', like)
                    if like > max_like:
                        max_like = like
                        predict = predict_category
                        #print('predict_category', predict_category)
                if predict == category:
                    correct_num += 1
                print('predict_category:',predict)
        print(correct_num / (3*5))