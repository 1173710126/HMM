import numpy as np
from get_mfc_data import get_mfc_data
from GaussianHMM import GaussianHMM

if __name__ == "__main__":
    datas = get_mfc_data('C:/Users/18341/Desktop/book/听觉/实验3-语音识别/语料/features/')

    # 每个类别创建一个hmm, 并用kmeans初始化hmm
    hmms = dict()
    for category in datas:
        Qs = datas[category]
        n_hidden = 5     
        n_dim = Qs[0].shape[1]

        hmm = GaussianHMM(n_hidden,n_dim)
        #hmm.viterbi_init(Qs, iter_max=5) # TODO:维特比算法没测试
        hmm.kmeans_init(Qs[:-3])
        hmms[category] = hmm

    # 训练每个hmm
    print('start fit')
    for category in hmms:
        hmm = hmms[category]
        #print(hmm.covs)
        Qs = datas[category]
        hmm.fit(Qs[:-3], iter_max = 5)
        hmms[category] = hmm
        print(category, ':fit success')
    
    # 测试, 打印得分和最终正确率
    correct_num = 0
    for real_category in datas:
        for test_sample in datas[real_category][-3:]:
            print('real_category:', real_category)
            max_score = -1 * np.inf
            predict_category = -1
            for test_category in hmms:
                hmm = hmms[test_category]
                score = hmm.generate_prob(test_sample)
                print('test category ', test_category, '\'s score: ', score)
                if score > max_score:
                    max_score = score
                    predict_category = test_category
            if predict_category == real_category:
                correct_num += 1
            print('predict_category:',predict_category)
    print(correct_num / (3*5))
    