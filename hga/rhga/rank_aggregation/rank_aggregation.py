import numpy as np
import math
import pandas as pd

from hga.rhga.rank_aggregation.correlation import Correlation
from hga.rhga.rank_aggregation.information_gain import InformationGain
from hga.rhga.rank_aggregation.relief import ReLief


class RankAggregation(object):
    '''
        Class RankAggregation : aggregate multiple list rank into one list rank
    '''
    # -------------------- Kendall and Sprearman -------

    def spearman_distance(self, dict_1,dict_2):
        '''

        :param dict_1: key:value = feature_name:rank
        :param dict_2: key:value = feature_name:rank
        :return: spearman distance between dict_1 and dict_2
        '''
        # difference rank between each feature in 2 dict
        diff = [abs(dict_1[i] - dict_2[i]) for i in dict_1.keys()]
        return sum(diff)

    def spearman_distance_l(self, array_1, array_2):
        '''

        :param array_1: array rank 1
        :param array_2: array rank 2
        :return: spearman distance
        '''
        l1 = list(array_1)
        l2 = list(array_2)
        diff = [abs(i-l2.index(v)) for i,v in enumerate(l1)]
        return sum(diff)


    def spearman_distance_multilple(self, x, y, importance=0):
        '''

        :param x: lists to be combined
        :param y: candidate lists
        :param importance: the weight factors indicating the importance of ordered lists
        :return: spearman distance
        '''
        if importance == 0:
            importance = np.ones(x.shape[0])
            importance = importance/np.sum(importance)
        k = y.shape[1]
        N = x.shape[0]
        result = [0 for i in range(y.shape[0])]
        for i,cand in enumerate(y):
            tmp = 0
            for j in x:
                tmp += self.spearman_distance_l(cand, j)
            result[i] = tmp
        return result


    def rank_aggregation(self, x,k,pop_size,CP,MP,max_iter,convIn):
        '''

        :param x: lists to be combined
        :param k: number element in list
        :param pop_size: population size of genetic algorithm
        :param CP: crossover rate of genetic algorithm
        :param MP: mutation rate of genetic algorithm
        :param max_iter: max generation in genetic algorithm
        :param convIn: number of generation can be converge
        :return: best final rank list
        '''
        importance = np.ones(x.shape[0])
        importance = importance/np.sum(importance)
        features_list = list(np.unique(x))
        n_features = len(features_list)
        # encode matrix of feature list
        for l in x:
            for i,v in enumerate(l):
                l[i] = int(features_list.index(v))
        x = x.astype(np.int)

        # genarate initial population randomly
        cands = np.zeros((pop_size,k),dtype=int)
        for i in range(pop_size):
            cands[i,:] = np.random.choice(np.arange(0,k),k,replace=False)
        # calculate objective function
        scores = self.spearman_distance_multilple(x,cands)
        best_cand = cands[scores.index(np.min(scores)),:]
        best_scores = np.min(scores)

        conv = False
        t = 1
        inter = 0
        while conv==False :
            print("+++++++++Iter {}++++++++".format(inter))
            # calculate probability for candidate by fitness scores
            min_scores = np.min(scores)
            prob_cands = (np.max(scores)+1-scores)/sum((np.max(scores)+1-scores))
            cum_prob_cands = np.cumsum(prob_cands)


            # selection candidate for the next generation
            ind = np.random.uniform(0,1,pop_size)
            ind2 = [0 for i in range(pop_size)]
            for i in range(pop_size):
                ind2[i] = sum(ind[i] > cum_prob_cands)
            cands = cands[ind2,:]
            # crossover
            pair2cross = math.floor(pop_size*CP/2)
            samp = np.random.choice(pop_size,pair2cross*2)
            point_of_cross  = np.random.choice(np.arange(2,k),pair2cross)
            for i in range(pair2cross):
                if point_of_cross[i] < k/2:
                    swap = list(range(point_of_cross[i]))
                else:
                    swap = list(range(point_of_cross[i],k))
                for j in swap:
                    t1 = cands[samp[i],j]
                    t2 = cands[samp[i+pair2cross],j]
                    if t2 in cands[samp[i]]:
                        t3 = list(cands[samp[i]]).index(t2)
                        cands[samp[i],t3] = t1
                    if t1 in cands[samp[i+pair2cross]]:
                        t3 = list(cands[samp[i+pair2cross]]).index(t1)
                        cands[samp[i+pair2cross],t3] = t2
                    cands[samp[i],j] = t2
                    cands[samp[i+pair2cross],j] = t1

            # random mutation with probability MP
            mutations = int(np.round(pop_size*k*MP)) # number of gene mutation
            rows = np.random.choice(pop_size,mutations)
            cols = np.random.choice(k,mutations)
            switch_with = np.random.choice(n_features,mutations)
            for i in range(mutations):
                tmp = cands[rows[i],cols[i]]
                if switch_with[i] in cands[rows[i]]:
                    ind_switch = list(cands[rows[i]]).index(switch_with[i])
                    cands[rows[i],ind_switch] = tmp
                    cands[rows[i],cols[i]] = switch_with[i]
            # calcualate objective function
            scores = self.spearman_distance_multilple(x,cands)
            if(min_scores == np.min(scores)):
                inter += 1
            else :
                inter = 1
            if inter == convIn:
                conv = True
            if np.min(scores) < best_scores:
                best_cand = cands[scores.index(np.min(scores)),:]
                best_cand = [features_list[i] for i in best_cand]
                best_scores = np.min(scores)
                print("Generation = {}, best candidate = {}, best result = {}".format(inter,best_cand,best_scores))

            t += 1
            if t > max_iter:
                print("Did not converge after {} iterations.".format(max_iter))
                break
            res = [best_cand,best_scores]

        return res

def feature_ranking(dataset,target_name,pop_size = 20, CP = 0.4, MP = 0.02, max_iter = 200, convIn = 30):
    features = dataset.drop(columns=[target_name]).columns.tolist()
    num_features = len(features)
    # Calculate correlation
    corr = Correlation(dataset, target_name)
    list_corr = corr.correlation_feature()
    print(list_corr)
    # Calculate information gain
    ig = InformationGain(dataset, target_name)
    list_ig = ig.calculate_ig_dataset()
    # Calculate relief
    relief = ReLief(dataset, target_name)
    list_relief = relief.reLief()

    # convert list of score to rank of feature
    dict_relief = {features[i]: list_relief[i] for i in range(len(list_relief))}
    dict_ig = {features[i]: list_ig[i] for i in range(len(list_ig))}
    dict_corr = {features[i]: list_corr[i] for i in range(len(list_corr))}
    print(dict_corr)
    # sort feature by score
    dict_relief = dict(sorted(dict_relief.items(), key=lambda x: x[1], reverse=True))
    dict_ig = dict(sorted(dict_ig.items(), key=lambda x: x[1], reverse=True))
    dict_corr = dict(sorted(dict_corr.items(), key=lambda x: x[1], reverse=True))
    print(dict_corr)

    # Create list of feature by rank
    list_relief = list(dict_relief.keys())
    list_ig = list(dict_ig.keys())
    list_corr = list(dict_corr.keys())
    print(list_corr)
    # concatenate 3 list features
    rank_list = np.row_stack([list_relief, list_ig, list_corr])

    # run rank aggregation algorithm
    agg = RankAggregation()
    res = agg.rank_aggregation(rank_list, num_features, pop_size, CP, MP, max_iter, convIn)
    print(res)
    return res

if __name__ == '__main__':
    print("------------------------Start ---------------------")
    dataset = pd.read_csv("../../../dataset/diabetes/diabetes_encoder.csv")
    target_name = 'Outcome'
    k = 8
    res = feature_ranking(dataset, target_name)
    print("--------------------- Done test -------------------")
