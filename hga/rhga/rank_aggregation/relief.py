from __future__ import print_function
from math import sqrt
import numpy as np

class ReLief(object):
    '''
    Class ReLief
    '''

    def __init__(self, dataset, target_name):
        '''
        Constructor
        :param dataset: dataframe of dataset
        :param target_name: target column name
        '''
        self.dataset = dataset
        self.target_name = target_name
        self.y = self.dataset[target_name].values
        self.X = self.dataset.drop(columns=[target_name]).values
        self.n_samples, self.n_features = np.shape(self.X)
        self.features = self.dataset.drop(columns=[target_name]).columns.tolist()
        self.dict_features_max = {v: max(self.X[:, i]) for i, v in enumerate(self.features)}
        self.dict_features_min = {v: min(self.X[:, i]) for i, v in enumerate(self.features)}

    def difference(self, X, i_features, i1, i2, cat=0):
        '''

        :param X: dataframe dataset
        :param i_features:index of feature name
        :param i1: index of instance 1
        :param i2: index of instance 2
        :param cat: is categorical or numeric default = numeric
        :return: difference between i1, i2 distinct by f_name
        '''
        if cat == 0:
            diff = abs(X[i1, i_features] - X[i2, i_features]) / (
                    self.dict_features_max[self.features[i_features]] - self.dict_features_min[
                self.features[i_features]])
        else:
            diff = 0 if X[i1, i_features] == X[i2, i_features] else 1
        return diff

    def hamming_distance(self, x1, x2):
        '''
        Method calculating hamming distance between bit strings
        :param x1: point 1
        :param x2: point 2
        :return: distance
        '''
        return sum(abs(x1 - x2)) / len(x1)

    def euclidean_distance(self, x1, x2):
        '''
        Method calculating euclidean distance between bit strings
        :param x1: point 1
        :param x2: point 2
        :return: distance
        '''
        return sqrt(sum((x1 - x2) ** 2))

    def mahattan_distance(self, x1, x2):
        '''
        Method calculating mahattan distance between bit strings
        :param x1: point 1
        :param x2: point 2
        :return: distance
        '''
        return sum(abs(x1 - x2))

    def reLief(self):
        '''
        Method compute relief
        :return: relief= List of reLief for each features
        '''
        near = np.zeros((self.n_samples, 2))
        for i in range(self.n_samples):
            hitDistance = 99999  # init as INF
            missDistance = 99999
            for j in range(self.n_samples):
                if j == i:
                    continue
                curDistance = self.mahattan_distance(self.X[i], self.X[j])
                if self.y[i] == self.y[j] and curDistance < hitDistance:
                    hitDistance = curDistance
                    near[i, 0] = j
                if self.y[i] != self.y[j] and curDistance < missDistance:
                    missDistance = curDistance
                    near[i, 1] = j

        relief = np.zeros(self.n_features)
        for j in range(self.n_features):
            for i in range(self.n_samples):
                relief[j] += self.difference(self.X, j, i, int(near[i, 1])) - \
                             self.difference(self.X, j, i,int(near[i, 0]))
        norm = np.linalg.norm(relief)
        relief = relief / norm
        return relief


