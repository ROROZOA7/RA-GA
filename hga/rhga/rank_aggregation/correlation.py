import pandas as pd
import numpy as np

class Correlation(object):
    '''
    Class Correlation compute correlation betwwen features and labels
    '''

    def __init__(self, dataset, target_name):
        '''
        Constructor
        :param dataset: dataframe of dataset
        :param target_name: target column name
        '''
        self.dataset =dataset
        self.target_name = target_name

    # --------------------  Correlation -----------

    def correlation_feature(self):
        '''
        Method compute correlation between features and labels
        :return corr: list of correlation scores
        '''
        corr = self.dataset.corr()
        list_feature = corr.columns.tolist()
        list_feature.remove(self.target_name)
        cor_target = abs(corr[self.target_name])
        del cor_target[self.target_name]
        corr = cor_target.values
        norm = np.linalg.norm(corr)
        corr = corr / norm
        return corr

