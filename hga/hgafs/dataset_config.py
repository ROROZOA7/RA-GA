import numpy as np
import itertools


class DatasetConfig(object):
    '''
    Class DatasetConfig define config for dataset
    '''

    def __init__(self, dataset, df_train, df_val, df_test, target_name, n_classes):
        '''
        Constructor
        :param dataset: dataframe dataset
        :param df_train: dataset for training
        :param df_val: dataset for validation
        :param df_test: dataset for test
        :param target_name: target name
        :param n_classes: number of classes
        '''
        self.dataset = dataset
        self.target_name = target_name
        self.n_classes = n_classes

        self.features_name = self.dataset.columns.tolist()
        self.features_name.remove(target_name)
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test

        def correlated_score():
            '''
            Method compute correlation score for features in dataset
            :return:  dictionary with key : values = features_name : score_corr
            '''
            df_corr = self.dataset.drop(columns=self.target_name).corr(method='pearson')
            df_corr = df_corr.abs()
            corr_score = (df_corr.select_dtypes(np.number).sum().rename('Corr') - 1) / (df_corr.shape[1] - 1)
            df_corr = df_corr.append(corr_score)
            corr_score = df_corr.iloc[df_corr.shape[0] - 1].to_list()
            features = df_corr.columns.to_list()
            dict_corr = {features[i]: corr_score[i] for i in range(len(features))}
            return dict_corr

        self.dict_corr = correlated_score()

        def split_correlated():
            '''
            Method split correlation score to 2 group : dissimilar and similar
            input dictionary with key : value = feature_name : score_corr
            :return: 2 dictionary X_d, X_s (dissimilar, similar correlated with N/2 values)
            '''
            # sort dict by values increase
            dict_corr = dict(sorted(self.dict_corr.items(), key=lambda x: x[1]))
            X_d = dict(itertools.islice(dict_corr.items(), int(len(dict_corr) / 2)))
            X_s = dict(itertools.islice(dict_corr.items(), int(len(dict_corr) / 2), len(dict_corr), 1))
            return X_d, X_s

        self.X_d, self.X_s = split_correlated()

