import numpy as np

class InformationGain(object):
    '''
    Class InformationGain compute information gain index for each features
    '''

    def __init__(self, dataset, target_name):
        '''
        Constructor
        ::param dataset: dataframe of dataset
        :param target_name: target column name
        '''
        self.dataset = dataset
        self.target_name = target_name


    def entropy(self, array_values):
        '''
        Method compute entropy of input list calculate H(s)=H(list_values)
        :param array_values: np.array of values
        :return: return entropy of the input list
        '''
        counts = np.bincount(array_values)
        freq = counts[np.nonzero(counts)]
        # remove prob 0
        freq_0 = freq[np.array(freq).nonzero()[0]]
        prob_0 = freq_0 / float(freq_0.sum())
        return -np.sum(prob_0 * np.log(prob_0))


    def entropy_x(self,class_, feature):
        '''
        function : calculate H(x,S) = H(class_, feature)
        :param class_: target column values
        :param feature: feature column values
        :return: return H(class_, feature)
        '''
        classes = set(class_)
        feature_values = set(feature)
        Hc_feature = 0
        feature = list(feature)
        for feat in feature_values:
            pf = feature.count(feat) / len(feature)
            indices = [i for i in range(len(feature)) if feature[i] == feat]
            clasess_of_feat = [class_[i] for i in indices]
            for c in classes:
                pcf = clasess_of_feat.count(c) / len(clasess_of_feat)
                if pcf != 0:
                    temp_H = - pf * pcf * np.log(pcf)
                    Hc_feature += temp_H
        return Hc_feature


    def information_gain(self,class_, feature):
        '''
        function : calculate information gain
        :param class_: target column values
        :param feature: feature column values
        :return: return information_gain(class_, feature)
        '''
        Hc = self.entropy(class_)
        Hc_feature = self.entropy_x(class_, feature)
        ig = Hc - Hc_feature
        return ig


    def calculate_ig_dataset(self):
        '''
        function : calculate information gain for every feature in dataset
        :return: return list of all feature's information gain
        '''
        y = self.dataset[self.target_name].values
        dataset = self.dataset.drop(columns=[self.target_name])
        X = dataset.values
        n_samples, n_features = np.shape(X)
        features = dataset.columns.tolist()
        Hc = self.entropy(y)
        result = np.zeros(n_features)
        for i in range(n_features):
            feature = dataset[features[i]].values
            Hc_feature = self.entropy_x(y, feature)
            ig = Hc - Hc_feature
            result[i] = ig
        # normalize np.array relief
        norm = np.linalg.norm(result)
        result = result / norm
        return result

