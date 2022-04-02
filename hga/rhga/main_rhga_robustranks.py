import time
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from hga.hgafs.nn_model2 import NeuralNetwork2
from hga.rhga.rank_aggregation.robust_rank_aggregation import feature_ranking
from hga.rhga.dataset_config import DatasetConfig
from hga.rhga.rhga import HybridGA


def convert_rank_2_prob(list_rank):
    '''
        :param list_rank: list of index features was ranked ex: [2,3,1,5,4] start index by 1
        :param features: list of features names
        :return:
        '''
    res = list()
    len_rank = len(list_rank)
    sum_rank = len_rank * (len_rank + 1) / 2
    prob = [(len_rank - i) / sum_rank for i, v in enumerate(list_rank)]
    res = [prob[list_rank.index(i+1)] for i,v in enumerate(list_rank)]
    return res

def preprocessing_dataset(data_path, target_name, fraction_split=[0.5, 0.25, 0.25]):
    '''
        Method handel preprocessing, split dataset into train, val, test set

        :param data_path: a file path of dataset
        :param target_name: target name
        :param fraction_split: list of fraction train, val, test
        :return: dataset, train set, val set, test set
        '''
    dataset = pd.read_csv(data_path)
    y = dataset[target_name].values
    # label encoder
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    # create scaler
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(dataset.drop(columns=[target_name]))
    normalized = pd.DataFrame(normalized)
    df_y = pd.DataFrame(y_enc, columns=[target_name])
    dataset = pd.concat([normalized, df_y], axis=1)
    # split dataset to: train, val, test set
    fractions = np.array(fraction_split)
    dataset = dataset.sample(frac=1)
    df_train, df_val, df_test = np.array_split(
        dataset, (fractions[:-1].cumsum() * len(dataset)).astype(int))
    return dataset, df_train, df_val, df_test


def main(data_path=None, target_name=None, output_path=None,subset_size_scheme=0.7,  crossover_rate=0.6, mutation_rate=0.02, pop_size=40,
         n_iter = 20):
    results = list()
    for i in range(20):
        np.random.seed(i)
        dataset, df_train, df_val, df_test = preprocessing_dataset(data_path, target_name)
        print('------------------------------ Calculate list rank ----------------------------')
        list_rank = feature_ranking(dataset, target_name)
        print("Ranking aggregation: {}".format(list_rank))
        list_rank_prob = convert_rank_2_prob(list_rank)
        print("Probability ranking: {}".format(list_rank_prob))
        print("------------------------------ Start training genetic algorithm -----------------")
        start = time.time()
        config = DatasetConfig(dataset, df_train, df_val, df_test, target_name)
        model = NeuralNetwork2()
        max_sub_features = int(np.round(subset_size_scheme * len(list_rank)))
        hga = HybridGA(dataset_config=config, neural_network=model, max_sub_features=max_sub_features,
                       pop_size=pop_size, crossover_rate=crossover_rate, mutation_rate=mutation_rate, n_iter=n_iter)
        best, result, res_prob_rank = hga.genetic_algorithm(list_rank_prob)
        end = time.time()
        print('Done! Time = {}'.format(end - start))
        print('{} = {}'.format(best, result))

        results.append(best)
        df = pd.DataFrame(result)
        df.to_csv(output_path + '/result_' + str(i) + ".csv", index=False)
        df_rank = pd.DataFrame(res_prob_rank)
        df_rank.to_csv(output_path+ '/res_rank_' + str(i) + ".csv", index=False)
        f = open(output_path+'/result_best.text', 'a')
        f.write(json.dumps(best))
        f.write('\t')
        f.write(str(end - start))
        f.write('\n')
        f.close()
        print("******************** Time = {} ****************".format(end - start))
    print("Results : {}".format(results))

if __name__ == '__main__':
    print('------------- Start -------------------')
    data_path = '../../dataset/diabetes/diabetes.csv'
    target_name = 'Outcome'
    n_classes = 2
    output_path = '../../dataset/diabetes/'
    main(data_path, target_name, output_path)
    print('----------------End--------------------------')