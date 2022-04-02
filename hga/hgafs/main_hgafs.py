import pandas as pd
import numpy as np
import time
import json
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from hga.hgafs.dataset_config import DatasetConfig
from hga.hgafs.nn_model2 import NeuralNetwork2
from hga.hgafs.hgafs import GeneticNeuralNetwork



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


def main(data_path=None, target_name=None, n_classes=0, output_path=None, crossover_rate=0.6, mutation_rate=0.02, pop_size=20,
         subset_size_scheme=0.7, distinct=0.65, stop_ga=3):
    results = list()
    for i in range(20):
        np.random.seed(i)
        print("------------------------------ Start training genetic algorithm -----------------")
        start = time.time()
        dataset, df_train, df_val, df_test = preprocessing_dataset(data_path, target_name)

        config = DatasetConfig(dataset, df_train, df_val, df_test, target_name, n_classes)
        model = NeuralNetwork2()
        ga = GeneticNeuralNetwork(config, model, crossover_rate=crossover_rate, mutation_rate=mutation_rate,
                                  pop_size=pop_size, subset_size_scheme=subset_size_scheme, distinct=distinct,
                                  stop_ga=stop_ga)
        best, result = ga.genetic_algorithm()
        end = time.time()
        print(best)
        results.append(best)
        df = pd.DataFrame(result)
        df.to_csv(output_path + 'result/result_' + str(i) + ".csv", index=False)
        f = open(output_path+ 'result/result_best.text', 'a')
        f.write(json.dumps(best))
        f.write('\t')
        f.write(str(end - start))
        f.write('\n')
        f.close()
        print("******************** Time = {} ****************".format(end - start))
    print(results)


if __name__ == '__main__':
    data_path = '../../dataset/diabetes/diabetes.csv'
    target_name = 'Outcome'
    n_classes = 2
    output_path = '../../dataset/diabetes/'
    main(data_path, target_name, n_classes, output_path)
