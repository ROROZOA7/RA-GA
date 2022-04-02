import copy

import numpy as np
from numpy.random import randint, rand


class GeneticNeuralNetwork(object):
    '''
    Main class to build genetic algorithm for feature selection
    '''

    def __init__(self, dataset_config, neural_network, crossover_rate=0.6, mutation_rate=0.02, pop_size=20,
                 subset_size_scheme=0.7, distinct=0.65, stop_ga=3):
        '''
        Constructor
        :param dataset_config: dataframe dataset
        :param neural_network: model neural network
        :param crossover_rate: crossover rate of genetic algorithm
        :param mutation_rate: mutation rate of genetic algorithm
        :param pop_size: population size of genetic algorithm
        :param subset_size_scheme: subset features size = subset_size_scheme * number of feature
        :param distinct: percentage features belong to group dissimilar features
        :param stop_ga: number of generation to be stop genetic algorithm when algorithm not improve result
        '''
        self.dataset_config = dataset_config
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.pop_size = pop_size
        self.subset_size_scheme = subset_size_scheme
        self.distinct = distinct
        self.stop_ga = stop_ga
        self.neural_network = neural_network

    def linear_k_subset(self, f, k_max):
        '''
        Lk maximizes linearly with minimizing k and the value of k_min <= k <= k_max
        :param f: number of total features
        :param k_max:  number of subset maximize
        :return: list of Lk
        '''
        linear_k = []
        for k in range(3, k_max + 1):
            lk = (f - k) / sum(range(k, f))
            linear_k.append(lk)
        # normalize linear_k
        tmp = sum(linear_k)
        linear_k = [float(i) / tmp for i in linear_k]
        return linear_k

    def random_selection_k_subset(self, f, k_max):
        '''
        Method choose subset features size
        :param f: number of total features
        :param k_max: maximize subset size
        :return: subset size of features
        '''
        linear_k = self.linear_k_subset(f, k_max)
        rand_k = np.random.rand()
        total = 0
        for i, k in enumerate(range(3, k_max + 1)):
            total = total + linear_k[i]
            if rand_k <= total:
                break
        return k

    def distinct_values(self, sub_features):
        '''
        Method calculate distinct values of one solution
        :param sub_features: a subset features solution
        :return: distinct score
        '''
        if sum(sub_features) > 0:
            scores = [self.dataset_config.dict_corr[self.dataset_config.features_name[i]] for i, v in
                      enumerate(sub_features) if v == 1]
            return 1 / sum(scores)
        return 0

    def calculate_fitness(self, df_train, df_val, df_test, sub_features):
        '''
        Method calculate fitness score
        :param df_train: dataframe data for training
        :param df_val: dataframe data for validation
        :param df_test: dataframe data for test
        :param sub_features: a subset features solution
        :return: fitness score
        '''
        distinct_score = self.distinct_values(sub_features)
        ca_score = self.neural_network.classification(df_train, df_val, df_test, sub_features,
                                                      self.dataset_config.target_name)
        return distinct_score + ca_score


    def initialize_population(self, f, k, pop_size):
        '''
        Create randomly chromosomes population
        :param f: number of total features
        :param k: subset size
        :param pop_size: number of population
        :return: a list of population
        '''
        pop = [randint(0, 2, f).tolist() for _ in range(pop_size)]
        for p in pop:
            for i in range(len(p)):
                if np.random.rand() < k / f:
                    p[i] = 1
                else:
                    p[i] = 0
        return pop

    def rank_selection(self, pop, fitness_values):
        '''
        Method to select population by fitness ranking
        :param pop: list population(pop_size, chromosomes_size)
        :param fitness_values: list of fitness score
        :return: selected index chromosomes
        '''

        seq = sorted(fitness_values)
        rank = [seq.index(v) + 1 for v in fitness_values]
        sum_rank = len(rank) * (len(rank) + 1) / 2
        prob = [i / sum_rank for i in rank]
        sum_prob = sum(prob)

        select_ind = 0
        rand_select = np.random.rand() * sum_prob
        tmp = 0
        for i in range(len(pop)):
            tmp += prob[i]
            if tmp >= rand_select:
                select_ind = i
                break
        return select_ind

    def crossover_single(self, p1, p2):
        '''
        Method proceed crossover to create new offspring by swap 1 gene between 2 parent
        :param p1: chromosomes parent 1
        :param p2: chromosomes parent 2
        :return: new offsprings
        '''
        c1, c2 = p1.copy(), p2.copy()
        if np.random.rand() < self.crossover_rate:
            pt = randint(1, len(p1) - 2)
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
        return [c1, c2]

    def crossover_double(self, p1, p2):
        '''
        Method proceed crossover to create new offspring by swap 2 genes between 2 parent
        :param p1: chromosomes parent 1
        :param p2: chromosomes parent 2
        :return: new offsprings
        '''
        c1, c2 = p1.copy(), p2.copy()
        if np.random.rand() < self.crossover_rate:
            pt1 = randint(1, len(p1) - 2)
            pt2 = randint(1, len(p1) - 2)
            while pt1 == pt2:
                pt2 = randint(1, len(p1) - 2)
            if pt1 > pt2:
                a = pt2
                pt2 = pt1
                pt1 = a
            c1 = p1[:pt1] + p2[pt1:pt2] + p1[pt2:]
            c2 = p2[:pt1] + p1[pt1:pt2] + p2[pt2:]
        return [c1, c2]

    def mutation(self, bitstring):
        '''
        Method proceed mutation by flip 1 gene in chromosome
        :param bitstring: a chromosome
        :return: new offspring
        '''
        offspring = bitstring.copy()
        for i in range(len(offspring)):
            if rand() < self.mutation_rate:
                offspring[i] = 1 - offspring[i]
        return offspring

    def replace_population(self,parent_pop, parent_scores,children_pop, children_scores):
        '''

        :param parent_pop: parent population
        :param parent_scores: parent fitness scores
        :param children_pop: children population
        :param children_scores: children fitness scores
        :return: new population, new socres
        '''
        new_pop = copy.deepcopy(parent_pop)
        new_scores = copy.deepcopy(parent_scores)
        parent_del = list()
        children_add = list()
        for i_parent, v_parent in enumerate(new_pop):
            for i_children, v_children in enumerate(children_pop):
                if children_scores[i_children] > parent_scores[i_parent]:
                    parent_del.append(i_parent)
                    children_add.append(i_children)
        parent_del = list(set(parent_del))
        children_add = list(set(children_add))
        new_pop = [v for i, v in enumerate(new_pop) if i not in parent_del]
        new_scores = [v for i, v in enumerate(new_scores) if i not in parent_del]
        for v in children_add:
            new_pop.append(children_pop[v])
            new_scores.append(children_scores[v])
        return new_pop, new_scores

    def local_search(self, bitstring, k, features_name, group_d, group_s):
        '''
        Method process local search to improve current solution to better solution

        :param bitstring: a chromosome
        :param k: subset features size
        :param features_name: list of features name
        :param group_d: group include dissimilar features
        :param group_s: group include similar features
        :return: new offspring
        '''
        num_d = round(self.distinct * k)
        num_s = k - num_d
        X_d = [1 if val == 1 and features_name[ind] in group_d.keys() else 0 for ind, val in enumerate(bitstring)]
        X_s = [1 if val == 1 and features_name[ind] in group_s.keys() else 0 for ind, val in enumerate(bitstring)]
        left_d = group_d.copy()
        dict_d = dict()
        for ind, val in enumerate(X_d):
            if val == 1:
                left_d.pop(features_name[ind])
                dict_d[features_name[ind]] = group_d[features_name[ind]]

        left_s = group_s.copy()
        dict_s = dict()
        for ind, val in enumerate(X_s):
            if val == 1:
                left_s.pop(features_name[ind])
                dict_s[features_name[ind]] = group_s[features_name[ind]]
        # readjust number of feature in X_d, X_s
        while sum(X_d) != num_d:
            if sum(X_d) > num_d:
                max_key = max(dict_d, key=dict_d.get)
                X_d[features_name.index(max_key)] = 0
                dict_d.pop(max_key)
            else:
                min_key = min(left_d, key=left_d.get)
                X_d[features_name.index(min_key)] = 1
                left_d.pop(min_key)

        while sum(X_s) != num_s:
            if sum(X_s) > num_s:
                max_key = max(dict_s, key=dict_s.get)
                X_s[features_name.index(max_key)] = 0
                dict_s.pop(max_key)
            else:
                min_key = min(left_s, key=left_s.get)
                X_s[features_name.index(min_key)] = 1
                left_s.pop(min_key)

        offspring = X_d.copy()
        offspring = [1 if X_d[i] == 1 or X_s[i] == 1 else 0 for i in range(len(X_d))]
        return offspring

    def check_stop_ga(self, gen,list_values):
        '''
        Method check stop criterion of genetic algorithm
        return True : if list l have 3 last value decrease
        :param list_values: list of values
        :return: boolean
        '''
        if (gen % self.stop_ga) != 0:
            return False
        if len(list_values) > 3:
            i = len(list_values) - 1
            if (list_values[i] - list_values[i - self.stop_ga]) <= 0 \
                    and (list_values[i-1] - list_values[i - self.stop_ga]) <= 0\
                    and (list_values[i-2] - list_values[i - self.stop_ga]) <= 0:
                return True
        return False

    def genetic_algorithm(self):
        '''
        Main method of genetic algortihm
        :return:
        '''

        features_name = self.dataset_config.features_name
        f = int(len(features_name))
        dict_corr = self.dataset_config.dict_corr
        X_d = self.dataset_config.X_d
        X_s = self.dataset_config.X_s
        k_max = int(self.subset_size_scheme * f)
        k = self.random_selection_k_subset(f, k_max)
        print("K subset features = {}".format(k))
        # initial population of random bitstring
        pop = self.initialize_population(f, k, self.pop_size)
        best_sol = 0
        best_score = 0
        scores = [self.calculate_fitness(self.dataset_config.df_train, self.dataset_config.df_val,
                                         self.dataset_config.df_test, c) for c in pop]

        gen = -1
        list_best_scores = list()
        list_best_sol = list()
        generation_best_scores = list()
        list_best_scores.append(best_score)
        list_best_sol.append(best_sol)
        generation_best_scores.append(best_score)
        result = list()
        while self.check_stop_ga(gen, generation_best_scores) == False:
            gen += 1
            print("============ Generation {}==========".format(gen))
            print(pop)
            print(scores)
            if gen > 1:
                max_score_in_generation = max(scores)
                max_index = scores.index(max_score_in_generation)
                generation_best_scores.append(max_score_in_generation)
                print("--------Best solution = {}, best score = {}".format(pop[max_index], max_score_in_generation))
                result.append([scores, pop])
                # check for new best solution
                for i in range(len(pop)):
                    if scores[i] > best_score:
                        best_sol, best_score = pop[i], scores[i]
                        list_best_scores.append(best_score)
                        list_best_sol.append(best_sol)
                        print("New best {} = {}".format(pop[i], scores[i]))
            # select parents
            selected_idx = [self.rank_selection(pop, scores) for _ in range(self.pop_size)]
            selected_pop = [pop[i] for i in selected_idx]
            selected_score = [scores[i] for i in selected_idx]
            # create the next generation
            children_pop = list()
            for i in range(0, self.pop_size, 2):
                p1, p2 = selected_pop[i], selected_pop[i + 1]
                for c in self.crossover_double(p1, p2):
                    c = self.mutation(c)
                    c = self.local_search(c, k, features_name, X_d, X_s)
                    children_pop.append(c)

            # compute children fitness score
            children_scores = [self.calculate_fitness(self.dataset_config.df_train, self.dataset_config.df_val,
                                             self.dataset_config.df_test, c) for c in children_pop]
            # replace population
            if gen > 1:
                pop, scores = self.replace_population(selected_pop, selected_score, children_pop, children_scores)
            else :
                pop = children_pop
                scores = children_scores

        return [best_sol, best_score], result
