import copy
import numpy as np
from numpy.random import randint, rand


class HybridGA(object):
    '''
    Main class to build Hybrid genetic algorithm
    '''

    def __init__(self, dataset_config, neural_network, max_sub_features=5, crossover_rate=0.6, mutation_rate=0.02,
                 pop_size=20, n_iter=30, distinct=0.65, alpha=0.9, stop_ga=3):
        '''
        Constructor
        :param dataset_config: dataframe dataset
        :param neural_network: model neural network
        :param max_sub_features: max size of subset features
        :param crossover_rate: crossover rate of genetic algorithm
        :param mutation_rate: mutation rate of genetic algorithm
        :param pop_size: population size of genetic algorithm
        :param n_iter: max generation in genetic algorithm
        :param distinct: percentage features belong to group dissimilar features
        :param alpha: weight of classification accuracy in fitness score
        '''
        self.dataset_config = dataset_config
        self.neural_network = neural_network
        self.max_sub_features = max_sub_features
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.pop_size = pop_size
        self.n_iter = n_iter
        self.distinct = distinct
        self.alpha = alpha
        self.stop_ga = stop_ga

    def initialize_population(self, f, k):
        '''
        Create randomly chromosomes population
        :param f: number of total features
        :param k: subset size
        :param pop_size: number of population
        :return: a list of population
        '''
        pop = [randint(0, 2, f).tolist() for _ in range(self.pop_size)]
        for p in pop:
            for i in range(len(p)):
                if np.random.rand() < k / f:
                    p[i] = 1
                else:
                    p[i] = 0
        return pop



    def calculate_fitness_score(self, df_train, df_val, df_test, sub_features):
        '''
        Method calculate fitness score
        :param df_train: dataframe data for training
        :param df_val: dataframe data for validation
        :param df_test: dataframe data for test
        :param sub_features: a subset features solution
        :return: fitness score
        '''
        sub_feature = list(sub_features)
        acc = self.neural_network.classification(df_train, df_val, df_test, sub_features,
                                                 self.dataset_config.target_name)
        print("Offspring = {}, val accuracy = {}".format(sub_feature, acc))
        n_feature_selection = np.sum(sub_feature)
        total_feature = len(sub_feature)
        return self.alpha * (1 - acc) + (1 - self.alpha) * n_feature_selection / total_feature

    def rank_selection(self, pop, fitness_values):
        '''
        Method to select population by fitness ranking
        :param pop: list population(pop_size, chromosomes_size)
        :param fitness_values: list of fitness score
        :return: return selected index in population
        '''
        # ranking fitness values by score
        seq = sorted(fitness_values)
        rank = [seq.index(v) + 1 for v in fitness_values]
        sum_rank = len(rank) * (len(rank) + 1) / 2
        # calcu new population score by rank
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


    def tournament_selection(self, pop, fitness_values, tournament_size=5):
        '''
        Method to select population by tournament rank
        :param pop: list population(pop_size, chromosomes_size)
        :param fitness_values: list of fitness score
        :return: return selected index in population
        '''
        selection_ind = randint(len(pop))
        for ind in randint(0, len(pop), tournament_size):
            if fitness_values[ind] < fitness_values[selection_ind]:
                selection_ind = ind
        return selection_ind

    def uniform_crossover(self, p1, p2):
        '''
        Method proceed crossover to create new offspring by swap n gene between 2 parent
        :param p1: chromosomes parent 1
        :param p2: chromosomes parent 2
        :return: new offsprings
        '''

        c1, c2 = p1.copy(), p2.copy()
        if np.random.rand() < self.crossover_rate:
            for i in range(len(p1)):
                prob = np.random.rand()
                if prob < 0.5:
                    c1[i] = p2[i]
                    c2[i] = p1[i]
        if sum(c1) == 0:
            c1[randint(1, len(p1) - 1)] = 1
        if sum(c2) == 0:
            c2[randint(1, len(p1) - 1)] = 1
        return [c1, c2]

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
        if sum(c1) == 0:
            c1[randint(1, len(p1) - 1)] = 1
        if sum(c2) == 0:
            c2[randint(1, len(p1) - 1)] = 1
        return [c1, c2]

    def crossover_double(self, p1, p2):
        '''
        Method proceed crossover to create new offspring by swap 2 gene between 2 parent
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
        if sum(c1) == 0:
            c1[randint(1, len(p1) - 1)] = 1
        if sum(c2) == 0:
            c2[randint(1, len(p1) - 1)] = 1
        return [c1, c2]

    def crossover_common(self, p1, p2):
        '''
        function: generate new offspring by common crossover
        :param p1: chromosome parent 1
        :param p2: chromosome parent 2
        :return: new offspring
        '''
        c1, c2 = p1.copy(), p2.copy()
        common = p1.copy()
        if np.random.rand() < self.crossover_rate:
            for i in range(len(c1)):
                if c1[i] != c2[i]:
                    common[i] = 0
        if sum(c1) == 0:
            c1[randint(1, len(p1) - 1)] = 1
        if sum(c2) == 0:
            c2[randint(1, len(p1) - 1)] = 1
        return common


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
        if sum(offspring) == 0:
            offspring[randint(1, len(offspring) - 1)]
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
        # delete poor parent and add new children
        new_pop = [v for i, v in enumerate(new_pop) if i not in parent_del]
        new_scores = [v for i, v in enumerate(new_scores) if i not in parent_del]
        for v in children_add:
            new_pop.append(children_pop[v])
            new_scores.append(children_scores[v])
        return new_pop, new_scores


    def replace_population_old(self, p1, p2, score_p1, score_p2, o1, o2, score_o1, score_o2):
        l_sol = [p1, p2, o1, o2]
        l_score = [score_p1, score_p2, score_o1, score_o2]
        d_score_sol = {i: [l_score[i], l_sol[i]] for i in range(len(l_score))}
        d_score_sol = sorted(d_score_sol.items(), key=lambda e: e[1][0], reverse=False)
        print(d_score_sol)
        score_1 = d_score_sol[0][1][0]
        score_2 = d_score_sol[1][1][0]
        sol_1 = d_score_sol[0][1][1]
        sol_2 = d_score_sol[1][1][1]
        return sol_1, sol_2, score_1, score_2


    def replace_not_common(self, offspring, common, list_prob_rank):
        '''
        function : replace a min rank feature in offspring by max rank feature in list features left
        :param offspring: the current solution
        :param common: the common between 2 parent of offspring
        :param list_prob_rank: list of proability of feature
        :return:  new offspring after remove minimum feature and add new maximum feature
        '''
        new_off = offspring.copy()
        value_min = [v for i, v in enumerate(list_prob_rank) if new_off[i] == 1 and common[i] == 0]
        value_max = [v for i, v in enumerate(list_prob_rank) if new_off[i] == 0 and common[i] == 0]
        if not value_min or not value_max:
            return new_off, common
        min_index = list_prob_rank.index(min(value_min))
        max_index = list_prob_rank.index(max(value_max))
        new_off[min_index] = 0
        new_off[max_index] = 1
        common[min_index] = 1
        common[max_index] = 1
        return new_off, common


    def replace_min_gene_by_max(self, offspring, list_prob_rank):
        '''
        function : replace a min rank feature in offspring by max rank feature in list features left
        :param offspring: the current solution
        :param common: the common between 2 parent of offspring
        :param list_prob_rank: list of proability of feature
        :return:  new offspring after remove minimum feature and add new maximum feature
        '''
        new_off = offspring.copy()
        value_min = [v for i, v in enumerate(list_prob_rank) if new_off[i] == 1]
        value_max = [v for i, v in enumerate(list_prob_rank) if new_off[i] == 0]
        if not value_min or not value_max:
            return new_off
        min_index = list_prob_rank.index(min(value_min))
        max_index = list_prob_rank.index(max(value_max))
        new_off[min_index] = 0
        new_off[max_index] = 1
        return new_off


    def local_search_3(self, offspring, offspring_score, k_max, list_prob_rank, max_try=3):
        '''
        Local search 3: just exchange features with min score by features with max score(not selected)
        :param offspring: offspring
        :param k_max: max size of subset features
        :param list_prob_rank: list of proability of feature
        :return: new offspring after
        '''
        list_prob_rank = list(list_prob_rank)
        current_sol = offspring.copy()
        current_score = offspring_score
        if sum(current_sol) > k_max:
            for i in range(sum(current_sol) - k_max):
                min_index = list_prob_rank.index(min([v for i, v in enumerate(list_prob_rank) if current_sol[i] == 1]))
                current_sol[min_index] = 0
        best_score = 9999
        best_sol = offspring.copy()
        while max_try != 0:
            if current_score < best_score:
                best_score = current_score
                best_sol = current_sol.copy()
            current_sol = self.replace_min_gene_by_max(current_sol, list_prob_rank)
            current_score = self.calculate_fitness_score(self.dataset_config.df_train,
                                                         self.dataset_config.df_val,
                                                         self.dataset_config.df_test, current_sol)
            max_try -= 1
        return best_sol, best_score


    def local_search_1(self, p1, p2, offspring, offspring_score, k_max, list_prob_rank, max_try=3):
        '''
        Method process local search to improve current solution to better solution
        :param p1: chromosome parent 1
        :param p2: chromosome parent 2
        :param offspring: offspring
        :param k_max: max size of subset features
        :param list_prob_rank: list of proability of feature
        :return: new offspring after
        '''
        list_prob_rank = list(list_prob_rank)
        p1_off = offspring.copy()
        p2_off = offspring.copy()
        common = offspring.copy()
        best_score = 9999
        best_sol = offspring.copy()
        for i in range(len(offspring)):
            p1_off[i] = offspring[i] - p1[i]
            p2_off[i] = offspring[i] - p2[i]
            if p1_off[i] == p2_off[i]:
                common[i] = 1
            else:
                common[i] = 0
        current_sol = offspring.copy()
        current_score = offspring_score
        if sum(current_sol) > k_max:
            for i in range(sum(current_sol) - k_max):
                min_index = list_prob_rank.index(min([v for i, v in enumerate(list_prob_rank) if current_sol[i] == 1]))
                current_sol[min_index] = 0

        while max_try != 0:
            if current_score < best_score:
                best_score = current_score
                best_sol = current_sol.copy()
            current_sol, common = self.replace_not_common(current_sol, common, list_prob_rank)
            current_score = self.calculate_fitness_score(self.dataset_config.df_train, self.dataset_config.df_val,
                                                         self.dataset_config.df_test, current_sol)
            max_try -= 1
        return best_sol, best_score

    def local_search_2(self, p1, p2, offspring, k_max, list_prob_rank):
        '''
        Method process local search to improve current solution to better solution
        :param p1: chromosome parent 1
        :param p2: chromosome parent 2
        :param offspring: offspring
        :param k_max: max size of subset features
        :param list_prob_rank: list of proability of feature
        :return: new offspring after
        '''
        list_prob_rank = list(list_prob_rank)
        common = offspring.copy()

        for i in range(len(offspring)):
            if (p1[i] == 0 and p2[i] == 1 and offspring[i] == 1) or (p1[i] == 1 and p2[i] == 0 and offspring[i] == 1):
                common[i] = 0
            else:
                common[i] = 1

        current_sol = offspring.copy()
        # if number of subset  feature > k_max : remove minimum redundant ranking feature
        if sum(current_sol) > k_max:
            for i in range(sum(current_sol) - k_max):
                min_index = list_prob_rank.index(min([v for i, v in enumerate(list_prob_rank) if current_sol[i] == 1]))
                current_sol[min_index] = 0


        current_sol, common = self.replace_not_common(current_sol, common, list_prob_rank)
        current_score = self.calculate_fitness_score(self.dataset_config.df_train, self.dataset_config.df_val,
                                                         self.dataset_config.df_test, current_sol)
        return current_sol, current_score


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
            if (list_values[i] - list_values[i - self.stop_ga]) >= 0 \
                    and (list_values[i-1] - list_values[i - self.stop_ga]) >= 0\
                    and (list_values[i-2] - list_values[i - self.stop_ga]) >= 0:
                return True
        return False

    def update_list_prob_rank(self, pop, scores):
        len_pop = len(pop)
        sum_rank = len_pop * (len_pop + 1) / 2
        prob_rank = [(len_pop - i) / sum_rank for i in range(len(pop))]
        pop_score_dict = {i: [pop[i], scores[i]] for i in range(len(pop))}
        pop_score_dict = sorted(pop_score_dict.items(), key=lambda e: e[1][1], reverse=True)
        res = np.zeros(len(pop[0]))
        for ind, value in enumerate(pop_score_dict):
            res += prob_rank[ind] * np.array(value[1][0])
        res = res / np.sum(res)
        return res

    def genetic_algorithm(self, list_prob_rank):
        '''
        Main method of genetic algortihm
        :param list_prob_rank: list of proability of feature
        :return:
        '''
        list_prob_rank = np.array(list_prob_rank)
        features_name = self.dataset_config.features_name
        f = int(len(features_name))
        pop = self.initialize_population(f, self.max_sub_features)
        best_sol = 0
        best_score = 9999
        scores = [self.calculate_fitness_score(self.dataset_config.df_train, self.dataset_config.df_val,
                                               self.dataset_config.df_test, c) for c in pop]
        gen = -1
        list_best_scores = list()
        list_best_sol = list()
        generation_best_scores = list()
        result = list()
        res_prob_rank = list()
        while self.check_stop_ga(gen,generation_best_scores) == False:
            gen += 1
            print("----------------------- Generation {}---------------------".format(gen))
            print(scores)
            if gen > 0:
                min_score_in_generation = min(scores)
                min_index = scores.index(min_score_in_generation)
                generation_best_scores.append(min_score_in_generation)
                print("-------------Best solution = {},best score = {}".format(pop[min_index], min_score_in_generation))
                result.append([scores, pop])
                res_prob_rank.append(list_prob_rank)
                # check for new best solution
                for i in range(len(pop)):
                    if scores[i] < best_score:
                        best_sol, best_score = pop[i], scores[i]
                        list_best_scores.append(best_score)
                        list_best_sol.append(best_sol)
                        print("New best {} = {}".format(pop[i], scores[i]))
            # select parents
            selected_ind = [self.tournament_selection(pop, scores) for _ in range(self.pop_size)]
            selected_pop = [pop[i] for i in selected_ind]
            selected_scores = [scores[i] for i in selected_ind]
            children_pop = list()
            children_scores = list()
            for i in range(0, self.pop_size, 2):
                p1, p2 = selected_pop[i], selected_pop[i + 1]
                p_score1, p_score2 = selected_scores[i], selected_scores[i+1]
                offspring_cross = self.uniform_crossover(p1, p2)
                for c in offspring_cross:
                    c_sol = self.mutation(c)
                    c_score = self.calculate_fitness_score(self.dataset_config.df_train, self.dataset_config.df_val,
                                                           self.dataset_config.df_test, c)

                    # ----------- Run local_search_3  ---------------
                    if c_score < selected_scores[i] or c_score < selected_scores[i + 1]:
                        c_sol, c_score = self.local_search_3(c_sol, c_score,
                                                             self.max_sub_features, list_prob_rank)
                    children_pop.append(c_sol)
                    children_scores.append(c_score)
            # update population
            pop = children_pop
            scores = children_scores
        return [best_sol, best_score], result, res_prob_rank
