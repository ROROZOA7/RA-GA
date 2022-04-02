import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skfeature.function.similarity_based import reliefF
from skfeature.function.statistical_based import chi_square
from skfeature.function.similarity_based import fisher_score
from skfeature.function.statistical_based import low_variance
from skfeature.function.information_theoretical_based import MRMR,MIFS
from skfeature.function.statistical_based import gini_index
from skfeature.function.statistical_based import low_variance

# Using R inside python
import rpy2
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.packages import importr
utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)

# Load package R
RobustRankAggreg = importr('RobustRankAggreg')
aggregateRanks = RobustRankAggreg.aggregateRanks
rankMatrix = RobustRankAggreg.rankMatrix

def feature_ranking(dataset, target_name):
    y = dataset[target_name].values
    X = dataset.drop(columns=[target_name]).values

    n_samples, n_features = X.shape
    print(n_samples, n_features)

    # Relief
    score_relief = reliefF.reliefF(X, y)
    print(score_relief)
    idx_relief = reliefF.feature_ranking(score_relief)
    print(idx_relief)

    # Chi-square
    score_chi_square = chi_square.chi_square(X, y)
    print(score_chi_square)
    # rank features in descending order according to score
    idx_chi_square = chi_square.feature_ranking(score_chi_square)
    print(idx_chi_square)

    # Fisher score
    # obtain the score of each feature on the training set
    score_fisher = fisher_score.fisher_score(X, y)
    print(score_fisher)
    # rank features in descending order according to score
    idx_fisher_score = fisher_score.feature_ranking(score_fisher)
    print(idx_fisher_score)

    glist = []
    for i in range(len(idx_relief)):
        glist.append(idx_relief[i])
        glist.append(idx_fisher_score[i])
        glist.append(idx_chi_square[i])

    r_l = ro.IntVector([i for i in glist])
    glist = ro.r['matrix'](r_l, nrow=3)
    r = rankMatrix(glist)
    score_test_fs = aggregateRanks(glist=glist, full=True, rmat=r, method='RRA')
    df_rank = ro.conversion.rpy2py(score_test_fs)
    rank = list(df_rank[0])
    return rank

if __name__ == '__main__':
    file_dataset = "../../../dataset/breast_cancer/breast_cancer_encoder.csv"
    dataset = pd.read_csv(file_dataset)
    target_name = 'Class'
    rank = feature_ranking(dataset, target_name)
    print(rank)
    print("Done")




