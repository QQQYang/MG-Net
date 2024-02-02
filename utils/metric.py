#encoding=utf-8

import numpy as np
from scipy.stats import spearmanr
from scipy.stats import kendalltau

def spearman_rank_corr(x1, x2):
    n = len(x1)
    x1 = np.array(x1)
    x2 = np.array(x2)
    # return 1 - 6*np.sum((x1 - x2)**2)/(n*(n**2-1))
    return spearmanr(x1, x2)[0]

def kendall_tau_rank_corr(x1, x2):
    s = 0
    n = len(x1)
    x1 = np.array(x1)
    x2 = np.array(x2)
    # for i in range(1, len(x1)):
    #     for j in range(i):
    #         s += np.sign(x1[j] - x1[i])*np.sign(x2[j] - x2[i])
    # return 2*s/(n*(n-1))
    return kendalltau(x1, x2)[0]

def R2(x1, x2):
    """https://zh.wikipedia.org/wiki/%E5%86%B3%E5%AE%9A%E7%B3%BB%E6%95%B0

    Args:
        x1 ([type]): [description]
        x2 ([type]): [description]

    Returns:
        [type]: [description]
    """
    x1 = np.array(x1)
    x2 = np.array(x2)
    tot = np.sum((x1 - np.mean(x1))**2)
    res = np.sum((x1 - x2)**2)
    return 1 - res/tot

def MSE(x1, x2):
    return np.mean((x1-x2)**2)

def worst_case(preds, labels):
    """[summary]

    Args:
        preds ([type]): [description]
        labels ([type]): [description]
    """

def best_case(preds, labels):
    """[summary]

    Args:
        preds ([type]): [description]
        labels ([type]): [description]
    """