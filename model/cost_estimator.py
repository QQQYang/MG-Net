#encoding=utf-8
import torch
import torch.nn as nn

from torch_geometric.nn import GraphConv, global_mean_pool

import math
    
class CostEstimator(nn.Module):
    def __init__(self, args):
        super(CostEstimator, self).__init__()


    def forward(self, ansatz_graph, prob_graph, n_layer):
        cost = 0
        return cost

