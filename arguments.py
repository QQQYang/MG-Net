#encoding=utf-8

import argparse
parser = argparse.ArgumentParser("MG-QAOA")

parser.add_argument('--config', type=str, default='exp/gnn/tfim_n16_l2-10.yaml')

# dataset
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--n_graph', type=int, default=1, help='number of graph')
parser.add_argument('--n_weight', type=int, default=100, help='number of edge weights')
parser.add_argument('--n_arch', type=int, default=20, help='number of H_B')
parser.add_argument('--n_param_group', type=int, default=2, help='number of parameter group')
parser.add_argument('--n_p', type=int, default=13, help='number of different p')
parser.add_argument('--n_two_qubit_gate', type=int, default=0, help='number of two qubit gates')
parser.add_argument('--n_node', type=int, default=16, help='number of nodes or qubits')
parser.add_argument('--degree', type=int, default=3, help='the degree of graph')
parser.add_argument('--graph_id_s', type=int, default=0, help='start index of graph')
parser.add_argument('--graph_id_e', type=int, default=100, help='start index of graph')
parser.add_argument('--n_layer', type=int, default=2, help='number of circuit layers')
parser.add_argument('--n_layer_max', type=int, default=30, help='maximum number of circuit layers')
parser.add_argument('--path_data', type=str, default='data/graph_n16_d3_s2.json', help='directory for saving samples') # train_n6_d3_l2-26_a0_g5_t0.json
parser.add_argument('--graph_data', type=str, default='data/graph_n64_d3_s1.json', help='directory for saving graphs')
parser.add_argument('--gt_data', type=str, default='data/graph_n6_d3_s2_gt.json', help='directory for saving samples')
parser.add_argument('--lr_qaoa', type=float, default=0.1, help='learning rate') # 1.0, 0.15
parser.add_argument('--model', choices=['cls', 'reg'], default='reg', help='classification model or regression model')
parser.add_argument('--n_class', type=int, default=1, help='classification model or regression model')
parser.add_argument('--share_hb_across_layer', type=bool, default=True, help='Whether the H_B of different layer is kept the same')
parser.add_argument('--n_cpu', type=int, default=1, help='the number of cpu')
parser.add_argument('--graph_divide', type=int, default=0)

# training
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate') # 0.01 for index predictor
parser.add_argument('--decay_step', type=int, default=200, help='step size for decaying the learning rate')
parser.add_argument('--n_epoch', type=int, default=1000, help='number of training epoch')
parser.add_argument('--save_freq', type=int, default=200, help='frequency of saving model')
parser.add_argument('--bs', type=int, default=128, help='batch size')
parser.add_argument('--n_worker', type=int, default=2, help='number of workers')
parser.add_argument('--cuda', type=bool, default=True, help='whether to use GPU')
parser.add_argument('--coef_rank_loss', type=float, default=1, help='')
parser.add_argument('--coef_mse_loss', type=float, default=1, help='')

# model
parser.add_argument('--param_group_encode', type=str, default='edge', help='must be one of (edge, node)')
parser.add_argument('--p_embed', type=str, default='position', help='must be one of (onehot, position)')
parser.add_argument('--p_feat', type=str, default='sum', help='must be one of (sum, concat)')
parser.add_argument('--arch_gnn_style', type=str, default='mul_gnn', help='must be one of (one_gnn, mul_gnn)')

# transformer
parser.add_argument('--t_nfeat', type=int, default=18, help='feature dimension')
parser.add_argument('--t_ninp', type=int, default=128, help='')
parser.add_argument('--t_nhead', type=int, default=8, help='')
parser.add_argument('--t_nhid', type=int, default=1024, help='')
parser.add_argument('--t_nout', type=int, default=128, help='')
parser.add_argument('--t_nlayers', type=int, default=4, help='')
parser.add_argument('--t_dropout', type=float, default=0.5, help='')

# graph
parser.add_argument('--g_nfeat', type=int, default=28, help='')
parser.add_argument('--g_nhid', type=int, default=64, help='')
parser.add_argument('--g_nout', type=int, default=128, help='')

# loss predictor
parser.add_argument('--l_nhid', type=int, default=64, help='')
parser.add_argument('--l_nout', type=int, default=128, help='')
parser.add_argument('--lr_lp', type=float, default=1e-4, help='learning rate')
parser.add_argument('--bs_lp', type=int, default=512, help='batch size')

# QAS
parser.add_argument('--noise', type=bool, default=False, help='')
parser.add_argument('--n_search', type=int, default=100, help='')
parser.add_argument('--n_expert', type=int, default=5, help='')
parser.add_argument('--qas_epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--warmup_epochs', type=int, default=20, help='num of warmup epochs')
parser.add_argument('--searcher', type=str, default='random', help='num of warmup epochs')

# QAOA
parser.add_argument('--n_group', type=int, default=1, help='')
parser.add_argument('--arch_search', type=int, default=1, help='')
parser.add_argument('--n_qaoa_epoch', type=int, default=40, help='')

parser.add_argument('--train', type=bool, default=False, help='train or test')
parser.add_argument('--retrain', type=bool, default=False, help='train or test')
parser.add_argument('--phase', type=str, default='arch', help='loss or arch')

parser.add_argument('--log_dir', type=str, default='logs', help='directory for saving logs')
args = parser.parse_args()
