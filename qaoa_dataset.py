#encoding=utf-8
import torch

from dgl.data import DGLDataset

import numpy as np
import json
from tqdm import tqdm
import networkx as nx
import pennylane as qml

import itertools
from arguments import args

# full arrangement
n_qubit = args.n_node
valid_Rs = [qml.RX, qml.RY]
valid_CZs = [[i, i+1] for i in range(n_qubit-1)]

Rs_space = list(itertools.product(*[valid_Rs for _ in range(n_qubit)]))#, valid_Rs, valid_Rs, valid_Rs, valid_Rs)) # 3**3=27
CZs_space = [[y for y in CNOTs if y is not None] for CNOTs in list(itertools.product(*([x, None] for x in valid_CZs)))]
CZs_space = [[]]
NAS_search_space = list(itertools.product(Rs_space, CZs_space))
NAS_search_space = Rs_space

gate_onehot_extent = {
    qml.RX: [0., 0., 0., 1.0, 0],
    qml.RY: [0., 0., 0., 0, 1.0],
}

def one_hot(v, length):
    code = np.zeros((length,), dtype=np.float32)
    code[v] = 1
    return code


def UBFeat(arch):
    n_qubit = len(arch)
    feat = []
    for i, qubit in enumerate(arch):
        feat.append(gate_onehot_extent[qubit]+one_hot(i, n_qubit).tolist())
    return np.array(feat, dtype=np.float32)

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = np.diag(r_inv_sqrt)

    return np.dot(np.dot(r_mat_inv_sqrt, mx), r_mat_inv_sqrt)

    # return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def group2edge(param_group):
    group_index = [int(i) for i in param_group.split('-')]
    edge_dict = {}
    for i in range(len(group_index)):
        if group_index[i] not in edge_dict:
            edge_dict[group_index[i]] = [i]
        else:
            edge_dict[group_index[i]].append(i)
    edge = []
    for key in edge_dict:
        if len(edge_dict[key]) > 1:
            for i in range(len(edge_dict[key])-1):
                for j in range(i+1, len(edge_dict[key])):
                    edge.append([edge_dict[key][i], edge_dict[key][j]])
    return edge

def graph2hc(graph_edges, n_node):
    jump2edge = {}
    circuit_edges = []
    for edge in graph_edges:
        if abs(edge[1] - edge[0]) not in jump2edge:
            jump2edge[abs(edge[1] - edge[0])] = [edge]
        else:
            jump2edge[abs(edge[1] - edge[0])].append(edge)
    qubit_start_idx, node_idx2qubit_idx = {}, {}
    for i in range(n_node):
        qubit_start_idx[i] = i
        node_idx2qubit_idx[i] = [i]
    node_idx = n_node
    keys = sorted(list(jump2edge.keys()))
    for jump in keys:
        for edge in jump2edge[jump]:
            circuit_edges.append([qubit_start_idx[edge[0]], node_idx])
            circuit_edges.append([qubit_start_idx[edge[1]], node_idx])
            node_idx2qubit_idx[node_idx] = [edge[0], edge[1]]
            qubit_start_idx[edge[0]] = node_idx
            qubit_start_idx[edge[1]] = node_idx
            node_idx += 1
    for i in range(n_node):
        circuit_edges.append([qubit_start_idx[i], node_idx])
        node_idx2qubit_idx[node_idx] = [i]
        node_idx += 1
    return circuit_edges, node_idx2qubit_idx

def path2graph(adj, n_city):
    circuit_edges = []
    weights = []
    for i in range(n_city):
        for j in range(n_city):
            for p in range(n_city):
                if adj[i][j] > 0:
                    circuit_edges.append([i*n_city+p, j*n_city+(p+1)//n_city])
                    weights.append(adj[i][j])
    return circuit_edges, weights

class CostEstimatorDataset(DGLDataset):
    """convert problem graph into HC, and then convert it into graph

    Args:
        DGLDataset (_type_): _description_
    """
    def __init__(self, args, phase='train'):
        self.args = args
        self.phase = phase
        super().__init__(name='DGL loss predictor dataset')

    def process(self):
        from torch_geometric.utils.convert import from_networkx
        if self.args.graph_divide == 1:
            self.args.n_node = self.args.n_node ** 2
        if self.phase == 'train':
            with open(self.args.train_data, 'r') as f:
                data = json.load(f)

            with open(self.args.train_label, 'r') as f:
                data_gt = json.load(f)
        else:
            with open(self.args.test_graph, 'r') as f:
                data = json.load(f)

            with open(self.args.test_label, 'r') as f:
                data_gt = json.load(f)

        self.ps = []
        self.prob_graphs = []
        self.ansatz_graphs = []

        self.label = []
        self.keys = []
        node_types = ['start', 'end', 'zz', 'x', 'y']
        full_graph = []
        for i in range(self.args.n_node):
            for j in range(self.args.n_node):
                full_graph.append([i, j])
        for key in tqdm(data, desc='key'):
            sample = data[key]

            ########### build problem graph ##################
            G = nx.DiGraph()
            if 'tsp' in self.args.train_data:
                edges, weights = path2graph(sample['weights'], int(np.sqrt(self.args.n_node)))
                circuit_edges, node_idx2qubit_idx = graph2hc(edges, self.args.n_node)
                edge2weight = {}
                for i, edge in enumerate(edges):
                    edge2weight['-'.join([str(min(edge)), str(max(edge))])] = weights[i]
            else:
                circuit_edges, node_idx2qubit_idx = graph2hc(sample['graph'], self.args.n_node)
                edge2weight = {}
                for i, edge in enumerate(sample['graph']):
                    edge2weight['-'.join([str(min(edge)), str(max(edge))])] = sample['weights'][i]
            G.add_edges_from(circuit_edges)
            for node in G.nodes:
                node_feat = []

                if node < self.args.n_node:
                    node_type = 'start'
                elif node >= len(G.nodes) - self.args.n_node:
                    node_type = 'end'
                else:
                    node_type = 'zz'
                node_feat += one_hot(node_types.index(node_type), len(node_types)).tolist()

                qubit_idx = 0
                for qubit in node_idx2qubit_idx[node]:
                    qubit_idx = qubit_idx + one_hot(qubit, self.args.n_node)
                node_feat += qubit_idx.tolist()

                node_feat.append(node)

                if node_type == 'zz':
                    node_feat.append(edge2weight['-'.join([str(min(node_idx2qubit_idx[node])), str(max(node_idx2qubit_idx[node]))])])
                else:
                    node_feat.append(0.0)

                if 'tfim' in self.args.train_data:
                    node_feat.append(sample['h'])

                G.nodes[node]['x'] = torch.from_numpy(np.array(node_feat, dtype=np.float32))

            d_feat = 0
            for node in G.nodes:
                d_feat = len(G.nodes[node]['x'])
                break
            x = torch.zeros((len(G.nodes()), d_feat))
            for idx, node in enumerate(G.nodes):
                x[idx] = G.nodes[node]['x']
            G = from_networkx(G)
            G.x = x
            self.prob_graphs.append(G)

            ################### build graph for mixer Hamiltonian #################
            arch_gate = []

            arch_idx = sample['arch'][0]

            arch_gate.append(NAS_search_space[arch_idx])

            param_group = []
            for isub in range(int(np.sqrt(self.args.n_node))):
                param_group_sub = '-'.join([str(int(ig)+isub*int(np.sqrt(self.args.n_node))) for ig in sample['param_group'].split('-')])
                param_group.append(param_group_sub)
            param_group = '-'.join(param_group)

            group_edge = group2edge(param_group)
            group_edge_str = []
            for edge in group_edge:
                group_edge_str.append('-'.join([str(edge[0]), str(edge[1])]))
                group_edge_str.append('-'.join([str(edge[1]), str(edge[0])]))

            ansatz_graph = nx.DiGraph()
            if self.args.param_group_encode == 'edge_attr':
                ansatz_graph.add_edges_from(full_graph)

                edge_attr = np.zeros((len(ansatz_graph.edges()), 1), np.float32)
                for idx, edge in enumerate(ansatz_graph.edges):
                    if '-'.join([str(edge[0]), str(edge[1])]) in group_edge_str:
                        edge_attr[idx, 0] = 1.0
            else:
                ansatz_graph.add_edges_from(group_edge)
            ansatz_graph.add_nodes_from([i for i in range(self.args.n_node)])

            node_feats = UBFeat(arch_gate[0])
            for node in ansatz_graph.nodes:
                ansatz_graph.nodes[node]['x'] = torch.from_numpy(np.array(node_feats[node], dtype=np.float32))

            d_feat = 0
            for node in ansatz_graph.nodes:
                d_feat = len(ansatz_graph.nodes[node]['x'])
                break
            x = torch.zeros((len(ansatz_graph.nodes()), d_feat))
            for idx, node in enumerate(ansatz_graph.nodes):
                x[idx] = ansatz_graph.nodes[node]['x']
            
            ansatz_graph = from_networkx(ansatz_graph)
            ansatz_graph.x = x
            if self.args.param_group_encode == 'edge_attr':
                ansatz_graph.edge_attr = torch.from_numpy(edge_attr)

            self.ansatz_graphs.append(ansatz_graph)

            p = sample['p']-1

            self.ps.append(p)

            if 'loss' in sample:
                self.label.append(sample['loss'])
            else:
                if 'tsp' in self.args.train_data:
                    self.label.append(data_gt[key]['label'])
                else:
                    self.label.append(data_gt['_'.join(key.split('_')[:2])])
            self.keys.append(key)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.prob_graphs[index], self.ansatz_graphs[index], torch.tensor(self.ps[index]), self.label[index], self.keys[index]
    
      
class MixerGeneratorDataset(DGLDataset):
    def __init__(self, args, phase='train'):
        self.args = args
        self.phase = phase
        super().__init__(name='DGL mixer predictor dataset')

    def process(self):
        # load collected data
        from torch_geometric.utils.convert import from_networkx
        if self.args.graph_divide == 1:
            self.args.n_node = self.args.n_node ** 2
        if self.phase == 'train':
            with open(self.args.train_data, 'r') as f:
                data = json.load(f)

            with open(self.args.train_label, 'r') as f:
                data_gt = json.load(f)
        else:
            with open(self.args.test_graph, 'r') as f:
                data = json.load(f)

            with open(self.args.test_label, 'r') as f:
                data_gt = json.load(f)

        self.ps = []
        self.prob_graphs = []
        self.ansatz_graphs = []

        self.label = []
        self.keys = []
        node_types = ['start', 'end', 'zz', 'x', 'y']

        full_graph = []
        for i in range(self.args.n_node):
            for j in range(self.args.n_node):
                full_graph.append([i, j])
        for key in tqdm(data, desc='key'):
            sample = data[key]

            ########### build problem graph ##################
            G = nx.DiGraph()
            circuit_edges, node_idx2qubit_idx = graph2hc(sample['graph'], self.args.n_node)
            edge2weight = {}
            for i, edge in enumerate(sample['graph']):
                edge2weight['-'.join([str(min(edge)), str(max(edge))])] = sample['weights'][i]
            G.add_edges_from(circuit_edges)
            for node in G.nodes:
                node_feat = []

                if node < self.args.n_node:
                    node_type = 'start'
                elif node >= len(G.nodes) - self.args.n_node:
                    node_type = 'end'
                else:
                    node_type = 'zz'
                node_feat += one_hot(node_types.index(node_type), len(node_types)).tolist()

                qubit_idx = 0
                for qubit in node_idx2qubit_idx[node]:
                    qubit_idx = qubit_idx + one_hot(qubit, self.args.n_node)
                node_feat += qubit_idx.tolist()

                node_feat.append(node)

                if node_type == 'zz':
                    node_feat.append(edge2weight['-'.join([str(min(node_idx2qubit_idx[node])), str(max(node_idx2qubit_idx[node]))])])
                else:
                    node_feat.append(0.0)

                if 'tfim' in self.args.train_data:
                    node_feat.append(sample['h'])

                G.nodes[node]['x'] = torch.from_numpy(np.array(node_feat, dtype=np.float32))

            d_feat = 0
            for node in G.nodes:
                d_feat = len(G.nodes[node]['x'])
                break
            x = torch.zeros((len(G.nodes()), d_feat))
            for idx, node in enumerate(G.nodes):
                x[idx] = G.nodes[node]['x']
            G = from_networkx(G)
            G.x = x
            self.prob_graphs.append(G)

            ################### build graph for mixer Hamiltonian #################
            arch_gate = []

            arch_idx = sample['arch'][0]

            mixer = []
            for _ in range(int(np.sqrt(self.args.n_node))):
                mixer += NAS_search_space[arch_idx]
            arch_gate.append(mixer)
            
            ansatz_graph = nx.DiGraph()
            ansatz_graph.add_edges_from(full_graph)
            edge_attr = np.zeros((len(ansatz_graph.edges()), 1), np.float32)
            ansatz_graph.add_nodes_from([i for i in range(self.args.n_node)])

            node_feats = UBFeat(arch_gate[0])
            for node in ansatz_graph.nodes:
                ansatz_graph.nodes[node]['x'] = torch.from_numpy(np.array(node_feats[node], dtype=np.float32))

            d_feat = 0
            for node in ansatz_graph.nodes:
                d_feat = len(ansatz_graph.nodes[node]['x'])
                break
            x = torch.zeros((len(ansatz_graph.nodes()), d_feat))
            for idx, node in enumerate(ansatz_graph.nodes):
                x[idx] = ansatz_graph.nodes[node]['x']
            ansatz_graph = from_networkx(ansatz_graph)
            ansatz_graph.x = x

            ansatz_graph.edge_attr = torch.from_numpy(edge_attr)

            self.ansatz_graphs.append(ansatz_graph)
            p = sample['p']-1
            self.ps.append(p)
            self.label.append(sample['loss'])
            self.keys.append(key)
        print(len(self.label))


    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.prob_graphs[index], self.ansatz_graphs[index], torch.tensor(self.ps[index]), self.label[index], self.keys[index]
