import torch
from torch_geometric.data import Data
from unifiedfl.models.mlp import MLPClassifier
from unifiedfl.models.cnn import CNNClassifier, CNNClassifierDeep
from .convert import mlp_to_graph, mlp_to_graph_ws, cnn_to_graph, cnn_to_graph_ws

def unify(model_list, input_shape):
    node_offset = 0
    E_idx, E_attr, B, A = [], [], [], []
    node_indices_list, layer_neurons_list, input_indices_list, output_indices_list, names = [], [], [], [], []
    for m in model_list:
        if isinstance(m, MLPClassifier):
            data, nodes, layers = mlp_to_graph(m)
        elif isinstance(m, CNNClassifier):
            data, nodes, layers = cnn_to_graph(m, input_shape)
        else:
            raise ValueError("unsupported model type")
        names.append(data.name)
        E_idx.append(data.edge_index + node_offset)
        E_attr.append(data.edge_attr)
        B.append(data.bias); A.append(data.activation)
        node_indices_list.append([i+node_offset for i in nodes])
        adj_layers = [[i+node_offset for i in L] for L in layers]
        layer_neurons_list.append(adj_layers)
        input_indices_list.append(adj_layers[0])
        output_indices_list.append(adj_layers[-1])
        node_offset += data.num_nodes
    edge_index = torch.cat(E_idx, dim=1)
    edge_attr  = torch.cat(E_attr, dim=0)
    bias       = torch.cat(B, dim=0)
    act        = torch.cat(A, dim=0)
    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=node_offset, names=names)
    data.bias = bias; data.activation = act
    return data, node_indices_list, layer_neurons_list, input_indices_list, output_indices_list

def unify_ws(model_list, input_shape):
    node_offset = 0
    E_idx, E_attr, B, A = [], [], [], []
    node_indices_list, layer_neurons_list, layer_types_lists = [], [], []
    input_indices_list, output_indices_list, names = [], [], []
    total_edge_to_kernel_idx = {}; total_node_to_layer_idx = {}; edge_idx_offset = 0

    for m in model_list:
        if isinstance(m, MLPClassifier):
            data, nodes, layers, e2k, n2l, ltypes = mlp_to_graph_ws(m)
        elif isinstance(m, (CNNClassifier, CNNClassifierDeep)):
            data, nodes, layers, e2k, n2l, ltypes = cnn_to_graph_ws(m, input_shape)
        else:
            raise ValueError("unsupported model type")
        names.append(data.name)
        E_idx.append(data.edge_index + node_offset)
        E_attr.append(data.edge_attr)
        B.append(data.bias); A.append(data.activation)
        node_indices_list.append([i+node_offset for i in nodes])
        adj_layers = [[i+node_offset for i in L] for L in layers]
        layer_neurons_list.append(adj_layers)
        layer_types_lists.append(ltypes)
        input_indices_list.append(adj_layers[0]); output_indices_list.append(adj_layers[-1])

        for e, k in e2k.items():
            total_edge_to_kernel_idx[e + edge_idx_offset] = k
        edge_idx_offset += data.edge_index.size(1)

        total_node_to_layer_idx.update({i+node_offset: ln for i, ln in n2l.items()})
        node_offset += data.num_nodes

    edge_index = torch.cat(E_idx, dim=1)
    edge_attr  = torch.cat(E_attr, dim=0)
    bias       = torch.cat(B, dim=0)
    act        = torch.cat(A, dim=0)
    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=node_offset, names=names)
    data.bias = bias; data.activation = act
    return (data, node_indices_list, layer_neurons_list, layer_types_lists,
            input_indices_list, output_indices_list, total_edge_to_kernel_idx, total_node_to_layer_idx)
