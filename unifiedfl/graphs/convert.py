import torch
from torch_geometric.data import Data
import torch.nn as nn

def _activation_map(): return {'relu': 1, None: 0}

# --- MLP to graph (simple) ---
def mlp_to_graph(mlp):
    neuron_counter = 0
    edge_index = [[], []]; edge_weight = []
    biases = {}; activations = {}
    # infer input
    for layer in mlp.net:
        if isinstance(layer, nn.Linear): input_size = layer.in_features; break
    input_neurons = list(range(neuron_counter, neuron_counter+input_size)); neuron_counter += input_size
    for idx in input_neurons: biases[idx]=0.0; activations[idx]=None
    prev = input_neurons; layer_neurons = [input_neurons]
    for layer in mlp.net:
        if isinstance(layer, nn.Linear):
            nf = layer.out_features
            curr = list(range(neuron_counter, neuron_counter+nf)); neuron_counter += nf
            for idx, b in zip(curr, layer.bias.data): biases[idx]=float(b.item()); activations[idx]=None
            W = layer.weight.data
            for t, tgt in enumerate(curr):
                for s, src in enumerate(prev):
                    edge_index[0].append(src); edge_index[1].append(tgt)
                    edge_weight.append(float(W[t, s].item()))
            prev = curr; layer_neurons.append(curr)
        elif isinstance(layer, nn.ReLU):
            for idx in prev: activations[idx]='relu'
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr  = torch.tensor(edge_weight, dtype=torch.float)
    num_nodes = neuron_counter
    node_bias = torch.zeros(num_nodes); 
    for i,b in biases.items(): node_bias[i]=b
    node_act = torch.zeros(num_nodes, dtype=torch.long)
    amap = _activation_map()
    for i,a in activations.items(): node_act[i] = amap[a]
    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes, name=str(mlp))
    data.bias = node_bias; data.activation = node_act
    nodes = list(range(num_nodes))
    return data, nodes, layer_neurons

# --- MLP to graph (WS) ---
def mlp_to_graph_ws(mlp):
    data, nodes, layers = mlp_to_graph(mlp)
    edge_to_kernel_idx = {}
    node_to_layer_idx = {}
    layer_types = ['input']
    edge_idx = 0; layer_num = 0
    # build maps walking again
    neuron_counter = 0
    # input
    for idx in layers[0]: node_to_layer_idx[idx] = 0
    prev = layers[0]
    for layer in mlp.net:
        if isinstance(layer, nn.Linear):
            layer_num += 1
            curr = layers[layer_num]
            for i in curr: node_to_layer_idx[i] = layer_num
            in_size = len(prev)
            for t, tgt in enumerate(curr):
                for s, src in enumerate(prev):
                    edge_to_kernel_idx[edge_idx] = {'layer_num': layer_num, 'target_idx': t, 'source_idx': s}
                    edge_idx += 1
            prev = curr; layer_types.append('linear')
        elif isinstance(layer, nn.ReLU):
            layer_types.append('relu')
    return data, nodes, layers, edge_to_kernel_idx, node_to_layer_idx, layer_types

# --- CNN to graph (simple) ---
def cnn_to_graph(cnn, input_shape):
    return _cnn_to_graph_impl(cnn, input_shape, want_ws=False)

# --- CNN to graph (WS) ---
def cnn_to_graph_ws(cnn, input_shape):
    return _cnn_to_graph_impl(cnn, input_shape, want_ws=True)

def _cnn_to_graph_impl(cnn, input_shape, want_ws: bool):
    import numpy as np
    neuron_counter = 0
    edge_index = [[], []]; edge_weight = []
    biases = {}; activations = {}
    layer_neurons = []; position_to_node = {}
    edge_to_kernel_idx = {}; node_to_layer_idx = {}; layer_types = []

    C_in, H_in, W_in = input_shape
    # input nodes
    curr = []
    for c in range(C_in):
        for i in range(H_in):
            for j in range(W_in):
                nid = neuron_counter; neuron_counter+=1
                curr.append(nid); position_to_node[('input', 0, c, i, j)] = nid
                biases[nid]=0.0; activations[nid]=None
                node_to_layer_idx[nid] = 0
    layer_neurons.append(curr); layer_types.append('input')

    prev = curr; prev_type='input'; prev_ln=0; prev_C, prev_H, prev_W = C_in, H_in, W_in
    edge_idx_counter = 0
    layer_num = 0
    for layer in cnn.net:
        if isinstance(layer, nn.Conv2d):
            layer_num += 1
            curr = []
            kH,kW = layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size,)*2
            sH,sW = layer.stride if isinstance(layer.stride, tuple) else (layer.stride,)*2
            pH,pW = layer.padding if isinstance(layer.padding, tuple) else (layer.padding,)*2
            dH,dW = layer.dilation if isinstance(layer.dilation, tuple) else (layer.dilation,)*2
            H_out = (prev_H + 2*pH - dH*(kH-1) - 1)//sH + 1
            W_out = (prev_W + 2*pW - dW*(kW-1) - 1)//sW + 1
            for co in range(layer.out_channels):
                for i in range(H_out):
                    for j in range(W_out):
                        nid = neuron_counter; neuron_counter+=1
                        curr.append(nid)
                        position_to_node[('conv', layer_num, co, i, j)] = nid
                        biases[nid] = float(layer.bias.data[co].item())
                        activations[nid] = None
                        node_to_layer_idx[nid] = layer_num
            W = layer.weight.data
            for co in range(layer.out_channels):
                for i in range(H_out):
                    for j in range(W_out):
                        out_id = position_to_node[('conv', layer_num, co, i, j)]
                        for ci in range(layer.in_channels):
                            for ki in range(kH):
                                for kj in range(kW):
                                    ii = i*sH - pH + ki*dH
                                    jj = j*sW - pW + kj*dW
                                    if 0 <= ii < prev_H and 0 <= jj < prev_W:
                                        src = position_to_node[(prev_type, prev_ln, ci, ii, jj)]
                                        edge_index[0].append(src); edge_index[1].append(out_id)
                                        w = float(W[co, ci, ki, kj].item()); edge_weight.append(w)
                                        if want_ws:
                                            edge_to_kernel_idx[edge_idx_counter] = (layer_num, co, ci, ki, kj)
                                        edge_idx_counter += 1
            prev = curr; prev_type='conv'; prev_ln=layer_num
            prev_C, prev_H, prev_W = layer.out_channels, H_out, W_out
            layer_neurons.append(curr); layer_types.append('conv')
        elif isinstance(layer, nn.ReLU):
            for idx in prev: activations[idx]='relu'
            layer_types.append('relu')
        elif isinstance(layer, nn.Flatten):
            layer_types.append('flatten')
        elif isinstance(layer, nn.Linear):
            layer_num += 1
            curr = list(range(neuron_counter, neuron_counter+layer.out_features)); neuron_counter += layer.out_features
            for idx, b in zip(curr, layer.bias.data): biases[idx]=float(b.item()); activations[idx]=None; node_to_layer_idx[idx]=layer_num
            W = layer.weight.data
            for t, tgt in enumerate(curr):
                for s, src in enumerate(prev):
                    edge_index[0].append(src); edge_index[1].append(tgt)
                    edge_weight.append(float(W[t, s].item()))
                    if want_ws:
                        edge_to_kernel_idx[edge_idx_counter] = {'layer_num': layer_num, 'target_idx': t, 'source_idx': s}
                    edge_idx_counter += 1
            prev = curr; prev_type='linear'; prev_ln=layer_num
            layer_neurons.append(curr); layer_types.append('linear')

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr  = torch.tensor(edge_weight, dtype=torch.float)
    num_nodes = neuron_counter
    node_bias = torch.zeros(num_nodes)
    for i,b in biases.items(): node_bias[i]=b
    amap = _activation_map()
    node_act = torch.zeros(num_nodes, dtype=torch.long)
    for i,a in activations.items(): node_act[i] = amap[a]
    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes, name=str(cnn))
    data.bias = node_bias; data.activation = node_act
    nodes = list(range(num_nodes))
    if want_ws:
        return data, nodes, layer_neurons, edge_to_kernel_idx, node_to_layer_idx, layer_types
    return data, nodes, layer_neurons
