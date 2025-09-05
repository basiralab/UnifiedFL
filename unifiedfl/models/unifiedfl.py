import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

# small activations
class ScaledTanh(nn.Module):
    def __init__(self, scale=1.0): super().__init__(); self.scale=scale
    def forward(self, x): return self.scale * torch.tanh(x / self.scale)

class ScaledSoftsign(nn.Module):
    def __init__(self, scale=1.0): super().__init__(); self.scale=scale
    def forward(self, x): return self.scale * x / (self.scale + x.abs())

def _act(act: str, scale: float):
    acts = {
        "tanh": nn.Tanh(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(),
        "leaky_relu": nn.LeakyReLU(), "identity": nn.Identity(),
        "softsign": nn.Softsign(), "scaled_tanh": ScaledTanh(scale),
        "scaled_softsign": ScaledSoftsign(scale),
    }
    if act not in acts: raise ValueError(f"Unsupported activation {act}")
    return acts[act]

class UGNN(MessagePassing):
    def __init__(self, data, layer_neurons_lists, input_indices_list, model2cluster, device,
                 k_edge_theta, k_bias_theta, act: str, scale: float, name=None, seed=42):
        super().__init__(aggr=None)
        self.data = data
        self.layer_neurons_lists = layer_neurons_lists
        self.input_indices_list = input_indices_list
        self.model2cluster = model2cluster
        self.activations = data.activation.to(device)
        self.edge_index = data.edge_index.to(device)
        self.device = device
        self.scale = scale

        g = torch.Generator(device=device).manual_seed(seed)
        self.register_buffer("edge_attr", data.edge_attr.clone().to(device))
        self.register_buffer("biases",    data.bias.clone().to(device))

        self.k_edge_theta = min(data.edge_attr.size(0), k_edge_theta)
        non_input = data.num_nodes - sum(len(x) for x in input_indices_list)
        self.k_bias_theta = min(non_input, k_bias_theta)

        self.theta_edge = nn.Parameter(torch.ones(self.k_edge_theta, device=device))
        self.theta_bias = nn.Parameter(torch.ones(self.k_bias_theta, device=device))
        self.theta_edge_shift = nn.Parameter(torch.zeros(self.k_edge_theta, device=device))
        self.theta_bias_shift = nn.Parameter(torch.zeros(self.k_bias_theta, device=device))
        self.model_loss_weights = nn.Parameter(torch.ones(len(model2cluster), device=device))
        self.theta_edge_act_scale = nn.Parameter(torch.ones(1, device=device))
        self.theta_bias_act_scale = nn.Parameter(torch.ones(1, device=device))
        self.alpha = nn.Parameter(torch.ones(1, device=device))

        # random groups
        self.edge_groups = torch.randint(0, self.k_edge_theta, (self.edge_index.size(1),), generator=g, device=device) \
            if self.k_edge_theta < self.edge_index.size(1) else torch.randperm(self.edge_index.size(1), generator=g, device=device)
        bias_groups = torch.full((data.num_nodes,), -1, dtype=torch.long, device=device)
        all_input = torch.tensor([i for inds in input_indices_list for i in inds], device=device, dtype=torch.long)
        mask = torch.ones(data.num_nodes, dtype=torch.bool, device=device)
        if all_input.numel(): mask[all_input]=False
        k = mask.sum().item()
        if self.k_bias_theta < k:
            bias_groups[mask] = torch.randint(0, self.k_bias_theta, (k,), generator=g, device=device)
        else:
            bias_groups[mask] = torch.randperm(k, generator=g, device=device)
        self.bias_groups = bias_groups

        self.act = _act(act, scale)
        self.name = name or f"uGNN-k_edge_theta-{k_edge_theta}-k_bias_theta-{k_bias_theta}-act-{act}-"

    def forward(self, x_batch):
        bs = x_batch[0][0].shape[0]
        if self.training:
            updated_edge_attr, updated_biases = self._update_edge_bias()
        else:
            updated_edge_attr, updated_biases = self.edge_attr, self.biases

        h = torch.zeros(bs, self.data.num_nodes, 1, device=self.device)
        for i, input_indices in enumerate(self.input_indices_list):
            h[:, input_indices] = x_batch[self.model2cluster[i]][0].view(bs, -1, 1).to(self.device)

        outs, h_shared = [], h.clone()
        for layer_neurons in self.layer_neurons_lists:
            h_model = h_shared.clone()
            for l in range(1, len(layer_neurons)):
                curr = layer_neurons[l]
                curr_t = torch.tensor(curr, device=self.device)
                mask = (self.edge_index[1].unsqueeze(0) == curr_t.unsqueeze(1)).any(0)
                edge_index_layer = self.edge_index[:, mask]
                edge_attr_layer  = updated_edge_attr[mask]
                h_current = self.propagate(edge_index_layer, x=h_model, edge_weight=edge_attr_layer, batch_size=bs)
                biases_current = updated_biases[curr].view(1, -1, 1)
                h_current[:, curr] += biases_current
                act_flags = self.activations[curr]
                nodes_to_act = curr_t[act_flags == 1]
                if nodes_to_act.numel() > 0:
                    h_current[:, nodes_to_act] = torch.relu(h_current[:, nodes_to_act])
                h_model = h_current
            out_nodes = layer_neurons[-1]
            outs.append(h_model[:, out_nodes].squeeze(2))
        return outs

    def message(self, x_j, edge_weight): return edge_weight.view(1, -1, 1) * x_j
    def aggregate(self, inputs, index, batch_size): 
        H = torch.zeros(batch_size, self.data.num_nodes, 1, device=inputs.device)
        H.index_add_(1, index, inputs); return H
    def update(self, inputs): return inputs

    def _scaled_softsign(self, scale, x): return scale * x / (scale + x.abs())

    def _update_edge_bias(self):
        te = self.theta_edge[self.edge_groups]
        te_s = self.theta_edge_shift[self.edge_groups]
        updated_edge_attr = self._scaled_softsign(self.theta_edge_act_scale, self.edge_attr * te + te_s)
        updated_biases = self.biases.clone()
        mask = self.bias_groups >= 0
        if mask.any():
            tb = self.theta_bias[self.bias_groups[mask]]
            tb_s = self.theta_bias_shift[self.bias_groups[mask]]
            updated_biases[mask] = self._scaled_softsign(self.theta_bias_act_scale, self.biases[mask] * tb + tb_s)
        return updated_edge_attr, updated_biases

    @torch.no_grad()
    def update_buffers(self):
        e, b = self._update_edge_bias()
        self.edge_attr.copy_(e); self.biases.copy_(b)

    def __repr__(self): return self.name

class UGNN_WS(UGNN):
    """Weight-sharing grouping by kernel/linear indices."""
    def __init__(self, data, layer_neurons_lists, layer_types_lists, input_indices_list, model2cluster, device,
                 edge_to_kernel_idx, node_to_layer_idx, k_edge_theta, k_bias_theta, act, scale, name=None, seed=42):
        super(MessagePassing, self).__init__()
        self.data = data
        self.layer_neurons_lists = layer_neurons_lists
        self.layer_types_lists = layer_types_lists
        self.input_indices_list = input_indices_list
        self.model2cluster = model2cluster
        self.device = device
        self.scale = scale
        self.activations = data.activation.to(device)
        self.edge_index = data.edge_index.to(device)

        g = torch.Generator(device=device).manual_seed(seed)
        self.register_buffer("edge_attr", data.edge_attr.clone().to(device))
        self.register_buffer("biases",    data.bias.clone().to(device))

        # unique kernels
        uniq = {}
        for _, info in edge_to_kernel_idx.items():
            if isinstance(info, tuple):
                key = ('conv', *info)
            else:
                key = ('linear', info['layer_num'], info['target_idx'], info['source_idx'])
            if key not in uniq: uniq[key] = len(uniq)
        total_kernels = len(uniq)
        self.k_edge_theta = min(total_kernels, k_edge_theta)

        non_input = data.num_nodes - sum(len(x) for x in input_indices_list)
        self.k_bias_theta = min(non_input, k_bias_theta)

        self.theta_edge = nn.Parameter(torch.ones(self.k_edge_theta, device=device))
        self.theta_bias = nn.Parameter(torch.ones(self.k_bias_theta, device=device))
        self.theta_edge_shift = nn.Parameter(torch.zeros(self.k_edge_theta, device=device))
        self.theta_bias_shift = nn.Parameter(torch.zeros(self.k_bias_theta, device=device))
        self.model_loss_weights = nn.Parameter(torch.ones(len(model2cluster), device=device))
        self.theta_edge_act_scale = nn.Parameter(torch.ones(1, device=device))
        self.theta_bias_act_scale = nn.Parameter(torch.ones(1, device=device))
        self.alpha = nn.Parameter(torch.ones(1, device=device))

        # edge groups map kernel ids -> [0..k_edge_theta)
        kernel_id = []
        for e in range(self.edge_index.size(1)):
            info = edge_to_kernel_idx[e]
            key = ('conv', *info) if isinstance(info, tuple) else ('linear', info['layer_num'], info['target_idx'], info['source_idx'])
            kernel_id.append(uniq[key])
        kernel_id = torch.tensor(kernel_id, device=device)
        if self.k_edge_theta >= total_kernels:
            self.edge_groups = kernel_id
        else:
            rand_groups = torch.randint(0, self.k_edge_theta, (total_kernels,), generator=g, device=device)
            self.edge_groups = rand_groups[kernel_id]

        # bias groups
        bias_groups = torch.full((data.num_nodes,), -1, dtype=torch.long, device=device)
        all_input = torch.tensor([i for inds in input_indices_list for i in inds], device=device, dtype=torch.long)
        mask = torch.ones(data.num_nodes, dtype=torch.bool, device=device)
        if all_input.numel(): mask[all_input]=False
        k = mask.sum().item()
        if self.k_bias_theta < k:
            bias_groups[mask] = torch.randint(0, self.k_bias_theta, (k,), generator=g, device=device)
        else:
            bias_groups[mask] = torch.randperm(k, generator=g, device=device)
        self.bias_groups = bias_groups

        self.act = _act(act, scale)
        self.name = name or f"uGNN-WS-k_edge_theta-{k_edge_theta}-k_bias_theta-{k_bias_theta}-act-{act}-"

    # forward/message/aggregate/update and buffer logic inherited from UGNN

class UGNNModelSpecific(MessagePassing):
    """Per-model θ groups (edge & bias) with shared graph state."""
    def __init__(self, data, node_indices_list, layer_neurons_list, input_indices_list, model2cluster,
                 device, k_edge_theta, k_bias_theta, act, scale, name=None, seed=42):
        super().__init__(aggr=None)
        self.data = data
        self.node_indices_list = node_indices_list
        self.layer_neurons_list = layer_neurons_list
        self.input_indices_list = input_indices_list
        self.model2cluster = model2cluster
        self.device = device
        self.scale = scale

        self.edge_index = data.edge_index.to(device)
        self.register_buffer("edge_attr", data.edge_attr.clone().to(device))
        self.register_buffer("biases",    data.bias.clone().to(device))
        self.activations = data.activation.to(device)

        self.num_models = len(model2cluster)
        self.theta_edge_per_model = nn.ParameterList()
        self.theta_bias_per_model = nn.ParameterList()
        self.edge_groups_per_model = []
        self.bias_groups_per_model = []
        self.model_edge_indices = []
        self.model_non_input_nodes = []

        g = torch.Generator(device=device).manual_seed(seed)
        input_nodes_all = torch.tensor([i for inds in input_indices_list for i in inds], device=device, dtype=torch.long)
        input_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
        if input_nodes_all.numel(): input_mask[input_nodes_all] = True

        node_model_id = torch.full((data.num_nodes,), -1, dtype=torch.long, device=device)
        for m, nodes in enumerate(node_indices_list):
            node_model_id[torch.tensor(nodes, device=device)] = m
        self.edge_model_id = node_model_id[self.edge_index[1]]

        for m in range(self.num_models):
            edge_mask = (self.edge_model_id == m)
            edge_ids = edge_mask.nonzero(as_tuple=True)[0]
            self.model_edge_indices.append(edge_ids)

            model_nodes = torch.tensor(node_indices_list[m], device=device)
            non_input_nodes = model_nodes[~input_mask[model_nodes]]
            self.model_non_input_nodes.append(non_input_nodes)

            k_e = min(edge_ids.numel(), k_edge_theta)
            k_b = min(non_input_nodes.numel(), k_bias_theta)
            self.theta_edge_per_model.append(nn.Parameter(torch.ones(k_e, device=device)))
            self.theta_bias_per_model.append(nn.Parameter(torch.ones(k_b, device=device)))

            if k_e > 0 and edge_ids.numel() > 0:
                if k_e >= edge_ids.numel():
                    edge_groups = torch.randperm(edge_ids.numel(), generator=g, device=device)
                else:
                    edge_groups = torch.randint(0, k_e, (edge_ids.numel(),), generator=g, device=device)
            else:
                edge_groups = torch.full((edge_ids.numel(),), -1, device=device, dtype=torch.long)
            self.edge_groups_per_model.append(edge_groups)

            if k_b > 0 and non_input_nodes.numel() > 0:
                if k_b >= non_input_nodes.numel():
                    bias_groups = torch.randperm(non_input_nodes.numel(), generator=g, device=device)
                else:
                    bias_groups = torch.randint(0, k_b, (non_input_nodes.numel(),), generator=g, device=device)
            else:
                bias_groups = torch.full((non_input_nodes.numel(),), -1, device=device, dtype=torch.long)
            self.bias_groups_per_model.append(bias_groups)

        self.act = _act(act, scale)
        self.name = name or f"UGNNModelSpecific-k_edge_theta-{k_edge_theta}-k_bias_theta-{k_bias_theta}-act-{act}-"

    def forward(self, x_batch):
        bs = x_batch[0][0].shape[0]
        upd_e, upd_b = self._update_edge_bias()
        h = torch.zeros(bs, self.data.num_nodes, 1, device=self.device)
        for i, inp in enumerate(self.input_indices_list):
            h[:, inp] = x_batch[self.model2cluster[i]][0].view(bs, -1, 1).to(self.device)
        outs = []
        for layers in self.layer_neurons_list:
            out_nodes = layers[-1]
            outs.append(h[:, out_nodes].squeeze(2))
        return outs

    def message(self, x_j, edge_weight): return edge_weight.view(1, -1, 1) * x_j
    def aggregate(self, inputs, index, batch_size=None):
        H = torch.zeros(inputs.size(0), self.data.num_nodes, 1, device=inputs.device)
        H.index_add_(1, index, inputs); return H
    def update(self, inputs): return inputs

    def _update_edge_bias(self):
        ue = self.edge_attr.clone()
        ub = self.biases.clone()
        for m in range(self.num_models):
            e_idx = self.model_edge_indices[m]
            non_in = self.model_non_input_nodes[m]
            eg = self.edge_groups_per_model[m]
            bg = self.bias_groups_per_model[m]
            if eg.numel() and self.theta_edge_per_model[m].numel():
                theta_e = self.theta_edge_per_model[m][eg]
                ue[e_idx] = ue[e_idx] * theta_e
            if bg.numel() and self.theta_bias_per_model[m].numel():
                theta_b = self.theta_bias_per_model[m][bg]
                ub[non_in] = ub[non_in] * theta_b
        return ue, ub

    def __repr__(self): return self.name
