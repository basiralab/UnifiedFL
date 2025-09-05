import argparse, torch, os, json
from unifiedfl.data import build_medmnist_splits
from unifiedfl.models import UGNN_WS, CNNClassifierDeep, CNNClassifier, MLPClassifier
from unifiedfl.graphs import unify_ws
from unifiedfl.eval import test_unified_gnn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=False, help="ugnn_ws.pt (optional to load state_dict)")
    args = ap.parse_args()

    (tr_loader, va_loader, te_loader), _ = build_medmnist_splits()
    gen = torch.Generator().manual_seed(42)
    models = [
        CNNClassifierDeep(generator=gen, in_channels=3, num_classes=9),
        CNNClassifier(generator=gen, in_channels=3, num_classes=9),
        MLPClassifier(3*28*28, [100,50,20], 9, generator=gen),
    ]
    data_tuple = unify_ws(models, (3,28,28))
    (data, node_idx_list, layer_neurons_list, layer_types_lists, input_idx_list, output_idx_list,
     edge_to_kernel_idx, node_to_layer_idx) = data_tuple

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model2cluster = {i:i for i in range(len(models))}
    model = UGNN_WS(
        data, layer_neurons_list, layer_types_lists, input_idx_list, model2cluster, device,
        edge_to_kernel_idx, node_to_layer_idx, k_edge_theta=5_000_000, k_bias_theta=1_000_000, act="softsign", scale=1.5
    ).to(device)

    if args.weights and os.path.isfile(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))

    metrics = test_unified_gnn(model, te_loader, device, num_mlps=len(models))
    print(json.dumps([{"model":i, "prec":p, "rec":r, "f1":f} for i,(p,r,f) in enumerate(metrics)], indent=2))

if __name__ == "__main__":
    main()
