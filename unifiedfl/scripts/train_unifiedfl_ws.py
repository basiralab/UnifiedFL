import argparse, os, torch
from unifiedfl.utils import get_logger, set_seed, count_learnable_parameters
from unifiedfl.data import build_medmnist_splits, MedMNISTDistShift, build_distshift_loader
from unifiedfl.models import CNNClassifierDeep, CNNClassifier, MLPClassifier, UGNN_WS
from unifiedfl.graphs import unify_ws

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--dataset", type=str, default="pathmnist")
    ap.add_argument("--num-clusters", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--k-edge-theta", type=int, default=5_000_000)
    ap.add_argument("--k-bias-theta", type=int, default=1_000_000)
    ap.add_argument("--act", type=str, default="softsign")
    ap.add_argument("--scale", type=float, default=1.5)
    ap.add_argument("--outdir", type=str, default="outputs")
    args = ap.parse_args()

    set_seed(args.seed)
    logger = get_logger("ugnn.scripts")

    (tr_loader, va_loader, te_loader), (tr_set, va_set, te_set) = build_medmnist_splits(batch_size=args.batch_size, seed=args.seed)

    # assume you saved cluster indices as .npy files: PathMNIST_cluster{0..k-1}.npy
    cluster_to_indices = {}
    for c in range(args.num_clusters):
        import numpy as np
        arr = np.load(f"PathMNIST_cluster{c}.npy")
        cluster_to_indices[c] = arr
        logger.info(f"Cluster {c}: {len(arr)} samples")

    distshift_tr = MedMNISTDistShift(tr_set, cluster_to_indices)
    train_loader = build_distshift_loader(distshift_tr, batch_size=args.batch_size, seed=args.seed)

    gen = torch.Generator().manual_seed(args.seed)
    models = [
        CNNClassifierDeep(generator=gen, in_channels=3, num_classes=9),
        CNNClassifier(generator=gen, in_channels=3, num_classes=9),
        MLPClassifier(3*28*28, [100,50,20], 9, generator=gen),
    ]
    for m in models:
        logger.info(f"{m} :: params={count_learnable_parameters(m)}")

    data_tuple = unify_ws(models, (3,28,28))
    (data, node_idx_list, layer_neurons_list, layer_types_lists, input_idx_list, output_idx_list,
     edge_to_kernel_idx, node_to_layer_idx) = data_tuple

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model2cluster = {i:i for i in range(len(models))}

    ugnn = UGNN_WS(
        data, layer_neurons_list, layer_types_lists, input_idx_list, model2cluster, device,
        edge_to_kernel_idx, node_to_layer_idx,
        k_edge_theta=args.k_edge_theta, k_bias_theta=args.k_bias_theta,
        act=args.act, scale=args.scale
    ).to(device)

    from unifiedfl.train import train_unified_gnn
    import torch.nn as nn, torch.optim as optim
    os.makedirs(args.outdir, exist_ok=True)
    optimz = optim.AdamW(ugnn.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    metrics_path = os.path.join(args.outdir, f"ugnn_ws_metrics.json")
    weight_prefix = os.path.join(args.outdir, "ugnn_ws")
    train_unified_gnn(
        ugnn, train_loader, va_loader, crit, optimz, args.epochs, device, len(models),
        model2cluster, scheduler_patience=50, validate_every_epoch=1,
        weight_save_path=weight_prefix, early_stopping_patience=100,
        metrics_save_file=metrics_path, alpha=5
    )

if __name__ == "__main__":
    main()
