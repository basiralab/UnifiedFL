import argparse, os, copy, torch
from unifiedfl.utils import get_logger, set_seed
from unifiedfl.data import build_medmnist_splits, MedMNISTDistShift, build_distshift_loader
from unifiedfl.models import CNNClassifierDeep, CNNClassifier, MLPClassifier
from unifiedfl.train import train_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--model-idx", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--outdir", type=str, default="outputs")
    args = ap.parse_args()

    set_seed(args.seed)
    logger = get_logger("ugnn.baseline")

    (tr_loader, va_loader, te_loader), (tr_set, va_set, te_set) = build_medmnist_splits(batch_size=args.batch_size, seed=args.seed)

    import numpy as np
    num_clusters = 3
    cluster_to_indices = {c: np.load(f"PathMNIST_cluster{c}.npy") for c in range(num_clusters)}
    distshift_tr = MedMNISTDistShift(tr_set, cluster_to_indices)
    train_loader = build_distshift_loader(distshift_tr, batch_size=args.batch_size, seed=args.seed)
    model2cluster = {i:i for i in range(3)} # one cluster per model in example

    gen = torch.Generator().manual_seed(args.seed)
    models = [
        CNNClassifierDeep(generator=gen, in_channels=3, num_classes=9),
        CNNClassifier(generator=gen, in_channels=3, num_classes=9),
        MLPClassifier(3*28*28, [100,50,20], 9, generator=gen),
    ]
    model = copy.deepcopy(models[args.model_idx])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crit = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.outdir, exist_ok=True)
    weight_path = os.path.join(args.outdir, f"baseline_model_{args.model_idx}.pt")
    metrics_path = os.path.join(args.outdir, f"baseline_model_{args.model_idx}_metrics.json")

    train_model(
        model, train_loader, va_loader, crit, opt,
        num_epochs=args.epochs, scheduler_patience=50, validate_every_epoch=1,
        early_stopping_patience=100, weight_save_path=weight_path,
        device=device, cluster=model2cluster[args.model_idx],
        metrics_save_path=metrics_path
    )

if __name__ == "__main__":
    main()
