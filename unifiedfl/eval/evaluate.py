import torch
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

@torch.no_grad()
def evaluate_unified_gnn(model, data_loader, num_models, device):
    model.eval()
    all_labels = []
    all_predictions_list = [[] for _ in range(num_models)]
    total_loss = 0
    crit = torch.nn.CrossEntropyLoss()
    for images, labels in tqdm(data_loader, desc="Validating"):
        images, labels = images.to(device), labels.to(device)
        outputs = model([(images, labels) for _ in range(num_models)])
        losses = [crit(o, labels) for o in outputs]
        loss = sum(losses) / num_models
        total_loss += float(loss.item())
        for i, o in enumerate(outputs):
            _, pred = torch.max(o.data, 1)
            all_predictions_list[i].extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    metrics_list = []
    for i in range(num_models):
        preds = all_predictions_list[i]
        p,r,f,_ = precision_recall_fscore_support(all_labels, preds, average='weighted')
        metrics_list.append((p,r,f))
    avg_loss = total_loss / max(1, len(data_loader))
    return metrics_list, avg_loss

@torch.no_grad()
def test_unified_gnn(model, test_loader, device, num_mlps):
    m, _ = evaluate_unified_gnn(model, test_loader, num_models=num_mlps, device=device)
    return m

@torch.no_grad()
def evaluate_model(model, data_loader, device):
    model.eval()
    all_labels, all_preds = [], []
    total_loss = 0
    crit = torch.nn.CrossEntropyLoss()
    for images, labels in tqdm(data_loader, desc="Validating"):
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        loss = crit(out, labels); total_loss += float(loss.item())
        _, pred = torch.max(out.data, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(pred.cpu().numpy())
    p,r,f,_ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    avg_loss = total_loss / max(1, len(data_loader))
    return p,r,f,avg_loss, all_preds, all_labels
