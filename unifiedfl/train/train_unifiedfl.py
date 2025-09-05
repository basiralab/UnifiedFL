import json, torch
from tqdm import tqdm
from unifiedfl.eval.evaluate import evaluate_unified_gnn

def train_unified_gnn(
    model, train_loader, val_loader, criterion, optimizer,
    num_epochs, device, num_models, model2cluster,
    scheduler_patience, validate_every_epoch,
    weight_save_path, early_stopping_patience,
    metrics_save_file, alpha, lr_decay_factor=0.1, min_lr=1e-6
):
    model.to(device); model.train()
    training_losses=[]; val_losses=[]; val_prec=[]; val_rec=[]; val_f1=[]
    best_f1s=[0.0]*num_models; best_avg_f1=0.0
    epochs_no_improve=0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")

    for epoch in range(num_epochs):
        model.train(); total_loss=0.0
        for batch in tqdm(train_loader, desc=f"Training {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            outputs = model(batch)
            labels = [batch[model2cluster[i]][1].to(device) for i in range(num_models)]
            losses = [criterion(o, y) for o,y in zip(outputs, labels)]
            loss = sum(losses)/num_models
            loss.backward(); optimizer.step()
            model.update_buffers()
            total_loss += float(loss.item())
        avg_loss = total_loss / max(1, len(train_loader))
        training_losses.append(avg_loss)
        scheduler.step(min(training_losses))

        if (epoch+1) % validate_every_epoch == 0:
            val_metrics, avg_val_loss = evaluate_unified_gnn(model, val_loader, num_models, device)
            val_losses.append(avg_val_loss)
            current_f1s=[]
            any_improved=False
            for i,(p,r,f) in enumerate(val_metrics):
                if f>best_f1s[i]:
                    best_f1s[i]=f; any_improved=True
                    torch.save({"data": model.data, "edge_attr": model.edge_attr, "biases": model.biases},
                               f"{weight_save_path}_graph_{i+1}_best_f1.pt")
                val_prec.append((i,p)); val_rec.append((i,r)); val_f1.append((i,f)); current_f1s.append(f)
            if any_improved: epochs_no_improve=0
            else: epochs_no_improve += 1
            avg_f1 = sum(current_f1s)/len(current_f1s)
            if avg_f1 > best_avg_f1:
                best_avg_f1 = avg_f1
                torch.save(model.state_dict(), weight_save_path + ".pt")
            if epochs_no_improve >= early_stopping_patience: break

    with open(metrics_save_file, "w") as f:
        json.dump({
            "training_losses": training_losses, "val_losses": val_losses,
            "val_prec": val_prec, "val_rec": val_rec, "val_f1": val_f1,
            "best_f1s": best_f1s, "best_avg_f1": best_avg_f1
        }, f)
    return training_losses, val_losses, val_prec, val_rec, val_f1
