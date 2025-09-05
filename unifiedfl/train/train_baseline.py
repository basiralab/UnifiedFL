import json, torch
from tqdm import tqdm
from unifiedfl.eval.evaluate import evaluate_model

def train_model(
    model, train_loader, val_loader, criterion, optimizer,
    num_epochs, scheduler_patience, validate_every_epoch,
    early_stopping_patience, weight_save_path, device, cluster, metrics_save_path
):
    model.to(device); model.train()
    training_losses=[]; val_losses=[]; val_prec=[]; val_rec=[]; val_f1=[]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=scheduler_patience, min_lr=1e-6)
    best_f1 = 0.0; epochs_no_improve=0

    for epoch in range(num_epochs):
        model.train(); total=0.0
        for batch in tqdm(train_loader, desc=f"Training {epoch+1}/{num_epochs}"):
            images, labels = batch[cluster]
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward(); optimizer.step()
            total += float(loss.item())
        training_losses.append(total/max(1,len(train_loader)))

        if (epoch+1) % validate_every_epoch == 0:
            p,r,f,vl,_,_ = evaluate_model(model, val_loader, device)
            val_losses.append(vl); val_prec.append(p); val_rec.append(r); val_f1.append(f)
            if f > best_f1:
                best_f1=f; epochs_no_improve=0
                torch.save(model.state_dict(), weight_save_path)
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience: break
            scheduler.step(f)

    with open(metrics_save_path, "w") as f:
        json.dump({
            "training_losses": training_losses, "val_losses": val_losses,
            "val_prec": val_prec, "val_rec": val_rec, "val_f1": val_f1,
            "best_f1": best_f1
        }, f)
    return training_losses, val_losses, val_prec, val_rec, val_f1
