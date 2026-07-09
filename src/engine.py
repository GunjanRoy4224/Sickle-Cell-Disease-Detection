import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from src.config import DEVICE

def run_epoch(model, loader, criterion, optimizer, scaler, use_amp, train=True):
    epoch_loss = 0.0
    all_probs, all_labels = [], []
    
    if train:
        model.train()
    else:
        model.eval()

    for images, labels, _ in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        if train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type='cuda', enabled=use_amp):
                logits = model(images).squeeze(1)
                loss = criterion(logits, labels) if criterion else torch.tensor(0.0)
            
            if train and optimizer is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        epoch_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    epoch_loss /= len(loader.dataset)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float('nan')
        
    return {'loss': epoch_loss, 'auc': auc, 'y_true': all_labels, 'y_prob': all_probs}