import random
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report, precision_recall_curve

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def optimize_medical_threshold(labels, probs, target_recall=0.95):
    """
    Finds the optimal probability threshold to ensure a minimum recall level,
    prioritizing patient safety (minimizing False Negatives).
    """
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    
    # Find the highest threshold that still gives us our target recall
    optimal_idx = np.where(recalls >= target_recall)[0][-1]
    
    if optimal_idx == len(thresholds):
        optimal_idx -= 1
        
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"\n--- Medical Risk Optimization ---")
    print(f"To guarantee {target_recall*100:.0f}% Recall, shifting decision threshold from 0.50 -> {optimal_threshold:.4f}")
    
    return optimal_threshold

def plot_training_curves(history, suffix=""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_title('Loss per Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('BCE Loss')
    axes[0].legend()

    axes[1].plot(history['train_auc'], label='Train')
    axes[1].plot(history['val_auc'], label='Val')
    axes[1].set_title('AUC per Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f'training_curves_{suffix}.png')
    plt.close()

def evaluate_and_plot(labels, probs, preds, class_names, suffix=""):
    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix ({suffix})')
    plt.savefig(f'confusion_matrix_{suffix}.png')
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = roc_auc_score(labels, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({suffix})')
    plt.legend(loc='lower right')
    plt.savefig(f'roc_curve_{suffix}.png')
    plt.close()