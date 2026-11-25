import json
import torch
import torch.nn as nn
import numpy as np
import optuna
import gc
from pathlib import Path
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             accuracy_score, precision_score, recall_score, f1_score)
from torch.utils.data import DataLoader

from src.config import SEED, BATCH_SIZE, EPOCHS, N_TRIALS, HPO_EPOCHS, NUM_WORKERS
from src.utils import seed_everything, plot_training_curves, evaluate_and_plot, optimize_medical_threshold
from src.data_setup import build_loaders, CellDataset
from src.model import get_model
from src.engine import run_epoch
from src.explainability import generate_grad_cam

def objective(trial, model_name, train_loader, val_loader):
    """Optuna objective function scoped for specific architectures."""
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    
    model = get_model(model_name)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=HPO_EPOCHS, eta_min=1e-6)
    
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    best_val_auc = -1

    for epoch in range(1, HPO_EPOCHS + 1):
        _ = run_epoch(model, train_loader, criterion, optimizer, scaler, use_amp, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, scaler, use_amp, train=False)
        scheduler.step()
        
        val_auc = val_metrics['auc']
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            
        trial.report(val_auc, epoch)
        
        if trial.should_prune():
            del model, optimizer, criterion
            torch.cuda.empty_cache()
            gc.collect()
            raise optuna.TrialPruned()
            
    del model, optimizer, criterion
    torch.cuda.empty_cache()
    gc.collect()
    
    return best_val_auc

def run_pipeline_for_architecture(model_name, train_loader, val_loader, test_loader, class_names, test_df, val_tfms):
    """Executes the full training, tuning, and evaluation pipeline for a single model."""
    print(f"\n{'='*60}")
    print(f"STARTING PIPELINE FOR: {model_name}")
    print(f"{'='*60}\n")
    
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    criterion = nn.BCEWithLogitsLoss()
    save_path = f"best_model_{model_name}.pth"

    # 1. HPO Search
    print(f"--- Starting Optuna Hyperparameter Search ({N_TRIALS} Trials) ---")
    study = optuna.create_study(
        direction="maximize", 
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=2)
    )
    # Pass arguments directly to the objective using a lambda
    study.optimize(lambda trial: objective(trial, model_name, train_loader, val_loader), n_trials=N_TRIALS)
    
    print("\n=== Hyperparameter Search Complete ===")
    print(f"Best Trial AUC: {study.best_trial.value:.4f}")
    print(f"Best Params: {study.best_params}")
    
    # 2. Final Training with Best Params
    print("\n--- Starting Final Training Pipeline ---")
    best_lr = study.best_params["lr"]
    best_wd = study.best_params["weight_decay"]
    
    model = get_model(model_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_lr, weight_decay=best_wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}
    best_val_auc = -1

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, scaler, use_amp, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, scaler, use_amp, train=False)
        scheduler.step()

        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_auc'].append(train_metrics['auc'])
        history['val_auc'].append(val_metrics['auc'])

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            torch.save(model.state_dict(), save_path)

        print(f"Epoch {epoch}/{EPOCHS} | Train loss {train_metrics['loss']:.4f} AUC {train_metrics['auc']:.4f} | Val loss {val_metrics['loss']:.4f} AUC {val_metrics['auc']:.4f}")
    
    plot_training_curves(history, suffix=model_name)

    # 3. Evaluation on Best Weights
    print(f"\nLoading best {model_name} weights for evaluation...")
    model.load_state_dict(torch.load(save_path, weights_only=True))
    
    # --- 1) THRESHOLD SELECTION: validation split only (the test split must not be used to tune anything) ---
    val_metrics = run_epoch(model, val_loader, criterion, None, scaler, use_amp, train=False)
    val_probs = np.array(val_metrics['y_prob'])
    val_labels = np.array(val_metrics['y_true'])
    optimal_threshold = float(optimize_medical_threshold(val_labels, val_probs, target_recall=0.95))

    # --- 2) TEST SET EVALUATION: reported once, using the validation-selected threshold ---
    test_metrics = run_epoch(model, test_loader, criterion, None, scaler, use_amp, train=False)
    test_probs = np.array(test_metrics['y_prob'])
    test_labels = np.array(test_metrics['y_true'])
    test_preds = (test_probs >= optimal_threshold).astype(int)
    test_preds_05 = (test_probs >= 0.5).astype(int)

    print('\n========================================')
    print(f'TEST SET CLASSIFICATION REPORT ({model_name}), threshold={optimal_threshold:.4f} (chosen on validation)')
    print('========================================')
    print(classification_report(test_labels, test_preds, target_names=class_names, digits=4))

    results_dir = Path("results"); results_dir.mkdir(exist_ok=True)
    def summarize(preds):
        return {
            "accuracy": float(accuracy_score(test_labels, preds)),
            "precision": float(precision_score(test_labels, preds, zero_division=0)),
            "recall": float(recall_score(test_labels, preds)),
            "f1": float(f1_score(test_labels, preds)),
            "confusion_matrix": confusion_matrix(test_labels, preds).tolist(),
        }
    summary = {
        "model": model_name,
        "classes": class_names,
        "n_test": int(len(test_labels)),
        "roc_auc": float(roc_auc_score(test_labels, test_probs)),
        "threshold_selected_on_validation": optimal_threshold,
        "at_validation_threshold": summarize(test_preds),
        "at_0.5": summarize(test_preds_05),
        "best_hpo_params": study.best_params,
    }
    (results_dir / f"metrics_{model_name}.json").write_text(json.dumps(summary, indent=2))
    thr_file = results_dir / "thresholds.json"
    thr = json.loads(thr_file.read_text()) if thr_file.exists() else {}
    thr[model_name] = optimal_threshold
    thr_file.write_text(json.dumps(thr, indent=2))
    print(f"Saved {results_dir / f'metrics_{model_name}.json'} and updated thresholds.json")

    # --- TRAIN SET EVALUATION ---
    print("\nEvaluating on training set...")
    # evaluate on un-augmented training images (the loader's dataset applies random augmentation)
    clean_train_ds = CellDataset(train_loader.dataset.df, val_tfms)
    sequential_train_loader = DataLoader(clean_train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    train_metrics = run_epoch(model, sequential_train_loader, criterion, None, scaler, use_amp, train=False)
    train_probs = np.array(train_metrics['y_prob'])
    train_labels = np.array(train_metrics['y_true'])
    train_preds = (train_probs >= optimal_threshold).astype(int)

    print('\n========================================')
    print(f'TRAIN SET CLASSIFICATION REPORT ({model_name})')
    print('========================================')
    print(classification_report(train_labels, train_preds, target_names=class_names, digits=4))

    # 4. Plotting & Explainability
    evaluate_and_plot(test_labels, test_probs, test_preds, class_names, suffix=model_name)
    print(f"\nGenerating Smoothed Grad-CAM visualizations for {model_name}...")
    generate_grad_cam(model, test_df, val_tfms, model_name=model_name)
    
    # Aggressively release VRAM before the next architecture starts
    del model
    torch.cuda.empty_cache()
    gc.collect()

def main():
    seed_everything(SEED)
    print("Loading data...")
    train_loader, val_loader, test_loader, class_names, test_df, val_tfms = build_loaders()
    
    # The ultimate showdown list
    architectures = ['swin_base_patch4_window7_224', 'resnet50']
    
    for arch in architectures:
        run_pipeline_for_architecture(arch, train_loader, val_loader, test_loader, class_names, test_df, val_tfms)

if __name__ == '__main__':
    main()