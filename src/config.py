import torch
import os

# System
SEED = 42
NUM_WORKERS = 8 if os.name != 'nt' else 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 8   # Optimized for 4GB VRAM
EPOCHS = 15      # Final training epochs

# Optuna Settings
N_TRIALS = 10    # Number of hyperparameter combinations to try
HPO_EPOCHS = 5   # Number of epochs per trial (fast evaluation)

# Normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Paths
MODEL_SAVE_PATH = "best_model.pth"