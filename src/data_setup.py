import kagglehub
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from src.config import IMG_SIZE, BATCH_SIZE, NUM_WORKERS, SEED, IMAGENET_MEAN, IMAGENET_STD

class CellDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row.path).convert('RGB')
        img = self.transform(img)
        label = torch.tensor(row.label, dtype=torch.float32)
        return img, label, row.path

def get_transforms():
    train_tfms = T.Compose([
        T.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_tfms = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tfms, val_tfms

def prepare_data():
    path = kagglehub.dataset_download("fenicxs/sickle-cell-anaemia")
    data_root = Path(path)
    
    extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    class_names = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
    records = []
    
    for label_idx, cls in enumerate(class_names):
        for img_path in sorted((data_root / cls).glob('*')):
            if img_path.suffix.lower() in extensions:
                records.append({'path': str(img_path), 'label': label_idx, 'class_name': cls})

    df = pd.DataFrame(records)
    
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=SEED)
    
    return train_df, val_df, test_df, class_names

def build_loaders():
    train_df, val_df, test_df, class_names = prepare_data()
    train_tfms, val_tfms = get_transforms()

    train_ds = CellDataset(train_df, train_tfms)
    val_ds = CellDataset(val_df, val_tfms)
    test_ds = CellDataset(test_df, val_tfms)

    class_counts = train_df['label'].value_counts().to_dict()
    weights = train_df['label'].map(lambda x: 1.0 / class_counts[x]).values.copy()
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    return train_loader, val_loader, test_loader, class_names, test_df, val_tfms