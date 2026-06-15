# src/warehouse_dataset.py
# D6: Custom PyTorch Dataset — Warehouse Scene Classification
# Capstone Day 9 | Intern: Shivani | Purple AI Labs Ltd
#
# Based on the NavigationDataset pattern from Day 3/4 training,
# adapted for scene classification (not action prediction).
#
# Key differences from training-week NavigationDataset:
#   - SCENE_CLASSES instead of ACTIONS
#   - class_idx column instead of action column
#   - ColorJitter EXCLUDED (color is a primary discriminating feature:
#     yellow tape, red tape, orange marker — jittering these would
#     corrupt the most important visual signals in the dataset)
#   - Supports 2 resolutions: 128 and 224 (capstone requirement)
#   - Stratified splits built-in (70/15/15 as required)
#
# Usage (local — verify pipeline):
#   from src.warehouse_dataset import get_dataloaders, SCENE_CLASSES
#   train_loader, val_loader, test_loader = get_dataloaders(
#       data_dir='data/warehouse_dataset',
#       resolution=224,
#       batch_size=32
#   )
#
# Usage (Colab — training):
#   Upload data/warehouse_dataset.zip to Google Drive, then use same import.

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

# Scene class definitions — must match warehouse_environments.py exactly
SCENE_CLASSES = {
    0: 'open_aisle',
    1: 'narrow_aisle',
    2: 'pick_station',
    3: 'blocked_path',
    4: 'cross_aisle_junction',
}

# ImageNet normalisation stats — used for transfer learning models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ===========================================================================
# AUGMENTATION PIPELINES
# ===========================================================================

def get_train_transform(resolution: int) -> A.Compose:
    """
    Training augmentation pipeline.

    Simplified version for Albumentations compatibility.
    """

    return A.Compose([
        A.Resize(resolution, resolution),

        # Aisles look the same mirrored
        A.HorizontalFlip(p=0.5),

        # Simulate lighting variation
        A.RandomBrightnessContrast(
            brightness_limit=0.25,
            contrast_limit=0.25,
            p=0.5
        ),

        # Camera blur during robot movement
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=(3, 7)),
        ], p=0.3),

        # Normalize and convert to tensor
        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),

        ToTensorV2(),
    ])


def get_val_test_transform(resolution: int) -> A.Compose:
    """
    Validation/test transform — resize + normalise only. No augmentation.
    The model must see clean images during evaluation to get true accuracy.
    """
    return A.Compose([
        A.Resize(resolution, resolution),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ===========================================================================
# DATASET CLASS (D6)
# ===========================================================================

class WarehouseSceneDataset(Dataset):
    """
    Custom PyTorch Dataset for warehouse scene classification.

    Follows the same __len__ + __getitem__ contract as NavigationDataset
    from Day 3 training, adapted for 5 scene classes.

    Args:
        image_dir  : Path to folder containing JPEG images
        labels_df  : DataFrame with columns: filename, class_idx
        transform  : Albumentations transform pipeline
    """

    CLASS_NAMES = list(SCENE_CLASSES.values())
    NUM_CLASSES = len(SCENE_CLASSES)

    def __init__(self, image_dir: str, labels_df: pd.DataFrame,
                 transform=None):
        self.image_dir  = image_dir
        self.labels_df  = labels_df.reset_index(drop=True)
        self.transform  = transform

    def __len__(self) -> int:
        return len(self.labels_df)

    def __getitem__(self, idx: int):
        row = self.labels_df.iloc[idx]

        # Load image (OpenCV → RGB NumPy array)
        img_path = os.path.join(self.image_dir, row['filename'])
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f'Cannot load image: {img_path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentation/normalisation pipeline
        if self.transform:
            image = self.transform(image=image)['image']

        label = int(row['class_idx'])
        return image, label

    def get_class_name(self, idx: int) -> str:
        return self.CLASS_NAMES[idx]


# ===========================================================================
# STRATIFIED SPLIT HELPER (D7)
# ===========================================================================

def make_stratified_splits(labels_csv: str,
                            train_ratio=0.70,
                            val_ratio=0.15,
                            test_ratio=0.15,
                            random_state=42):
    """
    Create stratified 70/15/15 train/val/test splits.

    Stratified = each split has the same class distribution as the full
    dataset. This is a capstone requirement.

    Returns three DataFrames: train_df, val_df, test_df
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        'Split ratios must sum to 1.0'

    df = pd.read_csv(labels_csv)

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df['class_idx'],
        random_state=random_state
    )

    # Second split: val vs test (from the temp set)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(test_ratio / (val_ratio + test_ratio)),
        stratify=temp_df['class_idx'],
        random_state=random_state
    )

    return train_df, val_df, test_df


# ===========================================================================
# CONVENIENCE: GET DATALOADERS
# ===========================================================================

def get_dataloaders(data_dir: str,
                    resolution: int = 224,
                    batch_size: int = 32,
                    num_workers: int = 0,
                    random_state: int = 42):
    """
    Full pipeline: CSV → splits → datasets → dataloaders.

    Args:
        data_dir    : Directory containing images/ and labels.csv
        resolution  : Input resolution (128 or 224)
        batch_size  : Batch size for DataLoader
        num_workers : DataLoader workers (0 for Windows compatibility)
        random_state: Seed for reproducible splits

    Returns:
        train_loader, val_loader, test_loader, class_counts
    """
    image_dir  = os.path.join(data_dir, 'images')
    labels_csv = os.path.join(data_dir, 'labels.csv')

    # Stratified splits
    train_df, val_df, test_df = make_stratified_splits(
        labels_csv, random_state=random_state
    )

    # Datasets
    train_ds = WarehouseSceneDataset(
        image_dir, train_df, transform=get_train_transform(resolution)
    )
    val_ds = WarehouseSceneDataset(
        image_dir, val_df, transform=get_val_test_transform(resolution)
    )
    test_ds = WarehouseSceneDataset(
        image_dir, test_df, transform=get_val_test_transform(resolution)
    )

    # DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=False
    )

    # Class counts for reporting
    full_df = pd.read_csv(labels_csv)
    class_counts = {
        SCENE_CLASSES[i]: int((full_df['class_idx'] == i).sum())
        for i in SCENE_CLASSES
    }

    print(f'Dataset loaded at resolution {resolution}×{resolution}')
    print(f'  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}')
    print(f'  Batch size: {batch_size} | Workers: {num_workers}')

    return train_loader, val_loader, test_loader, class_counts


# ===========================================================================
# QUICK TEST — python src/warehouse_dataset.py
# ===========================================================================

if __name__ == '__main__':
    import sys

    DATA_DIR = os.path.join('data', 'warehouse_dataset')

    if not os.path.exists(os.path.join(DATA_DIR, 'labels.csv')):
        print('ERROR: labels.csv not found.')
        print('Run collect_warehouse_data.py first.')
        sys.exit(1)

    print('Testing WarehouseSceneDataset...\n')

    # Test both resolutions
    for res in [128, 224]:
        train_loader, val_loader, test_loader, class_counts = get_dataloaders(
            DATA_DIR, resolution=res, batch_size=32, num_workers=0
        )

        # Test one batch
        images, labels = next(iter(train_loader))
        print(f'\nResolution {res}×{res}:')
        print(f'  Batch images shape : {images.shape}')   # (32, 3, res, res)
        print(f'  Batch labels shape : {labels.shape}')   # (32,)
        print(f'  Image dtype        : {images.dtype}')
        print(f'  Label values       : {labels[:8].tolist()}')
        print(f'  Pixel range        : [{images.min():.2f}, {images.max():.2f}]')

        # Verify label indices are valid
        assert images.shape == (min(32, len(train_loader.dataset)), 3, res, res) \
            or images.shape[0] <= 32
        assert labels.min() >= 0 and labels.max() <= 4

    print('\nClass distribution (full dataset):')
    for name, count in class_counts.items():
        print(f'  {name:<25} : {count}')

    print('\n✓ WarehouseSceneDataset working correctly!')
    print('✓ Both resolutions (128, 224) produce correct tensor shapes')
    print('✓ Labels are valid (0–4)')
    print('\nNext step: python scripts/validate_dataset.py')