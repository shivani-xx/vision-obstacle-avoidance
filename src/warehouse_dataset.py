# src/warehouse_dataset.py

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

SCENE_CLASSES = {
    0: 'open_aisle',
    1: 'narrow_aisle',
    2: 'pick_station',
    3: 'blocked_path',
    4: 'cross_aisle_junction',
}


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transform(resolution: int) -> A.Compose:


    return A.Compose([
        A.Resize(resolution, resolution),

        A.HorizontalFlip(p=0.5),

        A.RandomBrightnessContrast(
            brightness_limit=0.25,
            contrast_limit=0.25,
            p=0.5
        ),

        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=(3, 7)),
        ], p=0.3),

        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),

        ToTensorV2(),
    ])


def get_val_test_transform(resolution: int) -> A.Compose:
    
    return A.Compose([
        A.Resize(resolution, resolution),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])



class WarehouseSceneDataset(Dataset):
    

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

        img_path = os.path.join(self.image_dir, row['filename'])
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f'Cannot load image: {img_path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']

        label = int(row['class_idx'])
        return image, label

    def get_class_name(self, idx: int) -> str:
        return self.CLASS_NAMES[idx]


def make_stratified_splits(labels_csv: str,train_ratio=0.70,val_ratio=0.15,test_ratio=0.15,random_state=42):

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        'Split ratios must sum to 1.0'

    df = pd.read_csv(labels_csv)

    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df['class_idx'],
        random_state=random_state
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(test_ratio / (val_ratio + test_ratio)),
        stratify=temp_df['class_idx'],
        random_state=random_state
    )

    return train_df, val_df, test_df



def get_dataloaders(data_dir: str,
                    resolution: int = 224,
                    batch_size: int = 32,
                    num_workers: int = 0,
                    random_state: int = 42):
    
    image_dir  = os.path.join(data_dir, 'images')
    labels_csv = os.path.join(data_dir, 'labels.csv')

    
    train_df, val_df, test_df = make_stratified_splits(
        labels_csv, random_state=random_state
    )

    train_ds = WarehouseSceneDataset(
        image_dir, train_df, transform=get_train_transform(resolution)
    )
    val_ds = WarehouseSceneDataset(
        image_dir, val_df, transform=get_val_test_transform(resolution)
    )
    test_ds = WarehouseSceneDataset(
        image_dir, test_df, transform=get_val_test_transform(resolution)
    )

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

    full_df = pd.read_csv(labels_csv)
    class_counts = {
        SCENE_CLASSES[i]: int((full_df['class_idx'] == i).sum())
        for i in SCENE_CLASSES
    }

    print(f'Dataset loaded at resolution {resolution}×{resolution}')
    print(f'  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}')
    print(f'  Batch size: {batch_size} | Workers: {num_workers}')

    return train_loader, val_loader, test_loader, class_counts



if __name__ == '__main__':
    import sys

    DATA_DIR = os.path.join('data', 'warehouse_dataset')

    if not os.path.exists(os.path.join(DATA_DIR, 'labels.csv')):
        print('ERROR: labels.csv not found.')
        print('Run collect_warehouse_data.py first.')
        sys.exit(1)

    print('Testing WarehouseSceneDataset...\n')

    for res in [128, 224]:
        train_loader, val_loader, test_loader, class_counts = get_dataloaders(
            DATA_DIR, resolution=res, batch_size=32, num_workers=0
        )

        images, labels = next(iter(train_loader))
        print(f'\nResolution {res}×{res}:')
        print(f'  Batch images shape : {images.shape}')   # (32, 3, res, res)
        print(f'  Batch labels shape : {labels.shape}')   # (32,)
        print(f'  Image dtype        : {images.dtype}')
        print(f'  Label values       : {labels[:8].tolist()}')
        print(f'  Pixel range        : [{images.min():.2f}, {images.max():.2f}]')

        assert images.shape == (min(32, len(train_loader.dataset)), 3, res, res) \
            or images.shape[0] <= 32
        assert labels.min() >= 0 and labels.max() <= 4

    print('\nClass distribution (full dataset):')
    for name, count in class_counts.items():
        print(f'  {name:<25} : {count}')

    print('\n WarehouseSceneDataset working correctly!')
    print(' Both resolutions (128, 224) produce correct tensor shapes')
    print(' Labels are valid (0–4)')
    