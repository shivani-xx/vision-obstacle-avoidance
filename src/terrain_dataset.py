# src/terrain_dataset.py

import os
import cv2
import pandas as pd
from torch.utils.data import Dataset


class TerrainDataset(Dataset):
    CLASSES = [
        'flat_ground',
        'uphill_slope',
        'rough_terrain',
        'hazard'
    ]

    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        self.df = pd.read_csv(label_file)

        self.class_to_idx = {
            c: i
            for i, c in enumerate(self.CLASSES)
        }

        print(f'Loaded {len(self.df)} samples')

        for cls in self.CLASSES:
            n = (self.df['terrain'] == cls).sum()
            print(f'  {cls}: {n} ({n/len(self.df):.1%})')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img = cv2.imread(
            os.path.join(
                self.image_dir,
                row['filename']
            )
        )

        if img is None:
            raise FileNotFoundError(
                f"Cannot load: {row['filename']}"
            )

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        if self.transform:
            img = self.transform(
                image=img
            )['image']

        return img, self.class_to_idx[row['terrain']]
    
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_terrain_transforms(target_size=224, is_train=True):

    if is_train:

        return A.Compose([

            A.Resize(target_size, target_size),

            # Horizontal flip is safe because terrain type
            # does not depend on left/right orientation
            A.HorizontalFlip(p=0.5),

            # Simulates different lighting conditions
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),

            # Simulates slight camera blur
            A.GaussianBlur(
                blur_limit=(3, 5),
                p=0.3
            ),

            # Simulates camera sensor noise
            A.GaussNoise(
                p=0.3
            ),

            # Small colour variation while preserving terrain identity
            A.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05,
                p=0.3
            ),

            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),

            ToTensorV2(),
        ])

    else:

        return A.Compose([

            A.Resize(target_size, target_size),

            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),

            ToTensorV2(),
        ])