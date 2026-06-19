# scripts/validate_dataset.py

import os
import sys
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from warehouse_dataset import (
    SCENE_CLASSES, make_stratified_splits, get_dataloaders
)

DATA_DIR    = os.path.join('data', 'warehouse_dataset')
IMG_DIR     = os.path.join(DATA_DIR, 'images')
LABELS_CSV  = os.path.join(DATA_DIR, 'labels.csv')
RESULTS_DIR = 'results'

os.makedirs(RESULTS_DIR, exist_ok=True)


CLASS_COLORS = {
    0: '#F59E0B',   # Yellow  — Open Aisle (yellow floor tape)
    1: '#6B7280',   # Grey    — Narrow Aisle (grey shelves dominate)
    2: '#F97316',   # Orange  — Pick Station (orange marker)
    3: '#EF4444',   # Red     — Blocked Path (red tape)
    4: '#3B82F6',   # Blue    — Cross-Aisle Junction (white floor cross + open)
}


def check_dataset_exists():
    if not os.path.exists(LABELS_CSV):
        print('ERROR: labels.csv not found at', LABELS_CSV)
        print('Run scripts/collect_warehouse_data.py first.')
        sys.exit(1)


def validate_basic(df):
    
    print('--- Basic Validation ---')
    print(f'Total frames : {len(df)}')
    print(f'Columns      : {list(df.columns)}')


    missing = 0
    for fname in df['filename']:
        if not os.path.exists(os.path.join(IMG_DIR, fname)):
            missing += 1
    print(f'Missing images: {missing}')

    invalid = df[~df['class_idx'].isin(SCENE_CLASSES.keys())]
    print(f'Invalid labels: {len(invalid)}')

    if missing == 0 and len(invalid) == 0:
        print(' All files present and labels valid\n')
    else:
        print(' Issues found — fix before training\n')


def plot_class_distribution(df):
    
    counts = [int((df['class_idx'] == i).sum()) for i in SCENE_CLASSES]
    names  = [SCENE_CLASSES[i].replace('_', '\n') for i in SCENE_CLASSES]
    colors = [CLASS_COLORS[i] for i in SCENE_CLASSES]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, counts, color=colors, edgecolor='white', linewidth=0.8)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 15,
                f'{count}', ha='center', va='bottom',
                fontsize=11, fontweight='bold', color='#1F2937')

    mean_count = np.mean(counts)
    ax.axhline(mean_count, color='#6B7280', linestyle='--',
               linewidth=1.5, label=f'Mean ({mean_count:.0f})')
    ax.axhline(mean_count * 0.85, color='#EF4444', linestyle=':',
               linewidth=1.2, label='-15% threshold')
    ax.axhline(mean_count * 1.15, color='#EF4444', linestyle=':',
               linewidth=1.2, label='+15% threshold')

    ax.set_title('D7: Class Distribution — Warehouse Scene Dataset',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Number of Frames', fontsize=12)
    ax.set_ylim(0, max(counts) * 1.2)
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, 'D7_class_distribution.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


def plot_sample_grid(df):
    
    SAMPLES_PER_CLASS = 4
    fig, axes = plt.subplots(
        len(SCENE_CLASSES), SAMPLES_PER_CLASS,
        figsize=(SAMPLES_PER_CLASS * 3, len(SCENE_CLASSES) * 3)
    )
    fig.suptitle('D7: Sample Images per Scene Class (4 random per class)',
                 fontsize=13, fontweight='bold', y=1.01)

    for row, (class_idx, class_name) in enumerate(SCENE_CLASSES.items()):
        class_df = df[df['class_idx'] == class_idx]
        samples  = class_df.sample(min(SAMPLES_PER_CLASS, len(class_df)),
                                   random_state=42 + row)

        for col, (_, sample_row) in enumerate(samples.iterrows()):
            ax  = axes[row][col]
            img = cv2.imread(os.path.join(IMG_DIR, sample_row['filename']))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)

            
            if col == 0:
                ax.set_ylabel(
                    f'[{class_idx}] {class_name.replace("_", " ").title()}',
                    fontsize=9, fontweight='bold',
                    color=CLASS_COLORS[class_idx]
                )
            ax.axis('off')

            
            for spine in ax.spines.values():
                spine.set_edgecolor(CLASS_COLORS[class_idx])
                spine.set_linewidth(2)
                spine.set_visible(True)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, 'D7_sample_grid.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


def validate_splits(df):
    
    train_df, val_df, test_df = make_stratified_splits(LABELS_CSV)

    total = len(df)
    print('\n--- Split Validation (70/15/15 stratified) ---')
    print(f'{"Split":<8} {"Total":>7}  {"Ratio":>7}  '
          + '  '.join([f'cls{i}' for i in SCENE_CLASSES]))
    print('-' * 70)

    split_summary = []
    for split_name, split_df in [('Train', train_df),
                                  ('Val',   val_df),
                                  ('Test',  test_df)]:
        counts_per_class = [int((split_df['class_idx'] == i).sum())
                            for i in SCENE_CLASSES]
        ratio = len(split_df) / total
        row = (f'{split_name:<8} {len(split_df):>7}  {ratio:>7.1%}  '
               + '  '.join([f'{c:>5}' for c in counts_per_class]))
        print(row)
        split_summary.append({
            'split': split_name,
            'total': len(split_df),
            'ratio': f'{ratio:.1%}',
            **{SCENE_CLASSES[i]: counts_per_class[i] for i in SCENE_CLASSES}
        })

    summary_df = pd.DataFrame(split_summary)
    out = os.path.join(RESULTS_DIR, 'D7_split_summary.csv')
    summary_df.to_csv(out, index=False)
    print(f'\nSaved: {out}')

    train_ratio = len(train_df) / total
    val_ratio   = len(val_df)   / total
    test_ratio  = len(test_df)  / total
    print(f'\nActual ratios: train={train_ratio:.1%}, '
          f'val={val_ratio:.1%}, test={test_ratio:.1%}')

    if (0.68 <= train_ratio <= 0.72 and
            0.13 <= val_ratio <= 0.17 and
            0.13 <= test_ratio <= 0.17):
        print(' Splits are within acceptable range of 70/15/15')
    else:
        print(' Splits deviate from 70/15/15 — check make_stratified_splits()')


def validate_dataloader():
    print('\n--- DataLoader Validation ---')
    for res in [128, 224]:
        try:
            train_loader, val_loader, test_loader, _ = get_dataloaders(
                DATA_DIR, resolution=res, batch_size=32, num_workers=0
            )
            images, labels = next(iter(train_loader))
            assert images.shape[1] == 3
            assert images.shape[2] == res
            assert images.shape[3] == res
            assert labels.min() >= 0
            assert labels.max() <= 4
            print(f'   Resolution {res}×{res}: batch shape {tuple(images.shape)} OK')
        except Exception as e:
            print(f'   Resolution {res}×{res}: ERROR — {e}')


def print_augmentation_decisions():
    
    print('\n--- Augmentation Decisions (for report) ---')
    decisions = [
        ('HorizontalFlip',             'INCLUDED', 'Aisles look same mirrored'),
        ('RandomBrightnessContrast',   'INCLUDED', 'Warehouse lighting varies (fluorescents, shadows)'),
        ('GaussianBlur / MotionBlur',  'INCLUDED', 'Camera motion blur during robot movement'),
        ('GaussNoise',                 'INCLUDED', 'Sensor noise in cheap warehouse cameras'),
        ('CoarseDropout',              'INCLUDED', 'Partial occlusion from passing objects'),
        ('ColorJitter',                'EXCLUDED', 'Color is primary class discriminator — yellow/red/orange tape'),
        ('HueSaturationValue',         'EXCLUDED', 'Same reason as ColorJitter — would corrupt color signals'),
        ('RandomCrop',                 'EXCLUDED', 'Scene type often inferred from overall layout — cropping risks removing key cues'),
    ]
    for aug, decision, reason in decisions:
        symbol = '✓' if decision == 'INCLUDED' else '✗'
        print(f'  {symbol} {aug:<30} {decision:<10} — {reason}')



if __name__ == '__main__':
    check_dataset_exists()
    df = pd.read_csv(LABELS_CSV)

    print('=' * 60)
    print('D7: Data Validation — Warehouse Scene Dataset')
    print('=' * 60)

    validate_basic(df)

    print('--- Class Distribution ---')
    for i, name in SCENE_CLASSES.items():
        count = int((df['class_idx'] == i).sum())
        pct   = count / len(df) * 100
        bar   = '█' * int(pct / 2)
        print(f'  [{i}] {name:<25} {count:>5} ({pct:.1f}%)  {bar}')

    print('\nGenerating charts...')
    plot_class_distribution(df)
    plot_sample_grid(df)
    validate_splits(df)
    validate_dataloader()
    print_augmentation_decisions()

    print('\n' + '=' * 60)
    print('D7 VALIDATION COMPLETE')
    print(f'Charts saved to: {RESULTS_DIR}/')
    