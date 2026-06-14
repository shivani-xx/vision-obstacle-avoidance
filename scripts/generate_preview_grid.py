# scripts/generate_preview_grid.py
# Combines all 10 preview images into one comparison grid
# Run this after warehouse_environments.py has generated the preview images
#
# Usage:
#   python scripts/generate_preview_grid.py

import cv2
import numpy as np
import os

PREVIEW_DIR = 'preview_images'
OUTPUT_PATH = 'preview_images/COMPARISON_GRID.jpg'

CLASS_NAMES = [
    'Open Aisle',
    'Narrow Aisle',
    'Pick Station',
    'Blocked Path',
    'Cross-Aisle Junction'
]

SEEDS = ['seed_A', 'seed_B']
FILENAMES = [
    [f'class{i}_{name.lower().replace(" ", "_").replace("-", "_")}_{seed}.jpg'
     for seed in ['seed_A', 'seed_B']]
    for i, name in enumerate([
        'open_aisle', 'narrow_aisle', 'pick_station',
        'blocked_path', 'cross_aisle_junction'
    ])
]

IMG_SIZE   = 224
PADDING    = 12
LABEL_H    = 36
HEADER_H   = 50
BG_COLOR   = (30, 30, 30)       # Dark background
TEXT_COLOR = (255, 255, 255)
DIM_COLOR  = (180, 180, 180)
ACCENT     = (249, 115, 22)     # Orange accent (matches Purple AI Labs style)

COLS = 2   # seed A | seed B
ROWS = 5   # one per class

total_w = COLS * IMG_SIZE + (COLS + 1) * PADDING
total_h = HEADER_H + ROWS * (IMG_SIZE + LABEL_H + PADDING) + PADDING

canvas = np.full((total_h, total_w, 3), BG_COLOR, dtype=np.uint8)

# --- Top header bar ---
cv2.rectangle(canvas, (0, 0), (total_w, HEADER_H), (26, 60, 110), -1)
cv2.putText(canvas, 'D3: Scene Environment Comparison Grid',
            (PADDING, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, TEXT_COLOR, 1, cv2.LINE_AA)

# --- Column headers ---
for col, label in enumerate(['Seed A  (seed=42)', 'Seed B  (seed=137)']):
    x = PADDING + col * (IMG_SIZE + PADDING)
    cv2.putText(canvas, label,
                (x + 4, HEADER_H + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, DIM_COLOR, 1, cv2.LINE_AA)

# --- Place images ---
missing = []
for row, (class_name, file_pair) in enumerate(zip(CLASS_NAMES, FILENAMES)):
    y_top = HEADER_H + 28 + row * (IMG_SIZE + LABEL_H + PADDING)

    # Class label on left margin
    label_y = y_top + IMG_SIZE // 2 + 6
    cv2.putText(canvas,
                f'[{row}] {class_name}',
                (PADDING, label_y - IMG_SIZE // 2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, ACCENT, 1, cv2.LINE_AA)

    for col, fname in enumerate(file_pair):
        fpath = os.path.join(PREVIEW_DIR, fname)
        x = PADDING + col * (IMG_SIZE + PADDING)

        if not os.path.exists(fpath):
            missing.append(fpath)
            # Draw a red placeholder
            cv2.rectangle(canvas, (x, y_top), (x + IMG_SIZE, y_top + IMG_SIZE), (60, 0, 0), -1)
            cv2.putText(canvas, 'MISSING', (x + 60, y_top + IMG_SIZE // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
            continue

        img = cv2.imread(fpath)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        canvas[y_top: y_top + IMG_SIZE, x: x + IMG_SIZE] = img

        # Thin border around each image
        cv2.rectangle(canvas, (x, y_top), (x + IMG_SIZE - 1, y_top + IMG_SIZE - 1),
                      (80, 80, 80), 1)

        # Seed label below image
        seed_label = 'Seed A' if col == 0 else 'Seed B'
        cv2.putText(canvas, seed_label,
                    (x + 4, y_top + IMG_SIZE + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, DIM_COLOR, 1, cv2.LINE_AA)

# --- Save ---
cv2.imwrite(OUTPUT_PATH, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])

if missing:
    print(f"WARNING: {len(missing)} image(s) not found:")
    for m in missing:
        print(f"  {m}")
    print("Make sure you ran warehouse_environments.py first!")
else:
    print(f"Grid saved to: {OUTPUT_PATH}")
    print("Open preview_images/COMPARISON_GRID.jpg and send a screenshot!")