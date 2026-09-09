#!/usr/bin/env python3
"""
Reorganize YOLO annotation data into train/ and val/ subfolders.

Input:
- <base>/images/ - images with UUID prefixes
- <base>/labels/ - matching YOLO .txt labels

Output:
- <base>/images/{train,val}/  and  <base>/labels/{train,val}/

The test split is created separately and held out; this helper only partitions
the remaining images into train/val (default 80/20). The random seed is fixed
for reproducibility.
"""

import os
import shutil
from pathlib import Path
import random

# Set seed for reproducibility
random.seed(42)

# Paths
base_dir = Path("data/input/yolo")
images_dir = base_dir / "images"
labels_dir = base_dir / "labels"

# Get all image files
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
print(f"Found {len(image_files)} images")

# Create new directory structure
for split in ['train', 'val']:
    (images_dir / split).mkdir(exist_ok=True)
    (labels_dir / split).mkdir(exist_ok=True)

# Shuffle and split (80/20)
random.shuffle(image_files)
split_idx = int(len(image_files) * 0.8)
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

print(f"Train: {len(train_files)} images")
print(f"Val: {len(val_files)} images")

# Move files
for img_file in train_files:
    # Get corresponding label file
    label_file = img_file.replace('.png', '.txt')
    
    # Move to train
    shutil.move(images_dir / img_file, images_dir / 'train' / img_file)
    shutil.move(labels_dir / label_file, labels_dir / 'train' / label_file)

for img_file in val_files:
    # Get corresponding label file
    label_file = img_file.replace('.png', '.txt')
    
    # Move to val
    shutil.move(images_dir / img_file, images_dir / 'val' / img_file)
    shutil.move(labels_dir / label_file, labels_dir / 'val' / label_file)

print("\n✅ Reorganization complete!")
print(f"Train: {len(os.listdir(images_dir / 'train'))} images")
print(f"Val: {len(os.listdir(images_dir / 'val'))} images")
