#!/usr/bin/env python3
"""
YOLO Training Script

Fine-tunes YOLOv8s (small variant, to limit overfitting on a small training set)
from COCO-pretrained weights. Severe class imbalance is addressed through a
data-augmentation strategy: Mosaic augmentation to enlarge the effective
dataset, combined with brightness (HSV value) variation to simulate scan
conditions.

Supports:
- Pretrained COCO weights (toggle)
- Cosine-annealed learning rate (lr0 = 0.01)
- Configurable training parameters
"""

import os
import yaml
from ultralytics import YOLO
from pathlib import Path
import argparse


def load_config(config_path='configs/pipeline_config.yaml'):
    """Load training configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_yolo(config, use_pretrained=None):
    """
    Fine-tune the YOLO detector.

    Args:
        config: Configuration dictionary
        use_pretrained: Override config setting (True/False/None)
    """
    
    training_config = config['training']['yolo']
    aug_config = training_config.get('augmentation', {})
    
    # Override if specified
    if use_pretrained is not None:
        training_config['use_pretrained'] = use_pretrained
    
    # Initialize model
    if training_config['use_pretrained']:
        # Use pretrained weights from data/intermediate/models
        pretrained_path = training_config['pretrained_model']
        # If it's just a filename, prepend the models directory
        if not os.path.isabs(pretrained_path) and not pretrained_path.startswith('data/'):
            pretrained_path = f"data/intermediate/models/{pretrained_path}"
        
        print(f"✓ Using pretrained weights: {pretrained_path}")
        model = YOLO(pretrained_path)
    else:
        print("✓ Training from scratch (vanilla YOLO)")
        model = YOLO('yolov8s.yaml')  # Architecture only, no pretrained weights
    
    lr0 = training_config.get('lr0', 0.01)
    cos_lr = training_config.get('cos_lr', True)

    print(f"\nTraining configuration:")
    print(f"  Epochs: {training_config['epochs']}")
    print(f"  Batch size: {training_config['batch_size']}")
    print(f"  Image size: {training_config['image_size']}")
    print(f"  Device: {training_config['device']}")
    print(f"  Initial LR (lr0): {lr0}")
    print(f"  Cosine LR schedule: {cos_lr}")
    print(f"\nAugmentation strategy:")
    print(f"  Brightness (hsv_v): {aug_config.get('hsv_v', 0.3)}")
    print(f"  Mosaic: {aug_config.get('mosaic', 1.0)}")
    print(f"  Geometric transforms: DISABLED (tight bounding boxes)")
    print(f"\nStarting training...\n")
    
    # Class imbalance is addressed through the data-augmentation strategy
    # (Mosaic + brightness variation) configured below.

    # Train with augmentation
    results = model.train(
        data='data/input/yolo/yolo_data.yaml',
        epochs=training_config['epochs'],
        imgsz=training_config['image_size'],
        batch=training_config['batch_size'],
        patience=training_config['patience'],
        device=training_config['device'],
        lr0=lr0,
        cos_lr=cos_lr,

        # Augmentation parameters (conservative strategy)
        hsv_h=aug_config.get('hsv_h', 0.0),
        hsv_s=aug_config.get('hsv_s', 0.0),
        hsv_v=aug_config.get('hsv_v', 0.3),
        degrees=aug_config.get('degrees', 0.0),
        translate=aug_config.get('translate', 0.0),
        scale=aug_config.get('scale', 0.0),
        shear=aug_config.get('shear', 0.0),
        perspective=aug_config.get('perspective', 0.0),
        flipud=aug_config.get('flipud', 0.0),
        fliplr=aug_config.get('fliplr', 0.0),
        mosaic=aug_config.get('mosaic', 1.0),
        mixup=aug_config.get('mixup', 0.0),
        copy_paste=aug_config.get('copy_paste', 0.0),
        
        project='runs/detect',
        name='train',
        exist_ok=True,
        verbose=True
    )
    
    print("TRAINING COMPLETE!")
    print(f"Best model saved to: runs/detect/train/weights/best.pt")
    print(f"Last model saved to: runs/detect/train/weights/last.pt")
    print(f"\nTo use this model in the pipeline:")
    print(f"  The config already points to: runs/detect/train/weights/best.pt")
    print(f"\nTo run detection:")
    print(f"  python -m src.main --stage 1")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Fine-tune the YOLO element detector')
    parser.add_argument(
        '--pretrained',
        type=lambda x: x.lower() == 'true',
        default=None,
        help='Use pretrained weights (true/false, overrides config)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/pipeline_config.yaml',
        help='Path to config file'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Train
    train_yolo(config, use_pretrained=args.pretrained)


if __name__ == '__main__':
    main()
