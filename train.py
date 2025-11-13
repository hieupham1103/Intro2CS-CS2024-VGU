#!/usr/bin/env python3
"""
YOLOv8 Training Script for Drone Detection
Supports RGB and IR datasets
"""

import argparse
import os
import yaml
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv8 model for drone detection')
    parser.add_argument('--set', type=str, required=True, choices=['RGB', 'IR'],
                        help='Dataset to use (RGB or IR)')
    parser.add_argument('--output', type=str, default='outputs',
                        help='Output directory for training results')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to training config YAML file (default: configs/train_{set}.yml)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='Base model to use (default: yolov8n.pt)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config file)')
    parser.add_argument('--batch', type=int, default=None,
                        help='Batch size (overrides config file)')
    parser.add_argument('--imgsz', type=int, default=None,
                        help='Image size (overrides config file)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (e.g., 0, 1, cpu) (overrides config file)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    
    return parser.parse_args()


def load_training_config(config_path):
    """Load training configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def main():
    args = parse_args()
    
    if args.config is None:
        args.config = f'configs/train_{args.set}.yml'
    
    print(f"Loading configuration from: {args.config}")
    config = load_training_config(args.config)
    
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch is not None:
        config['batch'] = args.batch
    if args.imgsz is not None:
        config['imgsz'] = args.imgsz
    if args.device is not None:
        config['device'] = args.device
    
    config['project'] = args.output
    config['name'] = f'{args.set}_training'
    
    data_config = config.get('data', f'configs/data_{args.set}.yml')
    
    if not os.path.exists(data_config):
        raise FileNotFoundError(f"Data config file not found: {data_config}")
    
    print(f"\n{'='*60}")
    print(f"Training YOLOv8n on {args.set} dataset")
    print(f"{'='*60}")
    print(f"Base model: {args.model}")
    print(f"Data config: {data_config}")
    print(f"Epochs: {config.get('epochs', 'default')}")
    print(f"Batch size: {config.get('batch', 'default')}")
    print(f"Image size: {config.get('imgsz', 'default')}")
    print(f"Device: {config.get('device', 'default')}")
    print(f"Output: {args.output}/{args.set}_training")
    print(f"{'='*60}\n")
    
    model = YOLO(args.model)
    
    results = model.train(
        data=data_config,
        epochs=config.get('epochs', 100),
        batch=config.get('batch', 16),
        imgsz=config.get('imgsz', 640),
        device=config.get('device', 0),
        project=config['project'],
        name=config['name'],
        exist_ok=config.get('exist_ok', False),
        pretrained=config.get('pretrained', True),
        optimizer=config.get('optimizer', 'auto'),
        verbose=config.get('verbose', True),
        seed=config.get('seed', 0),
        deterministic=config.get('deterministic', True),
        single_cls=config.get('single_cls', False),
        rect=config.get('rect', False),
        cos_lr=config.get('cos_lr', False),
        close_mosaic=config.get('close_mosaic', 10),
        resume=args.resume,
        amp=config.get('amp', True),
        fraction=config.get('fraction', 1.0),
        profile=config.get('profile', False),
        freeze=config.get('freeze', None),
        lr0=config.get('lr0', 0.01),
        lrf=config.get('lrf', 0.01),
        momentum=config.get('momentum', 0.937),
        weight_decay=config.get('weight_decay', 0.0005),
        warmup_epochs=config.get('warmup_epochs', 3.0),
        warmup_momentum=config.get('warmup_momentum', 0.8),
        warmup_bias_lr=config.get('warmup_bias_lr', 0.1),
        box=config.get('box', 7.5),
        cls=config.get('cls', 0.5),
        dfl=config.get('dfl', 1.5),
        pose=config.get('pose', 12.0),
        kobj=config.get('kobj', 1.0),
        label_smoothing=config.get('label_smoothing', 0.0),
        nbs=config.get('nbs', 64),
        hsv_h=config.get('hsv_h', 0.015),
        hsv_s=config.get('hsv_s', 0.7),
        hsv_v=config.get('hsv_v', 0.4),
        degrees=config.get('degrees', 0.0),
        translate=config.get('translate', 0.1),
        scale=config.get('scale', 0.5),
        shear=config.get('shear', 0.0),
        perspective=config.get('perspective', 0.0),
        flipud=config.get('flipud', 0.0),
        fliplr=config.get('fliplr', 0.5),
        mosaic=config.get('mosaic', 1.0),
        mixup=config.get('mixup', 0.0),
        copy_paste=config.get('copy_paste', 0.0),
        patience=config.get('patience', 50),
        save=config.get('save', True),
        save_period=config.get('save_period', -1),
        cache=config.get('cache', False),
        plots=config.get('plots', True),
        overlap_mask=config.get('overlap_mask', True),
        mask_ratio=config.get('mask_ratio', 4),
    )
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Results saved to: {args.output}/{args.set}_training")
    print(f"Best model: {args.output}/{args.set}_training/weights/best.pt")
    print(f"{'='*60}\n")
    
    return results


if __name__ == '__main__':
    main()
