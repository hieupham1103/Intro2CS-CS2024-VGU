#!/usr/bin/env python3
"""
Video Visualization Tool for Drone Detection

This script creates visualization videos showing:
1. Ground truth bounding boxes
2. Model predictions (optional)
3. Side-by-side comparison (optional)

Usage:
    # Visualize ground truth only
    python visualize_videos.py --data-dir data/RGB_video --split test
    
    # Visualize with model predictions
    python visualize_videos.py --data-dir data/RGB_video --split test --model-path checkpoints/rgb.pt
    
    # Visualize OOD test set
    python visualize_videos.py --data-dir data/RGB_TEST_OD --split test --model-path checkpoints/rgb.pt
    
    # Side-by-side comparison
    python visualize_videos.py --data-dir data/RGB_video --split test --model-path checkpoints/rgb.pt --side-by-side
"""

import os
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Class names and colors
CLASS_NAMES = {0: 'BIRD', 1: 'DRONE'}
GT_COLORS = {0: (0, 255, 0), 1: (0, 255, 0)}  # Green for ground truth
PRED_COLORS = {0: (255, 0, 0), 1: (0, 0, 255)}  # Blue for bird, Red for drone predictions


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize videos with detection results')
    parser.add_argument('--data-dir', type=str, default='data/RGB_video',
                        help='Path to data directory (containing videos/ and labels/)')
    parser.add_argument('--split', type=str, choices=['train', 'test'], default='test',
                        help='Data split to visualize (default: test)')
    parser.add_argument('--output-dir', type=str, default='outputs/visualizations',
                        help='Output directory for visualization videos')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to model checkpoint for predictions (optional)')
    parser.add_argument('--model-type', type=str, choices=['yolo', 'multiscale'],
                        default='yolo', help='Type of model (default: yolo)')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                        help='Confidence threshold for detection (default: 0.25)')
    parser.add_argument('--iou-threshold', type=float, default=0.45,
                        help='IoU threshold for NMS (default: 0.45)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on (cuda/cpu)')
    parser.add_argument('--side-by-side', action='store_true',
                        help='Create side-by-side comparison (GT left, Pred right)')
    parser.add_argument('--gt-only', action='store_true',
                        help='Only show ground truth (no predictions)')
    parser.add_argument('--pred-only', action='store_true',
                        help='Only show predictions (no ground truth)')
    parser.add_argument('--video-name', type=str, default=None,
                        help='Specific video name to visualize (without extension)')
    parser.add_argument('--fps', type=int, default=None,
                        help='Output FPS (default: same as input)')
    parser.add_argument('--show-info', action='store_true',
                        help='Show frame info and statistics on video')
    parser.add_argument('--box-thickness', type=int, default=2,
                        help='Bounding box line thickness (default: 2)')
    parser.add_argument('--font-scale', type=float, default=0.6,
                        help='Font scale for labels (default: 0.6)')
    
    return parser.parse_args()


def load_video_labels(label_path):
    """
    Load video labels from text file
    Format: frame_id class_id center_x center_y width height (normalized 0-1)
    Returns: dict mapping frame_id to list of annotations
    """
    annotations = {}
    
    if not os.path.exists(label_path):
        print(f"Warning: Label file not found: {label_path}")
        return annotations
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 6:
                continue
            
            try:
                frame_id = int(parts[0])
                class_id = int(parts[1])
                center_x = float(parts[2])
                center_y = float(parts[3])
                width = float(parts[4])
                height = float(parts[5])
                
                if frame_id not in annotations:
                    annotations[frame_id] = []
                
                annotations[frame_id].append({
                    'class_id': class_id,
                    'center_x': center_x,
                    'center_y': center_y,
                    'width': width,
                    'height': height
                })
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse line: {line}")
                continue
    
    return annotations


def normalized_to_pixel(box, img_width, img_height):
    """Convert normalized box (center_x, center_y, width, height) to pixel coordinates (x1, y1, x2, y2)"""
    center_x = box['center_x'] * img_width
    center_y = box['center_y'] * img_height
    width = box['width'] * img_width
    height = box['height'] * img_height
    
    x1 = int(center_x - width / 2)
    y1 = int(center_y - height / 2)
    x2 = int(center_x + width / 2)
    y2 = int(center_y + height / 2)
    
    return x1, y1, x2, y2


def draw_boxes(frame, boxes, colors, is_gt=True, thickness=2, font_scale=0.6, show_conf=True):
    """
    Draw bounding boxes on frame
    
    Args:
        frame: Image to draw on
        boxes: List of boxes (different format for GT vs predictions)
        colors: Color dict for each class
        is_gt: Whether these are ground truth boxes
        thickness: Line thickness
        font_scale: Font scale for labels
        show_conf: Show confidence score (only for predictions)
    """
    frame = frame.copy()
    img_height, img_width = frame.shape[:2]
    
    for box in boxes:
        if is_gt:
            # Ground truth format: dict with normalized coordinates
            x1, y1, x2, y2 = normalized_to_pixel(box, img_width, img_height)
            class_id = box['class_id']
            conf = None
        else:
            # Prediction format: dict with pixel coordinates
            x1, y1, x2, y2 = box['box']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_id = int(box['class_id'])
            conf = box.get('conf', None)
        
        # Get color
        color = colors.get(class_id, (255, 255, 255))
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Create label
        class_name = CLASS_NAMES.get(class_id, f'Class_{class_id}')
        if conf is not None and show_conf:
            label = f"{class_name}: {conf:.2f}"
        else:
            label = class_name
        
        # Add prefix for GT/Pred
        if is_gt:
            label = f"GT: {label}"
        else:
            label = f"Pred: {label}"
        
        # Draw label background
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        
        # Position label above box if possible
        label_y = y1 - 5 if y1 > label_height + 10 else y2 + label_height + 5
        label_x = x1
        
        cv2.rectangle(
            frame,
            (label_x, label_y - label_height - baseline),
            (label_x + label_width, label_y + baseline),
            color, -1
        )
        
        # Draw label text
        cv2.putText(
            frame, label, (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA
        )
    
    return frame


def draw_info_panel(frame, frame_id, total_frames, gt_count, pred_count, video_name):
    """Draw information panel on the frame"""
    frame = frame.copy()
    img_height, img_width = frame.shape[:2]
    
    # Create semi-transparent overlay for info panel
    panel_height = 80
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (img_width, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Draw info text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)
    
    # Video name
    cv2.putText(frame, f"Video: {video_name}", (10, 25), font, font_scale, color, 1, cv2.LINE_AA)
    
    # Frame info
    cv2.putText(frame, f"Frame: {frame_id}/{total_frames}", (10, 50), font, font_scale, color, 1, cv2.LINE_AA)
    
    # Detection counts
    info_x = img_width // 2
    cv2.putText(frame, f"GT: {gt_count}", (info_x, 25), font, font_scale, GT_COLORS[0], 1, cv2.LINE_AA)
    cv2.putText(frame, f"Pred: {pred_count}", (info_x, 50), font, font_scale, PRED_COLORS[1], 1, cv2.LINE_AA)
    
    return frame


def load_model(model_path, model_type, conf_threshold, iou_threshold, device):
    """Load detection model"""
    if model_type == 'yolo':
        from models.model import DetectionModel
        model = DetectionModel(
            model_path=model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=device
        )
    elif model_type == 'multiscale':
        from models.multiscale_model import MultiscaleDetectionModel
        model = MultiscaleDetectionModel(
            model_path=model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=device
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def get_predictions(model, frame):
    """Get model predictions for a single frame"""
    detections = model.image_detect(frame)
    
    predictions = []
    for i in range(len(detections['boxes'])):
        box = detections['boxes'][i]
        if hasattr(box, 'numpy'):
            box = box.numpy()
        
        score = detections['scores'][i]
        if hasattr(score, 'item'):
            score = score.item()
        
        label = detections['labels'][i]
        if hasattr(label, 'item'):
            label = label.item()
        
        predictions.append({
            'box': (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            'conf': float(score),
            'class_id': int(label)
        })
    
    return predictions


def visualize_video(video_path, label_path, output_path, model=None, args=None):
    """
    Create visualization video
    
    Args:
        video_path: Path to input video
        label_path: Path to label file
        output_path: Path to output video
        model: Detection model (optional)
        args: Command line arguments
    """
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if args.fps is not None:
        fps = args.fps
    
    # Load ground truth labels
    gt_annotations = load_video_labels(label_path)
    
    # Setup video writer
    output_width = width * 2 if args.side_by_side else width
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, height))
    
    if not out.isOpened():
        print(f"Error: Could not create output video: {output_path}")
        cap.release()
        return False
    
    video_name = Path(video_path).stem
    
    frame_id = 0
    pbar = tqdm(total=total_frames, desc=f"Processing {video_name}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get ground truth for this frame
        gt_boxes = gt_annotations.get(frame_id, [])
        
        # Get predictions if model is provided
        pred_boxes = []
        if model is not None and not args.gt_only:
            pred_boxes = get_predictions(model, frame)
        
        if args.side_by_side:
            # Create side-by-side view
            frame_gt = frame.copy()
            frame_pred = frame.copy()
            
            # Draw ground truth on left
            if not args.pred_only:
                frame_gt = draw_boxes(
                    frame_gt, gt_boxes, GT_COLORS, is_gt=True,
                    thickness=args.box_thickness, font_scale=args.font_scale
                )
            
            # Draw predictions on right
            if pred_boxes:
                frame_pred = draw_boxes(
                    frame_pred, pred_boxes, PRED_COLORS, is_gt=False,
                    thickness=args.box_thickness, font_scale=args.font_scale
                )
            
            # Add labels
            cv2.putText(frame_gt, "Ground Truth", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame_pred, "Predictions", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Combine frames
            combined = np.hstack([frame_gt, frame_pred])
            
            if args.show_info:
                # Draw info on combined frame
                cv2.putText(combined, f"Frame: {frame_id}/{total_frames}", (10, height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            out.write(combined)
        else:
            # Single view with both GT and predictions
            vis_frame = frame.copy()
            
            # Draw ground truth
            if not args.pred_only:
                vis_frame = draw_boxes(
                    vis_frame, gt_boxes, GT_COLORS, is_gt=True,
                    thickness=args.box_thickness, font_scale=args.font_scale
                )
            
            # Draw predictions
            if pred_boxes and not args.gt_only:
                vis_frame = draw_boxes(
                    vis_frame, pred_boxes, PRED_COLORS, is_gt=False,
                    thickness=args.box_thickness, font_scale=args.font_scale
                )
            
            # Add info panel
            if args.show_info:
                vis_frame = draw_info_panel(
                    vis_frame, frame_id, total_frames,
                    len(gt_boxes), len(pred_boxes), video_name
                )
            
            out.write(vis_frame)
        
        frame_id += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    return True


def main():
    args = parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    video_dir = data_dir / 'videos' / args.split
    label_dir = data_dir / 'labels' / args.split
    output_dir = Path(args.output_dir) / data_dir.name / args.split
    
    # Validate paths
    if not video_dir.exists():
        print(f"Error: Video directory not found: {video_dir}")
        return
    
    if not label_dir.exists():
        print(f"Error: Label directory not found: {label_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model if specified
    model = None
    if args.model_path and not args.gt_only:
        print(f"Loading model from {args.model_path}...")
        model = load_model(
            args.model_path, args.model_type,
            args.conf_threshold, args.iou_threshold, args.device
        )
        print("Model loaded successfully!")
    
    # Get video files
    video_files = list(video_dir.glob('*.mp4'))
    
    if args.video_name:
        # Filter to specific video
        video_files = [v for v in video_files if v.stem == args.video_name]
        if not video_files:
            print(f"Error: Video not found: {args.video_name}")
            return
    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return
    
    print(f"\n{'='*60}")
    print("Video Visualization Tool")
    print(f"{'='*60}")
    print(f"Data directory:  {data_dir}")
    print(f"Split:           {args.split}")
    print(f"Videos found:    {len(video_files)}")
    print(f"Output directory: {output_dir}")
    print(f"Model:           {args.model_path if args.model_path else 'None (GT only)'}")
    print(f"Side-by-side:    {args.side_by_side}")
    print(f"Show info:       {args.show_info}")
    print(f"{'='*60}\n")
    
    # Process videos
    success_count = 0
    for video_path in video_files:
        video_name = video_path.stem
        label_path = label_dir / f"{video_name}.txt"
        
        # Create output filename
        suffix = ""
        if args.side_by_side:
            suffix = "_sbs"
        elif args.gt_only:
            suffix = "_gt"
        elif args.pred_only:
            suffix = "_pred"
        elif model is not None:
            suffix = "_comparison"
        
        output_path = output_dir / f"{video_name}{suffix}.mp4"
        
        print(f"\nProcessing: {video_name}")
        
        if visualize_video(video_path, label_path, output_path, model, args):
            print(f"  Output: {output_path}")
            success_count += 1
        else:
            print(f"  Failed!")
    
    print(f"\n{'='*60}")
    print(f"Completed: {success_count}/{len(video_files)} videos")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
