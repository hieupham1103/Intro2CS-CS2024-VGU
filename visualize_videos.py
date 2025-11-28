#!/usr/bin/env python3
"""
Video Visualization Tool for Drone Detection

This script creates visualization videos showing:
1. Ground truth bounding boxes
2. Model predictions (optional)
3. Side-by-side comparison (GT vs Pred, or YOLO vs Multiscale)

Usage:
    # Visualize ground truth only
    python visualize_videos.py --data-dir data/RGB_video --split test --gt-only
    
    # Visualize with model predictions
    python visualize_videos.py --data-dir data/RGB_video --split test --model-path checkpoints/rgb.pt
    
    # Side-by-side: GT vs Predictions
    python visualize_videos.py --data-dir data/RGB_video --split test --model-path checkpoints/rgb.pt --side-by-side
    
    # Side-by-side: YOLO vs Multiscale comparison
    python visualize_videos.py --data-dir data/RGB_video --split test \
        --model-path checkpoints/rgb.pt \
        --model-path-2 checkpoints/rgb.pt --model-type-2 multiscale \
        --compare-models --pred-only
"""

import os
import argparse
import cv2
import numpy as np
import subprocess
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
    
    # Model 1 (primary model)
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to first model checkpoint (optional)')
    parser.add_argument('--model-type', type=str, choices=['yolo', 'multiscale'],
                        default='yolo', help='Type of first model (default: yolo)')
    
    # Model 2 (for comparison)
    parser.add_argument('--model-path-2', type=str, default=None,
                        help='Path to second model checkpoint for comparison')
    parser.add_argument('--model-type-2', type=str, choices=['yolo', 'multiscale'],
                        default='multiscale', help='Type of second model (default: multiscale)')
    
    # Detection parameters
    parser.add_argument('--conf-threshold', type=float, default=0.3,
                        help='Confidence threshold for detection (default: 0.25)')
    parser.add_argument('--iou-threshold', type=float, default=0.45,
                        help='IoU threshold for NMS (default: 0.45)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on (cuda/cpu)')
    
    # Visualization modes
    parser.add_argument('--side-by-side', action='store_true',
                        help='Create side-by-side comparison (GT left, Pred right)')
    parser.add_argument('--compare-models', action='store_true',
                        help='Compare two models side-by-side (Model1 left, Model2 right)')
    parser.add_argument('--gt-only', action='store_true',
                        help='Only show ground truth (no predictions)')
    parser.add_argument('--pred-only', action='store_true',
                        help='Only show predictions (no ground truth)')
    
    # Other options
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
    parser.add_argument('--no-label-prefix', action='store_true',
                        help='Do not add GT/Pred prefix to labels')
    
    return parser.parse_args()


def check_ffmpeg():
    """Check if ffmpeg is available for H.264 encoding"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def convert_to_h264(input_path, output_path):
    """Convert video to H.264 codec using ffmpeg for better compatibility"""
    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', str(input_path),
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: Could not convert to H.264: {e}")
        return False


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
            except (ValueError, IndexError):
                print(f"Warning: Could not parse line: {line}")
                continue
    
    return annotations


def normalized_to_pixel(box, img_width, img_height):
    """Convert normalized box to pixel coordinates"""
    center_x = box['center_x'] * img_width
    center_y = box['center_y'] * img_height
    width = box['width'] * img_width
    height = box['height'] * img_height
    
    x1 = int(center_x - width / 2)
    y1 = int(center_y - height / 2)
    x2 = int(center_x + width / 2)
    y2 = int(center_y + height / 2)
    
    return x1, y1, x2, y2


def draw_boxes(frame, boxes, colors, label_prefix="", thickness=2, font_scale=0.6, 
               show_conf=True, is_gt=False):
    """Draw bounding boxes on frame"""
    frame = frame.copy()
    img_height, img_width = frame.shape[:2]
    
    for box in boxes:
        if is_gt:
            x1, y1, x2, y2 = normalized_to_pixel(box, img_width, img_height)
            class_id = box['class_id']
            conf = None
        else:
            x1, y1, x2, y2 = box['box']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_id = int(box['class_id'])
            conf = box.get('conf', None)
        
        color = colors.get(class_id, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        class_name = CLASS_NAMES.get(class_id, f'Class_{class_id}')
        if conf is not None and show_conf:
            label = f"{class_name}: {conf:.2f}"
        else:
            label = class_name
        
        if label_prefix:
            label = f"{label_prefix} {label}"
        
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        
        label_y = y1 - 5 if y1 > label_height + 10 else y2 + label_height + 5
        label_x = x1
        
        cv2.rectangle(
            frame,
            (label_x, label_y - label_height - baseline),
            (label_x + label_width, label_y + baseline),
            color, -1
        )
        
        cv2.putText(
            frame, label, (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA
        )
    
    return frame


def draw_title(frame, title, color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Draw a title on the frame"""
    frame = frame.copy()
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), bg_color, -1)
    cv2.putText(frame, title, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return frame


def draw_info_panel(frame, frame_id, total_frames, counts_dict, video_name):
    """Draw information panel on the frame"""
    frame = frame.copy()
    img_height, img_width = frame.shape[:2]
    
    panel_height = 80
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (img_width, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)
    
    cv2.putText(frame, f"Video: {video_name}", (10, 20), font, font_scale, color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Frame: {frame_id}/{total_frames}", (10, 45), font, font_scale, color, 1, cv2.LINE_AA)
    
    x_offset = img_width // 3
    y = 20
    for name, (count, c) in counts_dict.items():
        cv2.putText(frame, f"{name}: {count}", (x_offset, y), font, font_scale, c, 1, cv2.LINE_AA)
        y += 25
        if y > 70:
            y = 20
            x_offset += img_width // 4
    
    return frame


def load_model(model_path, model_type, conf_threshold, iou_threshold, device):
    """Load detection model"""
    if model_type == 'yolo':
        from models.model import DetectionModel
        model = DetectionModel(
            model_path=model_path,
            conf_threshold=0.7,
            iou_threshold=0.9,
            device=device
        )
    elif model_type == 'multiscale':
        from models.multiscale_model import DetectionModel
        model = DetectionModel(
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


def get_video_predictions(model, video_path, model_type='yolo'):
    """Get model predictions for entire video using video_detect for multiscale model"""
    if model_type == 'multiscale':
        print(f"  Running video_detect for {model_type} model...")
        all_frames_detections = model.video_detect(str(video_path))
    else:
        # Use image_detect frame by frame for yolo model
        print(f"  Running image_detect for {model_type} model...")
        all_frames_detections = []
        cap = cv2.VideoCapture(str(video_path))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            detections = model.image_detect(frame)
            all_frames_detections.append(detections)
        cap.release()
    
    # Convert to list of prediction dicts per frame
    all_predictions = []
    for frame_detections in all_frames_detections:
        predictions = []
        for i in range(len(frame_detections['boxes'])):
            box = frame_detections['boxes'][i]
            if hasattr(box, 'numpy'):
                box = box.numpy()
            
            score = frame_detections['scores'][i]
            if hasattr(score, 'item'):
                score = score.item()
            
            label = frame_detections['labels'][i]
            if hasattr(label, 'item'):
                label = label.item()
            
            predictions.append({
                'box': (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                'conf': float(score),
                'class_id': int(label)
            })
        all_predictions.append(predictions)
    
    return all_predictions


def visualize_video_compare_models(video_path, label_path, output_path, 
                                    model1, model2, model1_name, model2_name, 
                                    model1_type, model2_type, args):
    """Create visualization video comparing two models side-by-side"""
    
    # Pre-compute all predictions using video_detect for better tracking
    print(f"  Pre-computing predictions for {model1_name}...")
    all_pred1 = get_video_predictions(model1, video_path, model1_type) if model1 else []
    
    print(f"  Pre-computing predictions for {model2_name}...")
    all_pred2 = get_video_predictions(model2, video_path, model2_type) if model2 else []
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if args.fps is not None:
        fps = args.fps
    
    gt_annotations = load_video_labels(str(label_path))
    
    output_width = width * 2
    temp_output = str(output_path).replace('.mp4', '_temp.mp4')
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (output_width, height))
    
    if not out.isOpened():
        print(f"Error: Could not create output video: {output_path}")
        cap.release()
        return False
    
    video_name = Path(video_path).stem
    
    frame_id = 0
    pbar = tqdm(total=total_frames, desc=f"Rendering {video_name}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gt_boxes = gt_annotations.get(frame_id, [])
        
        # Use pre-computed predictions
        pred1 = all_pred1[frame_id] if frame_id < len(all_pred1) else []
        pred2 = all_pred2[frame_id] if frame_id < len(all_pred2) else []
        
        # Create left frame (Model 1)
        frame_left = frame.copy()
        
        if not args.pred_only and gt_boxes:
            frame_left = draw_boxes(
                frame_left, gt_boxes, GT_COLORS, 
                label_prefix="GT:" if not args.no_label_prefix else "",
                thickness=args.box_thickness, font_scale=args.font_scale, is_gt=True
            )
        
        if pred1:
            frame_left = draw_boxes(
                frame_left, pred1, PRED_COLORS,
                label_prefix=f"{model1_name}:" if not args.no_label_prefix else "",
                thickness=args.box_thickness, font_scale=args.font_scale, is_gt=False
            )
        
        frame_left = draw_title(frame_left, f"{model1_name}", (255, 165, 0))
        
        # Create right frame (Model 2)
        frame_right = frame.copy()
        
        if not args.pred_only and gt_boxes:
            frame_right = draw_boxes(
                frame_right, gt_boxes, GT_COLORS,
                label_prefix="GT:" if not args.no_label_prefix else "",
                thickness=args.box_thickness, font_scale=args.font_scale, is_gt=True
            )
        
        if pred2:
            frame_right = draw_boxes(
                frame_right, pred2, PRED_COLORS,
                label_prefix=f"{model2_name}:" if not args.no_label_prefix else "",
                thickness=args.box_thickness, font_scale=args.font_scale, is_gt=False
            )
        
        frame_right = draw_title(frame_right, f"{model2_name}", (255, 0, 255))
        
        combined = np.hstack([frame_left, frame_right])
        
        if args.show_info:
            info_text = f"Frame: {frame_id}/{total_frames} | GT: {len(gt_boxes)} | {model1_name}: {len(pred1)} | {model2_name}: {len(pred2)}"
            cv2.putText(combined, info_text, (10, height - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        out.write(combined)
        
        frame_id += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    if check_ffmpeg():
        print(f"  Converting to H.264 for better compatibility...")
        if convert_to_h264(temp_output, str(output_path)):
            os.remove(temp_output)
        else:
            os.rename(temp_output, str(output_path))
    else:
        os.rename(temp_output, str(output_path))
    
    return True


def visualize_video(video_path, label_path, output_path, model=None, model_type='yolo', args=None):
    """Create visualization video"""
    
    # Pre-compute predictions if model is provided
    all_predictions = []
    if model is not None and not args.gt_only:
        print(f"  Pre-computing predictions for {model_type} model...")
        all_predictions = get_video_predictions(model, video_path, model_type)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if args.fps is not None:
        fps = args.fps
    
    gt_annotations = load_video_labels(str(label_path))
    
    output_width = width * 2 if args.side_by_side else width
    temp_output = str(output_path).replace('.mp4', '_temp.mp4')
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (output_width, height))
    
    if not out.isOpened():
        print(f"Error: Could not create output video: {output_path}")
        cap.release()
        return False
    
    video_name = Path(video_path).stem
    
    frame_id = 0
    pbar = tqdm(total=total_frames, desc=f"Rendering {video_name}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gt_boxes = gt_annotations.get(frame_id, [])
        
        # Use pre-computed predictions
        pred_boxes = all_predictions[frame_id] if frame_id < len(all_predictions) else []
        
        if args.side_by_side:
            frame_gt = frame.copy()
            frame_pred = frame.copy()
            
            if not args.pred_only:
                frame_gt = draw_boxes(
                    frame_gt, gt_boxes, GT_COLORS, 
                    label_prefix="GT:" if not args.no_label_prefix else "",
                    thickness=args.box_thickness, font_scale=args.font_scale, is_gt=True
                )
            
            if pred_boxes:
                frame_pred = draw_boxes(
                    frame_pred, pred_boxes, PRED_COLORS,
                    label_prefix="Pred:" if not args.no_label_prefix else "",
                    thickness=args.box_thickness, font_scale=args.font_scale, is_gt=False
                )
            
            frame_gt = draw_title(frame_gt, "Ground Truth", (0, 255, 0))
            frame_pred = draw_title(frame_pred, "Predictions", (0, 0, 255))
            
            combined = np.hstack([frame_gt, frame_pred])
            
            if args.show_info:
                info_text = f"Frame: {frame_id}/{total_frames} | GT: {len(gt_boxes)} | Pred: {len(pred_boxes)}"
                cv2.putText(combined, info_text, (10, height - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            out.write(combined)
        else:
            vis_frame = frame.copy()
            
            if not args.pred_only:
                vis_frame = draw_boxes(
                    vis_frame, gt_boxes, GT_COLORS,
                    label_prefix="GT:" if not args.no_label_prefix else "",
                    thickness=args.box_thickness, font_scale=args.font_scale, is_gt=True
                )
            
            if pred_boxes and not args.gt_only:
                vis_frame = draw_boxes(
                    vis_frame, pred_boxes, PRED_COLORS,
                    label_prefix="Pred:" if not args.no_label_prefix else "",
                    thickness=args.box_thickness, font_scale=args.font_scale, is_gt=False
                )
            
            if args.show_info:
                counts = {
                    'GT': (len(gt_boxes), GT_COLORS[0]),
                    'Pred': (len(pred_boxes), PRED_COLORS[1])
                }
                vis_frame = draw_info_panel(
                    vis_frame, frame_id, total_frames, counts, video_name
                )
            
            out.write(vis_frame)
        
        frame_id += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    if check_ffmpeg():
        print(f"  Converting to H.264 for better compatibility...")
        if convert_to_h264(temp_output, str(output_path)):
            os.remove(temp_output)
        else:
            os.rename(temp_output, str(output_path))
    else:
        os.rename(temp_output, str(output_path))
    
    return True


def main():
    args = parse_args()
    
    data_dir = Path(args.data_dir)
    video_dir = data_dir / 'videos' / args.split
    label_dir = data_dir / 'labels' / args.split
    output_dir = Path(args.output_dir) / data_dir.name / args.split
    
    if not video_dir.exists():
        print(f"Error: Video directory not found: {video_dir}")
        return
    
    if not label_dir.exists():
        print(f"Error: Label directory not found: {label_dir}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    has_ffmpeg = check_ffmpeg()
    if not has_ffmpeg:
        print("Warning: ffmpeg not found. Output videos may not be compatible with all players.")
        print("Install ffmpeg for better compatibility: sudo apt install ffmpeg")
    
    model1 = None
    model2 = None
    model1_name = "YOLO"
    model2_name = "Multiscale"
    
    if args.compare_models:
        if not args.model_path:
            print("Error: --model-path is required for --compare-models")
            return
        if not args.model_path_2:
            print("Error: --model-path-2 is required for --compare-models")
            return
        
        print(f"Loading Model 1 ({args.model_type}) from {args.model_path}...")
        model1 = load_model(args.model_path, args.model_type,
                           args.conf_threshold, args.iou_threshold, args.device)
        model1_name = args.model_type.upper()
        
        print(f"Loading Model 2 ({args.model_type_2}) from {args.model_path_2}...")
        model2 = load_model(args.model_path_2, args.model_type_2,
                           args.conf_threshold, args.iou_threshold, args.device)
        model2_name = args.model_type_2.upper()
        
        print("Models loaded successfully!")
        
    elif args.model_path and not args.gt_only:
        print(f"Loading model ({args.model_type}) from {args.model_path}...")
        model1 = load_model(args.model_path, args.model_type,
                           args.conf_threshold, args.iou_threshold, args.device)
        print("Model loaded successfully!")
    
    video_files = list(video_dir.glob('*.mp4'))
    
    if args.video_name:
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
    print(f"Data directory:   {data_dir}")
    print(f"Split:            {args.split}")
    print(f"Videos found:     {len(video_files)}")
    print(f"Output directory: {output_dir}")
    
    if args.compare_models:
        print(f"Mode:             Compare Models")
        print(f"Model 1:          {model1_name} ({args.model_path})")
        print(f"Model 2:          {model2_name} ({args.model_path_2})")
    elif args.side_by_side:
        print(f"Mode:             Side-by-side (GT vs Pred)")
        print(f"Model:            {args.model_path if args.model_path else 'None'}")
    else:
        print(f"Mode:             {'GT Only' if args.gt_only else 'Single View'}")
        print(f"Model:            {args.model_path if args.model_path else 'None'}")
    
    print(f"H.264 encoding:   {'Yes' if has_ffmpeg else 'No (ffmpeg not found)'}")
    print(f"{'='*60}\n")
    
    success_count = 0
    for video_path in video_files:
        video_name = video_path.stem
        label_path = label_dir / f"{video_name}.txt"
        
        if args.compare_models:
            suffix = f"_{model1_name.lower()}_vs_{model2_name.lower()}"
        elif args.side_by_side:
            suffix = "_sbs"
        elif args.gt_only:
            suffix = "_gt"
        elif args.pred_only:
            suffix = "_pred"
        elif model1 is not None:
            suffix = "_comparison"
        else:
            suffix = "_gt"
        
        output_path = output_dir / f"{video_name}{suffix}.mp4"
        
        print(f"\nProcessing: {video_name}")
        
        if args.compare_models:
            success = visualize_video_compare_models(
                video_path, label_path, output_path,
                model1, model2, model1_name, model2_name,
                args.model_type, args.model_type_2, args
            )
        else:
            success = visualize_video(video_path, label_path, output_path, model1, args.model_type, args)
        
        if success:
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
