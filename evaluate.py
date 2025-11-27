import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import time

from models import multiscale_model as multiscale
from models import model as yolo_model


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Detection Models on Videos')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--model-type', type=str, choices=['yolo', 'multiscale'], 
                        default='yolo', help='Type of model to evaluate (default: yolo)')
    parser.add_argument('--iou-threshold', type=float, default=0.1,
                        help='IoU threshold for NMS (default: 0.1)')
    parser.add_argument('--conf-threshold', type=float, default=0.2,
                        help='Confidence threshold for detection (default: 0.2)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on (cuda/cpu)')
    parser.add_argument('--video-type', type=str, choices=['RGB', 'IR'], 
                        default='RGB', help='Type of video data (RGB or IR)')
    parser.add_argument('--data-dir', type=str, 
                        default='data/',
                        help='Path to dataset root directory')
    parser.add_argument('--scales', type=float, nargs='+', default=[1.0],
                        help='Scales for multiscale detection (only for multiscale model)')
    parser.add_argument('--crop-ratio', type=float, default=0.65,
                        help='Crop ratio for multiscale detection')
    parser.add_argument('--ood-test', action='store_true',
                        help='Evaluate on out-of-distribution test set (RGB_TEST_OD)')
    parser.add_argument('--ood-dir', type=str, default='RGB_TEST_OD',
                        help='Directory name for OOD test data (default: RGB_TEST_OD)')
    
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
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            
            frame_id = int(parts[0])
            class_id = int(parts[1])
            center_x = float(parts[2])
            center_y = float(parts[3])
            width = float(parts[4])
            height = float(parts[5])
            
            # Convert from center format to corner format (x1, y1, x2, y2) - normalized
            x1 = center_x - width / 2
            y1 = center_y - height / 2
            x2 = center_x + width / 2
            y2 = center_y + height / 2
            
            if frame_id not in annotations:
                annotations[frame_id] = []
            
            annotations[frame_id].append({
                'class': class_id,
                'bbox': [x1, y1, x2, y2]  # normalized coordinates
            })
    
    return annotations


def compute_iou(boxA, boxB):
    """
    Compute IoU between two boxes
    Boxes format: [x1, y1, x2, y2] (can be normalized or absolute)
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    union = boxAArea + boxBArea - interArea
    
    if union == 0:
        return 0
    
    return interArea / union


def normalize_detections(detections, img_width, img_height):
    """Normalize detection boxes to 0-1 range"""
    normalized = {
        "boxes": [],
        "scores": [],
        "labels": []
    }
    
    for box, score, label in zip(detections["boxes"], detections["scores"], detections["labels"]):
        # Convert box to list if it's not already
        if not isinstance(box, list):
            if hasattr(box, 'tolist'):
                box = box.tolist()
            else:
                box = list(box)
        
        # Normalize coordinates
        norm_box = [
            box[0] / img_width,
            box[1] / img_height,
            box[2] / img_width,
            box[3] / img_height
        ]
        
        # Extract scalar values
        if hasattr(score, 'item'):
            score = score.item()
        if hasattr(label, 'item'):
            label = int(label.item())
        else:
            label = int(label)
        
        normalized["boxes"].append(norm_box)
        normalized["scores"].append(float(score))
        normalized["labels"].append(label)
    
    return normalized


def calculate_ap(precisions, recalls):
    """Calculate Average Precision using 11-point interpolation"""
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    return ap


def evaluate_video(predictions, ground_truths, iou_threshold=0.5, num_classes=2):
    """
    Evaluate predictions against ground truth for video
    
    Args:
        predictions: list of detection dicts for each frame (normalized coordinates)
        ground_truths: dict mapping frame_id to list of ground truth annotations
        iou_threshold: IoU threshold for matching
        num_classes: number of classes
    
    Returns:
        metrics: dict with mAP@0.5, mAP@0.5:0.95
    """
    # Collect all detections and ground truths by class
    all_detections = {c: [] for c in range(num_classes)}
    all_ground_truths = {c: 0 for c in range(num_classes)}
    
    # Process each frame
    for frame_idx, pred in enumerate(predictions):
        # Get ground truth for this frame
        gt_list = ground_truths.get(frame_idx, [])
        
        # Mark ground truths as not matched
        gt_matched = [False] * len(gt_list)
        
        # Add predictions with frame info
        for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
            all_detections[label].append({
                'bbox': box,
                'score': score,
                'frame_idx': frame_idx,
                'matched': False
            })
        
        # Count ground truths by class
        for gt in gt_list:
            all_ground_truths[gt['class']] += 1
    
    # Calculate AP for each class and each IoU threshold
    class_metrics = {}
    iou_thresholds = np.arange(0.5, 1.0, 0.05)  # For mAP@0.5:0.95
    
    for class_id in range(num_classes):
        # Sort detections by confidence
        detections = sorted(all_detections[class_id], key=lambda x: x['score'], reverse=True)
        num_gt = all_ground_truths[class_id]
        
        if num_gt == 0:
            if len(detections) == 0:
                class_metrics[class_id] = {
                    'precision': 1.0,
                    'recall': 1.0,
                    'ap50': 1.0,
                    'ap': 1.0,
                    'tp': 0,
                    'fp': 0,
                    'fn': 0
                }
            else:
                class_metrics[class_id] = {
                    'precision': 0.0,
                    'recall': 0.0,
                    'ap50': 0.0,
                    'ap': 0.0,
                    'tp': 0,
                    'fp': len(detections),
                    'fn': 0
                }
            continue
        
        # Calculate AP for different IoU thresholds
        aps = []
        
        for iou_thresh in iou_thresholds:
            tp = np.zeros(len(detections))
            fp = np.zeros(len(detections))
            
            # Create ground truth tracking for this IoU threshold
            gt_by_frame = {}
            for frame_idx, gt_list in ground_truths.items():
                gt_class = [gt for gt in gt_list if gt['class'] == class_id]
                if gt_class:
                    gt_by_frame[frame_idx] = {'boxes': [gt['bbox'] for gt in gt_class],
                                               'matched': [False] * len(gt_class)}
            
            # Match detections to ground truths
            for det_idx, det in enumerate(detections):
                frame_idx = det['frame_idx']
                
                if frame_idx not in gt_by_frame:
                    fp[det_idx] = 1
                    continue
                
                gt_frame = gt_by_frame[frame_idx]
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt_box in enumerate(gt_frame['boxes']):
                    if gt_frame['matched'][gt_idx]:
                        continue
                    
                    iou = compute_iou(det['bbox'], gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= iou_thresh and best_gt_idx >= 0:
                    if not gt_frame['matched'][best_gt_idx]:
                        tp[det_idx] = 1
                        gt_frame['matched'][best_gt_idx] = True
                    else:
                        fp[det_idx] = 1
                else:
                    fp[det_idx] = 1
            
            # Compute precision and recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            recalls = tp_cumsum / num_gt
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
            
            # Compute AP
            ap = calculate_ap(precisions, recalls)
            aps.append(ap)
        
        # Get metrics at IoU=0.5
        ap50 = aps[0]
        
        # mAP@0.5:0.95 is the mean of all APs
        ap_mean = np.mean(aps)
        
        class_metrics[class_id] = {
            'ap50': ap50,
            'ap': ap_mean,
        }
    
    # Calculate overall metrics
    overall_map50 = np.mean([m['ap50'] for m in class_metrics.values()])
    overall_map = np.mean([m['ap'] for m in class_metrics.values()])
    
    return {
        'mAP50': overall_map50,
        'mAP': overall_map,
        'class_metrics': class_metrics,
    }


def main():
    args = parse_args()
    
    # Setup paths
    data_root = Path(args.data_dir)
    
    # Choose between normal test set and OOD test set
    if args.ood_test:
        video_dir = data_root / args.ood_dir / "videos" / "test"
        label_dir = data_root / args.ood_dir / "labels" / "test"
        test_name = f"OOD ({args.ood_dir})"
    else:
        video_dir = data_root / f"{args.video_type}_video" / "videos" / "test"
        label_dir = data_root / f"{args.video_type}_video" / "labels" / "test"
        test_name = f"{args.video_type} Videos"
    
    if not video_dir.exists():
        print(f"Error: Video directory not found: {video_dir}")
        if args.ood_test:
            print(f"Hint: Run 'python reorganize_ood_data.py' first to prepare OOD test data.")
        return
    
    if not label_dir.exists():
        print(f"Error: Label directory not found: {label_dir}")
        if args.ood_test:
            print(f"Hint: Run 'python reorganize_ood_data.py' first to prepare OOD test data.")
        return
    
    # Print configuration
    print("="*80)
    print(f"EVALUATION - {args.model_type.upper()} Model on {test_name}")
    print("="*80)
    print(f"Model path:         {args.model_path}")
    print(f"Video directory:    {video_dir}")
    print(f"Label directory:    {label_dir}")
    print(f"Device:             {args.device}")
    print(f"Model type:         {args.model_type}")
    print(f"IoU threshold:      {args.iou_threshold}")
    print(f"Conf threshold:     {args.conf_threshold}")
    if args.model_type == 'multiscale':
        print(f"Scales:             {args.scales}")
        print(f"Crop ratio:         {args.crop_ratio}")
    print("="*80)
    
    # Load model
    print(f"\nLoading {args.model_type} model from {args.model_path}...")
    if args.model_type == 'multiscale':
        detection_model = multiscale.DetectionModel(
            args.model_path,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
            device=args.device
        )
    else:
        detection_model = yolo_model.DetectionModel(
            args.model_path,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
            device=args.device
        )
    
    # Get list of test videos
    video_files = sorted(list(video_dir.glob("*.mp4")))
    
    if len(video_files) == 0:
        print(f"Error: No video files found in {video_dir}")
        return
    
    print(f"\nFound {len(video_files)} test videos")
    
    # Class names
    class_names = {0: 'bird', 1: 'drone'}
    num_classes = 2
    
    # Aggregate metrics across all videos
    all_predictions = []
    all_ground_truths = {}
    frame_offset = 0
    total_inference_time = 0
    total_frames_processed = 0
    
    # Process each video
    print("\nProcessing videos...")
    for video_path in tqdm(video_files, desc="Videos"):
        video_name = video_path.stem
        label_path = label_dir / f"{video_name}.txt"
        
        # Load ground truth
        gt_annotations = load_video_labels(label_path)
        
        # Get video dimensions
        cap = cv2.VideoCapture(str(video_path))
        img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Run detection on video with timing
        start_time = time.time()
        
        if args.model_type == 'multiscale':
            video_detections = detection_model.video_detect(
                str(video_path),
                scales=args.scales,
                crop_ratio=args.crop_ratio,
                conf_threshold=args.conf_threshold,
                iou_threshold=args.iou_threshold
            )
        else:
            video_detections = detection_model.video_detect(
                str(video_path),
                conf_threshold=args.conf_threshold,
                iou_threshold=args.iou_threshold
            )
        
        end_time = time.time()
        inference_time = end_time - start_time
        total_inference_time += inference_time
        total_frames_processed += len(video_detections)
        
        # Normalize detections and add to aggregate
        for frame_idx, frame_det in enumerate(video_detections):
            normalized_det = normalize_detections(frame_det, img_width, img_height)
            all_predictions.append(normalized_det)
            
            # Map ground truth to aggregate frame index
            if frame_idx in gt_annotations:
                all_ground_truths[frame_offset + frame_idx] = gt_annotations[frame_idx]
        
        frame_offset += len(video_detections)
    
    # Calculate FPS
    fps = total_frames_processed / total_inference_time if total_inference_time > 0 else 0
    
    # Evaluate
    print("\nCalculating metrics...")
    metrics = evaluate_video(
        all_predictions,
        all_ground_truths,
        iou_threshold=0.5,  # Standard IoU for evaluation
        num_classes=num_classes
    )
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print("\nOverall Metrics:")
    print(f"  mAP@0.50:      {metrics['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95:  {metrics['mAP']:.4f}")
    print(f"  FPS:           {fps:.2f}")
    
    print(f"\nPerformance:")
    print(f"  Total Frames:     {total_frames_processed}")
    print(f"  Inference Time:   {total_inference_time:.2f}s")
    print(f"  Avg Time/Frame:   {(total_inference_time/total_frames_processed)*1000:.2f}ms")
    
    print(f"\nPer-Class Metrics:")
    for class_id, class_metric in metrics['class_metrics'].items():
        class_name = class_names[class_id]
        print(f"\n  {class_name}:")
        print(f"    AP@0.50:      {class_metric['ap50']:.4f}")
        print(f"    AP@0.5:0.95:  {class_metric['ap']:.4f}")
    
    print("="*80)
    
    # Final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Model Type:      {args.model_type}")
    print(f"Model Path:      {args.model_path}")
    print(f"Test Set:        {test_name}")
    print(f"Videos Tested:   {len(video_files)}")
    print(f"Total Frames:    {frame_offset}")
    print(f"mAP@0.5:         {metrics['mAP50']:.4f}")
    print(f"mAP@0.5:0.95:    {metrics['mAP']:.4f}")
    print(f"FPS:             {fps:.2f}")
    print("="*80)


if __name__ == '__main__':
    main()
