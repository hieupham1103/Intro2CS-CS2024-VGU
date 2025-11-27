#!/usr/bin/env python3
"""
Reorganize IR_OD data (from Test_detection&tracking folder) to match IR_video structure.

This script:
1. Groups frame-level images by video name
2. Creates videos from frames (MP4 format)
3. Consolidates per-frame labels into single video label files
4. Outputs in the same structure as IR_video folder

Input structure:
    Test_detection&tracking/
        IR/
            images/
                IR_BIRD_068142_001.png
                IR_BIRD_068142_002.png
                ...
        IR_test_images/
            labels/
                train/
                    IR_BIRD_068142_001.txt
                    IR_BIRD_068142_002.txt
                    ...

Output structure:
    IR_TEST_OD_organized/
        videos/
            test/
                BIRD_068142.mp4
                ...
        labels/
            test/
                BIRD_068142.txt (format: frame_id class_id cx cy w h)
                ...
"""

import os
import re
import argparse
from pathlib import Path
from collections import defaultdict
import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Reorganize IR_OD data for evaluation')
    parser.add_argument('--input-dir', type=str, default='data/Test_detection&tracking',
                        help='Path to Test_detection&tracking directory')
    parser.add_argument('--output-dir', type=str, default='data/IR_TEST_OD_organized',
                        help='Path to output directory')
    parser.add_argument('--fps', type=int, default=30,
                        help='FPS for output videos (default: 30)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only show what would be done without actually doing it')
    return parser.parse_args()


def extract_video_info(filename):
    """
    Extract video name and frame number from filename.
    
    Examples:
        IR_BIRD_068142_001.png -> ('BIRD_068142', 1)
        IR_BIRD_068142_001.txt -> ('BIRD_068142', 1)
    """
    # Pattern: IR_<TYPE>_<ID>_<FRAME>.<ext>
    pattern = r'^IR_(BIRD|DRONE)_(\d+)_(\d+)\.(png|txt|jpg)$'
    match = re.match(pattern, filename)
    
    if match:
        obj_type = match.group(1)
        obj_id = match.group(2)
        frame_num = int(match.group(3))
        video_name = f"{obj_type}_{obj_id}"
        return video_name, frame_num
    
    return None, None


def find_all_files(input_dir):
    """Find all image and label files and group by video name."""
    input_path = Path(input_dir)
    
    # Find images in IR/images/
    image_dir = input_path / 'IR' / 'images'
    label_dir = input_path / 'IR_test_images' / 'labels' / 'train'
    
    image_files = defaultdict(list)
    label_files = defaultdict(list)
    
    # Scan images
    if image_dir.exists():
        for img_file in image_dir.iterdir():
            if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                video_name, frame_num = extract_video_info(img_file.name)
                if video_name and frame_num:
                    image_files[video_name].append((frame_num, img_file))
    
    # Scan labels
    if label_dir.exists():
        for label_file in label_dir.iterdir():
            if label_file.suffix.lower() == '.txt':
                video_name, frame_num = extract_video_info(label_file.name)
                if video_name and frame_num:
                    label_files[video_name].append((frame_num, label_file))
    
    # Sort by frame number
    for video_name in image_files:
        image_files[video_name].sort(key=lambda x: x[0])
    for video_name in label_files:
        label_files[video_name].sort(key=lambda x: x[0])
    
    return dict(image_files), dict(label_files)


def create_combined_label_file(label_files, output_path):
    """
    Create a single label file with frame_id prepended.
    
    Input format (per file): class_id cx cy w h
    Output format: frame_id class_id cx cy w h
    """
    all_labels = []
    
    for frame_num, label_path in label_files:
        # Frame numbers are 1-based in filenames, convert to 0-based for output
        frame_id = frame_num - 1
        
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Prepend frame_id to each line
                    all_labels.append(f"{frame_id} {line}")
    
    # Write combined label file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for label in all_labels:
            f.write(label + '\n')


def create_video_from_frames(image_files, output_path, fps=30):
    """Create MP4 video from list of frame images."""
    if not image_files:
        return False
    
    # Read first frame to get dimensions
    first_frame_path = image_files[0][1]
    first_frame = cv2.imread(str(first_frame_path))
    if first_frame is None:
        print(f"Error: Could not read {first_frame_path}")
        return False
    
    height, width = first_frame.shape[:2]
    
    # Setup video writer
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not create video writer for {output_path}")
        return False
    
    # Write frames
    for frame_num, frame_path in image_files:
        frame = cv2.imread(str(frame_path))
        if frame is not None:
            out.write(frame)
        else:
            print(f"Warning: Could not read {frame_path}")
    
    out.release()
    return True


def main():
    args = parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    print("=" * 80)
    print("IR_OD Data Reorganization")
    print("=" * 80)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"FPS:              {args.fps}")
    print(f"Dry run:          {args.dry_run}")
    print("=" * 80)
    
    # Find all files
    image_files, label_files = find_all_files(input_dir)
    
    print(f"\nFound {sum(len(v) for v in label_files.values())} label files (frame pattern)")
    print(f"Found {sum(len(v) for v in image_files.values())} image files (frame pattern)")
    
    # Print video info
    print(f"\nIdentified {len(label_files)} videos from labels:")
    for video_name, frames in sorted(label_files.items()):
        print(f"  - {video_name}: {len(frames)} frames")
    
    print(f"\nIdentified {len(image_files)} videos from images:")
    for video_name, frames in sorted(image_files.items()):
        print(f"  - {video_name}: {len(frames)} frames")
    
    if args.dry_run:
        print("\n[DRY RUN] Would create the following files:")
        for video_name in sorted(set(label_files.keys()) | set(image_files.keys())):
            if video_name in label_files:
                print(f"  - labels/test/{video_name}.txt")
            if video_name in image_files:
                print(f"  - videos/test/{video_name}.mp4")
        return
    
    # Create output directories
    video_output_dir = output_dir / 'videos' / 'test'
    label_output_dir = output_dir / 'labels' / 'test'
    video_output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create combined label files
    print("\nCreating combined label files...")
    label_count = 0
    for video_name, frames in tqdm(sorted(label_files.items()), desc="Labels"):
        output_path = label_output_dir / f"{video_name}.txt"
        create_combined_label_file(frames, output_path)
        label_count += 1
    
    # Create videos
    print("\nCreating videos from frames...")
    video_count = 0
    for video_name, frames in tqdm(sorted(image_files.items()), desc="Videos"):
        output_path = video_output_dir / f"{video_name}.mp4"
        if create_video_from_frames(frames, output_path, args.fps):
            video_count += 1
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Created {label_count} label files in {label_output_dir}")
    print(f"Created {video_count} video files in {video_output_dir}")
    
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"    ├── labels/")
    print(f"    │   └── test/")
    for video_name in sorted(label_files.keys())[:5]:
        print(f"    │       ├── {video_name}.txt")
    if len(label_files) > 5:
        print(f"    │       └── ... ({len(label_files) - 5} more)")
    print(f"    └── videos/")
    print(f"        └── test/")
    for video_name in sorted(image_files.keys())[:5]:
        print(f"            ├── {video_name}.mp4")
    if len(image_files) > 5:
        print(f"            └── ... ({len(image_files) - 5} more)")
    print("=" * 80)
    
    print(f"\nYou can now evaluate on IR OOD data using:")
    print(f"  python evaluate.py --model-path checkpoints/ir.pt --video-type IR --ood-test --ood-dir IR_TEST_OD_organized")


if __name__ == '__main__':
    main()
