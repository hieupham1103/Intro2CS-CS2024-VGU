import os
import re
from pathlib import Path
from collections import defaultdict
import cv2
from tqdm import tqdm

# Configuration
DATASETS = ['RGB', 'IR']
DATA_ROOT = 'data'
OUTPUT_ROOT = 'data'
FPS = 24
DURATION = 5
SPLIT = 'both'  # 'train' / 'test' / 'both'


def extract_sequence_info(filename):
    name = filename.replace('IR_', '')
    
    match = re.match(r'([A-Z]+_\d+)_(\d+)\.(jpg|png)', name)
    if match:
        sequence_id = match.group(1)
        frame_num = int(match.group(2))
        return sequence_id, frame_num
    
    return None, None


def group_images_by_sequence(image_dir):
    sequences = defaultdict(list)
    
    image_files = sorted(os.listdir(image_dir))
    for filename in image_files:
        if not filename.lower().endswith(('.jpg', '.png')):
            continue
        
        seq_id, frame_num = extract_sequence_info(filename)
        if seq_id and frame_num:
            filepath = os.path.join(image_dir, filename)
            sequences[seq_id].append((frame_num, filepath, filename))
    
    for seq_id in sequences:
        sequences[seq_id].sort(key=lambda x: x[0])
    
    return sequences


def read_label_file(label_path):
    """Read YOLO format label file and return list of bounding boxes"""
    if not os.path.exists(label_path):
        return []
    
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                boxes.append(line)
    return boxes


def create_video_label_file(frames, label_dir, output_path, data_set, fps=30, target_frames=150):
    if not frames:
        return False
    
    num_frames = len(frames)
    
    # Determine frame mapping (same logic as video creation)
    if num_frames >= target_frames:
        step = num_frames / target_frames
        selected_indices = [int(i * step) for i in range(target_frames)]
        frame_repeat = 1
    else:
        selected_indices = list(range(num_frames))
        frame_repeat = max(1, target_frames // num_frames)
    
    video_labels = []
    video_frame_num = 0
    
    for idx in selected_indices:
        frame_num, frame_path, filename = frames[idx]
        
        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(label_dir, label_filename)
        
        boxes = read_label_file(label_path)
        
        for _ in range(frame_repeat):
            for box in boxes:
                video_labels.append(f"{video_frame_num} {box}")
            
            video_frame_num += 1
            if video_frame_num >= target_frames:
                break
        
        if video_frame_num >= target_frames:
            break
    
    with open(output_path, 'w') as f:
        for label in video_labels:
            f.write(label + '\n')
    
    return True


def create_video_from_sequence(frames, output_path, fps=30, target_frames=150):
    if not frames:
        return False
    
    first_frame_path = frames[0][1]
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        print(f"Error: Could not read {first_frame_path}")
        return False
    
    height, width = first_frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not open VideoWriter for {output_path}")
        return False
    
    num_frames = len(frames)
    
    if num_frames >= target_frames:
        step = num_frames / target_frames
        selected_indices = [int(i * step) for i in range(target_frames)]
        frame_repeat = 1
    else:
        selected_indices = list(range(num_frames))
        frame_repeat = max(1, target_frames // num_frames)
    
    frames_written = 0
    for idx in selected_indices:
        frame_num, frame_path, filename = frames[idx]
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"Warning: Could not read {frame_path}")
            continue
        
        for _ in range(frame_repeat):
            out.write(frame)
            frames_written += 1
            
            if frames_written >= target_frames:
                break
        
        if frames_written >= target_frames:
            break
    
    out.release()
    return True


def process_dataset(data_set, data_root, output_root, split, fps, duration):
    print(f"\n{'='*60}")
    print(f"Processing {data_set} dataset - {split} split")
    print(f"{'='*60}")
    
    input_dir = os.path.join(data_root, data_set, 'images', split)
    label_dir = os.path.join(data_root, data_set, 'labels', split)
    output_video_dir = os.path.join(output_root, f'{data_set}_video', "videos", split)
    output_label_dir = os.path.join(output_root, f'{data_set}_video', 'labels', split)
    
    if not os.path.exists(input_dir):
        print(f"Warning: Input directory not found: {input_dir}")
        return
    
    os.makedirs(output_video_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # Group images by sequence
    print(f"Scanning images in {input_dir}...")
    sequences = group_images_by_sequence(input_dir)
    print(f"Found {len(sequences)} sequences")
    
    if not sequences:
        print("No valid sequences found!")
        return
    
    target_frames = fps * duration
    
    print(f"Creating videos and labels ({fps} fps, {duration}s duration = {target_frames} frames)...")
    success_count = 0
    label_count = 0
    
    for seq_id, frames in tqdm(sequences.items(), desc=f"Creating {data_set} videos"):
        # Create video
        video_output_path = os.path.join(output_video_dir, f"{seq_id}.mp4")
        video_success = create_video_from_sequence(frames, video_output_path, fps, target_frames)
        
        # Create label file
        label_output_path = os.path.join(output_label_dir, f"{seq_id}.txt")
        label_success = create_video_label_file(frames, label_dir, label_output_path, data_set, fps, target_frames)
        
        if video_success:
            success_count += 1
        if label_success:
            label_count += 1
    
    print(f"\nSuccessfully created {success_count}/{len(sequences)} videos")
    print(f"Successfully created {label_count}/{len(sequences)} label files")
    print(f"Output directory: {output_video_dir}")
    print(f"Label directory: {output_label_dir}")


def main():
    print(f"{'='*60}")
    print("Video Creation Tool")
    print(f"{'='*60}")
    print(f"Settings:")
    print(f"  FPS: {FPS}")
    print(f"  Duration: {DURATION}s")
    print(f"  Target frames per video: {FPS * DURATION}")
    print(f"  Processing: {', '.join(DATASETS)}")
    print(f"  Split: {SPLIT}")
    print(f"{'='*60}")
    
    splits = []
    if SPLIT == 'both':
        splits = ['train', 'test']
    else:
        splits = [SPLIT]
    
    for dataset in DATASETS:
        for split in splits:
            process_dataset(
                data_set=dataset,
                data_root=DATA_ROOT,
                output_root=OUTPUT_ROOT,
                split=split,
                fps=FPS,
                duration=DURATION
            )
    
    print(f"\n{'='*60}")
    print("Video creation completed!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
