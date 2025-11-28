import os
import json
from pathlib import Path

def get_video_info(filename, folder_type):
    name = filename.replace('.mp4', '')
    display_name = name
    for suffix in ['_comparison', '_sbs', '_yolo_vs_multiscale', '_output']:
        display_name = display_name.replace(suffix, '')
    
    is_drone = name.upper().startswith('DRONE')
    is_bird = name.upper().startswith('BIRD')
    
    category = 'drone' if is_drone else ('bird' if is_bird else 'other')
    
    return {
        'filename': filename,
        'name': display_name,
        'category': category,
        'type': folder_type.lower(),
        'path': f'videos/{folder_type}/{filename}'
    }

def scan_videos():
    script_dir = Path(__file__).parent
    videos_dir = script_dir / 'videos'
    
    all_videos = []
    
    for folder_type in ['RGB', 'IR']:
        folder_path = videos_dir / folder_type
        if folder_path.exists():
            for filename in sorted(os.listdir(folder_path)):
                if filename.endswith('.mp4'):
                    video_info = get_video_info(filename, folder_type)
                    all_videos.append(video_info)
                    print(f"Found: {video_info['path']}")
    
    return all_videos

def main():
    
    videos = scan_videos()
    
    script_dir = Path(__file__).parent
    output_file = script_dir / 'videos.json'
    
    with open(output_file, 'w') as f:
        json.dump(videos, f, indent=2)
    
    
    rgb_count = len([v for v in videos if v['type'] == 'rgb'])
    ir_count = len([v for v in videos if v['type'] == 'ir'])
    drone_count = len([v for v in videos if v['category'] == 'drone'])
    bird_count = len([v for v in videos if v['category'] == 'bird'])
    
    print(f"\n Summary:")
    print(f"   RGB videos: {rgb_count}")
    print(f"   IR videos:  {ir_count}")
    print(f"   Drones:     {drone_count}")
    print(f"   Birds:      {bird_count}")

if __name__ == '__main__':
    main()
