import os
import argparse
from ppg_process import process_video

def find_videos_to_process(directory):
    """Find all .avi files that need processing (have .json but no .pkl)"""
    videos_to_process = []
    
    # Get all .avi files
    for filename in os.listdir(directory):
        if filename.endswith('.avi'):
            video_path = os.path.join(directory, filename)
            json_path = video_path + '.json'
            pkl_path = video_path + '.pkl'
            
            # Check if JSON exists and PKL doesn't
            if os.path.exists(json_path) and not os.path.exists(pkl_path):
                videos_to_process.append((video_path, json_path))
    
    return videos_to_process

def batch_process(directory):
    """Process all videos in directory that have JSON but no PKL"""
    # Find videos that need processing
    videos = find_videos_to_process(directory)
    
    if not videos:
        print("No videos found that need processing.")
        print("(Looking for .avi files with .json files but no .pkl files)")
        return
    
    print(f"Found {len(videos)} videos to process:")
    for video_path, _ in videos:
        print(f"  {os.path.basename(video_path)}")
    print()
    
    # Process each video
    for i, (video_path, json_path) in enumerate(videos, 1):
        print(f"\nProcessing video {i}/{len(videos)}: {os.path.basename(video_path)}")
        try:
            process_video(video_path, json_path)
        except Exception as e:
            print(f"Error processing {os.path.basename(video_path)}: {str(e)}")
            continue
    
    print("\nBatch processing complete!")

def main():
    parser = argparse.ArgumentParser(description='Batch process PPG videos that have JSON but no PKL files')
    parser.add_argument('directory', nargs='?', default='.',
                      help='Directory containing the videos (default: current directory)')
    
    args = parser.parse_args()
    batch_process(args.directory)

if __name__ == '__main__':
    main() 