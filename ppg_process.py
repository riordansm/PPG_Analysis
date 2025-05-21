import cv2
import numpy as np
from scipy import ndimage
import scipy.signal as signal
import pickle
import json
import argparse
import tqdm
from ppg_alignment_gui import get_frame  # Reuse the grid functions

def post_process(l_in):
    """Post-process the frame data.
    
    Args:
        l_in (ndarray): Input frame data array
    
    Returns:
        ndarray: Processed frame data
    """
    try:
        # Remove DC component
        l = l_in - np.mean(l_in, axis=1)[:, None]
        
        # Normalize
        l = l / np.std(l, axis=1)[:, None]
        
        # Apply bandpass filter
        fs = 30  # Sampling frequency (assumed 30 fps)
        nyq = fs/2
        low = 0.5/nyq
        high = 4/nyq
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Filter each row
        for i in range(l.shape[0]):
            l[i,:] = signal.filtfilt(b, a, l[i,:])
            
        return l
        
    except Exception as e:
        raise ValueError(f"Error in post-processing: {str(e)}")

def process_video(input_video_path, json_path):
    """Process a video file using parameters from a JSON file.
    
    Args:
        input_video_path (str): Path to the input video file
        json_path (str): Path to the JSON file containing parameters
    """
    try:
        # Load parameters from JSON
        with open(json_path, 'r') as f:
            params = json.load(f)
        
        # Extract parameters
        rotation = params['r']
        x_shift = params['x']
        y_shift = params['y']
        proto = params['p']
        
        # Open video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
            
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize data arrays
        l = np.zeros((100, total_frames))
        r = np.zeros((100, total_frames))
        
        # Get grid pattern
        rr, ll = get_frame(proto, BINING=2)  # Use original BINING value
        
        # Process each frame
        for frame_idx in tqdm.tqdm(range(total_frames), desc="Processing frames"):
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Could not read frame {frame_idx}")
                
            # Apply transformations
            frame1 = ndimage.rotate(frame, rotation)
            frame1 = ndimage.shift(frame1, (x_shift, y_shift, 0))
            
            # Process right grid
            for y_offset in range(10):
                for x_offset in range(10):
                    # Right side
                    ya = rr[x_offset][y_offset]['ya']
                    yb = rr[x_offset][y_offset]['yb']
                    xa = rr[x_offset][y_offset]['xa']
                    xb = rr[x_offset][y_offset]['xb']
                    
                    r[y_offset*10 + x_offset, frame_idx] = np.mean(frame1[ya:yb, xa:xb])
                    
                    # Left side
                    ya = ll[x_offset][y_offset]['ya']
                    yb = ll[x_offset][y_offset]['yb']
                    xa = ll[x_offset][y_offset]['xa']
                    xb = ll[x_offset][y_offset]['xb']
                    
                    l[y_offset*10 + x_offset, frame_idx] = np.mean(frame1[ya:yb, xa:xb])
        
        # Post-process data
        l = post_process(l)
        r = post_process(r)
        
        # Save to PKL file with original format
        with open(f'{input_video_path}.pkl', 'wb') as f:
            pickle.dump({
                'l': l,
                'r': r,
                'x': x_shift,
                'y': y_shift,
                'rotation': rotation
            }, f)
            
        print(f"Processing complete. Results saved to {input_video_path}.pkl")
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if 'cap' in locals() and cap is not None:
            cap.release()
            cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Process PPG video using alignment parameters')
    parser.add_argument('video', help='Path to the input video file')
    parser.add_argument('json', help='Path to the JSON file containing alignment parameters')
    
    args = parser.parse_args()
    process_video(args.video, args.json)

if __name__ == '__main__':
    main() 