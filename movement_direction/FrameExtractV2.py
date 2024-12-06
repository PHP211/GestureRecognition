import cv2
import numpy as np
import os

def extract_frames(video_path, output_dir):
    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return
    
    # Get total frames and FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS of the video: {fps}")
    duration = total_frames / fps
    print(f"Duration: {duration} seconds")

    # Calculate skip frame value to capture approximately 30 frames
    # skip_frame = max(int(total_frames / 10), 1)
    # print(f"Skipping {skip_frame} frames between each capture.")

    # Extract frames
    extracted_frames = []
    current_frame = 0
    while current_frame < total_frames:
        # Set the position to read the frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break
        extracted_frames.append(frame)
        # Save the frames to files in the specified directory
        cv2.imwrite(os.path.join(output_dir, f'frame_{current_frame}.jpg'), frame)
        current_frame += 1

    cap.release()
    print(f"Extracted {len(extracted_frames)} frames from the video.")
    return extracted_frames

# Usage
video_path = 'interaction/data/chop/video_306.avi'
output_dir = 'interaction/extracted_frames'
frames = extract_frames(video_path, output_dir)
