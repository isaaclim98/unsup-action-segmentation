import argparse
import os
import numpy as np

from math import ceil
from moviepy.video.io.VideoFileClip import VideoFileClip
from python.read_utils import get_mapping, read_gt_label

def compare_frames(video_path, gt_path):
    # Load the video clip
    clip = VideoFileClip(video_path)
    video_frames = int(clip.fps * clip.duration)
    video_file_name = os.path.basename(video_path)
    
    # Read ground truth labels
    gt_labels, _ = read_gt_label(gt_path)
    gt_frames = len(gt_labels)

    # Print the comparison
    print(f"Number of frames in the video ({video_file_name}): {video_frames}")
    print(f"Number of frames in the ground truth ({video_file_name}): {gt_frames}")
    if video_frames == gt_frames:
        print("The number of frames in the video and ground truth match.")
    else:
        print("The number of frames in the video and ground truth do not match.")
    print("===================================================================")

    # Calculate the difference
    frame_difference = abs(video_frames - gt_frames)
    return frame_difference, video_frames == gt_frames

def calculate_metrics_directory(dataset_name, datasets_path):
    # Setup paths
    video_dir = os.path.join(datasets_path, dataset_name, 'videos')
    gt_dir = os.path.join(datasets_path, dataset_name, 'groundTruth')

    # Collect frame differences for each video
    frame_differences = []
    videos_matched = 0
    videos_not_matched = 0
    max_difference = 0
    max_difference_video = ""

    if dataset_name in ["Breakfast", "YTI"]:
        for root, dirs, files in os.walk(video_dir):
            for dir_name in dirs:
                curr_dir = os.path.join(video_dir, dir_name)
                video_files = [f for f in os.listdir(curr_dir) if f.lower().endswith(('.avi', '.mp4', '.mov', '.mkv', '.mpg'))]
                for video_file in video_files:
                    video_name = os.path.splitext(video_file)[0]
                    video_path = os.path.join(curr_dir, video_file)
                    if dataset_name == "YTI":
                        gt_path = os.path.join(gt_dir, video_name + "_idt")
                    else:
                        gt_path = os.path.join(gt_dir, video_name)
                    frame_difference, frames_match = compare_frames(video_path, gt_path)
                    frame_differences.append(frame_difference)
                    if frame_difference > max_difference:
                        max_difference = frame_difference
                        max_difference_video = video_name
                    if frames_match:
                        videos_matched += 1
                    else:
                        videos_not_matched += 1
    else:
        video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.avi', '.mp4', '.mov', '.mkv', '.mpg'))]
        for video_file in video_files:
            video_name = os.path.splitext(video_file)[0]
            video_path = os.path.join(video_dir, video_file)
            gt_path = os.path.join(gt_dir, video_name)
            frame_difference, frames_match = compare_frames(video_path, gt_path)
            frame_differences.append(frame_difference)
            if frame_difference > max_difference:
                max_difference = frame_difference
                max_difference_video = video_name
            if frames_match:
                videos_matched += 1
            else:
                videos_not_matched += 1

    # Calculate metrics
    variance = np.var(frame_differences)
    std_deviation = np.sqrt(variance)

    return max_difference, max_difference_video, videos_matched, videos_not_matched, variance, std_deviation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', required=True, help='Specify the name of dataset. Options: [Breakfast, 50Salads, YTI]')
    parser.add_argument('--datasets-path', default='/media/ntu/shengyang/Action_Segmentation_Datasets', help='Specify the root folder of all datsets')
    args = parser.parse_args()

    max_difference, max_difference_video, \
    videos_matched, videos_not_matched, variance, std_deviation = calculate_metrics_directory(args.dataset_name, args.datasets_path)

    print(f"Maximum frame count difference ({args.dataset_name}): {max_difference} in video {max_difference_video}")
    print(f"Number of videos where frames matched: {videos_matched}")
    print(f"Number of videos where frames did not match: {videos_not_matched}")
    print(f"Variance of frame count differences ({args.dataset_name}): {variance}")
    print(f"Standard deviation of frame count differences ({args.dataset_name}): {std_deviation}")