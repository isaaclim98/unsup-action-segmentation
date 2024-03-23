import argparse
import os

from math import ceil
from moviepy.video.io.VideoFileClip import VideoFileClip
from python.read_utils import get_mapping, read_gt_label

def resample_video(input_path, output_dir, dataset_name, target_total_frames=None, skip_existing=False):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract the filename and extension from the input video path
    if dataset_name in ["Breakfast", "YTI"]:
        parent_folder = os.path.basename(os.path.dirname(input_path))
        video_file_name = os.path.basename(input_path)
        os.makedirs(os.path.join(output_dir, parent_folder), exist_ok=True)

        video_name = os.path.join(parent_folder, video_file_name)
        video_name, video_extension = os.path.splitext(video_name)
    else:
        video_name, video_extension = os.path.splitext(os.path.basename(input_path))

    # Build the output video path in the resampled_videos folder
    output_path = os.path.join(output_dir, f"{video_name}.mp4")

    if os.path.exists(output_path):
        print(f"Skipping video {video_name}. Resampled video already exists.")
        return

    # Load the video clip
    clip = VideoFileClip(input_path)
    
    if target_total_frames is not None:

        # Calculate target frame rate
        duration = clip.duration
        target_frame_rate = target_total_frames / duration

        # Resample the video using set_fps
        resampled_clip = clip.set_fps(target_frame_rate)

        # Compare total frames with the target (rounding up)
        resampled_frames = ceil(resampled_clip.fps * resampled_clip.duration)
        if resampled_frames > target_total_frames:
            # Trim extra frames at the end
            resampled_clip = resampled_clip.subclip(0, target_total_frames / resampled_clip.fps)
        elif resampled_frames < target_total_frames:
            # Duplicate the final frame until it reaches the target
            last_frame = resampled_clip.get_frame(resampled_clip.duration - 1)
            num_duplicates = target_total_frames - resampled_frames

            # Generate duplicated frames
            duplicated_frames = [last_frame] * num_duplicates
            # Manually add duplicated frames to the end
            resampled_clip = resampled_clip.set_make_frame(lambda t: last_frame if t >= resampled_clip.duration else resampled_clip.get_frame(t))

        clip = resampled_clip

        # Print the number of frames in the resampled video
        print(f"Number of frames in the resampled video: {int(clip.fps * clip.duration)}")
        
    # Write the resampled video to the specified output path
    clip.write_videofile(output_path, codec="libx264", audio_codec="aac")


def run_on_existing(video_name='rgb-01-1.avi', dataset_name='50Salads', 
                    datasets_path='/media/ntu/shengyang/Action_Segmentation_Datasets',
                    skip_existing=False):
    
    # setup paths
    vid_name = video_name
    ds_name = dataset_name
    path_ds = os.path.join(datasets_path, ds_name)
    path_vid = os.path.join(datasets_path, ds_name, 'videos', vid_name)
    path_gt = os.path.join(path_ds, 'groundTruth/')
    if dataset_name == "YTI":
        path_mapping = os.path.join(path_ds, 'mapping', 'mapping_idxlabel.txt')
    else: 
        path_mapping = os.path.join(path_ds, 'mapping', 'mapping.txt')

    output_directory = os.path.join(datasets_path, ds_name, 'resampled_videos')

    video_name_no_ext = os.path.splitext(vid_name)[0]

    # %% Load all needed files: Descriptor, GT & Mapping
    if os.path.exists(path_mapping):
        # Create the Mapping dict
        mapping_dict = get_mapping(path_mapping)
    else:
        mapping_dict = None
        print("No mapping path exists!")


    # Load the GT_labels, map them to the corresponding ID    
    video_name = os.path.split(video_name_no_ext)[-1]
    print("video_name:", video_name)

    if dataset_name == "YTI":
        gt_label_path = os.path.join(path_gt, video_name + "_idt")
    else:
        gt_label_path = os.path.join(path_gt, video_name)
    print("gt_label_path:", gt_label_path)
    if not os.path.exists(gt_label_path):
        gt_label_path = os.path.join(path_gt, video_name + '.txt')

    gt_labels, n_labels = read_gt_label(gt_label_path, mapping_dict=mapping_dict)

    target_total_frames = len(gt_labels)

    print("Target number of frames:", target_total_frames)

    resample_video(path_vid, output_directory, dataset_name, target_total_frames, skip_existing)

def run_on_new(video_name='rgb-01-1.avi', dataset_name='Test', 
                    datasets_path='/media/ntu/shengyang/Action_Segmentation_Datasets'):
    
    # setup paths
    vid_name = video_name
    ds_name = dataset_name
    path_ds = os.path.join(datasets_path, ds_name)
    path_vid = os.path.join(datasets_path, ds_name, 'videos', vid_name)

    output_directory = os.path.join(path_ds, 'resampled_videos')

    video_name_no_ext = os.path.splitext(vid_name)[0]

    target_total_frames = None

    resample_video(path_vid, output_directory, dataset_name, target_total_frames)

def resample_directory(dataset_name, datasets_path, new_video_flag = False, skip_existing = False):

    path_dir = os.path.join(datasets_path, dataset_name, 'videos')

    if new_video_flag: # New videos (without GT labels and mapping)
        # List all video files in the input directory
        video_files = [f for f in os.listdir(path_dir) if f.lower().endswith(('.avi', '.mp4', '.mov', '.mkv', '.mpg'))]

        for video_file in video_files:
            run_on_new(video_file, dataset_name, datasets_path)
    else: # Existing videos (with GT labels and mapping)
        if dataset_name in ["Breakfast", "YTI"]:
            for root, dirs, files in os.walk(path_dir):
                for dir_name in dirs:
                    curr_dir = os.path.join(path_dir, dir_name)
                    print("current dir:", curr_dir)
                    video_files = [f for f in os.listdir(curr_dir) if f.lower().endswith(('.avi', '.mp4', '.mov', '.mkv', '.mpg'))]
                    for video_file in video_files:
                        curr_video = os.path.join(dir_name, video_file)
                        print("current video:", curr_video)
                        run_on_existing(curr_video, dataset_name, datasets_path, skip_existing)
        else:
            # List all video files in the input directory
            video_files = [f for f in os.listdir(path_dir) if f.lower().endswith(('.avi', '.mp4', '.mov', '.mkv', '.mpg'))]
            for video_file in video_files:
                run_on_existing(video_file, dataset_name, datasets_path, skip_existing)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', required=True, help='Specify the name of dataset. Options: [Breakfast, 50Salads, YTI]')
    parser.add_argument('--datasets-path', required=True, help='Specify the root folder of all datasets.')
    parser.add_argument('--video-name', default=None, help='Specify the video file name (including extension).')
    parser.add_argument('--new-video-flag', action='store_true', help='Specify whether to resample to a target framerate or not.')
    parser.add_argument('--run-on-directory', action='store_true', help='Specify whether to run visualisation on the whole dataset directory.')
    parser.add_argument('--skip-existing', action='store_true', help='Specify whether to skip videos that exist in the resampled folder.')
    args = parser.parse_args()

    if args.run_on_directory:
        if args.new_video_flag:
            print("Running on directory of new videos...")
            resample_directory(args.dataset_name, args.datasets_path, args.new_video_flag, args.skip_existing)
        else:
            print("Running on directory existing videos...")
            resample_directory(args.dataset_name, args.datasets_path, args.new_video_flag, args.skip_existing)

    else:
        if args.new_video_flag:
            print("Running on new video...")
            run_on_new(args.video_name, args.dataset_name, args.datasets_path)
        else:
            print("Running on existing video...")
            run_on_existing(args.video_name, args.dataset_name, args.datasets_path)

    