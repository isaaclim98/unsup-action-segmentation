import argparse
import cv2
import numpy as np
import os
import time
from resample import run_on_new
from extract import process_video
from run_on_video import run_video_clustering
from moviepy.editor import VideoFileClip, AudioFileClip

def process_single_video(video_name, dataset_name, datasets_path, extractor, dim, clustering_algo, num_clusters):

  video_name_no_ext = os.path.splitext(video_name)[0]
    
  output_directory = os.path.join(datasets_path, dataset_name, 'labeled_videos')
  os.makedirs(output_directory, exist_ok=True)

  # Resample the video
  resampled_video_path = os.path.join(datasets_path, dataset_name, 'resampled_videos', f"{video_name_no_ext}.mp4")
  run_on_new(video_name, dataset_name, datasets_path)

  # Extract features from the resampled video
  process_video(resampled_video_path, dataset_name, feature_extractor=extractor, dimension=dim)

  # Perform the video clustering (also saves the predicted cluster labels)
  run_video_clustering(video_name=video_name,
                        dataset_name=dataset_name,
                        datasets_path=datasets_path,
                        algo=clustering_algo,
                        features=extractor,
                        verbose=True,
                        existing_gt=False,
                        save_labels=True,
                        num_clusters=num_clusters)

  # Read the predicted cluster labels
  req_c_path = os.path.join(datasets_path, dataset_name, "req_c", extractor, f"{clustering_algo}", f"{video_name_no_ext}_req_c.txt")
  req_c = np.loadtxt(req_c_path, dtype=int)

  # Load the resampled video
  cap = cv2.VideoCapture(resampled_video_path)
  framerate_resampled = cap.get(cv2.CAP_PROP_FPS)

  # Add cluster labels to each frame and save the labeled video
  labeled_video_path = os.path.join(output_directory, f"{video_name_no_ext}_label.mp4")
  output_video_path = os.path.join(output_directory, f"{video_name_no_ext}.mp4")
  add_cluster_labels_to_video(cap, req_c, labeled_video_path, framerate_resampled, output_directory, video_name_no_ext)
  add_audio_to_video(resampled_video_path, labeled_video_path, output_video_path)

  print(f"Labeled video saved to: {output_video_path}")

  cap.release()

def add_cluster_labels_to_video(cap, cluster_labels, output_path, framerate, output_directory, video_name_no_ext):
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can choose other codecs based on your system and preferences
  out = cv2.VideoWriter(output_path, fourcc, framerate, (int(cap.get(3)), int(cap.get(4))))

  frame_idx = 0
  while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Add cluster label to the bottom left corner of the frame
    cluster_label = cluster_labels[frame_idx]
    # cv2.putText(frame, f'Cluster: {cluster_label}', (10, int(cap.get(4)) - 10),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    text = f'Cluster: {cluster_label}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_x = 10
    text_y = int(cap.get(4)) - 10
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    out.write(frame)

    # Saving the first frame of each cluster
    png_filename = f"{video_name_no_ext}_cluster{cluster_label}.png"
    png_path = os.path.join(output_directory, png_filename)
    if not os.path.exists(png_path):
        cv2.imwrite(png_path, frame)

    frame_idx += 1

  cap.release()
  out.release()

def add_audio_to_video(original_video_path, labeled_video_path, output_path):
  # Load the video clip
  original_video_clip = VideoFileClip(original_video_path)
  output_audio_path = output_path.replace('.mp4', '_audio.mp3')

  # Extract the audio from the video clip
  audio_clip = original_video_clip.audio

  if audio_clip is not None:

    # Write the extracted audio to a separate file
    audio_clip.write_audiofile(output_audio_path)
    audio_clip.close()
    audio_background = AudioFileClip(output_audio_path)

  original_video_clip.close()

  # Combine the video and audio
  final_video_clip = VideoFileClip(labeled_video_path)

  if audio_clip is not None:
    final_video_clip = final_video_clip.set_audio(audio_background)
    os.remove(output_audio_path)

  final_video_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

  os.remove(labeled_video_path)

if __name__ == "__main__":
    
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset-name', required=True, help='Specify the name of dataset. Options: [Breakfast, 50Salads, YTI]')
  parser.add_argument('--datasets-path', default='/media/ntu/shengyang/Action_Segmentation_Datasets', help='Specify the root folder of all datsets.')
  parser.add_argument('--algo', default='twfinch', help='Options: [twfinch, abd, spectral, optics, dbscan]')
  parser.add_argument('--features', default='orb', help='Options: [sift, orb]')
  parser.add_argument('--video-name', required=True, help='Specify the video file name (including extension).')
  parser.add_argument('--dimension', default=64, type=int, help='Specify the desired dimensionality of the feature vector.')
  parser.add_argument('--num-clusters', required=True, type=int, help='Specify the number of clusters desired.')
  args = parser.parse_args()

  start = time.time()

  # Process the single video
  process_single_video(args.video_name, args.dataset_name, args.datasets_path, args.features, args.dimension, args.algo, args.num_clusters)

  end = time.time()

  print(f'Overall Extraction, Clustering, Labelling on Video {args.dataset_name}/{args.video_name} finshed in {np.round(end - start)} seconds')
