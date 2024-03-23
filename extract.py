import numpy as np
import cv2
import os
from sklearn.cluster import KMeans


def extract_features(video_frame, feature_extractor):
    gray_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)

    if feature_extractor == 'sift':
        detector = cv2.SIFT_create()
    elif feature_extractor == 'orb':
        detector = cv2.ORB_create()
    elif feature_extractor == 'brisk':
        detector = cv2.BRISK_create()
    elif feature_extractor == 'akaze':
        detector = cv2.AKAZE_create()
    elif feature_extractor == 'fast':
        detector = cv2.FastFeatureDetector_create()
    else:
        raise ValueError("Invalid feature extractor. Choose 'sift' or 'orb'.")

    keypoints, descriptors = detector.detectAndCompute(gray_frame, None)

    if descriptors is not None:
        return descriptors
    else:
        print("Warning: Feature extraction failed for the current frame.")
        return np.array([])  # Return an empty array to handle the failure

def feature_extraction(video_path, feature_extractor='sift'):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f'Total number of frames in the video: {total_frames}')

    descriptor_list = []
    features_dict = {}

    for frame_number in range(total_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        print(f"Retrieved frame {frame_number}.")

        if ret:
            descriptors = extract_features(frame, feature_extractor)
            descriptor_list.extend(descriptors)
            features_dict[frame_number] = descriptors
        else:
            print(f"Error: Could not read frame {frame_number}.")

    cap.release()

    print(f'Total number of frames in the video: {total_frames}')

    return [descriptor_list, features_dict], total_frames

def kmeans_clustering(k, descriptor_list):
    print("Performing KMeans Clustering...")

    # Check if descriptor_list is not empty
    if not descriptor_list:
        print("Error: Descriptor list is empty.")
        return None, None

    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(descriptor_list)
    visual_words = kmeans.cluster_centers_ 
    return visual_words, kmeans


def save_histogram_to_txt(video_path, dataset_name, histo_list, feature_extractor='sift', dimension=64):
    video_dir, video_name = os.path.split(video_path)
    video_name_no_ext = os.path.splitext(video_name)[0]

    if dataset_name in ["Breakfast", "YTI"]:
        working_path = os.path.dirname(os.path.dirname(video_dir))
        parent_dir = os.path.split(video_dir)[-1]

        if feature_extractor == 'sift':
            output_file_path = os.path.join(working_path, f"histoListSift{dimension}", parent_dir, f"{video_name_no_ext}.txt")
            os.makedirs(os.path.join(working_path, f"histoListSift{dimension}"), exist_ok=True)
            os.makedirs(os.path.join(working_path, f"histoListSift{dimension}", parent_dir), exist_ok=True)
        elif feature_extractor == 'orb':
            output_file_path = os.path.join(working_path, f"histoListOrb{dimension}", parent_dir, f"{video_name_no_ext}.txt")
            os.makedirs(os.path.join(working_path, f"histoListOrb{dimension}"), exist_ok=True)
            os.makedirs(os.path.join(working_path, f"histoListOrb{dimension}", parent_dir), exist_ok=True)
        elif feature_extractor == 'brisk':
            output_file_path = os.path.join(working_path, f"histoListBrisk{dimension}", parent_dir, f"{video_name_no_ext}.txt")
            os.makedirs(os.path.join(working_path, f"histoListBrisk{dimension}"), exist_ok=True)
            os.makedirs(os.path.join(working_path, f"histoListBrisk{dimension}", parent_dir), exist_ok=True)
        elif feature_extractor == 'akaze':
            output_file_path = os.path.join(working_path, f"histoListAkaze{dimension}", parent_dir, f"{video_name_no_ext}.txt")
            os.makedirs(os.path.join(working_path, f"histoListAkaze{dimension}"), exist_ok=True)
            os.makedirs(os.path.join(working_path, f"histoListAkaze{dimension}", parent_dir), exist_ok=True)
        elif feature_extractor == 'fast':
            output_file_path = os.path.join(working_path, f"histoListFast{dimension}", parent_dir, f"{video_name_no_ext}.txt")
            os.makedirs(os.path.join(working_path, f"histoListFast{dimension}"), exist_ok=True)
            os.makedirs(os.path.join(working_path, f"histoListFast{dimension}", parent_dir), exist_ok=True)
        else:
            output_file_path = None

    else:
        working_path = os.path.dirname(video_dir)

        if feature_extractor == 'sift':
            output_file_path = os.path.join(working_path, f"histoListSift{dimension}", f"{video_name_no_ext}.txt")
            os.makedirs(os.path.join(working_path, f"histoListSift{dimension}"), exist_ok=True)
        elif feature_extractor == 'orb':
            output_file_path = os.path.join(working_path, f"histoListOrb{dimension}", f"{video_name_no_ext}.txt")
            os.makedirs(os.path.join(working_path, f"histoListOrb{dimension}"), exist_ok=True)
        elif feature_extractor == 'brisk':
            output_file_path = os.path.join(working_path, f"histoListBrisk{dimension}", f"{video_name_no_ext}.txt")
            os.makedirs(os.path.join(working_path, f"histoListBrisk{dimension}"), exist_ok=True)
        elif feature_extractor == 'akaze':
            output_file_path = os.path.join(working_path, f"histoListAkaze{dimension}", f"{video_name_no_ext}.txt")
            os.makedirs(os.path.join(working_path, f"histoListAkaze{dimension}"), exist_ok=True)
        elif feature_extractor == 'fast':
            output_file_path = os.path.join(working_path, f"histoListFast{dimension}", f"{video_name_no_ext}.txt")
            os.makedirs(os.path.join(working_path, f"histoListFast{dimension}"), exist_ok=True)
        else:
            output_file_path = None

    with open(output_file_path, 'w') as file:
        for hist in histo_list:
            line = ' '.join(map(str, hist))
            file.write(line + '\n')

    print(f"Histogram list saved to: {output_file_path}")


def process_directory(directory_path, dataset_name, feature_extractor='sift', dimension=64, skip_existing=True):
    video_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.avi', '.mp4', '.mov', '.mkv', '.mpg'))]

    for video_file in video_files:
        video_path = os.path.join(directory_path, video_file)

        # Check if histo_list.txt already exists
        video_dir, video_name = os.path.split(video_path)
        video_name_no_ext = os.path.splitext(video_name)[0]

        if dataset_name in ["Breakfast", "YTI"]:
            working_path = os.path.dirname(os.path.dirname(video_dir))
            parent_dir = os.path.split(video_dir)[-1]

            if feature_extractor == 'sift':
                histo_list_path = os.path.join(working_path, f"histoListSift{dimension}", parent_dir, f"{video_name_no_ext}.txt")
                os.makedirs(os.path.join(working_path, f"histoListSift{dimension}"), exist_ok=True)
                os.makedirs(os.path.join(working_path, f"histoListSift{dimension}", parent_dir), exist_ok=True)
            elif feature_extractor == 'orb':
                histo_list_path = os.path.join(working_path, f"histoListOrb{dimension}", parent_dir, f"{video_name_no_ext}.txt")
                os.makedirs(os.path.join(working_path, f"histoListOrb{dimension}"), exist_ok=True)
                os.makedirs(os.path.join(working_path, f"histoListOrb{dimension}", parent_dir), exist_ok=True)
            elif feature_extractor == 'brisk':
                histo_list_path = os.path.join(working_path, f"histoListBrisk{dimension}", parent_dir, f"{video_name_no_ext}.txt")
                os.makedirs(os.path.join(working_path, f"histoListBrisk{dimension}"), exist_ok=True)
                os.makedirs(os.path.join(working_path, f"histoListBrisk{dimension}", parent_dir), exist_ok=True)
            elif feature_extractor == 'akaze':
                histo_list_path = os.path.join(working_path, f"histoListAkaze{dimension}", parent_dir, f"{video_name_no_ext}.txt")
                os.makedirs(os.path.join(working_path, f"histoListAkaze{dimension}"), exist_ok=True)
                os.makedirs(os.path.join(working_path, f"histoListAkaze{dimension}", parent_dir), exist_ok=True)
            elif feature_extractor == 'fast':
                histo_list_path = os.path.join(working_path, f"histoListFast{dimension}", parent_dir, f"{video_name_no_ext}.txt")
                os.makedirs(os.path.join(working_path, f"histoListFast{dimension}"), exist_ok=True)
                os.makedirs(os.path.join(working_path, f"histoListFast{dimension}", parent_dir), exist_ok=True)
            else:
                histo_list_path = None

        else:
            working_path = os.path.dirname(video_dir)

            if feature_extractor == 'sift':
                histo_list_path = os.path.join(working_path, f"histoListSift{dimension}", f"{video_name_no_ext}.txt")
                os.makedirs(os.path.join(working_path, f"histoListSift{dimension}"), exist_ok=True)
            elif feature_extractor == 'orb':
                histo_list_path = os.path.join(working_path, f"histoListOrb{dimension}", f"{video_name_no_ext}.txt")
                os.makedirs(os.path.join(working_path, f"histoListOrb{dimension}"), exist_ok=True)
            elif feature_extractor == 'brisk':
                histo_list_path = os.path.join(working_path, f"histoListBrisk{dimension}", f"{video_name_no_ext}.txt")
                os.makedirs(os.path.join(working_path, f"histoListBrisk{dimension}"), exist_ok=True)
            elif feature_extractor == 'akaze':
                histo_list_path = os.path.join(working_path, f"histoListAkaze{dimension}", f"{video_name_no_ext}.txt")
                os.makedirs(os.path.join(working_path, f"histoListAkaze{dimension}"), exist_ok=True)
            elif feature_extractor == 'fast':
                histo_list_path = os.path.join(working_path, f"histoListFast{dimension}", f"{video_name_no_ext}.txt")
                os.makedirs(os.path.join(working_path, f"histoListFast{dimension}"), exist_ok=True)
            else:
                histo_list_path = None

        if skip_existing and os.path.exists(histo_list_path):
            print(f"Skipping video {video_name_no_ext}. features.txt already exists.")
        else:
            process_video(video_path, dataset_name, feature_extractor, dimension)

def process_video(video_path, dataset_name=None, feature_extractor='sift', dimension=64):
    print(f"Processing video: {video_path}")

    extractor = feature_extractor
    desc, total_frames = feature_extraction(video_path, feature_extractor=extractor)

    descriptor_list = desc[0]

    visual_words, kmeans = kmeans_clustering(k=dimension, descriptor_list=descriptor_list)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    histo_list = []

    for frame_number in range(total_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if ret:
            print(f"Creating Histogram for frame {frame_number}.")
            descriptors = extract_features(frame, feature_extractor=extractor)

            histo = np.zeros(dimension)
            nkp = np.size(descriptors)

            for d in descriptors:
                idx = kmeans.predict([d])
                histo[idx] += 1 / nkp

            histo_list.append(histo)
        else:
            print(f"Error: Could not read frame {frame_number}.")

    cap.release()

    save_histogram_to_txt(video_path, dataset_name, histo_list, extractor, dimension)

def process_video_batch(video_path, feature_extractor='sift', dimension=64, batch_size=10):
    extractor = feature_extractor
    desc, total_frames = feature_extraction(video_path, feature_extractor=extractor)
    descriptor_list = desc[0]
    visual_words, kmeans = kmeans_clustering(k=dimension, descriptor_list=descriptor_list)

    cap = cv2.VideoCapture(video_path)
    histo_list = []

    for start_frame in range(0, total_frames, batch_size):
        end_frame = min(start_frame + batch_size, total_frames)
        batch_frames = []

        for frame_number in range(start_frame, end_frame):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()

            if ret:
                batch_frames.append(frame)
            else:
                print(f"Error: Could not read frame {frame_number}.")

        if batch_frames:
            print(f"Creating Histograms for frames {start_frame} to {end_frame - 1}.")
            batch_descriptors = [extract_features(frame, feature_extractor=extractor) for frame in batch_frames]

            for frame_number, descriptors in enumerate(batch_descriptors):
                histo = np.zeros(dimension)
                nkp = np.size(descriptors)

                for d in descriptors:
                    idx = kmeans.predict([d])
                    histo[idx] += 1 / nkp

                histo_list.append(histo)

    cap.release()
    save_histogram_to_txt(video_path, dataset_name, histo_list, extractor, dimension)

if __name__ == '__main__':
    directory_path = '/media/ntu/shengyang/Action_Segmentation_Datasets/Breakfast/resampled_videos/'
    dataset_name = 'Breakfast'

    if dataset_name in ["Breakfast", "YTI"]:
        for root, dirs, files in os.walk(directory_path):
                for dir_name in dirs:
                    curr_dir = os.path.join(directory_path, dir_name)
                    process_directory(curr_dir, dataset_name, feature_extractor='sift', dimension=64, skip_existing=True)
                    
    else:
        process_directory(directory_path, dataset_name, feature_extractor='sift', dimension=64, skip_existing=True)
