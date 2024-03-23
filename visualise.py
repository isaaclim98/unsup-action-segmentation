import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from python.read_utils import get_mapping, read_gt_label

def setup_paths(vid_name, ds_name, datasets_path, algo, features, dir_name=None):
    video_name_no_ext = os.path.splitext(vid_name)[0]
    path_ds = os.path.join(datasets_path, ds_name)
    path_vid = os.path.join(datasets_path, ds_name, 'resampled_videos', vid_name)
    path_gt = os.path.join(path_ds, 'groundTruth/')
    if ds_name == "YTI" and not features == 'idt':
        path_mapping = os.path.join(path_ds, 'mapping', 'mapping_idxlabel.txt')
    else: 
        path_mapping = os.path.join(path_ds, 'mapping', 'mapping.txt')
    if ds_name in ["Breakfast", "YTI"]:
        path_y_pred = os.path.join(path_ds, "y_pred", features, f"{algo}", dir_name, video_name_no_ext + '_y_pred.txt')
    else:
        path_y_pred = os.path.join(path_ds, "y_pred", features, f"{algo}", video_name_no_ext + '_y_pred.txt')
    path_req_c = os.path.join(path_ds, "req_c", features, f"{algo}", video_name_no_ext + '_req_c.txt')

    return (video_name_no_ext, path_ds, path_vid, path_gt, path_mapping, path_y_pred, path_req_c)

def load_mapping(path_mapping):
    if os.path.exists(path_mapping):
        return get_mapping(path_mapping)
    else:
        print("No mapping path exists!")
        return None

def load_gt_labels(gt_label_path, dataset_name, video_name, mapping_dict):
    if os.path.isdir(gt_label_path):
        # If gt_label_path is a directory, find the file with the video_name
        if dataset_name == "YTI":
            gt_label_file = os.path.join(gt_label_path, video_name + "_idt")
        else:
            gt_label_file = os.path.join(gt_label_path, video_name)

        if os.path.exists(gt_label_file):
            gt_label_path = gt_label_file
        else:
            print(f"Error: Ground truth label file not found for {video_name} in {gt_label_path}.")
            return None

    return read_gt_label(gt_label_path, mapping_dict=mapping_dict)

def load_predicted_labels(y_pred_path):
    with open(y_pred_path, 'r') as file:
        return [int(line.strip()) for line in file]

def visualise_clusters(pred_labels, algo, ds_name, video_name_no_ext, features, gt_labels=None, save_path=None, show_plot=True):
    max_length = len(pred_labels)

    fig, ax = plt.subplots(figsize=(10, 5))

    if gt_labels is not None:
        unique_labels = sorted(set(pred_labels) | set(gt_labels))
        label_to_color = {label: i for i, label in enumerate(unique_labels)}

        for i, labels in enumerate([gt_labels, pred_labels]):
            for j, label in enumerate(labels):
                ax.plot([j, j+1], [i, i], color=plt.cm.tab20(label_to_color[label]), linewidth=15)

        ax.set_yticks([0, 1])
        ax.set_yticklabels(['gt_labels', 'pred_labels'])
        title_prefix = f"Algo: {algo.upper()} | Features: {features.upper()} | {ds_name}: {video_name_no_ext}"
    else:
        # Only plot the predicted labels
        for i, label in enumerate(pred_labels):
            ax.plot([i, i+1], [0, 0], color=plt.cm.tab20(label), linewidth=15)

        ax.set_yticks([0])
        ax.set_yticklabels(['pred_labels'])
        title_prefix = f"Algo: {algo.upper()} | Features: {features.upper()} | {ds_name}: {video_name_no_ext}"

    ax.set_xlabel('Frame Index')
    plt.title(title_prefix)

    if gt_labels is not None:
        colorbar_labels = [f'Cluster {label}' for label in unique_labels]
        colorbar_ticks = np.arange(len(unique_labels)) + 0.5

        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.tab20, norm=plt.Normalize(0, len(unique_labels))),
                            ax=ax, ticks=colorbar_ticks)

        cbar.ax.set_yticklabels(colorbar_labels)
        cbar.set_label('Cluster Label', rotation=270, labelpad=15)

    ax.set_ylim([-0.5, 1.5])

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Visualisation image saved to: {save_path}")

    if show_plot:
        plt.show()
        
    plt.close()

def run_visualisation(video_name = 'tie_01.mp4',
                      dir_name = None,
                      dataset_name = 'Test',
                      datasets_path = '/media/ntu/shengyang/Action_Segmentation_Datasets',
                      algo = 'twfinch',
                      features = 'orb',
                      new_video_flag = True,
                      show_plot = False,
                      run_on_directory = False):

    if run_on_directory:
        print("Running on directory...")
        path_dir = os.path.join(datasets_path, dataset_name, 'videos')
        if dataset_name in ["Breakfast", "YTI"]:
            for root, dirs, files in os.walk(path_dir):
                    for dir_name in dirs:
                        curr_dir = os.path.join(path_dir, dir_name)
        
                        # List all video files in the input directory
                        video_files = [f for f in os.listdir(curr_dir)]

                        for video_file in video_files:
                            video_name_no_ext, path_ds, path_vid, path_gt, path_mapping, path_y_pred, path_req_c = setup_paths(
                            video_file, dataset_name, datasets_path, algo, features, dir_name)
                            print(f"Working on video: {video_name_no_ext}")

                            mapping_dict = load_mapping(path_mapping)
                            if new_video_flag:
                                gt_labels = None
                            else:
                                gt_labels, n_labels = load_gt_labels(path_gt, dataset_name, video_name_no_ext, mapping_dict)
                            y_pred = load_predicted_labels(path_y_pred)

                            os.makedirs(os.path.join(path_ds, "visualisations", features, f"{algo}", dir_name), exist_ok=True)
                            save_path = os.path.join(path_ds, "visualisations", features, f"{algo}", dir_name, video_name_no_ext + '.png')

                            visualise_clusters(y_pred, algo, dataset_name, video_name_no_ext, features, gt_labels, save_path, show_plot)
        
        else:
            # List all video files in the input directory
            video_files = [f for f in os.listdir(path_dir)]

            for video_file in video_files:
                video_name_no_ext, path_ds, path_vid, path_gt, path_mapping, path_y_pred, path_req_c = setup_paths(
                video_file, dataset_name, datasets_path, algo, features)
                print(f"Working on video: {video_name_no_ext}")

                mapping_dict = load_mapping(path_mapping)
                if new_video_flag:
                    gt_labels = None
                    y_pred = load_predicted_labels(path_req_c)
                else:
                    gt_labels, n_labels = load_gt_labels(path_gt, dataset_name, video_name_no_ext, mapping_dict)
                    y_pred = load_predicted_labels(path_y_pred)

                os.makedirs(os.path.join(path_ds, "visualisations", features, f"{algo}"), exist_ok=True)
                save_path = os.path.join(path_ds, "visualisations", features, f"{algo}", video_name_no_ext + '.png')

                visualise_clusters(y_pred, algo, dataset_name, video_name_no_ext, features, gt_labels, save_path, show_plot)
            
    else:
        print("Running on single video...")
        video_name_no_ext, path_ds, path_vid, path_gt, path_mapping, path_y_pred, path_req_c = setup_paths(
        video_name, dataset_name, datasets_path, algo, features, dir_name)
        print(f"Working on video: {video_name_no_ext}")

        mapping_dict = load_mapping(path_mapping)
        if new_video_flag:
            gt_labels = None
            y_pred = load_predicted_labels(path_req_c)
        else:
            gt_labels, n_labels = load_gt_labels(path_gt, dataset_name, video_name_no_ext, mapping_dict)
            if dataset_name == 'Test':
                y_pred = load_predicted_labels(path_req_c)
            else:
                y_pred = load_predicted_labels(path_y_pred)

        os.makedirs(os.path.join(path_ds, "visualisations", features, f"{algo}"), exist_ok=True)
        save_path = os.path.join(path_ds, "visualisations", features, f"{algo}", video_name_no_ext + '.png')

        visualise_clusters(y_pred, algo, dataset_name, video_name_no_ext, features, gt_labels, save_path, show_plot)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', required=True, help='Specify the name of dataset. Options: [Breakfast, 50Salads, YTI]')
    parser.add_argument('--datasets-path', required=True, help='Specify the root folder of all datasets.')
    parser.add_argument('--algo', default='twfinch', help='Options: [twfinch, abd, spectral, optics, dbscan]')
    parser.add_argument('--features', default='orb', help='Options: [sift, orb]')
    parser.add_argument('--video-name', default=None, help='Specify the video file name (including extension).')
    parser.add_argument('--dir-name', default=None, help='Specify the parent folder of the video for Breakfast/YTI datasets.')
    parser.add_argument('--new-video-flag', action='store_true', help='Specify whether to use req_c (True) or y_pred (False).')
    parser.add_argument('--run-on-directory', action='store_true', help='Specify whether to run visualisation on the whole dataset directory.')
    parser.add_argument('--show-plot', action='store_true', help='Specify whether to show the visualised plot on screen.')
    args = parser.parse_args()

    _ = run_visualisation(video_name=args.video_name,
                          dir_name=args.dir_name,
                          dataset_name=args.dataset_name,
                          datasets_path=args.datasets_path,
                          algo=args.algo,
                          features=args.features,
                          new_video_flag=args.new_video_flag,
                          show_plot=args.show_plot,
                          run_on_directory=args.run_on_directory)