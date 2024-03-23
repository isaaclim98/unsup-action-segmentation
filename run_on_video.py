import glob
import os
import time
import numpy as np
import pandas as pd
from python.read_utils import get_mapping, read_gt_label, estimate_cost_matrix, avg_gt_activity_datasets
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from python.cluster_algo import FINCH, ABD
from sklearn.cluster import SpectralClustering, DBSCAN, OPTICS
import argparse

def run_video_clustering(
        video_name='rgb-01-1.avi',
        dataset_name='50Salads',
        datasets_path='./Action_Segmentation_Datasets',
        tw_finch=True,
        verbose=False,
        algo='twfinch',
        features='orb',
        existing_gt=True,
        save_labels=True,
        num_clusters=1
):
    # setup paths
    vid_name = video_name
    ds_name = dataset_name
    path_ds = os.path.join(datasets_path, ds_name)
    path_vid = os.path.join(datasets_path, ds_name, 'resampled_videos', vid_name)
    path_gt = os.path.join(path_ds, 'groundTruth/')
    if dataset_name == "YTI" and not features == 'idt':
        path_mapping = os.path.join(path_ds, 'mapping', 'mapping_idxlabel.txt')
    else: 
        path_mapping = os.path.join(path_ds, 'mapping', 'mapping.txt')
    
    if features == 'sift':
        path_features = os.path.join(path_ds, 'histoListSift64/')
    elif features == 'orb':
        path_features = os.path.join(path_ds, 'histoListOrb64/')
    elif features == 'brisk':
        path_features = os.path.join(path_ds, 'histoListBrisk64/')
    elif features == 'akaze':
        path_features = os.path.join(path_ds, 'histoListAkaze64/')
    elif features == 'fast':
        path_features = os.path.join(path_ds, 'histoListFast64/')
    elif features == 'idt':
        path_features = os.path.join(path_ds, 'features/')
    else:
        raise ValueError(f"Invalid value for features: {features}. Choose from [sift, orb, idt]")
    
    os.makedirs(path_features, exist_ok=True)
    os.makedirs(os.path.join(path_ds, "y_pred", features, f"{algo}"), exist_ok=True)
    os.makedirs(os.path.join(path_ds, "req_c", features, f"{algo}"), exist_ok=True)

    # %% Load all needed files: Descriptor, GT & Mapping
    if os.path.exists(path_mapping):
        # Create the Mapping dict
        mapping_dict = get_mapping(path_mapping)
        print("Mapping dictionary:")
        print(mapping_dict)
    else:
        mapping_dict = None
        print("No mapping path exists!")

    # Create the result matrix to hold all values in the end
    # results_matrix_ds = np.zeros(shape=(len(filenames), 7), dtype=object)

    # Load the Descriptor
    video_name_no_ext = os.path.splitext(vid_name)[0]
    cur_filename = os.path.join(path_features, video_name_no_ext + '.txt')
    print(cur_filename)
    cur_desc = np.loadtxt(cur_filename, dtype='float32')
    print(cur_desc)

    # Load the GT_labels, map them to the corresponding ID    
    video_name = os.path.split(video_name_no_ext)[-1]
    print("video_name:", video_name)

    if existing_gt:

        if ds_name == "MPII_Cooking":
            activity_name = video_name.split("-")[1]
        elif ds_name == "50Salads":
            activity_name = "features"
        else:
            activity_name = os.path.basename(os.path.split(cur_filename)[0])
        print("Activity name:", activity_name)

        # n_clusters for TWFINCH
        n_clusters = int(avg_gt_activity_datasets[ds_name][activity_name])
        if dataset_name == "YTI":
            gt_label_path = os.path.join(path_gt, video_name + "_idt")
        else:
            gt_label_path = os.path.join(path_gt, video_name)
        if not os.path.exists(gt_label_path):
            gt_label_path = os.path.join(path_gt, video_name + '.txt')

        gt_labels, n_labels = read_gt_label(gt_label_path, mapping_dict=mapping_dict)

    else:
        n_clusters = num_clusters

    # cluster data      
    start = time.time()
    # print("GT Labels:", gt_labels)
    # print("Number of Frames of GT Labels:", len(gt_labels))
    # print("Number of GT Labels:", n_labels)

    if algo == 'twfinch':
        c, num_clust, req_c = FINCH(cur_desc, req_clust=n_clusters, verbose=verbose, tw_finch=tw_finch)
    elif algo == 'spectral':
        # Spectral Clustering [maybe can look to use 'precomputed_nearest_neighbors' 
        # and pass in adj, orig_dist = clust_rank(mat, initial_rank, distance=distance, use_tw_finch=tw_finch)]
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
        req_c = spectral.fit_predict(cur_desc)
    elif algo == 'dbscan':
        # DBSCAN
        dbscan = DBSCAN(metric='cosine')
        req_c = dbscan.fit_predict(cur_desc)
    elif algo == 'optics':
        # OPTICS
        optics = OPTICS(metric='cosine')
        req_c = optics.fit_predict(cur_desc)
    elif algo == 'abd':
        # Action Boundary Detection
        req_c = ABD(cur_desc, K=n_clusters, alpha=0.4, filter='gaussian', verbose=verbose)
    else:
        raise ValueError("Invalid clustering algorithm. Choose 'twfinch', 'abd', 'spectral', 'optics', 'dbscan'.")
        

    # print("length of req_c:", len(req_c))
    # print("unique labels in req_c:", len(set(req_c)))
    # print("req_c:", req_c)
        
    if existing_gt:

        if len(req_c) > len(gt_labels):
            # Trim extra elements from req_c
            req_c = req_c[:len(gt_labels)]
        elif len(req_c) < len(gt_labels):
            # Duplicate the last element of req_c until lengths match
            last_element = req_c[-1]
            req_c = np.concatenate((req_c, np.tile(last_element, len(gt_labels) - len(req_c))))

        # Find best assignment through Hungarian Method
        cost_matrix = estimate_cost_matrix(gt_labels, req_c)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # decode the predicted labels
        y_pred = col_ind[req_c]

        # Calculate the metrics (External libraries)
        mof = -cost_matrix[row_ind, col_ind].sum() / len(
            cur_desc)  # MoF accuracy 
        cur_acc = metrics.accuracy_score(gt_labels, y_pred)

        f1_macro = metrics.f1_score(gt_labels, y_pred, average='macro') # F1-Score
        # iou_macro = metrics.jaccard_score(gt_labels, y_pred, average='macro')  # IOU
        # penalize equally over/under clustering
        iou = np.sum(metrics.jaccard_score(gt_labels, y_pred, average=None)) / n_clusters
    end = time.time()

    if save_labels:
        if existing_gt:
            output_file_path = os.path.join(path_ds, "y_pred", features, f"{algo}", f"{video_name}_y_pred.txt")
            cluster_labels = y_pred
        else:
            output_file_path = os.path.join(path_ds, "req_c", features, f"{algo}", f"{video_name}_req_c.txt")
            cluster_labels = req_c

        # Save the cluster labels to a text file
        with open(output_file_path, 'w') as file:
            for label in cluster_labels:
                line = str(label)
                file.write(line + '\n')

        print(f"Clustering labels saved to: {output_file_path}")

    def display_cluster_ranges(req_c):
        cluster_ranges = {}
        current_cluster = req_c[0]
        start_index = 0

        for i, label in enumerate(req_c[1:], start=1):
            if label != current_cluster:
                if current_cluster not in cluster_ranges:
                    cluster_ranges[current_cluster] = [(start_index, i - 1)]
                else:
                    cluster_ranges[current_cluster].append((start_index, i - 1))
                current_cluster = label
                start_index = i

        # Add the range for the last cluster
        if current_cluster not in cluster_ranges:
            cluster_ranges[current_cluster] = [(start_index, len(req_c) - 1)]
        else:
            cluster_ranges[current_cluster].append((start_index, len(req_c) - 1))

        for cluster, ranges in cluster_ranges.items():
            print(f"Cluster {cluster}: Index Ranges {', '.join([f'{start}-{end}' for start, end in ranges])}")
        
    display_cluster_ranges(cluster_labels)

    if verbose:
        if existing_gt:
            print(f'Algo: {algo.upper()} | Features: {features.upper()} | Evaluation on Video {activity_name}/{video_name} finshed in {np.round(end - start)} seconds: accuracy = {cur_acc} IoU = {iou} and f1 ={f1_macro}')
        else:
            print(f'Algo: {algo.upper()} | Features: {features.upper()} | Evaluation on Video {ds_name}/{video_name} finshed in {np.round(end - start)} seconds.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-name', required=True, help='Specify the name of the video.')
    parser.add_argument('--dataset-name', required=True, help='Specify the name of dataset. Options: [Breakfast, 50Salads, YTI]')
    parser.add_argument('--datasets-path', required=True, help='Specify the root folder of all datsets')
    parser.add_argument('--num-clusters', required=True, help='Specify the number of clusters desired.')
    parser.add_argument('--algo', default='twfinch', help='Options: [twfinch, abd, spectral, optics, dbscan]')
    parser.add_argument('--features', default='orb', help='Options: [sift, orb]')
    parser.add_argument('--tw-finch', action='store_true', default=True)
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--existing-gt', action='store_true', help='Specify if the video has existing ground truth labels.')
    parser.add_argument('--save-labels', action='store_true', help='Specify if you want to save prediction labels.')
    args = parser.parse_args()
    _ = run_video_clustering(video_name=args.video_name,
                          dataset_name=args.dataset_name,
                          datasets_path=args.datasets_path,
                          algo=args.algo,
                          features=args.features,
                          tw_finch=args.tw_finch,
                          verbose=args.verbose,
                          existing_gt=args.existing_gt,
                          save_labels=args.save_labels,
                          num_clusters=args.num_clusters)