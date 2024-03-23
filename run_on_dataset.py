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

def run_dataset_clustering(
        dataset_name='50Salads',
        datasets_path='./Action_Segmentation_Datasets',
        tw_finch=True,
        verbose=False,
        algo='twfinch',
        features='orb',
        save_labels=True,
        skip_existing=False
):
    # setup paths
    ds_name = dataset_name
    path_ds = os.path.join(datasets_path, ds_name)
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

    # %% Load all needed files: Descriptor, GT & Mapping
    if os.path.exists(path_mapping):
        # Create the Mapping dict
        mapping_dict = get_mapping(path_mapping)
        print("Mapping dictionary:")
        print(mapping_dict)
    else:
        mapping_dict = None
        print("No mapping path exists!")

    # Load all filenames from ds path        
    if ds_name == "MPII_Cooking":  # Load MPI test set files
        mpi_df = pd.read_csv(os.path.join(path_ds, 'mapping', 'sequencesTest.txt'), names=['filename'], header=None)
        mpi_df['filename'] = path_features + mpi_df['filename'] + '-cam-002.txt'
        filenames = mpi_df.filename.tolist()
    else:
        filenames = glob.glob(os.path.join(path_features, '**/*.txt'), recursive=True)

    # Create the result matrix to hold all values in the end
    results_matrix_ds = np.zeros(shape=(len(filenames), 7), dtype=object)

    # %% Loop over each file
    for file_num, cur_filename in enumerate(filenames):

        # Load the Descriptor        
        cur_desc = np.loadtxt(cur_filename, dtype='float32')

        # Load the GT_labels, map them to the corresponding ID    
        video_name = os.path.basename(cur_filename)[:-4]
        if ds_name == "MPII_Cooking":
            activity_name = video_name.split("-")[1]
        elif ds_name == "50Salads":
            activity_name = "features"
        else:
            activity_name = os.path.basename(os.path.split(cur_filename)[0])
        print("Activity name:", activity_name)

        if dataset_name in ["Breakfast", "YTI"]:
            video_dir, _ = os.path.split(cur_filename)
            parent_dir = os.path.split(video_dir)[-1]
            output_file_path = os.path.join(path_ds, "y_pred", features, f"{algo}", parent_dir, f"{video_name}_y_pred.txt")
            os.makedirs(os.path.join(path_ds, "y_pred", features, f"{algo}", parent_dir), exist_ok=True)
        else:
            output_file_path = os.path.join(path_ds, "y_pred", features, f"{algo}", f"{video_name}_y_pred.txt")

        print(f"Currently on video {video_name}")

        if skip_existing and os.path.exists(output_file_path):
            print(f"Skipping video {video_name}. Clustering label list already exists.")
            continue

        # n_clusters for TWFINCH
        n_clusters = int(avg_gt_activity_datasets[ds_name][activity_name])
        if dataset_name == "YTI":
            gt_label_path = os.path.join(path_gt, video_name + "_idt")
        else:
            gt_label_path = os.path.join(path_gt, video_name)
        if not os.path.exists(gt_label_path):
            gt_label_path = os.path.join(path_gt, video_name + '.txt')

        gt_labels, n_labels = read_gt_label(gt_label_path, mapping_dict=mapping_dict)

        # print("n_clusters:", n_clusters)
        # print("gt_label_path:", gt_label_path)
        # print("gt_labels:", gt_labels)
        # print("n_labels:", n_labels)

        # cluster data      
        start = time.time()
        # print("GT Labels:", gt_labels)
        # print("Number of Frames of GT Labels:", len(gt_labels))
        # print("Number of GT Labels:", n_labels)

        if algo == 'twfinch':
            c, num_clust, req_c = FINCH(cur_desc, req_clust=n_clusters, verbose=verbose, tw_finch=tw_finch, distance='cosine')
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

        print("Length of gt_labels:", len(gt_labels))
        print("Length of req_c:", len(req_c))
        print("Length of y_pred:", len(y_pred))

        # print("cost matrix:")
        # print(cost_matrix)

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
            
        display_cluster_ranges(req_c)

        # Calculate the metrics (External libraries)
        mof = -cost_matrix[row_ind, col_ind].sum() / len(
            cur_desc)  # MoF accuracy
        cur_acc = metrics.accuracy_score(gt_labels, y_pred)

        f1_macro = metrics.f1_score(gt_labels, y_pred, average='macro') # F1-Score
        # iou_macro = metrics.jaccard_score(gt_labels, y_pred, average='macro')  # IOU
        # penalize equally over/under clustering
        iou = np.sum(metrics.jaccard_score(gt_labels, y_pred, average=None)) / n_clusters
        end = time.time()
        if verbose:
            print(f'Evaluation on Video {activity_name}/{video_name} finshed in {np.round(end - start)} seconds: accuracy = {cur_acc} IoU = {iou} and f1 = {f1_macro}')

        if save_labels:
            with open(output_file_path, 'w') as file:
                for label in y_pred:
                    line = str(label)
                    file.write(line + '\n')

            print(f"Clustering labels saved to: {output_file_path}")

        # Transfer all the calculated metrics into the result matrix 
        results_matrix_ds[file_num][0] = activity_name + '/' + video_name
        results_matrix_ds[file_num][1] = cur_acc  # Accuracy
        results_matrix_ds[file_num][2] = iou  # IOU
        results_matrix_ds[file_num][3] = f1_macro  # F1 from Sklearn
        results_matrix_ds[file_num][4] = n_clusters  # Number of desired clusters for TW-Finch
        results_matrix_ds[file_num][5] = n_labels  # Number of ground truth clusters
        results_matrix_ds[file_num][5] = gt_labels  # Encoded GT Data
        results_matrix_ds[file_num][6] = y_pred  # Encoded cluster labels n_clusters     

    avg_score = np.mean(results_matrix_ds[:, 1:4], axis=0)
    print(f'Algo: {algo.upper()} | Features: {features.upper()} | Overall Results on {ds_name} Dataset : Acc: {avg_score[0]}, mIoU: {avg_score[1]}, Macro F1: {avg_score[2]}')
    return results_matrix_ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', required=True, help='Specify the name of dataset. Options: [Breakfast, 50Salads, YTI]')
    parser.add_argument('--datasets-path', required=True, help='Specify the root folder of all datsets.')
    parser.add_argument('--algo', default='twfinch', help='Options: [twfinch, abd, spectral, optics, dbscan]')
    parser.add_argument('--features', default='orb', help='Options: [sift, orb]')
    parser.add_argument('--tw-finch', action='store_true', default=True)
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--save-labels', action='store_true', help='Specify if you want to save prediction labels.')
    parser.add_argument('--skip-existing', action='store_true', help='Specify whether to skip videos that already have prediction labels in the label folder.')
    args = parser.parse_args()
    _ = run_dataset_clustering(dataset_name=args.dataset_name,
                          datasets_path=args.datasets_path,
                          algo=args.algo,
                          features=args.features,
                          tw_finch=args.tw_finch,
                          verbose=args.verbose,
                          save_labels=args.save_labels,
                          skip_existing=args.skip_existing)