import time
from math import floor, ceil
import argparse
import numpy as np
from sklearn import metrics
import scipy.sparse as sp
import warnings

from scipy.ndimage import gaussian_filter1d
from numpy.linalg import norm

try:
    from pynndescent import NNDescent

    pynndescent_available = True
except Exception as e:
    warnings.warn('pynndescent not installed: {}'.format(e))
    pynndescent_available = False
    pass

# use Approx NN to find first neighbor if samples more than ANN_THRESHOLD
ANN_THRESHOLD = 70000


def clust_rank(mat, initial_rank=None, distance='cosine', use_tw_finch=False):
    s = mat.shape[0]
    
    if initial_rank is not None:
        orig_dist = []
    elif s <= ANN_THRESHOLD:
        if use_tw_finch:
            loc = mat[:, -1]
            mat = mat[:, :-1]            
            loc_dist = np.sqrt((loc[:, None] - loc[:, None].T)**2)            
            
        else:
            loc_dist = 1.            

        orig_dist = metrics.pairwise.pairwise_distances(mat, mat, metric=distance)
        orig_dist = orig_dist * loc_dist
        np.fill_diagonal(orig_dist, 1e12)
        initial_rank = np.argmin(orig_dist, axis=1)
    else:
        if not pynndescent_available:
            raise MemoryError("You should use pynndescent for inputs larger than {} samples.".format(ANN_THRESHOLD))
        print('Using PyNNDescent to compute 1st-neighbours at this step ...')
        if use_tw_finch:
            print(f'Since the video is larger than {ANN_THRESHOLD} samples, we cannot compute all distances. Instead FINCH will be used')
        knn_index = NNDescent(
            mat,
            n_neighbors=2,
            metric=distance,
            )

        result, orig_dist = knn_index.neighbor_graph
        initial_rank = result[:, 1]
        orig_dist[:, 0] = 1e12
        print('Step PyNNDescent done ...')

    # The Clustering Equation
    A = sp.csr_matrix((np.ones_like(initial_rank, dtype=np.float32), (np.arange(0, s), initial_rank)), shape=(s, s))
    A = A + sp.eye(s, dtype=np.float32, format='csr')
    A = A @ A.T

    A = A.tolil()
    A.setdiag(0)   
    return A, orig_dist


def get_clust(a, orig_dist, min_sim=None):
    if min_sim is not None:
        a[np.where((orig_dist * a.toarray()) > min_sim)] = 0

    num_clust, u = sp.csgraph.connected_components(csgraph=a, directed=True, connection='weak', return_labels=True)
    return u, num_clust


def cool_mean(M, u):
    s = M.shape[0]
    un, nf = np.unique(u, return_counts=True)
    umat = sp.csr_matrix((np.ones(s, dtype='float32'), (np.arange(0, s), u)), shape=(s, len(un)))
    return (umat.T @ M) / nf[..., np.newaxis]


def get_merge(c, u, data):
    if len(c) != 0:
        _, ig = np.unique(c, return_inverse=True)
        c = u[ig]
    else:
        c = u

    mat = cool_mean(data, c)
    return c, mat


def update_adj(adj, d):
    # Update adj, keep one merge at a time
    idx = adj.nonzero()
    v = np.argsort(d[idx])
    v = v[:2]
    x = [idx[0][v[0]], idx[0][v[1]]]
    y = [idx[1][v[0]], idx[1][v[1]]]
    a = sp.lil_matrix(adj.get_shape())
    a[x, y] = 1
    return a


def req_numclust(c, data, req_clust, distance, use_tw_finch=False):
    iter_ = len(np.unique(c)) - req_clust
    c_, mat = get_merge([], c, data)
    for i in range(iter_):
        adj, orig_dist = clust_rank(mat, initial_rank=None, distance=distance, use_tw_finch=use_tw_finch)
        adj = update_adj(adj, orig_dist)
        u, _ = get_clust(adj, [], min_sim=None)
        c_, mat = get_merge(c_, u, data)
    return c_


def FINCH(data, initial_rank=None, req_clust=None, distance='cosine', tw_finch=True, ensure_early_exit=False, verbose=True):
    """ FINCH clustering algorithm.
    :param data: Input matrix with features in rows.
    :param initial_rank: Nx1 first integer neighbor indices (optional).
    :param req_clust: Set output number of clusters (optional). Not recommended.
    :param distance: One of ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'] Recommended 'cosine'.
    :param tw_finch: Run TW_FINCH on video data.
    :param ensure_early_exit: [Optional flag] may help in large, high dim datasets, ensure purity of merges and helps early exit
    :param verbose: Print verbose output.
    :return:
            c: NxP matrix where P is the partition. Cluster label for every partition.
            num_clust: Number of clusters.
            req_c: Labels of required clusters (Nx1). Only set if `req_clust` is not None.

    The code implements the FINCH algorithm described in our CVPR 2019 paper
        Sarfraz et al. "Efficient Parameter-free Clustering Using First Neighbor Relations", CVPR2019
         https://arxiv.org/abs/1902.11266
    For academic purpose only. The code or its re-implementation should not be used for commercial use.
    Please contact the author below for licensing information.
    Copyright
    M. Saquib Sarfraz (saquib.sarfraz@kit.edu)
    Karlsruhe Institute of Technology (KIT)
    """
    if tw_finch:
        n_frames = data.shape[0]
        time_index = (np.arange(n_frames) + 1.) / n_frames
        print("Shape of data before concatenation:", data.shape)
        data = np.concatenate([data, time_index[..., np.newaxis]], axis=1)
        ensure_early_exit = False
        verbose = False

    # Cast input data to float32
    data = data.astype(np.float32)
    
    min_sim = None
    adj, orig_dist = clust_rank(data, initial_rank, distance=distance, use_tw_finch=tw_finch)
    initial_rank = None
    group, num_clust = get_clust(adj, [], min_sim)
    c, mat = get_merge([], group, data)

    if verbose:
        print('Partition 0: {} clusters'.format(num_clust))

    if ensure_early_exit:
        if orig_dist.shape[-1] > 2:
            min_sim = np.max(orig_dist * adj.toarray())

    exit_clust = 2
    c_ = c
    k = 1
    num_clust = [num_clust]

    while exit_clust > 1:        
        adj, orig_dist = clust_rank(mat, initial_rank, distance=distance, use_tw_finch=tw_finch)
        u, num_clust_curr = get_clust(adj, orig_dist, min_sim)
        c_, mat = get_merge(c_, u, data)

        num_clust.append(num_clust_curr)
        c = np.column_stack((c, c_))
        exit_clust = num_clust[-2] - num_clust_curr

        if num_clust_curr == 1 or exit_clust < 1:
            num_clust = num_clust[:-1]
            c = c[:, :-1]
            break

        if verbose:
            print('Partition {}: {} clusters'.format(k, num_clust[k]))
        k += 1

    if req_clust is not None:
        if req_clust not in num_clust:
            ind = [i for i, v in enumerate(num_clust) if v >= req_clust]
            req_c = req_numclust(c[:, ind[-1]], data, req_clust, distance, use_tw_finch=tw_finch)
        else:
            req_c = c[:, num_clust.index(req_clust)]
    else:
        req_c = None

    return c, num_clust, req_c

def temporal_smoothing(x, t, k, sigma, filter="gaussian"):
    # Calculate the indices for slicing the data
    start_index = max(0, t - k)
    end_index = min(len(x), t + k + 1)

    # Slice the data for the kernel
    x_slice = x[start_index:end_index]

    if filter == "gaussian":
        # Calculate the weights using Gaussian kernel
        weights = gaussian_filter1d(np.ones_like(x_slice), sigma=sigma)

        # Apply the equation for temporal smoothing
        smoothed_value = np.sum(weights * x_slice) / np.sum(weights)

    elif filter == "average":
        # Apply the equation for temporal smoothing using an average filter
        smoothed_value = np.mean(x_slice)

    return smoothed_value

def smooth_features(features, kernel_size, filter='gaussian'):
    print("Smoothing features...")

    # Initialize smoothed features array
    smoothed_features = np.zeros_like(features)

    # Apply temporal smoothing to each feature dimension separately
    for d in range(features.shape[1]):
        for t in range(features.shape[0]):
            smoothed_features[t, d] = temporal_smoothing(features[:, d], t, kernel_size // 2, sigma=kernel_size / 6.0, filter=filter)

    return smoothed_features

def calculate_cosine_similarities(smoothed_features):
    cosine_similarities = np.zeros((1, smoothed_features.shape[0] - 1))  # create a 1 x (N-1) vector
    for i in range(smoothed_features.shape[0] - 1):
        cosine_similarities[0][i] = np.dot(smoothed_features[i],smoothed_features[i+1])\
            /(norm(smoothed_features[i])*norm(smoothed_features[i+1]))

    return cosine_similarities


def non_minimum_suppression(similarities, window_size):
    change_points = []

    for i in range(similarities.shape[1]):  # Iterate over columns
        start = max(0, i - window_size // 2)
        end = min(similarities.shape[1], i + window_size // 2 + 1)
        local_window = similarities[0, start:end]
        min_index = np.argmin(local_window)
        new_point = start + min_index

        if new_point not in change_points:
            change_points.append(new_point)

        # Print statements for debugging
        # print(f"Column {i}:")
        # print("Local Window:", local_window)
        # print("Min Index:", min_index)
        # print("Updated nms_result:")
        # print(nms_result)

    return change_points

# For detecting change points using threshold values
def detect_action_boundaries(similarity_curve, threshold=0):
    diff = np.diff(similarity_curve)
    print("Difference array:")
    print(diff)
    change_points = np.argwhere(np.abs(diff) > threshold)
    change_points = change_points[:, 1]
    return change_points

def assign_cluster_labels(features, change_points):
    labels = np.zeros(len(features), dtype=int)

    for i, cluster_idx in enumerate(change_points):
        if i == (len(change_points)-1): 
            break

        start = cluster_idx + 1
        end = change_points[i + 1] + 1
        labels[start:end] = i + 1

    # Assign the last cluster
    if len(change_points) > 0:
        labels[change_points[-1] + 1:] = len(change_points)

    return labels

def average_within_segments(features, labels):
    unique_labels = np.unique(labels)
    averaged_features = np.zeros((len(unique_labels), features.shape[1]))

    for i, label in enumerate(unique_labels):
        segment_indices = np.where(labels == label)[0]
        # print("segment_indices:", segment_indices)
        averaged_features[i] = np.mean(features[segment_indices], axis=0)

    return averaged_features

def calculate_similarity_matrix(averaged_features):
    num_clusters = len(averaged_features)
    similarity_matrix = np.zeros((num_clusters, num_clusters))

    for i in range(num_clusters):
        for j in range(num_clusters):
            if i != j:
                dot_product = np.dot(averaged_features[i], averaged_features[j])
                norm_i = norm(averaged_features[i])
                norm_j = norm(averaged_features[j])

                if norm_i != 0 and norm_j != 0 and not np.isnan(dot_product):
                    similarity_matrix[i, j] = dot_product / (norm_i * norm_j)
                else:
                    similarity_matrix[i, j] = 0  # invalid case

    return similarity_matrix

def most_similar_link(similarity_matrix):
    # Set the diagonal elements to a very small value to exclude them
    np.fill_diagonal(similarity_matrix, -np.inf)
    
    i, j = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
    return i, j

def update_labels_and_merge(labels, i, j):
    # Find cluster numbers corresponding to indices i and j
    cluster_i = np.unique(labels)[i]
    cluster_j = np.unique(labels)[j]

    # Merge clusters i and j, replace all occurrences of j with i
    labels[labels == cluster_j] = cluster_i

    return labels

def reassign_cluster_labels(labels):
    unique_labels = np.unique(labels)

    # Create a mapping from old labels to new labels
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}

    # Use the mapping to reassign labels
    new_labels = np.vectorize(label_mapping.get)(labels)

    return new_labels

def ABD(features, K=5, alpha=0.4, filter='gaussian', verbose=False):

    # print("Features:")
    # print(features)
    # print("Features shape:", features.shape)

    N = len(features) # Obtain number of frames in the video
    print("Number of Frames Received:", N)
    # alpha range: 0.2 to 0.8 for promising performance
    if floor(alpha * N / K) % 2 != 0: # number is odd
        filter_size = floor(alpha * N / K) # (2k+1) in paper
        window_size = floor(alpha * N / K) # L in paper
    else: 
        filter_size = floor(alpha * N / K) + 1 # (2k+1) in paper
        window_size = floor(alpha * N / K) + 1 # L in paper

    # Step 1: Smooth features
    smoothed_features = smooth_features(features, filter_size, filter)

    # print("Completed smoothing")
    # print(smoothed_features)
    # print(smoothed_features.shape)

    # Step 2: Calculate cosine similarities
    cosine_similarities = calculate_cosine_similarities(smoothed_features)

    # print("Completed calculating similarity")
    # print(cosine_similarities)
    # print(cosine_similarities.shape)
    
    # print("lower_quartile:", lower_quartile)
    # print("min_similarity:", min_similarity)
    # print("threshold:", thresh)

    # Step 3.1: Non-minimum suppression and detect action boundaries
    change_points = non_minimum_suppression(cosine_similarities, window_size)

    # print("Completed action boundary detection")
    # print(change_points)
    # print(change_points.shape)

    # Step 4: Clustering
    labels = assign_cluster_labels(features, change_points)

    # print("Completed initial clustering")
    # print("Number of labels:", len(labels))
    # print("Target number of clusters:", K)
    # print("Number of unique clusters:", len(np.unique(labels)))
    # print(labels)

    iteration = 1

    while len(np.unique(labels)) > K:

        if verbose:
            print("================Iteration:", iteration)
        # Step 5a: Average features within each segment
        averaged_features = average_within_segments(features, labels)

        # if verbose:
        #     print("Completed segment averaging")
        #     print(averaged_features)
        #     print(averaged_features.shape)

        # Step 5b: Calculate similarity matrix
        similarity_matrix = calculate_similarity_matrix(averaged_features)

        # if verbose:
        #     print("Completed similarity matrix")
        #     print(similarity_matrix)
        #     print(similarity_matrix.shape)

        # Step 5c: Detect most similar link
        i, j = most_similar_link(similarity_matrix)

        if verbose:
            print("Most similar link:", i, ",", j)

        # Step 5d: Update labels and merge
        labels = update_labels_and_merge(labels, i, j)

        # if verbose:
        #     print("Completed update labels with merge")
        #     print(features)
        #     print(labels)

        iteration += 1
        if verbose:
            print("Current Number of Clusters:", len(np.unique(labels)))

    labels = reassign_cluster_labels(labels)
    print("Final Labels:")
    print(labels)

    return labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True, help='Specify the path to your data csv file.')
    parser.add_argument('--output-path', default=None, help='Specify the folder to write back the results.')
    args = parser.parse_args()
    data = np.genfromtxt(args.data_path, delimiter=",").astype(np.float32)
    start = time.time()
    c, num_clust, req_c = FINCH(data, initial_rank=None, req_clust=None, distance='cosine', ensure_early_exit=True, verbose=True)
    print('Time Elapsed: {:2.2f} seconds'.format(time.time() - start))

    # Write back
    if args.output_path is not None:
        print('Writing back the results on the provided path ...')
        np.savetxt(args.output_path + '/c.csv', c, delimiter=',', fmt='%d')
        np.savetxt(args.output_path + '/num_clust.csv', np.array(num_clust), delimiter=',', fmt='%d')
        if req_c is not None:
            np.savetxt(args.output_path + '/req_c.csv', req_c, delimiter=',', fmt='%d')
    else:
        print('Results are not written back as the --output-path was not provided')


if __name__ == '__main__':
    main()