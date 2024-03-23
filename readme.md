# Unsupervised Action Segmentation in Videos with Clustering Algorithms

The code used for exploration and implementation in this project was based on and adapted from the [open-sourced code](https://github.com/ssarfraz/FINCH-Clustering/tree/master/TW-FINCH) by Sarfraz et al. that showcased their [TW-FINCH](https://openaccess.thecvf.com/content/CVPR2021/html/Sarfraz_Temporally-Weighted_Hierarchical_Clustering_for_Unsupervised_Action_Segmentation_CVPR_2021_paper.html) algorithm. The code implementation of the [ABD](https://openaccess.thecvf.com/content/CVPR2022/html/Du_Fast_and_Unsupervised_Action_Boundary_Detection_for_Action_Segmentation_CVPR_2022_paper.html) algorithm was replicated based on the algorithms and written work of Du et al., as there was no code available online regarding ABD. 

## Datasets:

This project utilised 3 datasets: 
* The [Breakfast Actions](https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/) dataset
* The [50 Salads](https://cvip.computing.dundee.ac.uk/datasets/foodpreparation/50salads/) dataset
* The [YouTube Instruction Videos](https://www.di.ens.fr/willow/research/instructionvideos/) dataset

The ground truth labels and label mapping were obtained from the download link in the README file in the [open-sourced code](https://github.com/ssarfraz/FINCH-Clustering/tree/master/TW-FINCH) by [Sarfraz et al.](https://openaccess.thecvf.com/content/CVPR2021/html/Sarfraz_Temporally-Weighted_Hierarchical_Clustering_for_Unsupervised_Action_Segmentation_CVPR_2021_paper.html), who in turn obtained it from the work of [Kukleva et al.](https://openaccess.thecvf.com/content_CVPR_2019/html/Kukleva_Unsupervised_Learning_of_Action_Classes_With_Continuous_Temporal_Embedding_CVPR_2019_paper.html). The raw video files for each dataset were obtained from their respective websites (linked above).

## Folder structure
After obtaning the ground truth labels and label mapping from [here](https://github.com/ssarfraz/FINCH-Clustering/tree/master/TW-FINCH), the original folder structure will look as such:

```
Action_Segmentation_Datasets
│
├── 50Salads/            
│   ├── features/       # IDT features (not used in this project)
│   ├── groundTruth/    # Ground Truth Labels
│   └── mapping/        # Label mappings
│
├── Breakfast/       
│   ├── features/       # IDT features (not used in this project)
│   ├── groundTruth/    # Ground Truth Labels
│   └── mapping/        # Label mappings
│
├── Hollywood_extended/ # Not used in this project
│   ├── features/       # IDT features (not used in this project)
│   ├── groundTruth/    # Ground Truth Labels
│   └── mapping/        # Label mappings
│
├── MPII_Cooking/       # Not used in this project
│   ├── features/       # IDT features (not used in this project)
│   ├── groundTruth/    # Ground Truth Labels
│   └── mapping/        # Label mappings
│
└── YTI/
    ├── features/       # IDT features (not used in this project)
    ├── groundTruth/    # Ground Truth Labels
    └── mapping/        # Label mappings

```

For each dataset, place the raw videos in a separate folder where the structure is as follows:

```
Action_Segmentation_Datasets
│
├── 50Salads/            
│   ├── features/              # IDT features (not used in this project)
│   ├── groundTruth/           # Ground Truth Labels
│   ├── mapping/               # Label mappings
│   └── videos/                # Raw videos
│       └── video files directly here
│
├── Breakfast/             
│   ├── features/              # IDT features (not used in this project)
│   ├── groundTruth/           # Ground Truth Labels
│   ├── mapping/               # Label mappings
│   └── videos/                # Raw videos
│       ├── cereals/ 
│       ├── coffee/
│       ├── friedegg/
│       ├── juice/
│       ├── milk/
│       ├── pancake/     
│       ├── salat/
│       ├── sandwich/
│       ├── scrambledegg/
│       └── tea/
│           └── video files for each activity type here
│
└── YTI/
    ├── features/              # IDT features (not used in this project)
    ├── groundTruth/           # Ground Truth Labels
    ├── mapping/               # Label mappings
    └── videos/                # Raw videos
        ├── changing_tire/ 
        ├── coffee/
        ├── cpr/
        ├── jump_car/
        └── repot/
            └── video files for each activity type here

```

## Usage on a single unlabelled video (Implementation):

The following line will take in a raw video file located at the path: /your/path/here/Action_Segmentation_Datasets/Test/videos/repot_0005.mpg and output a labelled video at the path: /your/path/here/Action_Segmentation_Datasets/Test/labeled_videos/repot_0005.mp4.

```
python single_video.py --dataset-name Test --datasets-path /your/path/here/Action_Segmentation_Datasets --video-name repot_0005.mpg --algo twfinch --num-clusters 7
```

Input:

* dataset-name:  Specify the name of dataset. Options: [Breakfast, 50Salads, YTI]
* datasets-path: Specify the root folder of all datasets.
* video-name: Specify the video file name (including extension).
* num-clusters: Specify the number of clusters desired.
* [OPTIONAL]
    * algo: Specify the desired clustering algorithm. Options: [twfinch, abd, spectral, optics, dbscan] (Default: twfinch)
    * features: Specify the feature extractor to use. Options: [sift, orb] (Default: orb)
    * dimension: Specify the desired dimensionality of the feature vector. (Default: 64)

## Usage on a dataset (Exploration):

### 1. Resampling Videos:

```
python resample.py --dataset-name 50Salads --datasets-path /your/path/here/Action_Segmentation_Datasets --run-on-directory --skip-existing
```
Input:

* dataset-name:  Specify the name of dataset. Options: [Breakfast, 50Salads, YTI]
* datasets-path: Specify the root folder of all datasets.
* [OPTIONAL]
    * video-name: Specify the video file name (including extension). (Default: None)
* [FLAGS]
    * new-video-flag: Specify whether to resample to a target framerate or not (whether this video has ground truth labels).
    * run-on-directory: Specify whether to run visualisation on the whole dataset directory (selected dataset only).
    * skip-existing: Specify whether to skip videos that exist in the resampled folder.

### 2. Feature Extraction:

```
python extract.py --dataset-name 50Salads --datasets-path /your/path/here/Action_Segmentation_Datasets --features orb --dimension 64 --skip-existing
```
Input:

* dataset-name:  Specify the name of dataset. Options: [Breakfast, 50Salads, YTI]
* datasets-path: Specify the root folder of all datasets.
* [OPTIONAL]
    * features: Specify the feature extractor to use. Options: [sift, orb] (Default: orb)
    * dimension: Specify the desired dimensionality of the feature vector. (Default: 64)
* [FLAGS]
    * skip-existing: Specify whether to skip videos that already have features in the feature folder.

### 3a. Run on a single video:

For the 50Salads dataset, the video name should just be the video filename, as shown below.
```
python run_on_video.py --video-name rgb-01-1.avi --dataset-name 50Salads --datasets-path /your/path/here/Action_Segmentation_Datasets --num-clusters 18
```

For the Breakfast or YTI datasets, the video filename should also include the parent directory as well, as shown below.
```
python run_on_video.py --video-name jump_car/jump_car_0002.mpg --dataset-name YTI --datasets-path /your/path/here/Action_Segmentation_Datasets --num-clusters 15 --save-labels
```
Input:

* dataset-name:  Specify the name of dataset. Options: [Breakfast, 50Salads, YTI]
* datasets-path: Specify the root folder of all datasets.
* video-name: Specify the video file name (including extension).
* num-clusters: Specify the number of clusters desired. Only required if the video does not have existing ground truth labels. (Default: 1)
* [OPTIONAL]
    * algo: Specify the desired clustering algorithm. Options: [twfinch, abd, spectral, optics, dbscan] (Default: twfinch)
    * features: Specify the feature extractor to use. Options: [sift, orb] (Default: orb)
* [FLAGS]
    * existing-gt: Specify if the video has existing ground truth labels.
    * save-labels: Specify if you want to save prediction labels.
    * verbose: Specify verbosity of the code.

### 3b. Run on a dataset:

```
python run_on_dataset.py --dataset-name Breakfast --datasets-path /your/path/here/Action_Segmentation_Datasets --algo abd --save-labels --skip-existing
```
Input:

* dataset-name:  Specify the name of dataset. Options: [Breakfast, 50Salads, YTI]
* datasets-path: Specify the root folder of all datasets.
* [OPTIONAL]
    * algo: Specify the desired clustering algorithm. Options: [twfinch, abd, spectral, optics, dbscan] (Default: twfinch)
    * features: Specify the feature extractor to use. Options: [sift, orb] (Default: orb)
* [FLAGS]
    * save-labels: Specify if you want to save prediction labels.
    * skip-existing: Specify whether to skip videos that already have prediction labels in the label folder.
    * verbose: Specify verbosity of the code.

### 4. Visualise cluster labels:

If running on a single video in the 50Salads dataset, --dir-name is not needed, as shown below.
```
python visualise.py --dataset-name 50Salads --algo spectral --video-name rgb-01-1.avi
```

If running on a single video in the Breakfast or YTI datasets, the video's parent directory should be input in --dir-name, as shown below.
```
python visualise.py --dataset-name YTI --algo spectral --video-name jump_car_0002.mpg --dir-name jump_car
```

Else, if running on the whole directory for a selected dataset, do as shown below.
```
python visualise.py --dataset-name YTI --algo dbscan --run-on-directory 
```
Input:

* dataset-name:  Specify the name of dataset. Options: [Breakfast, 50Salads, YTI]
* datasets-path: Specify the root folder of all datasets.
* video-name: Specify the video file name (including extension). Required if running on a single video only. (Default: None)
* dir-name: Specify the parent folder of the single video for Breakfast/YTI datasets. (Default: None)
* [OPTIONAL]
    * algo: Specify the desired clustering algorithm. Options: [twfinch, abd, spectral, optics, dbscan] (Default: twfinch)
    * features: Specify the feature extractor to use. Options: [sift, orb] (Default: orb)
* [FLAGS]
    * run-on-directory: Specify whether to run visualisation on the whole dataset directory (selected dataset only).
    * new-video-flag: Specify whether to use req_c (True) or y_pred (False).
    * show-plot: Specify whether to show the visualised plot on screen. Only for use on systems with graphical display capabilities.
