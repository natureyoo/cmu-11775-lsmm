#!/bin/bash

# This script performs a complete Media Event Detection pipeline (MED) using video features:
# a) preprocessing of videos, b) feature representation,
# c) computation of MAP scores

# You can pass arguments to this bash script defining which one of the steps you want to perform.
# This helps you to avoid rewriting the bash script whenever there are
# intermediate steps that you don't want to repeat.

# execute: bash run.pipeline.sh -p true -f true -m true -y filepath

# Reading of all arguments:
while getopts p:f:m:y: option		# p:f:m:y: is the optstring here
	do
	case "${option}"
	in
	p) PREPROCESSING=${OPTARG};;       # boolean true or false
	f) FEATURE_REPRESENTATION=${OPTARG};;  # boolean
	m) MAP=${OPTARG};;                 # boolean
    y) YAML=$OPTARG;;                  # path to yaml file containing parameters for feature extraction
	esac
	done

export PATH=~/anaconda3/bin:$PATH

if [ "$PREPROCESSING" = true ] ; then

    echo "#####################################"
    echo "#         PREPROCESSING             #"
    echo "#####################################"

    # steps only needed once
    video_path=~/video  # path to the directory containing all the videos.
    mkdir -p list downsampled_videos surf cnn kmeans  # create folders to save features
    awk '{print $1}' ../hw1_code/list/train > list/train.video  # save only video names in one file (keeping first column)
    awk '{print $1}' ../hw1_code/list/val > list/val.video
    cat list/train.video list/val.video list/test.video > list/all.video    #save all video names in one file
    downsampling_frame_len=60
    downsampling_frame_rate=15

    # 1. Downsample videos into shorter clips with lower frame rates.
    # TODO: Make this more efficient through multi-threading f.ex.
    start=`date +%s`
    for line in $(cat "list/all.video"); do
        ffmpeg -y -ss 0 -i $video_path/${line}.mp4 -strict experimental -t $downsampling_frame_len -r $downsampling_frame_rate downsampled_videos/$line.ds.mp4
    done
    end=`date +%s`
    runtime=$((end-start))
    echo "Downsampling took: $runtime" #28417 sec around 8h without parallelization

    # 2. TODO: Extract SURF features over keyframes of downsampled videos (0th, 5th, 10th frame, ...)
    python3.6 surf_feat_extraction.py -i list/all.video config.yaml

    # 3. TODO: Extract CNN features from keyframes of downsampled videos
    python3.6 cnn_feat_extraction.py
	

fi

if [ "$FEATURE_REPRESENTATION" = true ] ; then

    echo "#####################################"
    echo "#  SURF FEATURE REPRESENTATION      #"
    echo "#####################################"

    # 1. TODO: Train kmeans to obtain clusters for SURF features
    python3.6 train_kmeans.py surf

    # 2. TODO: Create kmeans representation for SURF features
    python3.6 create_kmeans.py surf

	echo "#####################################"
    echo "#   CNN FEATURE REPRESENTATION      #"
    echo "#####################################"

    # 1. TODO: Train kmeans to obtain clusters for CNN features
    python3.6 train_kmeans.py cnn

    # 2. TODO: Create kmeans representation for CNN features
    python3.6 create_kmeans.py cnn

fi

if [ "$MAP" = true ] ; then

    echo "#######################################"
    echo "# MED with SURF Features: MAP results #"
    echo "#######################################"

    # Paths to different tools;
    map_path=/home/ubuntu/tools/mAP
    export PATH=$map_path:$PATH

    # 1. TODO: Train SVM with OVR using only videos in training set.
    for i in P001 P002 P003; do python3.6 train_svm.py ${i} surf svm false; done
    # 2. TODO: Test SVM with val set and calculate its MAP scores for own info.
    for i in P001 P002 P003; do python3.6 test_svm.py svm/surf_${i}_only_train surf val result/${i}_surf_val.lst; ap list/${i}_val_label result/${i}_surf_val.lst; done

    # 3. TODO: Train SVM with OVR using videos in training and validation set.
    for i in P001 P002 P003; do python3.6 train_svm.py ${i} surf svm true; done
    # 4. TODO: Test SVM with test set saving scores for submission
    for i in P001 P002 P003; do python3.6 test_svm.py svm/surf_${i}_with_val surf test result/${i}_surf.lst; done

    echo "#######################################"
    echo "# MED with CNN Features: MAP results  #"
    echo "#######################################"


    # 1. TODO: Train SVM with OVR using only videos in training set.
    for i in P001 P002 P003; do python3.6 train_svm.py ${i} cnn svm false; done

    # 2. TODO: Test SVM with val set and calculate its MAP scores for own info.
    for i in P001 P002 P003; do python3.6 test_svm.py svm/cnn_${i}_only_train cnn val result/${i}_cnn_val.lst; ap list/${i}_val_label result/${i}_cnn_val.lst; done

    # 3. TODO: Train SVM with OVR using videos in training and validation set.
    for i in P001 P002 P003; do python3.6 train_svm.py ${i} cnn svm true; done
    # 4. TODO: Test SVM with test set saving scores for submission
    for i in P001 P002 P003; do python3.6 test_svm.py svm/cnn_${i}_with_val cnn test result/${i}_cnn.lst; done

fi

    echo "##############################"
    echo "# Ensemble for best results  #"
    echo "##############################"

    # ensemble model
    python3.6 ensemble.py cnn,surf result
    for event in P001 P002 P003; do 
        ap list/${event}_val_label result/${event}_best_val.lst
    done


