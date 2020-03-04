#!/bin/bash

# An example script for multimedia event detection (MED) of Homework 1
# Before running this script, you are supposed to have the features by running run.feature.sh 

# Note that this script gives you the very basic setup. Its configuration is by no means the optimal. 
# This is NOT the only solution by which you approach the problem. We highly encourage you to create
# your own setups. 

# Paths to different tools; 
map_path=/home/ubuntu/tools/mAP
export PATH=$map_path:$PATH

#echo "#####################################"
#echo "#       MED with MFCC Features      #"
#echo "#####################################"
sudo pip3 install imblearn
# iterate over the events
for feat_dim_mfcc in 50 100 150; do
  for event in P001 P002 P003; do
    echo "=========  Dim $feat_dim_mfcc Event $event  ========="
    # now train a svm model
    python scripts/train_svm.py $event mfcc $feat_dim_mfcc mfcc_pred_$feat_dim_mfcc/svm.$event.model || exit 1;
    # apply the svm model to *ALL* the testing videos;
    # output the score of each testing video to a file ${event}_pred 
    python scripts/test_svm.py mfcc_pred_$feat_dim_mfcc/svm.$event.model mfcc $feat_dim_mfcc mfcc_pred_$feat_dim_mfcc/${event}_mfcc_val.lst || exit 1;
    # compute the average precision by calling the mAP package
    ap list/${event}_val_label mfcc_pred_$feat_dim_mfcc/${event}_mfcc_val.lst
  done
done

echo ""
echo "#####################################"
echo "#       MED with ASR Features       #"
echo "#####################################"
# iterate over the events
feat_dim_asr=100
for event in P001 P002 P003; do
  echo "=========  Event $event  ========="
  # now train a svm model
  python scripts/train_svm.py $event asr $feat_dim_asr asr_pred_${feat_dim_asr}/svm.$event.model || exit 1;
  # apply the svm model to *ALL* the testing videos;
  # output the score of each testing video to a file ${event}_pred 
  python scripts/test_svm.py asr_pred/svm.$event.model asr $feat_dim_asr asr_pred/${event}_asr_val.lst || exit 1;
  # compute the average precision by calling the mAP package
  ap list/${event}_val_label asr_pred/${event}_asr_val.lst
done

echo ""
echo "#####################################"
echo "#    MED with SoundNet Features     #"
echo "#####################################"
# iterate over conv layer and the events
for event in P001 P002 P003; do
  echo "=========== Event $event ============"
  # now train a svm model
  python scripts/train_svm.py $event soundnet 0 soundnet_pred/svm.$event.model || exit 1;
  # apply the svm model to *ALL* the testing videos;
  # output the score of each testing video to a file ${event}_pred
  python scripts/test_svm.py soundnet_pred/svm.$event.model soundnet 0 soundnet_pred/${event}_soundnet_val.lst || exit 1;
  # compute the average precision by calling the mAP package
  ap list/${event}_val_label soundnet_pred/${event}_soundnet_val.lst
done

echo ""
echo "#####################################"
echo "#           Ensemble Model          #"
echo "#####################################"
# ensemble model
python scripts/ensemble.py mfcc_pred_50,mfcc_pred_100,asr_pred_50,asr_pred_100 ensemble
for event in P001 P002 P003; do 
  ap list/${event}_val_label ensemble/${event}_best_val.lst
done


echo ""
echo "Successfully Completed!"
echo ""

