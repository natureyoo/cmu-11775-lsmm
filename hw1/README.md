### cmu-11775-lsmm
This is the pipeline for video classification.
It extract 3 types of features, MFCC, ASR and soundnet.
#
#
# Data Required
To extract ASR feature, you should have asr transcription dataset.
To extract Soundnet feature, you should have the intermediate layer output for all videos using pretrained soundnet model. In this pipeline, we use the output of conv3, conv4, conv6, conv8 layers and save it in each directory soundnet/conv_03, 04, 06, 08.
#
#
#
# Python Required
Python2.7.12 for select_frames.py / train_kmeans.py / create_kmeans.py
Python3.6.6 for create_asrfeat.py / create_soundnet.py / train_svm.py / test_svm.py / ensemble.py
#
#
#
# Python Library Required
sklearn
imblearn
nltk
gensim
scipy
#
#
#
# How to run
sh run.feature.sh
sh run.med.sh

# Link to JIRA (test)
