#!/usr/local/bin/bash
P="python"
PROGRAM="lib27/classifier.py"

######### START CONSTANTS #########
RESTORE_PATH="models/vgg16a/vgg16_a_checkpoint-249"
META_PATH="models/vgg16a/vgg16_a_checkpoint-249.meta"
#META_PATH="models/vgg16a/vgg16a_graphdef.bin"
IMAGE_PATH="data/preprocessed/all/X_test.npy"
LABELS_PATH="data/preprocessed/all/y_test.npy"
######### END CONSTANTS #########

######### START VARIABLES #########
######### END VARIABLES #########


#############################
# SET UP ENVIRONMENT
#############################
echo "pwd: "
pwd
echo "PATH: "
echo $PATH
echo "PYTHONPATH: "
echo $PYTHONPATH
echo "LD_LIBRARY_PATH: "
echo $LD_LIBRARY_PATH
echo "CPATH: "
echo $CPATH
echo "LIBRARY_PATH: "
echo $LIBRARY_PATH
echo "CUDNN_PATH: "
echo $CUDNN_PATH

hostname
nvidia-smi
nvcc --version
#############################
# EXECUTION
#############################
time "$P" "$PROGRAM" \
"$META_PATH" \
"$RESTORE_PATH" \
"$IMAGE_PATH" \
"$LABELS_PATH" \
-v