#!/usr/local/bin/bash
P="python"
PROGRAM="lib27/classifier.py"

######### START CONSTANTS #########
#RESTORE_PATH="models/vgg16a/vgg16_a_checkpoint-249"
#META_PATH="models/vgg16a/vgg16_a_checkpoint-249.meta"
#RESTORE_PATH="models/vgg10a/vgg10_a_checkpoint-157"
#META_PATH="models/vgg10a/vgg10_a_checkpoint-157.meta"
RESTORE_PATH="models/vgg6t/vgg6t_a_checkpoint-57"
META_PATH="models/vgg6t/vgg6t_a_checkpoint-57.meta"
#RESTORE_PATH="models/vgg6f/vgg6f_a_checkpoint-25"
#META_PATH="models/vgg6f/vgg6f_a_checkpoint-25.meta"
#RESTORE_PATH="models/vgg5/vgg5_a_checkpoint-XX"
#META_PATH="models/vgg5/vgg5_a_checkpoint-XX.meta"


#META_PATH="models/vgg16a/vgg16a_graphdef.bin"
IMAGE_PATH="data/rgb8.npy"
#IMAGE_PATH="data/CORRECT.npy"
#IMAGE_PATH="data/preprocessed/all/X_test.npy"
#IMAGE_PATH="data/raw/all/000515.jpg"
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