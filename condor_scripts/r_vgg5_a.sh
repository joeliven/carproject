#!/usr/local/bin/bash
P="python"
PROGRAM="lib/vgg5.py"

######### START CONSTANTS #########
BATCH_SIZE_INT="128"
NB_CHANNELS_INT="3"
DIM_IMG_INT="224"
TRAIN_LIM="-1"
VAL_LIM="256"
TEST_LIM="128"
SAVE_SUMMARIES_EVERY="10"
DISPLAY_EVERY="1"
DISPLAY="False"
NB_TO_DISPLAY="5"
NB_EPOCHS="250"
SAVE_BEST_ONLY="save_all" # 'save_best_train' or 'save_best_val'
# WEIGHTS_PATH="models/vgg/vgg16_weights_pretrained.npz"
WEIGHTS_PATH="/scratch/cluster/joeliven/carproject/models/vgg16a/vgg16_weights_pretrained.npz"
#RESTORE_PATH="/scratch/cluster/joeliven/carproject/models/vgg5/vgg5_a_checkpoint-99"
# SAVE_PATH="models/vgg"
SAVE_PATH="/scratch/cluster/joeliven/carproject/models/vgg5"
X_TRAIN="/scratch/cluster/joeliven/carproject/data/preprocessed/all/X_train.npy"
X_VAL="/scratch/cluster/joeliven/carproject/data/preprocessed/all/X_val.npy"
X_TEST="/scratch/cluster/joeliven/carproject/data/preprocessed/all/X_test.npy"
Y_TRAIN="/scratch/cluster/joeliven/carproject/data/preprocessed/all/y_train.npy"
Y_VAL="/scratch/cluster/joeliven/carproject/data/preprocessed/all/y_val.npy"
Y_TEST="/scratch/cluster/joeliven/carproject/data/preprocessed/all/y_test.npy"


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
--batch-size "$BATCH_SIZE_INT" \
--nb-channels "$NB_CHANNELS_INT" \
--dim-img "$DIM_IMG_INT" \
--val-lim "$VAL_LIM" \
--test-lim "$TEST_LIM" \
--save-summaries-every "$SAVE_SUMMARIES_EVERY" \
--display-every "$DISPLAY_EVERY" \
--nb-to-display "$NB_TO_DISPLAY" \
--nb-epochs "$NB_EPOCHS" \
--save-best-only "$SAVE_BEST_ONLY" \
--weights-path "$WEIGHTS_PATH" \
--save-path "$SAVE_PATH" \
--X-train "$X_TRAIN" \
--X-val "$X_VAL" \
--X-test "$X_TEST" \
--y-train "$Y_TRAIN" \
--y-val "$Y_VAL" \
--y-test "$Y_TEST" \
--train