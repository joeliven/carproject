#!/usr/local/bin/bash
P="python"
PROGRAM="lib/vgg1.py"

######### START CONSTANTS #########
BATCH_SIZE_INT="32"
NB_CHANNELS_INT="3"
DIM_IMG_INT="224"
TRAIN_LIM="1000"
VAL_LIM="20"
SAVE_SUMMARIES_EVERY="100"
DISPLAY_EVERY="1"
DISPLAY="False"
NB_TO_DISPLAY="5"
NB_EPOCHS="100"
SAVE_BEST_ONLY="save_all" # 'save_best_train' or 'save_best_val'
# LOAD_PATH="models/vgg/vgg16_weights_pretrained.npz"
LOAD_PATH="/scratch/cluster/joeliven/carproject/models/vgg/vgg16_weights_pretrained.npz"
# SAVE_PATH="models/vgg"
SAVE_PATH="/scratch/cluster/joeliven/carproject/models/vgg"
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
--train-lim "$TRAIN_LIM" \
--val-limm "$VAL_LIM" \
--save-summaries-every "$SAVE_SUMMARIES_EVERY" \
--display-every "%DISPLAY_EVERY" \
--display "$DISPLAY" \
--nb-to-display "$NB_TO_DISPLAY" \
--nb-epochs "$NB_EPOCHS" \
--save-best-only "$SAVE_BEST_ONLY" \
--load-path "$LOAD_PATH" \
--save-path "$SAVE_PATH"