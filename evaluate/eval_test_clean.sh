#!/bin/bash

NAME=$1
GPU=$2
TAG=$3

if [ -z "$TAG" ]; then
    TAG="latest"
fi

CONFIG_DIR=conf/stage2_denoising/
CKPT_DIR=/data2/yoongi/NoiseRobustVRVQ/stage2/ ## Checkpoint directory
CKPT_PATH=${CKPT_DIR}/${NAME}

# SAVE_RESULT_DIR=evaluate/results/$NAME
SAVE_RESULT_DIR=evaluate/results/

CUDA_VISIBLE_DEVICES=${GPU} \
python evaluate/eval_ears_clean_test.py \
--args.load ${CONFIG_DIR}/${NAME}.yml \
--save_path ${CKPT_PATH} \
--tag ${TAG} \
--save_result_dir ${SAVE_RESULT_DIR}
# --infer_clean_without_denoising  ## With FeatureDenoiser or not. 
# --cal_visqol ## visqol calculation takes too long time.