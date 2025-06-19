#!/bin/bash

# EXP_NAME="VBR_feat_denoise_16k"
# EXP_NAME="CBR_feat_denoise_16k"
# EXP_NAME="VBR_feat_denoise_48k"
EXP_NAME="CBR_feat_denoise_48k"
DEVICE=7
# DEVICE=6

# EXP_NAME=$1
# DEVICE=$2

CKPT_BASE_DIR="/data2/yoongi/NoiseRobustVRVQ/"
TAG="latest"
DATA_DIR="/data2/yoongi/dataset/EARS_dataset/ears_benchmark/ears_wham/EARS-WHAM/test/"
OUTPUT_DIR="evaluate/output_samples"

python evaluate/infer_audio.py \
    --ckpt-base-dir $CKPT_BASE_DIR \
    --exp-name "${EXP_NAME}" \
    --tag "${TAG}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --device $DEVICE