#!/bin/bash

# CONFIG_DIR=conf/clean_recon_16k/
# SAVE_DIR=/data2/yoongi/dacruns_vbr_clean_16k
CONFIG_DIR=conf/stage1_clean_recon/
SAVE_DIR=/data2/yoongi/NoiseRobustVRVQ/stage1/

EXPNAME=${1} # CBR_16k or CBR_48k
GPU=${2} # 0

CUDA_VISIBLE_DEVICES=${GPU} \
taskset -c $((8*${GPU}))-$((8*${GPU}+7)) \
python scripts/stage1_train_clean.py \
--args.load ${CONFIG_DIR}/${EXPNAME}.yml \
--save_path ${SAVE_DIR}/${EXPNAME}

# torchrun --nproc_per_node gpu scripts/train_imp_grad.py --args.load ${CONFIG_DIR}/${EXPNAME}.yml --save_path ${SAVE_DIR}/${EXPNAME} --resume