#!/bin/bash

# ex) bash scripts/stage2_train_feat_dn_script.sh VBR_feat_denoise_16k 0
CONFIG_DIR=conf/stage2_denoising/
SAVE_DIR=/data2/yoongi/NoiseRobustVRVQ_mambatest/stage2/

EXPNAME=${1}
GPU=${2}

CUDA_VISIBLE_DEVICES=${GPU} \
taskset -c $((8*${GPU}))-$((8*${GPU}+7)) \
python scripts/stage2_train_feat_dn.py \
--args.load ${CONFIG_DIR}/${EXPNAME}.yml \
--save_path ${SAVE_DIR}/${EXPNAME}
# --resume true \ # Use this line to resume training.


# torchrun --nproc_per_node gpu scripts/train_imp_grad.py --args.load ${CONFIG_DIR}/${EXPNAME}.yml --save_path ${SAVE_DIR}/${EXPNAME} --resume