#!/bin/bash

GPU=6
# EXPs="CBR_feat_denoise_16k VBR_feat_denoise_16k CBR_feat_denoise_48k VBR_feat_denoise_48k"
EXPs="VBR_feat_denoise_16k_modulation VBR_feat_denoise_48k_modulation"

for EXP in ${EXPs}
do
    echo "Running evaluation for ${EXP} on GPU ${GPU}"
    bash evaluate/eval_test.sh ${EXP} ${GPU}
done