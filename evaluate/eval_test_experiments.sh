#!/bin/bash

GPU=7
EPXs="CBR_feat_denoise_16k VBR_feat_denoise_16k CBR_feat_denoise_48k VBR_feat_denoise_48k"

for EPX in ${EPXs}
do
    echo "Running evaluation for ${EPX} on GPU ${GPU}"
    bash evaluate/eval_test.sh ${EPX} ${GPU}
done