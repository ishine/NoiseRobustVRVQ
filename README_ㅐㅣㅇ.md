
This repository contains the official implementation of the paper **Towards Bitrate-Efficient and Noise-Robust Speech Coding with Variable Bitrate RVQ**, accepted at **INTERSPEECH 2025**.

## 📄 Paper Link
TO BE ADDED

## 🔊 Audio Samples
Importance map and audio samples are available in the following link:
[link](https://yoongi43.github.io/noise_robust_vrvq/)

## ⚙️ Environment Setup
To set up the environment, follow these steps:
```
# Create a conda environment
conda create -n noise_vrvq python=3.12

# Activate the environment
conda activate noise_vrvq

# Install dependencies
## We used this command for PyTorch install
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia

## Install Mamba
pip install mamba-ssm==1.2.0.post1 --no-build-isolation

## Install other dependencies
pip install -r requirements.txt
```

## Dataset
WE used EARS-WHAM benchmark dataset [1] for training and test for our model.
It can be downloaded from the following [github](https://github.com/sp-uhh/ears_benchmark).

## Training example
Training script can be run with the following command:
```
Ex) 16 kHz VBR model training, with GPU idx 0

## Stage 1 training
$ bash scripts/stage1_train_clean_script.sh VBR_16k 0

## Stage 2 training
$ bash scripts/stage2_train_feat_dn_script.sh VBR_feat_denoise_16k 0
```
Other experiment configurations can be found in the `conf/stage1_clean_recon` and `conf/stage2_denoising` directories.

## Evaluation
Can inference EARS-WHAM test set with the following code:
and evaluation metrics results are saved in evaluate/results/${EXPNAME}/evaluation_results.csv
```
Ex) 16 kHz VBR model inference, with GPU idx 0

$ bash evaluate/eval_test.sh VBR_feat_denoise_16k 0
```

## Inference example
Please refer to the evaluate/infer_audio.py for inference code.
Code is loading audio files from test loader, buut Code can be fixed to inference on a single audio file. 
This code reconstruct noisy audio into clean audio, and save the output audio file in the evaluate/output_audio/${EXPNAME} directory.
It also saves the importance map of various levels. 

```
# running example can be found int evaluate/infer_audio_example.sh

# can be run with:
bash evaluate/infer_audio_example.sh VBR_feat_denoise_16k 0
```


## Quantization example


## 📌 Code Base  
Codes are mainly based on **DAC** [2], **VRVQ** [3] and **SEMamba** [4].  
- **DAC GitHub:** [DAC GitHub Link](https://github.com/descriptinc/descript-audio-codec)  
- **VRVQ GitHub:** [VRVQ GitHub Link](https://github.com/SonyResearch/VRVQ)  
- **SEMamba GitHub:** [SEMamba GitHub Link](https://github.com/RoyChao19477/SEMamba)


## 📝 References
[1] Julius Richter, Yi-Chiao Wu, Steven Krenn, Simon Welker, Bunlong Lay, Shinji Watanabe, Alexander Richard, Timo Gerkmann,
**EARS: An Anechoic Fullband Speech Dataset Benchmarked for Speech Enhancement and Dereverberation**,
*INTERSPEECH 2024*.
[Paper Link](https://arxiv.org/abs/2406.06185)

[2] Rithesh Kumar, Prem Seetharaman, Alejandro Luebs, Ishaan Kumar, Kundan Kumar,  
**High-Fidelity Audio Compression with Improved RVQGAN**,  
*Advances in Neural Information Processing Systems*, 2023.  
[Paper Link](https://arxiv.org/abs/2306.06546)

[3] Yunkee Chae, Woosung Choi, Yuhta Takida, Junghyun Koo, Yukara Ikemiya, Zhi Zhong, Kin Wai Cheuk, Marco A. Martínez-Ramírez, Kyogu Lee, Wei-Hsiang Liao, Yuki Mitsufuji, 
**Variable Bitrate Residual Vector Quantization for Audio Coding**,
*ICASSP 2025*.
[Paper Link](https://arxiv.org/abs/2410.06016)

[4] Rong Chao, Wen-Huang Cheng, Moreno La Quatra, Sabato Marco Siniscalchi, Chao-Han Huck Yang, Szu-Wei Fu, Yu Tsao,
**An Investigation of Incorporating Mamba for Speech Enhancement**,
*ICASSP 2025*.
[Paper Link](https://arxiv.org/abs/2405.06573)

## 📚 Citation

If you find our work useful, please cite (To Be Updated):

<!-- ```bibtex
@INPROCEEDINGS{chae2025robustvrvq,
    title = {Towards Bitrate-Efficient and Noise-Robust Speech Coding with Variable Bitrate RVQ},
    author = {Yunkee Chae and Kyogu Lee},
    year = {2025},
    booktitle = {Interspeech 2025},
    pages = {},
    doi = {},
    issn = {}
}
``` -->