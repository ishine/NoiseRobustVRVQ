# Towards Bitrate-Efficient and Noise-Robust Speech Coding with Variable Bitrate RVQ


**Official implementation** of “Towards Bitrate-Efficient and Noise-Robust Speech Coding with Variable Bitrate RVQ” (INTERSPEECH 2025).

**Paper Link**: *To be added*



## 🔊 Audio Samples  
Model outputs (audio samples & importance maps) are available here:  
[sample_link](https://yoongi43.github.io/noise_robust_vrvq/)

---

## ⚙️ Environment Setup  

| Item                 | Command / Details                                                                                                          |
|----------------------|----------------------------------------------------------------------------------------------------------------------------|
| **Python**           | 3.12                                                                                                                       |
| **Create Conda env** | `conda create -n noise_vrvq python=3.12`                                                                                   |
| **Activate env**     | `conda activate noise_vrvq`                                                                                                |
| **PyTorch**          | `conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia`                  |
| **Mamba-SSM**        | `pip install mamba-ssm==1.2.0.post1 --no-build-isolation`                                                                   |
| **Other deps**       | `pip install -r requirements.txt`                                                                                          |

---

## 📂 Dataset  
- **EARS-WHAM** benchmark dataset (used for training & evaluation)  
- Download: [github.com/sp-uhh/ears_benchmark](https://github.com/sp-uhh/ears_benchmark)

---

## 🚀 Training Example  
Training script can be run with the following command:
```
Ex) 16 kHz VBR model training, with GPU idx 0

## Stage 1 training
$ bash scripts/stage1_train_clean_script.sh VBR_16k 0

## Stage 2 training
$ bash scripts/stage2_train_feat_dn_script.sh VBR_feat_denoise_16k 0
```
Other experiment configurations can be found in the `conf/stage1_clean_recon` and `conf/stage2_denoising` directories.

## 🧪 Evaluation
To evaluate on the EARS-WHAM test set, run:
```
Ex) 16 kHz VBR model inference, with GPU idx 0

$ bash evaluate/eval_test.sh VBR_feat_denoise_16k 0
```
Results (CSV of evaluation metrics) will be saved to:

```
evaluate/results/${EXPNAME}/evaluation_results.csv
```

## 🎧 Inference example
See `evaluate/infer_audio.py` for the inference script. 
By default it processes samples from the test loader, but you can adapt it to run on a single file. The script:

1. Reconstructs clean audio from noisy input

2. Saves output audio to evaluate/output_audio/${EXPNAME}

3. Saves importance maps at each quantization level

To run the provided example:
```
bash evaluate/infer_audio_example.sh VBR_feat_denoise_16k 0
```

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