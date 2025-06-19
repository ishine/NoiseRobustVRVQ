import argparse
import os; opj = os.path.join
from glob import glob

import numpy as np
import torch
from einops import rearrange

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from data.loaders import AudioLoader_EARS_Piared
from model.dac_vrvq import DAC_VRVQ_FeatureDenoise
from model.utils import cal_bpf_from_mask, generate_mask_hard, cal_metrics, cal_metrics_full, apply_straight
import matplotlib.pyplot as plt
from matplotlib import gridspec
import librosa.display

import warnings
from audiotools import AudioSignal
import math

## SISDR
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio, PerceptualEvaluationSpeechQuality


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for Noise Robust VRVQ")
    parser.add_argument("--ckpt-base-dir", type=str, 
                        help="Base directory for checkpoints")
    parser.add_argument("--exp-name", type=str, 
                        default="VBR_feat_denoise_16k",
                        help="Experiment name, without .yml")
    parser.add_argument("--tag", type=str,
                        default="latest",
                        help="Tag for the model checkpoint")
    parser.add_argument('--data-dir', type=str, 
                        default="/data2/yoongi/dataset/EARS_dataset/ears_benchmark/ears_wham/EARS-WHAM/test/",
                        help="Directory containing the audio data")
    parser.add_argument('--output-dir', type=str,
                         default='evaluate/output_samples',
                         help="Directory to save the output audio files")
    parser.add_argument('--device', type=str, default=7,
                        help="Device to run the inference on (e.g., 'cpu', 'cuda:0')")
    
    
    return parser.parse_args()

def infer_audio(
    *,
    model: DAC_VRVQ_FeatureDenoise, 
    audio_noisy, 
    audio_clean, 
    output_dir, 
    sample_name=None,
    vrvq_levels=[1], ## 0.2 ~ 6
    device):
    """
    infer single audio 
    """
    
    n_codebooks = model.n_codebooks
    # bits_per_codebook = math.ceil(np.log2(n_codebooks))
    bit_per_codebook = np.log2(model.quantizer.codebook_size)
    
    sr = model.sample_rate
    audio_noisy = audio_noisy.to(device)
    audio_clean = audio_clean.to(device)
    audio_noisy = model.preprocess(audio_noisy, sr) # just adjust length
    audio_clean = model.preprocess(audio_clean, sr) 
    
    # enc_out, fmaps = model.encoder(
    #     x_noisy=audio_noisy,
    #     x_gt=None
    # )
    # feat_enc = fmaps["noisy"]["imp_map_input"]
    # assert model.imp_map_input=="feature"
    # assert model.quantizer.operator_mode=="scaling"
    # imp_map = model.quantizer.imp_subnet(feat_enc) # imp_map:(B, 1, T)
    
    output = model.encode(
        audio_data_noisy=audio_noisy,
        audio_data_gt=None,
        n_quantizers=n_codebooks+1,
        level=1, # dummy value
    )
    # imp_map=output["imp_map"]
    # imp_net_input: (B, 1024, T)
    codes_fixed = output["codes"]

    _, _, _, z_q_stack_fixed = model.quantizer.from_codes(
        codes_fixed, return_z_q_is=True
    )
    
    if sample_name is not None:
        sample_out_dir = opj(output_dir, sample_name)
    else:
        count = 0
        while True:
            sample_out_dir = opj(output_dir, f"sample_{count:07d}")
            if not os.path.exists(sample_out_dir):
                break
            else:
                count += 1
    os.makedirs(sample_out_dir, exist_ok=True)
    
    ## save clean, noisy 
    save_from_audio(
        AudioSignal(audio_clean, sample_rate=sr),
        sample_out_dir,
        "clean"
    )
    save_from_audio(
        AudioSignal(audio_noisy, sample_rate=sr),
        sample_out_dir,
        "noisy"
    )
    
    if model.model_type == "VBR":
        imp_net_input=output["enc_fmaps"]["noisy"]["imp_map_input"]
        imp_map = model.quantizer.imp_subnet(imp_net_input) # imp_map:(B, 1, T)
        for level in vrvq_levels:
            imp_map_scaled = imp_map * n_codebooks * level
            mask_map = generate_mask_hard(
                x=imp_map_scaled,
                nq=n_codebooks
            ) # (B, Nq, T)
            # bpf = cal_bpf_from_mask(
            #     mask_map,
            #     bits_per_codebook=[bit_per_codebook] * n_codebooks,
            # )
            z_q_masked = z_q_stack_fixed * rearrange(mask_map, "b nq t -> b nq 1 t")
            z_q_masked_sum = torch.sum(z_q_masked, dim=1) # (B, C, T)
            recons = model.decoder(z_q_masked_sum) 
            
            signal_clean = AudioSignal(audio_clean, sample_rate=sr)
            signal_recons = AudioSignal(recons, sample_rate=sr)
            signal_noisy = AudioSignal(audio_noisy, sample_rate=sr)
            
            # sisdr = cal_metrics(signal_recons, signal_clean, state=None, loss_fn="SI-SDR")
            # pesq = cal_metrics(signal_recons, signal_clean, state=None, loss_fn="PESQ")
            
            level_out_dir = opj(sample_out_dir, f"level_{level:.2f}")
            os.makedirs(level_out_dir, exist_ok=True)
            save_results( ## impmaps, audios => cal metrics inside.
                mask_map=mask_map,
                cbr_n_codebooks=None,
                bits_per_codebook=[bit_per_codebook] * n_codebooks,
                downsample_ratio = np.prod(model.encoder_rates),
                signal_noisy=signal_noisy,
                signal_clean=signal_clean,
                signal_recons=signal_recons,
                output_dir = level_out_dir,
            )
    else:
        for nc in range(n_codebooks):
            # z_q_stack_fixed: (B, Nq, C, T)
            z_q_sum = torch.sum(z_q_stack_fixed[:, :nc+1], dim=1) # (B, C, T)
            recons = model.decoder(z_q_sum)
            
            signal_clean = AudioSignal(audio_clean, sample_rate=sr)
            signal_recons = AudioSignal(recons, sample_rate=sr)
            signal_noisy = AudioSignal(audio_noisy, sample_rate=sr)
            
            nc_out_dir = opj(sample_out_dir, f"CBR_Codebook_{nc+1:02d}")
            os.makedirs(nc_out_dir, exist_ok=True)
            
            ## save recons, metrics
            save_results(
                mask_map=None,  # No mask map in CBR mode
                cbr_n_codebooks=nc + 1,  # Number of codebooks used in CBR mode
                bits_per_codebook=[bit_per_codebook] * (nc + 1),  # Bits per codebook
                downsample_ratio = np.prod(model.encoder_rates),
                signal_noisy=signal_noisy,
                signal_clean=signal_clean,
                signal_recons=signal_recons,
                output_dir=nc_out_dir,
            )
            
        
        

def save_results(
    mask_map,
    cbr_n_codebooks, 
    bits_per_codebook,
    downsample_ratio,
    signal_noisy: AudioSignal,
    signal_clean: AudioSignal,
    signal_recons: AudioSignal,
    output_dir,
):
    n_codebooks = len(bits_per_codebook)
    
    sr = signal_noisy.sample_rate
    bits_transmit = math.ceil(math.log2(n_codebooks))
    
    sisdr = cal_metrics(signal_recons, signal_clean, state=None, loss_fn="SI-SDR")
    pesq = cal_metrics(signal_recons, signal_clean, state=None, loss_fn="PESQ")
    
    if mask_map is not None:
        assert mask_map.shape[1] == n_codebooks
        bpf = cal_bpf_from_mask(
            mask_map,
            bits_per_codebook=bits_per_codebook,
        )
        bps = (bpf + bits_transmit) * (sr / downsample_ratio) # sampling rate in latent domain
        kbps = bps / 1000.0
    else: ## CBR mode
        assert cbr_n_codebooks is not None
        bpf = sum(bits_per_codebook[:cbr_n_codebooks])
        bps = bpf * (sr / downsample_ratio) # sampling rate in latent domain
        kbps = bps / 1000.0
    
    with open(opj(output_dir, "metrics.txt"), "w") as f:
        f.write(f"SI-SDR: {sisdr:.2f}\n")
        f.write(f"PESQ: {pesq:.2f}\n")
        f.write(f"kbps: {kbps:.2f}\n")
        
    ## save audio, spec
    # save_from_audio(signal_clean, output_dir, "clean")
    # save_from_audio(signal_noisy, output_dir, "noisy")
    save_from_audio(signal_recons, output_dir, "recons")
    
    ## save imp_map
    if mask_map is not None:
        mask_map = mask_map.cpu().numpy()[0] # (B, Nq, T)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(mask_map, cmap='viridis', interpolation='none', aspect='auto')
        ax.set_yticks(np.arange(n_codebooks))
        ax.invert_yaxis()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(True)
        plt.savefig(opj(output_dir, "imp_map.png"), dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        
    print(f"Results saved to {output_dir}")
    
    
    
def save_from_audio(signal:AudioSignal, output_dir, name):
    # save .wav
    sig_cpu = signal[0].detach().cpu()
    sig_cpu.write(opj(output_dir, f"{name}.wav"))
    
    # save spec
    ref = sig_cpu.magnitude.max()
    logmag = sig_cpu.log_magnitude(ref_value=ref)
    logmag = logmag.numpy()[0][0] # (257, T)
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(
        logmag,
        x_axis='time',
        y_axis='linear',
        sr=signal.sample_rate,
        win_length=sig_cpu.stft_params.window_length,
        hop_length=sig_cpu.stft_params.hop_length,
        ax=ax,
    )
    # ax.set_yticks([])
    plt.tight_layout()
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(opj(output_dir, f"{name}_spec.png"), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    

if __name__=="__main__":

    warnings.filterwarnings("ignore", category=UserWarning)
    
    args = parse_args()
    
    output_dir = opj(args.output_dir, args.exp_name)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    
    # args.ckpt_base_dir = "/data2/yoongi/NoiseRobustVRVQ/" # default
    ckpt_dir = opj(args.ckpt_base_dir, "stage2", 
                   args.exp_name, args.tag, "dac_vrvq_featuredenoise")
    
    ckpt = torch.load(opj(ckpt_dir, "weights.pth"), map_location="cpu")
    weights = ckpt["state_dict"]
    config = ckpt["metadata"]["kwargs"]
    
    model = DAC_VRVQ_FeatureDenoise(**config)
    model.load_state_dict(weights, strict=True)
    model.eval()
    model.to(device)
    
    
    sr = model.sample_rate
    
    """
    We will use EARS dataset for inference example,
    but you can use any single audio file or dataset by fixing this code.
    """
    loader = AudioLoader_EARS_Piared(
        srcs_clean=[opj(args.data_dir, "clean")],
        srcs_noisy=[opj(args.data_dir, "noisy")],
        shuffle=False,
    )
    num_items = len(loader.clean_list)
    
    # item_indices = np.arange(num_items)
    num_indices = 30
    state = np.random.RandomState(0)
    # item_indices = np.random.choice(num_items, num_indices, replace=False)
    item_indices = state.choice(num_items, num_indices, replace=False)
    for idx in item_indices:
        ## load item from EARS-WHAM test loader.
        ## You can change this code into single audio file loading code.
        item = loader(
            state=state,
            sample_rate=sr,
            duration=10.0,
            loudness_cutoff=-35,
            num_channels=1,
            offset=None,
            item_idx=idx,
        )
        """
        => item.keys():
        signal_clean: AudioSignal
        signal_noisy: AudioSignal
        item_idx: item_idx,
        path_clean: str,
        path_noisy: str,
        """
        signal_noisy = item["signal_noisy"]
        signal_clean = item["signal_clean"]
        audio_noisy = signal_noisy.audio_data # (1, 1, len)
        audio_clean = signal_clean.audio_data
        audio_noisy = audio_noisy.to(device)
        audio_clean = audio_clean.to(device)
        
        sample_path = item["path_clean"]
        # sample_name = os.path.basename(sample_path).replace(".wav", "")
        sample_name = sample_path.split("/")[-2] + "_" + sample_path.split("/")[-1].replace(".wav", "")
        
        infer_audio(
            model=model,
            audio_noisy=audio_noisy,
            audio_clean=audio_clean,
            output_dir=output_dir,
            sample_name=sample_name,
            vrvq_levels=[0.5, 1, 2, 4, 6],
            device=device
        )