import argbind
import os
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
from einops import rearrange

from audiotools import AudioSignal
from audiotools import ml

import math

sys.path.append(str(Path(__file__).parent.parent))

from data.loaders import AudioLoader_EARS_Clean


from model.dac_vrvq import DAC_VRVQ_FeatureDenoise
from model.utils import cal_bpf_from_mask, generate_mask_hard, cal_metrics, cal_metrics_full, apply_straight


import matplotlib.pyplot as plt
from matplotlib import gridspec
import librosa
import time
from model import loss
from tqdm import tqdm
import pandas as pd


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
torch.backends.cudnn.benchmark = bool(int(os.getenv("CUDNN_BENCHMARK", 1)))
Accelerator = argbind.bind(ml.Accelerator, without_prefix=True)

## Model
DAC_VRVQ_FeatureDenoise = argbind.bind(DAC_VRVQ_FeatureDenoise)

## Test loader
AudioLoader = argbind.bind(AudioLoader_EARS_Clean, "test")

@argbind.bind("test")
def build_loader(
    folders: dict = None,
):
    loader = AudioLoader(
        srcs_clean = folders["clean"],
    )
    return loader

@dataclass
class State:
    generator: DAC_VRVQ_FeatureDenoise
    test_loader: AudioLoader
    tag:str
    batch_size: int = None
    downsample_ratio: int = 512
    

@argbind.bind(without_prefix=True)
def load(
    args,
    accel: ml.Accelerator,
    save_path: str,
    tag: str = "latest",
    # load_weights: bool = True,
):
    # kwargs = {
    #     "folder": f"{save_path}/{tag}",
    #     "map_location": "cpu",
    #     "package": False,
    # }
    model_dir = f"{save_path}/{tag}"
    ## DAC_VRVQ_FeatureDenoise
    if (Path(model_dir) / "dac_vrvq_featuredenoise").exists():
        print('### Loading from folder: ',  (Path(model_dir) / "dac_vrvq_featuredenoise"))
        generator = DAC_VRVQ_FeatureDenoise()
        ckpt_gen = Path(model_dir) / "dac_vrvq_featuredenoise" / "weights.pth"
        ckpt_gen = torch.load(ckpt_gen, map_location="cpu")
        generator.load_state_dict(ckpt_gen["state_dict"], strict=True)
    else:
        raise ValueError(f"Unknown model type in {model_dir}. ")
    
    generator = accel.prepare_model(generator)
    sample_rate = accel.unwrap(generator).sample_rate
    with argbind.scope(args, "test"):
        test_loader = build_loader()
    

    downsample_ratio = np.prod(generator.encoder_rates)
    
    return State(
        generator=generator,
        test_loader=test_loader,
        tag=tag,
        downsample_ratio=downsample_ratio
    )


# @timer()
@torch.no_grad()
def test_loop(signal_clean, state, save_figs=None, cal_visqol=True, infer_clean_without_denoising=False):
    ## samples: AudioSignal, (1, 1, T)
    
    st = time.time()
    state.generator.eval()
    device = state.generator.device
    signal_clean = signal_clean.to(device)

    audio_gt = signal_clean.audio_data
    audio_length = audio_gt.shape[-1]
    
    # audio_noisy = signal_noisy.audio_data
    sr = signal_clean.sample_rate

    out_fixed = state.generator(audio_data_noisy=audio_gt,
                                audio_data_clean=audio_gt,
                                sample_rate=sr,
                                n_quantizers=state.generator.n_codebooks+1,
                                infer_clean_without_denoising=infer_clean_without_denoising,
                                )   
    
    decoder = state.generator.decode
    quantizer = state.generator.quantizer
    codebook_size = state.generator.quantizer.codebook_size
    n_codebooks = len(quantizer.quantizers)
    
    # out: audio, z, codes, latents, vq/commitment_loss, vq/codebook_loss, imp_map, mask_imp

    ## Calculate VBR codebooks with importance map
    codes_fixed = out_fixed["codes"]
    

    ## Plot R-D curves of evaluation metrics
    num_levels = 12
    level_min = 0.2
    level_max = 6
    range_list = np.linspace(0, 1, num_levels)
    level_list = range_list * (math.log(level_max) - math.log(level_min)) + math.log(level_min)
    level_list = np.exp(level_list)

    level_list_figs = level_list[1::2]
    bits_per_frame_1c = math.ceil(math.log2(codebook_size)) # 1024=10bit
    
    if save_figs:
        os.makedirs(save_figs, exist_ok=True)
        save_idx = 0
        save_path_png = f'{save_figs}/mask_spec_{save_idx}.png'
        while os.path.exists(save_path_png):
            save_idx += 1
            save_path_png = f'{save_figs}/mask_spec_{save_idx}.png'    
            
        # fig, axes = plt.subplots(len(level_list_figs)+2, 1, figsize=(8, 20))            
        # fig = plt.figure(figsize=(2, 4))
        # fig = plt.figure(figsize=(8, 20))
        fig = plt.figure(figsize=(3, 10))
        add_figs = 2
        gs = gridspec.GridSpec(nrows=len(level_list_figs)+add_figs,## +1 for noisy plot only
                                ncols=1,
                                height_ratios=[1 for _ in range(len(level_list_figs))]+[2 for _ in range(add_figs)],
        )
            
        for jj, level in enumerate(level_list_figs):
            out_lev = state.generator(audio_data_noisy=audio_gt,
                                    audio_data_clean=audio_gt,
                                    sample_rate=sr,
                                    n_quantizers=state.generator.n_codebooks+1,
                                    level=level,
                                    infer_clean_without_denoising=infer_clean_without_denoising,
                                    )
            msk = out_lev["mask_imp"][0]
            
            bpf = cal_bpf_from_mask(msk.unsqueeze(0), 
                                    bits_per_codebook=[bits_per_frame_1c] * n_codebooks)
            recon = out_lev["audio"]
            recon = AudioSignal(recon, sr)
        
            loss_pesq = cal_metrics(recon, signal_clean, state, "PESQ")
            msk = msk.cpu().numpy()

            ax = plt.subplot(gs[jj])
            ax.imshow(msk, cmap='viridis', interpolation='none', aspect='auto')
            ax.set_yticks(np.arange(0, n_codebooks))
            ax.invert_yaxis()
            bps = (bpf+bits_transmit) * sr / state.downsample_ratio
            kbps = bps/1000
            ax.set_title(f'{loss_pesq:.2f} || {kbps:.2f}kbps')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_yticks([])
            if jj == len(level_list_figs) -1 :
                # ax.get_xaxis().set_visible(True)
                ax.set_xticks([])
            else:
                ax.set_xticks([])
                
        ax = plt.subplot(gs[-2])
        recon_full = out_fixed["audio"]
        recon_full = AudioSignal(recon_full, sr)
        recon_ad = recon_full[0].cpu() 
        ref_re = recon_ad.magnitude.max()
        logmag_re = recon_ad.log_magnitude(ref_value=ref_re)
        logmag_re = logmag_re.numpy()[0][0]
        librosa.display.specshow(
            logmag_re,
            x_axis='time',
            sr=sr,
            # ax=axes[-1],
            ax=ax
        )
        downsample_ratio = state.downsample_ratio
        loss_pesq = cal_metrics(recon_full, signal_clean, state, "PESQ")
        kbps = sr/downsample_ratio*math.log2(codebook_size) * n_codebooks /1000
        ax.set_title(f'FC: {loss_pesq:.2f} || {kbps:.2f}kbps')
        ax.set_yticks([])
        
        ### PLOT spec of input audio
        ax = plt.subplot(gs[-1])
        ad = signal_clean[0].cpu()
        ref = ad.magnitude.max()
        logmag = ad.log_magnitude(ref_value=ref)
        # logmag = logmag.numpy()[0].mean(axis=0)
        logmag = logmag.numpy()[0][0]
        librosa.display.specshow(
            logmag,
            x_axis='time',
            # y_axis='linar',
            sr=sr,
            # ax=axes[-1]
            ax=ax
        )
        ax.set_title(f'Clean Spec')
        ax.set_yticks([])
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=1)
        plt.savefig(save_path_png, bbox_inches='tight', pad_inches=0)
        plt.close()
            
    _, _, _, z_q_stack_fixed = quantizer.from_codes(codes_fixed, return_z_q_is=True) 
    ## => z_q_stack_fixed: (B, Nq, 1024, T)
    ## codes: (B, Nq, T)
    
    """ Calculate metrics"""
    output = {}

    """ CBR Mode Evaluation"""
    output["CBR_mode"] = {}
    for i in range(n_codebooks):
        z_q_sum = torch.sum(z_q_stack_fixed[:, :i+1], dim=1)
        recons = decoder(z_q_sum)
        # length = min(recons.shape[-1], signal_noisy.shape[-1], signal_clean.shape[-1])
        length = min(recons.shape[-1], signal_clean.shape[-1])
        recons = recons[..., :length]
        signal_clean = signal_clean[..., :length]
        # recons = recons[..., :signal_noisy.audio_data.shape[-1]]
        recons = AudioSignal(recons, sr)

        loss_dict = cal_metrics_full(
            recons=recons,
            signal=signal_clean,
            cal_visqol=cal_visqol,
        )
        # bpf = np.sum(bits_per_codebook[:i+1]).item()
        bpf = bits_per_frame_1c * (i + 1)
    
        output["CBR_mode"][f"Nq_{i}"] = {}

        output["CBR_mode"][f"Nq_{i}"]["bpf"] = bpf
        output["CBR_mode"][f"Nq_{i}"]["metrics"] = loss_dict
        
    ## Variable codebook:
    #### simple thresholding / sampling
    """ VBR Mode Evaluation"""
    if state.generator.model_type == "VBR":
        output["VBR_mode"] = {}

        bits_transmit = math.ceil(math.log2(n_codebooks))
        audio_gt_proc = state.generator.preprocess(audio_gt, sr)
        # audio_noisy_proc = state.generator.preprocess(audio_noisy, sr)
        # out_enc, feat_enc = state.generator.encoder(audio_noisy_proc, return_feat=True) # feat_enc: (B, 1024, T)
        outs_enc, fmaps = state.generator.encoder(x_noisy=audio_gt_proc,
                                                 x_gt=audio_gt_proc,
                                                 )
        if infer_clean_without_denoising:
            feat_enc = fmaps["gt"]["imp_map_input"]
        else:
            feat_enc = fmaps["noisy"]["imp_map_input"]
        if state.generator.imp_map_input=="feature":
            imp_map = state.generator.quantizer.imp_subnet(feat_enc) # imp_map: (B, 1, T)
        elif state.generator.imp_map_input=="zqis":
            zqis = rearrange(z_q_stack_fixed, 'b nq d t -> b (d nq) t')
            imp_map = state.generator.quantizer.imp_subnet(zqis) # imp_map: (B, 1, T)
        operator_mode = quantizer.operator_mode

        for level in level_list:
            if operator_mode=="scaling":
                imp_map_scaled = imp_map * level * n_codebooks
            elif operator_mode == "exponential":
                imp_map_scaled = n_codebooks * torch.pow(imp_map, 1/level)
            elif operator_mode == "transformed_scaling":
                imp_map_scaled = apply_straight(imp_map, level, n_codebooks)
            else:
                raise ValueError(f"Unknown operator mode: {operator_mode}")
            
            mask_map = generate_mask_hard(
                x=imp_map_scaled,
                nq=n_codebooks,
            ) # (B, Nq, T)
            
            bpf = cal_bpf_from_mask(mask_map, 
                                    bits_per_codebook=[bits_per_frame_1c] * n_codebooks)
            ### z_q_stack_fixed: (B, Nq, 1024, T)
            z_q_masked = z_q_stack_fixed * rearrange(mask_map, 'b nq t -> b nq 1 t')
            z_q_masked_sum = torch.sum(z_q_masked, dim=1)
            recons = decoder(z_q_masked_sum)
            # length = min(recons.shape[-1], signal_noisy.shape[-1], signal_clean.shape[-1])
            length = min(recons.shape[-1], signal_clean.shape[-1])
            recons = recons[..., :length]
            signal_clean = signal_clean[..., :length]
            # recons = recons[..., :signal_noisy.audio_data.shape[-1]]
            recons = AudioSignal(recons, signal_clean.sample_rate)

            loss_dict = cal_metrics_full(
                recons=recons,
                signal=signal_clean,
                cal_visqol=cal_visqol,
            )
            output["VBR_mode"][f"level_{level:.3f}"] = {}
            output["VBR_mode"][f"level_{level:.3f}"]["bpf"] = bpf+bits_transmit
            output["VBR_mode"][f"level_{level:.3f}"]["metrics"] = loss_dict

    return output, codes_fixed


@argbind.bind(without_prefix=True)
def eval(
    args,
    accel: ml.Accelerator,
    save_path: str = "ckpt",
    save_result_dir: str = None,
    cal_visqol: bool = False,
    infer_clean_without_denoising: bool = False,
):
    print(f"Evaluating Clean Test with args: {args}")
    print(f"## Infer clean without denoising: {infer_clean_without_denoising}")
    state = load(args, accel, save_path)
    assert save_result_dir is not None
    
    name = os.path.basename(save_path).split('.')[0]
    # save_path = Path(save_result_dir) / f'{name}_{state.tag}'
    if infer_clean_without_denoising:
        save_path = Path(save_result_dir) / f'{name}_{state.tag}_clean_woFD'
    else:
        save_path = Path(save_result_dir) / f'{name}_{state.tag}_clean_wFD'
    
    """If there is csv file, load it."""
    os.makedirs(save_path, exist_ok=True)
    csv_path_cur = save_path / "evaluation_results_cur.csv"
    try:
        df_load = pd.read_csv(csv_path_cur)
        # df_load = pd.read_csv(save_path / "evaluation_results_cur.csv")
        print(f"Loaded evaluation results from {csv_path_cur}")
        ## find the last idx
        start_idx = df_load["sample_idx"].max()+1
        print(f"Index of the last sample: {start_idx}")
        
    except:
        df_load = None
        start_idx = 0
        print(f"Could not find evaluation results from {csv_path_cur}")
        
    model_type = state.generator.model_type
    if model_type == "VBR":
        save_figs_path = os.path.join(save_path, 'maps')
        # os.makedirs(save_figs_path, exist_ok=True)
        
    sr = state.generator.sample_rate
    test_loader = state.test_loader
    
    # save_indices = [4, 8, 16, 20, 24, 28, 32] ## indices to save Plots (importance maps)
    save_indices = []

    df_results = []

    num_test_set = len(test_loader.clean_list)
    print("## Test Sample Length:", num_test_set)

    for idx in tqdm(range(start_idx, num_test_set),
                    desc=f"Eval {save_path}"):
        # if idx > 10:
        #     break ## for debugging
        items = test_loader(
            state=None,
            sample_rate=sr,
            duration=None,
            loudness_cutoff=None,
            num_channels=1,
            offset=None,
            item_idx=idx
        )
        signal_clean = items["signal_clean"]
        
        if idx in save_indices and model_type == "VBR":
            save_figs = save_figs_path
            print("Save figs:", idx, save_figs_path)
            os.makedirs(save_figs, exist_ok=True)
        else:
            save_figs = None
        
        output, codes = \
            test_loop(
                signal_clean=signal_clean,
                state=state,
                save_figs=save_figs,
                cal_visqol=cal_visqol,
                infer_clean_without_denoising=infer_clean_without_denoising,
            )

        ## CBR mode outputs
        for nq_key, values in output["CBR_mode"].items():
            nq = int(nq_key.split("_")[-1])  # "Nq_0" → 0
            bpf = values["bpf"]
            for metric, value in values["metrics"].items():
                df_results.append({
                    "sample_idx": idx,
                    "mode": "CBR",
                    "Nq": nq,
                    "level": None,
                    "bpf": bpf,
                    "metric": metric,
                    "value": value
                })

        ## VBR mode outputs (if model_type is VBR)
        for level_key, values in output.get("VBR_mode", {}).items():
            level = float(level_key.split("_")[-1])  # "level_0.200" → 0.2
            bpf = values["bpf"]
            for metric, value in values["metrics"].items():
                df_results.append({
                    "sample_idx": idx,
                    "mode": "VBR",
                    "Nq": None,
                    "level": level,
                    "bpf": bpf,
                    "metric": metric,
                    "value": value
                })

        df_cur = pd.DataFrame(df_results)
        df_cur.to_csv(csv_path_cur, index=False)

    ### Save results 

    df = pd.DataFrame(df_results)
    if df_load is not None:
        df = pd.concat([df_load, df], axis=0)
    os.makedirs(save_path, exist_ok=True)
    csv_path = save_path / "evaluation_results.csv"
    df.to_csv(csv_path, index=False)

    print(f"Saved evaluation results to {csv_path}")
    

if __name__=='__main__':
    args = argbind.parse_args()
    args['args.debug'] = int(os.getenv("LOCAL_RANK", 0)) == 0
    with argbind.scope(args):
        with Accelerator() as accel:
            if accel.local_rank != 0:
                sys.tracebacklimit = 0
            eval(args, accel)
    