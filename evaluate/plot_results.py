import argparse
import os; opj = os.path.join
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import re
import math
import seaborn as sns
import json
from copy import deepcopy

from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore
import pandas as pd

## 16k Experiments
SAMPLE_RATE = 16000
EXP_NAMES = [
    "VBR_feat_denoise_16k_latest",
    "VBR_feat_denoise_16k_modulation_latest",
    "VBR_feat_denoise_16k_modulation_latest_clean_wFD",
    # "VBR_feat_denoise_16k_modulation_latest_clean_woFD",
]

# ## 48k Experiments
# SAMPLE_RATE = 48000
# EXP_NAMES = [
#     "VBR_feat_denoise_48k_latest",
#     "VBR_feat_denoise_48k_modulation_latest",
# ]

# SAMPLE_RATE = 48000
EXP_RESULT_DIR = "results"
SAVE_DIR = "fig_curve"
SIZE = 35
MARKERSIZE=7
WIDTH=4.0
SCATTER_SIZE=300

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--save-dir', type=str, default='fig_curve')
#     return parser.parse_args()

def main():
    # loss_fn_list = ['mel', 'waveform', 'stft', 'SDR', 'SI-SDR', 'PESQ', 'STOI']
    # agg_plot_list = ['mel', 'SI-SDR', 'PESQ', 'STOI']
    
    ## Just for location of legend
    legend_UR = ["mel_loss", "stft_loss", "waveform_loss"] ## Lower is better
    legend_LR = ["SDR", "SI-SDR", "SI-SNR", "SNR", "PESQ", "STOI", "ESTOI", 
                 "SI-SDR-sg", "SI-SIR-sg", "SI-SAR-sg",
                 "p808_MOS", "MOS_SIG", "MOS_BAK", "MOS_OVR", "ViSQOL-speech",
                 "NISQA_overall_MOS",
                 "NISQA_noisiness",
                 "NISQA_discontinuity",
                 "NISQA_coloration",
                 "NISQA_loudness",
                 ] ## Higher is better
    
    metric_list = legend_UR + legend_LR
    
    agg_plot_list = ["PESQ", "STOI", "ESTOI", "SI-SDR", "ViSQOL-speech"]
    
    loss_fn_namechange = {
        'mel': 'Log Mel L1',
        'waveform': 'Waveform L1',
        'DAC-SISDR': 'DAC-SISDR',
        'stft': 'Mag Spec L1',
        'SDR': 'SDR',
        'SI-SDR': 'SI-SDR',
        'SNR': 'SNR',
        'SI-SNR': 'SI-SNR',
        'ViSQOL': 'ViSQOL',
        'PESQ': 'PESQ',
        'STOI': 'STOI',
        # 'ViSQOL-speech': 'ViSQOL',
    }
    

    sns.set_style("darkgrid")
    np.random.seed(39)
    random_colors = sns.color_palette("tab10", len(EXP_NAMES))
    plt.rc('font', size=SIZE)
    plt.rc('xtick', labelsize=SIZE)
    plt.rc('ytick', labelsize=SIZE)
    
    df_dict = {}
    for exp_name in EXP_NAMES:
        csv_path = opj(EXP_RESULT_DIR, exp_name, "evaluation_results.csv")
        df = pd.read_csv(csv_path)
        df_dict[exp_name] = df
        
        loss_types = df["metric"].unique()
        metric_list = list(set(loss_types))
    
    # fig_agg, axes_agg = plt.subplots(1, len(agg_plot_list), figsize=(25, 5)) 
    fig_agg, axes_agg = plt.subplots(1, len(agg_plot_list), figsize=(75, 15)) 
    
    ax_idx = -1
    
    for metric_fn in metric_list:
        if metric_fn in agg_plot_list:
            agg_plot = True
            ax_idx += 1
        else:
            agg_plot = False
        if metric_fn == "NISQA_overall_MOS":
            metric_fn_ylabel = "NISQA"
        else:
            metric_fn_ylabel = metric_fn
        fig, ax = plt.subplots(1, 1, figsize=(10,10))
        ax.set_xlabel("Bitrate [kbps]")
        ax.set_ylabel(f"{metric_fn_ylabel}")
        
        marker_list = ['s', '*', 'v', '^', 'd', 'x', 'P', '<', '>', 'p', 'h', 'H', 'D', 'd']
        max_sample_idx = None
        
        color_count = -1
        df_dict = dict(sorted(df_dict.items(), key=lambda x: x[0]))
        for ii, (exp_name, df) in enumerate(df_dict.items()):
            color_count += 1
            color = random_colors[color_count]
            marker = marker_list[color_count]
            
            if ii == 0:
                max_sample_idx = df["sample_idx"].max()
            else:
                assert max_sample_idx == df["sample_idx"].max(), "Sample indices do not match across experiments."
            df_metric = df[df["metric"] == metric_fn]
            df_cbr = df_metric[df_metric["mode"] == "CBR"]
            
            cbr_grouped = df_cbr.groupby(["Nq", "sample_idx"]).agg({"bpf": "mean", "value": "mean"}).reset_index()
            cbr_final = cbr_grouped.groupby("Nq").agg({"bpf": "mean", "value": "mean"}).reset_index()
            kbps = [bpf * SAMPLE_RATE / 512 / 1000 for bpf in cbr_final["bpf"]]
            df_vbr = df_metric[df_metric["mode"] == "VBR"]
            if len(df_vbr) > 0:
                alpha = 0.4
            else:
                alpha = 0.9
            ax.plot(kbps, cbr_final["value"], marker="o", color=color, alpha=alpha,
                    markersize=MARKERSIZE,
                    linewidth=WIDTH,
                    label=f"{exp_name} (CBR)"
                    )
            if agg_plot is True:
                axes_agg[ax_idx].plot(kbps, cbr_final["value"], marker="o", color=color, alpha=alpha,
                                      markersize=MARKERSIZE,
                                      linewidth=WIDTH,
                                      label=f"{exp_name} (CBR)",
                                      )
                axes_agg[ax_idx].set_xlabel("Bitrate [kbps]")
                axes_agg[ax_idx].set_ylabel(f"{metric_fn_ylabel}")
            
            df_vbr = df_metric[df_metric["mode"] == "VBR"]
            if len(df_vbr) > 0:
                vbr_grouped = df_vbr.groupby(["level", "sample_idx"]).agg({"bpf":"mean", "value": "mean"}).reset_index()
                vbr_final = vbr_grouped.groupby("level").agg({"bpf": "mean", "value": "mean"}).reset_index()
                kbps = [bpf * SAMPLE_RATE / 512 / 1000 for bpf in vbr_final["bpf"]]
                # import pdb; pdb.set_trace()
                ax.scatter(kbps, vbr_final["value"], marker=marker, color=color, s=SCATTER_SIZE,
                           label=f"{exp_name} (VBR)")
                if agg_plot:
                    axes_agg[ax_idx].scatter(kbps, vbr_final["value"], marker=marker, color=color, s=40,
                                             label=f"{exp_name} (VBR)")
                    
        fig.tight_layout()
        fig.savefig(opj(SAVE_DIR, f"{metric_fn}_RD.png"), 
                    bbox_inches='tight',
                    pad_inches=0.0,
                    dpi=300,
                    )
        plt.close(fig)
        print(f"Figure saved at: ", opj(SAVE_DIR, f"{metric_fn}_RD.png"))
    
    for ii, metric_fn in enumerate(agg_plot_list):
        if ii == len(agg_plot_list) - 1:
            axes_agg[ii].legend(loc="lower right", ncol=1)
    fig_agg.tight_layout()
    fig_agg.savefig(opj(SAVE_DIR, "agg_RD.png"), 
                    bbox_inches='tight',
                    pad_inches=0.0,
                    dpi=300,
                    )
    plt.close(fig_agg)
    print(f"Aggregated figure saved at: ", opj(SAVE_DIR, "agg_RD.png"))

if __name__ == "__main__":
    # args = parse_args()
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    main()