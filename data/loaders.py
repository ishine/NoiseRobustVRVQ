import os; opj = os.path.join
import re
import pandas as pd
from typing import Callable, List, Union, Dict
from audiotools import AudioSignal
from audiotools.core import util

from torch.utils.data import Dataset
import random




class AudioLoader_EARS_Piared:
# class AudioDataset_EARS_Paired:
    """
    adapted from audiotools.data.datasets.AudioLoader
    """
    """Loads audio endlessly from a list of audio sources
    containing paths to audio files. Audio sources can be
    folders full of audio files (which are found via file
    extension) or by providing a CSV file which contains paths
    to audio files.

    Parameters
    ----------
    sources : List[str], optional
        Sources containing folders, or CSVs with
        paths to audio files, by default None
    weights : List[float], optional
        Weights to sample audio files from each source, by default None
    relative_path : str, optional
        Path audio should be loaded relative to, by default ""
    transform : Callable, optional
        Transform to instantiate alongside audio sample,
        by default None
    ext : List[str]
        List of extensions to find audio within each source by. Can
        also be a file name (e.g. "vocals.wav"). by default
        ``['.wav', '.flac', '.mp3', '.mp4']``.
    shuffle: bool
        Whether to shuffle the files within the dataloader. Defaults to True.
    shuffle_state: int
        State to use to seed the shuffle of the files.
    """
    def __init__(
        self,
        srcs_clean: List[str],
        srcs_noisy: List[str],
        shuffle: bool = True,
        shuffle_state: int = 0,
    ):
        clean_list = util.read_sources(
            srcs_clean, relative_path="", ext=[".wav"]
        )
        noisy_list = util.read_sources(
            srcs_noisy, relative_path="", ext=[".wav"]
        )
        # assert len(self.clean_list) == len(self.noisy_list) == 1
        # self.clean_list = self.clean_list[0]
        # self.noisy_list = self.noisy_list[0]
        self.clean_list = []
        self.noisy_list = []
        for clist, nlist in zip(clean_list, noisy_list):
            assert len(clist) == len(nlist), "Clean and noisy lists must have the same length"
            self.clean_list.extend(clist)
            self.noisy_list.extend(nlist)
            
        self.clean_list = sorted([c['path'] for c in self.clean_list])
        self.noisy_list = sorted([n['path'] for n in self.noisy_list])
        
        if shuffle:
            state = util.random_state(shuffle_state)
            shuffle_idx = list(range(len(self.clean_list)))
            state.shuffle(shuffle_idx)
            self.clean_list = [self.clean_list[ii] for ii in shuffle_idx]
            self.noisy_list = [self.noisy_list[ii] for ii in shuffle_idx]
    
        ### Pair check    
        self.clean_names = [os.path.basename(c) for c in self.clean_list]
        self.clean_names = [os.path.splitext(n)[0] for n in self.clean_names]

        self.noisy_names = [os.path.basename(n) for n in self.noisy_list]
        self.noisy_names = [os.path.splitext(n)[0] for n in self.noisy_names]

        self.noisy_names = [re.sub(r'_-*[\d.]+(dB)*', '', n) for n in self.noisy_names]
        for ii in range(len(self.noisy_names)):
            assert self.clean_names[ii] == self.noisy_names[ii]


    def __call__(
        self,
        state,
        sample_rate: int, 
        duration: float,
        loudness_cutoff: float = -40,
        num_channels: int = 1,
        offset: float = None,
        item_idx: int = None,
    ):
        if item_idx is not None:
            item_idx = item_idx % len(self.clean_list)
            clean_path = self.clean_list[item_idx]
            noisy_path = self.noisy_list[item_idx]
        else:
            item_idx = state.randint(len(self.clean_list))
            clean_path = self.clean_list[item_idx]
            noisy_path = self.noisy_list[item_idx]
            
        if offset is None:
            if duration is not None:
                signal_clean = AudioSignal.salient_excerpt(
                    clean_path,
                    duration=duration,
                    state=state,
                    loudness_cutoff=loudness_cutoff,
                )
                offset = signal_clean.metadata["offset"]
                signal_noisy = AudioSignal(
                    noisy_path, 
                    offset=offset,
                    duration=duration,
                )
            else:
                signal_clean = AudioSignal(clean_path)
                signal_noisy = AudioSignal(noisy_path)
        else:
            signal_clean = AudioSignal(
                clean_path,
                offset=offset,
                duration=duration,
            )
            signal_noisy = AudioSignal(
                noisy_path,
                offset=offset,
                duration=duration,
            )
        
        if num_channels == 1:
            signal_clean = signal_clean.to_mono()
            signal_noisy = signal_noisy.to_mono()
        signal_clean = signal_clean.resample(sample_rate)
        signal_noisy = signal_noisy.resample(sample_rate)
        
        if duration is not None:
            if signal_clean.duration < duration:
                signal_clean = signal_clean.zero_pad_to(int(duration * sample_rate))
                signal_noisy = signal_noisy.zero_pad_to(int(duration * sample_rate))
            
        assert signal_clean.metadata["offset"] == signal_noisy.metadata["offset"], "Offset mismatch"
        
        items = {
            "signal_clean": signal_clean,
            "signal_noisy": signal_noisy,
            "item_idx": item_idx,
            "path_clean": str(clean_path),
            "path_noisy": str(noisy_path),
        }
        return items
            
        

class AudioDataset_EARS_Paired:
    def __init__(
        self,
        loader: AudioLoader_EARS_Piared,
        sample_rate: int,
        n_examples: int = 1000,
        duration: float = 0.5,
        loudness_cutoff: float = -40,
        num_channels: int = 1,
        without_replacement: bool = True,
    ):
        self.loader = loader
        self.sample_rate = sample_rate
        self.n_examples = n_examples
        self.duration = duration
        self.loudness_cutoff = loudness_cutoff
        self.num_channels = num_channels
        self.without_replacement = without_replacement
        
    def __len__(self):
        return self.n_examples
        
    def __getitem__(self, idx):
        state = util.random_state(idx)
        loader_kwargs = {
            "state": state,
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            "loudness_cutoff": self.loudness_cutoff,
            "num_channels": self.num_channels,
            "item_idx": idx if self.without_replacement else None,
        }
        item = self.loader(**loader_kwargs)
        item['idx'] = idx
        return item
    
    @staticmethod
    def collate(list_of_dicts: Union[list, dict], n_splits: int = None):
        """Collates items drawn from this dataset. Uses
        :py:func:`audiotools.core.util.collate`.

        Parameters
        ----------
        list_of_dicts : typing.Union[list, dict]
            Data drawn from each item.
        n_splits : int
            Number of splits to make when creating the batches (split into
            sub-batches). Useful for things like gradient accumulation.

        Returns
        -------
        dict
            Dictionary of batched data.
        """
        return util.collate(list_of_dicts, n_splits=n_splits)

class AudioLoader_EARS_Clean:
    """
    adapted from audiotools.data.datasets.AudioLoader
    """
    def __init__(
        self,
        srcs_clean: List[str],
        shuffle: bool = True,
        shuffle_state: int = 0,
    ):

        clean_list = util.read_sources(
            srcs_clean, relative_path="", ext=[".wav"]
        )
        self.clean_list = []
        for clist in clean_list:
            self.clean_list.extend(clist)
        self.clean_list = sorted([c['path'] for c in self.clean_list])
        
        if shuffle:
            state = util.random_state(shuffle_state)
            shuffle_idx = list(range(len(self.clean_list)))
            state.shuffle(shuffle_idx)
            self.clean_list = [self.clean_list[ii] for ii in shuffle_idx]

    def __call__(
        self,
        state,
        sample_rate: int, 
        duration: float,
        loudness_cutoff: float = -40,
        num_channels: int = 1,
        offset: float = None,
        item_idx: int = None,
    ):
        if item_idx is not None:
            item_idx = item_idx % len(self.clean_list)
            clean_path = self.clean_list[item_idx]
        else:
            item_idx = state.randint(len(self.clean_list))
            clean_path = self.clean_list[item_idx]
            
        if offset is None:
            if duration is not None:
                signal_clean = AudioSignal.salient_excerpt(
                    clean_path,
                    duration=duration,
                    state=state,
                    loudness_cutoff=loudness_cutoff,
                )
                offset = signal_clean.metadata["offset"]
            else:
                signal_clean = AudioSignal(clean_path)
        else:
            signal_clean = AudioSignal(
                clean_path,
                offset=offset,
                duration=duration,
            )
        
        if num_channels == 1:
            signal_clean = signal_clean.to_mono()
        signal_clean = signal_clean.resample(sample_rate)
        
        if duration is not None:
            if signal_clean.duration < duration:
                signal_clean = signal_clean.zero_pad_to(int(duration * sample_rate))
                    
        items = {
            "signal_clean": signal_clean,
            "item_idx": item_idx,
            "path_clean": str(clean_path),
        }
        
        return items
    

class AudioDataset_EARS_Clean:
    def __init__(
        self,
        loader: AudioLoader_EARS_Clean,
        sample_rate: int,
        n_examples: int = 1000,
        duration: float = 0.5,
        # offset: float = None,
        loudness_cutoff: float = -40,
        num_channels: int = 1,
        without_replacement: bool = True,
    ):
        self.loader = loader
        self.sample_rate = sample_rate
        self.n_examples = n_examples
        self.duration = duration
        # self.offset = offset
        self.loudness_cutoff = loudness_cutoff
        self.num_channels = num_channels
        self.without_replacement = without_replacement
        
    def __len__(self):
        return self.n_examples
        
    def __getitem__(self, idx):
        state = util.random_state(idx)
        # offset = None if self.offset is None else self.offset
        # item = {}
        loader_kwargs = {
            "state": state,
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            "loudness_cutoff": self.loudness_cutoff,
            "num_channels": self.num_channels,
            # "offset": offset,
            "item_idx": idx if self.without_replacement else None,
        }
        item = self.loader(**loader_kwargs)
        item['idx'] = idx
        return item
    
    @staticmethod
    def collate(list_of_dicts: Union[list, dict], n_splits: int = None):
        return util.collate(list_of_dicts, n_splits=n_splits)