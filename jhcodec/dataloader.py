import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio 
import os
import random
from glob import glob
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import traceback
import re
import itertools
import shutil
from contextlib import nullcontext
import time
from datetime import timedelta

import soundfile as sf



def match_file(path, spk_pattern, dataset_name):
    match = re.search(spk_pattern, path)
    if match:
        spk = dataset_name + "_" + match.group(1)
        return [spk, path]
    else:
        raise ValueError(f"Could not find spk in {path}")
    return None

def extract_spks(file_paths, spk_pattern, dataset_name):
    spks = set()
    spks_files = {}
    with mp.Pool(processes=8) as pool:
        results = pool.starmap_async(match_file, [(path, spk_pattern, dataset_name) for path in file_paths])

        for spk, path in tqdm(results.get(), total=len(file_paths), desc=f"Extracting spks from {dataset_name}"):
            spks.add(spk)
            if spk not in spks_files:
                spks_files[spk] = []
            spks_files[spk].append(path)

    return spks, spks_files
    


dataset_path ={
    'vctk': "/data/data/vctk/*/*.wav",
    'libritts_r': "/data/data/libritts_r/train-clean-*/*/*/*.wav",
    'hifitts': "/data/data/hifitts/*_clean/*/*.wav",
    'mls_en': "/data/data/mls_english/train/audio/*/*/*.flac",
    "ljspeech": "/data/data/ljspeech/*.wav",
    "librilight": "/data/libriheavy/download/librilight/large/*/*/*.flac",
    "ravdess": "/data/data/ravdess/*/*.wav",
    "emilia": "/data/emilia_en/*/*/mp3/*.mp3",
}
dataset_spk_path={
    'vctk': ("/data/data/vctk/*", "*.wav"),
    'libritts_r': ("/data/data/libritts_r/train-clean-*/*", "*/*.wav"),
    'hifitts': ("/data/data/hifitts/*_clean", "*/*.wav"),
    'mls_en': ("/data/data/mls_english/train/audio/*", "*/*.flac"),
    "ljspeech": ("/data/data/ljspeech", "*.wav"),
    "librilight": ("/data/libriheavy/download/librilight/large/*", "*/*.flac"),   
    "ravdess": ("/data/data/ravdess/*", "*.wav"),
    "emilia": ("/data/emilia_en/*/*", "mp3/*.mp3"),
}


val_dataset_path = {
    'libritts_r': "/data/data/libritts_r/dev-clean/*/*/*.wav",
}
# parenthesis is the spk id
dataset_spk_pattern ={
    'vctk' : r"/data/data/vctk/([^/]+)/[^/]+\.wav",
    'libritts_r': r"/data/data/libritts_r/train-clean-[^/]+/([^/]+)/[^/]+/[^/]+\.wav",
    'hifitts': r"/data/data/hifitts/([^/]+)_clean/[^/]+/[^/]+\.wav",
    'mls_en': r"/data/data/mls_english/train/audio/([^/]+)/[^/]+/[^/]+\.flac",
    "ljspeech": r"/data/data/(ljspeech)/[^/]+\.wav",
    "librilight": r"/data/libriheavy/download/librilight/large/([^/]+)/[^/]+/[^/]+\.flac",
    "ravdess": r"/data/data/ravdess/([^/]+)/[^/]+\.wav",
    "emilia": r"/data/emilia_en/[^/]+/([^/]+)/mp3/[^/]+\.mp3",
}
dataset_map_pattern = {
    "emilia": ("/data/emilia_en", "/data/maskvct/emilia"),
}
val_dataset_spk_pattern = {
    'libritts_r': r"/data/data/libritts_r/dev-clean/([^/]+)/[^/]+/[^/]+\.wav",
}
dataset_ratio = {
    'vctk': 0.2,
    'libritts_r': 0.3,
    'hifitts': 0.05,#0.25, # few speakers
    'mls_en': 0.3,#0.75,
    'ljspeech': 0.01,#0.01, # single speaker
    'librilight': 0.15,#1.5,
    'ravdess': 0.10,#0.1,
    'emilia': 0.05,
}

val_dataset_ratio = {
    'libritts_r': 1,
}

# number of files{
# 'vctk': 44,242, 
# libritts_r': 116,462, 
# 'daps': 100, 
# 'hifitts': 126,439, 
# 'mls_en': 10,808,037, 
# 'mls_es': 220,701
# }
# time in sec
#{'daps': 16096.28455782312, 
# 'hifitts': 415486.27999999776, 
# 'libritts_r': 685539.8806666582, 
# 'mls_en': 160775073.1196468, 
# 'mls_es': 3303663.0340000195, 
# 'vctk': 158541.3255625002}
#165354399.9244338
#time in hours
#{
#    'daps': 4.471190154950867,
#    'hifitts': 115.41285555555493,
#    'libritts_r': 190.42774462962726,
#    'mls_en': 44659.74253323522,
#    'mls_es': 917.6841761111166,
#    'vctk': 44.0392571006945
#    'l2artic': 10.0
#    'ljspeech': 24.0
#}

class AudioDataset(Dataset):
    def __init__(self,
                audio_dir,
                sample_rate=16000,
                segment_duration=10.24,
                training=True,
                use_mel=True,
                init_dataset=True, # gonna change this to False
                cache_dir=None,
                ):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        if training:
            self.dataset_path = dataset_path
            self.dataset_spk_path = dataset_spk_path
            self.dataset_spk_pattern = dataset_spk_pattern
            self.dataset_ratio = dataset_ratio
        else:
            self.dataset_path = val_dataset_path
            self.dataset_spk_pattern = val_dataset_spk_pattern
            self.dataset_ratio = val_dataset_ratio
        self.dataset_name = sorted(list(self.dataset_ratio.keys()))
        print(self.dataset_name)
        self.training = training
        self.cache_dir = os.path.join(cache_dir, 'train' if training else 'val')
        os.makedirs(self.cache_dir, exist_ok=True)
        if init_dataset:
            if self.training:
                self.audio_files = {
                    dataset: list(itertools.chain.from_iterable([glob(os.path.join(spk, self.dataset_spk_path[dataset][1])) for spk in tqdm(glob(self.dataset_spk_path[dataset][0]), desc=f"Loading spk files", position=1)])) \
                        for dataset in tqdm(self.dataset_name, desc=f"Loading audio files", position=0)
                }
                for dataset in self.dataset_name:
                    print(f"{dataset}: {len(self.audio_files[dataset]):,} files")
            else:
                # temporary for testing
                self.audio_files = {
                    dataset: glob(self.dataset_path[dataset])[:256] \
                        for dataset in self.dataset_name
                }
            np.save(os.path.join(self.cache_dir, "audio_files.npy"), self.audio_files)
            self.dataset_files_length = {
                dataset: len(self.audio_files[dataset]) \
                    for dataset in self.dataset_name
            }
            np.save(os.path.join(self.cache_dir, "dataset_files_length.npy"), self.dataset_files_length)
            self.dataset_total_files_length = sum(self.dataset_files_length.values())
            np.save(os.path.join(self.cache_dir, "dataset_total_files_length.npy"), self.dataset_total_files_length)
            spks_and_files = [
                extract_spks(self.audio_files[dataset], self.dataset_spk_pattern[dataset], dataset) \
                    for dataset in self.dataset_name
            ]
            self.spks = set()
            self.spks_files = {}
            for spks, spks_files in spks_and_files:
                self.spks.update(spks)
                self.spks_files.update(spks_files)
            np.save(os.path.join(self.cache_dir, "spks.npy"), self.spks)
            np.save(os.path.join(self.cache_dir, "spks_files.npy"), self.spks_files)
            print(self.dataset_files_length)
            print(self.spks)
            print([(spk, len(self.spks_files[spk])) for spk in self.spks])

        else:
            # load the data from the cache
            self.audio_files = np.load(os.path.join(self.cache_dir, "audio_files.npy"), allow_pickle=True).item()
            self.dataset_files_length = np.load(os.path.join(self.cache_dir, "dataset_files_length.npy"), allow_pickle=True).item()
            self.dataset_total_files_length = int(np.load(os.path.join(self.cache_dir, "dataset_total_files_length.npy")))
            self.spks = np.load(os.path.join(self.cache_dir, "spks.npy"), allow_pickle=True).item()
            self.spks_files = np.load(os.path.join(self.cache_dir, "spks_files.npy"), allow_pickle=True).item()

        ratio_sum = sum(self.dataset_ratio.values())
        self.dataset_ratio = {
            dataset: self.dataset_ratio[dataset] / ratio_sum \
                for dataset in self.dataset_name
        }

        target_batch_number = self.dataset_files_length['libritts_r'] /  self.dataset_ratio['libritts_r']
        self.iter_per_epoch = int(target_batch_number)
        print(self.iter_per_epoch)
        self.resample = torch.nn.ModuleDict()
        self.target_length = int(self.sample_rate * segment_duration)
        if use_mel:
            self.SAMPLE_RATE = 16000
            self.SEGMENT_DURATION = 10.24
            self.MEL_TARGET_LENGTH = 1024
            self.AUDIOMAE_PATCH_DURATION = 0.16
            self.SEGMENT_OVERLAP_RATIO = 0.0625
            self.stack_factor_K = 1.0
            assert self.SAMPLE_RATE == sample_rate
            assert self.SEGMENT_DURATION == segment_duration

        self.dataset_ratio_list = [self.dataset_ratio[dataset] for dataset in self.dataset_name]

        self.use_mel = use_mel
        print(f"Dataset Loaded")


    def __len__(self):
        return self.iter_per_epoch

    def find_spk_from_path(self, path, dataset=None):
        if dataset is None:
            for dataset in self.dataset_name:
                spk_pattern = self.dataset_spk_pattern[dataset]
            match = re.search(spk_pattern, path)
            if match:
                return dataset + "_" + match.group(1)
        else:
            spk_pattern = self.dataset_spk_pattern[dataset]
            match = re.search(spk_pattern, path)
            if match:
                return dataset + "_" + match.group(1)
        assert False, f"Could not find spk in {path}"

    def load_audio_with_length(self, audio_path, segment_duration=10.24, padding=True):
        # Load audio using soundfile (sf)
        if '.mp3' in audio_path:
            audio_path = self.get_mapped_audio_path(audio_path, 'emilia')
        info = sf.info(audio_path)
        len_audio = info.frames / info.samplerate
        length_sample = int(segment_duration * info.samplerate)
        length_target = int(segment_duration * self.sample_rate)

        if len_audio >= segment_duration + 0.01:
            max_start = info.frames - int((segment_duration + 0.01) * info.samplerate)
            frame_offset = random.randint(0, max_start)
            num_frames = length_sample + int(0.01 * info.samplerate)  # safety margin
        else:
            frame_offset = 0
            num_frames = -1

        if num_frames == -1:
            y = sf.read(audio_path, start=frame_offset, always_2d=True, dtype='float32')[0]
        else:
            y = sf.read(audio_path, start=frame_offset, frames=num_frames, always_2d=True, dtype='float32')[0]

        sr = info.samplerate

        # soundfile returns shape (num_samples, num_channels)
        if y.ndim == 2:  # multichannel (channels last)
            channel = random.randint(0, y.shape[1] - 1)
            y = y[:, channel]

        y = torch.from_numpy(y).unsqueeze(0)
        if sr != self.sample_rate:
            if str(sr) not in self.resample.keys():
                self.resample[str(sr)] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            y = self.resample[str(sr)](y)
        # Pad if audio is shorter than 8 seconds
        y = y[:, :length_target]  # truncate the audio to the desired length, consider upsampling

        # Apply amplitude fade-in/fade-out for 160 samples at start/end
        fade_len = min(160, y.shape[-1] // 2)  # avoid issues if audio is very short
        if fade_len > 0:
            fade_in = torch.linspace(0, 1, fade_len, device=y.device, dtype=y.dtype)


        if padding:
            if y.shape[-1] < length_target:
                padding =length_target - y.shape[-1]
                padding_start = random.randint(0, padding - fade_len)
                y = F.pad(y, (padding_start, padding-padding_start - fade_len))
                fade_out = torch.linspace(1, 0, fade_len, device=y.device, dtype=y.dtype)
                y[..., -fade_len:] = y[..., -fade_len:] * fade_out
        amp = random.uniform(0.8, max(1., min(10., 1/(np.abs(y).max() + 1e-3))))
        y = y * amp
        assert y.shape[-1] <= length_target, f"y shape: {y.shape} should be <= lengthtarget: {length_target}"
        if y.isnan().any():
            print(f"y is nan for {audio_path}")
            return torch.zeros([1, 1])
        return y

    def extract_kaldi_fbank_feature(self, waveform):
        assert self.sample_rate == 16000
        norm_mean = -4.2677393
        norm_std = 4.5689974

        waveform_16k = waveform - waveform.mean()
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform_16k,
            htk_compat=True,
            sample_frequency=16000,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=128,
            dither=0.0,
            frame_shift=10,
            frame_length=25,
            snip_edges=False,
        )

        fbank = (fbank - norm_mean) / (norm_std * 2)
        return fbank  # [1024, 128]

    def load_mel(self, waveform):
        assert self.sample_rate == 16000
        # audio is [1, T]
        original_duration = waveform.shape[-1] / self.sample_rate
        # This is to pad the audio to the multiplication of 0.16 seconds so that the original audio can be reconstructed
        original_duration = original_duration + (
                self.AUDIOMAE_PATCH_DURATION - original_duration % self.AUDIOMAE_PATCH_DURATION
        )
        mel = self.extract_kaldi_fbank_feature(waveform)
        
        segment_sample_length = int(self.SAMPLE_RATE * self.SEGMENT_DURATION)
        # Pad audio to the multiplication of 10.24 seconds for easier segmentations
        if waveform.shape[1] % segment_sample_length > 0:
            waveform = F.pad(waveform, (0, segment_sample_length - waveform.shape[1] % segment_sample_length))
        mel_target_length = self.MEL_TARGET_LENGTH * int(
            waveform.shape[-1] / segment_sample_length
        )

        target_token_len = (
            8 * original_duration / self.AUDIOMAE_PATCH_DURATION / self.stack_factor_K
        ) 
        # Calculate the mel spectrogram
        mel = self.extract_kaldi_fbank_feature(waveform)
        assert mel.shape[-1] == 128 and mel.shape[-2] % 1024 == 0, f"mel shape: {mel.shape}"
        return mel

    def get_mapped_audio_path(self, audio_path, dataset):
        orig, replace = dataset_map_pattern[dataset]
        mapped_audio_path = audio_path.replace(orig, replace)
        # change extension to wav
        mapped_audio_path = os.path.splitext(mapped_audio_path)[0] + ".wav"
        if not os.path.exists(mapped_audio_path):
            os.makedirs(os.path.dirname(mapped_audio_path), exist_ok=True)
            y, sr = torchaudio.load(audio_path)
            if sr != self.sample_rate:
                if str(sr) not in self.resample.keys():
                    self.resample[str(sr)] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                y = self.resample[str(sr)](y)
            if os.path.exists(mapped_audio_path):
                return mapped_audio_path
            assert y.isnan().any() == False, f"y is nan for {audio_path}"
            torchaudio.save(mapped_audio_path, y, self.sample_rate)
        return mapped_audio_path

    @torch.no_grad()
    def __getitem__(self, idx):
        # random sample from the dataset
        try:
            audio_paths =[]
            selected_dataset = np.random.choice(self.dataset_name, p=self.dataset_ratio_list)
            dataset_idx = np.random.randint(0, self.dataset_files_length[selected_dataset])
            
            audio_path = self.audio_files[selected_dataset][dataset_idx]
            spk = self.find_spk_from_path(audio_path, selected_dataset)
            #audio_path = self.get_mapped_audio_path(audio_path, selected_dataset)
            audio_paths.append(audio_path)
            spk_files = self.spks_files[spk]
            if len(spk_files) < 1:
                raise ValueError(f"No files found for speaker {spk} and {audio_path}")
            audio = self.load_audio_with_length(audio_path, self.segment_duration, padding=False)
            
            # Prevent infinite loop by limiting concatenation attempts
            max_attempts = 10
            attempt = 0
            while audio.shape[-1] < self.target_length and attempt < max_attempts:
                audio_path_ = random.choice(spk_files)
                #audio_path_ = self.get_mapped_audio_path(audio_path_, selected_dataset)
                audio_ = self.load_audio_with_length(audio_path_, self.segment_duration, padding=False)
                if audio_.shape[-1] + audio.shape[-1] < self.target_length:
                    # fade_out
                    fade_len = min(160, audio.shape[-1] // 2)
                    fade_out = torch.linspace(1, 0, fade_len, device=audio_.device, dtype=audio_.dtype)
                    audio_[..., -fade_len:] = audio_[..., -fade_len:] * fade_out
                audio = torch.cat([audio, audio_], dim=1)
                silence_duration = random.random() * 2 # 0. to 2.0
                audio = F.pad(audio, (0, int(silence_duration*self.sample_rate)))
                attempt += 1
                audio_paths.append(audio_path_)
            
            # If still not long enough, pad with zeros
            if audio.shape[-1] < self.target_length:
                padding = self.target_length - audio.shape[-1]
                audio = F.pad(audio, (0, padding))
            
            audio = audio[:, :self.target_length] 
            if audio.isnan().any():
                print(f"audio is nan for spk: {spk}, audio_path: {audio_paths}")
                assert False, "audio is nan"
            
            if self.use_mel:
                mel = self.load_mel(audio)
                # Return only the mono audio without any noise mixing
                return audio.view(1, -1), mel
            else:
                return audio.view(1, -1)
        except Exception as e:
            print(f"Error in __getitem__: {e} {audio_paths}")
            print(f"Original error: {str(e)}")
            print(f"Original traceback: {traceback.format_exc()}")
            return self.__getitem__(idx)

       
        

# Collate function for the dataloader
def collate_fn(batch):
    # Find the length of the longest audio in the batch
    len_batch = len(batch[0])
    if len_batch == 2:
        use_mel = True
    else:
        assert len_batch == 1, f"len_batch: {len_batch}"
        use_mel = False
        
    if use_mel:
        max_length = max([audio.shape[-1] for audio, mel in batch])
    else:
        max_length = max([audio.shape[-1] for audio in batch])
 
    
    
    # Pad each audio to the length of the longest one
    # for now, we assume all the audios have the same mel length
    if use_mel:
        padded_batch = []
        mel_batch = []
        for audio, mel in batch:
            padding_length = max_length - audio.shape[-1]
            padded_audio = F.pad(audio, (0, padding_length))
            padded_batch.append(padded_audio)
            mel_batch.append(mel)
        # Stack the padded audios
        stacked_audio = torch.cat(padded_batch, dim=0)
        stacked_mel = torch.stack(mel_batch, dim=0)
        return stacked_audio, stacked_mel
    else:
        padded_batch = []
        for audio in batch:
            padding_length = max_length - audio.shape[-1]
            padded_audio = F.pad(audio, (0, padding_length))
            padded_batch.append(padded_audio)
        # Stack the padded audios
        stacked_audio = torch.cat(padded_batch, dim=0)
        return stacked_audio

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = AudioDataset(audio_dir='./data', 
                            sample_rate=16000,
                            segment_duration=10.24,
                            training=True, 
                            init_dataset=False,
                            cache_dir='/data/dataloader/v9',
                            use_mel=False,
                            )
    print(len(dataset))
    # Print length of files for each dataset using dataset's internal data
    print("\nNumber of files per dataset:")
    for dataset_name in dataset.dataset_name:
       print(f"{dataset_name}: {len(dataset.audio_files[dataset_name]):,} files")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=16, collate_fn=collate_fn)
    for i, (audio, ref_audio, prompt_audio, pitch_shift, pitch_audio, pitch_prompt, pitch_ref) in enumerate(dataloader):
       print(audio.shape, ref_audio.shape, prompt_audio.shape, pitch_shift.shape, pitch_audio.shape, pitch_prompt.shape, pitch_ref.shape)
       if i > 10000:
           break
