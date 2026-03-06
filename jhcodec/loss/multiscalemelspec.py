# Adapted from https://huggingface.co/spaces/facebook/MusicGen/blob/main/audiocraft/losses/specloss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as tp
from math import sqrt
import torchaudio


class MelSpectrogramWrapper(nn.Module):
    """Wrapper for mel spectrogram computation with optional logarithmic scaling."""
    
    def __init__(self, n_fft: int, hop_length: int, win_length: int, n_mels: int,
                 sample_rate: int, f_min: float = 0.0, f_max: tp.Optional[float] = None,
                 log: bool = False, normalized: bool = False, floor_level: float = 1e-5):
        super().__init__()
        self.log = log
        self.floor_level = floor_level
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            normalized=normalized
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, channels, time)
        mel_spec = self.mel_transform(x)
        
        if self.log:
            # log10 is slow, so use log
            mel_spec = torch.log(mel_spec + self.floor_level)
        
        return mel_spec


class MultiScaleMelSpectrogramLoss(nn.Module):
    """Multi-Scale spectrogram loss (msspec).
    Args:
        sample_rate (int): Sample rate.
        range_start (int): Power of 2 to use for the first scale.
        range_stop (int): Power of 2 to use for the last scale.
        n_mels (int): Number of mel bins.
        f_min (float): Minimum frequency.
        f_max (float or None): Maximum frequency.
        normalized (bool): Whether to normalize the melspectrogram.
        use_log (bool): Whether to use log scale for the melspectrogram. ## I added this since log cause large gradient 
        alphas (bool): Whether to use alphas as coefficients or not.
        floor_level (float): Floor level value based on human perception (default=1e-5).
    """
    def __init__(self, sample_rate: int, range_start: int = 6, range_end: int = 11,
                 n_mels: int = 64, f_min: float = 0.0, f_max: tp.Optional[float] = None,
                 normalized: bool = False, use_log: bool = False, alphas: bool = True, floor_level: float = 1e-5):
        super().__init__()
        self.use_log = use_log
        l1s = list()
        if use_log:
            l2s = list()
        alphas_ = torch.ones((range_end - range_start,), dtype=torch.float32)
        total = torch.zeros((1,), dtype=torch.float32)
        self.normalized = normalized
        for i in range(range_start, range_end):
            l1s.append(
                MelSpectrogramWrapper(n_fft=2 ** i, hop_length=(2 ** i) // 4, win_length=2 ** i,
                                      n_mels=n_mels, sample_rate=sample_rate, f_min=f_min, f_max=f_max,
                                      log=False, normalized=normalized, floor_level=floor_level))
            if use_log:
                l2s.append(
                    MelSpectrogramWrapper(n_fft=2 ** i, hop_length=(2 ** i) // 4, win_length=2 ** i,
                                        n_mels=n_mels, sample_rate=sample_rate, f_min=f_min, f_max=f_max,
                                        log=True, normalized=normalized, floor_level=floor_level))
            if alphas:
                alphas_[i-range_start] = sqrt(2 ** i - 1)
            total += alphas_[-1] + 1 if use_log else 1
        self.register_buffer("alphas", alphas_)
        self.register_buffer("total", total)

        self.l1s = nn.ModuleList(l1s)
        if use_log:
            self.l2s = nn.ModuleList(l2s)

    def forward(self, x, y):
        loss = 0.0
        for i in range(len(self.alphas)):
            s_x_1 = self.l1s[i](x)
            s_y_1 = self.l1s[i](y)
            loss += F.l1_loss(s_x_1, s_y_1) 
            if self.use_log:
                s_x_2 = self.l2s[i](x)
                s_y_2 = self.l2s[i](y)
                loss += self.alphas[i] * F.mse_loss(s_x_2, s_y_2)
        if self.normalized:
            loss = loss / self.total
        return loss
