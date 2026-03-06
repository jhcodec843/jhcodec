#  MPD Adapted from https://github.com/jik876/hifi-gan/blob/master/models.py
#  MSSTFTD Adapted from https://github.com/facebookresearch/encodec/blob/main/encodec/msstftd.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm
import numpy as np
from typing import List, Tuple, Union


class MultiPeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator from VITS"""
    
    def __init__(self, periods: List[int] = [2, 3, 5, 7, 11]):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(period) for period in periods
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Args:
            x: Input audio tensor of shape [B, T]
        Returns:
            outputs: List of discriminator outputs
            feature_maps: List of feature maps from each discriminator
        """
        outputs = []
        feature_maps = []
        x = x.unsqueeze(1)
        for discriminator in self.discriminators:
            output, fmap = discriminator(x)
            outputs.append(output)
            feature_maps.append(fmap)
        
        return outputs, feature_maps


class PeriodDiscriminator(nn.Module):
    """Period Discriminator for a specific period"""
    
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3):
        super().__init__()
        self.period = period
        
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            weight_norm(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            weight_norm(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            weight_norm(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(kernel_size // 2, 0))),
        ])
        
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Input audio tensor of shape [B, 1, T]
        Returns:
            output: Discriminator output
            feature_maps: List of intermediate feature maps
        """
        feature_maps = []
        
        # Reshape to 2D for period-based processing
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        
        x = x.contiguous()
        x = x.view(b, c, t // self.period, self.period)
        
        for i, conv in enumerate(self.convs):
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            feature_maps.append(x)
        
        x = self.conv_post(x)
        feature_maps.append(x)
        x = torch.flatten(x, 1, -1)
        
        return x, feature_maps


def get_2d_padding(kernel_size: Tuple[int, int], dilation: Tuple[int, int] = (1, 1)):
    return (((kernel_size[0] - 1) * dilation[0]) // 2, ((kernel_size[1] - 1) * dilation[1]) // 2)


class DiscriminatorSTFT(nn.Module):
    """STFT sub-discriminator.
    Args:
        filters (int): Number of filters in convolutions
        in_channels (int): Number of input channels. Default: 1
        out_channels (int): Number of output channels. Default: 1
        n_fft (int): Size of FFT for each scale. Default: 1024
        hop_length (int): Length of hop between STFT windows for each scale. Default: 256
        kernel_size (tuple of int): Inner Conv2d kernel sizes. Default: ``(3, 9)``
        stride (tuple of int): Inner Conv2d strides. Default: ``(1, 2)``
        dilations (list of int): Inner Conv2d dilation on the time dimension. Default: ``[1, 2, 4]``
        win_length (int): Window size for each scale. Default: 1024
        normalized (bool): Whether to normalize by magnitude after stft. Default: True
        norm (str): Normalization method. Default: `'weight_norm'`
        activation (str): Activation function. Default: `'LeakyReLU'`
        activation_params (dict): Parameters to provide to the activation function.
        growth (int): Growth factor for the filters. Default: 1
    """
    def __init__(self, filters: int, in_channels: int = 1, out_channels: int = 1,
                 n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024, max_filters: int = 1024,
                 filters_scale: int = 1, kernel_size: Tuple[int, int] = (9, 3), dilations: List = [1, 2, 4],
                 stride: Tuple[int, int] = (1, 2), normalized: bool = False, norm: str = 'weight_norm'
                 ):
        super().__init__()
        assert len(kernel_size) == 2
        assert len(stride) == 2
        self.filters = filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        # torchaudio version issue
        #self.spec_transform = torchaudio.transforms.Spectrogram(
        #    n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window_fn=torch.hann_window,
        #    normalized=self.normalized, center=False, pad_mode=None, power=None)
        spec_channels = 2 * self.in_channels
        self.convs = nn.ModuleList()
        self.convs.append(
            weight_norm(nn.Conv2d(spec_channels, self.filters, kernel_size=kernel_size, padding=get_2d_padding(kernel_size)))
        )
        in_chs = min(filters_scale * self.filters, max_filters)
        for i, dilation in enumerate(dilations):
            out_chs = min((filters_scale ** (i + 1)) * self.filters, max_filters)
            self.convs.append(weight_norm(nn.Conv2d(in_chs, out_chs, kernel_size=kernel_size, stride=stride,
                                         dilation=(1, dilation), padding=get_2d_padding(kernel_size, (1, dilation)))))
            in_chs = out_chs
        out_chs = min((filters_scale ** (len(dilations) + 1)) * self.filters, max_filters)
        self.convs.append(weight_norm(nn.Conv2d(in_chs, out_chs, kernel_size=(kernel_size[1], kernel_size[1]),
                                     padding=get_2d_padding((kernel_size[1], kernel_size[1])))))
        self.conv_post = weight_norm(nn.Conv2d(out_chs, self.out_channels,
                                    kernel_size=(kernel_size[1], kernel_size[1]),
                                    padding=get_2d_padding((kernel_size[1], kernel_size[1]))))

    def forward(self, x: torch.Tensor):
        fmap = []
        x = x.contiguous()
        window = torch.hann_window(self.n_fft, requires_grad=False, dtype=x.dtype, device=x.device)
        z = torch.stft(x, self.n_fft, self.hop_length, self.win_length,
                       center=True, window=window, normalized=self.normalized, return_complex=True) # [B, Freq, Frames, 2] # Changed 
        z = torch.view_as_real(z)
        z = z.permute(0, 3, 1, 2)
        z = z.contiguous()
        #z = torch.stack([z.real, z.imag], dim=1) # changed due to torchaudio version issue
        #z = rearrange(z, 'b c w t -> b c t w') # change parameters to not doing this
        for i, layer in enumerate(self.convs):
            z = layer(z)
            z = F.leaky_relu(z, 0.2)
            fmap.append(z)
        z = self.conv_post(z)
        return z, fmap


class MultiScaleSTFTDiscriminator(nn.Module):
    """Multi-Scale STFT (MS-STFT) discriminator.
    Args:
        filters (int): Number of filters in convolutions
        in_channels (int): Number of input channels. Default: 1
        out_channels (int): Number of output channels. Default: 1
        n_ffts (Sequence[int]): Size of FFT for each scale
        hop_lengths (Sequence[int]): Length of hop between STFT windows for each scale
        win_lengths (Sequence[int]): Window size for each scale
        **kwargs: additional args for STFTDiscriminator
    """
    def __init__(self, filters: int, in_channels: int = 1, out_channels: int = 1,
                 n_ffts: List[int] = [1024, 2048, 512], hop_lengths: List[int] = [256, 512, 128],
                 win_lengths: List[int] = [1024, 2048, 512], **kwargs):
        super().__init__()
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        self.discriminators = nn.ModuleList([
            DiscriminatorSTFT(filters, in_channels=in_channels, out_channels=out_channels,
                              n_fft=n_ffts[i], win_length=win_lengths[i], hop_length=hop_lengths[i], **kwargs)
            for i in range(len(n_ffts))
        ])
        self.num_discriminators = len(self.discriminators)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        outputs = []
        feature_maps = []
        for disc in self.discriminators:
            output, fmap = disc(x)
            outputs.append(output)
            feature_maps.append(fmap)
        return outputs, feature_maps


class VocoderDiscriminator(nn.Module):
    """Combined discriminator with both MPD and MS-STFT discriminators"""
    
    def __init__(self,
                 mpd_periods: List[int] = [2, 3, 5, 7, 11],
                 msstft_filters: int = 32,
                 msstft_n_ffts: List[int] = [2048, 1024, 512, 256, 128],
                 msstft_hop_lengths: List[int] = [521, 257, 131, 67, 37], # set as the prime numbers to minimize aliasing
                 msstft_win_lengths: List[int] = [2048, 1024, 512, 256, 128]):
        super().__init__()
        
        self.mpd = MultiPeriodDiscriminator(periods=mpd_periods)
        self.msstft = MultiScaleSTFTDiscriminator(
            filters=msstft_filters,
            n_ffts=msstft_n_ffts,
            hop_lengths=msstft_hop_lengths,
            win_lengths=msstft_win_lengths
        )
        self.names = [f"mpd: {p}" for p in mpd_periods] + [f"msstft {i}" for i in msstft_n_ffts]
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Args:
            x: Input audio tensor of shape [B, 1, T]
        Returns:
            outputs: List of all discriminator outputs (MPD + MS-STFT)
            feature_maps: List of all feature maps (MPD + MS-STFT)
        """
        mpd_outputs, mpd_fmaps = self.mpd(x)
        msstft_outputs, msstft_fmaps = self.msstft(x)
        
        outputs = mpd_outputs + msstft_outputs
        feature_maps = mpd_fmaps + msstft_fmaps
        
        return outputs, feature_maps

def discriminator_loss(disc_real_outputs: List[torch.Tensor], 
                        disc_fake_outputs: List[torch.Tensor], apa_prob: Union[List[torch.Tensor], None] = None) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Discriminator loss (hinge loss)"""
    real_losses = []
    fake_losses = []
    real_accs = []
    fake_accs = []
    loss = 0.0
    
    for i, (dr, df) in enumerate(zip(disc_real_outputs, disc_fake_outputs)):
        target = torch.zeros_like(dr) if apa_prob is None else (torch.rand_like(dr) < apa_prob[i]).float()
        r_loss = torch.mean((1 - dr) ** 2)
        f_loss = torch.mean((target - df) ** 2)
        real_losses.append(r_loss)
        fake_losses.append(f_loss)
        loss = loss + r_loss + f_loss
        
        # Calculate accuracies
        real_acc = torch.mean((dr >= 0.5).float())
        fake_acc = torch.mean((df < 0.5).float())
        real_accs.append(real_acc)
        fake_accs.append(fake_acc)

    
    return loss, real_losses, fake_losses, real_accs, fake_accs


def generator_loss(disc_fake_outputs: List[torch.Tensor]) -> torch.Tensor:
    """Generator adversarial loss"""
    loss = 0.0
    for df in disc_fake_outputs:
        loss = loss + torch.mean((1 - df) ** 2)
    return loss


def feature_matching_loss(disc_real_fmaps: List[List[torch.Tensor]], 
                        disc_fake_fmaps: List[List[torch.Tensor]]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Feature matching loss"""
    losses = []
    total_loss = 0.0
    for dr_fmaps, df_fmaps in zip(disc_real_fmaps, disc_fake_fmaps):
        for dr_fmap, df_fmap in zip(dr_fmaps, df_fmaps):
            fm_loss = F.l1_loss(dr_fmap, df_fmap)
            total_loss = total_loss + fm_loss
            losses.append(fm_loss)
    return total_loss, losses
