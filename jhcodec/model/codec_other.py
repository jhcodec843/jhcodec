from configparser import NoSectionError
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torchaudio
import math

from transformers import DacModel, MimiModel, AutoFeatureExtractor
from functools import wraps
from tqdm import tqdm

def disable_tqdm(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        original_tqdm = tqdm._instances  # Save original instances
        tqdm._instances = set()  # Disable all progress bars
        result = func(*args, **kwargs)
        tqdm._instances = original_tqdm  # Restore original instances
        return result
    return wrapper

# --- StableCodecWrapper START ---

class StableCodecWrapper(nn.Module):
    """
    Thin wrapper for the StableCodec model from stabilityai/stable-codec.
    - Supports loading from local checkpoint/config or HuggingFace Hub.
    - Encoding: audio -> (latents, tokens)
    - Decoding: tokens -> audio
    """
    def __init__(
        self,
        model_config_path=None,
        ckpt_path=None,
        pretrained_model='stabilityai/stable-codec-speech-16k',
        device=None,
    ):
        """
        Args:
            model_config_path (str or None): Path to model config .yaml or .json. Required if loading local weights.
            ckpt_path (str or None): Path to model checkpoint. Optional if loading from HuggingFace, required locally.
            pretrained_model (str or None): HuggingFace repo id string, e.g. 'stabilityai/stable-codec-speech-16k'
            device (torch.device or str or None): Device to load model to.
        """
        super().__init__()
        # Lazy import to avoid mandatory install
        try:
            from stable_codec import StableCodec
        except ImportError:
            raise ImportError(
                "stable_codec is not installed. Install with `pip install stable_codec`."
            )

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        if pretrained_model is not None:
            self.model = StableCodec(
                pretrained_model=pretrained_model,
                device=self.device,
            )
        elif model_config_path is not None:
            self.model = StableCodec(
                model_config_path=model_config_path,
                ckpt_path=ckpt_path,
                device=self.device,
            )
        else:
            raise ValueError(
                "You must specify either `pretrained_model` or `model_config_path`."
            )

        # Model reports sample_rate
        self.SAMPLE_RATE = getattr(self.model, "sample_rate", 16000)
        self.sample_rate = self.SAMPLE_RATE
        # Empirically, segment limits can vary. Set None if unknown.
        self.SEGMENT_DURATION = None
        self.vocab_size = getattr(self.model, "vocab_size", None)

    @torch.no_grad()
    def encode(self, wav):
        if isinstance(wav, str):
            audio, sr = torchaudio.load(wav)
            if sr != self.SAMPLE_RATE:
                audio = torchaudio.functional.resample(audio, sr, self.SAMPLE_RATE)
            wav = audio  # Tensor: [ch, T]
        if isinstance(wav, torch.Tensor):
            if wav.ndim == 1:      # [T]
                wav = wav.unsqueeze(0).unsqueeze(0)
            elif wav.ndim == 2:      # [B, T] or [1, T]
                wav = wav.unsqueeze(1) # [B, 1, T]
            wav = wav.to(self.device).float()
        else:
            raise ValueError("wav must be a Tensor or path to audio file")
        
        # Remove batch dim for compatibility with some APIs
        batch_first = wav.shape[0] if wav.ndim == 3 else 1

        # [B, 1, T] -- we expect this shape
        latents, tokens = self.model.encode(wav)
        return latents, tokens

    @torch.no_grad()
    def decode(self, tokens):
        if isinstance(tokens, tuple) and len(tokens) == 2:
            # Accept output of encode: (latents, tokens)
            tokens = tokens[1]
        tokens = tokens.to(self.device)
        result = self.model.decode(tokens)
        # The StableCodec .decode returns a tensor (sometimes dict) with [B, 1, T]
        if isinstance(result, dict) and "audio" in result:
            audio = result["audio"]
        else:
            audio = result
        return audio

    @torch.no_grad()
    def forward(self, wav):
        "Encode then decode audio waveform (fully reconstruct through bottleneck)"
        latents, tokens = self.encode(wav)
        audio = self.decode(tokens)
        return audio

# --- StableCodecWrapper END ---


class DACWrapper(nn.Module):
    def __init__(
        self,
        model_path='descript/dac_16khz',
    ):
        super().__init__()
        self.model = DacModel.from_pretrained(model_path)
        self.SAMPLE_RATE = 16000
        self.SEGMENT_DURATION = 10.24 # it should be multiple of 512 samples current = 245760
        self.vocab_size = 1024

    @torch.no_grad()
    def encode(self, wav):
        if wav.ndim == 1:
            wav = wav.view(1, 1, -1)
        elif wav.ndim == 2:
            wav = wav.unsqueeze(1)
        elif wav.ndim == 3:
            assert wav.shape[1] == 1
        else:
            raise ValueError(f"wav must be 1D, 2D, or 3D tensor, but got {wav.ndim}D tensor with shape {wav.shape}")
        x = self.model.encode(wav) 
        codes = x.audio_codes
        return codes # [B, C, T]

    @torch.no_grad()
    def decode(self, tokens, num_codebooks: int = None):
        B, C, T = tokens.shape
        if num_codebooks is not None:
            tokens = tokens[:, :num_codebooks, :]
        waveform = self.model.decode(audio_codes=tokens).audio_values
        return waveform

    @torch.no_grad()
    def forward(self, wav, num_codebooks: int = None):
        codes = self.encode(wav)
        waveform = self.decode(codes, num_codebooks=num_codebooks)
        return waveform

# --- NeMoNanoWrapper ---

class NeMoNanoWrapper(nn.Module):
    """
    Wrapper for the NVIDIA NeMo Nano Codec 22kHz model.
    - Supports encoding (audio -> tokens) and decoding (tokens -> audio).
    """
    def __init__(self, model_name: str = "nvidia/nemo-nano-codec-22khz-1.78kbps-12.5fps", device: str = None):
        super().__init__()
        from nemo.collections.tts.models import AudioCodecModel
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        # Load model
        self.model = AudioCodecModel.from_pretrained(model_name).eval().to(self.device)

    @torch.no_grad()
    def encode(self, wav):
        # Load if path
        if isinstance(wav, torch.Tensor):
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            elif wav.dim() == 2:
                pass
            elif wav.dim() == 3:
                wav = wav.squeeze(1)
            else:
                raise ValueError(f"wav must be 1D or 2D tensor, but got {wav.dim()}D tensor with shape {wav.shape}")
        wav = wav.to(self.device)
        audio_len = torch.tensor([wav.shape[-1]]).to(self.device)
        encoded_tokens, encoded_len = self.model.encode(audio=wav, audio_len=audio_len)
        return encoded_tokens, encoded_len

    @torch.no_grad()
    def decode(self, tokens, tokens_len):
        output_audio, _ = self.model.decode(tokens=tokens, tokens_len=tokens_len)
        return output_audio

    @torch.no_grad()
    def forward(self, wav):
        tokens, tokens_len = self.encode(wav)
        output_audio = self.decode(tokens, tokens_len)
        return output_audio

class MimiWrapper(nn.Module):
    def __init__(
        self,
        model_path="kyutai/mimi",
    ):
        super().__init__()
        self.model = MimiModel.from_pretrained(model_path)
        self.SAMPLE_RATE = 24000
        self.SEGMENT_DURATION = 10.24 # -> [32, 128]
        self.vocab_size = 2048

    @torch.no_grad()
    def encode(self, wav):
        if wav.ndim == 1:
            wav = wav.view(1, 1, -1)
        elif wav.ndim == 2:
            wav = wav.unsqueeze(1)
        elif wav.ndim == 3:
            assert wav.shape[1] == 1
        else:
            raise ValueError(f"wav must be 1D, 2D, or 3D tensor, but got {wav.ndim}D tensor with shape {wav.shape}")
        encoder_outputs = self.model.encode(wav)
        codes = encoder_outputs.audio_codes # [B, C, T], C=32
        return codes # [B, C, T]

    @torch.no_grad()
    def decode(self, tokens, num_codebooks: int = None):
        B = tokens.shape[0]
        if num_codebooks is not None:
            tokens = tokens[:, :num_codebooks, :]
        audio_values = self.model.decode(tokens)[0]
        # audio_values shape [B, 1, T]
        return audio_values.view(B, -1)

    @torch.no_grad()
    def forward(self, wav, num_codebooks: int = None):
        codes = self.encode(wav)
        audio = self.decode(codes, num_codebooks=num_codebooks)
        return audio

class FocalWrapper(nn.Module):
    def __init__(
        self,
        config="lucadellalib/focalcodec_50hz_65k_causal",
    ):
        super().__init__()
        self.model = torch.hub.load(
            repo_or_dir="lucadellalib/focalcodec",
            model="focalcodec",
            config=config,
            force_reload=True,  # Fetch the latest FocalCodec version from Torch Hub
        )
        self.model =  self.model.eval()

        self.SAMPLE_RATE = self.model.sample_rate_input
        self.SEGMENT_DURATION = 10.24 # -> [32, 128]
        self.vocab_size = 65536

    @torch.no_grad()
    def encode(self, wav):
        if wav.ndim == 1:
            wav = wav.view(1, -1)
        elif wav.ndim == 2:
            wav = wav
        elif wav.ndim == 3:
            assert wav.shape[1] == 1
            wav = wav.squeeze(1)
        else:
            raise ValueError(f"wav must be 1D, 2D, or 3D tensor, but got {wav.ndim}D tensor with shape {wav.shape}")
        encoder_state = []
        sig_to_codes_state = []    
        tokens, encoder_state, sig_to_codes_state = self.model.sig_to_codes(wav, 
                                                                            encoder_state,
                                                                            sig_to_codes_state, return_state=True)  # Shape: (batch, time)

        return tokens

    @torch.no_grad()
    def decode(self, tokens):
        B = tokens.shape[0]
        decoder_state = []
        codes_to_sig_state = []    
        rec_sig, decoder_state, codes_to_sig_state = self.model.codes_to_sig(tokens, 
                                                               decoder_state,
                                                               codes_to_sig_state, return_state=True)  # Shape: (batch, code_time, log2 codebook_size)
        return rec_sig.view(B, -1)

    @torch.no_grad()
    def forward(self, wav):
        tokens = self.encode(wav)
        rec_sig = self.decode(tokens)
        return rec_sig

class Qwen3TTSTokenizerWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        from qwen_tts import Qwen3TTSTokenizer
        self.tokenizer = Qwen3TTSTokenizer.from_pretrained(
            "Qwen/Qwen3-TTS-Tokenizer-12Hz",
            device_map="cuda:0",
        )

    @torch.no_grad()
    def encode(self, wav, sr=24000):
        wav = wav.squeeze(0).squeeze(0).view(-1)
        wav = wav.detach().cpu().numpy()
        tokens = self.tokenizer.encode(wav, sr=sr)
        return tokens

    @torch.no_grad()
    def decode(self, tokens):
        return self.tokenizer.decode(tokens)[0][0] # second one is sr, 24000

    @torch.no_grad()
    def forward(self, wav, sr=24000):
        tokens = self.encode(wav, sr=sr)
        audio = self.decode(tokens)
        return audio

class Xcodec2Wrapper(nn.Module):
    def __init__(
        self,
        model_path="HKUSTAudio/xcodec2xcodec2",
    ):
        super().__init__()
        self.model = Xcodec2Model.from_pretrained(model_path)
        self.SAMPLE_RATE = 16000
        self.SEGMENT_DURATION = 10.24 # -> [32, 128]
    # Encoding/decoding implementation not provided in original selection;
    # Add encode/decode/forward if model API is known.

if __name__ == "__main__":
    # Example for StableCodecWrapper
    # Set PATHS as needed.
    # model = StableCodecWrapper(model_config_path="path/to/config.yaml", ckpt_path="path/to/ckpt.pt")
    # Or download from HuggingFace
    # model = StableCodecWrapper(pretrained_model="stabilityai/stable-codec-speech-16k")

    # Uncomment to test loading and running StableCodec
    # audiopath = "audio.wav"
    # latents, tokens = model.encode(audiopath)
    # decoded_audio = model.decode(tokens)
    # torchaudio.save("decoded.wav", decoded_audio.cpu(), model.SAMPLE_RATE)

    codec = DACWrapper()
    print(codec.decode(torch.randn(1, 512, 8)))
