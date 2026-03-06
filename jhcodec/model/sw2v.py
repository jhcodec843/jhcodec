import torch
import torch.nn as nn
from jhcodec.model.attention import Decoder, InferenceCache

class AudioEncoder(nn.Module):
    def __init__(self, config, compute_dtype: torch.dtype = torch.float32, training: bool = False):
        super().__init__()
        self.config = config
        self.compute_dtype = compute_dtype
        assert config.mlp_in.out_features == config.encoder.n_embd, f"mlp_in.out_features must be equal to encoder.n_embd, but got {config.mlp_in.out_features} and {config.encoder.n_embd}"
        self.linear_in = nn.Linear(config.mlp_in.in_features, config.encoder.n_embd)
        self.encoder = Decoder(config.encoder, compute_dtype)
        self.rvq_in = nn.Linear(config.encoder.n_embd, config.rvq.embedding_dim)
        self.in_features = config.mlp_in.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T]
        B, _ = x.shape
        x = x.view(B, -1, self.in_features) # [B, T_, C]
        x = self.linear_in(x)
        T = x.shape[1]
        if self.training and self.config.training.noise_masking > 0:
            use_noise = torch.rand((B,T,1), device=x.device) < self.config.training.noise_masking
            noise = torch.randn_like(x) #* std
            x = torch.where(use_noise, noise, x)
        else:
            x = x
        x = self.encoder(x)
        x_encoded = self.rvq_in(x)
        return x_encoded

    @torch.no_grad()
    def encode(self, x: torch.Tensor, inference_cache: InferenceCache=None) -> torch.Tensor:
        # x: [B,1,C]
        assert x.shape[1] % self.in_features == 0, f"x.shape[1] must be divisible by {self.in_features}, but got {x.shape[1]}"
        B, T = x.shape
        x = x.view(B, -1, self.in_features) # [B, T_, C]
        x = self.linear_in(x)
        x, encoder_cache = self.encoder.decode(x, inference_cache=inference_cache)
        x_encoded = self.rvq_in(x)
        return x_encoded, encoder_cache