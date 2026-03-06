import torch
import torch.nn as nn
import torch.nn.functional as F
from jhcodec.model.attention import Decoder, InferenceCache
from jhcodec.model.vq import RVQFixedSTE, RVQDAC, RVQMimi, RVQNaiveCommit, RVQNaiveGrad
from typing import Union



class JHCodecDAC(nn.Module):
    def __init__(self, config,# omegaconf.dictconfig.DictConfig
                 compute_dtype: torch.dtype = torch.float32, 
                 training: bool=False):
        super().__init__()
        self.config = config
        self.compute_dtype = compute_dtype
        assert config.mlp_in.out_features == config.decoder.n_embd, f"linear_in.out_features must be equal to decoder.n_embd, but got {config.mlp_in.out_features} and {config.decoder.n_embd}"
        assert config.mlp_out.in_features == config.decoder.n_embd, f"linear_out.in_features must be equal to decoder.n_embd, but got {config.mlp_out.in_features} and {config.decoder.n_embd}"
        assert config.encoder.n_embd == config.decoder.n_embd, f"encoder.n_embd must be equal to decoder.n_embd, but got {config.encoder.n_embd} and {config.decoder.n_embd}"
        self.feature_size = config.mlp_in.in_features
        self.linear_in = nn.Sequential(nn.Linear(config.mlp_in.in_features, config.mlp_in.hidden_features, bias=False),
                                       nn.Linear(config.mlp_in.hidden_features, config.decoder.n_embd))
        self.encoder = Decoder(config.encoder, compute_dtype)

        # mask
        self.encoder_mask_token = nn.Parameter(torch.randn(1,1,config.decoder.n_embd))
        self.decoder_mask_token = nn.Parameter(torch.randn(1,1,config.decoder.n_embd))
        self.encoder_mask_rate = config.training.encoder_mask_rate if 'training' in config and 'encoder_mask_rate' in config.training else 0.0
        self.decoder_mask_rate = config.training.decoder_mask_rate if 'training' in config and 'decoder_mask_rate' in config.training else 0.0
        #self.rvq_rectifier = Decoder(config.rvq_rectifier, compute_dtype)
        if config.decoder.n_embd == config.rvq.embedding_dim:
            self.rvq_in = nn.Identity()
        else:
            self.rvq_in = nn.Linear(config.decoder.n_embd, config.rvq.embedding_dim)
        if "naive" == config.rvq.type:
            if config.rvq.naive:
                if config.rvq.naive_grad:
                    self.rvq = RVQNaiveGrad(config.rvq, compute_dtype)
                else:
                    self.rvq = RVQNaiveCommit(config.rvq, compute_dtype)
            else:
                assert False, f"{config.rvq.type} is not supported"
        elif 'dac' == config.rvq.type:
            self.rvq = RVQDAC(config.rvq, compute_dtype)
        elif "fixedste" == config.rvq.type:
            self.rvq = RVQFixedSTE(config.rvq, compute_dtype)
        else:
            assert False, f"{config.rvq.type} is not supported"
        if config.decoder.n_embd == config.rvq.embedding_dim:
            self.rvq_out = nn.Identity()
        else:
            self.rvq_out = nn.Linear(config.rvq.embedding_dim, config.decoder.n_embd)
        self.decoder = Decoder(config.decoder, compute_dtype)
        self.linear_out = nn.Sequential(nn.Linear(config.decoder.n_embd, config.mlp_out.hidden_features),
                                       nn.Linear(config.mlp_out.hidden_features, config.mlp_out.out_features, bias=False))
        self.num_codebooks = config.rvq.num_codebooks

    # for all model, forward function is used for training
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T]
        B, T_in = x.shape
        if T_in % self.feature_size != 0:
            pad = self.feature_size - (T_in % self.feature_size)
            x = F.pad(x, (0, pad))
            B, T_in = x.shape
        x = x.view(B, -1, self.feature_size) # [B, T_, C]
        B, T, _ = x.shape
        if self.training:
            encoder_masked = torch.rand((B, T, 1), device=x.device) < self.encoder_mask_rate
            decoder_masked = torch.rand((B, T, 1), device=x.device) < self.decoder_mask_rate
        else:
            encoder_masked = torch.zeros((B, T, 1), device=x.device, dtype=torch.bool)
            decoder_masked = torch.zeros((B, T, 1), device=x.device, dtype=torch.bool)
        x = self.linear_in(x)
        x = torch.where(encoder_masked, self.encoder_mask_token, x)
        x_encoded = self.rvq_in(self.encoder(x))
        # quantizer dropout
        quantizer_drop = torch.rand(B, device=x.device) < self.config.training.quantizer_dropout if self.training else torch.zeros(B, dtype=torch.bool, device=x.device)
        n_codebooks = torch.where(quantizer_drop, 
                                  torch.randint(1, self.config.rvq.num_codebooks + 1, (B,), device=x.device), 
                                  self.config.rvq.num_codebooks)
        rvq_embed, indices, total_vq_loss, total_commit_loss = self.rvq(x_encoded, n_codebooks)
        # we do not use continuous training for this paper.
        # use_continuous = torch.rand((x.shape[0],1,1), device=x.device) < self.config.training.use_continuous
        # use_continuous_inbatch = torch.rand((x.shape[0], x.shape[1],1), device=x.device) < self.config.training.use_continuous

        # x = torch.where(use_continuous, 
        #                 torch.where(use_continuous_inbatch, x_encoded, rvq_embed), 
        #                 rvq_embed)
        x = torch.where(decoder_masked, self.decoder_mask_token, rvq_embed)
        x_decoded = self.decoder(self.rvq_out(x))
        x_out = self.linear_out(x_decoded)
        x_out = x_out.view(B, T_in)

        return x_out, indices, total_vq_loss, total_commit_loss

    @torch.no_grad()
    def encode(self, x: torch.Tensor, n_codebooks: Union[torch.Tensor, int, None]=None, inference_cache: InferenceCache=None) -> torch.Tensor:
        # x: [B,T,C] or [B,1,C] for frame-by-frame
        B, T_in = x.shape
        if T_in % self.feature_size != 0:
            pad = self.feature_size - (T_in % self.feature_size)
            x = F.pad(x, (0, pad))
            B, T_in = x.shape
        x = x.view(B, -1, self.feature_size) # [B, T_, C]
        x = self.linear_in(x)
        x_encoded, inference_cache = self.encoder.decode(x, inference_cache=inference_cache)
        if n_codebooks is int:
            n_codebooks = torch.tensor([n_codebooks]*B, dtype=torch.long, device=x.device)
        elif n_codebooks is None:
            n_codebooks = torch.tensor([self.num_codebooks]*B, dtype=torch.long, device=x.device)
        indices = self.rvq.encode(self.rvq_in(x_encoded), n_codebooks)
        return indices, inference_cache
    
    @torch.no_grad()
    def decode(self, indices: torch.Tensor, n_codebooks: Union[torch.Tensor, int, None]=None, inference_cache: InferenceCache=None) -> torch.Tensor:
        # indices: [B,T,D] - can be single frame [B,1,D] or full sequence
        B, T, D = indices.shape
        if isinstance(n_codebooks, int):
            n_codebooks = torch.tensor([n_codebooks]*B, dtype=torch.long, device=indices.device)
        elif n_codebooks is None:
            n_codebooks = torch.tensor([D]*B, dtype=torch.long, device=indices.device)
        x = self.rvq_out(self.rvq.decode(indices, n_codebooks))
        x_decoded, inference_cache = self.decoder.decode(x, inference_cache=inference_cache)
        x_out = self.linear_out(x_decoded)
        x_out = x_out.view(B, -1)
        return x_out, inference_cache


class JHCodecMimi(nn.Module):
    def __init__(self, config,# omegaconf.dictconfig.DictConfig
                 compute_dtype: torch.dtype = torch.float32, 
                 training: bool=False):
        super().__init__()
        self.config = config
        self.compute_dtype = compute_dtype
        assert config.mlp_in.out_features == config.decoder.n_embd, f"linear_in.out_features must be equal to decoder.n_embd, but got {config.mlp_in.out_features} and {config.decoder.n_embd}"
        assert config.mlp_out.in_features == config.decoder.n_embd, f"linear_out.in_features must be equal to decoder.n_embd, but got {config.mlp_out.in_features} and {config.decoder.n_embd}"
        self.linear_in = nn.Sequential(nn.Linear(config.mlp_in.in_features, config.mlp_in.hidden_features, bias=False),
                                       nn.Linear(config.mlp_in.hidden_features, config.decoder.n_embd))
        self.feature_size = config.mlp_in.in_features
        self.encoder = Decoder(config.encoder, compute_dtype)
        # mask
        self.encoder_mask_token = nn.Parameter(torch.randn(1,1,config.decoder.n_embd))
        self.decoder_mask_token = nn.Parameter(torch.randn(1,1,config.decoder.n_embd))
        self.encoder_mask_rate = config.training.encoder_mask_rate if 'training' in config and 'encoder_mask_rate' in config.training else 0.0
        self.decoder_mask_rate = config.training.decoder_mask_rate if 'training' in config and 'decoder_mask_rate' in config.training else 0.0
        #self.rvq_rectifier = Decoder(config.rvq_rectifier, compute_dtype)
        assert config.rvq.type == "mimi", f"{config.rvq.type} is not supported"
        if config.decoder.n_embd == config.rvq.embedding_dim:
            self.rvq_in = nn.Identity()
        else:
            self.rvq_in = nn.Linear(config.decoder.n_embd, config.rvq.embedding_dim)
   
        self.rvq = RVQMimi(config.rvq, compute_dtype)
        if config.decoder.n_embd == config.rvq.embedding_dim:
            self.rvq_out = nn.Identity()
        else:
            self.rvq_out = nn.Linear(config.rvq.embedding_dim, config.decoder.n_embd)
        self.decoder = Decoder(config.decoder, compute_dtype)
        self.linear_out = nn.Sequential(nn.Linear(config.decoder.n_embd, config.mlp_out.hidden_features),
                                       nn.Linear(config.mlp_out.hidden_features, config.mlp_out.out_features, bias=False))
        self.num_codebooks = config.rvq.num_codebooks

    # for all model, forward function is used for training
    def forward(self, x: torch.Tensor, semantic_gt: torch.Tensor) -> torch.Tensor:
        # x: [B,T]
        B, T_in = x.shape
        if T_in % self.feature_size != 0:
            pad = self.feature_size - (T_in % self.feature_size)
            x = F.pad(x, (0, pad))
            B, T_in = x.shape
        x = x.view(B, -1, self.feature_size) # [B, T_, C]
        B, T, _ = x.shape
        if self.training:
            encoder_masked = torch.rand((B, T, 1), device=x.device) < self.encoder_mask_rate
            decoder_masked = torch.rand((B, T, 1), device=x.device) < self.decoder_mask_rate
        else:
            encoder_masked = torch.zeros((B, T, 1), device=x.device, dtype=torch.bool)
            decoder_masked = torch.zeros((B, T, 1), device=x.device, dtype=torch.bool)
        x = self.linear_in(x)
        x = torch.where(encoder_masked, self.encoder_mask_token, x)
        x_encoded = self.rvq_in(self.encoder(x))
        # quantizer dropout
        quantizer_drop = torch.rand(x.shape[0], device=x.device) < self.config.training.quantizer_dropout if self.training else torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        n_codebooks = torch.where(quantizer_drop, 
                                  torch.randint(1, self.config.rvq.num_codebooks + 1, (x.shape[0],), device=x.device), 
                                  self.config.rvq.num_codebooks)
        rvq_embed, indices, semantic_loss, total_vq_loss, total_commit_loss = self.rvq(x_encoded, semantic_gt, n_codebooks)
        #use_continuous = torch.rand((x.shape[0],1,1), device=x.device) < self.config.training.use_continuous
        #use_continuous_inbatch = torch.rand((x.shape[0], x.shape[1],1), device=x.device) < self.config.training.use_continuous

        #x = torch.where(use_continuous, 
        #                torch.where(use_continuous_inbatch, x_encoded, rvq_embed), 
        #                rvq_embed)
        x = torch.where(decoder_masked, self.decoder_mask_token, rvq_embed)
        x_decoded = self.decoder(self.rvq_out(x))
        x_out = self.linear_out(x_decoded)
        x_out = x_out.view(B, T_in)

        return x_out, indices, semantic_loss, total_vq_loss, total_commit_loss

    @torch.no_grad()
    def encode(self, x: torch.Tensor, n_codebooks: Union[torch.Tensor, int, None]=None, inference_cache: InferenceCache=None) -> torch.Tensor:
        # x: [B,T,C] or [B,1,C] for frame-by-frame
        B, T_in = x.shape
        if T_in % self.feature_size != 0:
            pad = self.feature_size - (T_in % self.feature_size)
            x = F.pad(x, (0, pad))
            B, T_in = x.shape
        x = x.view(B, -1, self.feature_size) # [B, T_, C]
        x = self.linear_in(x)
        x_encoded, inference_cache = self.encoder.decode(x, inference_cache=inference_cache)
        if n_codebooks is int:
            n_codebooks = torch.tensor([n_codebooks]*B, dtype=torch.long, device=x.device)
        elif n_codebooks is None:
            n_codebooks = torch.tensor([self.num_codebooks]*B, dtype=torch.long, device=x.device)
        indices = self.rvq.encode(self.rvq_in(x_encoded), n_codebooks)
        return indices, inference_cache
    
    @torch.no_grad()
    def decode(self, indices: torch.Tensor, n_codebooks: Union[torch.Tensor, int, None]=None, inference_cache: InferenceCache=None) -> torch.Tensor:
        # indices: [B,T,D] - can be single frame [B,1,D] or full sequence
        B, T, D = indices.shape
        if isinstance(n_codebooks, int):
            n_codebooks = torch.tensor([n_codebooks]*B, dtype=torch.long, device=indices.device)
        elif n_codebooks is None:
            n_codebooks = torch.tensor([D]*B, dtype=torch.long, device=indices.device)
        x = self.rvq_out(self.rvq.decode(indices, n_codebooks))
        x_decoded, inference_cache = self.decoder.decode(x, inference_cache=inference_cache)
        x_out = self.linear_out(x_decoded)
        x_out = x_out.view(B, -1)
        return x_out, inference_cache