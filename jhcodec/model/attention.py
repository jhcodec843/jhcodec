import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RMSNorm
from math import sqrt
from jhcodec.model.rotary import RotaryEmbedding
import logging
logging.basicConfig(level=logging.INFO)

if torch.cuda.is_available():
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_with_kvcache
    FA_AVAILABLE = True
else:
    #assert False, "CUDA is not available. Please install it using the instructions in the README.md file."
    logging.info("Flash Attention is not available. Using torch attention instead.")
    logging.info("THIS DOES NOT INSURE PERFORMANCE. WE RECOMMEND YOU TO INSTALL FLASH ATTENTION.")
    FA_AVAILABLE = False

#Norm = "DynamicTanh" 
Norm = "LayerNorm"

class DynamicTanh(nn.Module):
    # remove affine transform
    def __init__(self, normalized_shape, channels_last=True, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value, requires_grad=True)
        # remove affine transform cuz it is usually followed by a linear layer

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.4, train_only=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.train_only = train_only

    def forward(self, x):
        if self.training or not self.train_only:
            shape = (x.shape[0],) + (1,) * (x.ndim - 1) # [B, 1, 1]
            keep_mask = x.new_empty(shape).bernoulli_(1 - self.drop_prob)
            x = x * keep_mask
            return x
        else:
            return x


class MlpBlock(nn.Module):
    def __init__(self, embed_dim: int, intermediate_dim: int, compute_dtype: torch.dtype):
        super().__init__()
        self.dtype = compute_dtype
        self.linear_up = nn.Linear(embed_dim, intermediate_dim *2)
        self.linear_down = nn.Linear(intermediate_dim, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x = self.linear_up(x) # [B, T, C] -> [B, T, D*2]
        x = x.view(B, T, 2, -1)
        gate = x[..., 0, :]
        up = x[..., 1, :]
        x = (F.silu(gate) * up).to(self.dtype)
        x = self.linear_down(x)
        return x 


# class RotaryEmbedding(nn.Module):
#     """Rotary Position Embedding (RoPE) implementation in PyTorch."""

#     def __init__(
#         self,
#         embedding_dims: int,
#         min_timescale: int = 1,
#         max_timescale: int = 10000,
#         dtype: torch.dtype = torch.float32,
#     ):
#         super().__init__()
#         if embedding_dims % 2 != 0:
#             raise ValueError("Embedding dim must be even for RoPE.")
#         self.embedding_dims = embedding_dims
#         self.min_timescale = min_timescale
#         self.max_timescale = max_timescale
#         self.compute_dtype = dtype

#         half_embedding_dim = embedding_dims // 2
#         fraction = (2.0 * torch.arange(0, half_embedding_dim)) / embedding_dims
#         timescale = (self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction).to(torch.float32)
#         self.register_buffer("timescale", timescale, persistent=False)

#     def forward(self, inputs: torch.Tensor, position: torch.Tensor):
#         """Applies RoPE."""
#         position = position.unsqueeze(-1).unsqueeze(-1) # b, 1, 1
#         sinusoid_inp = position / self.timescale
#         sin = torch.sin(sinusoid_inp)
#         cos = torch.cos(sinusoid_inp)
#         first_half, second_half = torch.chunk(inputs.to(torch.float32), 2, dim=-1)
#         first_part = first_half * cos - second_half * sin
#         second_part = second_half * cos + first_half * sin
#         return torch.cat((first_part.to(self.compute_dtype), second_part.to(self.compute_dtype)), dim=-1)

# make ring cache is needed for sliding window attention
class InferenceCache:
    """Inference cache for storing KV cache and position information."""
    
    def __init__(self, batch_size: int, max_seqlen: int, n_heads: int, head_dim: int, device: torch.device, dtype: torch.dtype = torch.bfloat16):
        self.batch_size = batch_size
        self.max_seqlen = max_seqlen
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.kv_cache = torch.empty(
            self.batch_size,
            self.max_seqlen,
            2,
            n_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )
        self.seqlen_offset = 0
    
    def update_max_seqlen(self):
        max_seqlen = self.max_seqlen * 2
        kv_cache = torch.empty(
            self.batch_size,
            max_seqlen,
            2,
            self.n_heads,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        kv_cache[:, :self.seqlen_offset, ...] = self.kv_cache[:, :self.seqlen_offset, ...]
        self.kv_cache = kv_cache
        self.max_seqlen = max_seqlen

    def update_position(self, seqlen: int):
        """Update position information."""
        self.seqlen_offset += seqlen
    
    def reset(self):
        """Reset position information."""
        self.seqlen_offset = 0
        self.kv_cache = torch.empty(
            self.batch_size,
            self.max_seqlen,
            2,
            self.n_heads,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
    def copy(self):
        new_cache = InferenceCache(
            batch_size=self.batch_size,
            max_seqlen=self.max_seqlen,
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            device=self.device,
            dtype=self.dtype,
        )
        new_cache.kv_cache = self.kv_cache.clone()
        new_cache.seqlen_offset = self.seqlen_offset
        return new_cache

class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(
        self,
        config,
        embed_dim: int,
        n_heads: int,
        head_dim: int,
        compute_dtype: torch.dtype,
        out_embed_dim: int | None = None,
        rotary_emb: RotaryEmbedding | None = None,
        window_size: int = -1,
        layer_idx: int | None = None,
        causal: bool | None = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.output_dim = out_embed_dim if out_embed_dim is not None else embed_dim
        self.layer_idx = layer_idx
        self.rotary_emb = rotary_emb
        self.window_size = window_size

        self.dropout_rate = config.dropout_rate
        assert embed_dim == n_heads * head_dim, f"embed_dim must be equal to n_heads * head_dim, but got {embed_dim} and {n_heads} * {head_dim}"
        if causal is not None:
            self.causal = causal
        else:
            self.causal = True

        # Self-attention: single QKV projection
        qkv_dim = self.head_dim * self.n_heads * 3
        self.Wqkv = nn.Linear(embed_dim, qkv_dim)
        self.out_proj = nn.Linear(self.output_dim, self.output_dim)

        softmax_scale = 1.0 / sqrt(head_dim)
        self.softmax_scale = softmax_scale

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for attention during training/prefill.
        
        Args:
            x: Query input tensor (B, T, D)
            
        Returns:
            output: Attention output tensor (B, T, output_dim)
        """
        original_dtype = x.dtype
        batch, seqlen = x.shape[:2]

        # Self-attention with equal heads
        qkv = self.Wqkv(x)
        qkv = qkv.view(batch, seqlen, 3, self.n_heads, self.head_dim)
        
        if self.rotary_emb is not None:
            qkv = self.rotary_emb(qkv, seqlen_offset=0, max_seqlen=None)
        
        # Use flash attention for training/prefill
        if FA_AVAILABLE:
            qkv = qkv.to(torch.bfloat16)
            context = flash_attn_qkvpacked_func(
                qkv,
                dropout_p=self.dropout_rate if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=self.causal,
                window_size=((self.window_size, 0) if self.window_size > 0 else (-1, 0)) if self.causal else (self.window_size, self.window_size)
            )

        else:
            qkv = qkv.to(torch.bfloat16)
            q = qkv[:, :, 0]  # (B, T, H, C)
            q = q.contiguous()
            k = qkv[:, :, 1]  # (B, T, H, C)
            k = k.contiguous()
            v = qkv[:, :, 2]  # (B, T, H, C)
            v = v.contiguous()
            attention_score = torch.einsum("bthc,bshc->btsh", q, k) * self.softmax_scale
            # Build attention mask: causal and optional window
            t_arange = torch.arange(seqlen, device=q.device)
            s_arange = torch.arange(seqlen, device=q.device)
            if self.causal:
                # [t, s] = True where s <= t (query t can see key s)
                causal_mask = (s_arange.unsqueeze(0) <= t_arange.unsqueeze(1))  # (1, T) <= (T, 1) -> (T, T)
                if self.window_size > 0:
                    # [t, s] = True where (t - s) <= window_size
                    window_mask = (t_arange.unsqueeze(1) - s_arange.unsqueeze(0) <= self.window_size)
                    attention_mask = (causal_mask.logical_and(window_mask)).view(1, seqlen, seqlen, 1)
                else:
                    attention_mask = causal_mask.view(1, seqlen, seqlen, 1)
            else:
                if self.window_size > 0:
                    window_mask = (t_arange.unsqueeze(1) - s_arange.unsqueeze(0) <= self.window_size).view(1, seqlen, seqlen, 1)
                    attention_mask = window_mask
                else:
                    attention_mask = None
            if attention_mask is not None:
                attention_score = attention_score.masked_fill(~attention_mask, float("-inf"))
            attention_weights = F.softmax(attention_score, dim=-2)
            context = torch.einsum("btsh,bshc->bthc", attention_weights, v)
        # Project output
        context = context.contiguous().view(batch, seqlen, -1)
        context = context.to(original_dtype)
        output = self.out_proj(context)
        
        return output

    def decode(
        self,
        x: torch.Tensor,
        inference_cache: InferenceCache | None = None,
    ) -> torch.Tensor:
        """
        Decode function for inference with KV cache.
        
        Args:
            x: Query input tensor (B, T, D)
            inference_cache: InferenceCache object for inference mode
            
        Returns:
            output: Attention output tensor (B, T, output_dim)
        """
       
        original_dtype = x.dtype
        batch, seqlen = x.shape[:2]

        # Self-attention with equal heads
        qkv = self.Wqkv(x)
        qkv = qkv.view(batch, seqlen, 3, self.n_heads, self.head_dim)
        
        if seqlen + inference_cache.seqlen_offset >= inference_cache.max_seqlen:
            inference_cache.update_max_seqlen()
            
        if self.rotary_emb is not None:
            qkv = self.rotary_emb(qkv, seqlen_offset=inference_cache.seqlen_offset, max_seqlen=inference_cache.max_seqlen)
        
        # Use flash attention for training/prefill
        qkv = qkv.to(torch.bfloat16)

        if FA_AVAILABLE:
            if inference_cache.seqlen_offset == 0:
                context = flash_attn_qkvpacked_func(
                    qkv,
                    dropout_p= 0.0,
                    softmax_scale=self.softmax_scale,
                    causal=True,
                    window_size=(self.window_size, 0) if self.window_size > 0 else (-1, 0)
                )
                inference_cache.kv_cache[:, :seqlen] = qkv[:, :, 1:3]
            else:
                # seqlens = torch.tensor([inference_cache.seqlen_offset] * batch, device=qkv.device, dtype=torch.int32)
                q = qkv[:, :, 0]
                q = q.contiguous()
                k = qkv[:, :, 1]
                k = k.contiguous()
                v = qkv[:, :, 2]
                v = v.contiguous()
                # torch.cuda.synchronize()
                context = flash_attn_with_kvcache(
                    q,
                    inference_cache.kv_cache[:, :, 0],
                    inference_cache.kv_cache[:, :, 1],
                    k=k,
                    v=v,
                    cache_seqlens=inference_cache.seqlen_offset,
                    softmax_scale=self.softmax_scale,
                    causal=True,
                    window_size=(self.window_size, 0) if self.window_size > 0 else (-1, 0)
                )
        else:
            q = qkv[:, :, 0]
            q = q.contiguous()
            k = qkv[:, :, 1]
            k = k.contiguous()
            v = qkv[:, :, 2]
            v = v.contiguous()
            inference_cache.kv_cache[:, inference_cache.seqlen_offset : inference_cache.seqlen_offset + seqlen, 0] = k
            inference_cache.kv_cache[:, inference_cache.seqlen_offset : inference_cache.seqlen_offset + seqlen, 1] = v

            if self.window_size > 0:
                start = 0 if inference_cache.seqlen_offset == 0 or inference_cache.seqlen_offset + seqlen < self.window_size else inference_cache.seqlen_offset + seqlen - self.window_size
                k_window = inference_cache.kv_cache[:, start : inference_cache.seqlen_offset + seqlen, 0]
                v_window = inference_cache.kv_cache[:, start : inference_cache.seqlen_offset + seqlen, 1]
            else:
                k_window = inference_cache.kv_cache[:, : inference_cache.seqlen_offset + seqlen, 0]
                v_window = inference_cache.kv_cache[:, : inference_cache.seqlen_offset + seqlen, 1]
            k_window = k_window.contiguous()
            v_window = v_window.contiguous()

            S = k_window.shape[1]
            attention_score = torch.einsum("bthc,bshc->btsh", q, k_window) * self.softmax_scale
            # Causal mask: query at (seqlen_offset + t) may only see keys at positions 0..(seqlen_offset + t)
            # mask[t, s] = True where s <= seqlen_offset + t
            causal_mask = (
                torch.arange(start, start + S, device=q.device).unsqueeze(0)
                <= torch.arange(inference_cache.seqlen_offset, inference_cache.seqlen_offset + seqlen, device=q.device).unsqueeze(1)
            )
            if self.window_size > 0:
                window_mask = (torch.arange(inference_cache.seqlen_offset, inference_cache.seqlen_offset + seqlen, device=q.device).unsqueeze(1)
                               - torch.arange(start, start + S, device=q.device).unsqueeze(0) <= self.window_size)
                causal_mask = causal_mask.logical_and(window_mask)
            attention_score = attention_score.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(-1), float("-inf"))
            attention_weights = F.softmax(attention_score, dim=-2)
            context = torch.einsum("btsh,bshc->bthc", attention_weights, v_window)
            
        inference_cache.update_position(seqlen)
        # Project output
        context = context.contiguous().view(batch, seqlen, -1)
        context = context.to(original_dtype)
        output = self.out_proj(context)
        return output


class DecoderLayer(nn.Module):
    """Transformer Decoder Layer."""

    def __init__(self, config, compute_dtype: torch.dtype, rotary_emb: RotaryEmbedding | None = None, layer_idx: int | None = None, total_layers: int | None = None):
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.compute_dtype = compute_dtype
        self.rotary_emb = rotary_emb
        self.layer_idx = layer_idx
        self.n_hidden = config.n_hidden
        self.window_size = config.window_size if rotary_emb is not None else -1 # SWAN-GPT
        self.total_layers = total_layers
        self.layer_scale_factor = 1 / sqrt(total_layers * 2)
        self.sa_scale = nn.Parameter(torch.ones((1,1, self.n_embd), dtype=compute_dtype))
        self.mlp_scale = nn.Parameter(torch.ones((1,1, self.n_embd), dtype=compute_dtype))
        self.eps = 1e-6
        self.dropout_rate = config.dropout_rate 
        self.drop_path_rate = config.drop_path_rate if 'drop_path_rate' in config else 0.0
        self.drop_path = DropPath(drop_prob=self.drop_path_rate) if self.drop_path_rate > 0.0 else nn.Identity()

        # Norms
        if Norm == "DynamicTanh":
            self.pre_sa_norm = DynamicTanh(self.n_embd)
            self.pre_mlp_norm = DynamicTanh(self.n_embd)
        elif Norm == "RMSNorm":
            self.pre_sa_norm = RMSNorm(
            self.n_embd,
            eps=self.eps,
            dtype=torch.float32,
        )
            self.pre_mlp_norm = RMSNorm(
            self.n_embd,
            eps=self.eps,
            dtype=torch.float32,
        )
        elif Norm == "LayerNorm":
            self.pre_sa_norm = nn.LayerNorm(self.n_embd, eps=self.eps)
            self.pre_mlp_norm = nn.LayerNorm(self.n_embd, eps=self.eps)
        else:
            raise ValueError(f"Norm {Norm} not supported")

        self.dropout = nn.Dropout(self.dropout_rate)

        # Self-Attention with Causal Masking
        self.self_attention = Attention(
            config,
            embed_dim=self.n_embd,
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            compute_dtype=compute_dtype,
            out_embed_dim=self.n_embd,
            rotary_emb=self.rotary_emb,
            layer_idx=layer_idx,
            window_size=self.window_size
        )
        # MLP
        self.mlp = MlpBlock(
            embed_dim=self.n_embd,
            intermediate_dim=self.n_hidden,
            compute_dtype=compute_dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for decoder layer."""
        residual = x
        x_norm = self.pre_sa_norm(x).to(self.compute_dtype)

        sa_out = self.self_attention(
            x=x_norm,
        )
        sa_layer_scale_factor = self.layer_scale_factor * self.sa_scale
        sa_out = sa_out * sa_layer_scale_factor
        sa_out = self.drop_path(self.dropout(sa_out))
        x = residual + sa_out

        residual = x
        x_norm = self.pre_mlp_norm(x).to(self.compute_dtype)
        mlp_out = self.mlp(x_norm)
        mlp_layer_scale_factor = self.layer_scale_factor * self.mlp_scale
        mlp_out = mlp_out * mlp_layer_scale_factor
        mlp_out = self.drop_path(self.dropout(mlp_out))
        x = residual + mlp_out

        return x
    
    def decode(self, x: torch.Tensor, inference_cache: InferenceCache | None = None) -> torch.Tensor:
        residual = x
        x_norm = self.pre_sa_norm(x).to(self.compute_dtype)

        sa_out = self.self_attention.decode(
            x=x_norm,
            inference_cache=inference_cache,
        )
        sa_layer_scale_factor = self.layer_scale_factor * self.sa_scale
        sa_out = sa_out * sa_layer_scale_factor
        x = residual + sa_out

        residual = x
        x_norm = self.pre_mlp_norm(x).to(self.compute_dtype)
        mlp_out = self.mlp(x_norm)
        mlp_layer_scale_factor = self.layer_scale_factor * self.mlp_scale
        mlp_out = mlp_out * mlp_layer_scale_factor
        x = residual + mlp_out

        return x


class Decoder(nn.Module):
    """Transformer Decoder Stack."""

    def __init__(self, config, compute_dtype: torch.dtype):
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.n_layers = config.n_layers
        
        if 'rotary_base' in config:
            self.rotary_base = config.rotary_base
        else:
            self.rotary_base = 10000.0
        self.rotary_emb = RotaryEmbedding(base=self.rotary_base, dim=self.n_embd // self.n_heads)
        if 'apply_nope' in config:
            self.apply_nope = config.apply_nope
        else:
            self.apply_nope = False
            self.nope_period = self.n_layers
        if self.apply_nope:
            self.nope_period = config.nope_period # if 4, nope, rope, rope rope, nope, rope ...
        # if  (not apply_nope) -->rotary_emb
        # if apply_nope and i % nope_period == 0 --> None
        # if apply_nope and i % nope_period != 0 --> rotary_emb
        #             apply_nope T   apply_nope F
        # %p==0       F               T
        # %p!=0       T               T
        # apply SWAN-GPT # https://arxiv.org/pdf/2504.08719
        self.layers = nn.ModuleList(
            [DecoderLayer(config=config, compute_dtype=compute_dtype, rotary_emb=self.rotary_emb if ((not self.apply_nope) or (i % self.nope_period != 0)) else None, layer_idx=i, total_layers=self.n_layers) for i in range(self.n_layers)]
        )
        self.max_seqlen = 2048 # corresponds to 10.24 seconds

    def forward(self, x_BxTxC: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Decoder stack.

        Args:
            x_BxTxC: Input tensor (B, T, C).
            inference_cache: InferenceCache object for inference mode.

        Returns:
            output: The final output tensor (B, T, C), cast to float32.
        """
        x = x_BxTxC
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def allocate_inference_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype, length: int = None) -> list[InferenceCache]:
        """
        Allocate inference cache for all layers.

        Args:
            batch_size: Batch size for the cache.
            device: Device to allocate the cache on.
            dtype: Data type for the cache.

        Returns:
            List of InferenceCache objects, one for each layer.
        """
        if length is None:
            length = self.max_seqlen
        return [
            InferenceCache(batch_size, length, self.n_heads, self.head_dim, device, dtype) 
            for i in range(self.n_layers)
        ]

    def decode(self, x_BxTxC: torch.Tensor, inference_cache: InferenceCache | None = None) -> torch.Tensor:
        """
        Forward pass for the Decoder stack.

        Args:
            x_BxTxC: Input tensor (B, T, C).
            inference_cache: InferenceCache object for inference mode.

        Returns:
            output: The final output tensor (B, T, C), cast to float32.
        """
        if inference_cache is None:
            inference_cache = self.allocate_inference_cache(x_BxTxC.shape[0], x_BxTxC.device, torch.bfloat16)
        x = x_BxTxC
        for i, layer in enumerate(self.layers):
            x = layer.decode(x, inference_cache=inference_cache[i])
        return x, inference_cache



class EncoderLayer(nn.Module):
    """Transformer Encoder Layer."""

    def __init__(self, config, compute_dtype: torch.dtype, rotary_emb: RotaryEmbedding | None = None, layer_idx: int | None = None, total_layers: int | None = None):
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.compute_dtype = compute_dtype
        self.rotary_emb = rotary_emb
        self.layer_idx = layer_idx
        self.n_hidden = config.n_hidden
        self.window_size = config.window_size if rotary_emb is not None else -1 # SWAN-GPT
        self.total_layers = total_layers
        self.layer_scale_factor = 1 / sqrt(total_layers * 2)
        self.sa_scale = nn.Parameter(torch.ones((1,1, self.n_embd), dtype=compute_dtype))
        self.mlp_scale = nn.Parameter(torch.ones((1,1, self.n_embd), dtype=compute_dtype))
        self.eps = 1e-6
        self.dropout_rate = config.dropout_rate 
        self.drop_path_rate = config.drop_path_rate if 'drop_path_rate' in config else 0.0
        self.drop_path = DropPath(drop_prob=self.drop_path_rate) if self.drop_path_rate > 0.0 else nn.Identity()

        # Norms
        if Norm == "DynamicTanh":
            self.pre_sa_norm = DynamicTanh(self.n_embd)
            self.pre_mlp_norm = DynamicTanh(self.n_embd)
        elif Norm == "RMSNorm":
            self.pre_sa_norm = RMSNorm(
            self.n_embd,
            eps=self.eps,
            dtype=torch.float32,
        )
            self.pre_mlp_norm = RMSNorm(
            self.n_embd,
            eps=self.eps,
            dtype=torch.float32,
        )
        elif Norm == "LayerNorm":
            self.pre_sa_norm = nn.LayerNorm(self.n_embd, eps=self.eps)
            self.pre_mlp_norm = nn.LayerNorm(self.n_embd, eps=self.eps)
        else:
            raise ValueError(f"Norm {Norm} not supported")

        self.dropout = nn.Dropout(self.dropout_rate)

        # Self-Attention with Causal Masking
        self.self_attention = Attention(
            config,
            embed_dim=self.n_embd,
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            compute_dtype=compute_dtype,
            out_embed_dim=self.n_embd,
            rotary_emb=self.rotary_emb,
            layer_idx=layer_idx,
            window_size=self.window_size,
            causal=False
        )
        # MLP
        self.mlp = MlpBlock(
            embed_dim=self.n_embd,
            intermediate_dim=self.n_hidden,
            compute_dtype=compute_dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for decoder layer."""
        residual = x
        x_norm = self.pre_sa_norm(x).to(self.compute_dtype)

        sa_out = self.self_attention(
            x=x_norm,
        )
        sa_layer_scale_factor = self.layer_scale_factor * self.sa_scale
        sa_out = sa_out * sa_layer_scale_factor
        sa_out = self.drop_path(self.dropout(sa_out))
        x = residual + sa_out

        residual = x
        x_norm = self.pre_mlp_norm(x).to(self.compute_dtype)
        mlp_out = self.mlp(x_norm)
        mlp_layer_scale_factor = self.layer_scale_factor * self.mlp_scale
        mlp_out = mlp_out * mlp_layer_scale_factor
        mlp_out = self.drop_path(self.dropout(mlp_out))
        x = residual + mlp_out

        return x
    
class Encoder(nn.Module):
    """Transformer Encoder Stack."""

    def __init__(self, config, compute_dtype: torch.dtype):
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.n_layers = config.n_layers
        
        if 'rotary_base' in config:
            self.rotary_base = config.rotary_base
        else:
            self.rotary_base = 10000.0
        self.rotary_emb = RotaryEmbedding(base=self.rotary_base, dim=self.n_embd // self.n_heads)
        self.layers = nn.ModuleList(
            [EncoderLayer(config=config, compute_dtype=compute_dtype, rotary_emb=self.rotary_emb, layer_idx=i, total_layers=self.n_layers) for i in range(self.n_layers)]
        )
        self.max_seqlen = 2048 # 512 corresponds to 10.24 seconds

    def forward(self, x_BxTxC: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Decoder stack.

        Args:
            x_BxTxC: Input tensor (B, T, C).
            inference_cache: InferenceCache object for inference mode.

        Returns:
            output: The final output tensor (B, T, C), cast to float32.
        """
        x = x_BxTxC
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

if __name__ == "__main__":
    import omegaconf
    config = omegaconf.OmegaConf.load("config/config_mimi_recon.json")
    config.model.decoder.dropout_rate = 0.0
    config.model.decoder.drop_path_rate = 0.0
    attention = Attention(config.model.decoder, embed_dim=config.model.decoder.n_embd, n_heads=config.model.decoder.n_heads, head_dim=config.model.decoder.head_dim, window_size=config.model.decoder.window_size, compute_dtype=torch.float32)
    attention = attention.to("cuda")
    x = torch.randn(1, 1024, config.model.decoder.n_embd, device="cuda")
    FA_AVAILABLE = True
    out = attention(x)
    FA_AVAILABLE = False
    out2 = attention(x)
    print(torch.allclose(out, out2))
    print(out - out2)
    print((out - out2).abs().max()) 