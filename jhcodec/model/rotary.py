# Adopted from FlashAttention
# Copyright (c) 2023, Tri Dao.

import math
from typing import Optional, Tuple, Union

import torch
from einops import rearrange, repeat

# Try to import the CUDA kernel, fallback to torch-only implementation
if torch.cuda.is_available():
    from jhcodec.kernel.rotary_kernel import apply_rotary
    CUDA_AVAILABLE = True
else:
    CUDA_AVAILABLE = False


def rotate_half(x, interleaved=True):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)


def apply_rotary_emb_torch(x, cos, sin, interleaved=True):
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    sin = repeat(sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    return torch.cat(
        [x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin, x[..., ro_dim:]],
        dim=-1,
    )


def apply_rotary_torch(
    x,
    cos,
    sin,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved=True,
    inplace=False,
    conjugate=False,
):
    """
    Pure PyTorch implementation of rotary embedding application.
    """
    if inplace:
        out = x
    else:
        out = x.clone()
    
    if conjugate:
        sin = -sin
    
    if cu_seqlens is not None:
        # Handle variable length sequences
        for i in range(len(cu_seqlens) - 1):
            start_idx = cu_seqlens[i]
            end_idx = cu_seqlens[i + 1]
            seqlen = end_idx - start_idx
            
            if isinstance(seqlen_offsets, torch.Tensor):
                offset = seqlen_offsets[i]
            else:
                offset = seqlen_offsets
            
            cos_slice = cos[offset:offset + seqlen]
            sin_slice = sin[offset:offset + seqlen]
            
            out[start_idx:end_idx] = apply_rotary_emb_torch(
                out[start_idx:end_idx], cos_slice, sin_slice, interleaved=interleaved
            )
    else:
        # Handle regular batched sequences
        if isinstance(seqlen_offsets, torch.Tensor):
            # Different offset for each sequence in batch
            for i in range(x.shape[0]):
                offset = seqlen_offsets[i]
                seqlen = x.shape[1]
                cos_slice = cos[offset:offset + seqlen]
                sin_slice = sin[offset:offset + seqlen]
                out[i] = apply_rotary_emb_torch(out[i], cos_slice, sin_slice, interleaved=interleaved)
        else:
            # Same offset for all sequences
            seqlen = x.shape[1]
            cos_slice = cos[seqlen_offsets:seqlen_offsets + seqlen]
            sin_slice = sin[seqlen_offsets:seqlen_offsets + seqlen]
            out = apply_rotary_emb_torch(out, cos_slice, sin_slice, interleaved=interleaved)
    
    return out


class ApplyRotaryEmb(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        cos,
        sin,
        interleaved=True,
        inplace=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        if CUDA_AVAILABLE and x.is_cuda:
            out = apply_rotary(
                x,
                cos,
                sin,
                seqlen_offsets=seqlen_offsets,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                interleaved=interleaved,
                inplace=inplace,
            )
        else:
            out = apply_rotary_torch(
                x,
                cos,
                sin,
                seqlen_offsets=seqlen_offsets,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                interleaved=interleaved,
                inplace=inplace,
            )
        
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin, cu_seqlens)  # Can't save int with save_for_backward
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cu_seqlens, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        ctx.max_seqlen = max_seqlen
        ctx.use_cuda = CUDA_AVAILABLE and x.is_cuda
        return out if not inplace else x

    @staticmethod
    def backward(ctx, do):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cu_seqlens, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cu_seqlens = ctx.saved_tensors
        # TD [2023-09-02]: For some reason Triton (2.0.0.post1) errors with
        # "[CUDA]: invalid device context", and cloning makes it work. Idk why. Triton 2.1.0 works.
        if not ctx.interleaved and not ctx.inplace:
            do = do.clone()
        
        if ctx.use_cuda:
            dx = apply_rotary(
                do,
                cos,
                sin,
                seqlen_offsets=seqlen_offsets,
                cu_seqlens=cu_seqlens,
                max_seqlen=ctx.max_seqlen,
                interleaved=ctx.interleaved,
                inplace=ctx.inplace,
                conjugate=True,
            )
        else:
            dx = apply_rotary_torch(
                do,
                cos,
                sin,
                seqlen_offsets=seqlen_offsets,
                cu_seqlens=cu_seqlens,
                max_seqlen=ctx.max_seqlen,
                interleaved=ctx.interleaved,
                inplace=ctx.inplace,
                conjugate=True,
            )
        return dx, None, None, None, None, None, None, None


def apply_rotary_emb(
    x,
    cos,
    sin,
    interleaved=True,
    inplace=False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
):
    """
    Arguments:
        x: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim)
        cos, sin: (seqlen_rotary, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        inplace: if True, apply rotary embedding in-place.
        seqlen_offsets: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Return:
        out: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding to the first rotary_dim of x.
    """
    return ApplyRotaryEmb.apply(
        x, cos, sin, interleaved, inplace, seqlen_offsets, cu_seqlens, max_seqlen
    )


# For backward compatibility
apply_rotary_emb_func = apply_rotary_emb


class ApplyRotaryEmbQKV_(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        cos,
        sin,
        cos_k=None,
        sin_k=None,
        interleaved=True,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
        num_heads_q: Union[int] = None,
    ):
        use_cuda = CUDA_AVAILABLE and qkv.is_cuda
        
        if cos_k is None and sin_k is None and qkv.is_contiguous():
            # Call 1 kernel instead of 2 kernels
            # We need qkv to be contiguous so that when we reshape to combine (3, nheads)
            # dimensions, we get the same tensor
            if qkv.dim() == 5:
                batch, seqlen, three, nheads, headdim = qkv.shape
                assert three == 3
                # qk = rearrange(qkv[:, :, :2], "b s t h d -> b s (t h) d")
                qk = qkv[:, :, :2].reshape(batch, seqlen, -1, headdim)
            else:
                assert qkv.dim() == 4
                assert num_heads_q is not None
                num_heads_k = (qkv.shape[2] - num_heads_q) // 2
                assert qkv.shape[2] == num_heads_q + 2 * num_heads_k
                qk = qkv[:, :, :num_heads_q + num_heads_k]
            
            if use_cuda:
                apply_rotary(
                    qk, cos, sin, seqlen_offsets=seqlen_offsets, interleaved=interleaved, inplace=True
                )
            else:
                qk_rotated = apply_rotary_torch(
                    qk, cos, sin, seqlen_offsets=seqlen_offsets, interleaved=interleaved, inplace=True
                )
        else:
            cos_k = cos if cos_k is None else cos_k
            sin_k = sin if sin_k is None else sin_k
            if qkv.dim() == 5:
                q, k = qkv[:, :, 0], qkv[:, :, 1]
            else:
                assert qkv.dim() == 4
                assert num_heads_q is not None
                num_heads_k = (qkv.shape[2] - num_heads_q) // 2
                assert qkv.shape[2] == num_heads_q + 2 * num_heads_k
                q, k = qkv[:, :, :num_heads_q], qkv[:, :, num_heads_q : num_heads_q + num_heads_k]
            
            if use_cuda:
                apply_rotary(q, cos, sin, seqlen_offsets, interleaved=interleaved, inplace=True)
                apply_rotary(k, cos_k, sin_k, seqlen_offsets, interleaved=interleaved, inplace=True)
            else:
                apply_rotary_torch(q, cos, sin, seqlen_offsets, interleaved=interleaved, inplace=True)
                apply_rotary_torch(k, cos_k, sin_k, seqlen_offsets, interleaved=interleaved, inplace=True)
            ctx.save_for_backward(cos, sin, cos_k, sin_k)
        
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin, cos_k, sin_k)
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cos_k, sin_k, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.num_heads_q = num_heads_q
        ctx.use_cuda = use_cuda
        return qkv

    @staticmethod
    def backward(ctx, dqkv):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cos_k, sin_k, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cos_k, sin_k = ctx.saved_tensors
        
        if cos_k is None and sin_k is None and dqkv.is_contiguous():
            # Call 1 kernel instead of 2 kernels
            # We need dqkv to be contiguous so that when we reshape to combine (3, nheads)
            # dimensions, we get the same tensor
            if dqkv.dim() == 5:
                dqk = rearrange(dqkv[:, :, :2], "b s t h d -> b s (t h) d")
            else:
                assert dqkv.dim() == 4
                assert ctx.num_heads_q is not None
                num_heads_k = (dqkv.shape[2] - ctx.num_heads_q) // 2
                assert dqkv.shape[2] == ctx.num_heads_q + 2 * num_heads_k
                dqk = dqkv[:, :, : ctx.num_heads_q + num_heads_k]
            
            if ctx.use_cuda:
                apply_rotary(
                    dqk,
                    cos,
                    sin,
                    seqlen_offsets=seqlen_offsets,
                    interleaved=ctx.interleaved,
                    inplace=True,
                    conjugate=True,
                )
            else:
                apply_rotary_torch(
                    dqk,
                    cos,
                    sin,
                    seqlen_offsets=seqlen_offsets,
                    interleaved=ctx.interleaved,
                    inplace=True,
                    conjugate=True,
                )
        else:
            cos_k = cos if cos_k is None else cos_k
            sin_k = sin if sin_k is None else sin_k
            if dqkv.dim() == 5:
                dq, dk = dqkv[:, :, 0], dqkv[:, :, 1]
            else:
                assert dqkv.dim() == 4
                assert ctx.num_heads_q is not None
                num_heads_k = (dqkv.shape[2] - ctx.num_heads_q) // 2
                assert dqkv.shape[2] == ctx.num_heads_q + 2 * num_heads_k
                dq = dqkv[:, :, : ctx.num_heads_q]
                dk = dqkv[:, :, ctx.num_heads_q : ctx.num_heads_q + num_heads_k]
            
            if ctx.use_cuda:
                apply_rotary(
                    dq,
                    cos,
                    sin,
                    seqlen_offsets,
                    interleaved=ctx.interleaved,
                    inplace=True,
                    conjugate=True,
                )
                apply_rotary(
                    dk,
                    cos_k,
                    sin_k,
                    seqlen_offsets,
                    interleaved=ctx.interleaved,
                    inplace=True,
                    conjugate=True,
                )
            else:
                apply_rotary_torch(
                    dq,
                    cos,
                    sin,
                    seqlen_offsets,
                    interleaved=ctx.interleaved,
                    inplace=True,
                    conjugate=True,
                )
                apply_rotary_torch(
                    dk,
                    cos_k,
                    sin_k,
                    seqlen_offsets,
                    interleaved=ctx.interleaved,
                    inplace=True,
                    conjugate=True,
                )
        return dqkv, None, None, None, None, None, None, None


def apply_rotary_emb_qkv_(
    qkv,
    cos,
    sin,
    cos_k=None,
    sin_k=None,
    interleaved=True,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    num_heads_q: Optional[int] = None,
):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim) or (batch_size, seqlen, num_heads_q + 2 * num_heads_k, headdim).
            If qkv has shape (batch_size, seqlen, num_heads_q + 2 * num_heads_k, headdim) (e.g. MQA / GQA),
            then num_heads_q must be provided.
        cos, sin: (seqlen, rotary_dim / 2)
        cos_k, sin_k: (seqlen, rotary_dim / 2), optional
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
            1st half and 2nd half (GPT-NeoX style).
        seqlen_offsets: (batch_size,) or int. Each sequence in Q and K is shifted by this amount.
            Most commonly used in inference when we have KV cache.
    Return:
        qkv: (batch_size, seqlen, 3, nheads, headdim) or (batch_size, seqlen, num_heads_q + 2 * num_heads_k, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding *inplace* to the first rotary_dim of Q and K.
    """
    return ApplyRotaryEmbQKV_.apply(
        qkv, cos, sin, cos_k, sin_k, interleaved, seqlen_offsets, num_heads_q
    )


class ApplyRotaryEmbKV_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, cos, sin, interleaved=True, seqlen_offsets: Union[int, torch.Tensor] = 0):
        batch, seqlen, two, nheads, headdim = kv.shape
        assert two == 2
        k = kv[:, :, 0]
        
        use_cuda = CUDA_AVAILABLE and kv.is_cuda
        if use_cuda:
            apply_rotary(
                k, cos, sin, seqlen_offsets=seqlen_offsets, interleaved=interleaved, inplace=True
            )
        else:
            apply_rotary_torch(
                k, cos, sin, seqlen_offsets=seqlen_offsets, interleaved=interleaved, inplace=True
            )
        
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin)  # Can't save int with save_for_backward
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.use_cuda = use_cuda
        return kv

    @staticmethod
    def backward(ctx, dkv):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin = ctx.saved_tensors
        
        if ctx.use_cuda:
            apply_rotary(
                dkv[:, :, 0],
                cos,
                sin,
                seqlen_offsets=seqlen_offsets,
                interleaved=ctx.interleaved,
                inplace=True,
                conjugate=True,
            )
        else:
            apply_rotary_torch(
                dkv[:, :, 0],
                cos,
                sin,
                seqlen_offsets=seqlen_offsets,
                interleaved=ctx.interleaved,
                inplace=True,
                conjugate=True,
            )
        return dkv, None, None, None, None


apply_rotary_emb_kv_ = ApplyRotaryEmbKV_.apply


def apply_rotary_emb_kv_(
    kv,
    cos,
    sin,
    interleaved=True,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
):
    """
    Arguments:
        kv: (batch_size, seqlen, 2, nheads, headdim)
        cos, sin: (seqlen, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
            1st half and 2nd half (GPT-NeoX style).
        seqlen_offsets: (batch_size,) or int. Each sequence in Q and K is shifted by this amount.
            Most commonly used in inference when we have KV cache.
    Return:
        kv: (batch_size, seqlen, 2, nheads, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding *inplace* to the first rotary_dim of K.
    """
    return ApplyRotaryEmbKV_.apply(kv, cos, sin, interleaved, seqlen_offsets)


class RotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox

    If scale_base is not None, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
    A recommended value for scale_base is 512: https://github.com/HazyResearch/flash-attention/issues/96
    Reference: https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py
    """

    def __init__(
        self,
        dim: int,
        base=10000.0,
        interleaved=True,
        scale_base=None,
        pos_idx_in_fp32=True,
        device=None,
    ):
        """
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        pos_idx_in_fp32: if True, the position indices [0.0, ..., seqlen - 1] are in fp32,
            otherwise they might be in lower precision.
            This option was added because previously (before 2023-07-02), when we construct
            the position indices, we use the dtype of self.inv_freq. In most cases this would
            be fp32, but if the model is trained in pure bf16 (not mixed precision), then
            self.inv_freq would be bf16, and the position indices are also in bf16.
            Because of the limited precision of bf16 (e.g. 1995.0 is rounded to 2000.0), the
            embeddings for some positions will coincide.
            To maintain compatibility with models previously trained in pure bf16,
            we add this option.
        """
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.interleaved = interleaved
        self.scale_base = scale_base
        scale = (
            (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim) / (1.4 * dim)
            if scale_base is not None
            else None
        )
        self.register_buffer("scale", scale, persistent=False)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _compute_inv_freq(self, device=None):
        return 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
        )

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
            # And the output of arange can be quite large, so bf16 would lose a lot of precision.
            # However, for compatibility reason, we add an option to use the dtype of self.inv_freq.
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                # We want fp32 here as well since inv_freq will be multiplied with t, and the output
                # will be large. Having it in bf16 will lose a lot of precision and cause the
                # cos & sin output to change significantly.
                # We want to recompute self.inv_freq if it was not loaded in fp32
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self._compute_inv_freq(device=device)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                inv_freq = self.inv_freq
            # Don't do einsum, it converts fp32 to fp16 under AMP
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, inv_freq)
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                power = (
                    torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device)
                    - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def forward(
        self,
        qkv: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        seqlen_offset: Union[int, torch.Tensor] = 0,
        max_seqlen: Optional[int] = None,
        num_heads_q: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        qkv: (batch, seqlen, 3, nheads, headdim) or (batch, seqlen, num_heads_q + 2 * num_heads_k, headdim)
            if kv is none, else it's just q of shape (batch, seqlen, nheads, headdim).
            If qkv has shape (batch, seqlen, num_heads_q + 2 * num_heads_k, headdim) (e.g. MQA / GQA),
            then num_heads_q must be provided.
        kv: (batch, seqlen, 2, nheads, headdim)
        seqlen_offset: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
            If it's a tensor of shape (batch_size,), then to update the cos / sin cache, one
            should pass in max_seqlen, which will update the cos / sin cache up to that length.
        Apply rotary embedding *inplace* to qkv and / or kv.
        """
        if kv is None:
            seqlen = qkv.shape[1]
        else:
            seqlen = max(qkv.shape[1], kv.shape[1])
        if max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)
        elif isinstance(seqlen_offset, int):
            self._update_cos_sin_cache(seqlen + seqlen_offset, device=qkv.device, dtype=qkv.dtype)
        if kv is None:
            if self.scale is None:
                return apply_rotary_emb_qkv_(
                    qkv,
                    self._cos_cached,
                    self._sin_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                    num_heads_q=num_heads_q,
                )
            else:
                return apply_rotary_emb_qkv_(
                    qkv,
                    self._cos_cached,
                    self._sin_cached,
                    self._cos_k_cached,
                    self._sin_k_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                    num_heads_q=num_heads_q,
                )
        else:
            q = qkv
            q = apply_rotary_emb_func(
                q,
                self._cos_cached,
                self._sin_cached,
                interleaved=self.interleaved,
                inplace=True,
                seqlen_offsets=seqlen_offset,
            )
            if self.scale is None:
                kv = apply_rotary_emb_kv_(
                    kv,
                    self._cos_cached,
                    self._sin_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                )
            else:
                kv = apply_rotary_emb_kv_(
                    kv,
                    self._cos_k_cached,
                    self._sin_k_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                )
            return q, kv
if __name__ == "__main__":
    a = torch.ones((1, 16, 3,1,256), device = "cuda")
    rotary = RotaryEmbedding(dim = 256)
    rotary = rotary.to(a.device)
    qkv = rotary(a)
    print(qkv.shape)
    q = qkv[:, :, 0]
    k = qkv[:, :, 1]
    t = torch.arange(16, device = a.device, dtype = torch.float32)
    attn_scores = torch.einsum("b t h d, b s h d -> b t s h", q, k).squeeze(-1)
    causal_mask  = (t.unsqueeze(0) <= t.unsqueeze(1)).view(1, 16, 16)
    attn_scores = attn_scores.masked_fill(~causal_mask, -torch.inf)
    attn_weights = torch.nn.functional.softmax(attn_scores/32, dim = -1)
    attn_weights = attn_weights.squeeze(0).detach().cpu().numpy()
    import matplotlib.pyplot as plt
    plt.imshow(attn_weights, aspect = "auto")
    plt.xlabel("Key position")
    plt.ylabel("Query position")
    plt.title("Attention Weights with Positional Embedding")
    plt.savefig("attn_weights.png")
    


if __name__ == "__main__":
    # Test code to verify CUDA and CPU implementations produce the same results
    import numpy as np
    
    def test_rotary_embedding_consistency():
        """Test that CUDA and CPU implementations produce the same results"""
        print("Testing rotary embedding consistency between CUDA and CPU implementations...")
        
        # Test parameters
        batch_size = 2
        seq_len = 16
        n_heads = 4
        head_dim = 64
        rotary_dim = 32
        
        # Create test data
        torch.manual_seed(42)
        
        # Test on CPU first
        device_cpu = torch.device('cpu')
        x_cpu = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device_cpu, requires_grad=True)
        
        # Create rotary embedding
        rotary_emb = RotaryEmbedding(rotary_dim, device=device_cpu)
        
        # Create QKV tensor for testing
        qkv_cpu = torch.randn(batch_size, seq_len, 3, n_heads, head_dim, device=device_cpu, requires_grad=True)
        
        # Test basic rotary embedding
        cos_cpu = rotary_emb._cos_cached if rotary_emb._cos_cached is not None else torch.cos(torch.outer(torch.arange(seq_len, dtype=torch.float32), rotary_emb.inv_freq))
        sin_cpu = rotary_emb._sin_cached if rotary_emb._sin_cached is not None else torch.sin(torch.outer(torch.arange(seq_len, dtype=torch.float32), rotary_emb.inv_freq))
        
        if cos_cpu is None or sin_cpu is None:
            rotary_emb._update_cos_sin_cache(seq_len, device=device_cpu, dtype=x_cpu.dtype)
            cos_cpu = rotary_emb._cos_cached
            sin_cpu = rotary_emb._sin_cached
        
        # Force CPU implementation by temporarily setting CUDA_AVAILABLE to False
        global CUDA_AVAILABLE
        original_cuda_available = CUDA_AVAILABLE
        
        # Test CPU implementation
        CUDA_AVAILABLE = False
        result_cpu = apply_rotary_emb(x_cpu.clone(), cos_cpu, sin_cpu, interleaved=True)
        qkv_result_cpu = apply_rotary_emb_qkv_(qkv_cpu.clone(), cos_cpu, sin_cpu, interleaved=True)
        
        # Test with CUDA available flag (but still on CPU device)
        CUDA_AVAILABLE = original_cuda_available
        result_cpu_with_cuda_flag = apply_rotary_emb(x_cpu.clone(), cos_cpu, sin_cpu, interleaved=True)
        qkv_result_cpu_with_cuda_flag = apply_rotary_emb_qkv_(qkv_cpu.clone(), cos_cpu, sin_cpu, interleaved=True)
        
        # Compare results
        print(f"Basic rotary embedding - Max difference: {torch.max(torch.abs(result_cpu - result_cpu_with_cuda_flag)).item():.2e}")
        print(f"QKV rotary embedding - Max difference: {torch.max(torch.abs(qkv_result_cpu - qkv_result_cpu_with_cuda_flag)).item():.2e}")
        
        # Test with different parameters
        print("\nTesting with different parameters:")
        
        # Test with seqlen_offsets
        seqlen_offset = 5
        result_offset_cpu_false = apply_rotary_emb(x_cpu.clone(), cos_cpu, sin_cpu, seqlen_offsets=seqlen_offset, interleaved=True)
        CUDA_AVAILABLE = False
        result_offset_cpu_true = apply_rotary_emb(x_cpu.clone(), cos_cpu, sin_cpu, seqlen_offsets=seqlen_offset, interleaved=True)
        CUDA_AVAILABLE = original_cuda_available
        
        print(f"With seqlen_offset - Max difference: {torch.max(torch.abs(result_offset_cpu_false - result_offset_cpu_true)).item():.2e}")
        
        # Test with tensor seqlen_offsets
        seqlen_offsets_tensor = torch.randint(0, 5, (batch_size,), device=device_cpu)
        result_tensor_offset_cpu_false = apply_rotary_emb(x_cpu.clone(), cos_cpu, sin_cpu, seqlen_offsets=seqlen_offsets_tensor, interleaved=True)
        CUDA_AVAILABLE = False
        result_tensor_offset_cpu_true = apply_rotary_emb(x_cpu.clone(), cos_cpu, sin_cpu, seqlen_offsets=seqlen_offsets_tensor, interleaved=True)
        CUDA_AVAILABLE = original_cuda_available
        
        print(f"With tensor seqlen_offsets - Max difference: {torch.max(torch.abs(result_tensor_offset_cpu_false - result_tensor_offset_cpu_true)).item():.2e}")
        
        # Test non-interleaved mode
        CUDA_AVAILABLE = original_cuda_available
        result_non_interleaved_cuda = apply_rotary_emb(x_cpu.clone(), cos_cpu, sin_cpu, interleaved=False)
        CUDA_AVAILABLE = False
        result_non_interleaved_cpu = apply_rotary_emb(x_cpu.clone(), cos_cpu, sin_cpu, interleaved=False)
        CUDA_AVAILABLE = original_cuda_available
        
        print(f"Non-interleaved mode - Max difference: {torch.max(torch.abs(result_non_interleaved_cuda - result_non_interleaved_cpu)).item():.2e}")
        
        # Test backward pass
        print("\nTesting backward pass:")
        
        # CPU implementation
        CUDA_AVAILABLE = False
        x_cpu_grad = x_cpu.clone().requires_grad_(True)
        result_cpu_grad = apply_rotary_emb(x_cpu_grad, cos_cpu, sin_cpu, interleaved=True)
        loss_cpu = result_cpu_grad.sum()
        loss_cpu.backward()
        grad_cpu = x_cpu_grad.grad.clone()
        
        # Reset and test with CUDA flag
        CUDA_AVAILABLE = original_cuda_available
        x_cpu_grad2 = x_cpu.clone().requires_grad_(True)
        result_cpu_grad2 = apply_rotary_emb(x_cpu_grad2, cos_cpu, sin_cpu, interleaved=True)
        loss_cpu2 = result_cpu_grad2.sum()
        loss_cpu2.backward()
        grad_cpu2 = x_cpu_grad2.grad.clone()
        
        print(f"Backward pass - Max gradient difference: {torch.max(torch.abs(grad_cpu - grad_cpu2)).item():.2e}")
        
        # Test RotaryEmbedding module
        print("\nTesting RotaryEmbedding module:")
        
        rotary_emb_cpu = RotaryEmbedding(rotary_dim, device=device_cpu)
        qkv_test = torch.randn(batch_size, seq_len, 3, n_heads, head_dim, device=device_cpu)
        
        CUDA_AVAILABLE = False
        result_module_cpu = rotary_emb_cpu(qkv_test.clone())
        
        CUDA_AVAILABLE = original_cuda_available
        result_module_cuda_flag = rotary_emb_cpu(qkv_test.clone())
        
        print(f"RotaryEmbedding module - Max difference: {torch.max(torch.abs(result_module_cpu - result_module_cuda_flag)).item():.2e}")
        
        # Restore original CUDA_AVAILABLE setting
        CUDA_AVAILABLE = original_cuda_available
        
        print("\nAll tests completed!")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA_AVAILABLE flag: {CUDA_AVAILABLE}")
        
        # If CUDA is available, test actual CUDA vs CPU
        if torch.cuda.is_available() and CUDA_AVAILABLE:
            print("\nTesting actual CUDA vs CPU implementation:")
            
            device_cuda = torch.device('cuda')
            x_cuda = x_cpu.clone().to(device_cuda)
            cos_cuda = cos_cpu.to(device_cuda)
            sin_cuda = sin_cpu.to(device_cuda)
            
            result_cuda = apply_rotary_emb(x_cuda, cos_cuda, sin_cuda, interleaved=True)
            result_cuda_cpu = result_cuda.cpu()
            
            print(f"CUDA vs CPU - Max difference: {torch.max(torch.abs(result_cpu - result_cuda_cpu)).item():.2e}")
    
    # Run the test
    test_rotary_embedding_consistency()

