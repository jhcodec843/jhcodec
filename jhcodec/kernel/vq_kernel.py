import triton
import triton.language as tl
import torch


@triton.jit
def vq_kernel_euclidean(
    value_ptr,                # pointer to [B, T, C] float32
    codebook_ptr,             # pointer to [V, C] float32
    codebook_norm_squared_ptr,        # pointer to [V] float32
    selected_codebook_ptr,    # pointer to [B, T] int32
    BT, C, V, 
    BLOCK_BT: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    bt = tl.program_id(0)
    offs_c = tl.arange(0, BLOCK_C)
    offs_v = tl.arange(0, BLOCK_V)
    offs_t = tl.arange(0, BLOCK_BT)
    value_ptr += bt * C * BLOCK_BT
    selected_codebook_ptr += bt * BLOCK_BT
    # Load first value and transpose
    codebook = tl.load(codebook_ptr + offs_v[None,:] * C + offs_c[:,None], 
                       mask=(offs_c[:,None] < C) & (offs_v[None,:] < V), other=0.0) #[C,V]
    #codebook_norm_squared = tl.sum(codebook * codebook, axis=0)[None,:] # [1,V]
    codebook_norm_squared = tl.load(codebook_norm_squared_ptr + offs_v[None,:], 
                       mask=(offs_v[None,:] < V), other=1000000.0) #[1,V]
    value = tl.load(value_ptr + offs_t[:,None] * C + offs_c[None,:], 
                    mask=(offs_t[:,None] + bt * BLOCK_BT < BT)&(offs_c[None,:] < C), other=0.0) #[bt,C]
    # actually, value_norm_square is not needed to calculate the argmin(distance(x,codebook)), since it is the same for all codebooks
    # value_norm_squared = tl.sum(value * value, axis=1)[:,None] #[bt,1]
   
    # Use manual dot product instead of tl.dot to avoid BLOCK_C<16 constraint
    if BLOCK_C < 16:
        dist_codebook = codebook_norm_squared - 2 * tl.sum(value[:,:,None] * codebook_norm_squared[None,:,:], axis=1) #[BT,V]
    else:
        dist_codebook = codebook_norm_squared - 2 * tl.dot(value, codebook) #[BT,V]
    selected_codebook = tl.argmin(dist_codebook, axis=1) #[BT]
    tl.store(selected_codebook_ptr + offs_t, selected_codebook, mask=offs_t + bt * BLOCK_BT < BT)



# @torch._dynamo.disable
# <1% indices are different, however it does not allocate extra memory for calculating the distance
# DO NOT CHANGE C And V, IT IS OPTIMIZED FOR C=16 and V=1024
def vq_triton(value, codebook):
    """
    Triton kernel for greedy segmentation.
    Args:
        value (torch.Tensor): (B, T, C), float32
        codebook (torch.Tensor): (V, C), float32
    Returns:

        selected_codebook: (B, T) int32
    """
    value = value.contiguous()
    codebook = codebook.contiguous()
    value = value.to(torch.float32)
    codebook = codebook.to(torch.float32)
    assert not value.isnan().any(), f"value: {value}"
    assert not codebook.isnan().any(), f"codebook: {codebook}"
    codebook_norm_squared = codebook.norm(dim=-1).square() # [V]
    B, T, C = value.shape
    selected_codebook = torch.empty((B, T), dtype=torch.long, device=value.device, requires_grad=False)
    assert value.shape[-1] == codebook.shape[-1], f"value.shape[-1] = {value.shape[-1]} != codebook.shape[-1] = {codebook.shape[-1]}"
    device = value.device
    V = codebook.shape[0]
    block_c = triton.next_power_of_2(C)
    block_v = triton.next_power_of_2(V)
    #block_bt = triton.next_power_of_2(B*T)
    BT = B*T
    block_bt = 128 if BT >= 128 else (triton.next_power_of_2(BT) if BT > 16 else 16)
    grid = (triton.cdiv(BT, block_bt), ) #lambda meta: (triton.cdiv(BT, meta['BLOCK_BT']), )
    vq_kernel = vq_kernel_euclidean 

    # Launch kernel
    with torch.cuda.device(value.device.index):
        vq_kernel[grid](
            value,                # value_ptr: [B, T, C] float32
            codebook,             # codebook_ptr: [C, V] float32
            codebook_norm_squared,        # codebook_norm_squared_ptr: [V] float32
            selected_codebook,    # selected_codebook_ptr: [B, T] int32 (output)
            BT, C, V,
            BLOCK_C=block_c,
            BLOCK_V=block_v,
            BLOCK_BT=block_bt,
        )
    return selected_codebook    