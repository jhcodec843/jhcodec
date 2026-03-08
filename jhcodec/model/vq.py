import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
if torch.cuda.is_available():
    USE_TRITON = True  
    # it will reduce the memory usage
    if USE_TRITON:
        from jhcodec.kernel.vq_kernel import vq_triton
else:
    USE_TRITON = False

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
logging.basicConfig(level=logging.INFO)



class VQ(nn.Module):
    def __init__(self, codebook_size, latent_dim=8, compute_dtype: torch.dtype = torch.float32, codebook_index: int = 0):
        super(VQ, self).__init__()
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        init_value = torch.randn((codebook_size, latent_dim), dtype=compute_dtype)
        init_value = init_value / torch.norm(init_value, dim=1, keepdim=True)
        self.codebook = nn.Embedding(codebook_size, latent_dim, _weight=init_value, dtype=compute_dtype)
        self.mse = nn.MSELoss(reduction='none')
        self.compute_dtype = compute_dtype
        counts_avg = torch.ones((self.codebook_size), device=self.codebook.weight.device, dtype=torch.float32)
        self.register_buffer('counts_avg', counts_avg)
        self.decay_rate = 0.99
        self.dead_criteria = 0.90 
        usage = torch.ones((), device=self.codebook.weight.device, dtype=torch.float32)
        self.register_buffer('usage', usage)
        self.codebook_index = codebook_index

    @torch.no_grad()
    def distance(self, x, y): 
        x_norm = torch.norm(x, dim=-1, keepdim=True) # [B,T,1]
        y_norm = torch.norm(y, dim=-1, keepdim=True) # [V,1]
        d = x_norm.square() + y_norm.square().view(1,1,-1) - 2 * torch.einsum('btc, vc->btv', x, y)
        return d
        
    # for the case of non-triton, euclidean distance
    def forward_(self, x, temperature=1.0): # x: [B,T,C]
        x = x.to(self.compute_dtype)
        with torch.no_grad():
            dist = self.distance(x, self.codebook.weight)
            #if self.training:
            #    noise = torch.zeros_like(dist).uniform_(0, 1)
            #    gumbel_noise = -torch.log(-torch.log(noise.clamp(min=1e-20))).clamp(min=1e-20)
            #    dist = dist/temperature + gumbel_noise
            selected_indices = torch.argmin(dist, dim=-1) # [B,T]
        vq_embed = self.codebook(selected_indices) # [B,T,C]
        if self.training:
            self.counts_ema(self.count(selected_indices))
            vq_loss = self.mse(vq_embed, x.detach()).mean(dim = [1,2]) # [B]
            commit_loss = self.mse(vq_embed.detach(), x).mean(dim = [1,2]) # [B]

            #vq_embed = self.rotation(x, vq_embed)
            vq_embed = self.straight_through_estimation(x, vq_embed)
            self.expires_code(x)  
        else:
            vq_loss = 0
            commit_loss = 0
        return vq_embed, selected_indices, vq_loss, commit_loss

    def get_very_efficient_rotation(self, u, q, e):
        w = ((u + q) / (torch.norm(u + q, dim=-1, keepdim=True).clamp(min=1e-6))).detach() #[B,T,C]
        # Original:
        # e = e - 2 * torch.bmm(torch.bmm(e, w.unsqueeze(-1)), w.unsqueeze(1)) + 2 * torch.bmm(
        #     torch.bmm(e, u.unsqueeze(-1).detach()), q.unsqueeze(1).detach())
        # Efficient version for [B, T, C] tensors:
        eww = (e*w).sum(-1, keepdim=True) * w  # [B, T, C]
        euq = (e * u.detach()).sum(-1, keepdim=True) * q.detach()  # [B, T, C]
        e = e - 2 * eww + 2 * euq
        return e

    def rotation(self, x, vq_embed):
        x_norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-6)
        vq_embed_norm = torch.norm(vq_embed, dim=-1, keepdim=True).clamp(min=1e-6)
        prenorm_vq_embed = self.get_very_efficient_rotation(x / x_norm, 
                                                            vq_embed / vq_embed_norm, 
                                                            x)
        vq_embed = prenorm_vq_embed * (vq_embed_norm / x_norm).detach()  # [B,T,C]
        return vq_embed

    def straight_through_estimation(self, x, vq_embed):
        return x + (vq_embed - x).detach() 
    
    def forward(self, x): # x: [B,T,C]
        with torch.no_grad():
            if USE_TRITON:
                selected_indices = vq_triton(x, self.codebook.weight) # [B,T]
            else:
                selected_indices = torch.argmin(self.distance(x, self.codebook.weight), dim=-1) # [B,T]
        vq_embed = self.codebook(selected_indices) # [B,T,C]
        if self.training:
            self.counts_ema(self.count(selected_indices))
            vq_loss = self.mse(vq_embed, x.detach()).mean(dim = [1,2]) # [B]
            commit_loss = self.mse(vq_embed.detach(), x).mean(dim = [1,2]) # [B]

            #vq_embed = self.rotation(x, vq_embed)
            vq_embed = self.straight_through_estimation(x, vq_embed)
            self.expires_code(x)
        else:
            vq_loss = torch.zeros([], device=x.device, dtype=x.dtype)
            commit_loss = torch.zeros([], device=x.device, dtype=x.dtype)
        return vq_embed, selected_indices, vq_loss, commit_loss

    @torch.no_grad()
    def count(self, indices):
        B, T = indices.shape
        counts = torch.zeros((self.codebook_size), device=indices.device, dtype=torch.int32)
        counts = torch.bincount(indices.view(-1), minlength=self.codebook_size)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM, async_op=False)
        return counts

    @torch.no_grad()
    def counts_ema(self, counts):
        self.counts_avg.mul_(self.decay_rate).add_(counts, alpha=1 - self.decay_rate)

    @torch.no_grad()
    def usage_update(self):
        self.usage = (self.counts_avg > self.dead_criteria).float().mean()

    @torch.no_grad()
    def expires_code(self, batch_samples):
        self.usage_update()
        unused_codebooks = (self.counts_avg < self.dead_criteria)
        if unused_codebooks.any():
            batch_samples_flatten = batch_samples.view(-1, batch_samples.shape[-1])
            bt = batch_samples_flatten.shape[0]
            random_sample_indices = torch.randint(0, bt, (self.codebook_size,), device=batch_samples.device, dtype=torch.long)
            random_sample = batch_samples_flatten[random_sample_indices] # [codebook_size, C]
            modified_codebook = torch.where(unused_codebooks[..., None], random_sample, self.codebook.weight.data)
            self.codebook.weight.data.copy_(modified_codebook)   
            dist.broadcast(self.codebook.weight.data, src=0, async_op=False)
            unused_codebooks_indices = torch.arange(self.codebook_size, device=batch_samples.device, dtype=torch.long)[unused_codebooks]
            logging.info(f"Expires codebooks: {unused_codebooks_indices} for codebook {self.codebook_index}")
        return

    def encode(self, x): # x: [B,T,C]
        with torch.no_grad():
            if USE_TRITON:
                selected_indices = vq_triton(x, self.codebook.weight) # [B,T]
            else:
                selected_indices = torch.argmin(self.distance(x, self.codebook.weight), dim=-1) # [B,T]
        return selected_indices

    def decode(self, indices): # indices: [B,T]
        return self.codebook(indices)
    

# neglect ste, vqloss and commit_loss for RVQfixed
class VQforRVQ(VQ):
    def __init__(self,  codebook_size, latent_dim=8, compute_dtype: torch.dtype = torch.float32, codebook_index: int=0):
        super(VQforRVQ, self).__init__(codebook_size, latent_dim, compute_dtype, codebook_index)

    def forward(self, x): # x: [B,T,C]
        with torch.no_grad():
            if USE_TRITON:
                selected_indices = vq_triton(x, self.codebook.weight) # [B,T]
            else:
                selected_indices = torch.argmin(self.distance(x, self.codebook.weight), dim=-1) # [B,T]
        vq_embed = self.codebook(selected_indices) # [B,T,C]
        if self.training:
            self.counts_ema(self.count(selected_indices))
            self.expires_code(x)
        return vq_embed, selected_indices


# Wrong case 1
class RVQNaiveGrad(nn.Module):
    def __init__(self, config, compute_dtype: torch.dtype = torch.float32):
        super(RVQNaiveGrad, self).__init__()
        self.num_codebooks = config.num_codebooks
        self.codebook_size = config.codebook_size
        self.embedding_dim = config.embedding_dim
        self.vqs = nn.ModuleList([VQ(config.codebook_size, config.latent_dim, compute_dtype, i)
                                  for i in range(config.num_codebooks)])
        self.updown_linears = config.updown_linears
        if self.updown_linears:
            self.downs = nn.ModuleList([nn.Linear(config.embedding_dim, config.latent_dim) for _ in range(config.num_codebooks)])
            self.ups = nn.ModuleList([nn.Linear(config.latent_dim, config.embedding_dim) for _ in range(config.num_codebooks)])
        self.compute_dtype = compute_dtype
        for i in range(config.num_codebooks):
            rand_init = torch.randn((config.codebook_size, config.latent_dim), dtype=compute_dtype)
            normalize = 2**(-0.5 * max(0.25, i))
            rand_init = rand_init * normalize
            self.vqs[i].codebook.weight.data.copy_(rand_init)

    def forward(self, x, n_codebooks=None): # x: [B,T,C], n_codebooks: [B]
        if n_codebooks is None:
            raise ValueError(f"n_codebooks must be provided")
        n_codebooks = n_codebooks.view(-1,1,1) # [B,1,1]
        #n_codebooks_max = n_codebooks.max() # it caused some error so we need to use all codebooks to prevent unused codebooks
        assert not x.isnan().any(), f"x: {x}"

        residuals = x
        rvq_embed = torch.zeros_like(x)
        indices = []
        total_vq_loss = torch.zeros([], device=x.device, dtype=x.dtype) # loss no dimension
        total_commit_loss = torch.zeros([], device=x.device, dtype=x.dtype) # loss no dimension

        with torch.no_grad():
            mask = torch.arange(self.num_codebooks, device=x.device).unsqueeze(0) < n_codebooks.squeeze(-1)  # [B,N]
            mask = mask.to(x.dtype)

        for i in range(self.num_codebooks):
            if self.updown_linears:
                latent = self.downs[i](residuals)
            else:
                latent = residuals
            vq_latent, selected_indices, vq_loss, commit_loss = self.vqs[i](latent)
            indices.append(selected_indices)
            if self.updown_linears:
                vq_embed = self.ups[i](vq_latent)
            else:
                vq_embed = vq_latent
            ###### WRONG PART ######
            # because of this detach, the gradients of partial{final rvq_embed}/partial{x} becomples multiple of n_codebooks
            residuals = residuals - vq_embed.detach()
            rvq_embed +=  vq_embed * mask[:,i].unsqueeze(-1).unsqueeze(-1)
            total_vq_loss += torch.mean(vq_loss * mask[:,i]) 
            total_commit_loss += torch.mean(commit_loss * mask[:,i])

        indices = torch.stack(indices, dim=-1) # [B,T,n_codebooks]
        return rvq_embed, indices, total_vq_loss, total_commit_loss

    def encode(self, x, n_codebooks=None): # x: [B,T,C], n_codebooks: [B]
        if n_codebooks is None:
            raise ValueError(f"n_codebooks must be provided")
        n_codebooks = n_codebooks.view(-1,1,1) # [B,1,1]
        n_codebooks_max = n_codebooks.max()
        mask = torch.arange(self.num_codebooks, device=x.device).unsqueeze(0) < n_codebooks.squeeze(-1)  # [B,N]
        mask = mask.to(x.dtype)       
        residuals = x
        #rvq_embed = torch.zeros_like(x)
        indices = []
        for i in range(n_codebooks_max):
            if self.updown_linears:
                latent = self.downs[i](residuals)
            else:
                latent = residuals
            encoded_indices = self.vqs[i].encode(latent) # [B,T]
            indices.append(encoded_indices)
            if self.updown_linears:
                vq_embed = self.ups[i](self.vqs[i].decode(encoded_indices))
            else:
                vq_embed = self.vqs[i].decode(encoded_indices)
            residuals = residuals - vq_embed
            #rvq_embed += vq_embed * mask[:,i]
        rvq_indices = torch.stack(indices, dim=-1) # [B,T,n_codebooks]
        return rvq_indices

    def decode(self, indices, n_codebooks=None): # indices: [B,T,n_codebooks]
        if n_codebooks is None:
            raise ValueError(f"n_codebooks must be provided")
        n_codebooks = n_codebooks.view(-1,1,1) # [B,1,1]
        n_codebooks_max = n_codebooks.max()

        B, T, C = indices.shape
        assert C >= n_codebooks_max, f"C = {C} < n_codebooks_max = {n_codebooks_max}"
        with torch.no_grad():
            mask = torch.arange(self.num_codebooks, device=indices.device).unsqueeze(0) < n_codebooks.squeeze(-1)  # [B,N]
            mask = mask.to(torch.float32)
        rvq_embed = torch.zeros((B, T, self.embedding_dim), device=indices.device, dtype=torch.float32)
        for i in range(n_codebooks_max):
            if self.updown_linears:
                # If precomputed up+VQ modules are registered, use them for faster decode.
                if getattr(self, "up_vqs", None) is not None:
                    decoded = self.up_vqs[i].decode(indices[:, :, i])
                else:
                    decoded = self.ups[i](self.vqs[i].decode(indices[:, :, i]))
            else:
                decoded = self.vqs[i].decode(indices[:, :, i])
            rvq_embed = rvq_embed + decoded * mask[:,i:i+1].unsqueeze(-1)
        return rvq_embed
      

# Wrong case 2
class RVQNaiveCommit(nn.Module):
    def __init__(self, config, compute_dtype: torch.dtype = torch.float32):
        super(RVQNaiveCommit, self).__init__()
        self.num_codebooks = config.num_codebooks
        self.codebook_size = config.codebook_size
        self.embedding_dim = config.embedding_dim
        self.vqs = nn.ModuleList([VQ(config.codebook_size, config.latent_dim, compute_dtype, i)
                                  for i in range(config.num_codebooks)])
        self.updown_linears = config.updown_linears
        if self.updown_linears:
            self.downs = nn.ModuleList([nn.Linear(config.embedding_dim, config.latent_dim) for _ in range(config.num_codebooks)])
            self.ups = nn.ModuleList([nn.Linear(config.latent_dim, config.embedding_dim) for _ in range(config.num_codebooks)])
        self.compute_dtype = compute_dtype
        # Optional precomputed up+VQ decoders for faster inference-time decode.
        # Populated by `register_up_vq`.
        self.up_vqs = None
        for i in range(config.num_codebooks):
            rand_init = torch.randn((config.codebook_size, config.latent_dim), dtype=compute_dtype)
            normalize = 2**(-0.5 * min(8, i)) # depends on init
            rand_init = rand_init * normalize
            self.vqs[i].codebook.weight.data.copy_(rand_init)

    def forward(self, x, n_codebooks=None): # x: [B,T,C], n_codebooks: [B]
        if n_codebooks is None:
            raise ValueError(f"n_codebooks must be provided")
        n_codebooks = n_codebooks.view(-1,1,1) # [B,1,1]
        #n_codebooks_max = n_codebooks.max() # it caused some error so we need to use all codebooks to prevent unused codebooks
        assert not x.isnan().any(), f"x: {x}"

        residuals = x
        rvq_embed = torch.zeros_like(x)
        indices = []
        total_vq_loss = torch.zeros([], device=x.device, dtype=x.dtype) # loss no dimension
        total_commit_loss = torch.zeros([], device=x.device, dtype=x.dtype) # loss no dimension

        with torch.no_grad():
            mask = torch.arange(self.num_codebooks, device=x.device).unsqueeze(0) < n_codebooks.squeeze(-1)  # [B,N]
            mask = mask.to(x.dtype)

        for i in range(self.num_codebooks):
            if self.updown_linears:
                latent = self.downs[i](residuals)
            else:
                latent = residuals
            vq_latent, selected_indices, vq_loss, commit_loss = self.vqs[i](latent)
            indices.append(selected_indices)
            if self.updown_linears:
                vq_embed = self.ups[i](vq_latent)
            else:
                vq_embed = vq_latent
            residuals = residuals - vq_embed
            rvq_embed +=  vq_embed * mask[:,i].unsqueeze(-1).unsqueeze(-1)
            total_vq_loss += torch.mean(vq_loss * mask[:,i])
            ###### WRONG PART ###### actually in vqs, to be sure, RVQDAC is separated
            # if no updonw_linears, gradient of commit_loss =0 for i>0
            total_commit_loss += torch.mean(commit_loss * mask[:,i])

        indices = torch.stack(indices, dim=-1) # [B,T,n_codebooks]
        return rvq_embed, indices, total_vq_loss, total_commit_loss

    def encode(self, x, n_codebooks=None): # x: [B,T,C], n_codebooks: [B]
        if n_codebooks is None:
            B = x.shape[0]
            n_codebooks = torch.tensor([self.num_codebooks]*B, dtype=torch.long, device=x.device)
        n_codebooks = n_codebooks.view(-1,1,1) # [B,1,1]
        n_codebooks_max = n_codebooks.max()
        mask = torch.arange(self.num_codebooks, device=x.device).unsqueeze(0) < n_codebooks.squeeze(-1)  # [B,N]
        mask = mask.to(x.dtype)       
        residuals = x
        #rvq_embed = torch.zeros_like(x)
        indices = []
        for i in range(n_codebooks_max):
            if self.updown_linears:
                latent = self.downs[i](residuals)
            else:
                latent = residuals
            encoded_indices = self.vqs[i].encode(latent) # [B,T]
            indices.append(encoded_indices)
            if self.updown_linears:
                vq_embed = self.ups[i](self.vqs[i].decode(encoded_indices))
            else:
                vq_embed = self.vqs[i].decode(encoded_indices)
            residuals = residuals - vq_embed
            #rvq_embed += vq_embed * mask[:,i]
        rvq_indices = torch.stack(indices, dim=-1) # [B,T,n_codebooks]
        return rvq_indices

    def decode(self, indices, n_codebooks=None): # indices: [B,T,n_codebooks]
        if n_codebooks is None:
            raise ValueError(f"n_codebooks must be provided")
        n_codebooks = n_codebooks.view(-1,1,1) # [B,1,1]
        n_codebooks_max = n_codebooks.max()

        B, T, C = indices.shape
        assert C >= n_codebooks_max, f"C = {C} < n_codebooks_max = {n_codebooks_max}"
        with torch.no_grad():
            mask = torch.arange(self.num_codebooks, device=indices.device).unsqueeze(0) < n_codebooks.squeeze(-1)  # [B,N]
            mask = mask.to(torch.float32)
        rvq_embed = torch.zeros((B, T, self.embedding_dim), device=indices.device, dtype=torch.float32)
        for i in range(n_codebooks_max):
            if self.updown_linears:
                # If precomputed up+VQ modules are registered, use them for faster decode.
                if getattr(self, "up_vqs", None) is not None:
                    decoded = self.up_vqs[i].decode(indices[:, :, i])
                else:
                    decoded = self.ups[i](self.vqs[i].decode(indices[:, :, i]))
            else:
                decoded = self.vqs[i].decode(indices[:, :, i])
            rvq_embed = rvq_embed + decoded * mask[:,i:i+1].unsqueeze(-1)
        return rvq_embed
     

# fix n * grad, vq/commit only at the end
# this case do not use updown_linears
class RVQFixedSTE(nn.Module):
    def __init__(self, config, compute_dtype: torch.dtype = torch.float32):
        super(RVQFixedSTE, self).__init__()
        self.num_codebooks = config.num_codebooks
        self.codebook_size = config.codebook_size
        self.embedding_dim = config.embedding_dim
        self.vqs = nn.ModuleList([VQforRVQ(config.codebook_size, config.latent_dim, compute_dtype, i)
                                  for i in range(config.num_codebooks)])
        self.updown_linears = config.updown_linears
        if not self.updown_linears:
            assert config.embedding_dim == config.latent_dim, f"embedding_dim must be equal to latent_dim when updown_linears is False, but got {config.embedding_dim} and {config.latent_dim}"
        if self.updown_linears:
            self.downs = nn.ModuleList([nn.Linear(config.embedding_dim, config.latent_dim) for _ in range(config.num_codebooks)])
            self.ups = nn.ModuleList([nn.Linear(config.latent_dim, config.embedding_dim) for _ in range(config.num_codebooks)])
        self.compute_dtype = compute_dtype
        # Optional precomputed up+VQ decoders for faster inference-time decode.
        # Populated by `register_up_vq`.
        self.up_vqs = None
        for i in range(config.num_codebooks):
            rand_init = torch.randn((config.codebook_size, config.latent_dim), dtype=compute_dtype)
            normalize = 2**(-0.5 * min(8, i)) # depends on init
            rand_init = rand_init * normalize
            self.vqs[i].codebook.weight.data.copy_(rand_init)
        self.mse = nn.MSELoss(reduction='none')

    def straight_through_estimation(self, x, vq_embed):
        return x + (vq_embed - x).detach() 

    @torch.no_grad()
    def register_up_vq(self):
        """
        Precompute and register VQ modules whose codebooks already include the
        effect of the corresponding `ups` linear layer.

        After calling this method, `decode` will use these faster lookup-only
        modules instead of computing `ups[i](vqs[i].decode(...))` on the fly.
        """
        if not self.updown_linears:
            # Nothing to register if we are not using updown linears.
            return

        device = self.vqs[0].codebook.weight.device
        up_vqs = nn.ModuleList()
        for vq, up in zip(self.vqs, self.ups):
            # Compose the codebook embedding with the linear "up" projection:
            # up(codebook[j]) for each entry j in the codebook.
            combined_weight = up(vq.codebook.weight.to(device))  # [codebook_size, embedding_dim]

            # Create a new VQ that will only be used for decode (embedding lookup).
            up_vq = VQ(
                codebook_size=self.codebook_size,
                latent_dim=self.embedding_dim,
                compute_dtype=self.compute_dtype,
                codebook_index=vq.codebook_index,
            ).to(device)
            up_vq.codebook.weight.data.copy_(combined_weight.to(up_vq.codebook.weight.data.dtype))
            # This is an inference-time helper; gradients are not required.
            up_vq.requires_grad_(False)
            up_vqs.append(up_vq)

        self.up_vqs = up_vqs

    def forward(self, x, n_codebooks=None): # x: [B,T,C], n_codebooks: [B]
        if n_codebooks is None:
            raise ValueError(f"n_codebooks must be provided")
        n_codebooks = n_codebooks.view(-1,1,1) # [B,1,1]
        #n_codebooks_max = n_codebooks.max() # it caused some error so we need to use all codebooks to prevent unused codebooks
        assert not x.isnan().any(), f"x: {x}"

        residuals = x
        rvq_embed = torch.zeros_like(x)
        indices = []
        total_vq_loss = torch.zeros([], device=x.device, dtype=x.dtype) # loss no dimension
        total_commit_loss = torch.zeros([], device=x.device, dtype=x.dtype) # loss no dimension

        with torch.no_grad():
            mask = torch.arange(self.num_codebooks, device=x.device).unsqueeze(0) < n_codebooks.squeeze(-1)  # [B,N]
            mask = mask.to(x.dtype)

        for i in range(self.num_codebooks):
            if self.updown_linears:
                latent = self.downs[i](residuals)
            else:
                latent = residuals
            vq_latent, selected_indices = self.vqs[i](latent)
            indices.append(selected_indices)
            if self.updown_linears:
                vq_embed = self.ups[i](vq_latent)
            else:
                vq_embed = vq_latent
            residuals = residuals - vq_embed
            rvq_embed +=  vq_embed * mask[:,i].unsqueeze(-1).unsqueeze(-1)
        total_vq_loss = self.mse(rvq_embed, x.detach()).mean()
        total_commit_loss = self.mse(rvq_embed.detach(), x).mean()
        rvq_embed = self.straight_through_estimation(x, rvq_embed)

        indices = torch.stack(indices, dim=-1) # [B,T,n_codebooks]
        return rvq_embed, indices, total_vq_loss, total_commit_loss

    def encode(self, x, n_codebooks=None): # x: [B,T,C], n_codebooks: [B]
        if n_codebooks is None:
            B = x.shape[0]
            n_codebooks = torch.tensor([self.num_codebooks]*B, dtype=torch.long, device=x.device)
        n_codebooks = n_codebooks.view(-1,1,1) # [B,1,1]
        n_codebooks_max = n_codebooks.max()
        mask = torch.arange(self.num_codebooks, device=x.device).unsqueeze(0) < n_codebooks.squeeze(-1)  # [B,N]
        mask = mask.to(x.dtype)       
        residuals = x
        #rvq_embed = torch.zeros_like(x)
        indices = []
        for i in range(n_codebooks_max):
            if self.updown_linears:
                latent = self.downs[i](residuals)
            else:
                latent = residuals
            encoded_indices = self.vqs[i].encode(latent) # [B,T]
            indices.append(encoded_indices)
            if self.updown_linears:
                vq_embed = self.ups[i](self.vqs[i].decode(encoded_indices))
            else:
                vq_embed = self.vqs[i].decode(encoded_indices)
            residuals = residuals - vq_embed
            #rvq_embed += vq_embed * mask[:,i]
        rvq_indices = torch.stack(indices, dim=-1) # [B,T,n_codebooks]
        return rvq_indices

    def decode(self, indices, n_codebooks=None): # indices: [B,T,n_codebooks]
        if n_codebooks is None:
            raise ValueError(f"n_codebooks must be provided")
        n_codebooks = n_codebooks.view(-1,1,1) # [B,1,1]
        n_codebooks_max = n_codebooks.max()

        B, T, C = indices.shape
        assert C >= n_codebooks_max, f"C = {C} < n_codebooks_max = {n_codebooks_max}"
        with torch.no_grad():
            mask = torch.arange(self.num_codebooks, device=indices.device).unsqueeze(0) < n_codebooks.squeeze(-1)  # [B,N]
            mask = mask.to(torch.float32)
        rvq_embed = torch.zeros((B, T, self.embedding_dim), device=indices.device, dtype=torch.float32)
        for i in range(n_codebooks_max):
            if self.updown_linears:
                # If precomputed up+VQ modules are registered, use them for faster decode.
                if getattr(self, "up_vqs", None) is not None:
                    decoded = self.up_vqs[i].decode(indices[:, :, i])
                else:
                    decoded = self.ups[i](self.vqs[i].decode(indices[:, :, i]))
            else:
                decoded = self.vqs[i].decode(indices[:, :, i])
            rvq_embed = rvq_embed + decoded * mask[:,i:i+1].unsqueeze(-1)
        return rvq_embed
      

# fix n * grad, vq/commit only at the end
class RVQDAC(nn.Module):
    def __init__(self, config, compute_dtype: torch.dtype = torch.float32):
        super(RVQDAC, self).__init__()
        self.num_codebooks = config.num_codebooks
        self.codebook_size = config.codebook_size
        self.embedding_dim = config.embedding_dim
        self.vqs = nn.ModuleList([VQ(config.codebook_size, config.latent_dim, compute_dtype, i) # RVQDAC should use VQ instead of VQforRVQ
                                  for i in range(config.num_codebooks)])
        self.updown_linears = config.updown_linears
        if not self.updown_linears:
            assert False, "updown_linears must be True for RVQDAC"
        if self.updown_linears:
            self.downs = nn.ModuleList([nn.Linear(config.embedding_dim, config.latent_dim) for _ in range(config.num_codebooks)])
            self.ups = nn.ModuleList([nn.Linear(config.latent_dim, config.embedding_dim) for _ in range(config.num_codebooks)])
        self.compute_dtype = compute_dtype
        for i in range(config.num_codebooks):
            rand_init = torch.randn((config.codebook_size, config.latent_dim), dtype=compute_dtype)
            normalize = 2**(-0.5 * min(8, i)) # depends on init
            rand_init = rand_init * normalize
            self.vqs[i].codebook.weight.data.copy_(rand_init)

    @torch.no_grad()
    def register_up_vq(self):
        """
        Precompute and register VQ modules whose codebooks already include the
        effect of the corresponding `ups` linear layer.

        After calling this method, `decode` will use these faster lookup-only
        modules instead of computing `ups[i](vqs[i].decode(...))` on the fly.
        """
        if not self.updown_linears:
            # Nothing to register if we are not using updown linears.
            return

        device = self.vqs[0].codebook.weight.device
        up_vqs = nn.ModuleList()
        for vq, up in zip(self.vqs, self.ups):
            # Compose the codebook embedding with the linear "up" projection:
            # up(codebook[j]) for each entry j in the codebook.
            combined_weight = up(vq.codebook.weight.to(device))  # [codebook_size, embedding_dim]

            # Create a new VQ that will only be used for decode (embedding lookup).
            up_vq = VQ(
                codebook_size=self.codebook_size,
                latent_dim=self.embedding_dim,
                compute_dtype=self.compute_dtype,
                codebook_index=vq.codebook_index,
            ).to(device)
            up_vq.codebook.weight.data.copy_(combined_weight.to(up_vq.codebook.weight.data.dtype))
            # This is an inference-time helper; gradients are not required.
            up_vq.requires_grad_(False)
            up_vqs.append(up_vq)

        self.up_vqs = up_vqs

    def train(self, mode: bool = True): #-> Self:
        if mode:
            # remove pre registered up+VQ decoders
            self.up_vqs = None
        else:
            self.register_up_vq()
            logging.info("Registered up+VQ decoders, DO NOT USE EVAL BEFORE LOADING CHECKPOINT")
        return super(RVQDAC, self).train(mode)

    def forward(self, x, n_codebooks=None): # x: [B,T,C], n_codebooks: [B]
        if n_codebooks is None:
            raise ValueError(f"n_codebooks must be provided")
        n_codebooks = n_codebooks.view(-1,1,1) # [B,1,1]
        #n_codebooks_max = n_codebooks.max() # it caused some error so we need to use all codebooks to prevent unused codebooks
        assert not x.isnan().any(), f"x: {x}"

        residuals = x
        rvq_embed = torch.zeros_like(x)
        indices = []
        total_vq_loss = torch.zeros([], device=x.device, dtype=x.dtype) # loss no dimension
        total_commit_loss = torch.zeros([], device=x.device, dtype=x.dtype) # loss no dimension

        with torch.no_grad():
            mask = torch.arange(self.num_codebooks, device=x.device).unsqueeze(0) < n_codebooks.squeeze(-1)  # [B,N]
            mask = mask.to(x.dtype)

        # different from RVQFixedSTE, RVQDAC calculate commit/vq loss inside the vq
        for i in range(self.num_codebooks):
            if self.updown_linears:
                latent = self.downs[i](residuals)
            else:
                latent = residuals
            vq_latent, selected_indices, vq_loss, commit_loss = self.vqs[i](latent)
            indices.append(selected_indices)
            if self.updown_linears:
                vq_embed = self.ups[i](vq_latent)
            else:
                vq_embed = vq_latent
            residuals = residuals - vq_embed
            rvq_embed +=  vq_embed * mask[:,i].unsqueeze(-1).unsqueeze(-1)
            total_vq_loss += torch.mean(vq_loss * mask[:,i])
            total_commit_loss += torch.mean(commit_loss * mask[:,i])

        indices = torch.stack(indices, dim=-1) # [B,T,n_codebooks]
        return rvq_embed, indices, total_vq_loss, total_commit_loss

    def encode(self, x, n_codebooks=None): # x: [B,T,C], n_codebooks: [B]
        if n_codebooks is None:
            B = x.shape[0]
            n_codebooks = torch.tensor([self.num_codebooks]*B, dtype=torch.long, device=x.device)
        n_codebooks = n_codebooks.view(-1,1,1) # [B,1,1]
        n_codebooks_max = n_codebooks.max()
        mask = torch.arange(self.num_codebooks, device=x.device).unsqueeze(0) < n_codebooks.squeeze(-1)  # [B,N]
        mask = mask.to(x.dtype)       
        residuals = x
        #rvq_embed = torch.zeros_like(x)
        indices = []
        for i in range(n_codebooks_max):
            if self.updown_linears:
                latent = self.downs[i](residuals)
            else:
                latent = residuals
            encoded_indices = self.vqs[i].encode(latent) # [B,T]
            indices.append(encoded_indices)
            if self.updown_linears:
                vq_embed = self.ups[i](self.vqs[i].decode(encoded_indices))
            else:
                vq_embed = self.vqs[i].decode(encoded_indices)
            residuals = residuals - vq_embed
            #rvq_embed += vq_embed * mask[:,i]
        rvq_indices = torch.stack(indices, dim=-1) # [B,T,n_codebooks]
        return rvq_indices

    def decode(self, indices, n_codebooks=None): # indices: [B,T,n_codebooks]
        if n_codebooks is None:
            raise ValueError(f"n_codebooks must be provided")
        n_codebooks = n_codebooks.view(-1,1,1) # [B,1,1]
        n_codebooks_max = n_codebooks.max()

        B, T, C = indices.shape
        assert C >= n_codebooks_max, f"C = {C} < n_codebooks_max = {n_codebooks_max}"
        with torch.no_grad():
            mask = torch.arange(self.num_codebooks, device=indices.device).unsqueeze(0) < n_codebooks.squeeze(-1)  # [B,N]
            mask = mask.to(torch.float32)
        rvq_embed = torch.zeros((B, T, self.embedding_dim), device=indices.device, dtype=torch.float32)
        for i in range(n_codebooks_max):
            if self.updown_linears:
                # Check if self.up_vqs exists and use it if available
                if getattr(self, "up_vqs", None) is not None:
                    rvq_embed = rvq_embed + self.up_vqs[i].decode(indices[:, :, i]) * mask[:,i:i+1].unsqueeze(-1)
                else:
                    rvq_embed = rvq_embed + self.ups[i](self.vqs[i].decode(indices[:,:,i])) * mask[:,i:i+1].unsqueeze(-1)
            else:
                rvq_embed = rvq_embed + self.vqs[i].decode(indices[:,:,i]) * mask[:,i:i+1].unsqueeze(-1)
        return rvq_embed

class RVQMimi(nn.Module):
    def __init__(self, config, compute_dtype: torch.dtype = torch.float32):
        super(RVQMimi, self).__init__()
        self.num_codebooks = config.num_codebooks
        self.codebook_size = config.codebook_size
        self.embedding_dim = config.embedding_dim
        self.semantic_vq = VQ(config.codebook_size, config.latent_dim, compute_dtype, 0)
        self.vqs = nn.ModuleList([VQ(config.codebook_size, config.latent_dim, compute_dtype, i+1) # RVQDAC should use VQ instead of VQforRVQ
                                  for i in range(config.num_codebooks-1)])
        self.updown_linears = config.updown_linears
        if not self.updown_linears:
            assert False, "updown_linears must be True for RVQMimi"
        if self.updown_linears:
            self.downs = nn.ModuleList([nn.Linear(config.embedding_dim, config.latent_dim) for _ in range(config.num_codebooks-1)])
            self.ups = nn.ModuleList([nn.Linear(config.latent_dim, config.embedding_dim) for _ in range(config.num_codebooks-1)])
            self.semantic_up = nn.Linear(config.latent_dim, config.embedding_dim)
            self.semantic_down = nn.Linear(config.embedding_dim, config.latent_dim)
            self.semantic_loss_up = nn.Linear(config.latent_dim, config.embedding_dim)
        self.compute_dtype = compute_dtype
        # Optional precomputed up+VQ decoders for faster inference-time decode.
        # Populated by `register_up_vq`.
        self.up_vqs = None
        self.semantic_up_vq = None
        rand_init = torch.randn((config.codebook_size, config.latent_dim), dtype=compute_dtype)
        normalize = 2**(-0.5 * min(8, 0)) # depends on init
        rand_init = rand_init * normalize
        self.semantic_vq.codebook.weight.data.copy_(rand_init)
        self.semantic_vq.dead_criteria = 0.001 # very conservative criteria
        for i in range(config.num_codebooks-1):
            rand_init = torch.randn((config.codebook_size, config.latent_dim), dtype=compute_dtype)
            normalize = 2**(-0.5 * min(8, i)) # depends on init
            rand_init = rand_init * normalize
            self.vqs[i].codebook.weight.data.copy_(rand_init)
        self.cossim = nn.CosineEmbeddingLoss()

    @torch.no_grad()
    def register_up_vq(self):
        """
        Precompute and register VQ modules whose codebooks already include the
        effect of the corresponding semantic / RVQ `up` linear layers.

        After calling this method, `decode` will use these faster lookup-only
        modules instead of computing `up(vq.decode(...))` on the fly.
        """
        if not self.updown_linears:
            # Nothing to register if we are not using updown linears.
            return

        device = self.semantic_vq.codebook.weight.device

        # Semantic (first) codebook.
        semantic_combined_weight = self.semantic_up(
            self.semantic_vq.codebook.weight.to(device)
        )  # [codebook_size, embedding_dim]
        semantic_up_vq = VQ(
            codebook_size=self.codebook_size,
            latent_dim=self.embedding_dim,
            compute_dtype=self.compute_dtype,
            codebook_index=self.semantic_vq.codebook_index,
        ).to(device)
        semantic_up_vq.codebook.weight.data.copy_(semantic_combined_weight.to(semantic_up_vq.codebook.weight.data.dtype))
        semantic_up_vq.requires_grad_(False)
        self.semantic_up_vq = semantic_up_vq

        # RVQ codebooks (rest).
        up_vqs = nn.ModuleList()
        for vq, up in zip(self.vqs, self.ups):
            combined_weight = up(vq.codebook.weight.to(device))  # [codebook_size, embedding_dim]
            up_vq = VQ(
                codebook_size=self.codebook_size,
                latent_dim=self.embedding_dim,
                compute_dtype=self.compute_dtype,
                codebook_index=vq.codebook_index,
            ).to(device)
            up_vq.codebook.weight.data.copy_(combined_weight.to(up_vq.codebook.weight.data.dtype))
            up_vq.requires_grad_(False)
            up_vqs.append(up_vq)

        self.up_vqs = up_vqs

    def train(self, mode: bool = True): #-> Self:
        if mode:
            # remove pre registered up+VQ decoders
            self.up_vqs = None
            self.semantic_up_vq = None
        else:
            self.register_up_vq()
            logging.info("Registered up+VQ decoders, DO NOT USE EVAL BEFORE LOADING CHECKPOINT")
        return super(RVQMimi, self).train(mode)

    def forward(self, x, semantic_gt, n_codebooks=None): # x: [B,T,C], n_codebooks: [B]
        if n_codebooks is None:
            raise ValueError(f"n_codebooks must be provided")
        n_codebooks = n_codebooks.view(-1, 1, 1)  # [B,1,1]
        assert not x.isnan().any(), f"x: {x}"

        B, T, C = x.shape
        residuals = x
        rvq_embed = torch.zeros_like(x)
        indices = []

        # Losses
        total_vq_loss = torch.zeros([], device=x.device, dtype=x.dtype)
        total_commit_loss = torch.zeros([], device=x.device, dtype=x.dtype)

        with torch.no_grad():
            mask = torch.arange(self.num_codebooks-1, device=x.device).unsqueeze(0) < (n_codebooks.squeeze(-1)-1)  # [1,N]
            mask = mask.to(x.dtype)

        # Semantic first codebook (special)
        semantic_latent = self.semantic_down(x)
        semantic_vq_embed, selected_semantic_indices, vq_loss, commit_loss = self.semantic_vq(semantic_latent)
        semantic_embed = self.semantic_up(semantic_vq_embed)
        rvq_embed = rvq_embed + semantic_embed

        # Cosine similarity (semantic loss), see CosineEmbeddingLoss API
        semantic_pred = self.semantic_loss_up(semantic_vq_embed).view(-1, semantic_gt.size(-1))
        semantic_gt = semantic_gt.view(-1, semantic_gt.size(-1))
        semantic_loss = self.cossim(semantic_pred, semantic_gt, torch.ones(semantic_pred.size(0), device=semantic_vq_embed.device))
        total_vq_loss += torch.mean(vq_loss)
        total_commit_loss += torch.mean(commit_loss)
        indices.append(selected_semantic_indices)


        # RVQ: apply rest of codebooks
        for i in range(self.num_codebooks-1):
            if self.updown_linears:
                latent = self.downs[i](residuals)
            else:
                latent = residuals
            vq_latent, selected_indices, vq_loss, commit_loss = self.vqs[i](latent)
            indices.append(selected_indices)
            if self.updown_linears:
                vq_embed = self.ups[i](vq_latent)
            else:
                vq_embed = vq_latent
            residuals = residuals - vq_embed
            rvq_embed += vq_embed * mask[:, i].unsqueeze(-1).unsqueeze(-1)
            total_vq_loss += torch.mean(vq_loss * mask[:, i])
            total_commit_loss += torch.mean(commit_loss * mask[:, i])

        indices = torch.stack(indices, dim=-1)  # [B,T,num_codebooks]
        return rvq_embed, indices, semantic_loss, total_vq_loss, total_commit_loss

    def encode(self, x, n_codebooks=None): # x: [B,T,C], n_codebooks: [B]
        # This should match the forward logic for indices. The output is [B,T,num_codebooks], where
        # the first index is for the semantic codebook, then (num_codebooks-1) RVQ codebooks.
        if n_codebooks is None:
            B = x.shape[0]
            n_codebooks = torch.tensor([self.num_codebooks]*B, dtype=torch.long, device=x.device)
        n_codebooks = n_codebooks.view(-1, 1, 1)  # [B, 1, 1]
        n_codebooks_max = n_codebooks.max()  

        B, T, C = x.shape
        indices = []

        # Semantic (first codebook)
        semantic_latent = self.semantic_down(x)
        selected_semantic_indices = self.semantic_vq.encode(semantic_latent)
        indices.append(selected_semantic_indices)

        # RVQ codebooks
        residuals = x
        for i in range(n_codebooks_max-1):
            if self.updown_linears:
                latent = self.downs[i](residuals)
            else:
                latent = residuals
            selected_indices = self.vqs[i].encode(latent)  # [B,T]
            indices.append(selected_indices)
            if self.updown_linears:
                vq_embed = self.ups[i](self.vqs[i].decode(selected_indices))
            else:
                vq_embed = self.vqs[i].decode(selected_indices)
            residuals = residuals - vq_embed

        rvq_indices = torch.stack(indices, dim=-1)  # [B,T,num_codebooks]
        return rvq_indices

    def decode(self, indices, n_codebooks=None): # indices: [B,T,num_codebooks]
        if n_codebooks is None:
            raise ValueError(f"n_codebooks must be provided")
        n_codebooks = n_codebooks.view(-1, 1, 1)  # [B,1,1]
        n_codebooks_max = n_codebooks.max()  
        
        B, T, C = indices.shape
        assert C >= n_codebooks_max, f"C = {C} < n_codebooks_max = {n_codebooks_max}"
        with torch.no_grad():
            mask = torch.arange(n_codebooks_max, device=indices.device).unsqueeze(0) < n_codebooks.squeeze(-1)  # [1,num_codebooks]
            mask = mask.to(indices.dtype)

        # First codebook: semantic
        if getattr(self, "semantic_up_vq", None) is not None:
            semantic_decoded = self.semantic_up_vq.decode(indices[:, :, 0]) 
        else:
            semantic_decoded = self.semantic_up(self.semantic_vq.decode(indices[:, :, 0]))
        rvq_embed = semantic_decoded

        # RVQ codebooks
        for i in range(n_codebooks_max-1):
            if getattr(self, "up_vqs", None) is not None:
                rvq_codebook_decode = self.up_vqs[i].decode(indices[:, :, i+1]) * mask[:, i+1].unsqueeze(-1).unsqueeze(-1)
            else:
                rvq_codebook_decode = self.ups[i](self.vqs[i].decode(indices[:, :, i+1])) * mask[:, i+1].unsqueeze(-1).unsqueeze(-1)
            rvq_embed += rvq_codebook_decode * mask[:, i+1].unsqueeze(-1).unsqueeze(-1)

        return rvq_embed




if __name__ == "__main__":
    B, T, C = 64, 1024, 16
    import jhcodec.utils as utils
    import os
    port = utils.find_free_port()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=1, rank=0)
    x = torch.randn(B, T, C, device='cuda') * 3
    vq = VQ(1024, 16)
    vq = vq.to('cuda')
    torch.cuda.synchronize()
    vq_embed, selected_indices, vq_loss, commit_loss = vq.forward_(x)
    print(vq_embed.shape, selected_indices.shape, vq_loss, commit_loss, selected_indices)
    vq_embed_triton, selected_indices_triton, vq_loss_triton, commit_loss_triton = vq.forward(x)
    print(vq_embed_triton.shape, selected_indices_triton.shape, vq_loss_triton, commit_loss_triton, selected_indices_triton)
    print(torch.allclose(vq_embed, vq_embed_triton))
    print(torch.allclose(selected_indices, selected_indices_triton))
    print((selected_indices==selected_indices_triton).float().sum())
    not_same_mask = (selected_indices!=selected_indices_triton)
    print(selected_indices[not_same_mask])
    print(selected_indices_triton[not_same_mask])
    print(vq_embed[not_same_mask] - vq_embed_triton[not_same_mask])
    dist.destroy_process_group()
