import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import argparse
from omegaconf import OmegaConf
from jhcodec.model.codec import JHCodecDAC, JHCodecMimi
from jhcodec.model.sw2v import AudioEncoder
from jhcodec.model.discriminator import VocoderDiscriminator, discriminator_loss, generator_loss, feature_matching_loss
from jhcodec.loss.multiscalemelspec import MultiScaleMelSpectrogramLoss
#from jhcodec.dataloader_librittsr import AudioDataset, collate_fn
from jhcodec.dataloader import AudioDataset, collate_fn
import jhcodec.utils as utils
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import logging
import time
from datetime import timedelta
import math
from phaseaug.phaseaug import PhaseAug

import torchaudio
import torch.nn.functional as F
import numpy as np

torch.autograd.set_detect_anomaly(True)


def run(rank, world_size, config):
    dist.init_process_group(backend="nccl", init_method="env://",
                            rank=rank, world_size=world_size, 
                            timeout=timedelta(seconds=2000))
    torch.cuda.set_device(rank)
    trainer = Trainer(rank, world_size, config)
    print(f"Rank {rank} is ready")
    trainer.train()

def get_lr_multiplier(current_step: int, warmup_steps: int, total_steps: int, min_lr: float):
    # Linear warmup phase
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    
    # Cosine decay phase after warmup
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

def slice_audios(audio1, audio2, segment_duration):
    # audio: [B, T] 
    # audio2: [B, T]
    # segment_duration: int
    # return: [B, segment_duration]
    B, T = audio1.shape
    B2, T2 = audio2.shape
    assert B == B2, f"audio1 and audio2 must have the same batch size but got {B} and {B2}"
    assert T == T2, f"audio1 and audio2 must have the same length but got {T} and {T2}"
    segment_start = torch.randint(0, T - segment_duration + 1, (B,))
    b = torch.randint(0, B, (1,)) # randomly select a batch to start from the beginning to prevent the strange artifact
    segment_start[b] = 0
    if B > 1:
        b2 = (b + torch.randint(1, max(B//3, 2), (1,))) % B # randomly select a batch to start from the beginning to prevent the strange artifact
        segment_start[b2] = 0
    if B > 2:
        e =  (b2 + torch.randint(1, max(B//3, 2), (1,))) % B # randomly select a batch to end at the ending to prevent the strange artifact
        segment_start[e] = T - segment_duration
    if B > 3:
        e2 = (e + torch.randint(1, max(B//3, 2), (1,))) % B # randomly select a batch to end at the ending to prevent the strange artifact
        segment_start[e2] = T - segment_duration
    segment_end = segment_start + segment_duration
    segments1 = torch.zeros((B, segment_duration), device=audio1.device, dtype=audio1.dtype)
    segments2 = torch.zeros((B, segment_duration), device=audio2.device, dtype=audio2.dtype)
    for i in range(B):
        segments1[i] = audio1[i, segment_start[i]:segment_end[i]]
        segments2[i] = audio2[i, segment_start[i]:segment_end[i]]
    return segments1, segments2


class Trainer:
    def __init__(self, rank, world_size, config):
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.initialize_components()
        if self.rank == 0:
            logging.basicConfig(level=logging.INFO)
            logging.info(f"Training with config: {self.config}")
            self.current_time = time.time()
        else:
            logging.basicConfig(level=logging.WARNING)
        self.segment_length = int(self.config.training.discriminator_segment_duration * self.config.data.sample_rate)

    def initialize_components(self):
        # Create model
        if self.config.model.rvq.type == "mimi":
            self.model = JHCodecMimi(self.config.model)
            self.model_type = "mimi"
        elif self.config.model.rvq.type == "dac":
            self.model = JHCodecDAC(self.config.model)
            self.model_type = "dac"
        else:
            raise ValueError(f"Invalid model type: {self.config.model.rvq.type}")
        self.model = self.model.to(self.rank, non_blocking=False)
       
        # Create discriminators
        self.discriminator = VocoderDiscriminator()
        self.discriminator = self.discriminator.to(self.rank, non_blocking=False)

        # Create w2v
        self.w2v = AudioEncoder(self.config.w2v)
        self.w2v = self.w2v.to(self.rank, non_blocking=False)
        self.w2v = self.w2v.eval()
        utils.load_checkpoint(self.w2v, None, None, self.config.training.sw2v_checkpoint, strict_model=True)
        
        
        # Define optimizers
        # normal optimizer
        self.optimizer_g = optim.AdamW(self.model.parameters(),
                                      lr=self.config.training.learning_rate,
                                      weight_decay=self.config.training.weight_decay,
                                      betas=(0.5, 0.99))
        # Got idea from lora+ https://arxiv.org/abs/2402.12354, since down linear shows higher gradient than the rest of the model,
        # To assign a lower learning rate to self.model.rvq.semantic_down and modules in self.model.rvq.downs,
        # group parameters and use different 'lr' for those params:
        # downs_params = []
        # Check if semantic_down exists and add its parameters to downs_params
        # if hasattr(self.model.rvq, "semantic_down") and self.model.rvq.semantic_down is not None:
        #     downs_params += list(self.model.rvq.semantic_down.parameters())
        # for down_module in self.model.rvq.downs:
        #     downs_params += list(down_module.parameters())

        # special_lr = self.config.training.learning_rate / 16  # e.g., decrease by 16 following lora+; adjust as needed

        # self.optimizer_g = optim.AdamW([
        #                                     {'params': [p for n, p in self.model.named_parameters()
        #                                                 if p not in set(downs_params)]},
        #                                     {'params': downs_params, 'lr': special_lr},
        #                                ],
        #                                lr=self.config.training.learning_rate,
        #                                weight_decay=self.config.training.weight_decay,
        #                                betas=(0.5, 0.99)
        # )

        if self.config.training.apply_apa:
            self.apa_prob = [torch.tensor(0.0) for _ in range(len(self.discriminator.names))]
        else:
            self.apa_prob = None
        
        # Combine discriminator parameters for single optimizer
        self.optimizer_d = optim.AdamW(self.discriminator.parameters(),
                                      lr=self.config.training.learning_rate,
                                      weight_decay=0,
                                      betas=(0.5, 0.99))
        
        # Calculate total steps for the scheduler
        total_steps = self.config.training.num_epochs * 100000  # Approximate
        warmup_steps = self.config.training.warmup_steps
        
        # Combined warmup and cosine decay scheduler
        self.lr_scheduler_g = optim.lr_scheduler.LambdaLR(
            self.optimizer_g,
            lambda step: get_lr_multiplier(step, warmup_steps, total_steps, self.config.training.min_lr))
        
        self.lr_scheduler_d = optim.lr_scheduler.LambdaLR(
            self.optimizer_d,
            lambda step: get_lr_multiplier(step, warmup_steps, total_steps, self.config.training.min_lr))

        # Initialize loss function
        self.mel_loss = MultiScaleMelSpectrogramLoss(
            sample_rate=self.config.data.sample_rate,
            range_start=6,
            range_end=11,
            n_mels=64,
            f_min=0.0,
            f_max=None,
            normalized=False,
            alphas=True,
            floor_level=1e-5
        ).to(self.rank)

        

        if self.config.training.resume:
            checkpoint_files = glob(os.path.join(self.config.logging.checkpoint_dir, "checkpoint_*.pt"))
            sorted_checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
            checkpoint_path = sorted_checkpoint_files[-1]
            del sorted_checkpoint_files
            del checkpoint_files
            _, _, _, self.epoch, self.global_step = utils.load_checkpoint(
                self.model, self.optimizer_g, self.lr_scheduler_g, checkpoint_path, 
                strict_model=self.config.training.strict_model)
            # Load discriminator checkpoint if exists
            if self.config.training.load_discriminator:
                disc_checkpoint_path = checkpoint_path.replace("checkpoint_", "discriminator_")
                if os.path.exists(disc_checkpoint_path):
                    utils.load_checkpoint(
                        self.discriminator, self.optimizer_d, self.lr_scheduler_d, disc_checkpoint_path,
                        strict_model=self.config.training.strict_model)
            self.global_step += 1
        else:
            if self.config.training.encoder_checkpoint is not None:
                utils.load_checkpoint(self.model, None, None, self.config.training.encoder_checkpoint, strict_model=False)
            if self.config.training.rect_checkpoint is not None:
                utils.load_checkpoint(self.model, None, None, self.config.training.rect_checkpoint, strict_model=False)
            if self.config.training.decoder_checkpoint is not None:
                utils.load_checkpoint(self.model, None, None, self.config.training.decoder_checkpoint, strict_model=False)
            if self.config.training.codec_checkpoint is not None:
                utils.load_checkpoint(self.model, None, None, self.config.training.codec_checkpoint, strict_model=False)
            self.epoch = 0
            self.global_step = 0

        self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=True)
        self.model_type = self.config.model.rvq.type
        self.discriminator = DDP(self.discriminator, device_ids=[self.rank], find_unused_parameters=True)



        if self.config.training.init_dataset:
            if self.rank == 0:
                dataset = AudioDataset(self.config.data.audio_dir,
                                        sample_rate=self.config.data.sample_rate,
                                        segment_duration=self.config.data.segment_duration,
                                        training=True,
                                        init_dataset=True,
                                        cache_dir=self.config.data.cache_dir,
                                        use_mel=False)
            if self.rank != 0:
                dataset = AudioDataset(self.config.data.audio_dir,
                                        sample_rate=self.config.data.sample_rate,
                                        segment_duration=self.config.data.segment_duration,
                                        training=True,
                                        init_dataset=False,
                                        cache_dir=self.config.data.cache_dir,
                                        use_mel=False)
        else:
            dataset = AudioDataset(self.config.data.audio_dir,
                                    sample_rate=self.config.data.sample_rate,
                                    segment_duration=self.config.data.segment_duration,
                                    training=True,
                                    init_dataset=False,
                                    cache_dir=self.config.data.cache_dir,
                                    use_mel=False)

        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True, drop_last=True)

        self.dataloader = DataLoader(dataset, 
                                    batch_size=self.config.training.batch_size, 
                                    sampler=sampler, 
                                    collate_fn=collate_fn, 
                                    pin_memory=True, 
                                    persistent_workers=True,
                                    prefetch_factor=2,
                                    num_workers=self.config.training.num_workers,
                                    multiprocessing_context="spawn")

        if self.rank == 0:
            self.validation_dataset = AudioDataset(self.config.data.audio_dir, 
                                                    sample_rate=self.config.data.sample_rate,
                                                    segment_duration=self.config.data.segment_duration,
                                                    training=False, 
                                                    init_dataset=True,
                                                    cache_dir=self.config.data.cache_dir,
                                                    use_mel=False)
        if self.rank != 0:
            self.validation_dataset = AudioDataset(self.config.data.audio_dir, 
                                                    sample_rate=self.config.data.sample_rate,
                                                    segment_duration=self.config.data.segment_duration,
                                                    training=False, 
                                                    init_dataset=False,
                                                    cache_dir=self.config.data.cache_dir,
                                                    use_mel=False)
        
        self.validation_dataloader = DataLoader(self.validation_dataset, 
                                                batch_size=self.config.training.batch_size, 
                                                shuffle=False, 
                                                collate_fn=collate_fn)
        
        # Initialize TensorBoard writer
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=self.config.logging.tensorboard_dir)
        if self.config.training.use_phaseaug:
            self.phaseaug = PhaseAug()
            self.phaseaug = self.phaseaug.to(self.rank, non_blocking=False)
        
        if self.config.training.loss_type == "cossim":
            self.loss = nn.CosineEmbeddingLoss()
        elif self.config.training.loss_type == "l2":
            self.loss = nn.MSELoss()
        elif self.config.training.loss_type == "l1":
            self.loss = nn.L1Loss()
        else:
            raise ValueError(f"Invalid loss type: {self.config.training.loss_type}")

    def train(self):
        start_epoch = self.epoch
        if self.global_step == 0:
            pass
        else:
            self.evaluate(self.epoch)
        
        for epoch in range(self.epoch, self.config.training.num_epochs):
            epoch, global_step = self.train_epoch(epoch, self.global_step)
        
        if self.rank == 0:
            self.writer.close()
        dist.destroy_process_group()

    def train_epoch(self, epoch, global_step):
        self.dataloader.sampler.set_epoch(epoch)
        counts_ema = None
        for batch_idx, batch in enumerate(self.dataloader):
            if self.global_step > self.config.training.masking_stop_steps:
                if self.global_step == self.config.training.masking_stop_steps:
                    logging.info(f"Masking stopped at step {self.global_step}")
                self.config.model.training.encoder_mask_rate = 0.0
                self.config.model.training.decoder_mask_rate = 0.0
                self.model.module.encoder_mask_rate = 0.0
                self.model.module.decoder_mask_rate = 0.0
            self.model.train()
            self.discriminator.train()
            
            # Unpack batch - assuming audio data without mel spectrograms
            clean_audios = batch
            B, T = clean_audios.shape
            if T % self.config.model.mlp_in.in_features > 0:
                pad_length = self.config.model.mlp_in.in_features - (T % self.config.model.mlp_in.in_features)
                clean_audios = F.pad(clean_audios, (0, pad_length))
                B, T = clean_audios.shape
            

            clean_audios = clean_audios.cuda(self.rank, non_blocking=True)
            #clean_audios = F.pad(clean_audios, (self.config.model.mlp_in.in_features * 3, 0 ))

            with torch.no_grad():
                use_noise = torch.rand((B,1), device=self.rank, dtype=torch.float32) < self.config.model.training.noise_augmentation
                use_noise = use_noise.float()
                noise_level = torch.rand((B,1), device=self.rank, dtype=torch.float32) * 2e-3
                noise = torch.randn_like(clean_audios) * noise_level
                time_idx = torch.arange(T, device=self.rank).unsqueeze(0)
                period = torch.rand((B,1), device=self.rank, dtype=torch.float32) * 16000
                sine_level = torch.rand((B,1), device=self.rank, dtype=torch.float32) * 0.001
                sine = torch.sin(2*3.141592653589793 * time_idx /period) * sine_level
                noise_audios = clean_audios + (noise + sine) * use_noise
            

            w2v_feature = self.w2v(clean_audios)
            if self.model_type == "mimi":
                decoded, indices, semantic_loss, total_vq_loss, total_commit_loss = self.model(noise_audios, w2v_feature)
            elif self.model_type == "semantic":
                decoded, indices, total_vq_loss, total_commit_loss = self.model(w2v_feature)
            else:
                decoded, indices, total_vq_loss, total_commit_loss = self.model(noise_audios)
            recon_loss = self.mel_loss(decoded, clean_audios)
            w2v_recon = self.w2v(decoded)


            with torch.no_grad():
                counts = utils.count(indices, self.config.model.rvq.codebook_size)
                counts_ema = counts.clone() if counts_ema is None else counts_ema * 0.9 + counts 
            # Train Discriminators
            if self.global_step > self.config.training.discriminator_start_steps:
                # slice
                clean_audios_slice, decoded_slice = slice_audios(clean_audios, decoded, self.segment_length)

                # if dataset contains noisy audios, g->d steps increased the stability
                self.optimizer_g.zero_grad()
                self.discriminator.eval()
                # recon losses

                if self.config.training.loss_type == "cossim":
                    w2v_feature = w2v_feature.view(-1, w2v_feature.size(-1))
                    w2v_recon = w2v_recon.view(-1, w2v_recon.size(-1))
                    w2v_loss = self.loss(w2v_feature, w2v_recon, torch.ones(w2v_feature.size(0), device=self.rank))
                elif self.config.training.loss_type == "l2" or self.config.training.loss_type == "l1":
                    w2v_loss = self.loss(w2v_feature, w2v_recon)
                else:
                    raise ValueError(f"Invalid loss type: {self.config.training.loss_type}")
                if self.config.training.use_phaseaug:
                    clean_audios_slice, decoded_slice = self.phaseaug.forward_sync(clean_audios_slice, decoded_slice)
                    clean_audios_slice = clean_audios_slice.squeeze(1)
                    decoded_slice = decoded_slice.squeeze(1)
            
                disc_fake_outputs_gen, disc_fake_features_gen = self.discriminator(decoded_slice)
                with torch.no_grad():
                    disc_real_outputs_gen, disc_real_features_gen = self.discriminator(clean_audios_slice)
                
                g_adv_loss = generator_loss(disc_fake_outputs_gen)
                
                # Feature matching loss
                fm_total_loss, fm_losses = feature_matching_loss(disc_real_features_gen, disc_fake_features_gen)
                
                # Total generator loss
                if self.model_type == "mimi":
                    g_loss = self.config.loss.recon_loss_weight * recon_loss + \
                            self.config.loss.adv_loss_weight * g_adv_loss + \
                            self.config.loss.fm_loss_weight * fm_total_loss + \
                            self.config.loss.vq_loss_weight * total_vq_loss + \
                            self.config.loss.commit_loss_weight * total_commit_loss + \
                            self.config.loss.w2v_loss_weight * w2v_loss + \
                            self.config.loss.semantic_loss_weight * semantic_loss
                else:
                    g_loss = self.config.loss.recon_loss_weight * recon_loss + \
                            self.config.loss.adv_loss_weight * g_adv_loss + \
                            self.config.loss.fm_loss_weight * fm_total_loss + \
                            self.config.loss.vq_loss_weight * total_vq_loss + \
                            self.config.loss.commit_loss_weight * total_commit_loss + \
                            self.config.loss.w2v_loss_weight * w2v_loss
                g_loss.backward()
                
                # Clip gradients by norm
                with torch.no_grad():
                    grad_norms = [(name, torch.norm(p.grad)) for name, p in self.model.named_parameters() if p.grad is not None]
                    grad_norm = torch.norm(torch.stack([norm for _, norm in grad_norms]))
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.training.max_grad_norm)
                
                self.optimizer_g.step()
                self.lr_scheduler_g.step()

                self.discriminator.train()
                self.optimizer_d.zero_grad()
                # apply phaseaug
                if self.config.training.use_phaseaug:
                    clean_audios_slice_orig = clean_audios_slice
                    decoded_slice_orig = decoded_slice
                    with torch.no_grad():
                        clean_audios_slice = self.phaseaug(clean_audios_slice_orig).squeeze(1)
                        decoded_slice = self.phaseaug(decoded_slice_orig).squeeze(1)
                # Discriminator outputs
                disc_real_outputs, disc_real_features = self.discriminator(clean_audios_slice)
                disc_fake_outputs, disc_fake_features = self.discriminator(decoded_slice.detach())
                
                # Discriminator loss using functions from discriminator.py
                d_loss, d_real_losses, d_fake_losses, d_real_accs, d_fake_accs = discriminator_loss(disc_real_outputs, disc_fake_outputs, self.apa_prob)
            
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 
                                            max_norm=self.config.training.max_grad_norm)
                self.optimizer_d.step()
                self.lr_scheduler_d.step()

                if self.config.training.apply_apa:
                    if self.apa_prob is not None:
                        for i, d_fake_acc in enumerate(d_fake_accs):
                            self.apa_prob[i] = self.apa_prob[i] + 0.001 * (1 if d_fake_acc > 0.7 else -2)
                            self.apa_prob[i] = torch.clamp(self.apa_prob[i], 0.0, 1.0)
               

            else:
                self.optimizer_g.zero_grad()
                if self.config.training.loss_type == "cossim":
                    w2v_feature = w2v_feature.view(-1, w2v_feature.size(-1))
                    w2v_recon = w2v_recon.view(-1, w2v_recon.size(-1))
                    w2v_loss = self.loss(w2v_feature, w2v_recon, torch.ones(w2v_feature.size(0), device=self.rank))
                elif self.config.training.loss_type == "l2" or self.config.training.loss_type == "l1":
                    w2v_loss = self.loss(w2v_feature, w2v_recon)
                else:
                    raise ValueError(f"Invalid loss type: {self.config.training.loss_type}")
                if self.model_type == "mimi":
                    g_loss = self.config.loss.recon_loss_weight * recon_loss + \
                            self.config.loss.vq_loss_weight * total_vq_loss + \
                            self.config.loss.commit_loss_weight * total_commit_loss + \
                            self.config.loss.w2v_loss_weight * w2v_loss + \
                            self.config.loss.semantic_loss_weight * semantic_loss
                else:
                    g_loss = self.config.loss.recon_loss_weight * recon_loss + \
                            self.config.loss.vq_loss_weight * total_vq_loss + \
                            self.config.loss.commit_loss_weight * total_commit_loss + \
                            self.config.loss.w2v_loss_weight * w2v_loss
                
                g_loss.backward()
                
                # Clip gradients by norm
                with torch.no_grad():
                    grad_norms = [(name, torch.norm(p.grad)) for name, p in self.model.named_parameters() if p.grad is not None]
                    grad_norm = torch.norm(torch.stack([norm for _, norm in grad_norms]))
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.training.max_grad_norm)
                
                self.optimizer_g.step()
                self.lr_scheduler_g.step()

            if self.rank == 0:
                if self.global_step % 20 == 0:
                    current = time.time()

                    usage = utils.count_to_usage(counts_ema)
                    if self.global_step > self.config.training.discriminator_start_steps:
                        logging.info(f"global_step: {self.global_step}, epoch: {self.epoch}, "
                                f"G_Loss: {g_loss:.4f}, D_Loss: {d_loss:.4f}, "
                                f"recon: {recon_loss:.4f}, "
                                f"w2v: {w2v_loss:.4f}, "
                                f"adv: {g_adv_loss:.4f}, fm: {fm_total_loss:.4f}, "
                                f"vq: {total_vq_loss:.4f}, commit: {total_commit_loss:.4f}, "
                                f"LR_G: {self.optimizer_g.param_groups[0]['lr']:.6f}, "
                                f"LR_D: {self.optimizer_d.param_groups[0]['lr']:.6f}, "
                                f"Grad_norm: {grad_norm:.4f}, "
                                f"Time: {(current - self.current_time) / 20:.4f} s/it")
                        if self.model_type == "mimi":
                            logging.info(f"semantic: {semantic_loss:.4f}")
                        for name, d_real_acc, d_fake_acc, fm_loss in zip(self.discriminator.module.names, d_real_accs, d_fake_accs, fm_losses):
                            logging.info(f"{name}: real_acc: {d_real_acc:.4f}, fake_acc: {d_fake_acc:.4f}, fm: {fm_loss:.4f}")
                        if self.config.training.apply_apa:
                            for name, apa_prob in zip(self.discriminator.module.names, self.apa_prob):
                                logging.info(f"{name}: apa_prob: {apa_prob:.4f}")
                    else:
                        logging.info(f"global_step: {self.global_step}, epoch: {self.epoch}, "
                                    f"G_Loss: {g_loss:.4f}, "
                                    f"recon: {recon_loss:.4f}, "
                                    f"w2v: {w2v_loss:.4f}, "
                                    f"vq: {total_vq_loss:.4f}, commit: {total_commit_loss:.4f}, "
                                    f"LR_G: {self.optimizer_g.param_groups[0]['lr']:.6f}, "
                                    f"Grad_norm: {grad_norm:.4f}, "
                                    f"Time: {(current - self.current_time) / 20:.4f} s/it")
                        if self.model_type == "mimi":
                            logging.info(f"semantic: {semantic_loss:.4f}")
                    logging.info(f"Usage: {usage}")
                    self.current_time = current
                
                if self.global_step % self.config.logging.save_interval == 0:
                    # Save generator checkpoint
                    checkpoint_path = os.path.join(self.config.logging.checkpoint_dir, f"checkpoint_{self.global_step}.pt")
                    utils.save_checkpoint(self.model.module, self.optimizer_g, self.lr_scheduler_g, 
                                        self.epoch, self.global_step, checkpoint_path)
                    
                    # Save discriminator checkpoint
                    if self.global_step > self.config.training.discriminator_start_steps:
                        disc_checkpoint_path = os.path.join(self.config.logging.checkpoint_dir, f"discriminator_{self.global_step}.pt")
                        utils.save_checkpoint(self.discriminator.module, self.optimizer_d, self.lr_scheduler_d,
                                            self.epoch, self.global_step, disc_checkpoint_path)
                
                # Log to TensorBoard
                if self.global_step % self.config.logging.log_interval == 0:
                    with torch.no_grad():
                        usage = utils.count_to_usage(counts_ema)
                        self.writer.add_scalar('G_Loss/generator_total', g_loss, self.global_step)
                        self.writer.add_scalar('LR/generator', self.optimizer_g.param_groups[0]['lr'], self.global_step)
                        self.writer.add_scalar('Grad_norm', grad_norm, self.global_step)
                        self.writer.add_scalar('G_Loss/reconstruction', recon_loss, self.global_step)
                        self.writer.add_scalar('G_Loss/w2v', w2v_loss, self.global_step)
                        self.writer.add_scalar('G_Loss/vq', total_vq_loss, self.global_step)
                        self.writer.add_scalar('G_Loss/commit', total_commit_loss, self.global_step)
                        if self.model_type == "mimi":
                            self.writer.add_scalar('G_Loss/semantic', semantic_loss, self.global_step)

                        for d in range(self.config.model.rvq.num_codebooks):
                            self.writer.add_scalar(f'Usage/codebook_{d}', usage[d], self.global_step)
                        if self.global_step > self.config.training.discriminator_start_steps:
                            self.writer.add_scalar('D_Loss/discriminator_total', d_loss, self.global_step)
                            self.writer.add_scalar('G_Loss/adversarial', g_adv_loss, self.global_step)
                            self.writer.add_scalar('G_Loss/feature_matching', fm_total_loss, self.global_step)
                            self.writer.add_scalar('LR/discriminator', self.optimizer_d.param_groups[0]['lr'], self.global_step)
                            for name, d_real_loss, d_fake_loss, d_real_acc, d_fake_acc, fm_loss in zip(self.discriminator.module.names, d_real_losses, d_fake_losses, d_real_accs, d_fake_accs, fm_losses):
                                self.writer.add_scalar(f'D_Loss_{name}/discriminator_{name}_real', d_real_loss, self.global_step)
                                self.writer.add_scalar(f'D_Loss_{name}/discriminator_{name}_fake', d_fake_loss, self.global_step)
                                self.writer.add_scalar(f'D_Loss_{name}/discriminator_{name}_real_acc', d_real_acc, self.global_step)
                                self.writer.add_scalar(f'D_Loss_{name}/discriminator_{name}_fake_acc', d_fake_acc, self.global_step)
                                self.writer.add_scalar(f'D_Loss_{name}/discriminator_{name}_fm', fm_loss, self.global_step)

                            if self.config.training.apply_apa:
                                for name, apa_prob in zip(self.discriminator.module.names, self.apa_prob):
                                    self.writer.add_scalar(f'D_Loss_{name}/discriminator_{name}_apa_prob', apa_prob, self.global_step)

                        # Log audio samples
                        for i in range(min(self.config.logging.n_samples, clean_audios.shape[0])):
                            self.writer.add_audio(f'train/Original_{i}', clean_audios[i].detach().cpu().numpy(), 
                                                self.global_step, sample_rate=self.config.data.sample_rate)
                            self.writer.add_audio(f'train/Reconstructed_{i}', decoded[i].detach().cpu().numpy(), 
                                                self.global_step, sample_rate=self.config.data.sample_rate)
                # if self.global_step % self.config.training.codebook_reset_interval == 0:
                #     #### codebook reset, not implemented for multiple GPUs ###
                #     logging.info(f"Resetting codebooks at global_step: {self.global_step}")
                #     with torch.no_grad():
                #         unused_codebooks = utils.reset_unused_codebooks(counts_ema)
                #         for d in range(self.config.model.rvq.num_codebooks):
                #             reset_codes = unused_codebooks[d] # [C]
                #             used_codebook_indices = torch.arange(self.config.model.rvq.num_codebooks, device=self.rank, dtype=torch.long)[torch.logical_not(reset_codes)]
                #             unused_codebook_indices = torch.arange(self.config.model.rvq.num_codebooks, device=self.rank, dtype=torch.long)[reset_codes]
                #             len_used_codebooks = len(used_codebook_indices)
                #             if self.config.rvq.codebook_size == len_used_codebooks:
                #                 continue
                #             else:
                #                 allocate_codes = used_codebook_indices[torch.randint(0, len_used_codebooks, (self.config.model.rvq.codebook_size - len_used_codebooks,), device=self.rank, dtype=torch.long)]
                #                 allocate_latents = self.rectifier.module.rvq.vqs[d](allocate_codes).detach().clone() 
                #                 allocate_latents = allocate_latents + torch.randn_like(allocate_latents) * (2**(-0.5 * (d+2)))
                #                 self.rectifier.module.rvq.vqs[d].codebook.weight.data[unused_codebook_indices].copy_(allocate_latents)
                #             logging.info(f"Codebooks reset at global_step: {self.global_step}, codebook_indices: {d}, len_unused_codebooks: {len(unused_codebook_indices)}")


            if self.global_step % self.config.logging.eval_interval == 0:
                logging.info(f"Evaluating at global_step: {self.global_step}")
                self.evaluate(epoch)
                self.model.train()
                self.discriminator.train()

            self.global_step += 1
                
        self.epoch += 1
        return self.epoch, self.global_step

    @torch.no_grad()
    def evaluate(self, epoch):
        self.model.eval()
        self.discriminator.eval()
        
        total_g_loss = 0
        total_d_loss = 0
        total_recon_loss = 0
        total_adv_loss = 0
        total_fm_total_loss = 0
        total_d_real_losses = None
        total_d_fake_losses = None
        total_d_real_accs = None
        total_d_fake_accs = None
        total_fm_losses = None
        total_vq_loss = 0
        total_commit_loss = 0
        total_w2v_loss = 0
        total_semantic_loss = 0
        for batch_idx, batch in enumerate(self.validation_dataloader):
            # Unpack batch
            clean_audios = batch

            clean_audios = clean_audios.to(self.rank, non_blocking=True)
            #clean_audios = F.pad(clean_audios, (self.config.model.mlp_in.in_features * 3, 0 ))
            
            w2v_feature = self.w2v(clean_audios)
            # Forward pass
            if self.model_type == "mimi":
                decoded, indices, semantic_loss, total_vq_loss, total_commit_loss = self.model.module(clean_audios, w2v_feature)
                total_semantic_loss += semantic_loss
            elif self.model_type == "semantic":
                decoded, indices, total_vq_loss, total_commit_loss = self.model.module(w2v_feature)
            else:
                decoded, indices, total_vq_loss, total_commit_loss = self.model.module(clean_audios)
            
            
            decoded_slice, clean_audios_slice = slice_audios(decoded, clean_audios, self.segment_length)
            # Calculate reconstruction loss
            recon_loss = self.mel_loss(decoded, clean_audios)
            w2v_recon = self.w2v(decoded)
            if self.config.training.loss_type == "cossim":
                w2v_feature = w2v_feature.view(-1, w2v_feature.size(-1))
                w2v_recon = w2v_recon.view(-1, w2v_recon.size(-1))
                w2v_loss = self.loss(w2v_feature, w2v_recon, torch.ones(w2v_feature.size(0), device=self.rank))
            elif self.config.training.loss_type == "l2" or self.config.training.loss_type == "l1":
                w2v_loss = self.loss(w2v_feature, w2v_recon)
            else:
                raise ValueError(f"Invalid loss type: {self.config.training.loss_type}")
 
            # Discriminator outputs
            disc_real_outputs, disc_real_features = self.discriminator.module(clean_audios_slice)
            disc_fake_outputs, disc_fake_features = self.discriminator.module(decoded_slice)
            
            # Losses using functions from discriminator.py
            d_loss, d_real_losses, d_fake_losses, d_real_accs, d_fake_accs = discriminator_loss(disc_real_outputs, disc_fake_outputs)
            
            g_adv_loss = generator_loss(disc_fake_outputs)
            
            fm_total_loss, fm_losses = feature_matching_loss(disc_real_features, disc_fake_features)
            
            # Total generator loss
            g_loss = self.config.loss.recon_loss_weight * recon_loss + \
                    self.config.loss.adv_loss_weight * g_adv_loss + \
                    self.config.loss.fm_loss_weight * fm_total_loss + \
                    self.config.loss.vq_loss_weight * total_vq_loss + \
                    self.config.loss.commit_loss_weight * total_commit_loss + \
                    self.config.loss.w2v_loss_weight * w2v_loss
            
            total_g_loss += g_loss
            total_d_loss += d_loss
            total_recon_loss += recon_loss
            total_adv_loss += g_adv_loss
            total_fm_total_loss += fm_total_loss
            total_vq_loss += total_vq_loss
            total_commit_loss += total_commit_loss
            total_w2v_loss += w2v_loss
            if total_d_real_losses is None:
                total_d_real_losses = d_real_losses
                total_d_fake_losses = d_fake_losses
                total_d_real_accs = d_real_accs
                total_d_fake_accs = d_fake_accs
                total_fm_losses = fm_losses
            else:
                for i, (d_real_loss, d_fake_loss, d_real_acc, d_fake_acc, fm_loss) in enumerate(zip(d_real_losses, d_fake_losses, d_real_accs, d_fake_accs, fm_losses)):
                    total_d_real_losses[i] += d_real_loss
                    total_d_fake_losses[i] += d_fake_loss
                    total_d_real_accs[i] += d_real_acc
                    total_d_fake_accs[i] += d_fake_acc
                    total_fm_losses[i] += fm_loss

            if self.rank == 0 and batch_idx == 0:
                for n in range(min(3, clean_audios.shape[0])):
                    self.writer.add_audio(f'Eval/original_{n}', clean_audios[n].cpu().numpy(), 
                                        self.global_step, sample_rate=self.config.data.sample_rate)
                    if n==0:
                        num_codebooks_i = torch.tensor([self.config.model.rvq.num_codebooks], device=self.rank, dtype=torch.long)
                    elif n==1:
                        num_codebooks_i = torch.tensor([self.config.model.rvq.num_codebooks-1], device=self.rank, dtype=torch.long)
                    else:
                        num_codebooks_i = torch.tensor([self.config.model.rvq.num_codebooks-2], device=self.rank, dtype=torch.long)
                    decoded_i = decoded[n:n+1]
                    self.writer.add_audio(f'Eval/reconstructed_{n}', decoded_i.cpu().numpy(), 
                                        self.global_step, sample_rate=self.config.data.sample_rate)

        avg_g_loss = total_g_loss / (batch_idx + 1)
        avg_d_loss = total_d_loss / (batch_idx + 1)
        avg_recon_loss = total_recon_loss / (batch_idx + 1)
        avg_adv_loss = total_adv_loss / (batch_idx + 1)
        avg_fm_loss = total_fm_total_loss / (batch_idx + 1)
        avg_vq_loss = total_vq_loss / (batch_idx + 1)
        avg_commit_loss = total_commit_loss / (batch_idx + 1)
        avg_w2v_loss = total_w2v_loss / (batch_idx + 1)
        avg_semantic_loss = total_semantic_loss / (batch_idx + 1)
        for i, (d_real_loss, d_fake_loss, d_real_acc, d_fake_acc, fm_loss) in enumerate(zip(total_d_real_losses, total_d_fake_losses, total_d_real_accs, total_d_fake_accs, total_fm_losses)):
            avg_d_real_loss = d_real_loss / (batch_idx + 1)
            avg_d_fake_loss = d_fake_loss / (batch_idx + 1)
            avg_d_real_acc = d_real_acc / (batch_idx + 1)
            avg_d_fake_acc = d_fake_acc / (batch_idx + 1)
            avg_fm_loss = fm_loss / (batch_idx + 1)
        logging.info(f"Evaluation - G_Loss: {avg_g_loss:.4f}, D_Loss: {avg_d_loss:.4f}, "
                    f"Recon Loss: {avg_recon_loss:.4f}, "
                    f"W2V Loss: {avg_w2v_loss:.4f}, "
                    f"Adv: {avg_adv_loss:.4f}, "
                    f"FM: {avg_fm_loss:.4f}, "
                    f"VQ: {avg_vq_loss:.4f}, "
                    f"Commit: {avg_commit_loss:.4f}")
        if self.model_type == "mimi":
            logging.info(f"Evaluation - Semantic Loss: {avg_semantic_loss:.4f}")
        
        if self.rank == 0:
            # Log to TensorBoard
            self.writer.add_scalar('Loss_eval/generator_total', avg_g_loss,self.global_step)
            self.writer.add_scalar('Loss_eval/reconstruction', avg_recon_loss,self.global_step)
            self.writer.add_scalar('Loss_eval/w2v', avg_w2v_loss, self.global_step)
            self.writer.add_scalar('Loss_eval/adversarial', avg_adv_loss, self.global_step)
            self.writer.add_scalar('Loss_eval/feature_matching', avg_fm_loss, self.global_step)
            self.writer.add_scalar('Loss_eval/vq', avg_vq_loss, self.global_step)
            self.writer.add_scalar('Loss_eval/commit', avg_commit_loss, self.global_step)
            if self.model_type == "mimi":
                self.writer.add_scalar('Loss_eval/semantic', avg_semantic_loss, self.global_step)
        self.model.train()
        self.discriminator.train()
        return

if __name__ == "__main__":
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config_codec.json", help="Path to the configuration file")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint")
    parser.add_argument("--experiment_name", type=str, default=datetime.now().strftime('%y%m%d%H%M'))
    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = 'localhost'
    port = utils.find_free_port()
    port2 = utils.find_free_port()
    print(f"Using master port: {port}")
    os.environ['MASTER_PORT'] = str(port)
    
    # Load configuration file using OmegaConf
    config = OmegaConf.load(args.config)

    # Merge command-line arguments with config
    cli_config = OmegaConf.from_dotlist([f"{k}={v}" for k, v in vars(args).items() if v is not None])
    config = OmegaConf.merge(config, cli_config)
    config.logging.experiment_name = args.experiment_name
    config.logging.experiment_dir = config.logging.experiment_dir.format(experiment_name=args.experiment_name)
    config.logging.checkpoint_dir = config.logging.checkpoint_dir.format(experiment_name=args.experiment_name)
    config.logging.tensorboard_dir = config.logging.tensorboard_dir.format(experiment_name=args.experiment_name)
    
    if args.resume:
        config.training.resume = True
        # check if the experiment name exists
        checkpoint_path = config.logging.checkpoint_dir
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Experiment name {args.experiment_name} does not exist")
    else:
        config.training.resume = False
    
    print(f"Experiment name: {args.experiment_name}, dir: {config.logging.experiment_dir}")
    
    # Create checkpoint and tensorboard directories
    os.makedirs(config.logging.experiment_dir, exist_ok=True)
    os.makedirs(config.logging.checkpoint_dir, exist_ok=True)
    os.makedirs(config.logging.tensorboard_dir, exist_ok=True)

    world_size = torch.cuda.device_count()
    print(f"World size: {world_size}")
    
    torch.multiprocessing.spawn(run, args=(world_size, config), nprocs=world_size)
