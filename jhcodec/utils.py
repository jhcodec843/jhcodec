import os
import torch
import matplotlib.pyplot as plt
import matplotlib
import random
matplotlib.use("Agg")
MATPLOTLIB_FLAG = True
import numpy as np
import torchaudio
import logging
import omegaconf

logging.basicConfig(level=logging.INFO)

def load_checkpoint(model, optimizer=None, warmup_scheduler=None, checkpoint_path=None, strict_model=False):
    logging.info(f"Trying to load checkpoint from {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        logging.info(f"No checkpoint found at {checkpoint_path}")
        return model, optimizer, warmup_scheduler, 0, 0
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if not strict_model:
        sd = model.state_dict()
        keys = list(checkpoint['model_state_dict'].keys())
        for name in keys:
            if name not in sd:
                logging.info(f"Skipping {name}")
                del checkpoint['model_state_dict'][name]
                continue
            if sd[name].shape != checkpoint['model_state_dict'][name].shape:
                logging.info(f"Skipping {name} because of shape mismatch")
                del checkpoint['model_state_dict'][name]
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict_model)
    if strict_model and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # optimizer_to(optimizer, device)
    #if True:
    #    #tmp
    #    epoch = 0#checkpoint['epoch']
    #    if 'global_step' in checkpoint:
    #        global_step = checkpoint['global_step']
    #    else:
    #        global_step = 3000
    #else:
    if strict_model and warmup_scheduler is not None and 'warmup_scheduler_state_dict' in checkpoint:
        warmup_scheduler.load_state_dict(checkpoint['warmup_scheduler_state_dict'])
        # scheduler_to(warmup_scheduler, device)
    epoch = checkpoint['epoch']
    global_step = checkpoint['global_step']

    logging.info(f"Loaded checkpoint from {checkpoint_path}")
    logging.info(f"Resuming from epoch {epoch}, global step {global_step}, strict_model: {strict_model}")

    # Delete the checkpoint from memory after loading
    del checkpoint

    return model, optimizer, warmup_scheduler, epoch, global_step

# def optimizer_to(optim, device=0):
#     for param in optim.state.values():
#         # Not sure there are any global tensors in the state dict
#         if isinstance(param, torch.Tensor):
#             param.data = param.data.to(device)
#             if param._grad is not None:
#                 param._grad.data = param._grad.data.to(device)
#         elif isinstance(param, dict):
#             for subparam in param.values():
#                 if isinstance(subparam, torch.Tensor):
#                     subparam.data = subparam.data.to(device)
#                     if subparam._grad is not None:
#                         subparam._grad.data = subparam._grad.data.to(device)

# def scheduler_to(scheduler, device=0):
#     for param in scheduler.state_dict().values():
#         if isinstance(param, torch.Tensor):
#             param.data = param.data.to(device)
#         elif isinstance(param, dict):
#             for subparam in param.values():
#                 if isinstance(subparam, torch.Tensor):
#                     subparam.data = subparam.data.to(device)
#         elif isinstance(param, list):
#             for subparam in param:
#                 if isinstance(subparam, torch.Tensor):
#                     subparam.data = subparam.data.to(device)


                

def save_checkpoint(model, optimizer, warmup_scheduler, epoch, global_step, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'warmup_scheduler_state_dict': warmup_scheduler.state_dict(),
    }, checkpoint_path)
    logging.info(f"Saved checkpoint to {checkpoint_path}")
    return


def load_audio(audio_path, sample_rate):
    y, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        y = torchaudio.transforms.Resample(sr, sample_rate)(y)
    return y

def plot_ids(prompt_ids, input_ids, output_ids, num_images=3, length=24, vmin=0, vmax=8191):
    images = []
    for i in range(num_images):
        random_idx = random.randint(0, input_ids.shape[1] - length)
        prompt_ids_np = prompt_ids[i, random_idx: random_idx + length, :].detach().cpu().numpy().transpose().astype(np.int32)
        input_ids_np = input_ids[i, random_idx: random_idx + length, :].detach().cpu().numpy().transpose().astype(np.int32)
        output_ids_np = output_ids[i, random_idx: random_idx + length, :].detach().cpu().numpy().transpose().astype(np.int32)
        fig, axs = plt.subplots(3, 1, figsize=(15, int((prompt_ids_np.shape[0] +1 + input_ids_np.shape[0]+1 + output_ids_np.shape[0]+1) * 15 /(length+1) )),
                                height_ratios=[prompt_ids_np.shape[0]+1, input_ids_np.shape[0]+1, output_ids_np.shape[0]+1])
        # using pcolormesh to plot the ids
        # show ids value text in pcolormesh

        axs[0].pcolormesh(prompt_ids_np, vmin=vmin, vmax=vmax)
        axs[0].set_title('Prompt IDs')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Codebook')
        axs[0].set_xticks(np.arange(prompt_ids_np.shape[1]))
        axs[0].set_yticks(np.arange(prompt_ids_np.shape[0]))
        axs[0].set_xticklabels(np.arange(prompt_ids_np.shape[1]))
        axs[0].set_yticklabels(np.arange(prompt_ids_np.shape[0]))

        # Annotate each cell with the value from prompt_ids_np
        for x in range(prompt_ids_np.shape[1]):
            for y in range(prompt_ids_np.shape[0]):
                axs[0].text(x + 0.5, y + 0.5, f'{prompt_ids_np[y, x]}', 
                            fontsize=8, ha='center', va='center', color='black')
        axs[0].set_aspect('equal')

        axs[1].pcolormesh(input_ids_np, vmin=vmin, vmax=vmax)
        axs[1].set_title('Input IDs')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Codebook')
        axs[1].set_xticks(np.arange(input_ids_np.shape[1]))
        axs[1].set_yticks(np.arange(input_ids_np.shape[0]))
        axs[1].set_xticklabels(np.arange(input_ids_np.shape[1]))
        axs[1].set_yticklabels(np.arange(input_ids_np.shape[0]))

        # Annotate each cell with the value from input_ids_np
        for x in range(input_ids_np.shape[1]):
            for y in range(input_ids_np.shape[0]):
                axs[1].text(x + 0.5, y + 0.5, f'{input_ids_np[y, x]}', 
                            fontsize=8, ha='center', va='center', color='black')
        axs[1].set_aspect('equal')

        axs[2].pcolormesh(output_ids_np, vmin=vmin, vmax=vmax)
        axs[2].set_title('Output IDs')
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Codebook')
        axs[2].set_xticks(np.arange(output_ids_np.shape[1]))
        axs[2].set_yticks(np.arange(output_ids_np.shape[0]))
        axs[2].set_xticklabels(np.arange(output_ids_np.shape[1]))
        axs[2].set_yticklabels(np.arange(output_ids_np.shape[0]))

        # Annotate each cell with the value from output_ids_np
        for x in range(output_ids_np.shape[1]):
            for y in range(output_ids_np.shape[0]):
                axs[2].text(x + 0.5, y + 0.5, f'{output_ids_np[y, x]}', 
                            fontsize=8, ha='center', va='center', color='black')
        axs[2].set_aspect('equal')
        plt.tight_layout()
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
        plt.close(fig)
    return images

def plot_pitch_prob(pitch_prob_input, pitch_prob_prompt, pitch_prob_shifted_roll_back, num_images=3):
    images = []
    # pitch_prob_input: (batch_size, length, num_index)
    # pitch_prob_prompt: (batch_size, length, num_index)
    # pitch_prob_shifted_roll_back: (batch_size, length, num_index)
    for i in range(num_images):
        # [batch_size, length, num_index] -> [num_index, length]
        pitch_prob_prompt_np = pitch_prob_prompt[i, :, :].detach().cpu().numpy().transpose().astype(np.float32)
        pitch_prob_input_np = pitch_prob_input[i, :, :].detach().cpu().numpy().transpose().astype(np.float32)
        pitch_prob_shifted_roll_back_np = pitch_prob_shifted_roll_back[i, :, :].detach().cpu().numpy().transpose().astype(np.float32)
        pitch_prob_argmax_prompt = np.argmax(pitch_prob_prompt_np, axis=0)
        pitch_prob_argmax_input = np.argmax(pitch_prob_input_np, axis=0)
        pitch_prob_argmax_shifted_roll_back = np.argmax(pitch_prob_shifted_roll_back_np, axis=0)
        fig, axs = plt.subplots(3, 1, figsize=(15, 15))
        im0 = axs[0].imshow(pitch_prob_prompt_np, aspect='auto')
        axs[0].set_title('Pitch Prob Prompt')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Pitch')
        axs[0].scatter(np.arange(pitch_prob_prompt_np.shape[1]), pitch_prob_argmax_prompt, color='blue', marker='x')
        axs[0].set_aspect('equal')
        plt.colorbar(im0, ax=axs[0])

        im1 = axs[1].imshow(pitch_prob_input_np, aspect='auto')
        axs[1].set_title('Pitch Prob Input')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Pitch')
        axs[1].scatter(np.arange(pitch_prob_input_np.shape[1]), pitch_prob_argmax_input, color='red', marker='x')
        axs[1].set_aspect('equal')
        plt.colorbar(im1, ax=axs[1])

        im2 = axs[2].imshow(pitch_prob_shifted_roll_back_np, aspect='auto')
        axs[2].set_title('Pitch Prob Shifted Roll Back')
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Pitch')
        axs[2].scatter(np.arange(pitch_prob_shifted_roll_back_np.shape[1]), pitch_prob_argmax_shifted_roll_back, color='green', marker='x')
        axs[2].scatter(np.arange(pitch_prob_input_np.shape[1]), pitch_prob_argmax_input, color='red', marker='x')
        axs[2].scatter(np.arange(pitch_prob_prompt_np.shape[1]), pitch_prob_argmax_prompt, color='blue', marker='x')
        axs[2].set_aspect('equal')
        plt.colorbar(im2, ax=axs[2])

        plt.tight_layout()
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
        plt.close(fig)
    return images

def plot_vq(encoding_indices, encoding_indices_input, num_images=3, length=24, vmin=0, vmax=255):
    images = []
    # encoding_indices: (batch_size, length)
    # encoding_indices_input: (batch_size, length)
    for i in range(num_images):
        random_idx = random.randint(0, encoding_indices.shape[1] - length)
        encoding_indices_np = encoding_indices[i, random_idx: random_idx + length].detach().unsqueeze(0).cpu().numpy().astype(np.int32)
        encoding_indices_input_np = encoding_indices_input[i, random_idx: random_idx + length].detach().unsqueeze(0).cpu().numpy().astype(np.int32)
        fig, axs = plt.subplots(2, 1, figsize=(15, 5))
        axs[0].pcolormesh(encoding_indices_np, vmin=vmin, vmax=vmax)
        axs[0].set_title('Encoding Indices Prompt')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Codebook')
        axs[0].set_xticks(np.arange(encoding_indices_np.shape[1]))
        axs[0].set_yticks(np.arange(encoding_indices_np.shape[0]))
        axs[0].set_xticklabels(np.arange(encoding_indices_np.shape[1]))
        axs[0].set_yticklabels(np.arange(encoding_indices_np.shape[0]))
        for x in range(encoding_indices_np.shape[1]):
            for y in range(encoding_indices_np.shape[0]):
                axs[0].text(x + 0.5, y + 0.5, f'{encoding_indices_np[y, x]}', 
                            fontsize=8, ha='center', va='center', color='black')
        axs[0].set_aspect('equal')

        axs[1].pcolormesh(encoding_indices_input_np, vmin=vmin, vmax=vmax)
        axs[1].set_title('Encoding Indices Input')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Codebook')
        axs[1].set_xticks(np.arange(encoding_indices_input_np.shape[1]))
        axs[1].set_yticks(np.arange(encoding_indices_input_np.shape[0]))
        axs[1].set_xticklabels(np.arange(encoding_indices_input_np.shape[1]))
        axs[1].set_yticklabels(np.arange(encoding_indices_input_np.shape[0]))
        for x in range(encoding_indices_input_np.shape[1]):
            for y in range(encoding_indices_input_np.shape[0]):
                axs[1].text(x + 0.5, y + 0.5, f'{encoding_indices_input_np[y, x]}', 
                            fontsize=8, ha='center', va='center', color='black')
        axs[1].set_aspect('equal')

        plt.tight_layout()
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
        plt.close(fig)
    return images

def plot_ling(w2v_feature_source, w2v_feature_prompt, w2vbert_source, w2vbert_prompt, 
              ling_source, ling_prompt, ling_cont, rvq_embed, ling_indices, 
              vmin=0, vmax=1024, num_images=3, length=24):
    """
    Plot linguistic features and embeddings for visualization.
    
    Args:
        w2v_feature_source: wav2vec features from source audio, [B, T, D]
        w2v_feature_prompt: wav2vec features from prompt audio, [B, T, D]
        w2vbert_source: wav2vecBERT features from source, [B, T, D]
        w2vbert_prompt: wav2vecBERT features from prompt, [B, T, D]
        ling_source: linguistic source features, [B, T, D]
        ling_prompt: linguistic prompt features
        ling_cont: continuous linguistic features
        rvq_embed: RVQ embeddings
        ling_indices: linguistic indices [B, T, R] R: number of codebooks
        vmin, vmax: value range for colormap
        num_images: number of samples to plot
        length: sequence length to plot
    
    Returns:
        image: numpy array of the plot
    """
    images = []
    for i in range(num_images):
        random_idx = random.randint(0, max(1, w2v_feature_source.shape[1] - length)) if w2v_feature_source is not None else 0
        
        # Prepare data arrays for plotting
        plot_data = []
        plot_titles = []
        
        # Add ling_indices if available (discrete values)
        if ling_indices is not None:
            ling_indices_np = ling_indices[i, random_idx:random_idx + length].detach().cpu().numpy().transpose().astype(np.int32)
            plot_data.append(ling_indices_np)
            plot_titles.append('Linguistic Indices')
        # Add continuous features (use all dimensions for visualization)
        if w2v_feature_source is not None:
            w2v_src = w2v_feature_source[i, random_idx:random_idx + length].detach().cpu().numpy().transpose()  # [D, T]
            plot_data.append(w2v_src)
            plot_titles.append('W2V Source Features')
        
        if w2v_feature_prompt is not None:
            w2v_prompt = w2v_feature_prompt[i, random_idx:random_idx + length].detach().cpu().numpy().transpose()  # [D, T]
            plot_data.append(w2v_prompt)
            plot_titles.append('W2V Prompt Features')
        
        if w2vbert_source is not None:
            w2vbert_src = w2vbert_source[i, random_idx:random_idx + length].detach().cpu().numpy().transpose()  # [D, T]
            plot_data.append(w2vbert_src)
            plot_titles.append('W2VBERT Source Features')
        
        if w2vbert_prompt is not None:
            w2vbert_pmt = w2vbert_prompt[i, random_idx:random_idx + length].detach().cpu().numpy().transpose()  # [D, T]
            plot_data.append(w2vbert_pmt)
            plot_titles.append('W2VBERT Prompt Features')
        
        if ling_source is not None:
            ling_src = ling_source[i, random_idx:random_idx + length].detach().cpu().numpy().transpose()  # [D, T]
            plot_data.append(ling_src)
            plot_titles.append('Linguistic Source')
        
        if ling_prompt is not None:
            ling_pmt = ling_prompt[i, random_idx:random_idx + length].detach().cpu().numpy().transpose()  # [D, T]
            plot_data.append(ling_pmt)
            plot_titles.append('Linguistic Prompt')
        
        if ling_cont is not None:
            ling_c = ling_cont[i, random_idx:random_idx + length].detach().cpu().numpy().transpose()  # [D, T]
            plot_data.append(ling_c)
            plot_titles.append('Linguistic Continuous')
        
        if rvq_embed is not None:
            rvq_emb = rvq_embed[i, random_idx:random_idx + length].detach().cpu().numpy().transpose()  # [D, T]
            plot_data.append(rvq_emb)
            plot_titles.append('RVQ Embeddings')
        if len(plot_data) == 0:
            continue
        
        # Calculate height ratios with scaling to limit pixel size
        max_height_per_subplot = 200  # Maximum pixels per subplot
        height_ratios = []
        for data in plot_data:
            # Scale height based on feature dimensions but cap it
            scaled_height = 100# min(max_height_per_subplot, max(50, data.shape[0] // 10))
            height_ratios.append(scaled_height)
        
        # Create subplots with controlled figure size
        total_height = sum(height_ratios) / 20  # Scale down total height
        fig, axs = plt.subplots(len(plot_data), 1, 
                               figsize=(15, min(20, total_height)),  # Cap total height at 20
                               height_ratios=height_ratios)
        
        if len(plot_data) == 1:
            axs = [axs]
        
        for idx, (data, title) in enumerate(zip(plot_data, plot_titles)):
            if 'Indices' in title:
                # Use discrete colormap for indices with specified vmin/vmax
                im = axs[idx].pcolormesh(data, vmin=vmin, vmax=vmax)
                # Add text annotations for indices
                for x in range(data.shape[1]):
                    for y in range(data.shape[0]):
                        axs[idx].text(x + 0.5, y + 0.5, f'{int(data[y, x])}', 
                                    fontsize=8, ha='center', va='center', color='black')
                # Add invisible placeholder colorbar to balance layout with other subplots
                cbar = plt.colorbar(im, ax=axs[idx])
                cbar.ax.set_visible(False)
                axs[idx].set_aspect('auto')

            else:
                # Use continuous colormap for features with controlled aspect ratio
                im = axs[idx].imshow(data, aspect="auto", origin="lower",
                  interpolation='none')
                plt.colorbar(im, ax=axs[idx])
                # Set aspect ratio to prevent excessive height

            
            axs[idx].set_title(title)
            axs[idx].set_xlabel('Time')
            axs[idx].set_ylabel('Feature Dim' if 'Indices' not in title else 'Index')
            
            # Reduce number of ticks for large feature dimensions
            # if data.shape[0] > 50:
            #     tick_step = max(1, data.shape[0] // 20)
            #     axs[idx].set_yticks(np.arange(0, data.shape[0], tick_step))
            #     axs[idx].set_yticklabels(np.arange(0, data.shape[0], tick_step))
            # else:
            #     axs[idx].set_yticks(np.arange(data.shape[0]))
            #     axs[idx].set_yticklabels(np.arange(data.shape[0]))
            
            axs[idx].set_xticks(np.arange(data.shape[1]))
            axs[idx].set_xticklabels(np.arange(data.shape[1]))
            plt.tight_layout()
        
        plt.tight_layout()
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
        plt.close(fig)
    return images

def plot_similarity(similarity):
    fig = plt.figure(figsize=(10, 10))
    similarity = similarity.detach().cpu().numpy()
    plt.imshow(similarity, aspect='auto')
    plt.title('Similarity')
    plt.xlabel('Sample')
    plt.ylabel('Sample')
    plt.gca().set_aspect('equal')
    plt.colorbar()
    plt.tight_layout()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image

@torch.no_grad()
def count(indices, codebook_size):
    # indices: [B,T,D]
    B,T,D = indices.shape
    counts = torch.zeros((D,codebook_size), device=indices.device, dtype=torch.int32)
    for d in range(D):
        count = torch.bincount(indices[:,:,d].view(-1), minlength=codebook_size)
        counts[d] = count
    return counts

@torch.no_grad()
def count_to_usage(counts):
    # counts: [D,codebook_size]
    D, _ = counts.shape
    usage = torch.zeros((D,), device=counts.device, dtype=torch.float32)
    usage = (counts>1e-6).float().mean(dim=-1)
    return usage

@torch.no_grad()
def reset_unused_codebooks(counts):
    # counts: [D,codebook_size]
    D, C = counts.shape
    unused_codebooks = (counts<1e-7) # set bar lower than count 
    return unused_codebooks



def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

def load_pretrained_jhcodec(repo_id='jhcodec/jhcodec'):
    from jhcodec.model.codec import JHCodecMimi
    from huggingface_hub import hf_hub_download
    # config: jhcodec/jhcodec/config.json
    # ckpt: jhcodec/jhcodec/jhcodec_mimi_1000000.pt
    # download config and ckpt from huggingface
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename="config.json",
    )
    ckpt_path = hf_hub_download(
        repo_id=repo_id,
        filename="jhcodec_mimi_1000000.pt",
    )
    config = omegaconf.OmegaConf.load(config_path)
    print(config)
    codec = JHCodecMimi(config.model, training=False)
    load_checkpoint(codec, None, None, ckpt_path, strict_model=True)
    return codec

def load_pretrained_sw2v(repo_id='jhcodec/sw2v_60k'):
    from jhcodec.model.sw2v import AudioEncoder
    from huggingface_hub import hf_hub_download
    assert repo_id in ['jhcodec/sw2v_60k', 'jhcodec/sw2v_120k'], f"Invalid repo_id: {repo_id}"

    # config: jhcodec/sw2v_60k/config.json
    # ckpt: jhcodec/sw2v_60k/sw2v_60k.pt
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename="config.json",
    )
    config = omegaconf.OmegaConf.load(config_path)
    try:
        ckpt_path = hf_hub_download(
            repo_id=repo_id,
            filename="sw2v_60000.pt",
        )
    except:
        ckpt_path = hf_hub_download(
            repo_id=repo_id,
            filename="sw2v_120000.pt",
        )
    w2v = AudioEncoder(config.w2v, training=False)
    load_checkpoint(w2v, None, None, ckpt_path, strict_model=True)
    return w2v