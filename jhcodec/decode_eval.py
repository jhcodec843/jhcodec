import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import omegaconf
from jhcodec.model.codec import JHCodecDAC, JHCodecMimi
import jhcodec.utils as utils
from glob import glob
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Decode eval for jhcodec")
    parser.add_argument("--config", default="config/config_mimi_recon.json", type=str, help="Path to config file")
    parser.add_argument("--checkpoint", default="/data/jhcodec/paper/mimi_recon/checkpoints/checkpoint_1000000.pt", type=str, help="Path to checkpoint file")
    parser.add_argument("--glob_pattern", type=str, default="/data/LibriSpeech/test-clean/*/*/*.flac",
                        help="Glob pattern for input WAV files")
    parser.add_argument("--out_dir", type=str, default=".",
                        help="Output directory")
    parser.add_argument("--hierarchy", type=int, default=4, help="How deep the hierarchy is")
    parser.add_argument("--name", type=str, default="jhcodec_recon", help="Name of the model")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    config = omegaconf.OmegaConf.load(args.config)
    print(config)
    if config.model.rvq.type == "mimi":
        codec = JHCodecMimi(config.model)
    elif config.model.rvq.type == "dac":
        codec = JHCodecDAC(config.model)
    else:
        raise ValueError(f"Invalid model type: {config.model.rvq.type}")

    checkpoint_path = args.checkpoint
    _, _, _, _, _ = utils.load_checkpoint(
        codec, None, None, checkpoint_path,
        strict_model=True)
    codec = codec.to("cuda")
    codec = codec.eval()


    target_files = glob(args.glob_pattern)
    for target_file in target_files:
        print(target_file)
        x, sr = torchaudio.load(target_file)
        if sr != config.data.sample_rate:
            x = torchaudio.transforms.Resample(sr, config.data.sample_rate)(x)
        x = x[0, :]
        x = x.view(1, -1)
        x = x.to("cuda")
        if x.shape[1] % 320 != 0:
            x = F.pad(x, (0, 320 - x.shape[1] % 320))
        indices, _ = codec.encode(x, torch.tensor([config.model.rvq.num_codebooks], device=x.device), inference_cache=None)
        for i in range(1, config.model.rvq.num_codebooks +1):
            out_dir = os.path.join(args.out_dir, f"{args.name}_{i}")
            decoded, _ = codec.decode(indices, torch.tensor([i], device=x.device), inference_cache=None)

            # keep the last two directory names and file name in the path
            path_parts = os.path.normpath(target_file).split(os.sep)
            out_file = os.path.join(out_dir, *path_parts[-args.hierarchy:])  # last three dirs + file
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            torchaudio.save(out_file, decoded.detach().cpu(), config.data.sample_rate)
            print(out_file)

if __name__ == "__main__":
    main()