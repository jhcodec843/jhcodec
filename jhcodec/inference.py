import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import omegaconf
from jhcodec.model.codec import JHCodecDAC, JHCodecMimi
import jhcodec.utils as utils
import argparse

def main():
    parser = argparse.ArgumentParser(description="Decode eval for jhcodec (single file)")
    parser.add_argument("--config", default="config/config_mimi_recon.json", type=str, help="Path to config file")
    parser.add_argument("--checkpoint", default="jhcodec_mimi_1000000.pt", type=str, help="Path to checkpoint file")
    parser.add_argument("--input_file", type=str, required=True, help="Input WAV/flac file")
    parser.add_argument("--output_file", type=str, required=True, help="Output WAV/flac file")
    parser.add_argument("--num_codebooks", type=int, default=8, help="Number of codebooks to use for decoding")
    args = parser.parse_args()

    config = omegaconf.OmegaConf.load(args.config)
    if config.model.rvq.type == "mimi":
        codec = JHCodecMimi(config.model)
    elif config.model.rvq.type == "dac":
        codec = JHCodecDAC(config.model)
    else:
        raise ValueError(f"Invalid model type: {config.model.rvq.type}")
    assert args.num_codebooks <= config.model.rvq.num_codebooks, f"Number of codebooks to use for decoding ({args.num_codebooks}) must be less than or equal to the number of codebooks in the model ({config.model.rvq.num_codebooks})"

    checkpoint_path = args.checkpoint
    _, _, _, _, _ = utils.load_checkpoint(
        codec, None, None, checkpoint_path,
        strict_model=True)
    codec = codec.to("cuda")
    codec = codec.eval()

    print(f'Processing {args.input_file}')
    x, sr = torchaudio.load(args.input_file)
    if sr != config.data.sample_rate:
        x = torchaudio.transforms.Resample(sr, config.data.sample_rate)(x)
    x = x[0, :]
    x = x.view(1, -1)
    x = x.to("cuda")
    if x.shape[1] % 320 != 0:
        x = F.pad(x, (0, 320 - x.shape[1] % 320))
    indices, _ = codec.encode(x, torch.tensor([args.num_codebooks], device=x.device), inference_cache=None)
    decoded, _ = codec.decode(indices, torch.tensor([args.num_codebooks], device=x.device), inference_cache=None)

    torchaudio.save(args.output_file, decoded.detach().cpu(), config.data.sample_rate)
    print(f'Saved to {args.output_file}')

if __name__ == "__main__":
    main()