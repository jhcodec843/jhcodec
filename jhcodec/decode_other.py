import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import omegaconf
from jhcodec.model.codec_other import DACWrapper, MimiWrapper, FocalWrapper, StableCodecWrapper, NeMoNanoWrapper
#
import jhcodec.utils as utils
from glob import glob
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Decode eval for jhcodec")
    parser.add_argument("--glob_pattern", type=str, default="/data/LibriSpeech/test-clean/*/*/*.flac",
                        help="Glob pattern for input WAV files")
    parser.add_argument("--out_dir", type=str, default=".",
                        help="Output directory")
    parser.add_argument("--name", type=str, default="dac", help="Name of the model")
    parser.add_argument("--model", type=str, default="dac", help="Model type: mimi or dac")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate, mimi: 24000, dac: 16000")
    parser.add_argument("--target_sample_rate", type=int, default=16000, help="Target sample rate")
    parser.add_argument("--hierarchy", type=int, default=4, help="How deep the hierarchy is")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.model == "mimi":
        codec = MimiWrapper()
    elif args.model == "dac":
        codec = DACWrapper()
    elif args.model == "focal":
        codec = FocalWrapper()
    elif args.model == "qwen3tts":
        args.sample_rate = 24000
        from jhcodec.model.codec_other import Qwen3TTSTokenizerWrapper
        codec = Qwen3TTSTokenizerWrapper()
    elif args.model == "stablecodec":
        codec = StableCodecWrapper()
    elif args.model == "nemonano":
        args.sample_rate = 22050
        codec = NeMoNanoWrapper()
    else:
        raise ValueError(f"Invalid model type: {args.model}")
    codec = codec.to("cuda")
    codec = codec.eval()
    resampler = torch.nn.ModuleDict()

    target_files = glob(args.glob_pattern)
    for target_file in target_files:
        print(target_file)
        x, sr = torchaudio.load(target_file)
        x = x[0, :]
        x = x.view(1, -1)
        x = x.to("cuda")
        if sr != args.sample_rate:
            if f"{sr}->{args.sample_rate}" not in resampler.keys():
                resampler_ = torchaudio.transforms.Resample(sr, args.sample_rate).to('cuda')
                resampler[f"{sr}->{args.sample_rate}"] = resampler_
            x = resampler[f"{sr}->{args.sample_rate}"](x)
        if args.model in ["mimi", "dac"]:
            indices = codec.encode(x)
            for i in range(1, 8 +1):
                out_dir = os.path.join(args.out_dir, f"{args.name}_{i}")
                decoded = codec.decode(indices, i)

                # keep the last two directory names and file name in the path
                path_parts = os.path.normpath(target_file).split(os.sep)
                out_file = os.path.join(out_dir, *path_parts[-args.hierarchy:])  # last three dirs + file
                os.makedirs(os.path.dirname(out_file), exist_ok=True)
                if args.target_sample_rate != args.sample_rate:
                    if f"{args.sample_rate}->{args.target_sample_rate}" not in resampler.keys():
                        resampler_ = torchaudio.transforms.Resample(args.sample_rate, args.target_sample_rate).to('cuda')
                        resampler[f"{args.sample_rate}->{args.target_sample_rate}"] = resampler_
                    decoded = resampler[f"{args.sample_rate}->{args.target_sample_rate}"](decoded)
                torchaudio.save(out_file, decoded.detach().cpu(), args.target_sample_rate)
                print(out_file)
            if args.model == "mimi":
                for i in [16,32]:
                    out_dir = os.path.join(args.out_dir, f"{args.name}_{i}")
                    decoded = codec.decode(indices, i)

                    # keep the last two directory names and file name in the path
                    path_parts = os.path.normpath(target_file).split(os.sep)
                    out_file = os.path.join(out_dir, *path_parts[-args.hierarchy:])  # last three dirs + file
                    os.makedirs(os.path.dirname(out_file), exist_ok=True)
                    if args.target_sample_rate != args.sample_rate:
                        if f"{args.sample_rate}->{args.target_sample_rate}" not in resampler.keys():
                            resampler_ = torchaudio.transforms.Resample(args.sample_rate, args.target_sample_rate).to('cuda')
                            resampler[f"{args.sample_rate}->{args.target_sample_rate}"] = resampler_
                        decoded = resampler[f"{args.sample_rate}->{args.target_sample_rate}"](decoded)
                    torchaudio.save(out_file, decoded.detach().cpu(), args.target_sample_rate)
                    print(out_file)
        elif args.model == "focal":
            indices = codec.encode(x)
            out_dir = os.path.join(args.out_dir, f"{args.name}")
            decoded = codec.decode(indices)
            # in and out of focal is different
            if args.target_sample_rate != 24000:
                if f"{24000}->{args.target_sample_rate}" not in resampler.keys():
                    resampler_ = torchaudio.transforms.Resample(24000, args.target_sample_rate).to('cuda')
                    resampler[f"{24000}->{args.target_sample_rate}"] = resampler_
                decoded = resampler[f"{24000}->{args.target_sample_rate}"](decoded)
            path_parts = os.path.normpath(target_file).split(os.sep)
            out_file = os.path.join(out_dir, *path_parts[-args.hierarchy:])  # last three dirs + file
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            torchaudio.save(out_file, decoded.detach().cpu(), args.target_sample_rate)
            print(out_file)
        elif args.model == "qwen3tts":
            tokens = codec.encode(x, sr=24000)
            out_dir = os.path.join(args.out_dir, f"{args.name}")
            decoded = torch.tensor(codec.decode(tokens)).to("cuda").view(1, -1)
            if args.target_sample_rate != 24000:
                if f"{24000}->{args.target_sample_rate}" not in resampler.keys():
                    resampler_ = torchaudio.transforms.Resample(24000, args.target_sample_rate).to('cuda')
                    resampler[f"{24000}->{args.target_sample_rate}"] = resampler_
                decoded = resampler[f"{24000}->{args.target_sample_rate}"](decoded)
            path_parts = os.path.normpath(target_file).split(os.sep)
            out_file = os.path.join(out_dir, *path_parts[-args.hierarchy:])  # last three dirs + file
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            torchaudio.save(out_file, decoded.detach().cpu(), args.target_sample_rate)
            print(out_file)
        elif args.model == "stablecodec":
            x = F.pad(x, (0, 320 - x.shape[-1] % 320))
            latents, tokens = codec.encode(x)
            out_dir = os.path.join(args.out_dir, f"{args.name}")
            decoded = codec.decode(tokens)
            if args.target_sample_rate != 16000:
                if f"{16000}->{args.target_sample_rate}" not in resampler.keys():
                    resampler_ = torchaudio.transforms.Resample(16000, args.target_sample_rate).to('cuda')
                    resampler[f"{16000}->{args.target_sample_rate}"] = resampler_
                decoded = resampler[f"{16000}->{args.target_sample_rate}"](decoded)
            decoded = decoded.view(1, -1)
            path_parts = os.path.normpath(target_file).split(os.sep)
            out_file = os.path.join(out_dir, *path_parts[-args.hierarchy:])  # last three dirs + file
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            torchaudio.save(out_file, decoded.detach().cpu(), args.target_sample_rate)
            print(out_file)
        elif args.model == "nemonano":
            tokens, tokens_len = codec.encode(x)
            out_dir = os.path.join(args.out_dir, f"{args.name}")
            decoded = codec.decode(tokens, tokens_len)
            if args.target_sample_rate != 22050:
                if f"{22050}->{args.target_sample_rate}" not in resampler.keys():
                    resampler_ = torchaudio.transforms.Resample(22050, args.target_sample_rate).to('cuda')
                    resampler[f"{22050}->{args.target_sample_rate}"] = resampler_
                decoded = resampler[f"{22050}->{args.target_sample_rate}"](decoded)
            decoded = decoded.view(1, -1)
            path_parts = os.path.normpath(target_file).split(os.sep)
            out_file = os.path.join(out_dir, *path_parts[-args.hierarchy:])  # last three dirs + file
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            torchaudio.save(out_file, decoded.detach().cpu(), args.target_sample_rate)
            print(out_file)


if __name__ == "__main__":
    main()