# Official Implementation of JHCodec

*Reconstruct! Don't Encode: Self-Supervised Representation Reconstruction Loss for High-Intelligibility and Low-Latency Streaming Neural Audio Codec*

JHCodec is a pure Transformer decoder based neural audio codec with residual vector quantization. It shows state-of-the-art performance with minimal latency.

## Overview

This repository contains the *implementation for training* neural audio codecs with end-to-end training capabilities. The codec supports:
- Multiple VQ architectures (DAC, MIMI)
- End-to-end training with w2v-bert-2.0 semantic features
- SSRR and non-SSRR variants


## TODO
- [ ] Revise Readme
- [ ] Upload checkpoint
- [ ] Upload to HuggingFace
- [ ] Upload to PyPI (probably after the review)
- [ ] Make non-anonymous (after the review)

## Installation

```bash
pip install -e .
```

### Requirements

- Python >= 3.10 (required for using the `X | None` union type syntax in type hints; see [PEP 604](https://peps.python.org/pep-0604/)), or manually remove this syntax if using an older Python version
- PyTorch/TorchAudio with CUDA support: tested with `torch==2.6.0+cu124` and `torch==2.9.1+cu128`
- [omegaconf==2.3.0](https://omegaconf.readthedocs.io/en/2.3_branch/): for configuration management
- [Flash-Attention](https://github.com/Dao-AILab/flash-attention): Non-fa code shows degraded quality. We tested with `flash-attn==2.7.4.post1` and `flash-attn==2.8.3`. 
- [HF transformers](https://huggingface.co/docs/transformers/index): Required only for running baselines and [w2v-bert2.0](https://huggingface.co/facebook/w2v-bert-2.0). JHCodec inference has no dependency on it.

We have provided a [Shell Script](/installcu128.sh) to help set up the environment. 
PLEASE DO NOT RUN It Directly. INSTEAD, REVIEW THE SCRIPT AND MODIFY IT AS NEEDED FOR YOUR SYSTEM.

OUR MODEL REQUIRES ONLY THE MINIMUM DEPENDENCIES LISTED ABOVE.

#### For training
- [PhaseAug](https://github.com/maum-ai/phaseaug)
- [alias-free-torch](https://github.com/junjun3518/alias-free-torch)

To install both required libraries, run:
```
pip install omegaconf==2.3.0
pip install alias-free-torch==0.0.6 phaseaug
```

Flash-Attention should be installed carefully. Please read the official [README](https://github.com/Dao-AILab/flash-attention).

## Project Structure

```
codec_paper/
├── jhcodec/                      # Main package
│   ├── model/                    # Model implementations
│   │   ├── codec.py              # Main codec models (JHCodec, JHCodecMimi)
│   │   ├── sw2v.py               # streaming wav2vec encoder
│   │   ├── discriminator.py      # Discriminator for adversarial training
│   │   └── vq.py                 # Vector quantization modules
│   ├── kernel/                   # Triton custom kernels 
│   │   ├── rotary_kernel.py      # Rotary positional embedding's kernel, adopted from FlashAttn
|   |   └── vq_kernel.py          # Vector quantization kernel
│   ├── loss/                     # Custom loss functions
│   │   └── multiscalemelspec.py  # Implements MultiScaleMelSpectrogramLoss used for perceptual audio training
│   ├── train_codec_e2e_w2v.py    # End-to-end training script
│   ├── decode_eval.py            # Decoding and evaluation script
│   └── dataloader.py             # Data loading utilities
├── config/                       # Configuration files
│   ├── config_dac_norecon.json   # DAC without reconstruction
│   ├── config_dac_recon.json     # DAC with reconstruction
│   ├── config_mimi_norecon.json  # MIMI without reconstruction
│   └── config_mimi_recon.json    # MIMI with reconstruction
└── setup.py
```

## Training

**Training Command:**

Data Preparation Using the `main` of `jhcodec/dataloader.py`

The main block of `dataloader.py` demonstrates how to construct and inspect an AudioDataset:

```python
from jhcodec.dataloader import AudioDataset, collate_fn
from torch.utils.data import DataLoader

dataset = AudioDataset(
    audio_dir='./data',                  # Path to your data
    sample_rate=16000,
    segment_duration=10.24,
    training=True,
    init_dataset=False,                  # Use True to scan files initially (slow), or False to load from cache
    cache_dir='cache_dir/dataloader/v9', # location of the cache
    use_mel=False,                       # Set True to return also Mel features
)
```

**Notes:**
- Initial dataset caching may take a while; once done, restart with `init_dataset=False` for faster loading.
- Requires all dependencies (see top part of `jhcodec/dataloader.py`).
- You can add a custom dataset by modifying the dictionary at the top of `dataloader.py`.


### For DAC with reconstruction:
```bash
python jhcodec/train_codec_e2e_w2v.py \
    --experiment_name paper/dac_recon \
    --config config/config_dac_recon.json \
    --resume # if resume
```

### For MIMI with reconstruction:
```bash
python jhcodec/train_codec_e2e_w2v.py \
    --experiment_name paper/mimi_recon \
    --config config/config_mimi_recon.json \
    --resume
```

**Available Configurations:**
- `config_dac_norecon.json`  - DAC without reconstruction
- `config_dac_recon.json`    - DAC with reconstruction
- `config_mimi_norecon.json` - MIMI without reconstruction
- `config_mimi_recon.json`   - MIMI with reconstruction (main)

### Training Parameters

Key training parameters (configurable in JSON config files):
- `learning_rate`: 1e-4
- `batch_size`: 42
- `num_epochs`: 100
- `warmup_steps`: 1000
- `discriminator_start_steps`: 10000
- Loss weights for reconstruction, VQ, commit, feature matching, and adversarial losses

## Decoding

Example
```bash
python jhcodec/decode_eval.py \
    --config config/config_dac_norecon.json \
    --checkpoint /path/to/checkpoint_300000.pt \
    --name jhcodec_dac_norecon \
    --glob_pattern "/path/to/audio/*.wav" \
    --out_dir "out_dir" \ 
    --hierarchy 4
```

**Arguments:**
- `--config`: Path to configuration file
- `--checkpoint`: Path to model checkpoint
- `--name`: Model name for output directory
- `--glob_pattern`: Glob pattern for input audio files
- `--hierarchy`: Depth of quantization hierarchy (default: 4)
- `--out_dir`: Output directory

### Supported Datasets

The decoding script supports various audio datasets:
- **LibriSpeech**: `/data/LibriSpeech/test-other/*/*/*.flac`
- **TITW**: `/data/titw/titw_hard/test/*.wav`
- **MLS**: `/data/MLS/mls_*/test/audio/*/*/*.flac`

## Evaluation

The project includes comprehensive evaluation scripts for:

### Word Error Rate (WER)
- `evaluation/wer_librispeech.py` - WER on LibriSpeech
- `evaluation/wer_titw.py` - WER on TITW
- `evaluation/wer_mls.py` - WER on MLS

### Mean Opinion Score (MOS)
- `evaluation/mos_librispeech.py` - MOS on LibriSpeech
- `evaluation/mos_titw.py` - MOS on TITW

### Speaker Similarity
- `evaluation/speaker_similarity_librispeech.py` - Speaker similarity on LibriSpeech
- `evaluation/speaker_similarity_titw.py` - Speaker similarity on TITW

Run evaluations using the corresponding SLURM scripts in the `evaluation/` directory.

## Configuration

Configuration files are JSON-based and include:

- **Model Architecture**: Encoder/decoder layers, attention heads, embedding dimensions
- **Vector Quantization**: Codebook size, number of codebooks, embedding dimensions
- **Training**: Learning rate, batch size, loss weights, discriminator settings
- **Data**: Sample rate, segment duration, data directories
- **Logging**: Checkpoint intervals, tensorboard settings

Example configuration structure:
```json
{
    "model": {
        "encoder": {...},
        "decoder": {...},
        "rvq": {
            "type": "dac",
            "num_codebooks": 8,
            "codebook_size": 1024
        }
    },
    "training": {...},
    "loss": {...},
    "data": {...}
}
```


## Main Contact

Anonymous. 
Contact: [jhcodec843@gmail.com](mailto:jhcodec843@gmail.com)
Submitted to Interspeech 2026

## References

- [EnCodec](https://github.com/facebookresearch/encodec)
- [DAC](https://github.com/descriptinc/descript-audio-codec)
- [Mimi](https://github.com/kyutai-labs/moshi)
- [AudioCraft](https://github.com/facebookresearch/audiocraft)
- [HiFiGAN](https://github.com/jik876/hifi-gan)
- [PhaseAug](https://github.com/maum-ai/phaseaug)
- [AliasFreeTorch](https://github.com/junjun3518/alias-free-torch)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)


## License

MIT License
