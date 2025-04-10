# Hanasu
Hanasu is a powerful voice library that allows you to convert text to speech (TTS) and perform voice conversion. It is designed to be easy to use and flexible.

# Hanasu TTS
Hanasu is a human-like TTS model based on the multilingual Himitsu V1 transformer-based encoder and VITS architecture. Hanasu is a Japanese word that means "to speak." This project aims to build a TTS model that can speak multiple languages and mimic human-like prosody.

## Table of Content
- [Hanasu](#hanasu)
- [Hanasu TTS](#hanasu-tts)
  - [Table of Content](#table-of-content)
  - [Installation](#installation)
    - [Requirements](#requirements)
  - [Usage](#usage)
    - [Language Support](#language-support)
    - [CLI](#cli)
    - [Python API](#python-api)
  - [Training](#training)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)
- [Hanasu Converter](#hanasu-converter)
  - [Features](#features)
  - [Installation](#installation-1)
  - [Quick Start](#quick-start)
    - [Converting Voice with Pre-trained Models](#converting-voice-with-pre-trained-models)
  - [Data Preparation](#data-preparation)
    - [Audio Requirements](#audio-requirements)
    - [Preprocessing](#preprocessing)
  - [Training](#training-1)
    - [Training from Scratch](#training-from-scratch)
    - [Continuous Training](#continuous-training)
    - [Training Options](#training-options)
    - [Monitoring Training](#monitoring-training)
  - [Inference](#inference)
    - [Single File Conversion](#single-file-conversion)
    - [Directory Conversion](#directory-conversion)
    - [Extract Speaker Embedding](#extract-speaker-embedding)
    - [Batch Conversion](#batch-conversion)
    - [Inference Options](#inference-options)
  - [Advanced Usage](#advanced-usage)
    - [Multi-GPU Training](#multi-gpu-training)
    - [Custom Audio Processing](#custom-audio-processing)
    - [Fine-tuning for Specific Voices](#fine-tuning-for-specific-voices)

## Installation
To install Hanasu, you can follow the instructions below:

```bash
git clone https://github.com/yukiarimo/hanasu.git
cd hanasu
pip install -e .
python -m unidic download
```

You can download the pre-trained models and encoders from the HF: [Hanasu V1 Encoder](https://huggingface.co/yukiarimo/yuna-ai-hanasu-v1).

### Requirements
- Python 3.8+
- PyTorch 1.9+
- torchaudio 0.9+
- transformers 4.9+
- librosa 0.8+
- unidic 1.0+
- 4GB+ GPU memory (for inference)
- 8GB+ GPU memory (for training)
- Supported GPUs: NVIDIA and Apple Silicon

## Usage
Hanasu can be used in two ways: CLI and Python API. The CLI is more user-friendly, while the Python API is more flexible.

### Language Support
Languages supported by Hanasu TTS:

- English (EN)
- Spanish (ES)
- French (FR)
- Chinese (ZH)
- Japanese (JP)
- Korean (KR)
- Russian (RU)

Additional languages can be added by transliterating the text to IPA.

### CLI
You may use the Hanasu CLI to interact with Hanasu. The CLI may be invoked using either `hanasu` or `hanasu`. Here are some examples:

**Read English text:**
```bash
hanasu "Text to read" output.wav
```

**Specify a language:**
```bash
hanasu "Text to read" output.wav --speaker EN-US
```

**Specify a speaker:**
```bash
hanasu "Text to read" output.wav --speaker EN-US --speaker Yuna
```

**Specify a speed:**
```bash
hanasu "Text to read" output.wav --speaker EN-US --speaker Yuna --speed 1.5
```

**Load from a file:**
```bash
hanasu file.txt out.wav --file
```

**Fuse models:**
```bash
# Basic Model Fusion: To fuse two models with default equal weighting:
python -m hanasu.fuse \
    --model_dirs /path/to/model1 /path/to/model2 \
    --model_steps 8600 108000 \
    --output_dir /path/to/output \
    --fallback_to_non_zero
```

**Fuse models with custom weighting:**
```bash
# Custom Model Fusion: Advanced Component-Specific Fusion with Three Models:
python -m hanasu.fuse \
    --model_dirs /path/to/model1 /path/to/model2 /path/to/model3 \
    --model_steps 8600 108000 21600 \
    --output_dir /path/to/output \
    --encoder_ratios 0.7 0.1 0.2 \
    --decoder_ratios 0.4 0.3 0.3 \
    --flow_attention_ratios 0.3 0.5 0.2 \
    --flow_other_ratios 0.3 0.4 0.3 \
    --duration_ratios 0.4 0.1 0.5 \
    --other_ratios 0.4 0.2 0.4 \
    --fallback_to_non_zero
```

**Fusion and Audio Generation:**
```bash
# Generate Audio from Fused Model:
python -m hanasu.fuse \
    --model_dirs /path/to/model1 /path/to/model2 \
    --model_steps 8600 108000 \
    --output_dir /path/to/output \
    --config_path /path/to/config.json \
    --generate_audio \
    --text "This is a test of the fused model synthesis." \
    --speaker_id 0 \
    --device cuda \
    --sdp_ratio 0.2 \
    --noise_scale 0.6 \
    --noise_scale_w 0.8 \
    --speed 1.0
```

### Python API
You may also use the Hanasu Python API to interact with Hanasu. Here is an example:

```python
from hanasu.api import TTS

# Speed is adjustable
text = "In a quiet neighborhood just west of Embassy Row in Washington, there exists a medieval-style walled garden whose roses, it is said, spring from twelfth-century plants. The garden's Carderock gazebo, known as Shadow House, sits elegantly amid meandering pathways of stones dug from George Washington's private quarry."
model = TTS(language='EN', device='cpu', use_hf=False, config_path="config.json", ckpt_path="G_100.pth")
model.tts_to_file(text=text, speaker_id=0, output_path='en-default.wav', sdp_ratio=0.8, noise_scale=0, noise_scale_w=0.2, speed=1.0, quiet=True)

"""
def tts_to_file(
 text,          # The input text to convert to speech
 speaker_id,    # ID of the speaker voice to use
 output_path,   # Where to save the audio file (optional)
 sdp_ratio,     # Controls the "cleanness" of the voice (0.0-1.0)
 noise_scale,   # Controls the variation in voice (0.0-1.0)
 noise_scale_w, # Controls the variation in speaking pace (0.0-1.0)
 speed,         # Speaking speed multiplier (1.0 = normal speed)
 pbar,          # Custom progress bar (optional)
 format,        # Audio format to save as (optional)
 position,      # Progress bar position (optional)
 quiet,         # Suppress progress output if True
):

- Higher sdp_ratio: Cleaner but more robotic voice
- Higher noise_scale: More variation/expressiveness but potentially less stable
- Higher noise_scale_w: More varied pacing but could sound unnatural
"""
```

## Training
To train Hanasu or its encoder, follow the steps in the `notebooks` directory. The training process is easy to follow and can be done on a single GPU. The training data is not included in this repository, but you can use your own data or download a dataset from a TTS dataset repository.

> Note 1: If you want to fine-tune the TTS model instead of training from scratch, place the pre-trained models into the `hanasu/logs/Yuna` directory.

> Note 2: If you plan to fine-tune or change the encoder, you must retrain the TTS models from scratch.

> Note 3: The TTS model has not been released yet, but you can train it yourself using the provided encoder. It will be released in the future once we have donations to support the project.

## Contributing
Contributions are welcome! Please open an issue in the repository for feature requests, bug reports, or other issues. If you want to contribute code, please fork the repository and submit a pull request.

## License
Hanasu is distributed under the OSI-approved [GNU Affero General Public License v3.0 (AGPLv3)](https://www.gnu.org/licenses/agpl-3.0.en.html); see `LICENSE.md` for more information. Additionally, on Hugging Face, encoder Hanasu and Himitsu are licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International Public License (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/), meaning they can only be used for non-commercial purposes without modification and with proper attribution. However, using these models for TTS as encoders are subject to the terms of the AGPLv3 license!

## Contact
For questions or support, please open an issue in the repository or contact the author at yukiarimo@gmail.com.

# Hanasu Converter
Hanasu Converter is a powerful voice-to-voice conversion system that allows you to transform audio from one voice to another. It supports training on multiple speakers and converting between their voices, as well as training on one speaker and converting to another using reference audio.

## Features
- **Direct Voice-to-Voice Conversion**: Convert between voices without intermediate text representation
- **Multiple Speaker Support**: Train on multiple speakers and convert between any of them
- **Reference-Based Conversion**: Use reference audio to define target voice characteristics
- **High-Quality Audio**: Works with 48kHz stereo audio for high-quality, high-pitched voice recordings
- **Cross-Platform Support**: Automatically detects and uses CPU, CUDA, or MPS (Apple Silicon)
- **Comprehensive Tools**: Includes training, inference, and utility scripts for a complete workflow

## Installation
Hanasu Converter is designed to be easy to install and use. Follow the instructions below to set up your environment.

1. Clone the repository:

```bash
git clone https://github.com/yourusername/hanasu-converter.git
cd hanasu-converter
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

Requirements include:

- Python 3.8 or higher
- PyTorch 1.12 or higher
- CUDA (optional, for GPU acceleration)
- MPS-enabled PyTorch (optional, for Apple Silicon)

## Quick Start

### Converting Voice with Pre-trained Models

1. Extract the target speaker embedding from reference audio:

```bash
python inference.py --config configs/config.json --checkpoint checkpoint.pth \
    extract --reference reference_audio.wav --output target_embedding.pth
```

2. Convert a source audio file to the target voice:

```bash
python inference.py --config configs/config.json --checkpoint checkpoint.pth \
    convert --source source_audio.wav --target_embedding target_embedding.pth \
    --output output_audio.wav
```

## Data Preparation

Hanasu Converter requires audio data for training. The data should be organized as follows:

```
data/
├── speaker1/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
├── speaker2/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
└── ...
```

### Audio Requirements

- Format: WAV, MP3, FLAC, or OGG (will be converted to WAV during preprocessing)
- Sample Rate: 48kHz (recommended, will be resampled if different)
- Channels: Mono or Stereo (stereo recommended for high-quality conversion)
- Duration: 3-10 seconds per file is ideal (longer files will be segmented)

### Preprocessing

You can preprocess your dataset using the utility functions:

```bash
# Create a CSV file listing all audio files
python -c "from hanasu.utils import create_dataset_csv; create_dataset_csv('dataset', 'dataset.csv')"

# Preprocess the dataset (resample, normalize, etc.)
python -c "from hanasu.utils import preprocess_dataset; preprocess_dataset('dataset.csv', 'processed_data')"
```

## Training

### Training from Scratch

To train a model from scratch:

```bash
python train.py --config configs/config.json --data_path dataset.csv --output_dir outputs
```

### Continuous Training

To continue training from a checkpoint:

```bash
python train.py --config configs/config.json --data_path dataset.csv --output_dir outputs \
    --checkpoint_path outputs/checkpoints/model_latest.pt
```

### Training Options

- `--config`: Path to the configuration file
- `--data_path`: Path to the data directory or CSV file
- `--output_dir`: Directory to save output files
- `--checkpoint_dir`: Directory to save checkpoints
- `--log_dir`: Directory to save logs
- `--checkpoint_path`: Path to checkpoint for continuing training
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--learning_rate`: Learning rate for optimization
- `--seed`: Random seed for reproducibility

### Monitoring Training

Training progress can be monitored using TensorBoard:

```bash
tensorboard --logdir outputs/logs
```

## Inference

Hanasu Converter provides several inference modes for different use cases.

### Single File Conversion

Convert a single audio file:

```bash
python inference.py --config configs/config.json --checkpoint outputs/checkpoints/model_latest.pt \
    convert --source test.wav --target_embedding target_embedding.pth \
    --output output_audio.wav
```

### Directory Conversion

Convert all audio files in a directory:

```bash
python inference.py --config configs/config.json --checkpoint checkpoint.pth \
    convert_dir --source_dir source_dir --target_embedding target_embedding.pth \
    --output_dir output_dir
```

### Extract Speaker Embedding

Extract speaker embedding from reference audio:

```bash
python inference.py --config configs/config.json --checkpoint outputs/checkpoints/model_latest.pt \
    extract --reference emb.wav --output target_embedding.pth
```

### Batch Conversion

Perform batch conversion between multiple speakers:

```bash
python inference.py --config configs/config.json --checkpoint checkpoint.pth \
    batch --source_dir source_dir --reference_dir reference_dir \
    --output_dir output_dir
```

### Inference Options

- `--config`: Path to the configuration file
- `--checkpoint`: Path to the model checkpoint
- `--device`: Device to use (cuda, mps, cpu)
- `--tau`: Temperature parameter for conversion (default: 0.3)

## Advanced Usage

### Multi-GPU Training

To train with multiple GPUs:

```bash
python train.py --config configs/config.json --data_path data --output_dir output
```

The system will automatically detect available GPUs and use distributed training if multiple GPUs are available.

### Custom Audio Processing

You can customize audio processing parameters in the configuration file:

```json
"audio": {
  "stereo": true,
  "normalize_audio": true,
  "target_level": -27,
  "vad_threshold": -40,
  "min_silence_duration": 500,
  "keep_silence_duration": 200
}
```

### Fine-tuning for Specific Voices

For better results with specific voice types (e.g., high-pitched anime voices):

1. Prepare a dataset with similar voice characteristics
2. Adjust the mel spectrogram parameters in the configuration:

```json
"data": {
  "sampling_rate": 48000,
  "filter_length": 2048,
  "hop_length": 512,
  "win_length": 2048,
  "n_mel_channels": 128,
  "mel_fmin": 0.0,
  "mel_fmax": 24000.0
}
```

3. Train the model with a smaller learning rate:

```bash
python train.py --config configs/config.json --data_path data --output_dir output --learning_rate 1e-4
```