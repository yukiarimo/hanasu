# Hanasu
Hanasu is a multilingual TTS model built in the style of VITS. It works with every language, and it is designed to be fast, easy-to-use, and sound 100% human.

## Table of Contents
- [Hanasu](#hanasu)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
  - [Inference and Training](#inference-and-training)
  - [Contribute](#contribute)
  - [License](#license)
  - [Contact](#contact)

## Installation
To put Hanasu on your system, execute the commands that follow:

```bash
git clone https://github.com/yukiarimo/hanasu.git
cd hanasu
pip install -e .
```

### Prerequisites
- Python 3.8 or higher
- PyTorch 1.9 or higher
- Over 1GB of GPU memory (for inference)
- At least 8GB of GPU memory (for training)
- Compatible GPUs: NVIDIA and Apple Silicon

## Inference and Training
To perform inference and training, utilize the scripts that have been given. Refer to the detailed instruction found in `notebooks/hanasu.ipynb` on how to train and infer Hanasu.

## Contribute
Contributing is welcome! For feature requests, bug reports, or other issues, please open an issue in the repo. If you would like to contribute code, please fork the repo and submit a pull request.

## License
Hanasu is distributed under the OSI-approved [GNU Affero General Public License v3.0 (AGPLv3)](https://www.gnu.org/licenses/agpl-3.0.en.html) license. Additionally, on Hugging Face, encoder Hanasu and Himitsu are licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International Public License (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/), meaning they can only be used for non-commercial purposes without modification and with proper attribution. However, using these models for TTS as encoders is subject to the terms of the AGPLv3 license!

> Note: The [Hanasu V1 Encoder](https://huggingface.co/yukiarimo/yuna-ai-hanasu-v1) is being deprecated in favor of the more straightforward and less labor-intensive VITS model.

## Contact
For questions or support, please open an issue in the repository or contact the author at yukiarimo@gmail.com.