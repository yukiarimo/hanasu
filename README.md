# Hanasu
Hanasu is a human-like TTS model based on the multilingual Himitsu V1 transformer-based encoder and VITS architecture. Hanasu is a Japanese word that means "to speak." This project aims to build a TTS model that can speak multiple languages and mimic human-like prosody.

## Installation
To install Hanasu, you can follow the instructions below:

```bash
git clone https://github.com/yukiarimo/hanasu.git
cd hanasu
pip install -e .
```

You can download the pre-trained models and encoders from the HF: [Hanasu V1 Encoder](https://huggingface.co/yukiarimo/yuna-ai-hanasu-v1).

### Requirements
- Python 3.8+
- PyTorch 1.9+
- 1GB+ GPU memory (for inference)
- 8GB+ GPU memory (for training)
- Supported GPUs: NVIDIA and Apple Silicon

### Language Support
Languages supported by Hanasu TTS:

- English (EN)
- Japanese (JP)
- Russian (RU)
- Any language by IPA transliteration.

### Training and Inference
For training and inference, you can use the provided scripts. Please refer to the `notebooks/hanasu.ipynb` for a detailed guide on how to use Hanasu for training and inference.

## Contributing
Contributions are welcome! Please open an issue in the repository for feature requests, bug reports, or other issues. If you want to contribute code, please fork the repository and submit a pull request.

## License
Hanasu is distributed under the OSI-approved [GNU Affero General Public License v3.0 (AGPLv3)](https://www.gnu.org/licenses/agpl-3.0.en.html); see `LICENSE.md` for more information. Additionally, on Hugging Face, encoder Hanasu and Himitsu are licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International Public License (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/), meaning they can only be used for non-commercial purposes without modification and with proper attribution. However, using these models for TTS as encoders are subject to the terms of the AGPLv3 license!

> Note: The Hanasu Encoder was deprecated in favor of the direct VITS model, which is more efficient and easier to use. The encoder is still available for those who prefer it.