# One for All: Synthesis-Free Fingerprint Learning for Attribution of In-the-Wild Synthetic Images

[![AAAI 2026](https://img.shields.io/badge/AAAI-2026-blue)](https://aaai.org/conference/aaai/aaai-26/)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](link-to-paper)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of "One for All: Synthesis-Free Fingerprint Learning for Attribution of In-the-Wild Synthetic Images" accepted at AAAI 2026.

## 📰 News
- **[2025-01]** Paper accepted at AAAI 2026! 🎉
- **[2025-01]** Code and models coming soon!

## 📄 Paper Information

**Title:** One for All: Synthesis-Free Fingerprint Learning for Attribution of In-the-Wild Synthetic Images

**Authors:** Jianwei Fei¹, Yunshu Dai², Peipeng Yu³, Zhihua Xia³, Dasara Shullani¹, Daniele Baracchi¹, Alessandro Piva¹

**Contact:** fei_jianwei@163.com


## Project Structure

- `denoiser.py`: Implementation of the DnCNN architecture, denoiser utilities, and inference scripts.
- `train_contrastivev_base_real_noiser_fft_bank.py`: Main training script for contrastive feature extraction using a supervised contrastive loss, memory banks, and FFT-based input transformations.


## Usage

### Training
To start the training process for the feature extractor:
```
python train_contrastivev_base_real_noiser_fft_bank.py --train_dir /path/to/dataset --batch_size 16
```

## 🚀 Timeline & Plan

- [ ] **Code Release**
- [ ] **Model Checkpoints**
- [ ] **Usage Instructions**


## 📋 Citation

If you find this work useful in your research, please consider citing:
```bibtex
@inproceedings{fei2026oneforall,
  title={One for All: Synthesis-Free Fingerprint Learning for Attribution of In-the-Wild Synthetic Images},
  author={Fei, Jianwei and Dai, Yunshu and Yu, Peipeng and Xia, Zhihua and Shullani, Dasara and Baracchi, Daniele and Piva, Alessandro},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```
