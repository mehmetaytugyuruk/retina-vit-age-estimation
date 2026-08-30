# Retinal Fundus Age Prediction (Vision Transformers)

> [!NOTE]
> **Retinal Age Prediction research series · Study 02**
>
> [Series overview](https://github.com/mehmetaytugyuruk/retinal-age-prediction) · [Study 01: ResNet baselines](https://github.com/mehmetaytugyuruk/retina-resnet-age-estimation) · [Study 03: Color spaces](https://github.com/mehmetaytugyuruk/retina-color-spaces-age-prediction)

**Resources:** [Pretrained weights](https://huggingface.co/mehmetaytugyuruk/retina-vit-age-estimation) · [Paper-to-code mapping](docs/paper-to-code-mapping.md) · [Citation](#citation) · [Companion ResNet study](https://github.com/mehmetaytugyuruk/retina-resnet-age-estimation)

## Publication

> M. A. Yürük and A. Memiş, "Decoding Chronological Age from the Retinal Fundus Images: A Deep Learning-based Analysis with Vision Transformers," in *2026 34th Signal Processing and Communications Applications Conference (SIU)*, İstanbul, Türkiye, Jul. 2026, pp. 1–4. doi: [10.1109/SIU71813.2026.11636734](https://doi.org/10.1109/SIU71813.2026.11636734)

[![DOI](https://img.shields.io/badge/DOI-10.1109%2FSIU71813.2026.11636734-blue)](https://doi.org/10.1109/SIU71813.2026.11636734)
[![IEEE Xplore](https://img.shields.io/badge/IEEE%20Xplore-11636734-00629B)](https://ieeexplore.ieee.org/document/11636734)

## Overview

Predicts chronological age from color retinal fundus images using four Vision Transformer variants (ViT-B/16, ViT-B/32, ViT-L/16, ViT-L/32), and derives an age-category classification from the regression output. Directly compares against a ResNet baseline from a companion study on the same dataset.

Companion study: [ResNet baseline study](https://github.com/mehmetaytugyuruk/retina-resnet-age-estimation) (IISEC 2026).

## Results

| Model | MAE (years) | Accuracy | F-measure |
|---|---|---|---|
| ViT-B/16 | 4.99 | 0.8368 | 0.6974 |
| ViT-B/32 | 5.52 | 0.8241 | 0.6764 |
| **ViT-L/16** | **4.86** | **0.8497** | **0.7219** |
| ViT-L/32 | 5.59 | 0.8211 | 0.6684 |

ViT-L/16 is the best-performing model overall. The paper reports this alongside the companion study's ResNet results (best: ResNet-101, MAE 5.01 as cited in this paper's Table VI) as a point-estimate comparison.

## Dataset

[Retina Age Analysis Dataset](https://huggingface.co/datasets/ramankamran/retina-age-analysis) (Kamran, 2025), MIT-licensed, 9,857 fundus images, same patient-level split as the companion ResNet study (6,902 / 1,493 / 1,462). Only the Graham-filtered preprocessing variant is used in this paper.

This repo does not redistribute the images. To reproduce:
1. Download the dataset from Hugging Face.
2. Run the preprocessing scripts below.
3. Place the results under `ImageFolders/filtered_images/` so the paths in `csvFiles/*/*.csv` resolve correctly (or edit the CSVs to point elsewhere).

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Training and evaluation automatically select CUDA, Apple Metal (MPS), or CPU,
in that order.

## Preprocessing

```bash
python src/preprocessing/crop_and_resize.py <raw_images_dir> ImageFolders/non_filtered_images
python src/preprocessing/ben_graham_filter.py ImageFolders/non_filtered_images ImageFolders/filtered_images
```

## Training

```bash
python src/train.py --model vit_l16 \
    --train-csv csvFiles/train/trainFilteredImages.csv \
    --val-csv csvFiles/validation/validationFilteredImages.csv \
    --output-dir checkpoints
```

`--model` accepts `vit_b16`, `vit_b32`, `vit_l16`, `vit_l32`. Each model has a per-variant default learning rate and batch size (see `MODEL_CONFIG` in `src/train.py`) — override with `--lr`/`--batch-size` if needed. All variants use a 5-epoch head-only warmup by default (`--no-warmup` to disable).

## Evaluation

```bash
python src/evaluate.py --model vit_l16 \
    --checkpoint checkpoints/best_vit_l16.pth \
    --test-csv csvFiles/test/testFilteredImages.csv
```

## Pretrained checkpoints

Not distributed via this repository (checkpoint files range from ~1GB to ~3.7GB). All 4 checkpoints are hosted on Hugging Face Hub:

**[Hugging Face checkpoint repository](https://huggingface.co/mehmetaytugyuruk/retina-vit-age-estimation)**

```python
from huggingface_hub import hf_hub_download
ckpt_path = hf_hub_download("mehmetaytugyuruk/retina-vit-age-estimation", "vit-l16-filtered.pth")
```

See the model card for the full file list, per-model results, and a loading example.

## Repository structure

```
src/
├── train.py                       # training, all 4 ViT variants via --model
├── evaluate.py                    # evaluation + per-class/overall metric tables
└── preprocessing/
    ├── crop_and_resize.py         # fundus disk crop + pad + resize
    └── ben_graham_filter.py       # Ben Graham vessel-enhancement filter
csvFiles/                          # train/val/test splits (Graham-filtered only)
docs/
└── paper-to-code-mapping.md
```

## Citation

```bibtex
@inproceedings{yuruk2026decoding,
  title     = {Decoding Chronological Age from the Retinal Fundus Images: A Deep Learning-based Analysis with Vision Transformers},
  author    = {Yürük, Mehmet Aytuğ and Memiş, Abbas},
  booktitle = {2026 34th Signal Processing and Communications Applications Conference (SIU)},
  year      = {2026},
  pages     = {1--4},
  address   = {İstanbul, Türkiye},
  publisher = {IEEE},
  doi       = {10.1109/SIU71813.2026.11636734}
}
```

## License

Code released under the [MIT License](LICENSE). The dataset is separately licensed by its authors (MIT, see the [dataset card](https://huggingface.co/datasets/ramankamran/retina-age-analysis)).
