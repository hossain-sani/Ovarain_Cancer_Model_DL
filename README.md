# Ovarian Cancer Detection and Classification using Attention-Based CNN

A reproduction of the pipeline described in:

> Md. Faruk Hosen, S. M. Hasan Mahmud, Francis Rudra D. Cruze, Kah Ong Michael Goh, Hosney Jahan, Watshara Shoombuatong. **"Ovarian cancer detection and classification using attention-based CNN and deep feature extraction techniques"**. *Scientific Reports* (2026). DOI: [10.1038/s41598-026-67832-z](https://doi.org/10.1038/s41598-026-67832-z)

## Overview

Binary classification of ovarian cancer (OC) vs. non-OC from histopathological images using a hybrid ML/DL pipeline:

1. **Preprocessing** — bilateral filtering + data augmentation (rotation, flip, affine, color jitter, gamma, sharpening), resized to 224×224 with ImageNet normalization.
2. **Feature extraction** — `AttResNet50`: a pre-trained ResNet50 augmented with a spatial attention module, fine-tuned via transfer learning to produce 2048 deep features per image.
3. **Feature selection** — SHAP (Shapley additive explanations) with an XGBoost model ranks the 2048 features; the top 500 are kept. Compare against PCA / LASSO / mRMR.
4. **Classification** — `AttCNN` (attention + FC layers) alongside the baseline classifiers DT, RF, XGBoost, and AdaBoost.

## Dataset

The **STRAMPN** dataset (Singh et al., 2024) is used — 987 histopathological images:

- `Ovarian_Cancer/` — 481 images
- `Ovarian_Non_Cancer/` — 506 images

Download it (requires credentials) from:

- **IEEE DataPort** (official): `STRAMPN-Histopathological Images for Ovarian Cancer Prediction`, DOI: [10.21227/w7w8-p960](https://doi.org/10.21227/w7w8-p960)
- **Kaggle** mirror: [shimantokumar/strampn-data-set-overian-cancer](https://www.kaggle.com/datasets/shimantokumar/strampn-data-set-overian-cancer)

```
kaggle datasets download -d shimantokumar/strampn-data-set-overian-cancer --unzip
```

Place the data so the two folders live under the dataset root:

```
Dataset/
├── Ovarian_Cancer/     481 images
└── Ovarian_Non_Cancer/ 506 images
```

## Installation

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

Verify PyTorch recognizes your device:

```bash
python -c "import torch; print('cuda', torch.cuda.is_available())"
```

## Usage

```bash
python main.py
```

Runs the full pipeline: preprocessing → AttResNet50 training → feature extraction → SHAP selection (500 features) → DT/RF/XGBoost/AdaBoost + AttCNN classification → metrics table.

### Command-line options

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `Dataset` | Path to dataset root |
| `--out` | `output` | Output directory for models & results |
| `--filter` | `bilateral` | Filter: `bilateral`, `clahe`, `histeq`, `gaussian` |
| `--fs` | `shap` | Feature selection: `shap`, `pca`, `lasso`, `mrmr` |
| `--n-selected` | `500` | Number of features to keep |
| `--fe-epochs` | `30` | AttResNet50 training epochs |
| `--fe-batch` | `16` | Feature extractor batch size |
| `--fe-lr` | `0.001` | Feature extractor learning rate |
| `--weight-decay` | `0.0005` | Adam weight decay |
| `--momentum` | `0.9` | Adam momentum (beta1) |
| `--cls-epochs` | `200` | AttCNN classifier epochs |
| `--cls-batch` | `32` | AttCNN batch size |
| `--seed` | `42` | Random seed |
| `--device` | `auto` | `cpu`, `cuda`, `mps`, or `auto` |
| `--no-augment` | — | Disable data augmentation |

### Example commands

```bash
# Compare feature selection techniques
python main.py --fs shap
python main.py --fs pca
python main.py --fs lasso
python main.py --fs mrmr

# Try different filter preprocessing
python main.py --filter clahe
python main.py --filter gaussian

# Fewer epochs for a quick smoke test
python main.py --fe-epochs 1 --cls-epochs 1
```

## Outputs

Everything is written to the `output/` directory:

- `attresnet50.pt` — trained feature extractor weights
- `attcnn.pt` — trained classifier weights
- `features.npz` — extracted 2048-d features for train/val/test
- `selected_features_idx.npy` — indices of the selected features
- `results.csv` / `results.json` — final metrics
- `output_config.json` — run configuration (written next to `output/`)

### Metrics reported

`ACC` (accuracy), `AUC` (area under ROC curve), `PRE` (precision), `SP` (specificity), `SN` (sensitivity / recall), `F1`, and `MCC` (Matthews correlation coefficient).

## Project structure

```
day2/
├── main.py                 # Entry point / CLI
├── requirements.txt
├── Dataset/                # STRAMPN data (not included)
└── src/
    ├── config.py           # Hyperparameters (Config dataclass)
    ├── data.py             # Loading, filtering, augmentation, splitting
    ├── models.py           # AttResNet50, AttCNN
    ├── pipeline.py         # Training + feature selection + classifier orchestration
    ├── eval.py             # Evaluation metrics
    └── __init__.py
```

## Data split

The paper reports a 70:15:15 holdout split. Note that the paper states 790/169/169, but 70% of 987 is 690; this code uses the exact ratio (690/148/149 stratified). 5-fold cross-validation is not implemented; the holdout split is stratified by class.

## Reference

Hosen, M. F., Mahmud, S. M. H., Cruze, F. R. D., Goh, K. O. M., Jahan, H., Shoombuatong, W. Ovarian cancer detection and classification using attention-based CNN and deep feature extraction techniques. *Scientific Reports* (2026). https://doi.org/10.1038/s41598-026-67832-z