# An Interpretable Hybrid Deep Learning and Interval Type-2 Fuzzy Framework for Breast Microcalcification Classification

This repository contains the implementation accompanying the manuscript:

**An Interpretable Hybrid Deep Learning and Interval Type-2 Fuzzy Framework for Breast Microcalcification Classification in Digital Mammography**

The implementation combines multi-scale wavelet descriptors, EfficientNetV2 with CBAM, Vision Transformer representations, attention-based feature fusion, Interval Type-2 Fuzzy inference, and Fuzzy ART classification.

## Main components

- Daubechies-4 (`db4`) wavelet decomposition and texture descriptors
- PCA fitted only on the training partition
- EfficientNetV2 + CBAM local feature branch
- ViT-B/16 global-context branch
- Attention-based multimodal fusion
- Interval Type-2 Fuzzy inference with 12 linguistic rules
- Karnik-Mendel type reduction
- Fuzzy ART incremental classifier
- Patient-level cross-validation and external validation
- Bootstrap confidence intervals
- Sensitivity, ablation, statistical, explainability, and benchmark utilities

## Repository structure

```text
.
├── main.py
├── README.md
├── REVIEWER_GUIDE.md
├── requirements.txt
├── pyproject.toml
├── environment.yml
├── config/
│   ├── default.yaml
│   └── paper.yaml
├── data/
│   └── examples/
│       ├── manifest_example.csv
│       └── annotations_example.csv
├── scripts/
├── src/mchf/
│   ├── art/
│   ├── data/
│   ├── features/
│   ├── fuzzy/
│   ├── models/
│   └── utils/
└── tests/
```

## Installation

Python 3.10 or 3.11 is recommended.

### Option 1: pip

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies and the package:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps --no-build-isolation
```

Check the environment:

```bash
python scripts/check_environment.py
```

### Option 2: conda

```bash
conda env create -f environment.yml
conda activate mchf-paper
pip install -e . --no-deps --no-build-isolation
```

## Reviewer smoke test

A short end-to-end test is provided so that the installation can be verified without downloading DDSM or INbreast.

```bash
python main.py smoke-test --device cpu
```

The smoke test:

1. runs the unit tests;
2. creates synthetic feature data;
3. runs patient-level cross-validation;
4. trains a feature-level model bundle;
5. evaluates the saved bundle;
6. validates the Karnik-Mendel implementation.

A successful run creates:

```text
results/reviewer_smoke_test/SMOKE_TEST_PASS.json
```

The synthetic data are used only to verify execution. They are not experimental results.

## Data

The experiments use:

- DDSM / CBIS-DDSM
- INbreast

The original mammograms are not included in the repository. Dataset use is subject to the corresponding provider terms.

### Canonical ROI manifest

Feature extraction uses a CSV file with the following required columns:

```text
image_path,label,patient_id,dataset
```

Example:

```text
image_path,label,patient_id,dataset
/path/to/roi001.png,0,P001,DDSM
/path/to/roi002.png,1,P002,INbreast
```

For explicit holdout evaluation, add a split column:

```text
image_path,label,patient_id,dataset,split
/path/to/roi001.png,0,P001,DDSM,train
/path/to/roi002.png,1,P002,INbreast,holdout
```

See `data/examples/manifest_example.csv`.

### Canonical annotation file for ROI generation

`prepare-rois` accepts a dataset-independent annotation CSV containing:

```text
image_path,dataset,patient_id,label,annotation_type,x1,y1,x2,y2,cx,cy
```

`annotation_type` can be `bbox` or `point`.

See `data/examples/annotations_example.csv`.

## Main workflow

All principal experiments are exposed through `main.py`:

```bash
python main.py --help
```

### 1. Prepare ROIs

```bash
python main.py prepare-rois \
  --annotations data/annotations.csv \
  --output-dir data/rois \
  --manifest data/manifest.csv
```

### 2. Optional deep-branch fine-tuning

EfficientNetV2 + CBAM:

```bash
python main.py train-branch \
  --manifest data/manifest.csv \
  --branch effnet \
  --output checkpoints/effnet.pt \
  --device cuda
```

Vision Transformer:

```bash
python main.py train-branch \
  --manifest data/manifest.csv \
  --branch vit \
  --output checkpoints/vit.pt \
  --device cuda
```

### 3. Extract features

```bash
python main.py extract \
  --manifest data/manifest.csv \
  --output features/all_features.npz \
  --eff-checkpoint checkpoints/effnet.pt \
  --vit-checkpoint checkpoints/vit.pt \
  --batch-size 16 \
  --device cuda
```

If branch checkpoints are omitted, pretrained backbones are loaded through `timm`. The first run may require network access unless those weights are already cached.

For an offline structural test only:

```bash
python main.py extract \
  --manifest data/manifest.csv \
  --output features/all_features.npz \
  --no-pretrained \
  --device cpu
```

### 4. Five-fold patient-level cross-validation

DDSM:

```bash
python main.py cv \
  --features features/all_features.npz \
  --dataset DDSM \
  --output results/ddsm_cv.csv \
  --device cuda
```

INbreast:

```bash
python main.py cv \
  --features features/all_features.npz \
  --dataset INbreast \
  --output results/inbreast_cv.csv \
  --device cuda
```

Pooled evaluation:

```bash
python main.py cv \
  --features features/all_features.npz \
  --dataset pooled \
  --output results/pooled_cv.csv \
  --device cuda
```

Default settings are:

- outer folds: 5
- inner folds: 3
- PCA retained variance: 0.95
- epochs: 50
- bootstrap resamples: 1000
- seed: 12345

### 5. Train final model bundle

```bash
python main.py train \
  --features features/all_features.npz \
  --dataset pooled \
  --output-dir checkpoints/final \
  --device cuda
```

The bundle contains the neural model state and the fitted training-only preprocessing objects required for later evaluation.

### 6. Evaluate a saved model

```bash
python main.py test \
  --features features/all_features.npz \
  --bundle checkpoints/final \
  --output results/test_predictions.csv \
  --device cuda
```

### 7. Locked holdout evaluation

Using a manifest split:

```bash
python main.py holdout \
  --features features/all_features.npz \
  --bundle checkpoints/final \
  --manifest data/manifest.csv \
  --split holdout \
  --output results/holdout_predictions.csv \
  --device cuda
```

Using exact stored feature indices:

```bash
python main.py holdout \
  --features features/all_features.npz \
  --bundle checkpoints/final \
  --indices data/locked_holdout_indices.npy \
  --output results/holdout_predictions.csv \
  --device cuda
```

The holdout command performs evaluation only and does not retrain the model.

### 8. External validation

DDSM to INbreast:

```bash
python main.py external \
  --features features/all_features.npz \
  --train-dataset DDSM \
  --test-dataset INbreast \
  --output results/ddsm_to_inbreast.csv \
  --device cuda
```

Both directions:

```bash
python main.py cross-dataset \
  --features features/all_features.npz \
  --output-dir results/cross_dataset \
  --device cuda
```

### 9. Sensitivity analysis

```bash
python main.py sensitivity \
  --features features/all_features.npz \
  --output-dir results/sensitivity \
  --device cuda
```

### 10. Ablation analysis

```bash
python main.py ablation \
  --features features/all_features.npz \
  --output-dir results/ablation \
  --device cuda
```

The generated table is written to:

```text
results/ablation/table6_ablation.csv
```

### 11. Karnik-Mendel validation

```bash
python main.py validate-km \
  --samples 1000 \
  --output results/km_validation.json
```

### 12. Computational benchmark

```bash
python main.py benchmark \
  --image path/to/example_roi.png \
  --output results/benchmark.json \
  --device cuda
```

## Feature pipeline

The three feature branches are combined as follows:

```text
ROI
 ├─ db4 Wavelet -> raw handcrafted descriptors -> PCA
 ├─ EfficientNetV2 + CBAM -> 128-D representation
 └─ ViT-B/16 -> 256-D representation
             |
             v
       Feature concatenation
             |
             v
       Training-only MinMaxScaler
             |
             v
       Linear projection to 256-D
             |
             v
       4-head self-attention
             |
             v
       Residual + LayerNorm
             |
             v
       Interval Type-2 Fuzzy inference
             |
             v
       Fuzzy ART classification
```

The PCA dimension is determined from the training fold by the retained-variance setting. With the manuscript dataset/configuration, this corresponds to the reported compact wavelet representation.

## Leakage prevention

The implementation applies the following safeguards:

- grouping by `patient_id` during validation;
- PCA fitted on training samples only;
- normalization fitted on training samples only;
- fuzzy rule-variable normalization fitted on training samples only;
- decision threshold selected using validation data;
- external test datasets are not used to fit preprocessing objects;
- holdout evaluation uses a saved model bundle and does not retrain.

Correct patient identifiers are therefore essential.

## Interval Type-2 Fuzzy module

The fuzzy module includes:

- Gaussian interval membership functions;
- footprint-of-uncertainty control;
- 12 linguistic rules;
- lower and upper firing strengths;
- exact iterative Karnik-Mendel type reduction;
- differentiable midpoint approximation for training-related validation;
- fuzzy malignancy score.

The rule base is located in:

```text
src/mchf/fuzzy/rules.py
```

The inference implementation is located in:

```text
src/mchf/fuzzy/type2.py
```

Some numerical implementation choices that are not fixed explicitly in the manuscript are exposed in the configuration files rather than being hidden in the code. These include the numerical mapping from learned embeddings to linguistic fuzzy variables and the singleton values used for fuzzy consequents.

## Fuzzy ART

The Fuzzy ART implementation includes:

- complement coding;
- fuzzy AND operation;
- category choice function;
- vigilance criterion;
- category update;
- category creation;
- supervised category-to-class mapping;
- continuous decision score.

Default manuscript setting:

```text
rho = 0.85
beta = 0.50
```

Implementation:

```text
src/mchf/art/fuzzy_art.py
```

## Evaluation metrics

The evaluation utilities include:

- Accuracy
- Sensitivity
- Specificity
- Precision
- F1-score
- ROC-AUC
- NPV
- FPR
- FNR
- Brier score
- bootstrap confidence interval

## Explainability utilities

Additional scripts are included for:

- Grad-CAM++ examples
- SHAP feature importance
- fuzzy workflow visualization
- result plotting

These utilities are located in `scripts/`.

## Tests

Run all unit tests with:

```bash
pytest -q
```

Expected behavior is that all applicable tests pass. Hardware-dependent tests may be skipped when the required device or optional runtime is unavailable.

## Reproducibility files

For a manuscript release, preserve the following alongside the code when licensing permits:

- exact package versions;
- random seed;
- `config/paper.yaml`;
- de-identified patient-level split manifests;
- extracted feature archives;
- out-of-fold prediction CSV files;
- final evaluation summaries;
- model checkpoints.

Package versions can be recorded with:

```bash
pip freeze > environment_frozen.txt
```

Raw mammograms should not be redistributed unless the dataset license permits it.

## Citation

If this implementation is used in academic work, please cite the associated manuscript:

**An Interpretable Hybrid Deep Learning and Interval Type-2 Fuzzy Framework for Breast Microcalcification Classification in Digital Mammography**

Complete bibliographic information and DOI can be added after publication.
