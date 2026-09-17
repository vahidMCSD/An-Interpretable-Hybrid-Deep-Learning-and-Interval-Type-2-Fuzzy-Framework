# Reviewer Execution Guide

This guide provides the shortest path for verifying the repository and reproducing the experimental workflow.

## 1. Create the environment

```bash
python -m venv .venv
```

Activate it and install the project:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps --no-build-isolation
```

Confirm package availability:

```bash
python scripts/check_environment.py
```

## 2. Verify the code without the mammography datasets

Run:

```bash
python main.py smoke-test --device cpu
```

This command runs the unit tests and a short synthetic end-to-end feature-level workflow. Successful completion creates:

```text
results/reviewer_smoke_test/SMOKE_TEST_PASS.json
```

Synthetic outputs are execution checks only and are not used as paper results.

## 3. Prepare real data

The repository does not contain DDSM/CBIS-DDSM or INbreast images.

Prepare a canonical manifest with:

```text
image_path,label,patient_id,dataset
```

`label` must be `0` for benign and `1` for malignant. `dataset` should be `DDSM` or `INbreast`. `patient_id` must identify the patient and is used to prevent patient overlap across folds.

If full mammograms and canonical annotations are available, ROIs can be generated with:

```bash
python main.py prepare-rois \
  --annotations data/annotations.csv \
  --output-dir data/rois \
  --manifest data/manifest.csv
```

## 4. Optional branch fine-tuning

```bash
python main.py train-branch --manifest data/manifest.csv --branch effnet --output checkpoints/effnet.pt --device cuda
python main.py train-branch --manifest data/manifest.csv --branch vit --output checkpoints/vit.pt --device cuda
```

## 5. Extract features

```bash
python main.py extract \
  --manifest data/manifest.csv \
  --output features/all_features.npz \
  --eff-checkpoint checkpoints/effnet.pt \
  --vit-checkpoint checkpoints/vit.pt \
  --batch-size 16 \
  --device cuda
```

## 6. Reproduce patient-level cross-validation

```bash
python main.py cv --features features/all_features.npz --dataset DDSM --output results/ddsm_cv.csv --device cuda
python main.py cv --features features/all_features.npz --dataset INbreast --output results/inbreast_cv.csv --device cuda
python main.py cv --features features/all_features.npz --dataset pooled --output results/pooled_cv.csv --device cuda
```

Each output CSV has a corresponding `.summary.json` file.

## 7. Reproduce cross-dataset validation

```bash
python main.py cross-dataset --features features/all_features.npz --output-dir results/cross_dataset --device cuda
```

## 8. Train the final bundle

```bash
python main.py train --features features/all_features.npz --dataset pooled --output-dir checkpoints/final --device cuda
```

## 9. Evaluate a locked holdout

A holdout must be identified before final evaluation either by manifest split or exact stored indices.

Manifest method:

```bash
python main.py holdout \
  --features features/all_features.npz \
  --bundle checkpoints/final \
  --manifest data/manifest.csv \
  --split holdout \
  --output results/holdout_predictions.csv \
  --device cuda
```

Exact-index method:

```bash
python main.py holdout \
  --features features/all_features.npz \
  --bundle checkpoints/final \
  --indices data/locked_holdout_indices.npy \
  --output results/holdout_predictions.csv \
  --device cuda
```

The holdout command never retrains the model.

## 10. Additional analyses

```bash
python main.py sensitivity --features features/all_features.npz --output-dir results/sensitivity --device cuda
python main.py ablation --features features/all_features.npz --output-dir results/ablation --device cuda
python main.py validate-km --samples 1000 --output results/km_validation.json
python main.py benchmark --image path/to/example_roi.png --output results/benchmark.json --device cuda
```

## Reproducibility settings

The main manuscript configuration is in:

```text
config/paper.yaml
```

Core defaults include five outer folds, three inner folds, PCA retained variance of 0.95, Fuzzy ART vigilance of 0.85, Fuzzy ART beta of 0.50, and 1000 bootstrap iterations.

All data-dependent transformations are fitted using training data only.
