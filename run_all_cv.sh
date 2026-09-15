#!/usr/bin/env bash
set -euo pipefail
FEATURES=${1:-features/all_features.npz}
mkdir -p results
python scripts/train_cv.py --features "$FEATURES" --dataset DDSM --output results/ddsm_oof.csv
python scripts/train_cv.py --features "$FEATURES" --dataset INbreast --output results/inbreast_oof.csv
python scripts/train_cv.py --features "$FEATURES" --dataset pooled --output results/pooled_oof.csv
python scripts/make_dataset_table.py --ddsm results/ddsm_oof.csv --inbreast results/inbreast_oof.csv --pooled results/pooled_oof.csv --output results/dataset_specific_performance.csv
