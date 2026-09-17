#!/usr/bin/env bash
set -euo pipefail
FEATURES=${1:-features/all_features.npz}
DEVICE=${DEVICE:-cuda}
mkdir -p results
python scripts/train_cv.py --features "$FEATURES" --dataset DDSM --output results/DDSM.csv --device "$DEVICE"
python scripts/train_cv.py --features "$FEATURES" --dataset INbreast --output results/INbreast.csv --device "$DEVICE"
python scripts/train_cv.py --features "$FEATURES" --dataset pooled --output results/pooled.csv --device "$DEVICE"
python scripts/make_dataset_table.py results/DDSM.csv results/INbreast.csv --output results/dataset_specific_performance.csv
