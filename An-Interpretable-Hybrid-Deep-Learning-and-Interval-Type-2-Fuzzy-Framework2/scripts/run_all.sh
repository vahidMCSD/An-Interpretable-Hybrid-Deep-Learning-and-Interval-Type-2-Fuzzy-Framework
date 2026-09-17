#!/usr/bin/env bash
set -euo pipefail
FEATURES="${1:-features/all_features.npz}"
DEVICE="${DEVICE:-cuda}"
mkdir -p results
python scripts/train_cv.py --features "$FEATURES" --dataset DDSM --output results/DDSM.csv --device "$DEVICE"
python scripts/train_cv.py --features "$FEATURES" --dataset INbreast --output results/INbreast.csv --device "$DEVICE"
python scripts/train_cv.py --features "$FEATURES" --dataset pooled --output results/pooled.csv --device "$DEVICE"
python scripts/make_dataset_table.py results/DDSM.csv results/INbreast.csv --output results/dataset_specific_performance.csv
python scripts/plot_results.py --predictions results/pooled.csv --output-dir results/figures
python scripts/run_cross_dataset.py --features "$FEATURES" --output-dir results/cross_dataset --device "$DEVICE"
python scripts/run_sensitivity.py --features "$FEATURES" --output-dir results/sensitivity --device "$DEVICE"
python scripts/run_ablation.py --features "$FEATURES" --output results/table6_ablation.csv
python scripts/validate_km.py --samples 1000 --output results/km_validation.json
python scripts/export_supplementary.py --output-dir results/supplementary
