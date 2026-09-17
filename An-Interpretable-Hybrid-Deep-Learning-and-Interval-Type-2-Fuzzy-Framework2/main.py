#!/usr/bin/env python
from __future__ import annotations

"""Single entry point for the manuscript code.

Examples
--------
python main.py extract --manifest data/manifest.csv --output features/all_features.npz --device cuda
python main.py cv --features features/all_features.npz --dataset DDSM --output results/ddsm.csv --device cuda
python main.py train --features features/all_features.npz --dataset pooled --output-dir checkpoints/final --device cuda
python main.py test --features features/all_features.npz --bundle checkpoints/final --output results/test.csv
python main.py holdout --features features/all_features.npz --bundle checkpoints/final --manifest data/manifest.csv --split holdout --output results/holdout.csv
python main.py cross-dataset --features features/all_features.npz --output-dir results/cross_dataset --device cuda
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / 'scripts'


def run_script(name: str, args: list[str]):
    cmd = [sys.executable, str(SCRIPTS / name), *args]
    print('RUN:', ' '.join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def common_device(p):
    p.add_argument('--device', default=None, help='cpu or cuda; omit to let the script choose')


def maybe_device(args, out):
    if getattr(args, 'device', None):
        out += ['--device', args.device]
    return out


def cmd_prepare(args):
    v = ['--annotations', args.annotations, '--output-dir', args.output_dir, '--manifest', args.manifest]
    run_script('prepare_rois.py', v)


def cmd_train_branch(args):
    v = ['--manifest', args.manifest, '--branch', args.branch, '--output', args.output]
    v = maybe_device(args, v)
    if args.epochs is not None: v += ['--epochs', str(args.epochs)]
    run_script('train_deep_branch.py', v)


def cmd_extract(args):
    v = ['--manifest', args.manifest, '--output', args.output, '--batch-size', str(args.batch_size)]
    v = maybe_device(args, v)
    if args.eff_checkpoint: v += ['--eff-checkpoint', args.eff_checkpoint]
    if args.vit_checkpoint: v += ['--vit-checkpoint', args.vit_checkpoint]
    if args.no_pretrained: v += ['--no-pretrained']
    if args.seed is not None: v += ['--seed', str(args.seed)]
    run_script('extract_features.py', v)


def cmd_cv(args):
    v = [
        '--features', args.features,
        '--output', args.output,
        '--dataset', args.dataset,
        '--outer-folds', str(args.outer_folds),
        '--inner-folds', str(args.inner_folds),
        '--pca-variance', str(args.pca_variance),
        '--epochs', str(args.epochs),
        '--bootstrap', str(args.bootstrap),
        '--seed', str(args.seed),
    ]
    v = maybe_device(args, v)
    run_script('train_cv.py', v)


def cmd_train(args):
    v = [
        '--features', args.features,
        '--output-dir', args.output_dir,
        '--dataset', args.dataset,
        '--epochs', str(args.epochs),
        '--seed', str(args.seed),
    ]
    v = maybe_device(args, v)
    run_script('train_final_model.py', v)


def cmd_test(args):
    v = ['--features', args.features, '--bundle', args.bundle, '--output', args.output]
    if args.indices: v += ['--indices', args.indices]
    if args.manifest: v += ['--manifest', args.manifest]
    if args.split: v += ['--split-value', args.split]
    if args.split_column: v += ['--split-column', args.split_column]
    v += ['--bootstrap', str(args.bootstrap), '--seed', str(args.seed)]
    v = maybe_device(args, v)
    run_script('evaluate_bundle.py', v)


def cmd_holdout(args):
    if not args.manifest and not args.indices:
        raise SystemExit('holdout requires --manifest with --split, or --indices with exact locked holdout indices')
    v = ['--features', args.features, '--bundle', args.bundle, '--output', args.output]
    if args.indices:
        v += ['--indices', args.indices]
    else:
        v += ['--manifest', args.manifest, '--split-column', args.split_column, '--split-value', args.split]
    v += ['--bootstrap', str(args.bootstrap), '--seed', str(args.seed)]
    v = maybe_device(args, v)
    run_script('evaluate_bundle.py', v)


def cmd_external(args):
    v = [
        '--features', args.features,
        '--train-dataset', args.train_dataset,
        '--test-dataset', args.test_dataset,
        '--output', args.output,
        '--epochs', str(args.epochs),
        '--bootstrap', str(args.bootstrap),
        '--seed', str(args.seed),
    ]
    v = maybe_device(args, v)
    run_script('train_external.py', v)


def cmd_cross_dataset(args):
    v = ['--features', args.features, '--output-dir', args.output_dir, '--epochs', str(args.epochs)]
    v = maybe_device(args, v)
    run_script('run_cross_dataset.py', v)


def cmd_sensitivity(args):
    v = ['--features', args.features, '--output-dir', args.output_dir, '--epochs', str(args.epochs)]
    v = maybe_device(args, v)
    run_script('run_sensitivity.py', v)


def cmd_ablation(args):
    output = str(Path(args.output_dir) / 'table6_ablation.csv')
    v = [
        '--features', args.features,
        '--output', output,
        '--epochs', str(args.epochs),
        '--outer-folds', str(args.outer_folds),
        '--seed', str(args.seed),
    ]
    v = maybe_device(args, v)
    run_script('run_ablation.py', v)


def cmd_smoke_test(args):
    v = ['--work-dir', args.work_dir, '--samples', str(args.samples), '--seed', str(args.seed)]
    v = maybe_device(args, v)
    run_script('reviewer_smoke_test.py', v)


def cmd_validate_km(args):
    v = ['--samples', str(args.samples), '--seed', str(args.seed), '--output', args.output]
    run_script('validate_km.py', v)


def cmd_benchmark(args):
    v = ['--image', args.image, '--output', args.output, '--iterations', str(args.iterations)]
    v = maybe_device(args, v)
    if args.no_pretrained: v += ['--no-pretrained']
    run_script('benchmark.py', v)


def build_parser():
    p = argparse.ArgumentParser(description='Single main.py entry point for the manuscript implementation.')
    sub = p.add_subparsers(dest='command', required=True)

    q = sub.add_parser('prepare-rois', help='Generate ROI images/manifest from canonical annotations.')
    q.add_argument('--annotations', required=True); q.add_argument('--output-dir', required=True); q.add_argument('--manifest', required=True); q.set_defaults(func=cmd_prepare)

    q = sub.add_parser('train-branch', help='Fine-tune EfficientNetV2+CBAM or ViT branch.')
    q.add_argument('--manifest', required=True); q.add_argument('--branch', choices=['effnet','vit'], required=True); q.add_argument('--output', required=True); q.add_argument('--epochs', type=int); common_device(q); q.set_defaults(func=cmd_train_branch)

    q = sub.add_parser('extract', help='Extract wavelet, EfficientNetV2+CBAM, and ViT features.')
    q.add_argument('--manifest', required=True); q.add_argument('--output', required=True); q.add_argument('--batch-size', type=int, default=16); q.add_argument('--eff-checkpoint'); q.add_argument('--vit-checkpoint'); q.add_argument('--no-pretrained', action='store_true'); q.add_argument('--seed', type=int, default=12345); common_device(q); q.set_defaults(func=cmd_extract)

    q = sub.add_parser('cv', help='Patient-level nested cross-validation.')
    q.add_argument('--features', required=True); q.add_argument('--output', required=True); q.add_argument('--dataset', choices=['DDSM','INbreast','pooled'], default='pooled'); q.add_argument('--outer-folds', type=int, default=5); q.add_argument('--inner-folds', type=int, default=3); q.add_argument('--pca-variance', type=float, default=0.95); q.add_argument('--epochs', type=int, default=50); q.add_argument('--bootstrap', type=int, default=1000); q.add_argument('--seed', type=int, default=12345); common_device(q); q.set_defaults(func=cmd_cv)

    q = sub.add_parser('train', help='Train and save the final feature-level model bundle.')
    q.add_argument('--features', required=True); q.add_argument('--output-dir', required=True); q.add_argument('--dataset', choices=['DDSM','INbreast','pooled'], default='pooled'); q.add_argument('--epochs', type=int, default=50); q.add_argument('--seed', type=int, default=12345); common_device(q); q.set_defaults(func=cmd_train)

    q = sub.add_parser('test', help='Evaluate a saved bundle without retraining.')
    q.add_argument('--features', required=True); q.add_argument('--bundle', required=True); q.add_argument('--output', required=True); q.add_argument('--manifest'); q.add_argument('--split'); q.add_argument('--split-column', default='split'); q.add_argument('--indices'); q.add_argument('--bootstrap', type=int, default=1000); q.add_argument('--seed', type=int, default=12345); common_device(q); q.set_defaults(func=cmd_test)

    q = sub.add_parser('holdout', help='Evaluate a saved bundle once on an explicitly locked holdout. Never retrains.')
    q.add_argument('--features', required=True); q.add_argument('--bundle', required=True); q.add_argument('--output', required=True); q.add_argument('--manifest'); q.add_argument('--split', default='holdout'); q.add_argument('--split-column', default='split'); q.add_argument('--indices'); q.add_argument('--bootstrap', type=int, default=1000); q.add_argument('--seed', type=int, default=12345); common_device(q); q.set_defaults(func=cmd_holdout)

    q = sub.add_parser('external', help='One external validation direction.')
    q.add_argument('--features', required=True); q.add_argument('--train-dataset', choices=['DDSM','INbreast'], required=True); q.add_argument('--test-dataset', choices=['DDSM','INbreast'], required=True); q.add_argument('--output', required=True); q.add_argument('--epochs', type=int, default=50); q.add_argument('--bootstrap', type=int, default=1000); q.add_argument('--seed', type=int, default=12345); common_device(q); q.set_defaults(func=cmd_external)

    q = sub.add_parser('cross-dataset', help='Run both DDSM->INbreast and INbreast->DDSM validations.')
    q.add_argument('--features', required=True); q.add_argument('--output-dir', default='results/cross_dataset'); q.add_argument('--epochs', type=int, default=50); common_device(q); q.set_defaults(func=cmd_cross_dataset)

    q = sub.add_parser('sensitivity', help='Run fuzzy/PCA/ART sensitivity analysis utilities.')
    q.add_argument('--features', required=True); q.add_argument('--output-dir', default='results/sensitivity'); q.add_argument('--epochs', type=int, default=50); common_device(q); q.set_defaults(func=cmd_sensitivity)

    q = sub.add_parser('ablation', help='Run component ablation analysis.')
    q.add_argument('--features', required=True); q.add_argument('--output-dir', default='results/ablation'); q.add_argument('--epochs', type=int, default=50); q.add_argument('--outer-folds', type=int, default=5); q.add_argument('--seed', type=int, default=12345); common_device(q); q.set_defaults(func=cmd_ablation)

    q = sub.add_parser('validate-km', help='Validate midpoint approximation against iterative Karnik-Mendel.')
    q.add_argument('--samples', type=int, default=1000); q.add_argument('--seed', type=int, default=12345); q.add_argument('--output', default='results/km_validation.json'); q.set_defaults(func=cmd_validate_km)


    q = sub.add_parser('smoke-test', help='Run a short reviewer validation without mammography data.')
    q.add_argument('--work-dir', default='results/reviewer_smoke_test'); q.add_argument('--samples', type=int, default=80); q.add_argument('--seed', type=int, default=7); common_device(q); q.set_defaults(func=cmd_smoke_test)

    q = sub.add_parser('benchmark', help='Benchmark inference components on one ROI.')
    q.add_argument('--image', required=True); q.add_argument('--output', default='results/benchmark.json'); q.add_argument('--iterations', type=int, default=50); q.add_argument('--no-pretrained', action='store_true'); common_device(q); q.set_defaults(func=cmd_benchmark)

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
