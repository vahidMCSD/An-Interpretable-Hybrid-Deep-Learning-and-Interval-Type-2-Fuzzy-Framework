#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Short end-to-end validation using synthetic feature data."
    )
    p.add_argument("--work-dir", default="results/reviewer_smoke_test")
    p.add_argument("--samples", type=int, default=80)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    work = Path(a.work_dir)
    if not work.is_absolute():
        work = ROOT / work
    work.mkdir(parents=True, exist_ok=True)

    features = work / "synthetic_features.npz"
    cv = work / "cv_predictions.csv"
    bundle = work / "final_bundle"
    test = work / "test_predictions.csv"
    km = work / "km_validation.json"

    run([sys.executable, "-m", "pytest", "-q"])
    run([
        sys.executable, str(ROOT / "scripts" / "synthetic_smoke_data.py"),
        "--output", str(features), "--n", str(a.samples), "--seed", str(a.seed),
    ])
    run([
        sys.executable, str(ROOT / "main.py"), "cv",
        "--features", str(features), "--dataset", "pooled", "--output", str(cv),
        "--device", a.device, "--outer-folds", "2", "--inner-folds", "2",
        "--epochs", "1", "--bootstrap", "20", "--seed", str(a.seed),
    ])
    run([
        sys.executable, str(ROOT / "main.py"), "train",
        "--features", str(features), "--dataset", "pooled",
        "--output-dir", str(bundle), "--device", a.device,
        "--epochs", "1", "--seed", str(a.seed),
    ])
    run([
        sys.executable, str(ROOT / "main.py"), "test",
        "--features", str(features), "--bundle", str(bundle),
        "--output", str(test), "--device", a.device,
        "--bootstrap", "20", "--seed", str(a.seed),
    ])
    run([
        sys.executable, str(ROOT / "main.py"), "validate-km",
        "--samples", "100", "--seed", str(a.seed), "--output", str(km),
    ])

    report = {
        "status": "PASS",
        "tests": "pytest",
        "synthetic_features": str(features),
        "cv_predictions": str(cv),
        "bundle": str(bundle),
        "test_predictions": str(test),
        "km_validation": str(km),
    }
    (work / "SMOKE_TEST_PASS.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nReviewer smoke test: PASS")
    print("Outputs:", work)


if __name__ == "__main__":
    main()
